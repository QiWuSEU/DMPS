#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vecaep_io.py

Strict I/O, validation, canonical-grid assembly, circular-polarization
conversion, and cache construction for the 187-port Stage3 Vec-AEP data set.

The module preserves the physical data contract of the HFSS exports:

* each Vec-AEP file is a 1 W, 0 deg single-port complex far-field response;
* Etheta/Ephi remain in the HFSS global spherical basis;
* no position-phase de-embedding is applied;
* no physical-rotation back transformation is applied;
* no spin compensation or spectral decomposition is applied;
* array synthesis uses ordinary matrix multiplication A @ w, not A.conj().T @ w.

Default project location on the user's computer:
    F:\\ningzhiwen\\rebuild\\stage3_vecaep_synthesis
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Current formal Stage3 baseline
# -----------------------------------------------------------------------------
SCHEMA_NAME = "stage3_vecaep_cache"
SCHEMA_VERSION = 2
SCRIPT_VERSION = "1.0.2-win-direct-cache"
CACHE_COMPLETE_NAME = "cache_complete.json"
CACHE_INCOMPLETE_NAME = ".cache_build_incomplete"

EXPECTED_PORT_COUNT = 187
EXPECTED_FREQUENCY_GHZ = 2.4
EXPECTED_CUT_L_MM = 3.60
EXPECTED_N_THETA = 91
EXPECTED_N_PHI = 180
EXPECTED_POINT_COUNT = EXPECTED_N_THETA * EXPECTED_N_PHI
EXPECTED_THETA_MIN_DEG = 0.0
EXPECTED_THETA_MAX_DEG = 90.0
EXPECTED_THETA_STEP_DEG = 1.0
EXPECTED_PHI_MIN_DEG = 0.0
EXPECTED_PHI_MAX_DEG = 358.0
EXPECTED_PHI_STEP_DEG = 2.0

DEFAULT_CP_BASIS_SIGN = -1
DEFAULT_VECAEP_ROOT = Path(
    r"F:\ningzhiwen\rebuild\hfssproject\vecaep_exports"
    r"\top1_sr_cut360_legacyenv_2p4GHz_allports\20260725_123606"
)
DEFAULT_CACHE_ROOT = Path(
    r"F:\ningzhiwen\rebuild\stage3_vecaep_synthesis\cache"
)

EPS = 1.0e-30
_FLOAT_TOKEN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
_VALUE_WITH_UNIT_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)='(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<unit>[A-Za-z]+)'"
)


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class HFSSFieldGrid:
    """One canonicalized HFSS complex vector-field export."""

    theta_axis_rad: np.ndarray
    phi_axis_rad: np.ndarray
    theta_points_rad: np.ndarray
    phi_points_rad: np.ndarray
    etheta: np.ndarray
    ephi: np.ndarray
    variation: str | None
    source_path: Path
    angle_mapping: Mapping[str, Any]
    duplicate_report: Mapping[str, Any]
    grid_report: Mapping[str, Any]


@dataclass(frozen=True)
class SourceWeights:
    """Port complex weights reconstructed from an HFSS source-state CSV."""

    weights: np.ndarray
    powers_w: np.ndarray
    phases_deg: np.ndarray
    table: pd.DataFrame
    source_path: Path

    @property
    def total_incident_power_w(self) -> float:
        return float(np.sum(self.powers_w))


@dataclass(frozen=True)
class VecAEPDataset:
    """Cached Stage3 Vec-AEP matrices in point-by-port orientation."""

    cache_dir: Path
    run_root: Path
    theta_axis_rad: np.ndarray
    phi_axis_rad: np.ndarray
    theta_points_rad: np.ndarray
    phi_points_rad: np.ndarray
    etheta_matrix: np.ndarray
    ephi_matrix: np.ndarray
    e_lhcp_matrix: np.ndarray | None
    e_rhcp_matrix: np.ndarray | None
    cp_basis_sign: int
    element_table: pd.DataFrame
    run_config: Mapping[str, Any]
    run_summary: Mapping[str, Any]
    cache_metadata: Mapping[str, Any]

    @property
    def point_count(self) -> int:
        return int(self.etheta_matrix.shape[0])

    @property
    def port_count(self) -> int:
        return int(self.etheta_matrix.shape[1])


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        with tmp.open("r", encoding="utf-8") as f:
            json.load(f)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def setup_logger(log_path: Path | None = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("stage3_vecaep_io")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(console)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        )
        logger.addHandler(file_handler)
    return logger


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _extract_named_value(variation: str, name: str, expected_unit: str | None = None) -> float:
    matches = {
        m.group("name"): (float(m.group("value")), m.group("unit"))
        for m in _VALUE_WITH_UNIT_RE.finditer(str(variation))
    }
    if name not in matches:
        raise ValueError(f"Variation does not contain {name!r}: {variation!r}")
    value, unit = matches[name]
    if expected_unit is not None and unit.lower() != expected_unit.lower():
        raise ValueError(
            f"Variation unit mismatch for {name}: expected {expected_unit!r}, got {unit!r}"
        )
    return value


def _close_scalar(a: float, b: float, atol: float, name: str) -> None:
    if not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=float(atol)):
        raise ValueError(f"{name} mismatch: actual={a!r}, expected={b!r}, atol={atol}")


# -----------------------------------------------------------------------------
# HFSS CSV parsing and canonical grid construction
# -----------------------------------------------------------------------------
def read_csv_with_variation(
    csv_path: Path,
    variation_selected: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one HFSS field CSV and select exactly one Variation value."""

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"HFSS CSV not found: {csv_path}")

    required = {
        "Variation",
        "Phi(rad)",
        "Theta(rad)",
        "rEphi(mag ang)",
        "rEtheta(mag ang)",
    }
    df_all = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        skipinitialspace=True,
        usecols=lambda c: str(c).strip() in required,
    )
    df_all = _normalize_column_names(df_all)
    missing = sorted(required.difference(df_all.columns))
    if missing:
        raise KeyError(f"Missing required columns {missing} in {csv_path}; got {list(df_all.columns)}")

    values = df_all["Variation"].astype(str).str.strip()
    unique = list(dict.fromkeys(v for v in values.tolist() if v))
    requested = None if variation_selected is None else str(variation_selected).strip()

    if not unique:
        if requested:
            raise ValueError(
                f"Variation was requested but the Variation column is empty in {csv_path}"
            )
        selected = None
        selected_df = df_all.copy()
        mode = "empty_variation_column"
    elif len(unique) == 1:
        selected = unique[0]
        if requested is not None and requested != selected:
            raise ValueError(
                f"Requested Variation does not match the only available value in {csv_path}:\n"
                f"requested={requested!r}\navailable={selected!r}"
            )
        selected_df = df_all.loc[values == selected].copy()
        mode = "single_unique"
    else:
        if requested is None:
            raise ValueError(
                f"CSV contains multiple Variation values; specify variation_selected exactly. "
                f"File={csv_path}, values={unique}"
            )
        if requested not in unique:
            raise ValueError(
                f"Requested Variation not present in {csv_path}: {requested!r}; values={unique}"
            )
        selected = requested
        selected_df = df_all.loc[values == selected].copy()
        mode = "explicit_multiple_unique"

    if selected_df.empty:
        raise ValueError(f"No rows remain after Variation selection in {csv_path}")

    return selected_df.reset_index(drop=True), {
        "row_count_total": int(len(df_all)),
        "row_count_selected": int(len(selected_df)),
        "unique_variation_count": int(len(unique)),
        "unique_variations": unique,
        "variation_selected": selected,
        "selection_mode": mode,
    }


def _infer_unit_from_max_abs(values: Iterable[np.ndarray]) -> tuple[str, float]:
    arrays = [np.asarray(v, dtype=np.float64).reshape(-1) for v in values]
    merged = np.concatenate(arrays)
    if merged.size == 0 or not np.all(np.isfinite(merged)):
        raise ValueError("Cannot infer units from empty or non-finite data")
    max_abs = float(np.max(np.abs(merged)))
    inferred = "deg" if max_abs > (2.0 * math.pi + 0.2) else "rad"
    return inferred, max_abs


def _require_unit_match(name: str, specified: str, inferred: str, max_abs: float) -> None:
    specified = str(specified).lower()
    if specified not in {"rad", "deg"}:
        raise ValueError(f"{name} must be 'rad' or 'deg', got {specified!r}")
    if specified != inferred:
        raise ValueError(
            f"[UNIT MISMATCH] {name}: specified={specified!r}, inferred={inferred!r}, "
            f"max_abs_raw={max_abs:.12g}"
        )


def resolve_angle_columns(
    theta_column_rad: np.ndarray,
    phi_column_rad: np.ndarray,
    mode: str = "auto",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Resolve physical theta/phi from HFSS columns using angular spans."""

    mode = str(mode).strip().lower()
    if mode not in {"auto", "standard", "swapped"}:
        raise ValueError("angle_column_mode must be auto, standard, or swapped")

    theta_column_rad = np.asarray(theta_column_rad, dtype=np.float64)
    phi_column_rad = np.asarray(phi_column_rad, dtype=np.float64)
    theta_span_deg = float(np.rad2deg(np.max(theta_column_rad) - np.min(theta_column_rad)))
    phi_span_deg = float(np.rad2deg(np.max(phi_column_rad) - np.min(phi_column_rad)))

    if mode == "standard":
        used = "standard"
        reason = "explicit_standard"
    elif mode == "swapped":
        used = "swapped"
        reason = "explicit_swapped"
    elif theta_span_deg <= 180.0 + 1e-8 and phi_span_deg > 180.0:
        used = "standard"
        reason = "Phi column spans the full-circle axis"
    elif phi_span_deg <= 180.0 + 1e-8 and theta_span_deg > 180.0:
        used = "swapped"
        reason = "Theta column spans the full-circle axis"
    else:
        raise ValueError(
            "Automatic angle-column mapping is ambiguous. "
            f"Theta span={theta_span_deg:.9g} deg, Phi span={phi_span_deg:.9g} deg. "
            "Specify --angle-column-mode standard or swapped."
        )

    if used == "standard":
        theta = theta_column_rad
        phi = phi_column_rad
        theta_from = "Theta(rad)"
        phi_from = "Phi(rad)"
    else:
        theta = phi_column_rad
        phi = theta_column_rad
        theta_from = "Phi(rad)"
        phi_from = "Theta(rad)"

    return theta, phi, {
        "mode_requested": mode,
        "mode_used": used,
        "reason": reason,
        "physical_theta_from": theta_from,
        "physical_phi_from": phi_from,
        "theta_column_span_deg": theta_span_deg,
        "phi_column_span_deg": phi_span_deg,
    }


def parse_mag_phase_series(
    series: pd.Series,
    *,
    column_name: str,
    phase_unit: str = "rad",
) -> np.ndarray:
    """Parse an HFSS 'magnitude phase' column into complex128 values."""

    phase_unit = str(phase_unit).lower()
    if phase_unit not in {"rad", "deg"}:
        raise ValueError(f"phase_unit must be rad or deg, got {phase_unit!r}")

    text = series.astype(str).str.strip()
    parts = text.str.split(pat=r"[\s,]+", n=1, expand=True)
    if parts.shape[1] != 2:
        bad = text.head(5).tolist()
        raise ValueError(
            f"Expected exactly two tokens in {column_name}; examples={bad}"
        )

    try:
        mag = pd.to_numeric(parts.iloc[:, 0], errors="raise").to_numpy(dtype=np.float64)
        ang = pd.to_numeric(parts.iloc[:, 1], errors="raise").to_numpy(dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Failed parsing magnitude/phase column {column_name}: {exc}") from exc

    if np.any(mag < 0.0):
        idx = int(np.flatnonzero(mag < 0.0)[0])
        raise ValueError(f"Negative magnitude in {column_name} at row {idx}: {mag[idx]}")
    if not np.all(np.isfinite(mag)) or not np.all(np.isfinite(ang)):
        raise ValueError(f"NaN or Inf in {column_name}")

    inferred, max_abs = _infer_unit_from_max_abs([ang])
    _require_unit_match(f"phase_unit({column_name})", phase_unit, inferred, max_abs)
    ang_rad = ang if phase_unit == "rad" else np.deg2rad(ang)
    return np.asarray(mag * np.exp(1j * ang_rad), dtype=np.complex128)


def _relative_group_spread(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.complex128)
    mean = complex(np.mean(values))
    floor = max(float(np.max(np.abs(values))) * 1e-14, 1e-30)
    return float(np.max(np.abs(values - mean)) / max(abs(mean), floor))


def canonicalize_field_grid(
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    etheta: np.ndarray,
    ephi: np.ndarray,
    *,
    angle_tolerance_rad: float = 1e-12,
    duplicate_relative_tolerance: float = 1e-8,
    strict_baseline: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
]:
    """Sort, deduplicate exact angle pairs, and validate a full theta-phi grid."""

    theta = np.asarray(theta_rad, dtype=np.float64).reshape(-1)
    phi = np.mod(np.asarray(phi_rad, dtype=np.float64).reshape(-1), 2.0 * math.pi)
    etheta = np.asarray(etheta, dtype=np.complex128).reshape(-1)
    ephi = np.asarray(ephi, dtype=np.complex128).reshape(-1)
    if not (theta.size == phi.size == etheta.size == ephi.size):
        raise ValueError("theta, phi, Etheta, and Ephi lengths differ")
    if angle_tolerance_rad <= 0.0:
        raise ValueError("angle_tolerance_rad must be positive")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(phi)):
        raise ValueError("Angle data contain NaN or Inf")
    if not np.all(np.isfinite(etheta)) or not np.all(np.isfinite(ephi)):
        raise ValueError("Field data contain NaN or Inf")

    qtheta = np.rint(theta / angle_tolerance_rad).astype(np.int64)
    qphi = np.rint(phi / angle_tolerance_rad).astype(np.int64)
    order = np.lexsort((qphi, qtheta))
    qtheta = qtheta[order]
    qphi = qphi[order]
    theta = theta[order]
    phi = phi[order]
    etheta = etheta[order]
    ephi = ephi[order]

    pair_change = np.ones(theta.size, dtype=bool)
    if theta.size > 1:
        pair_change[1:] = (qtheta[1:] != qtheta[:-1]) | (qphi[1:] != qphi[:-1])
    starts = np.flatnonzero(pair_change)
    ends = np.r_[starts[1:], theta.size]

    if np.all((ends - starts) == 1):
        duplicate_report = {
            "duplicate_group_count": 0,
            "duplicate_extra_row_count": 0,
            "max_relative_spread_etheta": 0.0,
            "max_relative_spread_ephi": 0.0,
            "merge_method": "none",
        }
    else:
        theta_out: list[float] = []
        phi_out: list[float] = []
        et_out: list[complex] = []
        ep_out: list[complex] = []
        qt_out: list[int] = []
        qp_out: list[int] = []
        max_rel_t = 0.0
        max_rel_p = 0.0
        duplicate_groups = 0
        duplicate_extra = 0
        for start, end in zip(starts, ends):
            vals_t = etheta[start:end]
            vals_p = ephi[start:end]
            count = int(end - start)
            if count > 1:
                duplicate_groups += 1
                duplicate_extra += count - 1
                rel_t = _relative_group_spread(vals_t)
                rel_p = _relative_group_spread(vals_p)
                max_rel_t = max(max_rel_t, rel_t)
                max_rel_p = max(max_rel_p, rel_p)
                if max(rel_t, rel_p) > duplicate_relative_tolerance:
                    raise ValueError(
                        "Duplicate angle rows are inconsistent: "
                        f"qtheta={qtheta[start]}, qphi={qphi[start]}, "
                        f"rel_spread(Etheta)={rel_t:.6g}, rel_spread(Ephi)={rel_p:.6g}"
                    )
            theta_out.append(float(qtheta[start] * angle_tolerance_rad))
            phi_out.append(float(qphi[start] * angle_tolerance_rad))
            et_out.append(complex(np.mean(vals_t)))
            ep_out.append(complex(np.mean(vals_p)))
            qt_out.append(int(qtheta[start]))
            qp_out.append(int(qphi[start]))
        theta = np.asarray(theta_out, dtype=np.float64)
        phi = np.asarray(phi_out, dtype=np.float64)
        etheta = np.asarray(et_out, dtype=np.complex128)
        ephi = np.asarray(ep_out, dtype=np.complex128)
        qtheta = np.asarray(qt_out, dtype=np.int64)
        qphi = np.asarray(qp_out, dtype=np.int64)
        duplicate_report = {
            "duplicate_group_count": duplicate_groups,
            "duplicate_extra_row_count": duplicate_extra,
            "max_relative_spread_etheta": max_rel_t,
            "max_relative_spread_ephi": max_rel_p,
            "merge_method": "complex_mean_after_consistency_check",
        }

    unique_qtheta = np.unique(qtheta)
    unique_qphi = np.unique(qphi)
    theta_axis = unique_qtheta.astype(np.float64) * angle_tolerance_rad
    phi_axis = unique_qphi.astype(np.float64) * angle_tolerance_rad
    n_theta = int(theta_axis.size)
    n_phi = int(phi_axis.size)
    if theta.size != n_theta * n_phi:
        raise ValueError(
            "HFSS data do not form a complete theta-phi Cartesian grid after deduplication: "
            f"rows={theta.size}, grid={n_theta}x{n_phi}={n_theta*n_phi}"
        )

    expected_theta = np.repeat(theta_axis, n_phi)
    expected_phi = np.tile(phi_axis, n_theta)
    if not np.allclose(theta, expected_theta, rtol=0.0, atol=2.0 * angle_tolerance_rad):
        raise ValueError("Canonical theta point order is not a full theta-major grid")
    if not np.allclose(phi, expected_phi, rtol=0.0, atol=2.0 * angle_tolerance_rad):
        raise ValueError("Canonical phi point order is not a full theta-major grid")

    if n_theta < 2 or n_phi < 3:
        raise ValueError("Angular grid is too small")
    dtheta = np.diff(theta_axis)
    dphi = np.diff(phi_axis)
    wrap_step = 2.0 * math.pi - float(phi_axis[-1]) + float(phi_axis[0])
    phi_steps = np.r_[dphi, wrap_step]
    theta_step = float(np.median(dtheta))
    phi_step = float(np.median(phi_steps))
    if not np.allclose(dtheta, theta_step, rtol=1e-9, atol=1e-12):
        raise ValueError("Theta grid is not uniformly sampled")
    if not np.allclose(phi_steps, phi_step, rtol=1e-9, atol=1e-12):
        raise ValueError("Phi grid is not uniformly sampled across the periodic wrap")
    if not math.isclose(n_phi * phi_step, 2.0 * math.pi, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError("Phi samples do not cover exactly one complete period")

    grid_report = {
        "n_theta": n_theta,
        "n_phi": n_phi,
        "point_count": int(theta.size),
        "theta_min_deg": float(np.rad2deg(theta_axis[0])),
        "theta_max_deg": float(np.rad2deg(theta_axis[-1])),
        "theta_step_deg": float(np.rad2deg(theta_step)),
        "phi_min_deg": float(np.rad2deg(phi_axis[0])),
        "phi_max_deg": float(np.rad2deg(phi_axis[-1])),
        "phi_step_deg": float(np.rad2deg(phi_step)),
        "phi_period_error_deg": float(np.rad2deg(abs(n_phi * phi_step - 2.0 * math.pi))),
        "sort_order": "theta_major_phi_minor",
        "phi_wrapped_to_0_2pi": True,
    }

    if strict_baseline:
        expected_checks = [
            (n_theta, EXPECTED_N_THETA, 0.0, "n_theta"),
            (n_phi, EXPECTED_N_PHI, 0.0, "n_phi"),
            (theta.size, EXPECTED_POINT_COUNT, 0.0, "point_count"),
        ]
        for actual, expected, _, name in expected_checks:
            if int(actual) != int(expected):
                raise ValueError(f"{name} mismatch: actual={actual}, expected={expected}")
        _close_scalar(grid_report["theta_min_deg"], EXPECTED_THETA_MIN_DEG, 1e-8, "theta_min_deg")
        _close_scalar(grid_report["theta_max_deg"], EXPECTED_THETA_MAX_DEG, 1e-8, "theta_max_deg")
        _close_scalar(grid_report["theta_step_deg"], EXPECTED_THETA_STEP_DEG, 1e-8, "theta_step_deg")
        _close_scalar(grid_report["phi_min_deg"], EXPECTED_PHI_MIN_DEG, 1e-8, "phi_min_deg")
        _close_scalar(grid_report["phi_max_deg"], EXPECTED_PHI_MAX_DEG, 1e-8, "phi_max_deg")
        _close_scalar(grid_report["phi_step_deg"], EXPECTED_PHI_STEP_DEG, 1e-8, "phi_step_deg")

    return (
        theta_axis,
        phi_axis,
        expected_theta,
        expected_phi,
        etheta,
        ephi,
        duplicate_report,
        grid_report,
    )


def read_hfss_vector_field_csv(
    csv_path: Path,
    *,
    variation_selected: str | None = None,
    angle_column_mode: str = "auto",
    angle_unit: str = "rad",
    phase_unit: str = "rad",
    angle_tolerance_rad: float = 1e-12,
    strict_baseline: bool = True,
    expected_cut_l_mm: float | None = EXPECTED_CUT_L_MM,
    expected_frequency_ghz: float | None = EXPECTED_FREQUENCY_GHZ,
) -> HFSSFieldGrid:
    """Read and canonicalize one raw HFSS Etheta/Ephi far-field CSV."""

    csv_path = Path(csv_path)
    df, variation_report = read_csv_with_variation(csv_path, variation_selected)

    theta_raw = pd.to_numeric(df["Theta(rad)"], errors="raise").to_numpy(dtype=np.float64)
    phi_raw = pd.to_numeric(df["Phi(rad)"], errors="raise").to_numpy(dtype=np.float64)
    inferred_angle_unit, max_abs_angle = _infer_unit_from_max_abs([theta_raw, phi_raw])
    _require_unit_match("angle_unit", angle_unit, inferred_angle_unit, max_abs_angle)
    theta_column_rad = theta_raw if angle_unit == "rad" else np.deg2rad(theta_raw)
    phi_column_rad = phi_raw if angle_unit == "rad" else np.deg2rad(phi_raw)
    theta_rad, phi_rad, mapping = resolve_angle_columns(
        theta_column_rad, phi_column_rad, mode=angle_column_mode
    )

    etheta = parse_mag_phase_series(
        df["rEtheta(mag ang)"], column_name="rEtheta(mag ang)", phase_unit=phase_unit
    )
    ephi = parse_mag_phase_series(
        df["rEphi(mag ang)"], column_name="rEphi(mag ang)", phase_unit=phase_unit
    )

    variation = variation_report["variation_selected"]
    if variation is not None:
        if expected_cut_l_mm is not None:
            cut_l = _extract_named_value(variation, "Cut_L", "mm")
            _close_scalar(cut_l, expected_cut_l_mm, 1e-9, "CSV Variation Cut_L_mm")
        if expected_frequency_ghz is not None:
            freq = _extract_named_value(variation, "Freq", "GHz")
            _close_scalar(freq, expected_frequency_ghz, 1e-9, "CSV Variation Freq_GHz")

    (
        theta_axis,
        phi_axis,
        theta_points,
        phi_points,
        etheta_sorted,
        ephi_sorted,
        duplicate_report,
        grid_report,
    ) = canonicalize_field_grid(
        theta_rad,
        phi_rad,
        etheta,
        ephi,
        angle_tolerance_rad=angle_tolerance_rad,
        strict_baseline=strict_baseline,
    )
    grid_report = {**grid_report, "variation_report": variation_report}

    return HFSSFieldGrid(
        theta_axis_rad=theta_axis,
        phi_axis_rad=phi_axis,
        theta_points_rad=theta_points,
        phi_points_rad=phi_points,
        etheta=etheta_sorted,
        ephi=ephi_sorted,
        variation=variation,
        source_path=csv_path.resolve(),
        angle_mapping=mapping,
        duplicate_report=duplicate_report,
        grid_report=grid_report,
    )


# -----------------------------------------------------------------------------
# Circular polarization and synthesis
# -----------------------------------------------------------------------------
def decompose_circular_components(
    etheta: np.ndarray,
    ephi: np.ndarray,
    *,
    cp_basis_sign: int = DEFAULT_CP_BASIS_SIGN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert global spherical Etheta/Ephi into the configured LHCP/RHCP basis.

    For the current project, cp_basis_sign = -1 gives:
        LHCP = (Etheta - j Ephi) / sqrt(2)
        RHCP = (Etheta + j Ephi) / sqrt(2)
    """

    if int(cp_basis_sign) not in {-1, 1}:
        raise ValueError(f"cp_basis_sign must be -1 or +1, got {cp_basis_sign!r}")
    etheta = np.asarray(etheta, dtype=np.complex128)
    ephi = np.asarray(ephi, dtype=np.complex128)
    if etheta.shape != ephi.shape:
        raise ValueError(f"Etheta/Ephi shape mismatch: {etheta.shape} vs {ephi.shape}")
    sign = float(cp_basis_sign)
    scale = 1.0 / math.sqrt(2.0)
    e_lhcp = (etheta + 1j * sign * ephi) * scale
    e_rhcp = (etheta - 1j * sign * ephi) * scale
    return np.asarray(e_lhcp, dtype=np.complex128), np.asarray(e_rhcp, dtype=np.complex128)


def _validate_weights(weights: np.ndarray, port_count: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.complex128).reshape(-1)
    if weights.size != port_count:
        raise ValueError(f"Weight count mismatch: actual={weights.size}, expected={port_count}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Weights contain NaN or Inf")
    return weights


def synthesize_spherical_fields(
    dataset: VecAEPDataset,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize Etheta/Ephi using ordinary point-by-port matrix multiplication."""

    weights = _validate_weights(weights, dataset.port_count)
    etheta = np.asarray(dataset.etheta_matrix @ weights, dtype=np.complex128)
    ephi = np.asarray(dataset.ephi_matrix @ weights, dtype=np.complex128)
    return etheta, ephi


def synthesize_circular_fields(
    dataset: VecAEPDataset,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize LHCP/RHCP directly from cached circular matrices when available."""

    weights = _validate_weights(weights, dataset.port_count)
    if dataset.e_lhcp_matrix is not None and dataset.e_rhcp_matrix is not None:
        e_lhcp = np.asarray(dataset.e_lhcp_matrix @ weights, dtype=np.complex128)
        e_rhcp = np.asarray(dataset.e_rhcp_matrix @ weights, dtype=np.complex128)
        return e_lhcp, e_rhcp
    etheta, ephi = synthesize_spherical_fields(dataset, weights)
    return decompose_circular_components(
        etheta, ephi, cp_basis_sign=dataset.cp_basis_sign
    )


def read_source_weights_csv(
    csv_path: Path,
    *,
    expected_port_count: int = EXPECTED_PORT_COUNT,
    power_sum_tolerance: float = 1e-10,
) -> SourceWeights:
    """Read HFSS source powers/phases and convert power to complex field weights."""

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Source weights CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", skipinitialspace=True)
    df = _normalize_column_names(df)
    required = {"global_index", "magnitude_W", "total_phase_wrapped_deg"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing source-weight columns {missing} in {csv_path}")
    df = df.sort_values("global_index", kind="mergesort").reset_index(drop=True)
    expected_indices = list(range(expected_port_count))
    if df["global_index"].astype(int).tolist() != expected_indices:
        raise ValueError(
            f"Source global_index must be exactly 0..{expected_port_count - 1}: {csv_path}"
        )
    if len(df) != expected_port_count:
        raise ValueError(f"Source count mismatch in {csv_path}: {len(df)}")

    powers = pd.to_numeric(df["magnitude_W"], errors="raise").to_numpy(dtype=np.float64)
    phases = pd.to_numeric(
        df["total_phase_wrapped_deg"], errors="raise"
    ).to_numpy(dtype=np.float64)
    if np.any(powers < 0.0) or not np.all(np.isfinite(powers)):
        raise ValueError("Source powers contain negative, NaN, or Inf values")
    if not np.all(np.isfinite(phases)):
        raise ValueError("Source phases contain NaN or Inf values")
    total_power = float(np.sum(powers))
    if not math.isclose(total_power, 1.0, rel_tol=0.0, abs_tol=power_sum_tolerance):
        raise ValueError(
            f"Source powers do not sum to 1 W: sum={total_power:.16g}, "
            f"tolerance={power_sum_tolerance}"
        )
    weights = np.sqrt(powers) * np.exp(1j * np.deg2rad(phases))
    return SourceWeights(
        weights=np.asarray(weights, dtype=np.complex128),
        powers_w=powers,
        phases_deg=phases,
        table=df,
        source_path=csv_path.resolve(),
    )


# -----------------------------------------------------------------------------
# Vec-AEP run discovery and metadata validation
# -----------------------------------------------------------------------------
def locate_vecaep_run(root: Path) -> Path:
    """Locate exactly one directory containing vecaep_manifest.csv below root."""

    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Vec-AEP root does not exist: {root}")
    if (root / "vecaep_manifest.csv").is_file():
        return root.resolve()
    matches = sorted(root.rglob("vecaep_manifest.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one vecaep_manifest.csv below {root}, found {len(matches)}: "
            f"{[str(p) for p in matches[:10]]}"
        )
    return matches[0].parent.resolve()


def _validate_baseline_variation(variation: str, *, strict_baseline: bool) -> dict[str, float]:
    cut_l = _extract_named_value(variation, "Cut_L", "mm")
    frequency = _extract_named_value(variation, "freq_", "GHz")
    if strict_baseline:
        _close_scalar(cut_l, EXPECTED_CUT_L_MM, 1e-9, "design_variation Cut_L_mm")
        _close_scalar(frequency, EXPECTED_FREQUENCY_GHZ, 1e-9, "design_variation freq_GHz")
    return {"cut_l_mm": cut_l, "frequency_ghz": frequency}


def load_and_validate_run_metadata(
    run_root: Path,
    *,
    strict_baseline: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_root = Path(run_root)
    run_config = read_json(run_root / "run_config.json")
    run_summary = read_json(run_root / "run_summary.json")

    if run_summary.get("ok") is not True:
        raise ValueError("run_summary.json does not report ok=true")
    expected_count = int(run_config.get("expected_port_count", -1))
    if strict_baseline and expected_count != EXPECTED_PORT_COUNT:
        raise ValueError(
            f"run_config expected_port_count={expected_count}, expected={EXPECTED_PORT_COUNT}"
        )
    frequency = float(run_config.get("frequency_ghz", float("nan")))
    if strict_baseline:
        _close_scalar(frequency, EXPECTED_FREQUENCY_GHZ, 1e-12, "run_config frequency_ghz")
    if float(run_config.get("active_port_power_w", float("nan"))) != 1.0:
        raise ValueError("Vec-AEP active_port_power_w must be 1.0")
    if float(run_config.get("inactive_port_power_w", float("nan"))) != 0.0:
        raise ValueError("Vec-AEP inactive_port_power_w must be 0.0")
    if float(run_config.get("source_phase_deg", float("nan"))) != 0.0:
        raise ValueError("Vec-AEP source_phase_deg must be 0.0")

    forbidden_true = {
        "position_phase_deembedded": run_config.get("position_phase_deembedded"),
        "physical_rotation_backrotated": run_config.get("physical_rotation_backrotated"),
        "circular_components_precomputed": run_config.get("circular_components_precomputed"),
    }
    bad = {k: v for k, v in forbidden_true.items() if v is not False}
    if bad:
        raise ValueError(f"Forbidden preprocessing flags are not false: {bad}")
    if str(run_config.get("coordinate_system", "")).strip() != "HFSS exported global spherical basis":
        raise ValueError(
            f"Unexpected coordinate system: {run_config.get('coordinate_system')!r}"
        )

    design_variation = str(run_config.get("design_variation_key", ""))
    variation_values = _validate_baseline_variation(
        design_variation, strict_baseline=strict_baseline
    )
    requested = run_summary.get("requested_global_indices", [])
    successful = run_summary.get("successful_global_indices", [])
    if strict_baseline:
        expected_indices = list(range(EXPECTED_PORT_COUNT))
        if requested != expected_indices or successful != expected_indices:
            raise ValueError("run_summary requested/successful global indices are not exactly 0..186")
        if run_summary.get("failed_global_indices") not in ([], None):
            raise ValueError("run_summary contains failed global indices")

    return run_config, run_summary, variation_values


def load_and_validate_manifest(
    run_root: Path,
    *,
    expected_port_count: int = EXPECTED_PORT_COUNT,
) -> pd.DataFrame:
    path = Path(run_root) / "vecaep_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Vec-AEP manifest not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True)
    df = _normalize_column_names(df)
    required = {
        "global_index",
        "unit_name",
        "port_name",
        "source_name",
        "ring_index",
        "ring_local_index",
        "x_mm",
        "y_mm",
        "rotation_deg",
        "excitation_magnitude_W",
        "excitation_phase_deg",
        "status",
        "vecaep_file",
        "vecaep_size_bytes",
        "metadata_file",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Manifest missing columns {missing}: {path}")
    df = df.sort_values("global_index", kind="mergesort").reset_index(drop=True)
    if len(df) != expected_port_count:
        raise ValueError(f"Manifest row count={len(df)}, expected={expected_port_count}")
    expected_indices = list(range(expected_port_count))
    if df["global_index"].astype(int).tolist() != expected_indices:
        raise ValueError("Manifest global_index is not exactly continuous 0..186")
    if df["global_index"].duplicated().any():
        raise ValueError("Manifest contains duplicate global_index values")

    expected_ports = [f"p{i:03d}" for i in range(1, expected_port_count + 1)]
    expected_units = [f"U{i:03d}" for i in range(expected_port_count)]
    if df["port_name"].astype(str).tolist() != expected_ports:
        raise ValueError("Manifest port_name order is not p001..p187")
    if df["unit_name"].astype(str).tolist() != expected_units:
        raise ValueError("Manifest unit_name order is not U000..U186")
    statuses = df["status"].astype(str).str.strip().str.lower()
    if not bool((statuses == "passed").all()):
        bad = df.loc[statuses != "passed", ["global_index", "port_name", "status"]]
        raise ValueError(f"Manifest contains non-passed rows:\n{bad.to_string(index=False)}")
    if not np.allclose(
        pd.to_numeric(df["excitation_magnitude_W"], errors="raise"),
        1.0,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Manifest excitation_magnitude_W is not exactly 1 W")
    if not np.allclose(
        pd.to_numeric(df["excitation_phase_deg"], errors="raise"),
        0.0,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Manifest excitation_phase_deg is not 0 deg")
    return df


def load_and_validate_element_plan(run_root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    path = Path(run_root) / "array_element_plan_snapshot.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Array element plan snapshot not found: {path}")
    plan = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True)
    plan = _normalize_column_names(plan)
    required = {
        "global_index",
        "unit_name",
        "ring_index",
        "ring_local_index",
        "x_mm",
        "y_mm",
        "rotation_deg",
        "port_name",
    }
    missing = sorted(required.difference(plan.columns))
    if missing:
        raise KeyError(f"Element plan missing columns {missing}: {path}")
    plan = plan.sort_values("global_index", kind="mergesort").reset_index(drop=True)
    if plan["global_index"].astype(int).tolist() != list(range(len(manifest))):
        raise ValueError("Element plan global_index is not exactly continuous")

    text_columns = ["unit_name", "port_name"]
    integer_columns = ["global_index", "ring_index", "ring_local_index"]
    float_columns = ["x_mm", "y_mm", "rotation_deg"]
    for col in text_columns:
        if plan[col].astype(str).tolist() != manifest[col].astype(str).tolist():
            raise ValueError(f"Manifest/plan mismatch in {col}")
    for col in integer_columns:
        if not np.array_equal(
            pd.to_numeric(plan[col], errors="raise").to_numpy(dtype=np.int64),
            pd.to_numeric(manifest[col], errors="raise").to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"Manifest/plan mismatch in {col}")
    for col in float_columns:
        if not np.allclose(
            pd.to_numeric(plan[col], errors="raise").to_numpy(dtype=np.float64),
            pd.to_numeric(manifest[col], errors="raise").to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(f"Manifest/plan mismatch in {col}")
    return plan


def resolve_vecaep_csv_path(run_root: Path, row: Mapping[str, Any]) -> Path:
    raw = str(row.get("vecaep_file", "")).strip()
    if raw:
        original = Path(raw)
        if original.is_file():
            return original.resolve()
    global_index = int(row["global_index"])
    port_name = str(row["port_name"])
    unit_name = str(row["unit_name"])
    label = f"g{global_index:03d}_{port_name}_{unit_name}"
    rebuilt = Path(run_root) / label / f"vecaep_{label}.csv"
    if rebuilt.is_file():
        return rebuilt.resolve()
    raise FileNotFoundError(
        "Vec-AEP CSV not found. "
        f"global_index={global_index}, port={port_name}, original={raw!r}, rebuilt={rebuilt}"
    )


def resolve_port_metadata_path(run_root: Path, row: Mapping[str, Any]) -> Path:
    raw = str(row.get("metadata_file", "")).strip()
    if raw:
        original = Path(raw)
        if original.is_file():
            return original.resolve()
    label = f"g{int(row['global_index']):03d}_{row['port_name']}_{row['unit_name']}"
    rebuilt = Path(run_root) / label / "metadata.json"
    if rebuilt.is_file():
        return rebuilt.resolve()
    raise FileNotFoundError(f"Port metadata not found: {rebuilt}")


def validate_port_metadata(metadata_path: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    data = read_json(metadata_path)
    expected = {
        "global_index": int(row["global_index"]),
        "unit_name": str(row["unit_name"]),
        "port_name": str(row["port_name"]),
        "source_name": str(row["source_name"]),
    }
    if data.get("ok") is not True:
        raise ValueError(f"Port metadata does not report ok=true: {metadata_path}")
    for key, expected_value in expected.items():
        if data.get(key) != expected_value:
            raise ValueError(
                f"Port metadata mismatch in {key}: actual={data.get(key)!r}, "
                f"expected={expected_value!r}, file={metadata_path}"
            )
    _close_scalar(data.get("excitation_magnitude_W"), 1.0, 1e-15, "port excitation W")
    _close_scalar(data.get("excitation_phase_deg"), 0.0, 1e-15, "port excitation phase")
    if int(data.get("inactive_source_count", -1)) != EXPECTED_PORT_COUNT - 1:
        raise ValueError(f"Port inactive_source_count is not 186: {metadata_path}")
    return data


def _source_inventory_digest(paths: Sequence[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted((Path(p) for p in paths), key=lambda p: str(p).lower()):
        stat = path.stat()
        record = f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8")
        h.update(record)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Cache construction and loading
# -----------------------------------------------------------------------------
def _cache_dir_for_run(cache_root: Path, run_root: Path) -> Path:
    return Path(cache_root) / f"vecaep_{Path(run_root).name}"


def _required_cache_files(cache_dir: Path, include_circular: bool) -> list[Path]:
    files = [
        cache_dir / "theta_axis_rad.npy",
        cache_dir / "phi_axis_rad.npy",
        cache_dir / "theta_points_rad.npy",
        cache_dir / "phi_points_rad.npy",
        cache_dir / "etheta_matrix.npy",
        cache_dir / "ephi_matrix.npy",
        cache_dir / "element_table.csv",
        cache_dir / "run_config.json",
        cache_dir / "run_summary.json",
        cache_dir / "cache_metadata.json",
        cache_dir / CACHE_COMPLETE_NAME,
    ]
    if include_circular:
        files.extend([cache_dir / "e_lhcp_matrix.npy", cache_dir / "e_rhcp_matrix.npy"])
    return files


def _load_cache_metadata(cache_dir: Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    incomplete = cache_dir / CACHE_INCOMPLETE_NAME
    complete = cache_dir / CACHE_COMPLETE_NAME
    if incomplete.exists():
        raise ValueError(f"Cache build is incomplete: {incomplete}")
    if not complete.is_file():
        raise ValueError(f"Cache completion marker is missing: {complete}")
    completion = read_json(complete)
    if completion.get("status") != "complete":
        raise ValueError(f"Invalid cache completion marker: {completion}")
    metadata = read_json(cache_dir / "cache_metadata.json")
    if metadata.get("schema_name") != SCHEMA_NAME:
        raise ValueError(f"Unknown cache schema: {metadata.get('schema_name')!r}")
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported cache schema version: {metadata.get('schema_version')!r}")
    return metadata


def build_vecaep_cache(
    vecaep_root: Path,
    cache_root: Path,
    *,
    cp_basis_sign: int = DEFAULT_CP_BASIS_SIGN,
    variation_selected: str | None = None,
    overwrite: bool = False,
    strict_baseline: bool = True,
    cache_circular_matrices: bool = True,
    progress_step: int = 10,
    logger: logging.Logger | None = None,
) -> Path:
    """Build a verified point-by-port Vec-AEP cache from all 187 raw CSV files.

    Version 1.0.2 deliberately writes directly into the final cache directory.
    It does not rename a populated temporary directory, because some Windows
    file systems, antivirus filters, and PyCharm/Explorer indexers can reject
    that directory rename with WinError 5 even after Python file handles are
    closed. A build-incomplete sentinel prevents partial caches from being used.
    """

    if int(cp_basis_sign) not in {-1, 1}:
        raise ValueError("cp_basis_sign must be -1 or +1")
    if progress_step <= 0:
        raise ValueError("progress_step must be positive")
    logger = logger or setup_logger(None)

    run_root = locate_vecaep_run(vecaep_root)
    cache_root = Path(cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = _cache_dir_for_run(cache_root, run_root)

    run_config, run_summary, variation_values = load_and_validate_run_metadata(
        run_root, strict_baseline=strict_baseline
    )
    manifest = load_and_validate_manifest(run_root)
    plan = load_and_validate_element_plan(run_root, manifest)

    csv_paths = [resolve_vecaep_csv_path(run_root, row) for row in manifest.to_dict("records")]
    metadata_paths = [
        resolve_port_metadata_path(run_root, row) for row in manifest.to_dict("records")
    ]
    source_inventory_digest = _source_inventory_digest(csv_paths + metadata_paths)

    if cache_dir.exists() and not overwrite:
        try:
            metadata = _load_cache_metadata(cache_dir)
            required = _required_cache_files(
                cache_dir, include_circular=bool(metadata.get("circular_matrices_cached"))
            )
            if all(p.is_file() for p in required):
                if metadata.get("source_inventory_digest") == source_inventory_digest:
                    if int(metadata.get("cp_basis_sign")) == int(cp_basis_sign):
                        logger.info("[CACHE] Existing compatible cache found: %s", cache_dir)
                        return cache_dir
        except Exception as exc:
            logger.warning("[CACHE] Existing cache is unusable: %s", exc)
        raise FileExistsError(
            f"Cache exists but is not compatible: {cache_dir}. Use --rebuild-cache."
        )

    # Direct-final-directory strategy for Windows reliability.
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=False)
    incomplete_marker = cache_dir / CACHE_INCOMPLETE_NAME
    incomplete_marker.write_text(
        json.dumps(
            {
                "status": "building",
                "script_version": SCRIPT_VERSION,
                "pid": os.getpid(),
                "started_unix_time": time.time(),
                "source_run_root": str(run_root),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("[VERSION] stage3_vecaep_io.py %s", SCRIPT_VERSION)
    logger.info("[CACHE] direct-final-directory mode enabled; no directory rename will be used")
    logger.info("[RUN] Vec-AEP run root: %s", run_root)
    logger.info("[PASS] run_summary ok=true")
    logger.info(
        "[PASS] formal baseline: ports=%d, frequency=%.6g GHz, Cut_L=%.6g mm",
        len(manifest),
        variation_values["frequency_ghz"],
        variation_values["cut_l_mm"],
    )
    logger.info("[PASS] manifest and element-plan mapping are consistent")

    etheta_matrix: np.ndarray | None = None
    ephi_matrix: np.ndarray | None = None
    reference_grid: HFSSFieldGrid | None = None
    field_sizes: list[int] = []

    try:
        for col, (row, csv_path, metadata_path) in enumerate(
            zip(manifest.to_dict("records"), csv_paths, metadata_paths)
        ):
            validate_port_metadata(metadata_path, row)
            expected_size = int(row["vecaep_size_bytes"])
            actual_size = int(csv_path.stat().st_size)
            if actual_size != expected_size:
                raise ValueError(
                    f"Vec-AEP size mismatch for port {row['port_name']}: "
                    f"manifest={expected_size}, actual={actual_size}, file={csv_path}"
                )
            field = read_hfss_vector_field_csv(
                csv_path,
                variation_selected=variation_selected,
                angle_column_mode="auto",
                angle_unit="rad",
                phase_unit="rad",
                strict_baseline=strict_baseline,
                expected_cut_l_mm=EXPECTED_CUT_L_MM if strict_baseline else None,
                expected_frequency_ghz=EXPECTED_FREQUENCY_GHZ if strict_baseline else None,
            )
            field_sizes.append(actual_size)

            if reference_grid is None:
                reference_grid = field
                point_count = field.etheta.size
                port_count = len(manifest)
                etheta_matrix = np.empty((point_count, port_count), dtype=np.complex128)
                ephi_matrix = np.empty((point_count, port_count), dtype=np.complex128)
                logger.info(
                    "[GRID] mapping=%s, theta=%g:%g:%g deg (%d), phi=%g:%g:%g deg (%d)",
                    field.angle_mapping["mode_used"],
                    field.grid_report["theta_min_deg"],
                    field.grid_report["theta_step_deg"],
                    field.grid_report["theta_max_deg"],
                    field.grid_report["n_theta"],
                    field.grid_report["phi_min_deg"],
                    field.grid_report["phi_step_deg"],
                    field.grid_report["phi_max_deg"],
                    field.grid_report["n_phi"],
                )
                logger.info("[GRID] canonical point count: %d", point_count)
            else:
                assert etheta_matrix is not None and ephi_matrix is not None
                for name, actual, expected in [
                    ("theta_axis_rad", field.theta_axis_rad, reference_grid.theta_axis_rad),
                    ("phi_axis_rad", field.phi_axis_rad, reference_grid.phi_axis_rad),
                    ("theta_points_rad", field.theta_points_rad, reference_grid.theta_points_rad),
                    ("phi_points_rad", field.phi_points_rad, reference_grid.phi_points_rad),
                ]:
                    if actual.shape != expected.shape or not np.array_equal(actual, expected):
                        max_diff = (
                            float(np.max(np.abs(actual - expected)))
                            if actual.shape == expected.shape
                            else float("inf")
                        )
                        raise ValueError(
                            f"Grid mismatch in {name} for port {row['port_name']}: max_diff={max_diff}"
                        )
                if field.variation != reference_grid.variation:
                    raise ValueError(
                        f"Variation mismatch for port {row['port_name']}: "
                        f"{field.variation!r} vs {reference_grid.variation!r}"
                    )

            assert etheta_matrix is not None and ephi_matrix is not None
            global_index = int(row["global_index"])
            if global_index != col:
                raise ValueError(
                    f"Internal manifest ordering failure: column={col}, global_index={global_index}"
                )
            etheta_matrix[:, global_index] = field.etheta
            ephi_matrix[:, global_index] = field.ephi

            done = col + 1
            if done == 1 or done % progress_step == 0 or done == len(manifest):
                logger.info("[LOAD] Vec-AEP %d/%d", done, len(manifest))

        assert reference_grid is not None
        assert etheta_matrix is not None and ephi_matrix is not None
        if not np.all(np.isfinite(etheta_matrix)) or not np.all(np.isfinite(ephi_matrix)):
            raise ValueError("Assembled Vec-AEP matrices contain NaN or Inf")

        e_lhcp_matrix: np.ndarray | None = None
        e_rhcp_matrix: np.ndarray | None = None
        if cache_circular_matrices:
            e_lhcp_matrix, e_rhcp_matrix = decompose_circular_components(
                etheta_matrix, ephi_matrix, cp_basis_sign=cp_basis_sign
            )

        np.save(cache_dir / "theta_axis_rad.npy", reference_grid.theta_axis_rad)
        np.save(cache_dir / "phi_axis_rad.npy", reference_grid.phi_axis_rad)
        np.save(cache_dir / "theta_points_rad.npy", reference_grid.theta_points_rad)
        np.save(cache_dir / "phi_points_rad.npy", reference_grid.phi_points_rad)
        np.save(cache_dir / "etheta_matrix.npy", etheta_matrix)
        np.save(cache_dir / "ephi_matrix.npy", ephi_matrix)
        if cache_circular_matrices:
            assert e_lhcp_matrix is not None and e_rhcp_matrix is not None
            np.save(cache_dir / "e_lhcp_matrix.npy", e_lhcp_matrix)
            np.save(cache_dir / "e_rhcp_matrix.npy", e_rhcp_matrix)

        plan.to_csv(cache_dir / "element_table.csv", index=False, encoding="utf-8-sig")
        write_json_atomic(cache_dir / "run_config.json", run_config)
        write_json_atomic(cache_dir / "run_summary.json", run_summary)

        metadata: dict[str, Any] = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "script_version": SCRIPT_VERSION,
            "cache_commit_strategy": "direct_final_directory_with_sentinel",
            "created_unix_time": time.time(),
            "source_run_root": str(run_root),
            "source_manifest_sha256": sha256_file(run_root / "vecaep_manifest.csv"),
            "source_plan_sha256": sha256_file(run_root / "array_element_plan_snapshot.csv"),
            "source_run_config_sha256": sha256_file(run_root / "run_config.json"),
            "source_run_summary_sha256": sha256_file(run_root / "run_summary.json"),
            "source_inventory_digest": source_inventory_digest,
            "source_total_vecaep_bytes": int(sum(field_sizes)),
            "source_variation": reference_grid.variation,
            "port_count": int(etheta_matrix.shape[1]),
            "point_count": int(etheta_matrix.shape[0]),
            "matrix_shape": list(etheta_matrix.shape),
            "matrix_dtype": str(etheta_matrix.dtype),
            "grid_report": reference_grid.grid_report,
            "angle_mapping": reference_grid.angle_mapping,
            "duplicate_report_first_port": reference_grid.duplicate_report,
            "cp_basis_sign": int(cp_basis_sign),
            "circular_matrices_cached": bool(cache_circular_matrices),
            "circular_definition": {
                "LHCP": "(Etheta + j*cp_basis_sign*Ephi)/sqrt(2)",
                "RHCP": "(Etheta - j*cp_basis_sign*Ephi)/sqrt(2)",
                "current_LHCP": "(Etheta - j*Ephi)/sqrt(2)" if cp_basis_sign == -1 else "(Etheta + j*Ephi)/sqrt(2)",
                "current_RHCP": "(Etheta + j*Ephi)/sqrt(2)" if cp_basis_sign == -1 else "(Etheta - j*Ephi)/sqrt(2)",
            },
            "physical_contract": {
                "single_port_excitation_w": 1.0,
                "single_port_phase_deg": 0.0,
                "coordinate_system": "HFSS exported global spherical basis",
                "normalization_applied": False,
                "position_phase_deembedded": False,
                "physical_rotation_backrotated": False,
                "spin_compensation_applied": False,
                "spectral_decomposition_applied": False,
                "synthesis_rule": "field = point_by_port_matrix @ complex_port_weights",
            },
            "baseline": {
                "frequency_ghz": variation_values["frequency_ghz"],
                "cut_l_mm": variation_values["cut_l_mm"],
                "strict_baseline": bool(strict_baseline),
            },
        }
        write_json_atomic(cache_dir / "cache_metadata.json", metadata)

        # Verify each file sequentially without mmap. No directory rename follows.
        theta_verify = np.load(cache_dir / "theta_axis_rad.npy", allow_pickle=False)
        phi_verify = np.load(cache_dir / "phi_axis_rad.npy", allow_pickle=False)
        if theta_verify.shape != reference_grid.theta_axis_rad.shape:
            raise ValueError("Written theta-axis cache shape is wrong")
        if phi_verify.shape != reference_grid.phi_axis_rad.shape:
            raise ValueError("Written phi-axis cache shape is wrong")
        del theta_verify, phi_verify

        et_verify = np.load(cache_dir / "etheta_matrix.npy", allow_pickle=False)
        if strict_baseline and et_verify.shape != (EXPECTED_POINT_COUNT, EXPECTED_PORT_COUNT):
            raise ValueError(f"Written Etheta cache shape is wrong: {et_verify.shape}")
        expected_shape = et_verify.shape
        if et_verify.dtype != np.complex128:
            raise ValueError(f"Written Etheta cache dtype is wrong: {et_verify.dtype}")
        del et_verify

        ep_verify = np.load(cache_dir / "ephi_matrix.npy", allow_pickle=False)
        if ep_verify.shape != expected_shape or ep_verify.dtype != np.complex128:
            raise ValueError("Written Ephi cache shape or dtype is wrong")
        del ep_verify

        if cache_circular_matrices:
            el_verify = np.load(cache_dir / "e_lhcp_matrix.npy", allow_pickle=False)
            er_verify = np.load(cache_dir / "e_rhcp_matrix.npy", allow_pickle=False)
            if el_verify.shape != expected_shape or el_verify.dtype != np.complex128:
                raise ValueError("Written LHCP cache shape or dtype is wrong")
            if er_verify.shape != expected_shape or er_verify.dtype != np.complex128:
                raise ValueError("Written RHCP cache shape or dtype is wrong")
            del el_verify, er_verify

        # Completion marker is written last. The loader requires it.
        write_json_atomic(
            cache_dir / CACHE_COMPLETE_NAME,
            {
                "status": "complete",
                "script_version": SCRIPT_VERSION,
                "completed_unix_time": time.time(),
                "matrix_shape": list(expected_shape),
                "cp_basis_sign": int(cp_basis_sign),
            },
        )
        incomplete_marker.unlink(missing_ok=True)
        _load_cache_metadata(cache_dir)

        logger.info("[PASS] Etheta matrix: %s, %s", etheta_matrix.shape, etheta_matrix.dtype)
        logger.info("[PASS] Ephi matrix:   %s, %s", ephi_matrix.shape, ephi_matrix.dtype)
        logger.info("[PASS] CP basis sign: %d", cp_basis_sign)
        logger.info("[PASS] no normalization, de-embedding, back-rotation, or spin compensation")
        logger.info("[PASS] cache committed without directory rename: %s", cache_dir)
        return cache_dir
    except Exception:
        # Keep a clear incomplete marker if cleanup is blocked by an external process.
        try:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        except Exception:
            pass
        raise


def load_vecaep_dataset(
    cache_dir: Path,
    *,
    mmap_mode: str | None = "r",
    load_circular: bool = True,
) -> VecAEPDataset:
    cache_dir = Path(cache_dir).expanduser().resolve()
    metadata = _load_cache_metadata(cache_dir)
    circular_cached = bool(metadata.get("circular_matrices_cached"))
    required = _required_cache_files(cache_dir, include_circular=circular_cached)
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Cache is incomplete; missing={missing}")

    theta_axis = np.load(cache_dir / "theta_axis_rad.npy", mmap_mode=mmap_mode, allow_pickle=False)
    phi_axis = np.load(cache_dir / "phi_axis_rad.npy", mmap_mode=mmap_mode, allow_pickle=False)
    theta_points = np.load(cache_dir / "theta_points_rad.npy", mmap_mode=mmap_mode, allow_pickle=False)
    phi_points = np.load(cache_dir / "phi_points_rad.npy", mmap_mode=mmap_mode, allow_pickle=False)
    etheta = np.load(cache_dir / "etheta_matrix.npy", mmap_mode=mmap_mode, allow_pickle=False)
    ephi = np.load(cache_dir / "ephi_matrix.npy", mmap_mode=mmap_mode, allow_pickle=False)
    e_lhcp: np.ndarray | None = None
    e_rhcp: np.ndarray | None = None
    if load_circular and circular_cached:
        e_lhcp = np.load(cache_dir / "e_lhcp_matrix.npy", mmap_mode=mmap_mode, allow_pickle=False)
        e_rhcp = np.load(cache_dir / "e_rhcp_matrix.npy", mmap_mode=mmap_mode, allow_pickle=False)

    shape = tuple(int(x) for x in metadata["matrix_shape"])
    if etheta.shape != shape or ephi.shape != shape:
        raise ValueError(
            f"Cache matrix shape mismatch: metadata={shape}, Etheta={etheta.shape}, Ephi={ephi.shape}"
        )
    if etheta.dtype != np.complex128 or ephi.dtype != np.complex128:
        raise ValueError(f"Cache matrices must be complex128: {etheta.dtype}, {ephi.dtype}")
    if e_lhcp is not None and (e_lhcp.shape != shape or e_lhcp.dtype != np.complex128):
        raise ValueError("LHCP cache matrix has wrong shape or dtype")
    if e_rhcp is not None and (e_rhcp.shape != shape or e_rhcp.dtype != np.complex128):
        raise ValueError("RHCP cache matrix has wrong shape or dtype")

    element_table = pd.read_csv(cache_dir / "element_table.csv", encoding="utf-8-sig")
    run_config = read_json(cache_dir / "run_config.json")
    run_summary = read_json(cache_dir / "run_summary.json")
    return VecAEPDataset(
        cache_dir=cache_dir,
        run_root=Path(metadata["source_run_root"]),
        theta_axis_rad=theta_axis,
        phi_axis_rad=phi_axis,
        theta_points_rad=theta_points,
        phi_points_rad=phi_points,
        etheta_matrix=etheta,
        ephi_matrix=ephi,
        e_lhcp_matrix=e_lhcp,
        e_rhcp_matrix=e_rhcp,
        cp_basis_sign=int(metadata["cp_basis_sign"]),
        element_table=element_table,
        run_config=run_config,
        run_summary=run_summary,
        cache_metadata=metadata,
    )


def load_or_build_vecaep_dataset(
    vecaep_root: Path,
    cache_root: Path,
    *,
    rebuild_cache: bool = False,
    cp_basis_sign: int = DEFAULT_CP_BASIS_SIGN,
    strict_baseline: bool = True,
    cache_circular_matrices: bool = True,
    mmap_mode: str | None = "r",
    logger: logging.Logger | None = None,
) -> VecAEPDataset:
    run_root = locate_vecaep_run(vecaep_root)
    cache_dir = build_vecaep_cache(
        run_root,
        cache_root,
        cp_basis_sign=cp_basis_sign,
        overwrite=rebuild_cache,
        strict_baseline=strict_baseline,
        cache_circular_matrices=cache_circular_matrices,
        logger=logger,
    )
    return load_vecaep_dataset(cache_dir, mmap_mode=mmap_mode, load_circular=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate the formal 187-port Stage3 Vec-AEP cache."
    )
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    parser.add_argument("--vecaep-root", type=Path, default=DEFAULT_VECAEP_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--cp-basis-sign", type=int, choices=[-1, 1], default=DEFAULT_CP_BASIS_SIGN)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-circular-cache", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--relaxed-baseline", action="store_true")
    parser.add_argument("--progress-step", type=int, default=10)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cache_root = Path(args.cache_root).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(cache_root / "stage3_vecaep_io.log", args.log_level)
    start = time.perf_counter()
    try:
        run_root = locate_vecaep_run(args.vecaep_root)
        cache_dir = build_vecaep_cache(
            run_root,
            cache_root,
            cp_basis_sign=args.cp_basis_sign,
            overwrite=args.rebuild_cache,
            strict_baseline=not args.relaxed_baseline,
            cache_circular_matrices=not args.no_circular_cache,
            progress_step=args.progress_step,
            logger=logger,
        )
        dataset = load_vecaep_dataset(
            cache_dir,
            mmap_mode=None if args.no_mmap else "r",
            load_circular=True,
        )
        logger.info("[SUMMARY] status=PASS")
        logger.info("[SUMMARY] cache=%s", cache_dir)
        logger.info("[SUMMARY] point_count=%d", dataset.point_count)
        logger.info("[SUMMARY] port_count=%d", dataset.port_count)
        logger.info("[SUMMARY] cp_basis_sign=%d", dataset.cp_basis_sign)
        logger.info("[SUMMARY] duration=%.3f s", time.perf_counter() - start)
        return 0
    except Exception as exc:
        logger.exception("[SUMMARY] status=FAILED error=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
