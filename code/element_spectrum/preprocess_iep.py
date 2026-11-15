#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract rotation-compatible sparse IEP spectra from HFSS far-field CSV data.

The processing includes angular-grid validation, circular-polarization
decomposition, spin-phase compensation, azimuthal Fourier analysis, sparse
modal selection, and complex-field reconstruction checks.

The NPZ/JSON output schema is fixed to ``hfss_sr_sparse`` version 2 for
compatibility with the IEP field model and downstream synthesis algorithms.
"""
import argparse
import hashlib
import json
import logging
import math
import os
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

MIN_PYTHON_VERSION = (3, 6)
if sys.version_info < MIN_PYTHON_VERSION:
    raise RuntimeError("Python 3.6 or newer is required")

_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
# Serialized identifiers and filenames are part of the downstream data contract.
# Renaming this script must not change them.
SCHEMA_NAME = "hfss_sr_sparse"
SCHEMA_VERSION = 2
SPECTRAL_FRAME = "sr_compatible_sparse_v2"
MAIN_NPZ_NAME = "hfss_sr_sparse.npz"
DIAG_NPZ_NAME = "hfss_sr_sparse_diagnostics.npz"
JSON_NAME = "hfss_sr_sparse.json"
LOG_NAME = "hfss_preprocess.log"


def _sha256_file(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def setup_logging(log_path: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("hfss_preprocess")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def _validate_output_targets(paths: Tuple[Path, ...], overwrite: bool) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output file(s) already exist; use --overwrite to replace them: " + ", ".join(existing)
        )


def _atomic_npz_write(path: Path, arrays: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        with np.load(tmp, allow_pickle=False) as z:
            if set(z.files) != set(arrays.keys()):
                raise IOError(f"NPZ verification failed for {path}: key set mismatch")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_json_write(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp, "r", encoding="utf-8") as f:
            json.load(f)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _array_inventory(arrays: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in arrays.items():
        arr = np.asarray(value)
        out[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return out


def read_hfss_csv(
    csv_path: Path,
    variation_selected: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select one HFSS design variation without mixing distinct field exports."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df_all = pd.read_csv(csv_path)
    n_rows_total = int(len(df_all))
    colmap = {str(c).strip(): c for c in df_all.columns}
    if "Variation" not in colmap:
        return df_all.copy(), {
            "n_rows_csv_total": n_rows_total,
            "has_variation_column": False,
            "n_unique_variations": 0,
            "unique_variations": [],
            "variation_selected": None,
            "selection_mode": "no_variation_column",
            "n_rows_selected": n_rows_total,
        }

    vcol = colmap["Variation"]
    vals = [str(v).strip() for v in df_all[vcol].dropna().tolist() if str(v).strip()]
    uniq = list(dict.fromkeys(vals))
    requested = None if variation_selected is None else str(variation_selected).strip()

    if len(uniq) == 0:
        if requested:
            raise ValueError("--variation-selected was supplied, but the Variation column contains no values")
        selected = None
        mode = "empty_variation_column"
        df_selected = df_all.copy()
    elif len(uniq) == 1:
        selected = uniq[0]
        if requested and requested != selected:
            raise ValueError(
                f"requested Variation {requested!r} does not match the only available value {selected!r}"
            )
        mode = "explicit_single_unique" if requested else "single_unique"
        df_selected = df_all[df_all[vcol].astype(str).str.strip() == selected].copy()
    else:
        if not requested:
            raise ValueError(
                "CSV contains multiple Variation values; specify --variation-selected exactly. "
                f"Available values: {uniq}"
            )
        if requested not in uniq:
            raise ValueError(
                f"requested Variation {requested!r} not found. Available values: {uniq}"
            )
        selected = requested
        mode = "explicit_multiple_unique"
        df_selected = df_all[df_all[vcol].astype(str).str.strip() == selected].copy()

    if len(df_selected) == 0:
        raise ValueError(f"no rows remain after Variation selection: {selected!r}")

    return df_selected, {
        "n_rows_csv_total": n_rows_total,
        "has_variation_column": True,
        "n_unique_variations": int(len(uniq)),
        "unique_variations": uniq,
        "variation_selected": selected,
        "selection_mode": mode,
        "n_rows_selected": int(len(df_selected)),
    }

def _is_float_token(s: str) -> bool:
    return bool(_FLOAT_RE.match(str(s).strip()))

def wrap_phi_to_0_2pi(phi_rad: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(phi_rad, dtype=np.float64), 2.0 * math.pi)

def deg2rad(x: np.ndarray) -> np.ndarray:
    return np.deg2rad(x)

def rad2deg(x: np.ndarray) -> np.ndarray:
    return np.rad2deg(x)

def infer_angle_unit_from_raw(theta_raw: np.ndarray, phi_raw: np.ndarray) -> Tuple[str, Dict[str, float]]:
    mx = float(np.nanmax(np.abs(np.concatenate([theta_raw, phi_raw]))))
    margin = 0.2
    inferred = "deg" if mx > (2.0 * math.pi + margin) else "rad"
    return inferred, {"max_abs_angle_raw": mx}

def _angle_min_max_span_deg(x_rad: np.ndarray) -> Tuple[float, float, float]:
    x_deg = rad2deg(np.asarray(x_rad, dtype=np.float64))
    mn = float(np.nanmin(x_deg))
    mx = float(np.nanmax(x_deg))
    return mn, mx, float(mx - mn)

def resolve_angle_column_mapping(theta_col_rad: np.ndarray,
                                 phi_col_rad: np.ndarray,
                                 *,
                                 mode: str = "auto") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Map CSV axes to physical theta/phi, using angular spans in auto mode."""
    mode = str(mode).strip().lower()
    if mode not in ("auto", "standard", "swapped"):
        raise ValueError("angle_column_mode must be 'auto', 'standard', or 'swapped'")

    theta_col_rad = np.asarray(theta_col_rad, dtype=np.float64)
    phi_col_rad = np.asarray(phi_col_rad, dtype=np.float64)
    th_min, th_max, th_span = _angle_min_max_span_deg(theta_col_rad)
    ph_min, ph_max, ph_span = _angle_min_max_span_deg(phi_col_rad)

    ambiguous = False
    if mode == "standard":
        used = "standard"
        reason = "explicit_standard"
        confidence = "explicit"
    elif mode == "swapped":
        used = "swapped"
        reason = "explicit_swapped"
        confidence = "explicit"
    else:
        if th_span > 180.0 and ph_span <= 180.0:
            used = "swapped"
            reason = "Theta column spans a full-circle axis and Phi column is elevation-like"
            confidence = "high"
        elif ph_span > 180.0 and th_span <= 180.0:
            used = "standard"
            reason = "Phi column spans a full-circle axis and Theta column is elevation-like"
            confidence = "high"
        else:
            ambiguous = True
            raise ValueError(
                "angle-column auto mapping is ambiguous: expected exactly one column to span more than 180 deg. "
                f"Theta span={th_span:.6g} deg, Phi span={ph_span:.6g} deg. "
                "Specify --angle-column-mode standard or swapped."
            )

    if used == "swapped":
        theta_phys = phi_col_rad
        phi_phys = theta_col_rad
        theta_from = "Phi(rad)"
        phi_from = "Theta(rad)"
    else:
        theta_phys = theta_col_rad
        phi_phys = phi_col_rad
        theta_from = "Theta(rad)"
        phi_from = "Phi(rad)"

    report = {
        "mode_requested": mode,
        "mode_used": used,
        "physical_theta_from": theta_from,
        "physical_phi_from": phi_from,
        "mapping_confidence": confidence,
        "mapping_reason": reason,
        "mapping_ambiguous": ambiguous,
        "csv_angle_ranges_before_mapping": {
            "Theta(rad)_min_deg": th_min,
            "Theta(rad)_max_deg": th_max,
            "Theta(rad)_span_deg": th_span,
            "Phi(rad)_min_deg": ph_min,
            "Phi(rad)_max_deg": ph_max,
            "Phi(rad)_span_deg": ph_span,
        },
    }
    return np.asarray(theta_phys, dtype=np.float64), np.asarray(phi_phys, dtype=np.float64), report

def infer_phase_unit_from_raw(ang_raw: np.ndarray) -> Tuple[str, Dict[str, float]]:
    mx = float(np.nanmax(np.abs(ang_raw)))
    margin = 0.2
    inferred = "deg" if mx > (2.0 * math.pi + margin) else "rad"
    return inferred, {"max_abs_phase_raw": mx}

def require_unit_match(name: str, specified: str, inferred: str, stats: Dict[str, float]) -> None:
    if specified not in ("rad", "deg"):
        raise ValueError(f"{name} must be 'rad' or 'deg'. got {specified!r}")
    if specified != inferred:
        raise ValueError(
            f"[UNIT MISMATCH] {name}: specified={specified!r} but inferred={inferred!r}. stats={stats}. "
            f"Fix your export units OR change the script parameters."
        )

def parse_mag_ang_column(col: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Parse HFSS magnitude-angle cells into separate real-valued arrays."""
    mags = np.empty(len(col), dtype=float)
    angs = np.empty(len(col), dtype=float)
    for i, v in enumerate(col.astype(str).values):
        parts = re.split(r"[\s,]+", v.strip())
        parts = [p for p in parts if p]
        if len(parts) < 2 or (not _is_float_token(parts[0])) or (not _is_float_token(parts[1])):
            raise ValueError(f"Cannot parse mag/ang from value: {v!r}")
        mags[i] = float(parts[0])
        angs[i] = float(parts[1])
    return mags, angs

def complex_from_mag_ang(mag: np.ndarray, ang: np.ndarray, phase_unit: str) -> np.ndarray:
    if phase_unit not in ("rad", "deg"):
        raise ValueError("phase_unit must be 'rad' or 'deg'")
    ang_rad = ang if phase_unit == "rad" else np.deg2rad(ang)
    return mag * np.exp(1j * ang_rad)

def _quantize_to_int(values: np.ndarray, tol: float) -> np.ndarray:
    if tol <= 0:
        raise ValueError("tol must be > 0")
    return np.round(np.asarray(values, dtype=np.float64) / tol).astype(np.int64)

def build_grid_from_quantized(qth: np.ndarray, qph: np.ndarray, angle_tol_rad: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uqth = np.unique(qth)
    uqph = np.unique(qph)
    uqth.sort()
    uqph.sort()
    theta_grid = uqth.astype(np.float64) * angle_tol_rad
    phi_grid = uqph.astype(np.float64) * angle_tol_rad
    return theta_grid, phi_grid, uqth, uqph

def dedup_and_validate_quantized_pairs(
    df: pd.DataFrame,
    theta_rad_col: str,
    phi_rad_col: str,
    angle_tol_rad: float,
    *,
    rel_tolerance: float = 1e-8,
    abs_floor_fraction: float = 1e-12,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Merge numerically identical angular samples only when fields agree."""
    qth = _quantize_to_int(df[theta_rad_col].to_numpy(dtype=float), angle_tol_rad)
    qph = _quantize_to_int(df[phi_rad_col].to_numpy(dtype=float), angle_tol_rad)
    df2 = df.copy()
    df2["_qth"] = qth
    df2["_qph"] = qph
    df2 = df2.sort_values(["_qth", "_qph"], kind="mergesort")

    peak_t = float(np.max(np.abs(df2["Etheta"].to_numpy(dtype=np.complex128))))
    peak_p = float(np.max(np.abs(df2["Ephi"].to_numpy(dtype=np.complex128))))
    floor_t = max(peak_t * abs_floor_fraction, 1e-30)
    floor_p = max(peak_p * abs_floor_fraction, 1e-30)

    duplicate_mask = df2.duplicated(subset=["_qth", "_qph"], keep=False)
    if not bool(duplicate_mask.any()):
        report = {
            "duplicate_group_count": 0,
            "duplicate_row_count": 0,
            "max_abs_diff_Etheta": 0.0,
            "max_abs_diff_Ephi": 0.0,
            "max_rel_diff_Etheta": 0.0,
            "max_rel_diff_Ephi": 0.0,
            "relative_tolerance": float(rel_tolerance),
            "consistency_passed": True,
            "merge_method": "complex_mean",
        }
        return df2.reset_index(drop=True), report

    rows = []
    duplicate_group_count = 0
    duplicate_row_count = 0
    max_abs_t = 0.0
    max_abs_p = 0.0
    max_rel_t = 0.0
    max_rel_p = 0.0

    for (_, _), grp in df2.groupby(["_qth", "_qph"], sort=True):
        et = grp["Etheta"].to_numpy(dtype=np.complex128)
        ep = grp["Ephi"].to_numpy(dtype=np.complex128)
        mean_t = complex(np.mean(et))
        mean_p = complex(np.mean(ep))
        if len(grp) > 1:
            duplicate_group_count += 1
            duplicate_row_count += int(len(grp) - 1)
            abs_t = float(np.max(np.abs(et - mean_t)))
            abs_p = float(np.max(np.abs(ep - mean_p)))
            rel_t = float(np.max(np.abs(et - mean_t) / max(abs(mean_t), floor_t)))
            rel_p = float(np.max(np.abs(ep - mean_p) / max(abs(mean_p), floor_p)))
            max_abs_t = max(max_abs_t, abs_t)
            max_abs_p = max(max_abs_p, abs_p)
            max_rel_t = max(max_rel_t, rel_t)
            max_rel_p = max(max_rel_p, rel_p)
        first = grp.iloc[0]
        rows.append({
            "theta_rad": float(np.real(first[theta_rad_col])),
            "phi_rad": float(np.real(first[phi_rad_col])),
            "Etheta": mean_t,
            "Ephi": mean_p,
            "_qth": int(np.real(first["_qth"])),
            "_qph": int(np.real(first["_qph"])),
        })

    consistency_passed = max(max_rel_t, max_rel_p) <= rel_tolerance
    if not consistency_passed:
        raise ValueError(
            "duplicate angle rows are inconsistent: "
            f"max_rel_diff_Etheta={max_rel_t:.6g}, max_rel_diff_Ephi={max_rel_p:.6g}, "
            f"tolerance={rel_tolerance:.6g}"
        )

    out = pd.DataFrame(rows).sort_values(["_qth", "_qph"], kind="mergesort").reset_index(drop=True)
    report = {
        "duplicate_group_count": int(duplicate_group_count),
        "duplicate_row_count": int(duplicate_row_count),
        "max_abs_diff_Etheta": float(max_abs_t),
        "max_abs_diff_Ephi": float(max_abs_p),
        "max_rel_diff_Etheta": float(max_rel_t),
        "max_rel_diff_Ephi": float(max_rel_p),
        "relative_tolerance": float(rel_tolerance),
        "consistency_passed": bool(consistency_passed),
        "merge_method": "complex_mean",
    }
    return out, report


def validate_angular_grid(
    theta_grid_rad: np.ndarray,
    phi_grid_rad: np.ndarray,
    *,
    required_theta_max_deg: float = 90.0,
    step_rtol: float = 1e-8,
    step_atol_rad: float = 1e-10,
) -> Dict[str, Any]:
    """Require a complete theta grid and one uniform, periodic phi revolution."""
    theta = np.asarray(theta_grid_rad, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi_grid_rad, dtype=np.float64).reshape(-1)
    if theta.size < 2 or phi.size < 3:
        raise ValueError("angular grid is too small")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(phi)):
        raise ValueError("angular grid contains NaN or Inf")
    if np.any(np.diff(theta) <= 0.0) or np.any(np.diff(phi) <= 0.0):
        raise ValueError("theta and phi grids must be strictly increasing after deduplication")

    theta_deg = rad2deg(theta)
    if float(theta_deg[0]) > 1e-6 or float(theta_deg[-1]) + 1e-8 < float(required_theta_max_deg):
        raise ValueError(
            f"theta grid must start at 0 deg and cover at least {required_theta_max_deg:g} deg; "
            f"actual range=[{theta_deg[0]:.9g},{theta_deg[-1]:.9g}] deg"
        )

    dtheta = np.diff(theta)
    dphi = np.diff(phi)
    wrap_step = 2.0 * math.pi - float(phi[-1]) + float(phi[0])
    all_phi_steps = np.concatenate([dphi, np.array([wrap_step], dtype=np.float64)])
    phi_step = float(np.median(all_phi_steps))
    uniform = bool(np.allclose(all_phi_steps, phi_step, rtol=step_rtol, atol=step_atol_rad))
    period_error = abs(float(phi.size) * phi_step - 2.0 * math.pi)
    full_period = period_error <= max(step_atol_rad * phi.size, step_rtol * 2.0 * math.pi)
    if not uniform:
        raise ValueError(
            "phi grid is not uniformly sampled over the periodic wrap: "
            f"min_step={rad2deg(np.min(all_phi_steps)):.9g} deg, "
            f"max_step={rad2deg(np.max(all_phi_steps)):.9g} deg"
        )
    if not full_period:
        raise ValueError(
            "phi grid does not cover exactly one full period: "
            f"Nphi*step error={rad2deg(period_error):.9g} deg"
        )

    return {
        "n_theta": int(theta.size),
        "n_phi": int(phi.size),
        "theta_min_deg": float(theta_deg[0]),
        "theta_max_deg": float(theta_deg[-1]),
        "theta_step_min_deg": float(rad2deg(np.min(dtheta))),
        "theta_step_median_deg": float(rad2deg(np.median(dtheta))),
        "theta_step_max_deg": float(rad2deg(np.max(dtheta))),
        "phi_min_deg": float(rad2deg(phi[0])),
        "phi_max_deg": float(rad2deg(phi[-1])),
        "phi_step_deg": float(rad2deg(phi_step)),
        "phi_step_max_deviation_deg": float(rad2deg(np.max(np.abs(all_phi_steps - phi_step)))),
        "periodic_span_error_deg": float(rad2deg(period_error)),
        "is_uniform_phi": uniform,
        "is_full_period_phi": full_period,
        "required_theta_max_deg": float(required_theta_max_deg),
    }

def fill_matrix_from_quantized(df_dedup_sorted: pd.DataFrame, n_th: int, n_ph: int, uqth: np.ndarray, uqph: np.ndarray, value_col: str) -> np.ndarray:
    qth = df_dedup_sorted["_qth"].to_numpy(dtype=np.int64)
    qph = df_dedup_sorted["_qph"].to_numpy(dtype=np.int64)
    vals = df_dedup_sorted[value_col].to_numpy()
    ri = np.searchsorted(uqth, qth)
    ci = np.searchsorted(uqph, qph)
    mat = np.empty((n_th, n_ph), dtype=vals.dtype)
    mat[ri, ci] = vals
    return mat

def _co_handedness_to_sign(co_handedness: str) -> int:
    h = str(co_handedness).strip().upper()
    if h == "LHCP":
        return +1
    if h == "RHCP":
        return -1
    raise ValueError(f"co_handedness must be 'LHCP' or 'RHCP', got {co_handedness!r}")

def decompose_circular(Eth: np.ndarray, Eph: np.ndarray, sign: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the explicit CP-basis sign associated with the HFSS field export."""
    ELH = (Eth + 1j * sign * Eph) / math.sqrt(2.0)
    ERH = (Eth - 1j * sign * Eph) / math.sqrt(2.0)
    return ELH, ERH

def route_cp_to_co_cx(E_LHCP: np.ndarray, E_RHCP: np.ndarray, co_handedness: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    co_h = str(co_handedness).strip().upper()
    if co_h == "LHCP":
        E_co_raw = np.asarray(E_LHCP, dtype=np.complex128)
        E_cx_raw = np.asarray(E_RHCP, dtype=np.complex128)
        cx_h = "RHCP"
    elif co_h == "RHCP":
        E_co_raw = np.asarray(E_RHCP, dtype=np.complex128)
        E_cx_raw = np.asarray(E_LHCP, dtype=np.complex128)
        cx_h = "LHCP"
    else:
        raise ValueError(f"co_handedness must be 'LHCP' or 'RHCP', got {co_handedness!r}")
    if E_co_raw.shape != E_cx_raw.shape:
        raise ValueError("Routed E_co_raw/E_cx_raw shape mismatch.")
    s = _co_handedness_to_sign(co_h)
    report = {
        "co_handedness": co_h,
        "cx_handedness": cx_h,
        "s": int(s),
        "routing_rule": f"{co_h}->co, {cx_h}->cx",
    }
    return E_co_raw, E_cx_raw, report

def compute_global_amp_norm_Aref_from_co(theta_grid_rad: np.ndarray, E_co_raw: np.ndarray, theta_ref_deg: float) -> Tuple[float, int]:
    theta_deg = rad2deg(np.asarray(theta_grid_rad, dtype=np.float64))
    idx = int(np.argmin(np.abs(theta_deg - theta_ref_deg)))
    row = np.abs(np.asarray(E_co_raw, dtype=np.complex128)[idx, :])
    A_ref = float(np.median(row))
    return A_ref, idx

def apply_co_spin_compensation(E_co: np.ndarray, E_cx: np.ndarray, phi_grid_rad: np.ndarray, co_handedness: str) -> Tuple[np.ndarray, np.ndarray]:
    """Remove the azimuthal spin phase before Fourier-mode extraction.

    The spin sign is +1 for LHCP and -1 for RHCP under this CP convention.
    Co and cx are multiplied by exp(-1j*s*phi) and exp(+1j*s*phi).
    """
    phi = np.asarray(phi_grid_rad, dtype=np.float64).reshape(-1)
    s = _co_handedness_to_sign(co_handedness)
    C_co = np.exp(-1j * float(s) * phi).astype(np.complex128)
    C_cx = np.exp(+1j * float(s) * phi).astype(np.complex128)
    Eco = np.asarray(E_co, dtype=np.complex128)
    Ecx = np.asarray(E_cx, dtype=np.complex128)
    if Eco.ndim != 2 or Ecx.ndim != 2:
        raise ValueError("E_co/E_cx must be 2D (n_theta,n_phi).")
    if Eco.shape != Ecx.shape:
        raise ValueError("E_co/E_cx shape mismatch.")
    if Eco.shape[1] != phi.shape[0]:
        raise ValueError("phi_grid_rad length mismatch with field matrices.")
    return Eco * C_co[None, :], Ecx * C_cx[None, :]

def _build_q_full(n_phi: int) -> np.ndarray:
    n_phi = int(n_phi)
    if n_phi < 3:
        raise ValueError("n_phi too small")
    if (n_phi % 2) == 0:
        qmax_unique = (n_phi // 2) - 1
    else:
        qmax_unique = n_phi // 2
    return np.arange(-qmax_unique, qmax_unique + 1, dtype=np.int64)

def fourier_coeffs_over_phi(E_mat: np.ndarray, phi_grid_rad: np.ndarray, q_list: np.ndarray) -> np.ndarray:
    """Extract complex coefficients of exp(1j*q*phi) on the periodic phi grid."""
    E = np.asarray(E_mat, dtype=np.complex128)
    phi = np.asarray(phi_grid_rad, dtype=np.float64)
    q = np.asarray(q_list, dtype=np.int64)
    exp_mat = np.exp(-1j * q[:, None].astype(np.float64) * phi[None, :]).astype(np.complex128)
    coeff = (E @ exp_mat.T) / float(phi.shape[0])
    return coeff.astype(np.complex128)

def load_hfss_dataframe_to_theta_phi_fields(
    df_selected: pd.DataFrame,
    *,
    csv_path: Path,
    variation_value: Optional[str],
    field_format: str = "mag_ang",
    angle_unit_in: str = "rad",
    phase_unit_in: str = "rad",
    angle_tol_rad: float = 1e-12,
    angle_column_mode: str = "auto",
    required_theta_max_deg: float = 90.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if field_format != "mag_ang":
        raise NotImplementedError("Only field_format='mag_ang' is implemented")
    df = df_selected.copy()
    colmap = {str(c).strip(): c for c in df.columns}
    for required in ["Theta(rad)", "Phi(rad)", "rEtheta(mag ang)", "rEphi(mag ang)"]:
        if required not in colmap:
            raise KeyError(f"Required column missing: {required}. Available={list(colmap.keys())}")

    csv_theta_raw = pd.to_numeric(df[colmap["Theta(rad)"]], errors="raise").to_numpy(dtype=float)
    csv_phi_raw = pd.to_numeric(df[colmap["Phi(rad)"]], errors="raise").to_numpy(dtype=float)
    angle_inferred, angle_stats = infer_angle_unit_from_raw(csv_theta_raw, csv_phi_raw)
    require_unit_match("angle_unit_in", angle_unit_in, angle_inferred, angle_stats)
    csv_theta_rad = csv_theta_raw if angle_unit_in == "rad" else np.deg2rad(csv_theta_raw)
    csv_phi_rad = csv_phi_raw if angle_unit_in == "rad" else np.deg2rad(csv_phi_raw)
    theta_rad, phi_rad_unwrapped, angle_mapping = resolve_angle_column_mapping(
        csv_theta_rad, csv_phi_rad, mode=angle_column_mode
    )
    phi_rad = wrap_phi_to_0_2pi(phi_rad_unwrapped)

    et_mag, et_ang = parse_mag_ang_column(df[colmap["rEtheta(mag ang)"]])
    ep_mag, ep_ang = parse_mag_ang_column(df[colmap["rEphi(mag ang)"]])
    phase_inferred_t, phase_stats_t = infer_phase_unit_from_raw(et_ang)
    phase_inferred_p, phase_stats_p = infer_phase_unit_from_raw(ep_ang)
    require_unit_match("phase_unit_in(theta)", phase_unit_in, phase_inferred_t, phase_stats_t)
    require_unit_match("phase_unit_in(phi)", phase_unit_in, phase_inferred_p, phase_stats_p)
    etheta = complex_from_mag_ang(et_mag, et_ang, phase_unit_in)
    ephi = complex_from_mag_ang(ep_mag, ep_ang, phase_unit_in)
    if not np.all(np.isfinite(etheta)) or not np.all(np.isfinite(ephi)):
        raise ValueError("field data contain NaN or Inf")

    df_work = pd.DataFrame({"theta_rad": theta_rad, "phi_rad": phi_rad, "Etheta": etheta, "Ephi": ephi})
    df_dedup, duplicate_report = dedup_and_validate_quantized_pairs(
        df_work, "theta_rad", "phi_rad", angle_tol_rad
    )
    qth = df_dedup["_qth"].to_numpy(dtype=np.int64)
    qph = df_dedup["_qph"].to_numpy(dtype=np.int64)
    theta_grid_rad, phi_grid_rad, uqth, uqph = build_grid_from_quantized(qth, qph, angle_tol_rad)
    n_th = int(theta_grid_rad.size)
    n_ph = int(phi_grid_rad.size)
    if len(df_dedup) != n_th * n_ph:
        raise ValueError(
            f"CSV does not form a full theta-phi grid after dedup: rows={len(df_dedup)}, "
            f"grid={n_th}x{n_ph}={n_th*n_ph}"
        )

    grid_validation = validate_angular_grid(
        theta_grid_rad, phi_grid_rad, required_theta_max_deg=required_theta_max_deg
    )
    etheta_mat = fill_matrix_from_quantized(df_dedup, n_th, n_ph, uqth, uqph, "Etheta")
    ephi_mat = fill_matrix_from_quantized(df_dedup, n_th, n_ph, uqth, uqph, "Ephi")
    sanity = {
        "input_csv": str(csv_path.resolve()),
        "input_sha256": _sha256_file(csv_path),
        "angle_unit_in": angle_unit_in,
        "phase_unit_in": phase_unit_in,
        "field_format": field_format,
        "phi_wrapped_to_0_2pi": True,
        "variation_selected": variation_value,
        "angle_column_mapping": angle_mapping,
        "duplicate_report": duplicate_report,
        "grid_validation": grid_validation,
        "n_rows_selected_before_dedup": int(len(df)),
        "n_rows_after_dedup": int(len(df_dedup)),
    }
    return {
        "theta_grid_rad": np.asarray(theta_grid_rad, dtype=np.float64),
        "phi_grid_rad": np.asarray(phi_grid_rad, dtype=np.float64),
        "E_theta": np.asarray(etheta_mat, dtype=np.complex128),
        "E_phi": np.asarray(ephi_mat, dtype=np.complex128),
    }, sanity


def build_circular_fields_once(
    data0: Dict[str, Any],
    *,
    cp_basis_sign: int,
    co_handedness: str,
) -> Dict[str, Any]:
    if int(cp_basis_sign) not in (-1, 1):
        raise ValueError("cp_basis_sign must be +1 or -1")
    e_lhcp, e_rhcp = decompose_circular(data0["E_theta"], data0["E_phi"], int(cp_basis_sign))
    e_co_raw, e_cx_raw, route_meta = route_cp_to_co_cx(e_lhcp, e_rhcp, co_handedness)
    return {
        **data0,
        "E_LHCP": np.asarray(e_lhcp, dtype=np.complex128),
        "E_RHCP": np.asarray(e_rhcp, dtype=np.complex128),
        "E_co_unscaled": np.asarray(e_co_raw, dtype=np.complex128),
        "E_cx_unscaled": np.asarray(e_cx_raw, dtype=np.complex128),
        "co_handedness": str(co_handedness).upper(),
        "cx_handedness": "RHCP" if str(co_handedness).upper() == "LHCP" else "LHCP",
        "s": int(_co_handedness_to_sign(co_handedness)),
        "cp_basis_sign": int(cp_basis_sign),
        "route_meta": route_meta,
    }


def build_prepared_fields(data_raw: Dict[str, Any], co_handedness: str) -> Dict[str, Any]:
    """Normalize the CP fields at broadside and apply spin compensation."""
    co_handedness = str(co_handedness).upper()
    if co_handedness != data_raw["co_handedness"]:
        raise ValueError("co_handedness does not match the already routed circular fields")
    e_co_raw = np.asarray(data_raw["E_co_unscaled"], dtype=np.complex128)
    e_cx_raw = np.asarray(data_raw["E_cx_unscaled"], dtype=np.complex128)
    a_ref, idx_norm = compute_global_amp_norm_Aref_from_co(data_raw["theta_grid_rad"], e_co_raw, 0.0)
    if not np.isfinite(a_ref) or a_ref <= 1e-15:
        raise ValueError(f"invalid normalization A_ref={a_ref!r}")
    scale = 1.0 / float(a_ref)
    e_co_raw_n = e_co_raw * scale
    e_cx_raw_n = e_cx_raw * scale
    e_co, e_cx = apply_co_spin_compensation(
        e_co_raw_n, e_cx_raw_n, data_raw["phi_grid_rad"], co_handedness
    )
    for name, arr in [("E_co_raw", e_co_raw_n), ("E_cx_raw", e_cx_raw_n), ("E_co", e_co), ("E_cx", e_cx)]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or Inf after normalization/compensation")
    theta_deg = rad2deg(np.asarray(data_raw["theta_grid_rad"], dtype=np.float64))
    return {
        "theta_grid_rad": np.asarray(data_raw["theta_grid_rad"], dtype=np.float64),
        "phi_grid_rad": np.asarray(data_raw["phi_grid_rad"], dtype=np.float64),
        "E_theta": np.asarray(data_raw["E_theta"], dtype=np.complex128),
        "E_phi": np.asarray(data_raw["E_phi"], dtype=np.complex128),
        "E_LHCP": np.asarray(data_raw["E_LHCP"], dtype=np.complex128),
        "E_RHCP": np.asarray(data_raw["E_RHCP"], dtype=np.complex128),
        "E_co_raw": np.asarray(e_co_raw_n, dtype=np.complex128),
        "E_cx_raw": np.asarray(e_cx_raw_n, dtype=np.complex128),
        "E_co": np.asarray(e_co, dtype=np.complex128),
        "E_cx": np.asarray(e_cx, dtype=np.complex128),
        "co_handedness": co_handedness,
        "cx_handedness": data_raw["cx_handedness"],
        "s": int(data_raw["s"]),
        "route_meta": data_raw["route_meta"],
        "normalization": {
            "method": "median_abs_co_at_nearest_theta_0",
            "A_ref": float(a_ref),
            "scale": float(scale),
            "theta_ref_requested_deg": 0.0,
            "theta_ref_actual_deg": float(theta_deg[idx_norm]),
            "theta_ref_row_index": int(idx_norm),
        },
        "cp_convention": {
            "basis_sign": int(data_raw["cp_basis_sign"]),
            "LHCP_definition": "(Etheta + j*sign*Ephi)/sqrt(2)",
            "RHCP_definition": "(Etheta - j*sign*Ephi)/sqrt(2)",
        },
    }

def extract_full_ab_spectra(prep: Dict[str, Any], qmax: Optional[int] = None) -> Dict[str, Any]:
    q_full = _build_q_full(prep["phi_grid_rad"].size)
    if qmax is not None:
        q_full = q_full[np.abs(q_full) <= int(qmax)]
    a_full = fourier_coeffs_over_phi(prep["E_co"], prep["phi_grid_rad"], q_full)
    b_full = fourier_coeffs_over_phi(prep["E_cx"], prep["phi_grid_rad"], q_full)
    return {"q_full": q_full, "a_full": a_full, "b_full": b_full}

def build_c_from_b(q_full: np.ndarray, b_full: np.ndarray, s: int) -> Dict[str, Any]:
    """Reindex cx modes as c_q = b_(q+2*s) for the common rotation order q.

    With this reindexing, both a_q and c_q acquire exp(-1j*q*chi) under
    the passive physical rotation represented by chi.
    """
    q_to_idx = {int(q): i for i, q in enumerate(np.asarray(q_full, dtype=np.int64).tolist())}
    q_c, cols = [], []
    for q in np.asarray(q_full, dtype=np.int64).tolist():
        src = int(q + 2 * int(s))
        if src in q_to_idx:
            q_c.append(int(q))
            cols.append(q_to_idx[src])
    return {
        "q_c": np.asarray(q_c, dtype=np.int64),
        "c_full": np.asarray(b_full[:, np.asarray(cols, dtype=np.int64)], dtype=np.complex128),
        "definition": f"c_q = b_(q+2s), s={int(s)}",
    }

def _compute_a0_and_A0max(spectra: Dict[str, Any]) -> Tuple[np.ndarray, float, int]:
    q_full = np.asarray(spectra["q_full"], dtype=np.int64)
    idx_q0 = int(np.where(q_full == 0)[0][0])
    a0 = np.asarray(spectra["a_full"], dtype=np.complex128)[:, idx_q0]
    return a0, float(np.max(np.abs(a0))), idx_q0

def build_aq_danger_maps(prep: Dict[str, Any], spectra: Dict[str, Any], a0: np.ndarray, A0_max: float, theta_split_deg: float, tau_near: float, tau_far: float, eps: float = 1e-12) -> Dict[str, Any]:
    """Score co modes against local |a0| or global max|a0| by theta region."""
    theta_deg = rad2deg(np.asarray(prep["theta_grid_rad"], dtype=np.float64))
    abs_a = np.abs(np.asarray(spectra["a_full"], dtype=np.complex128))
    abs_a0 = np.abs(np.asarray(a0, dtype=np.complex128))
    mask_near = theta_deg <= float(theta_split_deg) + 1e-12
    danger = np.empty_like(abs_a, dtype=np.float64)
    danger[mask_near, :] = abs_a[mask_near, :] / (abs_a0[mask_near, None] + eps)
    danger[~mask_near, :] = abs_a[~mask_near, :] / (float(A0_max) + eps)
    tau_map = np.where(mask_near, float(tau_near), float(tau_far)).astype(np.float64)
    return {"theta_deg": theta_deg, "danger_a": danger, "tau_map_a": tau_map, "mask_near_theta": mask_near}

def build_cq_danger_maps(prep: Dict[str, Any], c_spec: Dict[str, Any], A0_max: float, tau_c: float, eps: float = 1e-12) -> Dict[str, Any]:
    """Score cross-polarized modes against the global dominant-mode level."""
    danger = np.abs(np.asarray(c_spec["c_full"], dtype=np.complex128)) / (float(A0_max) + eps)
    tau_map = np.full(danger.shape[0], float(tau_c), dtype=np.float64)
    return {"danger_c": danger, "tau_map_c": tau_map}

def select_union_q_from_danger(q_list: np.ndarray, danger_map: np.ndarray, tau_map: np.ndarray, force_keep_q0: bool = True) -> Dict[str, Any]:
    q_arr = np.asarray(q_list, dtype=np.int64)
    danger = np.asarray(danger_map, dtype=np.float64)
    tau = np.asarray(tau_map, dtype=np.float64).reshape(-1)
    per_theta_mask = danger >= tau[:, None]
    union_mask = np.any(per_theta_mask, axis=0)
    union_before = int(np.sum(union_mask))
    if force_keep_q0:
        union_mask[int(np.where(q_arr == 0)[0][0])] = True
    sel_cols = np.where(union_mask)[0].astype(np.int64)
    return {
        "q_sel": q_arr[sel_cols],
        "sel_cols": sel_cols,
        "per_theta_candidate_counts": np.sum(per_theta_mask, axis=1).astype(np.int64),
        "union_count_before_force_keep": union_before,
        "union_count_after_force_keep": int(sel_cols.size),
    }

def build_keep_mask_from_danger(danger_map_sel: np.ndarray, tau_map: np.ndarray, q_sel: np.ndarray, force_keep_q0: bool = True) -> np.ndarray:
    keep = np.asarray(danger_map_sel, dtype=np.float64) >= np.asarray(tau_map, dtype=np.float64)[:, None]
    if force_keep_q0:
        keep[:, int(np.where(np.asarray(q_sel, dtype=np.int64) == 0)[0][0])] = True
    return keep.astype(bool)

def _reconstruct_sparse_aq(phi: np.ndarray, aq_grid: np.ndarray, aq_q_list: np.ndarray, aq_keep_mask: np.ndarray) -> np.ndarray:
    basis = np.exp(1j * np.asarray(aq_q_list, dtype=np.int64)[None, :].astype(np.float64) * np.asarray(phi, dtype=np.float64)[:, None]).astype(np.complex128)
    return np.asarray((np.asarray(aq_grid, dtype=np.complex128) * np.asarray(aq_keep_mask, dtype=bool)) @ basis.T, dtype=np.complex128)

def _reconstruct_sparse_cq(phi: np.ndarray, cq_grid: np.ndarray, cq_q_list: np.ndarray, cq_keep_mask: np.ndarray, s: int) -> np.ndarray:
    basis = np.exp(1j * (np.asarray(cq_q_list, dtype=np.int64)[None, :].astype(np.float64) + 2.0 * float(s)) * np.asarray(phi, dtype=np.float64)[:, None]).astype(np.complex128)
    return np.asarray((np.asarray(cq_grid, dtype=np.complex128) * np.asarray(cq_keep_mask, dtype=bool)) @ basis.T, dtype=np.complex128)

def build_relative_peak_mask(
    E_ref: np.ndarray,
    theta_grid_rad: np.ndarray,
    theta_max_deg: float,
    floor_db: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    e_ref = np.asarray(E_ref, dtype=np.complex128)
    theta_deg = rad2deg(np.asarray(theta_grid_rad, dtype=np.float64)).reshape(-1)
    if e_ref.ndim != 2 or e_ref.shape[0] != theta_deg.size:
        raise ValueError("E_ref shape is inconsistent with theta_grid_rad")
    if float(theta_deg[-1]) + 1e-8 < float(theta_max_deg):
        raise ValueError(
            f"field data only cover theta={theta_deg[-1]:.6g} deg, below required {theta_max_deg:g} deg"
        )
    theta_limit = float(theta_max_deg)
    theta_mask = (theta_deg < theta_limit) | np.isclose(
        theta_deg, theta_limit, rtol=0.0, atol=1e-8
    )
    domain_mask = np.broadcast_to(theta_mask[:, None], e_ref.shape)
    domain_abs = np.abs(e_ref[domain_mask])
    if domain_abs.size == 0:
        raise ValueError("fit domain contains no points")
    peak = float(np.max(domain_abs))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError(f"reference field peak is invalid in fit domain: {peak!r}")
    floor_linear = float(10.0 ** (float(floor_db) / 20.0))
    floor_absolute = peak * floor_linear
    valid_mask = domain_mask & (np.abs(e_ref) >= floor_absolute)
    n_domain = int(np.sum(domain_mask))
    n_valid = int(np.sum(valid_mask))
    if n_valid == 0:
        raise ValueError("no fit points remain above the relative peak floor")
    return valid_mask, {
        "theta_min_deg": float(theta_deg[0]),
        "theta_max_deg": float(theta_max_deg),
        "floor_db": float(floor_db),
        "floor_linear": floor_linear,
        "floor_absolute": float(floor_absolute),
        "reference_peak": peak,
        "n_points_domain": n_domain,
        "n_points_valid": n_valid,
        "n_points_excluded_by_floor": int(n_domain - n_valid),
        "valid_fraction": float(n_valid / n_domain),
    }


def _complex_error_summary_masked(
    E_ref: np.ndarray,
    E_test: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, float]:
    e_ref = np.asarray(E_ref, dtype=np.complex128)
    e_test = np.asarray(E_test, dtype=np.complex128)
    mask = np.asarray(valid_mask, dtype=bool)
    if e_ref.shape != e_test.shape or e_ref.shape != mask.shape:
        raise ValueError("E_ref, E_test, and valid_mask must have identical shapes")
    d = np.abs(e_test[mask] - e_ref[mask])
    ref_abs = np.abs(e_ref[mask])
    rel = d / ref_abs
    return {
        "relative_rms": float(np.sqrt(np.mean(rel ** 2))),
        "relative_p95": float(np.percentile(rel, 95.0)),
        "relative_max": float(np.max(rel)),
        "absolute_rms": float(np.sqrt(np.mean(d ** 2))),
        "absolute_p95": float(np.percentile(d, 95.0)),
        "absolute_max": float(np.max(d)),
    }


def run_sparse_fit_checks(
    prep: Dict[str, Any],
    aq_q_list: np.ndarray,
    aq_grid: np.ndarray,
    aq_keep_mask: np.ndarray,
    cq_q_list: np.ndarray,
    cq_grid: np.ndarray,
    cq_keep_mask: np.ndarray,
    *,
    co_theta_max_deg: float = 90.0,
    cx_theta_max_deg: float = 65.0,
    fit_error_floor_db: float = -40.0,
) -> Dict[str, Any]:
    """Measure complex reconstruction errors above a relative amplitude floor.

    These statistics describe spectral-fit accuracy; they do not impose an
    additional RMS or maximum-error acceptance threshold.
    """
    eco_sparse = _reconstruct_sparse_aq(prep["phi_grid_rad"], aq_grid, aq_q_list, aq_keep_mask)
    ecx_sparse = _reconstruct_sparse_cq(
        prep["phi_grid_rad"], cq_grid, cq_q_list, cq_keep_mask, prep["s"]
    )
    co_mask, co_mask_report = build_relative_peak_mask(
        prep["E_co"], prep["theta_grid_rad"], co_theta_max_deg, fit_error_floor_db
    )
    cx_mask, cx_mask_report = build_relative_peak_mask(
        prep["E_cx"], prep["theta_grid_rad"], cx_theta_max_deg, fit_error_floor_db
    )
    co_fit = _complex_error_summary_masked(prep["E_co"], eco_sparse, co_mask)
    cx_fit = _complex_error_summary_masked(prep["E_cx"], ecx_sparse, cx_mask)
    return {
        "co_sparse_fit_0to90": {**co_mask_report, **co_fit},
        "cx_sparse_fit_0to65": {**cx_mask_report, **cx_fit},
        "E_co_sparse": eco_sparse,
        "E_cx_sparse": ecx_sparse,
        "co_fit_valid_mask": co_mask,
        "cx_fit_valid_mask": cx_mask,
    }

def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {"shape": list(obj.shape), "real": np.real(obj).tolist(), "imag": np.imag(obj).tolist()}
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def validate_cli_config(args: argparse.Namespace) -> None:
    if args.qmax is not None and int(args.qmax) < 2:
        raise ValueError("--qmax must be at least 2 so that c0=b_(2s) is available")
    if not (0.0 <= args.aq_theta_split_deg <= 90.0):
        raise ValueError("--aq-theta-split-deg must be within [0, 90]")
    for name in ("aq_tau_near", "aq_tau_far", "cq_tau_global"):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if args.fit_error_floor_db >= 0.0:
        raise ValueError("--fit-error-floor-db must be negative")
    if not (0.0 < args.cx_fit_theta_max_deg <= args.co_fit_theta_max_deg <= 90.0):
        raise ValueError("require 0 < cx theta max <= co theta max <= 90")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HFSS preprocess: SR-compatible sparse a_q/c_q output with strict input/QC"
    )
    parser.add_argument("--csv", required=True, help="Explicit HFSS export CSV path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--variation-selected", default=None)
    parser.add_argument("--co-handedness", required=True, choices=["LHCP", "RHCP"])
    parser.add_argument("--cp-basis-sign", type=int, default=-1, choices=[-1, 1])
    parser.add_argument(
        "--angle-column-mode", default="auto", choices=["auto", "standard", "swapped"],
        help="standard: Theta->physical theta, Phi->physical phi; swapped: reverse; auto uses axis spans",
    )
    parser.add_argument("--qmax", type=int, default=None)
    parser.add_argument("--aq-theta-split-deg", type=float, default=65.0)
    parser.add_argument("--aq-tau-near", type=float, default=0.002)
    parser.add_argument("--aq-tau-far", type=float, default=0.001)
    parser.add_argument("--cq-tau-global", type=float, default=0.001)
    parser.add_argument("--fit-error-floor-db", type=float, default=-40.0)
    parser.add_argument("--co-fit-theta-max-deg", type=float, default=90.0)
    parser.add_argument("--cx-fit-theta-max-deg", type=float, default=65.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    diag = parser.add_mutually_exclusive_group()
    diag.add_argument("--save-diagnostics", dest="save_diagnostics", action="store_true")
    diag.add_argument("--no-save-diagnostics", dest="save_diagnostics", action="store_false")
    parser.set_defaults(save_diagnostics=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv).expanduser().resolve()
    main_npz_path = out_dir / MAIN_NPZ_NAME
    diag_npz_path = out_dir / DIAG_NPZ_NAME
    json_path = out_dir / JSON_NAME
    log_path = out_dir / LOG_NAME
    target_paths = [main_npz_path, json_path, log_path]
    if args.save_diagnostics:
        target_paths.append(diag_npz_path)
    _validate_output_targets(tuple(target_paths), args.overwrite)
    logger = setup_logging(log_path, args.log_level)
    start_perf = time.perf_counter()
    start_time = _utc_now_iso()
    warnings_list: List[str] = []

    try:
        validate_cli_config(args)
        logger.info("[INPUT] START")
        logger.info("[INPUT] csv=%s", csv_path)
        logger.info("[INPUT] out_dir=%s", out_dir)
        df_selected, variation_report = read_hfss_csv(csv_path, args.variation_selected)
        logger.info(
            "[VARIATION] mode=%s selected=%s rows=%d/%d",
            variation_report["selection_mode"], variation_report["variation_selected"],
            variation_report["n_rows_selected"], variation_report["n_rows_csv_total"],
        )

        logger.info("[GRID] START")
        data0, sanity = load_hfss_dataframe_to_theta_phi_fields(
            df_selected,
            csv_path=csv_path,
            variation_value=variation_report["variation_selected"],
            field_format="mag_ang",
            angle_unit_in="rad",
            phase_unit_in="rad",
            angle_tol_rad=1e-12,
            angle_column_mode=args.angle_column_mode,
            required_theta_max_deg=args.co_fit_theta_max_deg,
        )
        grid = sanity["grid_validation"]
        logger.info(
            "[GRID] mapping=%s theta=[%.3f,%.3f] deg n_theta=%d phi_step=%.6g deg n_phi=%d",
            sanity["angle_column_mapping"]["mode_used"], grid["theta_min_deg"], grid["theta_max_deg"],
            grid["n_theta"], grid["phi_step_deg"], grid["n_phi"],
        )
        dup = sanity["duplicate_report"]
        logger.info(
            "[GRID] duplicate_groups=%d removed_rows=%d merge=%s",
            dup["duplicate_group_count"], dup["duplicate_row_count"], dup["merge_method"],
        )

        logger.info("[POLARIZATION] START")
        circular = build_circular_fields_once(
            data0, cp_basis_sign=args.cp_basis_sign, co_handedness=args.co_handedness
        )
        prep = build_prepared_fields(circular, args.co_handedness)
        logger.info(
            "[POLARIZATION] co=%s cx=%s spin_sign=%d cp_basis_sign=%d",
            prep["co_handedness"], prep["cx_handedness"], prep["s"], args.cp_basis_sign,
        )
        logger.info(
            "[NORMALIZATION] A_ref=%.12g scale=%.12g theta_ref=%.6g deg",
            prep["normalization"]["A_ref"], prep["normalization"]["scale"],
            prep["normalization"]["theta_ref_actual_deg"],
        )

        logger.info("[SPECTRUM] START")
        spectra = extract_full_ab_spectra(prep, args.qmax)
        c_spec = build_c_from_b(spectra["q_full"], spectra["b_full"], prep["s"])
        a0, a0_max, idx_q0 = _compute_a0_and_A0max(spectra)
        idx_c0_candidates = np.where(np.asarray(c_spec["q_c"], dtype=np.int64) == 0)[0]
        if idx_c0_candidates.size != 1:
            raise ValueError("c0 is unavailable or ambiguous; ensure qmax>=2 and q grid is valid")
        idx_c0 = int(idx_c0_candidates[0])
        c0 = np.asarray(c_spec["c_full"], dtype=np.complex128)[:, idx_c0]
        logger.info(
            "[SPECTRUM] q_full=[%d,%d] count=%d q_c=[%d,%d] count=%d",
            int(np.min(spectra["q_full"])), int(np.max(spectra["q_full"])), int(spectra["q_full"].size),
            int(np.min(c_spec["q_c"])), int(np.max(c_spec["q_c"])), int(c_spec["q_c"].size),
        )

        logger.info("[SPARSE] START")
        aq_danger = build_aq_danger_maps(
            prep, spectra, a0, a0_max, args.aq_theta_split_deg, args.aq_tau_near, args.aq_tau_far
        )
        cq_danger = build_cq_danger_maps(prep, c_spec, a0_max, args.cq_tau_global)
        aq_sel = select_union_q_from_danger(
            spectra["q_full"], aq_danger["danger_a"], aq_danger["tau_map_a"], True
        )
        cq_sel = select_union_q_from_danger(
            c_spec["q_c"], cq_danger["danger_c"], cq_danger["tau_map_c"], True
        )
        aq_q_list = aq_sel["q_sel"]
        cq_q_list = cq_sel["q_sel"]
        aq_grid = spectra["a_full"][:, aq_sel["sel_cols"]]
        cq_grid = c_spec["c_full"][:, cq_sel["sel_cols"]]
        aq_keep_mask = build_keep_mask_from_danger(
            aq_danger["danger_a"][:, aq_sel["sel_cols"]], aq_danger["tau_map_a"], aq_q_list, True
        )
        cq_keep_mask = build_keep_mask_from_danger(
            cq_danger["danger_c"][:, cq_sel["sel_cols"]], cq_danger["tau_map_c"], cq_q_list, True
        )
        logger.info("[SPARSE] aq_q_list=%s keep_density=%.6f", aq_q_list.tolist(), float(np.mean(aq_keep_mask)))
        logger.info("[SPARSE] cq_q_list=%s keep_density=%.6f", cq_q_list.tolist(), float(np.mean(cq_keep_mask)))

        logger.info("[FIT] START")
        fit_checks = run_sparse_fit_checks(
            prep, aq_q_list, aq_grid, aq_keep_mask, cq_q_list, cq_grid, cq_keep_mask,
            co_theta_max_deg=args.co_fit_theta_max_deg,
            cx_theta_max_deg=args.cx_fit_theta_max_deg,
            fit_error_floor_db=args.fit_error_floor_db,
        )
        co_fit = fit_checks["co_sparse_fit_0to90"]
        cx_fit = fit_checks["cx_sparse_fit_0to65"]
        logger.info(
            "[FIT] co theta=0-%.1f deg valid=%.3f rel_rms=%.12g rel_p95=%.12g rel_max=%.12g",
            args.co_fit_theta_max_deg, co_fit["valid_fraction"], co_fit["relative_rms"],
            co_fit["relative_p95"], co_fit["relative_max"],
        )
        logger.info(
            "[FIT] cx theta=0-%.1f deg valid=%.3f rel_rms=%.12g rel_p95=%.12g rel_max=%.12g",
            args.cx_fit_theta_max_deg, cx_fit["valid_fraction"], cx_fit["relative_rms"],
            cx_fit["relative_p95"], cx_fit["relative_max"],
        )
        if co_fit["valid_fraction"] < 0.01:
            warnings_list.append("co fit valid_fraction is below 1%")
        if cx_fit["valid_fraction"] < 0.01:
            warnings_list.append("cx fit valid_fraction is below 1%")

        main_arrays: Dict[str, Any] = {
            "schema_name": np.asarray(SCHEMA_NAME),
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
            "spectral_frame": np.asarray(SPECTRAL_FRAME),
            "co_handedness": np.asarray(prep["co_handedness"]),
            "cx_handedness": np.asarray(prep["cx_handedness"]),
            "spin_sign": np.asarray(prep["s"], dtype=np.int64),
            "cp_basis_sign": np.asarray(args.cp_basis_sign, dtype=np.int64),
            "normalization_A_ref": np.asarray(prep["normalization"]["A_ref"], dtype=np.float64),
            "normalization_scale": np.asarray(prep["normalization"]["scale"], dtype=np.float64),
            "theta_grid_rad": np.asarray(prep["theta_grid_rad"], dtype=np.float64),
            "phi_grid_rad": np.asarray(prep["phi_grid_rad"], dtype=np.float64),
            "aq_q_list": np.asarray(aq_q_list, dtype=np.int64),
            "aq_grid": np.asarray(aq_grid, dtype=np.complex128),
            "aq_keep_mask": np.asarray(aq_keep_mask, dtype=bool),
            "a0": np.asarray(a0, dtype=np.complex128),
            "cq_q_list": np.asarray(cq_q_list, dtype=np.int64),
            "cq_grid": np.asarray(cq_grid, dtype=np.complex128),
            "cq_keep_mask": np.asarray(cq_keep_mask, dtype=bool),
            "c0": np.asarray(c0, dtype=np.complex128),
        }
        diag_arrays: Dict[str, Any] = {
            "E_theta": prep["E_theta"],
            "E_phi": prep["E_phi"],
            "E_LHCP": prep["E_LHCP"],
            "E_RHCP": prep["E_RHCP"],
            "E_co_raw_normalized": prep["E_co_raw"],
            "E_cx_raw_normalized": prep["E_cx_raw"],
            "E_co_spin_compensated": prep["E_co"],
            "E_cx_spin_compensated": prep["E_cx"],
            "E_co_sparse_reconstructed": fit_checks["E_co_sparse"],
            "E_cx_sparse_reconstructed": fit_checks["E_cx_sparse"],
            "co_fit_valid_mask": fit_checks["co_fit_valid_mask"],
            "cx_fit_valid_mask": fit_checks["cx_fit_valid_mask"],
        }

        logger.info("[OUTPUT] writing %s", main_npz_path)
        _atomic_npz_write(main_npz_path, main_arrays)
        if args.save_diagnostics:
            logger.info("[OUTPUT] writing %s", diag_npz_path)
            _atomic_npz_write(diag_npz_path, diag_arrays)

        end_time = _utc_now_iso()
        duration = float(time.perf_counter() - start_perf)
        status = "WARNING" if warnings_list else "PASS"
        report: Dict[str, Any] = {
            "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION, "spectral_frame": SPECTRAL_FRAME},
            "run": {
                "status": status,
                "start_time_utc": start_time,
                "end_time_utc": end_time,
                "duration_seconds": duration,
                "command_line": sys.argv,
                "script_path": str(Path(__file__).resolve()),
                "script_sha256": _sha256_file(Path(__file__).resolve()),
                "python_version": sys.version,
                "platform": platform.platform(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
            },
            "input": {
                "csv_path": str(csv_path),
                "csv_sha256": _sha256_file(csv_path),
                "csv_size_bytes": int(csv_path.stat().st_size),
                **variation_report,
            },
            "resolved_config": {
                "co_handedness": args.co_handedness,
                "cp_basis_sign": args.cp_basis_sign,
                "angle_column_mode": args.angle_column_mode,
                "qmax": args.qmax,
                "aq_theta_split_deg": args.aq_theta_split_deg,
                "aq_tau_near": args.aq_tau_near,
                "aq_tau_far": args.aq_tau_far,
                "cq_tau_global": args.cq_tau_global,
                "fit_error_floor_db": args.fit_error_floor_db,
                "co_fit_theta_max_deg": args.co_fit_theta_max_deg,
                "cx_fit_theta_max_deg": args.cx_fit_theta_max_deg,
                "save_diagnostics": args.save_diagnostics,
            },
            "grid": sanity,
            "polarization": {
                "cp_basis_sign": args.cp_basis_sign,
                "LHCP_definition": "(Etheta + j*sign*Ephi)/sqrt(2)",
                "RHCP_definition": "(Etheta - j*sign*Ephi)/sqrt(2)",
                "co_handedness": prep["co_handedness"],
                "cx_handedness": prep["cx_handedness"],
                "spin_sign": prep["s"],
                "co_spin_compensation": "E_co_tilde = E_co_raw * exp(-j*s*phi)",
                "cx_spin_compensation": "E_cx_tilde = E_cx_raw * exp(+j*s*phi)",
            },
            "normalization": prep["normalization"],
            "spectral_model": {
                "aq_definition": "E_co_tilde = sum_q a_q exp(j q phi)",
                "bq_definition": "E_cx_tilde = sum_m b_m exp(j m phi)",
                "cq_definition": c_spec["definition"],
                "sr_phase_law": "a_q,c_q -> coefficient * exp(-j q chi)",
                "q_full_min": int(np.min(spectra["q_full"])),
                "q_full_max": int(np.max(spectra["q_full"])),
                "q_full_count": int(spectra["q_full"].size),
                "q_c_min": int(np.min(c_spec["q_c"])),
                "q_c_max": int(np.max(c_spec["q_c"])),
                "q_c_count": int(c_spec["q_c"].size),
                "qmax_user": args.qmax,
                "nyquist_policy": "for even Nphi, omit the non-unique Nyquist order",
                "anchors": {"a0_abs_max": float(np.max(np.abs(a0))), "c0_abs_max": float(np.max(np.abs(c0)))},
            },
            "sparse_selection": {
                "aq": {
                    "theta_split_deg": args.aq_theta_split_deg,
                    "tau_near": args.aq_tau_near,
                    "tau_far": args.aq_tau_far,
                    "q_list": aq_q_list.tolist(),
                    "q_count": int(aq_q_list.size),
                    "union_count_before_force_keep": aq_sel["union_count_before_force_keep"],
                    "union_count_after_force_keep": aq_sel["union_count_after_force_keep"],
                    "keep_density": float(np.mean(aq_keep_mask)),
                    "per_theta_keep_min": int(np.min(np.sum(aq_keep_mask, axis=1))),
                    "per_theta_keep_median": float(np.median(np.sum(aq_keep_mask, axis=1))),
                    "per_theta_keep_max": int(np.max(np.sum(aq_keep_mask, axis=1))),
                },
                "cq": {
                    "tau_global": args.cq_tau_global,
                    "q_list": cq_q_list.tolist(),
                    "q_count": int(cq_q_list.size),
                    "union_count_before_force_keep": cq_sel["union_count_before_force_keep"],
                    "union_count_after_force_keep": cq_sel["union_count_after_force_keep"],
                    "keep_density": float(np.mean(cq_keep_mask)),
                    "per_theta_keep_min": int(np.min(np.sum(cq_keep_mask, axis=1))),
                    "per_theta_keep_median": float(np.median(np.sum(cq_keep_mask, axis=1))),
                    "per_theta_keep_max": int(np.max(np.sum(cq_keep_mask, axis=1))),
                },
            },
            "fit_checks": {
                "co_0to90": co_fit,
                "cx_0to65": cx_fit,
            },
            "quality_control": {"status": status, "warnings": warnings_list},
            "outputs": {
                "main_npz": {
                    "path": str(main_npz_path),
                    "size_bytes": int(main_npz_path.stat().st_size),
                    "sha256": _sha256_file(main_npz_path),
                    "array_inventory": _array_inventory(main_arrays),
                },
                "diagnostics_npz": None if not args.save_diagnostics else {
                    "path": str(diag_npz_path),
                    "size_bytes": int(diag_npz_path.stat().st_size),
                    "sha256": _sha256_file(diag_npz_path),
                    "array_inventory": _array_inventory(diag_arrays),
                },
                "log": {"path": str(log_path)},
            },
        }
        _atomic_json_write(json_path, report)
        logger.info("[SUMMARY] status=%s duration=%.3f s", status, duration)
        logger.info("[SUMMARY] main_npz=%s", main_npz_path)
        logger.info("[SUMMARY] json=%s", json_path)
        if args.save_diagnostics:
            logger.info("[SUMMARY] diagnostics_npz=%s", diag_npz_path)
        if status != "PASS":
            logger.warning("[SUMMARY] Non-PASS data cannot be loaded by the strict IEP model")
            return 2
        return 0
    except Exception as exc:
        logger.error("[SUMMARY] status=FAILED error=%s", exc)
        logger.debug("[SUMMARY] traceback:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
