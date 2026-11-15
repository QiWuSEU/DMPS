#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-cut Vec-AEP residual-closure codebook pipeline for the sparse circular CP array.

This runner does not reimplement the optimization kernels.  It orchestrates:

1. discovery of all HFSS full-array scan states from scan_manifest.csv;
2. HFSS-vs-Vec-AEP baseline reconstruction validation;
3. fixed u-v main/side domain construction or reuse;
4. automatic dual-threshold classification (AR and target-referenced LHCP PSLL);
5. strategy routing: direct minimum-PSLL when baseline AR is already good,
   otherwise minimum-AR followed by minimum-PSLL;
6. representative PSLL capping at the global PSLL specification;
7. deterministic selection and packaging of a complete single-cut codebook.

Run this script from the repository root, which contains the migrated modules:
    vecaep_residual_closure/vecaep_io.py
    vecaep_residual_closure/vecaep_grid_domain.py
    vecaep_residual_closure/vecaep_fullgrid_socp.py
    vecaep_residual_closure/bounded_ap_ar_socp.py
    vecaep_residual_closure/bounded_ap_sll_socp.py

The optimization cores remain the source of truth for feasibility, bisection,
representative-codeword generation, and independent NumPy auditing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from vecaep_residual_closure.vecaep_io import (
        decompose_circular_components,
        load_vecaep_dataset,
        read_hfss_vector_field_csv,
        read_source_weights_csv,
        synthesize_circular_fields,
        synthesize_spherical_fields,
    )
    from vecaep_residual_closure.vecaep_grid_domain import load_grid_domain, make_domain_name
    from vecaep_residual_closure.vecaep_fullgrid_socp import rho_to_ar_db, safe_db20, wrap_phase_deg
except Exception as exc:  # pragma: no cover - deployment guard
    raise ImportError(
        "run_vecaep_residual_closure.py requires the DMPS Vec-AEP, grid-domain, full-grid SOCP, "
        "bounded-AR, and bounded-SLL modules in the repository package"
    ) from exc


SCRIPT_VERSION = "1.1.0"
EXPECTED_PORT_COUNT = 187
EXPECTED_POINT_COUNT = 16380

AR_CORE_NAME = "vecaep_residual_closure/bounded_ap_ar_socp.py"
SLL_CORE_NAME = "vecaep_residual_closure/bounded_ap_sll_socp.py"
GRID_CORE_NAME = "vecaep_residual_closure/vecaep_grid_domain.py"

PIPELINE_LOG_NAME = "run_stage3_cut_codebook_pipeline.log"
PIPELINE_COMPLETE_NAME = "pipeline_complete.json"
PIPELINE_INCOMPLETE_NAME = ".pipeline_incomplete"


@dataclass(frozen=True)
class StateFiles:
    scan_theta_deg: float
    scan_phi_deg: float
    state_tag: str
    state_dir: Path
    farfield_csv: Path
    source_weights_csv: Path
    metadata_json: Path
    grid_domain_dir: Path


@dataclass(frozen=True)
class PatternMetrics:
    target_ar_db: float
    target_rho: float
    target_lhcp_abs: float
    target_rhcp_abs: float
    target_ref_lhcp_psll_db: float
    peak_ref_lhcp_psll_db: float
    target_to_main_peak_db: float
    main_peak_index: int
    main_peak_theta_deg: float
    main_peak_phi_deg: float
    worst_side_index: int
    worst_side_theta_deg: float
    worst_side_phi_deg: float


@dataclass(frozen=True)
class ReconstructionMetrics:
    raw_lhcp_relative_l2: float
    aligned_lhcp_relative_l2: float
    complex_correlation_lhcp: float
    global_phase_alignment_deg: float
    target_lhcp_difference_db: float
    whole_pattern_norm_difference_db: float
    etheta_aligned_relative_l2: float
    ephi_aligned_relative_l2: float


@dataclass(frozen=True)
class ThresholdClassification:
    label: str
    ar_passed: bool
    sll_passed: bool
    dual_spec_passed: bool


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, data: Mapping[str, Any] | Sequence[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    os.replace(temp, path)


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def setup_logger(path: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("stage3_cut_codebook_pipeline")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def environment_summary() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def normalize_phi_deg(phi_deg: float) -> float:
    value = float(phi_deg) % 360.0
    if math.isclose(value, 360.0, rel_tol=0.0, abs_tol=1e-10):
        return 0.0
    return value


def integer_angle_token(value: float, width: int = 3) -> str:
    rounded = round(float(value))
    if not math.isclose(float(value), rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"This first pipeline version requires integer-degree scan angles: {value}")
    return f"{int(rounded):0{width}d}"


def signed_phi_token(phi_deg: float) -> str:
    value = normalize_phi_deg(phi_deg)
    # The formal cut is phi=0. Preserve the existing exporter naming convention.
    if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-9):
        return "p000"
    if value > 180.0:
        value -= 360.0
    sign = "p" if value >= 0.0 else "m"
    return sign + integer_angle_token(abs(value))


def state_tag(theta_deg: float, phi_deg: float) -> str:
    return f"theta_{integer_angle_token(theta_deg)}_phi_{signed_phi_token(phi_deg)}"


def ceil_to_step(value: float, step: float, epsilon: float = 1e-10) -> float:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value cannot be rounded: {value}")
    if step <= 0.0 or not math.isfinite(step):
        raise ValueError(f"Round step must be positive and finite: {step}")
    return float(step * math.ceil((float(value) - epsilon) / step))


def _safe_ratio_db(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("Denominator must be positive")
    return float(20.0 * math.log10(max(numerator, 1e-300) / denominator))


def _prepare_output_root(output_root: Path, *, overwrite: bool, resume: bool) -> Path:
    output_root = Path(output_root).expanduser().resolve()
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        elif not resume:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Use --resume or --overwrite."
            )
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _require_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    logger: logging.Logger,
    command_record: Path | None = None,
) -> None:
    command = [str(item) for item in command]
    logger.info("[SUBPROCESS] %s", subprocess.list2cmdline(command))
    if command_record is not None:
        write_json(
            command_record,
            {
                "cwd": str(Path(cwd).resolve()),
                "command": command,
                "command_line": subprocess.list2cmdline(command),
            },
        )
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info("[CORE] %s", line.rstrip())
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Subprocess failed with exit code {return_code}: {subprocess.list2cmdline(command)}"
        )


# -----------------------------------------------------------------------------
# State discovery and fixed-domain construction
# -----------------------------------------------------------------------------


def discover_states(
    hfss_scan_root: Path,
    *,
    scan_phi_deg: float,
    theta_start_deg: float,
    theta_stop_deg: float,
    theta_step_deg: float,
    grid_root: Path,
    mainlobe_radius_uv: float,
) -> list[StateFiles]:
    hfss_scan_root = _require_dir(hfss_scan_root, "HFSS scan root")
    manifest_path = _require_file(hfss_scan_root / "scan_manifest.csv", "HFSS scan manifest")
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", skipinitialspace=True)
    required = {"scan_theta_deg", "scan_phi_deg", "status"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise KeyError(f"Manifest is missing columns {missing}: {manifest_path}")

    expected_count_float = (theta_stop_deg - theta_start_deg) / theta_step_deg
    expected_count = int(round(expected_count_float)) + 1
    if not math.isclose(expected_count_float, round(expected_count_float), abs_tol=1e-9):
        raise ValueError("Theta interval is not an integer number of steps")
    expected_thetas = [theta_start_deg + i * theta_step_deg for i in range(expected_count)]

    selected = manifest.loc[
        manifest["status"].astype(str).str.lower().eq("passed")
        & np.isclose(
            manifest["scan_phi_deg"].astype(float).map(normalize_phi_deg),
            normalize_phi_deg(scan_phi_deg),
            atol=1e-9,
            rtol=0.0,
        )
    ].copy()
    selected = selected.loc[
        (selected["scan_theta_deg"].astype(float) >= theta_start_deg - 1e-9)
        & (selected["scan_theta_deg"].astype(float) <= theta_stop_deg + 1e-9)
    ]
    if selected["scan_theta_deg"].duplicated().any():
        duplicates = selected.loc[selected["scan_theta_deg"].duplicated(), "scan_theta_deg"].tolist()
        raise ValueError(f"Duplicate scan theta rows in manifest: {duplicates}")

    actual_thetas = sorted(float(x) for x in selected["scan_theta_deg"].tolist())
    if len(actual_thetas) != len(expected_thetas) or not np.allclose(
        actual_thetas, expected_thetas, atol=1e-9, rtol=0.0
    ):
        raise ValueError(
            f"HFSS scan states do not match expected theta grid. "
            f"expected={expected_thetas}, actual={actual_thetas}"
        )

    states: list[StateFiles] = []
    for theta in expected_thetas:
        tag = state_tag(theta, scan_phi_deg)
        directory = _require_dir(hfss_scan_root / tag, f"HFSS state directory {tag}")
        theta_token = integer_angle_token(theta)
        phi_token = signed_phi_token(scan_phi_deg)
        farfield = _require_file(
            directory / f"farfield_theta_{theta_token}_phi_{phi_token}.csv",
            f"HFSS far-field CSV for {tag}",
        )
        weights = _require_file(
            directory / f"source_weights_theta_{theta_token}_phi_{phi_token}.csv",
            f"HFSS source weights for {tag}",
        )
        metadata = _require_file(directory / "metadata.json", f"HFSS metadata for {tag}")
        metadata_obj = read_json(metadata)
        if not bool(metadata_obj.get("ok", False)):
            raise ValueError(f"HFSS metadata reports failure: {metadata}")
        if not math.isclose(
            float(metadata_obj.get("scan_theta_deg")), theta, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"Metadata theta mismatch: {metadata}")
        if not math.isclose(
            normalize_phi_deg(float(metadata_obj.get("scan_phi_deg"))),
            normalize_phi_deg(scan_phi_deg),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"Metadata phi mismatch: {metadata}")
        domain_dir = Path(grid_root).resolve() / make_domain_name(
            theta, scan_phi_deg, mainlobe_radius_uv
        )
        states.append(
            StateFiles(
                scan_theta_deg=float(theta),
                scan_phi_deg=float(scan_phi_deg),
                state_tag=tag,
                state_dir=directory,
                farfield_csv=farfield,
                source_weights_csv=weights,
                metadata_json=metadata,
                grid_domain_dir=domain_dir,
            )
        )
    return states


def _domain_is_compatible(
    domain_dir: Path,
    *,
    theta_deg: float,
    phi_deg: float,
    mainlobe_radius_uv: float,
) -> bool:
    summary_path = Path(domain_dir) / "grid_domain_summary.json"
    complete_path = Path(domain_dir) / "domain_complete.json"
    arrays_path = Path(domain_dir) / "grid_domain_arrays.npz"
    if not (summary_path.is_file() and complete_path.is_file() and arrays_path.is_file()):
        return False
    try:
        summary = read_json(summary_path)
        complete = read_json(complete_path)
        return (
            str(complete.get("status", "")).lower() == "complete"
            and math.isclose(
                float(summary["scan"]["theta_deg"]), theta_deg, rel_tol=0.0, abs_tol=1e-9
            )
            and math.isclose(
                normalize_phi_deg(float(summary["scan"]["phi_deg"])),
                normalize_phi_deg(phi_deg),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(summary["domain_definition"]["mainlobe_radius_uv"]),
                mainlobe_radius_uv,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    except Exception:
        return False


def ensure_grid_domain(
    state: StateFiles,
    *,
    project_root: Path,
    vecaep_cache: Path,
    grid_root: Path,
    mainlobe_radius_uv: float,
    boundary_half_width_uv: float,
    target_tolerance_deg: float,
    rebuild: bool,
    logger: logging.Logger,
) -> None:
    if _domain_is_compatible(
        state.grid_domain_dir,
        theta_deg=state.scan_theta_deg,
        phi_deg=state.scan_phi_deg,
        mainlobe_radius_uv=mainlobe_radius_uv,
    ) and not rebuild:
        logger.info("[GRID] reuse %s", state.grid_domain_dir)
        return

    command = [
        sys.executable,
        "-u",
        "-m",
        "vecaep_residual_closure.vecaep_grid_domain",
        "--vecaep-cache",
        str(vecaep_cache),
        "--output-root",
        str(grid_root),
        "--scan-theta-deg",
        f"{state.scan_theta_deg:.12g}",
        "--scan-phi-deg",
        f"{state.scan_phi_deg:.12g}",
        "--mainlobe-radius-uv",
        f"{mainlobe_radius_uv:.12g}",
        "--boundary-half-width-uv",
        f"{boundary_half_width_uv:.12g}",
        "--target-tolerance-deg",
        f"{target_tolerance_deg:.12g}",
        "--baseline-fullwave-dir",
        str(state.state_dir),
        "--overwrite",
    ]
    _run_subprocess(command, cwd=project_root, logger=logger)
    if not _domain_is_compatible(
        state.grid_domain_dir,
        theta_deg=state.scan_theta_deg,
        phi_deg=state.scan_phi_deg,
        mainlobe_radius_uv=mainlobe_radius_uv,
    ):
        raise RuntimeError(f"Grid-domain build did not produce a compatible domain: {state.grid_domain_dir}")


# -----------------------------------------------------------------------------
# Field metrics, reconstruction validation, and threshold selection
# -----------------------------------------------------------------------------


def compute_pattern_metrics(
    e_lhcp: np.ndarray,
    e_rhcp: np.ndarray,
    *,
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    target_index: int,
    main_indices: np.ndarray,
    side_indices: np.ndarray,
) -> PatternMetrics:
    e_lhcp = np.asarray(e_lhcp, dtype=np.complex128).reshape(-1)
    e_rhcp = np.asarray(e_rhcp, dtype=np.complex128).reshape(-1)
    target_lhcp = float(abs(e_lhcp[target_index]))
    target_rhcp = float(abs(e_rhcp[target_index]))
    if not math.isfinite(target_lhcp) or target_lhcp <= 0.0:
        raise ValueError("Target LHCP field magnitude is invalid")
    rho = float(target_rhcp / target_lhcp)
    ar_db = float(rho_to_ar_db(rho))

    side_magnitudes = np.abs(e_lhcp[side_indices])
    main_magnitudes = np.abs(e_lhcp[main_indices])
    worst_side_index = int(side_indices[int(np.argmax(side_magnitudes))])
    main_peak_index = int(main_indices[int(np.argmax(main_magnitudes))])
    target_ref_psll = float(safe_db20(float(abs(e_lhcp[worst_side_index])) / target_lhcp))
    peak_ref_psll = float(
        safe_db20(float(abs(e_lhcp[worst_side_index])) / float(abs(e_lhcp[main_peak_index])))
    )
    target_to_peak = float(
        safe_db20(float(abs(e_lhcp[main_peak_index])) / target_lhcp)
    )
    return PatternMetrics(
        target_ar_db=ar_db,
        target_rho=rho,
        target_lhcp_abs=target_lhcp,
        target_rhcp_abs=target_rhcp,
        target_ref_lhcp_psll_db=target_ref_psll,
        peak_ref_lhcp_psll_db=peak_ref_psll,
        target_to_main_peak_db=target_to_peak,
        main_peak_index=main_peak_index,
        main_peak_theta_deg=float(np.rad2deg(theta_rad[main_peak_index])),
        main_peak_phi_deg=float(np.rad2deg(phi_rad[main_peak_index])),
        worst_side_index=worst_side_index,
        worst_side_theta_deg=float(np.rad2deg(theta_rad[worst_side_index])),
        worst_side_phi_deg=float(np.rad2deg(phi_rad[worst_side_index])),
    )


def compare_reconstruction(
    *,
    hfss_etheta: np.ndarray,
    hfss_ephi: np.ndarray,
    reconstructed_etheta: np.ndarray,
    reconstructed_ephi: np.ndarray,
    hfss_lhcp: np.ndarray,
    reconstructed_lhcp: np.ndarray,
    target_index: int,
) -> ReconstructionMetrics:
    hfss_lhcp = np.asarray(hfss_lhcp, dtype=np.complex128).reshape(-1)
    reconstructed_lhcp = np.asarray(reconstructed_lhcp, dtype=np.complex128).reshape(-1)
    inner = complex(np.vdot(reconstructed_lhcp, hfss_lhcp))
    if abs(inner) <= 0.0:
        raise ValueError("HFSS/reconstructed LHCP correlation inner product is zero")
    alignment = inner / abs(inner)
    aligned_lhcp = alignment * reconstructed_lhcp
    raw_rel = float(np.linalg.norm(reconstructed_lhcp - hfss_lhcp) / np.linalg.norm(hfss_lhcp))
    aligned_rel = float(np.linalg.norm(aligned_lhcp - hfss_lhcp) / np.linalg.norm(hfss_lhcp))
    corr = float(
        abs(np.vdot(reconstructed_lhcp, hfss_lhcp))
        / (np.linalg.norm(reconstructed_lhcp) * np.linalg.norm(hfss_lhcp))
    )
    aligned_etheta = alignment * np.asarray(reconstructed_etheta, dtype=np.complex128)
    aligned_ephi = alignment * np.asarray(reconstructed_ephi, dtype=np.complex128)
    etheta_rel = float(
        np.linalg.norm(aligned_etheta - hfss_etheta) / np.linalg.norm(hfss_etheta)
    )
    ephi_rel = float(np.linalg.norm(aligned_ephi - hfss_ephi) / np.linalg.norm(hfss_ephi))
    return ReconstructionMetrics(
        raw_lhcp_relative_l2=raw_rel,
        aligned_lhcp_relative_l2=aligned_rel,
        complex_correlation_lhcp=corr,
        global_phase_alignment_deg=float(np.rad2deg(np.angle(alignment))),
        target_lhcp_difference_db=_safe_ratio_db(
            float(abs(reconstructed_lhcp[target_index])), float(abs(hfss_lhcp[target_index]))
        ),
        whole_pattern_norm_difference_db=_safe_ratio_db(
            float(np.linalg.norm(reconstructed_lhcp)), float(np.linalg.norm(hfss_lhcp))
        ),
        etheta_aligned_relative_l2=etheta_rel,
        ephi_aligned_relative_l2=ephi_rel,
    )


def classify_metrics(
    metrics: PatternMetrics,
    *,
    ar_spec_db: float,
    sll_spec_db: float,
    tolerance_db: float = 1e-9,
) -> ThresholdClassification:
    ar_pass = bool(metrics.target_ar_db <= ar_spec_db + tolerance_db)
    sll_pass = bool(metrics.target_ref_lhcp_psll_db <= sll_spec_db + tolerance_db)
    if ar_pass and sll_pass:
        label = "PASS_BOTH"
    elif not ar_pass and sll_pass:
        label = "FAIL_AR_ONLY"
    elif ar_pass and not sll_pass:
        label = "FAIL_SLL_ONLY"
    else:
        label = "FAIL_BOTH"
    return ThresholdClassification(
        label=label,
        ar_passed=ar_pass,
        sll_passed=sll_pass,
        dual_spec_passed=bool(ar_pass and sll_pass),
    )


def validate_reconstruction_state(
    *,
    hfss_metrics: PatternMetrics,
    reconstructed_metrics: PatternMetrics,
    comparison: ReconstructionMetrics,
    hfss_classification: ThresholdClassification,
    reconstructed_classification: ThresholdClassification,
    ar_tolerance_db: float,
    psll_tolerance_db: float,
    target_tolerance_db: float,
    correlation_min: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if abs(reconstructed_metrics.target_ar_db - hfss_metrics.target_ar_db) > ar_tolerance_db:
        failures.append("AR_DIFFERENCE")
    if (
        abs(
            reconstructed_metrics.target_ref_lhcp_psll_db
            - hfss_metrics.target_ref_lhcp_psll_db
        )
        > psll_tolerance_db
    ):
        failures.append("PSLL_DIFFERENCE")
    if abs(comparison.target_lhcp_difference_db) > target_tolerance_db:
        failures.append("TARGET_LHCP_DIFFERENCE")
    if comparison.complex_correlation_lhcp < correlation_min:
        failures.append("COMPLEX_CORRELATION")
    if hfss_classification.label != reconstructed_classification.label:
        failures.append("CLASSIFICATION_MISMATCH")
    # Exact peak indices are diagnostic only.  At broadside, multiple phi rows
    # represent the same physical theta=0 direction, and minimax sidelobe
    # solutions can also contain numerically tied grid points.  AR, PSLL, field
    # correlation, and target amplitude are the hard reconstruction gates.
    return len(failures) == 0, failures


def evaluate_baselines(
    states: Sequence[StateFiles],
    *,
    vecaep_cache: Path,
    output_root: Path,
    ar_spec_db: float,
    sll_spec_db: float,
    reconstruction_ar_tolerance_db: float,
    reconstruction_psll_tolerance_db: float,
    reconstruction_target_tolerance_db: float,
    reconstruction_correlation_min: float,
    allow_reconstruction_mismatch: bool,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[StateFiles]]:
    dataset = load_vecaep_dataset(vecaep_cache, mmap_mode="r", load_circular=True)
    if dataset.point_count != EXPECTED_POINT_COUNT or dataset.port_count != EXPECTED_PORT_COUNT:
        raise ValueError(
            f"Unexpected Vec-AEP shape: points={dataset.point_count}, ports={dataset.port_count}"
        )
    rows: list[dict[str, Any]] = []
    failed_states: list[StateFiles] = []
    per_state_root = output_root / "baseline_validation" / "per_state"
    per_state_root.mkdir(parents=True, exist_ok=True)

    for state in states:
        logger.info("[BASELINE] evaluate %s", state.state_tag)
        domain = load_grid_domain(state.grid_domain_dir)
        hfss_grid = read_hfss_vector_field_csv(state.farfield_csv)
        source = read_source_weights_csv(
            state.source_weights_csv,
            expected_port_count=EXPECTED_PORT_COUNT,
            power_sum_tolerance=1e-8,
        )
        if not np.allclose(
            hfss_grid.theta_points_rad,
            dataset.theta_points_rad,
            atol=1e-14,
            rtol=0.0,
        ) or not np.allclose(
            hfss_grid.phi_points_rad,
            dataset.phi_points_rad,
            atol=1e-14,
            rtol=0.0,
        ):
            raise ValueError(f"HFSS/Vec-AEP grid ordering mismatch: {state.state_tag}")
        if not np.allclose(domain.theta_rad, dataset.theta_points_rad, atol=1e-14, rtol=0.0):
            raise ValueError(f"Grid-domain/Vec-AEP theta mismatch: {state.state_tag}")
        if not np.allclose(domain.phi_rad, dataset.phi_points_rad, atol=1e-14, rtol=0.0):
            raise ValueError(f"Grid-domain/Vec-AEP phi mismatch: {state.state_tag}")

        hfss_lhcp, hfss_rhcp = decompose_circular_components(
            hfss_grid.etheta,
            hfss_grid.ephi,
            cp_basis_sign=dataset.cp_basis_sign,
        )
        reconstructed_etheta, reconstructed_ephi = synthesize_spherical_fields(
            dataset, source.weights
        )
        reconstructed_lhcp, reconstructed_rhcp = synthesize_circular_fields(
            dataset, source.weights
        )

        hfss_metrics = compute_pattern_metrics(
            hfss_lhcp,
            hfss_rhcp,
            theta_rad=domain.theta_rad,
            phi_rad=domain.phi_rad,
            target_index=domain.target_index,
            main_indices=domain.main_indices,
            side_indices=domain.side_indices,
        )
        reconstructed_metrics = compute_pattern_metrics(
            reconstructed_lhcp,
            reconstructed_rhcp,
            theta_rad=domain.theta_rad,
            phi_rad=domain.phi_rad,
            target_index=domain.target_index,
            main_indices=domain.main_indices,
            side_indices=domain.side_indices,
        )
        comparison = compare_reconstruction(
            hfss_etheta=hfss_grid.etheta,
            hfss_ephi=hfss_grid.ephi,
            reconstructed_etheta=reconstructed_etheta,
            reconstructed_ephi=reconstructed_ephi,
            hfss_lhcp=hfss_lhcp,
            reconstructed_lhcp=reconstructed_lhcp,
            target_index=domain.target_index,
        )
        hfss_class = classify_metrics(
            hfss_metrics, ar_spec_db=ar_spec_db, sll_spec_db=sll_spec_db
        )
        rec_class = classify_metrics(
            reconstructed_metrics, ar_spec_db=ar_spec_db, sll_spec_db=sll_spec_db
        )
        validation_passed, failure_reasons = validate_reconstruction_state(
            hfss_metrics=hfss_metrics,
            reconstructed_metrics=reconstructed_metrics,
            comparison=comparison,
            hfss_classification=hfss_class,
            reconstructed_classification=rec_class,
            ar_tolerance_db=reconstruction_ar_tolerance_db,
            psll_tolerance_db=reconstruction_psll_tolerance_db,
            target_tolerance_db=reconstruction_target_tolerance_db,
            correlation_min=reconstruction_correlation_min,
        )
        row: dict[str, Any] = {
            "scan_theta_deg": state.scan_theta_deg,
            "scan_phi_deg": state.scan_phi_deg,
            "state_tag": state.state_tag,
            "farfield_csv": str(state.farfield_csv),
            "source_weights_csv": str(state.source_weights_csv),
            "grid_domain_dir": str(state.grid_domain_dir),
            "source_power_w": float(np.sum(source.powers_w)),
            **{f"hfss_{key}": value for key, value in asdict(hfss_metrics).items()},
            **{
                f"reconstructed_{key}": value
                for key, value in asdict(reconstructed_metrics).items()
            },
            **asdict(comparison),
            "delta_ar_db": float(
                reconstructed_metrics.target_ar_db - hfss_metrics.target_ar_db
            ),
            "delta_target_ref_psll_db": float(
                reconstructed_metrics.target_ref_lhcp_psll_db
                - hfss_metrics.target_ref_lhcp_psll_db
            ),
            "hfss_threshold_class": hfss_class.label,
            "reconstructed_threshold_class": rec_class.label,
            "hfss_dual_spec_passed": hfss_class.dual_spec_passed,
            "reconstructed_dual_spec_passed": rec_class.dual_spec_passed,
            "reconstruction_validation_passed": validation_passed,
            "reconstruction_validation_failures": ";".join(failure_reasons),
        }
        rows.append(row)

        state_output = per_state_root / state.state_tag
        state_output.mkdir(parents=True, exist_ok=True)
        write_json(
            state_output / "baseline_comparison.json",
            {
                "state": asdict(state),
                "hfss_metrics": asdict(hfss_metrics),
                "reconstructed_metrics": asdict(reconstructed_metrics),
                "reconstruction_metrics": asdict(comparison),
                "hfss_classification": asdict(hfss_class),
                "reconstructed_classification": asdict(rec_class),
                "validation_passed": validation_passed,
                "validation_failures": failure_reasons,
            },
        )
        np.savez_compressed(
            state_output / "baseline_fields_comparison.npz",
            theta_rad=np.asarray(domain.theta_rad),
            phi_rad=np.asarray(domain.phi_rad),
            target_index=np.asarray(domain.target_index, dtype=np.int64),
            main_indices=np.asarray(domain.main_indices, dtype=np.int64),
            side_indices=np.asarray(domain.side_indices, dtype=np.int64),
            hfss_lhcp=np.asarray(hfss_lhcp, dtype=np.complex128),
            hfss_rhcp=np.asarray(hfss_rhcp, dtype=np.complex128),
            reconstructed_lhcp=np.asarray(reconstructed_lhcp, dtype=np.complex128),
            reconstructed_rhcp=np.asarray(reconstructed_rhcp, dtype=np.complex128),
        )
        logger.info(
            "[BASELINE] %s HFSS AR=%.6f dB PSLL=%.6f dB class=%s "
            "reconstruction=%s",
            state.state_tag,
            hfss_metrics.target_ar_db,
            hfss_metrics.target_ref_lhcp_psll_db,
            hfss_class.label,
            "PASS" if validation_passed else f"FAIL:{failure_reasons}",
        )
        if not hfss_class.dual_spec_passed:
            failed_states.append(state)
        if not validation_passed and not allow_reconstruction_mismatch:
            raise RuntimeError(
                f"Reconstruction validation failed for {state.state_tag}: {failure_reasons}"
            )

    validation_root = output_root / "baseline_validation"
    details_df = pd.DataFrame(rows).sort_values(["scan_phi_deg", "scan_theta_deg"])
    details_df.to_csv(
        validation_root / "fullwave_vecaep_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline_cols = [
        "scan_theta_deg",
        "scan_phi_deg",
        "state_tag",
        "hfss_target_ar_db",
        "hfss_target_ref_lhcp_psll_db",
        "hfss_peak_ref_lhcp_psll_db",
        "reconstructed_target_ar_db",
        "reconstructed_target_ref_lhcp_psll_db",
        "reconstructed_peak_ref_lhcp_psll_db",
        "delta_ar_db",
        "delta_target_ref_psll_db",
        "hfss_threshold_class",
        "reconstructed_threshold_class",
        "reconstruction_validation_passed",
    ]
    details_df[baseline_cols].to_csv(
        validation_root / "baseline_scan_metrics.csv", index=False, encoding="utf-8-sig"
    )
    classification_cols = [
        "scan_theta_deg",
        "scan_phi_deg",
        "state_tag",
        "hfss_target_ar_db",
        "hfss_target_ref_lhcp_psll_db",
        "hfss_threshold_class",
        "hfss_dual_spec_passed",
        "reconstructed_threshold_class",
        "reconstruction_validation_passed",
    ]
    details_df[classification_cols].to_csv(
        validation_root / "threshold_classification.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(
        validation_root / "threshold_selection.json",
        {
            "ar_spec_db": ar_spec_db,
            "sll_spec_db": sll_spec_db,
            "selection_rule": "HFSS AR > spec OR HFSS target-referenced LHCP PSLL > spec",
            "failed_state_count": len(failed_states),
            "failed_states": [state.state_tag for state in failed_states],
            "class_counts": details_df["hfss_threshold_class"].value_counts().to_dict(),
        },
    )
    write_json(
        validation_root / "reconstruction_summary.json",
        {
            "all_states_passed": bool(details_df["reconstruction_validation_passed"].all()),
            "state_count": int(len(details_df)),
            "maximum_absolute_delta_ar_db": float(details_df["delta_ar_db"].abs().max()),
            "maximum_absolute_delta_psll_db": float(
                details_df["delta_target_ref_psll_db"].abs().max()
            ),
            "minimum_complex_correlation_lhcp": float(
                details_df["complex_correlation_lhcp"].min()
            ),
            "maximum_aligned_lhcp_relative_l2": float(
                details_df["aligned_lhcp_relative_l2"].max()
            ),
        },
    )
    return rows, failed_states


# -----------------------------------------------------------------------------
# Optimization orchestration and result selection
# -----------------------------------------------------------------------------


def _core_complete(output_dir: Path, summary_name: str) -> bool:
    marker = Path(output_dir) / "solution_complete.json"
    summary = Path(output_dir) / summary_name
    if not (marker.is_file() and summary.is_file()):
        return False
    try:
        marker_obj = read_json(marker)
        return str(marker_obj.get("status", "")).lower() not in {"failed", ""}
    except Exception:
        return False


def _core_base_args(args: argparse.Namespace, state: StateFiles, output_dir: Path) -> list[str]:
    return [
        "--vecaep-cache",
        str(args.vecaep_cache),
        "--grid-domain",
        str(state.grid_domain_dir),
        "--source-weights",
        str(state.source_weights_csv),
        "--output-dir",
        str(output_dir),
        "--solver",
        args.solver,
        "--constraint-model",
        "fixed-inner",
        "--amplitude-limit-db",
        f"{args.amplitude_limit_db:.12g}",
        "--phase-limit-deg",
        f"{args.phase_limit_deg:.12g}",
        "--target-co-fraction",
        f"{args.target_co_fraction:.12g}",
        "--main-peak-cap-db",
        f"{args.main_peak_cap_db:.12g}",
        "--power-budget-w",
        f"{args.power_budget_w:.12g}",
        "--margin-accept-tol",
        f"{args.margin_accept_tol:.12g}",
        "--margin-reject-tol",
        f"{args.margin_reject_tol:.12g}",
        "--mosek-time-limit-s",
        f"{args.mosek_time_limit_s:.12g}",
        "--power-accept-tol",
        f"{args.power_accept_tol:.12g}",
        "--power-reject-tol",
        f"{args.power_reject_tol:.12g}",
        "--constraint-abs-tol",
        f"{args.constraint_abs_tol:.12g}",
        "--correction-abs-tol",
        f"{args.correction_abs_tol:.12g}",
        "--phase-deg-tol",
        f"{args.phase_deg_tol:.12g}",
        "--log-level",
        args.log_level,
    ] + (["--no-solver-fallback"] if args.no_solver_fallback else [])


def run_min_ar_case(
    args: argparse.Namespace,
    state: StateFiles,
    case_root: Path,
    *,
    logger: logging.Logger,
) -> dict[str, Any]:
    output_dir = case_root / "min_ar"
    summary_path = output_dir / "bounded_ap_ar_summary.json"
    if args.resume and _core_complete(output_dir, summary_path.name):
        logger.info("[RESUME] reuse min-AR case %s", state.state_tag)
        return read_json(summary_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    command = [
        sys.executable,
        "-u",
        "-m",
        "vecaep_residual_closure.bounded_ap_ar_socp",
        *_core_base_args(args, state, output_dir),
        "--sll-db",
        f"{args.sll_spec_db:.12g}",
        "--rho-tol",
        f"{args.rho_tol:.12g}",
        "--max-bisection-iterations",
        str(args.ar_max_bisection_iterations),
        "--representative-rho-margin",
        f"{args.representative_rho_margin:.12g}",
        "--overwrite",
    ]
    _run_subprocess(
        command,
        cwd=args.project_root,
        logger=logger,
        command_record=case_root / "min_ar_command.json",
    )
    return read_json(summary_path)


def determine_sll_ar_limit(
    ar_summary: Mapping[str, Any],
    *,
    ar_spec_db: float,
    round_step_db: float,
) -> tuple[float, dict[str, Any]]:
    status = str(ar_summary.get("status", ""))
    bisection = ar_summary.get("bisection")
    if status == "complete" and isinstance(bisection, Mapping):
        upper = float(bisection["ar_upper_db"])
        upper_certified = bool(bisection.get("upper_is_certified_feasible", False))
        if upper_certified and upper <= ar_spec_db + 1e-9:
            rounded = min(ar_spec_db, ceil_to_step(upper, round_step_db))
            return rounded, {
                "mode": "rounded_min_ar_certified_upper",
                "min_ar_certified_upper_db": upper,
                "round_step_db": round_step_db,
                "final_ar_limit_db": rounded,
            }
    return float(ar_spec_db), {
        "mode": "diagnostic_fixed_ar_spec",
        "min_ar_status": status,
        "round_step_db": round_step_db,
        "final_ar_limit_db": float(ar_spec_db),
        "reason": "No certified min-AR upper bound at or below the global AR specification",
    }


def run_min_sll_case(
    args: argparse.Namespace,
    state: StateFiles,
    case_root: Path,
    *,
    ar_limit_db: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    output_dir = case_root / "min_sll"
    summary_path = output_dir / "bounded_ap_sll_summary.json"
    if args.resume and _core_complete(output_dir, summary_path.name):
        existing = read_json(summary_path)
        existing_problem = existing.get("problem", {})
        existing_limit = existing_problem.get("ar_limit_db")
        existing_cap = existing_problem.get("representative_sll_cap_db")
        limit_matches = existing_limit is not None and math.isclose(
            float(existing_limit), ar_limit_db, rel_tol=0.0, abs_tol=1e-10
        )
        cap_matches = existing_cap is not None and math.isclose(
            float(existing_cap), args.sll_spec_db, rel_tol=0.0, abs_tol=1e-10
        )
        if limit_matches and cap_matches:
            logger.info("[RESUME] reuse min-SLL case %s", state.state_tag)
            return existing
        logger.warning(
            "[RESUME] min-SLL AR limit or representative cap changed; rerun %s",
            state.state_tag,
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    command = [
        sys.executable,
        "-u",
        "-m",
        "vecaep_residual_closure.bounded_ap_sll_socp",
        *_core_base_args(args, state, output_dir),
        "--ar-limit-db",
        f"{ar_limit_db:.12g}",
        "--ar-round-step-db",
        f"{args.ar_round_step_db:.12g}",
        "--sll-loose-db",
        f"{args.sll_loose_db:.12g}",
        "--sll-strict-db",
        f"{args.sll_strict_db:.12g}",
        "--sll-tol-db",
        f"{args.sll_tol_db:.12g}",
        "--max-bisection-iterations",
        str(args.sll_max_bisection_iterations),
        "--representative-sll-margin-db",
        f"{args.representative_sll_margin_db:.12g}",
        "--representative-sll-cap-db",
        f"{args.sll_spec_db:.12g}",
        "--overwrite",
    ]
    _run_subprocess(
        command,
        cwd=args.project_root,
        logger=logger,
        command_record=case_root / "min_sll_command.json",
    )
    return read_json(summary_path)


def _audit_meets_dual_spec(
    audit: Mapping[str, Any] | None,
    *,
    ar_spec_db: float,
    sll_spec_db: float,
    tolerance_db: float = 2e-5,
) -> bool:
    if not audit:
        return False
    feasible = bool(audit.get("feasible_under_budget", False))
    ar_value = audit.get("target_ar_actual_db")
    sll_value = audit.get("target_ref_lhcp_psll_db", audit.get("max_side_db"))
    if ar_value is None or sll_value is None:
        return False
    return bool(
        feasible
        and float(ar_value) <= ar_spec_db + tolerance_db
        and float(sll_value) <= sll_spec_db + tolerance_db
    )


def select_case_codeword(
    *,
    case_root: Path,
    ar_summary: Mapping[str, Any] | None,
    sll_summary: Mapping[str, Any],
    ar_spec_db: float,
    sll_spec_db: float,
    optimization_strategy: str,
) -> dict[str, Any]:
    sll_selected = sll_summary.get("selected_solution") or {}
    ar_selected = (ar_summary or {}).get("selected_solution") or {}
    sll_audit = sll_selected.get("audit")
    ar_audit = ar_selected.get("audit")
    if _audit_meets_dual_spec(
        sll_audit, ar_spec_db=ar_spec_db, sll_spec_db=sll_spec_db
    ):
        if optimization_strategy == "DIRECT_SLL_BASELINE_AR_CAP":
            return {
                "selection_status": "SELECTED_DIRECT_SLL_BASELINE_AR_PRESERVED",
                "selection_mode": "direct_sll_baseline_ar_cap",
                "source_role": "stage3_direct_min_sll_representative",
                "source_dir": str(case_root / "min_sll"),
                "optimized_weights_csv": str(case_root / "min_sll" / "optimized_weights.csv"),
                "summary_json": str(case_root / "min_sll" / "bounded_ap_sll_summary.json"),
                "audit": dict(sll_audit),
            }
        return {
            "selection_status": "SELECTED_LEXICOGRAPHIC_AR_THEN_SLL",
            "selection_mode": "lexicographic_ar_then_psll",
            "source_role": "stage3_min_sll_representative",
            "source_dir": str(case_root / "min_sll"),
            "optimized_weights_csv": str(case_root / "min_sll" / "optimized_weights.csv"),
            "summary_json": str(case_root / "min_sll" / "bounded_ap_sll_summary.json"),
            "audit": dict(sll_audit),
        }
    if optimization_strategy != "DIRECT_SLL_BASELINE_AR_CAP" and _audit_meets_dual_spec(
        ar_audit, ar_spec_db=ar_spec_db, sll_spec_db=sll_spec_db
    ):
        return {
            "selection_status": "SELECTED_FALLBACK",
            "selection_mode": "min_ar_fallback",
            "source_role": "stage3_min_ar_representative",
            "source_dir": str(case_root / "min_ar"),
            "optimized_weights_csv": str(case_root / "min_ar" / "optimized_weights.csv"),
            "summary_json": str(case_root / "min_ar" / "bounded_ap_ar_summary.json"),
            "audit": dict(ar_audit),
            "reason": "The minimum-SLL representative did not close both global specifications",
        }
    return {
        "selection_status": "NO_DUAL_SPEC_CODEWORD",
        "selection_mode": "none",
        "source_role": None,
        "optimized_weights_csv": None,
        "summary_json": None,
        "audit": None,
        "reason": (
            "The direct minimum-SLL representative did not close both global specifications"
            if optimization_strategy == "DIRECT_SLL_BASELINE_AR_CAP"
            else "Neither optimized representative was independently audited to satisfy both specifications"
        ),
    }


def _copy_selected_outputs(selection: Mapping[str, Any], selected_dir: Path) -> None:
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True, exist_ok=True)
    write_json(selected_dir / "selection.json", dict(selection))
    source_dir_value = selection.get("source_dir")
    if not source_dir_value:
        return
    source_dir = Path(str(source_dir_value))
    for name in [
        "optimized_weights.csv",
        "grid_field_audit.npz",
        "top_sidelobe_points.csv",
        "bounded_ap_ar_summary.json",
        "bounded_ap_sll_summary.json",
        "bounded_ap_ar_solution.npz",
        "bounded_ap_sll_solution.npz",
        "solution_complete.json",
    ]:
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, selected_dir / name)


def _load_baseline_ar_guard(
    output_root: Path,
    state: StateFiles,
) -> tuple[float, float, float]:
    comparison_path = (
        Path(output_root)
        / "baseline_validation"
        / "per_state"
        / state.state_tag
        / "baseline_comparison.json"
    )
    comparison = read_json(comparison_path)
    hfss_ar_db = float(comparison["hfss_metrics"]["target_ar_db"])
    reconstructed_ar_db = float(
        comparison["reconstructed_metrics"]["target_ar_db"]
    )
    guard_db = max(hfss_ar_db, reconstructed_ar_db)
    return hfss_ar_db, reconstructed_ar_db, guard_db


def _write_min_ar_skipped(
    case_root: Path,
    *,
    hfss_ar_db: float,
    reconstructed_ar_db: float,
    baseline_ar_guard_db: float,
    threshold_db: float,
) -> dict[str, Any]:
    output_dir = case_root / "min_ar"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "skipped",
        "reason": "baseline_ar_below_direct_sll_threshold",
        "baseline_hfss_ar_db": hfss_ar_db,
        "baseline_reconstructed_ar_db": reconstructed_ar_db,
        "baseline_ar_guard_db": baseline_ar_guard_db,
        "direct_sll_ar_threshold_db": threshold_db,
        "ar_limit_db": baseline_ar_guard_db,
    }
    write_json(output_dir / "min_ar_skipped.json", summary)
    return summary


def optimize_failed_states(
    args: argparse.Namespace,
    failed_states: Sequence[StateFiles],
    *,
    output_root: Path,
    logger: logging.Logger,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    cases_root = output_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    for index, state in enumerate(failed_states, start=1):
        logger.info(
            "[CASE %d/%d] optimize %s", index, len(failed_states), state.state_tag
        )
        case_root = cases_root / state.state_tag
        case_root.mkdir(parents=True, exist_ok=True)

        hfss_ar_db, reconstructed_ar_db, baseline_ar_guard_db = _load_baseline_ar_guard(
            output_root, state
        )
        direct_sll = bool(
            baseline_ar_guard_db <= args.direct_sll_ar_threshold_db + 1e-12
        )

        if direct_sll:
            optimization_strategy = "DIRECT_SLL_BASELINE_AR_CAP"
            min_ar_skipped = True
            min_ar_skip = _write_min_ar_skipped(
                case_root,
                hfss_ar_db=hfss_ar_db,
                reconstructed_ar_db=reconstructed_ar_db,
                baseline_ar_guard_db=baseline_ar_guard_db,
                threshold_db=args.direct_sll_ar_threshold_db,
            )
            ar_summary: dict[str, Any] | None = None
            ar_limit_db = baseline_ar_guard_db
            ar_limit_strategy = {
                "mode": "baseline_ar_guard_direct_sll",
                "baseline_hfss_ar_db": hfss_ar_db,
                "baseline_reconstructed_ar_db": reconstructed_ar_db,
                "baseline_ar_guard_db": baseline_ar_guard_db,
                "direct_sll_ar_threshold_db": args.direct_sll_ar_threshold_db,
                "final_ar_limit_db": ar_limit_db,
            }
            logger.info(
                "[STRATEGY] %s baseline AR guard=%.9f dB <= %.6f dB; "
                "skip min-AR and directly minimize PSLL with AR<=%.9f dB",
                state.state_tag,
                baseline_ar_guard_db,
                args.direct_sll_ar_threshold_db,
                ar_limit_db,
            )
        else:
            optimization_strategy = "MIN_AR_THEN_SLL"
            min_ar_skipped = False
            min_ar_skip = None
            ar_summary = run_min_ar_case(args, state, case_root, logger=logger)
            ar_limit_db, ar_limit_strategy = determine_sll_ar_limit(
                ar_summary,
                ar_spec_db=args.ar_spec_db,
                round_step_db=args.ar_round_step_db,
            )
            logger.info(
                "[STRATEGY] %s baseline AR guard=%.9f dB > %.6f dB; "
                "run min-AR then min-PSLL with AR<=%.9f dB",
                state.state_tag,
                baseline_ar_guard_db,
                args.direct_sll_ar_threshold_db,
                ar_limit_db,
            )

        write_json(case_root / "ar_limit_strategy.json", ar_limit_strategy)
        sll_summary = run_min_sll_case(
            args,
            state,
            case_root,
            ar_limit_db=ar_limit_db,
            logger=logger,
        )
        selection = select_case_codeword(
            case_root=case_root,
            ar_summary=ar_summary,
            sll_summary=sll_summary,
            ar_spec_db=args.ar_spec_db,
            sll_spec_db=args.sll_spec_db,
            optimization_strategy=optimization_strategy,
        )
        _copy_selected_outputs(selection, case_root / "selected")
        case_summary = {
            "state": asdict(state),
            "optimization_strategy": optimization_strategy,
            "min_ar_skipped": min_ar_skipped,
            "min_ar_skip_reason": (
                None if min_ar_skip is None else min_ar_skip.get("reason")
            ),
            "baseline_hfss_ar_db": hfss_ar_db,
            "baseline_reconstructed_ar_db": reconstructed_ar_db,
            "baseline_ar_guard_db": baseline_ar_guard_db,
            "direct_sll_ar_threshold_db": args.direct_sll_ar_threshold_db,
            "ar_limit_strategy": ar_limit_strategy,
            "min_ar_status": (
                "skipped" if ar_summary is None else ar_summary.get("status")
            ),
            "min_ar_bisection": (
                None if ar_summary is None else ar_summary.get("bisection")
            ),
            "min_ar_selected_audit": (
                None
                if ar_summary is None
                else (ar_summary.get("selected_solution") or {}).get("audit")
            ),
            "min_sll_status": sll_summary.get("status"),
            "min_sll_bisection": sll_summary.get("bisection"),
            "min_sll_selected_audit": (
                sll_summary.get("selected_solution") or {}
            ).get("audit"),
            "selection": selection,
        }
        write_json(case_root / "case_summary.json", case_summary)
        write_json(
            case_root / "case_manifest.json",
            {
                "state": asdict(state),
                "inputs": {
                    "vecaep_cache": str(args.vecaep_cache),
                    "grid_domain": str(state.grid_domain_dir),
                    "source_weights": str(state.source_weights_csv),
                },
                "strategy": {
                    "optimization_strategy": optimization_strategy,
                    "min_ar_skipped": min_ar_skipped,
                    "baseline_ar_guard_db": baseline_ar_guard_db,
                    "direct_sll_ar_threshold_db": args.direct_sll_ar_threshold_db,
                    "ar_limit_db": ar_limit_db,
                    "representative_sll_cap_db": args.sll_spec_db,
                },
                "constraints": {
                    "ar_spec_db": args.ar_spec_db,
                    "sll_spec_db": args.sll_spec_db,
                    "amplitude_limit_db": args.amplitude_limit_db,
                    "phase_limit_deg": args.phase_limit_deg,
                    "target_co_fraction": args.target_co_fraction,
                    "main_peak_cap_db": args.main_peak_cap_db,
                    "power_budget_w": args.power_budget_w,
                },
            },
        )
        results[state.state_tag] = case_summary
        logger.info(
            "[CASE] %s strategy=%s selection=%s",
            state.state_tag,
            optimization_strategy,
            selection["selection_status"],
        )
    return results


# -----------------------------------------------------------------------------
# Complete cut codebook packaging
# -----------------------------------------------------------------------------


def _load_optimized_weights(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    table = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True)
    required = {"global_index", "optimized_real", "optimized_imag", "optimized_power_W", "optimized_phase_deg"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"Optimized weights CSV is missing columns {missing}: {path}")
    table = table.sort_values("global_index", kind="mergesort").reset_index(drop=True)
    if table["global_index"].astype(int).tolist() != list(range(EXPECTED_PORT_COUNT)):
        raise ValueError(f"Optimized weights global_index mapping is invalid: {path}")
    weights = (
        table["optimized_real"].astype(float).to_numpy()
        + 1j * table["optimized_imag"].astype(float).to_numpy()
    )
    return np.asarray(weights, dtype=np.complex128), table


def write_hfss_source_weights(
    *,
    baseline_source_csv: Path,
    final_weights: np.ndarray,
    output_csv: Path,
) -> pd.DataFrame:
    source = pd.read_csv(baseline_source_csv, encoding="utf-8-sig", skipinitialspace=True)
    source = source.sort_values("global_index", kind="mergesort").reset_index(drop=True)
    final_weights = np.asarray(final_weights, dtype=np.complex128).reshape(-1)
    if len(source) != EXPECTED_PORT_COUNT or final_weights.size != EXPECTED_PORT_COUNT:
        raise ValueError("Final codeword does not contain exactly 187 ports")
    baseline_wrapped = source["total_phase_wrapped_deg"].astype(float).to_numpy()
    baseline_unwrapped = source["total_phase_unwrapped_deg"].astype(float).to_numpy()
    final_wrapped = np.asarray(wrap_phase_deg(np.rad2deg(np.angle(final_weights))), dtype=float)
    phase_change = np.asarray(wrap_phase_deg(final_wrapped - baseline_wrapped), dtype=float)
    final_unwrapped = baseline_unwrapped + phase_change
    source["magnitude_W"] = np.abs(final_weights) ** 2
    source["total_phase_unwrapped_deg"] = final_unwrapped
    source["total_phase_wrapped_deg"] = final_wrapped
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return source


def _final_metrics_for_weights(
    dataset: Any,
    state: StateFiles,
    weights: np.ndarray,
) -> PatternMetrics:
    domain = load_grid_domain(state.grid_domain_dir)
    e_lhcp, e_rhcp = synthesize_circular_fields(dataset, weights)
    return compute_pattern_metrics(
        e_lhcp,
        e_rhcp,
        theta_rad=domain.theta_rad,
        phi_rad=domain.phi_rad,
        target_index=domain.target_index,
        main_indices=domain.main_indices,
        side_indices=domain.side_indices,
    )


def build_complete_codebook(
    args: argparse.Namespace,
    states: Sequence[StateFiles],
    baseline_rows: Sequence[Mapping[str, Any]],
    case_results: Mapping[str, Mapping[str, Any]],
    *,
    output_root: Path,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    dataset = load_vecaep_dataset(args.vecaep_cache, mmap_mode="r", load_circular=True)
    baseline_by_tag = {str(row["state_tag"]): row for row in baseline_rows}
    codebook_root = output_root / "codebook"
    codebook_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []

    for state in states:
        baseline_row = baseline_by_tag[state.state_tag]
        baseline_source = read_source_weights_csv(
            state.source_weights_csv,
            expected_port_count=EXPECTED_PORT_COUNT,
            power_sum_tolerance=1e-8,
        )
        case = case_results.get(state.state_tag)
        selection = None if case is None else case.get("selection")
        selected_path = None if selection is None else selection.get("optimized_weights_csv")

        if selected_path:
            final_weights, optimized_table = _load_optimized_weights(Path(str(selected_path)))
            codeword_source = str(selection.get("source_role"))
            stage3_applied = True
            selection_status = str(selection.get("selection_status"))
        else:
            final_weights = np.asarray(baseline_source.weights, dtype=np.complex128)
            optimized_table = None
            stage3_applied = False
            if case is None:
                codeword_source = "stage2_baseline_passed_dual_spec"
                selection_status = "BASELINE_RETAINED"
            else:
                codeword_source = "stage2_baseline_no_certified_stage3_replacement"
                selection_status = "NO_DUAL_SPEC_CODEWORD"

        final_metrics = _final_metrics_for_weights(dataset, state, final_weights)
        final_class = classify_metrics(
            final_metrics, ar_spec_db=args.ar_spec_db, sll_spec_db=args.sll_spec_db
        )
        state_output = codebook_root / state.state_tag
        state_output.mkdir(parents=True, exist_ok=True)
        final_source_name = (
            f"final_source_weights_theta_{integer_angle_token(state.scan_theta_deg)}_"
            f"phi_{signed_phi_token(state.scan_phi_deg)}.csv"
        )
        final_source_path = state_output / final_source_name
        write_hfss_source_weights(
            baseline_source_csv=state.source_weights_csv,
            final_weights=final_weights,
            output_csv=final_source_path,
        )
        if optimized_table is not None:
            optimized_table.to_csv(
                state_output / "final_weights_detailed.csv",
                index=False,
                encoding="utf-8-sig",
            )
        np.savez_compressed(
            state_output / "final_codeword.npz",
            weights=np.asarray(final_weights, dtype=np.complex128),
            powers_w=np.abs(final_weights) ** 2,
            phases_deg=np.asarray(wrap_phase_deg(np.rad2deg(np.angle(final_weights))), dtype=float),
        )
        metadata = {
            "state": asdict(state),
            "codeword_source": codeword_source,
            "selection_status": selection_status,
            "stage3_correction_applied": stage3_applied,
            "baseline_hfss_metrics": {
                key.removeprefix("hfss_"): value
                for key, value in baseline_row.items()
                if str(key).startswith("hfss_")
            },
            "baseline_reconstructed_metrics": {
                key.removeprefix("reconstructed_"): value
                for key, value in baseline_row.items()
                if str(key).startswith("reconstructed_")
            },
            "final_vecaep_metrics": asdict(final_metrics),
            "final_classification": asdict(final_class),
            "final_source_weights_csv": str(final_source_path),
            "selected_core_result": selection,
        }
        write_json(state_output / "codeword_metadata.json", metadata)
        manifest_rows.append(
            {
                "scan_theta_deg": state.scan_theta_deg,
                "scan_phi_deg": state.scan_phi_deg,
                "state_tag": state.state_tag,
                "baseline_or_corrected": "corrected" if stage3_applied else "baseline",
                "codeword_source": codeword_source,
                "selection_status": selection_status,
                "source_weights_path": str(final_source_path),
                "baseline_hfss_ar_db": baseline_row["hfss_target_ar_db"],
                "baseline_hfss_psll_db": baseline_row["hfss_target_ref_lhcp_psll_db"],
                "baseline_reconstructed_ar_db": baseline_row["reconstructed_target_ar_db"],
                "baseline_reconstructed_psll_db": baseline_row[
                    "reconstructed_target_ref_lhcp_psll_db"
                ],
                "final_reconstructed_ar_db": final_metrics.target_ar_db,
                "final_reconstructed_target_ref_psll_db": final_metrics.target_ref_lhcp_psll_db,
                "final_reconstructed_peak_ref_psll_db": final_metrics.peak_ref_lhcp_psll_db,
                "final_target_lhcp_abs": final_metrics.target_lhcp_abs,
                "final_power_w": float(np.vdot(final_weights, final_weights).real),
                "dual_spec_passed": final_class.dual_spec_passed,
                "requires_hfss_export": bool(stage3_applied),
            }
        )
        logger.info(
            "[CODEBOOK] %s source=%s final AR=%.6f dB PSLL=%.6f dB pass=%s",
            state.state_tag,
            codeword_source,
            final_metrics.target_ar_db,
            final_metrics.target_ref_lhcp_psll_db,
            final_class.dual_spec_passed,
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["scan_phi_deg", "scan_theta_deg"])
    manifest_df.to_csv(
        codebook_root / "cut_codebook_manifest.csv", index=False, encoding="utf-8-sig"
    )
    manifest_df[
        [
            "scan_theta_deg",
            "scan_phi_deg",
            "source_weights_path",
            "final_reconstructed_ar_db",
            "final_reconstructed_target_ref_psll_db",
            "requires_hfss_export",
        ]
    ].to_csv(codebook_root / "hfss_replay_manifest.csv", index=False, encoding="utf-8-sig")
    write_json(
        codebook_root / "cut_codebook_summary.json",
        {
            "state_count": int(len(manifest_df)),
            "corrected_state_count": int((manifest_df["baseline_or_corrected"] == "corrected").sum()),
            "dual_spec_pass_count": int(manifest_df["dual_spec_passed"].sum()),
            "all_states_dual_spec_passed": bool(manifest_df["dual_spec_passed"].all()),
            "worst_final_ar_db": float(manifest_df["final_reconstructed_ar_db"].max()),
            "worst_final_target_ref_psll_db": float(
                manifest_df["final_reconstructed_target_ref_psll_db"].max()
            ),
            "manifest_csv": str(codebook_root / "cut_codebook_manifest.csv"),
        },
    )
    return manifest_rows


# -----------------------------------------------------------------------------
# Summaries and plots
# -----------------------------------------------------------------------------


def write_optimization_summaries(
    case_results: Mapping[str, Mapping[str, Any]],
    *,
    output_root: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for tag, case in case_results.items():
        state = case["state"]
        ar_b = case.get("min_ar_bisection") or {}
        ar_a = case.get("min_ar_selected_audit") or {}
        sll_b = case.get("min_sll_bisection") or {}
        sll_a = case.get("min_sll_selected_audit") or {}
        selection = case.get("selection") or {}
        rows.append(
            {
                "state_tag": tag,
                "scan_theta_deg": state["scan_theta_deg"],
                "scan_phi_deg": state["scan_phi_deg"],
                "optimization_strategy": case.get("optimization_strategy"),
                "min_ar_skipped": case.get("min_ar_skipped"),
                "baseline_ar_guard_db": case.get("baseline_ar_guard_db"),
                "direct_sll_ar_threshold_db": case.get("direct_sll_ar_threshold_db"),
                "min_ar_status": case.get("min_ar_status"),
                "min_ar_lower_db": ar_b.get("ar_lower_db"),
                "min_ar_upper_db": ar_b.get("ar_upper_db"),
                "min_ar_selected_actual_db": ar_a.get("target_ar_actual_db"),
                "min_ar_selected_psll_db": ar_a.get("max_side_db"),
                "rounded_ar_limit_db": case.get("ar_limit_strategy", {}).get(
                    "final_ar_limit_db"
                ),
                "min_sll_status": case.get("min_sll_status"),
                "min_sll_infeasible_db": sll_b.get("sll_infeasible_db"),
                "min_sll_feasible_db": sll_b.get("sll_feasible_db"),
                "min_sll_selected_actual_db": sll_a.get("target_ref_lhcp_psll_db"),
                "min_sll_selected_ar_db": sll_a.get("target_ar_actual_db"),
                "selection_status": selection.get("selection_status"),
                "selection_mode": selection.get("selection_mode"),
            }
        )
    summary_root = output_root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["scan_phi_deg", "scan_theta_deg"]).to_csv(
        summary_root / "failed_state_optimization_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(summary_root / "failed_state_optimization_summary.json", rows)


def write_complete_cut_summary(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
) -> None:
    summary_root = output_root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(manifest_rows).sort_values(["scan_phi_deg", "scan_theta_deg"])
    frame.to_csv(summary_root / "complete_cut_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(summary_root / "complete_cut_metrics.json", frame.to_dict("records"))


def create_plots(
    baseline_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]] | None,
    *,
    output_root: Path,
    ar_spec_db: float,
    sll_spec_db: float,
    logger: logging.Logger,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.warning("[PLOT] matplotlib unavailable; skip plots: %s", exc)
        return
    plots_root = output_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    baseline = pd.DataFrame(baseline_rows).sort_values("scan_theta_deg")

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(baseline["scan_theta_deg"], baseline["hfss_target_ar_db"], marker="o", label="HFSS full array")
    plt.plot(baseline["scan_theta_deg"], baseline["reconstructed_target_ar_db"], marker="x", linestyle="--", label="Vec-AEP reconstruction")
    plt.axhline(ar_spec_db, linestyle=":", label=f"AR specification = {ar_spec_db:g} dB")
    plt.xlabel("Scan theta (deg)")
    plt.ylabel("Target AR (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_root / "baseline_hfss_vs_vecaep_ar.png", dpi=240)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(baseline["scan_theta_deg"], baseline["hfss_target_ref_lhcp_psll_db"], marker="o", label="HFSS full array")
    plt.plot(baseline["scan_theta_deg"], baseline["reconstructed_target_ref_lhcp_psll_db"], marker="x", linestyle="--", label="Vec-AEP reconstruction")
    plt.axhline(sll_spec_db, linestyle=":", label=f"PSLL specification = {sll_spec_db:g} dB")
    plt.xlabel("Scan theta (deg)")
    plt.ylabel("Target-referenced LHCP PSLL (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_root / "baseline_hfss_vs_vecaep_psll.png", dpi=240)
    plt.close()

    if manifest_rows is None:
        return
    final = pd.DataFrame(manifest_rows).sort_values("scan_theta_deg")
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(baseline["scan_theta_deg"], baseline["hfss_target_ar_db"], marker="o", label="Stage2 HFSS baseline")
    plt.plot(final["scan_theta_deg"], final["final_reconstructed_ar_db"], marker="s", label="Final Vec-AEP codebook")
    plt.axhline(ar_spec_db, linestyle=":", label=f"AR specification = {ar_spec_db:g} dB")
    plt.xlabel("Scan theta (deg)")
    plt.ylabel("Target AR (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_root / "baseline_and_final_ar_vs_theta.png", dpi=240)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(baseline["scan_theta_deg"], baseline["hfss_target_ref_lhcp_psll_db"], marker="o", label="Stage2 HFSS baseline")
    plt.plot(final["scan_theta_deg"], final["final_reconstructed_target_ref_psll_db"], marker="s", label="Final Vec-AEP codebook")
    plt.axhline(sll_spec_db, linestyle=":", label=f"PSLL specification = {sll_spec_db:g} dB")
    plt.xlabel("Scan theta (deg)")
    plt.ylabel("Target-referenced LHCP PSLL (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_root / "baseline_and_final_psll_vs_theta.png", dpi=240)
    plt.close()


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, optimize, and package one Stage3 scan-cut codebook."
    )
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--vecaep-cache", type=Path, required=True)
    parser.add_argument("--hfss-scan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--grid-root", type=Path, default=None)

    parser.add_argument("--scan-phi-deg", type=float, default=0.0)
    parser.add_argument("--scan-theta-start-deg", type=float, default=0.0)
    parser.add_argument("--scan-theta-stop-deg", type=float, default=60.0)
    parser.add_argument("--scan-theta-step-deg", type=float, default=5.0)
    parser.add_argument("--mainlobe-radius-uv", type=float, default=0.13)
    parser.add_argument(
        "--grid-boundary-half-width-uv",
        type=float,
        default=0.01,
        help="0.01 avoids an empty boundary-audit annulus at broadside on the 1deg/2deg grid.",
    )
    parser.add_argument(
        "--grid-target-tolerance-deg",
        type=float,
        default=1e-5,
        help="Numerical target lookup tolerance; much smaller than the 1 deg theta grid step.",
    )
    parser.add_argument("--rebuild-grid-domains", action="store_true")

    parser.add_argument("--ar-spec-db", type=float, default=3.0)
    parser.add_argument("--sll-spec-db", type=float, default=-20.0)
    parser.add_argument("--amplitude-limit-db", type=float, default=3.0)
    parser.add_argument("--phase-limit-deg", type=float, default=20.0)
    parser.add_argument("--target-co-fraction", type=float, default=0.85)
    parser.add_argument("--main-peak-cap-db", type=float, default=0.2)
    parser.add_argument("--power-budget-w", type=float, default=1.0)

    parser.add_argument("--ar-round-step-db", type=float, default=0.05)
    parser.add_argument(
        "--direct-sll-ar-threshold-db",
        type=float,
        default=2.5,
        help=(
            "If the conservative baseline AR max(HFSS, Vec-AEP) is at or below "
            "this threshold, skip minimum-AR and directly minimize PSLL while "
            "capping AR at the baseline value."
        ),
    )
    parser.add_argument("--rho-tol", type=float, default=1e-4)
    parser.add_argument("--ar-max-bisection-iterations", type=int, default=30)
    parser.add_argument("--representative-rho-margin", type=float, default=5e-5)

    parser.add_argument("--sll-loose-db", type=float, default=-15.0)
    parser.add_argument("--sll-strict-db", type=float, default=-30.0)
    parser.add_argument("--sll-tol-db", type=float, default=0.02)
    parser.add_argument("--sll-max-bisection-iterations", type=int, default=20)
    parser.add_argument("--representative-sll-margin-db", type=float, default=0.05)

    parser.add_argument("--solver", default="MOSEK", choices=["AUTO", "MOSEK", "CLARABEL", "SCS"])
    parser.add_argument("--no-solver-fallback", action="store_true", default=True)
    parser.add_argument("--allow-solver-fallback", action="store_false", dest="no_solver_fallback")
    parser.add_argument("--mosek-time-limit-s", type=float, default=60.0)
    parser.add_argument("--margin-accept-tol", type=float, default=2e-6)
    parser.add_argument("--margin-reject-tol", type=float, default=5e-5)
    parser.add_argument("--power-accept-tol", type=float, default=2e-6)
    parser.add_argument("--power-reject-tol", type=float, default=1e-5)
    parser.add_argument("--constraint-abs-tol", type=float, default=2e-6)
    parser.add_argument("--correction-abs-tol", type=float, default=2e-6)
    parser.add_argument("--phase-deg-tol", type=float, default=2e-5)

    parser.add_argument("--reconstruction-ar-tol-db", type=float, default=0.05)
    parser.add_argument("--reconstruction-psll-tol-db", type=float, default=0.10)
    parser.add_argument("--reconstruction-target-tol-db", type=float, default=0.10)
    parser.add_argument("--reconstruction-correlation-min", type=float, default=0.999)
    parser.add_argument("--allow-reconstruction-mismatch", action="store_true")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run-optimization", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    args.project_root = _require_dir(args.project_root, "Stage3 project root")
    args.vecaep_cache = _require_dir(args.vecaep_cache, "Vec-AEP cache")
    args.hfss_scan_root = _require_dir(args.hfss_scan_root, "HFSS scan root")
    args.grid_root = (
        (args.project_root / "cache").resolve()
        if args.grid_root is None
        else Path(args.grid_root).expanduser().resolve()
    )
    args.grid_root.mkdir(parents=True, exist_ok=True)
    for name in [GRID_CORE_NAME, AR_CORE_NAME, SLL_CORE_NAME, "vecaep_residual_closure/vecaep_io.py", "vecaep_residual_closure/vecaep_fullgrid_socp.py"]:
        _require_file(args.project_root / name, f"Required Stage3 script {name}")
    if args.scan_theta_step_deg <= 0.0:
        raise ValueError("Theta step must be positive")
    if args.scan_theta_stop_deg < args.scan_theta_start_deg:
        raise ValueError("Theta stop must not be below theta start")
    if args.mainlobe_radius_uv <= 0.0:
        raise ValueError("Mainlobe radius must be positive")
    if args.grid_boundary_half_width_uv <= 0.0:
        raise ValueError("Grid boundary half width must be positive")
    if args.grid_target_tolerance_deg <= 0.0:
        raise ValueError("Grid target tolerance must be positive")
    if args.ar_spec_db < 0.0:
        raise ValueError("AR specification must be nonnegative")
    if not (0.0 <= args.direct_sll_ar_threshold_db <= args.ar_spec_db):
        raise ValueError(
            "--direct-sll-ar-threshold-db must lie in [0, --ar-spec-db]"
        )
    if args.sll_spec_db >= 0.0:
        raise ValueError("PSLL specification must be negative")
    if args.sll_strict_db >= args.sll_loose_db:
        raise ValueError("Strict PSLL endpoint must be more negative than loose endpoint")
    if args.sll_tol_db <= 0.0 or args.rho_tol <= 0.0:
        raise ValueError("Bisection tolerances must be positive")
    if not (0.0 < args.target_co_fraction <= 1.0):
        raise ValueError("Target LHCP fraction must be in (0,1]")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    try:
        validate_args(args)
        output_root = _prepare_output_root(
            args.output_root, overwrite=args.overwrite, resume=args.resume
        )
    except Exception as exc:
        print(f"ERROR | [SUMMARY] status=FAILED error={exc}", file=sys.stderr)
        return 2

    logger = setup_logger(output_root / PIPELINE_LOG_NAME, args.log_level)
    incomplete = output_root / PIPELINE_INCOMPLETE_NAME
    incomplete.write_text("running\n", encoding="utf-8")
    try:
        logger.info("[VERSION] run_vecaep_residual_closure.py %s", SCRIPT_VERSION)
        logger.info("[ENV] python=%s", sys.executable)
        logger.info("[ROOT] project=%s", args.project_root)
        logger.info("[ROOT] vecaep=%s", args.vecaep_cache)
        logger.info("[ROOT] hfss_scan=%s", args.hfss_scan_root)
        logger.info("[ROOT] output=%s", output_root)
        if args.run_optimization:
            (output_root / PIPELINE_COMPLETE_NAME).unlink(missing_ok=True)
            (output_root / "pipeline_failure.json").unlink(missing_ok=True)

        states = discover_states(
            args.hfss_scan_root,
            scan_phi_deg=args.scan_phi_deg,
            theta_start_deg=args.scan_theta_start_deg,
            theta_stop_deg=args.scan_theta_stop_deg,
            theta_step_deg=args.scan_theta_step_deg,
            grid_root=args.grid_root,
            mainlobe_radius_uv=args.mainlobe_radius_uv,
        )
        logger.info("[DISCOVERY] validated %d scan states", len(states))

        config = {
            "script_version": SCRIPT_VERSION,
            "environment": environment_summary(),
            "inputs": {
                "project_root": str(args.project_root),
                "vecaep_cache": str(args.vecaep_cache),
                "hfss_scan_root": str(args.hfss_scan_root),
                "grid_root": str(args.grid_root),
                "scan_manifest_sha256": sha256_file(args.hfss_scan_root / "scan_manifest.csv"),
                "ar_core_sha256": sha256_file(args.project_root / AR_CORE_NAME),
                "sll_core_sha256": sha256_file(args.project_root / SLL_CORE_NAME),
            },
            "scan": {
                "phi_deg": args.scan_phi_deg,
                "theta_start_deg": args.scan_theta_start_deg,
                "theta_stop_deg": args.scan_theta_stop_deg,
                "theta_step_deg": args.scan_theta_step_deg,
                "state_count": len(states),
            },
            "specifications": {
                "ar_spec_db": args.ar_spec_db,
                "sll_spec_db": args.sll_spec_db,
                "amplitude_limit_db": args.amplitude_limit_db,
                "phase_limit_deg": args.phase_limit_deg,
                "target_co_fraction": args.target_co_fraction,
                "main_peak_cap_db": args.main_peak_cap_db,
                "power_budget_w": args.power_budget_w,
            },
            "strategy": {
                "selection_source": "HFSS full-array baseline",
                "failed_state_rule": "AR > ar_spec OR target-reference LHCP PSLL > sll_spec",
                "direct_sll_ar_threshold_db": args.direct_sll_ar_threshold_db,
                "low_ar_rule": (
                    "if max(HFSS AR, Vec-AEP AR) <= threshold, skip min-AR and "
                    "directly minimize PSLL with AR capped at the baseline guard"
                ),
                "high_ar_rule": (
                    "otherwise minimize AR at fixed PSLL, then ceil certified min-AR "
                    "upper to 0.05 dB and minimize PSLL"
                ),
                "min_ar_fixed_sll_db": args.sll_spec_db,
                "representative_sll_cap_db": args.sll_spec_db,
                "default_final_codeword": "minimum-SLL representative under routed AR policy",
                "public_common_phase_sweep": False,
            },
        }
        write_json(output_root / "pipeline_config.json", config)
        write_json(output_root / "pipeline_plan.json", {"states": [asdict(state) for state in states]})

        for state in states:
            ensure_grid_domain(
                state,
                project_root=args.project_root,
                vecaep_cache=args.vecaep_cache,
                grid_root=args.grid_root,
                mainlobe_radius_uv=args.mainlobe_radius_uv,
                boundary_half_width_uv=args.grid_boundary_half_width_uv,
                target_tolerance_deg=args.grid_target_tolerance_deg,
                rebuild=args.rebuild_grid_domains,
                logger=logger,
            )

        baseline_rows, failed_states = evaluate_baselines(
            states,
            vecaep_cache=args.vecaep_cache,
            output_root=output_root,
            ar_spec_db=args.ar_spec_db,
            sll_spec_db=args.sll_spec_db,
            reconstruction_ar_tolerance_db=args.reconstruction_ar_tol_db,
            reconstruction_psll_tolerance_db=args.reconstruction_psll_tol_db,
            reconstruction_target_tolerance_db=args.reconstruction_target_tol_db,
            reconstruction_correlation_min=args.reconstruction_correlation_min,
            allow_reconstruction_mismatch=args.allow_reconstruction_mismatch,
            logger=logger,
        )
        logger.info(
            "[SELECTION] %d/%d states fail at least one baseline threshold: %s",
            len(failed_states),
            len(states),
            [state.state_tag for state in failed_states],
        )
        create_plots(
            baseline_rows,
            None,
            output_root=output_root,
            ar_spec_db=args.ar_spec_db,
            sll_spec_db=args.sll_spec_db,
            logger=logger,
        )

        if args.preflight_only:
            complete = {
                "status": "preflight_complete",
                "script_version": SCRIPT_VERSION,
                "duration_s": float(time.perf_counter() - started),
                "state_count": len(states),
                "failed_state_count": len(failed_states),
                "failed_states": [state.state_tag for state in failed_states],
                "optimization_run": False,
            }
            write_json(output_root / PIPELINE_COMPLETE_NAME, complete)
            incomplete.unlink(missing_ok=True)
            logger.info("[SUMMARY] preflight complete")
            return 0

        case_results = optimize_failed_states(
            args, failed_states, output_root=output_root, logger=logger
        )
        write_optimization_summaries(case_results, output_root=output_root)
        manifest_rows = build_complete_codebook(
            args,
            states,
            baseline_rows,
            case_results,
            output_root=output_root,
            logger=logger,
        )
        write_complete_cut_summary(manifest_rows, output_root=output_root)
        create_plots(
            baseline_rows,
            manifest_rows,
            output_root=output_root,
            ar_spec_db=args.ar_spec_db,
            sll_spec_db=args.sll_spec_db,
            logger=logger,
        )
        frame = pd.DataFrame(manifest_rows)
        complete = {
            "status": "complete",
            "script_version": SCRIPT_VERSION,
            "duration_s": float(time.perf_counter() - started),
            "state_count": len(states),
            "failed_baseline_state_count": len(failed_states),
            "failed_baseline_states": [state.state_tag for state in failed_states],
            "stage3_corrected_state_count": int(
                (frame["baseline_or_corrected"] == "corrected").sum()
            ),
            "final_dual_spec_pass_count": int(frame["dual_spec_passed"].sum()),
            "all_final_states_dual_spec_passed": bool(frame["dual_spec_passed"].all()),
            "codebook_manifest": str(output_root / "codebook" / "cut_codebook_manifest.csv"),
            "hfss_replay_manifest": str(output_root / "codebook" / "hfss_replay_manifest.csv"),
        }
        write_json(output_root / PIPELINE_COMPLETE_NAME, complete)
        incomplete.unlink(missing_ok=True)
        logger.info(
            "[SUMMARY] complete duration=%.3f s final_pass=%d/%d",
            complete["duration_s"],
            complete["final_dual_spec_pass_count"],
            complete["state_count"],
        )
        return 0
    except Exception as exc:
        duration = float(time.perf_counter() - started)
        logger.error("[SUMMARY] status=FAILED error=%s", exc)
        logger.error("%s", traceback.format_exc())
        write_json(
            output_root / "pipeline_failure.json",
            {
                "status": "failed",
                "script_version": SCRIPT_VERSION,
                "duration_s": duration,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "environment": environment_summary(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
