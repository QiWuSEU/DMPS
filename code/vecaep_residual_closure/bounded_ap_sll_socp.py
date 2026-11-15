#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-3 bounded amplitude/phase minimum-PSLL synthesis.

For one fixed scan state, one fixed target axial-ratio ceiling, and one fixed
per-port amplitude/phase correction budget, this script minimizes target-
referenced LHCP peak sidelobe level (PSLL).

The final weights are written as

    w_n = w0_n * c_n,

where w0 is the immutable Stage-2 baseline codeword and c is the Stage-3
residual correction.  The default hardware model uses the same safe fixed-
tangent convex inner approximation as bounded_ap_ar_socp.py:

    Re(c_n) >= g_min,
    |c_n| <= g_max,
    |Im(c_n)| <= tan(Phi_max) Re(c_n).

For a fixed candidate sidelobe ceiling L_sll (dB), the inner SOCP minimizes the
physical incident-source norm ||w||_2 subject to the fixed AR ceiling, target-
LHCP retention, main-lobe peak guard, full-grid sidelobe ceiling, and bounded
amplitude/phase corrections.  An outer certified bisection searches L_sll over
an explicitly supplied bracket.  Near a feasible/infeasible boundary, the
solver permanently switches to the same always-feasible hard-power-budget
margin model used by the validated minimum-AR script.

The reported feasible PSLL endpoint is an independently audited achievable
upper bound for the fixed convex inner approximation.  It is not a globally
exact optimum of the original nonconvex double-sided annular-sector problem.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import cvxpy as cp  # type: ignore
except Exception as exc:  # pragma: no cover - preflight-only environments
    cp = None  # type: ignore
    CVXPY_IMPORT_ERROR = repr(exc)
else:
    CVXPY_IMPORT_ERROR = None

try:
    from vecaep_residual_closure.vecaep_fullgrid_socp import (
        CERTIFIED_FEASIBLE,
        CERTIFIED_INFEASIBLE,
        SOLVER_ERROR,
        UNCERTAIN,
        TRUSTED_INFEASIBILITY_SOLVERS,
        AuditTolerances,
        SocpInputData,
        ar_db_to_rho,
        environment_summary,
        installed_solver_names,
        load_socp_input_data,
        prepare_output_dir,
        require_cvxpy,
        resolve_solver_sequence,
        safe_db20,
        solver_options,
        wrap_phase_deg,
        write_json,
        write_optimized_weights_csv,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "bounded_ap_sll_socp.py requires the installed "
        "vecaep_residual_closure.vecaep_fullgrid_socp module"
    ) from exc


SCRIPT_VERSION = "1.1.0"
EXPECTED_POINT_COUNT = 16380
EXPECTED_PORT_COUNT = 187

DEFAULT_TARGET_CO_FRACTION = 0.85
DEFAULT_MAIN_PEAK_CAP_DB = 0.2
DEFAULT_POWER_BUDGET_W = 1.0
DEFAULT_AMPLITUDE_LIMIT_DB = 3.0
DEFAULT_PHASE_LIMIT_DEG = 20.0
DEFAULT_SLL_LOOSE_DB = -15.0
DEFAULT_SLL_STRICT_DB = -30.0
DEFAULT_SLL_TOL_DB = 0.05
DEFAULT_MAX_BISECTION_ITERATIONS = 20
DEFAULT_REPRESENTATIVE_SLL_MARGIN_DB = 0.05
DEFAULT_AR_ROUND_STEP_DB = 0.05
DEFAULT_MARGIN_ACCEPT_TOL = 2.0e-6
DEFAULT_MARGIN_REJECT_TOL = 5.0e-5
DEFAULT_MOSEK_TIME_LIMIT_S = 60.0

CONSTRAINT_FIXED_INNER = "fixed-inner"
CONSTRAINT_UNBOUNDED = "unbounded"

INCOMPLETE_MARKER = ".solve_incomplete"
COMPLETE_MARKER = "solution_complete.json"
SUMMARY_NAME = "bounded_ap_sll_summary.json"
SOLUTION_NAME = "bounded_ap_sll_solution.npz"
WEIGHTS_NAME = "optimized_weights.csv"
HISTORY_NAME = "bounded_ap_sll_bisection_history.csv"
LOG_NAME = "stage3_bounded_ap_sll_socp.log"
PREFLIGHT_NAME = "bounded_ap_sll_preflight_summary.json"
GRID_AUDIT_NAME = "grid_field_audit.npz"
TOP_SIDE_NAME = "top_sidelobe_points.csv"

CASE_COMPLETE = "complete"
CASE_LOOSE_ENDPOINT_INFEASIBLE = "loose_endpoint_infeasible"
CASE_LOOSE_ENDPOINT_UNCERTAIN = "loose_endpoint_uncertain"
CASE_STRICT_ENDPOINT_UNCERTAIN = "strict_endpoint_uncertain"
CASE_STRICT_ENDPOINT_STILL_FEASIBLE = "strict_endpoint_still_feasible"

SOLVE_MODE_MIN_POWER = "min_power"
SOLVE_MODE_BUDGET_MARGIN = "budget_margin"

MOSEK_PRIMARY_TOL = 1.0e-8
MOSEK_RETRY_TOL = 1.0e-7
MOSEK_FORCE_DUAL = True


def bounded_solver_options(
    solver_name: str,
    *,
    strict: bool,
    accept_unknown: bool = False,
    save_file: Path | None = None,
    time_limit_s: float | None = None,
) -> dict[str, Any]:
    solver_name = solver_name.upper()
    if solver_name != "MOSEK":
        return solver_options(solver_name, strict=strict)

    tol = MOSEK_PRIMARY_TOL if strict else MOSEK_RETRY_TOL
    params: dict[str, Any] = {
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tol,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tol,
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol,
    }
    if MOSEK_FORCE_DUAL:
        params["MSK_IPAR_INTPNT_SOLVE_FORM"] = "MSK_SOLVE_DUAL"
    options: dict[str, Any] = {"mosek_params": params}
    if accept_unknown:
        options["accept_unknown"] = True
    if time_limit_s is not None and float(time_limit_s) > 0.0:
        params["MSK_DPAR_OPTIMIZER_MAX_TIME"] = float(time_limit_s)
    if save_file is not None:
        options["save_file"] = str(save_file)
    return options


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def setup_logger(log_path: Path | None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("stage3_bounded_ap_sll_socp")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    formatter = logging.Formatter("%(levelname)s | %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def amplitude_bounds_from_db(amplitude_limit_db: float) -> tuple[float, float]:
    amplitude_limit_db = float(amplitude_limit_db)
    if not math.isfinite(amplitude_limit_db) or amplitude_limit_db < 0.0:
        raise ValueError("Amplitude limit must be finite and nonnegative")
    return (
        float(10.0 ** (-amplitude_limit_db / 20.0)),
        float(10.0 ** (+amplitude_limit_db / 20.0)),
    )


def effective_phase_limit_deg(amplitude_limit_db: float, nominal_phase_limit_deg: float) -> float:
    nominal_phase_limit_deg = float(nominal_phase_limit_deg)
    if not math.isfinite(nominal_phase_limit_deg) or not (0.0 <= nominal_phase_limit_deg <= 90.0):
        raise ValueError("Nominal phase limit must lie in [0, 90] degrees")
    g_min, g_max = amplitude_bounds_from_db(amplitude_limit_db)
    ratio = min(1.0, max(0.0, g_min / g_max))
    inner_limit = math.degrees(math.acos(ratio))
    return float(min(nominal_phase_limit_deg, inner_limit))


def round_up_to_step(value: float, step: float, *, epsilon: float = 1.0e-10) -> float:
    value = float(value)
    step = float(step)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Value to round must be finite and nonnegative")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("Round step must be finite and positive")
    units = math.ceil((value - epsilon) / step)
    rounded = units * step
    decimals = max(0, int(math.ceil(-math.log10(step))) + 3) if step < 1.0 else 6
    return float(round(rounded, decimals))


def resolve_ar_limit(
    *,
    explicit_ar_limit_db: float | None,
    reference_summary: Path | None,
    round_step_db: float,
) -> tuple[float, dict[str, Any]]:
    if (explicit_ar_limit_db is None) == (reference_summary is None):
        raise ValueError(
            "Specify exactly one of --ar-limit-db or --ar-reference-summary"
        )

    if explicit_ar_limit_db is not None:
        value = float(explicit_ar_limit_db)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("--ar-limit-db must be finite and nonnegative")
        return value, {
            "mode": "explicit",
            "reference_ar_db": None,
            "round_step_db": None,
            "final_ar_limit_db": value,
            "reference_summary_path": None,
        }

    assert reference_summary is not None
    summary = read_json(reference_summary)
    if summary.get("status") != "complete":
        raise ValueError(
            f"AR reference summary is not complete: status={summary.get('status')!r}"
        )
    bisection = summary.get("bisection")
    if not isinstance(bisection, dict):
        raise ValueError("AR reference summary has no bisection object")
    if not bool(bisection.get("converged")):
        raise ValueError("AR reference bisection did not converge")
    if not bool(bisection.get("upper_is_certified_feasible")):
        raise ValueError("AR reference upper endpoint is not certified feasible")
    reference_ar = float(bisection["ar_upper_db"])
    rounded = round_up_to_step(reference_ar, round_step_db)
    return rounded, {
        "mode": "ceil_reference_upper",
        "reference_ar_db": reference_ar,
        "round_step_db": float(round_step_db),
        "final_ar_limit_db": rounded,
        "reference_summary_path": str(Path(reference_summary).resolve()),
    }


def correction_from_weights(weights: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.complex128).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.complex128).reshape(-1)
    if weights.size != baseline.size:
        raise ValueError("Weight/baseline size mismatch")
    if np.any(np.abs(baseline) <= 1.0e-300):
        raise ValueError("Baseline contains a zero-magnitude port; correction ratio is undefined")
    return np.asarray(weights / baseline, dtype=np.complex128)


def weights_from_correction(correction: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    correction = np.asarray(correction, dtype=np.complex128).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.complex128).reshape(-1)
    if correction.size != baseline.size:
        raise ValueError("Correction/baseline size mismatch")
    return np.asarray(baseline * correction, dtype=np.complex128)


def normalized_response_rows(
    data: SocpInputData,
    indices: np.ndarray | Sequence[int],
    *,
    rhcp: bool,
) -> np.ndarray:
    matrix = data.a_rhcp_raw if rhcp else data.a_lhcp_raw
    rows = np.asarray(matrix[np.asarray(indices, dtype=np.int64), :], dtype=np.complex128)
    return np.asarray(
        data.scale_complex * rows * data.w_baseline[None, :],
        dtype=np.complex128,
    )


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundedApSllConfig:
    constraint_model: str
    amplitude_limit_db: float
    nominal_phase_limit_deg: float
    target_co_fraction: float
    ar_limit_db: float
    rho_limit: float
    main_peak_cap_db: float
    power_budget_w: float

    g_min: float
    g_max: float
    effective_phase_limit_deg: float
    main_peak_factor: float

    @classmethod
    def build(
        cls,
        *,
        constraint_model: str,
        amplitude_limit_db: float,
        phase_limit_deg: float,
        target_co_fraction: float,
        ar_limit_db: float,
        main_peak_cap_db: float,
        power_budget_w: float,
    ) -> "BoundedApSllConfig":
        constraint_model = str(constraint_model)
        if constraint_model not in {CONSTRAINT_FIXED_INNER, CONSTRAINT_UNBOUNDED}:
            raise ValueError(f"Unsupported constraint model: {constraint_model}")
        if not (0.0 < target_co_fraction <= 1.0):
            raise ValueError("target_co_fraction must lie in (0, 1]")
        if not math.isfinite(ar_limit_db) or ar_limit_db < 0.0:
            raise ValueError("ar_limit_db must be finite and nonnegative")
        if main_peak_cap_db < 0.0:
            raise ValueError("main_peak_cap_db must be nonnegative")
        if power_budget_w <= 0.0:
            raise ValueError("power_budget_w must be positive")
        if not (0.0 <= phase_limit_deg <= 90.0):
            raise ValueError("phase_limit_deg must lie in [0, 90]")

        if constraint_model == CONSTRAINT_FIXED_INNER:
            g_min, g_max = amplitude_bounds_from_db(amplitude_limit_db)
            effective = effective_phase_limit_deg(amplitude_limit_db, phase_limit_deg)
        else:
            g_min, g_max, effective = 0.0, float("inf"), 90.0

        return cls(
            constraint_model=constraint_model,
            amplitude_limit_db=float(amplitude_limit_db),
            nominal_phase_limit_deg=float(phase_limit_deg),
            target_co_fraction=float(target_co_fraction),
            ar_limit_db=float(ar_limit_db),
            rho_limit=float(ar_db_to_rho(ar_limit_db)),
            main_peak_cap_db=float(main_peak_cap_db),
            power_budget_w=float(power_budget_w),
            g_min=float(g_min),
            g_max=float(g_max),
            effective_phase_limit_deg=float(effective),
            main_peak_factor=float(10.0 ** (main_peak_cap_db / 20.0)),
        )


@dataclass
class BoundedSllAttempt:
    iteration: int
    sll_test_db: float
    eta_test: float
    solver_name: str
    solver_status: str
    classification: str
    correction_value: np.ndarray | None
    weights_value: np.ndarray | None
    minimum_norm: float | None
    minimum_power_w: float | None
    audit_passed: bool
    audit: dict[str, Any]
    solve_time_s: float
    solver_iterations: int | None
    retry_count: int
    message: str
    solve_mode: str = SOLVE_MODE_MIN_POWER
    margin_value: float | None = None
    margin_accept_tol: float | None = None
    margin_reject_tol: float | None = None
    probe_kind: str = "unspecified"


@dataclass
class BoundedSllBisectionResult:
    converged: bool
    sll_infeasible_db: float
    sll_feasible_db: float
    lower_is_certified_infeasible: bool
    upper_is_certified_feasible: bool
    best_feasible_c: np.ndarray | None
    best_feasible_w: np.ndarray | None
    best_feasible_attempt: BoundedSllAttempt | None
    history: list[BoundedSllAttempt] = field(default_factory=list)
    stop_reason: str = ""
    elapsed_s: float = 0.0
    strict_endpoint_still_feasible: bool = False


@dataclass
class RepresentativeResult:
    success: bool
    sll_limit_db: float
    correction_value: np.ndarray | None
    weights_value: np.ndarray | None
    solver_name: str
    solver_status: str
    distance_to_baseline_c: float | None
    audit: dict[str, Any]
    solve_time_s: float
    message: str


# -----------------------------------------------------------------------------
# Independent physical audit
# -----------------------------------------------------------------------------


def audit_bounded_sll_candidate(
    data: SocpInputData,
    weights: np.ndarray,
    *,
    sll_test_db: float,
    config: BoundedApSllConfig,
    tolerances: AuditTolerances,
    correction_abs_tol: float,
    phase_deg_tol: float,
) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=np.complex128).reshape(-1)
    if weights.size != data.port_count:
        raise ValueError(f"Weight count mismatch: {weights.size} vs {data.port_count}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Candidate weights contain NaN or Inf")

    eta_test = float(10.0 ** (float(sll_test_db) / 20.0))
    correction = correction_from_weights(weights, data.w_baseline)
    correction_abs = np.abs(correction)
    correction_phase_deg = np.asarray(wrap_phase_deg(np.rad2deg(np.angle(correction))))
    correction_real = correction.real

    e_lhcp = np.asarray(data.a_lhcp_raw @ weights, dtype=np.complex128)
    e_rhcp = np.asarray(data.a_rhcp_raw @ weights, dtype=np.complex128)
    target_l = complex(e_lhcp[data.target_index])
    target_r = complex(e_rhcp[data.target_index])
    target_l_abs = abs(target_l)
    target_r_abs = abs(target_r)
    target_fraction = float(target_l_abs / data.field_scale)
    rho_actual = float(target_r_abs / target_l_abs) if target_l_abs > 0.0 else float("inf")
    if rho_actual < 1.0:
        ar_actual = float(20.0 * math.log10((1.0 + rho_actual) / (1.0 - rho_actual)))
    else:
        ar_actual = float("inf")
    power_w = float(np.vdot(weights, weights).real)

    scaled_target = data.scale_complex * target_l
    side_abs = np.abs(e_lhcp[data.side_indices])
    side_ratio = float(np.max(side_abs) / target_l_abs) if target_l_abs > 0.0 else float("inf")
    side_db = safe_db20(side_ratio)
    side_local = int(np.argmax(side_abs)) if side_abs.size else -1
    side_global = int(data.side_indices[side_local]) if side_local >= 0 else -1

    main_indices = np.asarray(data.main_indices, dtype=np.int64)
    main_no_target = main_indices[main_indices != data.target_index]
    if main_no_target.size:
        main_abs = np.abs(e_lhcp[main_no_target])
        main_ratio = float(np.max(main_abs) / target_l_abs) if target_l_abs > 0.0 else float("inf")
        main_local = int(np.argmax(main_abs))
        main_global = int(main_no_target[main_local])
    else:
        main_ratio = 1.0
        main_global = data.target_index
    main_excess_db = safe_db20(main_ratio)

    main_peak_abs = float(max(np.max(np.abs(e_lhcp[main_indices])), target_l_abs))
    peak_ref_side_ratio = float(np.max(side_abs) / main_peak_abs) if main_peak_abs > 0.0 else float("inf")
    peak_ref_side_db = safe_db20(peak_ref_side_ratio)

    target_floor_passed = bool(
        target_fraction >= config.target_co_fraction - tolerances.target_fraction_abs
    )
    ar_passed = bool(rho_actual <= config.rho_limit + tolerances.rho_abs)
    side_passed = bool(side_ratio <= eta_test + tolerances.side_ratio_abs)
    main_passed = bool(main_ratio <= config.main_peak_factor + tolerances.side_ratio_abs)
    phase_anchor_passed = bool(
        abs(scaled_target.imag) <= tolerances.phase_anchor_abs
        and scaled_target.real >= config.target_co_fraction - tolerances.target_fraction_abs
    )
    power_passed = bool(power_w <= config.power_budget_w + tolerances.power_accept_abs)

    if config.constraint_model == CONSTRAINT_FIXED_INNER:
        lower_passed = bool(np.min(correction_abs) >= config.g_min - correction_abs_tol)
        upper_passed = bool(np.max(correction_abs) <= config.g_max + correction_abs_tol)
        inner_tangent_passed = bool(np.min(correction_real) >= config.g_min - correction_abs_tol)
        phase_passed = bool(
            np.max(np.abs(correction_phase_deg))
            <= config.nominal_phase_limit_deg + phase_deg_tol
        )
        amplitude_phase_passed = bool(
            lower_passed and upper_passed and inner_tangent_passed and phase_passed
        )
    else:
        lower_passed = True
        upper_passed = True
        inner_tangent_passed = True
        phase_passed = True
        amplitude_phase_passed = True

    shape_constraints_passed = bool(
        target_floor_passed
        and ar_passed
        and side_passed
        and main_passed
        and phase_anchor_passed
        and amplitude_phase_passed
    )

    amplitude_change_db = 20.0 * np.log10(np.maximum(correction_abs, 1.0e-300))
    lower_active_count = 0
    upper_active_count = 0
    phase_active_count = 0
    if config.constraint_model == CONSTRAINT_FIXED_INNER:
        active_ratio_tol = max(5.0e-5, 10.0 * correction_abs_tol)
        active_phase_tol = max(1.0e-3, 10.0 * phase_deg_tol)
        lower_active_count = int(np.count_nonzero(correction_abs <= config.g_min + active_ratio_tol))
        upper_active_count = int(np.count_nonzero(correction_abs >= config.g_max - active_ratio_tol))
        phase_active_count = int(
            np.count_nonzero(
                np.abs(correction_phase_deg)
                >= max(0.0, config.nominal_phase_limit_deg - active_phase_tol)
            )
        )

    return {
        "power_w": power_w,
        "minimum_norm_from_weights": float(math.sqrt(max(power_w, 0.0))),
        "power_budget_w": config.power_budget_w,
        "power_passed": power_passed,
        "target_lhcp_abs": target_l_abs,
        "target_rhcp_abs": target_r_abs,
        "target_fraction": target_fraction,
        "target_floor_required": config.target_co_fraction,
        "target_floor_passed": target_floor_passed,
        "target_rho_actual": rho_actual,
        "target_rho_limit": config.rho_limit,
        "target_ar_actual_db": ar_actual,
        "target_ar_limit_db": config.ar_limit_db,
        "ar_passed": ar_passed,
        "max_side_ratio": side_ratio,
        "target_ref_lhcp_psll_db": side_db,
        "peak_ref_lhcp_psll_db": peak_ref_side_db,
        "sll_test_db": float(sll_test_db),
        "eta_test": eta_test,
        "sll_passed": side_passed,
        "worst_side_global_index": side_global,
        "worst_side_theta_deg": float(np.rad2deg(data.theta_rad[side_global])) if side_global >= 0 else None,
        "worst_side_phi_deg": float(np.rad2deg(data.phi_rad[side_global])) if side_global >= 0 else None,
        "max_main_ratio": main_ratio,
        "max_main_excess_db": main_excess_db,
        "main_peak_cap_db": config.main_peak_cap_db,
        "main_peak_passed": main_passed,
        "worst_main_global_index": main_global,
        "worst_main_theta_deg": float(np.rad2deg(data.theta_rad[main_global])),
        "worst_main_phi_deg": float(np.rad2deg(data.phi_rad[main_global])),
        "scaled_target_real": float(scaled_target.real),
        "scaled_target_imag": float(scaled_target.imag),
        "phase_anchor_passed": phase_anchor_passed,
        "constraint_model": config.constraint_model,
        "amplitude_limit_db": config.amplitude_limit_db,
        "g_min": _finite_or_none(config.g_min),
        "g_max": _finite_or_none(config.g_max),
        "nominal_phase_limit_deg": config.nominal_phase_limit_deg,
        "effective_phase_limit_deg": config.effective_phase_limit_deg,
        "min_correction_abs": float(np.min(correction_abs)),
        "max_correction_abs": float(np.max(correction_abs)),
        "min_correction_real": float(np.min(correction_real)),
        "min_amplitude_change_db": float(np.min(amplitude_change_db)),
        "max_amplitude_change_db": float(np.max(amplitude_change_db)),
        "max_abs_phase_change_deg": float(np.max(np.abs(correction_phase_deg))),
        "mean_abs_phase_change_deg": float(np.mean(np.abs(correction_phase_deg))),
        "amplitude_lower_passed": lower_passed,
        "amplitude_upper_passed": upper_passed,
        "fixed_inner_tangent_passed": inner_tangent_passed,
        "phase_limit_passed": phase_passed,
        "amplitude_phase_passed": amplitude_phase_passed,
        "amplitude_lower_active_count": lower_active_count,
        "amplitude_upper_active_count": upper_active_count,
        "phase_active_count": phase_active_count,
        "shape_constraints_passed": shape_constraints_passed,
        "feasible_under_budget": bool(shape_constraints_passed and power_passed),
        "max_port_power_w": float(np.max(np.abs(weights) ** 2)),
        "min_port_power_w": float(np.min(np.abs(weights) ** 2)),
        "complex_correlation_to_baseline": float(
            abs(np.vdot(data.w_baseline, weights))
            / max(np.linalg.norm(data.w_baseline) * np.linalg.norm(weights), 1.0e-300)
        ),
    }


def baseline_preflight_audit(
    data: SocpInputData,
    *,
    config: BoundedApSllConfig,
    tolerances: AuditTolerances,
    correction_abs_tol: float,
    phase_deg_tol: float,
) -> dict[str, Any]:
    audit = audit_bounded_sll_candidate(
        data,
        data.w_baseline,
        sll_test_db=-1.0e-9,
        config=config,
        tolerances=tolerances,
        correction_abs_tol=correction_abs_tol,
        phase_deg_tol=phase_deg_tol,
    )
    integrity_passed = bool(
        abs(data.baseline_power_w - 1.0) <= 1.0e-7
        and abs(audit["target_fraction"] - 1.0) <= 1.0e-7
        and audit["phase_anchor_passed"]
        and data.point_count == EXPECTED_POINT_COUNT
        and data.port_count == EXPECTED_PORT_COUNT
    )
    return {
        "data_integrity_passed": integrity_passed,
        "baseline_meets_fixed_ar_hardware_without_sll_test": bool(
            audit["target_floor_passed"]
            and audit["ar_passed"]
            and audit["main_peak_passed"]
            and audit["phase_anchor_passed"]
            and audit["amplitude_phase_passed"]
            and audit["power_passed"]
        ),
        "reference_audit": audit,
        "known_baseline": {
            "power_w": data.baseline_power_w,
            "target_ar_db": data.baseline_ar_db,
            "max_side_db_to_target": data.baseline_side_db,
            "target_rho": data.baseline_rho,
        },
    }


# -----------------------------------------------------------------------------
# Parametric SOCP models
# -----------------------------------------------------------------------------


class BoundedApSllSocpModel:
    """Minimum-physical-power SOCP for a fixed candidate PSLL ceiling."""

    def __init__(self, data: SocpInputData, *, config: BoundedApSllConfig) -> None:
        require_cvxpy()
        self.data = data
        self.config = config
        self.port_count = data.port_count

        self.cr = cp.Variable(self.port_count, name="sll_correction_real")
        self.ci = cp.Variable(self.port_count, name="sll_correction_imag")
        self.eta_parameter = cp.Parameter(nonneg=True, name="sll_eta_test")

        target_l = normalized_response_rows(data, [data.target_index], rhcp=False)[0]
        target_r = normalized_response_rows(data, [data.target_index], rhcp=True)[0]
        side_l = normalized_response_rows(data, data.side_indices, rhcp=False)
        main_indices = np.asarray(data.main_indices, dtype=np.int64)
        self.main_no_target = main_indices[main_indices != data.target_index]
        main_l = normalized_response_rows(data, self.main_no_target, rhcp=False)

        self.target_l_re = target_l.real @ self.cr - target_l.imag @ self.ci
        self.target_l_im = target_l.real @ self.ci + target_l.imag @ self.cr
        self.target_r_re = target_r.real @ self.cr - target_r.imag @ self.ci
        self.target_r_im = target_r.real @ self.ci + target_r.imag @ self.cr
        side_re = side_l.real @ self.cr - side_l.imag @ self.ci
        side_im = side_l.real @ self.ci + side_l.imag @ self.cr
        main_re = main_l.real @ self.cr - main_l.imag @ self.ci
        main_im = main_l.real @ self.ci + main_l.imag @ self.cr

        constraints: list[Any] = [
            self.target_l_im == 0.0,
            self.target_l_re >= config.target_co_fraction,
            cp.norm(cp.hstack([self.target_r_re, self.target_r_im]), 2)
            <= config.rho_limit * self.target_l_re,
            cp.norm(cp.vstack([side_re, side_im]), 2, axis=0)
            <= self.eta_parameter * self.target_l_re,
        ]
        if self.main_no_target.size:
            constraints.append(
                cp.norm(cp.vstack([main_re, main_im]), 2, axis=0)
                <= config.main_peak_factor * self.target_l_re
            )

        if config.constraint_model == CONSTRAINT_FIXED_INNER:
            constraints.append(self.cr >= config.g_min)
            constraints.append(
                cp.norm(cp.vstack([self.cr, self.ci]), 2, axis=0) <= config.g_max
            )
            phi = math.radians(config.nominal_phase_limit_deg)
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            constraints.extend(
                [
                    cos_phi * self.ci <= sin_phi * self.cr,
                    -cos_phi * self.ci <= sin_phi * self.cr,
                ]
            )

        w0_re = data.w_baseline.real
        w0_im = data.w_baseline.imag
        self.wr = cp.multiply(w0_re, self.cr) - cp.multiply(w0_im, self.ci)
        self.wi = cp.multiply(w0_re, self.ci) + cp.multiply(w0_im, self.cr)
        self.stacked_weights = cp.hstack([self.wr, self.wi])

        self.problem = cp.Problem(
            cp.Minimize(cp.norm(self.stacked_weights, 2)),
            constraints,
        )
        if not self.problem.is_dcp():
            raise RuntimeError("The generated bounded minimum-PSLL model is not DCP")
        self.is_dpp = bool(self.problem.is_dpp())

    def set_warm_start(self, correction: np.ndarray | None) -> None:
        if correction is None:
            self.cr.value = None
            self.ci.value = None
            return
        correction = np.asarray(correction, dtype=np.complex128).reshape(-1)
        if correction.size != self.port_count:
            raise ValueError("Warm-start correction count mismatch")
        self.cr.value = correction.real.copy()
        self.ci.value = correction.imag.copy()

    def solve_once(
        self,
        sll_test_db: float,
        *,
        solver_name: str,
        warm_start_c: np.ndarray | None,
        warm_start_enabled: bool,
        strict_solver_tolerances: bool,
        accept_unknown: bool,
        attempt_label: str,
        solver_verbose: bool,
        tolerances: AuditTolerances,
        correction_abs_tol: float,
        phase_deg_tol: float,
        iteration: int,
        retry_count: int,
        save_file: Path | None = None,
        time_limit_s: float | None = None,
    ) -> BoundedSllAttempt:
        solver_name = solver_name.upper()
        sll_test_db = float(sll_test_db)
        eta_test = float(10.0 ** (sll_test_db / 20.0))
        self.eta_parameter.value = eta_test
        self.set_warm_start(warm_start_c if warm_start_enabled else None)
        started = time.perf_counter()
        try:
            value = self.problem.solve(
                solver=solver_name,
                verbose=bool(solver_verbose),
                warm_start=bool(warm_start_enabled),
                **bounded_solver_options(
                    solver_name,
                    strict=bool(strict_solver_tolerances),
                    accept_unknown=bool(accept_unknown),
                    save_file=save_file,
                    time_limit_s=time_limit_s,
                ),
            )
            elapsed = time.perf_counter() - started
        except Exception as exc:
            elapsed = time.perf_counter() - started
            return BoundedSllAttempt(
                iteration=iteration,
                sll_test_db=sll_test_db,
                eta_test=eta_test,
                solver_name=solver_name,
                solver_status="exception",
                classification=SOLVER_ERROR,
                correction_value=None,
                weights_value=None,
                minimum_norm=None,
                minimum_power_w=None,
                audit_passed=False,
                audit={},
                solve_time_s=elapsed,
                solver_iterations=None,
                retry_count=retry_count,
                message=f"{attempt_label}: {type(exc).__name__}: {exc}",
                solve_mode=SOLVE_MODE_MIN_POWER,
            )

        status = str(self.problem.status or "unknown").lower()
        stats = self.problem.solver_stats
        iterations = getattr(stats, "num_iters", None)
        correction = None
        weights = None
        minimum_norm = None
        minimum_power = None
        audit: dict[str, Any] = {}
        audit_passed = False
        classification = UNCERTAIN
        message = ""

        if status in {"optimal", "optimal_inaccurate"}:
            if self.cr.value is None or self.ci.value is None:
                message = "Solver returned an optimal status without primal correction values"
            else:
                correction = np.asarray(self.cr.value, dtype=np.float64) + 1j * np.asarray(
                    self.ci.value, dtype=np.float64
                )
                correction = np.asarray(correction, dtype=np.complex128)
                weights = weights_from_correction(correction, self.data.w_baseline)
                minimum_norm = float(np.linalg.norm(weights))
                minimum_power = float(minimum_norm**2)
                audit = audit_bounded_sll_candidate(
                    self.data,
                    weights,
                    sll_test_db=sll_test_db,
                    config=self.config,
                    tolerances=tolerances,
                    correction_abs_tol=correction_abs_tol,
                    phase_deg_tol=phase_deg_tol,
                )
                audit_passed = bool(audit["shape_constraints_passed"])

                if audit_passed and minimum_power <= self.config.power_budget_w + tolerances.power_accept_abs:
                    classification = CERTIFIED_FEASIBLE
                    message = f"{attempt_label}: Audited feasible witness within the physical power budget"
                elif (
                    status == "optimal"
                    and audit_passed
                    and minimum_power > self.config.power_budget_w + tolerances.power_reject_abs
                    and solver_name in TRUSTED_INFEASIBILITY_SOLVERS
                ):
                    classification = CERTIFIED_INFEASIBLE
                    message = f"{attempt_label}: Optimal minimum-power solution exceeds the physical power budget"
                else:
                    classification = UNCERTAIN
                    message = (
                        f"{attempt_label}: Optimal result lies in a numerical gray zone, "
                        "uses an untrusted lower-bound solver, or fails independent audit"
                    )
        elif status == "infeasible":
            if solver_name in TRUSTED_INFEASIBILITY_SOLVERS:
                classification = CERTIFIED_INFEASIBLE
                message = f"{attempt_label}: Trusted conic solver certified the unbudgeted inner model infeasible"
            else:
                classification = UNCERTAIN
                message = f"{attempt_label}: SCS infeasibility is not used as a certified lower bound"
        elif status in {"infeasible_inaccurate", "unbounded_inaccurate", "user_limit"}:
            classification = UNCERTAIN
            message = f"{attempt_label}: Noncertifying solver status: {status}"
        else:
            classification = UNCERTAIN
            message = f"{attempt_label}: Unhandled solver status: {status}; objective={value!r}"

        return BoundedSllAttempt(
            iteration=iteration,
            sll_test_db=sll_test_db,
            eta_test=eta_test,
            solver_name=solver_name,
            solver_status=status,
            classification=classification,
            correction_value=correction,
            weights_value=weights,
            minimum_norm=minimum_norm,
            minimum_power_w=minimum_power,
            audit_passed=audit_passed,
            audit=audit,
            solve_time_s=elapsed,
            solver_iterations=int(iterations) if iterations is not None else None,
            retry_count=retry_count,
            message=message,
            solve_mode=SOLVE_MODE_MIN_POWER,
        )


class BoundedSllBudgetMarginSocpModel:
    """Always-feasible hard-budget margin SOCP near the PSLL boundary."""

    def __init__(self, data: SocpInputData, *, config: BoundedApSllConfig) -> None:
        require_cvxpy()
        self.data = data
        self.config = config
        self.port_count = data.port_count

        self.cr = cp.Variable(self.port_count, name="sll_margin_correction_real")
        self.ci = cp.Variable(self.port_count, name="sll_margin_correction_imag")
        self.margin = cp.Variable(nonneg=True, name="sll_normalized_field_margin")
        self.eta_parameter = cp.Parameter(nonneg=True, name="sll_margin_eta_test")

        target_l = normalized_response_rows(data, [data.target_index], rhcp=False)[0]
        target_r = normalized_response_rows(data, [data.target_index], rhcp=True)[0]
        side_l = normalized_response_rows(data, data.side_indices, rhcp=False)
        main_indices = np.asarray(data.main_indices, dtype=np.int64)
        self.main_no_target = main_indices[main_indices != data.target_index]
        main_l = normalized_response_rows(data, self.main_no_target, rhcp=False)

        self.target_l_re = target_l.real @ self.cr - target_l.imag @ self.ci
        self.target_l_im = target_l.real @ self.ci + target_l.imag @ self.cr
        target_r_re = target_r.real @ self.cr - target_r.imag @ self.ci
        target_r_im = target_r.real @ self.ci + target_r.imag @ self.cr
        side_re = side_l.real @ self.cr - side_l.imag @ self.ci
        side_im = side_l.real @ self.ci + side_l.imag @ self.cr
        main_re = main_l.real @ self.cr - main_l.imag @ self.ci
        main_im = main_l.real @ self.ci + main_l.imag @ self.cr

        w0_re = data.w_baseline.real
        w0_im = data.w_baseline.imag
        self.wr = cp.multiply(w0_re, self.cr) - cp.multiply(w0_im, self.ci)
        self.wi = cp.multiply(w0_re, self.ci) + cp.multiply(w0_im, self.cr)
        self.stacked_weights = cp.hstack([self.wr, self.wi])

        constraints: list[Any] = [
            self.target_l_im == 0.0,
            self.target_l_re >= 0.0,
            self.target_l_re + self.margin >= config.target_co_fraction,
            cp.norm(cp.hstack([target_r_re, target_r_im]), 2)
            <= config.rho_limit * self.target_l_re + self.margin,
            cp.norm(cp.vstack([side_re, side_im]), 2, axis=0)
            <= self.eta_parameter * self.target_l_re + self.margin,
            cp.norm(self.stacked_weights, 2) <= math.sqrt(config.power_budget_w),
        ]
        if self.main_no_target.size:
            constraints.append(
                cp.norm(cp.vstack([main_re, main_im]), 2, axis=0)
                <= config.main_peak_factor * self.target_l_re + self.margin
            )

        if config.constraint_model == CONSTRAINT_FIXED_INNER:
            constraints.append(self.cr >= config.g_min)
            constraints.append(
                cp.norm(cp.vstack([self.cr, self.ci]), 2, axis=0) <= config.g_max
            )
            phi = math.radians(config.nominal_phase_limit_deg)
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            constraints.extend(
                [
                    cos_phi * self.ci <= sin_phi * self.cr,
                    -cos_phi * self.ci <= sin_phi * self.cr,
                ]
            )

        self.problem = cp.Problem(cp.Minimize(self.margin), constraints)
        if not self.problem.is_dcp():
            raise RuntimeError("The generated hard-budget minimum-PSLL margin model is not DCP")
        self.is_dpp = bool(self.problem.is_dpp())

    def set_warm_start(self, correction: np.ndarray | None, margin_value: float | None = None) -> None:
        if correction is None:
            self.cr.value = None
            self.ci.value = None
        else:
            correction = np.asarray(correction, dtype=np.complex128).reshape(-1)
            if correction.size != self.port_count:
                raise ValueError("Margin warm-start correction count mismatch")
            self.cr.value = correction.real.copy()
            self.ci.value = correction.imag.copy()
        self.margin.value = None if margin_value is None else max(0.0, float(margin_value))

    def solve_once(
        self,
        sll_test_db: float,
        *,
        solver_name: str,
        warm_start_c: np.ndarray | None,
        warm_start_enabled: bool,
        strict_solver_tolerances: bool,
        accept_unknown: bool,
        attempt_label: str,
        solver_verbose: bool,
        tolerances: AuditTolerances,
        correction_abs_tol: float,
        phase_deg_tol: float,
        iteration: int,
        retry_count: int,
        margin_accept_tol: float,
        margin_reject_tol: float,
        time_limit_s: float | None,
        save_file: Path | None = None,
    ) -> BoundedSllAttempt:
        solver_name = solver_name.upper()
        sll_test_db = float(sll_test_db)
        eta_test = float(10.0 ** (sll_test_db / 20.0))
        self.eta_parameter.value = eta_test
        self.set_warm_start(
            warm_start_c if warm_start_enabled else None,
            margin_value=0.0 if warm_start_enabled else None,
        )
        started = time.perf_counter()
        try:
            value = self.problem.solve(
                solver=solver_name,
                verbose=bool(solver_verbose),
                warm_start=bool(warm_start_enabled),
                **bounded_solver_options(
                    solver_name,
                    strict=bool(strict_solver_tolerances),
                    accept_unknown=bool(accept_unknown),
                    save_file=save_file,
                    time_limit_s=time_limit_s,
                ),
            )
            elapsed = time.perf_counter() - started
        except Exception as exc:
            elapsed = time.perf_counter() - started
            return BoundedSllAttempt(
                iteration=iteration,
                sll_test_db=sll_test_db,
                eta_test=eta_test,
                solver_name=solver_name,
                solver_status="exception",
                classification=SOLVER_ERROR,
                correction_value=None,
                weights_value=None,
                minimum_norm=None,
                minimum_power_w=None,
                audit_passed=False,
                audit={},
                solve_time_s=elapsed,
                solver_iterations=None,
                retry_count=retry_count,
                message=f"{attempt_label}: {type(exc).__name__}: {exc}",
                solve_mode=SOLVE_MODE_BUDGET_MARGIN,
                margin_value=None,
                margin_accept_tol=margin_accept_tol,
                margin_reject_tol=margin_reject_tol,
            )

        status = str(self.problem.status or "unknown").lower()
        stats = self.problem.solver_stats
        iterations = getattr(stats, "num_iters", None)
        margin_value = None
        correction = None
        weights = None
        witness_norm = None
        witness_power = None
        audit: dict[str, Any] = {}
        audit_passed = False
        classification = UNCERTAIN
        message = ""

        if self.margin.value is not None:
            margin_value = float(np.asarray(self.margin.value).reshape(()))

        if status in {"optimal", "optimal_inaccurate"}:
            if self.cr.value is None or self.ci.value is None or margin_value is None:
                message = f"{attempt_label}: Solver returned no usable primal margin witness"
            else:
                correction = np.asarray(self.cr.value, dtype=np.float64) + 1j * np.asarray(
                    self.ci.value, dtype=np.float64
                )
                correction = np.asarray(correction, dtype=np.complex128)
                weights = weights_from_correction(correction, self.data.w_baseline)
                witness_norm = float(np.linalg.norm(weights))
                witness_power = float(witness_norm**2)
                audit = audit_bounded_sll_candidate(
                    self.data,
                    weights,
                    sll_test_db=sll_test_db,
                    config=self.config,
                    tolerances=tolerances,
                    correction_abs_tol=correction_abs_tol,
                    phase_deg_tol=phase_deg_tol,
                )
                audit["margin_value"] = margin_value
                audit["margin_accept_tol"] = margin_accept_tol
                audit["margin_reject_tol"] = margin_reject_tol
                audit_passed = bool(audit["feasible_under_budget"])

                if audit_passed:
                    classification = CERTIFIED_FEASIBLE
                    message = (
                        f"{attempt_label}: Hard-budget margin model returned an independently "
                        "audited zero-violation witness"
                    )
                elif (
                    status == "optimal"
                    and solver_name in TRUSTED_INFEASIBILITY_SOLVERS
                    and margin_value >= margin_reject_tol
                ):
                    classification = CERTIFIED_INFEASIBLE
                    message = (
                        f"{attempt_label}: Optimal normalized field margin {margin_value:.9g} "
                        f"exceeds rejection threshold {margin_reject_tol:.9g}; no zero-margin "
                        "solution exists within the physical power budget"
                    )
                elif status == "optimal" and margin_value <= margin_accept_tol:
                    classification = UNCERTAIN
                    message = (
                        f"{attempt_label}: Margin is numerically near zero but the original "
                        "constraints failed the independent audit"
                    )
                else:
                    classification = UNCERTAIN
                    message = (
                        f"{attempt_label}: Margin result lies in the numerical gray zone or "
                        "has a noncertifying solver status"
                    )
        elif status == "infeasible":
            classification = SOLVER_ERROR
            message = f"{attempt_label}: Hard-budget margin model unexpectedly reported infeasible"
        elif status in {"infeasible_inaccurate", "unbounded_inaccurate", "user_limit"}:
            classification = UNCERTAIN
            message = f"{attempt_label}: Noncertifying margin-model status: {status}"
        else:
            classification = UNCERTAIN
            message = f"{attempt_label}: Unhandled margin-model status: {status}; objective={value!r}"

        return BoundedSllAttempt(
            iteration=iteration,
            sll_test_db=sll_test_db,
            eta_test=eta_test,
            solver_name=solver_name,
            solver_status=status,
            classification=classification,
            correction_value=correction,
            weights_value=weights,
            minimum_norm=witness_norm,
            minimum_power_w=witness_power,
            audit_passed=audit_passed,
            audit=audit,
            solve_time_s=elapsed,
            solver_iterations=int(iterations) if iterations is not None else None,
            retry_count=retry_count,
            message=message,
            solve_mode=SOLVE_MODE_BUDGET_MARGIN,
            margin_value=margin_value,
            margin_accept_tol=margin_accept_tol,
            margin_reject_tol=margin_reject_tol,
        )


class HybridSllSolver:
    """Min-power first, then permanent hard-budget margin mode near a boundary."""

    def __init__(
        self,
        min_power_model: BoundedApSllSocpModel,
        *,
        solver_sequence: Sequence[str],
        solver_verbose: bool,
        tolerances: AuditTolerances,
        correction_abs_tol: float,
        phase_deg_tol: float,
        margin_accept_tol: float,
        margin_reject_tol: float,
        mosek_time_limit_s: float,
        cross_solver_boundary_fallback: bool,
        logger: logging.Logger,
    ) -> None:
        if not solver_sequence:
            raise ValueError("At least one conic solver is required")
        self.min_power_model: BoundedApSllSocpModel | None = min_power_model
        self.data = min_power_model.data
        self.config = min_power_model.config
        self.solver_sequence = [str(name).upper() for name in solver_sequence]
        self.primary_solver = self.solver_sequence[0]
        self.solver_verbose = bool(solver_verbose)
        self.tolerances = tolerances
        self.correction_abs_tol = float(correction_abs_tol)
        self.phase_deg_tol = float(phase_deg_tol)
        self.margin_accept_tol = float(margin_accept_tol)
        self.margin_reject_tol = float(margin_reject_tol)
        self.mosek_time_limit_s = float(mosek_time_limit_s)
        self.cross_solver_boundary_fallback = bool(cross_solver_boundary_fallback)
        self.logger = logger
        self.margin_model: BoundedSllBudgetMarginSocpModel | None = None
        self.margin_only = False
        self.margin_activation_reason: str | None = None

    def _get_margin_model(self) -> BoundedSllBudgetMarginSocpModel:
        if self.margin_model is None:
            self.logger.info("[MODEL] building always-feasible hard-budget margin fallback")
            self.margin_model = BoundedSllBudgetMarginSocpModel(self.data, config=self.config)
            self.logger.info(
                "[MODEL] margin DPP=%s side=%d main_without_target=%d",
                self.margin_model.is_dpp,
                self.data.side_indices.size,
                self.margin_model.main_no_target.size,
            )
            if self.margin_only and self.min_power_model is not None:
                self.min_power_model = None
                gc.collect()
                self.logger.info("[MODEL] released min-power model after boundary-mode activation")
        return self.margin_model

    def _log_attempt(self, attempt: BoundedSllAttempt, label: str) -> None:
        self.logger.info(
            "[ATTEMPT] iter=%s SLL=%.6f dB mode=%s label=%s solver=%s status=%s "
            "class=%s time=%.3fs P=%s eps=%s",
            attempt.iteration,
            attempt.sll_test_db,
            attempt.solve_mode,
            label,
            attempt.solver_name,
            attempt.solver_status,
            attempt.classification,
            attempt.solve_time_s,
            "n/a" if attempt.minimum_power_w is None else f"{attempt.minimum_power_w:.9g}",
            "n/a" if attempt.margin_value is None else f"{attempt.margin_value:.9g}",
        )

    def solve(
        self,
        sll_test_db: float,
        *,
        warm_start_c: np.ndarray | None,
        iteration: int,
    ) -> BoundedSllAttempt:
        messages: list[str] = []
        retry_count = 0

        if not self.margin_only:
            if self.min_power_model is None:
                raise RuntimeError("Min-power model was released before margin-mode activation")
            primary = self.min_power_model.solve_once(
                sll_test_db,
                solver_name=self.primary_solver,
                warm_start_c=warm_start_c,
                warm_start_enabled=warm_start_c is not None,
                strict_solver_tolerances=True,
                accept_unknown=False,
                attempt_label="primary_min_power",
                solver_verbose=self.solver_verbose,
                tolerances=self.tolerances,
                correction_abs_tol=self.correction_abs_tol,
                phase_deg_tol=self.phase_deg_tol,
                iteration=iteration,
                retry_count=retry_count,
                time_limit_s=self.mosek_time_limit_s if self.primary_solver == "MOSEK" else None,
            )
            self._log_attempt(primary, "primary_min_power")
            messages.append(
                f"{primary.solver_name}/primary_min_power:{primary.solver_status}:"
                f"{primary.classification}:{primary.message}"
            )
            if primary.classification in {CERTIFIED_FEASIBLE, CERTIFIED_INFEASIBLE}:
                primary.message = " | ".join(messages)
                return primary
            self.margin_only = True
            self.margin_activation_reason = (
                f"SLL={sll_test_db:.12g} dB, status={primary.solver_status}, "
                f"classification={primary.classification}"
            )
            self.logger.warning(
                "[BOUNDARY] min-power model became noncertifying; switching this case "
                "permanently to hard-budget margin mode (%s)",
                self.margin_activation_reason,
            )
            retry_count += 1

        margin_model = self._get_margin_model()
        margin_profiles = [("budget_margin_strict_cold", self.primary_solver, True, False)]
        if self.primary_solver == "MOSEK":
            margin_profiles.append(("budget_margin_relaxed_accept_unknown", "MOSEK", False, True))
        if self.cross_solver_boundary_fallback:
            for solver_name in self.solver_sequence[1:]:
                margin_profiles.append(
                    (f"budget_margin_{solver_name.lower()}_fallback", solver_name, False, False)
                )

        last_attempt: BoundedSllAttempt | None = None
        for label, solver_name, strict_tols, accept_unknown in margin_profiles:
            attempt = margin_model.solve_once(
                sll_test_db,
                solver_name=solver_name,
                warm_start_c=warm_start_c,
                warm_start_enabled=False,
                strict_solver_tolerances=strict_tols,
                accept_unknown=accept_unknown,
                attempt_label=label,
                solver_verbose=self.solver_verbose,
                tolerances=self.tolerances,
                correction_abs_tol=self.correction_abs_tol,
                phase_deg_tol=self.phase_deg_tol,
                iteration=iteration,
                retry_count=retry_count,
                margin_accept_tol=self.margin_accept_tol,
                margin_reject_tol=self.margin_reject_tol,
                time_limit_s=self.mosek_time_limit_s if solver_name == "MOSEK" else None,
            )
            self._log_attempt(attempt, label)
            messages.append(
                f"{attempt.solver_name}/{label}:{attempt.solver_status}:"
                f"{attempt.classification}:{attempt.message}"
            )
            last_attempt = attempt
            if attempt.classification in {CERTIFIED_FEASIBLE, CERTIFIED_INFEASIBLE}:
                attempt.message = " | ".join(messages)
                return attempt
            retry_count += 1

        assert last_attempt is not None
        last_attempt.message = " | ".join(messages)
        return last_attempt


# -----------------------------------------------------------------------------
# Certified PSLL search
# -----------------------------------------------------------------------------


def run_sll_gate_and_bisection(
    sll_solver: HybridSllSolver,
    *,
    sll_loose_db: float,
    sll_strict_db: float,
    sll_tolerance_db: float,
    max_iterations: int,
    logger: logging.Logger,
) -> tuple[BoundedSllAttempt, BoundedSllAttempt | None, BoundedSllBisectionResult | None]:
    started = time.perf_counter()
    baseline_c = np.ones(sll_solver.data.port_count, dtype=np.complex128)

    loose = sll_solver.solve(sll_loose_db, warm_start_c=baseline_c, iteration=-1)
    loose.probe_kind = "sll_loose_gate"
    logger.info(
        "[GATE] loose=%.6f dB class=%s mode=%s solver=%s status=%s P=%s eps=%s",
        sll_loose_db,
        loose.classification,
        loose.solve_mode,
        loose.solver_name,
        loose.solver_status,
        "n/a" if loose.minimum_power_w is None else f"{loose.minimum_power_w:.9g}",
        "n/a" if loose.margin_value is None else f"{loose.margin_value:.9g}",
    )
    if loose.classification != CERTIFIED_FEASIBLE:
        return loose, None, None
    if loose.correction_value is None or loose.weights_value is None:
        raise RuntimeError("Loose endpoint was certified feasible without a retained witness")

    best_c = np.asarray(loose.correction_value, dtype=np.complex128).copy()
    best_w = np.asarray(loose.weights_value, dtype=np.complex128).copy()
    best_attempt: BoundedSllAttempt | None = loose
    warm_c = best_c.copy()
    history: list[BoundedSllAttempt] = [loose]

    strict = sll_solver.solve(sll_strict_db, warm_start_c=warm_c, iteration=0)
    strict.probe_kind = "sll_strict_gate"
    history.append(strict)
    logger.info(
        "[GATE] strict=%.6f dB class=%s mode=%s solver=%s status=%s P=%s eps=%s",
        sll_strict_db,
        strict.classification,
        strict.solve_mode,
        strict.solver_name,
        strict.solver_status,
        "n/a" if strict.minimum_power_w is None else f"{strict.minimum_power_w:.9g}",
        "n/a" if strict.margin_value is None else f"{strict.margin_value:.9g}",
    )

    if strict.classification == CERTIFIED_FEASIBLE:
        assert strict.correction_value is not None and strict.weights_value is not None
        elapsed = time.perf_counter() - started
        return loose, strict, BoundedSllBisectionResult(
            converged=True,
            sll_infeasible_db=sll_strict_db,
            sll_feasible_db=sll_strict_db,
            lower_is_certified_infeasible=False,
            upper_is_certified_feasible=True,
            best_feasible_c=np.asarray(strict.correction_value, dtype=np.complex128),
            best_feasible_w=np.asarray(strict.weights_value, dtype=np.complex128),
            best_feasible_attempt=strict,
            history=history,
            stop_reason="strict search endpoint is still certified feasible",
            elapsed_s=elapsed,
            strict_endpoint_still_feasible=True,
        )
    if strict.classification != CERTIFIED_INFEASIBLE:
        return loose, strict, None

    sll_infeasible = float(sll_strict_db)
    sll_feasible = float(sll_loose_db)
    uncertain_slls: list[float] = []
    attempted_slls: list[float] = [float(sll_loose_db), float(sll_strict_db)]

    def active_gray_points() -> list[float]:
        eps = max(1.0e-12, 1.0e-12 * max(1.0, abs(sll_infeasible), abs(sll_feasible)))
        return sorted(
            {
                float(value)
                for value in uncertain_slls
                if sll_infeasible + eps < float(value) < sll_feasible - eps
            }
        )

    def choose_probe() -> tuple[float, str]:
        gray = active_gray_points()
        if not gray:
            return 0.5 * (sll_infeasible + sll_feasible), "midpoint_db"
        gray_low = gray[0]
        gray_high = gray[-1]
        left_gap = max(0.0, gray_low - sll_infeasible)
        right_gap = max(0.0, sll_feasible - gray_high)
        if left_gap >= right_gap and left_gap > 0.0:
            return 0.5 * (sll_infeasible + gray_low), "gray_strict_refine"
        if right_gap > 0.0:
            return 0.5 * (gray_high + sll_feasible), "gray_loose_refine"
        return 0.5 * (sll_infeasible + sll_feasible), "gray_span_probe"

    converged = False
    stop_reason = "maximum certified-refinement iterations reached"
    for iteration in range(1, max_iterations + 1):
        width_before = sll_feasible - sll_infeasible
        if width_before <= sll_tolerance_db:
            converged = True
            stop_reason = "certified PSLL interval tolerance reached"
            break

        sll_probe, probe_kind = choose_probe()
        spacing_tol = max(1.0e-12, 1.0e-12 * max(1.0, abs(sll_probe)))
        if any(abs(sll_probe - prior) <= spacing_tol for prior in attempted_slls):
            gray = active_gray_points()
            alternatives: list[tuple[float, str]] = []
            if gray:
                alternatives.extend(
                    [
                        (0.5 * (sll_infeasible + gray[0]), "gray_strict_refine_alt"),
                        (0.5 * (gray[-1] + sll_feasible), "gray_loose_refine_alt"),
                    ]
                )
            alternatives.extend(
                [
                    (sll_infeasible + 0.25 * width_before, "strict_quarter_alt"),
                    (sll_feasible - 0.25 * width_before, "loose_quarter_alt"),
                ]
            )
            replacement = next(
                (
                    (candidate, label)
                    for candidate, label in alternatives
                    if sll_infeasible < candidate < sll_feasible
                    and not any(abs(candidate - prior) <= spacing_tol for prior in attempted_slls)
                ),
                None,
            )
            if replacement is None:
                stop_reason = (
                    "gray-zone refinement exhausted distinct probe points; "
                    f"certified interval width={width_before:.12g} dB"
                )
                break
            sll_probe, probe_kind = replacement

        attempt = sll_solver.solve(sll_probe, warm_start_c=warm_c, iteration=iteration)
        attempt.probe_kind = probe_kind
        history.append(attempt)
        attempted_slls.append(float(sll_probe))
        logger.info(
            "[BISECTION] iter=%02d probe=%s SLL=%.6f dB class=%s mode=%s solver=%s "
            "P=%s eps=%s bracket=[%.6f, %.6f] dB",
            iteration,
            probe_kind,
            sll_probe,
            attempt.classification,
            attempt.solve_mode,
            attempt.solver_name,
            "n/a" if attempt.minimum_power_w is None else f"{attempt.minimum_power_w:.9g}",
            "n/a" if attempt.margin_value is None else f"{attempt.margin_value:.9g}",
            sll_infeasible,
            sll_feasible,
        )

        if attempt.classification == CERTIFIED_FEASIBLE:
            sll_feasible = sll_probe
            assert attempt.correction_value is not None and attempt.weights_value is not None
            best_c = np.asarray(attempt.correction_value, dtype=np.complex128).copy()
            best_w = np.asarray(attempt.weights_value, dtype=np.complex128).copy()
            best_attempt = attempt
            warm_c = best_c.copy()
        elif attempt.classification == CERTIFIED_INFEASIBLE:
            sll_infeasible = sll_probe
        else:
            uncertain_slls.append(float(sll_probe))
            logger.warning(
                "[GRAY] retained noncertifying SLL=%.9f dB; certified bracket remains "
                "[%.9f, %.9f] dB, active_gray=[%s]",
                sll_probe,
                sll_infeasible,
                sll_feasible,
                ", ".join(f"{value:.9f}" for value in active_gray_points()),
            )

        uncertain_slls = active_gray_points()
        if sll_feasible - sll_infeasible <= sll_tolerance_db:
            converged = True
            stop_reason = (
                "certified PSLL interval tolerance reached after gray-zone refinement"
                if uncertain_slls
                else "certified PSLL interval tolerance reached"
            )
            break
    else:
        if sll_feasible - sll_infeasible <= sll_tolerance_db:
            converged = True
            stop_reason = "certified PSLL interval tolerance reached at iteration limit"

    if not converged and uncertain_slls:
        gray = active_gray_points()
        gray_span = (gray[-1] - gray[0]) if len(gray) >= 2 else 0.0
        stop_reason = (
            f"{stop_reason}; certified bracket=[{sll_infeasible:.12g}, "
            f"{sll_feasible:.12g}] dB, width={sll_feasible - sll_infeasible:.12g} dB, "
            f"gray_count={len(gray)}, gray_span={gray_span:.12g} dB"
        )

    elapsed = time.perf_counter() - started
    return loose, strict, BoundedSllBisectionResult(
        converged=converged,
        sll_infeasible_db=sll_infeasible,
        sll_feasible_db=sll_feasible,
        lower_is_certified_infeasible=True,
        upper_is_certified_feasible=True,
        best_feasible_c=best_c,
        best_feasible_w=best_w,
        best_feasible_attempt=best_attempt,
        history=history,
        stop_reason=stop_reason,
        elapsed_s=elapsed,
        strict_endpoint_still_feasible=False,
    )


# -----------------------------------------------------------------------------
# Representative solution
# -----------------------------------------------------------------------------


def resolve_representative_sll_limit(
    feasible_sll_db: float,
    margin_db: float,
    cap_db: float | None,
) -> tuple[float, float]:
    """Return (final_limit, uncapped_limit) for representative-codeword synthesis."""
    uncapped = float(feasible_sll_db + max(0.0, margin_db))
    final = uncapped if cap_db is None else min(uncapped, float(cap_db))
    return float(final), float(uncapped)


def solve_representative_correction(
    data: SocpInputData,
    *,
    config: BoundedApSllConfig,
    sll_limit_db: float,
    solver_sequence: Sequence[str],
    warm_start_c: np.ndarray,
    solver_verbose: bool,
    tolerances: AuditTolerances,
    correction_abs_tol: float,
    phase_deg_tol: float,
    mosek_time_limit_s: float,
) -> RepresentativeResult:
    require_cvxpy()
    count = data.port_count
    cr = cp.Variable(count, name="rep_sll_correction_real")
    ci = cp.Variable(count, name="rep_sll_correction_imag")
    eta_limit = float(10.0 ** (sll_limit_db / 20.0))

    target_l = normalized_response_rows(data, [data.target_index], rhcp=False)[0]
    target_r = normalized_response_rows(data, [data.target_index], rhcp=True)[0]
    side_l = normalized_response_rows(data, data.side_indices, rhcp=False)
    main_indices = np.asarray(data.main_indices, dtype=np.int64)
    main_no_target = main_indices[main_indices != data.target_index]
    main_l = normalized_response_rows(data, main_no_target, rhcp=False)

    target_l_re = target_l.real @ cr - target_l.imag @ ci
    target_l_im = target_l.real @ ci + target_l.imag @ cr
    target_r_re = target_r.real @ cr - target_r.imag @ ci
    target_r_im = target_r.real @ ci + target_r.imag @ cr
    side_re = side_l.real @ cr - side_l.imag @ ci
    side_im = side_l.real @ ci + side_l.imag @ cr
    main_re = main_l.real @ cr - main_l.imag @ ci
    main_im = main_l.real @ ci + main_l.imag @ cr

    w0_re = data.w_baseline.real
    w0_im = data.w_baseline.imag
    wr = cp.multiply(w0_re, cr) - cp.multiply(w0_im, ci)
    wi = cp.multiply(w0_re, ci) + cp.multiply(w0_im, cr)
    stacked_w = cp.hstack([wr, wi])

    constraints: list[Any] = [
        target_l_im == 0.0,
        target_l_re >= config.target_co_fraction,
        cp.norm(cp.hstack([target_r_re, target_r_im]), 2)
        <= config.rho_limit * target_l_re,
        cp.norm(cp.vstack([side_re, side_im]), 2, axis=0)
        <= eta_limit * target_l_re,
        cp.norm(stacked_w, 2) <= math.sqrt(config.power_budget_w),
    ]
    if main_no_target.size:
        constraints.append(
            cp.norm(cp.vstack([main_re, main_im]), 2, axis=0)
            <= config.main_peak_factor * target_l_re
        )
    if config.constraint_model == CONSTRAINT_FIXED_INNER:
        constraints.append(cr >= config.g_min)
        constraints.append(cp.norm(cp.vstack([cr, ci]), 2, axis=0) <= config.g_max)
        phi = math.radians(config.nominal_phase_limit_deg)
        constraints.extend(
            [
                math.cos(phi) * ci <= math.sin(phi) * cr,
                -math.cos(phi) * ci <= math.sin(phi) * cr,
            ]
        )

    correction_vector = cp.hstack([cr - 1.0, ci])
    problem = cp.Problem(cp.Minimize(cp.norm(correction_vector, 2)), constraints)
    if not problem.is_dcp():
        raise RuntimeError("Representative bounded minimum-PSLL model is not DCP")

    warm_start_c = np.asarray(warm_start_c, dtype=np.complex128)
    cr.value = warm_start_c.real.copy()
    ci.value = warm_start_c.imag.copy()

    messages: list[str] = []
    last_result: RepresentativeResult | None = None
    for solver_name in solver_sequence:
        started = time.perf_counter()
        try:
            value = problem.solve(
                solver=solver_name,
                verbose=bool(solver_verbose),
                warm_start=True,
                **bounded_solver_options(
                    solver_name,
                    strict=True,
                    time_limit_s=mosek_time_limit_s if solver_name.upper() == "MOSEK" else None,
                ),
            )
            elapsed = time.perf_counter() - started
        except Exception as exc:
            elapsed = time.perf_counter() - started
            messages.append(f"{solver_name}:exception:{type(exc).__name__}:{exc}")
            last_result = RepresentativeResult(
                success=False,
                sll_limit_db=sll_limit_db,
                correction_value=None,
                weights_value=None,
                solver_name=solver_name,
                solver_status="exception",
                distance_to_baseline_c=None,
                audit={},
                solve_time_s=elapsed,
                message=" | ".join(messages),
            )
            continue

        status = str(problem.status or "unknown").lower()
        if status in {"optimal", "optimal_inaccurate"} and cr.value is not None and ci.value is not None:
            correction = np.asarray(cr.value, dtype=np.float64) + 1j * np.asarray(
                ci.value, dtype=np.float64
            )
            correction = np.asarray(correction, dtype=np.complex128)
            weights = weights_from_correction(correction, data.w_baseline)
            audit = audit_bounded_sll_candidate(
                data,
                weights,
                sll_test_db=sll_limit_db,
                config=config,
                tolerances=tolerances,
                correction_abs_tol=correction_abs_tol,
                phase_deg_tol=phase_deg_tol,
            )
            success = bool(audit["feasible_under_budget"])
            messages.append(f"{solver_name}:{status}:audit={success}")
            result = RepresentativeResult(
                success=success,
                sll_limit_db=sll_limit_db,
                correction_value=correction if success else None,
                weights_value=weights if success else None,
                solver_name=solver_name,
                solver_status=status,
                distance_to_baseline_c=float(np.linalg.norm(correction - 1.0)),
                audit=audit,
                solve_time_s=elapsed,
                message=" | ".join(messages),
            )
            if success:
                return result
            last_result = result
        else:
            messages.append(f"{solver_name}:{status}:objective={value!r}")
            last_result = RepresentativeResult(
                success=False,
                sll_limit_db=sll_limit_db,
                correction_value=None,
                weights_value=None,
                solver_name=solver_name,
                solver_status=status,
                distance_to_baseline_c=None,
                audit={},
                solve_time_s=elapsed,
                message=" | ".join(messages),
            )
    assert last_result is not None
    return last_result


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------


def attempt_to_row(attempt: BoundedSllAttempt) -> dict[str, Any]:
    audit = attempt.audit or {}
    return {
        "iteration": attempt.iteration,
        "sll_test_db": attempt.sll_test_db,
        "eta_test": attempt.eta_test,
        "solver": attempt.solver_name,
        "solver_status": attempt.solver_status,
        "classification": attempt.classification,
        "solve_mode": attempt.solve_mode,
        "margin_value": attempt.margin_value,
        "margin_accept_tol": attempt.margin_accept_tol,
        "margin_reject_tol": attempt.margin_reject_tol,
        "minimum_norm": attempt.minimum_norm,
        "minimum_power_W": attempt.minimum_power_w,
        "power_is_minimized": attempt.solve_mode == SOLVE_MODE_MIN_POWER,
        "target_fraction_actual": audit.get("target_fraction"),
        "target_rho_actual": audit.get("target_rho_actual"),
        "target_ar_actual_db": audit.get("target_ar_actual_db"),
        "target_ref_lhcp_psll_db_actual": audit.get("target_ref_lhcp_psll_db"),
        "peak_ref_lhcp_psll_db_actual": audit.get("peak_ref_lhcp_psll_db"),
        "max_main_excess_db_actual": audit.get("max_main_excess_db"),
        "min_amplitude_change_db": audit.get("min_amplitude_change_db"),
        "max_amplitude_change_db": audit.get("max_amplitude_change_db"),
        "max_abs_phase_change_deg": audit.get("max_abs_phase_change_deg"),
        "shape_audit_passed": audit.get("shape_constraints_passed"),
        "budget_audit_passed": audit.get("power_passed"),
        "solve_time_s": attempt.solve_time_s,
        "solver_iterations": attempt.solver_iterations,
        "retry_count": attempt.retry_count,
        "probe_kind": attempt.probe_kind,
        "message": attempt.message,
    }


def write_history_csv(path: Path, history: Sequence[BoundedSllAttempt]) -> None:
    rows = [attempt_to_row(item) for item in history]
    fields = list(rows[0].keys()) if rows else ["iteration", "sll_test_db", "classification"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_grid_audit(output_dir: Path, data: SocpInputData, weights: np.ndarray) -> None:
    weights = np.asarray(weights, dtype=np.complex128)
    e_lhcp = np.asarray(data.a_lhcp_raw @ weights, dtype=np.complex128)
    e_rhcp = np.asarray(data.a_rhcp_raw @ weights, dtype=np.complex128)
    correction = correction_from_weights(weights, data.w_baseline)
    np.savez(
        output_dir / GRID_AUDIT_NAME,
        e_lhcp=e_lhcp,
        e_rhcp=e_rhcp,
        correction=correction,
        theta_rad=np.asarray(data.theta_rad, dtype=np.float64),
        phi_rad=np.asarray(data.phi_rad, dtype=np.float64),
        target_index=np.array(data.target_index, dtype=np.int64),
        main_indices=np.asarray(data.main_indices, dtype=np.int64),
        side_indices=np.asarray(data.side_indices, dtype=np.int64),
    )


def write_top_sidelobe_points(
    output_dir: Path,
    data: SocpInputData,
    weights: np.ndarray,
    *,
    count: int = 30,
) -> None:
    e_lhcp = np.asarray(data.a_lhcp_raw @ weights, dtype=np.complex128)
    target_abs = abs(e_lhcp[data.target_index])
    side_indices = np.asarray(data.side_indices, dtype=np.int64)
    side_level = 20.0 * np.log10(
        np.maximum(np.abs(e_lhcp[side_indices]) / max(target_abs, 1.0e-300), 1.0e-300)
    )
    order = np.argsort(side_level)[::-1][: min(count, side_indices.size)]
    rows: list[dict[str, Any]] = []
    for rank, local in enumerate(order, start=1):
        global_index = int(side_indices[int(local)])
        rows.append(
            {
                "rank": rank,
                "global_index": global_index,
                "theta_deg": float(np.rad2deg(data.theta_rad[global_index])),
                "phi_deg": float(np.rad2deg(data.phi_rad[global_index])),
                "lhcp_target_ref_db": float(side_level[int(local)]),
                "lhcp_abs": float(abs(e_lhcp[global_index])),
            }
        )
    with (output_dir / TOP_SIDE_NAME).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["rank"])
        writer.writeheader()
        writer.writerows(rows)


def bisection_summary(result: BoundedSllBisectionResult) -> dict[str, Any]:
    mode_counts: dict[str, int] = {}
    for attempt in result.history:
        mode_counts[attempt.solve_mode] = mode_counts.get(attempt.solve_mode, 0) + 1
    margin_values = [
        float(attempt.margin_value)
        for attempt in result.history
        if attempt.margin_value is not None and math.isfinite(float(attempt.margin_value))
    ]
    uncertain_attempts = [
        attempt for attempt in result.history if attempt.classification in {UNCERTAIN, SOLVER_ERROR}
    ]
    uncertain_values = sorted({float(attempt.sll_test_db) for attempt in uncertain_attempts})
    return {
        "converged": result.converged,
        "sll_infeasible_db": result.sll_infeasible_db,
        "sll_feasible_db": result.sll_feasible_db,
        "sll_width_db": result.sll_feasible_db - result.sll_infeasible_db,
        "eta_infeasible": float(10.0 ** (result.sll_infeasible_db / 20.0)),
        "eta_feasible": float(10.0 ** (result.sll_feasible_db / 20.0)),
        "lower_is_certified_infeasible": result.lower_is_certified_infeasible,
        "upper_is_certified_feasible": result.upper_is_certified_feasible,
        "strict_endpoint_still_feasible": result.strict_endpoint_still_feasible,
        "history_count": len(result.history),
        "solve_mode_counts": mode_counts,
        "margin_mode_used": bool(mode_counts.get(SOLVE_MODE_BUDGET_MARGIN, 0)),
        "max_margin_value_seen": max(margin_values) if margin_values else None,
        "uncertain_attempt_count": len(uncertain_attempts),
        "uncertain_sll_values_db": uncertain_values,
        "uncertain_sll_span_db": (
            uncertain_values[-1] - uncertain_values[0] if len(uncertain_values) >= 2 else 0.0
        ),
        "stop_reason": result.stop_reason,
        "elapsed_s": result.elapsed_s,
    }


def config_summary(config: BoundedApSllConfig) -> dict[str, Any]:
    return {
        "constraint_model": config.constraint_model,
        "amplitude_limit_db": config.amplitude_limit_db,
        "g_min": _finite_or_none(config.g_min),
        "g_max": _finite_or_none(config.g_max),
        "nominal_phase_limit_deg": config.nominal_phase_limit_deg,
        "effective_phase_limit_deg": config.effective_phase_limit_deg,
        "target_co_fraction": config.target_co_fraction,
        "target_lhcp_power_density_fraction": config.target_co_fraction**2,
        "target_lhcp_equivalent_loss_db": -20.0 * math.log10(config.target_co_fraction),
        "ar_limit_db": config.ar_limit_db,
        "rho_limit": config.rho_limit,
        "main_peak_cap_db": config.main_peak_cap_db,
        "main_peak_factor": config.main_peak_factor,
        "power_budget_w": config.power_budget_w,
    }


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


def run_preflight(
    data: SocpInputData,
    *,
    config: BoundedApSllConfig,
    ar_limit_strategy: Mapping[str, Any],
    sll_loose_db: float,
    sll_strict_db: float,
    sll_tolerance_db: float,
    tolerances: AuditTolerances,
    correction_abs_tol: float,
    phase_deg_tol: float,
) -> dict[str, Any]:
    baseline = baseline_preflight_audit(
        data,
        config=config,
        tolerances=tolerances,
        correction_abs_tol=correction_abs_tol,
        phase_deg_tol=phase_deg_tol,
    )
    return {
        "status": "PASS" if baseline["data_integrity_passed"] else "FAILED",
        "script_version": SCRIPT_VERSION,
        "point_count": data.point_count,
        "port_count": data.port_count,
        "target_index": data.target_index,
        "main_count": int(data.main_indices.size),
        "side_count": int(data.side_indices.size),
        "configuration": {
            **config_summary(config),
            "sll_loose_db": sll_loose_db,
            "sll_strict_db": sll_strict_db,
            "sll_tolerance_db": sll_tolerance_db,
        },
        "ar_limit_strategy": dict(ar_limit_strategy),
        "baseline": baseline,
        "environment": environment_summary(),
        "input_paths": {
            "vecaep_cache": str(data.vecaep_cache),
            "grid_domain": str(data.grid_domain_dir),
            "source_weights": str(data.source_weights_path),
        },
    }


def _base_summary(
    data: SocpInputData,
    *,
    config: BoundedApSllConfig,
    ar_limit_strategy: Mapping[str, Any],
    solver_sequence: Sequence[str],
    sll_loose_db: float,
    sll_strict_db: float,
    sll_tolerance_db: float,
    max_bisection_iterations: int,
    representative_sll_margin_db: float,
    representative_sll_cap_db: float | None,
    margin_accept_tol: float,
    margin_reject_tol: float,
    mosek_time_limit_s: float,
    cross_solver_boundary_fallback: bool,
    tolerances: AuditTolerances,
    correction_abs_tol: float,
    phase_deg_tol: float,
) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "environment": environment_summary(),
        "input_paths": {
            "vecaep_cache": str(data.vecaep_cache),
            "grid_domain": str(data.grid_domain_dir),
            "source_weights": str(data.source_weights_path),
        },
        "ar_limit_strategy": dict(ar_limit_strategy),
        "problem": {
            "point_count": data.point_count,
            "port_count": data.port_count,
            "target_index": data.target_index,
            "main_count": int(data.main_indices.size),
            "side_count": int(data.side_indices.size),
            **config_summary(config),
            "sll_loose_db": sll_loose_db,
            "sll_strict_db": sll_strict_db,
            "sll_tolerance_db": sll_tolerance_db,
            "max_bisection_iterations": max_bisection_iterations,
            "representative_sll_margin_db": representative_sll_margin_db,
            "representative_sll_cap_db": representative_sll_cap_db,
            "boundary_policy": "min-power-then-budget-margin-with-gray-aware-certified-refinement",
            "margin_accept_tol": margin_accept_tol,
            "margin_reject_tol": margin_reject_tol,
            "mosek_time_limit_s": mosek_time_limit_s,
            "cross_solver_boundary_fallback": cross_solver_boundary_fallback,
        },
        "solver_sequence": list(solver_sequence),
        "baseline": baseline_preflight_audit(
            data,
            config=config,
            tolerances=tolerances,
            correction_abs_tol=correction_abs_tol,
            phase_deg_tol=phase_deg_tol,
        ),
    }


def run_optimization_pipeline(
    data: SocpInputData,
    *,
    output_dir: Path,
    config: BoundedApSllConfig,
    ar_limit_strategy: Mapping[str, Any],
    requested_solver: str,
    allow_solver_fallback: bool,
    sll_loose_db: float,
    sll_strict_db: float,
    sll_tolerance_db: float,
    max_bisection_iterations: int,
    representative_sll_margin_db: float,
    representative_sll_cap_db: float | None,
    skip_representative: bool,
    feasibility_only_sll_db: float | None,
    solver_verbose: bool,
    tolerances: AuditTolerances,
    correction_abs_tol: float,
    phase_deg_tol: float,
    margin_accept_tol: float,
    margin_reject_tol: float,
    mosek_time_limit_s: float,
    cross_solver_boundary_fallback: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    require_cvxpy()
    solver_sequence = resolve_solver_sequence(
        requested_solver,
        allow_fallback=allow_solver_fallback,
    )
    logger.info("[SOLVER] sequence=%s", solver_sequence)
    logger.info(
        "[MODEL] constraint=%s amp=+/-%.6g dB phase=+/-%.6g deg effective=%.6g deg "
        "AR<=%.6f dB rho<=%.9g",
        config.constraint_model,
        config.amplitude_limit_db,
        config.nominal_phase_limit_deg,
        config.effective_phase_limit_deg,
        config.ar_limit_db,
        config.rho_limit,
    )

    model = BoundedApSllSocpModel(data, config=config)
    logger.info(
        "[MODEL] min-power DPP=%s side=%d main_without_target=%d",
        model.is_dpp,
        data.side_indices.size,
        model.main_no_target.size,
    )
    logger.info(
        "[BOUNDARY] policy=min-power-then-budget-margin margin_accept=%.3g "
        "margin_reject=%.3g mosek_time_limit=%.3gs cross_solver=%s",
        margin_accept_tol,
        margin_reject_tol,
        mosek_time_limit_s,
        cross_solver_boundary_fallback,
    )
    sll_solver = HybridSllSolver(
        model,
        solver_sequence=solver_sequence,
        solver_verbose=solver_verbose,
        tolerances=tolerances,
        correction_abs_tol=correction_abs_tol,
        phase_deg_tol=phase_deg_tol,
        margin_accept_tol=margin_accept_tol,
        margin_reject_tol=margin_reject_tol,
        mosek_time_limit_s=mosek_time_limit_s,
        cross_solver_boundary_fallback=cross_solver_boundary_fallback,
        logger=logger,
    )

    base_summary = _base_summary(
        data,
        config=config,
        ar_limit_strategy=ar_limit_strategy,
        solver_sequence=solver_sequence,
        sll_loose_db=sll_loose_db,
        sll_strict_db=sll_strict_db,
        sll_tolerance_db=sll_tolerance_db,
        max_bisection_iterations=max_bisection_iterations,
        representative_sll_margin_db=representative_sll_margin_db,
        representative_sll_cap_db=representative_sll_cap_db,
        margin_accept_tol=margin_accept_tol,
        margin_reject_tol=margin_reject_tol,
        mosek_time_limit_s=mosek_time_limit_s,
        cross_solver_boundary_fallback=cross_solver_boundary_fallback,
        tolerances=tolerances,
        correction_abs_tol=correction_abs_tol,
        phase_deg_tol=phase_deg_tol,
    )

    if feasibility_only_sll_db is not None:
        attempt = sll_solver.solve(
            feasibility_only_sll_db,
            warm_start_c=np.ones(data.port_count, dtype=np.complex128),
            iteration=0,
        )
        attempt.probe_kind = "feasibility_only"
        write_history_csv(output_dir / HISTORY_NAME, [attempt])
        summary = {
            "status": "feasibility_only",
            **base_summary,
            "feasibility_test": attempt_to_row(attempt),
            "selected_solution": None,
            "output_files": {"history_csv": HISTORY_NAME},
        }
        if attempt.classification == CERTIFIED_FEASIBLE and attempt.weights_value is not None:
            selected_w = np.asarray(attempt.weights_value, dtype=np.complex128)
            selected_c = correction_from_weights(selected_w, data.w_baseline)
            np.savez(
                output_dir / SOLUTION_NAME,
                w_baseline=np.asarray(data.w_baseline, dtype=np.complex128),
                c_selected=selected_c,
                w_selected=selected_w,
                ar_limit_db=np.array(config.ar_limit_db, dtype=np.float64),
                rho_limit=np.array(config.rho_limit, dtype=np.float64),
                sll_test_db=np.array(feasibility_only_sll_db, dtype=np.float64),
            )
            write_optimized_weights_csv(
                output_dir / WEIGHTS_NAME,
                data,
                selected_w,
                solution_role="feasibility_only",
            )
            write_grid_audit(output_dir, data, selected_w)
            write_top_sidelobe_points(output_dir, data, selected_w)
            summary["selected_solution"] = {
                "role": "feasibility_only",
                "sll_limit_db": feasibility_only_sll_db,
                "audit": attempt.audit,
            }
            summary["output_files"].update(
                {
                    "solution_npz": SOLUTION_NAME,
                    "optimized_weights_csv": WEIGHTS_NAME,
                    "grid_field_audit_npz": GRID_AUDIT_NAME,
                    "top_sidelobe_points_csv": TOP_SIDE_NAME,
                }
            )
        write_json(output_dir / SUMMARY_NAME, summary)
        return summary

    loose, strict, result = run_sll_gate_and_bisection(
        sll_solver,
        sll_loose_db=sll_loose_db,
        sll_strict_db=sll_strict_db,
        sll_tolerance_db=sll_tolerance_db,
        max_iterations=max_bisection_iterations,
        logger=logger,
    )

    history = [loose] if strict is None else (
        result.history if result is not None else [loose, strict]
    )
    write_history_csv(output_dir / HISTORY_NAME, history)

    if result is None:
        if loose.classification == CERTIFIED_INFEASIBLE:
            status = CASE_LOOSE_ENDPOINT_INFEASIBLE
            interpretation = (
                "The loose PSLL endpoint is certified infeasible within the fixed convex inner "
                "approximation and physical power budget. This is not global infeasibility for "
                "the original nonconvex amplitude/phase set."
            )
        elif loose.classification != CERTIFIED_FEASIBLE:
            status = CASE_LOOSE_ENDPOINT_UNCERTAIN
            interpretation = "The loose PSLL endpoint was not numerically certified."
        else:
            status = CASE_STRICT_ENDPOINT_UNCERTAIN
            interpretation = "The strict PSLL endpoint was not numerically certified, so no certified bracket was formed."
        summary = {
            "status": status,
            **base_summary,
            "loose_endpoint_gate": attempt_to_row(loose),
            "strict_endpoint_gate": None if strict is None else attempt_to_row(strict),
            "bisection": None,
            "representative": None,
            "selected_solution": None,
            "interpretation": interpretation,
            "output_files": {"history_csv": HISTORY_NAME},
        }
        write_json(output_dir / SUMMARY_NAME, summary)
        return summary

    if result.best_feasible_c is None or result.best_feasible_w is None:
        raise RuntimeError("PSLL search did not retain a feasible correction/weight vector")
    if not result.converged:
        raise RuntimeError(f"Bounded minimum-PSLL search did not converge: {result.stop_reason}")

    if result.strict_endpoint_still_feasible:
        representative_sll_uncapped = result.sll_feasible_db
        case_status = CASE_STRICT_ENDPOINT_STILL_FEASIBLE
        logger.warning(
            "[BOUND] strict endpoint %.6f dB is still feasible; the optimum lies below the configured search range",
            result.sll_feasible_db,
        )
    else:
        representative_sll, representative_sll_uncapped = resolve_representative_sll_limit(
            result.sll_feasible_db,
            representative_sll_margin_db,
            representative_sll_cap_db,
        )
        case_status = CASE_COMPLETE

    if result.strict_endpoint_still_feasible:
        representative_sll, representative_sll_uncapped = resolve_representative_sll_limit(
            result.sll_feasible_db,
            0.0,
            representative_sll_cap_db,
        )

    if (
        representative_sll_cap_db is not None
        and representative_sll < representative_sll_uncapped - 1e-12
    ):
        logger.info(
            "[REPRESENTATIVE] cap applied: uncapped=%.6f dB cap=%.6f dB final=%.6f dB",
            representative_sll_uncapped,
            representative_sll_cap_db,
            representative_sll,
        )

    representative: RepresentativeResult | None = None
    if not skip_representative:
        logger.info(
            "[REPRESENTATIVE] closest-to-baseline correction at SLL<=%.6f dB",
            representative_sll,
        )
        representative = solve_representative_correction(
            data,
            config=config,
            sll_limit_db=representative_sll,
            solver_sequence=solver_sequence,
            warm_start_c=result.best_feasible_c,
            solver_verbose=solver_verbose,
            tolerances=tolerances,
            correction_abs_tol=correction_abs_tol,
            phase_deg_tol=phase_deg_tol,
            mosek_time_limit_s=mosek_time_limit_s,
        )
        if not representative.success:
            raise RuntimeError(f"Representative solve failed audit: {representative.message}")

    selected_c = (
        representative.correction_value
        if representative is not None and representative.success
        else result.best_feasible_c
    )
    selected_w = (
        representative.weights_value
        if representative is not None and representative.success
        else result.best_feasible_w
    )
    assert selected_c is not None and selected_w is not None
    selected_role = "representative" if representative is not None else "bisection_best_feasible"
    selected_sll_limit = representative.sll_limit_db if representative is not None else result.sll_feasible_db
    selected_audit = audit_bounded_sll_candidate(
        data,
        selected_w,
        sll_test_db=selected_sll_limit,
        config=config,
        tolerances=tolerances,
        correction_abs_tol=correction_abs_tol,
        phase_deg_tol=phase_deg_tol,
    )
    if not selected_audit["feasible_under_budget"]:
        raise RuntimeError(f"Final selected solution failed independent audit: {selected_audit}")

    representative_c = (
        representative.correction_value
        if representative is not None and representative.correction_value is not None
        else np.empty(0, dtype=np.complex128)
    )
    representative_w = (
        representative.weights_value
        if representative is not None and representative.weights_value is not None
        else np.empty(0, dtype=np.complex128)
    )
    np.savez(
        output_dir / SOLUTION_NAME,
        w_baseline=np.asarray(data.w_baseline, dtype=np.complex128),
        c_best_feasible=np.asarray(result.best_feasible_c, dtype=np.complex128),
        w_best_feasible=np.asarray(result.best_feasible_w, dtype=np.complex128),
        c_representative=np.asarray(representative_c, dtype=np.complex128),
        w_representative=np.asarray(representative_w, dtype=np.complex128),
        c_selected=np.asarray(selected_c, dtype=np.complex128),
        w_selected=np.asarray(selected_w, dtype=np.complex128),
        ar_limit_db=np.array(config.ar_limit_db, dtype=np.float64),
        rho_limit=np.array(config.rho_limit, dtype=np.float64),
        sll_infeasible_db=np.array(result.sll_infeasible_db, dtype=np.float64),
        sll_feasible_db=np.array(result.sll_feasible_db, dtype=np.float64),
        sll_representative_db=np.array(representative_sll, dtype=np.float64),
        target_index=np.array(data.target_index, dtype=np.int64),
        main_indices=np.asarray(data.main_indices, dtype=np.int64),
        side_indices=np.asarray(data.side_indices, dtype=np.int64),
    )
    write_optimized_weights_csv(
        output_dir / WEIGHTS_NAME,
        data,
        selected_w,
        solution_role=selected_role,
    )
    write_grid_audit(output_dir, data, selected_w)
    write_top_sidelobe_points(output_dir, data, selected_w)

    summary = {
        "status": case_status,
        **base_summary,
        "loose_endpoint_gate": attempt_to_row(loose),
        "strict_endpoint_gate": None if strict is None else attempt_to_row(strict),
        "bisection": bisection_summary(result),
        "representative": None
        if representative is None
        else {
            "success": representative.success,
            "sll_limit_db": representative.sll_limit_db,
            "solver": representative.solver_name,
            "solver_status": representative.solver_status,
            "distance_to_baseline_c": representative.distance_to_baseline_c,
            "solve_time_s": representative.solve_time_s,
            "audit": representative.audit,
            "message": representative.message,
        },
        "selected_solution": {
            "role": selected_role,
            "sll_limit_db": selected_sll_limit,
            "audit": selected_audit,
            "ar_requirement_met": bool(selected_audit["ar_passed"]),
        },
        "interpretation": (
            "The feasible PSLL endpoint is an independently audited achievable upper bound "
            "for the fixed-tangent convex inner approximation. It is conservative relative "
            "to the original nonconvex double-sided amplitude/phase problem."
            if config.constraint_model == CONSTRAINT_FIXED_INNER
            else "The feasible PSLL endpoint uses continuous unbounded complex corrections."
        ),
        "output_files": {
            "solution_npz": SOLUTION_NAME,
            "optimized_weights_csv": WEIGHTS_NAME,
            "history_csv": HISTORY_NAME,
            "grid_field_audit_npz": GRID_AUDIT_NAME,
            "top_sidelobe_points_csv": TOP_SIDE_NAME,
        },
    }
    write_json(output_dir / SUMMARY_NAME, summary)
    return summary


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimum target-referenced LHCP PSLL Vec-AEP synthesis with a fixed AR ceiling "
            "and fixed-inner bounded amplitude/phase corrections"
        )
    )
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    parser.add_argument("--vecaep-cache", type=Path)
    parser.add_argument("--grid-domain", type=Path)
    parser.add_argument("--source-weights", type=Path)
    parser.add_argument("--output-dir", type=Path)

    parser.add_argument("--solver", default="AUTO", choices=["AUTO", "MOSEK", "CLARABEL", "SCS"])
    parser.add_argument("--no-solver-fallback", action="store_true")
    parser.add_argument(
        "--constraint-model",
        default=CONSTRAINT_FIXED_INNER,
        choices=[CONSTRAINT_FIXED_INNER, CONSTRAINT_UNBOUNDED],
    )
    parser.add_argument("--amplitude-limit-db", type=float, default=DEFAULT_AMPLITUDE_LIMIT_DB)
    parser.add_argument("--phase-limit-deg", type=float, default=DEFAULT_PHASE_LIMIT_DEG)
    parser.add_argument("--target-co-fraction", type=float, default=DEFAULT_TARGET_CO_FRACTION)
    parser.add_argument("--main-peak-cap-db", type=float, default=DEFAULT_MAIN_PEAK_CAP_DB)
    parser.add_argument("--power-budget-w", type=float, default=DEFAULT_POWER_BUDGET_W)

    parser.add_argument("--ar-limit-db", type=float)
    parser.add_argument("--ar-reference-summary", type=Path)
    parser.add_argument("--ar-round-step-db", type=float, default=DEFAULT_AR_ROUND_STEP_DB)

    parser.add_argument("--sll-loose-db", type=float, default=DEFAULT_SLL_LOOSE_DB)
    parser.add_argument("--sll-strict-db", type=float, default=DEFAULT_SLL_STRICT_DB)
    parser.add_argument("--sll-tol-db", type=float, default=DEFAULT_SLL_TOL_DB)
    parser.add_argument(
        "--max-bisection-iterations",
        type=int,
        default=DEFAULT_MAX_BISECTION_ITERATIONS,
    )
    parser.add_argument(
        "--representative-sll-margin-db",
        type=float,
        default=DEFAULT_REPRESENTATIVE_SLL_MARGIN_DB,
    )
    parser.add_argument(
        "--representative-sll-cap-db",
        type=float,
        default=None,
        help=(
            "Optional upper cap for the representative PSLL limit in dB. "
            "Because more-negative PSLL is stricter, the representative limit is "
            "min(feasible+margin, cap). Use -20 to prevent representative relaxation "
            "from crossing the global -20 dB specification."
        ),
    )
    parser.add_argument(
        "--feasibility-only-sll-db",
        type=float,
        help="Solve one fixed PSLL feasibility/minimum-power case and skip the outer bisection.",
    )

    parser.add_argument("--margin-accept-tol", type=float, default=DEFAULT_MARGIN_ACCEPT_TOL)
    parser.add_argument("--margin-reject-tol", type=float, default=DEFAULT_MARGIN_REJECT_TOL)
    parser.add_argument("--mosek-time-limit-s", type=float, default=DEFAULT_MOSEK_TIME_LIMIT_S)
    parser.add_argument(
        "--cross-solver-boundary-fallback",
        action="store_true",
        help=(
            "After hard-budget margin MOSEK retries remain uncertain, also try the "
            "remaining solvers. Disabled by default to avoid long fallback stalls."
        ),
    )

    parser.add_argument("--power-accept-tol", type=float, default=2.0e-6)
    parser.add_argument("--power-reject-tol", type=float, default=1.0e-5)
    parser.add_argument("--constraint-abs-tol", type=float, default=2.0e-6)
    parser.add_argument("--correction-abs-tol", type=float, default=2.0e-6)
    parser.add_argument("--phase-deg-tol", type=float, default=2.0e-5)

    parser.add_argument("--skip-representative", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--solver-verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    required = ["vecaep_cache", "grid_domain", "source_weights", "output_dir"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Missing required arguments: {missing}")
    if (args.ar_limit_db is None) == (args.ar_reference_summary is None):
        raise ValueError("Specify exactly one of --ar-limit-db or --ar-reference-summary")
    if args.ar_limit_db is not None and (not math.isfinite(args.ar_limit_db) or args.ar_limit_db < 0.0):
        raise ValueError("--ar-limit-db must be finite and nonnegative")
    if args.ar_round_step_db <= 0.0:
        raise ValueError("--ar-round-step-db must be positive")
    if not (0.0 < args.target_co_fraction <= 1.0):
        raise ValueError("--target-co-fraction must lie in (0, 1]")
    if args.main_peak_cap_db < 0.0:
        raise ValueError("--main-peak-cap-db must be nonnegative")
    if args.power_budget_w <= 0.0:
        raise ValueError("--power-budget-w must be positive")
    if args.amplitude_limit_db < 0.0:
        raise ValueError("--amplitude-limit-db must be nonnegative")
    if not (0.0 <= args.phase_limit_deg <= 90.0):
        raise ValueError("--phase-limit-deg must lie in [0, 90]")
    if not (args.sll_strict_db < args.sll_loose_db < 0.0):
        raise ValueError("Require --sll-strict-db < --sll-loose-db < 0")
    if args.sll_tol_db <= 0.0:
        raise ValueError("--sll-tol-db must be positive")
    if args.max_bisection_iterations <= 0:
        raise ValueError("--max-bisection-iterations must be positive")
    if args.representative_sll_margin_db < 0.0:
        raise ValueError("--representative-sll-margin-db must be nonnegative")
    if args.representative_sll_cap_db is not None:
        if not math.isfinite(args.representative_sll_cap_db) or args.representative_sll_cap_db >= 0.0:
            raise ValueError("--representative-sll-cap-db must be finite and negative")
    if args.feasibility_only_sll_db is not None and args.feasibility_only_sll_db >= 0.0:
        raise ValueError("--feasibility-only-sll-db must be negative")
    if args.power_accept_tol < 0.0 or args.power_reject_tol <= args.power_accept_tol:
        raise ValueError("Power rejection tolerance must exceed acceptance tolerance")
    if args.margin_accept_tol < 0.0 or args.margin_reject_tol <= args.margin_accept_tol:
        raise ValueError("Margin rejection tolerance must exceed margin acceptance tolerance")
    if args.mosek_time_limit_s <= 0.0:
        raise ValueError("--mosek-time-limit-s must be positive")
    if args.constraint_abs_tol < 0.0 or args.correction_abs_tol < 0.0 or args.phase_deg_tol < 0.0:
        raise ValueError("Audit tolerances must be nonnegative")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_args(args)
        ar_limit_db, ar_limit_strategy = resolve_ar_limit(
            explicit_ar_limit_db=args.ar_limit_db,
            reference_summary=args.ar_reference_summary,
            round_step_db=args.ar_round_step_db,
        )
        output_dir = prepare_output_dir(args.output_dir, overwrite=args.overwrite)
        config = BoundedApSllConfig.build(
            constraint_model=args.constraint_model,
            amplitude_limit_db=args.amplitude_limit_db,
            phase_limit_deg=args.phase_limit_deg,
            target_co_fraction=args.target_co_fraction,
            ar_limit_db=ar_limit_db,
            main_peak_cap_db=args.main_peak_cap_db,
            power_budget_w=args.power_budget_w,
        )
    except Exception as exc:
        print(f"ERROR | [SUMMARY] status=FAILED error={exc}", file=sys.stderr)
        return 2

    logger = setup_logger(output_dir / LOG_NAME, level=args.log_level)
    incomplete_path = output_dir / INCOMPLETE_MARKER
    incomplete_path.write_text("running\n", encoding="utf-8")
    started = time.perf_counter()

    tolerances = AuditTolerances(
        target_fraction_abs=args.constraint_abs_tol,
        rho_abs=args.constraint_abs_tol,
        side_ratio_abs=args.constraint_abs_tol,
        phase_anchor_abs=args.constraint_abs_tol,
        power_accept_abs=args.power_accept_tol,
        power_reject_abs=args.power_reject_tol,
    )

    try:
        logger.info("[VERSION] stage3_bounded_ap_sll_socp.py %s", SCRIPT_VERSION)
        logger.info("[ENV] CVXPY available=%s solvers=%s", cp is not None, installed_solver_names())
        logger.info("[LOAD] vecaep_cache=%s", args.vecaep_cache)
        logger.info("[LOAD] grid_domain=%s", args.grid_domain)
        logger.info("[LOAD] source_weights=%s", args.source_weights)
        data = load_socp_input_data(args.vecaep_cache, args.grid_domain, args.source_weights)
        logger.info(
            "[PASS] data loaded points=%d ports=%d target=%d main=%d side=%d",
            data.point_count,
            data.port_count,
            data.target_index,
            data.main_indices.size,
            data.side_indices.size,
        )

        preflight = run_preflight(
            data,
            config=config,
            ar_limit_strategy=ar_limit_strategy,
            sll_loose_db=args.sll_loose_db,
            sll_strict_db=args.sll_strict_db,
            sll_tolerance_db=args.sll_tol_db,
            tolerances=tolerances,
            correction_abs_tol=args.correction_abs_tol,
            phase_deg_tol=args.phase_deg_tol,
        )
        write_json(output_dir / PREFLIGHT_NAME, preflight)
        if preflight["status"] != "PASS":
            raise RuntimeError(f"Preflight data-integrity audit failed: {preflight['baseline']}")
        logger.info(
            "[PASS] baseline reconstruction power=%.9g W AR=%.9f dB side=%.9f dB",
            data.baseline_power_w,
            data.baseline_ar_db,
            data.baseline_side_db,
        )
        logger.info(
            "[MODEL] fixed AR limit=%.6f dB rho=%.9g strategy=%s",
            config.ar_limit_db,
            config.rho_limit,
            ar_limit_strategy["mode"],
        )
        logger.info(
            "[SEARCH] PSLL bracket=[%.6f, %.6f] dB tol=%.6f dB",
            args.sll_strict_db,
            args.sll_loose_db,
            args.sll_tol_db,
        )

        if args.preflight_only:
            summary = {
                "status": "preflight_only",
                "script_version": SCRIPT_VERSION,
                "preflight_file": PREFLIGHT_NAME,
                "ar_limit_strategy": ar_limit_strategy,
            }
            write_json(output_dir / SUMMARY_NAME, summary)
        else:
            summary = run_optimization_pipeline(
                data,
                output_dir=output_dir,
                config=config,
                ar_limit_strategy=ar_limit_strategy,
                requested_solver=args.solver,
                allow_solver_fallback=not args.no_solver_fallback,
                sll_loose_db=args.sll_loose_db,
                sll_strict_db=args.sll_strict_db,
                sll_tolerance_db=args.sll_tol_db,
                max_bisection_iterations=args.max_bisection_iterations,
                representative_sll_margin_db=args.representative_sll_margin_db,
                representative_sll_cap_db=args.representative_sll_cap_db,
                skip_representative=args.skip_representative,
                feasibility_only_sll_db=args.feasibility_only_sll_db,
                solver_verbose=args.solver_verbose,
                tolerances=tolerances,
                correction_abs_tol=args.correction_abs_tol,
                phase_deg_tol=args.phase_deg_tol,
                margin_accept_tol=args.margin_accept_tol,
                margin_reject_tol=args.margin_reject_tol,
                mosek_time_limit_s=args.mosek_time_limit_s,
                cross_solver_boundary_fallback=args.cross_solver_boundary_fallback,
                logger=logger,
            )

        elapsed = time.perf_counter() - started
        if incomplete_path.exists():
            incomplete_path.unlink()
        write_json(
            output_dir / COMPLETE_MARKER,
            {
                "status": summary.get("status"),
                "script_version": SCRIPT_VERSION,
                "elapsed_s": elapsed,
                "summary_file": SUMMARY_NAME,
            },
        )
        logger.info("[SUMMARY] status=%s", summary.get("status"))
        if summary.get("bisection"):
            b = summary["bisection"]
            logger.info(
                "[SUMMARY] PSLL interval=[%.6f, %.6f] dB selected=%s",
                b["sll_infeasible_db"],
                b["sll_feasible_db"],
                "n/a"
                if summary.get("selected_solution") is None
                else f"{summary['selected_solution']['audit']['target_ref_lhcp_psll_db']:.6f} dB",
            )
        logger.info("[SUMMARY] duration=%.3f s", elapsed)
        return 0

    except Exception as exc:
        elapsed = time.perf_counter() - started
        logger.error("[SUMMARY] status=FAILED error=%s", exc)
        logger.debug("%s", traceback.format_exc())
        try:
            write_json(
                output_dir / SUMMARY_NAME,
                {
                    "status": "failed",
                    "script_version": SCRIPT_VERSION,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "elapsed_s": elapsed,
                },
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
