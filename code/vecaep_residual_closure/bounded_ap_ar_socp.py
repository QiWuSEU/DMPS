#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-3 bounded amplitude/phase minimum-AR synthesis.

The fixed Stage-1 geometry, Stage-2 physical rotations, full-array Vec-AEP
matrices, and Stage-2 source weights are treated as immutable inputs.  The
optimization variable is the per-port complex correction coefficient

    c_n = w_n / w0_n.

For the first research version, the nonconvex lower magnitude constraint is
replaced by the safe fixed-tangent inner approximation

    Re(c_n) >= g_min,

while the magnitude upper bound and phase-sector bound remain exact.  Every
returned solution therefore satisfies the original amplitude and phase limits,
but the feasible set is conservative.

For a fixed rho = |E_RHCP(target)| / |E_LHCP(target)|, the normal inner SOCP
minimizes the physical incident-source norm ||w||_2.  If that formulation
becomes numerically noncertifying near a feasible/infeasible boundary, the case
switches permanently to an always-feasible hard-power-budget SOCP that minimizes
a single normalized field-specification margin.  This avoids weak-infeasibility
certificates while preserving an independently audited feasible AR upper bound.
"""

from __future__ import annotations

import argparse
import csv
import gc
import dataclasses
import json
import logging
import math
import os
import shutil
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
        environment_summary,
        installed_solver_names,
        load_socp_input_data,
        prepare_output_dir,
        require_cvxpy,
        resolve_solver_sequence,
        rho_to_ar_db,
        safe_db20,
        solver_options,
        wrap_phase_deg,
        write_json,
        write_optimized_weights_csv,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "bounded_ap_ar_socp.py requires the installed "
        "vecaep_residual_closure.vecaep_fullgrid_socp module"
    ) from exc


SCRIPT_VERSION = "1.3.0"
EXPECTED_POINT_COUNT = 16380
EXPECTED_PORT_COUNT = 187

DEFAULT_TARGET_CO_FRACTION = 0.85
DEFAULT_SLL_DB = -20.0
DEFAULT_MAIN_PEAK_CAP_DB = 0.2
DEFAULT_POWER_BUDGET_W = 1.0
DEFAULT_AMPLITUDE_LIMIT_DB = 2.0
DEFAULT_PHASE_LIMIT_DEG = 30.0
DEFAULT_RHO_TOL = 1.0e-4
DEFAULT_MAX_BISECTION_ITERATIONS = 30
DEFAULT_REPRESENTATIVE_RHO_MARGIN = 5.0e-5
DEFAULT_MARGIN_ACCEPT_TOL = 2.0e-6
DEFAULT_MARGIN_REJECT_TOL = 5.0e-5
DEFAULT_MOSEK_TIME_LIMIT_S = 60.0

CONSTRAINT_FIXED_INNER = "fixed-inner"
CONSTRAINT_UNBOUNDED = "unbounded"

INCOMPLETE_MARKER = ".solve_incomplete"
COMPLETE_MARKER = "solution_complete.json"
SUMMARY_NAME = "bounded_ap_ar_summary.json"
SOLUTION_NAME = "bounded_ap_ar_solution.npz"
WEIGHTS_NAME = "optimized_weights.csv"
HISTORY_NAME = "bounded_ap_ar_bisection_history.csv"
LOG_NAME = "stage3_bounded_ap_ar_socp.log"
PREFLIGHT_NAME = "bounded_ap_ar_preflight_summary.json"
GRID_AUDIT_NAME = "grid_field_audit.npz"
TOP_SIDE_NAME = "top_sidelobe_points.csv"

CASE_COMPLETE = "complete"
CASE_INNER_APPROX_INFEASIBLE = "inner_approx_infeasible"

SOLVE_MODE_MIN_POWER = "min_power"
SOLVE_MODE_BUDGET_MARGIN = "budget_margin"


# MOSEK robustness policy for near-feasible / near-infeasible rho points.
# The 1 W acceptance/rejection tolerances are post-solve audit thresholds and
# do not affect whether MOSEK can solve the conic model.  Numerical retries
# therefore act on the conic solver settings, not on the power thresholds.
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
    """Return solver options used by the bounded amplitude/phase model.

    CVXPY dualizes continuous models for its MOSEK interface.  CVXPY's own
    documentation recommends explicitly asking MOSEK to solve the dual form
    in this situation.  A relaxed retry uses 1e-7 termination tolerances, but
    any returned primal witness must still pass the independent NumPy audit.
    """
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
    logger = logging.getLogger("stage3_bounded_ap_ar_socp")
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


def _jsonable_dataclass(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
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
    """Return the actual maximum phase admitted by the fixed inner approximation."""

    nominal_phase_limit_deg = float(nominal_phase_limit_deg)
    if not math.isfinite(nominal_phase_limit_deg) or not (0.0 <= nominal_phase_limit_deg <= 90.0):
        raise ValueError("Nominal phase limit must lie in [0, 90] degrees")
    g_min, g_max = amplitude_bounds_from_db(amplitude_limit_db)
    ratio = min(1.0, max(0.0, g_min / g_max))
    inner_limit = math.degrees(math.acos(ratio))
    return float(min(nominal_phase_limit_deg, inner_limit))


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


def normalized_response_rows(data: SocpInputData, indices: np.ndarray | Sequence[int], *, rhcp: bool) -> np.ndarray:
    matrix = data.a_rhcp_raw if rhcp else data.a_lhcp_raw
    rows = np.asarray(matrix[np.asarray(indices, dtype=np.int64), :], dtype=np.complex128)
    return np.asarray(data.scale_complex * rows * data.w_baseline[None, :], dtype=np.complex128)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundedApConfig:
    constraint_model: str
    amplitude_limit_db: float
    nominal_phase_limit_deg: float
    target_co_fraction: float
    sll_db: float
    main_peak_cap_db: float
    power_budget_w: float

    g_min: float
    g_max: float
    effective_phase_limit_deg: float
    eta_sll: float
    main_peak_factor: float

    @classmethod
    def build(
        cls,
        *,
        constraint_model: str,
        amplitude_limit_db: float,
        phase_limit_deg: float,
        target_co_fraction: float,
        sll_db: float,
        main_peak_cap_db: float,
        power_budget_w: float,
    ) -> "BoundedApConfig":
        constraint_model = str(constraint_model)
        if constraint_model not in {CONSTRAINT_FIXED_INNER, CONSTRAINT_UNBOUNDED}:
            raise ValueError(f"Unsupported constraint model: {constraint_model}")
        if not (0.0 < target_co_fraction <= 1.0):
            raise ValueError("target_co_fraction must lie in (0, 1]")
        if sll_db >= 0.0:
            raise ValueError("sll_db must be negative")
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
            sll_db=float(sll_db),
            main_peak_cap_db=float(main_peak_cap_db),
            power_budget_w=float(power_budget_w),
            g_min=float(g_min),
            g_max=float(g_max),
            effective_phase_limit_deg=float(effective),
            eta_sll=float(10.0 ** (sll_db / 20.0)),
            main_peak_factor=float(10.0 ** (main_peak_cap_db / 20.0)),
        )


@dataclass
class BoundedAttempt:
    iteration: int
    rho_test: float
    ar_test_db: float
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
class BoundedBisectionResult:
    converged: bool
    rho_lower: float
    rho_upper: float
    ar_lower_db: float
    ar_upper_db: float
    lower_is_certified_infeasible: bool
    upper_is_certified_feasible: bool
    best_feasible_c: np.ndarray | None
    best_feasible_w: np.ndarray | None
    best_feasible_attempt: BoundedAttempt | None
    history: list[BoundedAttempt] = field(default_factory=list)
    stop_reason: str = ""
    elapsed_s: float = 0.0


@dataclass
class RepresentativeResult:
    success: bool
    rho_limit: float
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


def audit_bounded_candidate(
    data: SocpInputData,
    weights: np.ndarray,
    *,
    rho_test: float,
    config: BoundedApConfig,
    tolerances: AuditTolerances,
    correction_abs_tol: float,
    phase_deg_tol: float,
) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=np.complex128).reshape(-1)
    if weights.size != data.port_count:
        raise ValueError(f"Weight count mismatch: {weights.size} vs {data.port_count}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Candidate weights contain NaN or Inf")

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
    ar_actual = rho_to_ar_db(rho_actual) if rho_actual < 1.0 else float("inf")
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

    target_floor_passed = bool(
        target_fraction >= config.target_co_fraction - tolerances.target_fraction_abs
    )
    rho_passed = bool(rho_actual <= rho_test + tolerances.rho_abs)
    side_passed = bool(side_ratio <= config.eta_sll + tolerances.side_ratio_abs)
    main_passed = bool(
        main_ratio <= config.main_peak_factor + tolerances.side_ratio_abs
    )
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
        and rho_passed
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
        "target_rho_limit": rho_test,
        "target_ar_actual_db": ar_actual,
        "rho_passed": rho_passed,
        "max_side_ratio": side_ratio,
        "max_side_db": side_db,
        "side_limit_db": config.sll_db,
        "side_passed": side_passed,
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
    config: BoundedApConfig,
    tolerances: AuditTolerances,
    correction_abs_tol: float,
    phase_deg_tol: float,
) -> dict[str, Any]:
    audit = audit_bounded_candidate(
        data,
        data.w_baseline,
        rho_test=max(1.0, data.baseline_rho),
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
        "baseline_meets_requested_spec": bool(audit["feasible_under_budget"]),
        "requested_spec_audit": audit,
        "known_baseline": {
            "power_w": data.baseline_power_w,
            "target_ar_db": data.baseline_ar_db,
            "max_side_db_to_target": data.baseline_side_db,
            "target_rho": data.baseline_rho,
        },
    }


# -----------------------------------------------------------------------------
# Parametric SOCP
# -----------------------------------------------------------------------------


class BoundedApSocpModel:
    """Minimum-physical-power SOCP for one fixed amplitude/phase case."""

    def __init__(self, data: SocpInputData, *, config: BoundedApConfig) -> None:
        require_cvxpy()
        self.data = data
        self.config = config
        self.port_count = data.port_count

        self.cr = cp.Variable(self.port_count, name="correction_real")
        self.ci = cp.Variable(self.port_count, name="correction_imag")
        self.rho_parameter = cp.Parameter(nonneg=True, name="rho_test")

        target_index_array = np.array([data.target_index], dtype=np.int64)
        target_l = normalized_response_rows(data, target_index_array, rhcp=False)[0]
        target_r = normalized_response_rows(data, target_index_array, rhcp=True)[0]
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
            <= self.rho_parameter * self.target_l_re,
            cp.norm(cp.vstack([side_re, side_im]), 2, axis=0)
            <= config.eta_sll * self.target_l_re,
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
            raise RuntimeError("The generated bounded amplitude/phase model is not DCP")
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
        rho_test: float,
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
    ) -> BoundedAttempt:
        solver_name = solver_name.upper()
        rho_test = float(rho_test)
        self.rho_parameter.value = rho_test
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
            return BoundedAttempt(
                iteration=iteration,
                rho_test=rho_test,
                ar_test_db=rho_to_ar_db(rho_test),
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
                audit = audit_bounded_candidate(
                    self.data,
                    weights,
                    rho_test=rho_test,
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

        return BoundedAttempt(
            iteration=iteration,
            rho_test=rho_test,
            ar_test_db=rho_to_ar_db(rho_test),
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


class BoundedBudgetMarginSocpModel:
    """Always-feasible 1 W budget-margin SOCP used near numerical boundaries.

    The amplitude/phase sector, target phase anchor, nonnegative target LHCP,
    and physical power budget are hard constraints.  A single nonnegative
    normalized field slack relaxes the target floor, target polarization ratio,
    outside-mask LHCP ceiling, and main-lobe peak guard.  The baseline
    correction c=1 is feasible for every fixed-inner case when the slack is
    sufficiently large, so MOSEK is not asked to construct a weak
    infeasibility certificate.
    """

    def __init__(self, data: SocpInputData, *, config: BoundedApConfig) -> None:
        require_cvxpy()
        self.data = data
        self.config = config
        self.port_count = data.port_count

        self.cr = cp.Variable(self.port_count, name="margin_correction_real")
        self.ci = cp.Variable(self.port_count, name="margin_correction_imag")
        self.margin = cp.Variable(nonneg=True, name="normalized_field_margin")
        self.rho_parameter = cp.Parameter(nonneg=True, name="margin_rho_test")

        target_index_array = np.array([data.target_index], dtype=np.int64)
        target_l = normalized_response_rows(data, target_index_array, rhcp=False)[0]
        target_r = normalized_response_rows(data, target_index_array, rhcp=True)[0]
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
            <= self.rho_parameter * self.target_l_re + self.margin,
            cp.norm(cp.vstack([side_re, side_im]), 2, axis=0)
            <= config.eta_sll * self.target_l_re + self.margin,
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
            raise RuntimeError("The generated hard-budget margin model is not DCP")
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
        rho_test: float,
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
    ) -> BoundedAttempt:
        solver_name = solver_name.upper()
        rho_test = float(rho_test)
        self.rho_parameter.value = rho_test
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
            return BoundedAttempt(
                iteration=iteration,
                rho_test=rho_test,
                ar_test_db=rho_to_ar_db(rho_test),
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
                audit = audit_bounded_candidate(
                    self.data,
                    weights,
                    rho_test=rho_test,
                    config=self.config,
                    tolerances=tolerances,
                    correction_abs_tol=correction_abs_tol,
                    phase_deg_tol=phase_deg_tol,
                )
                audit["margin_value"] = margin_value
                audit["margin_accept_tol"] = margin_accept_tol
                audit["margin_reject_tol"] = margin_reject_tol
                audit_passed = bool(audit["feasible_under_budget"])

                # A directly audited witness is a constructive feasibility
                # certificate even if the solver labels it inaccurate.
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
            # This model should always admit c=1 with sufficiently large margin
            # for fixed-inner cases.  Treat infeasibility as a numerical/model
            # failure rather than a physical certificate.
            classification = SOLVER_ERROR
            message = (
                f"{attempt_label}: Hard-budget margin model unexpectedly reported infeasible"
            )
        elif status in {"infeasible_inaccurate", "unbounded_inaccurate", "user_limit"}:
            classification = UNCERTAIN
            message = f"{attempt_label}: Noncertifying margin-model status: {status}"
        else:
            classification = UNCERTAIN
            message = f"{attempt_label}: Unhandled margin-model status: {status}; objective={value!r}"

        return BoundedAttempt(
            iteration=iteration,
            rho_test=rho_test,
            ar_test_db=rho_to_ar_db(rho_test),
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


class HybridRhoSolver:
    """Min-power first, then permanent hard-budget margin mode near a boundary."""

    def __init__(
        self,
        min_power_model: BoundedApSocpModel,
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
        self.min_power_model: BoundedApSocpModel | None = min_power_model
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
        self.margin_model: BoundedBudgetMarginSocpModel | None = None
        self.margin_only = False
        self.margin_activation_reason: str | None = None

    def _get_margin_model(self) -> BoundedBudgetMarginSocpModel:
        if self.margin_model is None:
            self.logger.info("[MODEL] building always-feasible hard-budget margin fallback")
            self.margin_model = BoundedBudgetMarginSocpModel(self.data, config=self.config)
            self.logger.info(
                "[MODEL] margin DPP=%s side=%d main_without_target=%d",
                self.margin_model.is_dpp,
                self.data.side_indices.size,
                self.margin_model.main_no_target.size,
            )
            if self.margin_only and self.min_power_model is not None:
                # Margin mode is permanent for the rest of this case. Release
                # the large min-power CVXPY graph/canonicalization cache before
                # solving the second large SOCP to limit peak memory.
                self.min_power_model = None
                gc.collect()
                self.logger.info("[MODEL] released min-power model after boundary-mode activation")
        return self.margin_model

    def _log_attempt(self, attempt: BoundedAttempt, label: str) -> None:
        self.logger.info(
            "[ATTEMPT] iter=%s rho=%.9g mode=%s label=%s solver=%s status=%s "
            "class=%s time=%.3fs P=%s eps=%s",
            attempt.iteration,
            attempt.rho_test,
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
        rho_test: float,
        *,
        warm_start_c: np.ndarray | None,
        iteration: int,
    ) -> BoundedAttempt:
        messages: list[str] = []
        retry_count = 0

        if not self.margin_only:
            if self.min_power_model is None:
                raise RuntimeError("Min-power model was released before margin-mode activation")
            primary = self.min_power_model.solve_once(
                rho_test,
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
                f"rho={rho_test:.12g}, status={primary.solver_status}, "
                f"classification={primary.classification}"
            )
            self.logger.warning(
                "[BOUNDARY] min-power model became noncertifying; switching this case "
                "permanently to hard-budget margin mode (%s)",
                self.margin_activation_reason,
            )
            retry_count += 1

        margin_model = self._get_margin_model()
        margin_profiles = [
            ("budget_margin_strict_cold", self.primary_solver, True, False),
        ]
        if self.primary_solver == "MOSEK":
            margin_profiles.append(
                ("budget_margin_relaxed_accept_unknown", "MOSEK", False, True)
            )
        if self.cross_solver_boundary_fallback:
            for solver_name in self.solver_sequence[1:]:
                margin_profiles.append(
                    (f"budget_margin_{solver_name.lower()}_fallback", solver_name, False, False)
                )

        last_attempt: BoundedAttempt | None = None
        for label, solver_name, strict_tols, accept_unknown in margin_profiles:
            attempt = margin_model.solve_once(
                rho_test,
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

def run_rho_gate_and_bisection(
    rho_solver: HybridRhoSolver,
    *,
    rho_tolerance: float,
    max_iterations: int,
    logger: logging.Logger,
) -> tuple[BoundedAttempt, BoundedBisectionResult | None]:
    """Run a certified tri-state rho search.

    The hybrid inner solver can return three meaningful classifications:

    * CERTIFIED_INFEASIBLE: update the lower rho bound;
    * CERTIFIED_FEASIBLE: update the upper rho bound and retain its witness;
    * UNCERTAIN/SOLVER_ERROR: retain the point as a gray-zone sample.

    A conventional bisection must not abort at a gray-zone midpoint.  Doing so
    is especially problematic for the hard-budget margin model because the
    deliberately conservative acceptance/rejection thresholds define a finite
    numerical gray band.  This routine therefore keeps certified endpoints and
    refines the larger gap between those endpoints and the known gray samples.
    The reported interval always remains an outer bracket bounded by a
    certified-infeasible point (when available) and a certified-feasible point.
    """
    started = time.perf_counter()
    baseline_c = np.ones(rho_solver.data.port_count, dtype=np.complex128)

    gate = rho_solver.solve(1.0, warm_start_c=baseline_c, iteration=-1)
    gate.probe_kind = "rho_gate"
    logger.info(
        "[GATE] rho=1 class=%s mode=%s solver=%s status=%s P=%s eps=%s",
        gate.classification,
        gate.solve_mode,
        gate.solver_name,
        gate.solver_status,
        "n/a" if gate.minimum_power_w is None else f"{gate.minimum_power_w:.9g}",
        "n/a" if gate.margin_value is None else f"{gate.margin_value:.9g}",
    )
    if gate.classification == CERTIFIED_INFEASIBLE:
        return gate, None
    if gate.classification != CERTIFIED_FEASIBLE or gate.correction_value is None or gate.weights_value is None:
        raise RuntimeError(f"rho=1 feasibility gate was not certified: {gate.message}")

    rho_low = 0.0
    rho_high = 1.0
    lower_certified = False
    best_c = np.asarray(gate.correction_value, dtype=np.complex128).copy()
    best_w = np.asarray(gate.weights_value, dtype=np.complex128).copy()
    best_attempt: BoundedAttempt | None = gate
    warm_c = best_c.copy()
    history: list[BoundedAttempt] = [gate]
    uncertain_rhos: list[float] = []
    attempted_rhos: list[float] = [1.0]

    zero = rho_solver.solve(0.0, warm_start_c=warm_c, iteration=0)
    zero.probe_kind = "rho_zero_test"
    history.append(zero)
    attempted_rhos.append(0.0)
    logger.info(
        "[BISECTION] zero test class=%s mode=%s solver=%s status=%s P=%s eps=%s",
        zero.classification,
        zero.solve_mode,
        zero.solver_name,
        zero.solver_status,
        "n/a" if zero.minimum_power_w is None else f"{zero.minimum_power_w:.9g}",
        "n/a" if zero.margin_value is None else f"{zero.margin_value:.9g}",
    )
    if zero.classification == CERTIFIED_FEASIBLE:
        elapsed = time.perf_counter() - started
        assert zero.correction_value is not None and zero.weights_value is not None
        return gate, BoundedBisectionResult(
            converged=True,
            rho_lower=0.0,
            rho_upper=0.0,
            ar_lower_db=0.0,
            ar_upper_db=0.0,
            lower_is_certified_infeasible=False,
            upper_is_certified_feasible=True,
            best_feasible_c=np.asarray(zero.correction_value),
            best_feasible_w=np.asarray(zero.weights_value),
            best_feasible_attempt=zero,
            history=history,
            stop_reason="rho=0 is certified feasible",
            elapsed_s=elapsed,
        )
    if zero.classification == CERTIFIED_INFEASIBLE:
        lower_certified = True
    elif zero.classification in {UNCERTAIN, SOLVER_ERROR}:
        uncertain_rhos.append(0.0)
        logger.warning("[BISECTION] rho=0 not certified; retaining mathematical lower bound rho>=0")

    def active_gray_points() -> list[float]:
        eps = max(1.0e-15, 1.0e-12 * max(1.0, abs(rho_low), abs(rho_high)))
        return sorted({
            float(value)
            for value in uncertain_rhos
            if rho_low + eps < float(value) < rho_high - eps
        })

    def choose_probe() -> tuple[float, str]:
        gray = active_gray_points()
        if not gray:
            return 0.5 * (rho_low + rho_high), "midpoint"

        gray_low = gray[0]
        gray_high = gray[-1]
        left_gap = max(0.0, gray_low - rho_low)
        right_gap = max(0.0, rho_high - gray_high)

        # Refine the larger certified-to-gray gap.  This avoids re-solving the
        # same gray midpoint after symmetric endpoint updates.
        if left_gap >= right_gap and left_gap > 0.0:
            return 0.5 * (rho_low + gray_low), "gray_left_refine"
        if right_gap > 0.0:
            return 0.5 * (gray_high + rho_high), "gray_right_refine"
        return 0.5 * (rho_low + rho_high), "gray_span_probe"

    converged = False
    stop_reason = "maximum certified-refinement iterations reached"
    for iteration in range(1, max_iterations + 1):
        width_before = rho_high - rho_low
        if width_before <= rho_tolerance:
            converged = True
            stop_reason = "certified rho interval tolerance reached"
            break

        rho_probe, probe_kind = choose_probe()
        spacing_tol = max(1.0e-15, 1.0e-12 * max(1.0, abs(rho_probe)))
        if any(abs(rho_probe - prior) <= spacing_tol for prior in attempted_rhos):
            gray = active_gray_points()
            alternatives: list[tuple[float, str]] = []
            if gray:
                alternatives.extend([
                    (0.5 * (rho_low + gray[0]), "gray_left_refine_alt"),
                    (0.5 * (gray[-1] + rho_high), "gray_right_refine_alt"),
                ])
            alternatives.extend([
                (rho_low + 0.25 * width_before, "lower_quarter_alt"),
                (rho_high - 0.25 * width_before, "upper_quarter_alt"),
            ])
            replacement = next(
                (
                    (candidate, label)
                    for candidate, label in alternatives
                    if rho_low < candidate < rho_high
                    and not any(abs(candidate - prior) <= spacing_tol for prior in attempted_rhos)
                ),
                None,
            )
            if replacement is None:
                stop_reason = (
                    "gray-zone refinement exhausted distinct probe points; "
                    f"certified interval width={width_before:.12g}"
                )
                break
            rho_probe, probe_kind = replacement

        attempt = rho_solver.solve(rho_probe, warm_start_c=warm_c, iteration=iteration)
        attempt.probe_kind = probe_kind
        history.append(attempt)
        attempted_rhos.append(float(rho_probe))
        logger.info(
            "[BISECTION] iter=%02d probe=%s rho=%.9g AR=%.6f class=%s mode=%s "
            "solver=%s P=%s eps=%s bracket=[%.9g, %.9g]",
            iteration,
            probe_kind,
            rho_probe,
            attempt.ar_test_db,
            attempt.classification,
            attempt.solve_mode,
            attempt.solver_name,
            "n/a" if attempt.minimum_power_w is None else f"{attempt.minimum_power_w:.9g}",
            "n/a" if attempt.margin_value is None else f"{attempt.margin_value:.9g}",
            rho_low,
            rho_high,
        )

        if attempt.classification == CERTIFIED_FEASIBLE:
            rho_high = rho_probe
            assert attempt.correction_value is not None and attempt.weights_value is not None
            best_c = np.asarray(attempt.correction_value, dtype=np.complex128).copy()
            best_w = np.asarray(attempt.weights_value, dtype=np.complex128).copy()
            best_attempt = attempt
            warm_c = best_c.copy()
        elif attempt.classification == CERTIFIED_INFEASIBLE:
            rho_low = rho_probe
            lower_certified = True
        else:
            uncertain_rhos.append(float(rho_probe))
            gray = active_gray_points()
            logger.warning(
                "[GRAY] retained noncertifying rho=%.12g; certified bracket remains "
                "[%.12g, %.12g], active_gray=[%s]",
                rho_probe,
                rho_low,
                rho_high,
                ", ".join(f"{value:.12g}" for value in gray),
            )

        # Discard gray samples that are no longer inside the certified bracket.
        uncertain_rhos = active_gray_points()
        width_after = rho_high - rho_low
        if width_after <= rho_tolerance:
            converged = True
            stop_reason = (
                "certified rho interval tolerance reached after gray-zone refinement"
                if uncertain_rhos
                else "certified rho interval tolerance reached"
            )
            break
    else:
        if rho_high - rho_low <= rho_tolerance:
            converged = True
            stop_reason = "certified rho interval tolerance reached at iteration limit"

    if not converged and uncertain_rhos:
        gray = active_gray_points()
        gray_span = (gray[-1] - gray[0]) if len(gray) >= 2 else 0.0
        stop_reason = (
            f"{stop_reason}; certified bracket=[{rho_low:.12g}, {rho_high:.12g}], "
            f"width={rho_high - rho_low:.12g}, gray_count={len(gray)}, "
            f"gray_span={gray_span:.12g}"
        )

    elapsed = time.perf_counter() - started
    return gate, BoundedBisectionResult(
        converged=converged,
        rho_lower=rho_low,
        rho_upper=rho_high,
        ar_lower_db=rho_to_ar_db(rho_low),
        ar_upper_db=rho_to_ar_db(rho_high),
        lower_is_certified_infeasible=lower_certified,
        upper_is_certified_feasible=True,
        best_feasible_c=best_c,
        best_feasible_w=best_w,
        best_feasible_attempt=best_attempt,
        history=history,
        stop_reason=stop_reason,
        elapsed_s=elapsed,
    )

# -----------------------------------------------------------------------------
# Representative solution
# -----------------------------------------------------------------------------


def solve_representative_correction(
    data: SocpInputData,
    *,
    config: BoundedApConfig,
    rho_limit: float,
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
    cr = cp.Variable(count, name="rep_correction_real")
    ci = cp.Variable(count, name="rep_correction_imag")

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
        cp.norm(cp.hstack([target_r_re, target_r_im]), 2) <= rho_limit * target_l_re,
        cp.norm(cp.vstack([side_re, side_im]), 2, axis=0)
        <= config.eta_sll * target_l_re,
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
        raise RuntimeError("Representative bounded amplitude/phase model is not DCP")

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
                rho_limit=rho_limit,
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
            audit = audit_bounded_candidate(
                data,
                weights,
                rho_test=rho_limit,
                config=config,
                tolerances=tolerances,
                correction_abs_tol=correction_abs_tol,
                phase_deg_tol=phase_deg_tol,
            )
            success = bool(audit["feasible_under_budget"])
            messages.append(f"{solver_name}:{status}:audit={success}")
            result = RepresentativeResult(
                success=success,
                rho_limit=rho_limit,
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
                rho_limit=rho_limit,
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


def attempt_to_row(attempt: BoundedAttempt) -> dict[str, Any]:
    audit = attempt.audit or {}
    return {
        "iteration": attempt.iteration,
        "rho_test": attempt.rho_test,
        "ar_test_db": attempt.ar_test_db,
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
        "max_side_db_actual": audit.get("max_side_db"),
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


def write_history_csv(path: Path, history: Sequence[BoundedAttempt]) -> None:
    rows = [attempt_to_row(item) for item in history]
    fields = list(rows[0].keys()) if rows else ["iteration", "rho_test", "classification"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_grid_audit(
    output_dir: Path,
    data: SocpInputData,
    weights: np.ndarray,
) -> None:
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


def bisection_summary(result: BoundedBisectionResult) -> dict[str, Any]:
    mode_counts: dict[str, int] = {}
    for attempt in result.history:
        mode_counts[attempt.solve_mode] = mode_counts.get(attempt.solve_mode, 0) + 1
    margin_values = [
        float(attempt.margin_value)
        for attempt in result.history
        if attempt.margin_value is not None and math.isfinite(float(attempt.margin_value))
    ]
    uncertain_attempts = [
        attempt for attempt in result.history
        if attempt.classification in {UNCERTAIN, SOLVER_ERROR}
    ]
    uncertain_rhos = sorted({float(attempt.rho_test) for attempt in uncertain_attempts})
    return {
        "converged": result.converged,
        "rho_lower": result.rho_lower,
        "rho_upper": result.rho_upper,
        "rho_width": result.rho_upper - result.rho_lower,
        "ar_lower_db": result.ar_lower_db,
        "ar_upper_db": result.ar_upper_db,
        "ar_width_db": result.ar_upper_db - result.ar_lower_db,
        "lower_is_certified_infeasible": result.lower_is_certified_infeasible,
        "upper_is_certified_feasible": result.upper_is_certified_feasible,
        "history_count": len(result.history),
        "solve_mode_counts": mode_counts,
        "margin_mode_used": bool(mode_counts.get(SOLVE_MODE_BUDGET_MARGIN, 0)),
        "max_margin_value_seen": max(margin_values) if margin_values else None,
        "uncertain_attempt_count": len(uncertain_attempts),
        "uncertain_rho_values": uncertain_rhos,
        "uncertain_rho_span": (uncertain_rhos[-1] - uncertain_rhos[0]) if len(uncertain_rhos) >= 2 else 0.0,
        "stop_reason": result.stop_reason,
        "elapsed_s": result.elapsed_s,
    }


def config_summary(config: BoundedApConfig) -> dict[str, Any]:
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
        "sll_db": config.sll_db,
        "eta_sll": config.eta_sll,
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
    config: BoundedApConfig,
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
        "configuration": config_summary(config),
        "baseline": baseline,
        "environment": environment_summary(),
        "input_paths": {
            "vecaep_cache": str(data.vecaep_cache),
            "grid_domain": str(data.grid_domain_dir),
            "source_weights": str(data.source_weights_path),
        },
    }


def run_optimization_pipeline(
    data: SocpInputData,
    *,
    output_dir: Path,
    config: BoundedApConfig,
    requested_solver: str,
    allow_solver_fallback: bool,
    rho_tolerance: float,
    max_bisection_iterations: int,
    representative_rho_margin: float,
    skip_representative: bool,
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
        requested_solver, allow_fallback=allow_solver_fallback
    )
    logger.info("[SOLVER] sequence=%s", solver_sequence)
    logger.info(
        "[MODEL] constraint=%s amp=+/-%.6g dB phase=+/-%.6g deg effective=%.6g deg",
        config.constraint_model,
        config.amplitude_limit_db,
        config.nominal_phase_limit_deg,
        config.effective_phase_limit_deg,
    )

    model = BoundedApSocpModel(data, config=config)
    logger.info("[MODEL] min-power DPP=%s side=%d main_without_target=%d", model.is_dpp, data.side_indices.size, model.main_no_target.size)
    logger.info(
        "[BOUNDARY] policy=min-power-then-budget-margin margin_accept=%.3g "
        "margin_reject=%.3g mosek_time_limit=%.3gs cross_solver=%s",
        margin_accept_tol,
        margin_reject_tol,
        mosek_time_limit_s,
        cross_solver_boundary_fallback,
    )
    rho_solver = HybridRhoSolver(
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

    gate, result = run_rho_gate_and_bisection(
        rho_solver,
        rho_tolerance=rho_tolerance,
        max_iterations=max_bisection_iterations,
        logger=logger,
    )

    if result is None:
        write_history_csv(output_dir / HISTORY_NAME, [gate])
        summary = {
            "status": CASE_INNER_APPROX_INFEASIBLE,
            "script_version": SCRIPT_VERSION,
            "environment": environment_summary(),
            "input_paths": {
                "vecaep_cache": str(data.vecaep_cache),
                "grid_domain": str(data.grid_domain_dir),
                "source_weights": str(data.source_weights_path),
            },
            "problem": {
                "point_count": data.point_count,
                "port_count": data.port_count,
                "target_index": data.target_index,
                "main_count": int(data.main_indices.size),
                "side_count": int(data.side_indices.size),
                **config_summary(config),
                "rho_tolerance": rho_tolerance,
                "max_bisection_iterations": max_bisection_iterations,
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
            "rho_gate": attempt_to_row(gate),
            "bisection": None,
            "selected_solution": None,
            "interpretation": (
                "No audited solution exists within the requested physical power budget at rho=1 "
                "for the fixed convex inner approximation.  This is not a certificate of global "
                "infeasibility for the original nonconvex double-sided amplitude problem."
            ),
            "output_files": {"history_csv": HISTORY_NAME},
        }
        write_json(output_dir / SUMMARY_NAME, summary)
        return summary

    write_history_csv(output_dir / HISTORY_NAME, result.history)
    if not result.converged:
        raise RuntimeError(f"Bounded amplitude/phase bisection did not converge: {result.stop_reason}")
    if result.best_feasible_c is None or result.best_feasible_w is None:
        raise RuntimeError("Bisection did not retain a feasible correction/weight vector")

    rho_rep = min(1.0, result.rho_upper + max(representative_rho_margin, 5.0 * rho_tolerance))
    representative: RepresentativeResult | None = None
    if not skip_representative:
        logger.info(
            "[REPRESENTATIVE] closest-to-baseline correction at rho_limit=%.9g AR=%.6f dB",
            rho_rep,
            rho_to_ar_db(rho_rep),
        )
        representative = solve_representative_correction(
            data,
            config=config,
            rho_limit=rho_rep,
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
    selected_rho_limit = representative.rho_limit if representative is not None else result.rho_upper
    selected_audit = audit_bounded_candidate(
        data,
        selected_w,
        rho_test=selected_rho_limit,
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
        rho_lower=np.array(result.rho_lower, dtype=np.float64),
        rho_upper=np.array(result.rho_upper, dtype=np.float64),
        rho_representative=np.array(rho_rep, dtype=np.float64),
        target_index=np.array(data.target_index, dtype=np.int64),
        main_indices=np.asarray(data.main_indices, dtype=np.int64),
        side_indices=np.asarray(data.side_indices, dtype=np.int64),
    )
    write_optimized_weights_csv(output_dir / WEIGHTS_NAME, data, selected_w, solution_role=selected_role)
    write_grid_audit(output_dir, data, selected_w)
    write_top_sidelobe_points(output_dir, data, selected_w)

    summary = {
        "status": CASE_COMPLETE,
        "script_version": SCRIPT_VERSION,
        "environment": environment_summary(),
        "input_paths": {
            "vecaep_cache": str(data.vecaep_cache),
            "grid_domain": str(data.grid_domain_dir),
            "source_weights": str(data.source_weights_path),
        },
        "problem": {
            "point_count": data.point_count,
            "port_count": data.port_count,
            "target_index": data.target_index,
            "main_count": int(data.main_indices.size),
            "side_count": int(data.side_indices.size),
            **config_summary(config),
            "rho_tolerance": rho_tolerance,
            "max_bisection_iterations": max_bisection_iterations,
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
        "rho_gate": attempt_to_row(gate),
        "bisection": bisection_summary(result),
        "representative": None
        if representative is None
        else {
            "success": representative.success,
            "rho_limit": representative.rho_limit,
            "ar_limit_db": rho_to_ar_db(representative.rho_limit),
            "solver": representative.solver_name,
            "solver_status": representative.solver_status,
            "distance_to_baseline_c": representative.distance_to_baseline_c,
            "solve_time_s": representative.solve_time_s,
            "audit": representative.audit,
            "message": representative.message,
        },
        "selected_solution": {
            "role": selected_role,
            "rho_limit": selected_rho_limit,
            "ar_limit_db": rho_to_ar_db(selected_rho_limit),
            "audit": selected_audit,
            "below_3db": bool(selected_audit["target_ar_actual_db"] < 3.0),
        },
        "interpretation": (
            "The reported AR upper bound is certified for the fixed-tangent convex inner "
            "approximation.  It is a conservative achievable result for the original "
            "double-sided amplitude/phase-constrained problem."
            if config.constraint_model == CONSTRAINT_FIXED_INNER
            else "The reported AR upper bound uses continuous unbounded complex corrections."
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
            "Minimum-AR Vec-AEP synthesis with fixed-inner bounded amplitude/phase corrections"
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
    parser.add_argument("--sll-db", type=float, default=DEFAULT_SLL_DB)
    parser.add_argument("--main-peak-cap-db", type=float, default=DEFAULT_MAIN_PEAK_CAP_DB)
    parser.add_argument("--power-budget-w", type=float, default=DEFAULT_POWER_BUDGET_W)
    parser.add_argument("--rho-tol", type=float, default=DEFAULT_RHO_TOL)
    parser.add_argument("--max-bisection-iterations", type=int, default=DEFAULT_MAX_BISECTION_ITERATIONS)
    parser.add_argument("--representative-rho-margin", type=float, default=DEFAULT_REPRESENTATIVE_RHO_MARGIN)

    parser.add_argument("--margin-accept-tol", type=float, default=DEFAULT_MARGIN_ACCEPT_TOL)
    parser.add_argument("--margin-reject-tol", type=float, default=DEFAULT_MARGIN_REJECT_TOL)
    parser.add_argument("--mosek-time-limit-s", type=float, default=DEFAULT_MOSEK_TIME_LIMIT_S)
    parser.add_argument(
        "--cross-solver-boundary-fallback",
        action="store_true",
        help=(
            "After the hard-budget margin MOSEK retries remain uncertain, also try the "
            "remaining solvers. Disabled by default to avoid multi-minute CLARABEL/SCS stalls."
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
    if not (0.0 < args.target_co_fraction <= 1.0):
        raise ValueError("--target-co-fraction must lie in (0, 1]")
    if args.sll_db >= 0.0:
        raise ValueError("--sll-db must be negative")
    if args.main_peak_cap_db < 0.0:
        raise ValueError("--main-peak-cap-db must be nonnegative")
    if args.power_budget_w <= 0.0:
        raise ValueError("--power-budget-w must be positive")
    if args.amplitude_limit_db < 0.0:
        raise ValueError("--amplitude-limit-db must be nonnegative")
    if not (0.0 <= args.phase_limit_deg <= 90.0):
        raise ValueError("--phase-limit-deg must lie in [0, 90]")
    if args.rho_tol <= 0.0:
        raise ValueError("--rho-tol must be positive")
    if args.max_bisection_iterations <= 0:
        raise ValueError("--max-bisection-iterations must be positive")
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
        output_dir = prepare_output_dir(args.output_dir, overwrite=args.overwrite)
        config = BoundedApConfig.build(
            constraint_model=args.constraint_model,
            amplitude_limit_db=args.amplitude_limit_db,
            phase_limit_deg=args.phase_limit_deg,
            target_co_fraction=args.target_co_fraction,
            sll_db=args.sll_db,
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
        logger.info("[VERSION] stage3_bounded_ap_ar_socp.py %s", SCRIPT_VERSION)
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
            "[INFO] baseline_meets_requested_spec=%s (expected False for -20 dB side in this case)",
            preflight["baseline"]["baseline_meets_requested_spec"],
        )

        if args.preflight_only:
            complete = {
                "status": "preflight_complete",
                "script_version": SCRIPT_VERSION,
                "duration_s": time.perf_counter() - started,
                "preflight_summary": PREFLIGHT_NAME,
            }
            write_json(output_dir / "preflight_complete.json", complete)
            incomplete_path.unlink(missing_ok=True)
            logger.info("[SUMMARY] status=PASS mode=preflight-only")
            return 0

        summary = run_optimization_pipeline(
            data,
            output_dir=output_dir,
            config=config,
            requested_solver=args.solver,
            allow_solver_fallback=not args.no_solver_fallback,
            rho_tolerance=args.rho_tol,
            max_bisection_iterations=args.max_bisection_iterations,
            representative_rho_margin=args.representative_rho_margin,
            skip_representative=args.skip_representative,
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
        duration = time.perf_counter() - started

        if summary["status"] == CASE_INNER_APPROX_INFEASIBLE:
            complete = {
                "status": CASE_INNER_APPROX_INFEASIBLE,
                "script_version": SCRIPT_VERSION,
                "duration_s": duration,
                "bisection_converged": False,
                "numpy_audit_passed": True,
                "summary_file": SUMMARY_NAME,
                "interpretation": "Valid completed case: fixed convex inner approximation is infeasible.",
            }
            write_json(output_dir / COMPLETE_MARKER, complete)
            incomplete_path.unlink(missing_ok=True)
            logger.info("[SUMMARY] status=%s duration=%.3f s", CASE_INNER_APPROX_INFEASIBLE, duration)
            return 0

        complete = {
            "status": CASE_COMPLETE,
            "script_version": SCRIPT_VERSION,
            "duration_s": duration,
            "bisection_converged": bool(summary["bisection"]["converged"]),
            "numpy_audit_passed": bool(summary["selected_solution"]["audit"]["feasible_under_budget"]),
            "summary_file": SUMMARY_NAME,
            "solution_file": SOLUTION_NAME,
            "weights_file": WEIGHTS_NAME,
        }
        write_json(output_dir / COMPLETE_MARKER, complete)
        incomplete_path.unlink(missing_ok=True)
        logger.info("[SUMMARY] status=PASS")
        logger.info(
            "[SUMMARY] AR interval=[%.9f, %.9f] dB selected=%.9f dB",
            summary["bisection"]["ar_lower_db"],
            summary["bisection"]["ar_upper_db"],
            summary["selected_solution"]["audit"]["target_ar_actual_db"],
        )
        logger.info("[SUMMARY] duration=%.3f s", duration)
        return 0
    except Exception as exc:
        duration = time.perf_counter() - started
        logger.error("[SUMMARY] status=FAILED error=%s", exc)
        logger.error(traceback.format_exc())
        write_json(
            output_dir / "failure_summary.json",
            {
                "status": "FAILED",
                "script_version": SCRIPT_VERSION,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "duration_s": duration,
                "environment": environment_summary(),
                "cvxpy_import_error_local": CVXPY_IMPORT_ERROR,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
