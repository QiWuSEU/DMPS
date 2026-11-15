#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3 full-grid SOCP synthesis for the 50-degree Vec-AEP baseline.

The script solves two quasiconvex polarization-synthesis problems:

1. target-only: target LHCP floor + target RHCP/LHCP ratio + total-power budget;
2. full-grid: the same constraints plus all fixed sidelobe-grid constraints.

For a fixed candidate rho = |ER| / |EL|, the inner problem minimizes the
Euclidean norm of all complex port weights.  Because the AEPs are normalized
to 1 W incident power per excited port, ||w||_2**2 is the total incident
source power in watts.  A rho candidate is feasible under the project budget
when the minimum required power is no greater than 1 W.

The outer bisection encloses the globally optimal rho of the discretized convex
model between a lower bound and an audited feasible upper bound.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import cvxpy as cp  # type: ignore
except Exception as exc:  # pragma: no cover - exercised by preflight-only environments
    cp = None  # type: ignore
    CVXPY_IMPORT_ERROR = repr(exc)
else:
    CVXPY_IMPORT_ERROR = None

try:
    from vecaep_residual_closure.vecaep_io import load_vecaep_dataset, read_source_weights_csv
    from vecaep_residual_closure.vecaep_grid_domain import load_grid_domain
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "vecaep_fullgrid_socp.py requires the "
        "vecaep_residual_closure.vecaep_io and vecaep_grid_domain modules"
    ) from exc


SCRIPT_VERSION = "1.0.0"
EXPECTED_POINT_COUNT = 16380
EXPECTED_PORT_COUNT = 187
DEFAULT_TARGET_CO_FRACTION = 0.95
DEFAULT_SLL_DB = -15.0
DEFAULT_POWER_BUDGET_W = 1.0
DEFAULT_RHO_TOL = 1.0e-5
DEFAULT_MAX_BISECTION_ITERATIONS = 25
DEFAULT_REPRESENTATIVE_RHO_MARGIN = 5.0e-5

INCOMPLETE_MARKER = ".solve_incomplete"
COMPLETE_MARKER = "solution_complete.json"
SUMMARY_NAME = "fullgrid_socp_summary.json"
SOLUTION_NAME = "fullgrid_socp_solution.npz"
WEIGHTS_NAME = "optimized_weights.csv"
TARGET_HISTORY_NAME = "target_only_bisection_history.csv"
FULL_HISTORY_NAME = "fullgrid_bisection_history.csv"
LOG_NAME = "stage3_fullgrid_socp.log"
PREFLIGHT_NAME = "fullgrid_socp_preflight_summary.json"

CERTIFIED_FEASIBLE = "CERTIFIED_FEASIBLE"
CERTIFIED_INFEASIBLE = "CERTIFIED_INFEASIBLE"
UNCERTAIN = "UNCERTAIN"
SOLVER_ERROR = "SOLVER_ERROR"

TRUSTED_INFEASIBILITY_SOLVERS = {"MOSEK", "CLARABEL"}


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    os.replace(temp_path, path)


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def setup_logger(log_path: Path | None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("stage3_fullgrid_socp")
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


def wrap_phase_deg(values: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(values, dtype=np.float64)
    wrapped = (array + 180.0) % 360.0 - 180.0
    if np.ndim(values) == 0:
        return float(wrapped)
    return wrapped


def rho_to_ar_db(rho: float) -> float:
    rho = float(rho)
    if not math.isfinite(rho) or rho < 0.0:
        raise ValueError(f"rho must be finite and nonnegative, got {rho!r}")
    if rho >= 1.0:
        return float("inf")
    return float(20.0 * math.log10((1.0 + rho) / (1.0 - rho)))


def ar_db_to_rho(ar_db: float) -> float:
    ar_db = float(ar_db)
    if not math.isfinite(ar_db) or ar_db < 0.0:
        raise ValueError(f"AR must be finite and nonnegative, got {ar_db!r}")
    linear = 10.0 ** (ar_db / 20.0)
    return float((linear - 1.0) / (linear + 1.0))


def safe_db20(ratio: float, floor_db: float = -400.0) -> float:
    ratio = float(ratio)
    if ratio <= 0.0:
        return float(floor_db)
    return float(20.0 * math.log10(ratio))


def complex_matrix_to_real_operators(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R/I operators for x=[real(w), imag(w)]."""

    matrix = np.asarray(matrix, dtype=np.complex128)
    real_operator = np.concatenate([matrix.real, -matrix.imag], axis=1)
    imag_operator = np.concatenate([matrix.imag, matrix.real], axis=1)
    return np.asarray(real_operator, dtype=np.float64), np.asarray(imag_operator, dtype=np.float64)


def weights_to_real_vector(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.complex128).reshape(-1)
    return np.concatenate([weights.real, weights.imag]).astype(np.float64, copy=False)


def real_vector_to_weights(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.size % 2:
        raise ValueError(f"Real weight vector length must be even, got {vector.size}")
    count = vector.size // 2
    return np.asarray(vector[:count] + 1j * vector[count:], dtype=np.complex128)


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditTolerances:
    target_fraction_abs: float = 2.0e-6
    rho_abs: float = 2.0e-6
    side_ratio_abs: float = 2.0e-6
    phase_anchor_abs: float = 2.0e-6
    power_accept_abs: float = 2.0e-6
    power_reject_abs: float = 1.0e-5


@dataclass(frozen=True)
class SocpInputData:
    vecaep_cache: Path
    grid_domain_dir: Path
    source_weights_path: Path

    a_lhcp_raw: np.ndarray
    a_rhcp_raw: np.ndarray
    theta_rad: np.ndarray
    phi_rad: np.ndarray

    target_index: int
    main_indices: np.ndarray
    side_indices: np.ndarray

    w_baseline: np.ndarray
    source_table: pd.DataFrame
    element_table: pd.DataFrame

    baseline_target_lhcp: complex
    baseline_target_rhcp: complex
    baseline_rho: float
    baseline_ar_db: float
    baseline_power_w: float
    baseline_side_ratio: float
    baseline_side_db: float

    field_scale: float
    phase_anchor: complex
    scale_complex: complex

    @property
    def point_count(self) -> int:
        return int(self.a_lhcp_raw.shape[0])

    @property
    def port_count(self) -> int:
        return int(self.a_lhcp_raw.shape[1])


@dataclass
class SocpAttempt:
    iteration: int
    rho_test: float
    ar_test_db: float
    include_sidelobes: bool

    solver_name: str
    solver_status: str
    classification: str

    w_value: np.ndarray | None
    minimum_norm: float | None
    minimum_power_w: float | None

    audit_passed: bool
    audit: dict[str, Any]

    solve_time_s: float
    solver_iterations: int | None
    retry_count: int
    message: str


@dataclass
class BisectionResult:
    name: str
    converged: bool
    rho_lower: float
    rho_upper: float
    ar_lower_db: float
    ar_upper_db: float
    lower_is_certified_infeasible: bool
    upper_is_certified_feasible: bool
    best_feasible_w: np.ndarray | None
    best_feasible_attempt: SocpAttempt | None
    history: list[SocpAttempt] = field(default_factory=list)
    stop_reason: str = ""
    elapsed_s: float = 0.0


@dataclass
class RepresentativeResult:
    success: bool
    rho_limit: float
    w_value: np.ndarray | None
    solver_name: str
    solver_status: str
    distance_to_baseline: float | None
    audit: dict[str, Any]
    solve_time_s: float
    message: str


# -----------------------------------------------------------------------------
# Input loading and baseline validation
# -----------------------------------------------------------------------------


def _assert_matching_columns(
    source_table: pd.DataFrame,
    element_table: pd.DataFrame,
    columns: Iterable[str] = ("global_index", "port_name", "unit_name"),
) -> None:
    for column in columns:
        if column not in source_table.columns or column not in element_table.columns:
            raise KeyError(f"Required mapping column is missing: {column}")
        left = source_table[column].astype(str).tolist()
        right = element_table[column].astype(str).tolist()
        if left != right:
            for index, (a, b) in enumerate(zip(left, right)):
                if a != b:
                    raise ValueError(
                        f"Source/Vec-AEP mapping mismatch at row {index}, "
                        f"column={column}: source={a!r}, element={b!r}"
                    )
            raise ValueError(f"Source/Vec-AEP mapping length mismatch for column {column}")


def load_socp_input_data(
    vecaep_cache: Path,
    grid_domain_dir: Path,
    source_weights_path: Path,
    *,
    expected_point_count: int = EXPECTED_POINT_COUNT,
    expected_port_count: int = EXPECTED_PORT_COUNT,
) -> SocpInputData:
    vecaep_cache = Path(vecaep_cache).expanduser().resolve()
    grid_domain_dir = Path(grid_domain_dir).expanduser().resolve()
    source_weights_path = Path(source_weights_path).expanduser().resolve()

    dataset = load_vecaep_dataset(vecaep_cache, mmap_mode="r", load_circular=True)
    if dataset.e_lhcp_matrix is None or dataset.e_rhcp_matrix is None:
        raise ValueError("Vec-AEP cache does not contain LHCP/RHCP matrices")
    if dataset.e_lhcp_matrix.shape != (expected_point_count, expected_port_count):
        raise ValueError(
            f"Unexpected LHCP matrix shape: {dataset.e_lhcp_matrix.shape}, "
            f"expected={(expected_point_count, expected_port_count)}"
        )
    if dataset.e_rhcp_matrix.shape != dataset.e_lhcp_matrix.shape:
        raise ValueError("LHCP/RHCP matrix shapes do not match")

    domain = load_grid_domain(grid_domain_dir)
    if domain.theta_rad.shape != dataset.theta_points_rad.shape:
        raise ValueError("Grid-domain and Vec-AEP point counts do not match")
    if not np.allclose(domain.theta_rad, dataset.theta_points_rad, atol=1.0e-14, rtol=0.0):
        raise ValueError("Grid-domain theta ordering does not match Vec-AEP cache")
    if not np.allclose(domain.phi_rad, dataset.phi_points_rad, atol=1.0e-14, rtol=0.0):
        raise ValueError("Grid-domain phi ordering does not match Vec-AEP cache")

    source = read_source_weights_csv(
        source_weights_path,
        expected_port_count=expected_port_count,
        power_sum_tolerance=1.0e-8,
    )
    element_table = dataset.element_table.sort_values("global_index", kind="mergesort").reset_index(drop=True)
    source_table = source.table.sort_values("global_index", kind="mergesort").reset_index(drop=True)
    _assert_matching_columns(source_table, element_table)

    w0 = np.asarray(source.weights, dtype=np.complex128)
    a_lhcp = dataset.e_lhcp_matrix
    a_rhcp = dataset.e_rhcp_matrix
    target_index = int(domain.target_index)
    side_indices = np.asarray(domain.side_indices, dtype=np.int64)
    main_indices = np.asarray(domain.main_indices, dtype=np.int64)

    target_lhcp = complex(np.asarray(a_lhcp[target_index, :]) @ w0)
    target_rhcp = complex(np.asarray(a_rhcp[target_index, :]) @ w0)
    field_scale = abs(target_lhcp)
    if not math.isfinite(field_scale) or field_scale <= 0.0:
        raise ValueError(f"Invalid baseline target LHCP magnitude: {field_scale}")
    phase_anchor = target_lhcp / field_scale
    scale_complex = np.conj(phase_anchor) / field_scale

    power = float(np.vdot(w0, w0).real)
    rho = float(abs(target_rhcp) / field_scale)
    ar_db = rho_to_ar_db(rho)
    baseline_lhcp = np.asarray(a_lhcp @ w0, dtype=np.complex128)
    side_ratio = float(np.max(np.abs(baseline_lhcp[side_indices])) / field_scale)
    side_db = safe_db20(side_ratio)

    return SocpInputData(
        vecaep_cache=vecaep_cache,
        grid_domain_dir=grid_domain_dir,
        source_weights_path=source_weights_path,
        a_lhcp_raw=a_lhcp,
        a_rhcp_raw=a_rhcp,
        theta_rad=np.asarray(dataset.theta_points_rad),
        phi_rad=np.asarray(dataset.phi_points_rad),
        target_index=target_index,
        main_indices=main_indices,
        side_indices=side_indices,
        w_baseline=w0,
        source_table=source_table,
        element_table=element_table,
        baseline_target_lhcp=target_lhcp,
        baseline_target_rhcp=target_rhcp,
        baseline_rho=rho,
        baseline_ar_db=ar_db,
        baseline_power_w=power,
        baseline_side_ratio=side_ratio,
        baseline_side_db=side_db,
        field_scale=field_scale,
        phase_anchor=phase_anchor,
        scale_complex=scale_complex,
    )


def audit_baseline(
    data: SocpInputData,
    *,
    target_co_fraction: float,
    sll_db: float,
    power_budget_w: float,
    tolerances: AuditTolerances,
) -> dict[str, Any]:
    eta_sll = 10.0 ** (sll_db / 20.0)
    scaled_target = data.scale_complex * data.baseline_target_lhcp
    result = {
        "power_w": data.baseline_power_w,
        "target_lhcp_abs": abs(data.baseline_target_lhcp),
        "target_rhcp_abs": abs(data.baseline_target_rhcp),
        "target_fraction": 1.0,
        "target_rho": data.baseline_rho,
        "target_ar_db": data.baseline_ar_db,
        "max_side_ratio_to_target": data.baseline_side_ratio,
        "max_side_db_to_target": data.baseline_side_db,
        "scaled_target_real": float(scaled_target.real),
        "scaled_target_imag": float(scaled_target.imag),
        "target_index": data.target_index,
        "side_count": int(data.side_indices.size),
    }
    result["power_passed"] = bool(
        data.baseline_power_w <= power_budget_w + tolerances.power_accept_abs
    )
    result["target_floor_passed"] = bool(
        1.0 >= target_co_fraction - tolerances.target_fraction_abs
    )
    result["side_passed"] = bool(
        data.baseline_side_ratio <= eta_sll + tolerances.side_ratio_abs
    )
    result["phase_anchor_passed"] = bool(
        abs(scaled_target.imag) <= tolerances.phase_anchor_abs
        and scaled_target.real > 0.0
    )
    result["passed"] = bool(
        result["power_passed"]
        and result["target_floor_passed"]
        and result["side_passed"]
        and result["phase_anchor_passed"]
    )
    return result


# -----------------------------------------------------------------------------
# Candidate audit
# -----------------------------------------------------------------------------


def audit_candidate(
    data: SocpInputData,
    weights: np.ndarray,
    *,
    rho_test: float,
    target_co_fraction: float,
    sll_db: float,
    power_budget_w: float,
    include_sidelobes: bool,
    tolerances: AuditTolerances,
) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=np.complex128).reshape(-1)
    if weights.size != data.port_count:
        raise ValueError(f"Weight count mismatch: {weights.size} vs {data.port_count}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Candidate weights contain NaN or Inf")

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
    side_ratio = float("nan")
    side_db = float("nan")
    if include_sidelobes:
        side_ratio = float(np.max(np.abs(e_lhcp[data.side_indices])) / target_l_abs)
        side_db = safe_db20(side_ratio)

    eta_sll = 10.0 ** (sll_db / 20.0)
    target_floor_passed = bool(
        target_fraction >= target_co_fraction - tolerances.target_fraction_abs
    )
    rho_passed = bool(rho_actual <= rho_test + tolerances.rho_abs)
    side_passed = bool(
        (not include_sidelobes)
        or side_ratio <= eta_sll + tolerances.side_ratio_abs
    )
    phase_anchor_passed = bool(
        abs(scaled_target.imag) <= tolerances.phase_anchor_abs
        and scaled_target.real >= target_co_fraction - tolerances.target_fraction_abs
    )
    power_passed = bool(power_w <= power_budget_w + tolerances.power_accept_abs)
    shape_constraints_passed = bool(
        target_floor_passed and rho_passed and side_passed and phase_anchor_passed
    )

    return {
        "power_w": power_w,
        "minimum_norm_from_weights": float(math.sqrt(max(power_w, 0.0))),
        "power_budget_w": power_budget_w,
        "power_passed": power_passed,
        "target_lhcp_abs": target_l_abs,
        "target_rhcp_abs": target_r_abs,
        "target_fraction": target_fraction,
        "target_floor_required": target_co_fraction,
        "target_floor_passed": target_floor_passed,
        "target_rho_actual": rho_actual,
        "target_rho_limit": rho_test,
        "target_ar_actual_db": ar_actual,
        "rho_passed": rho_passed,
        "max_side_ratio": side_ratio,
        "max_side_db": side_db,
        "side_limit_db": sll_db,
        "side_passed": side_passed,
        "scaled_target_real": float(scaled_target.real),
        "scaled_target_imag": float(scaled_target.imag),
        "phase_anchor_passed": phase_anchor_passed,
        "shape_constraints_passed": shape_constraints_passed,
        "feasible_under_budget": bool(shape_constraints_passed and power_passed),
        "max_port_power_w": float(np.max(np.abs(weights) ** 2)),
        "min_port_power_w": float(np.min(np.abs(weights) ** 2)),
    }


# -----------------------------------------------------------------------------
# CVXPY solver support
# -----------------------------------------------------------------------------


def require_cvxpy() -> None:
    if cp is None:
        raise RuntimeError(
            "CVXPY is not available in this Python environment. "
            f"Import error: {CVXPY_IMPORT_ERROR}. Install CVXPY and at least one "
            "conic solver before running SOCP optimization."
        )


def installed_solver_names() -> list[str]:
    if cp is None:
        return []
    return [str(name).upper() for name in cp.installed_solvers()]


def resolve_solver_sequence(requested: str, allow_fallback: bool = True) -> list[str]:
    require_cvxpy()
    installed = installed_solver_names()
    preference = ["MOSEK", "CLARABEL", "SCS"]
    requested = requested.upper()

    if requested == "AUTO":
        sequence = [name for name in preference if name in installed]
    else:
        if requested not in installed:
            raise RuntimeError(
                f"Requested solver {requested!r} is not installed. "
                f"Installed CVXPY solvers: {installed}"
            )
        sequence = [requested]
        if allow_fallback:
            sequence.extend(name for name in preference if name in installed and name != requested)

    if not sequence:
        raise RuntimeError(
            "No supported conic solver is installed. Expected one of MOSEK, "
            f"CLARABEL, or SCS. Installed CVXPY solvers: {installed}"
        )
    return sequence


def solver_options(solver_name: str, *, strict: bool = True) -> dict[str, Any]:
    solver_name = solver_name.upper()
    if solver_name == "MOSEK":
        tolerance = 1.0e-8 if strict else 1.0e-7
        return {
            "mosek_params": {
                "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tolerance,
                "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tolerance,
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tolerance,
            }
        }
    if solver_name == "CLARABEL":
        tolerance = 1.0e-8 if strict else 1.0e-7
        return {
            "tol_gap_abs": tolerance,
            "tol_gap_rel": tolerance,
            "tol_feas": tolerance,
            "max_iter": 500,
        }
    if solver_name == "SCS":
        return {
            "eps": 5.0e-6 if strict else 1.0e-5,
            "max_iters": 200000,
            "acceleration_lookback": 20,
        }
    raise ValueError(f"Unsupported solver: {solver_name}")


class ParametricSocpModel:
    """A fixed target-only or full-grid minimum-norm SOCP model."""

    def __init__(
        self,
        data: SocpInputData,
        *,
        include_sidelobes: bool,
        target_co_fraction: float,
        sll_db: float,
    ) -> None:
        require_cvxpy()
        self.data = data
        self.include_sidelobes = bool(include_sidelobes)
        self.target_co_fraction = float(target_co_fraction)
        self.sll_db = float(sll_db)
        self.eta_sll = float(10.0 ** (self.sll_db / 20.0))
        self.port_count = data.port_count

        self.wr = cp.Variable(self.port_count, name="wr")
        self.wi = cp.Variable(self.port_count, name="wi")
        self.rho_parameter = cp.Parameter(nonneg=True, name="rho_test")

        target_l = np.asarray(
            data.scale_complex * np.asarray(data.a_lhcp_raw[data.target_index, :]),
            dtype=np.complex128,
        )
        target_r = np.asarray(
            data.scale_complex * np.asarray(data.a_rhcp_raw[data.target_index, :]),
            dtype=np.complex128,
        )

        self.target_l_re = target_l.real @ self.wr - target_l.imag @ self.wi
        self.target_l_im = target_l.real @ self.wi + target_l.imag @ self.wr
        self.target_r_re = target_r.real @ self.wr - target_r.imag @ self.wi
        self.target_r_im = target_r.real @ self.wi + target_r.imag @ self.wr

        constraints: list[Any] = [
            self.target_l_im == 0.0,
            self.target_l_re >= self.target_co_fraction,
            cp.norm(cp.hstack([self.target_r_re, self.target_r_im]), 2)
            <= self.rho_parameter * self.target_l_re,
        ]

        self.side_count = 0
        if self.include_sidelobes:
            side_l = np.asarray(
                data.scale_complex * np.asarray(data.a_lhcp_raw[data.side_indices, :]),
                dtype=np.complex128,
            )
            self.side_count = int(side_l.shape[0])
            side_re = side_l.real @ self.wr - side_l.imag @ self.wi
            side_im = side_l.real @ self.wi + side_l.imag @ self.wr
            constraints.append(
                cp.norm(cp.vstack([side_re, side_im]), 2, axis=0)
                <= self.eta_sll * self.target_l_re
            )

        stacked_weights = cp.hstack([self.wr, self.wi])
        self.problem = cp.Problem(cp.Minimize(cp.norm(stacked_weights, 2)), constraints)
        if not self.problem.is_dcp():
            raise RuntimeError("The generated minimum-norm model is not DCP")
        self.is_dpp = bool(self.problem.is_dpp())

    def set_warm_start(self, weights: np.ndarray | None) -> None:
        if weights is None:
            self.wr.value = None
            self.wi.value = None
            return
        weights = np.asarray(weights, dtype=np.complex128).reshape(-1)
        if weights.size != self.port_count:
            raise ValueError("Warm-start weight count mismatch")
        self.wr.value = weights.real.copy()
        self.wi.value = weights.imag.copy()

    def solve_once(
        self,
        rho_test: float,
        *,
        solver_name: str,
        warm_start_w: np.ndarray | None,
        solver_verbose: bool,
        strict: bool,
        power_budget_w: float,
        tolerances: AuditTolerances,
        iteration: int,
        retry_count: int,
    ) -> SocpAttempt:
        solver_name = solver_name.upper()
        rho_test = float(rho_test)
        self.rho_parameter.value = rho_test
        self.set_warm_start(warm_start_w)
        started = time.perf_counter()
        try:
            value = self.problem.solve(
                solver=solver_name,
                verbose=bool(solver_verbose),
                warm_start=True,
                **solver_options(solver_name, strict=strict),
            )
            elapsed = time.perf_counter() - started
        except Exception as exc:
            elapsed = time.perf_counter() - started
            return SocpAttempt(
                iteration=iteration,
                rho_test=rho_test,
                ar_test_db=rho_to_ar_db(rho_test),
                include_sidelobes=self.include_sidelobes,
                solver_name=solver_name,
                solver_status="exception",
                classification=SOLVER_ERROR,
                w_value=None,
                minimum_norm=None,
                minimum_power_w=None,
                audit_passed=False,
                audit={},
                solve_time_s=elapsed,
                solver_iterations=None,
                retry_count=retry_count,
                message=f"{type(exc).__name__}: {exc}",
            )

        status = str(self.problem.status or "unknown").lower()
        stats = self.problem.solver_stats
        iterations = getattr(stats, "num_iters", None)
        minimum_norm = None
        minimum_power = None
        weights = None
        audit: dict[str, Any] = {}
        audit_passed = False
        classification = UNCERTAIN
        message = ""

        if status in {"optimal", "optimal_inaccurate"}:
            if self.wr.value is None or self.wi.value is None:
                classification = UNCERTAIN
                message = "Solver returned an optimal status without primal weights"
            else:
                weights = np.asarray(self.wr.value, dtype=np.float64) + 1j * np.asarray(
                    self.wi.value, dtype=np.float64
                )
                weights = np.asarray(weights, dtype=np.complex128)
                minimum_norm = float(np.linalg.norm(weights))
                minimum_power = float(minimum_norm**2)
                audit = audit_candidate(
                    self.data,
                    weights,
                    rho_test=rho_test,
                    target_co_fraction=self.target_co_fraction,
                    sll_db=self.sll_db,
                    power_budget_w=power_budget_w,
                    include_sidelobes=self.include_sidelobes,
                    tolerances=tolerances,
                )
                audit_passed = bool(audit["shape_constraints_passed"])

                if audit_passed and minimum_power <= power_budget_w + tolerances.power_accept_abs:
                    classification = CERTIFIED_FEASIBLE
                    message = "Audited feasible witness within the total-power budget"
                elif (
                    status == "optimal"
                    and audit_passed
                    and minimum_power > power_budget_w + tolerances.power_reject_abs
                    and solver_name in TRUSTED_INFEASIBILITY_SOLVERS
                ):
                    classification = CERTIFIED_INFEASIBLE
                    message = (
                        "Optimal minimum-power solution exceeds the total-power budget"
                    )
                else:
                    classification = UNCERTAIN
                    message = (
                        "Optimal result lies in a numerical gray zone, uses an untrusted "
                        "lower-bound solver, or fails the independent shape audit"
                    )

        elif status == "infeasible":
            if solver_name in TRUSTED_INFEASIBILITY_SOLVERS:
                classification = CERTIFIED_INFEASIBLE
                message = "Trusted conic solver certified the unbudgeted inner model infeasible"
            else:
                classification = UNCERTAIN
                message = "SCS infeasibility is not used as a certified lower bound"
        elif status in {"infeasible_inaccurate", "unbounded_inaccurate", "user_limit"}:
            classification = UNCERTAIN
            message = f"Noncertifying solver status: {status}"
        else:
            classification = UNCERTAIN
            message = f"Unhandled solver status: {status}; objective={value!r}"

        return SocpAttempt(
            iteration=iteration,
            rho_test=rho_test,
            ar_test_db=rho_to_ar_db(rho_test),
            include_sidelobes=self.include_sidelobes,
            solver_name=solver_name,
            solver_status=status,
            classification=classification,
            w_value=weights,
            minimum_norm=minimum_norm,
            minimum_power_w=minimum_power,
            audit_passed=audit_passed,
            audit=audit,
            solve_time_s=elapsed,
            solver_iterations=int(iterations) if iterations is not None else None,
            retry_count=retry_count,
            message=message,
        )


def solve_rho_with_retries(
    model: ParametricSocpModel,
    rho_test: float,
    *,
    solver_sequence: Sequence[str],
    warm_start_w: np.ndarray | None,
    solver_verbose: bool,
    power_budget_w: float,
    tolerances: AuditTolerances,
    iteration: int,
) -> SocpAttempt:
    messages: list[str] = []
    last_attempt: SocpAttempt | None = None

    for retry_count, solver_name in enumerate(solver_sequence):
        attempt = model.solve_once(
            rho_test,
            solver_name=solver_name,
            warm_start_w=warm_start_w,
            solver_verbose=solver_verbose,
            strict=True,
            power_budget_w=power_budget_w,
            tolerances=tolerances,
            iteration=iteration,
            retry_count=retry_count,
        )
        messages.append(
            f"{solver_name}:{attempt.solver_status}:{attempt.classification}:{attempt.message}"
        )
        last_attempt = attempt
        if attempt.classification in {CERTIFIED_FEASIBLE, CERTIFIED_INFEASIBLE}:
            attempt.message = " | ".join(messages)
            return attempt

    assert last_attempt is not None
    last_attempt.message = " | ".join(messages)
    return last_attempt


# -----------------------------------------------------------------------------
# Bisection and representative solve
# -----------------------------------------------------------------------------


def run_rho_bisection(
    name: str,
    model: ParametricSocpModel,
    *,
    rho_initial_lower: float,
    lower_is_certified_infeasible: bool,
    rho_initial_upper: float,
    initial_feasible_w: np.ndarray,
    rho_tolerance: float,
    max_iterations: int,
    solver_sequence: Sequence[str],
    solver_verbose: bool,
    power_budget_w: float,
    tolerances: AuditTolerances,
    logger: logging.Logger,
    test_zero: bool = True,
) -> BisectionResult:
    started = time.perf_counter()
    rho_low = float(rho_initial_lower)
    rho_high = float(rho_initial_upper)
    if not (0.0 <= rho_low <= rho_high < 1.0):
        raise ValueError(f"Invalid initial rho interval: [{rho_low}, {rho_high}]")

    best_w = np.asarray(initial_feasible_w, dtype=np.complex128).copy()
    baseline_audit = audit_candidate(
        model.data,
        best_w,
        rho_test=rho_high,
        target_co_fraction=model.target_co_fraction,
        sll_db=model.sll_db,
        power_budget_w=power_budget_w,
        include_sidelobes=model.include_sidelobes,
        tolerances=tolerances,
    )
    if not baseline_audit["feasible_under_budget"]:
        raise RuntimeError(
            f"Initial upper-bound weights are not feasible for {name}: {baseline_audit}"
        )

    history: list[SocpAttempt] = []
    best_attempt: SocpAttempt | None = None
    warm_start_w = best_w.copy()

    logger.info(
        "[%s] initial interval rho=[%.9g, %.9g], AR=[%.6f, %.6f] dB",
        name,
        rho_low,
        rho_high,
        rho_to_ar_db(rho_low),
        rho_to_ar_db(rho_high),
    )

    if test_zero and rho_low == 0.0 and not lower_is_certified_infeasible:
        attempt_zero = solve_rho_with_retries(
            model,
            0.0,
            solver_sequence=solver_sequence,
            warm_start_w=warm_start_w,
            solver_verbose=solver_verbose,
            power_budget_w=power_budget_w,
            tolerances=tolerances,
            iteration=0,
        )
        history.append(attempt_zero)
        logger.info(
            "[%s] zero test: status=%s class=%s Pmin=%s",
            name,
            attempt_zero.solver_status,
            attempt_zero.classification,
            "n/a" if attempt_zero.minimum_power_w is None else f"{attempt_zero.minimum_power_w:.9g}",
        )
        if attempt_zero.classification == CERTIFIED_FEASIBLE:
            elapsed = time.perf_counter() - started
            return BisectionResult(
                name=name,
                converged=True,
                rho_lower=0.0,
                rho_upper=0.0,
                ar_lower_db=0.0,
                ar_upper_db=0.0,
                lower_is_certified_infeasible=False,
                upper_is_certified_feasible=True,
                best_feasible_w=attempt_zero.w_value,
                best_feasible_attempt=attempt_zero,
                history=history,
                stop_reason="rho=0 is feasible",
                elapsed_s=elapsed,
            )
        if attempt_zero.classification == CERTIFIED_INFEASIBLE:
            lower_is_certified_infeasible = True
        elif attempt_zero.classification in {UNCERTAIN, SOLVER_ERROR}:
            logger.warning(
                "[%s] rho=0 was not certified; continuing with the mathematical lower bound rho>=0",
                name,
            )

    converged = False
    stop_reason = "maximum iterations reached"
    start_iteration = 1

    for iteration in range(start_iteration, max_iterations + 1):
        if rho_high - rho_low <= rho_tolerance:
            converged = True
            stop_reason = "rho interval tolerance reached"
            break

        rho_mid = 0.5 * (rho_low + rho_high)
        attempt = solve_rho_with_retries(
            model,
            rho_mid,
            solver_sequence=solver_sequence,
            warm_start_w=warm_start_w,
            solver_verbose=solver_verbose,
            power_budget_w=power_budget_w,
            tolerances=tolerances,
            iteration=iteration,
        )
        history.append(attempt)
        logger.info(
            "[%s] iter=%02d rho=%.9g AR=%.6f dB class=%s solver=%s status=%s Pmin=%s",
            name,
            iteration,
            rho_mid,
            attempt.ar_test_db,
            attempt.classification,
            attempt.solver_name,
            attempt.solver_status,
            "n/a" if attempt.minimum_power_w is None else f"{attempt.minimum_power_w:.9g}",
        )

        if attempt.classification == CERTIFIED_FEASIBLE:
            rho_high = rho_mid
            best_w = np.asarray(attempt.w_value, dtype=np.complex128).copy()
            best_attempt = attempt
            warm_start_w = best_w.copy()
        elif attempt.classification == CERTIFIED_INFEASIBLE:
            rho_low = rho_mid
            lower_is_certified_infeasible = True
        else:
            stop_reason = f"uncertain solver result at rho={rho_mid:.12g}"
            converged = False
            break
    else:
        if rho_high - rho_low <= rho_tolerance:
            converged = True
            stop_reason = "rho interval tolerance reached at iteration limit"

    elapsed = time.perf_counter() - started
    return BisectionResult(
        name=name,
        converged=converged,
        rho_lower=rho_low,
        rho_upper=rho_high,
        ar_lower_db=rho_to_ar_db(rho_low),
        ar_upper_db=rho_to_ar_db(rho_high),
        lower_is_certified_infeasible=lower_is_certified_infeasible,
        upper_is_certified_feasible=True,
        best_feasible_w=best_w,
        best_feasible_attempt=best_attempt,
        history=history,
        stop_reason=stop_reason,
        elapsed_s=elapsed,
    )


def solve_representative_weights(
    data: SocpInputData,
    *,
    rho_limit: float,
    target_co_fraction: float,
    sll_db: float,
    power_budget_w: float,
    solver_sequence: Sequence[str],
    warm_start_w: np.ndarray,
    solver_verbose: bool,
    tolerances: AuditTolerances,
) -> RepresentativeResult:
    require_cvxpy()
    port_count = data.port_count
    wr = cp.Variable(port_count, name="rep_wr")
    wi = cp.Variable(port_count, name="rep_wi")

    target_l = np.asarray(
        data.scale_complex * np.asarray(data.a_lhcp_raw[data.target_index, :]),
        dtype=np.complex128,
    )
    target_r = np.asarray(
        data.scale_complex * np.asarray(data.a_rhcp_raw[data.target_index, :]),
        dtype=np.complex128,
    )
    side_l = np.asarray(
        data.scale_complex * np.asarray(data.a_lhcp_raw[data.side_indices, :]),
        dtype=np.complex128,
    )

    target_l_re = target_l.real @ wr - target_l.imag @ wi
    target_l_im = target_l.real @ wi + target_l.imag @ wr
    target_r_re = target_r.real @ wr - target_r.imag @ wi
    target_r_im = target_r.real @ wi + target_r.imag @ wr
    side_re = side_l.real @ wr - side_l.imag @ wi
    side_im = side_l.real @ wi + side_l.imag @ wr

    eta_sll = 10.0 ** (sll_db / 20.0)
    stacked = cp.hstack([wr, wi])
    baseline_vector = weights_to_real_vector(data.w_baseline)
    constraints = [
        target_l_im == 0.0,
        target_l_re >= target_co_fraction,
        cp.norm(cp.hstack([target_r_re, target_r_im]), 2) <= rho_limit * target_l_re,
        cp.norm(cp.vstack([side_re, side_im]), 2, axis=0) <= eta_sll * target_l_re,
        cp.norm(stacked, 2) <= math.sqrt(power_budget_w),
    ]
    problem = cp.Problem(cp.Minimize(cp.norm(stacked - baseline_vector, 2)), constraints)
    if not problem.is_dcp():
        raise RuntimeError("Representative-weight model is not DCP")

    warm_start_w = np.asarray(warm_start_w, dtype=np.complex128)
    wr.value = warm_start_w.real.copy()
    wi.value = warm_start_w.imag.copy()

    messages: list[str] = []
    last_result: RepresentativeResult | None = None
    for solver_name in solver_sequence:
        started = time.perf_counter()
        try:
            value = problem.solve(
                solver=solver_name,
                verbose=bool(solver_verbose),
                warm_start=True,
                **solver_options(solver_name, strict=True),
            )
            elapsed = time.perf_counter() - started
        except Exception as exc:
            elapsed = time.perf_counter() - started
            messages.append(f"{solver_name}:exception:{type(exc).__name__}:{exc}")
            last_result = RepresentativeResult(
                success=False,
                rho_limit=rho_limit,
                w_value=None,
                solver_name=solver_name,
                solver_status="exception",
                distance_to_baseline=None,
                audit={},
                solve_time_s=elapsed,
                message=" | ".join(messages),
            )
            continue

        status = str(problem.status or "unknown").lower()
        if status in {"optimal", "optimal_inaccurate"} and wr.value is not None and wi.value is not None:
            weights = np.asarray(wr.value, dtype=np.float64) + 1j * np.asarray(
                wi.value, dtype=np.float64
            )
            weights = np.asarray(weights, dtype=np.complex128)
            audit = audit_candidate(
                data,
                weights,
                rho_test=rho_limit,
                target_co_fraction=target_co_fraction,
                sll_db=sll_db,
                power_budget_w=power_budget_w,
                include_sidelobes=True,
                tolerances=tolerances,
            )
            success = bool(audit["feasible_under_budget"])
            messages.append(f"{solver_name}:{status}:audit={success}")
            result = RepresentativeResult(
                success=success,
                rho_limit=rho_limit,
                w_value=weights if success else None,
                solver_name=solver_name,
                solver_status=status,
                distance_to_baseline=float(np.linalg.norm(weights - data.w_baseline)),
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
                w_value=None,
                solver_name=solver_name,
                solver_status=status,
                distance_to_baseline=None,
                audit={},
                solve_time_s=elapsed,
                message=" | ".join(messages),
            )

    assert last_result is not None
    return last_result


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------


def attempt_to_history_row(attempt: SocpAttempt) -> dict[str, Any]:
    audit = attempt.audit or {}
    return {
        "iteration": attempt.iteration,
        "rho_test": attempt.rho_test,
        "ar_test_db": attempt.ar_test_db,
        "include_sidelobes": attempt.include_sidelobes,
        "solver": attempt.solver_name,
        "solver_status": attempt.solver_status,
        "classification": attempt.classification,
        "minimum_norm": attempt.minimum_norm,
        "minimum_power_W": attempt.minimum_power_w,
        "target_co_fraction_actual": audit.get("target_fraction"),
        "target_rho_actual": audit.get("target_rho_actual"),
        "target_ar_actual_db": audit.get("target_ar_actual_db"),
        "max_side_ratio_actual": audit.get("max_side_ratio"),
        "max_side_db_actual": audit.get("max_side_db"),
        "phase_anchor_imag": audit.get("scaled_target_imag"),
        "shape_audit_passed": audit.get("shape_constraints_passed"),
        "budget_audit_passed": audit.get("power_passed"),
        "solve_time_s": attempt.solve_time_s,
        "solver_iterations": attempt.solver_iterations,
        "retry_count": attempt.retry_count,
        "message": attempt.message,
    }


def write_history_csv(path: Path, history: Sequence[SocpAttempt]) -> None:
    rows = [attempt_to_history_row(item) for item in history]
    fieldnames = list(rows[0].keys()) if rows else [
        "iteration",
        "rho_test",
        "ar_test_db",
        "classification",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_optimized_weights_csv(
    path: Path,
    data: SocpInputData,
    optimized_weights: np.ndarray,
    *,
    solution_role: str,
) -> None:
    optimized_weights = np.asarray(optimized_weights, dtype=np.complex128)
    baseline = data.w_baseline
    baseline_amp = np.abs(baseline)
    optimized_amp = np.abs(optimized_weights)
    baseline_power = baseline_amp**2
    optimized_power = optimized_amp**2
    baseline_phase = np.rad2deg(np.angle(baseline))
    optimized_phase = np.rad2deg(np.angle(optimized_weights))
    phase_change = np.asarray(wrap_phase_deg(optimized_phase - baseline_phase))
    amplitude_ratio = optimized_amp / np.maximum(baseline_amp, 1.0e-300)
    power_ratio = optimized_power / np.maximum(baseline_power, 1.0e-300)
    amplitude_change_db = 20.0 * np.log10(np.maximum(amplitude_ratio, 1.0e-300))

    output = data.element_table.copy()
    output["solution_role"] = solution_role
    output["baseline_real"] = baseline.real
    output["baseline_imag"] = baseline.imag
    output["baseline_amplitude_sqrtW"] = baseline_amp
    output["baseline_power_W"] = baseline_power
    output["baseline_phase_deg"] = baseline_phase
    output["optimized_real"] = optimized_weights.real
    output["optimized_imag"] = optimized_weights.imag
    output["optimized_amplitude_sqrtW"] = optimized_amp
    output["optimized_power_W"] = optimized_power
    output["optimized_phase_deg"] = optimized_phase
    output["power_ratio_to_baseline"] = power_ratio
    output["amplitude_ratio_to_baseline"] = amplitude_ratio
    output["amplitude_change_db"] = amplitude_change_db
    output["phase_change_deg"] = phase_change
    output.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.16g")


def bisection_summary(result: BisectionResult) -> dict[str, Any]:
    return {
        "name": result.name,
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
        "stop_reason": result.stop_reason,
        "elapsed_s": result.elapsed_s,
    }


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists; use --overwrite to replace it: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def environment_summary() -> dict[str, Any]:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cvxpy_available": cp is not None,
        "cvxpy_import_error": CVXPY_IMPORT_ERROR,
        "cvxpy_version": getattr(cp, "__version__", None) if cp is not None else None,
        "installed_solvers": installed_solver_names(),
    }
    return result


def run_preflight(
    data: SocpInputData,
    *,
    target_co_fraction: float,
    sll_db: float,
    power_budget_w: float,
    tolerances: AuditTolerances,
) -> dict[str, Any]:
    baseline = audit_baseline(
        data,
        target_co_fraction=target_co_fraction,
        sll_db=sll_db,
        power_budget_w=power_budget_w,
        tolerances=tolerances,
    )
    return {
        "status": "PASS" if baseline["passed"] else "FAILED",
        "script_version": SCRIPT_VERSION,
        "point_count": data.point_count,
        "port_count": data.port_count,
        "target_index": data.target_index,
        "main_count": int(data.main_indices.size),
        "side_count": int(data.side_indices.size),
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
    requested_solver: str,
    allow_solver_fallback: bool,
    target_co_fraction: float,
    sll_db: float,
    power_budget_w: float,
    rho_tolerance: float,
    max_bisection_iterations: int,
    representative_rho_margin: float,
    skip_target_only: bool,
    skip_representative: bool,
    solver_verbose: bool,
    tolerances: AuditTolerances,
    logger: logging.Logger,
) -> dict[str, Any]:
    require_cvxpy()
    solver_sequence = resolve_solver_sequence(
        requested_solver, allow_fallback=allow_solver_fallback
    )
    logger.info("[SOLVER] sequence=%s", solver_sequence)

    baseline = audit_baseline(
        data,
        target_co_fraction=target_co_fraction,
        sll_db=sll_db,
        power_budget_w=power_budget_w,
        tolerances=tolerances,
    )
    if not baseline["passed"]:
        raise RuntimeError(f"Baseline is not a valid initial feasible upper bound: {baseline}")

    rho_upper_initial = min(
        0.999999,
        data.baseline_rho + max(1.0e-10, 1.0e-9 * max(1.0, data.baseline_rho)),
    )

    target_result: BisectionResult | None = None
    if not skip_target_only:
        logger.info("[TARGET-ONLY] building model")
        target_model = ParametricSocpModel(
            data,
            include_sidelobes=False,
            target_co_fraction=target_co_fraction,
            sll_db=sll_db,
        )
        logger.info("[TARGET-ONLY] model DPP=%s", target_model.is_dpp)
        target_result = run_rho_bisection(
            "TARGET-ONLY",
            target_model,
            rho_initial_lower=0.0,
            lower_is_certified_infeasible=False,
            rho_initial_upper=rho_upper_initial,
            initial_feasible_w=data.w_baseline,
            rho_tolerance=rho_tolerance,
            max_iterations=max_bisection_iterations,
            solver_sequence=solver_sequence,
            solver_verbose=solver_verbose,
            power_budget_w=power_budget_w,
            tolerances=tolerances,
            logger=logger,
            test_zero=True,
        )
        write_history_csv(output_dir / TARGET_HISTORY_NAME, target_result.history)
        if not target_result.converged:
            raise RuntimeError(
                f"Target-only bisection did not converge: {target_result.stop_reason}"
            )

    logger.info("[FULL-GRID] building model with %d sidelobe points", data.side_indices.size)
    full_model = ParametricSocpModel(
        data,
        include_sidelobes=True,
        target_co_fraction=target_co_fraction,
        sll_db=sll_db,
    )
    logger.info("[FULL-GRID] model DPP=%s", full_model.is_dpp)

    full_lower = 0.0
    full_lower_certified = False
    if target_result is not None:
        full_lower = target_result.rho_lower
        full_lower_certified = target_result.lower_is_certified_infeasible

    full_result = run_rho_bisection(
        "FULL-GRID",
        full_model,
        rho_initial_lower=full_lower,
        lower_is_certified_infeasible=full_lower_certified,
        rho_initial_upper=rho_upper_initial,
        initial_feasible_w=data.w_baseline,
        rho_tolerance=rho_tolerance,
        max_iterations=max_bisection_iterations,
        solver_sequence=solver_sequence,
        solver_verbose=solver_verbose,
        power_budget_w=power_budget_w,
        tolerances=tolerances,
        logger=logger,
        test_zero=(full_lower == 0.0 and not full_lower_certified),
    )
    write_history_csv(output_dir / FULL_HISTORY_NAME, full_result.history)
    if not full_result.converged:
        raise RuntimeError(f"Full-grid bisection did not converge: {full_result.stop_reason}")
    if full_result.best_feasible_w is None:
        raise RuntimeError("Full-grid bisection did not retain a feasible weight vector")

    rho_representative = min(
        data.baseline_rho,
        full_result.rho_upper + max(representative_rho_margin, 5.0 * rho_tolerance),
    )
    representative: RepresentativeResult | None = None
    if not skip_representative:
        logger.info(
            "[REPRESENTATIVE] solving closest-to-baseline weights at rho_limit=%.9g (AR=%.6f dB)",
            rho_representative,
            rho_to_ar_db(rho_representative),
        )
        representative = solve_representative_weights(
            data,
            rho_limit=rho_representative,
            target_co_fraction=target_co_fraction,
            sll_db=sll_db,
            power_budget_w=power_budget_w,
            solver_sequence=solver_sequence,
            warm_start_w=full_result.best_feasible_w,
            solver_verbose=solver_verbose,
            tolerances=tolerances,
        )
        if not representative.success:
            raise RuntimeError(
                f"Representative solve failed independent audit: {representative.message}"
            )

    selected_w = (
        representative.w_value
        if representative is not None and representative.success
        else full_result.best_feasible_w
    )
    assert selected_w is not None
    selected_role = "representative" if representative is not None else "fullgrid_best_feasible"
    selected_rho_limit = (
        representative.rho_limit if representative is not None else full_result.rho_upper
    )
    selected_audit = audit_candidate(
        data,
        selected_w,
        rho_test=selected_rho_limit,
        target_co_fraction=target_co_fraction,
        sll_db=sll_db,
        power_budget_w=power_budget_w,
        include_sidelobes=True,
        tolerances=tolerances,
    )
    if not selected_audit["feasible_under_budget"]:
        raise RuntimeError(f"Final selected weights failed independent audit: {selected_audit}")

    target_best_w = (
        target_result.best_feasible_w
        if target_result is not None and target_result.best_feasible_w is not None
        else np.empty(0, dtype=np.complex128)
    )
    representative_w = (
        representative.w_value
        if representative is not None and representative.w_value is not None
        else np.empty(0, dtype=np.complex128)
    )

    np.savez(
        output_dir / SOLUTION_NAME,
        w_baseline=np.asarray(data.w_baseline, dtype=np.complex128),
        w_target_only_best=np.asarray(target_best_w, dtype=np.complex128),
        w_fullgrid_best_feasible=np.asarray(full_result.best_feasible_w, dtype=np.complex128),
        w_representative=np.asarray(representative_w, dtype=np.complex128),
        w_selected=np.asarray(selected_w, dtype=np.complex128),
        rho_target_lower=np.array(
            np.nan if target_result is None else target_result.rho_lower, dtype=np.float64
        ),
        rho_target_upper=np.array(
            np.nan if target_result is None else target_result.rho_upper, dtype=np.float64
        ),
        rho_fullgrid_lower=np.array(full_result.rho_lower, dtype=np.float64),
        rho_fullgrid_upper=np.array(full_result.rho_upper, dtype=np.float64),
        rho_representative=np.array(rho_representative, dtype=np.float64),
        target_index=np.array(data.target_index, dtype=np.int64),
        side_indices=np.asarray(data.side_indices, dtype=np.int64),
    )
    write_optimized_weights_csv(
        output_dir / WEIGHTS_NAME,
        data,
        selected_w,
        solution_role=selected_role,
    )

    summary = {
        "status": "complete",
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
            "target_co_fraction": target_co_fraction,
            "sll_db": sll_db,
            "power_budget_w": power_budget_w,
            "rho_tolerance": rho_tolerance,
            "max_bisection_iterations": max_bisection_iterations,
        },
        "solver_sequence": list(solver_sequence),
        "baseline": baseline,
        "target_only": None if target_result is None else bisection_summary(target_result),
        "full_grid": bisection_summary(full_result),
        "representative": None
        if representative is None
        else {
            "success": representative.success,
            "rho_limit": representative.rho_limit,
            "ar_limit_db": rho_to_ar_db(representative.rho_limit),
            "solver": representative.solver_name,
            "solver_status": representative.solver_status,
            "distance_to_baseline": representative.distance_to_baseline,
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
        "output_files": {
            "solution_npz": SOLUTION_NAME,
            "optimized_weights_csv": WEIGHTS_NAME,
            "target_history_csv": None if target_result is None else TARGET_HISTORY_NAME,
            "full_history_csv": FULL_HISTORY_NAME,
        },
    }
    write_json(output_dir / SUMMARY_NAME, summary)
    return summary


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-grid Vec-AEP axis-ratio synthesis via bisection and SOCP"
    )
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    parser.add_argument("--vecaep-cache", type=Path)
    parser.add_argument("--grid-domain", type=Path)
    parser.add_argument("--source-weights", type=Path)
    parser.add_argument("--output-dir", type=Path)

    parser.add_argument("--solver", default="AUTO", choices=["AUTO", "MOSEK", "CLARABEL", "SCS"])
    parser.add_argument("--no-solver-fallback", action="store_true")
    parser.add_argument("--target-co-fraction", type=float, default=DEFAULT_TARGET_CO_FRACTION)
    parser.add_argument("--sll-db", type=float, default=DEFAULT_SLL_DB)
    parser.add_argument("--power-budget-w", type=float, default=DEFAULT_POWER_BUDGET_W)
    parser.add_argument("--rho-tol", type=float, default=DEFAULT_RHO_TOL)
    parser.add_argument(
        "--max-bisection-iterations", type=int, default=DEFAULT_MAX_BISECTION_ITERATIONS
    )
    parser.add_argument(
        "--representative-rho-margin",
        type=float,
        default=DEFAULT_REPRESENTATIVE_RHO_MARGIN,
    )

    parser.add_argument("--power-accept-tol", type=float, default=2.0e-6)
    parser.add_argument("--power-reject-tol", type=float, default=1.0e-5)
    parser.add_argument("--constraint-abs-tol", type=float, default=2.0e-6)

    parser.add_argument("--skip-target-only", action="store_true")
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
        raise ValueError("--target-co-fraction must be in (0, 1]")
    if args.sll_db >= 0.0:
        raise ValueError("--sll-db must be negative")
    if args.power_budget_w <= 0.0:
        raise ValueError("--power-budget-w must be positive")
    if args.rho_tol <= 0.0:
        raise ValueError("--rho-tol must be positive")
    if args.max_bisection_iterations <= 0:
        raise ValueError("--max-bisection-iterations must be positive")
    if args.power_accept_tol < 0.0 or args.power_reject_tol <= args.power_accept_tol:
        raise ValueError("Power rejection tolerance must be greater than acceptance tolerance")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_args(args)
        output_dir = prepare_output_dir(args.output_dir, overwrite=args.overwrite)
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
        logger.info("[VERSION] stage3_fullgrid_socp.py %s", SCRIPT_VERSION)
        logger.info("[ENV] CVXPY available=%s, solvers=%s", cp is not None, installed_solver_names())
        logger.info("[LOAD] vecaep_cache=%s", args.vecaep_cache)
        logger.info("[LOAD] grid_domain=%s", args.grid_domain)
        logger.info("[LOAD] source_weights=%s", args.source_weights)
        data = load_socp_input_data(
            args.vecaep_cache,
            args.grid_domain,
            args.source_weights,
        )
        logger.info(
            "[PASS] formal data loaded: points=%d ports=%d target=%d side=%d",
            data.point_count,
            data.port_count,
            data.target_index,
            data.side_indices.size,
        )
        preflight = run_preflight(
            data,
            target_co_fraction=args.target_co_fraction,
            sll_db=args.sll_db,
            power_budget_w=args.power_budget_w,
            tolerances=tolerances,
        )
        write_json(output_dir / PREFLIGHT_NAME, preflight)
        if preflight["status"] != "PASS":
            raise RuntimeError(f"Preflight baseline audit failed: {preflight['baseline']}")
        logger.info(
            "[PASS] baseline: power=%.9g W, AR=%.9f dB, side=%.9f dB",
            data.baseline_power_w,
            data.baseline_ar_db,
            data.baseline_side_db,
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
            requested_solver=args.solver,
            allow_solver_fallback=not args.no_solver_fallback,
            target_co_fraction=args.target_co_fraction,
            sll_db=args.sll_db,
            power_budget_w=args.power_budget_w,
            rho_tolerance=args.rho_tol,
            max_bisection_iterations=args.max_bisection_iterations,
            representative_rho_margin=args.representative_rho_margin,
            skip_target_only=args.skip_target_only,
            skip_representative=args.skip_representative,
            solver_verbose=args.solver_verbose,
            tolerances=tolerances,
            logger=logger,
        )
        duration = time.perf_counter() - started
        complete = {
            "status": "complete",
            "script_version": SCRIPT_VERSION,
            "duration_s": duration,
            "bisection_converged": bool(summary["full_grid"]["converged"]),
            "numpy_audit_passed": bool(
                summary["selected_solution"]["audit"]["feasible_under_budget"]
            ),
            "summary_file": SUMMARY_NAME,
            "solution_file": SOLUTION_NAME,
            "weights_file": WEIGHTS_NAME,
        }
        write_json(output_dir / COMPLETE_MARKER, complete)
        incomplete_path.unlink(missing_ok=True)
        logger.info("[SUMMARY] status=PASS")
        logger.info(
            "[SUMMARY] full-grid AR interval=[%.9f, %.9f] dB",
            summary["full_grid"]["ar_lower_db"],
            summary["full_grid"]["ar_upper_db"],
        )
        logger.info(
            "[SUMMARY] selected AR=%.9f dB power=%.9g W",
            summary["selected_solution"]["audit"]["target_ar_actual_db"],
            summary["selected_solution"]["audit"]["power_w"],
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
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
