#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrate the intrinsic 1D-core / 2D-sidelobe sampling partition.

This version intentionally uses one fixed intrinsic sampling grid for every
(K, d_min, b) case.  It does not use base/medium/fine refinement, candidate-b
screening, or adaptive convergence checks.

The program compares the exact finite-element q=0 concentric-ring array factor
with its J0 proxy in rho = sin(alpha_intrinsic), extracts the first persistent
dual-gate failure boundary, and takes the most conservative boundary over b for
each (K, d_min) pair.

This is a geometry-only calibration, independent of the IEP dataset. It does
not calibrate scan-induced uv distortion or the polarization response. The
q0cra_stage1_partition_v2 data schema is intentionally frozen.

Supported runtime: Python 3.6+.
"""

import argparse
import csv
import datetime
import hashlib
import json
import logging
import math
import os
import platform
import sys
import time

import numpy as np
import scipy
from scipy.special import j0


FORMAT_VERSION = "q0cra_stage1_partition_v2"
CALIBRATION_REVISION = "single_fixed_grid_full_sweep_v1"
# These serialized identifiers are consumed by the static pool generator.
BOUNDARY_MODE_CODE = {
    "no_bad_run": 0,
    "dual_gate_linear": 1,
    "last_safe_grid_fallback": 2,
    "boundary_at_zero": 3,
}


# =============================================================================
# Python-3.6-compatible record classes
# =============================================================================


class IntrinsicPoolConfig(object):
    __slots__ = (
        "alpha_step_deg",
        "alpha_max_deg",
        "eta_arc_step_deg",
        "n_min_per_ring",
    )

    def __init__(self, alpha_step_deg=0.5, alpha_max_deg=90.0,
                 eta_arc_step_deg=4.0, n_min_per_ring=12):
        self.alpha_step_deg = float(alpha_step_deg)
        self.alpha_max_deg = float(alpha_max_deg)
        self.eta_arc_step_deg = float(eta_arc_step_deg)
        self.n_min_per_ring = int(n_min_per_ring)


class AlphaRingSpec(object):
    __slots__ = (
        "ring_id", "alpha_deg", "alpha_rad", "rho", "n_eta", "start", "stop",
    )

    def __init__(self, ring_id, alpha_deg, alpha_rad, rho, n_eta, start, stop):
        self.ring_id = int(ring_id)
        self.alpha_deg = float(alpha_deg)
        self.alpha_rad = float(alpha_rad)
        self.rho = float(rho)
        self.n_eta = int(n_eta)
        self.start = int(start)
        self.stop = int(stop)


class IntrinsicRingPool(object):
    __slots__ = (
        "cfg", "alpha_grid_deg", "alpha_grid_rad", "rho_grid",
        "gamma_eval_deg", "gamma_eval_rad", "rho_eval", "eta_eval_rad",
        "ring_specs", "total_points",
    )

    def __init__(self, cfg, alpha_grid_deg, alpha_grid_rad, rho_grid,
                 gamma_eval_deg, gamma_eval_rad, rho_eval, eta_eval_rad,
                 ring_specs, total_points):
        self.cfg = cfg
        self.alpha_grid_deg = alpha_grid_deg
        self.alpha_grid_rad = alpha_grid_rad
        self.rho_grid = rho_grid
        self.gamma_eval_deg = gamma_eval_deg
        self.gamma_eval_rad = gamma_eval_rad
        self.rho_eval = rho_eval
        self.eta_eval_rad = eta_eval_rad
        self.ring_specs = ring_specs
        self.total_points = int(total_points)


class EvalConfig(object):
    __slots__ = (
        "beta", "eps", "xi_core", "peak_gate_db",
        "persistent_bad_width_deg", "isolated_bad_policy",
        "n_consecutive_bad_legacy",
    )

    def __init__(self, beta=2.0 * math.pi, eps=1e-12, xi_core=0.01,
                 peak_gate_db=-40.0, persistent_bad_width_deg=0.5,
                 isolated_bad_policy="warn", n_consecutive_bad_legacy=2):
        self.beta = float(beta)
        self.eps = float(eps)
        self.xi_core = float(xi_core)
        self.peak_gate_db = float(peak_gate_db)
        self.persistent_bad_width_deg = float(persistent_bad_width_deg)
        self.isolated_bad_policy = str(isolated_bad_policy)
        self.n_consecutive_bad_legacy = int(n_consecutive_bad_legacy)


class StructureCase(object):
    __slots__ = ("K", "d_min", "b", "r_vec", "N_vec")

    def __init__(self, K, d_min, b, r_vec, N_vec):
        self.K = int(K)
        self.d_min = float(d_min)
        self.b = float(b)
        self.r_vec = np.asarray(r_vec, dtype=np.float64)
        self.N_vec = np.asarray(N_vec, dtype=np.int64)


class CaseCurves(object):
    __slots__ = (
        "K", "d_min", "b", "gamma_inf", "gamma_prefix", "ring_peak_db",
        "proxy_abs", "exact_peak_abs", "proxy_db", "exact_peak_db",
        "absolute_error_peak", "absolute_error_normalized", "r_vec", "N_vec",
    )

    def __init__(self, K, d_min, b, gamma_inf, gamma_prefix, ring_peak_db,
                 proxy_abs, exact_peak_abs, proxy_db, exact_peak_db,
                 absolute_error_peak, absolute_error_normalized, r_vec, N_vec):
        self.K = int(K)
        self.d_min = float(d_min)
        self.b = float(b)
        self.gamma_inf = gamma_inf
        self.gamma_prefix = gamma_prefix
        self.ring_peak_db = ring_peak_db
        self.proxy_abs = proxy_abs
        self.exact_peak_abs = exact_peak_abs
        self.proxy_db = proxy_db
        self.exact_peak_db = exact_peak_db
        self.absolute_error_peak = absolute_error_peak
        self.absolute_error_normalized = absolute_error_normalized
        self.r_vec = r_vec
        self.N_vec = N_vec


class BoundaryResult(object):
    __slots__ = (
        "xi", "peak_gate_db", "alpha_boundary_deg", "alpha_boundary_rad",
        "rho_boundary", "error_bad_mask", "amplitude_active_mask",
        "dual_bad_mask", "first_dual_bad_idx", "first_bad_run_start_idx",
        "first_bad_run_stop_idx", "last_safe_idx", "triggered_by",
        "interpolation_mode", "alpha_gamma_cross_deg", "alpha_peak_cross_deg",
        "isolated_bad_indices", "isolated_bad_count", "first_isolated_bad_deg",
        "max_isolated_gamma", "max_isolated_peak_db", "alpha_grid_step_deg",
        "persistent_bad_width_deg", "required_bad_sample_count",
        "actual_bad_span_deg", "warnings",
    )

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs[name])


class CasePartitionResult(object):
    __slots__ = (
        "K", "d_min", "b", "N_total", "N_ring_total", "N_total_with_center",
        "include_center", "alpha_1d_deg", "alpha_1d_rad", "rho_1d",
        "first_dual_bad_idx", "first_bad_run_start_idx", "first_bad_run_stop_idx",
        "last_safe_idx", "triggered_by", "interpolation_mode",
        "alpha_gamma_cross_deg", "alpha_peak_cross_deg", "isolated_bad_count",
        "first_isolated_bad_deg", "max_isolated_gamma", "max_isolated_peak_db",
        "gamma_inf_max", "ring_peak_db_max", "alpha_grid_step_deg",
        "persistent_bad_width_deg", "required_bad_sample_count",
        "actual_bad_span_deg", "r_vec", "N_vec", "warnings",
    )

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs[name])


class ComboPartitionResult(object):
    __slots__ = (
        "K", "d_min", "d_eff", "R0", "alpha_core_calib_deg",
        "alpha_core_used_deg", "rho_core_used", "alpha_outer_start_deg",
        "worst_b", "alpha_1d_min_over_b_deg", "alpha_1d_mean_over_b_deg",
        "alpha_1d_max_over_b_deg", "rho_1d_min_over_b",
        "rho_1d_mean_over_b", "rho_1d_max_over_b", "N_total_at_worst_b",
        "N_ring_total_at_worst_b", "N_total_with_center_at_worst_b",
        "worst_case_interpolation_mode", "worst_case_triggered_by",
        "isolated_bad_case_count", "max_isolated_bad_count",
        "all_cases_have_persistent_boundary",
    )

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs[name])


class MainConfig(object):
    __slots__ = (
        "out_dir", "R0", "d_eff", "bind_d_eff_to_d_min", "K_list",
        "d_min_list", "b_min", "b_max", "b_step", "beta",
        "safety_margin_deg", "alpha_min_allowed_deg", "alpha_outer_start_deg",
        "outer5_reduce_mult4", "outer5_threshold_gt", "overwrite",
    )

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs[name])


# =============================================================================
# Utilities
# =============================================================================


def _record_to_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _record_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_record_to_dict(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if hasattr(obj, "__slots__"):
        return {name: _record_to_dict(getattr(obj, name)) for name in obj.__slots__}
    return obj


def _jsonable(obj):
    return _record_to_dict(obj)


def _ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def _float_range(start, stop, step):
    if float(step) <= 0.0:
        raise ValueError("step must be > 0")
    if float(stop) < float(start):
        raise ValueError("stop must be >= start")
    n = int(np.floor((float(stop) - float(start)) / float(step) + 1e-12)) + 1
    arr = float(start) + float(step) * np.arange(n, dtype=np.float64)
    arr = arr[arr <= float(stop) + 1e-12]
    return np.round(arr, 12).astype(np.float64)


def _parse_int_list(s):
    out = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not out:
        raise ValueError("empty integer list")
    return out


def _parse_float_list(s):
    out = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if not out:
        raise ValueError("empty float list")
    return out


def _safe_min(x, default=float("nan")):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else float(default)


def _safe_max(x, default=float("nan")):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else float(default)


def _safe_mean(x, default=float("nan")):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float(default)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _utc_now_iso():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_csv(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        if rows:
            keys = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_savez(path, payload):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def _setup_logger(log_path):
    logger = logging.getLogger("calibrate_q0_partition")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False
    formatter_file = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    formatter_console = logging.Formatter("%(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(formatter_file)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter_console)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _close_logger(logger):
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)


def _check_output_paths(paths, overwrite):
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not overwrite:
        raise IOError(
            "Output files already exist; use --overwrite or a new --out-dir:\n  "
            + "\n  ".join(existing)
        )


def resolve_persistent_bad_width(alpha_step_deg, n_consecutive_bad,
                                 persistent_bad_width_deg):
    if persistent_bad_width_deg is not None:
        width = float(persistent_bad_width_deg)
    else:
        n_bad = int(n_consecutive_bad)
        if n_bad < 1:
            raise ValueError("n_consecutive_bad must be >= 1")
        width = float(max(0, n_bad - 1)) * float(alpha_step_deg)
    if width < 0.0 or not np.isfinite(width):
        raise ValueError("persistent_bad_width_deg must be finite and >= 0")
    return float(width)


# =============================================================================
# Geometry / structure generation
# =============================================================================


def init_r_gompertz(*, K, R0, d_min, b):
    if int(K) < 1:
        raise ValueError("K must be >= 1")
    if float(d_min) <= 0.0 or float(R0) <= 0.0:
        raise ValueError("d_min and R0 must be > 0")
    dK = int(K) * float(d_min)
    if float(R0) < dK - 1e-12:
        raise ValueError("R0({0}) < K*d_min({1})".format(R0, dK))
    m = np.arange(1, int(K) + 1, dtype=np.float64)
    d_m = m * float(d_min)
    if int(K) == 1:
        return np.array([float(R0)], dtype=np.float64)
    t_m = (m - 1.0) / (int(K) - 1.0)
    if abs(float(b)) < 1e-12:
        gbar = t_m
    else:
        denom = np.exp(float(b)) - 1.0
        gbar = (np.exp(float(b) * t_m) - 1.0) / denom if abs(denom) > 1e-15 else t_m
    l_m = (float(R0) - dK) * gbar
    r = d_m + l_m
    r[0] = float(d_min)
    r[-1] = float(R0)
    if np.any(np.diff(r) <= 0.0):
        raise ValueError("Non-monotone Gompertz radii produced")
    return r.astype(np.float64)


def derive_ring_counts(r_over_lambda, *, d_eff=0.5,
                       outer5_reduce_mult4=True, outer5_threshold_gt=16):
    r = np.asarray(r_over_lambda, dtype=np.float64).reshape(-1)
    if r.size == 0 or np.any(~np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("r_over_lambda must contain finite positive values")
    if float(d_eff) <= 0.0 or not np.isfinite(float(d_eff)):
        raise ValueError("d_eff must be finite and > 0")
    n_full = np.floor((2.0 * math.pi * r) / float(d_eff) + 1e-12).astype(np.int64)
    n_full = np.maximum(n_full, 1)
    n_used = n_full.copy()
    if bool(outer5_reduce_mult4) and r.size >= 5:
        start = max(0, r.size - 5)
        for k in range(start, r.size):
            if int(n_used[k]) > int(outer5_threshold_gt) and int(n_used[k]) % 4 == 0:
                n_used[k] -= 1
    return n_used.astype(np.int64)


def build_element_angles(N_vec):
    out = []
    for N in np.asarray(N_vec, dtype=np.int64).reshape(-1):
        if int(N) <= 0:
            raise ValueError("N_vec must contain positive integers")
        psi = 2.0 * math.pi * np.arange(int(N), dtype=np.float64) / float(N)
        out.append(psi.astype(np.float64))
    return out


# =============================================================================
# Intrinsic spectral-ring pool
# =============================================================================


def build_intrinsic_ring_pool(cfg):
    if cfg.alpha_step_deg <= 0.0:
        raise ValueError("alpha_step_deg must be > 0")
    if cfg.alpha_max_deg <= 0.0 or cfg.alpha_max_deg > 90.0 + 1e-12:
        raise ValueError("alpha_max_deg must be in (0, 90]")
    if cfg.eta_arc_step_deg <= 0.0:
        raise ValueError("eta_arc_step_deg must be > 0")
    if cfg.n_min_per_ring < 1:
        raise ValueError("n_min_per_ring must be >= 1")

    alpha_grid_deg = _float_range(0.0, cfg.alpha_max_deg, cfg.alpha_step_deg)
    if alpha_grid_deg.size < 2 or np.any(np.diff(alpha_grid_deg) <= 0.0):
        raise RuntimeError("alpha grid must contain at least two strictly increasing samples")
    alpha_grid_rad = np.deg2rad(alpha_grid_deg)
    rho_grid = np.sin(alpha_grid_rad)
    eta_arc_step_rad = math.radians(cfg.eta_arc_step_deg)

    gamma_eval_deg = [0.0]
    gamma_eval_rad = [0.0]
    rho_eval = [0.0]
    eta_eval_rad = [0.0]
    ring_specs = [AlphaRingSpec(0, 0.0, 0.0, 0.0, 1, 0, 1)]
    start = 1
    for ring_id in range(1, alpha_grid_deg.size):
        alpha_deg = float(alpha_grid_deg[ring_id])
        alpha_rad = float(alpha_grid_rad[ring_id])
        rho = float(rho_grid[ring_id])
        circumference = 2.0 * math.pi * rho
        n_eta = max(cfg.n_min_per_ring, int(math.ceil(circumference / eta_arc_step_rad)))
        eta = 2.0 * math.pi * (np.arange(n_eta, dtype=np.float64) + 0.5) / float(n_eta)
        gamma_eval_deg.extend([alpha_deg] * n_eta)
        gamma_eval_rad.extend([alpha_rad] * n_eta)
        rho_eval.extend([rho] * n_eta)
        eta_eval_rad.extend(eta.tolist())
        stop = start + n_eta
        ring_specs.append(AlphaRingSpec(ring_id, alpha_deg, alpha_rad, rho, n_eta, start, stop))
        start = stop

    return IntrinsicRingPool(
        cfg=cfg,
        alpha_grid_deg=alpha_grid_deg.astype(np.float64),
        alpha_grid_rad=alpha_grid_rad.astype(np.float64),
        rho_grid=rho_grid.astype(np.float64),
        gamma_eval_deg=np.asarray(gamma_eval_deg, dtype=np.float64),
        gamma_eval_rad=np.asarray(gamma_eval_rad, dtype=np.float64),
        rho_eval=np.asarray(rho_eval, dtype=np.float64),
        eta_eval_rad=np.asarray(eta_eval_rad, dtype=np.float64),
        ring_specs=ring_specs,
        total_points=len(gamma_eval_deg),
    )


def required_bad_sample_count(alpha_grid_deg, persistent_bad_width_deg):
    alpha = np.asarray(alpha_grid_deg, dtype=np.float64).reshape(-1)
    if alpha.size < 2:
        raise ValueError("alpha grid must contain at least two points")
    if np.any(~np.isfinite(alpha)):
        raise ValueError("alpha grid contains NaN/Inf")
    diff = np.diff(alpha)
    if np.any(diff <= 0.0):
        raise ValueError("alpha grid must be strictly increasing")
    step = float(np.median(diff))
    if not np.allclose(diff, step, atol=1e-10, rtol=0.0):
        raise ValueError("alpha grid must be uniformly spaced")
    width = float(persistent_bad_width_deg)
    if width < 0.0 or not np.isfinite(width):
        raise ValueError("persistent_bad_width_deg must be finite and >= 0")
    count = int(math.ceil(width / step - 1e-12)) + 1
    count = max(1, count)
    actual_span = float(count - 1) * step
    return {
        "alpha_grid_step_deg": float(step),
        "persistent_bad_width_deg": float(width),
        "required_bad_sample_count": int(count),
        "actual_bad_span_deg": float(actual_span),
    }


# =============================================================================
# Exact finite-ring q=0 response versus its continuous-ring J0 proxy
# =============================================================================


def evaluate_case_intrinsic(case, pool, cfg):
    r_vec = np.asarray(case.r_vec, dtype=np.float64).reshape(-1)
    N_vec = np.asarray(case.N_vec, dtype=np.int64).reshape(-1)
    if r_vec.size == 0 or r_vec.size != N_vec.size:
        raise ValueError("r_vec/N_vec must be non-empty and have matching lengths")
    if np.any(~np.isfinite(r_vec)) or np.any(r_vec <= 0.0):
        raise ValueError("r_vec must contain finite positive values")
    if np.any(N_vec <= 0):
        raise ValueError("N_vec must contain positive integers")
    if np.any(np.diff(r_vec) <= 0.0):
        raise ValueError("r_vec must be strictly increasing")

    psi_elem = build_element_angles(N_vec)
    shape = pool.alpha_grid_rad.shape
    gamma_inf = np.zeros(shape, dtype=np.float64)
    ring_peak_db = np.full(shape, -np.inf, dtype=np.float64)
    proxy_abs = np.zeros(shape, dtype=np.float64)
    exact_peak_abs = np.zeros(shape, dtype=np.float64)
    proxy_db = np.full(shape, -np.inf, dtype=np.float64)
    exact_peak_db = np.full(shape, -np.inf, dtype=np.float64)
    absolute_error_peak = np.zeros(shape, dtype=np.float64)
    absolute_error_normalized = np.zeros(shape, dtype=np.float64)

    n_float = N_vec.astype(np.float64)
    a_ref = 1.0 + float(np.sum(n_float))
    a_ref = max(abs(a_ref), cfg.eps)

    for rs in pool.ring_specs:
        i = rs.ring_id
        rho = rs.rho
        proxy = 1.0 + float(np.sum(n_float * j0(cfg.beta * r_vec * rho)))
        p_abs = abs(proxy)
        if i == 0:
            exact_vec = np.asarray([1.0 + float(np.sum(n_float))], dtype=np.complex128)
        else:
            eta = pool.eta_eval_rad[rs.start:rs.stop]
            exact_vec = np.ones((eta.size,), dtype=np.complex128)
            for rk, pe in zip(r_vec.tolist(), psi_elem):
                phase = cfg.beta * float(rk) * rho * np.cos(eta[:, None] - pe[None, :])
                exact_vec += np.sum(np.exp(1j * phase), axis=1)

        abs_err = np.abs(exact_vec - proxy)
        e_peak = float(np.max(abs_err)) if abs_err.size else 0.0
        x_peak = float(np.max(np.abs(exact_vec))) if exact_vec.size else 0.0
        gamma_inf[i] = e_peak / (p_abs + cfg.eps)
        proxy_abs[i] = p_abs
        exact_peak_abs[i] = x_peak
        absolute_error_peak[i] = e_peak
        absolute_error_normalized[i] = e_peak / a_ref
        proxy_db[i] = 20.0 * math.log10(max(p_abs, cfg.eps) / a_ref)
        exact_peak_db[i] = 20.0 * math.log10(max(x_peak, cfg.eps) / a_ref)
        ring_peak_db[i] = max(proxy_db[i], exact_peak_db[i])

    arrays = (
        gamma_inf, ring_peak_db, proxy_abs, exact_peak_abs,
        proxy_db, exact_peak_db, absolute_error_peak, absolute_error_normalized,
    )
    if any(np.any(~np.isfinite(arr)) for arr in arrays):
        raise FloatingPointError("case evaluation produced NaN/Inf")

    gamma_prefix = np.maximum.accumulate(gamma_inf)
    return CaseCurves(
        K=case.K, d_min=case.d_min, b=case.b,
        gamma_inf=gamma_inf, gamma_prefix=gamma_prefix,
        ring_peak_db=ring_peak_db, proxy_abs=proxy_abs,
        exact_peak_abs=exact_peak_abs, proxy_db=proxy_db,
        exact_peak_db=exact_peak_db,
        absolute_error_peak=absolute_error_peak,
        absolute_error_normalized=absolute_error_normalized,
        r_vec=r_vec.copy(), N_vec=N_vec.copy(),
    )


def find_first_consecutive_true(mask, run_length):
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    n = int(run_length)
    if n < 1:
        raise ValueError("run_length must be >= 1")
    run = 0
    for j, value in enumerate(arr.tolist()):
        if value:
            run += 1
            if run >= n:
                start = j - n + 1
                return int(start), int(j)
        else:
            run = 0
    return -1, -1


def _linear_upward_crossing(x0, x1, y0, y1, threshold):
    values = (x0, x1, y0, y1, threshold)
    if not all(np.isfinite(float(v)) for v in values):
        return None
    if not (float(y0) <= float(threshold) < float(y1)):
        return None
    dy = float(y1) - float(y0)
    if abs(dy) < 1e-15:
        return None
    t = (float(threshold) - float(y0)) / dy
    t = float(np.clip(t, 0.0, 1.0))
    return float(x0) + t * (float(x1) - float(x0))


def extract_boundary_from_curves(gamma_inf, ring_peak_db,
                                 alpha_grid_deg, alpha_grid_rad, *,
                                 xi, peak_gate_db, persistent_bad_width_deg,
                                 isolated_bad_policy="warn"):
    gamma = np.asarray(gamma_inf, dtype=np.float64).reshape(-1)
    peak = np.asarray(ring_peak_db, dtype=np.float64).reshape(-1)
    alpha_deg = np.asarray(alpha_grid_deg, dtype=np.float64).reshape(-1)
    alpha_rad = np.asarray(alpha_grid_rad, dtype=np.float64).reshape(-1)
    if not (gamma.size == peak.size == alpha_deg.size == alpha_rad.size):
        raise ValueError("curve/grid lengths must match")
    if gamma.size == 0:
        raise ValueError("curves must be non-empty")
    if np.any(~np.isfinite(gamma)) or np.any(~np.isfinite(peak)):
        raise ValueError("gamma and peak curves must be finite")
    if np.any(~np.isfinite(alpha_deg)) or np.any(np.diff(alpha_deg) <= 0.0):
        raise ValueError("alpha_grid_deg must be finite and strictly increasing")
    if isolated_bad_policy not in ("warn", "error", "ignore"):
        raise ValueError("isolated_bad_policy must be warn, error, or ignore")

    bad_spec = required_bad_sample_count(alpha_deg, persistent_bad_width_deg)
    run_length = int(bad_spec["required_bad_sample_count"])

    error_bad = gamma > float(xi)
    amplitude_active = peak > float(peak_gate_db)
    dual_bad = error_bad & amplitude_active
    true_idx = np.flatnonzero(dual_bad)
    first_dual_bad_idx = int(true_idx[0]) if true_idx.size else -1
    run_start, run_stop = find_first_consecutive_true(dual_bad, run_length)

    isolated_limit = run_start if run_start >= 0 else dual_bad.size
    isolated_idx = np.flatnonzero(dual_bad[:isolated_limit]).astype(np.int64)
    isolated_count = int(isolated_idx.size)
    warnings = []
    if isolated_count:
        msg = (
            "Found {0} isolated dual-gate bad alpha samples before the persistent bad run"
            .format(isolated_count)
        )
        if isolated_bad_policy == "error":
            raise ValueError(msg)
        if isolated_bad_policy == "warn":
            warnings.append(msg)

    first_isolated_deg = float(alpha_deg[isolated_idx[0]]) if isolated_count else float("nan")
    max_isolated_gamma = float(np.max(gamma[isolated_idx])) if isolated_count else float("nan")
    max_isolated_peak = float(np.max(peak[isolated_idx])) if isolated_count else float("nan")

    alpha_gamma_cross = float("nan")
    alpha_peak_cross = float("nan")
    triggered_by = "none"
    last_safe_idx = -1

    if run_start < 0:
        boundary_deg = float(alpha_deg[-1])
        mode = "no_bad_run"
    elif run_start == 0:
        boundary_deg = 0.0
        mode = "boundary_at_zero"
        triggered_by = "both"
    else:
        j = int(run_start)
        last_safe_idx = j - 1
        x0 = float(alpha_deg[j - 1])
        x1 = float(alpha_deg[j])

        if gamma[j - 1] > float(xi):
            alpha_gamma = x0
        else:
            alpha_gamma = _linear_upward_crossing(
                x0, x1, gamma[j - 1], gamma[j], float(xi)
            )

        if peak[j - 1] > float(peak_gate_db):
            alpha_peak = x0
        else:
            alpha_peak = _linear_upward_crossing(
                x0, x1, peak[j - 1], peak[j], float(peak_gate_db)
            )

        if alpha_gamma is not None:
            alpha_gamma_cross = float(alpha_gamma)
        if alpha_peak is not None:
            alpha_peak_cross = float(alpha_peak)

        if alpha_gamma is not None and alpha_peak is not None:
            boundary_deg = max(float(alpha_gamma), float(alpha_peak))
            mode = "dual_gate_linear"
            if abs(float(alpha_gamma) - float(alpha_peak)) <= 1e-12:
                triggered_by = "both"
            elif float(alpha_gamma) > float(alpha_peak):
                triggered_by = "gamma"
            else:
                triggered_by = "amplitude"
        else:
            boundary_deg = x0
            mode = "last_safe_grid_fallback"
            triggered_by = "fallback"

    boundary_rad = math.radians(boundary_deg)
    return BoundaryResult(
        xi=float(xi),
        peak_gate_db=float(peak_gate_db),
        alpha_boundary_deg=float(boundary_deg),
        alpha_boundary_rad=float(boundary_rad),
        rho_boundary=float(math.sin(boundary_rad)),
        error_bad_mask=error_bad.astype(bool),
        amplitude_active_mask=amplitude_active.astype(bool),
        dual_bad_mask=dual_bad.astype(bool),
        first_dual_bad_idx=int(first_dual_bad_idx),
        first_bad_run_start_idx=int(run_start),
        first_bad_run_stop_idx=int(run_stop),
        last_safe_idx=int(last_safe_idx),
        triggered_by=str(triggered_by),
        interpolation_mode=str(mode),
        alpha_gamma_cross_deg=float(alpha_gamma_cross),
        alpha_peak_cross_deg=float(alpha_peak_cross),
        isolated_bad_indices=isolated_idx,
        isolated_bad_count=int(isolated_count),
        first_isolated_bad_deg=float(first_isolated_deg),
        max_isolated_gamma=float(max_isolated_gamma),
        max_isolated_peak_db=float(max_isolated_peak),
        alpha_grid_step_deg=float(bad_spec["alpha_grid_step_deg"]),
        persistent_bad_width_deg=float(bad_spec["persistent_bad_width_deg"]),
        required_bad_sample_count=int(bad_spec["required_bad_sample_count"]),
        actual_bad_span_deg=float(bad_spec["actual_bad_span_deg"]),
        warnings=warnings,
    )


def make_used_boundary(*, alpha_core_calib_deg, safety_margin_deg,
                       alpha_min_allowed_deg, alpha_max_allowed_deg=90.0):
    if not np.isfinite(float(alpha_core_calib_deg)):
        raise ValueError("alpha_core_calib_deg must be finite")
    if float(safety_margin_deg) < 0.0:
        raise ValueError("safety_margin_deg must be >= 0")
    alpha_used = math.floor(float(alpha_core_calib_deg)) - float(safety_margin_deg)
    alpha_used = max(float(alpha_min_allowed_deg), min(float(alpha_max_allowed_deg), alpha_used))
    return float(alpha_used), float(math.sin(math.radians(alpha_used)))


def make_case_result(curves, boundary):
    n_ring = int(np.sum(curves.N_vec))
    return CasePartitionResult(
        K=curves.K,
        d_min=curves.d_min,
        b=curves.b,
        N_total=n_ring,
        N_ring_total=n_ring,
        N_total_with_center=n_ring + 1,
        include_center=True,
        alpha_1d_deg=boundary.alpha_boundary_deg,
        alpha_1d_rad=boundary.alpha_boundary_rad,
        rho_1d=boundary.rho_boundary,
        first_dual_bad_idx=boundary.first_dual_bad_idx,
        first_bad_run_start_idx=boundary.first_bad_run_start_idx,
        first_bad_run_stop_idx=boundary.first_bad_run_stop_idx,
        last_safe_idx=boundary.last_safe_idx,
        triggered_by=boundary.triggered_by,
        interpolation_mode=boundary.interpolation_mode,
        alpha_gamma_cross_deg=boundary.alpha_gamma_cross_deg,
        alpha_peak_cross_deg=boundary.alpha_peak_cross_deg,
        isolated_bad_count=boundary.isolated_bad_count,
        first_isolated_bad_deg=boundary.first_isolated_bad_deg,
        max_isolated_gamma=boundary.max_isolated_gamma,
        max_isolated_peak_db=boundary.max_isolated_peak_db,
        gamma_inf_max=float(np.max(curves.gamma_inf)),
        ring_peak_db_max=float(np.max(curves.ring_peak_db)),
        alpha_grid_step_deg=boundary.alpha_grid_step_deg,
        persistent_bad_width_deg=boundary.persistent_bad_width_deg,
        required_bad_sample_count=boundary.required_bad_sample_count,
        actual_bad_span_deg=boundary.actual_bad_span_deg,
        r_vec=curves.r_vec.copy(),
        N_vec=curves.N_vec.copy(),
        warnings=list(boundary.warnings),
    )


def aggregate_combo_results(case_results, *, R0, d_eff, safety_margin_deg,
                            alpha_min_allowed_deg, alpha_outer_start_deg):
    if not case_results:
        raise ValueError("case_results is empty")
    K = int(case_results[0].K)
    d_min = float(case_results[0].d_min)
    alpha_vals = np.asarray([x.alpha_1d_deg for x in case_results], dtype=np.float64)
    rho_vals = np.asarray([x.rho_1d for x in case_results], dtype=np.float64)
    min_alpha = float(np.nanmin(alpha_vals))
    candidates = [x for x in case_results if abs(float(x.alpha_1d_deg) - min_alpha) <= 1e-12]
    candidates.sort(key=lambda x: (-int(x.isolated_bad_count), -float(x.gamma_inf_max), float(x.b)))
    worst = candidates[0]
    alpha_used, rho_used = make_used_boundary(
        alpha_core_calib_deg=min_alpha,
        safety_margin_deg=safety_margin_deg,
        alpha_min_allowed_deg=alpha_min_allowed_deg,
        alpha_max_allowed_deg=alpha_outer_start_deg,
    )
    return ComboPartitionResult(
        K=K,
        d_min=d_min,
        d_eff=float(d_eff),
        R0=float(R0),
        alpha_core_calib_deg=min_alpha,
        alpha_core_used_deg=alpha_used,
        rho_core_used=rho_used,
        alpha_outer_start_deg=float(alpha_outer_start_deg),
        worst_b=float(worst.b),
        alpha_1d_min_over_b_deg=_safe_min(alpha_vals),
        alpha_1d_mean_over_b_deg=_safe_mean(alpha_vals),
        alpha_1d_max_over_b_deg=_safe_max(alpha_vals),
        rho_1d_min_over_b=_safe_min(rho_vals),
        rho_1d_mean_over_b=_safe_mean(rho_vals),
        rho_1d_max_over_b=_safe_max(rho_vals),
        N_total_at_worst_b=int(worst.N_total),
        N_ring_total_at_worst_b=int(worst.N_ring_total),
        N_total_with_center_at_worst_b=int(worst.N_total_with_center),
        worst_case_interpolation_mode=str(worst.interpolation_mode),
        worst_case_triggered_by=str(worst.triggered_by),
        isolated_bad_case_count=int(sum(1 for x in case_results if x.isolated_bad_count > 0)),
        max_isolated_bad_count=int(max(x.isolated_bad_count for x in case_results)),
        all_cases_have_persistent_boundary=bool(
            all(x.first_bad_run_start_idx >= 0 for x in case_results)
        ),
    )


def evaluate_custom_structure_boundary(r_vec, N_vec, pool_cfg, eval_cfg,
                                       K=None, d_min=float("nan"), b=float("nan")):
    r = np.asarray(r_vec, dtype=np.float64).reshape(-1)
    n = np.asarray(N_vec, dtype=np.int64).reshape(-1)
    case = StructureCase(K=(r.size if K is None else K), d_min=d_min, b=b, r_vec=r, N_vec=n)
    pool = build_intrinsic_ring_pool(pool_cfg)
    curves = evaluate_case_intrinsic(case, pool, eval_cfg)
    boundary = extract_boundary_from_curves(
        curves.gamma_inf,
        curves.ring_peak_db,
        pool.alpha_grid_deg,
        pool.alpha_grid_rad,
        xi=eval_cfg.xi_core,
        peak_gate_db=eval_cfg.peak_gate_db,
        persistent_bad_width_deg=eval_cfg.persistent_bad_width_deg,
        isolated_bad_policy=eval_cfg.isolated_bad_policy,
    )
    return curves, boundary


# =============================================================================
# Validation and output rows
# =============================================================================


def validate_configs(cfg, pool_cfg, eval_cfg):
    if cfg.R0 <= 0.0 or cfg.d_eff <= 0.0:
        raise ValueError("R0 and d_eff must be > 0")
    if cfg.beta <= 0.0 or eval_cfg.beta <= 0.0:
        raise ValueError("beta must be > 0")
    if eval_cfg.xi_core <= 0.0:
        raise ValueError("xi_core must be > 0")
    if eval_cfg.peak_gate_db > 0.0:
        raise ValueError("peak_gate_db must be <= 0")
    if eval_cfg.persistent_bad_width_deg < 0.0:
        raise ValueError("persistent_bad_width_deg must be >= 0")
    if eval_cfg.isolated_bad_policy not in ("warn", "error", "ignore"):
        raise ValueError("isolated_bad_policy must be warn, error, or ignore")
    if cfg.safety_margin_deg < 0.0:
        raise ValueError("safety_margin_deg must be >= 0")
    if not (0.0 <= cfg.alpha_min_allowed_deg <= cfg.alpha_outer_start_deg):
        raise ValueError("invalid alpha bounds")
    if cfg.b_step <= 0.0 or cfg.b_max < cfg.b_min:
        raise ValueError("invalid b range")
    if not cfg.K_list or any(int(x) < 1 for x in cfg.K_list):
        raise ValueError("K_list must contain positive integers")
    if not cfg.d_min_list or any(float(x) <= 0.0 for x in cfg.d_min_list):
        raise ValueError("d_min_list must contain positive values")
    pool = build_intrinsic_ring_pool(pool_cfg)
    required_bad_sample_count(pool.alpha_grid_deg, eval_cfg.persistent_bad_width_deg)
    feasible = any(
        cfg.R0 >= int(K) * float(d) - 1e-12
        for K in cfg.K_list
        for d in cfg.d_min_list
    )
    if not feasible:
        raise ValueError("No feasible (K, d_min) combination satisfies R0 >= K*d_min")


def case_result_to_row(res, K_max):
    row = {
        "K": int(res.K),
        "d_min": float(res.d_min),
        "b": float(res.b),
        "N_total": int(res.N_total),
        "N_total_semantics": "ring_elements_only",
        "N_ring_total": int(res.N_ring_total),
        "N_total_with_center": int(res.N_total_with_center),
        "include_center": bool(res.include_center),
        "alpha_1d_deg": float(res.alpha_1d_deg),
        "alpha_1d_rad": float(res.alpha_1d_rad),
        "rho_1d": float(res.rho_1d),
        "first_dual_bad_idx": int(res.first_dual_bad_idx),
        "first_bad_run_start_idx": int(res.first_bad_run_start_idx),
        "first_bad_run_stop_idx": int(res.first_bad_run_stop_idx),
        "last_safe_idx": int(res.last_safe_idx),
        "triggered_by": str(res.triggered_by),
        "interpolation_mode": str(res.interpolation_mode),
        "alpha_gamma_cross_deg": float(res.alpha_gamma_cross_deg),
        "alpha_peak_cross_deg": float(res.alpha_peak_cross_deg),
        "isolated_bad_count": int(res.isolated_bad_count),
        "first_isolated_bad_deg": float(res.first_isolated_bad_deg),
        "max_isolated_gamma": float(res.max_isolated_gamma),
        "max_isolated_peak_db": float(res.max_isolated_peak_db),
        "gamma_inf_max": float(res.gamma_inf_max),
        "ring_peak_db_max": float(res.ring_peak_db_max),
        "alpha_grid_step_deg": float(res.alpha_grid_step_deg),
        "persistent_bad_width_deg": float(res.persistent_bad_width_deg),
        "required_bad_sample_count": int(res.required_bad_sample_count),
        "actual_bad_span_deg": float(res.actual_bad_span_deg),
    }
    for k in range(int(K_max)):
        row["r_{0}".format(k + 1)] = float(res.r_vec[k]) if k < res.r_vec.size else float("nan")
    for k in range(int(K_max)):
        row["N_{0}".format(k + 1)] = int(res.N_vec[k]) if k < res.N_vec.size else ""
    return row


def combo_result_to_row(res):
    return _record_to_dict(res)


def make_partition_table_payload(cfg, pool_cfg, eval_cfg, combo_results,
                                 outputs, run_info, status, warnings, bad_spec):
    # The intrinsic alpha boundary divides the cheap 1D core from the 2D
    # buffer. The outer zone uses a scan-centered spherical angle instead.
    return {
        "version": FORMAT_VERSION,
        "calibration_revision": CALIBRATION_REVISION,
        "run": run_info,
        "quality_control": {
            "status": str(status),
            "warnings": list(warnings),
        },
        "definition": {
            "core_1d": "0 <= alpha_intrinsic <= alpha_core_used_deg",
            "buffer_2d": "alpha_core_used_deg < alpha_intrinsic <= 90 deg",
            "outer_2d": "90 deg < local spherical alpha <= 90 deg + theta_s",
            "alpha_outer_start_deg": float(cfg.alpha_outer_start_deg),
        },
        "coordinate": {
            "name": "intrinsic_spectral_radius",
            "rho_definition": "rho = sin(alpha_intrinsic)",
            "alpha_definition": "alpha_intrinsic = asin(rho)",
            "scan_angle_included": False,
        },
        "model_scope": {
            "calibrated_effect": "finite-ring azimuthal aliasing relative to the q=0 J0 proxy",
            "not_calibrated": [
                "wide-angle Delta-u/Delta-v geometric distortion",
                "IEP angular dependence",
                "optimized geometry outside the swept Gompertz family",
                "nonzero inter-ring angular offsets",
            ],
            "ring_orientation": "all rings start at psi=0",
            "include_center": True,
            "sampling_strategy": "single fixed grid applied independently to every case",
        },
        "calibration": {
            "method": "intrinsic q0 exact finite-ring direct sum versus J0 proxy",
            "R0": float(cfg.R0),
            "d_eff_mode": "bound_to_d_min" if cfg.bind_d_eff_to_d_min else "fixed",
            "d_eff": "d_min" if cfg.bind_d_eff_to_d_min else float(cfg.d_eff),
            "K_list": [int(x) for x in cfg.K_list],
            "d_min_list": [float(x) for x in cfg.d_min_list],
            "b_grid": _float_range(cfg.b_min, cfg.b_max, cfg.b_step).tolist(),
            "xi_core": float(eval_cfg.xi_core),
            "peak_gate_db": float(eval_cfg.peak_gate_db),
            "isolated_bad_policy": str(eval_cfg.isolated_bad_policy),
            "persistent_bad_rule": dict(bad_spec),
            "legacy_n_consecutive_bad": int(eval_cfg.n_consecutive_bad_legacy),
            "dual_gate_boundary_rule": "max(gamma threshold crossing, amplitude threshold crossing)",
            "safety_rule": "floor(alpha_core_calib_deg) - safety_margin_deg",
            "safety_margin_deg": float(cfg.safety_margin_deg),
            "alpha_min_allowed_deg": float(cfg.alpha_min_allowed_deg),
            "pool": _record_to_dict(pool_cfg),
            "outer5_reduce_mult4": bool(cfg.outer5_reduce_mult4),
            "outer5_threshold_gt": int(cfg.outer5_threshold_gt),
            "N_total_semantics": "ring_elements_only; N_total_with_center includes the central element",
        },
        "entries": [combo_result_to_row(x) for x in combo_results],
        "outputs": dict(outputs),
    }


# =============================================================================
# CLI and main run
# =============================================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Offline fixed-grid calibration of the Q0CRA intrinsic 1D/2D partition (Python 3.6 compatible)"
    )
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--R0", type=float, default=5.0)
    p.add_argument("--d-eff", type=float, default=0.5)
    p.add_argument("--fixed-d-eff", action="store_true")
    p.add_argument("--K-list", type=str, default="6,7,8")
    p.add_argument("--d-min-list", type=str, default="")
    p.add_argument("--d-min-min", type=float, default=0.50)
    p.add_argument("--d-min-max", type=float, default=0.60)
    p.add_argument("--d-min-step", type=float, default=0.01)
    p.add_argument("--b-min", type=float, default=6.0)
    p.add_argument("--b-max", type=float, default=12.0)
    p.add_argument("--b-step", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=2.0 * math.pi)
    p.add_argument("--xi-core", type=float, default=0.01)
    p.add_argument("--peak-gate-db", type=float, default=-40.0)
    p.add_argument(
        "--persistent-bad-width-deg",
        type=float,
        default=None,
        help="Required physical alpha span of a persistent dual-gate bad run. "
             "When omitted, it is derived from --n-consecutive-bad and --alpha-step-deg.",
    )
    p.add_argument(
        "--n-consecutive-bad",
        type=int,
        default=2,
        help="Legacy compatibility option used only to derive persistent width when "
             "--persistent-bad-width-deg is omitted.",
    )
    p.add_argument("--isolated-bad-policy", choices=["warn", "error", "ignore"], default="warn")
    p.add_argument("--safety-margin-deg", type=float, default=1.0)
    p.add_argument("--alpha-min-allowed-deg", type=float, default=0.0)
    p.add_argument("--alpha-outer-start-deg", type=float, default=90.0)
    p.add_argument("--outer5-threshold-gt", type=int, default=16)
    p.add_argument("--disable-outer5-reduce-mult4", action="store_true")
    p.add_argument("--alpha-step-deg", type=float, default=0.5)
    p.add_argument("--alpha-max-deg", type=float, default=90.0)
    p.add_argument("--eta-arc-step-deg", type=float, default=4.0)
    p.add_argument("--n-min-per-ring", type=int, default=12)
    return p.parse_args(argv)


def run_calibration(args):
    t_start = time.perf_counter()
    start_utc = _utc_now_iso()
    out_dir = os.path.abspath(args.out_dir)
    _ensure_dir(out_dir)

    paths = {
        "partition_table_json": os.path.join(out_dir, "q0cra_stage1_partition_table.json"),
        "summary_json": os.path.join(out_dir, "q0cra_stage1_partition_summary.json"),
        "per_case_csv": os.path.join(out_dir, "q0cra_stage1_partition_per_case.csv"),
        "per_combo_csv": os.path.join(out_dir, "q0cra_stage1_partition_per_combo.csv"),
        "curves_npz": os.path.join(out_dir, "q0cra_stage1_partition_curves.npz"),
        "log": os.path.join(out_dir, "q0cra_stage1_partition_calibrate.log"),
    }
    _check_output_paths(list(paths.values()), bool(args.overwrite))
    logger = _setup_logger(paths["log"])

    try:
        K_list = sorted(_parse_int_list(args.K_list))
        if str(args.d_min_list).strip():
            d_min_list = sorted(_parse_float_list(args.d_min_list))
        else:
            d_min_list = _float_range(args.d_min_min, args.d_min_max, args.d_min_step).tolist()
        b_grid = _float_range(args.b_min, args.b_max, args.b_step)

        persistent_width = resolve_persistent_bad_width(
            args.alpha_step_deg,
            args.n_consecutive_bad,
            args.persistent_bad_width_deg,
        )
        pool_cfg = IntrinsicPoolConfig(
            args.alpha_step_deg,
            args.alpha_max_deg,
            args.eta_arc_step_deg,
            args.n_min_per_ring,
        )
        cfg = MainConfig(
            out_dir=out_dir,
            R0=float(args.R0),
            d_eff=float(args.d_eff),
            bind_d_eff_to_d_min=not bool(args.fixed_d_eff),
            K_list=[int(x) for x in K_list],
            d_min_list=[float(x) for x in d_min_list],
            b_min=float(args.b_min),
            b_max=float(args.b_max),
            b_step=float(args.b_step),
            beta=float(args.beta),
            safety_margin_deg=float(args.safety_margin_deg),
            alpha_min_allowed_deg=float(args.alpha_min_allowed_deg),
            alpha_outer_start_deg=float(args.alpha_outer_start_deg),
            outer5_reduce_mult4=not bool(args.disable_outer5_reduce_mult4),
            outer5_threshold_gt=int(args.outer5_threshold_gt),
            overwrite=bool(args.overwrite),
        )
        eval_cfg = EvalConfig(
            beta=args.beta,
            xi_core=args.xi_core,
            peak_gate_db=args.peak_gate_db,
            persistent_bad_width_deg=persistent_width,
            isolated_bad_policy=args.isolated_bad_policy,
            n_consecutive_bad_legacy=args.n_consecutive_bad,
        )
        validate_configs(cfg, pool_cfg, eval_cfg)
        pool = build_intrinsic_ring_pool(pool_cfg)
        bad_spec = required_bad_sample_count(
            pool.alpha_grid_deg,
            eval_cfg.persistent_bad_width_deg,
        )

        logger.info("[CONFIG] out_dir=%s", out_dir)
        logger.info("[CONFIG] sampling_strategy=single_fixed_grid_full_sweep")
        logger.info("[CONFIG] K_list=%s d_min_list=%s", cfg.K_list, cfg.d_min_list)
        logger.info(
            "[CONFIG] b=[%g,%g] step=%g count=%d",
            cfg.b_min, cfg.b_max, cfg.b_step, b_grid.size,
        )
        logger.info(
            "[POOL] alpha_step=%g eta_step=%g n_alpha=%d total_points=%d",
            pool_cfg.alpha_step_deg, pool_cfg.eta_arc_step_deg,
            pool.alpha_grid_deg.size, pool.total_points,
        )
        logger.info(
            "[BOUNDARY] xi=%g peak_gate_db=%g persistent_width=%g "
            "required_bad_samples=%d actual_span=%g isolated_policy=%s",
            eval_cfg.xi_core, eval_cfg.peak_gate_db,
            eval_cfg.persistent_bad_width_deg,
            bad_spec["required_bad_sample_count"],
            bad_spec["actual_bad_span_deg"],
            eval_cfg.isolated_bad_policy,
        )

        K_max = max(cfg.K_list)
        nK = len(cfg.K_list)
        nD = len(cfg.d_min_list)
        nB = int(b_grid.size)
        nA = int(pool.alpha_grid_deg.size)
        shape_case = (nK, nD, nB)
        shape_curve = shape_case + (nA,)

        alpha_deg_arr = np.full(shape_case, np.nan, dtype=np.float64)
        alpha_rad_arr = np.full(shape_case, np.nan, dtype=np.float64)
        rho_arr = np.full(shape_case, np.nan, dtype=np.float64)
        gamma_arr = np.full(shape_curve, np.nan, dtype=np.float64)
        gamma_prefix_arr = np.full(shape_curve, np.nan, dtype=np.float64)
        peak_arr = np.full(shape_curve, np.nan, dtype=np.float64)
        proxy_abs_arr = np.full(shape_curve, np.nan, dtype=np.float64)
        exact_peak_abs_arr = np.full(shape_curve, np.nan, dtype=np.float64)
        proxy_db_arr = np.full(shape_curve, np.nan, dtype=np.float64)
        exact_peak_db_arr = np.full(shape_curve, np.nan, dtype=np.float64)
        abs_err_arr = np.full(shape_curve, np.nan, dtype=np.float64)
        abs_err_norm_arr = np.full(shape_curve, np.nan, dtype=np.float64)
        error_bad_arr = np.zeros(shape_curve, dtype=bool)
        amp_active_arr = np.zeros(shape_curve, dtype=bool)
        dual_bad_arr = np.zeros(shape_curve, dtype=bool)
        first_dual_arr = np.full(shape_case, -1, dtype=np.int32)
        run_start_arr = np.full(shape_case, -1, dtype=np.int32)
        run_stop_arr = np.full(shape_case, -1, dtype=np.int32)
        isolated_count_arr = np.zeros(shape_case, dtype=np.int32)
        gamma_cross_arr = np.full(shape_case, np.nan, dtype=np.float64)
        peak_cross_arr = np.full(shape_case, np.nan, dtype=np.float64)
        mode_code_arr = np.full(shape_case, -1, dtype=np.int8)
        r_mat = np.full(shape_case + (K_max,), np.nan, dtype=np.float64)
        N_mat = np.full(shape_case + (K_max,), -1, dtype=np.int64)
        N_ring_arr = np.zeros(shape_case, dtype=np.int32)
        N_center_arr = np.zeros(shape_case, dtype=np.int32)

        all_cases = []
        all_combos = []
        warnings = []
        isolated_case_count_total = 0
        feasible_total_cases = sum(
            int(b_grid.size)
            for K in cfg.K_list
            for d_min in cfg.d_min_list
            if cfg.R0 >= int(K) * float(d_min) - 1e-12
        )
        counter = 0

        for ki, K in enumerate(cfg.K_list):
            for di, d_min in enumerate(cfg.d_min_list):
                if cfg.R0 < K * d_min - 1e-12:
                    logger.warning("[CASE] skip infeasible K=%d d_min=%g", K, d_min)
                    continue
                combo_cases = []
                for bi, b in enumerate(b_grid.tolist()):
                    counter += 1
                    d_eff_case = d_min if cfg.bind_d_eff_to_d_min else cfg.d_eff
                    r_vec = init_r_gompertz(K=K, R0=cfg.R0, d_min=d_min, b=b)
                    N_vec = derive_ring_counts(
                        r_vec,
                        d_eff=d_eff_case,
                        outer5_reduce_mult4=cfg.outer5_reduce_mult4,
                        outer5_threshold_gt=cfg.outer5_threshold_gt,
                    )
                    case = StructureCase(K, d_min, b, r_vec, N_vec)
                    curves = evaluate_case_intrinsic(case, pool, eval_cfg)
                    boundary = extract_boundary_from_curves(
                        curves.gamma_inf,
                        curves.ring_peak_db,
                        pool.alpha_grid_deg,
                        pool.alpha_grid_rad,
                        xi=eval_cfg.xi_core,
                        peak_gate_db=eval_cfg.peak_gate_db,
                        persistent_bad_width_deg=eval_cfg.persistent_bad_width_deg,
                        isolated_bad_policy=eval_cfg.isolated_bad_policy,
                    )
                    result = make_case_result(curves, boundary)
                    combo_cases.append(result)
                    all_cases.append(result)
                    if boundary.isolated_bad_count > 0:
                        isolated_case_count_total += 1

                    idx = (ki, di, bi)
                    alpha_deg_arr[idx] = result.alpha_1d_deg
                    alpha_rad_arr[idx] = result.alpha_1d_rad
                    rho_arr[idx] = result.rho_1d
                    gamma_arr[idx] = curves.gamma_inf
                    gamma_prefix_arr[idx] = curves.gamma_prefix
                    peak_arr[idx] = curves.ring_peak_db
                    proxy_abs_arr[idx] = curves.proxy_abs
                    exact_peak_abs_arr[idx] = curves.exact_peak_abs
                    proxy_db_arr[idx] = curves.proxy_db
                    exact_peak_db_arr[idx] = curves.exact_peak_db
                    abs_err_arr[idx] = curves.absolute_error_peak
                    abs_err_norm_arr[idx] = curves.absolute_error_normalized
                    error_bad_arr[idx] = boundary.error_bad_mask
                    amp_active_arr[idx] = boundary.amplitude_active_mask
                    dual_bad_arr[idx] = boundary.dual_bad_mask
                    first_dual_arr[idx] = boundary.first_dual_bad_idx
                    run_start_arr[idx] = boundary.first_bad_run_start_idx
                    run_stop_arr[idx] = boundary.first_bad_run_stop_idx
                    isolated_count_arr[idx] = boundary.isolated_bad_count
                    gamma_cross_arr[idx] = boundary.alpha_gamma_cross_deg
                    peak_cross_arr[idx] = boundary.alpha_peak_cross_deg
                    mode_code_arr[idx] = BOUNDARY_MODE_CODE[boundary.interpolation_mode]
                    r_mat[idx + (slice(0, K),)] = r_vec
                    N_mat[idx + (slice(0, K),)] = N_vec
                    N_ring_arr[idx] = result.N_ring_total
                    N_center_arr[idx] = result.N_total_with_center

                    logger.info(
                        "[CASE %04d/%04d] K=%d d_min=%.2f b=%.2f "
                        "N_ring=%d alpha=%.6f mode=%s isolated=%d",
                        counter, feasible_total_cases, K, d_min, b,
                        result.N_ring_total, result.alpha_1d_deg,
                        result.interpolation_mode, result.isolated_bad_count,
                    )

                if combo_cases:
                    combo = aggregate_combo_results(
                        combo_cases,
                        R0=cfg.R0,
                        d_eff=(d_min if cfg.bind_d_eff_to_d_min else cfg.d_eff),
                        safety_margin_deg=cfg.safety_margin_deg,
                        alpha_min_allowed_deg=cfg.alpha_min_allowed_deg,
                        alpha_outer_start_deg=cfg.alpha_outer_start_deg,
                    )
                    all_combos.append(combo)
                    logger.info(
                        "[COMBO] K=%d d_min=%.2f alpha_calib=%.6f "
                        "alpha_used=%.6f worst_b=%.2f",
                        K, d_min, combo.alpha_core_calib_deg,
                        combo.alpha_core_used_deg, combo.worst_b,
                    )

        if not all_combos:
            raise RuntimeError("No feasible combo produced a partition result")

        if isolated_case_count_total > 0 and eval_cfg.isolated_bad_policy == "warn":
            warnings.append(
                "Isolated dual-gate bad samples were detected in {0} case(s); "
                "inspect per-case CSV and NPZ diagnostics"
                .format(isolated_case_count_total)
            )

        case_rows = [case_result_to_row(x, K_max) for x in all_cases]
        combo_rows = [combo_result_to_row(x) for x in all_combos]
        _atomic_write_csv(paths["per_case_csv"], case_rows)
        _atomic_write_csv(paths["per_combo_csv"], combo_rows)

        npz_payload = {
            "format_version": np.asarray(FORMAT_VERSION),
            "calibration_revision": np.asarray(CALIBRATION_REVISION),
            "K_list": np.asarray(cfg.K_list, dtype=np.int64),
            "d_min_list": np.asarray(cfg.d_min_list, dtype=np.float64),
            "d_eff_list": np.asarray(
                [d if cfg.bind_d_eff_to_d_min else cfg.d_eff for d in cfg.d_min_list],
                dtype=np.float64,
            ),
            "b_grid": b_grid.astype(np.float64),
            "alpha_grid_deg": pool.alpha_grid_deg,
            "alpha_grid_rad": pool.alpha_grid_rad,
            "rho_grid": pool.rho_grid,
            "persistent_bad_width_deg": np.asarray(eval_cfg.persistent_bad_width_deg),
            "required_bad_sample_count": np.asarray(
                bad_spec["required_bad_sample_count"], dtype=np.int64
            ),
            "actual_bad_span_deg": np.asarray(bad_spec["actual_bad_span_deg"]),
            "alpha_1d_deg": alpha_deg_arr,
            "alpha_1d_rad": alpha_rad_arr,
            "rho_1d": rho_arr,
            "gamma_inf": gamma_arr,
            "gamma_prefix": gamma_prefix_arr,
            "ring_peak_db": peak_arr,
            "proxy_abs": proxy_abs_arr,
            "exact_peak_abs": exact_peak_abs_arr,
            "proxy_db": proxy_db_arr,
            "exact_peak_db": exact_peak_db_arr,
            "absolute_error_peak": abs_err_arr,
            "absolute_error_normalized": abs_err_norm_arr,
            "error_bad_mask": error_bad_arr,
            "amplitude_active_mask": amp_active_arr,
            "dual_bad_mask": dual_bad_arr,
            "first_dual_bad_idx": first_dual_arr,
            "first_bad_run_start_idx": run_start_arr,
            "first_bad_run_stop_idx": run_stop_arr,
            "isolated_bad_count": isolated_count_arr,
            "alpha_gamma_cross_deg": gamma_cross_arr,
            "alpha_peak_cross_deg": peak_cross_arr,
            "boundary_mode_code": mode_code_arr,
            "r_mat_padded": r_mat,
            "N_mat_padded": N_mat,
            "N_total": N_ring_arr,
            "N_ring_total": N_ring_arr,
            "N_total_with_center": N_center_arr,
        }
        _atomic_savez(paths["curves_npz"], npz_payload)

        elapsed = time.perf_counter() - t_start
        status = "WARNING" if warnings else "PASS"
        run_info = {
            "status": status,
            "start_time_utc": start_utc,
            "end_time_utc": _utc_now_iso(),
            "elapsed_seconds": float(elapsed),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "platform": platform.platform(),
            "command_line": list(sys.argv),
            "script_path": os.path.abspath(__file__),
            "script_sha256": _sha256_file(os.path.abspath(__file__)),
        }
        table_payload = make_partition_table_payload(
            cfg, pool_cfg, eval_cfg, all_combos, paths, run_info,
            status, warnings, bad_spec,
        )
        _atomic_write_json(paths["partition_table_json"], table_payload)

        alpha_used = np.asarray([x.alpha_core_used_deg for x in all_combos], dtype=np.float64)
        alpha_calib = np.asarray([x.alpha_core_calib_deg for x in all_combos], dtype=np.float64)
        summary_payload = {
            "version": FORMAT_VERSION,
            "calibration_revision": CALIBRATION_REVISION,
            "run": run_info,
            "quality_control": {
                "status": status,
                "warnings": list(warnings),
            },
            "config": _record_to_dict(cfg),
            "pool": _record_to_dict(pool_cfg),
            "eval": _record_to_dict(eval_cfg),
            "persistent_bad_rule": dict(bad_spec),
            "coordinate": table_payload["coordinate"],
            "model_scope": table_payload["model_scope"],
            "n_cases": len(all_cases),
            "n_combos": len(all_combos),
            "isolated_bad_case_count": int(isolated_case_count_total),
            "alpha_core_calib_deg_min": _safe_min(alpha_calib),
            "alpha_core_calib_deg_mean": _safe_mean(alpha_calib),
            "alpha_core_calib_deg_max": _safe_max(alpha_calib),
            "alpha_core_used_deg_min": _safe_min(alpha_used),
            "alpha_core_used_deg_mean": _safe_mean(alpha_used),
            "alpha_core_used_deg_max": _safe_max(alpha_used),
            "most_conservative_combo": combo_result_to_row(
                all_combos[int(np.argmin(alpha_used))]
            ),
            "least_conservative_combo": combo_result_to_row(
                all_combos[int(np.argmax(alpha_used))]
            ),
            "outputs": paths,
        }
        _atomic_write_json(paths["summary_json"], summary_payload)

        logger.info("[OUTPUT] partition_table=%s", paths["partition_table_json"])
        logger.info("[OUTPUT] summary=%s", paths["summary_json"])
        logger.info("[OUTPUT] per_case=%s", paths["per_case_csv"])
        logger.info("[OUTPUT] per_combo=%s", paths["per_combo_csv"])
        logger.info("[OUTPUT] curves=%s", paths["curves_npz"])
        logger.info(
            "[SUMMARY] status=%s n_cases=%d n_combos=%d isolated_cases=%d elapsed=%.3f s",
            status, len(all_cases), len(all_combos),
            isolated_case_count_total, elapsed,
        )
        return summary_payload
    finally:
        _close_logger(logger)


def main():
    args = parse_args()
    run_calibration(args)


if __name__ == "__main__":
    main()
