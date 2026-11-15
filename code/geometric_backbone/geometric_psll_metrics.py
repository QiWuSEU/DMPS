#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate geometric PSLL and rank sidelobe candidates on mixed 1D/2D pools.

Standalone metric module for the geometric backbone synthesis stage.

Design intent
-------------
This module ONLY performs stage-1 PSLL judging and raw sidelobe ranking.
It does NOT:
- build point pools
- evaluate a0 / AF / co-channel fields
- perform active-set thresholding / clustering / capping
- solve convex optimization subproblems

Upstream should provide, for each scan angle, already-fused judging values:
- core_value   = a0_max(alpha; theta_s) * |AF_core|
- buffer_value = |a0(theta)| * |AF_buffer|
- outer_value  = |a0(theta)| * |AF_outer|

The denominator is taken ONLY from the core main-guard region.
The sidelobe numerator is the maximum across:
- core 1D sidelobe region
- buffer 2D sidelobe region
- outer 2D sidelobe region
"""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Record classes (Python 3.6 compatible)
# -----------------------------------------------------------------------------


class Stage1ScanMetricInput(object):
    __slots__ = (
        "theta_s_deg",
        "core_alpha_deg",
        "core_value",
        "core_is_main_guard",
        "core_is_side_1d",
        "buffer_alpha_deg",
        "buffer_eta_deg",
        "buffer_theta_deg",
        "buffer_phi_deg",
        "buffer_value",
        "buffer_is_side_2d",
        "outer_alpha_deg",
        "outer_eta_deg",
        "outer_theta_deg",
        "outer_phi_deg",
        "outer_value",
        "outer_is_side_2d",
        "meta",
    )

    def __init__(
        self,
        theta_s_deg,
        core_alpha_deg,
        core_value,
        core_is_main_guard,
        core_is_side_1d,
        buffer_alpha_deg,
        buffer_eta_deg,
        buffer_theta_deg,
        buffer_phi_deg,
        buffer_value,
        buffer_is_side_2d,
        outer_alpha_deg,
        outer_eta_deg,
        outer_theta_deg,
        outer_phi_deg,
        outer_value,
        outer_is_side_2d,
        meta=None,
    ):
        self.theta_s_deg = theta_s_deg
        self.core_alpha_deg = core_alpha_deg
        self.core_value = core_value
        self.core_is_main_guard = core_is_main_guard
        self.core_is_side_1d = core_is_side_1d
        self.buffer_alpha_deg = buffer_alpha_deg
        self.buffer_eta_deg = buffer_eta_deg
        self.buffer_theta_deg = buffer_theta_deg
        self.buffer_phi_deg = buffer_phi_deg
        self.buffer_value = buffer_value
        self.buffer_is_side_2d = buffer_is_side_2d
        self.outer_alpha_deg = outer_alpha_deg
        self.outer_eta_deg = outer_eta_deg
        self.outer_theta_deg = outer_theta_deg
        self.outer_phi_deg = outer_phi_deg
        self.outer_value = outer_value
        self.outer_is_side_2d = outer_is_side_2d
        self.meta = {} if meta is None else dict(meta)


class Stage1ScanPSLLResult(object):
    __slots__ = (
        "theta_s_deg",
        "D_core",
        "U_core_side",
        "U_buffer",
        "U_outer",
        "U_side",
        "psll_db",
        "zone_wc",
        "idx_wc_local",
        "argmax_core_main",
        "argmax_core_side",
        "argmax_buffer",
        "argmax_outer",
        "counts_core_main",
        "counts_core_side",
        "counts_buffer",
        "counts_outer",
        "meta",
    )

    def __init__(
        self,
        theta_s_deg,
        D_core,
        U_core_side,
        U_buffer,
        U_outer,
        U_side,
        psll_db,
        zone_wc,
        idx_wc_local,
        argmax_core_main,
        argmax_core_side,
        argmax_buffer,
        argmax_outer,
        counts_core_main,
        counts_core_side,
        counts_buffer,
        counts_outer,
        meta=None,
    ):
        self.theta_s_deg = theta_s_deg
        self.D_core = D_core
        self.U_core_side = U_core_side
        self.U_buffer = U_buffer
        self.U_outer = U_outer
        self.U_side = U_side
        self.psll_db = psll_db
        self.zone_wc = zone_wc
        self.idx_wc_local = idx_wc_local
        self.argmax_core_main = argmax_core_main
        self.argmax_core_side = argmax_core_side
        self.argmax_buffer = argmax_buffer
        self.argmax_outer = argmax_outer
        self.counts_core_main = counts_core_main
        self.counts_core_side = counts_core_side
        self.counts_buffer = counts_buffer
        self.counts_outer = counts_outer
        self.meta = {} if meta is None else dict(meta)


class Stage1BatchPSLLResult(object):
    __slots__ = (
        "psll_s_db",
        "psll_wc_db",
        "idx_wc_s",
        "D_s",
        "U_s",
        "zone_wc_s",
        "per_scan_results",
        "meta",
    )

    def __init__(
        self,
        psll_s_db,
        psll_wc_db,
        idx_wc_s,
        D_s,
        U_s,
        zone_wc_s,
        per_scan_results,
        meta=None,
    ):
        self.psll_s_db = psll_s_db
        self.psll_wc_db = psll_wc_db
        self.idx_wc_s = idx_wc_s
        self.D_s = D_s
        self.U_s = U_s
        self.zone_wc_s = zone_wc_s
        self.per_scan_results = per_scan_results
        self.meta = {} if meta is None else dict(meta)


class Stage1SideRankingResult(object):
    __slots__ = (
        "core_side_idx_sorted",
        "core_side_val_sorted",
        "core_side_ratio_db_sorted",
        "buffer_idx_sorted",
        "buffer_val_sorted",
        "buffer_ratio_db_sorted",
        "outer_idx_sorted",
        "outer_val_sorted",
        "outer_ratio_db_sorted",
        "global_zone_sorted",
        "global_idx_sorted",
        "global_val_sorted",
        "global_ratio_db_sorted",
        "meta",
    )

    def __init__(
        self,
        core_side_idx_sorted,
        core_side_val_sorted,
        core_side_ratio_db_sorted,
        buffer_idx_sorted,
        buffer_val_sorted,
        buffer_ratio_db_sorted,
        outer_idx_sorted,
        outer_val_sorted,
        outer_ratio_db_sorted,
        global_zone_sorted,
        global_idx_sorted,
        global_val_sorted,
        global_ratio_db_sorted,
        meta=None,
    ):
        self.core_side_idx_sorted = core_side_idx_sorted
        self.core_side_val_sorted = core_side_val_sorted
        self.core_side_ratio_db_sorted = core_side_ratio_db_sorted
        self.buffer_idx_sorted = buffer_idx_sorted
        self.buffer_val_sorted = buffer_val_sorted
        self.buffer_ratio_db_sorted = buffer_ratio_db_sorted
        self.outer_idx_sorted = outer_idx_sorted
        self.outer_val_sorted = outer_val_sorted
        self.outer_ratio_db_sorted = outer_ratio_db_sorted
        self.global_zone_sorted = global_zone_sorted
        self.global_idx_sorted = global_idx_sorted
        self.global_val_sorted = global_val_sorted
        self.global_ratio_db_sorted = global_ratio_db_sorted
        self.meta = {} if meta is None else dict(meta)


# -----------------------------------------------------------------------------
# Validation / helpers
# -----------------------------------------------------------------------------


def _as_1d_f64(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf.")
    return arr



def _as_1d_bool(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x).reshape(-1)
    if arr.dtype == np.bool_:
        return arr.astype(bool, copy=False)
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("{0} must contain boolean or 0/1 values.".format(name))
    arr_f64 = np.asarray(arr, dtype=np.float64)
    if np.any(~np.isfinite(arr_f64)):
        raise ValueError("{0} contains NaN/Inf.".format(name))
    valid = (arr_f64 == 0.0) | (arr_f64 == 1.0)
    if not np.all(valid):
        raise ValueError("{0} must contain only 0, 1, False or True.".format(name))
    return arr_f64.astype(bool)


def _validate_eps(eps: float) -> float:
    value = float(eps)
    if not np.isfinite(value):
        raise ValueError("eps must be finite.")
    if value <= 0.0:
        raise ValueError("eps must be > 0.")
    return value



def _empty_f64() -> np.ndarray:
    return np.asarray([], dtype=np.float64)



def _empty_bool() -> np.ndarray:
    return np.asarray([], dtype=bool)



def _validate_scan(scan: Stage1ScanMetricInput) -> Stage1ScanMetricInput:
    theta_s_deg = float(scan.theta_s_deg)
    if not np.isfinite(theta_s_deg):
        raise ValueError("theta_s_deg must be finite.")

    core_alpha_deg = _as_1d_f64(scan.core_alpha_deg, "core_alpha_deg")
    core_value = _as_1d_f64(scan.core_value, "core_value")
    core_is_main_guard = _as_1d_bool(scan.core_is_main_guard, "core_is_main_guard")
    core_is_side_1d = _as_1d_bool(scan.core_is_side_1d, "core_is_side_1d")

    n_core = core_value.size
    if core_alpha_deg.size != n_core or core_is_main_guard.size != n_core or core_is_side_1d.size != n_core:
        raise ValueError("core arrays must all have the same length.")
    if np.any(core_value < 0):
        raise ValueError("core_value must be non-negative.")
    if np.any(core_is_main_guard & core_is_side_1d):
        raise ValueError("core_is_main_guard and core_is_side_1d overlap.")
    if np.any(~(core_is_main_guard | core_is_side_1d)):
        raise ValueError("every core point must belong to either main_guard or side_1d.")

    buffer_alpha_deg = _as_1d_f64(scan.buffer_alpha_deg, "buffer_alpha_deg") if len(np.asarray(scan.buffer_alpha_deg).reshape(-1)) else _empty_f64()
    buffer_eta_deg = _as_1d_f64(scan.buffer_eta_deg, "buffer_eta_deg") if len(np.asarray(scan.buffer_eta_deg).reshape(-1)) else _empty_f64()
    buffer_theta_deg = _as_1d_f64(scan.buffer_theta_deg, "buffer_theta_deg") if len(np.asarray(scan.buffer_theta_deg).reshape(-1)) else _empty_f64()
    buffer_phi_deg = _as_1d_f64(scan.buffer_phi_deg, "buffer_phi_deg") if len(np.asarray(scan.buffer_phi_deg).reshape(-1)) else _empty_f64()
    buffer_value = _as_1d_f64(scan.buffer_value, "buffer_value") if len(np.asarray(scan.buffer_value).reshape(-1)) else _empty_f64()
    buffer_is_side_2d = _as_1d_bool(scan.buffer_is_side_2d, "buffer_is_side_2d") if len(np.asarray(scan.buffer_is_side_2d).reshape(-1)) else _empty_bool()

    n_buf = buffer_value.size
    for arr, name in [
        (buffer_alpha_deg, "buffer_alpha_deg"),
        (buffer_eta_deg, "buffer_eta_deg"),
        (buffer_theta_deg, "buffer_theta_deg"),
        (buffer_phi_deg, "buffer_phi_deg"),
        (buffer_is_side_2d, "buffer_is_side_2d"),
    ]:
        if arr.size != n_buf:
            raise ValueError(f"{name} must have the same length as buffer_value.")
    if np.any(buffer_value < 0):
        raise ValueError("buffer_value must be non-negative.")

    outer_alpha_deg = _as_1d_f64(scan.outer_alpha_deg, "outer_alpha_deg") if len(np.asarray(scan.outer_alpha_deg).reshape(-1)) else _empty_f64()
    outer_eta_deg = _as_1d_f64(scan.outer_eta_deg, "outer_eta_deg") if len(np.asarray(scan.outer_eta_deg).reshape(-1)) else _empty_f64()
    outer_theta_deg = _as_1d_f64(scan.outer_theta_deg, "outer_theta_deg") if len(np.asarray(scan.outer_theta_deg).reshape(-1)) else _empty_f64()
    outer_phi_deg = _as_1d_f64(scan.outer_phi_deg, "outer_phi_deg") if len(np.asarray(scan.outer_phi_deg).reshape(-1)) else _empty_f64()
    outer_value = _as_1d_f64(scan.outer_value, "outer_value") if len(np.asarray(scan.outer_value).reshape(-1)) else _empty_f64()
    outer_is_side_2d = _as_1d_bool(scan.outer_is_side_2d, "outer_is_side_2d") if len(np.asarray(scan.outer_is_side_2d).reshape(-1)) else _empty_bool()

    n_out = outer_value.size
    for arr, name in [
        (outer_alpha_deg, "outer_alpha_deg"),
        (outer_eta_deg, "outer_eta_deg"),
        (outer_theta_deg, "outer_theta_deg"),
        (outer_phi_deg, "outer_phi_deg"),
        (outer_is_side_2d, "outer_is_side_2d"),
    ]:
        if arr.size != n_out:
            raise ValueError(f"{name} must have the same length as outer_value.")
    if np.any(outer_value < 0):
        raise ValueError("outer_value must be non-negative.")

    return Stage1ScanMetricInput(
        theta_s_deg=theta_s_deg,
        core_alpha_deg=core_alpha_deg,
        core_value=core_value,
        core_is_main_guard=core_is_main_guard,
        core_is_side_1d=core_is_side_1d,
        buffer_alpha_deg=buffer_alpha_deg,
        buffer_eta_deg=buffer_eta_deg,
        buffer_theta_deg=buffer_theta_deg,
        buffer_phi_deg=buffer_phi_deg,
        buffer_value=buffer_value,
        buffer_is_side_2d=buffer_is_side_2d,
        outer_alpha_deg=outer_alpha_deg,
        outer_eta_deg=outer_eta_deg,
        outer_theta_deg=outer_theta_deg,
        outer_phi_deg=outer_phi_deg,
        outer_value=outer_value,
        outer_is_side_2d=outer_is_side_2d,
        meta=dict(scan.meta),
    )



def _masked_max(arr: np.ndarray, mask: np.ndarray) -> Tuple[float, int, int]:
    idx = np.flatnonzero(mask)
    n = int(idx.size)
    if n == 0:
        return float("nan"), -1, 0
    sub = arr[idx]
    j = int(np.argmax(sub))
    return float(sub[j]), int(idx[j]), n



def _ratio_db(num: np.ndarray, den: float, eps: float) -> np.ndarray:
    den_safe = max(float(den), float(eps))
    out = np.full_like(num, -np.inf, dtype=np.float64)
    pos = num > 0.0
    out[pos] = 20.0 * np.log10(num[pos] / den_safe)
    return out



def _top_desc_indices(values: np.ndarray, topk: Optional[int] = None) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return np.asarray([], dtype=np.int64)
    if np.any(np.isnan(vals)):
        raise ValueError("ranking values contain NaN.")
    local_idx = np.arange(vals.size, dtype=np.int64)
    order = np.lexsort((local_idx, -vals)).astype(np.int64)
    if topk is None:
        return order
    k = int(topk)
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    return order[:min(k, order.size)]


def _sort_desc(values: np.ndarray) -> np.ndarray:
    return _top_desc_indices(values, topk=None)


# -----------------------------------------------------------------------------
# Optional helper: build one scan input from pool-like point dicts + fused values
# -----------------------------------------------------------------------------


def _required_mask(points, key, size, region_name):
    if int(size) == 0:
        return np.asarray([], dtype=bool)
    if key not in points:
        raise KeyError(
            "{0} points missing required mask: {1}".format(region_name, key)
        )
    mask = _as_1d_bool(points[key], key)
    if mask.size != int(size):
        raise ValueError(
            "{0} mask length does not match value length.".format(region_name)
        )
    return mask


def _require_same_size(arrays, expected_size, region_name):
    for name, arr in arrays:
        if arr.size != int(expected_size):
            raise ValueError(
                "{0}.{1} length does not match value length.".format(
                    region_name, name
                )
            )


def _validate_optional_theta_s(points, theta_s_deg, region_name, tol=1e-9):
    if "theta_s_deg" not in points:
        return
    arr = _as_1d_f64(
        points["theta_s_deg"],
        region_name + "_theta_s_deg",
    )
    if arr.size == 0:
        return
    if np.any(np.abs(arr - float(theta_s_deg)) > float(tol)):
        raise ValueError(
            "{0} points contain a different scan angle.".format(region_name)
        )


def build_stage1_scan_metric_input(
    *,
    theta_s_deg: float,
    core_points: Mapping[str, np.ndarray],
    core_value: Sequence[float],
    buffer_points: Mapping[str, np.ndarray],
    buffer_value: Sequence[float],
    outer_points: Mapping[str, np.ndarray],
    outer_value: Sequence[float],
    meta: Optional[Mapping[str, Any]] = None,
    core_main_key: str = "is_main_guard",
    core_side_key: str = "is_side_1d",
    buffer_side_key: str = "is_side_2d",
    outer_side_key: str = "is_side_2d",
) -> Stage1ScanMetricInput:
    meta_dict = dict(meta) if meta is not None else {}
    theta_value = float(theta_s_deg)
    if not np.isfinite(theta_value):
        raise ValueError("theta_s_deg must be finite.")

    _validate_optional_theta_s(core_points, theta_value, "core")
    _validate_optional_theta_s(buffer_points, theta_value, "buffer")
    _validate_optional_theta_s(outer_points, theta_value, "outer")

    core_alpha = _as_1d_f64(core_points.get("alpha_deg", []), "core.alpha_deg")
    core_val = _as_1d_f64(core_value, "core_value")
    if core_alpha.size != core_val.size:
        raise ValueError("core_points['alpha_deg'] and core_value must have the same length.")
    core_main = _required_mask(core_points, core_main_key, core_val.size, "core")
    core_side = _required_mask(core_points, core_side_key, core_val.size, "core")

    buffer_alpha = _as_1d_f64(buffer_points.get("alpha_deg", []), "buffer.alpha_deg")
    buffer_eta = _as_1d_f64(buffer_points.get("eta_deg", []), "buffer.eta_deg")
    buffer_theta = _as_1d_f64(buffer_points.get("theta_deg", []), "buffer.theta_deg")
    buffer_phi = _as_1d_f64(buffer_points.get("phi_deg", []), "buffer.phi_deg")
    buffer_val = _as_1d_f64(buffer_value, "buffer_value")
    _require_same_size(
        [
            ("alpha_deg", buffer_alpha),
            ("eta_deg", buffer_eta),
            ("theta_deg", buffer_theta),
            ("phi_deg", buffer_phi),
        ],
        buffer_val.size,
        "buffer",
    )
    buffer_side = _required_mask(
        buffer_points, buffer_side_key, buffer_val.size, "buffer"
    )

    outer_alpha = _as_1d_f64(outer_points.get("alpha_deg", []), "outer.alpha_deg")
    outer_eta = _as_1d_f64(outer_points.get("eta_deg", []), "outer.eta_deg")
    outer_theta = _as_1d_f64(outer_points.get("theta_deg", []), "outer.theta_deg")
    outer_phi = _as_1d_f64(outer_points.get("phi_deg", []), "outer.phi_deg")
    outer_val = _as_1d_f64(outer_value, "outer_value")
    _require_same_size(
        [
            ("alpha_deg", outer_alpha),
            ("eta_deg", outer_eta),
            ("theta_deg", outer_theta),
            ("phi_deg", outer_phi),
        ],
        outer_val.size,
        "outer",
    )
    outer_side = _required_mask(
        outer_points, outer_side_key, outer_val.size, "outer"
    )

    return Stage1ScanMetricInput(
        theta_s_deg=theta_value,
        core_alpha_deg=core_alpha,
        core_value=core_val,
        core_is_main_guard=core_main,
        core_is_side_1d=core_side,
        buffer_alpha_deg=buffer_alpha,
        buffer_eta_deg=buffer_eta,
        buffer_theta_deg=buffer_theta,
        buffer_phi_deg=buffer_phi,
        buffer_value=buffer_val,
        buffer_is_side_2d=buffer_side,
        outer_alpha_deg=outer_alpha,
        outer_eta_deg=outer_eta,
        outer_theta_deg=outer_theta,
        outer_phi_deg=outer_phi,
        outer_value=outer_val,
        outer_is_side_2d=outer_side,
        meta=meta_dict,
    )


# -----------------------------------------------------------------------------
# Core PSLL evaluation
# -----------------------------------------------------------------------------


def eval_psll_stage1_single(
    scan: Stage1ScanMetricInput,
    *,
    eps: float = 1e-300,
    on_empty: str = "error",
    verbose: bool = False,
    validate: bool = True,
) -> Stage1ScanPSLLResult:
    if on_empty not in {"error", "nan"}:
        raise ValueError("on_empty must be 'error' or 'nan'.")
    eps = _validate_eps(eps)
    sc = _validate_scan(scan) if validate else scan

    D_core, argmax_core_main, counts_core_main = _masked_max(sc.core_value, sc.core_is_main_guard)
    if counts_core_main == 0:
        if on_empty == "error":
            raise ValueError(f"theta_s={sc.theta_s_deg:.6g}: core main-guard set is empty.")
        return Stage1ScanPSLLResult(
            theta_s_deg=sc.theta_s_deg,
            D_core=float("nan"),
            U_core_side=float("nan"),
            U_buffer=float("nan"),
            U_outer=float("nan"),
            U_side=float("nan"),
            psll_db=float("nan"),
            zone_wc="none",
            idx_wc_local=-1,
            argmax_core_main=-1,
            argmax_core_side=-1,
            argmax_buffer=-1,
            argmax_outer=-1,
            counts_core_main=0,
            counts_core_side=0,
            counts_buffer=0,
            counts_outer=0,
            meta=dict(dict(sc.meta), empty_reason="core_main_guard_empty"),
        )

    U_core_side, argmax_core_side, counts_core_side = _masked_max(sc.core_value, sc.core_is_side_1d)
    U_buffer, argmax_buffer, counts_buffer = _masked_max(sc.buffer_value, sc.buffer_is_side_2d)
    U_outer, argmax_outer, counts_outer = _masked_max(sc.outer_value, sc.outer_is_side_2d)

    candidates = [
        ("core_side_1d", U_core_side, argmax_core_side),
        ("buffer_side_2d", U_buffer, argmax_buffer),
        ("outer_side_2d", U_outer, argmax_outer),
    ]
    finite_candidates = [(z, u, idx) for (z, u, idx) in candidates if np.isfinite(u)]
    result_meta = dict(sc.meta)
    if len(finite_candidates) == 0:
        if on_empty == "error":
            raise ValueError(f"theta_s={sc.theta_s_deg:.6g}: all sidelobe regions are empty.")
        zone_wc = "none"
        idx_wc_local = -1
        U_side = float("nan")
        psll_db = float("nan")
        result_meta["empty_reason"] = "all_sidelobe_regions_empty"
    else:
        zone_wc, U_side, idx_wc_local = max(finite_candidates, key=lambda x: x[1])
        if D_core <= eps:
            if on_empty == "error":
                raise ValueError(
                    "theta_s={0:.6g}: core main-guard denominator is non-positive.".format(
                        sc.theta_s_deg
                    )
                )
            psll_db = float("nan")
            result_meta["invalid_reason"] = "core_main_guard_nonpositive"
        elif U_side <= 0.0:
            psll_db = float("-inf")
        else:
            psll_db = float(20.0 * np.log10(U_side / D_core))

    if verbose:
        print(f"=== GEOMETRIC_PSLL_METRICS SINGLE theta_s={sc.theta_s_deg:.3f} deg ===")
        print(f"[counts] core_main={counts_core_main} core_side={counts_core_side} buffer={counts_buffer} outer={counts_outer}")
        print(f"[D_core] value={D_core:.6g} argmax={argmax_core_main}")
        print(f"[U_core_side] value={U_core_side:.6g} argmax={argmax_core_side}")
        print(f"[U_buffer] value={U_buffer:.6g} argmax={argmax_buffer}")
        print(f"[U_outer] value={U_outer:.6g} argmax={argmax_outer}")
        print(f"[worst_side] zone={zone_wc} idx={idx_wc_local} U_side={U_side:.6g} psll_db={psll_db:.6f}")

    return Stage1ScanPSLLResult(
        theta_s_deg=sc.theta_s_deg,
        D_core=float(D_core),
        U_core_side=float(U_core_side),
        U_buffer=float(U_buffer),
        U_outer=float(U_outer),
        U_side=float(U_side),
        psll_db=float(psll_db),
        zone_wc=str(zone_wc),
        idx_wc_local=int(idx_wc_local),
        argmax_core_main=int(argmax_core_main),
        argmax_core_side=int(argmax_core_side),
        argmax_buffer=int(argmax_buffer),
        argmax_outer=int(argmax_outer),
        counts_core_main=int(counts_core_main),
        counts_core_side=int(counts_core_side),
        counts_buffer=int(counts_buffer),
        counts_outer=int(counts_outer),
        meta=result_meta,
    )



def eval_psll_stage1_batch(
    scans: Sequence[Stage1ScanMetricInput],
    *,
    eps: float = 1e-300,
    on_empty: str = "error",
    verbose: bool = False,
    validate: bool = True,
) -> Stage1BatchPSLLResult:
    if len(scans) == 0:
        raise ValueError("scans must be non-empty.")
    eps = _validate_eps(eps)

    results = tuple(
        eval_psll_stage1_single(
            s,
            eps=eps,
            on_empty=on_empty,
            verbose=verbose,
            validate=validate,
        )
        for s in scans
    )
    psll_s_db = np.asarray([r.psll_db for r in results], dtype=np.float64)
    D_s = np.asarray([r.D_core for r in results], dtype=np.float64)
    U_s = np.asarray([r.U_side for r in results], dtype=np.float64)
    zone_wc_s = np.asarray([r.zone_wc for r in results], dtype=object)

    valid = ~np.isnan(psll_s_db)
    if not np.any(valid):
        idx_wc_s = -1
        psll_wc_db = float("nan")
    else:
        idxs = np.flatnonzero(valid)
        j = int(np.argmax(psll_s_db[idxs]))
        idx_wc_s = int(idxs[j])
        psll_wc_db = float(psll_s_db[idx_wc_s])

    return Stage1BatchPSLLResult(
        psll_s_db=psll_s_db,
        psll_wc_db=psll_wc_db,
        idx_wc_s=idx_wc_s,
        D_s=D_s,
        U_s=U_s,
        zone_wc_s=zone_wc_s,
        per_scan_results=results,
        meta={"n_scans": len(results)},
    )


# -----------------------------------------------------------------------------
# Raw sidelobe ranking (for future active-set / tracking consumers)
# -----------------------------------------------------------------------------


def _ranking_one_region(
    values: np.ndarray,
    mask: np.ndarray,
    D_core: float,
    eps: float,
    rank_by: str,
    topk: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    vals = values[idx].astype(np.float64)
    ratio_db = _ratio_db(vals, D_core, eps)
    if rank_by == "value":
        order = _top_desc_indices(vals, topk=topk)
    elif rank_by == "ratio_db":
        order = _top_desc_indices(ratio_db, topk=topk)
    else:
        raise ValueError("rank_by must be 'value' or 'ratio_db'.")
    return idx[order].astype(np.int64), vals[order].astype(np.float64), ratio_db[order].astype(np.float64)



def rank_stage1_side_candidates_single(
    scan: Stage1ScanMetricInput,
    *,
    D_core: Optional[float] = None,
    eps: float = 1e-300,
    rank_by: str = "ratio_db",
    topk_per_zone: Optional[int] = None,
    topk_global: Optional[int] = None,
    validate: bool = True,
    verbose: bool = False,
) -> Stage1SideRankingResult:
    eps = _validate_eps(eps)
    sc = _validate_scan(scan) if validate else scan
    if D_core is None:
        D_core, _, n_main = _masked_max(sc.core_value, sc.core_is_main_guard)
        if n_main == 0:
            raise ValueError(f"theta_s={sc.theta_s_deg:.6g}: core main-guard set is empty.")
    D_core = float(D_core)
    if not np.isfinite(D_core):
        raise ValueError("D_core must be finite.")
    if D_core <= eps:
        raise ValueError("D_core must be greater than eps.")

    core_idx, core_val, core_ratio = _ranking_one_region(sc.core_value, sc.core_is_side_1d, D_core, eps, rank_by, topk=topk_per_zone)
    buf_idx, buf_val, buf_ratio = _ranking_one_region(sc.buffer_value, sc.buffer_is_side_2d, D_core, eps, rank_by, topk=topk_per_zone)
    out_idx, out_val, out_ratio = _ranking_one_region(sc.outer_value, sc.outer_is_side_2d, D_core, eps, rank_by, topk=topk_per_zone)

    global_zone_list = [
        np.asarray(["core_side_1d"] * core_idx.size, dtype=object),
        np.asarray(["buffer_side_2d"] * buf_idx.size, dtype=object),
        np.asarray(["outer_side_2d"] * out_idx.size, dtype=object),
    ]
    global_idx_list = [core_idx, buf_idx, out_idx]
    global_val_list = [core_val, buf_val, out_val]
    global_ratio_list = [core_ratio, buf_ratio, out_ratio]

    global_zone = np.concatenate(global_zone_list) if any(x.size for x in global_zone_list) else np.asarray([], dtype=object)
    global_idx = np.concatenate(global_idx_list) if any(x.size for x in global_idx_list) else np.asarray([], dtype=np.int64)
    global_val = np.concatenate(global_val_list) if any(x.size for x in global_val_list) else np.asarray([], dtype=np.float64)
    global_ratio = np.concatenate(global_ratio_list) if any(x.size for x in global_ratio_list) else np.asarray([], dtype=np.float64)

    if global_val.size == 0:
        global_order = np.asarray([], dtype=np.int64)
    else:
        global_score = global_val if rank_by == "value" else global_ratio
        global_order = _top_desc_indices(global_score, topk=topk_global)

    global_zone = global_zone[global_order]
    global_idx = global_idx[global_order]
    global_val = global_val[global_order]
    global_ratio = global_ratio[global_order]

    if verbose:
        print(f"=== GEOMETRIC_PSLL_METRICS RANKING theta_s={sc.theta_s_deg:.3f} deg ===")
        print(
            f"[ranking] core_side={core_idx.size} buffer={buf_idx.size} outer={out_idx.size} "
            f"global={global_idx.size} rank_by={rank_by} "
            f"topk_per_zone={topk_per_zone} topk_global={topk_global}"
        )
        if global_idx.size:
            print(f"[ranking top1] zone={global_zone[0]} idx={int(global_idx[0])} val={float(global_val[0]):.6g} ratio_db={float(global_ratio[0]):.6f}")

    return Stage1SideRankingResult(
        core_side_idx_sorted=core_idx,
        core_side_val_sorted=core_val,
        core_side_ratio_db_sorted=core_ratio,
        buffer_idx_sorted=buf_idx,
        buffer_val_sorted=buf_val,
        buffer_ratio_db_sorted=buf_ratio,
        outer_idx_sorted=out_idx,
        outer_val_sorted=out_val,
        outer_ratio_db_sorted=out_ratio,
        global_zone_sorted=global_zone,
        global_idx_sorted=global_idx,
        global_val_sorted=global_val,
        global_ratio_db_sorted=global_ratio,
        meta={
            "theta_s_deg": float(sc.theta_s_deg),
            "rank_by": str(rank_by),
            "topk_per_zone": None if topk_per_zone is None else int(topk_per_zone),
            "topk_global": None if topk_global is None else int(topk_global),
            "validated": bool(validate),
        },
    )


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------


def _run_self_tests() -> None:
    print("=== GEOMETRIC_PSLL_METRICS SELF TESTS: START ===")

    scan = Stage1ScanMetricInput(
        theta_s_deg=30.0,
        core_alpha_deg=np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0]),
        core_value=np.array([0.92, 1.00, 0.97, 0.83, 0.17, 0.11]),
        core_is_main_guard=np.array([True, True, True, True, False, False]),
        core_is_side_1d=np.array([False, False, False, False, True, True]),
        buffer_alpha_deg=np.array([60.0, 66.0, 72.0]),
        buffer_eta_deg=np.array([180.0, 180.0, 180.0]),
        buffer_theta_deg=np.array([80.0, 82.0, 85.0]),
        buffer_phi_deg=np.array([0.0, 0.0, 0.0]),
        buffer_value=np.array([0.10, 0.08, 0.09]),
        buffer_is_side_2d=np.array([True, True, True]),
        outer_alpha_deg=np.array([95.0, 100.0]),
        outer_eta_deg=np.array([180.0, 180.0]),
        outer_theta_deg=np.array([88.0, 89.0]),
        outer_phi_deg=np.array([180.0, 180.0]),
        outer_value=np.array([0.07, 0.06]),
        outer_is_side_2d=np.array([True, True]),
        meta={},
    )

    r = eval_psll_stage1_single(scan, verbose=True)
    assert abs(r.D_core - 1.0) < 1e-12
    assert abs(r.U_side - 0.17) < 1e-12
    assert r.zone_wc == "core_side_1d"
    assert r.psll_db < -15.0

    rk = rank_stage1_side_candidates_single(scan, verbose=True)
    assert rk.global_idx_sorted.size == 7
    assert rk.global_zone_sorted[0] == "core_side_1d"

    rk_top = rank_stage1_side_candidates_single(
        scan,
        topk_per_zone=1,
        topk_global=2,
        validate=True,
        verbose=True,
    )
    assert rk_top.core_side_idx_sorted.size == 1
    assert rk_top.buffer_idx_sorted.size == 1
    assert rk_top.outer_idx_sorted.size == 1
    assert rk_top.global_idx_sorted.size == 2
    assert rk_top.global_zone_sorted[0] == "core_side_1d"

    r_fast = eval_psll_stage1_single(scan, validate=False)
    assert abs(r_fast.psll_db - r.psll_db) < 1e-12
    assert r_fast.zone_wc == r.zone_wc

    batch = eval_psll_stage1_batch([scan, scan], verbose=False)
    assert batch.psll_s_db.size == 2
    assert batch.idx_wc_s in (0, 1)

    batch_fast = eval_psll_stage1_batch([scan, scan], validate=False, verbose=False)
    assert np.allclose(batch_fast.psll_s_db, batch.psll_s_db)
    assert batch_fast.idx_wc_s == batch.idx_wc_s
    zero_side = Stage1ScanMetricInput(
        theta_s_deg=30.0,
        core_alpha_deg=np.array([0.0, 8.0]),
        core_value=np.array([1.0, 0.0]),
        core_is_main_guard=np.array([True, False]),
        core_is_side_1d=np.array([False, True]),
        buffer_alpha_deg=np.array([], dtype=np.float64),
        buffer_eta_deg=np.array([], dtype=np.float64),
        buffer_theta_deg=np.array([], dtype=np.float64),
        buffer_phi_deg=np.array([], dtype=np.float64),
        buffer_value=np.array([], dtype=np.float64),
        buffer_is_side_2d=np.array([], dtype=bool),
        outer_alpha_deg=np.array([], dtype=np.float64),
        outer_eta_deg=np.array([], dtype=np.float64),
        outer_theta_deg=np.array([], dtype=np.float64),
        outer_phi_deg=np.array([], dtype=np.float64),
        outer_value=np.array([], dtype=np.float64),
        outer_is_side_2d=np.array([], dtype=bool),
        meta={},
    )
    zero_result = eval_psll_stage1_single(zero_side)
    assert np.isneginf(zero_result.psll_db)
    zero_batch = eval_psll_stage1_batch([zero_side, zero_side])
    assert zero_batch.idx_wc_s == 0
    assert np.isneginf(zero_batch.psll_wc_db)

    print("=== GEOMETRIC_PSLL_METRICS SELF TESTS: PASS ===")


if __name__ == "__main__":
    _run_self_tests()
