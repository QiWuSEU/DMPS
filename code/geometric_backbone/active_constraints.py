#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Select active geometric sidelobe constraints from mixed 1D/2D rankings.

Active-set builder for the geometric backbone synthesis stage.

Responsibilities
----------------
This module ONLY performs layer-2 active-set selection:
- consumes per-scan raw sidelobe rankings (already computed upstream)
- applies thresholding, de-redundancy / NMS, per-zone bounds, per-scan cap,
  and global cap
- returns the final constraint-point set to be consumed by downstream solver
  adapters / convex subproblems

This module intentionally does NOT:
- build point pools
- compute a0 / AF / co-channel values
- compute PSLL itself
- solve convex optimization subproblems

Design notes
------------
The stage-1 sidelobe domain is split into three zones:
- core_side_1d
- buffer_side_2d
- outer_side_2d

The output indexing scheme is no longer the legacy flat (t,s) work-grid index.
Instead, each selected point is represented by:
- zone_name
- idx_local (local index inside that zone for the given scan)
"""

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Python 3.6-compatible record classes
# -----------------------------------------------------------------------------


class Stage1ScanRankingInput(object):
    __slots__ = (
        "theta_s_deg", "D_core", "psll_db",
        "core_side_idx_sorted", "core_side_val_sorted",
        "core_side_ratio_db_sorted", "core_side_alpha_deg_sorted",
        "buffer_idx_sorted", "buffer_val_sorted", "buffer_ratio_db_sorted",
        "buffer_alpha_deg_sorted", "buffer_eta_deg_sorted",
        "buffer_theta_deg_sorted", "buffer_phi_deg_sorted",
        "outer_idx_sorted", "outer_val_sorted", "outer_ratio_db_sorted",
        "outer_alpha_deg_sorted", "outer_eta_deg_sorted",
        "outer_theta_deg_sorted", "outer_phi_deg_sorted", "meta",
    )

    def __init__(
        self,
        theta_s_deg,
        D_core,
        psll_db,
        core_side_idx_sorted,
        core_side_val_sorted,
        core_side_ratio_db_sorted,
        core_side_alpha_deg_sorted,
        buffer_idx_sorted,
        buffer_val_sorted,
        buffer_ratio_db_sorted,
        buffer_alpha_deg_sorted,
        buffer_eta_deg_sorted,
        buffer_theta_deg_sorted,
        buffer_phi_deg_sorted,
        outer_idx_sorted,
        outer_val_sorted,
        outer_ratio_db_sorted,
        outer_alpha_deg_sorted,
        outer_eta_deg_sorted,
        outer_theta_deg_sorted,
        outer_phi_deg_sorted,
        meta=None,
    ):
        self.theta_s_deg = theta_s_deg
        self.D_core = D_core
        self.psll_db = psll_db
        self.core_side_idx_sorted = core_side_idx_sorted
        self.core_side_val_sorted = core_side_val_sorted
        self.core_side_ratio_db_sorted = core_side_ratio_db_sorted
        self.core_side_alpha_deg_sorted = core_side_alpha_deg_sorted
        self.buffer_idx_sorted = buffer_idx_sorted
        self.buffer_val_sorted = buffer_val_sorted
        self.buffer_ratio_db_sorted = buffer_ratio_db_sorted
        self.buffer_alpha_deg_sorted = buffer_alpha_deg_sorted
        self.buffer_eta_deg_sorted = buffer_eta_deg_sorted
        self.buffer_theta_deg_sorted = buffer_theta_deg_sorted
        self.buffer_phi_deg_sorted = buffer_phi_deg_sorted
        self.outer_idx_sorted = outer_idx_sorted
        self.outer_val_sorted = outer_val_sorted
        self.outer_ratio_db_sorted = outer_ratio_db_sorted
        self.outer_alpha_deg_sorted = outer_alpha_deg_sorted
        self.outer_eta_deg_sorted = outer_eta_deg_sorted
        self.outer_theta_deg_sorted = outer_theta_deg_sorted
        self.outer_phi_deg_sorted = outer_phi_deg_sorted
        self.meta = {} if meta is None else dict(meta)


class Stage1ScanActiveSet(object):
    __slots__ = (
        "theta_s_deg", "zone_name", "idx_local", "alpha_deg", "eta_deg",
        "theta_deg", "phi_deg", "val_side", "ratio_db", "n_core_side",
        "n_buffer_side", "n_outer_side", "mandatory_mask", "meta",
    )

    def __init__(
        self,
        theta_s_deg,
        zone_name,
        idx_local,
        alpha_deg,
        eta_deg,
        theta_deg,
        phi_deg,
        val_side,
        ratio_db,
        n_core_side,
        n_buffer_side,
        n_outer_side,
        mandatory_mask,
        meta=None,
    ):
        self.theta_s_deg = theta_s_deg
        self.zone_name = zone_name
        self.idx_local = idx_local
        self.alpha_deg = alpha_deg
        self.eta_deg = eta_deg
        self.theta_deg = theta_deg
        self.phi_deg = phi_deg
        self.val_side = val_side
        self.ratio_db = ratio_db
        self.n_core_side = n_core_side
        self.n_buffer_side = n_buffer_side
        self.n_outer_side = n_outer_side
        self.mandatory_mask = mandatory_mask
        self.meta = {} if meta is None else dict(meta)


class Stage1BatchActiveSet(object):
    __slots__ = (
        "per_scan", "theta_s_deg", "per_scan_counts", "total_M", "meta",
    )

    def __init__(
        self,
        per_scan,
        theta_s_deg,
        per_scan_counts,
        total_M,
        meta=None,
    ):
        self.per_scan = tuple(per_scan)
        self.theta_s_deg = theta_s_deg
        self.per_scan_counts = per_scan_counts
        self.total_M = total_M
        self.meta = {} if meta is None else dict(meta)


ZONE_PRIORITY = {
    "core_side_1d": 0,
    "buffer_side_2d": 1,
    "outer_side_2d": 2,
}
VALID_ZONE_NAMES = set(ZONE_PRIORITY.keys())


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _as_1d_f64(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(arr)):
        raise ValueError("{0} contains NaN/Inf.".format(name))
    return arr


def _as_1d_ratio_db(x: Any, name: str) -> np.ndarray:
    """Return dB ratios, allowing -Inf for exactly zero sidelobe values."""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if np.any(np.isnan(arr)):
        raise ValueError("{0} contains NaN.".format(name))
    if np.any(np.isposinf(arr)):
        raise ValueError("{0} contains +Inf.".format(name))
    return arr


def _as_1d_i64(x: Any, name: str) -> np.ndarray:
    raw = np.asarray(x).reshape(-1)
    if raw.size == 0:
        return np.asarray([], dtype=np.int64)
    try:
        raw_f64 = np.asarray(raw, dtype=np.float64)
    except Exception:
        raise ValueError("{0} must contain integer indices.".format(name))
    if np.any(~np.isfinite(raw_f64)):
        raise ValueError("{0} contains NaN/Inf.".format(name))
    if np.any(raw_f64 != np.floor(raw_f64)):
        raise ValueError("{0} contains non-integer indices.".format(name))
    if np.any(raw_f64 < 0.0):
        raise ValueError("{0} contains negative indices.".format(name))
    if np.any(raw_f64 > float(np.iinfo(np.int64).max)):
        raise ValueError("{0} contains indices outside int64 range.".format(name))
    return raw_f64.astype(np.int64)


def _as_1d_obj(x: Any, name: str) -> np.ndarray:
    return np.asarray(x, dtype=object).reshape(-1)


def _empty_f64() -> np.ndarray:
    return np.asarray([], dtype=np.float64)


def _empty_i64() -> np.ndarray:
    return np.asarray([], dtype=np.int64)


def wrapped_eta_diff_deg(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def _sort_desc(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return _empty_i64()
    if np.any(np.isnan(vals)):
        raise ValueError("sort values contain NaN.")
    if np.any(np.isposinf(vals)):
        raise ValueError("sort values contain +Inf.")
    idx = np.arange(vals.size, dtype=np.int64)
    return np.lexsort((idx, -vals)).astype(np.int64)


def _ordered_unique_i64(x: Iterable[int]) -> np.ndarray:
    seen = set()
    out: List[int] = []
    for v in x:
        iv = int(v)
        if iv not in seen:
            seen.add(iv)
            out.append(iv)
    return np.asarray(out, dtype=np.int64)


def _validate_nonnegative_float(value: Any, name: str) -> float:
    x = float(value)
    if not np.isfinite(x):
        raise ValueError("{0} must be finite.".format(name))
    if x < 0.0:
        raise ValueError("{0} must be >= 0.".format(name))
    return x


def _validate_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{0} must be an integer.".format(name))
    try:
        x_float = float(value)
    except Exception:
        raise ValueError("{0} must be a non-negative integer.".format(name))
    if (not np.isfinite(x_float)) or x_float != math.floor(x_float) or x_float < 0.0:
        raise ValueError("{0} must be a non-negative integer.".format(name))
    return int(x_float)


def _validate_optional_nonnegative_int(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    return _validate_nonnegative_int(value, name)


def _expected_ratio_db(values: np.ndarray, D_core: float) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full(vals.shape, -np.inf, dtype=np.float64)
    positive = vals > 0.0
    out[positive] = 20.0 * np.log10(vals[positive] / float(D_core))
    return out


def _validate_value_ratio_pair(
    values: np.ndarray,
    ratio_db: np.ndarray,
    D_core: float,
    region_name: str,
    atol: float = 1e-9,
) -> None:
    expected = _expected_ratio_db(values, D_core)
    if not np.array_equal(np.isneginf(expected), np.isneginf(ratio_db)):
        raise ValueError(
            "{0} value/ratio zero pattern is inconsistent.".format(region_name)
        )
    finite = np.isfinite(expected)
    if np.any(np.abs(expected[finite] - ratio_db[finite]) > float(atol)):
        raise ValueError(
            "{0} value and ratio_db are inconsistent.".format(region_name)
        )


def _validate_descending_ratio(
    ratio_db: np.ndarray,
    region_name: str,
    tol: float = 1e-12,
) -> None:
    if ratio_db.size <= 1:
        return
    left = ratio_db[:-1]
    right = ratio_db[1:]
    if np.any(left + float(tol) < right):
        raise ValueError(
            "{0} candidates are not sorted by descending ratio_db.".format(
                region_name
            )
        )


def _validate_unique_indices(idx: np.ndarray, region_name: str) -> None:
    if np.unique(idx).size != idx.size:
        raise ValueError(
            "duplicate local indices in {0} zone.".format(region_name)
        )


def _validate_index_bounds(
    idx: np.ndarray,
    source_size: int,
    region_name: str,
) -> None:
    if idx.size == 0:
        return
    if int(np.max(idx)) >= int(source_size):
        raise IndexError(
            "{0} ranking index exceeds source point count.".format(region_name)
        )


def _preclip_arrays(preclip_candidates: Optional[int], *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Keep only the first N already-sorted candidates from a zone."""
    if preclip_candidates is None:
        return tuple(np.asarray(a) for a in arrays)
    m = int(preclip_candidates)
    return tuple(np.asarray(a)[:m] for a in arrays)


# -----------------------------------------------------------------------------
# Input validation / builder from metrics objects
# -----------------------------------------------------------------------------


def _validate_scan_ranking_input(inp: Stage1ScanRankingInput) -> Stage1ScanRankingInput:
    th = float(inp.theta_s_deg)
    if not np.isfinite(th):
        raise ValueError("theta_s_deg must be finite.")

    D_core = float(inp.D_core)
    if not np.isfinite(D_core):
        raise ValueError("D_core must be finite.")
    if D_core <= 0.0:
        raise ValueError("D_core must be > 0.")

    psll_db = float(inp.psll_db)
    if np.isnan(psll_db):
        raise ValueError("psll_db must not be NaN.")
    if np.isposinf(psll_db):
        raise ValueError("psll_db must not be +Inf.")

    core_idx = _as_1d_i64(inp.core_side_idx_sorted, "core_side_idx_sorted")
    core_val = _as_1d_f64(inp.core_side_val_sorted, "core_side_val_sorted")
    core_ratio = _as_1d_ratio_db(inp.core_side_ratio_db_sorted, "core_side_ratio_db_sorted")
    core_alpha = _as_1d_f64(inp.core_side_alpha_deg_sorted, "core_side_alpha_deg_sorted")
    n_core = core_idx.size
    for arr, name in [
        (core_val, "core_side_val_sorted"),
        (core_ratio, "core_side_ratio_db_sorted"),
        (core_alpha, "core_side_alpha_deg_sorted"),
    ]:
        if arr.size != n_core:
            raise ValueError("{0} must have the same length as core_side_idx_sorted.".format(name))
    if np.any(core_val < 0.0):
        raise ValueError("core_side_val_sorted must be non-negative.")
    _validate_unique_indices(core_idx, "core_side_1d")
    _validate_value_ratio_pair(core_val, core_ratio, D_core, "core_side_1d")
    _validate_descending_ratio(core_ratio, "core_side_1d")

    buf_idx = _as_1d_i64(inp.buffer_idx_sorted, "buffer_idx_sorted")
    buf_val = _as_1d_f64(inp.buffer_val_sorted, "buffer_val_sorted")
    buf_ratio = _as_1d_ratio_db(inp.buffer_ratio_db_sorted, "buffer_ratio_db_sorted")
    buf_alpha = _as_1d_f64(inp.buffer_alpha_deg_sorted, "buffer_alpha_deg_sorted")
    buf_eta = _as_1d_f64(inp.buffer_eta_deg_sorted, "buffer_eta_deg_sorted")
    buf_theta = _as_1d_f64(inp.buffer_theta_deg_sorted, "buffer_theta_deg_sorted")
    buf_phi = _as_1d_f64(inp.buffer_phi_deg_sorted, "buffer_phi_deg_sorted")
    n_buf = buf_idx.size
    for arr, name in [
        (buf_val, "buffer_val_sorted"),
        (buf_ratio, "buffer_ratio_db_sorted"),
        (buf_alpha, "buffer_alpha_deg_sorted"),
        (buf_eta, "buffer_eta_deg_sorted"),
        (buf_theta, "buffer_theta_deg_sorted"),
        (buf_phi, "buffer_phi_deg_sorted"),
    ]:
        if arr.size != n_buf:
            raise ValueError("{0} must have the same length as buffer_idx_sorted.".format(name))
    if np.any(buf_val < 0.0):
        raise ValueError("buffer_val_sorted must be non-negative.")
    _validate_unique_indices(buf_idx, "buffer_side_2d")
    _validate_value_ratio_pair(buf_val, buf_ratio, D_core, "buffer_side_2d")
    _validate_descending_ratio(buf_ratio, "buffer_side_2d")

    out_idx = _as_1d_i64(inp.outer_idx_sorted, "outer_idx_sorted")
    out_val = _as_1d_f64(inp.outer_val_sorted, "outer_val_sorted")
    out_ratio = _as_1d_ratio_db(inp.outer_ratio_db_sorted, "outer_ratio_db_sorted")
    out_alpha = _as_1d_f64(inp.outer_alpha_deg_sorted, "outer_alpha_deg_sorted")
    out_eta = _as_1d_f64(inp.outer_eta_deg_sorted, "outer_eta_deg_sorted")
    out_theta = _as_1d_f64(inp.outer_theta_deg_sorted, "outer_theta_deg_sorted")
    out_phi = _as_1d_f64(inp.outer_phi_deg_sorted, "outer_phi_deg_sorted")
    n_out = out_idx.size
    for arr, name in [
        (out_val, "outer_val_sorted"),
        (out_ratio, "outer_ratio_db_sorted"),
        (out_alpha, "outer_alpha_deg_sorted"),
        (out_eta, "outer_eta_deg_sorted"),
        (out_theta, "outer_theta_deg_sorted"),
        (out_phi, "outer_phi_deg_sorted"),
    ]:
        if arr.size != n_out:
            raise ValueError("{0} must have the same length as outer_idx_sorted.".format(name))
    if np.any(out_val < 0.0):
        raise ValueError("outer_val_sorted must be non-negative.")
    _validate_unique_indices(out_idx, "outer_side_2d")
    _validate_value_ratio_pair(out_val, out_ratio, D_core, "outer_side_2d")
    _validate_descending_ratio(out_ratio, "outer_side_2d")

    return Stage1ScanRankingInput(
        theta_s_deg=th,
        D_core=D_core,
        psll_db=psll_db,
        core_side_idx_sorted=core_idx,
        core_side_val_sorted=core_val,
        core_side_ratio_db_sorted=core_ratio,
        core_side_alpha_deg_sorted=core_alpha,
        buffer_idx_sorted=buf_idx,
        buffer_val_sorted=buf_val,
        buffer_ratio_db_sorted=buf_ratio,
        buffer_alpha_deg_sorted=buf_alpha,
        buffer_eta_deg_sorted=buf_eta,
        buffer_theta_deg_sorted=buf_theta,
        buffer_phi_deg_sorted=buf_phi,
        outer_idx_sorted=out_idx,
        outer_val_sorted=out_val,
        outer_ratio_db_sorted=out_ratio,
        outer_alpha_deg_sorted=out_alpha,
        outer_eta_deg_sorted=out_eta,
        outer_theta_deg_sorted=out_theta,
        outer_phi_deg_sorted=out_phi,
        meta=dict(inp.meta),
    )


def build_stage1_scan_ranking_input_from_metric(
    *,
    scan_input: Any,
    psll_result: Any,
    ranking_result: Any,
    meta: Optional[Mapping[str, Any]] = None,
) -> Stage1ScanRankingInput:
    """Convenience adapter from geometric_psll_metrics outputs."""
    meta_dict = dict(meta) if meta is not None else {}

    theta_scan = float(scan_input.theta_s_deg)
    theta_psll = float(psll_result.theta_s_deg)
    if not np.isfinite(theta_scan) or not np.isfinite(theta_psll):
        raise ValueError("scan angles must be finite.")
    if abs(theta_scan - theta_psll) > 1e-9:
        raise ValueError(
            "scan_input and psll_result refer to different scan angles."
        )

    ranking_meta = dict(getattr(ranking_result, "meta", {}))
    if "theta_s_deg" in ranking_meta:
        theta_rank = float(ranking_meta["theta_s_deg"])
        if not np.isfinite(theta_rank) or abs(theta_rank - theta_psll) > 1e-9:
            raise ValueError(
                "ranking_result and psll_result refer to different scan angles."
            )

    core_idx = _as_1d_i64(ranking_result.core_side_idx_sorted, "core_side_idx_sorted")
    buf_idx = _as_1d_i64(ranking_result.buffer_idx_sorted, "buffer_idx_sorted")
    out_idx = _as_1d_i64(ranking_result.outer_idx_sorted, "outer_idx_sorted")

    core_alpha = _as_1d_f64(scan_input.core_alpha_deg, "scan_input.core_alpha_deg")
    buf_alpha = _as_1d_f64(scan_input.buffer_alpha_deg, "scan_input.buffer_alpha_deg")
    buf_eta = _as_1d_f64(scan_input.buffer_eta_deg, "scan_input.buffer_eta_deg")
    buf_theta = _as_1d_f64(scan_input.buffer_theta_deg, "scan_input.buffer_theta_deg")
    buf_phi = _as_1d_f64(scan_input.buffer_phi_deg, "scan_input.buffer_phi_deg")
    out_alpha = _as_1d_f64(scan_input.outer_alpha_deg, "scan_input.outer_alpha_deg")
    out_eta = _as_1d_f64(scan_input.outer_eta_deg, "scan_input.outer_eta_deg")
    out_theta = _as_1d_f64(scan_input.outer_theta_deg, "scan_input.outer_theta_deg")
    out_phi = _as_1d_f64(scan_input.outer_phi_deg, "scan_input.outer_phi_deg")

    if not (buf_alpha.size == buf_eta.size == buf_theta.size == buf_phi.size):
        raise ValueError("buffer source coordinate arrays must have the same length.")
    if not (out_alpha.size == out_eta.size == out_theta.size == out_phi.size):
        raise ValueError("outer source coordinate arrays must have the same length.")

    _validate_index_bounds(core_idx, core_alpha.size, "core_side_1d")
    _validate_index_bounds(buf_idx, buf_alpha.size, "buffer_side_2d")
    _validate_index_bounds(out_idx, out_alpha.size, "outer_side_2d")

    meta_dict.update(dict(getattr(scan_input, "meta", {})))
    meta_dict.update(dict(getattr(psll_result, "meta", {})))

    result = Stage1ScanRankingInput(
        theta_s_deg=theta_psll,
        D_core=float(psll_result.D_core),
        psll_db=float(psll_result.psll_db),
        core_side_idx_sorted=core_idx,
        core_side_val_sorted=np.asarray(ranking_result.core_side_val_sorted, dtype=np.float64).reshape(-1),
        core_side_ratio_db_sorted=np.asarray(ranking_result.core_side_ratio_db_sorted, dtype=np.float64).reshape(-1),
        core_side_alpha_deg_sorted=core_alpha[core_idx] if core_idx.size else _empty_f64(),
        buffer_idx_sorted=buf_idx,
        buffer_val_sorted=np.asarray(ranking_result.buffer_val_sorted, dtype=np.float64).reshape(-1),
        buffer_ratio_db_sorted=np.asarray(ranking_result.buffer_ratio_db_sorted, dtype=np.float64).reshape(-1),
        buffer_alpha_deg_sorted=buf_alpha[buf_idx] if buf_idx.size else _empty_f64(),
        buffer_eta_deg_sorted=buf_eta[buf_idx] if buf_idx.size else _empty_f64(),
        buffer_theta_deg_sorted=buf_theta[buf_idx] if buf_idx.size else _empty_f64(),
        buffer_phi_deg_sorted=buf_phi[buf_idx] if buf_idx.size else _empty_f64(),
        outer_idx_sorted=out_idx,
        outer_val_sorted=np.asarray(ranking_result.outer_val_sorted, dtype=np.float64).reshape(-1),
        outer_ratio_db_sorted=np.asarray(ranking_result.outer_ratio_db_sorted, dtype=np.float64).reshape(-1),
        outer_alpha_deg_sorted=out_alpha[out_idx] if out_idx.size else _empty_f64(),
        outer_eta_deg_sorted=out_eta[out_idx] if out_idx.size else _empty_f64(),
        outer_theta_deg_sorted=out_theta[out_idx] if out_idx.size else _empty_f64(),
        outer_phi_deg_sorted=out_phi[out_idx] if out_idx.size else _empty_f64(),
        meta=meta_dict,
    )
    return _validate_scan_ranking_input(result)


# -----------------------------------------------------------------------------
# Core selection helpers
# -----------------------------------------------------------------------------


def threshold_keep_mask(ratio_db: np.ndarray, ref_best_db: float, delta_keep_db: float) -> np.ndarray:
    ratio = _as_1d_ratio_db(ratio_db, "ratio_db")
    delta = _validate_nonnegative_float(delta_keep_db, "delta_keep_db")
    ref = float(ref_best_db)
    if np.isnan(ref):
        raise ValueError("ref_best_db must not be NaN.")
    if np.isposinf(ref):
        raise ValueError("ref_best_db must not be +Inf.")
    if ratio.size == 0:
        return np.zeros(ratio.shape, dtype=bool)
    if np.isneginf(ref):
        return np.ones(ratio.shape, dtype=bool)
    return ratio >= (ref - delta)


def greedy_merge_1d_alpha(
    alpha_deg: np.ndarray,
    ratio_db: np.ndarray,
    merge_alpha_deg: float,
) -> np.ndarray:
    alpha = _as_1d_f64(alpha_deg, "alpha_deg")
    ratio = _as_1d_ratio_db(ratio_db, "ratio_db")
    merge = _validate_nonnegative_float(merge_alpha_deg, "merge_alpha_deg")
    if alpha.size != ratio.size:
        raise ValueError("alpha_deg and ratio_db must have the same length.")
    if alpha.size == 0:
        return _empty_i64()
    kept: List[int] = []
    for i in range(alpha.size):
        ai = float(alpha[i])
        if all(abs(ai - float(alpha[j])) > merge for j in kept):
            kept.append(i)
    return np.asarray(kept, dtype=np.int64)


def greedy_nms_2d_alpha_eta(
    alpha_deg: np.ndarray,
    eta_deg: np.ndarray,
    merge_alpha_deg: float,
    merge_eta_deg: float,
) -> np.ndarray:
    alpha = _as_1d_f64(alpha_deg, "alpha_deg")
    eta = _as_1d_f64(eta_deg, "eta_deg")
    merge_alpha = _validate_nonnegative_float(merge_alpha_deg, "merge_alpha_deg")
    merge_eta = _validate_nonnegative_float(merge_eta_deg, "merge_eta_deg")
    if alpha.size != eta.size:
        raise ValueError("alpha_deg and eta_deg must have the same length.")
    if alpha.size == 0:
        return _empty_i64()
    kept: List[int] = []
    for i in range(alpha.size):
        ai = float(alpha[i])
        ei = float(eta[i])
        accept = True
        for j in kept:
            aj = float(alpha[j])
            ej = float(eta[j])
            if abs(ai - aj) <= merge_alpha and wrapped_eta_diff_deg(ei, ej) <= merge_eta:
                accept = False
                break
        if accept:
            kept.append(i)
    return np.asarray(kept, dtype=np.int64)


class _ZoneSelected(object):
    __slots__ = (
        "zone_name", "idx_local", "alpha_deg", "eta_deg", "theta_deg",
        "phi_deg", "val_side", "ratio_db", "mandatory_count",
    )

    def __init__(
        self,
        zone_name,
        idx_local,
        alpha_deg,
        eta_deg,
        theta_deg,
        phi_deg,
        val_side,
        ratio_db,
        mandatory_count,
    ):
        self.zone_name = zone_name
        self.idx_local = idx_local
        self.alpha_deg = alpha_deg
        self.eta_deg = eta_deg
        self.theta_deg = theta_deg
        self.phi_deg = phi_deg
        self.val_side = val_side
        self.ratio_db = ratio_db
        self.mandatory_count = mandatory_count


def _select_zone_core(
    *,
    idx_local_sorted: np.ndarray,
    val_sorted: np.ndarray,
    ratio_sorted: np.ndarray,
    alpha_sorted: np.ndarray,
    ref_best_db: float,
    delta_keep_db: float,
    merge_alpha_deg: float,
    min_keep: int,
    max_keep: int,
    preclip_candidates: Optional[int] = None,
) -> _ZoneSelected:
    idx_local_sorted, val_sorted, ratio_sorted, alpha_sorted = _preclip_arrays(
        preclip_candidates, idx_local_sorted, val_sorted, ratio_sorted, alpha_sorted
    )
    n = idx_local_sorted.size
    if n == 0:
        return _ZoneSelected("core_side_1d", _empty_i64(), _empty_f64(), _empty_f64(), _empty_f64(), _empty_f64(), _empty_f64(), _empty_f64(), 0)

    thr_mask = threshold_keep_mask(ratio_sorted, ref_best_db, delta_keep_db)
    cand = np.flatnonzero(thr_mask)
    if cand.size == 0:
        cand = np.arange(n, dtype=np.int64)
    merged_rel = greedy_merge_1d_alpha(alpha_sorted[cand], ratio_sorted[cand], merge_alpha_deg)
    selected = cand[merged_rel].tolist()

    if len(selected) < min_keep:
        used = set(selected)
        # strict fill first
        fill_candidates = [i for i in range(n) if i not in used]
        if fill_candidates:
            fill_alpha = alpha_sorted[np.asarray(fill_candidates, dtype=np.int64)]
            merged_fill_rel = greedy_merge_1d_alpha(fill_alpha, ratio_sorted[np.asarray(fill_candidates, dtype=np.int64)], merge_alpha_deg)
            for rr in merged_fill_rel.tolist():
                ii = fill_candidates[int(rr)]
                if ii not in used:
                    selected.append(ii)
                    used.add(ii)
                if len(selected) >= min_keep:
                    break
        # hard fill if still too few
        if len(selected) < min_keep:
            for ii in range(n):
                if ii not in used:
                    selected.append(ii)
                    used.add(ii)
                if len(selected) >= min_keep:
                    break

    selected = selected[:max_keep]
    sel = np.asarray(selected, dtype=np.int64)
    mandatory_count = min(sel.size, min_keep)
    return _ZoneSelected(
        "core_side_1d",
        idx_local=idx_local_sorted[sel].astype(np.int64),
        alpha_deg=alpha_sorted[sel].astype(np.float64),
        eta_deg=np.full(sel.size, np.nan, dtype=np.float64),
        theta_deg=np.full(sel.size, np.nan, dtype=np.float64),
        phi_deg=np.full(sel.size, np.nan, dtype=np.float64),
        val_side=val_sorted[sel].astype(np.float64),
        ratio_db=ratio_sorted[sel].astype(np.float64),
        mandatory_count=int(mandatory_count),
    )



def _select_zone_2d(
    *,
    zone_name: str,
    idx_local_sorted: np.ndarray,
    val_sorted: np.ndarray,
    ratio_sorted: np.ndarray,
    alpha_sorted: np.ndarray,
    eta_sorted: np.ndarray,
    theta_sorted: np.ndarray,
    phi_sorted: np.ndarray,
    ref_best_db: float,
    delta_keep_db: float,
    merge_alpha_deg: float,
    merge_eta_deg: float,
    min_keep: int,
    max_keep: int,
    preclip_candidates: Optional[int] = None,
) -> _ZoneSelected:
    (
        idx_local_sorted,
        val_sorted,
        ratio_sorted,
        alpha_sorted,
        eta_sorted,
        theta_sorted,
        phi_sorted,
    ) = _preclip_arrays(
        preclip_candidates,
        idx_local_sorted,
        val_sorted,
        ratio_sorted,
        alpha_sorted,
        eta_sorted,
        theta_sorted,
        phi_sorted,
    )
    n = idx_local_sorted.size
    if n == 0:
        return _ZoneSelected(zone_name, _empty_i64(), _empty_f64(), _empty_f64(), _empty_f64(), _empty_f64(), _empty_f64(), _empty_f64(), 0)

    thr_mask = threshold_keep_mask(ratio_sorted, ref_best_db, delta_keep_db)
    cand = np.flatnonzero(thr_mask)
    if cand.size == 0:
        cand = np.arange(n, dtype=np.int64)
    merged_rel = greedy_nms_2d_alpha_eta(alpha_sorted[cand], eta_sorted[cand], merge_alpha_deg, merge_eta_deg)
    selected = cand[merged_rel].tolist()

    if len(selected) < min_keep:
        used = set(selected)
        fill_candidates = [i for i in range(n) if i not in used]
        if fill_candidates:
            fill_alpha = alpha_sorted[np.asarray(fill_candidates, dtype=np.int64)]
            fill_eta = eta_sorted[np.asarray(fill_candidates, dtype=np.int64)]
            merged_fill_rel = greedy_nms_2d_alpha_eta(fill_alpha, fill_eta, merge_alpha_deg, merge_eta_deg)
            for rr in merged_fill_rel.tolist():
                ii = fill_candidates[int(rr)]
                if ii not in used:
                    selected.append(ii)
                    used.add(ii)
                if len(selected) >= min_keep:
                    break
        if len(selected) < min_keep:
            for ii in range(n):
                if ii not in used:
                    selected.append(ii)
                    used.add(ii)
                if len(selected) >= min_keep:
                    break

    selected = selected[:max_keep]
    sel = np.asarray(selected, dtype=np.int64)
    mandatory_count = min(sel.size, min_keep)
    return _ZoneSelected(
        zone_name,
        idx_local=idx_local_sorted[sel].astype(np.int64),
        alpha_deg=alpha_sorted[sel].astype(np.float64),
        eta_deg=eta_sorted[sel].astype(np.float64),
        theta_deg=theta_sorted[sel].astype(np.float64),
        phi_deg=phi_sorted[sel].astype(np.float64),
        val_side=val_sorted[sel].astype(np.float64),
        ratio_db=ratio_sorted[sel].astype(np.float64),
        mandatory_count=int(mandatory_count),
    )



def _validate_scan_active_set_result(result: Stage1ScanActiveSet) -> None:
    theta_s = float(result.theta_s_deg)
    if not np.isfinite(theta_s):
        raise ValueError("active-set theta_s_deg must be finite.")

    zone = np.asarray(result.zone_name, dtype=object).reshape(-1)
    idx = _as_1d_i64(result.idx_local, "active_set.idx_local")
    alpha = _as_1d_f64(result.alpha_deg, "active_set.alpha_deg")
    eta = np.asarray(result.eta_deg, dtype=np.float64).reshape(-1)
    theta = np.asarray(result.theta_deg, dtype=np.float64).reshape(-1)
    phi = np.asarray(result.phi_deg, dtype=np.float64).reshape(-1)
    val = _as_1d_f64(result.val_side, "active_set.val_side")
    ratio = _as_1d_ratio_db(result.ratio_db, "active_set.ratio_db")
    mandatory = np.asarray(result.mandatory_mask).reshape(-1)

    n = idx.size
    for arr, name in [
        (zone, "zone_name"), (alpha, "alpha_deg"), (eta, "eta_deg"),
        (theta, "theta_deg"), (phi, "phi_deg"), (val, "val_side"),
        (ratio, "ratio_db"), (mandatory, "mandatory_mask"),
    ]:
        if arr.size != n:
            raise ValueError("active-set {0} length mismatch.".format(name))
    if np.any(val < 0.0):
        raise ValueError("active-set val_side must be non-negative.")
    if mandatory.dtype != np.bool_:
        if not np.all((mandatory == 0) | (mandatory == 1)):
            raise ValueError("active-set mandatory_mask must be boolean/0/1.")
        mandatory = mandatory.astype(bool)

    for name in zone.tolist():
        if str(name) not in VALID_ZONE_NAMES:
            raise ValueError("active-set contains unknown zone: {0}".format(name))

    keys = [(str(zone[i]), int(idx[i])) for i in range(n)]
    if len(set(keys)) != len(keys):
        raise ValueError("active-set contains duplicate (zone_name, idx_local).")

    core_mask = zone == "core_side_1d"
    two_d_mask = ~core_mask
    if np.any(~np.isfinite(eta[two_d_mask])) or np.any(~np.isfinite(theta[two_d_mask])) or np.any(~np.isfinite(phi[two_d_mask])):
        raise ValueError("2D active-set coordinates must be finite.")
    if np.any(~np.isnan(eta[core_mask])) or np.any(~np.isnan(theta[core_mask])) or np.any(~np.isnan(phi[core_mask])):
        raise ValueError("core 1D eta/theta/phi placeholders must be NaN.")

    n_core = int(np.sum(core_mask))
    n_buf = int(np.sum(zone == "buffer_side_2d"))
    n_out = int(np.sum(zone == "outer_side_2d"))
    if int(result.n_core_side) != n_core or int(result.n_buffer_side) != n_buf or int(result.n_outer_side) != n_out:
        raise ValueError("active-set zone counts do not match output rows.")


def _validate_batch_active_set_result(result: Stage1BatchActiveSet) -> None:
    theta = _as_1d_f64(result.theta_s_deg, "batch.theta_s_deg")
    counts = _as_1d_i64(result.per_scan_counts, "batch.per_scan_counts")
    if theta.size != len(result.per_scan) or counts.size != len(result.per_scan):
        raise ValueError("batch arrays must match per_scan length.")
    for i, scan in enumerate(result.per_scan):
        _validate_scan_active_set_result(scan)
        if abs(float(scan.theta_s_deg) - float(theta[i])) > 1e-12:
            raise ValueError("batch theta_s_deg does not match per_scan result.")
        if int(scan.idx_local.size) != int(counts[i]):
            raise ValueError("batch per_scan_counts is inconsistent.")
    if int(result.total_M) != int(np.sum(counts)):
        raise ValueError("batch total_M is inconsistent.")


def _combine_scan_zone_results(
    theta_s_deg: float,
    zone_results: Sequence[_ZoneSelected],
    max_keep_per_scan: int,
) -> Stage1ScanActiveSet:
    mandatory_rows: List[Tuple[str, int, float, float, float, float, float, float]] = []
    optional_rows: List[Tuple[str, int, float, float, float, float, float, float]] = []

    pre_cap_counts = {
        "core_side_1d": 0,
        "buffer_side_2d": 0,
        "outer_side_2d": 0,
    }

    for zr in zone_results:
        rows = [
            (
                zr.zone_name,
                int(zr.idx_local[i]),
                float(zr.alpha_deg[i]),
                float(zr.eta_deg[i]),
                float(zr.theta_deg[i]),
                float(zr.phi_deg[i]),
                float(zr.val_side[i]),
                float(zr.ratio_db[i]),
            )
            for i in range(zr.idx_local.size)
        ]
        mandatory_rows.extend(rows[: zr.mandatory_count])
        optional_rows.extend(rows[zr.mandatory_count :])
        pre_cap_counts[zr.zone_name] = int(zr.idx_local.size)

    if len(mandatory_rows) > max_keep_per_scan:
        raise ValueError(
            "mandatory selection count ({0}) exceeds max_keep_per_scan ({1}).".format(
                len(mandatory_rows), max_keep_per_scan
            )
        )

    optional_rows.sort(
        key=lambda row: (
            -float(row[7]),
            int(ZONE_PRIORITY[row[0]]),
            int(row[1]),
        )
    )
    rows = mandatory_rows + optional_rows[: max(0, max_keep_per_scan - len(mandatory_rows))]

    if len(rows) == 0:
        zone_name = np.asarray([], dtype=object)
        idx_local = _empty_i64()
        alpha_deg = _empty_f64()
        eta_deg = _empty_f64()
        theta_deg = _empty_f64()
        phi_deg = _empty_f64()
        val_side = _empty_f64()
        ratio_db = _empty_f64()
        mandatory_mask = np.asarray([], dtype=bool)
    else:
        zone_name = np.asarray([r[0] for r in rows], dtype=object)
        idx_local = np.asarray([r[1] for r in rows], dtype=np.int64)
        alpha_deg = np.asarray([r[2] for r in rows], dtype=np.float64)
        eta_deg = np.asarray([r[3] for r in rows], dtype=np.float64)
        theta_deg = np.asarray([r[4] for r in rows], dtype=np.float64)
        phi_deg = np.asarray([r[5] for r in rows], dtype=np.float64)
        val_side = np.asarray([r[6] for r in rows], dtype=np.float64)
        ratio_db = np.asarray([r[7] for r in rows], dtype=np.float64)
        mandatory_mask = np.asarray(
            [i < len(mandatory_rows) for i in range(len(rows))],
            dtype=bool,
        )

    n_core = int(np.sum(zone_name == "core_side_1d"))
    n_buf = int(np.sum(zone_name == "buffer_side_2d"))
    n_out = int(np.sum(zone_name == "outer_side_2d"))

    meta = {
        "selected_M_scan": int(len(rows)),
        "mandatory_M_scan": int(len(mandatory_rows)),
        "max_keep_per_scan": int(max_keep_per_scan),
        "pre_scan_cap_core": int(pre_cap_counts["core_side_1d"]),
        "pre_scan_cap_buffer": int(pre_cap_counts["buffer_side_2d"]),
        "pre_scan_cap_outer": int(pre_cap_counts["outer_side_2d"]),
        "pre_scan_cap_total": int(sum(pre_cap_counts.values())),
    }

    result = Stage1ScanActiveSet(
        theta_s_deg=float(theta_s_deg),
        zone_name=zone_name,
        idx_local=idx_local,
        alpha_deg=alpha_deg,
        eta_deg=eta_deg,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        val_side=val_side,
        ratio_db=ratio_db,
        n_core_side=n_core,
        n_buffer_side=n_buf,
        n_outer_side=n_out,
        mandatory_mask=mandatory_mask,
        meta=meta,
    )
    _validate_scan_active_set_result(result)
    return result


# -----------------------------------------------------------------------------
# Public builders
# -----------------------------------------------------------------------------


def build_active_set_stage1_single(
    ranking: Stage1ScanRankingInput,
    *,
    delta_keep_db: float = 5.0,
    min_keep_core_side_per_scan: int = 15,
    min_keep_buffer_per_scan: int = 9,
    min_keep_outer_per_scan: int = 10,
    max_keep_core_side_per_scan: int = 20,
    max_keep_buffer_per_scan: int = 14,
    max_keep_outer_per_scan: int = 14,
    core_merge_alpha_deg: float = 0.5,
    buffer_merge_alpha_deg: float = 3.0,
    buffer_merge_eta_deg: float = 12.0,
    outer_merge_alpha_deg: float = 4.0,
    outer_merge_eta_deg: float = 15.0,
    max_keep_per_scan: int = 48,
    preclip_core_candidates: Optional[int] = 200,
    preclip_buffer_candidates: Optional[int] = 200,
    preclip_outer_candidates: Optional[int] = 200,
    validate: bool = True,
    on_empty: str = "error",
    verbose: bool = False,
) -> Stage1ScanActiveSet:
    if on_empty not in {"error", "nan"}:
        raise ValueError("on_empty must be 'error' or 'nan'.")

    delta_keep_db = _validate_nonnegative_float(delta_keep_db, "delta_keep_db")
    min_keep_core_side_per_scan = _validate_nonnegative_int(min_keep_core_side_per_scan, "min_keep_core_side_per_scan")
    min_keep_buffer_per_scan = _validate_nonnegative_int(min_keep_buffer_per_scan, "min_keep_buffer_per_scan")
    min_keep_outer_per_scan = _validate_nonnegative_int(min_keep_outer_per_scan, "min_keep_outer_per_scan")
    max_keep_core_side_per_scan = _validate_nonnegative_int(max_keep_core_side_per_scan, "max_keep_core_side_per_scan")
    max_keep_buffer_per_scan = _validate_nonnegative_int(max_keep_buffer_per_scan, "max_keep_buffer_per_scan")
    max_keep_outer_per_scan = _validate_nonnegative_int(max_keep_outer_per_scan, "max_keep_outer_per_scan")
    core_merge_alpha_deg = _validate_nonnegative_float(core_merge_alpha_deg, "core_merge_alpha_deg")
    buffer_merge_alpha_deg = _validate_nonnegative_float(buffer_merge_alpha_deg, "buffer_merge_alpha_deg")
    buffer_merge_eta_deg = _validate_nonnegative_float(buffer_merge_eta_deg, "buffer_merge_eta_deg")
    outer_merge_alpha_deg = _validate_nonnegative_float(outer_merge_alpha_deg, "outer_merge_alpha_deg")
    outer_merge_eta_deg = _validate_nonnegative_float(outer_merge_eta_deg, "outer_merge_eta_deg")
    max_keep_per_scan = _validate_nonnegative_int(max_keep_per_scan, "max_keep_per_scan")
    preclip_core_candidates = _validate_optional_nonnegative_int(preclip_core_candidates, "preclip_core_candidates")
    preclip_buffer_candidates = _validate_optional_nonnegative_int(preclip_buffer_candidates, "preclip_buffer_candidates")
    preclip_outer_candidates = _validate_optional_nonnegative_int(preclip_outer_candidates, "preclip_outer_candidates")

    zone_bounds = [
        ("core", min_keep_core_side_per_scan, max_keep_core_side_per_scan, preclip_core_candidates),
        ("buffer", min_keep_buffer_per_scan, max_keep_buffer_per_scan, preclip_buffer_candidates),
        ("outer", min_keep_outer_per_scan, max_keep_outer_per_scan, preclip_outer_candidates),
    ]
    for zone_name, min_keep, max_keep, preclip in zone_bounds:
        if max_keep < min_keep:
            raise ValueError("max_keep_{0} must be >= min_keep_{0}.".format(zone_name))
        if preclip is not None and preclip < min_keep:
            raise ValueError("preclip_{0}_candidates must be >= the corresponding min_keep.".format(zone_name))

    r = _validate_scan_ranking_input(ranking) if validate else ranking

    input_candidate_counts = {
        "core_side_1d": int(np.asarray(r.core_side_idx_sorted).reshape(-1).size),
        "buffer_side_2d": int(np.asarray(r.buffer_idx_sorted).reshape(-1).size),
        "outer_side_2d": int(np.asarray(r.outer_idx_sorted).reshape(-1).size),
    }
    candidate_shortage = {
        "core_side_1d": input_candidate_counts["core_side_1d"] < int(min_keep_core_side_per_scan),
        "buffer_side_2d": input_candidate_counts["buffer_side_2d"] < int(min_keep_buffer_per_scan),
        "outer_side_2d": input_candidate_counts["outer_side_2d"] < int(min_keep_outer_per_scan),
    }

    sum_min = int(min_keep_core_side_per_scan + min_keep_buffer_per_scan + min_keep_outer_per_scan)
    if max_keep_per_scan < sum_min:
        raise ValueError("max_keep_per_scan must be >= the sum of per-zone minimum keeps.")

    # Use the strongest side ratio across all zones as the scan-local reference.
    ratio_heads = []
    for arr in [r.core_side_ratio_db_sorted, r.buffer_ratio_db_sorted, r.outer_ratio_db_sorted]:
        if arr.size > 0:
            head = float(arr[0])
            if np.isnan(head) or np.isposinf(head):
                raise ValueError("invalid ratio head.")
            ratio_heads.append(head)
    if len(ratio_heads) == 0:
        if on_empty == "error":
            raise ValueError(f"theta_s={r.theta_s_deg:.6g}: no side candidates available.")
        empty = Stage1ScanActiveSet(
            theta_s_deg=float(r.theta_s_deg),
            zone_name=np.asarray([], dtype=object),
            idx_local=_empty_i64(),
            alpha_deg=_empty_f64(),
            eta_deg=_empty_f64(),
            theta_deg=_empty_f64(),
            phi_deg=_empty_f64(),
            val_side=_empty_f64(),
            ratio_db=_empty_f64(),
            n_core_side=0,
            n_buffer_side=0,
            n_outer_side=0,
            mandatory_mask=np.asarray([], dtype=bool),
            meta={
                "empty_reason": "no_side_candidates",
                "input_candidate_counts": input_candidate_counts,
                "candidate_shortage": candidate_shortage,
                "validated": bool(validate),
            },
        )
        return empty
    ref_best_db = max(ratio_heads)

    core_sel = _select_zone_core(
        idx_local_sorted=r.core_side_idx_sorted,
        val_sorted=r.core_side_val_sorted,
        ratio_sorted=r.core_side_ratio_db_sorted,
        alpha_sorted=r.core_side_alpha_deg_sorted,
        ref_best_db=ref_best_db,
        delta_keep_db=delta_keep_db,
        merge_alpha_deg=core_merge_alpha_deg,
        min_keep=min_keep_core_side_per_scan,
        max_keep=max_keep_core_side_per_scan,
        preclip_candidates=preclip_core_candidates,
    )
    buf_sel = _select_zone_2d(
        zone_name="buffer_side_2d",
        idx_local_sorted=r.buffer_idx_sorted,
        val_sorted=r.buffer_val_sorted,
        ratio_sorted=r.buffer_ratio_db_sorted,
        alpha_sorted=r.buffer_alpha_deg_sorted,
        eta_sorted=r.buffer_eta_deg_sorted,
        theta_sorted=r.buffer_theta_deg_sorted,
        phi_sorted=r.buffer_phi_deg_sorted,
        ref_best_db=ref_best_db,
        delta_keep_db=delta_keep_db,
        merge_alpha_deg=buffer_merge_alpha_deg,
        merge_eta_deg=buffer_merge_eta_deg,
        min_keep=min_keep_buffer_per_scan,
        max_keep=max_keep_buffer_per_scan,
        preclip_candidates=preclip_buffer_candidates,
    )
    out_sel = _select_zone_2d(
        zone_name="outer_side_2d",
        idx_local_sorted=r.outer_idx_sorted,
        val_sorted=r.outer_val_sorted,
        ratio_sorted=r.outer_ratio_db_sorted,
        alpha_sorted=r.outer_alpha_deg_sorted,
        eta_sorted=r.outer_eta_deg_sorted,
        theta_sorted=r.outer_theta_deg_sorted,
        phi_sorted=r.outer_phi_deg_sorted,
        ref_best_db=ref_best_db,
        delta_keep_db=delta_keep_db,
        merge_alpha_deg=outer_merge_alpha_deg,
        merge_eta_deg=outer_merge_eta_deg,
        min_keep=min_keep_outer_per_scan,
        max_keep=max_keep_outer_per_scan,
        preclip_candidates=preclip_outer_candidates,
    )

    scan_as = _combine_scan_zone_results(
        theta_s_deg=r.theta_s_deg,
        zone_results=[core_sel, buf_sel, out_sel],
        max_keep_per_scan=max_keep_per_scan,
    )

    meta = dict(r.meta)
    meta.update(scan_as.meta)
    meta.update(
        {
            "D_core": float(r.D_core),
            "psll_db": float(r.psll_db),
            "delta_keep_db": float(delta_keep_db),
            "ref_best_db": float(ref_best_db),
            "min_keep_core_side_per_scan": int(min_keep_core_side_per_scan),
            "min_keep_buffer_per_scan": int(min_keep_buffer_per_scan),
            "min_keep_outer_per_scan": int(min_keep_outer_per_scan),
            "max_keep_core_side_per_scan": int(max_keep_core_side_per_scan),
            "max_keep_buffer_per_scan": int(max_keep_buffer_per_scan),
            "max_keep_outer_per_scan": int(max_keep_outer_per_scan),
            "core_merge_alpha_deg": float(core_merge_alpha_deg),
            "buffer_merge_alpha_deg": float(buffer_merge_alpha_deg),
            "buffer_merge_eta_deg": float(buffer_merge_eta_deg),
            "outer_merge_alpha_deg": float(outer_merge_alpha_deg),
            "outer_merge_eta_deg": float(outer_merge_eta_deg),
            "preclip_core_candidates": None if preclip_core_candidates is None else int(preclip_core_candidates),
            "preclip_buffer_candidates": None if preclip_buffer_candidates is None else int(preclip_buffer_candidates),
            "preclip_outer_candidates": None if preclip_outer_candidates is None else int(preclip_outer_candidates),
            "input_candidate_counts": input_candidate_counts,
            "candidate_shortage": candidate_shortage,
            "validated": bool(validate),
        }
    )

    if verbose:
        print(f"=== ACTIVE_CONSTRAINTS SINGLE theta_s={r.theta_s_deg:.3f} deg ===")
        print(f"[ref] D_core={r.D_core:.6g} psll_db={r.psll_db:.6f} ref_best_db={ref_best_db:.6f}")
        print(f"[zone selected] core={scan_as.n_core_side} buffer={scan_as.n_buffer_side} outer={scan_as.n_outer_side}")
        print(f"[scan selected] M={scan_as.idx_local.size} mandatory={int(np.sum(scan_as.mandatory_mask))}")

    meta["selected_after_scan_cap"] = {
        "core_side_1d": int(scan_as.n_core_side),
        "buffer_side_2d": int(scan_as.n_buffer_side),
        "outer_side_2d": int(scan_as.n_outer_side),
        "total": int(scan_as.idx_local.size),
    }

    result = Stage1ScanActiveSet(
        theta_s_deg=scan_as.theta_s_deg,
        zone_name=scan_as.zone_name,
        idx_local=scan_as.idx_local,
        alpha_deg=scan_as.alpha_deg,
        eta_deg=scan_as.eta_deg,
        theta_deg=scan_as.theta_deg,
        phi_deg=scan_as.phi_deg,
        val_side=scan_as.val_side,
        ratio_db=scan_as.ratio_db,
        n_core_side=scan_as.n_core_side,
        n_buffer_side=scan_as.n_buffer_side,
        n_outer_side=scan_as.n_outer_side,
        mandatory_mask=scan_as.mandatory_mask,
        meta=meta,
    )
    _validate_scan_active_set_result(result)
    return result



def build_active_set_stage1_batch(
    rankings: Sequence[Stage1ScanRankingInput],
    *,
    delta_keep_db: float = 5.0,
    min_keep_core_side_per_scan: int = 15,
    min_keep_buffer_per_scan: int = 9,
    min_keep_outer_per_scan: int = 10,
    max_keep_core_side_per_scan: int = 20,
    max_keep_buffer_per_scan: int = 14,
    max_keep_outer_per_scan: int = 14,
    core_merge_alpha_deg: float = 0.5,
    buffer_merge_alpha_deg: float = 3.0,
    buffer_merge_eta_deg: float = 12.0,
    outer_merge_alpha_deg: float = 4.0,
    outer_merge_eta_deg: float = 15.0,
    max_keep_per_scan: int = 48,
    max_total_points: int = 550,
    preclip_core_candidates: Optional[int] = 200,
    preclip_buffer_candidates: Optional[int] = 200,
    preclip_outer_candidates: Optional[int] = 200,
    validate: bool = True,
    on_empty: str = "error",
    verbose: bool = False,
) -> Stage1BatchActiveSet:
    if len(rankings) == 0:
        raise ValueError("rankings must be non-empty.")
    max_total_points = _validate_nonnegative_int(max_total_points, "max_total_points")

    per_scan = [
        build_active_set_stage1_single(
            rk,
            delta_keep_db=delta_keep_db,
            min_keep_core_side_per_scan=min_keep_core_side_per_scan,
            min_keep_buffer_per_scan=min_keep_buffer_per_scan,
            min_keep_outer_per_scan=min_keep_outer_per_scan,
            max_keep_core_side_per_scan=max_keep_core_side_per_scan,
            max_keep_buffer_per_scan=max_keep_buffer_per_scan,
            max_keep_outer_per_scan=max_keep_outer_per_scan,
            core_merge_alpha_deg=core_merge_alpha_deg,
            buffer_merge_alpha_deg=buffer_merge_alpha_deg,
            buffer_merge_eta_deg=buffer_merge_eta_deg,
            outer_merge_alpha_deg=outer_merge_alpha_deg,
            outer_merge_eta_deg=outer_merge_eta_deg,
            max_keep_per_scan=max_keep_per_scan,
            preclip_core_candidates=preclip_core_candidates,
            preclip_buffer_candidates=preclip_buffer_candidates,
            preclip_outer_candidates=preclip_outer_candidates,
            validate=validate,
            on_empty=on_empty,
            verbose=verbose,
        )
        for rk in rankings
    ]

    # Global cap with mandatory-first retention.
    mandatory_rows: List[Tuple[int, int, float]] = []  # (scan_idx, local_row_idx, ratio_db)
    optional_rows: List[Tuple[int, int, float]] = []
    for s, scan_as in enumerate(per_scan):
        for i in range(scan_as.idx_local.size):
            row = (int(s), int(i), float(scan_as.ratio_db[i]))
            if bool(scan_as.mandatory_mask[i]):
                mandatory_rows.append(row)
            else:
                optional_rows.append(row)

    if len(mandatory_rows) > max_total_points:
        raise ValueError(
            f"mandatory active-set size ({len(mandatory_rows)}) exceeds max_total_points ({max_total_points})."
        )

    optional_rows.sort(key=lambda row: (-float(row[2]), int(row[0]), int(row[1])))
    keep_optional_n = max(0, int(max_total_points) - len(mandatory_rows))
    kept_optional = optional_rows[:keep_optional_n]
    keep_map: Dict[Tuple[int, int], None] = {}
    for s, i, _ in mandatory_rows + kept_optional:
        keep_map[(int(s), int(i))] = None

    capped_scans: List[Stage1ScanActiveSet] = []
    for s, scan_as in enumerate(per_scan):
        keep_rows = [i for i in range(scan_as.idx_local.size) if (s, i) in keep_map]
        keep_rows_arr = np.asarray(keep_rows, dtype=np.int64)
        capped_meta = dict(scan_as.meta)
        capped_meta["after_global_cap_M"] = int(keep_rows_arr.size)
        capped = Stage1ScanActiveSet(
            theta_s_deg=scan_as.theta_s_deg,
            zone_name=scan_as.zone_name[keep_rows_arr].copy(),
            idx_local=scan_as.idx_local[keep_rows_arr].copy(),
            alpha_deg=scan_as.alpha_deg[keep_rows_arr].copy(),
            eta_deg=scan_as.eta_deg[keep_rows_arr].copy(),
            theta_deg=scan_as.theta_deg[keep_rows_arr].copy(),
            phi_deg=scan_as.phi_deg[keep_rows_arr].copy(),
            val_side=scan_as.val_side[keep_rows_arr].copy(),
            ratio_db=scan_as.ratio_db[keep_rows_arr].copy(),
            n_core_side=int(np.sum(scan_as.zone_name[keep_rows_arr] == "core_side_1d")),
            n_buffer_side=int(np.sum(scan_as.zone_name[keep_rows_arr] == "buffer_side_2d")),
            n_outer_side=int(np.sum(scan_as.zone_name[keep_rows_arr] == "outer_side_2d")),
            mandatory_mask=scan_as.mandatory_mask[keep_rows_arr].copy(),
            meta=capped_meta,
        )
        _validate_scan_active_set_result(capped)
        capped_scans.append(capped)

    theta_s_deg = np.asarray([s.theta_s_deg for s in capped_scans], dtype=np.float64)
    per_scan_counts = np.asarray([s.idx_local.size for s in capped_scans], dtype=np.int64)
    total_M = int(np.sum(per_scan_counts))
    meta = {
        "delta_keep_db": float(delta_keep_db),
        "max_total_points": int(max_total_points),
        "mandatory_M_total": int(len(mandatory_rows)),
        "selected_M_total": int(total_M),
        "preclip_core_candidates": None if preclip_core_candidates is None else int(preclip_core_candidates),
        "preclip_buffer_candidates": None if preclip_buffer_candidates is None else int(preclip_buffer_candidates),
        "preclip_outer_candidates": None if preclip_outer_candidates is None else int(preclip_outer_candidates),
        "validated": bool(validate),
    }

    if verbose:
        print("=== ACTIVE_CONSTRAINTS BATCH ===")
        print(f"[global] mandatory={len(mandatory_rows)} total={total_M} cap={max_total_points}")
        print(f"[per_scan_counts] {per_scan_counts.tolist()}")

    result = Stage1BatchActiveSet(
        per_scan=tuple(capped_scans),
        theta_s_deg=theta_s_deg,
        per_scan_counts=per_scan_counts,
        total_M=total_M,
        meta=meta,
    )
    _validate_batch_active_set_result(result)
    return result


# -----------------------------------------------------------------------------
# Self tests
# -----------------------------------------------------------------------------


def _make_demo_ranking(theta_s_deg: float, core_n: int, buf_n: int, out_n: int, shift: float = 0.0) -> Stage1ScanRankingInput:
    core_idx = np.arange(core_n, dtype=np.int64)
    core_ratio = np.linspace(-10.0 + shift, -18.0 + shift, core_n, dtype=np.float64)
    core_val = 10.0 ** (core_ratio / 20.0)
    core_alpha = np.linspace(8.0, 25.0, core_n, dtype=np.float64)

    buf_idx = np.arange(buf_n, dtype=np.int64)
    buf_ratio = np.linspace(-8.0 + shift, -20.0 + shift, buf_n, dtype=np.float64)
    buf_val = 10.0 ** (buf_ratio / 20.0)
    buf_alpha = np.linspace(60.0, 90.0, buf_n, dtype=np.float64)
    buf_eta = np.mod(np.linspace(150.0, 230.0, buf_n, dtype=np.float64), 360.0)
    buf_theta = np.clip(buf_alpha - 20.0, 0.0, 90.0)
    buf_phi = np.mod(buf_eta, 360.0)

    out_idx = np.arange(out_n, dtype=np.int64)
    out_ratio = np.linspace(-9.0 + shift, -22.0 + shift, out_n, dtype=np.float64)
    out_val = 10.0 ** (out_ratio / 20.0)
    out_alpha = np.linspace(95.0, 140.0, out_n, dtype=np.float64) if out_n else _empty_f64()
    out_eta = np.mod(np.linspace(165.0, 205.0, out_n, dtype=np.float64), 360.0) if out_n else _empty_f64()
    out_theta = np.clip(180.0 - out_alpha, 0.0, 90.0) if out_n else _empty_f64()
    out_phi = np.mod(out_eta, 360.0) if out_n else _empty_f64()

    return Stage1ScanRankingInput(
        theta_s_deg=float(theta_s_deg),
        D_core=1.0,
        psll_db=float(max(core_ratio[0] if core_n else -np.inf, buf_ratio[0] if buf_n else -np.inf, out_ratio[0] if out_n else -np.inf)),
        core_side_idx_sorted=core_idx,
        core_side_val_sorted=core_val,
        core_side_ratio_db_sorted=core_ratio,
        core_side_alpha_deg_sorted=core_alpha,
        buffer_idx_sorted=buf_idx,
        buffer_val_sorted=buf_val,
        buffer_ratio_db_sorted=buf_ratio,
        buffer_alpha_deg_sorted=buf_alpha,
        buffer_eta_deg_sorted=buf_eta,
        buffer_theta_deg_sorted=buf_theta,
        buffer_phi_deg_sorted=buf_phi,
        outer_idx_sorted=out_idx,
        outer_val_sorted=out_val,
        outer_ratio_db_sorted=out_ratio,
        outer_alpha_deg_sorted=out_alpha,
        outer_eta_deg_sorted=out_eta,
        outer_theta_deg_sorted=out_theta,
        outer_phi_deg_sorted=out_phi,
        meta={},
    )


if __name__ == "__main__":
    print("=== ACTIVE_CONSTRAINTS SELF TESTS: START ===")
    rk1 = _make_demo_ranking(theta_s_deg=20.0, core_n=40, buf_n=30, out_n=25, shift=0.0)
    rk2 = _make_demo_ranking(theta_s_deg=55.0, core_n=45, buf_n=32, out_n=28, shift=-1.0)
    a1 = build_active_set_stage1_single(rk1, verbose=True)
    a2 = build_active_set_stage1_single(rk1, verbose=True)
    if not np.array_equal(a1.zone_name, a2.zone_name) or not np.array_equal(a1.idx_local, a2.idx_local):
        raise AssertionError("Determinism failed for single-scan active set.")
    print(f"- Determinism: PASS (M={a1.idx_local.size})")

    a_fast = build_active_set_stage1_single(rk1, validate=False, verbose=False)
    if not np.array_equal(a1.zone_name, a_fast.zone_name) or not np.array_equal(a1.idx_local, a_fast.idx_local):
        raise AssertionError("validate=False fast path changed the single-scan active set.")
    print("- validate=False single path: PASS")

    a_clip = build_active_set_stage1_single(
        rk1,
        preclip_core_candidates=25,
        preclip_buffer_candidates=20,
        preclip_outer_candidates=20,
        verbose=False,
    )
    if a_clip.idx_local.size > 48:
        raise AssertionError("Preclip single result exceeded max_keep_per_scan.")
    if "input_candidate_counts" not in a_clip.meta or "candidate_shortage" not in a_clip.meta:
        raise AssertionError("Active-set meta does not include candidate diagnostics.")
    print(f"- Preclip + diagnostics: PASS (M={a_clip.idx_local.size})")

    rk_short = _make_demo_ranking(theta_s_deg=10.0, core_n=3, buf_n=2, out_n=0, shift=0.0)
    a_short = build_active_set_stage1_single(
        rk_short,
        min_keep_core_side_per_scan=5,
        min_keep_buffer_per_scan=4,
        min_keep_outer_per_scan=3,
        max_keep_per_scan=20,
        on_empty="nan",
        verbose=False,
    )
    if not a_short.meta["candidate_shortage"]["core_side_1d"]:
        raise AssertionError("Core candidate shortage was not reported.")
    if not a_short.meta["candidate_shortage"]["outer_side_2d"]:
        raise AssertionError("Outer candidate shortage was not reported.")
    print("- Candidate shortage diagnostics: PASS")

    batch = build_active_set_stage1_batch([rk1, rk2], verbose=True)
    if batch.total_M > 550:
        raise AssertionError("Global cap failed.")
    print(f"- Global cap: PASS (M={batch.total_M})")
    batch_fast = build_active_set_stage1_batch([rk1, rk2], validate=False, verbose=False)
    if batch_fast.total_M != batch.total_M or not np.array_equal(batch_fast.per_scan_counts, batch.per_scan_counts):
        raise AssertionError("validate=False fast path changed the batch active set.")
    print("- validate=False batch path: PASS")
    print(f"- Per-scan counts: {batch.per_scan_counts.tolist()}")
    print("=== ACTIVE_CONSTRAINTS SELF TESTS: PASS ===")
