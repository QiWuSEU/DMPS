from __future__ import annotations

"""
passive_screening.py

Stage-2 screening module for the current Q0CRA/A0CRA flow.

This module is intentionally NOT a direct command-line runner.  Use
run_stage2_auto.py as the executable entry point.  The auto runner resolves
paths, generates the Stage-2 static point pool, creates an output directory,
and then calls run_stage2_screen_main(...).

Responsibilities in this module:
- Load fixed stage-1 geometry r*, N*.
- Load the externally generated Stage-2 static pool.
- Build Pool-A / Pool-C views for the current score_smain_2d and
  score_ar_psll_2d evaluators.
- Enumerate all-ring m_k candidates around N_k/4.
- Stage-1: score S_main + AR_main over the full Pool-A 3 dB main domain,
  using three delta anchors, then keep the best anchors under relative bands.
- Stage-2: score strict PSLL for Stage-1 survivors and keep topJ.
- Final: optionally build a shared dynamic patch from topJ worst points, run
  a coarse (c0,c1) grid and an optional controller-level local fine grid, then
  keep final_topJ seeds.

Non-responsibilities:
- No static pool generation here.
- No r-step / N-step / SOCP.
- No third-stage DE / 4-bit quantization / full-wave signoff.
"""

import csv
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from element_spectrum import iep_model as iep_io
from passive_preconditioning import passive_sampling_pool as p2
from passive_preconditioning.passive_smain_score import (
    SMain2DConfig,
    SMain2DEvaluator,
    SMainPoolView,
    SMainResult,
    build_pool_view_from_scan_point_dicts as build_smain_pool_view,
)
from passive_preconditioning.passive_ar_psll_score import (
    ARPSLL2DConfig,
    ARPSLL2DEvaluator,
    ARPSLLPoolView,
    ARFloor2DResult,
    ARMain2DResult,
    PSLL2DResult,
    ar_floor_result_to_dict,
    ar_main_result_to_dict,
    psll_result_to_dict,
    build_pool_view_from_scan_point_dicts as build_arpsll_pool_view,
)

try:
    from passive_preconditioning.passive_field_batch import Stage2BatchConfig, Stage2FieldBatchBackend
except Exception:  # pragma: no cover - handled at runtime with a clear error
    Stage2BatchConfig = None
    Stage2FieldBatchBackend = None

TWO_PI = 2.0 * math.pi
GOLDEN_GAMMA = math.pi * (3.0 - math.sqrt(5.0))


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage2ScreenConfig:
    stage2_topJ: int = 10
    final_topJ: int = 3

    # Candidate generation.  The default keeps every ring active but compresses
    # each ring to two physically distinct m candidates: a biased-quarter primary
    # and an alias-risk-complement / quarter-opposite secondary.
    candidate_mode: str = "two_per_ring_alias_complement"
    m_rule: str = "around_quarter_window"  # legacy alias retained for compatibility
    min_m: int = 1
    active_rings: str = "all"  # "all" or e.g. "1:8" in 1-based inclusive form
    max_designs_stage1: Optional[int] = None  # debug only; None = full enumeration
    stage1_to_stage2: int = 50

    delta_base: str = "golden"
    q_anchors: Tuple[float, ...] = (0.0, math.pi / 8.0, math.pi / 4.0)
    delta_reduce: str = "best"

    S_band: float = 0.01
    AR_band_db: float = 0.1

    enable_dynamic_final: bool = True
    dynamic_cap_total: int = 2000
    dynamic_main_rho_offsets: Tuple[float, ...] = (-0.006, 0.0, 0.006)
    dynamic_main_psi_offsets_deg: Tuple[float, ...] = (-6.0, 0.0, 6.0)
    dynamic_side_rho_offsets: Tuple[float, ...] = (-0.012, 0.0, 0.012)
    dynamic_side_psi_offsets_deg: Tuple[float, ...] = (-8.0, 0.0, 8.0)
    dynamic_quant_deg: float = 0.02

    final_c_grid: Tuple[float, ...] = (-math.pi / 16.0, -math.pi / 32.0, 0.0, math.pi / 32.0, math.pi / 16.0)
    final_fine_delta_grid: Tuple[float, ...] = (-math.pi / 64.0, 0.0, math.pi / 64.0)
    final_use_fine: bool = True

    # Batch backend.  Default path uses stage2_field_batch.py for S/AR/PSLL.
    # Legacy score_smain_2d.py / score_ar_psll_2d.py can be selected by setting
    # use_batch_backend=False, mainly for regression/debug.
    use_batch_backend: bool = True
    batch_stage1_chunk: int = 128
    batch_stage2_chunk: int = 8
    batch_final_chunk: int = 8
    batch_verbose: bool = False

    verbose: bool = True


@dataclass(frozen=True)
class Stage2ScreenResult:
    output_dir: str
    config_json_path: str
    inputs_json_path: str
    m_options_csv_path: str
    m_combos_npz_path: str
    stage1_pairs_csv_path: str
    stage1_survivors_csv_path: str
    stage2_psll_csv_path: str
    stage2_top10_csv_path: str
    final_delta_grid_csv_path: str
    final_top3_csv_path: str
    seeds_json_path: str
    seeds_npz_path: str
    report_json_path: str
    stage1_pair_count: int
    stage1_survivor_count: int
    stage2_scored_count: int
    stage2_topJ_count: int
    final_grid_count: int
    final_topJ_count: int
    best_final_psll_db: Optional[float]
    best_final_ar_main_db: Optional[float]
    best_final_smain: Optional[float]
    backend_A_summary_json_path: str = ""
    backend_C_summary_json_path: str = ""
    backend_Final_summary_json_path: str = ""


# -----------------------------------------------------------------------------
# JSON / CSV helpers
# -----------------------------------------------------------------------------

def _to_jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (Path,)):
        return str(x)
    if dataclasses.is_dataclass(x):
        return dataclasses.asdict(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_to_jsonable(dict(payload)), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for k in fieldnames:
                v = row.get(k, "")
                if isinstance(v, (list, tuple, dict, np.ndarray)):
                    out[k] = json.dumps(_to_jsonable(v), ensure_ascii=False)
                elif isinstance(v, (np.floating, np.integer)):
                    out[k] = v.item()
                else:
                    out[k] = v
            w.writerow(out)


def _wp_smain_to_dict(res: SMainResult) -> Optional[Dict[str, Any]]:
    w = res.worst
    if w is None:
        return None
    return {
        "theta_s_deg": float(w.theta_s_deg),
        "theta_deg": float(w.theta_deg),
        "phi_deg": float(w.phi_deg),
        "rho": float(w.rho),
        "psi_rad": float(w.psi_rad),
        "psi_deg": float(np.rad2deg(w.psi_rad)),
        "delta_id": int(w.delta_id),
        "ratio": float(w.ratio),
        "global_point_idx": int(w.global_point_idx),
        "local_point_idx": int(w.local_point_idx),
        "scan_idx": int(w.scan_idx),
    }


def _wp_arpsll_to_dict(wp: Any) -> Optional[Dict[str, Any]]:
    if wp is None:
        return None
    return {
        "theta_s_deg": float(wp.theta_s_deg),
        "theta_deg": float(wp.theta_deg),
        "phi_deg": float(wp.phi_deg),
        "rho": float(wp.rho),
        "psi_rad": float(wp.psi_rad),
        "psi_deg": float(np.rad2deg(wp.psi_rad)),
        "value": float(wp.value),
        "global_point_idx": int(wp.global_point_idx),
        "local_point_idx": int(wp.local_point_idx),
        "scan_idx": int(wp.scan_idx),
        "meta": dict(getattr(wp, "meta", {})),
    }


# -----------------------------------------------------------------------------
# Input loading / validation
# -----------------------------------------------------------------------------

def _load_npz_dict(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_stage1_geometry(npz_path: str | Path, json_path: str | Path) -> Dict[str, Any]:
    d = _load_npz_dict(npz_path)
    meta = _load_json(json_path)
    r_keys = ["r_best", "r_final", "r"]
    N_keys = ["N_best", "N_final", "N"]
    r = None
    N_raw = None
    r_key = None
    N_key = None
    for k in r_keys:
        if k in d:
            r = np.asarray(d[k], dtype=np.float64).reshape(-1)
            r_key = k
            break
    for k in N_keys:
        if k in d:
            N_raw = np.asarray(d[k], dtype=np.float64).reshape(-1)
            N_key = k
            break
    if r is None or N_raw is None:
        raise KeyError(f"stage1 npz missing r/N keys. keys={sorted(d.keys())}")
    if r.size == 0 or r.size != N_raw.size:
        raise ValueError(f"stage1 r/N shape mismatch: {r.shape}, {N_raw.shape}")
    if np.any(~np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("stage1 radii must contain positive finite values")
    if np.any(~np.isfinite(N_raw)):
        raise ValueError("stage1 N contains NaN/Inf")
    if not np.allclose(N_raw, np.rint(N_raw), atol=1e-12, rtol=0.0):
        raise ValueError("stage1 N must be integer-valued")
    N = np.rint(N_raw).astype(np.int64)
    if np.any(N <= 0):
        raise ValueError("stage1 N must contain positive integers")
    return {
        "npz": d,
        "json": meta,
        "r": r,
        "N": N,
        "K": int(r.size),
        "r_key": r_key,
        "N_key": N_key,
    }

def _check_boundary_pool_alignment(
    boundary_json_path: str | Path,
    *,
    pool_ctx: p2.Stage2PoolContext,
    K: int,
) -> Dict[str, Any]:
    """Verify that the reused/generated static pool matches the boundary source."""
    p = Path(boundary_json_path)
    if not p.exists():
        raise FileNotFoundError(f"boundary JSON not found: {p}")
    expected = p2.load_stage2_boundary_tables_from_json(
        str(p),
        pool_ctx.angle_sets,
        boundary_format="auto",
        rho_from_alpha="sin_alpha",
        K=int(K),
    )
    actual = pool_ctx.boundaries
    names = ("rho_3db_A", "rho_3db_B", "rho_3db_C", "rho_null_B", "rho_null_C")
    diffs: Dict[str, float] = {}
    for name in names:
        a = np.asarray(getattr(actual, name), dtype=np.float64).reshape(-1)
        e = np.asarray(getattr(expected, name), dtype=np.float64).reshape(-1)
        if a.shape != e.shape:
            raise ValueError(
                f"static pool boundary {name} shape {a.shape} does not match "
                f"boundary source shape {e.shape}"
            )
        diff = float(np.max(np.abs(a - e))) if a.size else 0.0
        diffs[name] = diff
        if not np.allclose(a, e, atol=1e-12, rtol=0.0):
            raise ValueError(
                f"static pool boundary {name} does not match boundary JSON; "
                f"max_abs_diff={diff:.6e}"
            )
    payload = _load_json(p)
    return {
        "path": str(p),
        "format": payload.get("format", None) if isinstance(payload, dict) else None,
        "K": int(K),
        "rho_from_alpha": "sin_alpha",
        "max_abs_diff": diffs,
        "aligned": True,
    }

def _validate_screen_config(config: Stage2ScreenConfig, *, K: int) -> None:
    if int(K) <= 0:
        raise ValueError("K must be positive")
    for name in ("stage2_topJ", "final_topJ", "stage1_to_stage2"):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(config.min_m) < 1:
        raise ValueError("min_m must be >= 1")
    if config.max_designs_stage1 is not None and int(config.max_designs_stage1) <= 0:
        raise ValueError("max_designs_stage1 must be None or positive")
    if not np.isfinite(float(config.S_band)) or float(config.S_band) < 0.0:
        raise ValueError("S_band must be finite and non-negative")
    if not np.isfinite(float(config.AR_band_db)) or float(config.AR_band_db) < 0.0:
        raise ValueError("AR_band_db must be finite and non-negative")
    for name in ("batch_stage1_chunk", "batch_stage2_chunk", "batch_final_chunk"):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(config.dynamic_cap_total) < 0:
        raise ValueError("dynamic_cap_total must be non-negative")
    if str(config.delta_base) != "golden":
        raise NotImplementedError("Only delta_base='golden' is implemented")
    if str(config.delta_reduce) != "best":
        raise NotImplementedError("Only delta_reduce='best' is implemented")

    q = np.asarray(config.q_anchors, dtype=np.float64).reshape(-1)
    if q.size == 0 or np.any(~np.isfinite(q)):
        raise ValueError("q_anchors must be non-empty finite values")
    coarse = np.asarray(config.final_c_grid, dtype=np.float64).reshape(-1)
    if coarse.size == 0 or np.any(~np.isfinite(coarse)):
        raise ValueError("final_c_grid must be non-empty finite values")
    if bool(config.final_use_fine):
        fine = np.asarray(config.final_fine_delta_grid, dtype=np.float64).reshape(-1)
        if fine.size == 0 or np.any(~np.isfinite(fine)):
            raise ValueError("final_fine_delta_grid must be non-empty finite values when final_use_fine=True")


# -----------------------------------------------------------------------------
# Pool views
# -----------------------------------------------------------------------------

def _points_by_pool(ctx: p2.Stage2PoolContext, pool: str, *, include_dynamic: bool = False) -> Dict[float, Dict[str, np.ndarray]]:
    pool = str(pool).upper()
    if pool == "A":
        return {float(t): p2.get_poolA_points_for_scan(ctx, float(t)) for t in np.asarray(ctx.angle_sets.theta_A_deg, dtype=np.float64).tolist()}
    if pool == "C":
        return p2.get_all_poolC_points(ctx, include_dynamic=include_dynamic)
    if pool == "B":
        return p2.get_all_poolB_points(ctx, include_dynamic=include_dynamic)
    raise ValueError(f"unknown pool {pool}")


def _build_smain_view_A(ctx: p2.Stage2PoolContext) -> SMainPoolView:
    scan_dicts = _points_by_pool(ctx, "A", include_dynamic=False)

    def main_mask(pts: Mapping[str, np.ndarray]) -> np.ndarray:
        zones = np.asarray(pts["zone_code"], dtype=np.int64).reshape(-1)
        return zones == int(p2.ZONE_MAIN)

    return build_smain_pool_view(scan_dicts, main_mask_fn=main_mask)

def _build_arpsll_view_A(ctx: p2.Stage2PoolContext) -> ARPSLLPoolView:
    scan_dicts = _points_by_pool(ctx, "A", include_dynamic=False)

    def main_mask(pts: Mapping[str, np.ndarray]) -> np.ndarray:
        zones = np.asarray(pts["zone_code"], dtype=np.int64).reshape(-1)
        return zones == int(p2.ZONE_MAIN)

    def side_mask(pts: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.zeros_like(np.asarray(pts["rho"], dtype=np.float64).reshape(-1), dtype=bool)

    return build_arpsll_pool_view(scan_dicts, main_mask_fn=main_mask, side_mask_fn=side_mask)

def _build_smain_view_C(ctx: p2.Stage2PoolContext, *, include_dynamic: bool = False) -> SMainPoolView:
    scan_dicts = _points_by_pool(ctx, "C", include_dynamic=include_dynamic)
    def main_mask(pts: Mapping[str, np.ndarray]) -> np.ndarray:
        tags = np.asarray(pts["tag_code"], dtype=np.int64).reshape(-1)
        zones = np.asarray(pts["zone_code"], dtype=np.int64).reshape(-1)
        mask = (tags == int(p2.TAG_C_MAIN)) | ((zones == int(p2.ZONE_MAIN)) & (np.asarray(pts.get("is_dynamic", np.zeros_like(zones)), dtype=np.int64).reshape(-1) == 1))
        if not np.any(mask):
            mask = zones == int(p2.ZONE_MAIN)
        return mask
    return build_smain_pool_view(scan_dicts, main_mask_fn=main_mask)


def _build_arpsll_view_C(ctx: p2.Stage2PoolContext, *, include_dynamic: bool = False) -> ARPSLLPoolView:
    scan_dicts = _points_by_pool(ctx, "C", include_dynamic=include_dynamic)
    def main_mask(pts: Mapping[str, np.ndarray]) -> np.ndarray:
        tags = np.asarray(pts["tag_code"], dtype=np.int64).reshape(-1)
        zones = np.asarray(pts["zone_code"], dtype=np.int64).reshape(-1)
        is_dyn = np.asarray(pts.get("is_dynamic", np.zeros_like(zones)), dtype=np.int64).reshape(-1)
        mask = (tags == int(p2.TAG_C_MAIN)) | ((zones == int(p2.ZONE_MAIN)) & (is_dyn == 1))
        if not np.any(mask):
            mask = zones == int(p2.ZONE_MAIN)
        return mask
    def side_mask(pts: Mapping[str, np.ndarray]) -> np.ndarray:
        zones = np.asarray(pts["zone_code"], dtype=np.int64).reshape(-1)
        return zones == int(p2.ZONE_SIDE)
    return build_arpsll_pool_view(scan_dicts, main_mask_fn=main_mask, side_mask_fn=side_mask)


# -----------------------------------------------------------------------------
# m / delta candidates
# -----------------------------------------------------------------------------

def _parse_active_rings(active_rings: str, K: int) -> np.ndarray:
    s = str(active_rings).strip().lower()
    if s in ("", "all", "*"):
        return np.arange(K, dtype=np.int64)
    if ":" in s:
        a, b = s.split(":", 1)
        lo = int(a)
        hi = int(b)
        if lo < 1 or hi > K or lo > hi:
            raise ValueError(f"invalid active_rings {active_rings!r} for K={K}; use 1-based inclusive range")
        return np.arange(lo - 1, hi, dtype=np.int64)
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    out = []
    for v in vals:
        if v < 1 or v > K:
            raise ValueError(f"active ring {v} out of 1..{K}")
        out.append(v - 1)
    return np.asarray(sorted(set(out)), dtype=np.int64)


def _is_legal_m(Nk: int, m: int, *, min_m: int = 1) -> bool:
    """Legal SR order constraint: m_k >= min_m and m_k < N_k/2."""
    Nk = int(Nk)
    m = int(m)
    return (m >= int(min_m)) and (m < (float(Nk) / 2.0))


def m_candidates_around_quarter(Nk: int, *, min_m: int = 1) -> List[int]:
    """Legacy/full local candidate set around N_k/4.

    This is the candidate pool from which the compressed two-candidate mode
    selects its primary/secondary representatives.
    """
    Nk = int(Nk)
    x = Nk / 4.0
    mL = int(math.floor(x))
    mU = int(math.ceil(x))
    raw = [mL - 1, mL, mL + 1, mU - 1, mU, mU + 1]
    keep: List[int] = []
    for m in raw:
        m = int(m)
        if not _is_legal_m(Nk, m, min_m=min_m):
            continue
        if m not in keep:
            keep.append(m)
    keep.sort()
    if not keep:
        raise ValueError(f"No legal m candidate for N={Nk}, min_m={min_m}")
    return keep


def _biased_quarter_primary(Nk: int, full_set: Sequence[int], *, min_m: int = 1) -> int:
    """Primary candidate using the agreed biased N/4 rounding rule.

    N/4 = n + f:
      f=0.00 -> n
      f=0.25 -> n-1
      f=0.50 -> n-1
      f=0.75 -> n+2
    """
    Nk = int(Nk)
    q = Nk // 4
    rem = Nk % 4
    if rem == 0:
        m0 = q
    elif rem == 1:
        m0 = q - 1
    elif rem == 2:
        m0 = q - 1
    else:
        m0 = q + 2
    full = [int(x) for x in full_set]
    if _is_legal_m(Nk, m0, min_m=min_m) and int(m0) in full:
        return int(m0)
    x = Nk / 4.0
    return int(min(full, key=lambda m: (abs(float(m) - x), int(m))))


def _has_zero_alias_risk(Nk: int, mk: int, q_alias: Sequence[int]) -> bool:
    """Whether exists nonzero q with N_k | q*m_k."""
    Nk = int(Nk)
    mk = int(mk)
    for q in np.asarray(q_alias, dtype=np.int64).reshape(-1).tolist():
        q = int(q)
        if q == 0:
            continue
        if (q * mk) % Nk == 0:
            return True
    return False


def _quarter_opposite_candidate(Nk: int, m0: int, full_set: Sequence[int], *, min_m: int = 1) -> Optional[int]:
    """Fallback secondary on the opposite side of the biased-quarter choice."""
    Nk = int(Nk)
    q = Nk // 4
    rem = Nk % 4
    if rem == 0:
        raw = [q - 1, q + 1]
    elif rem == 1:
        raw = [q + 1]
    elif rem == 2:
        raw = [q + 2, q]
    else:
        raw = [q]
    full = set(int(x) for x in full_set)
    for m in raw:
        m = int(m)
        if m != int(m0) and m in full and _is_legal_m(Nk, m, min_m=min_m):
            return int(m)
    return None


def _select_two_alias_complement_for_ring(Nk: int, q_alias: Sequence[int], *, min_m: int = 1) -> Dict[str, Any]:
    """Return two representative m candidates for one ring.

    The selected set is:
      1. biased-quarter primary;
      2. alias-risk complement from the local full set if available;
         otherwise quarter-opposite; otherwise nearest other.
    """
    Nk = int(Nk)
    full_set = m_candidates_around_quarter(Nk, min_m=min_m)
    m0 = _biased_quarter_primary(Nk, full_set, min_m=min_m)
    risk0 = _has_zero_alias_risk(Nk, m0, q_alias)
    others = [int(m) for m in full_set if int(m) != int(m0)]
    target_risk = not bool(risk0)
    complement = [m for m in others if _has_zero_alias_risk(Nk, m, q_alias) == target_risk]
    x = Nk / 4.0
    if complement:
        m1 = int(min(complement, key=lambda m: (abs(float(m) - x), abs(int(m) - int(m0)), int(m))))
        reason = "alias_complement"
    else:
        opp = _quarter_opposite_candidate(Nk, m0, full_set, min_m=min_m)
        if opp is not None:
            m1 = int(opp)
            reason = "quarter_opposite"
        elif others:
            m1 = int(min(others, key=lambda m: (abs(float(m) - x), abs(int(m) - int(m0)), int(m))))
            reason = "nearest_other"
        else:
            m1 = int(m0)
            reason = "single_candidate_only"
    selected = sorted(set([int(m0), int(m1)]))
    return {
        "N": int(Nk),
        "full_set": [int(x) for x in full_set],
        "selected": [int(x) for x in selected],
        "primary": int(m0),
        "secondary": int(m1),
        "primary_risk": bool(risk0),
        "secondary_risk": bool(_has_zero_alias_risk(Nk, m1, q_alias)),
        "secondary_reason": reason,
    }


def build_m_options(
    N: np.ndarray,
    *,
    active_rings: np.ndarray,
    min_m: int = 1,
    candidate_mode: str = "two_per_ring_alias_complement",
    q_alias: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    N = np.asarray(N, dtype=np.int64).reshape(-1)
    K = int(N.size)
    active_set = set(int(k) for k in np.asarray(active_rings, dtype=np.int64).reshape(-1).tolist())
    mode = str(candidate_mode or "two_per_ring_alias_complement")
    if mode == "m_rule":
        mode = "around_quarter_window"
    q_alias_arr = np.asarray([] if q_alias is None else q_alias, dtype=np.int64).reshape(-1)
    rows: List[Dict[str, Any]] = []
    opts_by_ring: List[np.ndarray] = []
    for k in range(K):
        if k not in active_set:
            opts = np.asarray([0], dtype=np.int64)
            row_extra: Dict[str, Any] = {
                "full_set": [],
                "primary": 0,
                "secondary": 0,
                "primary_risk": False,
                "secondary_risk": False,
                "secondary_reason": "inactive_ring",
            }
        elif mode == "two_per_ring_alias_complement":
            info = _select_two_alias_complement_for_ring(int(N[k]), q_alias_arr, min_m=min_m)
            opts = np.asarray(info["selected"], dtype=np.int64)
            row_extra = dict(info)
        elif mode == "around_quarter_window":
            full_set = m_candidates_around_quarter(int(N[k]), min_m=min_m)
            opts = np.asarray(full_set, dtype=np.int64)
            row_extra = {
                "full_set": [int(x) for x in full_set],
                "primary": int(full_set[0]),
                "secondary": int(full_set[-1]),
                "primary_risk": bool(_has_zero_alias_risk(int(N[k]), int(full_set[0]), q_alias_arr)),
                "secondary_risk": bool(_has_zero_alias_risk(int(N[k]), int(full_set[-1]), q_alias_arr)),
                "secondary_reason": "legacy_full_window",
            }
        else:
            raise ValueError(f"Unknown candidate_mode={candidate_mode!r}")
        opts_by_ring.append(opts)
        rows.append({
            "ring_idx0": int(k),
            "ring_idx1": int(k + 1),
            "N_k": int(N[k]),
            "active": bool(k in active_set),
            "candidate_mode": mode,
            "N_over_4": float(N[k] / 4.0),
            "m_candidates": opts.astype(int).tolist(),
            "n_candidates": int(opts.size),
            "full_set_json": row_extra.get("full_set", []),
            "primary": int(row_extra.get("primary", 0)),
            "secondary": int(row_extra.get("secondary", 0)),
            "primary_risk": bool(row_extra.get("primary_risk", False)),
            "secondary_risk": bool(row_extra.get("secondary_risk", False)),
            "secondary_reason": str(row_extra.get("secondary_reason", "")),
        })
    return rows, opts_by_ring

def enumerate_m_combos(opts_by_ring: Sequence[np.ndarray], *, max_designs: Optional[int] = None) -> np.ndarray:
    import itertools
    lists = [np.asarray(x, dtype=np.int64).reshape(-1).tolist() for x in opts_by_ring]
    combos_iter = itertools.product(*lists)
    if max_designs is None:
        combos = np.asarray(list(combos_iter), dtype=np.int64)
        if combos.ndim == 1:
            combos = combos.reshape(1, -1)
        return combos
    rows = []
    for i, combo in enumerate(combos_iter):
        if i >= int(max_designs):
            break
        rows.append(combo)
    return np.asarray(rows, dtype=np.int64)


def make_delta_gold(K: int, *, offset_rank: float = 0.0) -> np.ndarray:
    k = np.arange(int(K), dtype=np.float64)
    return np.mod((k + float(offset_rank)) * GOLDEN_GAMMA, TWO_PI)


def make_delta_with_anchor(delta_gold: np.ndarray, q_anchor: float) -> np.ndarray:
    return np.mod(np.asarray(delta_gold, dtype=np.float64).reshape(-1) + float(q_anchor), TWO_PI)


def _normalized_ring_axis(K: int) -> np.ndarray:
    if K <= 1:
        return np.zeros((K,), dtype=np.float64)
    return (-1.0 + 2.0 * np.arange(K, dtype=np.float64) / float(K - 1)).astype(np.float64)


# -----------------------------------------------------------------------------
# Evaluator row helpers
# -----------------------------------------------------------------------------

def _smain_row(prefix: str, res: SMainResult) -> Dict[str, Any]:
    wp = _wp_smain_to_dict(res)
    return {
        f"{prefix}_status": res.status,
        f"{prefix}_S_main": float(res.S_main),
        f"{prefix}_worst_scan_idx": int(res.idx_wc_scan),
        f"{prefix}_worst_theta_s_deg": None if wp is None else wp["theta_s_deg"],
        f"{prefix}_worst_theta_deg": None if wp is None else wp["theta_deg"],
        f"{prefix}_worst_phi_deg": None if wp is None else wp["phi_deg"],
        f"{prefix}_worst_rho": None if wp is None else wp["rho"],
        f"{prefix}_worst_point_json": wp,
    }


def _ar_row(prefix: str, res: ARMain2DResult) -> Dict[str, Any]:
    wp = _wp_arpsll_to_dict(res.worst_point)
    return {
        f"{prefix}_status": res.status,
        f"{prefix}_AR_main_db": float(res.ar_wc_db),
        f"{prefix}_eta_max": float(res.eta_max),
        f"{prefix}_bad_eta_count": int(res.bad_eta_count),
        f"{prefix}_worst_scan_idx": int(res.worst_scan_idx),
        f"{prefix}_worst_theta_s_deg": None if wp is None else wp["theta_s_deg"],
        f"{prefix}_worst_theta_deg": None if wp is None else wp["theta_deg"],
        f"{prefix}_worst_phi_deg": None if wp is None else wp["phi_deg"],
        f"{prefix}_worst_rho": None if wp is None else wp["rho"],
        f"{prefix}_worst_point_json": wp,
    }


def _psll_row(prefix: str, res: PSLL2DResult) -> Dict[str, Any]:
    wsp = _wp_arpsll_to_dict(res.worst_side_point)
    wmp = _wp_arpsll_to_dict(res.worst_main_point)
    return {
        f"{prefix}_status": res.status,
        f"{prefix}_PSLL_wc_db": float(res.psll_wc_db),
        f"{prefix}_worst_scan_idx": int(res.worst_scan_idx),
        f"{prefix}_worst_theta_s_deg": None if wsp is None else wsp["theta_s_deg"],
        f"{prefix}_worst_side_theta_deg": None if wsp is None else wsp["theta_deg"],
        f"{prefix}_worst_side_phi_deg": None if wsp is None else wsp["phi_deg"],
        f"{prefix}_worst_side_rho": None if wsp is None else wsp["rho"],
        f"{prefix}_worst_side_point_json": wsp,
        f"{prefix}_worst_main_point_json": wmp,
    }


# -----------------------------------------------------------------------------
# Dynamic point patch generation
# -----------------------------------------------------------------------------

def _local_to_global(theta_s_deg: float, phi_s_deg: float, rho: np.ndarray, psi_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta_s_rad = np.deg2rad(float(theta_s_deg))
    phi_s_rad = np.deg2rad(float(phi_s_deg))
    u_s = math.sin(theta_s_rad) * math.cos(phi_s_rad)
    v_s = math.sin(theta_s_rad) * math.sin(phi_s_rad)
    u = u_s + rho * np.cos(psi_rad)
    v = v_s + rho * np.sin(psi_rad)
    rr = np.sqrt(u * u + v * v)
    mask_clip = np.clip(rr, 0.0, 1.0)
    theta_rad = np.arcsin(mask_clip)
    phi_rad = np.mod(np.arctan2(v, u), TWO_PI)
    return np.rad2deg(theta_rad), np.rad2deg(phi_rad), u, v


def _empty_dyn_points() -> Dict[str, np.ndarray]:
    return {
        "scan_idx": np.zeros((0,), dtype=np.int32),
        "theta_s_deg": np.zeros((0,), dtype=np.float64),
        "phi_s_deg": np.zeros((0,), dtype=np.float64),
        "rho": np.zeros((0,), dtype=np.float64),
        "psi_deg": np.zeros((0,), dtype=np.float64),
        "theta_deg": np.zeros((0,), dtype=np.float64),
        "phi_deg": np.zeros((0,), dtype=np.float64),
        "u": np.zeros((0,), dtype=np.float64),
        "v": np.zeros((0,), dtype=np.float64),
        "zone_code": np.zeros((0,), dtype=np.int8),
        "tag_code": np.zeros((0,), dtype=np.int16),
        "layer_code": np.zeros((0,), dtype=np.int16),
        "priority": np.zeros((0,), dtype=np.int16),
        "is_dynamic": np.zeros((0,), dtype=np.int8),
    }


def _concat_dyn(parts: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    parts = [p for p in parts if int(np.asarray(p.get("rho", np.zeros((0,)))).size) > 0]
    if not parts:
        return _empty_dyn_points()
    keys = _empty_dyn_points().keys()
    return {k: np.concatenate([np.asarray(p[k]) for p in parts], axis=0) for k in keys}


def _dedup_dyn(pts: Dict[str, np.ndarray], *, quant_deg: float, cap: Optional[int] = None) -> Dict[str, np.ndarray]:
    n = int(np.asarray(pts["rho"]).size)
    if n == 0:
        return pts
    q = float(quant_deg)
    kth = np.round(np.asarray(pts["theta_deg"], dtype=np.float64) / q).astype(np.int64)
    kph = np.round(np.asarray(pts["phi_deg"], dtype=np.float64) / q).astype(np.int64)
    scan = np.asarray(pts["scan_idx"], dtype=np.int64)
    zone = np.asarray(pts["zone_code"], dtype=np.int64)
    pr = np.asarray(pts["priority"], dtype=np.int64)
    order = np.lexsort((kph, kth, zone, scan, pr))
    keep = np.ones((order.size,), dtype=bool)
    for ii in range(1, order.size):
        a = order[ii - 1]
        b = order[ii]
        same = scan[a] == scan[b] and zone[a] == zone[b] and kth[a] == kth[b] and kph[a] == kph[b]
        if same:
            keep[ii] = False
    idx = order[keep]
    if cap is not None and idx.size > int(cap):
        idx = idx[: int(cap)]
    return {k: np.asarray(v)[idx] for k, v in pts.items()}


def _make_patch_around_point(
    *,
    scan_idx: int,
    theta_s_deg: float,
    phi_s_deg: float,
    seed_rho: float,
    seed_psi_rad: float,
    rho_offsets: Sequence[float],
    psi_offsets_deg: Sequence[float],
    rho_min: float,
    rho_max: float,
    zone_code: int,
    tag_code: int,
    layer_code: int,
    priority: int,
) -> Dict[str, np.ndarray]:
    rr: List[float] = []
    pp: List[float] = []
    seed_psi_deg = float(np.rad2deg(seed_psi_rad))
    for dr in rho_offsets:
        r = float(seed_rho) + float(dr)
        for dp in psi_offsets_deg:
            rr.append(r)
            pp.append(seed_psi_deg + float(dp))
    rho = np.asarray(rr, dtype=np.float64)
    psi_deg = np.mod(np.asarray(pp, dtype=np.float64), 360.0)
    mask = (rho >= float(rho_min) - 1e-12) & (rho <= float(rho_max) + 1e-12)
    rho = rho[mask]
    psi_deg = psi_deg[mask]
    if rho.size == 0:
        return _empty_dyn_points()
    psi_rad = np.deg2rad(psi_deg)
    theta_deg, phi_deg, u, v = _local_to_global(theta_s_deg, phi_s_deg, rho, psi_rad)
    vis = (u * u + v * v) <= 1.0 + 1e-12
    rho = rho[vis]
    psi_deg = psi_deg[vis]
    theta_deg = theta_deg[vis]
    phi_deg = phi_deg[vis]
    u = u[vis]
    v = v[vis]
    n = int(rho.size)
    if n == 0:
        return _empty_dyn_points()
    return {
        "scan_idx": np.full((n,), int(scan_idx), dtype=np.int32),
        "theta_s_deg": np.full((n,), float(theta_s_deg), dtype=np.float64),
        "phi_s_deg": np.full((n,), float(phi_s_deg), dtype=np.float64),
        "rho": rho.astype(np.float64),
        "psi_deg": psi_deg.astype(np.float64),
        "theta_deg": theta_deg.astype(np.float64),
        "phi_deg": phi_deg.astype(np.float64),
        "u": u.astype(np.float64),
        "v": v.astype(np.float64),
        "zone_code": np.full((n,), int(zone_code), dtype=np.int8),
        "tag_code": np.full((n,), int(tag_code), dtype=np.int16),
        "layer_code": np.full((n,), int(layer_code), dtype=np.int16),
        "priority": np.full((n,), int(priority), dtype=np.int16),
        "is_dynamic": np.ones((n,), dtype=np.int8),
    }


def _attach_mixed_poolC_dynamic_points_for_scan(
    ctx: p2.Stage2PoolContext,
    theta_s_deg: float,
    points: Mapping[str, np.ndarray],
) -> None:
    """Attach the controller's mixed main/side final patch to Pool-C.

    The generic pool module's public attach helper is side-only because its
    native Pool-C tie-break builder creates side patches.  The Stage-2 final
    controller intentionally adds both main and side worst-point patches, so it
    validates and stores that mixed patch locally without changing the pool
    module's side-only contract.
    """
    theta_grid = np.asarray(ctx.angle_sets.theta_C_deg, dtype=np.float64).reshape(-1)
    hit = np.where(np.isclose(theta_grid, float(theta_s_deg), atol=float(ctx.tol_deg), rtol=0.0))[0]
    if hit.size != 1:
        raise KeyError(f"theta_s_deg={theta_s_deg} not uniquely found in Pool-C scan grid")
    scan_idx = int(hit[0])
    key = float(theta_grid[scan_idx])
    required = (
        "scan_idx", "theta_s_deg", "phi_s_deg", "rho", "psi_deg",
        "theta_deg", "phi_deg", "u", "v", "zone_code", "tag_code",
        "layer_code", "priority", "is_dynamic",
    )
    missing = [name for name in required if name not in points]
    if missing:
        raise KeyError(f"mixed Pool-C dynamic points missing fields: {missing}")
    arrays = {name: np.asarray(points[name]).reshape(-1).copy() for name in required}
    n = int(arrays["rho"].size)
    if any(arr.size != n for arr in arrays.values()):
        raise ValueError("mixed Pool-C dynamic point fields have inconsistent lengths")
    if n:
        for name in ("theta_s_deg", "phi_s_deg", "rho", "psi_deg", "theta_deg", "phi_deg", "u", "v"):
            if np.any(~np.isfinite(np.asarray(arrays[name], dtype=np.float64))):
                raise ValueError(f"mixed Pool-C dynamic field {name} contains NaN/Inf")
        if np.any(np.asarray(arrays["rho"], dtype=np.float64) < 0.0):
            raise ValueError("mixed Pool-C dynamic rho must be non-negative")
        if np.any(np.asarray(arrays["u"], dtype=np.float64) ** 2 + np.asarray(arrays["v"], dtype=np.float64) ** 2 > 1.0 + 1e-12):
            raise ValueError("mixed Pool-C dynamic points must lie in the visible u-v disk")
        zones = np.asarray(arrays["zone_code"], dtype=np.int64)
        if np.any((zones != int(p2.ZONE_MAIN)) & (zones != int(p2.ZONE_SIDE))):
            raise ValueError("mixed Pool-C dynamic zone_code must be main or side")
        if np.any(np.asarray(arrays["is_dynamic"], dtype=np.int64) != 1):
            raise ValueError("mixed Pool-C dynamic points require is_dynamic=1")
        if np.any(~np.isclose(np.asarray(arrays["theta_s_deg"], dtype=np.float64), key, atol=float(ctx.tol_deg), rtol=0.0)):
            raise ValueError("mixed Pool-C dynamic theta_s_deg does not match target scan")
    arrays["scan_idx"] = np.full((n,), scan_idx, dtype=np.int32)
    ctx.dynamic_poolC_by_scan[key] = arrays


def build_dynamic_patch_from_top_rows(
    top_rows: Sequence[Mapping[str, Any]],
    *,
    ctx: p2.Stage2PoolContext,
    config: Stage2ScreenConfig,
    out_npz_path: str,
    out_json_path: str,
) -> Dict[str, Any]:
    """Create a shared final dynamic patch from top-row worst points.

    This helper is deliberately conservative and simple: it builds small local
    patches around S/AR main worst points and PSLL side worst points, then
    attaches the union to ctx.dynamic_poolC_by_scan.
    """
    p2.clear_all_poolC_dynamic_points(ctx, verbose=False)
    rho3_by_theta = {float(t): float(r) for t, r in zip(np.asarray(ctx.angle_sets.theta_C_deg, dtype=np.float64), np.asarray(ctx.boundaries.rho_3db_C, dtype=np.float64))}
    rhon_by_theta = {float(t): float(r) for t, r in zip(np.asarray(ctx.angle_sets.theta_C_deg, dtype=np.float64), np.asarray(ctx.boundaries.rho_null_C, dtype=np.float64))}
    theta_to_scan = {float(t): i for i, t in enumerate(np.asarray(ctx.angle_sets.theta_C_deg, dtype=np.float64).tolist())}
    phi_s_deg = float(ctx.angle_sets.phi_s_deg)
    parts_by_theta: Dict[float, List[Dict[str, np.ndarray]]] = {}

    def add_point(wp: Optional[Mapping[str, Any]], *, domain: str) -> None:
        if not wp:
            return
        th = float(wp.get("theta_s_deg"))
        # snap to known C theta grid
        th_key = min(theta_to_scan.keys(), key=lambda x: abs(x - th))
        if abs(th_key - th) > 1e-6:
            return
        scan_idx = int(theta_to_scan[th_key])
        rho = float(wp.get("rho", 0.0))
        psi_rad = float(wp.get("psi_rad", math.radians(float(wp.get("psi_deg", 0.0)))))
        if domain == "main":
            part = _make_patch_around_point(
                scan_idx=scan_idx,
                theta_s_deg=th_key,
                phi_s_deg=phi_s_deg,
                seed_rho=rho,
                seed_psi_rad=psi_rad,
                rho_offsets=config.dynamic_main_rho_offsets,
                psi_offsets_deg=config.dynamic_main_psi_offsets_deg,
                rho_min=0.0,
                rho_max=rho3_by_theta[th_key],
                zone_code=p2.ZONE_MAIN,
                tag_code=p2.TAG_DYNAMIC_NEAR_PATCH,
                layer_code=p2.LAYER_C_DYNAMIC,
                priority=1,
            )
        else:
            part = _make_patch_around_point(
                scan_idx=scan_idx,
                theta_s_deg=th_key,
                phi_s_deg=phi_s_deg,
                seed_rho=rho,
                seed_psi_rad=psi_rad,
                rho_offsets=config.dynamic_side_rho_offsets,
                psi_offsets_deg=config.dynamic_side_psi_offsets_deg,
                rho_min=rhon_by_theta[th_key] + 1e-6,
                rho_max=min(float(ctx.summary.get("config", {}).get("rho_cap", 1.50)), 1.0 + math.sin(math.radians(th_key))),
                zone_code=p2.ZONE_SIDE,
                tag_code=p2.TAG_DYNAMIC_FAR_PATCH,
                layer_code=p2.LAYER_C_DYNAMIC,
                priority=2,
            )
        if int(np.asarray(part["rho"]).size) > 0:
            parts_by_theta.setdefault(th_key, []).append(part)

    for row in top_rows:
        add_point(row.get("S_worst_point_json"), domain="main")
        add_point(row.get("AR_worst_point_json"), domain="main")
        add_point(row.get("PSLL_worst_side_point_json"), domain="side")

    attached_counts: Dict[str, int] = {}
    all_parts: List[Dict[str, np.ndarray]] = []
    for th, parts in sorted(parts_by_theta.items()):
        pts = _concat_dyn(parts)
        pts = _dedup_dyn(pts, quant_deg=float(config.dynamic_quant_deg), cap=None)
        attached_counts[f"{th:.9g}"] = int(np.asarray(pts["rho"]).size)
        if int(np.asarray(pts["rho"]).size) > 0:
            _attach_mixed_poolC_dynamic_points_for_scan(ctx, th, pts)
            all_parts.append(pts)
    all_dyn_before_cap = _concat_dyn(all_parts)
    points_before_cap = int(np.asarray(all_dyn_before_cap["rho"]).size)
    all_dyn = _dedup_dyn(
        all_dyn_before_cap,
        quant_deg=float(config.dynamic_quant_deg),
        cap=int(config.dynamic_cap_total),
    )
    points_after_cap = int(np.asarray(all_dyn["rho"]).size)

    # Re-attach cap-limited dynamic points per scan.
    p2.clear_all_poolC_dynamic_points(ctx, verbose=False)
    for th in np.unique(np.asarray(all_dyn["theta_s_deg"], dtype=np.float64)).tolist():
        mask = np.isclose(np.asarray(all_dyn["theta_s_deg"], dtype=np.float64), float(th), atol=1e-9)
        pts_th = {k: np.asarray(v)[mask] for k, v in all_dyn.items()}
        _attach_mixed_poolC_dynamic_points_for_scan(ctx, float(th), pts_th)

    Path(out_npz_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz_path, **all_dyn)
    attached_counts = {}
    theta_final = np.asarray(all_dyn["theta_s_deg"], dtype=np.float64)
    for th in np.unique(theta_final).tolist():
        attached_counts[f"{float(th):.9g}"] = int(np.count_nonzero(np.isclose(theta_final, float(th), atol=1e-9)))
    summary = {
        "enabled": True,
        "source_topJ": int(len(top_rows)),
        "total_dynamic_points": int(np.asarray(all_dyn["rho"]).size),
        "n_main": int(np.count_nonzero(np.asarray(all_dyn["zone_code"], dtype=np.int64) == int(p2.ZONE_MAIN))),
        "n_side": int(np.count_nonzero(np.asarray(all_dyn["zone_code"], dtype=np.int64) == int(p2.ZONE_SIDE))),
        "attached_counts_by_theta": attached_counts,
        "points_before_cap": int(points_before_cap),
        "points_after_cap": int(points_after_cap),
        "cap_applied": bool(points_before_cap > points_after_cap),
        "npz_path": str(out_npz_path),
    }
    _write_json(out_json_path, summary)
    return summary



# -----------------------------------------------------------------------------
# Batch backend helpers
# -----------------------------------------------------------------------------

def _write_progress(out_dir: Path, payload: Mapping[str, Any]) -> None:
    """Write a small heartbeat JSON so long runs are inspectable."""
    try:
        _write_json(out_dir / "stage2_progress.json", dict(payload))
    except Exception:
        # Progress writing must never kill the optimization run.
        pass


def _make_m_options_by_ring(opts_by_ring: Sequence[np.ndarray]) -> Dict[int, np.ndarray]:
    return {int(k): np.asarray(v, dtype=np.int64).reshape(-1) for k, v in enumerate(opts_by_ring)}


def _require_batch_backend() -> None:
    if Stage2BatchConfig is None or Stage2FieldBatchBackend is None:
        raise RuntimeError(
            "Stage2ScreenConfig.use_batch_backend=True but stage2_field_batch.py "
            "could not be imported. Put stage2_field_batch.py in the project root "
            "or run with --no-batch-backend."
        )


def _make_batch_backend(
    *,
    name: str,
    pool_view: ARPSLLPoolView,
    iep_model: iep_io.IEPModel,
    r: np.ndarray,
    N: np.ndarray,
    m_options_by_ring: Mapping[int, np.ndarray],
    config: Stage2ScreenConfig,
    out_summary_path: str,
) -> Any:
    _require_batch_backend()
    bcfg = Stage2BatchConfig(
        beta=2.0 * np.pi,
        eps=1e-12,
        B_alias_co=2,
        B_alias_cx=2,
        use_active_q_only=True,
        stage1_chunk_size=int(config.batch_stage1_chunk),
        stage2_chunk_size=int(config.batch_stage2_chunk),
        final_chunk_size=int(config.batch_final_chunk),
        verbose=bool(config.batch_verbose),
    )
    if config.verbose:
        print(f"[Stage2][{name}] precompute START")
    t0 = time.perf_counter()
    backend = Stage2FieldBatchBackend(
        pool_view=pool_view,
        iep_model=iep_model,
        r=np.asarray(r, dtype=np.float64),
        N=np.asarray(N, dtype=np.int64),
        m_options_by_ring=m_options_by_ring,
        config=bcfg,
    )
    backend.precompute_all()
    summary = backend.summary()
    _write_json(out_summary_path, summary)
    if config.verbose:
        print(
            f"[Stage2][{name}] precompute DONE "
            f"P={summary.get('P')} P_main={summary.get('P_main')} P_side={summary.get('P_side')} "
            f"S={summary.get('S')} q_a={len(summary.get('q_a_non0', []))} "
            f"q_c={len(summary.get('q_c_non0', []))} "
            f"basis_co={summary.get('basis_count_co')} basis_cx={summary.get('basis_count_cx')} "
            f"mem={float(summary.get('memory_estimate_mb', 0.0)):.2f} MB "
            f"elapsed={time.perf_counter()-t0:.2f}s"
        )
    return backend


def _apply_final_row_diagnostics(
    row: Dict[str, Any],
    *,
    ar_floor_db: float,
    ar_floor_bad_eta_count: int,
    ar_floor_eta_max: float,
    ar_floor_worst_point_json: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Refresh final-row diagnostic fields from the final Pool-C evaluation.

    Final rows inherit Stage-1/Stage-2 columns from their base rows.  Metrics are
    recomputed on the final Pool-C view, so every scalar worst-scan field and the
    AR-floor worst-point record must be refreshed from that same final view.
    This helper changes output metadata only; it does not affect feasibility,
    ranking, or any field calculation.
    """
    floor_wp = None if ar_floor_worst_point_json is None else dict(ar_floor_worst_point_json)
    row["AR_floor_db"] = float(ar_floor_db)
    row["AR_floor_bad_eta_count"] = int(ar_floor_bad_eta_count)
    row["AR_floor_eta_max"] = float(ar_floor_eta_max)
    row["AR_floor_worst_point_json"] = floor_wp
    row["AR_floor_worst_scan_idx"] = -1 if floor_wp is None else int(floor_wp["scan_idx"])
    row["AR_floor_worst_theta_s_deg"] = None if floor_wp is None else float(floor_wp["theta_s_deg"])
    row["AR_floor_worst_theta_deg"] = None if floor_wp is None else float(floor_wp["theta_deg"])
    row["AR_floor_worst_phi_deg"] = None if floor_wp is None else float(floor_wp["phi_deg"])
    row["AR_floor_worst_rho"] = None if floor_wp is None else float(floor_wp["rho"])

    s_wp = row.get("S_worst_point_json")
    if isinstance(s_wp, Mapping):
        row["S_worst_scan_idx"] = int(s_wp["scan_idx"])
        row["S_worst_theta_s_deg"] = float(s_wp["theta_s_deg"])
        row["S_worst_theta_deg"] = float(s_wp["theta_deg"])
        row["S_worst_phi_deg"] = float(s_wp["phi_deg"])
        row["S_worst_rho"] = float(s_wp["rho"])

    ar_wp = row.get("AR_worst_point_json")
    if isinstance(ar_wp, Mapping):
        row["AR_worst_scan_idx"] = int(ar_wp["scan_idx"])
        row["AR_worst_theta_s_deg"] = float(ar_wp["theta_s_deg"])
        row["AR_worst_theta_deg"] = float(ar_wp["theta_deg"])
        row["AR_worst_phi_deg"] = float(ar_wp["phi_deg"])
        row["AR_worst_rho"] = float(ar_wp["rho"])

    psll_side_wp = row.get("PSLL_worst_side_point_json")
    if isinstance(psll_side_wp, Mapping):
        row["PSLL_worst_scan_idx"] = int(psll_side_wp["scan_idx"])
        row["PSLL_worst_theta_s_deg"] = float(psll_side_wp["theta_s_deg"])
        row["PSLL_worst_side_theta_deg"] = float(psll_side_wp["theta_deg"])
        row["PSLL_worst_side_phi_deg"] = float(psll_side_wp["phi_deg"])
        row["PSLL_worst_side_rho"] = float(psll_side_wp["rho"])

    row["AR_delta_to_floor_db"] = float(float(row["AR_main_db"]) - float(ar_floor_db))
    return row


def _normalize_final_batch_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float],
    ar_floor_db: Optional[float] = None,
    ar_floor_bad_eta_count: Optional[int] = None,
    ar_floor_eta_max: Optional[float] = None,
    ar_floor_worst_point_json: Optional[Mapping[str, Any]] = None,
    ar_bad_value_db: float = 300.0,
) -> List[Dict[str, Any]]:
    """Normalize coarse backend rows to the controller's final-row contract."""
    S_thr = float(thresholds["S_threshold"])
    AR_thr = float(thresholds["AR_threshold_db"])
    out: List[Dict[str, Any]] = []
    floor_db_default = float("nan") if ar_floor_db is None else float(ar_floor_db)
    floor_bad_default = 0 if ar_floor_bad_eta_count is None else int(ar_floor_bad_eta_count)
    floor_eta_default = float("nan") if ar_floor_eta_max is None else float(ar_floor_eta_max)
    for r0 in rows:
        r = dict(r0)
        if "delta_pre_json" not in r and "delta_json" in r:
            r["delta_pre_json"] = list(r["delta_json"])
        if "delta_json" not in r and "delta_pre_json" in r:
            r["delta_json"] = list(r["delta_pre_json"])
        r.setdefault("grid_stage", "coarse")
        r.setdefault("base_top_rank", int(r.get("final_candidate_id", -1)) + 1)

        s = float(r.get("S_main", np.nan))
        ar = float(r.get("AR_main_db", np.nan))
        psll = float(r.get("PSLL_wc_db", np.nan))
        S_status = "OK" if np.isfinite(s) else "INVALID_METRIC"
        if not np.isfinite(ar):
            AR_status = "INVALID_METRIC"
        elif ar >= float(ar_bad_value_db) - 1e-9:
            AR_status = "HANDEDNESS_FLIP"
        else:
            AR_status = "OK"
        PSLL_status = "OK" if np.isfinite(psll) else "INVALID_METRIC"
        status_ok = S_status == "OK" and AR_status == "OK" and PSLL_status == "OK"
        pass_S = bool(S_status == "OK" and s <= S_thr)
        pass_AR = bool(AR_status == "OK" and ar <= AR_thr)
        feasible = bool(status_ok and pass_S and pass_AR)
        penalty = (
            max(0.0, s - S_thr) * 1000.0
            + max(0.0, ar - AR_thr) * 10.0
            + (psll if np.isfinite(psll) else 1e6)
        )
        r.update({
            "S_status": S_status,
            "AR_status": AR_status,
            "PSLL_status": PSLL_status,
            "status_ok": bool(status_ok),
            "pass_S": pass_S,
            "pass_AR": pass_AR,
            "feasible": feasible,
            "penalty": float(penalty),
        })
        _apply_final_row_diagnostics(
            r,
            ar_floor_db=floor_db_default,
            ar_floor_bad_eta_count=floor_bad_default,
            ar_floor_eta_max=floor_eta_default,
            ar_floor_worst_point_json=ar_floor_worst_point_json,
        )
        out.append(r)
    return out

def _score_stage1_pairs_batch(
    *,
    backend_A: Any,
    m_combos: np.ndarray,
    delta_gold: np.ndarray,
    q_anchors: Sequence[float],
    config: Stage2ScreenConfig,
) -> List[Dict[str, Any]]:
    t0 = time.perf_counter()
    if config.verbose:
        print(
            f"[Stage2][Stage1-Batch] START designs={int(np.asarray(m_combos).shape[0])} "
            f"q_anchors={len(q_anchors)} pairs={int(np.asarray(m_combos).shape[0]) * len(q_anchors)} "
            f"chunk={int(config.batch_stage1_chunk)}"
        )
    res = backend_A.score_stage1_s_ar(
        m_combos=m_combos,
        delta_gold=delta_gold,
        q_anchors=q_anchors,
        chunk_size=int(config.batch_stage1_chunk),
        verbose=bool(config.verbose or config.batch_verbose),
    )
    rows: List[Dict[str, Any]] = []
    for r0 in res.rows:
        r = dict(r0)
        s = float(r.get("S_main", np.nan))
        ar = float(r.get("AR_main_db", np.nan))
        bad_eta = int(r.get("AR_bad_eta_count", 0))
        S_status = "OK" if np.isfinite(s) else "INVALID_METRIC"
        if bad_eta > 0:
            AR_status = "HANDEDNESS_FLIP"
        elif not np.isfinite(ar):
            AR_status = "INVALID_METRIC"
        else:
            AR_status = "OK"
        r["S_status"] = S_status
        r["AR_status"] = AR_status
        r["status"] = "OK" if S_status == "OK" and AR_status == "OK" else "PARTIAL"
        rows.append(r)
    if config.verbose:
        valid_s = [float(r["S_main"]) for r in rows if str(r.get("status")) == "OK"]
        valid_ar = [float(r["AR_main_db"]) for r in rows if str(r.get("status")) == "OK"]
        s_star = min(valid_s) if valid_s else float("nan")
        ar_star = min(valid_ar) if valid_ar else float("nan")
        print(
            f"[Stage2][Stage1-Batch] DONE rows={len(rows)} "
            f"S_star={s_star:.6g} AR_star={ar_star:.4f} dB "
            f"elapsed={time.perf_counter()-t0:.2f}s"
        )
    return rows

def _score_stage2_psll_batch(
    *,
    backend_C: Any,
    survivors: Sequence[Mapping[str, Any]],
    config: Stage2ScreenConfig,
) -> List[Dict[str, Any]]:
    t0 = time.perf_counter()
    if config.verbose:
        print(f"[Stage2][Stage2-Batch] START survivors={len(survivors)} chunk={int(config.batch_stage2_chunk)}")
    res = backend_C.score_psll_batch(
        rows=survivors,
        chunk_size=int(config.batch_stage2_chunk),
        verbose=bool(config.verbose or config.batch_verbose),
    )
    rows: List[Dict[str, Any]] = []
    for r0 in res.rows:
        r = dict(r0)
        psll = float(r.get("PSLL_wc_db", np.nan))
        r["PSLL_status"] = "OK" if np.isfinite(psll) else "INVALID_METRIC"
        rows.append(r)
    rows.sort(key=lambda r: (
        float(r.get("PSLL_wc_db", np.inf)),
        float(r.get("S_main", np.inf)),
        float(r.get("AR_main_db", np.inf)),
        int(r.get("m_id", -1)),
    ))
    if config.verbose:
        best = None if not rows else rows[0].get("PSLL_wc_db")
        print(f"[Stage2][Stage2-Batch] DONE scored={len(rows)} best_PSLL={best} elapsed={time.perf_counter()-t0:.2f}s")
    return rows

def _final_row_sort_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    feasible = bool(row.get("feasible", False))
    if feasible:
        return (
            0,
            float(row.get("PSLL_wc_db", np.inf)),
            float(row.get("S_main", np.inf)),
            float(row.get("AR_main_db", np.inf)),
            float(row.get("c0_star", 0.0)),
            float(row.get("c1_star", 0.0)),
        )
    return (
        1,
        float(row.get("penalty", np.inf)),
        float(row.get("PSLL_wc_db", np.inf)),
        float(row.get("S_main", np.inf)),
        float(row.get("AR_main_db", np.inf)),
    )


def _best_final_rows_by_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_candidates: int,
) -> List[Dict[str, Any]]:
    best: List[Dict[str, Any]] = []
    for cid in range(int(n_candidates)):
        local = [dict(r) for r in rows if int(r.get("final_candidate_id", -1)) == cid]
        if not local:
            raise RuntimeError(f"Final grid produced no rows for candidate {cid}")
        best.append(sorted(local, key=_final_row_sort_key)[0])
    best.sort(key=_final_row_sort_key)
    return best


def _eval_final_point_reference(
    *,
    smain_eval: SMain2DEvaluator,
    arpsll_eval: ARPSLL2DEvaluator,
    base_row: Mapping[str, Any],
    final_candidate_id: int,
    base_top_rank: int,
    grid_id: int,
    grid_stage: str,
    c0: float,
    c1: float,
    delta: np.ndarray,
    thresholds: Mapping[str, float],
    ar_floor_result: ARFloor2DResult,
    coarse_parent_grid_id: Optional[int] = None,
) -> Dict[str, Any]:
    m = np.asarray(base_row["m_json"], dtype=np.int64)
    sres = smain_eval.score_single_delta(m=m, delta=delta)
    ares = arpsll_eval.score_ar_main(m=m, delta=delta)
    pres = arpsll_eval.score_psll(m=m, delta=delta)
    status_ok = sres.status == "OK" and ares.status == "OK" and pres.status == "OK"
    pass_S = bool(sres.status == "OK" and float(sres.S_main) <= float(thresholds["S_threshold"]))
    pass_AR = bool(ares.status == "OK" and float(ares.ar_wc_db) <= float(thresholds["AR_threshold_db"]))
    feasible = bool(status_ok and pass_S and pass_AR)
    penalty = (
        max(0.0, float(sres.S_main) - float(thresholds["S_threshold"])) * 1000.0
        + max(0.0, float(ares.ar_wc_db) - float(thresholds["AR_threshold_db"])) * 10.0
        + (float(pres.psll_wc_db) if np.isfinite(float(pres.psll_wc_db)) else 1e6)
    )
    row = dict(base_row)
    row.update({
        "final_candidate_id": int(final_candidate_id),
        "base_stage2_id": int(base_row.get("stage2_id", -1)),
        "base_top_rank": int(base_top_rank),
        "grid_id": int(grid_id),
        "grid_stage": str(grid_stage),
        "coarse_parent_grid_id": None if coarse_parent_grid_id is None else int(coarse_parent_grid_id),
        "m_json": m.astype(int).tolist(),
        "c0_star": float(c0),
        "c1_star": float(c1),
        "delta_json": np.asarray(delta, dtype=np.float64).tolist(),
        "delta_pre_json": np.asarray(delta, dtype=np.float64).tolist(),
        "S_main": float(sres.S_main),
        "AR_floor_db": float(ar_floor_result.ar_floor_wc_db),
        "AR_main_db": float(ares.ar_wc_db),
        "PSLL_wc_db": float(pres.psll_wc_db),
        "pass_S": pass_S,
        "pass_AR": pass_AR,
        "status_ok": bool(status_ok),
        "feasible": feasible,
        "penalty": float(penalty),
        "S_status": sres.status,
        "AR_status": ares.status,
        "PSLL_status": pres.status,
        "S_worst_point_json": _wp_smain_to_dict(sres),
        "AR_worst_point_json": _wp_arpsll_to_dict(ares.worst_point),
        "PSLL_worst_side_point_json": _wp_arpsll_to_dict(pres.worst_side_point),
        "PSLL_worst_main_point_json": _wp_arpsll_to_dict(pres.worst_main_point),
    })
    _apply_final_row_diagnostics(
        row,
        ar_floor_db=float(ar_floor_result.ar_floor_wc_db),
        ar_floor_bad_eta_count=int(ar_floor_result.bad_eta_count),
        ar_floor_eta_max=float(ar_floor_result.eta_floor_max),
        ar_floor_worst_point_json=_wp_arpsll_to_dict(ar_floor_result.worst_point),
    )
    return row


def _run_final_fine_from_coarse(
    *,
    smain_eval: SMain2DEvaluator,
    arpsll_eval: ARPSLL2DEvaluator,
    top_rows: Sequence[Mapping[str, Any]],
    coarse_rows: Sequence[Mapping[str, Any]],
    delta_gold: np.ndarray,
    thresholds: Mapping[str, float],
    config: Stage2ScreenConfig,
) -> List[Dict[str, Any]]:
    K = int(np.asarray(delta_gold).size)
    axis = _normalized_ring_axis(K)
    coarse_best_by_cid = {
        cid: sorted(
            [dict(r) for r in coarse_rows if int(r.get("final_candidate_id", -1)) == cid],
            key=_final_row_sort_key,
        )[0]
        for cid in range(len(top_rows))
    }
    fine_rows: List[Dict[str, Any]] = []
    ar_floor_result = arpsll_eval.score_ar_floor_main()
    fine_axis = tuple(float(x) for x in config.final_fine_delta_grid)
    coarse_count = len(tuple(config.final_c_grid)) ** 2
    for cid, base in enumerate(top_rows):
        parent = coarse_best_by_cid[cid]
        c0_center = float(parent["c0_star"])
        c1_center = float(parent["c1_star"])
        q_anchor = float(base.get("q_anchor_rad", 0.0))
        for i0, dc0 in enumerate(fine_axis):
            for i1, dc1 in enumerate(fine_axis):
                c0 = c0_center + dc0
                c1 = c1_center + dc1
                delta = np.mod(np.asarray(delta_gold, dtype=np.float64) + q_anchor + c0 + c1 * axis, TWO_PI)
                fine_rows.append(_eval_final_point_reference(
                    smain_eval=smain_eval,
                    arpsll_eval=arpsll_eval,
                    base_row=base,
                    final_candidate_id=cid,
                    base_top_rank=cid + 1,
                    grid_id=coarse_count + i0 * len(fine_axis) + i1,
                    grid_stage="fine",
                    c0=c0,
                    c1=c1,
                    delta=delta,
                    thresholds=thresholds,
                    ar_floor_result=ar_floor_result,
                    coarse_parent_grid_id=int(parent.get("grid_id", -1)),
                ))
    return fine_rows


def _run_final_grid_batch(
    *,
    backend_Final: Any,
    top_rows: Sequence[Mapping[str, Any]],
    delta_gold: np.ndarray,
    thresholds: Mapping[str, float],
    config: Stage2ScreenConfig,
    smain_fine_eval: Optional[SMain2DEvaluator] = None,
    arpsll_fine_eval: Optional[ARPSLL2DEvaluator] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    t0 = time.perf_counter()
    if config.verbose:
        print(
            f"[Stage2][Final-Batch] START top_rows={len(top_rows)} "
            f"grid={len(config.final_c_grid)}x{len(config.final_c_grid)} "
            f"chunk={int(config.batch_final_chunk)}"
        )
    coarse_rows_raw, _ = backend_Final.score_final_grid_batch(
        top_rows=top_rows,
        delta_gold=delta_gold,
        c_grid=config.final_c_grid,
        thresholds=thresholds,
        fine_delta_grid=None,
        chunk_size=int(config.batch_final_chunk),
        verbose=bool(config.verbose or config.batch_verbose),
    )
    (
        ar_floor_db,
        _ar_floor_scan_values,
        ar_floor_bad_eta_count,
        ar_floor_eta_max,
        ar_floor_worst_point_json,
    ) = backend_Final._reduce_ar_floor()
    coarse_rows = _normalize_final_batch_rows(
        coarse_rows_raw,
        thresholds=thresholds,
        ar_floor_db=float(ar_floor_db),
        ar_floor_bad_eta_count=int(ar_floor_bad_eta_count),
        ar_floor_eta_max=float(ar_floor_eta_max),
        ar_floor_worst_point_json=ar_floor_worst_point_json,
    )
    fine_rows: List[Dict[str, Any]] = []
    if bool(config.final_use_fine):
        if smain_fine_eval is None or arpsll_fine_eval is None:
            raise RuntimeError("Final fine refinement requires reference S/AR/PSLL evaluators")
        fine_rows = _run_final_fine_from_coarse(
            smain_eval=smain_fine_eval,
            arpsll_eval=arpsll_fine_eval,
            top_rows=top_rows,
            coarse_rows=coarse_rows,
            delta_gold=delta_gold,
            thresholds=thresholds,
            config=config,
        )
    grid_rows = coarse_rows + fine_rows
    best_rows = _best_final_rows_by_candidate(grid_rows, n_candidates=len(top_rows))
    if config.verbose:
        best = None if not best_rows else best_rows[0].get("PSLL_wc_db")
        print(
            f"[Stage2][Final-Batch] DONE coarse_rows={len(coarse_rows)} "
            f"fine_rows={len(fine_rows)} best_rows={len(best_rows)} "
            f"best_PSLL={best} elapsed={time.perf_counter()-t0:.2f}s"
        )
    return grid_rows, best_rows

# -----------------------------------------------------------------------------
# Core pipeline
# -----------------------------------------------------------------------------

def _make_iep_model(unit_npz_path: str, unit_json_path: Optional[str]) -> iep_io.IEPModel:
    cfg = iep_io.IEPModelConfig(phi_main_rad=0.0, bounds="clamp", enable_bilinear_phi=True)
    return iep_io.IEPModel(unit_npz_path, meta_json_path=None if not unit_json_path else unit_json_path, config=cfg)


def _stage1_composite_score(row: Mapping[str, Any], thresholds: Mapping[str, float], config: Stage2ScreenConfig) -> float:
    eps = 1e-12
    s_part = (float(row["S_main"]) - float(thresholds["S_star"])) / (float(config.S_band) + eps)
    ar_part = (float(row["AR_main_db"]) - float(thresholds["AR_star_db"])) / (float(config.AR_band_db) + eps)
    return float(s_part + ar_part)


def _select_stage1_survivors(pair_rows: List[Dict[str, Any]], config: Stage2ScreenConfig) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Apply S/AR pass bands, then select at most stage1_to_stage2 by composite score.

    Composite is evaluated only on passing pairs, then each m_id keeps its best
    q-anchor.  If the pass set is empty, fall back to global composite top-N and
    mark the fallback in thresholds.
    """
    valid = [r for r in pair_rows if str(r.get("status", "")) == "OK" and np.isfinite(float(r["S_main"])) and np.isfinite(float(r["AR_main_db"]))]
    if not valid:
        raise RuntimeError("Stage-1 produced no valid S/AR rows")
    S_star = min(float(r["S_main"]) for r in valid)
    C_s = [r for r in valid if float(r["S_main"]) <= S_star + float(config.S_band)]
    if not C_s:
        C_s = sorted(valid, key=lambda r: float(r["S_main"]))[: max(1, min(10, len(valid)))]
    AR_star = min(float(r["AR_main_db"]) for r in C_s)
    C_sa = [r for r in C_s if float(r["AR_main_db"]) <= AR_star + float(config.AR_band_db)]
    fallback_global = False
    if not C_sa:
        C_sa = list(valid)
        fallback_global = True
    thresholds: Dict[str, float] = {
        "S_star": float(S_star),
        "AR_star_db": float(AR_star),
        "S_threshold": float(S_star + config.S_band),
        "AR_threshold_db": float(AR_star + config.AR_band_db),
        "stage1_valid_pair_count": float(len(valid)),
        "stage1_pass_pair_count_before_unique_m": float(len(C_sa)),
        "stage1_fallback_global_composite": float(1 if fallback_global else 0),
        "stage1_to_stage2_cap": float(max(1, int(config.stage1_to_stage2))),
    }
    for r in C_sa:
        r["pass_stage1"] = True if not fallback_global else False
        r["stage1_composite"] = _stage1_composite_score(r, thresholds, config)
    if fallback_global:
        for r in valid:
            r["stage1_composite"] = _stage1_composite_score(r, thresholds, config)
        source_rows = valid
    else:
        source_rows = C_sa
    # For every m, keep best q by composite.  Ties go to lower S, lower AR.
    by_m: Dict[int, Dict[str, Any]] = {}
    for r in source_rows:
        mid = int(r["m_id"])
        old = by_m.get(mid)
        if old is None or (
            float(r["stage1_composite"]), float(r["S_main"]), float(r["AR_main_db"]), float(r["q_anchor_rad"])
        ) < (
            float(old["stage1_composite"]), float(old["S_main"]), float(old["AR_main_db"]), float(old["q_anchor_rad"])
        ):
            by_m[mid] = dict(r)
    survivors = list(by_m.values())
    survivors.sort(key=lambda r: (float(r["stage1_composite"]), float(r["S_main"]), float(r["AR_main_db"]), int(r["m_id"])))
    cap = max(1, int(config.stage1_to_stage2))
    survivors = survivors[:cap]
    for rank, r in enumerate(survivors, start=1):
        r["stage1_to_stage2_rank"] = int(rank)
        r["survivor_rank_key"] = [float(r["stage1_composite"]), float(r["S_main"]), float(r["AR_main_db"])]
    thresholds["stage1_unique_m_before_cap"] = float(len(by_m))
    thresholds["stage1_selected_to_stage2"] = float(len(survivors))
    return survivors, thresholds

def _score_stage1_pairs(
    *,
    smain_eval: SMain2DEvaluator,
    ar_eval: ARPSLL2DEvaluator,
    m_combos: np.ndarray,
    delta_gold: np.ndarray,
    q_anchors: Sequence[float],
) -> List[Dict[str, Any]]:
    ar_floor = ar_eval.score_ar_floor_main()
    ar_floor_db = float(ar_floor.ar_floor_wc_db)
    rows: List[Dict[str, Any]] = []
    pair_id = 0
    for mid, m in enumerate(np.asarray(m_combos, dtype=np.int64)):
        for qid, q in enumerate(q_anchors):
            delta = make_delta_with_anchor(delta_gold, float(q))
            sres = smain_eval.score_single_delta(m=m, delta=delta)
            ares = ar_eval.score_ar_main(m=m, delta=delta)
            status = "OK" if sres.status == "OK" and ares.status == "OK" else "PARTIAL"
            row: Dict[str, Any] = {
                "pair_id": int(pair_id),
                "m_id": int(mid),
                "q_id": int(qid),
                "q_anchor_rad": float(q),
                "q_anchor_pi_units": float(q / math.pi),
                "m_json": m.astype(int).tolist(),
                "delta_json": delta.astype(float).tolist(),
                "status": status,
                "S_main": float(sres.S_main),
                "AR_floor_db": ar_floor_db,
                "AR_main_db": float(ares.ar_wc_db),
                "AR_delta_to_floor_db": float(ares.ar_wc_db - ar_floor_db),
                "S_worst_point_json": _wp_smain_to_dict(sres),
                "AR_worst_point_json": _wp_arpsll_to_dict(ares.worst_point),
            }
            row.update(_smain_row("S", sres))
            row.update(_ar_row("AR", ares))
            rows.append(row)
            pair_id += 1
    return rows


def _score_stage2_psll(
    *,
    psll_eval: ARPSLL2DEvaluator,
    survivors: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sid, row in enumerate(survivors):
        m = np.asarray(row["m_json"], dtype=np.int64)
        delta = np.asarray(row["delta_json"], dtype=np.float64)
        pres = psll_eval.score_psll(m=m, delta=delta)
        out = dict(row)
        out["stage2_id"] = int(sid)
        out["PSLL_wc_db"] = float(pres.psll_wc_db)
        out["PSLL_status"] = pres.status
        out["PSLL_worst_side_point_json"] = _wp_arpsll_to_dict(pres.worst_side_point)
        out["PSLL_worst_main_point_json"] = _wp_arpsll_to_dict(pres.worst_main_point)
        out.update(_psll_row("PSLL", pres))
        rows.append(out)
    rows.sort(key=lambda r: (float(r.get("PSLL_wc_db", np.inf)), float(r.get("S_main", np.inf)), float(r.get("AR_main_db", np.inf))))
    return rows


def _select_stage2_top_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    topJ: int,
) -> List[Dict[str, Any]]:
    valid = [
        dict(r)
        for r in rows
        if str(r.get("PSLL_status", "OK")) == "OK"
        and np.isfinite(float(r.get("PSLL_wc_db", np.nan)))
    ]
    if not valid:
        raise RuntimeError("Stage2 produced no valid PSLL rows")
    valid.sort(key=lambda r: (
        float(r.get("PSLL_wc_db", np.inf)),
        float(r.get("S_main", np.inf)),
        float(r.get("AR_main_db", np.inf)),
        int(r.get("m_id", -1)),
    ))
    out = valid[: max(1, int(topJ))]
    for rank, r in enumerate(out, start=1):
        r["stage2_top_rank"] = int(rank)
    return out


def _final_delta_candidates(delta_gold: np.ndarray, q_anchor: float, K: int, config: Stage2ScreenConfig) -> List[Tuple[float, float, np.ndarray, str]]:
    axis = _normalized_ring_axis(K)
    out: List[Tuple[float, float, np.ndarray, str]] = []
    seen = set()
    for c0 in config.final_c_grid:
        for c1 in config.final_c_grid:
            delta = np.mod(delta_gold + float(q_anchor) + float(c0) + float(c1) * axis, TWO_PI)
            key = tuple(np.round(delta, 12).tolist())
            if key not in seen:
                seen.add(key)
                out.append((float(c0), float(c1), delta, "coarse"))
    return out


def _run_final_grid(
    *,
    smain_eval: SMain2DEvaluator,
    arpsll_eval: ARPSLL2DEvaluator,
    top_rows: Sequence[Mapping[str, Any]],
    delta_gold: np.ndarray,
    thresholds: Mapping[str, float],
    config: Stage2ScreenConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    K = int(delta_gold.size)
    ar_floor = arpsll_eval.score_ar_floor_main()
    ar_floor_db = float(ar_floor.ar_floor_wc_db)
    grid_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    def eval_one(base_row: Mapping[str, Any], c0: float, c1: float, delta: np.ndarray, stage: str, grid_id: int) -> Dict[str, Any]:
        m = np.asarray(base_row["m_json"], dtype=np.int64)
        sres = smain_eval.score_single_delta(m=m, delta=delta)
        ares = arpsll_eval.score_ar_main(m=m, delta=delta)
        pres = arpsll_eval.score_psll(m=m, delta=delta)
        status_ok = sres.status == "OK" and ares.status == "OK" and pres.status == "OK"
        pass_S = bool(sres.status == "OK" and float(sres.S_main) <= float(thresholds["S_threshold"]))
        pass_AR = bool(ares.status == "OK" and float(ares.ar_wc_db) <= float(thresholds["AR_threshold_db"]))
        penalty = max(0.0, float(sres.S_main) - float(thresholds["S_threshold"])) * 1000.0 + max(0.0, float(ares.ar_wc_db) - float(thresholds["AR_threshold_db"])) * 10.0 + float(pres.psll_wc_db)
        row: Dict[str, Any] = {
            "final_candidate_id": int(base_row.get("stage2_top_rank", 1)) - 1,
            "base_stage2_id": int(base_row.get("stage2_id", -1)),
            "m_id": int(base_row.get("m_id", -1)),
            "grid_id": int(grid_id),
            "grid_stage": stage,
            "m_json": m.astype(int).tolist(),
            "q_anchor_rad": float(base_row.get("q_anchor_rad", 0.0)),
            "c0_star": float(c0),
            "c1_star": float(c1),
            "delta_json": np.asarray(delta, dtype=float).tolist(),
            "delta_pre_json": np.asarray(delta, dtype=float).tolist(),
            "S_main": float(sres.S_main),
            "AR_floor_db": ar_floor_db,
            "AR_main_db": float(ares.ar_wc_db),
            "AR_delta_to_floor_db": float(ares.ar_wc_db - ar_floor_db),
            "PSLL_wc_db": float(pres.psll_wc_db),
            "pass_S": pass_S,
            "pass_AR": pass_AR,
            "status_ok": bool(status_ok),
            "feasible": bool(status_ok and pass_S and pass_AR),
            "penalty": float(penalty),
            "S_worst_point_json": _wp_smain_to_dict(sres),
            "AR_worst_point_json": _wp_arpsll_to_dict(ares.worst_point),
            "PSLL_worst_side_point_json": _wp_arpsll_to_dict(pres.worst_side_point),
            "PSLL_worst_main_point_json": _wp_arpsll_to_dict(pres.worst_main_point),
            "S_status": sres.status,
            "AR_status": ares.status,
            "PSLL_status": pres.status,
        }
        _apply_final_row_diagnostics(
            row,
            ar_floor_db=float(ar_floor.ar_floor_wc_db),
            ar_floor_bad_eta_count=int(ar_floor.bad_eta_count),
            ar_floor_eta_max=float(ar_floor.eta_floor_max),
            ar_floor_worst_point_json=_wp_arpsll_to_dict(ar_floor.worst_point),
        )
        return row

    for base_idx, base in enumerate(top_rows):
        candidates = _final_delta_candidates(delta_gold, float(base.get("q_anchor_rad", 0.0)), K, config)
        coarse_rows: List[Dict[str, Any]] = []
        for gid, (c0, c1, delta, stage) in enumerate(candidates):
            row = eval_one(base, c0, c1, delta, stage, gid)
            row["base_top_rank"] = int(base_idx + 1)
            coarse_rows.append(row)
            grid_rows.append(row)
        feasible = [r for r in coarse_rows if r["feasible"]]
        best_coarse = sorted(feasible if feasible else coarse_rows, key=lambda r: (0 if r["feasible"] else 1, float(r["PSLL_wc_db"]) if r["feasible"] else float(r["penalty"]))) [0]
        best_pool = list(coarse_rows)
        if config.final_use_fine:
            fine_start_id = len(coarse_rows)
            for i0, dc0 in enumerate(config.final_fine_delta_grid):
                for i1, dc1 in enumerate(config.final_fine_delta_grid):
                    c0 = float(best_coarse["c0_star"]) + float(dc0)
                    c1 = float(best_coarse["c1_star"]) + float(dc1)
                    axis = _normalized_ring_axis(K)
                    delta = np.mod(delta_gold + float(base.get("q_anchor_rad", 0.0)) + c0 + c1 * axis, TWO_PI)
                    row = eval_one(base, c0, c1, delta, "fine", fine_start_id + i0 * len(config.final_fine_delta_grid) + i1)
                    row["base_top_rank"] = int(base_idx + 1)
                    grid_rows.append(row)
                    best_pool.append(row)
        feasible2 = [r for r in best_pool if r["feasible"]]
        best = sorted(feasible2 if feasible2 else best_pool, key=lambda r: (0 if r["feasible"] else 1, float(r["PSLL_wc_db"]) if r["feasible"] else float(r["penalty"]))) [0]
        best_rows.append(best)
    best_rows.sort(key=lambda r: (0 if r["feasible"] else 1, float(r["PSLL_wc_db"]) if r["feasible"] else float(r["penalty"]), float(r["S_main"]), float(r["AR_main_db"])))
    return grid_rows, best_rows


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------

def run_stage2_screen_main(
    *,
    stage1_npz_path: str,
    stage1_json_path: str,
    unit_npz_path: str,
    unit_json_path: Optional[str],
    mainlobe_table_json_path: str,
    static_pool_npz_path: str,
    static_pool_json_path: str,
    output_dir: str,
    config: Stage2ScreenConfig = Stage2ScreenConfig(),
) -> Stage2ScreenResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1 = _load_stage1_geometry(stage1_npz_path, stage1_json_path)
    r = stage1["r"]
    N = stage1["N"]
    K = int(stage1["K"])
    _validate_screen_config(config, K=K)

    pool_ctx = p2.load_stage2_pool_context(static_pool_npz_path, static_pool_json_path, verbose=bool(config.verbose))
    boundary_alignment = _check_boundary_pool_alignment(
        mainlobe_table_json_path,
        pool_ctx=pool_ctx,
        K=K,
    )
    if abs(float(pool_ctx.angle_sets.phi_s_deg)) > 1e-12:
        raise NotImplementedError("Current formal Stage-2 flow supports phi_s_deg=0 only")
    iep_model = _make_iep_model(unit_npz_path, unit_json_path)
    q_a = np.asarray([] if getattr(iep_model, "aq_q_list", None) is None else iep_model.aq_q_list, dtype=np.int64).reshape(-1)
    q_c = np.asarray([] if getattr(iep_model, "cq_q_list", None) is None else iep_model.cq_q_list, dtype=np.int64).reshape(-1)
    q_alias = np.unique(np.concatenate([q_a[q_a != 0], q_c[q_c != 0]])).astype(np.int64)
    if q_alias.size == 0:
        raise RuntimeError("IEP contains no active nonzero q orders; Stage2 sequence-rotation search is undefined")

    # Save config / resolved inputs early.
    config_path = str(out_dir / "stage2_config.json")
    inputs_path = str(out_dir / "stage2_inputs_resolved.json")
    _write_json(config_path, dataclasses.asdict(config))
    _write_json(inputs_path, {
        "stage1_npz_path": str(stage1_npz_path),
        "stage1_json_path": str(stage1_json_path),
        "unit_npz_path": str(unit_npz_path),
        "unit_json_path": None if unit_json_path is None else str(unit_json_path),
        "mainlobe_table_json_path": str(mainlobe_table_json_path),
        "static_pool_npz_path": str(static_pool_npz_path),
        "static_pool_json_path": str(static_pool_json_path),
        "K": int(K),
        "r": r.tolist(),
        "N": N.astype(int).tolist(),
        "q_alias": q_alias.astype(int).tolist(),
        "boundary_alignment": boundary_alignment,
        "pool_summary_excerpt": {
            "angle_sets": pool_ctx.summary.get("angle_sets", {}),
            "counts_A_first": pool_ctx.summary.get("counts_A", [])[:2],
            "counts_C_first": pool_ctx.summary.get("counts_C", [])[:2],
        },
    })

    # Build static views and evaluators.
    viewA_s = _build_smain_view_A(pool_ctx)
    viewA_ar = _build_arpsll_view_A(pool_ctx)
    viewC_s = _build_smain_view_C(pool_ctx, include_dynamic=False)
    viewC_ar = _build_arpsll_view_C(pool_ctx, include_dynamic=False)

    # Legacy evaluators are only constructed when explicitly requested.
    smain_A = None
    ar_A = None
    psll_C = None
    if not bool(config.use_batch_backend):
        smain_A = SMain2DEvaluator(pool_view=viewA_s, iep_model=iep_model, r=r, N=N, config=SMain2DConfig(verbose=False))
        ar_A = ARPSLL2DEvaluator(pool_view=viewA_ar, iep_model=iep_model, r=r, N=N, config=ARPSLL2DConfig(verbose=False))
        psll_C = ARPSLL2DEvaluator(pool_view=viewC_ar, iep_model=iep_model, r=r, N=N, config=ARPSLL2DConfig(verbose=False))

    active_rings = _parse_active_rings(config.active_rings, K)
    m_option_rows, opts_by_ring = build_m_options(
        N,
        active_rings=active_rings,
        min_m=int(config.min_m),
        candidate_mode=str(config.candidate_mode),
        q_alias=q_alias,
    )
    m_combos = enumerate_m_combos(opts_by_ring, max_designs=config.max_designs_stage1)
    delta_gold = make_delta_gold(K)

    m_options_csv = str(out_dir / "stage2_m_options.csv")
    m_combos_npz = str(out_dir / "stage2_m_combos.npz")
    backend_A_summary_json = str(out_dir / "stage2_backend_A_summary.json")
    backend_C_summary_json = str(out_dir / "stage2_backend_C_summary.json")
    backend_Final_summary_json = str(out_dir / "stage2_backend_Final_summary.json")
    _write_csv(m_options_csv, m_option_rows)
    np.savez(m_combos_npz, m_combos=m_combos, N=N, active_rings=active_rings, delta_gold=delta_gold, q_alias=q_alias, candidate_mode=np.asarray([str(config.candidate_mode)]))

    if config.verbose:
        print(f"[Stage2] K={K} candidate_mode={config.candidate_mode} designs={m_combos.shape[0]} q_anchors={len(config.q_anchors)} stage1_pairs={m_combos.shape[0] * len(config.q_anchors)} stage1_to_stage2={config.stage1_to_stage2}")
        print(f"[Stage2][backend] use_batch_backend={config.use_batch_backend} stage1_chunk={config.batch_stage1_chunk} stage2_chunk={config.batch_stage2_chunk} final_chunk={config.batch_final_chunk}")
        print(f"[Stage2] q_alias={q_alias.astype(int).tolist()}")
        for row in m_option_rows:
            print(f"[Stage2][m-options] ring={row['ring_idx1']} N={row['N_k']} selected={row['m_candidates']} full={row['full_set_json']} primary={row['primary']} secondary={row['secondary']} risk={row['primary_risk']}/{row['secondary_risk']} reason={row['secondary_reason']}")
    _write_progress(out_dir, {
        "stage": "m_candidates_ready",
        "candidate_mode": str(config.candidate_mode),
        "use_batch_backend": bool(config.use_batch_backend),
        "designs": int(m_combos.shape[0]),
        "stage1_pairs": int(m_combos.shape[0] * len(config.q_anchors)),
        "stage1_to_stage2": int(config.stage1_to_stage2),
    })

    # Stage-1 S/AR.
    m_options_by_ring = _make_m_options_by_ring(opts_by_ring)
    backend_A = None
    backend_C = None
    backend_Final = None
    if bool(config.use_batch_backend):
        backend_A = _make_batch_backend(
            name="Batch-A",
            pool_view=viewA_ar,
            iep_model=iep_model,
            r=r,
            N=N,
            m_options_by_ring=m_options_by_ring,
            config=config,
            out_summary_path=backend_A_summary_json,
        )
        pair_rows = _score_stage1_pairs_batch(
            backend_A=backend_A,
            m_combos=m_combos,
            delta_gold=delta_gold,
            q_anchors=config.q_anchors,
            config=config,
        )
    else:
        pair_rows = _score_stage1_pairs(
            smain_eval=smain_A,
            ar_eval=ar_A,
            m_combos=m_combos,
            delta_gold=delta_gold,
            q_anchors=config.q_anchors,
        )
    stage1_pairs_csv = str(out_dir / "stage2_stage1_smain_ar_pairs.csv")
    _write_csv(stage1_pairs_csv, pair_rows)
    survivors, thresholds = _select_stage1_survivors(pair_rows, config)
    stage1_survivors_csv = str(out_dir / "stage2_stage1_survivors.csv")
    _write_csv(stage1_survivors_csv, survivors)
    if config.verbose:
        print(f"[Stage2][Stage1] pairs={len(pair_rows)} survivors={len(survivors)} thresholds={thresholds}")
    _write_progress(out_dir, {
        "stage": "stage1_done",
        "pairs": int(len(pair_rows)),
        "survivors": int(len(survivors)),
        "thresholds": thresholds,
    })

    # Stage-2 PSLL on static Pool-C, topJ.
    if bool(config.use_batch_backend):
        backend_C = _make_batch_backend(
            name="Batch-C",
            pool_view=viewC_ar,
            iep_model=iep_model,
            r=r,
            N=N,
            m_options_by_ring=m_options_by_ring,
            config=config,
            out_summary_path=backend_C_summary_json,
        )
        stage2_rows = _score_stage2_psll_batch(backend_C=backend_C, survivors=survivors, config=config)
    else:
        stage2_rows = _score_stage2_psll(psll_eval=psll_C, survivors=survivors)
    stage2_psll_csv = str(out_dir / "stage2_stage2_psll.csv")
    _write_csv(stage2_psll_csv, stage2_rows)
    top10 = _select_stage2_top_rows(stage2_rows, topJ=int(config.stage2_topJ))
    stage2_top10_csv = str(out_dir / "stage2_stage2_top10.csv")
    _write_csv(stage2_top10_csv, top10)
    if config.verbose:
        best_psll = None if not top10 else top10[0].get("PSLL_wc_db")
        print(f"[Stage2][Stage2] scored={len(stage2_rows)} topJ={len(top10)} best_PSLL={best_psll}")
    _write_progress(out_dir, {
        "stage": "stage2_done",
        "scored": int(len(stage2_rows)),
        "topJ": int(len(top10)),
        "best_PSLL_wc_db": None if not top10 else top10[0].get("PSLL_wc_db"),
    })

    # Final dynamic patch and delta refinement.
    dynamic_summary: Dict[str, Any] = {"enabled": False}
    if config.enable_dynamic_final and top10:
        # Collect updated static-C S/AR worst points for top10 before generating dynamic patch.
        smain_C_static = SMain2DEvaluator(pool_view=viewC_s, iep_model=iep_model, r=r, N=N, config=SMain2DConfig(verbose=False))
        ar_C_static = ARPSLL2DEvaluator(pool_view=viewC_ar, iep_model=iep_model, r=r, N=N, config=ARPSLL2DConfig(verbose=False))
        top10_for_dyn: List[Dict[str, Any]] = []
        for row in top10:
            m = np.asarray(row["m_json"], dtype=np.int64)
            delta = np.asarray(row["delta_json"], dtype=np.float64)
            sr = smain_C_static.score_single_delta(m=m, delta=delta)
            ar = ar_C_static.score_ar_main(m=m, delta=delta)
            out = dict(row)
            out["S_worst_point_json"] = _wp_smain_to_dict(sr)
            out["AR_worst_point_json"] = _wp_arpsll_to_dict(ar.worst_point)
            top10_for_dyn.append(out)
        dynamic_summary = build_dynamic_patch_from_top_rows(
            top10_for_dyn,
            ctx=pool_ctx,
            config=config,
            out_npz_path=str(out_dir / "stage2_final_dynamic_patch.npz"),
            out_json_path=str(out_dir / "stage2_final_dynamic_patch_summary.json"),
        )

    viewFinal_s = _build_smain_view_C(pool_ctx, include_dynamic=bool(dynamic_summary.get("enabled", False)))
    viewFinal_ar = _build_arpsll_view_C(pool_ctx, include_dynamic=bool(dynamic_summary.get("enabled", False)))

    if bool(config.use_batch_backend):
        if (not bool(dynamic_summary.get("enabled", False))) and backend_C is not None:
            backend_Final = backend_C
            _write_json(backend_Final_summary_json, backend_Final.summary())
            if config.verbose:
                print("[Stage2][Batch-Final] reuse Batch-C backend because dynamic final is disabled")
        else:
            backend_Final = _make_batch_backend(
                name="Batch-Final",
                pool_view=viewFinal_ar,
                iep_model=iep_model,
                r=r,
                N=N,
                m_options_by_ring=m_options_by_ring,
                config=config,
                out_summary_path=backend_Final_summary_json,
            )
        smain_final_fine = None
        arpsll_final_fine = None
        if bool(config.final_use_fine):
            smain_final_fine = SMain2DEvaluator(
                pool_view=viewFinal_s,
                iep_model=iep_model,
                r=r,
                N=N,
                config=SMain2DConfig(verbose=False),
            )
            arpsll_final_fine = ARPSLL2DEvaluator(
                pool_view=viewFinal_ar,
                iep_model=iep_model,
                r=r,
                N=N,
                config=ARPSLL2DConfig(verbose=False),
            )
        final_grid_rows, final_best_per_base = _run_final_grid_batch(
            backend_Final=backend_Final,
            top_rows=top10,
            delta_gold=delta_gold,
            thresholds=thresholds,
            config=config,
            smain_fine_eval=smain_final_fine,
            arpsll_fine_eval=arpsll_final_fine,
        )
    else:
        smain_final = SMain2DEvaluator(pool_view=viewFinal_s, iep_model=iep_model, r=r, N=N, config=SMain2DConfig(verbose=False))
        arpsll_final = ARPSLL2DEvaluator(pool_view=viewFinal_ar, iep_model=iep_model, r=r, N=N, config=ARPSLL2DConfig(verbose=False))
        final_grid_rows, final_best_per_base = _run_final_grid(
            smain_eval=smain_final,
            arpsll_eval=arpsll_final,
            top_rows=top10,
            delta_gold=delta_gold,
            thresholds=thresholds,
            config=config,
        )
    final_top3 = final_best_per_base[: int(config.final_topJ)]
    final_grid_csv = str(out_dir / "stage2_final_delta_grid.csv")
    final_top3_csv = str(out_dir / "stage2_final_top3.csv")
    _write_csv(final_grid_csv, final_grid_rows)
    _write_csv(final_top3_csv, final_top3)
    _write_progress(out_dir, {
        "stage": "final_done",
        "grid_rows": int(len(final_grid_rows)),
        "final_topJ": int(len(final_top3)),
        "best_final_PSLL_wc_db": None if not final_top3 else final_top3[0].get("PSLL_wc_db"),
        "dynamic_enabled": bool(dynamic_summary.get("enabled", False)),
    })

    seeds_json = str(out_dir / "stage2_shortlist_seeds.json")
    seeds_npz = str(out_dir / "stage2_shortlist_seeds.npz")
    seeds_payload = {
        "version": "stage2_screen_main_batch_v3_metadata_patch1",
        "backend": {"use_batch_backend": bool(config.use_batch_backend), "module": "stage2_field_batch.py" if bool(config.use_batch_backend) else "legacy_score_scripts"},
        "stage1_source": {"npz": str(stage1_npz_path), "json": str(stage1_json_path)},
        "pool_source": {"npz": str(static_pool_npz_path), "json": str(static_pool_json_path)},
        "dynamic_final": dynamic_summary,
        "thresholds": thresholds,
        "top3": final_top3,
    }
    _write_json(seeds_json, seeds_payload)
    if final_top3:
        np.savez(
            seeds_npz,
            r_star=r,
            N_star=N,
            m_top=np.asarray([np.asarray(row["m_json"], dtype=np.int64) for row in final_top3], dtype=np.int64),
            delta_pre_top=np.asarray([np.asarray(row["delta_pre_json"], dtype=np.float64) for row in final_top3], dtype=np.float64),
            psll_top=np.asarray([float(row["PSLL_wc_db"]) for row in final_top3], dtype=np.float64),
            smain_top=np.asarray([float(row["S_main"]) for row in final_top3], dtype=np.float64),
            ar_main_top=np.asarray([float(row["AR_main_db"]) for row in final_top3], dtype=np.float64),
        )
    else:
        np.savez(seeds_npz, r_star=r, N_star=N)

    report_json = str(out_dir / "stage2_run_report.json")
    report = {
        "ok": True,
        "config": dataclasses.asdict(config),
        "inputs": {
            "stage1_npz": str(stage1_npz_path),
            "unit_npz": str(unit_npz_path),
            "boundary_json": str(mainlobe_table_json_path),
        "mainlobe_table_json": str(mainlobe_table_json_path),
            "static_pool_npz": str(static_pool_npz_path),
        },
        "geometry": {"K": K, "r": r.tolist(), "N": N.astype(int).tolist()},
        "m_generation": {
            "active_rings_0based": active_rings.astype(int).tolist(),
            "candidate_mode": str(config.candidate_mode),
            "q_alias": q_alias.astype(int).tolist(),
            "n_designs": int(m_combos.shape[0]),
            "m_options": m_option_rows,
        },
        "thresholds": thresholds,
        "counts": {
            "stage1_pair_count": int(len(pair_rows)),
            "stage1_survivor_count": int(len(survivors)),
            "stage2_scored_count": int(len(stage2_rows)),
            "stage2_topJ_count": int(len(top10)),
            "final_grid_count": int(len(final_grid_rows)),
            "final_topJ_count": int(len(final_top3)),
        },
        "dynamic_final": dynamic_summary,
        "backend": {
            "use_batch_backend": bool(config.use_batch_backend),
            "backend_module": "stage2_field_batch.py" if bool(config.use_batch_backend) else "legacy_score_scripts",
            "batch_stage1_chunk": int(config.batch_stage1_chunk),
            "batch_stage2_chunk": int(config.batch_stage2_chunk),
            "batch_final_chunk": int(config.batch_final_chunk),
            "backend_A_summary_json": backend_A_summary_json if bool(config.use_batch_backend) else "",
            "backend_C_summary_json": backend_C_summary_json if bool(config.use_batch_backend) else "",
            "backend_Final_summary_json": backend_Final_summary_json if bool(config.use_batch_backend) else "",
            "legacy_score_scripts_used_for_dynamic_patch": bool(config.enable_dynamic_final and bool(top10)),
        },
        "poolA_main_domain": "all ZONE_MAIN points, rho <= rho_3db",
        "fine_orchestration": {
            "coarse_backend": "stage2_field_batch" if bool(config.use_batch_backend) else "legacy_reference_evaluators",
            "fine_backend": "legacy_reference_evaluators",
            "fine_enabled": bool(config.final_use_fine),
        },
        "status_contract": {
            "handedness_flip_is_feasible": False,
            "ar_bad_value_db": 300.0,
        },
        "best_final": None if not final_top3 else final_top3[0],
        "outputs": {
            "stage1_pairs_csv": stage1_pairs_csv,
            "stage1_survivors_csv": stage1_survivors_csv,
            "stage2_psll_csv": stage2_psll_csv,
            "stage2_top10_csv": stage2_top10_csv,
            "final_delta_grid_csv": final_grid_csv,
            "final_top3_csv": final_top3_csv,
            "seeds_json": seeds_json,
            "seeds_npz": seeds_npz,
            "backend_A_summary_json": backend_A_summary_json if bool(config.use_batch_backend) else "",
            "backend_C_summary_json": backend_C_summary_json if bool(config.use_batch_backend) else "",
            "backend_Final_summary_json": backend_Final_summary_json if bool(config.use_batch_backend) else "",
        },
    }
    _write_json(report_json, report)

    best = final_top3[0] if final_top3 else None
    return Stage2ScreenResult(
        output_dir=str(out_dir),
        config_json_path=config_path,
        inputs_json_path=inputs_path,
        m_options_csv_path=m_options_csv,
        m_combos_npz_path=m_combos_npz,
        stage1_pairs_csv_path=stage1_pairs_csv,
        stage1_survivors_csv_path=stage1_survivors_csv,
        stage2_psll_csv_path=stage2_psll_csv,
        stage2_top10_csv_path=stage2_top10_csv,
        final_delta_grid_csv_path=final_grid_csv,
        final_top3_csv_path=final_top3_csv,
        seeds_json_path=seeds_json,
        seeds_npz_path=seeds_npz,
        report_json_path=report_json,
        stage1_pair_count=int(len(pair_rows)),
        stage1_survivor_count=int(len(survivors)),
        stage2_scored_count=int(len(stage2_rows)),
        stage2_topJ_count=int(len(top10)),
        final_grid_count=int(len(final_grid_rows)),
        final_topJ_count=int(len(final_top3)),
        best_final_psll_db=None if best is None else float(best["PSLL_wc_db"]),
        best_final_ar_main_db=None if best is None else float(best["AR_main_db"]),
        best_final_smain=None if best is None else float(best["S_main"]),
        backend_A_summary_json_path=backend_A_summary_json if bool(config.use_batch_backend) else "",
        backend_C_summary_json_path=backend_C_summary_json if bool(config.use_batch_backend) else "",
        backend_Final_summary_json_path=backend_Final_summary_json if bool(config.use_batch_backend) else "",
    )


__all__ = [
    "Stage2ScreenConfig",
    "Stage2ScreenResult",
    "run_stage2_screen_main",
    "m_candidates_around_quarter",
    "build_m_options",
    "_has_zero_alias_risk",
    "enumerate_m_combos",
]


if __name__ == "__main__":
    raise SystemExit("stage2_screen_main.py is a module. Run run_stage2_auto.py instead.")
