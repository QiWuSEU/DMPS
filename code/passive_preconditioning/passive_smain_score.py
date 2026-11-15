from __future__ import annotations

"""
passive_smain_score.py

Stage-2 / Stage-3 shared S_main evaluator.

Design intent
-------------
- Single spectral backend: iep_io.IEPModel only.
- The caller owns the pool (static or dynamic). This module never builds pools.
- The evaluator owns runtime precomputation and scoring:
    * flatten main-point pool view
    * theta-key spectral interpolation via iep_io
    * pointwise a0 reference denominator
    * lowest-2 co-channel alias residual basis
    * stage-2 ring-local mk candidate cache (optional)
    * stage-3 frozen tensor cache for fixed m* (optional)

S_main definition
-----------------
For candidate (m, delta):

  S_main(m, delta)
    = max_{scan} max_{p in Omega_main(scan)}
        |Delta E_co(p; m, delta)| / ( |E_a0_ref(p)| + eps )

with

  E_a0_ref(p) = a0(theta_p) * [ 1 + sum_k w_k N_k J0(beta r_k rho_p) ]

and residual

  Delta E_co(p; m, delta)
    = sum_k sum_{q != 0} exp(-j q delta_k) *
        sum_{b=1..B_alias} C_{k,q,b}(p; m_k)

with

  C_{k,q,b}(p; m_k)
    = w_k * N_k * a_q(theta_p) * exp(j q phi_p)
      * i^{nu_{k,q}^{(b)}}
      * J_{nu_{k,q}^{(b)}}(beta r_k rho_p)
      * exp(j nu_{k,q}^{(b)} psi_p)

where the co-channel alias orders are

  nu = q m_k + l N_k,  l in Z

and we keep the lowest-1 or lowest-2 orders according to config.B_alias.

Current model conventions
-------------------------
- The center element contributes to the q=0 reference term only.
- B_alias is restricted to 1 or 2.
- Delta vectors are validated and evaluated modulo 2*pi.
- Every scan used by S_main must contain at least one main-domain point.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import j0 as _j0, jv as _jv

from element_spectrum import iep_model as iep_io


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SMain2DConfig:
    beta: float = 2.0 * np.pi
    eps: float = 1e-12

    B_alias: int = 2

    cache_bessel: bool = True
    cache_phase: bool = True
    cache_basis_by_mk: bool = True
    cache_union_q: bool = True

    use_active_q_only: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class SMainPoolView:
    theta_s_deg: np.ndarray
    theta_deg_all: np.ndarray
    phi_deg_all: np.ndarray
    rho_all: np.ndarray
    psi_rad_all: np.ndarray
    slice_list: Tuple[Tuple[int, int], ...]
    main_idx_list: Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class SMainWorstPoint:
    theta_s_deg: float
    theta_deg: float
    phi_deg: float
    rho: float
    psi_rad: float
    delta_id: int
    ratio: float
    global_point_idx: int
    local_point_idx: int
    scan_idx: int


@dataclass(frozen=True)
class SMainScanResult:
    theta_s_deg: float
    S_scan: float
    delta_id_wc: int
    idx_wc_local: int
    idx_wc_global: int
    counts_main: int
    meta: Dict[str, Any]


@dataclass(frozen=True)
class SMainResult:
    S_main: float
    status: str
    idx_wc_scan: int
    per_scan_results: Tuple[SMainScanResult, ...]
    worst: Optional[SMainWorstPoint]
    meta: Dict[str, Any]


@dataclass(frozen=True)
class RingMCandidateTable:
    ring_idx: int
    m_candidates: np.ndarray


@dataclass(frozen=True)
class Stage2MCandidateSet:
    tables: Tuple[RingMCandidateTable, ...]


@dataclass(frozen=True)
class SMainCandidate:
    m: np.ndarray
    delta_set: Tuple[np.ndarray, ...]
    candidate_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class SMainAnchorResult:
    q_offset: float
    delta: np.ndarray
    result: SMainResult


@dataclass(frozen=True)
class SMainAnchorSweepResult:
    best_index: int
    best_q_offset: float
    best_delta: np.ndarray
    best_result: SMainResult
    all_results: Tuple[SMainAnchorResult, ...]


@dataclass(frozen=True)
class SMainStage2ScreenRow:
    candidate_id: str
    m: np.ndarray
    best_anchor_index: int
    best_q_offset: float
    best_delta: np.ndarray
    S_main: float
    status: str
    worst: Optional[SMainWorstPoint]
    per_anchor_S: Tuple[float, ...]
    anchor_results: Tuple[SMainAnchorResult, ...]
    meta: Dict[str, Any]


# -----------------------------------------------------------------------------
# Pool-view helpers (optional; main script may also build views directly)
# -----------------------------------------------------------------------------


def build_pool_view_from_scan_point_dicts(
    scan_point_dicts: Mapping[float, Mapping[str, np.ndarray]],
    *,
    main_mask_fn: Optional[Callable[[Mapping[str, np.ndarray]], np.ndarray]] = None,
) -> SMainPoolView:
    """
    Build an SMainPoolView from per-scan point dicts.

    Expected per-scan keys:
      - theta_deg, phi_deg, rho
      - psi_deg or psi_rad

    main_mask_fn:
      - if None, all points of each scan are treated as main points
      - otherwise receives the per-scan dict and returns a boolean mask
    """
    theta_s_list = []
    theta_deg_all: List[np.ndarray] = []
    phi_deg_all: List[np.ndarray] = []
    rho_all: List[np.ndarray] = []
    psi_all: List[np.ndarray] = []
    slice_list: List[Tuple[int, int]] = []
    main_idx_list: List[np.ndarray] = []

    pos = 0
    for theta_s_deg in sorted(float(x) for x in scan_point_dicts.keys()):
        pts = scan_point_dicts[theta_s_deg]
        theta_deg = np.asarray(pts["theta_deg"], dtype=np.float64).reshape(-1)
        phi_deg = np.asarray(pts["phi_deg"], dtype=np.float64).reshape(-1)
        rho = np.asarray(pts["rho"], dtype=np.float64).reshape(-1)
        if "psi_rad" in pts:
            psi = np.asarray(pts["psi_rad"], dtype=np.float64).reshape(-1)
        elif "psi_deg" in pts:
            psi = np.deg2rad(np.asarray(pts["psi_deg"], dtype=np.float64).reshape(-1))
        else:
            raise KeyError("point dict must contain psi_rad or psi_deg")

        n = int(theta_deg.size)
        if n == 0:
            raise ValueError(f"scan theta_s={theta_s_deg}: contains zero points")
        if not (phi_deg.size == n and rho.size == n and psi.size == n):
            raise ValueError(f"scan theta_s={theta_s_deg}: point arrays length mismatch")
        if not (np.all(np.isfinite(theta_deg)) and np.all(np.isfinite(phi_deg)) and np.all(np.isfinite(rho)) and np.all(np.isfinite(psi))):
            raise ValueError(f"scan theta_s={theta_s_deg}: point geometry contains NaN/Inf")
        if np.any(rho < 0.0):
            raise ValueError(f"scan theta_s={theta_s_deg}: rho must be non-negative")

        if main_mask_fn is None:
            main_idx = np.arange(n, dtype=np.int64)
        else:
            mask = np.asarray(main_mask_fn(pts)).reshape(-1).astype(bool)
            if mask.size != n:
                raise ValueError(f"scan theta_s={theta_s_deg}: main_mask_fn returned wrong length")
            main_idx = np.flatnonzero(mask).astype(np.int64)
        if main_idx.size == 0:
            raise ValueError(f"scan theta_s={theta_s_deg}: contains zero main points")

        theta_s_list.append(float(theta_s_deg))
        theta_deg_all.append(theta_deg)
        phi_deg_all.append(phi_deg)
        rho_all.append(rho)
        psi_all.append(psi)
        slice_list.append((pos, pos + n))
        main_idx_list.append(main_idx)
        pos += n

    if pos == 0:
        raise ValueError("scan_point_dicts is empty")

    return SMainPoolView(
        theta_s_deg=np.asarray(theta_s_list, dtype=np.float64),
        theta_deg_all=np.concatenate(theta_deg_all, axis=0).astype(np.float64),
        phi_deg_all=np.concatenate(phi_deg_all, axis=0).astype(np.float64),
        rho_all=np.concatenate(rho_all, axis=0).astype(np.float64),
        psi_rad_all=np.concatenate(psi_all, axis=0).astype(np.float64),
        slice_list=tuple((int(a), int(b)) for (a, b) in slice_list),
        main_idx_list=tuple(np.asarray(x, dtype=np.int64) for x in main_idx_list),
    )


def smain_worst_point_to_dict(w: Optional[SMainWorstPoint]) -> Optional[Dict[str, Any]]:
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


def smain_scan_result_to_dict(row: SMainScanResult) -> Dict[str, Any]:
    return {
        "theta_s_deg": float(row.theta_s_deg),
        "S_scan": float(row.S_scan),
        "delta_id_wc": int(row.delta_id_wc),
        "idx_wc_local": int(row.idx_wc_local),
        "idx_wc_global": int(row.idx_wc_global),
        "counts_main": int(row.counts_main),
        "meta": dict(row.meta),
    }


def smain_result_to_dict(res: SMainResult, *, include_per_scan: bool = True) -> Dict[str, Any]:
    out = {
        "S_main": float(res.S_main),
        "status": str(res.status),
        "idx_wc_scan": int(res.idx_wc_scan),
        "worst": smain_worst_point_to_dict(res.worst),
        "meta": dict(res.meta),
    }
    if include_per_scan:
        out["per_scan_results"] = [smain_scan_result_to_dict(x) for x in res.per_scan_results]
    return out


def smain_screen_row_to_record(row: SMainStage2ScreenRow) -> Dict[str, Any]:
    worst = smain_worst_point_to_dict(row.worst) or {}
    return {
        "candidate_id": row.candidate_id,
        "S_main": float(row.S_main),
        "status": row.status,
        "best_anchor_index": int(row.best_anchor_index),
        "best_q_offset": float(row.best_q_offset),
        "per_anchor_S": [float(x) for x in row.per_anchor_S],
        "worst_theta_s_deg": worst.get("theta_s_deg"),
        "worst_theta_deg": worst.get("theta_deg"),
        "worst_phi_deg": worst.get("phi_deg"),
        "worst_rho": worst.get("rho"),
        "worst_psi_deg": worst.get("psi_deg"),
        "worst_scan_idx": worst.get("scan_idx"),
        "worst_delta_id": worst.get("delta_id"),
        "meta": dict(row.meta),
    }


def smain_screen_rows_to_records(rows: Sequence[SMainStage2ScreenRow]) -> List[Dict[str, Any]]:
    return [smain_screen_row_to_record(x) for x in rows]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _wrap_rad(phi: np.ndarray) -> np.ndarray:
    two_pi = 2.0 * np.pi
    y = np.mod(phi, two_pi)
    y[y == two_pi] = 0.0
    return y


def _jpow_j(n: int) -> complex:
    m = int(n) % 4
    if m == 0:
        return 1.0 + 0.0j
    if m == 1:
        return 0.0 + 1.0j
    if m == 2:
        return -1.0 + 0.0j
    return 0.0 - 1.0j


def _extract_unique_theta_keys(theta_rad: np.ndarray, decimals: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    key = np.round(np.asarray(theta_rad, dtype=np.float64), decimals)
    uniq, inv = np.unique(key, return_inverse=True)
    theta_key = np.zeros((uniq.size,), dtype=np.float64)
    for i in range(int(uniq.size)):
        theta_key[i] = float(theta_rad[np.where(inv == i)[0][0]])
    return theta_key, inv.astype(np.int64)


def _normalize_delta_vector(delta: np.ndarray, *, K: int, name: str = "delta") -> np.ndarray:
    arr = np.asarray(delta, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector with shape ({int(K)},), got {arr.shape}")
    if arr.size != int(K):
        raise ValueError(f"{name} must have shape ({int(K)},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr)).reshape(-1).astype(int).tolist()
        raise ValueError(f"{name} contains NaN or Inf at indices {bad}")
    return np.mod(arr, 2.0 * np.pi).astype(np.float64, copy=False)


def _lowest_two_orders_for_aN(a: int, N: int) -> Tuple[int, int]:
    if N <= 0:
        raise ValueError("N must be positive")
    l0 = int(np.rint(-float(a) / float(N)))
    cand_ls = [l0 - 2, l0 - 1, l0, l0 + 1, l0 + 2]
    cand = []
    seen = set()
    for l in cand_ls:
        nu = int(a + l * N)
        if nu in seen:
            continue
        seen.add(nu)
        cand.append((abs(nu), abs(l), nu))
    cand.sort()
    nu1 = int(cand[0][2])
    nu2 = nu1
    for item in cand[1:]:
        if int(item[2]) != nu1:
            nu2 = int(item[2])
            break
    return nu1, nu2


def compute_alias_orders_co(*, N: np.ndarray, m: np.ndarray, q_values: np.ndarray, B: int = 2) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    N = np.asarray(N, dtype=np.int64).reshape(-1)
    m = np.asarray(m, dtype=np.int64).reshape(-1)
    q_values = np.asarray(q_values, dtype=np.int64).reshape(-1)
    if N.size != m.size:
        raise ValueError("N and m must have same length")
    K = int(N.size)
    nQ = int(q_values.size)
    nu1 = np.zeros((K, nQ), dtype=np.int64)
    nu2 = np.zeros((K, nQ), dtype=np.int64) if B == 2 else None
    for k in range(K):
        for j in range(nQ):
            a = int(q_values[j] * m[k])
            o1, o2 = _lowest_two_orders_for_aN(a, int(N[k]))
            nu1[k, j] = o1
            if nu2 is not None:
                nu2[k, j] = o2
    return nu1, nu2


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------


class SMain2DEvaluator:
    def __init__(
        self,
        *,
        pool_view: SMainPoolView,
        iep_model: iep_io.IEPModel,
        r: np.ndarray,
        N: np.ndarray,
        w: Optional[np.ndarray] = None,
        alpha_ring_rad: Optional[np.ndarray] = None,
        config: Optional[SMain2DConfig] = None,
    ) -> None:
        self.pool_view = pool_view
        self.iep_model = iep_model
        self.config = config or SMain2DConfig()
        if int(self.config.B_alias) not in (1, 2):
            raise ValueError("SMain2DConfig.B_alias must be 1 or 2; this evaluator only implements lowest-1/lowest-2 alias truncation")
        self.beta = float(self.config.beta)
        self.eps = float(self.config.eps)
        self.verbose = bool(self.config.verbose)

        self.r = np.asarray(r, dtype=np.float64).reshape(-1)
        self.N = np.asarray(N, dtype=np.int64).reshape(-1)
        if self.r.size == 0 or self.r.size != self.N.size:
            raise ValueError("r and N must be non-empty and same length")
        self.K = int(self.r.size)
        if np.any(self.N <= 0):
            raise ValueError("N must be positive")

        if w is None:
            self.w = np.ones((self.K,), dtype=np.float64)
        else:
            self.w = np.asarray(w, dtype=np.float64).reshape(-1)
            if self.w.size != self.K:
                raise ValueError("w must have same length as r")

        if alpha_ring_rad is None:
            self.alpha_ring_rad = np.zeros((self.K,), dtype=np.float64)
        else:
            self.alpha_ring_rad = np.asarray(alpha_ring_rad, dtype=np.float64).reshape(-1)
            if self.alpha_ring_rad.size != self.K:
                raise ValueError("alpha_ring_rad must have same length as r")

        self._build_main_point_view()
        self._build_theta_key_map()
        self._build_interp_spectral_assets()
        self._build_pointwise_spectral_factors()
        self._build_reference_terms()

        self._m_last: Optional[np.ndarray] = None
        self._q_non0: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._q_non0_col_idx: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._nu1: Optional[np.ndarray] = None
        self._nu2: Optional[np.ndarray] = None
        self._orders_per_k: Tuple[np.ndarray, ...] = tuple()
        self._Jn_cache_per_k: List[Dict[int, np.ndarray]] = []
        self._phase_cache_per_k: List[Dict[int, np.ndarray]] = []
        self._Ck_list: List[np.ndarray] = []
        self._C_tensor: Optional[np.ndarray] = None
        self._frozen_m_tensor: Optional[np.ndarray] = None

        self._stage2_basis_cache: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self.validate()
        if self.verbose:
            print("=== [SMAIN2D] INIT DONE ===")
            print(f"[SMAIN2D] S={self.S} K={self.K} P_main={self.P} theta_keys={self.theta_key_rad.size}")
            print(f"[SMAIN2D] nQa={self.aq_q_list.size} q_list_a={self.aq_q_list.tolist()}")

    # ------------------------------------------------------------------
    # Summary / validation
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        eref = np.asarray(self.Eref_abs_p, dtype=np.float64)
        points_per_scan = [int(x.size) for x in self.point_pos_by_scan]
        return {
            "S": int(self.S),
            "K": int(self.K),
            "P_main": int(self.P),
            "points_per_scan": points_per_scan,
            "theta_s_deg": self.theta_s_deg.tolist(),
            "theta_key_count": int(self.theta_key_rad.size),
            "nQa": int(self.aq_q_list.size),
            "q_list_a": self.aq_q_list.tolist(),
            "q0_idx": None if self.q0_idx is None else int(self.q0_idx),
            "q_non0_global": self.q_non0_global.tolist(),
            "nQ_non0_global": int(self.q_non0_global.size),
            "r": self.r.tolist(),
            "N": self.N.tolist(),
            "w": self.w.tolist(),
            "B_alias": int(self.config.B_alias),
            "use_active_q_only": bool(self.config.use_active_q_only),
            "Eref_abs_min": float(np.min(eref)) if eref.size else float("nan"),
            "Eref_abs_p01": float(np.percentile(eref, 1.0)) if eref.size else float("nan"),
            "Eref_abs_median": float(np.median(eref)) if eref.size else float("nan"),
            "Eref_abs_max": float(np.max(eref)) if eref.size else float("nan"),
            "Eref_near_eps_count": int(np.count_nonzero(eref <= 10.0 * float(self.eps))) if eref.size else 0,
            "has_stage2_cache": bool(self._stage2_basis_cache),
            "has_stage3_tensor": bool(self._C_tensor is not None),
        }

    def validate(self) -> None:
        if self.P <= 0:
            raise ValueError("No main points in pool_view")
        if int(self.config.B_alias) not in (1, 2):
            raise ValueError("SMain2DConfig.B_alias must be 1 or 2")
        if np.any(self.N <= 0):
            raise ValueError("N must be positive")
        if not (np.all(np.isfinite(self.r)) and np.all(np.isfinite(self.w)) and np.all(np.isfinite(self.alpha_ring_rad))):
            raise ValueError("r/w/alpha_ring_rad contains NaN/Inf")
        if np.any(~np.isfinite(self.theta_rad)) or np.any(~np.isfinite(self.phi_rad)) or np.any(~np.isfinite(self.rho)) or np.any(~np.isfinite(self.psi)):
            raise ValueError("Point geometry contains NaN/Inf")
        if self.aq_phase_p_full.shape != (self.P, self.aq_q_list.size):
            raise ValueError("aq_phase_p_full shape mismatch")
        if np.any(~np.isfinite(self.Eref_abs_p)):
            raise ValueError("Eref_abs_p contains NaN/Inf")
        if np.any(self.Eref_abs_p < 0.0):
            raise ValueError("Eref_abs_p must be non-negative")
        if np.any(self.rho < 0.0):
            raise ValueError("rho must be non-negative")
        empty_scans = [int(s) for s, pos in enumerate(self.point_pos_by_scan) if pos.size == 0]
        if empty_scans:
            raise ValueError(f"main domain is empty for scan indices {empty_scans}")
        if self.q0_idx is None and self.verbose:
            print("[SMAIN2D][WARN] aq_q_list has no q=0 column; denominator still uses assets['a0']")

    # ------------------------------------------------------------------
    # Fixed precompute
    # ------------------------------------------------------------------
    def _build_main_point_view(self) -> None:
        pv = self.pool_view
        self.theta_s_deg = np.asarray(pv.theta_s_deg, dtype=np.float64).reshape(-1)
        self.S = int(self.theta_s_deg.size)
        if self.S == 0:
            raise ValueError("pool_view.theta_s_deg must be non-empty")
        if not np.all(np.isfinite(self.theta_s_deg)):
            raise ValueError("pool_view.theta_s_deg contains NaN/Inf")

        theta_all = np.asarray(pv.theta_deg_all, dtype=np.float64).reshape(-1)
        phi_all = np.asarray(pv.phi_deg_all, dtype=np.float64).reshape(-1)
        rho_all = np.asarray(pv.rho_all, dtype=np.float64).reshape(-1)
        psi_all = np.asarray(pv.psi_rad_all, dtype=np.float64).reshape(-1)
        self.P_total = int(theta_all.size)
        if self.P_total == 0:
            raise ValueError("pool_view contains zero points")
        if not (phi_all.size == self.P_total and rho_all.size == self.P_total and psi_all.size == self.P_total):
            raise ValueError("pool_view point arrays length mismatch")
        if not (np.all(np.isfinite(theta_all)) and np.all(np.isfinite(phi_all)) and np.all(np.isfinite(rho_all)) and np.all(np.isfinite(psi_all))):
            raise ValueError("pool_view point geometry contains NaN/Inf")
        if np.any(rho_all < 0.0):
            raise ValueError("pool_view.rho_all must be non-negative")
        if len(pv.slice_list) != self.S:
            raise ValueError(f"slice_list length={len(pv.slice_list)} does not match scans={self.S}")
        if len(pv.main_idx_list) != self.S:
            raise ValueError(f"main_idx_list length={len(pv.main_idx_list)} does not match scans={self.S}")

        slice_list = tuple((int(a), int(b)) for a, b in pv.slice_list)
        if slice_list[0][0] != 0:
            raise ValueError(f"slice_list must start at 0, got {slice_list[0][0]}")
        prev_end = 0
        for s, (a, b) in enumerate(slice_list):
            if a != prev_end:
                raise ValueError(f"slice_list is not contiguous at scan {s}: expected start={prev_end}, got {a}")
            if a < 0 or b <= a or b > self.P_total:
                raise ValueError(f"bad slice for scan {s}: {(a, b)} with P_total={self.P_total}")
            prev_end = b
        if prev_end != self.P_total:
            raise ValueError(f"slice_list does not cover all points: final end={prev_end}, P_total={self.P_total}")
        self.slice_list = slice_list

        main_gidx: List[np.ndarray] = []
        scan_id: List[np.ndarray] = []
        local_idx: List[np.ndarray] = []
        point_pos_by_scan: List[np.ndarray] = []

        pos = 0
        for s in range(self.S):
            a, b = slice_list[s]
            loc = np.asarray(pv.main_idx_list[s], dtype=np.int64).reshape(-1)
            if loc.size == 0:
                raise ValueError(f"main domain is empty for scan {s}")
            if np.any(loc < 0) or np.any(loc >= (b - a)):
                raise ValueError(f"main_idx out of range for scan {s}")
            if np.unique(loc).size != loc.size:
                raise ValueError(f"main_idx contains duplicates for scan {s}")
            g = (a + loc).astype(np.int64)
            n = int(g.size)
            main_gidx.append(g)
            scan_id.append(np.full((n,), s, dtype=np.int64))
            local_idx.append(loc.astype(np.int64))
            point_pos_by_scan.append(np.arange(pos, pos + n, dtype=np.int64))
            pos += n

        self.main_gidx = np.concatenate(main_gidx, axis=0).astype(np.int64)
        self.scan_id = np.concatenate(scan_id, axis=0).astype(np.int64)
        self.local_idx_in_scan = np.concatenate(local_idx, axis=0).astype(np.int64)
        self.point_pos_by_scan = tuple(point_pos_by_scan)
        self.P = int(self.main_gidx.size)

        self.theta_deg = theta_all[self.main_gidx]
        self.phi_deg = phi_all[self.main_gidx]
        self.rho = rho_all[self.main_gidx]
        self.psi = psi_all[self.main_gidx]
        self.theta_rad = np.deg2rad(self.theta_deg).astype(np.float64)
        self.phi_rad = _wrap_rad(np.deg2rad(self.phi_deg).astype(np.float64))

    def _build_theta_key_map(self) -> None:
        self.theta_key_rad, self.theta_idx_p = _extract_unique_theta_keys(self.theta_rad)

    def _build_interp_spectral_assets(self) -> None:
        assets = self.iep_model.build_interp_spectral_assets(
            theta_key_rad=self.theta_key_rad,
            verbose=self.verbose,
        )
        if "a0" not in assets or "aq_q_list" not in assets or "aq_grid" not in assets:
            raise RuntimeError("iep_model.build_interp_spectral_assets returned incomplete aq assets")
        self.assets = assets
        self.a0_key = np.asarray(assets["a0"], dtype=np.complex128).reshape(-1)
        self.aq_q_list = np.asarray(assets["aq_q_list"], dtype=np.int64).reshape(-1)
        self.aq_key_grid = np.asarray(assets["aq_grid"], dtype=np.complex128)
        if self.aq_key_grid.shape != (self.theta_key_rad.size, self.aq_q_list.size):
            raise RuntimeError("aq_grid shape mismatch")
        self.active_a_qidx_by_theta_key = tuple(
            np.asarray(x, dtype=np.int64).reshape(-1)
            for x in assets.get("active_a_qidx_by_theta", [np.arange(self.aq_q_list.size, dtype=np.int64) for _ in range(self.theta_key_rad.size)])
        )
        q0_idx = np.where(self.aq_q_list == 0)[0]
        self.q0_idx = int(q0_idx[0]) if q0_idx.size > 0 else None

    def _build_pointwise_spectral_factors(self) -> None:
        self.a0_p = self.a0_key[self.theta_idx_p].astype(np.complex128)
        self.aq_p_full = self.aq_key_grid[self.theta_idx_p, :].astype(np.complex128)
        if self.aq_p_full.shape != (self.P, self.aq_q_list.size):
            raise RuntimeError("aq_p_full shape mismatch")

        q = self.aq_q_list.astype(np.float64)
        expjqphi = np.exp(1j * self.phi_rad.reshape(-1, 1) * q.reshape(1, -1)).astype(np.complex128)
        self.aq_phase_p_full = (self.aq_p_full * expjqphi).astype(np.complex128)

        # Per-point active mask derived from theta-key active lists; q=0 excluded from residual use.
        active_mask = np.zeros((self.P, self.aq_q_list.size), dtype=bool)
        for tk, qidx in enumerate(self.active_a_qidx_by_theta_key):
            pts = np.flatnonzero(self.theta_idx_p == tk)
            if pts.size == 0:
                continue
            active_mask[np.ix_(pts, qidx)] = True
        if self.q0_idx is not None:
            active_mask[:, self.q0_idx] = False
        self.active_mask_pq = active_mask

        if bool(self.config.use_active_q_only):
            self.aq_phase_p_full = np.where(self.active_mask_pq, self.aq_phase_p_full, 0.0 + 0.0j)

        nz_cols = np.where(np.any(np.abs(self.aq_phase_p_full) > 0.0, axis=0))[0].astype(np.int64)
        if self.q0_idx is not None:
            nz_cols = nz_cols[nz_cols != self.q0_idx]
        self.q_non0_global_col_idx = nz_cols
        self.q_non0_global = self.aq_q_list[nz_cols].astype(np.int64)

    def _build_reference_terms(self) -> None:
        # x_pk = beta * rho_p * r_k
        self.x_pk = (self.beta * self.r.reshape(1, -1)) * self.rho.reshape(-1, 1)
        self.J0_pk = _j0(self.x_pk).astype(np.float64)
        self.AF0_p = (1.0 + (self.J0_pk * (self.w * self.N.astype(np.float64)).reshape(1, -1)).sum(axis=1)).astype(np.float64)
        self.Eref_abs_p = np.abs(self.a0_p * self.AF0_p).astype(np.float64) + float(self.eps)

    # ------------------------------------------------------------------
    # m checks and caches
    # ------------------------------------------------------------------
    def check_m_feasible(self, m: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        m = np.asarray(m, dtype=np.int64).reshape(-1)
        if m.size != self.K:
            raise ValueError(f"m must have shape ({self.K},)")
        bad_lt0 = np.where(m < 0)[0].astype(int).tolist()
        bad_half = np.where(self.N - 2 * m <= 0)[0].astype(int).tolist()
        ok = (len(bad_lt0) == 0) and (len(bad_half) == 0)
        return ok, {
            "bad_m_lt0": bad_lt0,
            "bad_N_minus_2m_le0": bad_half,
        }

    def precompute_stage2_mk_bases(self, *, m_candidates_by_ring: Dict[int, np.ndarray]) -> None:
        if not bool(self.config.cache_basis_by_mk):
            return
        q_non0 = self.q_non0_global.astype(np.int64)
        q_cols = self.q_non0_global_col_idx.astype(np.int64)
        self._stage2_basis_cache = {}
        for k, mk_list in m_candidates_by_ring.items():
            kk = int(k)
            if kk < 0 or kk >= self.K:
                raise ValueError(f"ring_idx out of range: {kk}")
            ring_cache: Dict[int, Dict[str, Any]] = {}
            mk_arr = np.asarray(mk_list, dtype=np.int64).reshape(-1)
            for mk in np.unique(mk_arr).tolist():
                mk_int = int(mk)
                ok, info = self.check_m_feasible(np.array([mk_int if i == kk else max(1, int(min(self.N[i] // 4, self.N[i] // 2 - 1))) for i in range(self.K)], dtype=np.int64))
                # Feasibility for ring-local cache: only enforce mk > 0 and mk < N_k/2 for this ring.
                if mk_int <= 0 or (self.N[kk] - 2 * mk_int) <= 0:
                    continue
                nu1, nu2 = compute_alias_orders_co(
                    N=np.asarray([self.N[kk]], dtype=np.int64),
                    m=np.asarray([mk_int], dtype=np.int64),
                    q_values=q_non0,
                    B=int(self.config.B_alias),
                )
                n1 = nu1[0, :].astype(np.int64)
                n2 = nu2[0, :].astype(np.int64) if nu2 is not None else n1
                Ck = self._build_ring_basis_for_orders(kk, q_non0, q_cols, n1, n2)
                ring_cache[mk_int] = {
                    "q_non0": q_non0.copy(),
                    "q_cols": q_cols.copy(),
                    "nu1": n1.copy(),
                    "nu2": n2.copy(),
                    "Ck": Ck,
                }
            self._stage2_basis_cache[kk] = ring_cache
        if self.verbose:
            sizes = {k: len(v) for k, v in self._stage2_basis_cache.items()}
            print(f"[SMAIN2D] stage2 mk cache built: {sizes}")

    def precompute_stage2_candidate_set(self, candidate_set: Stage2MCandidateSet) -> None:
        self.precompute_stage2_mk_bases(
            m_candidates_by_ring={int(t.ring_idx): np.asarray(t.m_candidates, dtype=np.int64) for t in candidate_set.tables}
        )

    def set_m(self, m: np.ndarray) -> None:
        m = np.asarray(m, dtype=np.int64).reshape(-1)
        if m.size != self.K:
            raise ValueError(f"m must have shape ({self.K},)")
        ok, info = self.check_m_feasible(m)
        if not ok:
            self._m_last = m.copy()
            self._q_non0 = np.zeros((0,), dtype=np.int64)
            self._q_non0_col_idx = np.zeros((0,), dtype=np.int64)
            self._Ck_list = []
            self._C_tensor = None
            self._frozen_m_tensor = None
            return

        self._m_last = m.copy()
        self._C_tensor = None
        self._frozen_m_tensor = None

        # Current q set comes from global nonzero active columns.
        self._q_non0 = self.q_non0_global.copy()
        self._q_non0_col_idx = self.q_non0_global_col_idx.copy()
        q_non0 = self._q_non0
        q_cols = self._q_non0_col_idx
        nQ = int(q_non0.size)
        if nQ == 0:
            self._Ck_list = [np.zeros((0, self.P), dtype=np.complex128) for _ in range(self.K)]
            return

        all_cached = bool(self._stage2_basis_cache) and all(
            (int(k) in self._stage2_basis_cache) and (int(m[k]) in self._stage2_basis_cache[int(k)])
            for k in range(self.K)
        )
        if all_cached:
            self._assemble_residual_basis_from_stage2_cache(m)
            return

        self._assemble_residual_basis_direct(m)

    def freeze_stage3_tensor(self, m: np.ndarray) -> None:
        ok, info = self.check_m_feasible(m)
        if not ok:
            raise ValueError(f"infeasible m for freeze_stage3_tensor: {info}")
        self.set_m(m)
        if len(self._Ck_list) == 0:
            self._C_tensor = np.zeros((self.K, 0, self.P), dtype=np.complex128)
        else:
            self._C_tensor = np.stack(self._Ck_list, axis=0).astype(np.complex128)
        self._frozen_m_tensor = np.asarray(m, dtype=np.int64).reshape(-1).copy()

    # ------------------------------------------------------------------
    # Residual basis assembly
    # ------------------------------------------------------------------
    def _build_ring_basis_for_orders(
        self,
        k: int,
        q_non0: np.ndarray,
        q_cols: np.ndarray,
        n1: np.ndarray,
        n2: np.ndarray,
    ) -> np.ndarray:
        orders = np.unique(np.concatenate([n1, n2], axis=0)).astype(np.int64)
        Jn_dict = self._build_ring_bessel_cache(k, orders)
        Ph_dict = self._build_ring_phase_cache(k, orders)
        aq_phase_sel = self.aq_phase_p_full[:, q_cols].T.astype(np.complex128)  # (nQ,P)
        nQ = int(q_non0.size)
        Ck = np.zeros((nQ, self.P), dtype=np.complex128)
        wk = float(self.w[k])
        Nk = float(self.N[k])
        for iq in range(nQ):
            n1i = int(n1[iq])
            n2i = int(n2[iq])
            t1 = _jpow_j(n1i) * (Jn_dict[n1i].astype(np.complex128) * Ph_dict[n1i].astype(np.complex128))
            if n2i == n1i:
                t2 = 0.0
            else:
                t2 = _jpow_j(n2i) * (Jn_dict[n2i].astype(np.complex128) * Ph_dict[n2i].astype(np.complex128))
            Ck[iq, :] = wk * Nk * aq_phase_sel[iq, :] * (t1 + t2)
        return Ck

    def _assemble_residual_basis_from_stage2_cache(self, m: np.ndarray) -> None:
        self._Ck_list = []
        self._nu1 = np.zeros((self.K, self._q_non0.size), dtype=np.int64)
        self._nu2 = np.zeros((self.K, self._q_non0.size), dtype=np.int64)
        for k in range(self.K):
            entry = self._stage2_basis_cache[int(k)][int(m[k])]
            self._Ck_list.append(np.asarray(entry["Ck"], dtype=np.complex128))
            self._nu1[k, :] = np.asarray(entry["nu1"], dtype=np.int64)
            self._nu2[k, :] = np.asarray(entry["nu2"], dtype=np.int64)
        self._orders_per_k = tuple(np.unique(np.concatenate([self._nu1[k, :], self._nu2[k, :]], axis=0)).astype(np.int64) for k in range(self.K))

    def _assemble_residual_basis_direct(self, m: np.ndarray) -> None:
        q_non0 = self._q_non0
        q_cols = self._q_non0_col_idx
        nu1, nu2 = compute_alias_orders_co(N=self.N, m=m, q_values=q_non0, B=int(self.config.B_alias))
        self._nu1 = nu1
        self._nu2 = nu2 if nu2 is not None else nu1
        self._orders_per_k = tuple(np.unique(np.concatenate([self._nu1[k, :], self._nu2[k, :]], axis=0)).astype(np.int64) for k in range(self.K))
        self._Ck_list = []
        for k in range(self.K):
            self._Ck_list.append(self._build_ring_basis_for_orders(k, q_non0, q_cols, self._nu1[k, :], self._nu2[k, :]))

    def _build_ring_bessel_cache(self, k: int, orders: np.ndarray) -> Dict[int, np.ndarray]:
        xk = self.x_pk[:, int(k)].astype(np.float64)
        out: Dict[int, np.ndarray] = {}
        for n in np.asarray(orders, dtype=np.int64).tolist():
            out[int(n)] = _jv(int(n), xk).astype(np.float64)
        return out

    def _build_ring_phase_cache(self, k: int, orders: np.ndarray) -> Dict[int, np.ndarray]:
        phi = (self.psi - float(self.alpha_ring_rad[int(k)])).astype(np.float64)
        out: Dict[int, np.ndarray] = {}
        for n in np.asarray(orders, dtype=np.int64).tolist():
            out[int(n)] = np.exp(1j * float(n) * phi).astype(np.complex128)
        return out

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self, *, m: np.ndarray, delta_set: Sequence[np.ndarray]) -> SMainResult:
        if self.P == 0:
            return SMainResult(S_main=float("nan"), status="EMPTY_MAIN", idx_wc_scan=-1, per_scan_results=tuple(), worst=None, meta={"P_main": 0})
        empty_scans = [int(s) for s, pos in enumerate(self.point_pos_by_scan) if pos.size == 0]
        if empty_scans:
            return SMainResult(
                S_main=float("nan"),
                status="EMPTY_MAIN_SCAN",
                idx_wc_scan=-1,
                per_scan_results=tuple(),
                worst=None,
                meta={"empty_scan_indices": empty_scans},
            )

        ok, info = self.check_m_feasible(m)
        if not ok:
            return SMainResult(
                S_main=float("inf"),
                status="INFEASIBLE_M",
                idx_wc_scan=-1,
                per_scan_results=tuple(),
                worst=None,
                meta=info,
            )

        if isinstance(delta_set, np.ndarray):
            delta_arr = np.asarray(delta_set, dtype=np.float64)
            if delta_arr.ndim == 1:
                raw_delta_list = [delta_arr]
            elif delta_arr.ndim == 2:
                raw_delta_list = [delta_arr[i, :] for i in range(delta_arr.shape[0])]
            else:
                raise ValueError(f"delta_set ndarray must be one- or two-dimensional, got shape {delta_arr.shape}")
        else:
            raw_delta_list = list(delta_set)
        if len(raw_delta_list) == 0:
            raise ValueError("delta_set must be non-empty")
        delta_list = [
            _normalize_delta_vector(d, K=self.K, name=f"delta_set[{i}]")
            for i, d in enumerate(raw_delta_list)
        ]

        m_arr = np.asarray(m, dtype=np.int64).reshape(-1)
        if self._m_last is None or np.any(self._m_last != m_arr):
            self.set_m(m_arr)

        if self._q_non0.size == 0:
            scan_results = tuple(
                SMainScanResult(
                    theta_s_deg=float(self.theta_s_deg[s]),
                    S_scan=0.0,
                    delta_id_wc=0,
                    idx_wc_local=0 if self.point_pos_by_scan[s].size else -1,
                    idx_wc_global=int(self.main_gidx[self.point_pos_by_scan[s][0]]) if self.point_pos_by_scan[s].size else -1,
                    counts_main=int(self.point_pos_by_scan[s].size),
                    meta={},
                )
                for s in range(self.S)
            )
            wp = None
            if self.P > 0:
                wp = SMainWorstPoint(
                    theta_s_deg=float(self.theta_s_deg[int(self.scan_id[0])]),
                    theta_deg=float(self.theta_deg[0]),
                    phi_deg=float(self.phi_deg[0]),
                    rho=float(self.rho[0]),
                    psi_rad=float(self.psi[0]),
                    delta_id=0,
                    ratio=0.0,
                    global_point_idx=int(self.main_gidx[0]),
                    local_point_idx=int(self.local_idx_in_scan[0]),
                    scan_idx=int(self.scan_id[0]),
                )
            return SMainResult(S_main=0.0, status="OK", idx_wc_scan=0 if self.S > 0 else -1, per_scan_results=scan_results, worst=wp, meta={"nQ_non0": 0, "P_main": int(self.P)})

        best_scan_val = np.full((self.S,), -np.inf, dtype=np.float64)
        best_scan_delta = np.full((self.S,), -1, dtype=np.int64)
        best_scan_pos = np.full((self.S,), -1, dtype=np.int64)

        for did, delta in enumerate(delta_list):
            resid = self._eval_residual_for_delta(delta)
            ratio = (np.abs(resid) / self.Eref_abs_p).astype(np.float64)
            for s in range(self.S):
                pos = self.point_pos_by_scan[s]
                if pos.size == 0:
                    continue
                sub = ratio[pos]
                jloc = int(np.argmax(sub))
                val = float(sub[jloc])
                if val > float(best_scan_val[s]):
                    best_scan_val[s] = val
                    best_scan_delta[s] = int(did)
                    best_scan_pos[s] = int(pos[jloc])

        if not np.any(np.isfinite(best_scan_val)):
            return SMainResult(S_main=float("nan"), status="EMPTY_MAIN", idx_wc_scan=-1, per_scan_results=tuple(), worst=None, meta={"P_main": int(self.P)})

        idx_wc_scan = int(np.nanargmax(best_scan_val))
        S_main_val = float(best_scan_val[idx_wc_scan])
        pos_wc = int(best_scan_pos[idx_wc_scan])
        did_wc = int(best_scan_delta[idx_wc_scan])

        per_scan_results: List[SMainScanResult] = []
        for s in range(self.S):
            pos = self.point_pos_by_scan[s]
            scan_wc_pos = int(best_scan_pos[s]) if best_scan_pos[s] >= 0 else -1
            idx_wc_local = int(self.local_idx_in_scan[scan_wc_pos]) if scan_wc_pos >= 0 else -1
            idx_wc_global = int(self.main_gidx[scan_wc_pos]) if scan_wc_pos >= 0 else -1
            per_scan_results.append(
                SMainScanResult(
                    theta_s_deg=float(self.theta_s_deg[s]),
                    S_scan=float(best_scan_val[s]),
                    delta_id_wc=int(best_scan_delta[s]),
                    idx_wc_local=idx_wc_local,
                    idx_wc_global=idx_wc_global,
                    counts_main=int(pos.size),
                    meta={},
                )
            )

        worst = SMainWorstPoint(
            theta_s_deg=float(self.theta_s_deg[idx_wc_scan]),
            theta_deg=float(self.theta_deg[pos_wc]),
            phi_deg=float(self.phi_deg[pos_wc]),
            rho=float(self.rho[pos_wc]),
            psi_rad=float(self.psi[pos_wc]),
            delta_id=did_wc,
            ratio=S_main_val,
            global_point_idx=int(self.main_gidx[pos_wc]),
            local_point_idx=int(self.local_idx_in_scan[pos_wc]),
            scan_idx=int(idx_wc_scan),
        )

        return SMainResult(
            S_main=S_main_val,
            status="OK",
            idx_wc_scan=idx_wc_scan,
            per_scan_results=tuple(per_scan_results),
            worst=worst,
            meta={
                "nQ_non0": int(self._q_non0.size),
                "q_non0": self._q_non0.tolist(),
                "P_main": int(self.P),
                "S": int(self.S),
                "delta_count": int(len(delta_list)),
            },
        )

    def score_single_delta(self, *, m: np.ndarray, delta: np.ndarray) -> SMainResult:
        delta_use = _normalize_delta_vector(delta, K=self.K, name="delta")
        return self.score(m=m, delta_set=[delta_use])

    def score_many(self, candidates: Sequence[SMainCandidate]) -> Tuple[SMainResult, ...]:
        return tuple(self.score(m=c.m, delta_set=c.delta_set) for c in candidates)

    def score_anchor_offsets(
        self,
        *,
        m: np.ndarray,
        delta_gold: np.ndarray,
        q_offsets: Sequence[float] = (0.0, np.pi / 8.0, np.pi / 4.0),
        wrap_2pi: bool = True,
    ) -> SMainAnchorSweepResult:
        if len(q_offsets) == 0:
            raise ValueError("q_offsets must be non-empty")
        delta0 = _normalize_delta_vector(np.asarray(delta_gold, dtype=np.float64), K=self.K, name="delta_gold")
        q_offsets_use = [float(x) for x in q_offsets]
        if not np.all(np.isfinite(np.asarray(q_offsets_use, dtype=np.float64))):
            raise ValueError("q_offsets contains NaN or Inf")

        all_results: List[SMainAnchorResult] = []
        best_idx = -1
        best_val = float("inf")
        for i, qoff in enumerate(q_offsets_use):
            delta = delta0 + float(qoff)
            if wrap_2pi:
                delta = np.mod(delta, 2.0 * np.pi)
            res = self.score_single_delta(m=m, delta=delta)
            all_results.append(SMainAnchorResult(q_offset=float(qoff), delta=delta.copy(), result=res))
            val = float(res.S_main)
            if res.status == "OK" and val < best_val:
                best_val = val
                best_idx = int(i)

        if best_idx < 0:
            # All anchors failed. Keep the numerically smallest S_main for diagnostics.
            vals = [float(x.result.S_main) for x in all_results]
            best_idx = int(np.nanargmin(np.asarray(vals, dtype=np.float64)))

        best = all_results[best_idx]
        return SMainAnchorSweepResult(
            best_index=int(best_idx),
            best_q_offset=float(best.q_offset),
            best_delta=np.asarray(best.delta, dtype=np.float64).copy(),
            best_result=best.result,
            all_results=tuple(all_results),
        )

    def score_stage2_anchor_grid(
        self,
        *,
        m_list: Sequence[np.ndarray],
        delta_gold: np.ndarray,
        q_offsets: Sequence[float] = (0.0, np.pi / 8.0, np.pi / 4.0),
        candidate_ids: Optional[Sequence[str]] = None,
        wrap_2pi: bool = True,
    ) -> Tuple[SMainStage2ScreenRow, ...]:
        if candidate_ids is not None and len(candidate_ids) != len(m_list):
            raise ValueError("candidate_ids must have the same length as m_list")
        rows: List[SMainStage2ScreenRow] = []
        for i, m in enumerate(m_list):
            cid = str(candidate_ids[i]) if candidate_ids is not None else f"cand_{i:06d}"
            sweep = self.score_anchor_offsets(m=np.asarray(m, dtype=np.int64), delta_gold=delta_gold, q_offsets=q_offsets, wrap_2pi=wrap_2pi)
            per_anchor_S = tuple(float(x.result.S_main) for x in sweep.all_results)
            rows.append(
                SMainStage2ScreenRow(
                    candidate_id=cid,
                    m=np.asarray(m, dtype=np.int64).reshape(-1).copy(),
                    best_anchor_index=int(sweep.best_index),
                    best_q_offset=float(sweep.best_q_offset),
                    best_delta=np.asarray(sweep.best_delta, dtype=np.float64).copy(),
                    S_main=float(sweep.best_result.S_main),
                    status=str(sweep.best_result.status),
                    worst=sweep.best_result.worst,
                    per_anchor_S=per_anchor_S,
                    anchor_results=tuple(sweep.all_results),
                    meta={
                        "q_offsets": [float(x) for x in q_offsets],
                        "best_result_meta": dict(sweep.best_result.meta),
                    },
                )
            )
        return tuple(rows)

    def _eval_residual_for_delta(self, delta: np.ndarray) -> np.ndarray:
        delta = _normalize_delta_vector(np.asarray(delta, dtype=np.float64), K=self.K, name="delta")
        q = self._q_non0.astype(np.float64)
        if self._C_tensor is not None and self._frozen_m_tensor is not None and self._m_last is not None and np.all(self._frozen_m_tensor == self._m_last):
            phase = np.exp(-1j * (delta.reshape(-1, 1) * q.reshape(1, -1))).astype(np.complex128)  # (K,nQ)
            return np.einsum("kq,kqp->p", phase, self._C_tensor, optimize=True).astype(np.complex128)
        resid = np.zeros((self.P,), dtype=np.complex128)
        for k in range(self.K):
            phase = np.exp(-1j * (q * float(delta[k]))).astype(np.complex128)
            resid += phase @ self._Ck_list[k]
        return resid

    def export_debug_snapshot(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "q_non0_current": self._q_non0.tolist() if self._q_non0.size else [],
            "m_last": None if self._m_last is None else self._m_last.tolist(),
            "has_frozen_tensor": bool(self._C_tensor is not None),
            "stage2_cache_rings": sorted(int(k) for k in self._stage2_basis_cache.keys()),
            "stage2_cache_sizes": {int(k): len(v) for k, v in self._stage2_basis_cache.items()},
        }


__all__ = [
    "SMain2DConfig",
    "SMainPoolView",
    "SMainWorstPoint",
    "SMainScanResult",
    "SMainResult",
    "RingMCandidateTable",
    "Stage2MCandidateSet",
    "SMainCandidate",
    "SMainAnchorResult",
    "SMainAnchorSweepResult",
    "SMainStage2ScreenRow",
    "build_pool_view_from_scan_point_dicts",
    "smain_worst_point_to_dict",
    "smain_scan_result_to_dict",
    "smain_result_to_dict",
    "smain_screen_row_to_record",
    "smain_screen_rows_to_records",
    "compute_alias_orders_co",
    "SMain2DEvaluator",
]
