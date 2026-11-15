from __future__ import annotations

"""
passive_ar_psll_score.py

Stage-2 / Stage-3 shared AR_floor / AR_main / PSLL evaluator.

Design intent
-------------
- Use the current SR-compatible IEP interface: iep_io.IEPModel.
- Use a0/aq for co-pol and c0/cq for cx-pol.
- The caller owns the pool. This module never builds pools.
- The caller owns stage logic. This module only evaluates metrics.
- Main/side domains are provided through ARPSLLPoolView:
    * main_idx_list: rho <= rho_3dB domain
    * side_idx_list: rho > rho_null domain

Metric definitions
------------------
For candidate (m, delta), define

  AF0(p) = 1 + sum_k w_k N_k J0(beta r_k rho_p)

  E_co(p) = a0(theta_p) AF0(p)
            + sum_k sum_{q in Qa, q != 0} exp(-j q delta_k)
                sum_b C^a_{kqb}(p; m_k)

  E_cx(p) = c0(theta_p) AF0(p)
            + sum_k sum_{q in Qc, q != 0} exp(-j q delta_k)
                sum_b C^c_{kqb}(p; m_k)

where, for channel x in {a,c},

  C^x_{kqb}(p; m_k) = w_k N_k x_q(theta_p) exp(j q phi_p)
                       i^nu J_nu(beta r_k rho_p)
                       exp(j nu (psi_p - alpha_k))

and alias orders are

  nu = q m_k + ell N_k, ell in Z.

We keep the lowest-1 or lowest-2 |nu| orders according to config.

AR_floor is a candidate-independent unit-cell floor:

  eta_floor(p) = |c0(theta_p)| / (|a0(theta_p)| + eps)
  AR_floor_main = max over main points of 20 log10((1+eta)/(1-eta)).

AR_main uses full fields:

  eta(p) = |E_cx(p)| / (|E_co(p)| + eps)
  AR_main = max over main points of 20 log10((1+eta)/(1-eta)).

PSLL is strict normal PSLL:

  D_s = max_{p in main(scan s)} |E_co(p)|
  U_s = max_{p in side(scan s)} |E_co(p)|
  PSLL_s = 20 log10(U_s / (D_s + eps))
  PSLL_wc = max_s PSLL_s.

Current model conventions
-------------------------
- The center element contributes to q=0 terms only.
- eta >= 1 keeps AR at ar_bad_eta_value_db and marks HANDEDNESS_FLIP.
- Every scan used for AR must contain main points.
- Every scan used for PSLL must contain both main and side points.
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
class ARPSLL2DConfig:
    beta: float = 2.0 * np.pi
    eps: float = 1e-12

    B_alias_co: int = 2
    B_alias_cx: int = 2

    use_active_q_only: bool = True
    ar_bad_eta_value_db: float = 300.0
    verbose: bool = False


@dataclass(frozen=True)
class ARPSLLPoolView:
    theta_s_deg: np.ndarray
    theta_deg_all: np.ndarray
    phi_deg_all: np.ndarray
    rho_all: np.ndarray
    psi_rad_all: np.ndarray
    slice_list: Tuple[Tuple[int, int], ...]
    main_idx_list: Tuple[np.ndarray, ...]
    side_idx_list: Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class MetricWorstPoint:
    theta_s_deg: float
    theta_deg: float
    phi_deg: float
    rho: float
    psi_rad: float
    value: float
    global_point_idx: int
    local_point_idx: int
    scan_idx: int
    meta: Dict[str, Any]


@dataclass(frozen=True)
class ARFloor2DResult:
    ar_floor_wc_db: float
    ar_floor_s_db: np.ndarray
    eta_floor_max: float
    bad_eta_count: int
    worst_scan_idx: int
    worst_point: Optional[MetricWorstPoint]
    status: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class ARMain2DResult:
    ar_wc_db: float
    ar_s_db: np.ndarray
    eta_max: float
    bad_eta_count: int
    worst_scan_idx: int
    worst_point: Optional[MetricWorstPoint]
    status: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class PSLL2DResult:
    psll_wc_db: float
    psll_s_db: np.ndarray
    D_s: np.ndarray
    U_s: np.ndarray
    worst_scan_idx: int
    worst_side_point: Optional[MetricWorstPoint]
    worst_main_point: Optional[MetricWorstPoint]
    status: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class ARPSLL2DResult:
    ar_floor: Optional[ARFloor2DResult]
    ar_main: Optional[ARMain2DResult]
    psll: Optional[PSLL2DResult]
    status: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class ARPSLLCandidate:
    m: np.ndarray
    delta: np.ndarray
    candidate_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# Pool view helpers
# -----------------------------------------------------------------------------


def build_pool_view_from_scan_point_dicts(
    scan_point_dicts: Mapping[float, Mapping[str, np.ndarray]],
    *,
    main_mask_fn: Optional[Callable[[Mapping[str, np.ndarray]], np.ndarray]] = None,
    side_mask_fn: Optional[Callable[[Mapping[str, np.ndarray]], np.ndarray]] = None,
) -> ARPSLLPoolView:
    """
    Build an ARPSLLPoolView from per-scan point dicts.

    Required per-scan keys:
      - theta_deg, phi_deg, rho
      - psi_deg or psi_rad

    main_mask_fn:
      - if None, all points are treated as main points
      - otherwise returns a boolean mask of main points for that scan

    side_mask_fn:
      - if None, all points not in main are treated as side points
      - otherwise returns a boolean mask of side points for that scan
    """
    theta_s_list: List[float] = []
    theta_deg_all: List[np.ndarray] = []
    phi_deg_all: List[np.ndarray] = []
    rho_all: List[np.ndarray] = []
    psi_all: List[np.ndarray] = []
    slice_list: List[Tuple[int, int]] = []
    main_idx_list: List[np.ndarray] = []
    side_idx_list: List[np.ndarray] = []

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
            main_mask = np.ones((n,), dtype=bool)
        else:
            main_mask = np.asarray(main_mask_fn(pts)).reshape(-1).astype(bool)
            if main_mask.size != n:
                raise ValueError(f"scan theta_s={theta_s_deg}: main_mask_fn returned wrong length")
            main_idx = np.flatnonzero(main_mask).astype(np.int64)

        if side_mask_fn is None:
            side_mask = ~main_mask
            side_idx = np.flatnonzero(side_mask).astype(np.int64)
        else:
            side_mask = np.asarray(side_mask_fn(pts)).reshape(-1).astype(bool)
            if side_mask.size != n:
                raise ValueError(f"scan theta_s={theta_s_deg}: side_mask_fn returned wrong length")
            side_idx = np.flatnonzero(side_mask).astype(np.int64)
        overlap = main_mask & side_mask
        if np.any(overlap):
            raise ValueError(
                f"scan theta_s={theta_s_deg}: main and side masks overlap at "
                f"{np.flatnonzero(overlap).astype(int).tolist()}"
            )

        theta_s_list.append(float(theta_s_deg))
        theta_deg_all.append(theta_deg)
        phi_deg_all.append(phi_deg)
        rho_all.append(rho)
        psi_all.append(psi)
        slice_list.append((pos, pos + n))
        main_idx_list.append(main_idx)
        side_idx_list.append(side_idx)
        pos += n

    if pos == 0:
        raise ValueError("scan_point_dicts is empty")

    return ARPSLLPoolView(
        theta_s_deg=np.asarray(theta_s_list, dtype=np.float64),
        theta_deg_all=np.concatenate(theta_deg_all, axis=0).astype(np.float64),
        phi_deg_all=np.concatenate(phi_deg_all, axis=0).astype(np.float64),
        rho_all=np.concatenate(rho_all, axis=0).astype(np.float64),
        psi_rad_all=np.concatenate(psi_all, axis=0).astype(np.float64),
        slice_list=tuple((int(a), int(b)) for (a, b) in slice_list),
        main_idx_list=tuple(np.asarray(x, dtype=np.int64) for x in main_idx_list),
        side_idx_list=tuple(np.asarray(x, dtype=np.int64) for x in side_idx_list),
    )


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


def _lowest_orders_for_aN(a: int, N: int, B: int) -> Tuple[int, ...]:
    if N <= 0:
        raise ValueError("N must be positive")
    if B not in (1, 2):
        raise ValueError("B must be 1 or 2")
    l0 = int(np.rint(-float(a) / float(N)))
    cand_ls = [l0 - 3, l0 - 2, l0 - 1, l0, l0 + 1, l0 + 2, l0 + 3]
    cand = []
    seen = set()
    for ell in cand_ls:
        nu = int(a + ell * N)
        if nu in seen:
            continue
        seen.add(nu)
        cand.append((abs(nu), abs(ell), nu))
    cand.sort()
    out: List[int] = []
    for item in cand:
        nu = int(item[2])
        if nu not in out:
            out.append(nu)
        if len(out) >= B:
            break
    if not out:
        out.append(0)
    if B == 2 and len(out) == 1:
        out.append(out[0])
    return tuple(out[:B])


def compute_alias_orders(*, N: np.ndarray, m: np.ndarray, q_values: np.ndarray, B: int = 2) -> np.ndarray:
    N = np.asarray(N, dtype=np.int64).reshape(-1)
    m = np.asarray(m, dtype=np.int64).reshape(-1)
    q_values = np.asarray(q_values, dtype=np.int64).reshape(-1)
    if B not in (1, 2):
        raise ValueError("B must be 1 or 2")
    if N.size != m.size:
        raise ValueError("N and m must have same length")
    K = int(N.size)
    nQ = int(q_values.size)
    orders = np.zeros((K, nQ, B), dtype=np.int64)
    for k in range(K):
        for j in range(nQ):
            a = int(q_values[j] * m[k])
            oo = _lowest_orders_for_aN(a, int(N[k]), B)
            orders[k, j, :] = np.asarray(oo, dtype=np.int64)
    return orders


def _stable_ar_db_from_eta(eta: np.ndarray, *, eps: float, bad_value_db: float) -> Tuple[np.ndarray, int, float]:
    eta = np.asarray(eta, dtype=np.float64)
    bad = ~np.isfinite(eta) | (eta >= 1.0)
    eta_clip = np.clip(np.where(np.isfinite(eta), eta, 1.0), 0.0, 1.0 - 1e-12)
    ar = 20.0 * np.log10((1.0 + eta_clip) / np.maximum(1.0 - eta_clip, eps))
    if np.any(bad):
        ar = ar.astype(np.float64, copy=True)
        ar[bad] = float(bad_value_db)
    eta_max = float(np.nanmax(eta)) if eta.size and np.any(np.isfinite(eta)) else float("nan")
    return ar.astype(np.float64), int(np.count_nonzero(bad)), eta_max


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------


class ARPSLL2DEvaluator:
    def __init__(
        self,
        *,
        pool_view: ARPSLLPoolView,
        iep_model: iep_io.IEPModel,
        r: np.ndarray,
        N: np.ndarray,
        w: Optional[np.ndarray] = None,
        alpha_ring_rad: Optional[np.ndarray] = None,
        config: Optional[ARPSLL2DConfig] = None,
    ) -> None:
        self.pool_view = pool_view
        self.iep_model = iep_model
        self.config = config or ARPSLL2DConfig()
        self.beta = float(self.config.beta)
        self.eps = float(self.config.eps)
        self.verbose = bool(self.config.verbose)

        if int(self.config.B_alias_co) not in (1, 2):
            raise ValueError("ARPSLL2DConfig.B_alias_co must be 1 or 2")
        if int(self.config.B_alias_cx) not in (1, 2):
            raise ValueError("ARPSLL2DConfig.B_alias_cx must be 1 or 2")

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

        self._build_point_view()
        self._build_theta_key_map()
        self._build_interp_spectral_assets()
        self._build_pointwise_spectral_factors()
        self._build_reference_terms()

        self._m_last: Optional[np.ndarray] = None
        self._orders_co: Optional[np.ndarray] = None
        self._orders_cx: Optional[np.ndarray] = None
        self._Ck_co: List[np.ndarray] = []
        self._Ck_cx: List[np.ndarray] = []

        self.validate()
        if self.verbose:
            print("=== [ARPSLL2D] INIT DONE ===")
            print(f"[ARPSLL2D] S={self.S} K={self.K} P={self.P} P_main={self.main_gidx.size} P_side={self.side_gidx.size}")
            print(f"[ARPSLL2D] q_a_non0={self.q_a_non0.tolist()}")
            print(f"[ARPSLL2D] q_c_non0={self.q_c_non0.tolist()}")

    # ------------------------------------------------------------------
    # Summary / validation
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "S": int(self.S),
            "K": int(self.K),
            "P": int(self.P),
            "P_main": int(self.main_gidx.size),
            "P_side": int(self.side_gidx.size),
            "theta_key_count": int(self.theta_key_rad.size),
            "r": self.r.tolist(),
            "N": self.N.tolist(),
            "w": self.w.tolist(),
            "B_alias_co": int(self.config.B_alias_co),
            "B_alias_cx": int(self.config.B_alias_cx),
            "use_active_q_only": bool(self.config.use_active_q_only),
            "spectral_source": "iep_io.build_interp_spectral_assets",
            "co_spectral_source": "a0/aq",
            "cx_spectral_source": "c0/cq",
            "has_a0": bool(self.a0_p.size == self.P),
            "has_c0": bool(self.c0_p.size == self.P),
            "q_list_a": self.aq_q_list.tolist(),
            "q_list_c": self.cq_q_list.tolist(),
            "q_a_non0": self.q_a_non0.tolist(),
            "q_c_non0": self.q_c_non0.tolist(),
            "nQ_a_non0": int(self.q_a_non0.size),
            "nQ_c_non0": int(self.q_c_non0.size),
            "theta_s_deg": self.theta_s_deg.tolist(),
            "main_points_per_scan": [int(x.size) for x in self.main_pos_by_scan],
            "side_points_per_scan": [int(x.size) for x in self.side_pos_by_scan],
            "AF0_abs_min": float(np.min(np.abs(self.AF0_p))) if self.P else float("nan"),
            "a0_abs_min": float(np.min(np.abs(self.a0_p))) if self.P else float("nan"),
            "c0_abs_max": float(np.max(np.abs(self.c0_p))) if self.P else float("nan"),
        }

    def validate(self) -> None:
        if self.P <= 0:
            raise ValueError("No points in pool_view")
        if self.main_gidx.size <= 0:
            raise ValueError("No main points in pool_view")
        if np.any(self.N <= 0):
            raise ValueError("N must be positive")
        if not (np.all(np.isfinite(self.r)) and np.all(np.isfinite(self.w)) and np.all(np.isfinite(self.alpha_ring_rad))):
            raise ValueError("r/w/alpha_ring_rad contains NaN/Inf")
        if np.any(~np.isfinite(self.theta_rad)) or np.any(~np.isfinite(self.rho)):
            raise ValueError("Point geometry contains NaN/Inf")
        if np.any(~np.isfinite(self.phi_rad)) or np.any(~np.isfinite(self.psi)):
            raise ValueError("Point angular geometry contains NaN/Inf")
        if self.aq_phase_p_full.shape != (self.P, self.aq_q_list.size):
            raise ValueError("aq_phase_p_full shape mismatch")
        if self.cq_phase_p_full.shape != (self.P, self.cq_q_list.size):
            raise ValueError("cq_phase_p_full shape mismatch")
        if np.any(~np.isfinite(self.AF0_p)):
            raise ValueError("AF0_p contains NaN/Inf")
        if np.any(self.rho < 0.0):
            raise ValueError("rho must be non-negative")
        if len(self.slice_list) != self.S:
            raise ValueError("slice_list length mismatch")

    # ------------------------------------------------------------------
    # Fixed precompute
    # ------------------------------------------------------------------
    def _build_point_view(self) -> None:
        pv = self.pool_view
        self.theta_s_deg = np.asarray(pv.theta_s_deg, dtype=np.float64).reshape(-1)
        self.S = int(self.theta_s_deg.size)
        if self.S == 0:
            raise ValueError("pool_view.theta_s_deg must be non-empty")
        if not np.all(np.isfinite(self.theta_s_deg)):
            raise ValueError("pool_view.theta_s_deg contains NaN/Inf")

        self.theta_deg = np.asarray(pv.theta_deg_all, dtype=np.float64).reshape(-1)
        self.phi_deg = np.asarray(pv.phi_deg_all, dtype=np.float64).reshape(-1)
        self.rho = np.asarray(pv.rho_all, dtype=np.float64).reshape(-1)
        self.psi = np.asarray(pv.psi_rad_all, dtype=np.float64).reshape(-1)
        self.P = int(self.theta_deg.size)
        if self.P == 0:
            raise ValueError("pool_view contains zero points")
        if not (self.phi_deg.size == self.P and self.rho.size == self.P and self.psi.size == self.P):
            raise ValueError("pool_view point arrays length mismatch")
        if not (np.all(np.isfinite(self.theta_deg)) and np.all(np.isfinite(self.phi_deg)) and np.all(np.isfinite(self.rho)) and np.all(np.isfinite(self.psi))):
            raise ValueError("pool_view point geometry contains NaN/Inf")
        if np.any(self.rho < 0.0):
            raise ValueError("pool_view.rho_all must be non-negative")
        if len(pv.slice_list) != self.S:
            raise ValueError(f"slice_list length={len(pv.slice_list)} does not match scans={self.S}")
        if len(pv.main_idx_list) != self.S:
            raise ValueError(f"main_idx_list length={len(pv.main_idx_list)} does not match scans={self.S}")
        if len(pv.side_idx_list) != self.S:
            raise ValueError(f"side_idx_list length={len(pv.side_idx_list)} does not match scans={self.S}")

        self.theta_rad = np.deg2rad(self.theta_deg).astype(np.float64)
        self.phi_rad = _wrap_rad(np.deg2rad(self.phi_deg).astype(np.float64))
        self.slice_list = tuple((int(x), int(y)) for x, y in pv.slice_list)
        if self.slice_list[0][0] != 0:
            raise ValueError(f"slice_list must start at 0, got {self.slice_list[0][0]}")
        prev_end = 0
        for s, (a0, b0) in enumerate(self.slice_list):
            if a0 != prev_end:
                raise ValueError(f"slice_list is not contiguous at scan {s}: expected start={prev_end}, got {a0}")
            if a0 < 0 or b0 <= a0 or b0 > self.P:
                raise ValueError(f"bad slice for scan {s}: {(a0, b0)} with P={self.P}")
            prev_end = b0
        if prev_end != self.P:
            raise ValueError(f"slice_list does not cover all points: final end={prev_end}, P={self.P}")

        main_gidx: List[np.ndarray] = []
        side_gidx: List[np.ndarray] = []
        main_pos_by_scan: List[np.ndarray] = []
        side_pos_by_scan: List[np.ndarray] = []
        main_local_by_pos: Dict[int, int] = {}
        side_local_by_pos: Dict[int, int] = {}

        for s in range(self.S):
            a0, b0 = self.slice_list[s]
            n = b0 - a0
            main_local = np.asarray(pv.main_idx_list[s], dtype=np.int64).reshape(-1)
            side_local = np.asarray(pv.side_idx_list[s], dtype=np.int64).reshape(-1)
            if np.any(main_local < 0) or np.any(main_local >= n):
                raise ValueError(f"main_idx out of range for scan {s}")
            if np.any(side_local < 0) or np.any(side_local >= n):
                raise ValueError(f"side_idx out of range for scan {s}")
            if np.unique(main_local).size != main_local.size:
                raise ValueError(f"main_idx contains duplicates for scan {s}")
            if np.unique(side_local).size != side_local.size:
                raise ValueError(f"side_idx contains duplicates for scan {s}")
            overlap = np.intersect1d(main_local, side_local)
            if overlap.size:
                raise ValueError(f"main_idx and side_idx overlap for scan {s}: {overlap.astype(int).tolist()}")
            mg = (a0 + main_local).astype(np.int64)
            sg = (a0 + side_local).astype(np.int64)
            main_gidx.append(mg)
            side_gidx.append(sg)
            main_pos_by_scan.append(mg)
            side_pos_by_scan.append(sg)
            for loc, gp in zip(main_local.tolist(), mg.tolist()):
                main_local_by_pos[int(gp)] = int(loc)
            for loc, gp in zip(side_local.tolist(), sg.tolist()):
                side_local_by_pos[int(gp)] = int(loc)

        self.main_gidx_list = tuple(main_gidx)
        self.side_gidx_list = tuple(side_gidx)
        self.main_gidx = np.unique(np.concatenate(main_gidx)) if main_gidx else np.zeros((0,), dtype=np.int64)
        self.side_gidx = np.unique(np.concatenate(side_gidx)) if side_gidx else np.zeros((0,), dtype=np.int64)
        self.main_pos_by_scan = tuple(main_pos_by_scan)
        self.side_pos_by_scan = tuple(side_pos_by_scan)
        self._main_local_by_pos = main_local_by_pos
        self._side_local_by_pos = side_local_by_pos

    def _build_theta_key_map(self) -> None:
        self.theta_key_rad, self.theta_idx_p = _extract_unique_theta_keys(self.theta_rad)

    def _build_interp_spectral_assets(self) -> None:
        assets = self.iep_model.build_interp_spectral_assets(
            theta_key_rad=self.theta_key_rad,
            verbose=self.verbose,
        )
        required = ["a0", "c0", "aq_q_list", "cq_q_list", "aq_grid", "cq_grid"]
        missing = [k for k in required if k not in assets]
        if missing:
            raise RuntimeError(f"iep_model.build_interp_spectral_assets missing SR-compatible keys: {missing}")
        self.assets = assets
        self.a0_key = np.asarray(assets["a0"], dtype=np.complex128).reshape(-1)
        self.c0_key = np.asarray(assets["c0"], dtype=np.complex128).reshape(-1)
        self.aq_q_list = np.asarray(assets["aq_q_list"], dtype=np.int64).reshape(-1)
        self.cq_q_list = np.asarray(assets["cq_q_list"], dtype=np.int64).reshape(-1)
        self.aq_key_grid = np.asarray(assets["aq_grid"], dtype=np.complex128)
        self.cq_key_grid = np.asarray(assets["cq_grid"], dtype=np.complex128)
        T = int(self.theta_key_rad.size)
        if self.a0_key.size != T or self.c0_key.size != T:
            raise RuntimeError("a0/c0 shape mismatch")
        if self.aq_key_grid.shape != (T, self.aq_q_list.size):
            raise RuntimeError("aq_grid shape mismatch")
        if self.cq_key_grid.shape != (T, self.cq_q_list.size):
            raise RuntimeError("cq_grid shape mismatch")
        self.active_a_qidx_by_theta_key = tuple(
            np.asarray(x, dtype=np.int64).reshape(-1)
            for x in assets.get("active_a_qidx_by_theta", [np.arange(self.aq_q_list.size, dtype=np.int64) for _ in range(T)])
        )
        self.active_c_qidx_by_theta_key = tuple(
            np.asarray(x, dtype=np.int64).reshape(-1)
            for x in assets.get("active_c_qidx_by_theta", [np.arange(self.cq_q_list.size, dtype=np.int64) for _ in range(T)])
        )
        q0a = np.where(self.aq_q_list == 0)[0]
        q0c = np.where(self.cq_q_list == 0)[0]
        self.q0a_idx = int(q0a[0]) if q0a.size else None
        self.q0c_idx = int(q0c[0]) if q0c.size else None

    def _build_pointwise_spectral_factors(self) -> None:
        self.a0_p = self.a0_key[self.theta_idx_p].astype(np.complex128)
        self.c0_p = self.c0_key[self.theta_idx_p].astype(np.complex128)
        self.aq_p_full = self.aq_key_grid[self.theta_idx_p, :].astype(np.complex128)
        self.cq_p_full = self.cq_key_grid[self.theta_idx_p, :].astype(np.complex128)

        qa = self.aq_q_list.astype(np.float64)
        qc = self.cq_q_list.astype(np.float64)
        self.aq_phase_p_full = (self.aq_p_full * np.exp(1j * self.phi_rad.reshape(-1, 1) * qa.reshape(1, -1))).astype(np.complex128)
        self.cq_phase_p_full = (self.cq_p_full * np.exp(1j * self.phi_rad.reshape(-1, 1) * qc.reshape(1, -1))).astype(np.complex128)

        active_a = np.zeros((self.P, self.aq_q_list.size), dtype=bool)
        for tk, qidx in enumerate(self.active_a_qidx_by_theta_key):
            pts = np.flatnonzero(self.theta_idx_p == tk)
            if pts.size:
                active_a[np.ix_(pts, qidx)] = True
        if self.q0a_idx is not None:
            active_a[:, self.q0a_idx] = False

        active_c = np.zeros((self.P, self.cq_q_list.size), dtype=bool)
        for tk, qidx in enumerate(self.active_c_qidx_by_theta_key):
            pts = np.flatnonzero(self.theta_idx_p == tk)
            if pts.size:
                active_c[np.ix_(pts, qidx)] = True
        if self.q0c_idx is not None:
            active_c[:, self.q0c_idx] = False

        if bool(self.config.use_active_q_only):
            self.aq_phase_p_full = np.where(active_a, self.aq_phase_p_full, 0.0 + 0.0j)
            self.cq_phase_p_full = np.where(active_c, self.cq_phase_p_full, 0.0 + 0.0j)

        a_cols = np.where(np.any(np.abs(self.aq_phase_p_full) > 0.0, axis=0))[0].astype(np.int64)
        if self.q0a_idx is not None:
            a_cols = a_cols[a_cols != self.q0a_idx]
        c_cols = np.where(np.any(np.abs(self.cq_phase_p_full) > 0.0, axis=0))[0].astype(np.int64)
        if self.q0c_idx is not None:
            c_cols = c_cols[c_cols != self.q0c_idx]
        self.q_a_non0_col_idx = a_cols
        self.q_a_non0 = self.aq_q_list[a_cols].astype(np.int64)
        self.q_c_non0_col_idx = c_cols
        self.q_c_non0 = self.cq_q_list[c_cols].astype(np.int64)

    def _build_reference_terms(self) -> None:
        self.x_pk = (self.beta * self.r.reshape(1, -1)) * self.rho.reshape(-1, 1)
        self.J0_pk = _j0(self.x_pk).astype(np.float64)
        self.AF0_p = (1.0 + (self.J0_pk * (self.w * self.N.astype(np.float64)).reshape(1, -1)).sum(axis=1)).astype(np.float64)
        self.Ea0_p = (self.a0_p * self.AF0_p.astype(np.complex128)).astype(np.complex128)
        self.Ec0_p = (self.c0_p * self.AF0_p.astype(np.complex128)).astype(np.complex128)
        self.eta_floor_p = (np.abs(self.c0_p) / (np.abs(self.a0_p) + float(self.eps))).astype(np.float64)
        self.ar_floor_p_db, self.ar_floor_bad_count_all, self.eta_floor_max_all = _stable_ar_db_from_eta(
            self.eta_floor_p,
            eps=float(self.eps),
            bad_value_db=float(self.config.ar_bad_eta_value_db),
        )

    # ------------------------------------------------------------------
    # m and basis
    # ------------------------------------------------------------------
    def check_m_feasible(self, m: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        m = np.asarray(m, dtype=np.int64).reshape(-1)
        if m.size != self.K:
            raise ValueError(f"m must have shape ({self.K},)")
        bad_lt0 = np.where(m < 0)[0].astype(int).tolist()
        bad_half = np.where(self.N - 2 * m <= 0)[0].astype(int).tolist()
        ok = (len(bad_lt0) == 0) and (len(bad_half) == 0)
        return ok, {"bad_m_lt0": bad_lt0, "bad_N_minus_2m_le0": bad_half}

    def set_m(self, m: np.ndarray) -> None:
        m = np.asarray(m, dtype=np.int64).reshape(-1)
        if m.size != self.K:
            raise ValueError(f"m must have shape ({self.K},)")
        ok, _ = self.check_m_feasible(m)
        if not ok:
            self._m_last = m.copy()
            self._Ck_co = []
            self._Ck_cx = []
            self._orders_co = None
            self._orders_cx = None
            return
        if self._m_last is not None and np.array_equal(self._m_last, m) and self._Ck_co and self._Ck_cx:
            return
        self._m_last = m.copy()
        self._orders_co = compute_alias_orders(N=self.N, m=m, q_values=self.q_a_non0, B=int(self.config.B_alias_co)) if self.q_a_non0.size else np.zeros((self.K, 0, int(self.config.B_alias_co)), dtype=np.int64)
        self._orders_cx = compute_alias_orders(N=self.N, m=m, q_values=self.q_c_non0, B=int(self.config.B_alias_cx)) if self.q_c_non0.size else np.zeros((self.K, 0, int(self.config.B_alias_cx)), dtype=np.int64)
        self._Ck_co = []
        self._Ck_cx = []
        for k in range(self.K):
            self._Ck_co.append(self._build_ring_basis(k, self.q_a_non0, self.q_a_non0_col_idx, self._orders_co[k, :, :] if self.q_a_non0.size else np.zeros((0, int(self.config.B_alias_co)), dtype=np.int64), self.aq_phase_p_full))
            self._Ck_cx.append(self._build_ring_basis(k, self.q_c_non0, self.q_c_non0_col_idx, self._orders_cx[k, :, :] if self.q_c_non0.size else np.zeros((0, int(self.config.B_alias_cx)), dtype=np.int64), self.cq_phase_p_full))

    def _build_ring_basis(
        self,
        k: int,
        q_non0: np.ndarray,
        q_cols: np.ndarray,
        orders_qb: np.ndarray,
        phase_grid_full: np.ndarray,
    ) -> np.ndarray:
        q_non0 = np.asarray(q_non0, dtype=np.int64).reshape(-1)
        q_cols = np.asarray(q_cols, dtype=np.int64).reshape(-1)
        nQ = int(q_non0.size)
        if nQ == 0:
            return np.zeros((0, self.P), dtype=np.complex128)
        phase_sel = phase_grid_full[:, q_cols].T.astype(np.complex128)  # (nQ,P)
        Ck = np.zeros((nQ, self.P), dtype=np.complex128)
        wkNk = float(self.w[k]) * float(self.N[k])
        psi_rel = (self.psi - float(self.alpha_ring_rad[k])).astype(np.float64)
        xk = self.x_pk[:, int(k)].astype(np.float64)
        for iq in range(nQ):
            acc = np.zeros((self.P,), dtype=np.complex128)
            for nu in np.asarray(orders_qb[iq, :], dtype=np.int64).reshape(-1).tolist():
                nu = int(nu)
                Jn = _jv(nu, xk).astype(np.float64)
                phase_psi = np.exp(1j * float(nu) * psi_rel).astype(np.complex128)
                acc += _jpow_j(nu) * Jn.astype(np.complex128) * phase_psi
            Ck[iq, :] = wkNk * phase_sel[iq, :] * acc
        return Ck

    # ------------------------------------------------------------------
    # Field evaluation
    # ------------------------------------------------------------------
    def _eval_residual(self, *, delta: np.ndarray, channel: str) -> np.ndarray:
        delta = _normalize_delta_vector(np.asarray(delta, dtype=np.float64), K=self.K, name="delta")
        if channel == "co":
            q = self.q_a_non0.astype(np.float64)
            Ck_list = self._Ck_co
        elif channel == "cx":
            q = self.q_c_non0.astype(np.float64)
            Ck_list = self._Ck_cx
        else:
            raise ValueError("channel must be 'co' or 'cx'")
        if q.size == 0:
            return np.zeros((self.P,), dtype=np.complex128)
        resid = np.zeros((self.P,), dtype=np.complex128)
        for k in range(self.K):
            phase = np.exp(-1j * (q * float(delta[k]))).astype(np.complex128)
            resid += phase @ Ck_list[k]
        return resid

    def eval_fields(self, *, m: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ok, info = self.check_m_feasible(m)
        if not ok:
            raise ValueError(f"infeasible m: {info}")
        delta_use = _normalize_delta_vector(np.asarray(delta, dtype=np.float64), K=self.K, name="delta")
        self.set_m(m)
        Eco = self.Ea0_p + self._eval_residual(delta=delta_use, channel="co")
        Ecx = self.Ec0_p + self._eval_residual(delta=delta_use, channel="cx")
        return Eco.astype(np.complex128), Ecx.astype(np.complex128)

    # ------------------------------------------------------------------
    # Metric scoring
    # ------------------------------------------------------------------
    def score_ar_floor_main(self) -> ARFloor2DResult:
        if self.main_gidx.size == 0:
            return ARFloor2DResult(float("nan"), np.full((self.S,), np.nan), float("nan"), 0, -1, None, "EMPTY_MAIN", {"P_main": 0})
        empty_main_scans = [int(s) for s, g in enumerate(self.main_gidx_list) if g.size == 0]
        if empty_main_scans:
            return ARFloor2DResult(
                float("nan"), np.full((self.S,), np.nan), float("nan"), 0, -1, None,
                "EMPTY_MAIN_SCAN", {"empty_scan_indices": empty_main_scans},
            )
        ar_s = np.full((self.S,), np.nan, dtype=np.float64)
        worst_val = -np.inf
        worst_s = -1
        worst_g = -1
        for s in range(self.S):
            g = self.main_gidx_list[s]
            if g.size == 0:
                continue
            vals = self.ar_floor_p_db[g]
            j = int(np.nanargmax(vals))
            ar_s[s] = float(vals[j])
            if ar_s[s] > worst_val:
                worst_val = float(ar_s[s])
                worst_s = int(s)
                worst_g = int(g[j])
        wp = self._make_worst_point(worst_g, worst_s, worst_val, domain="main", extra={"metric": "AR_floor_db", "eta_floor": float(self.eta_floor_p[worst_g]) if worst_g >= 0 else None})
        bad_count = int(np.count_nonzero(~np.isfinite(self.eta_floor_p[self.main_gidx]) | (self.eta_floor_p[self.main_gidx] >= 1.0)))
        eta_max = float(np.nanmax(self.eta_floor_p[self.main_gidx])) if self.main_gidx.size else float("nan")
        return ARFloor2DResult(
            ar_floor_wc_db=float(np.nanmax(ar_s)) if np.any(np.isfinite(ar_s)) else float("nan"),
            ar_floor_s_db=ar_s,
            eta_floor_max=eta_max,
            bad_eta_count=bad_count,
            worst_scan_idx=int(worst_s),
            worst_point=wp,
            status="HANDEDNESS_FLIP" if bad_count > 0 else "OK",
            meta={"P_main": int(self.main_gidx.size), "definition": "max_main_AR_from_abs_c0_over_abs_a0"},
        )

    def score_ar_main(self, *, m: np.ndarray, delta: np.ndarray) -> ARMain2DResult:
        ok, info = self.check_m_feasible(m)
        if not ok:
            return ARMain2DResult(float("inf"), np.full((self.S,), np.inf), float("inf"), 0, -1, None, "INFEASIBLE_M", info)
        if self.main_gidx.size == 0:
            return ARMain2DResult(float("nan"), np.full((self.S,), np.nan), float("nan"), 0, -1, None, "EMPTY_MAIN", {"P_main": 0})
        empty_main_scans = [int(s) for s, g in enumerate(self.main_gidx_list) if g.size == 0]
        if empty_main_scans:
            return ARMain2DResult(
                float("nan"), np.full((self.S,), np.nan), float("nan"), 0, -1, None,
                "EMPTY_MAIN_SCAN", {"empty_scan_indices": empty_main_scans},
            )
        Eco, Ecx = self.eval_fields(m=m, delta=delta)
        idx = self.main_gidx
        eta = np.abs(Ecx[idx]) / (np.abs(Eco[idx]) + float(self.eps))
        ar_vals, bad_count, eta_max = _stable_ar_db_from_eta(eta, eps=float(self.eps), bad_value_db=float(self.config.ar_bad_eta_value_db))
        ar_map = np.full((self.P,), np.nan, dtype=np.float64)
        eta_map = np.full((self.P,), np.nan, dtype=np.float64)
        ar_map[idx] = ar_vals
        eta_map[idx] = eta
        ar_s = np.full((self.S,), np.nan, dtype=np.float64)
        worst_val = -np.inf
        worst_s = -1
        worst_g = -1
        for s in range(self.S):
            g = self.main_gidx_list[s]
            if g.size == 0:
                continue
            vals = ar_map[g]
            j = int(np.nanargmax(vals))
            ar_s[s] = float(vals[j])
            if ar_s[s] > worst_val:
                worst_val = float(ar_s[s])
                worst_s = int(s)
                worst_g = int(g[j])
        wp = self._make_worst_point(worst_g, worst_s, worst_val, domain="main", extra={"metric": "AR_main_db", "eta": float(eta_map[worst_g]) if worst_g >= 0 else None})
        return ARMain2DResult(
            ar_wc_db=float(np.nanmax(ar_s)) if np.any(np.isfinite(ar_s)) else float("nan"),
            ar_s_db=ar_s,
            eta_max=float(eta_max),
            bad_eta_count=int(bad_count),
            worst_scan_idx=int(worst_s),
            worst_point=wp,
            status="HANDEDNESS_FLIP" if bad_count > 0 else "OK",
            meta={"P_main": int(self.main_gidx.size), "definition": "max_main_AR_from_full_Ecx_over_full_Eco"},
        )

    def score_psll(self, *, m: np.ndarray, delta: np.ndarray) -> PSLL2DResult:
        ok, info = self.check_m_feasible(m)
        if not ok:
            return PSLL2DResult(float("inf"), np.full((self.S,), np.inf), np.full((self.S,), np.nan), np.full((self.S,), np.nan), -1, None, None, "INFEASIBLE_M", info)
        if self.main_gidx.size == 0:
            return PSLL2DResult(float("nan"), np.full((self.S,), np.nan), np.full((self.S,), np.nan), np.full((self.S,), np.nan), -1, None, None, "EMPTY_MAIN", {"P_main": 0})
        if self.side_gidx.size == 0:
            return PSLL2DResult(float("nan"), np.full((self.S,), np.nan), np.full((self.S,), np.nan), np.full((self.S,), np.nan), -1, None, None, "EMPTY_SIDE", {"P_side": 0})
        empty_main_scans = [int(s) for s, g in enumerate(self.main_gidx_list) if g.size == 0]
        if empty_main_scans:
            return PSLL2DResult(
                float("nan"), np.full((self.S,), np.nan), np.full((self.S,), np.nan), np.full((self.S,), np.nan),
                -1, None, None, "EMPTY_MAIN_SCAN", {"empty_scan_indices": empty_main_scans},
            )
        empty_side_scans = [int(s) for s, g in enumerate(self.side_gidx_list) if g.size == 0]
        if empty_side_scans:
            return PSLL2DResult(
                float("nan"), np.full((self.S,), np.nan), np.full((self.S,), np.nan), np.full((self.S,), np.nan),
                -1, None, None, "EMPTY_SIDE_SCAN", {"empty_scan_indices": empty_side_scans},
            )
        Eco, _ = self.eval_fields(m=m, delta=delta)
        absE = np.abs(Eco).astype(np.float64)
        D_s = np.full((self.S,), np.nan, dtype=np.float64)
        U_s = np.full((self.S,), np.nan, dtype=np.float64)
        idx_main_wc = np.full((self.S,), -1, dtype=np.int64)
        idx_side_wc = np.full((self.S,), -1, dtype=np.int64)
        for s in range(self.S):
            gm = self.main_gidx_list[s]
            gs = self.side_gidx_list[s]
            if gm.size:
                jm = int(np.argmax(absE[gm]))
                D_s[s] = float(absE[gm[jm]])
                idx_main_wc[s] = int(gm[jm])
            if gs.size:
                js = int(np.argmax(absE[gs]))
                U_s[s] = float(absE[gs[js]])
                idx_side_wc[s] = int(gs[js])
        psll_s = 20.0 * np.log10(np.maximum(U_s, 1e-300) / np.maximum(D_s, float(self.eps)))
        if not np.any(np.isfinite(psll_s)):
            return PSLL2DResult(float("nan"), psll_s, D_s, U_s, -1, None, None, "EMPTY_VALID_SCAN", {"P_main": int(self.main_gidx.size), "P_side": int(self.side_gidx.size)})
        worst_s = int(np.nanargmax(psll_s))
        worst_side = self._make_worst_point(int(idx_side_wc[worst_s]), worst_s, float(U_s[worst_s]), domain="side", extra={"metric": "U_side_abs_Eco"})
        worst_main = self._make_worst_point(int(idx_main_wc[worst_s]), worst_s, float(D_s[worst_s]), domain="main", extra={"metric": "D_main_abs_Eco"})
        return PSLL2DResult(
            psll_wc_db=float(psll_s[worst_s]),
            psll_s_db=psll_s.astype(np.float64),
            D_s=D_s,
            U_s=U_s,
            worst_scan_idx=int(worst_s),
            worst_side_point=worst_side,
            worst_main_point=worst_main,
            status="OK",
            meta={
                "P_main": int(self.main_gidx.size),
                "P_side": int(self.side_gidx.size),
                "definition": "strict_per_scan_max_side_over_max_main_then_worst_scan",
            },
        )

    def score_all(self, *, m: np.ndarray, delta: np.ndarray, include_floor: bool = True) -> ARPSLL2DResult:
        ar_floor = self.score_ar_floor_main() if include_floor else None
        ar_main = self.score_ar_main(m=m, delta=delta)
        psll = self.score_psll(m=m, delta=delta)
        status = "OK" if ar_main.status == "OK" and psll.status == "OK" and (ar_floor is None or ar_floor.status == "OK") else "PARTIAL"
        return ARPSLL2DResult(
            ar_floor=ar_floor,
            ar_main=ar_main,
            psll=psll,
            status=status,
            meta={"summary": self.summary()},
        )

    def score_many(self, candidates: Sequence[ARPSLLCandidate], *, include_floor: bool = True) -> Tuple[ARPSLL2DResult, ...]:
        return tuple(self.score_all(m=c.m, delta=c.delta, include_floor=include_floor) for c in candidates)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _make_worst_point(self, gidx: int, scan_idx: int, value: float, *, domain: str, extra: Optional[Dict[str, Any]] = None) -> Optional[MetricWorstPoint]:
        if gidx < 0 or gidx >= self.P or scan_idx < 0 or scan_idx >= self.S:
            return None
        local = self._main_local_by_pos.get(int(gidx), self._side_local_by_pos.get(int(gidx), -1))
        meta = {"domain": str(domain)}
        if extra:
            meta.update(dict(extra))
        return MetricWorstPoint(
            theta_s_deg=float(self.theta_s_deg[int(scan_idx)]),
            theta_deg=float(self.theta_deg[int(gidx)]),
            phi_deg=float(self.phi_deg[int(gidx)]),
            rho=float(self.rho[int(gidx)]),
            psi_rad=float(self.psi[int(gidx)]),
            value=float(value),
            global_point_idx=int(gidx),
            local_point_idx=int(local),
            scan_idx=int(scan_idx),
            meta=meta,
        )

    def export_debug_snapshot(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "m_last": None if self._m_last is None else self._m_last.tolist(),
            "orders_co_shape": None if self._orders_co is None else list(self._orders_co.shape),
            "orders_cx_shape": None if self._orders_cx is None else list(self._orders_cx.shape),
        }


# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------


def _wp_to_dict(wp: Optional[MetricWorstPoint]) -> Optional[Dict[str, Any]]:
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
        "meta": dict(wp.meta),
    }


def ar_floor_result_to_dict(res: ARFloor2DResult) -> Dict[str, Any]:
    return {
        "ar_floor_wc_db": float(res.ar_floor_wc_db),
        "ar_floor_s_db": np.asarray(res.ar_floor_s_db, dtype=np.float64).tolist(),
        "eta_floor_max": float(res.eta_floor_max),
        "bad_eta_count": int(res.bad_eta_count),
        "worst_scan_idx": int(res.worst_scan_idx),
        "worst_point": _wp_to_dict(res.worst_point),
        "status": str(res.status),
        "meta": dict(res.meta),
    }


def ar_main_result_to_dict(res: ARMain2DResult) -> Dict[str, Any]:
    return {
        "ar_wc_db": float(res.ar_wc_db),
        "ar_s_db": np.asarray(res.ar_s_db, dtype=np.float64).tolist(),
        "eta_max": float(res.eta_max),
        "bad_eta_count": int(res.bad_eta_count),
        "worst_scan_idx": int(res.worst_scan_idx),
        "worst_point": _wp_to_dict(res.worst_point),
        "status": str(res.status),
        "meta": dict(res.meta),
    }


def psll_result_to_dict(res: PSLL2DResult) -> Dict[str, Any]:
    return {
        "psll_wc_db": float(res.psll_wc_db),
        "psll_s_db": np.asarray(res.psll_s_db, dtype=np.float64).tolist(),
        "D_s": np.asarray(res.D_s, dtype=np.float64).tolist(),
        "U_s": np.asarray(res.U_s, dtype=np.float64).tolist(),
        "worst_scan_idx": int(res.worst_scan_idx),
        "worst_side_point": _wp_to_dict(res.worst_side_point),
        "worst_main_point": _wp_to_dict(res.worst_main_point),
        "status": str(res.status),
        "meta": dict(res.meta),
    }


def arpsll_result_to_dict(res: ARPSLL2DResult) -> Dict[str, Any]:
    return {
        "status": str(res.status),
        "ar_floor": None if res.ar_floor is None else ar_floor_result_to_dict(res.ar_floor),
        "ar_main": None if res.ar_main is None else ar_main_result_to_dict(res.ar_main),
        "psll": None if res.psll is None else psll_result_to_dict(res.psll),
        "meta": dict(res.meta),
    }


__all__ = [
    "ARPSLL2DConfig",
    "ARPSLLPoolView",
    "MetricWorstPoint",
    "ARFloor2DResult",
    "ARMain2DResult",
    "PSLL2DResult",
    "ARPSLL2DResult",
    "ARPSLLCandidate",
    "build_pool_view_from_scan_point_dicts",
    "compute_alias_orders",
    "ARPSLL2DEvaluator",
    "ar_floor_result_to_dict",
    "ar_main_result_to_dict",
    "psll_result_to_dict",
    "arpsll_result_to_dict",
]
