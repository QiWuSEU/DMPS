from __future__ import annotations

"""
passive_field_batch.py

Shared batch field backend for the current Q0CRA/A0CRA Stage-2 flow.

Purpose
-------
This module is a computation backend, not a runner.  It precomputes the
candidate-independent and ring-option-dependent terms that are repeatedly used
by Stage-2 metrics, then evaluates many (m, delta) candidates in chunks.

It follows the same physical model as score_smain_2d.py and
score_ar_psll_2d.py:

  Eco = a0 * AF0 + sum_k sum_q exp(-j q delta_k) Cco[k, m_k, q]
  Ecx = c0 * AF0 + sum_k sum_q exp(-j q delta_k) Ccx[k, m_k, q]

where Cco/Ccx are the lowest-|nu| alias-truncated Bessel spectral bases for a
single ring and a single local m_k option.

Intended use
------------
- Build one backend for Pool-A to batch score Stage-1 S_main + AR_main.
- Build one backend for Pool-C to batch score Stage-2 PSLL and Final grids.
- Keep the existing score_smain_2d.py / score_ar_psll_2d.py as reference
  evaluators until this backend is validated and wired in.

Current model conventions
-------------------------
1. The center element contributes to the q=0 AF0 term only.
2. Nonzero aq/cq residual terms are summed over the circular rings.
3. B_alias_co and B_alias_cx are restricted to 1 or 2.
4. eta >= 1 is treated as target-handedness failure and mapped to
   ar_bad_eta_value_db.
5. AR_floor is a simple |c0|/|a0|-based prediction metric.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import time

import numpy as np
from scipy.special import j0 as _j0, jv as _jv

from element_spectrum import iep_model as iep_io

TWO_PI = 2.0 * math.pi
GOLDEN_GAMMA = math.pi * (3.0 - math.sqrt(5.0))


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage2BatchConfig:
    beta: float = 2.0 * np.pi
    eps: float = 1e-12

    B_alias_co: int = 2
    B_alias_cx: int = 2
    use_active_q_only: bool = True

    ar_bad_eta_value_db: float = 300.0

    stage1_chunk_size: int = 128
    stage2_chunk_size: int = 8
    final_chunk_size: int = 8

    cache_bessel_by_order: bool = True
    cache_ring_option_basis: bool = True
    cache_delta_phase: bool = True

    verbose: bool = True


@dataclass(frozen=True)
class Stage2BatchStage1Result:
    rows: Tuple[Dict[str, Any], ...]
    S_star: float
    AR_star_db: float
    ar_floor_wc_db: float
    status: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class Stage2BatchPSLLResult:
    rows: Tuple[Dict[str, Any], ...]
    status: str
    meta: Dict[str, Any]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _wrap_rad(phi: np.ndarray) -> np.ndarray:
    y = np.mod(np.asarray(phi, dtype=np.float64), TWO_PI)
    y[y == TWO_PI] = 0.0
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
    theta_rad = np.asarray(theta_rad, dtype=np.float64).reshape(-1)
    key = np.round(theta_rad, decimals)
    uniq, inv = np.unique(key, return_inverse=True)
    theta_key = np.zeros((uniq.size,), dtype=np.float64)
    for i in range(int(uniq.size)):
        theta_key[i] = float(theta_rad[np.where(inv == i)[0][0]])
    return theta_key, inv.astype(np.int64)


def lowest_orders_for_aN(a: int, N: int, B: int) -> Tuple[int, ...]:
    if int(N) <= 0:
        raise ValueError("N must be positive")
    if int(B) not in (1, 2):
        raise ValueError("B must be 1 or 2")
    l0 = int(np.rint(-float(a) / float(N)))
    cand_ls = [l0 - 3, l0 - 2, l0 - 1, l0, l0 + 1, l0 + 2, l0 + 3]
    cand: List[Tuple[int, int, int]] = []
    seen = set()
    for ell in cand_ls:
        nu = int(a + ell * int(N))
        if nu in seen:
            continue
        seen.add(nu)
        cand.append((abs(nu), abs(int(ell)), nu))
    cand.sort()
    out: List[int] = []
    for _, _, nu in cand:
        if int(nu) not in out:
            out.append(int(nu))
        if len(out) >= int(B):
            break
    if not out:
        out.append(0)
    if int(B) == 2 and len(out) == 1:
        out.append(out[0])
    return tuple(out[: int(B)])


def compute_alias_orders(Nk: int, mk: int, q_values: np.ndarray, B: int) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=np.int64).reshape(-1)
    orders = np.zeros((q_values.size, int(B)), dtype=np.int64)
    for iq, q in enumerate(q_values.tolist()):
        a = int(q) * int(mk)
        orders[iq, :] = np.asarray(lowest_orders_for_aN(a, int(Nk), int(B)), dtype=np.int64)
    return orders


def stable_ar_db_from_eta(eta: np.ndarray, *, eps: float, bad_value_db: float) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    eta = np.asarray(eta, dtype=np.float64)
    bad = ~np.isfinite(eta) | (eta >= 1.0)
    eta_clip = np.clip(np.where(np.isfinite(eta), eta, 1.0), 0.0, 1.0 - 1e-12)
    ar = 20.0 * np.log10((1.0 + eta_clip) / np.maximum(1.0 - eta_clip, float(eps)))
    if np.any(bad):
        ar = ar.astype(np.float64, copy=True)
        ar[bad] = float(bad_value_db)
    return ar.astype(np.float64), bad.astype(bool), int(np.count_nonzero(bad)), eta


def make_delta_gold(K: int, *, offset_rank: float = 0.0) -> np.ndarray:
    k = np.arange(int(K), dtype=np.float64)
    return np.mod((k + float(offset_rank)) * GOLDEN_GAMMA, TWO_PI)


def make_delta_with_anchor(delta_gold: np.ndarray, q_anchor: float) -> np.ndarray:
    return np.mod(np.asarray(delta_gold, dtype=np.float64).reshape(-1) + float(q_anchor), TWO_PI)


def normalized_ring_axis(K: int) -> np.ndarray:
    if int(K) <= 1:
        return np.zeros((int(K),), dtype=np.float64)
    return (-1.0 + 2.0 * np.arange(int(K), dtype=np.float64) / float(int(K) - 1)).astype(np.float64)


def _as_list_jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _as_list_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_as_list_jsonable(v) for v in x]
    return x


def _validate_channels(channels: str) -> str:
    channels = str(channels)
    allowed = {"both", "co", "cx"}
    if channels not in allowed:
        raise ValueError(f"channels must be one of {sorted(allowed)}, got {channels!r}")
    return channels


def _validate_and_wrap_delta_batch(delta_batch: np.ndarray, *, K: int) -> np.ndarray:
    delta = np.asarray(delta_batch, dtype=np.float64)
    if delta.ndim == 1:
        delta = delta.reshape(1, -1)
    if delta.ndim != 2:
        raise ValueError(f"delta_batch must be a 1D or 2D array, got ndim={delta.ndim}")
    if delta.shape[1] != int(K):
        raise ValueError(f"delta_batch must have K={int(K)} columns, got shape={delta.shape}")
    if not np.all(np.isfinite(delta)):
        bad = np.argwhere(~np.isfinite(delta))
        shown = bad[:10].tolist()
        suffix = "" if bad.shape[0] <= 10 else f" (showing first 10 of {bad.shape[0]})"
        raise ValueError(f"delta_batch contains NaN or Inf at indices {shown}{suffix}")
    return np.mod(delta, TWO_PI).astype(np.float64, copy=False)


# -----------------------------------------------------------------------------
# Backend
# -----------------------------------------------------------------------------


class Stage2FieldBatchBackend:
    """Shared batch field backend with compatibility metric reductions.

    Metric definitions remain compatible with the current score_smain_2d.py
    and score_ar_psll_2d.py implementations.

    Parameters
    ----------
    pool_view:
        An object with the same attributes as score_ar_psll_2d.ARPSLLPoolView.
        It must contain all points, scan slices, main_idx_list and side_idx_list.
    iep_model:
        iep_io.IEPModel instance.
    r, N:
        Fixed stage-1 geometry.
    m_options_by_ring:
        Mapping ring index -> local m candidates to precompute. Missing m values
        requested later are built lazily.
    """

    def __init__(
        self,
        *,
        pool_view: Any,
        iep_model: iep_io.IEPModel,
        r: np.ndarray,
        N: np.ndarray,
        m_options_by_ring: Optional[Mapping[int, Sequence[int]]] = None,
        w: Optional[np.ndarray] = None,
        alpha_ring_rad: Optional[np.ndarray] = None,
        config: Optional[Stage2BatchConfig] = None,
    ) -> None:
        self.pool_view = pool_view
        self.iep_model = iep_model
        self.config = config or Stage2BatchConfig()
        if int(self.config.B_alias_co) not in (1, 2) or int(self.config.B_alias_cx) not in (1, 2):
            raise ValueError("B_alias_co/B_alias_cx must be 1 or 2")
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

        self.m_options_by_ring: Dict[int, np.ndarray] = {}
        if m_options_by_ring:
            for k, vals in m_options_by_ring.items():
                kk = int(k)
                if kk < 0 or kk >= self.K:
                    raise ValueError(f"ring index out of range in m_options_by_ring: {kk}")
                self.m_options_by_ring[kk] = np.asarray(vals, dtype=np.int64).reshape(-1)

        # Built by precompute_all()
        self._precomputed = False
        self.Cco: Dict[Tuple[int, int], np.ndarray] = {}
        self.Ccx: Dict[Tuple[int, int], np.ndarray] = {}
        self.orders_co: Dict[Tuple[int, int], np.ndarray] = {}
        self.orders_cx: Dict[Tuple[int, int], np.ndarray] = {}
        self.B_cache: Dict[Tuple[int, int], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public setup / summary
    # ------------------------------------------------------------------

    def precompute_all(self) -> "Stage2FieldBatchBackend":
        t0 = time.perf_counter()
        self._build_point_view()
        self._build_theta_key_map()
        self._build_interp_spectral_assets()
        self._build_pointwise_spectral_factors()
        self._build_reference_terms()
        self._build_all_ring_option_basis()
        self._precomputed = True
        if self.verbose:
            print(
                f"[Stage2Batch] precompute_all DONE P={self.P} S={self.S} K={self.K} "
                f"q_a={self.q_a_non0.size} q_c={self.q_c_non0.size} "
                f"basis_co={len(self.Cco)} basis_cx={len(self.Ccx)} "
                f"mem_est={self.estimate_memory_mb():.1f} MB elapsed={time.perf_counter()-t0:.2f}s",
                flush=True,
            )
        return self

    @property
    def q_alias(self) -> np.ndarray:
        if not hasattr(self, "q_a_non0"):
            raise RuntimeError("precompute_all() must be called before q_alias")
        return np.unique(np.concatenate([self.q_a_non0, self.q_c_non0]).astype(np.int64))

    def summary(self) -> Dict[str, Any]:
        self._require_ready()
        return {
            "P": int(self.P),
            "S": int(self.S),
            "K": int(self.K),
            "r": self.r.tolist(),
            "N": self.N.tolist(),
            "w": self.w.tolist(),
            "theta_s_deg": self.theta_s_deg.tolist(),
            "P_main": int(self.main_gidx.size),
            "P_side": int(self.side_gidx.size),
            "main_points_per_scan": [int(x.size) for x in self.main_gidx_list],
            "side_points_per_scan": [int(x.size) for x in self.side_gidx_list],
            "theta_key_count": int(self.theta_key_rad.size),
            "q_a_non0": self.q_a_non0.tolist(),
            "q_c_non0": self.q_c_non0.tolist(),
            "q_alias": self.q_alias.tolist(),
            "B_alias_co": int(self.config.B_alias_co),
            "B_alias_cx": int(self.config.B_alias_cx),
            "use_active_q_only": bool(self.config.use_active_q_only),
            "basis_count_co": int(len(self.Cco)),
            "basis_count_cx": int(len(self.Ccx)),
            "m_options_by_ring": {int(k): np.asarray(v, dtype=np.int64).tolist() for k, v in self.m_options_by_ring.items()},
            "AF0_abs_min": float(np.min(np.abs(self.AF0_p))) if self.P else float("nan"),
            "a0_abs_min": float(np.min(np.abs(self.a0_p))) if self.P else float("nan"),
            "c0_abs_max": float(np.max(np.abs(self.c0_p))) if self.P else float("nan"),
            "memory_estimate_mb": float(self.estimate_memory_mb()),
            "ar_bad_eta_value_db": float(self.config.ar_bad_eta_value_db),
            "slice_count": int(len(self.slice_list)),
            "model_conventions": {
                "center_element_model": "q0_only",
                "alias_relation": "nu = q*m + p*N",
                "delta_phase": "exp(-1j*q*delta_k)",
                "bessel_phase": "j**nu * exp(1j*nu*(psi-alpha_k))",
                "ar_bad_eta_rule": "eta >= 1 -> ar_bad_eta_value_db",
                "ar_floor_definition": "|c0| / |a0|",
            },
        }

    def estimate_memory_mb(self) -> float:
        total = 0
        for name in (
            "theta_deg", "phi_deg", "rho", "psi", "theta_rad", "phi_rad",
            "a0_p", "c0_p", "aq_phase_p_full", "cq_phase_p_full", "Ea0_p", "Ec0_p", "AF0_p",
        ):
            if hasattr(self, name):
                total += int(getattr(self, name).nbytes)
        for d in (self.Cco, self.Ccx, self.B_cache):
            total += sum(int(v.nbytes) for v in d.values())
        return total / (1024.0 * 1024.0)

    def export_cache_summary(self, path: str | Path) -> None:
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(_as_list_jsonable(self.summary()), ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Fixed precompute
    # ------------------------------------------------------------------

    def _build_point_view(self) -> None:
        pv = self.pool_view

        required_attrs = (
            "theta_s_deg",
            "theta_deg_all",
            "phi_deg_all",
            "rho_all",
            "psi_rad_all",
            "slice_list",
            "main_idx_list",
            "side_idx_list",
        )
        for name in required_attrs:
            if not hasattr(pv, name):
                raise AttributeError(f"pool_view is missing required attribute {name!r}")

        self.theta_s_deg = np.asarray(pv.theta_s_deg, dtype=np.float64).reshape(-1)
        self.S = int(self.theta_s_deg.size)
        if self.S == 0:
            raise ValueError("pool_view.theta_s_deg must be non-empty")

        slice_list_raw = tuple(pv.slice_list)
        main_idx_list_raw = tuple(pv.main_idx_list)
        side_idx_list_raw = tuple(pv.side_idx_list)
        if len(slice_list_raw) != self.S:
            raise ValueError(f"pool_view.slice_list length={len(slice_list_raw)} does not match S={self.S}")
        if len(main_idx_list_raw) != self.S:
            raise ValueError(f"pool_view.main_idx_list length={len(main_idx_list_raw)} does not match S={self.S}")
        if len(side_idx_list_raw) != self.S:
            raise ValueError(f"pool_view.side_idx_list length={len(side_idx_list_raw)} does not match S={self.S}")

        self.theta_deg = np.asarray(pv.theta_deg_all, dtype=np.float64).reshape(-1)
        self.phi_deg = np.asarray(pv.phi_deg_all, dtype=np.float64).reshape(-1)
        self.rho = np.asarray(pv.rho_all, dtype=np.float64).reshape(-1)
        self.psi = np.asarray(pv.psi_rad_all, dtype=np.float64).reshape(-1)
        self.P = int(self.theta_deg.size)
        if self.P == 0:
            raise ValueError("pool_view point arrays must be non-empty")
        if not (self.phi_deg.size == self.P and self.rho.size == self.P and self.psi.size == self.P):
            raise ValueError("pool_view point arrays length mismatch")
        for name, arr in (
            ("theta_deg_all", self.theta_deg),
            ("phi_deg_all", self.phi_deg),
            ("rho_all", self.rho),
            ("psi_rad_all", self.psi),
        ):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"pool_view.{name} contains NaN or Inf")
        if np.any(self.rho < 0.0):
            raise ValueError("pool_view.rho_all must be non-negative")
        self.theta_rad = np.deg2rad(self.theta_deg).astype(np.float64)
        self.phi_rad = _wrap_rad(np.deg2rad(self.phi_deg).astype(np.float64))

        main_gidx: List[np.ndarray] = []
        side_gidx: List[np.ndarray] = []
        main_local: Dict[int, int] = {}
        side_local: Dict[int, int] = {}
        self.slice_list = tuple((int(a), int(b)) for a, b in slice_list_raw)
        if self.slice_list[0][0] != 0:
            raise ValueError(f"pool_view.slice_list must start at 0, got {self.slice_list[0][0]}")
        prev_end = 0
        for s, (a, b) in enumerate(self.slice_list):
            if not (0 <= int(a) < int(b) <= self.P):
                raise ValueError(f"invalid slice for scan {s}: ({a}, {b}), P={self.P}")
            if int(a) != int(prev_end):
                raise ValueError(
                    f"pool_view.slice_list is not contiguous before scan {s}: "
                    f"previous end={prev_end}, current start={a}"
                )
            prev_end = int(b)
        if int(prev_end) != self.P:
            raise ValueError(f"pool_view.slice_list ends at {prev_end}, but P={self.P}")

        for s in range(self.S):
            a, b = self.slice_list[s]
            n = b - a
            mi = np.asarray(main_idx_list_raw[s], dtype=np.int64).reshape(-1)
            si = np.asarray(side_idx_list_raw[s], dtype=np.int64).reshape(-1)
            if np.any(mi < 0) or np.any(mi >= n):
                raise ValueError(f"main_idx out of range for scan {s}")
            if np.any(si < 0) or np.any(si >= n):
                raise ValueError(f"side_idx out of range for scan {s}")
            if np.unique(mi).size != mi.size:
                raise ValueError(f"duplicated main_idx for scan {s}")
            if np.unique(si).size != si.size:
                raise ValueError(f"duplicated side_idx for scan {s}")
            overlap = np.intersect1d(mi, si)
            if overlap.size:
                raise ValueError(f"main_idx and side_idx overlap for scan {s}: {overlap[:10].tolist()}")
            mg = (a + mi).astype(np.int64)
            sg = (a + si).astype(np.int64)
            main_gidx.append(mg)
            side_gidx.append(sg)
            for loc, gp in zip(mi.tolist(), mg.tolist()):
                main_local[int(gp)] = int(loc)
            for loc, gp in zip(si.tolist(), sg.tolist()):
                side_local[int(gp)] = int(loc)
        self.main_gidx_list = tuple(main_gidx)
        self.side_gidx_list = tuple(side_gidx)
        self.main_gidx = np.unique(np.concatenate(main_gidx)) if main_gidx else np.zeros((0,), dtype=np.int64)
        self.side_gidx = np.unique(np.concatenate(side_gidx)) if side_gidx else np.zeros((0,), dtype=np.int64)
        if self.main_gidx.size == 0:
            raise ValueError("pool_view contains no main-domain points")
        self._main_local_by_pos = main_local
        self._side_local_by_pos = side_local

    def _build_theta_key_map(self) -> None:
        self.theta_key_rad, self.theta_idx_p = _extract_unique_theta_keys(self.theta_rad)

    def _build_interp_spectral_assets(self) -> None:
        assets = self.iep_model.build_interp_spectral_assets(theta_key_rad=self.theta_key_rad, verbose=self.verbose)
        required = ["a0", "c0", "aq_q_list", "cq_q_list", "aq_grid", "cq_grid"]
        missing = [k for k in required if k not in assets]
        if missing:
            raise RuntimeError(f"IEP assets missing keys: {missing}")
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
        c_cols = np.where(np.any(np.abs(self.cq_phase_p_full) > 0.0, axis=0))[0].astype(np.int64)
        if self.q0a_idx is not None:
            a_cols = a_cols[a_cols != self.q0a_idx]
        if self.q0c_idx is not None:
            c_cols = c_cols[c_cols != self.q0c_idx]
        self.q_a_non0_col_idx = a_cols
        self.q_c_non0_col_idx = c_cols
        self.q_a_non0 = self.aq_q_list[a_cols].astype(np.int64)
        self.q_c_non0 = self.cq_q_list[c_cols].astype(np.int64)

    def _build_reference_terms(self) -> None:
        self.x_pk = (self.beta * self.r.reshape(1, -1)) * self.rho.reshape(-1, 1)
        self.J0_pk = _j0(self.x_pk).astype(np.float64)
        self.AF0_p = (1.0 + (self.J0_pk * (self.w * self.N.astype(np.float64)).reshape(1, -1)).sum(axis=1)).astype(np.float64)
        self.Ea0_p = (self.a0_p * self.AF0_p.astype(np.complex128)).astype(np.complex128)
        self.Ec0_p = (self.c0_p * self.AF0_p.astype(np.complex128)).astype(np.complex128)
        self.Eref_abs_p = (np.abs(self.Ea0_p) + float(self.eps)).astype(np.float64)
        self.eta_floor_p = (np.abs(self.c0_p) / (np.abs(self.a0_p) + float(self.eps))).astype(np.float64)
        self.ar_floor_p_db, _, _, _ = stable_ar_db_from_eta(
            self.eta_floor_p,
            eps=float(self.eps),
            bad_value_db=float(self.config.ar_bad_eta_value_db),
        )

    # ------------------------------------------------------------------
    # Basis precompute
    # ------------------------------------------------------------------

    def _build_all_ring_option_basis(self) -> None:
        for k in range(self.K):
            opts = np.asarray(self.m_options_by_ring.get(k, np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
            for mk in np.unique(opts).tolist():
                self.ensure_ring_option_basis(int(k), int(mk))

    def _basis_cache_vector(self, k: int, nu: int) -> np.ndarray:
        key = (int(k), int(nu))
        if key in self.B_cache:
            return self.B_cache[key]
        psi_rel = (self.psi - float(self.alpha_ring_rad[int(k)])).astype(np.float64)
        xk = self.x_pk[:, int(k)].astype(np.float64)
        Jn = _jv(int(nu), xk).astype(np.float64)
        phase = np.exp(1j * float(int(nu)) * psi_rel).astype(np.complex128)
        vec = (_jpow_j(int(nu)) * Jn.astype(np.complex128) * phase).astype(np.complex128)
        self.B_cache[key] = vec
        return vec

    def ensure_ring_option_basis(self, k: int, mk: int) -> None:
        k = int(k)
        mk = int(mk)
        if k < 0 or k >= self.K:
            raise ValueError(f"ring index out of range: {k}")
        if mk < 0 or not (2 * mk < int(self.N[k]) or mk == 0):
            # Allow mk=0 for compatibility with legacy/debug cases; otherwise enforce m < N/2.
            raise ValueError(f"illegal m option for ring {k}: N={int(self.N[k])}, m={mk}")

        key = (k, mk)
        if key not in self.Cco:
            orders = compute_alias_orders(int(self.N[k]), mk, self.q_a_non0, int(self.config.B_alias_co)) if self.q_a_non0.size else np.zeros((0, int(self.config.B_alias_co)), dtype=np.int64)
            self.orders_co[key] = orders
            self.Cco[key] = self._build_ring_basis(k, self.q_a_non0, self.q_a_non0_col_idx, orders, self.aq_phase_p_full)
        if key not in self.Ccx:
            orders = compute_alias_orders(int(self.N[k]), mk, self.q_c_non0, int(self.config.B_alias_cx)) if self.q_c_non0.size else np.zeros((0, int(self.config.B_alias_cx)), dtype=np.int64)
            self.orders_cx[key] = orders
            self.Ccx[key] = self._build_ring_basis(k, self.q_c_non0, self.q_c_non0_col_idx, orders, self.cq_phase_p_full)

    def _build_ring_basis(self, k: int, q_non0: np.ndarray, q_cols: np.ndarray, orders_qb: np.ndarray, phase_grid_full: np.ndarray) -> np.ndarray:
        q_non0 = np.asarray(q_non0, dtype=np.int64).reshape(-1)
        q_cols = np.asarray(q_cols, dtype=np.int64).reshape(-1)
        nQ = int(q_non0.size)
        if nQ == 0:
            return np.zeros((0, self.P), dtype=np.complex128)
        phase_sel = phase_grid_full[:, q_cols].T.astype(np.complex128)  # (nQ,P)
        Ck = np.zeros((nQ, self.P), dtype=np.complex128)
        wkNk = float(self.w[k]) * float(self.N[k])
        for iq in range(nQ):
            acc = np.zeros((self.P,), dtype=np.complex128)
            for nu in np.asarray(orders_qb[iq, :], dtype=np.int64).reshape(-1).tolist():
                acc += self._basis_cache_vector(int(k), int(nu))
            Ck[iq, :] = wkNk * phase_sel[iq, :] * acc
        return Ck

    # ------------------------------------------------------------------
    # Field batch evaluation
    # ------------------------------------------------------------------

    def _require_ready(self) -> None:
        if not bool(self._precomputed):
            raise RuntimeError("Stage2FieldBatchBackend.precompute_all() must be called first")

    def check_m_feasible(self, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = np.asarray(m, dtype=np.int64)
        if m.ndim == 1:
            m = m.reshape(1, -1)
        if m.ndim != 2:
            raise ValueError(f"m must be a 1D or 2D array, got ndim={m.ndim}")
        if m.shape[1] != self.K:
            raise ValueError(f"m must have K={self.K} columns, got shape={m.shape}")
        bad_lt0 = np.any(m < 0, axis=1)
        # m=0 is allowed by legacy evaluator. New candidate generator should avoid it.
        bad_half = np.any(((2 * m) >= self.N.reshape(1, -1)) & (m != 0), axis=1)
        bad = bad_lt0 | bad_half
        return ~bad, bad

    def _delta_phase(self, delta_batch: np.ndarray, channel: str) -> np.ndarray:
        channel = _validate_channels(channel)
        if channel == "both":
            raise ValueError("_delta_phase channel must be 'co' or 'cx', not 'both'")
        delta_batch = _validate_and_wrap_delta_batch(delta_batch, K=self.K)
        if channel == "co":
            q = self.q_a_non0.astype(np.float64)
        else:
            q = self.q_c_non0.astype(np.float64)
        return np.exp(-1j * delta_batch[:, :, None] * q.reshape(1, 1, -1)).astype(np.complex128)

    def eval_residual_batch(self, *, m_batch: np.ndarray, delta_batch: np.ndarray, channels: str = "both") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        self._require_ready()
        channels = _validate_channels(channels)
        m_batch = np.asarray(m_batch, dtype=np.int64)
        if m_batch.ndim == 1:
            m_batch = m_batch.reshape(1, -1)
        if m_batch.ndim != 2:
            raise ValueError(f"m_batch must be a 1D or 2D array, got ndim={m_batch.ndim}")
        delta_batch = _validate_and_wrap_delta_batch(delta_batch, K=self.K)
        if m_batch.shape != delta_batch.shape or m_batch.shape[1] != self.K:
            raise ValueError(
                f"m_batch shape={m_batch.shape} and delta_batch shape={delta_batch.shape}; "
                f"both must be (D,{self.K})"
            )
        valid_m, bad_m = self.check_m_feasible(m_batch)
        if np.any(bad_m):
            details: List[str] = []
            for row_idx in np.flatnonzero(bad_m).tolist()[:10]:
                for k in range(self.K):
                    mk = int(m_batch[int(row_idx), k])
                    if mk < 0:
                        details.append(f"row={row_idx}, ring={k}, N={int(self.N[k])}, m={mk}, reason=m<0")
                    elif mk != 0 and 2 * mk >= int(self.N[k]):
                        details.append(f"row={row_idx}, ring={k}, N={int(self.N[k])}, m={mk}, reason=2*m>=N")
            suffix = "" if int(np.count_nonzero(bad_m)) <= 10 else " (showing first 10 bad rows)"
            raise ValueError(f"illegal m candidates: {'; '.join(details)}{suffix}")
        D = int(m_batch.shape[0])
        need_co = channels in ("both", "co")
        need_cx = channels in ("both", "cx")
        Eco_res = np.zeros((D, self.P), dtype=np.complex128) if need_co else None
        Ecx_res = np.zeros((D, self.P), dtype=np.complex128) if need_cx else None
        phase_co = self._delta_phase(delta_batch, "co") if need_co and self.q_a_non0.size else None
        phase_cx = self._delta_phase(delta_batch, "cx") if need_cx and self.q_c_non0.size else None

        for k in range(self.K):
            mk_col = m_batch[:, k].astype(np.int64)
            for mk in np.unique(mk_col).tolist():
                rows = np.flatnonzero(mk_col == int(mk))
                if rows.size == 0:
                    continue
                self.ensure_ring_option_basis(k, int(mk))
                if need_co and self.q_a_non0.size:
                    C = self.Cco[(k, int(mk))]
                    Eco_res[rows, :] += phase_co[rows, k, :] @ C
                if need_cx and self.q_c_non0.size:
                    C = self.Ccx[(k, int(mk))]
                    Ecx_res[rows, :] += phase_cx[rows, k, :] @ C
        return Eco_res, Ecx_res

    def eval_fields_batch(self, *, m_batch: np.ndarray, delta_batch: np.ndarray, channels: str = "both") -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        Eco_res, Ecx_res = self.eval_residual_batch(m_batch=m_batch, delta_batch=delta_batch, channels=channels)
        Eco = self.Ea0_p.reshape(1, -1) + Eco_res if Eco_res is not None else None
        Ecx = self.Ec0_p.reshape(1, -1) + Ecx_res if Ecx_res is not None else None
        return Eco, Ecx, Eco_res, Ecx_res

    # ------------------------------------------------------------------
    # Reductions / metric rows
    # ------------------------------------------------------------------

    def _make_worst_point(self, gidx: int, scan_idx: int, value: float, *, domain: str, meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if int(gidx) < 0 or int(scan_idx) < 0:
            return None
        local = self._main_local_by_pos.get(int(gidx), -1) if domain == "main" else self._side_local_by_pos.get(int(gidx), -1)
        return {
            "theta_s_deg": float(self.theta_s_deg[int(scan_idx)]),
            "theta_deg": float(self.theta_deg[int(gidx)]),
            "phi_deg": float(self.phi_deg[int(gidx)]),
            "rho": float(self.rho[int(gidx)]),
            "psi_rad": float(self.psi[int(gidx)]),
            "psi_deg": float(np.rad2deg(self.psi[int(gidx)])),
            "value": float(value),
            "global_point_idx": int(gidx),
            "local_point_idx": int(local),
            "scan_idx": int(scan_idx),
            "domain": str(domain),
            "meta": dict(meta or {}),
        }

    def _reduce_smain(self, Eco_res: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        empty_scans = [s for s, g in enumerate(self.main_gidx_list) if g.size == 0]
        if empty_scans:
            raise ValueError(f"S_main requires main-domain points for every scan; empty scans={empty_scans}")
        D = int(Eco_res.shape[0])
        S_scan = np.full((D, self.S), np.nan, dtype=np.float64)
        idx_scan = np.full((D, self.S), -1, dtype=np.int64)
        for s, g in enumerate(self.main_gidx_list):
            if g.size == 0:
                continue
            vals = np.abs(Eco_res[:, g]) / self.Eref_abs_p[g].reshape(1, -1)
            j = np.argmax(vals, axis=1)
            S_scan[:, s] = vals[np.arange(D), j]
            idx_scan[:, s] = g[j]
        wc_scan = np.nanargmax(S_scan, axis=1).astype(np.int64)
        S_main = S_scan[np.arange(D), wc_scan]
        wc_g = idx_scan[np.arange(D), wc_scan]
        return S_main, wc_scan, wc_g, S_scan

    def _reduce_ar_main(self, Eco: np.ndarray, Ecx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        empty_scans = [s for s, g in enumerate(self.main_gidx_list) if g.size == 0]
        if empty_scans:
            raise ValueError(f"AR_main requires main-domain points for every scan; empty scans={empty_scans}")
        D = int(Eco.shape[0])
        AR_scan = np.full((D, self.S), np.nan, dtype=np.float64)
        idx_scan = np.full((D, self.S), -1, dtype=np.int64)
        bad_count = np.zeros((D,), dtype=np.int64)
        eta_max = np.full((D,), np.nan, dtype=np.float64)
        for s, g in enumerate(self.main_gidx_list):
            if g.size == 0:
                continue
            eta = np.abs(Ecx[:, g]) / (np.abs(Eco[:, g]) + float(self.eps))
            ar, bad, _, eta_raw = stable_ar_db_from_eta(eta, eps=float(self.eps), bad_value_db=float(self.config.ar_bad_eta_value_db))
            bad_count += np.count_nonzero(bad, axis=1).astype(np.int64)
            eta_s_max = np.nanmax(eta_raw, axis=1)
            eta_max = np.fmax(np.where(np.isfinite(eta_max), eta_max, -np.inf), eta_s_max)
            j = np.nanargmax(ar, axis=1)
            AR_scan[:, s] = ar[np.arange(D), j]
            idx_scan[:, s] = g[j]
        wc_scan = np.nanargmax(AR_scan, axis=1).astype(np.int64)
        AR_main = AR_scan[np.arange(D), wc_scan]
        wc_g = idx_scan[np.arange(D), wc_scan]
        return AR_main, wc_scan, wc_g, bad_count, eta_max

    def _reduce_ar_floor(self) -> Tuple[float, np.ndarray, int, float, Optional[Dict[str, Any]]]:
        empty_scans = [s for s, g in enumerate(self.main_gidx_list) if g.size == 0]
        if empty_scans:
            raise ValueError(f"AR_floor requires main-domain points for every scan; empty scans={empty_scans}")
        ar_s = np.full((self.S,), np.nan, dtype=np.float64)
        idx_s = np.full((self.S,), -1, dtype=np.int64)
        for s, g in enumerate(self.main_gidx_list):
            if g.size == 0:
                continue
            vals = self.ar_floor_p_db[g]
            j = int(np.nanargmax(vals))
            ar_s[s] = float(vals[j])
            idx_s[s] = int(g[j])
        if not np.any(np.isfinite(ar_s)):
            return float("nan"), ar_s, 0, float("nan"), None
        wc_s = int(np.nanargmax(ar_s))
        wc_g = int(idx_s[wc_s])
        bad = int(np.count_nonzero(~np.isfinite(self.eta_floor_p[self.main_gidx]) | (self.eta_floor_p[self.main_gidx] >= 1.0)))
        eta_max = float(np.nanmax(self.eta_floor_p[self.main_gidx])) if self.main_gidx.size else float("nan")
        wp = self._make_worst_point(wc_g, wc_s, float(ar_s[wc_s]), domain="main", meta={"metric": "AR_floor_db", "eta_floor": float(self.eta_floor_p[wc_g])})
        return float(ar_s[wc_s]), ar_s, bad, eta_max, wp

    def _reduce_psll(self, Eco: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        empty_main = [s for s, g in enumerate(self.main_gidx_list) if g.size == 0]
        empty_side = [s for s, g in enumerate(self.side_gidx_list) if g.size == 0]
        if empty_main or empty_side:
            raise ValueError(
                "PSLL requires both main-domain and side-domain points for every scan; "
                f"empty main scans={empty_main}, empty side scans={empty_side}"
            )
        D = int(Eco.shape[0])
        absE = np.abs(Eco).astype(np.float64)
        D_s = np.full((D, self.S), np.nan, dtype=np.float64)
        U_s = np.full((D, self.S), np.nan, dtype=np.float64)
        idx_main = np.full((D, self.S), -1, dtype=np.int64)
        idx_side = np.full((D, self.S), -1, dtype=np.int64)
        for s in range(self.S):
            gm = self.main_gidx_list[s]
            gs = self.side_gidx_list[s]
            if gm.size:
                jm = np.argmax(absE[:, gm], axis=1)
                D_s[:, s] = absE[np.arange(D), gm[jm]]
                idx_main[:, s] = gm[jm]
            if gs.size:
                js = np.argmax(absE[:, gs], axis=1)
                U_s[:, s] = absE[np.arange(D), gs[js]]
                idx_side[:, s] = gs[js]
        psll_s = 20.0 * np.log10(np.maximum(U_s, 1e-300) / np.maximum(D_s, float(self.eps)))
        wc_scan = np.nanargmax(psll_s, axis=1).astype(np.int64)
        psll_wc = psll_s[np.arange(D), wc_scan]
        wc_side = idx_side[np.arange(D), wc_scan]
        wc_main = idx_main[np.arange(D), wc_scan]
        return psll_wc, wc_scan, wc_side, wc_main, D_s, U_s

    # ------------------------------------------------------------------
    # Public scoring APIs
    # ------------------------------------------------------------------

    def score_stage1_s_ar(
        self,
        *,
        m_combos: np.ndarray,
        delta_gold: np.ndarray,
        q_anchors: Sequence[float],
        chunk_size: Optional[int] = None,
        candidate_ids: Optional[Sequence[str]] = None,
        verbose: Optional[bool] = None,
    ) -> Stage2BatchStage1Result:
        self._require_ready()
        verbose = self.verbose if verbose is None else bool(verbose)
        m_combos = np.asarray(m_combos, dtype=np.int64)
        if m_combos.ndim == 1:
            m_combos = m_combos.reshape(1, -1)
        if m_combos.shape[1] != self.K:
            raise ValueError(f"m_combos must have K={self.K} columns")
        delta_gold = np.asarray(delta_gold, dtype=np.float64).reshape(-1)
        if delta_gold.size != self.K:
            raise ValueError(f"delta_gold must have shape ({self.K},)")
        q_anchors = [float(x) for x in q_anchors]
        chunk = int(chunk_size or self.config.stage1_chunk_size)
        chunk = max(1, chunk)

        D = int(m_combos.shape[0])
        if candidate_ids is None:
            candidate_ids_use: Tuple[Any, ...] = tuple(int(i) for i in range(D))
        else:
            candidate_ids_use = tuple(candidate_ids)
            if len(candidate_ids_use) != D:
                raise ValueError(f"candidate_ids length={len(candidate_ids_use)} does not match designs={D}")

        ar_floor_wc_db, _, ar_floor_bad, ar_floor_eta_max, ar_floor_wp = self._reduce_ar_floor()
        rows: List[Dict[str, Any]] = []
        pair_id = 0
        t0 = time.perf_counter()
        if verbose:
            print(f"[Stage2Batch][Stage1] START designs={D} q_anchors={len(q_anchors)} pairs={D*len(q_anchors)} chunk={chunk}", flush=True)
        for start in range(0, D, chunk):
            stop = min(D, start + chunk)
            m_chunk_base = m_combos[start:stop, :]
            for qid, q in enumerate(q_anchors):
                delta = make_delta_with_anchor(delta_gold, q)
                delta_chunk = np.repeat(delta.reshape(1, -1), stop - start, axis=0)
                Eco, Ecx, Eco_res, _ = self.eval_fields_batch(m_batch=m_chunk_base, delta_batch=delta_chunk, channels="both")
                S_main, S_wc_scan, S_wc_g, _ = self._reduce_smain(Eco_res)
                AR_main, AR_wc_scan, AR_wc_g, AR_bad, AR_eta_max = self._reduce_ar_main(Eco, Ecx)
                for ii in range(stop - start):
                    mid = start + ii
                    s_g = int(S_wc_g[ii])
                    ar_g = int(AR_wc_g[ii])
                    row = {
                        "pair_id": int(pair_id),
                        "candidate_id": _as_list_jsonable(candidate_ids_use[mid]),
                        "m_id": int(mid),
                        "q_id": int(qid),
                        "q_anchor_rad": float(q),
                        "q_anchor_pi_units": float(q / math.pi),
                        "m_json": m_chunk_base[ii, :].astype(int).tolist(),
                        "delta_json": delta.astype(float).tolist(),
                        "status": "OK",
                        "S_main": float(S_main[ii]),
                        "AR_floor_db": float(ar_floor_wc_db),
                        "AR_floor_bad_eta_count": int(ar_floor_bad),
                        "AR_floor_eta_max": float(ar_floor_eta_max),
                        "AR_floor_worst_point_json": ar_floor_wp,
                        "AR_main_db": float(AR_main[ii]),
                        "AR_delta_to_floor_db": float(AR_main[ii] - ar_floor_wc_db),
                        "AR_bad_eta_count": int(AR_bad[ii]),
                        "AR_eta_max": float(AR_eta_max[ii]),
                        "S_worst_scan_idx": int(S_wc_scan[ii]),
                        "S_worst_point_json": self._make_worst_point(s_g, int(S_wc_scan[ii]), float(S_main[ii]), domain="main", meta={"metric": "S_main_ratio"}),
                        "AR_worst_scan_idx": int(AR_wc_scan[ii]),
                        "AR_worst_point_json": self._make_worst_point(ar_g, int(AR_wc_scan[ii]), float(AR_main[ii]), domain="main", meta={"metric": "AR_main_db"}),
                    }
                    rows.append(row)
                    pair_id += 1
            if verbose:
                done = stop
                pct = 100.0 * done / max(1, D)
                elapsed = time.perf_counter() - t0
                rate = done / max(elapsed, 1e-12)
                print(f"[Stage2Batch][Stage1] progress designs={done}/{D} ({pct:.1f}%) elapsed={elapsed:.1f}s rate={rate:.2f} design/s", flush=True)

        valid_S = [float(r["S_main"]) for r in rows if np.isfinite(float(r["S_main"]))]
        valid_AR = [float(r["AR_main_db"]) for r in rows if np.isfinite(float(r["AR_main_db"]))]
        S_star = min(valid_S) if valid_S else float("nan")
        AR_star = min(valid_AR) if valid_AR else float("nan")
        if verbose:
            print(f"[Stage2Batch][Stage1] DONE rows={len(rows)} S_star={S_star:.6g} AR_star={AR_star:.4f}dB elapsed={time.perf_counter()-t0:.1f}s", flush=True)
        return Stage2BatchStage1Result(
            rows=tuple(rows),
            S_star=float(S_star),
            AR_star_db=float(AR_star),
            ar_floor_wc_db=float(ar_floor_wc_db),
            status="OK" if rows else "EMPTY",
            meta={"D": int(D), "q_anchors": q_anchors, "chunk_size": int(chunk), "backend_summary": self.summary()},
        )

    def score_psll_batch(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        chunk_size: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> Stage2BatchPSLLResult:
        self._require_ready()
        verbose = self.verbose if verbose is None else bool(verbose)
        chunk = int(chunk_size or self.config.stage2_chunk_size)
        chunk = max(1, chunk)
        in_rows = [dict(r) for r in rows]
        out_rows: List[Dict[str, Any]] = []
        n = int(len(in_rows))
        t0 = time.perf_counter()
        if n == 0:
            return Stage2BatchPSLLResult(rows=tuple(), status="EMPTY", meta={"input_count": 0, "chunk_size": chunk})
        for i, row in enumerate(in_rows):
            for key in ("m_json", "delta_json"):
                if key not in row:
                    raise KeyError(f"PSLL input row {i} missing required key {key!r}")
        if verbose:
            print(f"[Stage2Batch][PSLL] START rows={n} chunk={chunk}", flush=True)
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            m_batch = np.asarray([in_rows[i]["m_json"] for i in range(start, stop)], dtype=np.int64)
            delta_batch = np.asarray([in_rows[i]["delta_json"] for i in range(start, stop)], dtype=np.float64)
            Eco, _, _, _ = self.eval_fields_batch(m_batch=m_batch, delta_batch=delta_batch, channels="co")
            psll_wc, wc_scan, wc_side, wc_main, D_s, U_s = self._reduce_psll(Eco)
            for ii in range(stop - start):
                base = dict(in_rows[start + ii])
                ws = int(wc_scan[ii])
                side_g = int(wc_side[ii])
                main_g = int(wc_main[ii])
                base.update({
                    "PSLL_wc_db": float(psll_wc[ii]),
                    "PSLL_worst_scan_idx": ws,
                    "PSLL_D_s_json": D_s[ii, :].astype(float).tolist(),
                    "PSLL_U_s_json": U_s[ii, :].astype(float).tolist(),
                    "PSLL_worst_side_point_json": self._make_worst_point(side_g, ws, float(U_s[ii, ws]), domain="side", meta={"metric": "U_side_abs_Eco"}),
                    "PSLL_worst_main_point_json": self._make_worst_point(main_g, ws, float(D_s[ii, ws]), domain="main", meta={"metric": "D_main_abs_Eco"}),
                    "PSLL_status": "OK",
                })
                out_rows.append(base)
            if verbose:
                print(f"[Stage2Batch][PSLL] progress {stop}/{n} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
        out_rows.sort(key=lambda r: (float(r.get("PSLL_wc_db", np.inf)), float(r.get("S_main", np.inf)), float(r.get("AR_main_db", np.inf))))
        return Stage2BatchPSLLResult(rows=tuple(out_rows), status="OK" if out_rows else "EMPTY", meta={"input_count": n, "chunk_size": chunk})

    def score_final_grid_batch(
        self,
        *,
        top_rows: Sequence[Mapping[str, Any]],
        delta_gold: np.ndarray,
        c_grid: Sequence[float],
        thresholds: Optional[Mapping[str, float]] = None,
        fine_delta_grid: Optional[Sequence[float]] = None,
        chunk_size: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Score one explicit coarse c0/c1 grid for each top row.

        fine_delta_grid is retained for API compatibility but must be None or
        empty. Fine refinement belongs to stage2_screen_main.py orchestration.
        """
        self._require_ready()
        verbose = self.verbose if verbose is None else bool(verbose)
        thresholds = dict(thresholds or {})
        S_thr = float(thresholds.get("S_threshold", np.inf))
        AR_thr = float(thresholds.get("AR_threshold_db", np.inf))
        c_axis = tuple(float(x) for x in c_grid)
        if not c_axis:
            raise ValueError("c_grid must be non-empty")
        if not np.all(np.isfinite(np.asarray(c_axis, dtype=np.float64))):
            raise ValueError("c_grid contains NaN or Inf")
        fine_axis = tuple() if fine_delta_grid is None else tuple(float(x) for x in fine_delta_grid)
        if fine_axis:
            raise NotImplementedError(
                "score_final_grid_batch currently performs coarse c0/c1 grid only; "
                "fine_delta_grid is reserved for stage2_screen_main orchestration"
            )
        axis = normalized_ring_axis(self.K)
        delta_gold = np.asarray(delta_gold, dtype=np.float64).reshape(-1)
        if delta_gold.size != self.K:
            raise ValueError(f"delta_gold must have shape ({self.K},)")
        grid_rows: List[Dict[str, Any]] = []
        best_rows: List[Dict[str, Any]] = []
        if verbose:
            print(f"[Stage2Batch][Final] START top_rows={len(top_rows)} grid={len(c_axis)}x{len(c_axis)}", flush=True)
        for cid, base in enumerate(top_rows):
            if "m_json" not in base:
                raise KeyError(f"Final input row {cid} missing required key 'm_json'")
            m = np.asarray(base["m_json"], dtype=np.int64).reshape(1, -1)
            q_anchor = float(base.get("q_anchor_rad", 0.0))
            local_rows: List[Dict[str, Any]] = []
            m_batch: List[np.ndarray] = []
            d_batch: List[np.ndarray] = []
            meta_batch: List[Tuple[float, float]] = []
            for c0 in c_axis:
                for c1 in c_axis:
                    delta = np.mod(delta_gold + q_anchor + float(c0) + float(c1) * axis, TWO_PI)
                    m_batch.append(m.reshape(-1).copy())
                    d_batch.append(delta.copy())
                    meta_batch.append((float(c0), float(c1)))
            # Score S/AR/PSLL in chunks by reusing stage1 and psll logic pieces.
            M = np.asarray(m_batch, dtype=np.int64)
            Dlt = np.asarray(d_batch, dtype=np.float64)
            chunk = max(1, int(chunk_size or self.config.final_chunk_size))
            for start in range(0, M.shape[0], chunk):
                stop = min(M.shape[0], start + chunk)
                Eco, Ecx, Eco_res, _ = self.eval_fields_batch(m_batch=M[start:stop], delta_batch=Dlt[start:stop], channels="both")
                S_main, S_wc_scan, S_wc_g, _ = self._reduce_smain(Eco_res)
                AR_main, AR_wc_scan, AR_wc_g, AR_bad, AR_eta = self._reduce_ar_main(Eco, Ecx)
                psll_wc, psll_wc_scan, psll_side_g, psll_main_g, D_s, U_s = self._reduce_psll(Eco)
                for ii in range(stop - start):
                    j = start + ii
                    c0, c1 = meta_batch[j]
                    feasible = bool((float(S_main[ii]) <= S_thr) and (float(AR_main[ii]) <= AR_thr))
                    row = dict(base)
                    row.update({
                        "final_candidate_id": int(cid),
                        "grid_id": int(j),
                        "c0_star": float(c0),
                        "c1_star": float(c1),
                        "delta_json": Dlt[j].astype(float).tolist(),
                        "S_main": float(S_main[ii]),
                        "AR_main_db": float(AR_main[ii]),
                        "AR_delta_to_floor_db": float(AR_main[ii] - float(base.get("AR_floor_db", np.nan))),
                        "PSLL_wc_db": float(psll_wc[ii]),
                        "pass_S": bool(float(S_main[ii]) <= S_thr),
                        "pass_AR": bool(float(AR_main[ii]) <= AR_thr),
                        "feasible": feasible,
                        "S_worst_point_json": self._make_worst_point(int(S_wc_g[ii]), int(S_wc_scan[ii]), float(S_main[ii]), domain="main", meta={"metric": "S_main_ratio"}),
                        "AR_worst_point_json": self._make_worst_point(int(AR_wc_g[ii]), int(AR_wc_scan[ii]), float(AR_main[ii]), domain="main", meta={"metric": "AR_main_db"}),
                        "PSLL_worst_side_point_json": self._make_worst_point(int(psll_side_g[ii]), int(psll_wc_scan[ii]), float(U_s[ii, int(psll_wc_scan[ii])]), domain="side", meta={"metric": "U_side_abs_Eco"}),
                        "PSLL_worst_main_point_json": self._make_worst_point(int(psll_main_g[ii]), int(psll_wc_scan[ii]), float(D_s[ii, int(psll_wc_scan[ii])]), domain="main", meta={"metric": "D_main_abs_Eco"}),
                    })
                    local_rows.append(row)
                    grid_rows.append(row)
            feasible_rows = [r for r in local_rows if bool(r.get("feasible", False))]
            if feasible_rows:
                best = sorted(feasible_rows, key=lambda r: (float(r["PSLL_wc_db"]), float(r["S_main"]), float(r["AR_main_db"])))[0]
            else:
                best = sorted(local_rows, key=lambda r: (
                    max(0.0, float(r["S_main"]) - S_thr) + max(0.0, float(r["AR_main_db"]) - AR_thr),
                    float(r["PSLL_wc_db"]),
                ))[0]
            best_rows.append(dict(best))
            if verbose:
                print(f"[Stage2Batch][Final] candidate {cid+1}/{len(top_rows)} best_PSLL={best['PSLL_wc_db']:.4f} feasible={best.get('feasible')}", flush=True)
        best_rows.sort(key=lambda r: (float(r.get("PSLL_wc_db", np.inf)), float(r.get("S_main", np.inf)), float(r.get("AR_main_db", np.inf))))
        return grid_rows, best_rows


__all__ = [
    "Stage2BatchConfig",
    "Stage2BatchStage1Result",
    "Stage2BatchPSLLResult",
    "Stage2FieldBatchBackend",
    "lowest_orders_for_aN",
    "compute_alias_orders",
    "stable_ar_db_from_eta",
    "make_delta_gold",
    "make_delta_with_anchor",
    "normalized_ring_axis",
]
