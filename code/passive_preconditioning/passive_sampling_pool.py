from __future__ import annotations

"""
passive_sampling_pool.py

Stage-2 point-pool builder for the screening / light-pre-optimization pipeline.

Responsibilities
----------------
1. Build three fixed scan-angle sets (A/B/C).
2. Build static pools:
   - Pool-A: main-domain screening pool (3 dB mainlobe only)
   - Pool-B: side-domain screening pool (visible domain minus null-mainlobe)
   - Pool-C: audit pool (main + side)
3. Provide dynamic side-patch builders for Pool-B and Pool-C tie-break use.
4. Save/load static assets in a stage-1-like npz/json style.
5. Provide runtime context helpers to fetch per-scan static + dynamic merged points.

Non-responsibilities
--------------------
- Does NOT evaluate S_main / AR / PSLL.
- Does NOT choose seeds.
- Does NOT optimize (c0, c1).
- Does NOT read IEP / HFSS files.

Design convention
-----------------
Point generation is performed in the scan-centered local domain using (rho, psi),
then exported with both local and global coordinates:
- local: rho, psi_deg
- global: theta_deg, phi_deg, u, v

Current formal-model conventions:
- Stage-1 angular boundaries are converted with rho = sin(alpha) by default.
- The converted rho boundary is applied as a circular scan-centered u-v radius.
- The formal pool builder currently supports phi_s_deg = 0 only.
"""

import argparse
import csv
import json
import math
import hashlib
import os
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

TAU = 2.0 * math.pi
ArrayLike = Union[float, np.ndarray]

# -----------------------------------------------------------------------------
# Zone / layer / tag constants
# -----------------------------------------------------------------------------

ZONE_MAIN = 1
ZONE_SIDE = 2

LAYER_A = 10
LAYER_B = 20
LAYER_C = 30
LAYER_B_DYNAMIC = 21
LAYER_C_DYNAMIC = 31

TAG_A_MAIN = 11
TAG_A_GUARD = 12

TAG_B_SIDE_BASE = 21
TAG_B_SIDE_NEAR = 22
TAG_B_SIDE_NULL_RING = 23
TAG_B_SIDE_FAR = 24
TAG_B_HORIZON = 25
TAG_B_BACK = 26

TAG_C_MAIN = 31
TAG_C_GUARD = 32
TAG_C_SIDE_BASE = 33
TAG_C_SIDE_NEAR = 34
TAG_C_SIDE_NULL_RING = 35
TAG_C_SIDE_FAR = 36
TAG_C_HORIZON = 37
TAG_C_BACK = 38

TAG_DYNAMIC_NEAR_PATCH = 41
TAG_DYNAMIC_FAR_PATCH = 42
TAG_DYNAMIC_NULL_ARC = 43
TAG_DYNAMIC_HORIZON_PATCH = 44
TAG_DYNAMIC_BACK_PATCH = 45

ZONE_NAME = {
    ZONE_MAIN: "main",
    ZONE_SIDE: "side",
}

LAYER_NAME = {
    LAYER_A: "A_static",
    LAYER_B: "B_static",
    LAYER_C: "C_static",
    LAYER_B_DYNAMIC: "B_dynamic",
    LAYER_C_DYNAMIC: "C_dynamic",
}

TAG_NAME = {
    TAG_A_MAIN: "A_MAIN",
    TAG_A_GUARD: "A_GUARD",
    TAG_B_SIDE_BASE: "B_SIDE_BASE",
    TAG_B_SIDE_NEAR: "B_SIDE_NEAR",
    TAG_B_SIDE_NULL_RING: "B_SIDE_NULL_RING",
    TAG_B_SIDE_FAR: "B_SIDE_FAR",
    TAG_B_HORIZON: "B_HORIZON",
    TAG_B_BACK: "B_BACK",
    TAG_C_MAIN: "C_MAIN",
    TAG_C_GUARD: "C_GUARD",
    TAG_C_SIDE_BASE: "C_SIDE_BASE",
    TAG_C_SIDE_NEAR: "C_SIDE_NEAR",
    TAG_C_SIDE_NULL_RING: "C_SIDE_NULL_RING",
    TAG_C_SIDE_FAR: "C_SIDE_FAR",
    TAG_C_HORIZON: "C_HORIZON",
    TAG_C_BACK: "C_BACK",
    TAG_DYNAMIC_NEAR_PATCH: "DYN_NEAR_PATCH",
    TAG_DYNAMIC_FAR_PATCH: "DYN_FAR_PATCH",
    TAG_DYNAMIC_NULL_ARC: "DYN_NULL_ARC",
    TAG_DYNAMIC_HORIZON_PATCH: "DYN_HORIZON_PATCH",
    TAG_DYNAMIC_BACK_PATCH: "DYN_BACK_PATCH",
}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage2AngleSets:
    theta_A_deg: np.ndarray
    theta_B_deg: np.ndarray
    theta_C_deg: np.ndarray
    phi_s_deg: float = 0.0


@dataclass(frozen=True)
class Stage2BoundaryTables:
    rho_3db_A: np.ndarray
    rho_3db_B: np.ndarray
    rho_3db_C: np.ndarray
    rho_null_B: np.ndarray
    rho_null_C: np.ndarray


@dataclass(frozen=True)
class Stage2DynamicPatchConfig:
    n_worst_scan_B: int = 2
    n_near_seed_B: int = 2
    n_other_seed_B: int = 2

    near_dr_list_B: Tuple[float, ...] = (0.01, 0.02)
    near_dpsi_list_deg_B: Tuple[float, ...] = (-8.0, -4.0, 0.0, 4.0, 8.0)

    other_dr_list_B: Tuple[float, ...] = (0.02, 0.04)
    other_dpsi_list_deg_B: Tuple[float, ...] = (-12.0, -6.0, 0.0, 6.0, 12.0)

    null_arc_dr_list_B: Tuple[float, ...] = (0.01, 0.02)
    null_arc_dpsi_list_deg_B: Tuple[float, ...] = (-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0)

    n_worst_scan_C: int = 1
    n_near_seed_C: int = 1
    n_other_seed_C: int = 1


@dataclass(frozen=True)
class Stage2PoolConfig:
    phi_s_deg: float = 0.0

    rho_cap: float = 1.50
    rim_margin: float = 0.02
    eps_side: float = 1e-6

    quant_deg: float = 0.02
    global_seed: int = 12345
    fill_tries: int = 20
    fill_batch: int = 256

    gamma_A: float = 0.70

    n_A_main: int = 160
    n_A_guard: int = 64

    n_ring_A_main: int = 10
    npsi0_A_main: int = 10
    kpsi_A_main: int = 16
    npsi_A_main_boundary: int = 32

    n_ring_A_guard: int = 4
    npsi_A_guard: int = 16
    npsi_A_guard_boundary: int = 32

    n_B_side_base: int = 256
    n_B_side_near: int = 160
    n_B_side_null_ring: int = 32
    n_B_side_far: int = 96
    n_B_horizon: int = 48
    n_B_back: int = 48

    near_band_width_B: float = 0.15
    null_ring_offset_B: float = 0.01
    far_alpha_B: float = 2.0
    horizon_band_width_B: float = 0.02
    back_r1_B: float = 0.03
    back_r2_B: float = 0.07

    n_C_main: int = 96
    # Formal stage-2 Pool-C leaves the 3dB-to-null gap empty by default.
    # C_GUARD is retained only as an optional/debug hook.
    include_C_guard: bool = False
    n_C_guard: int = 0
    n_C_side_base: int = 160
    n_C_side_near: int = 96
    n_C_side_null_ring: int = 32
    n_C_side_far: int = 64
    n_C_horizon: int = 32
    n_C_back: int = 32

    near_band_width_C: float = 0.12
    null_ring_offset_C: float = 0.01
    far_alpha_C: float = 2.0
    horizon_band_width_C: float = 0.02
    back_r1_C: float = 0.03
    back_r2_C: float = 0.07

    dynamic: Stage2DynamicPatchConfig = field(default_factory=Stage2DynamicPatchConfig)


@dataclass(frozen=True)
class Stage2SideSeedSet:
    theta_s_deg: float
    near_seed_rho: np.ndarray
    near_seed_psi_deg: np.ndarray
    other_seed_rho: np.ndarray
    other_seed_psi_deg: np.ndarray
    other_seed_tag: np.ndarray


@dataclass
class PointBatch2DStage2:
    scan_idx: List[int]
    theta_s_deg: List[float]
    phi_s_deg: List[float]
    rho: List[float]
    psi_deg: List[float]
    theta_deg: List[float]
    phi_deg: List[float]
    u: List[float]
    v: List[float]
    zone_code: List[int]
    tag_code: List[int]
    layer_code: List[int]
    priority: List[int]
    is_dynamic: List[int]

    @classmethod
    def empty(cls) -> "PointBatch2DStage2":
        return cls([], [], [], [], [], [], [], [], [], [], [], [], [], [])

    def append(
        self,
        scan_idx: int,
        theta_s_deg: float,
        phi_s_deg: float,
        rho: float,
        psi_deg: float,
        theta_deg: float,
        phi_deg: float,
        u: float,
        v: float,
        zone_code: int,
        tag_code: int,
        layer_code: int,
        priority: int,
        is_dynamic: int,
    ) -> None:
        self.scan_idx.append(int(scan_idx))
        self.theta_s_deg.append(float(theta_s_deg))
        self.phi_s_deg.append(float(phi_s_deg))
        self.rho.append(float(rho))
        self.psi_deg.append(float(psi_deg))
        self.theta_deg.append(float(theta_deg))
        self.phi_deg.append(float(phi_deg))
        self.u.append(float(u))
        self.v.append(float(v))
        self.zone_code.append(int(zone_code))
        self.tag_code.append(int(tag_code))
        self.layer_code.append(int(layer_code))
        self.priority.append(int(priority))
        self.is_dynamic.append(int(is_dynamic))

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "scan_idx": np.asarray(self.scan_idx, dtype=np.int32),
            "theta_s_deg": np.asarray(self.theta_s_deg, dtype=np.float64),
            "phi_s_deg": np.asarray(self.phi_s_deg, dtype=np.float64),
            "rho": np.asarray(self.rho, dtype=np.float64),
            "psi_deg": np.asarray(self.psi_deg, dtype=np.float64),
            "theta_deg": np.asarray(self.theta_deg, dtype=np.float64),
            "phi_deg": np.asarray(self.phi_deg, dtype=np.float64),
            "u": np.asarray(self.u, dtype=np.float64),
            "v": np.asarray(self.v, dtype=np.float64),
            "zone_code": np.asarray(self.zone_code, dtype=np.int8),
            "tag_code": np.asarray(self.tag_code, dtype=np.int16),
            "layer_code": np.asarray(self.layer_code, dtype=np.int16),
            "priority": np.asarray(self.priority, dtype=np.int16),
            "is_dynamic": np.asarray(self.is_dynamic, dtype=np.int8),
        }


@dataclass
class Stage2PoolPackage:
    config: Stage2PoolConfig
    angle_sets: Stage2AngleSets
    boundaries: Stage2BoundaryTables
    poolA: PointBatch2DStage2
    poolB_static: PointBatch2DStage2
    poolC_static: PointBatch2DStage2
    counts_A: List[Dict[str, Any]]
    counts_B: List[Dict[str, Any]]
    counts_C: List[Dict[str, Any]]


@dataclass
class Stage2PoolContext:
    static_npz: Dict[str, np.ndarray]
    summary: Dict[str, Any]
    angle_sets: Stage2AngleSets
    boundaries: Stage2BoundaryTables
    tol_deg: float
    canonical_poolA_by_scan: Dict[float, Dict[str, np.ndarray]]
    canonical_poolB_by_scan: Dict[float, Dict[str, np.ndarray]]
    canonical_poolC_by_scan: Dict[float, Dict[str, np.ndarray]]
    dynamic_poolB_by_scan: Dict[float, Dict[str, np.ndarray]]
    dynamic_poolC_by_scan: Dict[float, Dict[str, np.ndarray]]
    static_slice_validated: bool


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def deg2rad(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) * (math.pi / 180.0)


def rad2deg(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) * (180.0 / math.pi)


def wrap_phi_0_2pi(phi: ArrayLike) -> np.ndarray:
    return np.mod(np.asarray(phi, dtype=np.float64), TAU)


def uv_to_theta_phi(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(u * u + v * v)
    theta = np.arcsin(np.clip(r, 0.0, 1.0))
    phi = wrap_phi_0_2pi(np.arctan2(v, u))
    return theta, phi


def compute_rho_psi(u: np.ndarray, v: np.ndarray, u_s: float, v_s: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    du = u - u_s
    dv = v - v_s
    rho = np.sqrt(du * du + dv * dv)
    psi = wrap_phi_0_2pi(np.arctan2(dv, du))
    return rho, psi


def visible_mask(u: np.ndarray, v: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    return (u * u + v * v) <= (1.0 + tol)


def stable_seed_int(*parts: object) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:8], "big", signed=False)


def _van_der_corput(n: int, base: int) -> float:
    vdc = 0.0
    denom = 1.0
    nn = int(n)
    while nn > 0:
        nn, rem = divmod(nn, base)
        denom *= base
        vdc += rem / denom
    return vdc


def halton_2d_indices(start_idx: int, n: int, base1: int = 2, base2: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    t1 = np.empty(n, dtype=np.float64)
    t2 = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = int(start_idx) + i
        t1[i] = _van_der_corput(idx, base1)
        t2[i] = _van_der_corput(idx, base2)
    return t1, t2


def _visible_rho_cap(theta_s_deg: float, cfg: Stage2PoolConfig) -> float:
    theta_s_rad = float(deg2rad(theta_s_deg))
    return float(min(cfg.rho_cap, 1.0 + math.sin(theta_s_rad)) - cfg.rim_margin)


def _local_to_global(theta_s_deg: float, phi_s_deg: float, rho: float, psi_deg: float) -> Tuple[float, float, float, float]:
    theta_s_rad = float(deg2rad(theta_s_deg))
    phi_s_rad = float(deg2rad(phi_s_deg))
    psi_rad = float(deg2rad(psi_deg))
    u_s = math.sin(theta_s_rad) * math.cos(phi_s_rad)
    v_s = math.sin(theta_s_rad) * math.sin(phi_s_rad)
    u = u_s + float(rho) * math.cos(psi_rad)
    v = v_s + float(rho) * math.sin(psi_rad)
    theta_rad, phi_rad = uv_to_theta_phi(np.asarray([u]), np.asarray([v]))
    return float(rad2deg(theta_rad)[0]), float(rad2deg(phi_rad)[0]), float(u), float(v)


def quantize_theta_phi_deg(theta_rad: np.ndarray, phi_rad: np.ndarray, quant_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    theta_deg = rad2deg(theta_rad)
    phi_deg = rad2deg(wrap_phi_0_2pi(phi_rad))
    q = float(quant_deg)
    return np.round(theta_deg / q).astype(np.int64), np.round(phi_deg / q).astype(np.int64)


def stable_dedup_by_theta_phi(
    u: np.ndarray,
    v: np.ndarray,
    rho: np.ndarray,
    psi_deg: np.ndarray,
    priority: np.ndarray,
    tag: np.ndarray,
    zone: np.ndarray,
    layer: np.ndarray,
    is_dynamic: np.ndarray,
    quant_deg: float,
) -> Tuple[np.ndarray, ...]:
    theta_rad, phi_rad = uv_to_theta_phi(u, v)
    k_th, k_ph = quantize_theta_phi_deg(theta_rad, phi_rad, quant_deg)
    order = np.lexsort((k_ph, k_th, priority))
    u2 = u[order]
    v2 = v[order]
    rho2 = rho[order]
    psi2 = psi_deg[order]
    pr2 = priority[order]
    tg2 = tag[order]
    zn2 = zone[order]
    ly2 = layer[order]
    dy2 = is_dynamic[order]
    kth2 = k_th[order]
    kph2 = k_ph[order]
    keep = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        same = (kth2[1:] == kth2[:-1]) & (kph2[1:] == kph2[:-1])
        keep[1:] = ~same
    return u2[keep], v2[keep], rho2[keep], psi2[keep], pr2[keep], tg2[keep], zn2[keep], ly2[keep], dy2[keep]


def trim_to_n(
    u: np.ndarray,
    v: np.ndarray,
    rho: np.ndarray,
    psi_deg: np.ndarray,
    priority: np.ndarray,
    tag: np.ndarray,
    zone: np.ndarray,
    layer: np.ndarray,
    is_dynamic: np.ndarray,
    n_target: int,
) -> Tuple[np.ndarray, ...]:
    if len(u) <= n_target:
        return u, v, rho, psi_deg, priority, tag, zone, layer, is_dynamic
    order = np.lexsort((psi_deg, rho, priority))[: int(n_target)]
    return u[order], v[order], rho[order], psi_deg[order], priority[order], tag[order], zone[order], layer[order], is_dynamic[order]


def _empty_point_dict() -> Dict[str, np.ndarray]:
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


def _count_summary(theta_s_deg: float, pts: Dict[str, np.ndarray]) -> Dict[str, Any]:
    tags = np.asarray(pts["tag_code"], dtype=np.int16)
    zones = np.asarray(pts["zone_code"], dtype=np.int8)
    return {
        "theta_s_deg": float(theta_s_deg),
        "n_total": int(tags.size),
        "n_main": int(np.count_nonzero(zones == ZONE_MAIN)),
        "n_side": int(np.count_nonzero(zones == ZONE_SIDE)),
        "n_dynamic": int(np.count_nonzero(np.asarray(pts["is_dynamic"], dtype=np.int8) == 1)),
        "tag_hist": {TAG_NAME.get(int(t), str(int(t))): int(np.count_nonzero(tags == t)) for t in np.unique(tags)},
    }


def _canonical_theta_deg(theta_s_deg: float, tol_deg: float) -> float:
    scale = max(1, int(round(1.0 / max(float(tol_deg), 1e-12))))
    return float(round(float(theta_s_deg) * scale) / scale)


def _theta_mask(theta_vec_deg: np.ndarray, theta_s_deg: float, tol_deg: float) -> np.ndarray:
    return np.isclose(np.asarray(theta_vec_deg, dtype=np.float64), float(theta_s_deg), atol=float(tol_deg))


_POINT_DICT_KEYS = (
    "scan_idx",
    "theta_s_deg",
    "phi_s_deg",
    "rho",
    "psi_deg",
    "theta_deg",
    "phi_deg",
    "u",
    "v",
    "zone_code",
    "tag_code",
    "layer_code",
    "priority",
    "is_dynamic",
)


def _validate_point_dict(
    pts: Mapping[str, np.ndarray],
    *,
    name: str,
    allow_empty: bool,
    expected_scan_idx: Optional[int] = None,
    expected_theta_s_deg: Optional[float] = None,
    expected_dynamic: Optional[int] = None,
    tol: float = 1e-9,
) -> None:
    missing = [key for key in _POINT_DICT_KEYS if key not in pts]
    if missing:
        raise ValueError(f"{name}: missing point fields {missing}")

    arrays = {key: np.asarray(pts[key]).reshape(-1) for key in _POINT_DICT_KEYS}
    lengths = {key: int(arr.size) for key, arr in arrays.items()}
    n = lengths["rho"]
    bad_lengths = {key: size for key, size in lengths.items() if size != n}
    if bad_lengths:
        raise ValueError(f"{name}: point-field length mismatch; rho={n}, mismatches={bad_lengths}")
    if n == 0:
        if allow_empty:
            return
        raise ValueError(f"{name}: point set is empty")

    for key in ("theta_s_deg", "phi_s_deg", "rho", "psi_deg", "theta_deg", "phi_deg", "u", "v"):
        arr = np.asarray(arrays[key], dtype=np.float64)
        if np.any(~np.isfinite(arr)):
            bad = np.flatnonzero(~np.isfinite(arr))[:10].astype(int).tolist()
            raise ValueError(f"{name}: {key} contains NaN/Inf at indices {bad}")

    rho = np.asarray(arrays["rho"], dtype=np.float64)
    if np.any(rho < -float(tol)):
        bad = np.flatnonzero(rho < -float(tol))[:10].astype(int).tolist()
        raise ValueError(f"{name}: rho contains negative values at indices {bad}")

    u = np.asarray(arrays["u"], dtype=np.float64)
    v = np.asarray(arrays["v"], dtype=np.float64)
    outside = (u * u + v * v) > (1.0 + float(tol))
    if np.any(outside):
        bad = np.flatnonzero(outside)[:10].astype(int).tolist()
        raise ValueError(f"{name}: points outside visible u-v disk at indices {bad}")

    zone = np.asarray(arrays["zone_code"], dtype=np.int64)
    bad_zone = ~np.isin(zone, np.asarray([ZONE_MAIN, ZONE_SIDE], dtype=np.int64))
    if np.any(bad_zone):
        values = np.unique(zone[bad_zone]).astype(int).tolist()
        raise ValueError(f"{name}: invalid zone_code values {values}")

    dynamic = np.asarray(arrays["is_dynamic"], dtype=np.int64)
    bad_dynamic = ~np.isin(dynamic, np.asarray([0, 1], dtype=np.int64))
    if np.any(bad_dynamic):
        values = np.unique(dynamic[bad_dynamic]).astype(int).tolist()
        raise ValueError(f"{name}: invalid is_dynamic values {values}")

    if expected_scan_idx is not None:
        scan_idx = np.asarray(arrays["scan_idx"], dtype=np.int64)
        if np.any(scan_idx != int(expected_scan_idx)):
            values = np.unique(scan_idx).astype(int).tolist()
            raise ValueError(f"{name}: expected scan_idx={expected_scan_idx}, got {values}")

    if expected_theta_s_deg is not None:
        theta_s = np.asarray(arrays["theta_s_deg"], dtype=np.float64)
        if not np.all(np.isclose(theta_s, float(expected_theta_s_deg), atol=float(tol), rtol=0.0)):
            raise ValueError(f"{name}: theta_s_deg does not match expected {expected_theta_s_deg}")

    if expected_dynamic is not None and np.any(dynamic != int(expected_dynamic)):
        values = np.unique(dynamic).astype(int).tolist()
        raise ValueError(f"{name}: expected is_dynamic={expected_dynamic}, got {values}")


def _validate_scan_pool_domains(
    pts: Mapping[str, np.ndarray],
    *,
    pool_name: str,
    rho_3db: Optional[float],
    rho_null: Optional[float],
    cfg: Stage2PoolConfig,
    require_main: bool,
    require_side: bool,
    tol: float = 1e-9,
) -> None:
    name = str(pool_name).upper().strip()
    rho = np.asarray(pts["rho"], dtype=np.float64).reshape(-1)
    zone = np.asarray(pts["zone_code"], dtype=np.int64).reshape(-1)
    tag = np.asarray(pts["tag_code"], dtype=np.int64).reshape(-1)

    n_main = int(np.count_nonzero(zone == ZONE_MAIN))
    n_side = int(np.count_nonzero(zone == ZONE_SIDE))
    if require_main and n_main == 0:
        raise ValueError(f"Pool-{name}: required main-domain points are missing")
    if require_side and n_side == 0:
        raise ValueError(f"Pool-{name}: required side-domain points are missing")

    if name == "A":
        if n_side != 0:
            raise ValueError("Pool-A must not contain side-domain points")
        if rho_3db is None or not np.isfinite(float(rho_3db)):
            raise ValueError("Pool-A requires finite rho_3db")
        bad = (zone == ZONE_MAIN) & (rho > float(rho_3db) + float(tol))
        if np.any(bad):
            raise ValueError("Pool-A contains main points outside rho_3db")
        return

    if name == "B":
        if n_main != 0:
            raise ValueError("Pool-B must not contain main-domain points")
        if rho_null is None or not np.isfinite(float(rho_null)):
            raise ValueError("Pool-B requires finite rho_null")
        bad = (zone == ZONE_SIDE) & (rho <= float(rho_null) + float(cfg.eps_side) - float(tol))
        if np.any(bad):
            raise ValueError("Pool-B contains side points inside the null exclusion boundary")
        return

    if name == "C":
        if rho_3db is None or rho_null is None:
            raise ValueError("Pool-C requires rho_3db and rho_null")
        r3 = float(rho_3db)
        rn = float(rho_null)
        guard = tag == TAG_C_GUARD
        ordinary_main = (zone == ZONE_MAIN) & ~guard
        side = zone == ZONE_SIDE
        if np.any(ordinary_main & (rho > r3 + float(tol))):
            raise ValueError("Pool-C contains ordinary main points outside rho_3db")
        if np.any(side & (rho <= rn + float(cfg.eps_side) - float(tol))):
            raise ValueError("Pool-C contains side points inside the null exclusion boundary")
        if bool(getattr(cfg, "include_C_guard", False)):
            if np.any(guard & ((rho <= r3 + float(tol)) | (rho > rn + float(tol)))):
                raise ValueError("Pool-C guard points must satisfy rho_3db < rho <= rho_null")
        elif np.any(guard):
            raise ValueError("Pool-C contains guard points while include_C_guard=False")
        gap = (rho > r3 + float(tol)) & (rho <= rn + float(cfg.eps_side) - float(tol))
        if not bool(getattr(cfg, "include_C_guard", False)) and np.any(gap):
            raise ValueError("Pool-C formal mode must leave rho_3db-to-rho_null gap empty")
        return

    raise ValueError("pool_name must be A, B, or C")


# -----------------------------------------------------------------------------
# Primitive subset builders
# -----------------------------------------------------------------------------


def subset_ring(u_s: float, rho: float, npsi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    psi = np.linspace(0.0, TAU, int(npsi), endpoint=False, dtype=np.float64)
    u = u_s + float(rho) * np.cos(psi)
    v = float(rho) * np.sin(psi)
    return u, v, rad2deg(psi)


def subset_equal_area_rings_disk(u_s: float, rho_max: float, n_ring: int, npsi0: int, kpsi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rho_max <= 0 or n_ring <= 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    us: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    rhos: List[np.ndarray] = []
    psis: List[np.ndarray] = []
    for i in range(1, int(n_ring)):
        rho_i = float(rho_max) * math.sqrt(i / float(n_ring))
        frac = rho_i / max(float(rho_max), 1e-12)
        npsi = int(np.clip(int(round(int(npsi0) + int(kpsi) * frac)), 8, 256))
        u_i, v_i, psi_i_deg = subset_ring(u_s, rho_i, npsi)
        us.append(u_i)
        vs.append(v_i)
        rhos.append(np.full(npsi, rho_i, dtype=np.float64))
        psis.append(psi_i_deg)
    if not us:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(us), np.concatenate(vs), np.concatenate(rhos), np.concatenate(psis)


def subset_guard_equal_area_annulus(u_s: float, rho_lo: float, rho_hi: float, n_ring: int, npsi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rho_hi <= rho_lo or n_ring <= 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    us: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    rhos: List[np.ndarray] = []
    psis: List[np.ndarray] = []
    r2_lo = float(rho_lo) ** 2
    r2_hi = float(rho_hi) ** 2
    for i in range(1, int(n_ring) + 1):
        t = i / float(n_ring)
        rho_i = math.sqrt(r2_lo + t * (r2_hi - r2_lo))
        u_i, v_i, psi_i_deg = subset_ring(u_s, rho_i, int(npsi))
        us.append(u_i)
        vs.append(v_i)
        rhos.append(np.full(int(npsi), rho_i, dtype=np.float64))
        psis.append(psi_i_deg)
    return np.concatenate(us), np.concatenate(vs), np.concatenate(rhos), np.concatenate(psis)


def subset_side_area_uniform(u_s: float, rho_min: float, rho_max: float, n: int, seed: int, stream: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0 or rho_max <= rho_min:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    start = 1 + (seed % 100000) + (stable_seed_int(seed, stream) % 100000)
    t1, t2 = halton_2d_indices(start, int(n))
    psi = TAU * t1
    r2_lo = float(rho_min) ** 2
    r2_hi = float(rho_max) ** 2
    rho = np.sqrt(r2_lo + t2 * (r2_hi - r2_lo))
    u = u_s + rho * np.cos(psi)
    v = rho * np.sin(psi)
    return u, v, rho, rad2deg(psi)


def subset_side_far_tail(u_s: float, rho_min: float, rho_max: float, n: int, alpha: float, seed: int, stream: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0 or rho_max <= rho_min:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    start = 1 + (seed % 100000) + (stable_seed_int(seed, stream) % 100000)
    t1, t2 = halton_2d_indices(start, int(n))
    psi = TAU * t1
    p = 1.0 / (1.0 + float(alpha))
    rho = float(rho_min) + (float(rho_max) - float(rho_min)) * np.power(t2, p)
    u = u_s + rho * np.cos(psi)
    v = rho * np.sin(psi)
    return u, v, rho, rad2deg(psi)


def subset_horizon_band(n: int, band_width: float, seed: int, stream: str) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        return np.zeros(0), np.zeros(0)
    start = 1 + (seed % 100000) + (stable_seed_int(seed, stream) % 100000)
    t1, t2 = halton_2d_indices(start, int(n))
    phi = TAU * t1
    r_uv = 1.0 - float(band_width) * t2
    return r_uv * np.cos(phi), r_uv * np.sin(phi)


def subset_disk_centered(u_c: float, v_c: float, radius: float, n: int, seed: int, stream: str) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0 or radius <= 0:
        return np.zeros(0), np.zeros(0)
    start = 1 + (seed % 100000) + (stable_seed_int(seed, stream) % 100000)
    t1, t2 = halton_2d_indices(start, int(n))
    ang = TAU * t1
    r = float(radius) * np.sqrt(t2)
    return float(u_c) + r * np.cos(ang), float(v_c) + r * np.sin(ang)


# -----------------------------------------------------------------------------
# Internal point-dict builder helpers
# -----------------------------------------------------------------------------


def _make_point_dict_from_uv(
    scan_idx: int,
    theta_s_deg: float,
    phi_s_deg: float,
    u: np.ndarray,
    v: np.ndarray,
    rho: np.ndarray,
    psi_deg: np.ndarray,
    zone_code: int,
    tag_code: int,
    layer_code: int,
    priority: int,
    is_dynamic: int,
) -> Dict[str, np.ndarray]:
    if len(u) == 0:
        return _empty_point_dict()
    theta_rad, phi_rad = uv_to_theta_phi(u, v)
    n = int(len(u))
    return {
        "scan_idx": np.full((n,), int(scan_idx), dtype=np.int32),
        "theta_s_deg": np.full((n,), float(theta_s_deg), dtype=np.float64),
        "phi_s_deg": np.full((n,), float(phi_s_deg), dtype=np.float64),
        "rho": np.asarray(rho, dtype=np.float64).reshape(-1),
        "psi_deg": np.asarray(psi_deg, dtype=np.float64).reshape(-1),
        "theta_deg": rad2deg(theta_rad).astype(np.float64),
        "phi_deg": rad2deg(phi_rad).astype(np.float64),
        "u": np.asarray(u, dtype=np.float64).reshape(-1),
        "v": np.asarray(v, dtype=np.float64).reshape(-1),
        "zone_code": np.full((n,), int(zone_code), dtype=np.int8),
        "tag_code": np.full((n,), int(tag_code), dtype=np.int16),
        "layer_code": np.full((n,), int(layer_code), dtype=np.int16),
        "priority": np.full((n,), int(priority), dtype=np.int16),
        "is_dynamic": np.full((n,), int(is_dynamic), dtype=np.int8),
    }


def _concat_point_dicts_raw(dicts: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    valid = [d for d in dicts if d and int(np.asarray(d.get("rho", np.zeros((0,)))).size) > 0]
    if not valid:
        return _empty_point_dict()
    keys = sorted(valid[0].keys())
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = np.concatenate([np.asarray(d[k]) for d in valid], axis=0)
    return out


def _filter_point_dict(
    pts: Dict[str, np.ndarray],
    *,
    theta_s_deg: float,
    rho_lo: Optional[float],
    rho_hi: Optional[float],
    visible_only: bool = True,
) -> Dict[str, np.ndarray]:
    if int(np.asarray(pts["rho"]).size) == 0:
        return _empty_point_dict()
    u = np.asarray(pts["u"], dtype=np.float64)
    v = np.asarray(pts["v"], dtype=np.float64)
    rho = np.asarray(pts["rho"], dtype=np.float64)
    mask = np.ones((rho.size,), dtype=bool)
    if visible_only:
        mask &= visible_mask(u, v, tol=1e-12)
    rho_rim = _visible_rho_cap(float(theta_s_deg), Stage2PoolConfig())  # only used as permissive cap; overwritten by explicit hi if present
    mask &= rho <= rho_rim + 1e-12 if rho_hi is None else rho <= float(rho_hi) + 1e-12
    if rho_lo is not None:
        mask &= rho >= float(rho_lo) - 1e-12
    out = {k: np.asarray(vv)[mask].copy() for k, vv in pts.items()}
    return out


def _dedup_trim_fill(
    pts: Dict[str, np.ndarray],
    *,
    n_target: int,
    quant_deg: float,
) -> Dict[str, np.ndarray]:
    if int(np.asarray(pts["rho"]).size) == 0:
        return _empty_point_dict()
    u = np.asarray(pts["u"], dtype=np.float64)
    v = np.asarray(pts["v"], dtype=np.float64)
    rho = np.asarray(pts["rho"], dtype=np.float64)
    psi_deg = np.asarray(pts["psi_deg"], dtype=np.float64)
    priority = np.asarray(pts["priority"], dtype=np.int16)
    tag = np.asarray(pts["tag_code"], dtype=np.int16)
    zone = np.asarray(pts["zone_code"], dtype=np.int8)
    layer = np.asarray(pts["layer_code"], dtype=np.int16)
    is_dynamic = np.asarray(pts["is_dynamic"], dtype=np.int8)

    u, v, rho, psi_deg, priority, tag, zone, layer, is_dynamic = stable_dedup_by_theta_phi(
        u, v, rho, psi_deg, priority, tag, zone, layer, is_dynamic, quant_deg
    )
    u, v, rho, psi_deg, priority, tag, zone, layer, is_dynamic = trim_to_n(
        u, v, rho, psi_deg, priority, tag, zone, layer, is_dynamic, n_target
    )
    theta_rad, phi_rad = uv_to_theta_phi(u, v)
    scan_idx = np.asarray(pts["scan_idx"], dtype=np.int32)
    theta_s_deg = np.asarray(pts["theta_s_deg"], dtype=np.float64)
    phi_s_deg = np.asarray(pts["phi_s_deg"], dtype=np.float64)
    scan_idx = np.full((len(u),), int(scan_idx[0]), dtype=np.int32)
    theta_s_deg = np.full((len(u),), float(theta_s_deg[0]), dtype=np.float64)
    phi_s_deg = np.full((len(u),), float(phi_s_deg[0]), dtype=np.float64)
    return {
        "scan_idx": scan_idx,
        "theta_s_deg": theta_s_deg,
        "phi_s_deg": phi_s_deg,
        "rho": rho.astype(np.float64),
        "psi_deg": psi_deg.astype(np.float64),
        "theta_deg": rad2deg(theta_rad).astype(np.float64),
        "phi_deg": rad2deg(phi_rad).astype(np.float64),
        "u": u.astype(np.float64),
        "v": v.astype(np.float64),
        "zone_code": zone.astype(np.int8),
        "tag_code": tag.astype(np.int16),
        "layer_code": layer.astype(np.int16),
        "priority": priority.astype(np.int16),
        "is_dynamic": is_dynamic.astype(np.int8),
    }




def _topup_uniform_local_domain(
    *,
    scan_idx: int,
    theta_s_deg: float,
    phi_s_deg: float,
    rho_lo: float,
    rho_hi: float,
    n_need: int,
    seed: int,
    stream: str,
    zone_code: int,
    tag_code: int,
    layer_code: int,
    priority: int,
    is_dynamic: int,
    quant_deg: float,
    existing: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    if n_need <= 0 or rho_hi <= rho_lo:
        return _empty_point_dict()
    theta_s_rad = float(deg2rad(theta_s_deg))
    phi_s_rad = float(deg2rad(phi_s_deg))
    u_s = math.sin(theta_s_rad) * math.cos(phi_s_rad)
    v_s = math.sin(theta_s_rad) * math.sin(phi_s_rad)
    acc = _empty_point_dict() if existing is None else {k: np.asarray(v).copy() for k, v in existing.items()}
    target = int(np.asarray(acc.get('rho', np.zeros((0,)))).size) + int(n_need)
    for tries in range(1, 21):
        cur = int(np.asarray(acc.get('rho', np.zeros((0,)))).size)
        if cur >= target:
            break
        n_cand = max(256, target - cur)
        u, v, rho, psi_deg = subset_side_area_uniform(u_s, rho_lo, rho_hi, n_cand, seed, f'{stream}_{tries}')
        cand = _make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u, v, rho, psi_deg, zone_code, tag_code, layer_code, priority, is_dynamic)
        cand = _filter_point_dict(cand, theta_s_deg=theta_s_deg, rho_lo=rho_lo, rho_hi=rho_hi, visible_only=True)
        acc = _concat_point_dicts_raw([acc, cand])
        acc = _dedup_trim_fill(acc, n_target=target, quant_deg=quant_deg)
    if int(np.asarray(acc.get('rho', np.zeros((0,)))).size) <= (0 if existing is None else int(np.asarray(existing['rho']).size)):
        return _empty_point_dict()
    if existing is None:
        return acc
    n_exist = int(np.asarray(existing['rho']).size)
    return {k: np.asarray(v)[n_exist:].copy() for k, v in acc.items()}


def _append_point_dict_to_batch(batch: PointBatch2DStage2, pts: Dict[str, np.ndarray]) -> None:
    n = int(np.asarray(pts["rho"]).size)
    for i in range(n):
        batch.append(
            scan_idx=int(np.asarray(pts["scan_idx"])[i]),
            theta_s_deg=float(np.asarray(pts["theta_s_deg"])[i]),
            phi_s_deg=float(np.asarray(pts["phi_s_deg"])[i]),
            rho=float(np.asarray(pts["rho"])[i]),
            psi_deg=float(np.asarray(pts["psi_deg"])[i]),
            theta_deg=float(np.asarray(pts["theta_deg"])[i]),
            phi_deg=float(np.asarray(pts["phi_deg"])[i]),
            u=float(np.asarray(pts["u"])[i]),
            v=float(np.asarray(pts["v"])[i]),
            zone_code=int(np.asarray(pts["zone_code"])[i]),
            tag_code=int(np.asarray(pts["tag_code"])[i]),
            layer_code=int(np.asarray(pts["layer_code"])[i]),
            priority=int(np.asarray(pts["priority"])[i]),
            is_dynamic=int(np.asarray(pts["is_dynamic"])[i]),
        )


# -----------------------------------------------------------------------------
# Per-scan builders
# -----------------------------------------------------------------------------


def build_scan_pool_A(scan_idx: int, theta_s_deg: float, rho_3db: float, cfg: Stage2PoolConfig) -> Dict[str, np.ndarray]:
    phi_s_deg = float(cfg.phi_s_deg)
    theta_s_rad = float(deg2rad(theta_s_deg))
    u_s = math.sin(theta_s_rad) * math.cos(float(deg2rad(phi_s_deg)))
    gamma_u = float(cfg.gamma_A * float(rho_3db))
    rho_rim = _visible_rho_cap(theta_s_deg, cfg)

    # Main: center + equal-area rings + gamma boundary ring
    center = _make_point_dict_from_uv(
        scan_idx, theta_s_deg, phi_s_deg,
        np.asarray([u_s]), np.asarray([0.0]), np.asarray([0.0]), np.asarray([0.0]),
        ZONE_MAIN, TAG_A_MAIN, LAYER_A, 0, 0,
    )
    u_ea, v_ea, rho_ea, psi_ea = subset_equal_area_rings_disk(u_s, gamma_u, cfg.n_ring_A_main, cfg.npsi0_A_main, cfg.kpsi_A_main)
    ea = _make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_ea, v_ea, rho_ea, psi_ea, ZONE_MAIN, TAG_A_MAIN, LAYER_A, 0, 0)
    u_bd, v_bd, psi_bd = subset_ring(u_s, gamma_u, cfg.npsi_A_main_boundary)
    bd = _make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_bd, v_bd, np.full(len(u_bd), gamma_u), psi_bd, ZONE_MAIN, TAG_A_MAIN, LAYER_A, 1, 0)
    main = _concat_point_dicts_raw([center, ea, bd])
    main = _filter_point_dict(main, theta_s_deg=theta_s_deg, rho_lo=0.0, rho_hi=min(gamma_u, rho_rim), visible_only=True)
    main = _dedup_trim_fill(main, n_target=cfg.n_A_main, quant_deg=cfg.quant_deg)
    if int(np.asarray(main['rho']).size) < cfg.n_A_main:
        fill = _topup_uniform_local_domain(
            scan_idx=scan_idx, theta_s_deg=theta_s_deg, phi_s_deg=phi_s_deg,
            rho_lo=0.0, rho_hi=min(gamma_u, rho_rim), n_need=cfg.n_A_main - int(np.asarray(main['rho']).size),
            seed=stable_seed_int('A_main_fill', cfg.global_seed, theta_s_deg), stream='A_main_fill',
            zone_code=ZONE_MAIN, tag_code=TAG_A_MAIN, layer_code=LAYER_A, priority=9, is_dynamic=0, quant_deg=cfg.quant_deg, existing=main,
        )
        main = _concat_point_dicts_raw([main, fill])
        main = _dedup_trim_fill(main, n_target=cfg.n_A_main, quant_deg=cfg.quant_deg)

    # Guard annulus
    rho_lo = gamma_u * 1.0005
    rho_hi = min(float(rho_3db), rho_rim)
    u_g, v_g, rho_g, psi_g = subset_guard_equal_area_annulus(u_s, rho_lo, rho_hi, cfg.n_ring_A_guard, cfg.npsi_A_guard)
    guard_base = _make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_g, v_g, rho_g, psi_g, ZONE_MAIN, TAG_A_GUARD, LAYER_A, 2, 0)
    guard_parts = [guard_base]
    rho_gb = min(0.97 * float(rho_3db), rho_rim)
    if rho_gb > rho_lo + 1e-9:
        u_b, v_b, psi_b = subset_ring(u_s, rho_gb, cfg.npsi_A_guard_boundary)
        guard_parts.append(_make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_b, v_b, np.full(len(u_b), rho_gb), psi_b, ZONE_MAIN, TAG_A_GUARD, LAYER_A, 3, 0))
    guard = _concat_point_dicts_raw(guard_parts)
    guard = _filter_point_dict(guard, theta_s_deg=theta_s_deg, rho_lo=rho_lo, rho_hi=rho_hi, visible_only=True)
    guard = _dedup_trim_fill(guard, n_target=cfg.n_A_guard, quant_deg=cfg.quant_deg)
    if int(np.asarray(guard['rho']).size) < cfg.n_A_guard and rho_hi > rho_lo + 1e-9:
        fill = _topup_uniform_local_domain(
            scan_idx=scan_idx, theta_s_deg=theta_s_deg, phi_s_deg=phi_s_deg,
            rho_lo=rho_lo, rho_hi=rho_hi, n_need=cfg.n_A_guard - int(np.asarray(guard['rho']).size),
            seed=stable_seed_int('A_guard_fill', cfg.global_seed, theta_s_deg), stream='A_guard_fill',
            zone_code=ZONE_MAIN, tag_code=TAG_A_GUARD, layer_code=LAYER_A, priority=9, is_dynamic=0, quant_deg=cfg.quant_deg, existing=guard,
        )
        guard = _concat_point_dicts_raw([guard, fill])
        guard = _dedup_trim_fill(guard, n_target=cfg.n_A_guard, quant_deg=cfg.quant_deg)

    return _concat_point_dicts_raw([main, guard])


def build_scan_pool_B_static(scan_idx: int, theta_s_deg: float, rho_null: float, cfg: Stage2PoolConfig) -> Dict[str, np.ndarray]:
    phi_s_deg = float(cfg.phi_s_deg)
    theta_s_rad = float(deg2rad(theta_s_deg))
    u_s = math.sin(theta_s_rad) * math.cos(float(deg2rad(phi_s_deg)))
    rho_rim = _visible_rho_cap(theta_s_deg, cfg)
    seed0 = stable_seed_int("stage2_B", cfg.global_seed, theta_s_deg, rho_null)
    rho_min_side = min(max(float(rho_null) + cfg.eps_side, 0.0), rho_rim)

    parts: List[Dict[str, np.ndarray]] = []

    # B side base
    u_b, v_b, rho_b, psi_b = subset_side_area_uniform(u_s, rho_min_side, rho_rim, cfg.n_B_side_base, seed0, "B_side_base")
    parts.append(_make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_b, v_b, rho_b, psi_b, ZONE_SIDE, TAG_B_SIDE_BASE, LAYER_B, 4, 0))

    # B near
    rho_near_max = min(float(rho_null) + float(cfg.near_band_width_B), rho_rim)
    if rho_near_max > rho_min_side + 1e-9:
        u_n, v_n, rho_n, psi_n = subset_side_area_uniform(u_s, rho_min_side, rho_near_max, cfg.n_B_side_near, seed0, "B_side_near")
        parts.append(_make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_n, v_n, rho_n, psi_n, ZONE_SIDE, TAG_B_SIDE_NEAR, LAYER_B, 5, 0))
    u_nr, v_nr, psi_nr = subset_ring(u_s, min(float(rho_null) + float(cfg.null_ring_offset_B), rho_rim), cfg.n_B_side_null_ring)
    parts.append(_make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_nr, v_nr, np.full(len(u_nr), min(float(rho_null) + float(cfg.null_ring_offset_B), rho_rim)), psi_nr, ZONE_SIDE, TAG_B_SIDE_NULL_RING, LAYER_B, 3, 0))

    # B far
    rho_far_min = min(max(float(rho_null) + 0.25, 0.65), rho_rim)
    if rho_rim > rho_far_min + 1e-9:
        u_f, v_f, rho_f, psi_f = subset_side_far_tail(u_s, rho_far_min, rho_rim, cfg.n_B_side_far, cfg.far_alpha_B, seed0, "B_side_far")
        parts.append(_make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_f, v_f, rho_f, psi_f, ZONE_SIDE, TAG_B_SIDE_FAR, LAYER_B, 6, 0))

    # Horizon
    u_h, v_h = subset_horizon_band(cfg.n_B_horizon, cfg.horizon_band_width_B, seed0, "B_horizon")
    rho_h, psi_h = compute_rho_psi(u_h, v_h, u_s=u_s, v_s=0.0)
    parts.append(_make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_h, v_h, rho_h, rad2deg(psi_h), ZONE_SIDE, TAG_B_HORIZON, LAYER_B, 7, 0))

    # Back
    u_c = -u_s
    n1 = cfg.n_B_back // 2
    n2 = cfg.n_B_back - n1
    u1, v1 = subset_disk_centered(u_c, 0.0, cfg.back_r1_B, n1, seed0, "B_back1")
    u2, v2 = subset_disk_centered(u_c, 0.0, cfg.back_r2_B, n2, seed0, "B_back2")
    u_back = np.concatenate([u1, u2])
    v_back = np.concatenate([v1, v2])
    rho_back, psi_back = compute_rho_psi(u_back, v_back, u_s=u_s, v_s=0.0)
    parts.append(_make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_back, v_back, rho_back, rad2deg(psi_back), ZONE_SIDE, TAG_B_BACK, LAYER_B, 7, 0))

    pts = _concat_point_dicts_raw(parts)
    pts = _filter_point_dict(pts, theta_s_deg=theta_s_deg, rho_lo=rho_min_side, rho_hi=rho_rim, visible_only=True)
    target_B = cfg.n_B_side_base + cfg.n_B_side_near + cfg.n_B_side_null_ring + cfg.n_B_side_far + cfg.n_B_horizon + cfg.n_B_back
    pts = _dedup_trim_fill(pts, n_target=target_B, quant_deg=cfg.quant_deg)
    if int(np.asarray(pts['rho']).size) < target_B and rho_rim > rho_min_side + 1e-9:
        fill = _topup_uniform_local_domain(
            scan_idx=scan_idx, theta_s_deg=theta_s_deg, phi_s_deg=phi_s_deg,
            rho_lo=rho_min_side, rho_hi=rho_rim, n_need=target_B - int(np.asarray(pts['rho']).size),
            seed=stable_seed_int('B_fill', cfg.global_seed, theta_s_deg), stream='B_fill',
            zone_code=ZONE_SIDE, tag_code=TAG_B_SIDE_BASE, layer_code=LAYER_B, priority=9, is_dynamic=0, quant_deg=cfg.quant_deg, existing=pts,
        )
        pts = _concat_point_dicts_raw([pts, fill])
        pts = _dedup_trim_fill(pts, n_target=target_B, quant_deg=cfg.quant_deg)
    return pts


def build_scan_pool_C_static(scan_idx: int, theta_s_deg: float, rho_3db: float, rho_null: float, cfg: Stage2PoolConfig) -> Dict[str, np.ndarray]:
    phi_s_deg = float(cfg.phi_s_deg)
    theta_s_rad = float(deg2rad(theta_s_deg))
    u_s = math.sin(theta_s_rad) * math.cos(float(deg2rad(phi_s_deg)))
    rho_rim = _visible_rho_cap(theta_s_deg, cfg)
    seed0 = stable_seed_int("stage2_C", cfg.global_seed, theta_s_deg, rho_3db, rho_null)
    parts: List[Dict[str, np.ndarray]] = []

    # C main: formal stage-2 uses the whole 3dB main domain.
    # The null-main skirt (rho_3db, rho_null] is intentionally empty unless
    # include_C_guard is explicitly enabled for debugging/audit experiments.
    rho_main_hi = min(float(rho_3db), rho_rim)
    if rho_main_hi > 0.0:
        center = _make_point_dict_from_uv(
            scan_idx, theta_s_deg, phi_s_deg,
            np.asarray([u_s]), np.asarray([0.0]), np.asarray([0.0]), np.asarray([0.0]),
            ZONE_MAIN, TAG_C_MAIN, LAYER_C, 0, 0,
        )
        u_cm, v_cm, rho_cm, psi_cm = subset_equal_area_rings_disk(
            u_s,
            rho_main_hi,
            max(6, cfg.n_ring_A_main - 2),
            max(8, cfg.npsi0_A_main - 2),
            max(10, cfg.kpsi_A_main - 4),
        )
        u_bd, v_bd, psi_bd = subset_ring(u_s, rho_main_hi, max(16, cfg.npsi_A_main_boundary))
        main = _concat_point_dicts_raw([
            center,
            _make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_cm, v_cm, rho_cm, psi_cm, ZONE_MAIN, TAG_C_MAIN, LAYER_C, 0, 0),
            _make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_bd, v_bd, np.full(len(u_bd), rho_main_hi), psi_bd, ZONE_MAIN, TAG_C_MAIN, LAYER_C, 1, 0),
        ])
        main = _filter_point_dict(main, theta_s_deg=theta_s_deg, rho_lo=0.0, rho_hi=rho_main_hi, visible_only=True)
        main = _dedup_trim_fill(main, n_target=cfg.n_C_main, quant_deg=cfg.quant_deg)
        if int(np.asarray(main['rho']).size) < cfg.n_C_main:
            fill = _topup_uniform_local_domain(
                scan_idx=scan_idx, theta_s_deg=theta_s_deg, phi_s_deg=phi_s_deg,
                rho_lo=0.0, rho_hi=rho_main_hi, n_need=cfg.n_C_main - int(np.asarray(main['rho']).size),
                seed=stable_seed_int('C_main_fill', cfg.global_seed, theta_s_deg), stream='C_main_fill',
                zone_code=ZONE_MAIN, tag_code=TAG_C_MAIN, layer_code=LAYER_C, priority=9, is_dynamic=0, quant_deg=cfg.quant_deg, existing=main,
            )
            main = _concat_point_dicts_raw([main, fill])
            main = _dedup_trim_fill(main, n_target=cfg.n_C_main, quant_deg=cfg.quant_deg)
        parts.append(main)

    # Optional C guard: disabled in the formal stage-2 pool.  When enabled, it
    # occupies rho_3db < rho <= rho_null and is marked as TAG_C_GUARD.
    if bool(getattr(cfg, 'include_C_guard', False)) and int(cfg.n_C_guard) > 0:
        rho_lo = min(float(rho_3db) * 1.0005, rho_rim)
        rho_hi = min(float(rho_null), rho_rim)
        if rho_hi > rho_lo + 1e-9:
            u_g, v_g, rho_g, psi_g = subset_guard_equal_area_annulus(
                u_s, rho_lo, rho_hi, max(3, cfg.n_ring_A_guard), max(12, cfg.npsi_A_guard)
            )
            guard = _make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_g, v_g, rho_g, psi_g, ZONE_MAIN, TAG_C_GUARD, LAYER_C, 2, 0)
            guard = _filter_point_dict(guard, theta_s_deg=theta_s_deg, rho_lo=rho_lo, rho_hi=rho_hi, visible_only=True)
            guard = _dedup_trim_fill(guard, n_target=cfg.n_C_guard, quant_deg=cfg.quant_deg)
            parts.append(guard)

    # Side domain starts outside rho_null only.  Thus Pool-C has an intentional
    # empty gap between rho_3db and rho_null when include_C_guard=False.
    rho_min_side = min(max(float(rho_null) + cfg.eps_side, 0.0), rho_rim)
    if rho_rim > rho_min_side + 1e-9:
        u_sb, v_sb, rho_sb, psi_sb = subset_side_area_uniform(u_s, rho_min_side, rho_rim, cfg.n_C_side_base, seed0, "C_side_base")
        parts.append(_make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_sb, v_sb, rho_sb, psi_sb, ZONE_SIDE, TAG_C_SIDE_BASE, LAYER_C, 4, 0))

        rho_near_max = min(float(rho_null) + float(cfg.near_band_width_C), rho_rim)
        if rho_near_max > rho_min_side + 1e-9:
            u_sn, v_sn, rho_sn, psi_sn = subset_side_area_uniform(u_s, rho_min_side, rho_near_max, cfg.n_C_side_near, seed0, "C_side_near")
            parts.append(_make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_sn, v_sn, rho_sn, psi_sn, ZONE_SIDE, TAG_C_SIDE_NEAR, LAYER_C, 5, 0))

        u_nr, v_nr, psi_nr = subset_ring(u_s, min(float(rho_null) + float(cfg.null_ring_offset_C), rho_rim), cfg.n_C_side_null_ring)
        parts.append(_make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_nr, v_nr, np.full(len(u_nr), min(float(rho_null) + float(cfg.null_ring_offset_C), rho_rim)), psi_nr, ZONE_SIDE, TAG_C_SIDE_NULL_RING, LAYER_C, 3, 0))

        rho_far_min = min(max(float(rho_null) + 0.25, 0.65), rho_rim)
        if rho_rim > rho_far_min + 1e-9:
            u_sf, v_sf, rho_sf, psi_sf = subset_side_far_tail(u_s, rho_far_min, rho_rim, cfg.n_C_side_far, cfg.far_alpha_C, seed0, "C_side_far")
            parts.append(_make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_sf, v_sf, rho_sf, psi_sf, ZONE_SIDE, TAG_C_SIDE_FAR, LAYER_C, 6, 0))

        u_h, v_h = subset_horizon_band(cfg.n_C_horizon, cfg.horizon_band_width_C, seed0, "C_horizon")
        rho_h, psi_h = compute_rho_psi(u_h, v_h, u_s=u_s, v_s=0.0)
        parts.append(_make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_h, v_h, rho_h, rad2deg(psi_h), ZONE_SIDE, TAG_C_HORIZON, LAYER_C, 7, 0))

        u_c = -u_s
        n1 = cfg.n_C_back // 2
        n2 = cfg.n_C_back - n1
        u1, v1 = subset_disk_centered(u_c, 0.0, cfg.back_r1_C, n1, seed0, "C_back1")
        u2, v2 = subset_disk_centered(u_c, 0.0, cfg.back_r2_C, n2, seed0, "C_back2")
        u_back = np.concatenate([u1, u2])
        v_back = np.concatenate([v1, v2])
        rho_back, psi_back = compute_rho_psi(u_back, v_back, u_s=u_s, v_s=0.0)
        parts.append(_make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u_back, v_back, rho_back, rad2deg(psi_back), ZONE_SIDE, TAG_C_BACK, LAYER_C, 7, 0))

    pts = _concat_point_dicts_raw(parts)
    pts = _filter_point_dict(pts, theta_s_deg=theta_s_deg, rho_lo=0.0, rho_hi=rho_rim, visible_only=True)
    if int(np.asarray(pts['rho']).size) > 0:
        zones0 = np.asarray(pts['zone_code'], dtype=np.int8)
        tags0 = np.asarray(pts['tag_code'], dtype=np.int16)
        rho0 = np.asarray(pts['rho'], dtype=np.float64)
        mask0 = np.ones((rho0.size,), dtype=bool)
        mask_main = (zones0 == ZONE_MAIN) & (tags0 != TAG_C_GUARD)
        mask_guard = (tags0 == TAG_C_GUARD)
        mask_side = zones0 == ZONE_SIDE
        mask0 &= np.where(mask_main, rho0 <= min(float(rho_3db), rho_rim) + 1e-12, True)
        mask0 &= np.where(mask_guard, (rho0 > float(rho_3db) + 1e-12) & (rho0 <= min(float(rho_null), rho_rim) + 1e-12), True)
        mask0 &= np.where(mask_side, rho0 > rho_min_side - 1e-12, True)
        pts = {k: np.asarray(vv)[mask0].copy() for k, vv in pts.items()}

    n_guard_target = int(cfg.n_C_guard) if bool(getattr(cfg, 'include_C_guard', False)) else 0
    n_side_target = cfg.n_C_side_base + cfg.n_C_side_near + cfg.n_C_side_null_ring + cfg.n_C_side_far + cfg.n_C_horizon + cfg.n_C_back
    target_C = cfg.n_C_main + n_guard_target + n_side_target
    pts = _dedup_trim_fill(pts, n_target=target_C, quant_deg=cfg.quant_deg)
    if int(np.asarray(pts['rho']).size) < target_C:
        # Top up main first, then side.  Formal mode does not top up the
        # rho_3db-to-rho_null gap.
        n_main_cur = int(np.count_nonzero((np.asarray(pts['zone_code']) == ZONE_MAIN) & (np.asarray(pts['tag_code']) != TAG_C_GUARD)))
        if n_main_cur < cfg.n_C_main and rho_main_hi > 1e-12:
            fill_main = _topup_uniform_local_domain(
                scan_idx=scan_idx, theta_s_deg=theta_s_deg, phi_s_deg=phi_s_deg,
                rho_lo=0.0, rho_hi=rho_main_hi, n_need=cfg.n_C_main - n_main_cur,
                seed=stable_seed_int('C_main_fill', cfg.global_seed, theta_s_deg), stream='C_main_fill',
                zone_code=ZONE_MAIN, tag_code=TAG_C_MAIN, layer_code=LAYER_C, priority=9, is_dynamic=0, quant_deg=cfg.quant_deg, existing=None,
            )
            pts = _concat_point_dicts_raw([pts, fill_main])
            pts = _dedup_trim_fill(pts, n_target=min(target_C, int(np.asarray(pts['rho']).size)), quant_deg=cfg.quant_deg)
        n_side_cur = int(np.count_nonzero(np.asarray(pts['zone_code']) == ZONE_SIDE))
        if n_side_cur < n_side_target and rho_rim > rho_min_side + 1e-9:
            fill_side = _topup_uniform_local_domain(
                scan_idx=scan_idx, theta_s_deg=theta_s_deg, phi_s_deg=phi_s_deg,
                rho_lo=rho_min_side, rho_hi=rho_rim, n_need=n_side_target - n_side_cur,
                seed=stable_seed_int('C_side_fill', cfg.global_seed, theta_s_deg), stream='C_side_fill',
                zone_code=ZONE_SIDE, tag_code=TAG_C_SIDE_BASE, layer_code=LAYER_C, priority=9, is_dynamic=0, quant_deg=cfg.quant_deg, existing=None,
            )
            pts = _concat_point_dicts_raw([pts, fill_side])
        pts = _dedup_trim_fill(pts, n_target=target_C, quant_deg=cfg.quant_deg)
    return pts

# -----------------------------------------------------------------------------
# Package builders
# -----------------------------------------------------------------------------


def _validate_angles_and_boundaries(angle_sets: Stage2AngleSets, boundaries: Stage2BoundaryTables) -> None:
    theta_A = np.asarray(angle_sets.theta_A_deg, dtype=np.float64).reshape(-1)
    theta_B = np.asarray(angle_sets.theta_B_deg, dtype=np.float64).reshape(-1)
    theta_C = np.asarray(angle_sets.theta_C_deg, dtype=np.float64).reshape(-1)
    for arr, name in [(theta_A, "theta_A_deg"), (theta_B, "theta_B_deg"), (theta_C, "theta_C_deg")]:
        if arr.size == 0 or np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} must be non-empty finite")
        if np.any(np.diff(arr) <= 0.0):
            raise ValueError(f"{name} must be strictly increasing without duplicates")
    if not np.isfinite(float(angle_sets.phi_s_deg)):
        raise ValueError("phi_s_deg must be finite")

    if len(boundaries.rho_3db_A) != len(theta_A):
        raise ValueError("rho_3db_A length mismatch")
    if len(boundaries.rho_3db_B) != len(theta_B):
        raise ValueError("rho_3db_B length mismatch")
    if len(boundaries.rho_3db_C) != len(theta_C):
        raise ValueError("rho_3db_C length mismatch")
    if len(boundaries.rho_null_B) != len(theta_B):
        raise ValueError("rho_null_B length mismatch")
    if len(boundaries.rho_null_C) != len(theta_C):
        raise ValueError("rho_null_C length mismatch")

    for arr, name in [
        (boundaries.rho_3db_A, "rho_3db_A"),
        (boundaries.rho_3db_B, "rho_3db_B"),
        (boundaries.rho_3db_C, "rho_3db_C"),
        (boundaries.rho_null_B, "rho_null_B"),
        (boundaries.rho_null_C, "rho_null_C"),
    ]:
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        if a.size == 0 or np.any(~np.isfinite(a)) or np.any(a <= 0.0):
            raise ValueError(f"{name} must contain positive finite values")
    if np.any(np.asarray(boundaries.rho_null_B) <= np.asarray(boundaries.rho_3db_B)):
        raise ValueError("rho_null_B must be greater than rho_3db_B")
    if np.any(np.asarray(boundaries.rho_null_C) <= np.asarray(boundaries.rho_3db_C)):
        raise ValueError("rho_null_C must be greater than rho_3db_C")


def _batch_points_for_scan(batch: PointBatch2DStage2, scan_idx: int) -> Dict[str, np.ndarray]:
    pts = batch.to_dict()
    mask = np.asarray(pts["scan_idx"], dtype=np.int64) == int(scan_idx)
    return {key: np.asarray(value)[mask].copy() for key, value in pts.items()}


def _validate_stage2_pool_package(package: Stage2PoolPackage) -> None:
    _validate_angles_and_boundaries(package.angle_sets, package.boundaries)
    cfg = package.config
    if not np.isfinite(float(cfg.phi_s_deg)):
        raise ValueError("Stage2PoolConfig.phi_s_deg must be finite")
    if abs(float(cfg.phi_s_deg) - float(package.angle_sets.phi_s_deg)) > 1e-12:
        raise ValueError("Stage2PoolConfig.phi_s_deg must match Stage2AngleSets.phi_s_deg")
    if abs(float(cfg.phi_s_deg)) > 1e-12:
        raise NotImplementedError("Current formal Stage-2 pool builder supports phi_s_deg=0 only")

    for i, (theta_s, rho_3db) in enumerate(zip(
        np.asarray(package.angle_sets.theta_A_deg, dtype=np.float64).tolist(),
        np.asarray(package.boundaries.rho_3db_A, dtype=np.float64).tolist(),
    )):
        pts = _batch_points_for_scan(package.poolA, i)
        _validate_point_dict(pts, name=f"Pool-A scan {i}", allow_empty=False, expected_scan_idx=i, expected_theta_s_deg=theta_s, expected_dynamic=0)
        _validate_scan_pool_domains(pts, pool_name="A", rho_3db=rho_3db, rho_null=None, cfg=cfg, require_main=True, require_side=False)

    for i, (theta_s, rho_3db, rho_null) in enumerate(zip(
        np.asarray(package.angle_sets.theta_B_deg, dtype=np.float64).tolist(),
        np.asarray(package.boundaries.rho_3db_B, dtype=np.float64).tolist(),
        np.asarray(package.boundaries.rho_null_B, dtype=np.float64).tolist(),
    )):
        pts = _batch_points_for_scan(package.poolB_static, i)
        _validate_point_dict(pts, name=f"Pool-B scan {i}", allow_empty=False, expected_scan_idx=i, expected_theta_s_deg=theta_s, expected_dynamic=0)
        _validate_scan_pool_domains(pts, pool_name="B", rho_3db=rho_3db, rho_null=rho_null, cfg=cfg, require_main=False, require_side=True)

    for i, (theta_s, rho_3db, rho_null) in enumerate(zip(
        np.asarray(package.angle_sets.theta_C_deg, dtype=np.float64).tolist(),
        np.asarray(package.boundaries.rho_3db_C, dtype=np.float64).tolist(),
        np.asarray(package.boundaries.rho_null_C, dtype=np.float64).tolist(),
    )):
        pts = _batch_points_for_scan(package.poolC_static, i)
        _validate_point_dict(pts, name=f"Pool-C scan {i}", allow_empty=False, expected_scan_idx=i, expected_theta_s_deg=theta_s, expected_dynamic=0)
        _validate_scan_pool_domains(pts, pool_name="C", rho_3db=rho_3db, rho_null=rho_null, cfg=cfg, require_main=True, require_side=True)

def build_pool_A_stack(angle_sets: Stage2AngleSets, boundaries: Stage2BoundaryTables, cfg: Stage2PoolConfig) -> Tuple[PointBatch2DStage2, List[Dict[str, Any]], List[Tuple[int, int]], List[np.ndarray], List[np.ndarray]]:
    batch = PointBatch2DStage2.empty()
    counts: List[Dict[str, Any]] = []
    slices: List[Tuple[int, int]] = []
    main_idx_list: List[np.ndarray] = []
    side_idx_list: List[np.ndarray] = []
    theta_A = np.asarray(angle_sets.theta_A_deg, dtype=np.float64).reshape(-1)
    rho_A = np.asarray(boundaries.rho_3db_A, dtype=np.float64).reshape(-1)
    cursor = 0
    for i, (th, r3) in enumerate(zip(theta_A.tolist(), rho_A.tolist())):
        pts = build_scan_pool_A(i, th, r3, cfg)
        _append_point_dict_to_batch(batch, pts)
        n = int(np.asarray(pts["rho"]).size)
        slices.append((cursor, cursor + n))
        zones = np.asarray(pts["zone_code"], dtype=np.int8)
        main_idx_list.append(np.nonzero(zones == ZONE_MAIN)[0].astype(np.int64))
        side_idx_list.append(np.nonzero(zones == ZONE_SIDE)[0].astype(np.int64))
        counts.append(_count_summary(th, pts))
        cursor += n
    return batch, counts, slices, main_idx_list, side_idx_list


def build_pool_B_static_stack(angle_sets: Stage2AngleSets, boundaries: Stage2BoundaryTables, cfg: Stage2PoolConfig) -> Tuple[PointBatch2DStage2, List[Dict[str, Any]], List[Tuple[int, int]], List[np.ndarray], List[np.ndarray]]:
    batch = PointBatch2DStage2.empty()
    counts: List[Dict[str, Any]] = []
    slices: List[Tuple[int, int]] = []
    main_idx_list: List[np.ndarray] = []
    side_idx_list: List[np.ndarray] = []
    theta_B = np.asarray(angle_sets.theta_B_deg, dtype=np.float64).reshape(-1)
    rho_null_B = np.asarray(boundaries.rho_null_B, dtype=np.float64).reshape(-1)
    cursor = 0
    for i, (th, rn) in enumerate(zip(theta_B.tolist(), rho_null_B.tolist())):
        pts = build_scan_pool_B_static(i, th, rn, cfg)
        _append_point_dict_to_batch(batch, pts)
        n = int(np.asarray(pts["rho"]).size)
        slices.append((cursor, cursor + n))
        zones = np.asarray(pts["zone_code"], dtype=np.int8)
        main_idx_list.append(np.nonzero(zones == ZONE_MAIN)[0].astype(np.int64))
        side_idx_list.append(np.nonzero(zones == ZONE_SIDE)[0].astype(np.int64))
        counts.append(_count_summary(th, pts))
        cursor += n
    return batch, counts, slices, main_idx_list, side_idx_list


def build_pool_C_static_stack(angle_sets: Stage2AngleSets, boundaries: Stage2BoundaryTables, cfg: Stage2PoolConfig) -> Tuple[PointBatch2DStage2, List[Dict[str, Any]], List[Tuple[int, int]], List[np.ndarray], List[np.ndarray]]:
    batch = PointBatch2DStage2.empty()
    counts: List[Dict[str, Any]] = []
    slices: List[Tuple[int, int]] = []
    main_idx_list: List[np.ndarray] = []
    side_idx_list: List[np.ndarray] = []
    theta_C = np.asarray(angle_sets.theta_C_deg, dtype=np.float64).reshape(-1)
    rho_3db_C = np.asarray(boundaries.rho_3db_C, dtype=np.float64).reshape(-1)
    rho_null_C = np.asarray(boundaries.rho_null_C, dtype=np.float64).reshape(-1)
    cursor = 0
    for i, (th, r3, rn) in enumerate(zip(theta_C.tolist(), rho_3db_C.tolist(), rho_null_C.tolist())):
        pts = build_scan_pool_C_static(i, th, r3, rn, cfg)
        _append_point_dict_to_batch(batch, pts)
        n = int(np.asarray(pts["rho"]).size)
        slices.append((cursor, cursor + n))
        zones = np.asarray(pts["zone_code"], dtype=np.int8)
        main_idx_list.append(np.nonzero(zones == ZONE_MAIN)[0].astype(np.int64))
        side_idx_list.append(np.nonzero(zones == ZONE_SIDE)[0].astype(np.int64))
        counts.append(_count_summary(th, pts))
        cursor += n
    return batch, counts, slices, main_idx_list, side_idx_list


def build_stage2_pool_package(angle_sets: Stage2AngleSets, boundaries: Stage2BoundaryTables, cfg: Stage2PoolConfig = Stage2PoolConfig()) -> Stage2PoolPackage:
    _validate_angles_and_boundaries(angle_sets, boundaries)
    if not np.isfinite(float(cfg.phi_s_deg)):
        raise ValueError("Stage2PoolConfig.phi_s_deg must be finite")
    if abs(float(cfg.phi_s_deg) - float(angle_sets.phi_s_deg)) > 1e-12:
        raise ValueError("Stage2PoolConfig.phi_s_deg must match Stage2AngleSets.phi_s_deg")
    if abs(float(cfg.phi_s_deg)) > 1e-12:
        raise NotImplementedError("Current formal Stage-2 pool builder supports phi_s_deg=0 only")
    poolA, counts_A, _, _, _ = build_pool_A_stack(angle_sets, boundaries, cfg)
    poolB, counts_B, _, _, _ = build_pool_B_static_stack(angle_sets, boundaries, cfg)
    poolC, counts_C, _, _, _ = build_pool_C_static_stack(angle_sets, boundaries, cfg)
    package = Stage2PoolPackage(config=cfg, angle_sets=angle_sets, boundaries=boundaries, poolA=poolA, poolB_static=poolB, poolC_static=poolC, counts_A=counts_A, counts_B=counts_B, counts_C=counts_C)
    _validate_stage2_pool_package(package)
    return package


# -----------------------------------------------------------------------------
# External boundary table loading
# -----------------------------------------------------------------------------


def _sort_unique_theta_values(theta: Any, value: Any, *, name: str, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    val = np.asarray(value, dtype=np.float64).reshape(-1)
    if th.size == 0 or val.size == 0 or th.size != val.size:
        raise ValueError(f"{name}: theta/value arrays must be non-empty and equal length")
    if np.any(~np.isfinite(th)) or np.any(~np.isfinite(val)):
        raise ValueError(f"{name}: theta/value arrays contain NaN/Inf")
    order = np.argsort(th, kind="mergesort")
    th = th[order]
    val = val[order]

    out_th: List[float] = []
    out_val: List[float] = []
    i = 0
    while i < th.size:
        j = i + 1
        while j < th.size and abs(float(th[j]) - float(th[i])) <= float(tol):
            j += 1
        group = val[i:j]
        if group.size > 1 and np.max(np.abs(group - group[0])) > float(tol):
            raise ValueError(f"{name}: duplicate theta={float(th[i])} has conflicting values {group.tolist()}")
        out_th.append(float(th[i]))
        out_val.append(float(group[0]))
        i = j
    return np.asarray(out_th, dtype=np.float64), np.asarray(out_val, dtype=np.float64)


def _interp_by_theta(theta_src: Any, value_src: Any, theta_dst: Any, *, name: str) -> np.ndarray:
    th, val = _sort_unique_theta_values(theta_src, value_src, name=name)
    dst = np.asarray(theta_dst, dtype=np.float64).reshape(-1)
    if dst.size == 0 or np.any(~np.isfinite(dst)):
        raise ValueError(f"{name}: destination theta array must be non-empty finite")
    if dst.min() < th[0] - 1e-9 or dst.max() > th[-1] + 1e-9:
        raise ValueError(f"{name}: requested theta range [{dst.min()}, {dst.max()}] outside source range [{th[0]}, {th[-1]}]")
    return np.interp(dst, th, val).astype(np.float64)


def _extract_stage1_mainlobe_block(data: Mapping[str, Any]) -> Tuple[Mapping[str, Any], str]:
    final_solution = data.get("final_solution")
    if isinstance(final_solution, Mapping):
        block = final_solution.get("mainlobe")
        if isinstance(block, Mapping) and block:
            return block, "final_solution.mainlobe"
    block = data.get("final_mainlobe")
    if isinstance(block, Mapping) and block:
        return block, "final_mainlobe"
    block = data.get("mainlobe")
    if isinstance(block, Mapping) and block:
        return block, "mainlobe"
    raise ValueError(
        "stage1_main_out requires one of: final_solution.mainlobe, "
        "final_mainlobe, mainlobe"
    )


def _auto_detect_boundary_format(data: Mapping[str, Any]) -> str:
    if data.get("format") == "a0cra_alpha_main_table_v1":
        return "alpha_main_table"
    if all(key in data for key in ("theta_s_deg", "rho_3db", "rho_null")):
        return "rho_table"
    try:
        _extract_stage1_mainlobe_block(data)
        return "stage1_main_out"
    except ValueError:
        pass
    raise ValueError("Cannot auto-detect boundary JSON format")

def _alpha_deg_to_rho(alpha_deg: Any, mode: str = "sin_alpha") -> np.ndarray:
    arr = np.asarray(alpha_deg, dtype=np.float64)
    m = str(mode).lower().strip()
    if m in ("sin_alpha", "sin", "sine"):
        return np.sin(np.deg2rad(arr)).astype(np.float64)
    if m in ("identity", "rho"):
        return arr.astype(np.float64)
    raise ValueError("rho_from_alpha must be 'sin_alpha' or 'identity'")


def _find_alpha_main_table_entry(
    table: Mapping[str, Any],
    *,
    K: int,
    d_min: float,
    d_eff: float,
    R0: float,
    include_center: bool = True,
    outer5_reduce_mult4: bool = True,
    outer5_threshold_gt: int = 16,
    tol: float = 1e-9,
) -> Mapping[str, Any]:
    for entry in list(table.get("entries", [])):
        key = dict(entry.get("key", {}))
        if int(key.get("K")) != int(K):
            continue
        if abs(float(key.get("d_min")) - float(d_min)) > float(tol):
            continue
        if abs(float(key.get("d_eff")) - float(d_eff)) > float(tol):
            continue
        if abs(float(key.get("R0")) - float(R0)) > float(tol):
            continue
        if bool(key.get("include_center", True)) != bool(include_center):
            continue
        if bool(key.get("outer5_reduce_mult4", True)) != bool(outer5_reduce_mult4):
            continue
        if int(key.get("outer5_threshold_gt", 16)) != int(outer5_threshold_gt):
            continue
        return entry
    raise KeyError(
        "No a0cra_alpha_main_table entry for "
        f"K={K}, d_min={d_min}, d_eff={d_eff}, R0={R0}, include_center={include_center}, "
        f"outer5_reduce_mult4={outer5_reduce_mult4}, outer5_threshold_gt={outer5_threshold_gt}"
    )


def load_stage2_boundary_tables_from_json(
    json_path: str,
    angle_sets: Stage2AngleSets,
    *,
    boundary_format: str = "auto",
    rho_from_alpha: str = "sin_alpha",
    K: int = 8,
    d_min: float = 0.5,
    d_eff: float = 0.5,
    R0: float = 5.0,
    include_center: bool = True,
    outer5_reduce_mult4: bool = True,
    outer5_threshold_gt: int = 16,
    key_tol: float = 1e-9,
) -> Stage2BoundaryTables:
    """Load externally calibrated 3dB/null-main boundaries for Pool-A/B/C.

    Supported formats:
    - a0cra_alpha_main_table_v1: uses alpha_3db_deg and alpha_main_deg from the
      selected K/d/R0 entry.  alpha_main_deg is treated as the null-main side
      exclusion boundary for stage-2 pools.
    - stage1_main_out: reads final_solution.mainlobe first, then legacy
      final_mainlobe/mainlobe blocks.  alpha_guard_null_deg is preferred over
      alpha_null_deg for the stage-2 side exclusion boundary.
    - rho_table: uses theta_s_deg, rho_3db, rho_null directly.
    """
    path = Path(str(json_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Boundary JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    fmt = str(boundary_format).lower().strip()
    if fmt == "auto":
        fmt = _auto_detect_boundary_format(data)

    if fmt == "alpha_main_table":
        if data.get("format") != "a0cra_alpha_main_table_v1":
            raise ValueError(f"Expected a0cra_alpha_main_table_v1, got {data.get('format')!r}")
        entry = _find_alpha_main_table_entry(
            data,
            K=K,
            d_min=d_min,
            d_eff=d_eff,
            R0=R0,
            include_center=include_center,
            outer5_reduce_mult4=outer5_reduce_mult4,
            outer5_threshold_gt=outer5_threshold_gt,
            tol=key_tol,
        )
        theta_src = np.asarray(entry["theta_s_deg"], dtype=np.float64).reshape(-1)
        alpha_3db = np.asarray(entry["alpha_3db_deg"], dtype=np.float64).reshape(-1)
        alpha_null = np.asarray(entry["alpha_main_deg"], dtype=np.float64).reshape(-1)
        if not (theta_src.size == alpha_3db.size == alpha_null.size):
            raise ValueError("alpha_main_table theta/alpha arrays length mismatch")
        if np.any(alpha_3db <= 0.0) or np.any(alpha_null <= alpha_3db):
            raise ValueError("alpha_main_table requires alpha_main_deg > alpha_3db_deg > 0")
        rho3_src = _alpha_deg_to_rho(alpha_3db, mode=rho_from_alpha)
        rhon_src = _alpha_deg_to_rho(alpha_null, mode=rho_from_alpha)
    elif fmt == "stage1_main_out":
        ml, _ = _extract_stage1_mainlobe_block(data)
        required = [key for key in ("theta_s_deg", "alpha_3db_deg") if key not in ml]
        if required:
            raise ValueError(f"stage1 mainlobe block missing keys: {required}")
        theta_src = np.asarray(ml["theta_s_deg"], dtype=np.float64).reshape(-1)
        alpha_3db = np.asarray(ml["alpha_3db_deg"], dtype=np.float64).reshape(-1)
        if "alpha_guard_null_deg" in ml:
            alpha_null = np.asarray(ml["alpha_guard_null_deg"], dtype=np.float64).reshape(-1)
        elif "alpha_null_deg" in ml:
            alpha_null = np.asarray(ml["alpha_null_deg"], dtype=np.float64).reshape(-1)
        else:
            raise ValueError("stage1 mainlobe block requires alpha_guard_null_deg or alpha_null_deg")
        if not (theta_src.size == alpha_3db.size == alpha_null.size) or theta_src.size == 0:
            raise ValueError("stage1 mainlobe theta/alpha arrays must be non-empty and equal length")
        if np.any(~np.isfinite(theta_src)) or np.any(~np.isfinite(alpha_3db)) or np.any(~np.isfinite(alpha_null)):
            raise ValueError("stage1 mainlobe theta/alpha arrays contain NaN/Inf")
        if np.any(alpha_3db <= 0.0):
            raise ValueError("stage1 mainlobe alpha_3db_deg must be positive")
        if np.any(alpha_null <= alpha_3db):
            raise ValueError("stage1 mainlobe null boundary must be greater than alpha_3db_deg")
        rho3_src = _alpha_deg_to_rho(alpha_3db, mode=rho_from_alpha)
        rhon_src = _alpha_deg_to_rho(alpha_null, mode=rho_from_alpha)
    elif fmt == "rho_table":
        theta_src = np.asarray(data["theta_s_deg"], dtype=np.float64).reshape(-1)
        rho3_src = np.asarray(data["rho_3db"], dtype=np.float64).reshape(-1)
        rhon_src = np.asarray(data["rho_null"], dtype=np.float64).reshape(-1)
        if not (theta_src.size == rho3_src.size == rhon_src.size) or theta_src.size == 0:
            raise ValueError("rho_table theta/rho arrays must be non-empty and equal length")
        if np.any(~np.isfinite(theta_src)) or np.any(~np.isfinite(rho3_src)) or np.any(~np.isfinite(rhon_src)):
            raise ValueError("rho_table theta/rho arrays contain NaN/Inf")
        if np.any(rho3_src <= 0.0) or np.any(rhon_src <= rho3_src):
            raise ValueError("rho_table requires rho_null > rho_3db > 0")
    else:
        raise ValueError("boundary_format must be auto, alpha_main_table, stage1_main_out, or rho_table")

    theta_A = np.asarray(angle_sets.theta_A_deg, dtype=np.float64)
    theta_B = np.asarray(angle_sets.theta_B_deg, dtype=np.float64)
    theta_C = np.asarray(angle_sets.theta_C_deg, dtype=np.float64)
    out = Stage2BoundaryTables(
        rho_3db_A=_interp_by_theta(theta_src, rho3_src, theta_A, name="rho_3db_A"),
        rho_3db_B=_interp_by_theta(theta_src, rho3_src, theta_B, name="rho_3db_B"),
        rho_3db_C=_interp_by_theta(theta_src, rho3_src, theta_C, name="rho_3db_C"),
        rho_null_B=_interp_by_theta(theta_src, rhon_src, theta_B, name="rho_null_B"),
        rho_null_C=_interp_by_theta(theta_src, rhon_src, theta_C, name="rho_null_C"),
    )
    _validate_angles_and_boundaries(angle_sets, out)
    return out

def make_metric_masks_for_points(points: Mapping[str, np.ndarray], rho_3db: float, rho_null: float, eps_side: float = 1e-6) -> Dict[str, np.ndarray]:
    rho = np.asarray(points["rho"], dtype=np.float64).reshape(-1)
    return {
        "main_3db": rho <= float(rho_3db) + 1e-12,
        "side_null": rho > float(rho_null) + float(eps_side),
        "gap_3db_to_null": (rho > float(rho_3db) + 1e-12) & (rho <= float(rho_null) + float(eps_side)),
    }


def get_stage2_boundaries_for_scan(ctx: Stage2PoolContext, pool_name: str, theta_s_deg: float) -> Tuple[float, float]:
    name = str(pool_name).upper().strip()
    if name == "A":
        theta = np.asarray(ctx.angle_sets.theta_A_deg, dtype=np.float64)
        idx = _find_scan_index(theta, theta_s_deg, ctx.tol_deg)
        return float(ctx.boundaries.rho_3db_A[idx]), float("nan")
    if name == "B":
        theta = np.asarray(ctx.angle_sets.theta_B_deg, dtype=np.float64)
        idx = _find_scan_index(theta, theta_s_deg, ctx.tol_deg)
        return float(ctx.boundaries.rho_3db_B[idx]), float(ctx.boundaries.rho_null_B[idx])
    if name == "C":
        theta = np.asarray(ctx.angle_sets.theta_C_deg, dtype=np.float64)
        idx = _find_scan_index(theta, theta_s_deg, ctx.tol_deg)
        return float(ctx.boundaries.rho_3db_C[idx]), float(ctx.boundaries.rho_null_C[idx])
    raise ValueError("pool_name must be A, B, or C")



# -----------------------------------------------------------------------------
# Save / load assets
# -----------------------------------------------------------------------------


def _write_counts_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = ["theta_s_deg", "n_total", "n_main", "n_side", "n_dynamic", "tag_hist"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            out = dict(row)
            out["tag_hist"] = json.dumps(out["tag_hist"], ensure_ascii=False)
            w.writerow(out)


def save_stage2_pool_package(package: Stage2PoolPackage, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "stage2_pool_static.npz")
    json_path = os.path.join(out_dir, "stage2_pool_summary.json")
    counts_A_csv = os.path.join(out_dir, "stage2_counts_A.csv")
    counts_B_csv = os.path.join(out_dir, "stage2_counts_B.csv")
    counts_C_csv = os.path.join(out_dir, "stage2_counts_C.csv")

    pA = package.poolA.to_dict()
    pB = package.poolB_static.to_dict()
    pC = package.poolC_static.to_dict()

    np.savez(
        npz_path,
        theta_A_deg=np.asarray(package.angle_sets.theta_A_deg, dtype=np.float64),
        theta_B_deg=np.asarray(package.angle_sets.theta_B_deg, dtype=np.float64),
        theta_C_deg=np.asarray(package.angle_sets.theta_C_deg, dtype=np.float64),
        phi_s_deg=np.asarray([package.angle_sets.phi_s_deg], dtype=np.float64),
        rho_3db_A=np.asarray(package.boundaries.rho_3db_A, dtype=np.float64),
        rho_3db_B=np.asarray(package.boundaries.rho_3db_B, dtype=np.float64),
        rho_3db_C=np.asarray(package.boundaries.rho_3db_C, dtype=np.float64),
        rho_null_B=np.asarray(package.boundaries.rho_null_B, dtype=np.float64),
        rho_null_C=np.asarray(package.boundaries.rho_null_C, dtype=np.float64),
        poolA_scan_idx=pA["scan_idx"],
        poolA_theta_s_deg=pA["theta_s_deg"],
        poolA_phi_s_deg=pA["phi_s_deg"],
        poolA_rho=pA["rho"],
        poolA_psi_deg=pA["psi_deg"],
        poolA_theta_deg=pA["theta_deg"],
        poolA_phi_deg=pA["phi_deg"],
        poolA_u=pA["u"],
        poolA_v=pA["v"],
        poolA_zone_code=pA["zone_code"],
        poolA_tag_code=pA["tag_code"],
        poolA_layer_code=pA["layer_code"],
        poolA_priority=pA["priority"],
        poolA_is_dynamic=pA["is_dynamic"],
        poolB_scan_idx=pB["scan_idx"],
        poolB_theta_s_deg=pB["theta_s_deg"],
        poolB_phi_s_deg=pB["phi_s_deg"],
        poolB_rho=pB["rho"],
        poolB_psi_deg=pB["psi_deg"],
        poolB_theta_deg=pB["theta_deg"],
        poolB_phi_deg=pB["phi_deg"],
        poolB_u=pB["u"],
        poolB_v=pB["v"],
        poolB_zone_code=pB["zone_code"],
        poolB_tag_code=pB["tag_code"],
        poolB_layer_code=pB["layer_code"],
        poolB_priority=pB["priority"],
        poolB_is_dynamic=pB["is_dynamic"],
        poolC_scan_idx=pC["scan_idx"],
        poolC_theta_s_deg=pC["theta_s_deg"],
        poolC_phi_s_deg=pC["phi_s_deg"],
        poolC_rho=pC["rho"],
        poolC_psi_deg=pC["psi_deg"],
        poolC_theta_deg=pC["theta_deg"],
        poolC_phi_deg=pC["phi_deg"],
        poolC_u=pC["u"],
        poolC_v=pC["v"],
        poolC_zone_code=pC["zone_code"],
        poolC_tag_code=pC["tag_code"],
        poolC_layer_code=pC["layer_code"],
        poolC_priority=pC["priority"],
        poolC_is_dynamic=pC["is_dynamic"],
    )

    summary = {
        "config": asdict(package.config),
        "angle_sets": {
            "theta_A_deg": np.asarray(package.angle_sets.theta_A_deg, dtype=np.float64).tolist(),
            "theta_B_deg": np.asarray(package.angle_sets.theta_B_deg, dtype=np.float64).tolist(),
            "theta_C_deg": np.asarray(package.angle_sets.theta_C_deg, dtype=np.float64).tolist(),
            "phi_s_deg": float(package.angle_sets.phi_s_deg),
        },
        "boundaries": {
            "rho_3db_A": np.asarray(package.boundaries.rho_3db_A, dtype=np.float64).tolist(),
            "rho_3db_B": np.asarray(package.boundaries.rho_3db_B, dtype=np.float64).tolist(),
            "rho_3db_C": np.asarray(package.boundaries.rho_3db_C, dtype=np.float64).tolist(),
            "rho_null_B": np.asarray(package.boundaries.rho_null_B, dtype=np.float64).tolist(),
            "rho_null_C": np.asarray(package.boundaries.rho_null_C, dtype=np.float64).tolist(),
        },
        "pool_semantics": {
            "Pool-A": "rho <= rho_3db; used for S_main and AR_main screening",
            "Pool-B": "rho > rho_null; used for PSLL_screen side screening",
            "Pool-C": "rho <= rho_3db plus rho > rho_null; rho_3db < rho <= rho_null is intentionally empty unless include_C_guard=True",
            "C_guard_enabled": bool(getattr(package.config, "include_C_guard", False)),
            "main_metric_mask": "rho <= rho_3db",
            "side_metric_mask": "rho > rho_null + eps_side"
        },
        "boundary_contract": {
            "stage1_mainlobe_paths": ["final_solution.mainlobe", "final_mainlobe", "mainlobe"],
            "rho_3db_key": "alpha_3db_deg",
            "rho_null_priority": ["alpha_guard_null_deg", "alpha_null_deg"],
            "rho_from_alpha": "sin_alpha",
            "coordinate_assumption": "circular scan-centered u-v radius"
        },
        "validation": {
            "static_pool_validated": True,
            "phi_s_supported_deg": [0.0]
        },
        "zone_name": ZONE_NAME,
        "layer_name": LAYER_NAME,
        "tag_name": TAG_NAME,
        "responsibility_boundary": {
            "pool_script_handles": [
                "geometry-only stage-2 point pools",
                "static angle-set specific pools A/B/C",
                "dynamic patch building for pool B and pool C tie-break",
                "local+global coordinate export",
            ],
            "pool_script_does_not_handle": [
                "S_main evaluation",
                "AR evaluation",
                "PSLL_screen evaluation",
                "candidate ranking",
                "(c0,c1) optimization",
                "seed extraction",
            ],
        },
        "counts_A": package.counts_A,
        "counts_B": package.counts_B,
        "counts_C": package.counts_C,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _write_counts_csv(counts_A_csv, package.counts_A)
    _write_counts_csv(counts_B_csv, package.counts_B)
    _write_counts_csv(counts_C_csv, package.counts_C)

    return {
        "npz": npz_path,
        "summary_json": json_path,
        "counts_A_csv": counts_A_csv,
        "counts_B_csv": counts_B_csv,
        "counts_C_csv": counts_C_csv,
    }


def _load_npz_dict(npz_path: str) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def _build_pool_subset(static_npz: Dict[str, np.ndarray], prefix: str, theta_s_deg: float, tol_deg: float) -> Dict[str, np.ndarray]:
    theta_key = f"{prefix}_theta_s_deg"
    mask = _theta_mask(static_npz[theta_key], theta_s_deg, tol_deg)
    out: Dict[str, np.ndarray] = {}
    stem = f"{prefix}_"
    for key, arr in static_npz.items():
        if key.startswith(stem):
            out[key[len(stem):]] = np.asarray(arr)[mask].copy()
    return out


def _build_canonical_by_scan(static_npz: Dict[str, np.ndarray], theta_list: np.ndarray, prefix: str, tol_deg: float) -> Dict[float, Dict[str, np.ndarray]]:
    out: Dict[float, Dict[str, np.ndarray]] = {}
    for theta_s_deg in np.asarray(theta_list, dtype=np.float64).tolist():
        key = _canonical_theta_deg(float(theta_s_deg), tol_deg)
        out[key] = _build_pool_subset(static_npz, prefix, key, tol_deg)
    return out


def _config_for_loaded_context(ctx: Stage2PoolContext) -> Stage2PoolConfig:
    config = dict(ctx.summary.get("config", {})) if isinstance(ctx.summary, Mapping) else {}
    return Stage2PoolConfig(
        phi_s_deg=float(config.get("phi_s_deg", ctx.angle_sets.phi_s_deg)),
        eps_side=float(config.get("eps_side", 1e-6)),
        include_C_guard=bool(config.get("include_C_guard", False)),
        n_C_guard=int(config.get("n_C_guard", 0)),
    )


def validate_stage2_static_slices(ctx: Stage2PoolContext) -> None:
    _validate_angles_and_boundaries(ctx.angle_sets, ctx.boundaries)
    cfg = _config_for_loaded_context(ctx)
    if abs(float(cfg.phi_s_deg)) > 1e-12:
        raise NotImplementedError("Current formal Stage-2 pool builder supports phi_s_deg=0 only")

    total_A = 0
    for i, (theta_s_deg, rho_3db) in enumerate(zip(
        np.asarray(ctx.angle_sets.theta_A_deg, dtype=np.float64).tolist(),
        np.asarray(ctx.boundaries.rho_3db_A, dtype=np.float64).tolist(),
    )):
        key = _canonical_theta_deg(theta_s_deg, ctx.tol_deg)
        if key not in ctx.canonical_poolA_by_scan:
            raise AssertionError(f"theta_s={key} missing in poolA")
        pts = ctx.canonical_poolA_by_scan[key]
        _validate_point_dict(pts, name=f"loaded Pool-A scan {i}", allow_empty=False, expected_scan_idx=i, expected_theta_s_deg=theta_s_deg, expected_dynamic=0, tol=ctx.tol_deg)
        _validate_scan_pool_domains(pts, pool_name="A", rho_3db=rho_3db, rho_null=None, cfg=cfg, require_main=True, require_side=False)
        total_A += int(np.asarray(pts["rho"]).size)

    total_B = 0
    for i, (theta_s_deg, rho_3db, rho_null) in enumerate(zip(
        np.asarray(ctx.angle_sets.theta_B_deg, dtype=np.float64).tolist(),
        np.asarray(ctx.boundaries.rho_3db_B, dtype=np.float64).tolist(),
        np.asarray(ctx.boundaries.rho_null_B, dtype=np.float64).tolist(),
    )):
        key = _canonical_theta_deg(theta_s_deg, ctx.tol_deg)
        if key not in ctx.canonical_poolB_by_scan:
            raise AssertionError(f"theta_s={key} missing in poolB")
        pts = ctx.canonical_poolB_by_scan[key]
        _validate_point_dict(pts, name=f"loaded Pool-B scan {i}", allow_empty=False, expected_scan_idx=i, expected_theta_s_deg=theta_s_deg, expected_dynamic=0, tol=ctx.tol_deg)
        _validate_scan_pool_domains(pts, pool_name="B", rho_3db=rho_3db, rho_null=rho_null, cfg=cfg, require_main=False, require_side=True)
        total_B += int(np.asarray(pts["rho"]).size)

    total_C = 0
    for i, (theta_s_deg, rho_3db, rho_null) in enumerate(zip(
        np.asarray(ctx.angle_sets.theta_C_deg, dtype=np.float64).tolist(),
        np.asarray(ctx.boundaries.rho_3db_C, dtype=np.float64).tolist(),
        np.asarray(ctx.boundaries.rho_null_C, dtype=np.float64).tolist(),
    )):
        key = _canonical_theta_deg(theta_s_deg, ctx.tol_deg)
        if key not in ctx.canonical_poolC_by_scan:
            raise AssertionError(f"theta_s={key} missing in poolC")
        pts = ctx.canonical_poolC_by_scan[key]
        _validate_point_dict(pts, name=f"loaded Pool-C scan {i}", allow_empty=False, expected_scan_idx=i, expected_theta_s_deg=theta_s_deg, expected_dynamic=0, tol=ctx.tol_deg)
        _validate_scan_pool_domains(pts, pool_name="C", rho_3db=rho_3db, rho_null=rho_null, cfg=cfg, require_main=True, require_side=True)
        total_C += int(np.asarray(pts["rho"]).size)

    expected_A = int(np.asarray(ctx.static_npz.get("poolA_rho", np.zeros((0,)))).size)
    expected_B = int(np.asarray(ctx.static_npz.get("poolB_rho", np.zeros((0,)))).size)
    expected_C = int(np.asarray(ctx.static_npz.get("poolC_rho", np.zeros((0,)))).size)
    if total_A != expected_A or total_B != expected_B or total_C != expected_C:
        raise AssertionError(
            f"static pool slice coverage mismatch: A {total_A}/{expected_A}, "
            f"B {total_B}/{expected_B}, C {total_C}/{expected_C}"
        )

def load_stage2_pool_context(npz_path: str, json_path: str, *, tol_deg: float = 1e-9, verbose: bool = True) -> Stage2PoolContext:
    static_npz = _load_npz_dict(npz_path)
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    angle_sets = Stage2AngleSets(
        theta_A_deg=np.asarray(static_npz["theta_A_deg"], dtype=np.float64),
        theta_B_deg=np.asarray(static_npz["theta_B_deg"], dtype=np.float64),
        theta_C_deg=np.asarray(static_npz["theta_C_deg"], dtype=np.float64),
        phi_s_deg=float(np.asarray(static_npz["phi_s_deg"], dtype=np.float64).reshape(-1)[0]),
    )
    boundaries = Stage2BoundaryTables(
        rho_3db_A=np.asarray(static_npz["rho_3db_A"], dtype=np.float64),
        rho_3db_B=np.asarray(static_npz["rho_3db_B"], dtype=np.float64),
        rho_3db_C=np.asarray(static_npz["rho_3db_C"], dtype=np.float64),
        rho_null_B=np.asarray(static_npz["rho_null_B"], dtype=np.float64),
        rho_null_C=np.asarray(static_npz["rho_null_C"], dtype=np.float64),
    )
    ctx = Stage2PoolContext(
        static_npz=static_npz,
        summary=summary,
        angle_sets=angle_sets,
        boundaries=boundaries,
        tol_deg=float(tol_deg),
        canonical_poolA_by_scan=_build_canonical_by_scan(static_npz, angle_sets.theta_A_deg, "poolA", tol_deg),
        canonical_poolB_by_scan=_build_canonical_by_scan(static_npz, angle_sets.theta_B_deg, "poolB", tol_deg),
        canonical_poolC_by_scan=_build_canonical_by_scan(static_npz, angle_sets.theta_C_deg, "poolC", tol_deg),
        dynamic_poolB_by_scan={},
        dynamic_poolC_by_scan={},
        static_slice_validated=False,
    )
    validate_stage2_static_slices(ctx)
    ctx.static_slice_validated = True
    if verbose:
        print("=== [STAGE2][POOL] LOAD CONTEXT: DONE ===")
        print(f"[POOL] A scans={len(angle_sets.theta_A_deg)} B scans={len(angle_sets.theta_B_deg)} C scans={len(angle_sets.theta_C_deg)}")
        print(f"[POOL] canonical A/B/C cached={len(ctx.canonical_poolA_by_scan)}/{len(ctx.canonical_poolB_by_scan)}/{len(ctx.canonical_poolC_by_scan)}")
    return ctx


# -----------------------------------------------------------------------------
# Runtime merge helpers
# -----------------------------------------------------------------------------


def concat_point_dicts(static_points: Dict[str, np.ndarray], dynamic_points: Optional[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if dynamic_points is None or int(np.asarray(dynamic_points.get("rho", np.zeros((0,)))).size) == 0:
        return {k: np.asarray(v).copy() for k, v in static_points.items()}
    out: Dict[str, np.ndarray] = {}
    all_keys = set(static_points.keys()) | set(dynamic_points.keys())
    n_static = int(np.asarray(static_points.get("rho", np.zeros((0,)))).size)
    n_dynamic = int(np.asarray(dynamic_points.get("rho", np.zeros((0,)))).size)
    for key in sorted(all_keys):
        s = static_points.get(key, None)
        d = dynamic_points.get(key, None)
        if s is None:
            dtype = np.asarray(d).dtype
            s = np.full((n_static,), np.nan if np.issubdtype(dtype, np.floating) else -1, dtype=dtype)
        if d is None:
            dtype = np.asarray(s).dtype
            d = np.full((n_dynamic,), np.nan if np.issubdtype(dtype, np.floating) else -1, dtype=dtype)
        out[key] = np.concatenate([np.asarray(s), np.asarray(d)], axis=0)
    return out


def get_poolA_points_for_scan(ctx: Stage2PoolContext, theta_s_deg: float) -> Dict[str, np.ndarray]:
    key = _canonical_theta_deg(theta_s_deg, ctx.tol_deg)
    if key not in ctx.canonical_poolA_by_scan:
        raise KeyError(f"theta_s_deg={theta_s_deg} not found in Pool-A")
    return {k: np.asarray(v).copy() for k, v in ctx.canonical_poolA_by_scan[key].items()}


def get_poolB_points_for_scan(ctx: Stage2PoolContext, theta_s_deg: float, *, include_dynamic: bool = True) -> Dict[str, np.ndarray]:
    key = _canonical_theta_deg(theta_s_deg, ctx.tol_deg)
    if key not in ctx.canonical_poolB_by_scan:
        raise KeyError(f"theta_s_deg={theta_s_deg} not found in Pool-B")
    static_pts = {k: np.asarray(v).copy() for k, v in ctx.canonical_poolB_by_scan[key].items()}
    return concat_point_dicts(static_pts, ctx.dynamic_poolB_by_scan.get(key, None) if include_dynamic else None)


def get_poolC_points_for_scan(ctx: Stage2PoolContext, theta_s_deg: float, *, include_dynamic: bool = True) -> Dict[str, np.ndarray]:
    key = _canonical_theta_deg(theta_s_deg, ctx.tol_deg)
    if key not in ctx.canonical_poolC_by_scan:
        raise KeyError(f"theta_s_deg={theta_s_deg} not found in Pool-C")
    static_pts = {k: np.asarray(v).copy() for k, v in ctx.canonical_poolC_by_scan[key].items()}
    return concat_point_dicts(static_pts, ctx.dynamic_poolC_by_scan.get(key, None) if include_dynamic else None)


def get_all_poolB_points(ctx: Stage2PoolContext, *, include_dynamic: bool = True) -> Dict[float, Dict[str, np.ndarray]]:
    return {float(t): get_poolB_points_for_scan(ctx, float(t), include_dynamic=include_dynamic) for t in np.asarray(ctx.angle_sets.theta_B_deg, dtype=np.float64).tolist()}


def get_all_poolC_points(ctx: Stage2PoolContext, *, include_dynamic: bool = True) -> Dict[float, Dict[str, np.ndarray]]:
    return {float(t): get_poolC_points_for_scan(ctx, float(t), include_dynamic=include_dynamic) for t in np.asarray(ctx.angle_sets.theta_C_deg, dtype=np.float64).tolist()}


# -----------------------------------------------------------------------------
# Dynamic patch builders and context operations
# -----------------------------------------------------------------------------


def _build_dynamic_patch_from_offsets(
    scan_idx: int,
    theta_s_deg: float,
    phi_s_deg: float,
    seed_rho: np.ndarray,
    seed_psi_deg: np.ndarray,
    dr_list: Sequence[float],
    dpsi_list_deg: Sequence[float],
    *,
    zone_code: int,
    tag_code: int,
    layer_code: int,
    priority: int,
    rho_min: float,
    rho_max: float,
) -> Dict[str, np.ndarray]:
    if len(seed_rho) == 0:
        return _empty_point_dict()
    theta_s_rad = float(deg2rad(theta_s_deg))
    u_s = math.sin(theta_s_rad) * math.cos(float(deg2rad(phi_s_deg)))
    v_s = math.sin(theta_s_rad) * math.sin(float(deg2rad(phi_s_deg)))
    rho_all: List[float] = []
    psi_all: List[float] = []
    for r0, p0 in zip(np.asarray(seed_rho, dtype=np.float64).tolist(), np.asarray(seed_psi_deg, dtype=np.float64).tolist()):
        for dr in dr_list:
            for sign in (-1.0, 1.0):
                rr = float(r0) + sign * float(dr)
                for dp in dpsi_list_deg:
                    rho_all.append(rr)
                    psi_all.append(float(p0) + float(dp))
    rho_arr = np.asarray(rho_all, dtype=np.float64)
    psi_deg_arr = np.mod(np.asarray(psi_all, dtype=np.float64), 360.0)
    mask = (rho_arr > float(rho_min) + 1e-12) & (rho_arr <= float(rho_max) + 1e-12)
    rho_arr = rho_arr[mask]
    psi_deg_arr = psi_deg_arr[mask]
    psi_rad = deg2rad(psi_deg_arr)
    u = u_s + rho_arr * np.cos(psi_rad)
    v = v_s + rho_arr * np.sin(psi_rad)
    pts = _make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u, v, rho_arr, psi_deg_arr, zone_code, tag_code, layer_code, priority, 1)
    return pts


def _build_local_null_arc(
    scan_idx: int,
    theta_s_deg: float,
    phi_s_deg: float,
    rho_null: float,
    center_psi_deg: float,
    dr_list: Sequence[float],
    dpsi_list_deg: Sequence[float],
    *,
    layer_code: int,
    rho_max: float,
) -> Dict[str, np.ndarray]:
    theta_s_rad = float(deg2rad(theta_s_deg))
    u_s = math.sin(theta_s_rad) * math.cos(float(deg2rad(phi_s_deg)))
    v_s = math.sin(theta_s_rad) * math.sin(float(deg2rad(phi_s_deg)))
    rho_all: List[float] = []
    psi_all: List[float] = []
    for dr in dr_list:
        rr = float(rho_null) + float(dr)
        for dp in dpsi_list_deg:
            rho_all.append(rr)
            psi_all.append(float(center_psi_deg) + float(dp))
    rho_arr = np.asarray(rho_all, dtype=np.float64)
    psi_deg_arr = np.mod(np.asarray(psi_all, dtype=np.float64), 360.0)
    mask = rho_arr <= float(rho_max) + 1e-12
    rho_arr = rho_arr[mask]
    psi_deg_arr = psi_deg_arr[mask]
    psi_rad = deg2rad(psi_deg_arr)
    u = u_s + rho_arr * np.cos(psi_rad)
    v = v_s + rho_arr * np.sin(psi_rad)
    return _make_point_dict_from_uv(scan_idx, theta_s_deg, phi_s_deg, u, v, rho_arr, psi_deg_arr, ZONE_SIDE, TAG_DYNAMIC_NULL_ARC, layer_code, 3, 1)


def _validate_side_seed_set(seeds: Stage2SideSeedSet, *, theta_s_deg: float, name: str, tol: float = 1e-9) -> None:
    if not np.isfinite(float(seeds.theta_s_deg)) or abs(float(seeds.theta_s_deg) - float(theta_s_deg)) > float(tol):
        raise ValueError(f"{name}: seeds.theta_s_deg={seeds.theta_s_deg} does not match theta_s_deg={theta_s_deg}")
    near_rho = np.asarray(seeds.near_seed_rho, dtype=np.float64).reshape(-1)
    near_psi = np.asarray(seeds.near_seed_psi_deg, dtype=np.float64).reshape(-1)
    other_rho = np.asarray(seeds.other_seed_rho, dtype=np.float64).reshape(-1)
    other_psi = np.asarray(seeds.other_seed_psi_deg, dtype=np.float64).reshape(-1)
    other_tag = np.asarray(seeds.other_seed_tag).reshape(-1)
    if near_rho.size != near_psi.size:
        raise ValueError(f"{name}: near seed rho/psi length mismatch")
    if not (other_rho.size == other_psi.size == other_tag.size):
        raise ValueError(f"{name}: other seed rho/psi/tag length mismatch")
    for label, arr in (("near_seed_rho", near_rho), ("near_seed_psi_deg", near_psi), ("other_seed_rho", other_rho), ("other_seed_psi_deg", other_psi)):
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name}: {label} contains NaN/Inf")
    if np.any(near_rho < 0.0) or np.any(other_rho < 0.0):
        raise ValueError(f"{name}: seed rho values must be non-negative")


def build_poolB_dynamic_patch(theta_s_deg: float, rho_null: float, seeds: Stage2SideSeedSet, cfg: Stage2PoolConfig, *, verbose: bool = True) -> Dict[str, np.ndarray]:
    if not np.isfinite(float(rho_null)) or float(rho_null) <= 0.0:
        raise ValueError("Pool-B dynamic rho_null must be positive finite")
    if abs(float(cfg.phi_s_deg)) > 1e-12:
        raise NotImplementedError("Current formal Stage-2 pool builder supports phi_s_deg=0 only")
    _validate_side_seed_set(seeds, theta_s_deg=theta_s_deg, name="Pool-B dynamic seeds")
    phi_s_deg = float(cfg.phi_s_deg)
    rho_rim = _visible_rho_cap(theta_s_deg, cfg)
    parts: List[Dict[str, np.ndarray]] = []
    near_seed_rho = np.asarray(seeds.near_seed_rho, dtype=np.float64)[: cfg.dynamic.n_near_seed_B]
    near_seed_psi = np.asarray(seeds.near_seed_psi_deg, dtype=np.float64)[: cfg.dynamic.n_near_seed_B]
    if near_seed_rho.size:
        parts.append(_build_dynamic_patch_from_offsets(
            0, theta_s_deg, phi_s_deg, near_seed_rho, near_seed_psi,
            cfg.dynamic.near_dr_list_B, cfg.dynamic.near_dpsi_list_deg_B,
            zone_code=ZONE_SIDE, tag_code=TAG_DYNAMIC_NEAR_PATCH, layer_code=LAYER_B_DYNAMIC,
            priority=1, rho_min=float(rho_null) + cfg.eps_side, rho_max=rho_rim,
        ))
        parts.append(_build_local_null_arc(
            0, theta_s_deg, phi_s_deg, float(rho_null), float(near_seed_psi[0]),
            cfg.dynamic.null_arc_dr_list_B, cfg.dynamic.null_arc_dpsi_list_deg_B,
            layer_code=LAYER_B_DYNAMIC, rho_max=rho_rim,
        ))

    other_seed_rho = np.asarray(seeds.other_seed_rho, dtype=np.float64)[: cfg.dynamic.n_other_seed_B]
    other_seed_psi = np.asarray(seeds.other_seed_psi_deg, dtype=np.float64)[: cfg.dynamic.n_other_seed_B]
    other_tags = np.asarray(seeds.other_seed_tag, dtype=np.int16)[: cfg.dynamic.n_other_seed_B]
    for r0, p0, tg in zip(other_seed_rho.tolist(), other_seed_psi.tolist(), other_tags.tolist()):
        dyn_tag = TAG_DYNAMIC_FAR_PATCH
        if int(tg) in (TAG_B_HORIZON, TAG_C_HORIZON):
            dyn_tag = TAG_DYNAMIC_HORIZON_PATCH
        elif int(tg) in (TAG_B_BACK, TAG_C_BACK):
            dyn_tag = TAG_DYNAMIC_BACK_PATCH
        parts.append(_build_dynamic_patch_from_offsets(
            0, theta_s_deg, phi_s_deg, np.asarray([r0]), np.asarray([p0]),
            cfg.dynamic.other_dr_list_B, cfg.dynamic.other_dpsi_list_deg_B,
            zone_code=ZONE_SIDE, tag_code=dyn_tag, layer_code=LAYER_B_DYNAMIC,
            priority=2, rho_min=float(rho_null) + cfg.eps_side, rho_max=rho_rim,
        ))

    pts = _concat_point_dicts_raw(parts)
    pts = _filter_point_dict(pts, theta_s_deg=theta_s_deg, rho_lo=float(rho_null) + cfg.eps_side, rho_hi=rho_rim, visible_only=True)
    pts = _dedup_trim_fill(pts, n_target=124, quant_deg=cfg.quant_deg)
    _validate_point_dict(pts, name="Pool-B dynamic patch", allow_empty=True, expected_theta_s_deg=theta_s_deg, expected_dynamic=1)
    if int(np.asarray(pts["rho"]).size) > 0:
        if np.any(np.asarray(pts["zone_code"], dtype=np.int64) != ZONE_SIDE):
            raise ValueError("Pool-B dynamic patch must contain side-domain points only")
    if verbose:
        print(f"[STAGE2][POOL][B-DYN] theta_s={theta_s_deg:.3f} dynamic_points={int(np.asarray(pts['rho']).size)}")
    return pts


def build_poolC_tiebreak_patch(theta_s_deg: float, rho_null: float, seeds: Stage2SideSeedSet, cfg: Stage2PoolConfig, *, verbose: bool = True) -> Dict[str, np.ndarray]:
    if not np.isfinite(float(rho_null)) or float(rho_null) <= 0.0:
        raise ValueError("Pool-C dynamic rho_null must be positive finite")
    if abs(float(cfg.phi_s_deg)) > 1e-12:
        raise NotImplementedError("Current formal Stage-2 pool builder supports phi_s_deg=0 only")
    _validate_side_seed_set(seeds, theta_s_deg=theta_s_deg, name="Pool-C dynamic seeds")
    phi_s_deg = float(cfg.phi_s_deg)
    rho_rim = _visible_rho_cap(theta_s_deg, cfg)
    parts: List[Dict[str, np.ndarray]] = []
    near_seed_rho = np.asarray(seeds.near_seed_rho, dtype=np.float64)[: cfg.dynamic.n_near_seed_C]
    near_seed_psi = np.asarray(seeds.near_seed_psi_deg, dtype=np.float64)[: cfg.dynamic.n_near_seed_C]
    if near_seed_rho.size:
        parts.append(_build_dynamic_patch_from_offsets(
            0, theta_s_deg, phi_s_deg, near_seed_rho, near_seed_psi,
            (0.01,), (-6.0, 0.0, 6.0),
            zone_code=ZONE_SIDE, tag_code=TAG_DYNAMIC_NEAR_PATCH, layer_code=LAYER_C_DYNAMIC,
            priority=1, rho_min=float(rho_null) + cfg.eps_side, rho_max=rho_rim,
        ))
    other_seed_rho = np.asarray(seeds.other_seed_rho, dtype=np.float64)[: cfg.dynamic.n_other_seed_C]
    other_seed_psi = np.asarray(seeds.other_seed_psi_deg, dtype=np.float64)[: cfg.dynamic.n_other_seed_C]
    other_tags = np.asarray(seeds.other_seed_tag, dtype=np.int16)[: cfg.dynamic.n_other_seed_C]
    for r0, p0, tg in zip(other_seed_rho.tolist(), other_seed_psi.tolist(), other_tags.tolist()):
        dyn_tag = TAG_DYNAMIC_FAR_PATCH
        if int(tg) in (TAG_B_HORIZON, TAG_C_HORIZON):
            dyn_tag = TAG_DYNAMIC_HORIZON_PATCH
        elif int(tg) in (TAG_B_BACK, TAG_C_BACK):
            dyn_tag = TAG_DYNAMIC_BACK_PATCH
        parts.append(_build_dynamic_patch_from_offsets(
            0, theta_s_deg, phi_s_deg, np.asarray([r0]), np.asarray([p0]),
            (0.02,), (-8.0, 0.0, 8.0),
            zone_code=ZONE_SIDE, tag_code=dyn_tag, layer_code=LAYER_C_DYNAMIC,
            priority=2, rho_min=float(rho_null) + cfg.eps_side, rho_max=rho_rim,
        ))
    pts = _concat_point_dicts_raw(parts)
    pts = _filter_point_dict(pts, theta_s_deg=theta_s_deg, rho_lo=float(rho_null) + cfg.eps_side, rho_hi=rho_rim, visible_only=True)
    pts = _dedup_trim_fill(pts, n_target=32, quant_deg=cfg.quant_deg)
    _validate_point_dict(pts, name="Pool-C dynamic patch", allow_empty=True, expected_theta_s_deg=theta_s_deg, expected_dynamic=1)
    if int(np.asarray(pts["rho"]).size) > 0:
        if np.any(np.asarray(pts["zone_code"], dtype=np.int64) != ZONE_SIDE):
            raise ValueError("Pool-C dynamic patch must contain side-domain points only")
    if verbose:
        print(f"[STAGE2][POOL][C-DYN] theta_s={theta_s_deg:.3f} dynamic_points={int(np.asarray(pts['rho']).size)}")
    return pts


def clear_poolB_dynamic_points_for_scan(ctx: Stage2PoolContext, theta_s_deg: float, *, verbose: bool = True) -> None:
    key = _canonical_theta_deg(theta_s_deg, ctx.tol_deg)
    ctx.dynamic_poolB_by_scan.pop(key, None)
    if verbose:
        print(f"[STAGE2][POOL][B-DYN] cleared theta_s={key:.3f}")


def clear_poolC_dynamic_points_for_scan(ctx: Stage2PoolContext, theta_s_deg: float, *, verbose: bool = True) -> None:
    key = _canonical_theta_deg(theta_s_deg, ctx.tol_deg)
    ctx.dynamic_poolC_by_scan.pop(key, None)
    if verbose:
        print(f"[STAGE2][POOL][C-DYN] cleared theta_s={key:.3f}")


def clear_all_poolB_dynamic_points(ctx: Stage2PoolContext, *, verbose: bool = True) -> None:
    ctx.dynamic_poolB_by_scan.clear()
    if verbose:
        print("[STAGE2][POOL][B-DYN] cleared all")


def clear_all_poolC_dynamic_points(ctx: Stage2PoolContext, *, verbose: bool = True) -> None:
    ctx.dynamic_poolC_by_scan.clear()
    if verbose:
        print("[STAGE2][POOL][C-DYN] cleared all")




def _find_scan_index(theta_list_deg: np.ndarray, theta_s_deg: float, tol_deg: float) -> int:
    theta = np.asarray(theta_list_deg, dtype=np.float64).reshape(-1)
    hit = np.where(np.isclose(theta, float(theta_s_deg), atol=float(tol_deg), rtol=0.0))[0]
    if hit.size == 0:
        raise KeyError(f"theta_s_deg={theta_s_deg} not found in scan list")
    return int(hit[0])


def _with_scan_idx(points: Dict[str, np.ndarray], scan_idx: int) -> Dict[str, np.ndarray]:
    out = {k: np.asarray(v).copy() for k, v in points.items()}
    n = int(np.asarray(out.get("rho", np.zeros((0,)))).size)
    if n > 0:
        out["scan_idx"] = np.full((n,), int(scan_idx), dtype=np.int32)
    return out

def attach_poolB_dynamic_points_for_scan(ctx: Stage2PoolContext, theta_s_deg: float, dynamic_points: Dict[str, np.ndarray], *, verbose: bool = True) -> None:
    key = _canonical_theta_deg(theta_s_deg, ctx.tol_deg)
    scan_idx = _find_scan_index(ctx.angle_sets.theta_B_deg, theta_s_deg, ctx.tol_deg)
    _validate_point_dict(dynamic_points, name="attached Pool-B dynamic points", allow_empty=True, expected_theta_s_deg=theta_s_deg, expected_dynamic=1, tol=ctx.tol_deg)
    if int(np.asarray(dynamic_points["rho"]).size) > 0 and np.any(np.asarray(dynamic_points["zone_code"], dtype=np.int64) != ZONE_SIDE):
        raise ValueError("attached Pool-B dynamic points must be side-domain points")
    stored = _with_scan_idx(dynamic_points, scan_idx)
    _validate_point_dict(stored, name="stored Pool-B dynamic points", allow_empty=True, expected_scan_idx=scan_idx, expected_theta_s_deg=theta_s_deg, expected_dynamic=1, tol=ctx.tol_deg)
    ctx.dynamic_poolB_by_scan[key] = stored
    if verbose:
        print(f"[STAGE2][POOL][B-DYN] attached theta_s={key:.3f} count={int(np.asarray(dynamic_points['rho']).size)}")


def attach_poolC_dynamic_points_for_scan(ctx: Stage2PoolContext, theta_s_deg: float, dynamic_points: Dict[str, np.ndarray], *, verbose: bool = True) -> None:
    key = _canonical_theta_deg(theta_s_deg, ctx.tol_deg)
    scan_idx = _find_scan_index(ctx.angle_sets.theta_C_deg, theta_s_deg, ctx.tol_deg)
    _validate_point_dict(dynamic_points, name="attached Pool-C dynamic points", allow_empty=True, expected_theta_s_deg=theta_s_deg, expected_dynamic=1, tol=ctx.tol_deg)
    if int(np.asarray(dynamic_points["rho"]).size) > 0 and np.any(np.asarray(dynamic_points["zone_code"], dtype=np.int64) != ZONE_SIDE):
        raise ValueError("attached Pool-C dynamic points must be side-domain points")
    stored = _with_scan_idx(dynamic_points, scan_idx)
    _validate_point_dict(stored, name="stored Pool-C dynamic points", allow_empty=True, expected_scan_idx=scan_idx, expected_theta_s_deg=theta_s_deg, expected_dynamic=1, tol=ctx.tol_deg)
    ctx.dynamic_poolC_by_scan[key] = stored
    if verbose:
        print(f"[STAGE2][POOL][C-DYN] attached theta_s={key:.3f} count={int(np.asarray(dynamic_points['rho']).size)}")

def rebuild_poolB_dynamic_points_for_scan(ctx: Stage2PoolContext, theta_s_deg: float, rho_null: float, seeds: Stage2SideSeedSet, cfg: Stage2PoolConfig, *, verbose: bool = True) -> None:
    clear_poolB_dynamic_points_for_scan(ctx, theta_s_deg, verbose=False)
    dyn = build_poolB_dynamic_patch(theta_s_deg, rho_null, seeds, cfg, verbose=verbose)
    attach_poolB_dynamic_points_for_scan(ctx, theta_s_deg, dyn, verbose=verbose)


def rebuild_poolC_dynamic_points_for_scan(ctx: Stage2PoolContext, theta_s_deg: float, rho_null: float, seeds: Stage2SideSeedSet, cfg: Stage2PoolConfig, *, verbose: bool = True) -> None:
    clear_poolC_dynamic_points_for_scan(ctx, theta_s_deg, verbose=False)
    dyn = build_poolC_tiebreak_patch(theta_s_deg, rho_null, seeds, cfg, verbose=verbose)
    attach_poolC_dynamic_points_for_scan(ctx, theta_s_deg, dyn, verbose=verbose)


# -----------------------------------------------------------------------------
# Defaults and CLI
# -----------------------------------------------------------------------------


def make_default_angle_sets() -> Stage2AngleSets:
    return Stage2AngleSets(
        theta_A_deg=np.asarray([0, 15, 30, 45, 50, 55, 58, 60], dtype=np.float64),
        theta_B_deg=np.asarray([0, 10, 15, 20, 25, 30, 35, 40, 45, 48, 50, 52, 55, 57, 58, 60], dtype=np.float64),
        theta_C_deg=np.arange(0.0, 61.0, 1.0, dtype=np.float64),
        phi_s_deg=0.0,
    )


def synthetic_boundary_tables(angle_sets: Stage2AngleSets) -> Stage2BoundaryTables:
    tA = np.asarray(angle_sets.theta_A_deg, dtype=np.float64)
    tB = np.asarray(angle_sets.theta_B_deg, dtype=np.float64)
    tC = np.asarray(angle_sets.theta_C_deg, dtype=np.float64)
    rho3_A = 0.22 + 0.0012 * tA
    rho3_B = 0.22 + 0.0012 * tB
    rho3_C = 0.22 + 0.0012 * tC
    rhoN_B = rho3_B + 0.12 + 0.0006 * tB
    rhoN_C = rho3_C + 0.12 + 0.0006 * tC
    return Stage2BoundaryTables(rho_3db_A=rho3_A, rho_3db_B=rho3_B, rho_3db_C=rho3_C, rho_null_B=rhoN_B, rho_null_C=rhoN_C)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate stage-2 static point pools.")
    p.add_argument("--base-dir", type=str, default=".", help="Base directory used for automatic boundary JSON lookup.")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--boundary-json", type=str, default="", help="External calibrated boundary JSON. Recommended: a0cra_stage1_main_out.json.")
    p.add_argument("--boundary-format", type=str, default="auto", choices=["auto", "alpha_main_table", "stage1_main_out", "rho_table"])
    p.add_argument("--rho-from-alpha", type=str, default="sin_alpha", choices=["sin_alpha", "identity"])
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--d-min", type=float, default=0.5)
    p.add_argument("--d-eff", type=float, default=0.5)
    p.add_argument("--R0", type=float, default=5.0)
    p.add_argument("--include-center", action="store_true", default=True)
    p.add_argument("--no-include-center", dest="include_center", action="store_false")
    p.add_argument("--outer5-reduce-mult4", action="store_true", default=True)
    p.add_argument("--no-outer5-reduce-mult4", dest="outer5_reduce_mult4", action="store_false")
    p.add_argument("--outer5-threshold-gt", type=int, default=16)
    p.add_argument("--use-synth-boundaries", action="store_true", help="Use synthetic boundary tables for demo generation. Formal runs should not use this.")
    return p.parse_args()


def _resolve_boundary_json_for_cli(base_dir: str, user_path: str) -> str:
    if str(user_path).strip():
        p = Path(str(user_path)).expanduser()
        if not p.is_absolute():
            p = Path(base_dir).expanduser().resolve() / p
        if not p.is_file():
            raise FileNotFoundError(f"Boundary JSON not found: {p}")
        return str(p)

    roots = [Path(base_dir).expanduser().resolve(), Path.cwd().resolve(), Path(__file__).resolve().parent]
    names = ["a0cra_stage1_main_out.json", "stage2_boundary_table.json", "a0cra_alpha_main_table.json"]
    seen = set()
    for root in roots:
        for name in names:
            cand = (root / name).resolve()
            key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            if cand.is_file():
                print(f"[boundary] auto-selected JSON: {cand}")
                return key

    run_candidates: List[Path] = []
    for root in roots:
        run_candidates.extend(root.glob("stage1_runs/*/a0cra_stage1_main_out.json"))
    run_candidates = sorted({p.resolve() for p in run_candidates if p.is_file()}, key=lambda p: str(p))
    if run_candidates:
        selected = run_candidates[-1]
        print(f"[boundary] auto-selected latest Stage1 JSON: {selected}")
        return str(selected)

    raise FileNotFoundError(
        "No boundary JSON found. Pass --boundary-json, or use --use-synth-boundaries for demo generation."
    )


def _describe_boundary_json_source(json_path: str, requested_format: str) -> Dict[str, Any]:
    path = Path(str(json_path)).expanduser()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    fmt = str(requested_format).lower().strip()
    if fmt == "auto":
        fmt = _auto_detect_boundary_format(data)
    out: Dict[str, Any] = {"resolved_format": fmt}
    if fmt == "stage1_main_out":
        ml, block_path = _extract_stage1_mainlobe_block(data)
        out["mainlobe_block"] = block_path
        out["rho_3db_key"] = "alpha_3db_deg"
        out["rho_null_key"] = "alpha_guard_null_deg" if "alpha_guard_null_deg" in ml else "alpha_null_deg"
    elif fmt == "alpha_main_table":
        out["rho_3db_key"] = "alpha_3db_deg"
        out["rho_null_key"] = "alpha_main_deg"
    elif fmt == "rho_table":
        out["rho_3db_key"] = "rho_3db"
        out["rho_null_key"] = "rho_null"
    return out

def main() -> int:
    args = _parse_args()
    angle_sets = make_default_angle_sets()
    boundary_source: Dict[str, Any]
    if bool(args.use_synth_boundaries):
        boundaries = synthetic_boundary_tables(angle_sets)
        boundary_source = {"mode": "synthetic"}
    else:
        boundary_json = _resolve_boundary_json_for_cli(args.base_dir, args.boundary_json)
        boundaries = load_stage2_boundary_tables_from_json(
            boundary_json,
            angle_sets,
            boundary_format=args.boundary_format,
            rho_from_alpha=args.rho_from_alpha,
            K=int(args.K),
            d_min=float(args.d_min),
            d_eff=float(args.d_eff),
            R0=float(args.R0),
            include_center=bool(args.include_center),
            outer5_reduce_mult4=bool(args.outer5_reduce_mult4),
            outer5_threshold_gt=int(args.outer5_threshold_gt),
        )
        source_detail = _describe_boundary_json_source(boundary_json, args.boundary_format)
        boundary_source = {
            "mode": "external_json",
            "path": str(boundary_json),
            "format": str(args.boundary_format),
            "rho_from_alpha": str(args.rho_from_alpha),
            "K": int(args.K),
            "d_min": float(args.d_min),
            "d_eff": float(args.d_eff),
            "R0": float(args.R0),
            **source_detail,
        }
    package = build_stage2_pool_package(angle_sets, boundaries, Stage2PoolConfig())
    out = save_stage2_pool_package(package, os.path.abspath(args.out_dir))

    # Record the boundary source without changing the core npz schema.
    try:
        with open(out["summary_json"], "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["boundary_source"] = boundary_source
        with open(out["summary_json"], "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[warn] failed to append boundary_source to summary: {exc}")

    print("=== STAGE-2 POINT POOL: DONE ===")
    print(f"[boundary_source] {boundary_source}")
    print(f"[saved] npz={out['npz']}")
    print(f"[saved] summary_json={out['summary_json']}")
    print(f"[saved] counts_A_csv={out['counts_A_csv']}")
    print(f"[saved] counts_B_csv={out['counts_B_csv']}")
    print(f"[saved] counts_C_csv={out['counts_C_csv']}")
    return 0


__all__ = [
    "Stage2AngleSets",
    "Stage2BoundaryTables",
    "Stage2DynamicPatchConfig",
    "Stage2PoolConfig",
    "Stage2SideSeedSet",
    "PointBatch2DStage2",
    "Stage2PoolPackage",
    "Stage2PoolContext",
    "build_scan_pool_A",
    "build_scan_pool_B_static",
    "build_scan_pool_C_static",
    "build_stage2_pool_package",
    "save_stage2_pool_package",
    "load_stage2_pool_context",
    "get_poolA_points_for_scan",
    "get_poolB_points_for_scan",
    "get_poolC_points_for_scan",
    "get_all_poolB_points",
    "get_all_poolC_points",
    "build_poolB_dynamic_patch",
    "build_poolC_tiebreak_patch",
    "attach_poolB_dynamic_points_for_scan",
    "attach_poolC_dynamic_points_for_scan",
    "rebuild_poolB_dynamic_points_for_scan",
    "rebuild_poolC_dynamic_points_for_scan",
    "clear_poolB_dynamic_points_for_scan",
    "clear_poolC_dynamic_points_for_scan",
    "clear_all_poolB_dynamic_points",
    "clear_all_poolC_dynamic_points",
    "validate_stage2_static_slices",
    "make_default_angle_sets",
    "synthetic_boundary_tables",
    "load_stage2_boundary_tables_from_json",
    "make_metric_masks_for_points",
    "get_stage2_boundaries_for_scan",
]

if __name__ == "__main__":
    raise SystemExit(main())
