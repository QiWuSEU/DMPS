"""Estimate scan mainlobe domains and calibrate offline exclusion tables.

The co-polar direct-sum pattern is evaluated on radial rays around each
requested scan direction. The first-null/guard-null and 3-dB boundaries are
retained separately; the resulting alpha-main table is distinct from the
intrinsic 1D/2D partition table produced by calibrate_q0_partition.py.

The frozen alpha-main table schema and public function names are preserved.
Run the command-line entry point as a module from the repository root:
    python -m geometric_backbone.scan_mainlobe_domain --help
"""

import math
import json
import argparse
import time
from pathlib import Path
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


# =============================================================================
# Geometry helpers
# =============================================================================

def deg2rad(x: Union[np.ndarray, float]) -> np.ndarray:
    return np.deg2rad(np.asarray(x, dtype=np.float64))


def rad2deg(x: Union[np.ndarray, float]) -> np.ndarray:
    return np.rad2deg(np.asarray(x, dtype=np.float64))


def wrap_phi_0_2pi(phi_rad: Union[np.ndarray, float]) -> np.ndarray:
    x = np.asarray(phi_rad, dtype=np.float64)
    twopi = 2.0 * math.pi
    y = np.mod(x, twopi)
    y = np.where(y < 0.0, y + twopi, y)
    return y.astype(np.float64)


def uv_from_theta_phi(theta_rad: Union[np.ndarray, float], phi_rad: Union[np.ndarray, float]) -> Tuple[np.ndarray, np.ndarray]:
    th = np.asarray(theta_rad, dtype=np.float64)
    ph = np.asarray(phi_rad, dtype=np.float64)
    s = np.sin(th)
    return (s * np.cos(ph)).astype(np.float64), (s * np.sin(ph)).astype(np.float64)


def theta_phi_from_unit_vectors(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    zc = np.clip(z, -1.0, 1.0)
    theta = np.arccos(zc).astype(np.float64)
    phi = wrap_phi_0_2pi(np.arctan2(y, x))
    return theta, phi


def local_spherical_points(theta_s_rad: float, phi_s_rad: float, alpha_rad: np.ndarray, psi_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scan-centered local spherical coordinates:
      alpha: spherical angular distance from scan center
      psi:   local azimuth around scan center

    Returns global (theta, phi).
    """
    ths = float(theta_s_rad)
    phs = float(phi_s_rad)
    a = np.asarray(alpha_rad, dtype=np.float64)
    psi = float(psi_rad)

    s = np.array([
        math.sin(ths) * math.cos(phs),
        math.sin(ths) * math.sin(phs),
        math.cos(ths),
    ], dtype=np.float64)

    e1 = np.array([
        math.cos(ths) * math.cos(phs),
        math.cos(ths) * math.sin(phs),
        -math.sin(ths),
    ], dtype=np.float64)
    e2 = np.array([
        -math.sin(phs),
        math.cos(phs),
        0.0,
    ], dtype=np.float64)

    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)

    vec = (
        np.cos(a)[:, None] * s[None, :]
        + np.sin(a)[:, None] * (math.cos(psi) * e1[None, :] + math.sin(psi) * e2[None, :])
    )
    theta, phi = theta_phi_from_unit_vectors(vec[:, 0], vec[:, 1], vec[:, 2])
    return theta, phi


# =============================================================================
# Array geometry
# =============================================================================


def asdict(obj):
    """Small Python-3.6-compatible replacement for dataclasses.asdict()."""
    if hasattr(obj, "_asdict"):
        return dict(obj._asdict())
    if isinstance(obj, dict):
        return dict(obj)
    raise TypeError("asdict() expects one of this module's record objects")


ArrayGeometry = namedtuple(
    "ArrayGeometry",
    ["r_over_lambda", "N_per_ring", "include_center"],
)
ArrayGeometry.__new__.__defaults__ = (True,)


ScanSpecMainlobe = namedtuple(
    "ScanSpecMainlobe",
    ["theta_s_deg_list", "phi_s_deg"],
)
ScanSpecMainlobe.__new__.__defaults__ = (0.0,)


MainlobeDomainConfig = namedtuple(
    "MainlobeDomainConfig",
    [
        "alpha_eval_max_deg",
        "n_dir",
        "n_alpha_coarse",
        "n_alpha_refine",
        "valley_rel_max",
        "null_guard_scale",
        "db_drop",
        "combine_rule",
        "peak_mode",
        "near_center_max_deg",
        "directsum_block_size",
    ],
)
MainlobeDomainConfig.__new__.__defaults__ = (
    30.0,
    12,
    1200,
    400,
    0.60,
    0.98,
    3.0,
    "median",
    "center",
    1.0,
    256,
)


MainlobeDomainResult = namedtuple(
    "MainlobeDomainResult",
    [
        "theta_s_deg",
        "phi_s_deg",
        "alpha_null_deg",
        "alpha_guard_null_deg",
        "alpha_3db_deg",
        "alpha_null_dirs_deg",
        "alpha_guard_null_dirs_deg",
        "alpha_3db_dirs_deg",
        "psi_dirs_rad",
        "stats",
    ],
)


AlphaMainTablePolicy = namedtuple(
    "AlphaMainTablePolicy",
    [
        "alpha_main_source",
        "reduction_rule",
        "mainlobe_margin_deg",
        "apply_margin_to_alpha_main",
        "alpha_3db_reduction_rule",
        "apply_margin_to_alpha_3db",
        "apply_margin_to_debug_null",
        "apply_margin_to_debug_guard_null",
    ],
)
AlphaMainTablePolicy.__new__.__defaults__ = (
    "guard_null",
    "min_over_b",
    0.5,
    True,
    "min_over_b",
    False,
    True,
    True,
)


AlphaMainTableSweepConfig = namedtuple(
    "AlphaMainTableSweepConfig",
    [
        "unit_cell_npz_path",
        "unit_cell_meta_json_path",
        "out_dir",
        "K_list",
        "d_min_list",
        "R0",
        "bind_d_eff_to_d_min",
        "d_eff_default",
        "b_min",
        "b_max",
        "b_step",
        "theta_s_min_deg",
        "theta_s_max_deg",
        "theta_s_step_deg",
        "include_center",
        "outer5_reduce_mult4",
        "outer5_threshold_gt",
        "alpha_eval_max_deg",
        "n_dir",
        "n_alpha_coarse",
        "n_alpha_refine",
        "valley_rel_max",
        "null_guard_scale",
        "db_drop",
        "combine_rule",
        "peak_mode",
        "near_center_max_deg",
        "directsum_block_size",
        "alpha_main_source",
        "mainlobe_margin_deg",
    ],
)
AlphaMainTableSweepConfig.__new__.__defaults__ = (
    "",
    "",
    ".",
    (6, 7, 8),
    tuple(np.round(np.arange(0.50, 0.601, 0.01), 2).tolist()),
    5.0,
    True,
    0.5,
    6.0,
    12.0,
    0.25,
    0.0,
    60.0,
    1.0,
    True,
    True,
    16,
    30.0,
    12,
    1200,
    400,
    0.60,
    0.98,
    3.0,
    "median",
    "center",
    1.0,
    256,
    "guard_null",
    0.5,
)


# =============================================================================
# AOSCRA-compatible initialization helpers
# =============================================================================

def init_r_gompertz(*, K: int = 8, R0: float = 5.0, d_min: float = 0.5, b: float = 6.0) -> np.ndarray:
    if K < 1:
        raise ValueError("K must be >= 1")
    if d_min <= 0 or R0 <= 0:
        raise ValueError("d_min and R0 must be > 0")
    dK = K * d_min
    if R0 < dK - 1e-12:
        raise ValueError(f"R0({R0}) < K*d_min({dK})")

    m = np.arange(1, K + 1, dtype=np.float64)
    d_m = m * d_min
    if K == 1:
        return np.array([R0], dtype=np.float64)

    t_m = (m - 1.0) / (K - 1.0)
    if abs(b) < 1e-12:
        gbar = t_m
    else:
        denom = np.exp(b) - 1.0
        gbar = (np.exp(b * t_m) - 1.0) / denom if abs(denom) > 1e-15 else t_m

    l_m = (R0 - dK) * gbar
    r = d_m + l_m
    r[0] = d_min
    r[-1] = R0
    if np.any(np.diff(r) <= -1e-9):
        raise ValueError("Non-monotone Gompertz radii produced.")
    return r.astype(np.float64)


def derive_ring_counts(
    r_over_lambda: np.ndarray,
    *,
    d_eff: float = 0.5,
    outer5_reduce_mult4: bool = True,
    outer5_threshold_gt: int = 16,
) -> np.ndarray:
    r = np.asarray(r_over_lambda, dtype=np.float64).reshape(-1)
    if d_eff <= 0:
        raise ValueError("d_eff must be > 0")
    n_full = np.floor((2.0 * math.pi * r) / d_eff + 1e-12).astype(np.int64)
    n_full = np.maximum(n_full, 1)
    n_used = n_full.copy()
    if outer5_reduce_mult4 and r.size >= 5:
        start = max(0, r.size - 5)
        for k in range(start, r.size):
            if int(n_used[k]) > int(outer5_threshold_gt) and int(n_used[k]) % 4 == 0:
                n_used[k] -= 1
    return n_used.astype(np.int64)


def build_element_angles(N_vec: np.ndarray) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for N in np.asarray(N_vec, dtype=np.int64).reshape(-1):
        psi = 2.0 * math.pi * np.arange(int(N), dtype=np.float64) / float(N)
        out.append(psi)
    return out


def build_array_geometry_from_init_b6(
    *,
    K: int = 8,
    R0: float = 5.0,
    d_min: float = 0.5,
    b: float = 6.0,
    d_eff: float = 0.5,
    include_center: bool = True,
    outer5_reduce_mult4: bool = True,
    outer5_threshold_gt: int = 16,
) -> ArrayGeometry:
    r = init_r_gompertz(K=K, R0=R0, d_min=d_min, b=b)
    N = derive_ring_counts(
        r,
        d_eff=d_eff,
        outer5_reduce_mult4=outer5_reduce_mult4,
        outer5_threshold_gt=outer5_threshold_gt,
    )
    return ArrayGeometry(r_over_lambda=r, N_per_ring=N, include_center=include_center)


# =============================================================================
# Direct-sum co pattern evaluator
# =============================================================================

def build_directsum_pattern_evaluator(
    eco_eval: Callable[[np.ndarray, np.ndarray], np.ndarray],
    geometry: ArrayGeometry,
    *,
    block_size: int = 256,
) -> Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]:
    """
    Returns pattern_eval(theta_rad, phi_rad, theta_s_rad, phi_s_rad) -> complex ndarray.

    Array-factor direct sum with ring first element fixed at y=0:
      psi_{k,n} = 2*pi*n / N_k
    """
    r = np.asarray(geometry.r_over_lambda, dtype=np.float64).reshape(-1)
    N = np.asarray(geometry.N_per_ring, dtype=np.int64).reshape(-1)
    if r.size != N.size:
        raise ValueError("r_over_lambda and N_per_ring must have same length.")
    if np.any(N <= 0):
        raise ValueError("N_per_ring must be positive.")

    xs_list = []
    ys_list = []
    if geometry.include_center:
        xs_list.append(np.array([0.0], dtype=np.float64))
        ys_list.append(np.array([0.0], dtype=np.float64))
    for rk, Nk in zip(r.tolist(), N.tolist()):
        psi = 2.0 * math.pi * np.arange(int(Nk), dtype=np.float64) / float(Nk)
        xs_list.append(rk * np.cos(psi))
        ys_list.append(rk * np.sin(psi))
    x_elem = np.concatenate(xs_list, axis=0).astype(np.float64)
    y_elem = np.concatenate(ys_list, axis=0).astype(np.float64)
    bs = int(block_size)
    if bs <= 0:
        raise ValueError("block_size must be > 0")

    def pattern_eval(theta_rad: np.ndarray, phi_rad: np.ndarray, theta_s_rad: float, phi_s_rad: float) -> np.ndarray:
        th = np.asarray(theta_rad, dtype=np.float64)
        ph = wrap_phi_0_2pi(np.asarray(phi_rad, dtype=np.float64))
        th, ph = np.broadcast_arrays(th, ph)
        shp = th.shape
        th1 = th.reshape(-1)
        ph1 = ph.reshape(-1)

        u = np.sin(th1) * np.cos(ph1)
        v = np.sin(th1) * np.sin(ph1)
        us = math.sin(float(theta_s_rad)) * math.cos(float(phi_s_rad))
        vs = math.sin(float(theta_s_rad)) * math.sin(float(phi_s_rad))

        af = np.empty(th1.shape, dtype=np.complex128)
        for s0 in range(0, int(th1.size), bs):
            s1 = min(s0 + bs, int(th1.size))
            phase = 2.0 * math.pi * (
                (u[s0:s1, None] - us) * x_elem[None, :]
                + (v[s0:s1, None] - vs) * y_elem[None, :]
            )
            af[s0:s1] = np.exp(1j * phase).sum(axis=1)
        eco = np.asarray(eco_eval(th1, ph1))
        if not np.iscomplexobj(eco):
            if eco.ndim >= 1 and eco.shape[-1] == 2:
                eco = eco[..., 0] + 1j * eco[..., 1]
            else:
                eco = eco.astype(np.complex128)
        eco = eco.reshape(-1)
        out = eco * af
        return out.reshape(shp).astype(np.complex128)

    return pattern_eval


# =============================================================================
# Mainlobe domain extraction
# =============================================================================
# The scan-centered angle alpha is measured along each local radial ray;
# the null guard and 3-dB crossing have different roles in sidelobe scoring.

def _build_theta_s_list(scan: ScanSpecMainlobe) -> np.ndarray:
    ths = np.asarray(scan.theta_s_deg_list, dtype=np.float64).reshape(-1)
    if ths.size == 0:
        raise ValueError("theta_s_deg_list is empty")
    if np.any(~np.isfinite(ths)):
        raise ValueError("theta_s_deg_list has NaN/Inf")
    ths = np.unique(ths)
    ths.sort()
    return ths.astype(np.float64)


def _combine(vals: np.ndarray, rule: str) -> float:
    x = np.asarray(vals, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if rule == "median":
        return float(np.median(x))
    if rule == "q75":
        return float(np.quantile(x, 0.75))
    if rule == "max":
        return float(np.max(x))
    raise ValueError("combine_rule must be 'median', 'q75', or 'max'.")


def _center_peak_mag(mag: np.ndarray, alpha_rad: np.ndarray, cfg: MainlobeDomainConfig) -> float:
    if cfg.peak_mode == "center":
        return float(mag[0])
    if cfg.peak_mode == "local_max_near_center":
        amax = math.radians(float(cfg.near_center_max_deg))
        mask = alpha_rad <= amax + 1e-15
        if not np.any(mask):
            return float(mag[0])
        return float(np.max(mag[mask]))
    raise ValueError("peak_mode must be 'center' or 'local_max_near_center'.")


def _first_valley_alpha(alpha_rad: np.ndarray, mag: np.ndarray, mag0: float, valley_rel_max: float) -> Tuple[bool, float]:
    alpha = np.asarray(alpha_rad, dtype=np.float64).reshape(-1)
    mag = np.asarray(mag, dtype=np.float64).reshape(-1)
    if alpha.size != mag.size or alpha.size < 5:
        return False, float(alpha[-1]) if alpha.size else 0.0
    thr = float(valley_rel_max) * float(mag0)
    for i in range(2, alpha.size - 2):
        if mag[i] <= mag[i - 1] and mag[i] <= mag[i + 1] and mag[i] <= thr:
            return True, float(alpha[i])
    return False, float(alpha[-1])


def _first_crossing_alpha(alpha_rad: np.ndarray, mag: np.ndarray, mag0: float, db_drop: float) -> Tuple[bool, float]:
    alpha = np.asarray(alpha_rad, dtype=np.float64).reshape(-1)
    mag = np.asarray(mag, dtype=np.float64).reshape(-1)
    if alpha.size != mag.size or alpha.size < 2:
        return False, float(alpha[-1]) if alpha.size else 0.0
    thr = float(mag0) * (10.0 ** (-float(db_drop) / 20.0))
    idx = np.where(mag <= thr)[0]
    if idx.size == 0:
        return False, float(alpha[-1])
    i = int(idx[0])
    if i == 0:
        return True, float(alpha[0])
    a0 = float(alpha[i - 1])
    a1 = float(alpha[i])
    m0 = float(mag[i - 1])
    m1 = float(mag[i])
    if abs(m1 - m0) < 1e-15:
        return True, a1
    t = (thr - m0) / (m1 - m0)
    t = max(0.0, min(1.0, t))
    return True, float(a0 + t * (a1 - a0))


def _eval_mag_on_ray(
    pattern_eval: Callable[[np.ndarray, np.ndarray, float, float], np.ndarray],
    theta_s_rad: float,
    phi_s_rad: float,
    psi_rad: float,
    alpha_rad: np.ndarray,
) -> np.ndarray:
    theta_rad, phi_rad = local_spherical_points(theta_s_rad, phi_s_rad, alpha_rad, psi_rad)
    e = np.asarray(pattern_eval(theta_rad, phi_rad, theta_s_rad, phi_s_rad))
    return np.abs(e).astype(np.float64)


def estimate_mainlobe_domains(
    pattern_eval: Callable[[np.ndarray, np.ndarray, float, float], np.ndarray],
    scan: ScanSpecMainlobe,
    cfg: MainlobeDomainConfig = MainlobeDomainConfig(),
    psi_dirs_rad: Optional[np.ndarray] = None,
) -> MainlobeDomainResult:
    theta_s_deg = _build_theta_s_list(scan)
    phi_s_deg = float(scan.phi_s_deg)
    if abs(phi_s_deg) > 1e-12:
        raise ValueError("First version fixes phi_s_deg = 0.0")
    phi_s_rad = 0.0

    if psi_dirs_rad is None:
        n = int(cfg.n_dir)
        if n <= 0:
            raise ValueError("n_dir must be >= 1")
        psi_dirs = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False, dtype=np.float64)
    else:
        psi_dirs = wrap_phi_0_2pi(np.asarray(psi_dirs_rad, dtype=np.float64).reshape(-1))
        if psi_dirs.size == 0:
            raise ValueError("psi_dirs_rad is empty")

    alpha_max_rad = math.radians(float(cfg.alpha_eval_max_deg))
    if alpha_max_rad <= 0.0:
        raise ValueError("alpha_eval_max_deg must be > 0")

    S = theta_s_deg.size
    D = psi_dirs.size
    alpha_null_dirs = np.zeros((S, D), dtype=np.float64)
    alpha_guard_null_dirs = np.zeros((S, D), dtype=np.float64)
    alpha_3db_dirs = np.zeros((S, D), dtype=np.float64)
    null_found = np.zeros((S, D), dtype=bool)
    db_found = np.zeros((S, D), dtype=bool)

    alpha_coarse = np.linspace(0.0, alpha_max_rad, int(cfg.n_alpha_coarse), dtype=np.float64)

    for i, ths_deg in enumerate(theta_s_deg.tolist()):
        ths_rad = math.radians(float(ths_deg))
        for j, psi in enumerate(psi_dirs.tolist()):
            mag_c = _eval_mag_on_ray(pattern_eval, ths_rad, phi_s_rad, float(psi), alpha_coarse)
            mag0 = _center_peak_mag(mag_c, alpha_coarse, cfg)

            ok_null, a_null = _first_valley_alpha(alpha_coarse, mag_c, mag0, float(cfg.valley_rel_max))
            ok_3db, a_3db = _first_crossing_alpha(alpha_coarse, mag_c, mag0, float(cfg.db_drop))

            # refine null
            if ok_null:
                k = int(np.argmin(np.abs(alpha_coarse - a_null)))
                k0 = max(0, k - 1)
                k1 = min(alpha_coarse.size - 1, k + 1)
                a0 = float(alpha_coarse[k0])
                a1 = float(alpha_coarse[k1])
                if a1 > a0 + 1e-15:
                    alpha_ref = np.linspace(a0, a1, int(cfg.n_alpha_refine), dtype=np.float64)
                    mag_ref = _eval_mag_on_ray(pattern_eval, ths_rad, phi_s_rad, float(psi), alpha_ref)
                    mag0_ref = _center_peak_mag(mag_ref, alpha_ref, cfg)
                    ok_null2, a_null2 = _first_valley_alpha(alpha_ref, mag_ref, mag0_ref, float(cfg.valley_rel_max))
                    if ok_null2:
                        a_null = a_null2

            # refine 3db
            if ok_3db:
                k = int(np.argmin(np.abs(alpha_coarse - a_3db)))
                k0 = max(0, k - 1)
                k1 = min(alpha_coarse.size - 1, k + 1)
                a0 = float(alpha_coarse[k0])
                a1 = float(alpha_coarse[k1])
                if a1 > a0 + 1e-15:
                    alpha_ref = np.linspace(a0, a1, int(cfg.n_alpha_refine), dtype=np.float64)
                    mag_ref = _eval_mag_on_ray(pattern_eval, ths_rad, phi_s_rad, float(psi), alpha_ref)
                    mag0_ref = _center_peak_mag(mag_ref, alpha_ref, cfg)
                    ok_3db2, a_3db2 = _first_crossing_alpha(alpha_ref, mag_ref, mag0_ref, float(cfg.db_drop))
                    if ok_3db2:
                        a_3db = a_3db2

            alpha_null_dirs[i, j] = float(a_null)
            alpha_guard_null_dirs[i, j] = float(cfg.null_guard_scale) * float(a_null)
            alpha_3db_dirs[i, j] = float(a_3db)
            null_found[i, j] = bool(ok_null)
            db_found[i, j] = bool(ok_3db)

    alpha_null = np.asarray([_combine(alpha_null_dirs[i, :], cfg.combine_rule) for i in range(S)], dtype=np.float64)
    alpha_guard_null = np.asarray([_combine(alpha_guard_null_dirs[i, :], cfg.combine_rule) for i in range(S)], dtype=np.float64)
    alpha_3db = np.asarray([_combine(alpha_3db_dirs[i, :], cfg.combine_rule) for i in range(S)], dtype=np.float64)

    stats = {
        "cfg": asdict(cfg),
        "scan": {"theta_s_deg_list": theta_s_deg.tolist(), "phi_s_deg": phi_s_deg},
        "null_found_ratio": float(np.mean(null_found)),
        "db_found_ratio": float(np.mean(db_found)),
        "alpha_null_min_deg": float(rad2deg(np.min(alpha_null))),
        "alpha_null_max_deg": float(rad2deg(np.max(alpha_null))),
        "alpha_3db_min_deg": float(rad2deg(np.min(alpha_3db))),
        "alpha_3db_max_deg": float(rad2deg(np.max(alpha_3db))),
    }

    return MainlobeDomainResult(
        theta_s_deg=theta_s_deg.astype(np.float64),
        phi_s_deg=phi_s_deg,
        alpha_null_deg=rad2deg(alpha_null).astype(np.float64),
        alpha_guard_null_deg=rad2deg(alpha_guard_null).astype(np.float64),
        alpha_3db_deg=rad2deg(alpha_3db).astype(np.float64),
        alpha_null_dirs_deg=rad2deg(alpha_null_dirs).astype(np.float64),
        alpha_guard_null_dirs_deg=rad2deg(alpha_guard_null_dirs).astype(np.float64),
        alpha_3db_dirs_deg=rad2deg(alpha_3db_dirs).astype(np.float64),
        psi_dirs_rad=psi_dirs.astype(np.float64),
        stats=stats,
    )


def estimate_mainlobe_domains_from_directsum(
    eco_eval: Callable[[np.ndarray, np.ndarray], np.ndarray],
    geometry: ArrayGeometry,
    scan: ScanSpecMainlobe,
    cfg: MainlobeDomainConfig = MainlobeDomainConfig(),
    psi_dirs_rad: Optional[np.ndarray] = None,
) -> MainlobeDomainResult:
    pattern_eval = build_directsum_pattern_evaluator(
        eco_eval,
        geometry,
        block_size=int(cfg.directsum_block_size),
    )
    return estimate_mainlobe_domains(pattern_eval, scan, cfg=cfg, psi_dirs_rad=psi_dirs_rad)


# =============================================================================
# Offline alpha-main table helpers
# =============================================================================
# Preserve the existing a0cra_alpha_main_table_v1 payload fields: downstream
# backbone synthesis reads them as a scientific data interface.

def _validate_by_b_theta_matrix(name: str, x: np.ndarray, b_size: int, theta_size: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    expected = (int(b_size), int(theta_size))
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected}; got {arr.shape}")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr.astype(np.float64)


def build_alpha_main_table_entry(
    *,
    key: Dict[str, Any],
    theta_s_deg: np.ndarray,
    b_values: np.ndarray,
    raw_alpha_null_deg_by_b_theta: np.ndarray,
    raw_alpha_guard_null_deg_by_b_theta: np.ndarray,
    raw_alpha_3db_deg_by_b_theta: np.ndarray,
    r_over_lambda_by_b: Optional[np.ndarray] = None,
    N_per_ring_by_b: Optional[np.ndarray] = None,
    raw_stats_by_b: Optional[Sequence[Dict[str, Any]]] = None,
    policy: AlphaMainTablePolicy = AlphaMainTablePolicy(),
) -> Dict[str, Any]:
    """Build one runtime alpha-main/3dB table entry from raw per-b diagnostics.

    Runtime consumers should use the top-level ``theta_s_deg``, ``alpha_main_deg``,
    and ``alpha_3db_deg`` fields.  The raw null / guard-null / 3 dB arrays are
    retained under ``debug`` for audit.
    """
    theta = np.asarray(theta_s_deg, dtype=np.float64).reshape(-1)
    b = np.asarray(b_values, dtype=np.float64).reshape(-1)

    if theta.size == 0:
        raise ValueError("theta_s_deg is empty")
    if b.size == 0:
        raise ValueError("b_values is empty")
    if np.any(~np.isfinite(theta)) or np.any(~np.isfinite(b)):
        raise ValueError("theta_s_deg and b_values must be finite")

    raw_null = _validate_by_b_theta_matrix(
        "raw_alpha_null_deg_by_b_theta",
        raw_alpha_null_deg_by_b_theta,
        b.size,
        theta.size,
    )
    raw_guard = _validate_by_b_theta_matrix(
        "raw_alpha_guard_null_deg_by_b_theta",
        raw_alpha_guard_null_deg_by_b_theta,
        b.size,
        theta.size,
    )
    raw_3db = _validate_by_b_theta_matrix(
        "raw_alpha_3db_deg_by_b_theta",
        raw_alpha_3db_deg_by_b_theta,
        b.size,
        theta.size,
    )

    if str(policy.reduction_rule).lower().strip() != "min_over_b":
        raise ValueError("First version only supports reduction_rule='min_over_b'")
    if str(policy.alpha_3db_reduction_rule).lower().strip() != "min_over_b":
        raise ValueError("First version only supports alpha_3db_reduction_rule='min_over_b'")

    margin = float(policy.mainlobe_margin_deg)
    if margin < 0.0:
        raise ValueError("mainlobe_margin_deg must be >= 0")

    null_min = np.min(raw_null, axis=0)
    guard_min = np.min(raw_guard, axis=0)
    db3_min = np.min(raw_3db, axis=0)

    null_used = null_min.copy()
    guard_used = guard_min.copy()

    if bool(policy.apply_margin_to_debug_null):
        null_used = np.maximum(0.0, null_used - margin)
    if bool(policy.apply_margin_to_debug_guard_null):
        guard_used = np.maximum(0.0, guard_used - margin)

    src = str(policy.alpha_main_source).lower().strip()
    if src == "guard_null":
        alpha_main = guard_min.copy()
    elif src == "null":
        alpha_main = null_min.copy()
    elif src == "3db":
        alpha_main = db3_min.copy()
    else:
        raise ValueError("alpha_main_source must be 'guard_null', 'null', or '3db'")

    if bool(policy.apply_margin_to_alpha_main):
        alpha_main = np.maximum(0.0, alpha_main - margin)

    alpha_3db = db3_min.copy()
    if bool(policy.apply_margin_to_alpha_3db):
        alpha_3db = np.maximum(0.0, alpha_3db - margin)

    worst_b_null = b[np.argmin(raw_null, axis=0)]
    worst_b_guard = b[np.argmin(raw_guard, axis=0)]
    worst_b_3db = b[np.argmin(raw_3db, axis=0)]

    debug_raw: Dict[str, Any] = {
        "alpha_null_deg_by_b_theta": raw_null.astype(float).tolist(),
        "alpha_guard_null_deg_by_b_theta": raw_guard.astype(float).tolist(),
        "alpha_3db_deg_by_b_theta": raw_3db.astype(float).tolist(),
    }
    if r_over_lambda_by_b is not None:
        debug_raw["r_over_lambda_by_b"] = np.asarray(r_over_lambda_by_b, dtype=np.float64).tolist()
    if N_per_ring_by_b is not None:
        debug_raw["N_per_ring_by_b"] = np.asarray(N_per_ring_by_b, dtype=np.int64).tolist()
    if raw_stats_by_b is not None:
        debug_raw["stats_by_b"] = list(raw_stats_by_b)

    return {
        "key": dict(key),
        "theta_s_deg": theta.astype(float).tolist(),
        "alpha_main_deg": alpha_main.astype(float).tolist(),
        "alpha_3db_deg": alpha_3db.astype(float).tolist(),
        "debug": {
            "policy": asdict(policy),
            "b_values": b.astype(float).tolist(),
            "alpha_null_used_deg": null_used.astype(float).tolist(),
            "alpha_guard_null_used_deg": guard_used.astype(float).tolist(),
            "alpha_3db_used_deg": alpha_3db.astype(float).tolist(),
            "worst_b_null_by_theta": worst_b_null.astype(float).tolist(),
            "worst_b_guard_by_theta": worst_b_guard.astype(float).tolist(),
            "worst_b_3db_by_theta": worst_b_3db.astype(float).tolist(),
            "raw": debug_raw,
        },
    }


def _validate_alpha_table_entry(entry: Dict[str, Any]) -> None:
    for name in ("key", "theta_s_deg", "alpha_main_deg", "alpha_3db_deg"):
        if name not in entry:
            raise KeyError(f"alpha table entry missing {name}")
    theta = np.asarray(entry["theta_s_deg"], dtype=np.float64).reshape(-1)
    alpha_main = np.asarray(entry["alpha_main_deg"], dtype=np.float64).reshape(-1)
    alpha_3db = np.asarray(entry["alpha_3db_deg"], dtype=np.float64).reshape(-1)
    if not (theta.size == alpha_main.size == alpha_3db.size):
        raise ValueError("entry theta_s_deg / alpha_main_deg / alpha_3db_deg length mismatch")
    if np.any(~np.isfinite(theta)) or np.any(~np.isfinite(alpha_main)) or np.any(~np.isfinite(alpha_3db)):
        raise ValueError("entry theta_s_deg / alpha_main_deg / alpha_3db_deg contains NaN/Inf")

def build_alpha_main_table(
    *,
    entries: Sequence[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entries_list = list(entries)
    if len(entries_list) == 0:
        raise ValueError("entries must be non-empty")
    for entry in entries_list:
        _validate_alpha_table_entry(entry)
    return {
        "format": "a0cra_alpha_main_table_v1",
        "config": dict(config or {}),
        "entries": entries_list,
    }


def find_alpha_main_table_entry(
    table: Dict[str, Any],
    *,
    K: int,
    d_min: float,
    d_eff: float,
    R0: float,
    include_center: Optional[bool] = None,
    outer5_reduce_mult4: Optional[bool] = None,
    outer5_threshold_gt: Optional[int] = None,
    tol: float = 1e-9,
) -> Dict[str, Any]:
    if str(table.get("format", "")) != "a0cra_alpha_main_table_v1":
        raise ValueError("Unsupported or missing alpha-main table format")

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

        if include_center is not None and bool(key.get("include_center")) != bool(include_center):
            continue
        if outer5_reduce_mult4 is not None and bool(key.get("outer5_reduce_mult4")) != bool(outer5_reduce_mult4):
            continue
        if outer5_threshold_gt is not None and int(key.get("outer5_threshold_gt")) != int(outer5_threshold_gt):
            continue

        return entry

    raise KeyError(
        f"No alpha-main table entry for K={K}, d_min={d_min}, d_eff={d_eff}, R0={R0}"
    )


def slice_alpha_table_entry_by_theta(
    entry: Dict[str, Any],
    theta_s_deg_required: np.ndarray,
    *,
    tol: float = 1e-9,
) -> Dict[str, np.ndarray]:
    theta_all = np.asarray(entry["theta_s_deg"], dtype=np.float64).reshape(-1)
    alpha_main_all = np.asarray(entry["alpha_main_deg"], dtype=np.float64).reshape(-1)
    alpha_3db_all = np.asarray(entry["alpha_3db_deg"], dtype=np.float64).reshape(-1)

    if not (theta_all.size == alpha_main_all.size == alpha_3db_all.size):
        raise ValueError("entry theta_s_deg / alpha_main_deg / alpha_3db_deg length mismatch")

    req = np.asarray(theta_s_deg_required, dtype=np.float64).reshape(-1)
    if req.size == 0:
        raise ValueError("theta_s_deg_required is empty")

    idx = []
    missing = []
    for th in req.tolist():
        hit = np.where(np.isclose(theta_all, float(th), atol=float(tol), rtol=0.0))[0]
        if hit.size == 0:
            missing.append(float(th))
        else:
            idx.append(int(hit[0]))

    if missing:
        raise KeyError(f"alpha table entry missing theta_s_deg={missing}")

    idx_arr = np.asarray(idx, dtype=np.int64)
    return {
        "theta_s_deg": theta_all[idx_arr].astype(np.float64),
        "alpha_main_deg": alpha_main_all[idx_arr].astype(np.float64),
        "alpha_3db_deg": alpha_3db_all[idx_arr].astype(np.float64),
    }


def slice_alpha_main_entry_by_theta(
    entry: Dict[str, Any],
    theta_s_deg_required: np.ndarray,
    *,
    tol: float = 1e-9,
) -> Dict[str, np.ndarray]:
    sliced = slice_alpha_table_entry_by_theta(entry, theta_s_deg_required, tol=tol)
    return {
        "theta_s_deg": sliced["theta_s_deg"],
        "alpha_main_deg": sliced["alpha_main_deg"],
    }

def save_alpha_main_table_json(table: Dict[str, Any], path: str) -> None:
    if str(table.get("format", "")) != "a0cra_alpha_main_table_v1":
        raise ValueError("Unsupported or missing alpha-main table format")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)


def load_alpha_main_table_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)
    if str(table.get("format", "")) != "a0cra_alpha_main_table_v1":
        raise ValueError("Unsupported or missing alpha-main table format")
    return table


def save_alpha_main_table_npz(table: Dict[str, Any], path: str) -> None:
    if str(table.get("format", "")) != "a0cra_alpha_main_table_v1":
        raise ValueError("Unsupported or missing alpha-main table format")

    entries = list(table.get("entries", []))
    if len(entries) == 0:
        raise ValueError("table has no entries")

    for entry in entries:
        _validate_alpha_table_entry(entry)

    theta0 = np.asarray(entries[0]["theta_s_deg"], dtype=np.float64)

    K_list = []
    d_min_list = []
    d_eff_list = []
    R0_list = []
    alpha_main_list = []
    alpha_3db_list = []

    for entry in entries:
        key = dict(entry["key"])
        theta = np.asarray(entry["theta_s_deg"], dtype=np.float64)
        if theta.shape != theta0.shape or not np.allclose(theta, theta0, atol=1e-9, rtol=0.0):
            raise ValueError("All entries must share the same theta_s_deg grid for NPZ export")

        K_list.append(int(key["K"]))
        d_min_list.append(float(key["d_min"]))
        d_eff_list.append(float(key["d_eff"]))
        R0_list.append(float(key["R0"]))
        alpha_main_list.append(np.asarray(entry["alpha_main_deg"], dtype=np.float64))
        alpha_3db_list.append(np.asarray(entry["alpha_3db_deg"], dtype=np.float64))

    np.savez(
        path,
        K=np.asarray(K_list, dtype=np.int64),
        d_min=np.asarray(d_min_list, dtype=np.float64),
        d_eff=np.asarray(d_eff_list, dtype=np.float64),
        R0=np.asarray(R0_list, dtype=np.float64),
        theta_s_deg=theta0.astype(np.float64),
        alpha_main_deg=np.asarray(alpha_main_list, dtype=np.float64),
        alpha_3db_deg=np.asarray(alpha_3db_list, dtype=np.float64),
    )


# =============================================================================
# Offline alpha-main table sweep CLI
# =============================================================================

def _arange_inclusive(start: float, stop: float, step: float, *, ndigits: int = 10) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("step must be > 0")
    n = int(math.floor((float(stop) - float(start)) / float(step) + 0.5)) + 1
    arr = float(start) + float(step) * np.arange(n, dtype=np.float64)
    arr = arr[arr <= float(stop) + 1e-9]
    return np.asarray([round(float(x), ndigits) for x in arr], dtype=np.float64)


def _parse_int_list(s: str) -> Tuple[int, ...]:
    vals = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty int list")
    return tuple(vals)


def _parse_float_range_or_list(s: str) -> Tuple[float, ...]:
    text = str(s).strip()
    if ":" in text:
        parts = [float(x.strip()) for x in text.split(":")]
        if len(parts) != 3:
            raise ValueError("range format must be start:stop:step")
        return tuple(_arange_inclusive(parts[0], parts[1], parts[2], ndigits=6).tolist())
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty float list")
    return tuple(vals)



def _resolve_cli_input_paths(args: argparse.Namespace) -> argparse.Namespace:
    unit_arg = str(getattr(args, "unit_cell_npz", "") or "").strip()
    if not unit_arg:
        raise FileNotFoundError("--unit-cell-npz must be provided explicitly")
    unit_path = Path(unit_arg)
    if not unit_path.is_absolute():
        unit_path = Path.cwd() / unit_path
    if not unit_path.exists() or not unit_path.is_file():
        raise FileNotFoundError("unit-cell NPZ not found: %s" % unit_path)

    meta_arg = str(getattr(args, "unit_cell_meta_json", "") or "").strip()
    if not meta_arg:
        raise FileNotFoundError("--unit-cell-meta-json must be provided explicitly")
    meta_path = Path(meta_arg)
    if not meta_path.is_absolute():
        meta_path = Path.cwd() / meta_path
    if not meta_path.exists() or not meta_path.is_file():
        raise FileNotFoundError("unit-cell meta JSON not found: %s" % meta_path)

    out_dir_arg = str(getattr(args, "out_dir", "") or "").strip()
    if not out_dir_arg or out_dir_arg == ".":
        # Keep generated tables out of the installed source-code directory.
        resolved_out = str(Path.cwd())
    else:
        out_path = Path(out_dir_arg)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        resolved_out = str(out_path)

    args.unit_cell_npz = str(unit_path)
    args.unit_cell_meta_json = str(meta_path)
    args.out_dir = resolved_out
    return args


def _load_iep_model_for_cli(unit_cell_npz_path: str, unit_cell_meta_json_path: str):
    from element_spectrum import iep_model

    cfg = iep_model.IEPModelConfig(
        phi_main_rad=0.0,
        bounds="error",
        enable_bilinear_phi=True,
    )
    model = iep_model.IEPModel(
        str(unit_cell_npz_path),
        meta_json_path=str(unit_cell_meta_json_path),
        config=cfg,
    )
    if model.schema_name != "hfss_sr_sparse" or int(model.schema_version) != 2:
        raise RuntimeError("scan_mainlobe_domain requires hfss_sr_sparse schema v2")
    if model.spectral_frame != "sr_compatible_sparse_v2":
        raise RuntimeError("scan_mainlobe_domain requires sr_compatible_sparse_v2")
    if model.co_handedness != "LHCP":
        raise RuntimeError("current Stage1 configuration requires co_handedness='LHCP'")
    ready = model.validate_stage1_ready(require_sr_compatible=True)
    if not bool(ready.get("ok", False)):
        raise RuntimeError("IEP model is not Stage1-ready: %s" % ready.get("issues", []))
    return model


def _load_iep_eco_eval_for_cli(unit_cell_npz_path: str, unit_cell_meta_json_path: str = "") -> Callable:
    model = _load_iep_model_for_cli(
        unit_cell_npz_path,
        unit_cell_meta_json_path,
    )
    return model.query_E_co


def _make_mainlobe_solver_config_from_sweep(cfg: AlphaMainTableSweepConfig) -> MainlobeDomainConfig:
    return MainlobeDomainConfig(
        alpha_eval_max_deg=float(cfg.alpha_eval_max_deg),
        n_dir=int(cfg.n_dir),
        n_alpha_coarse=int(cfg.n_alpha_coarse),
        n_alpha_refine=int(cfg.n_alpha_refine),
        valley_rel_max=float(cfg.valley_rel_max),
        null_guard_scale=float(cfg.null_guard_scale),
        db_drop=float(cfg.db_drop),
        combine_rule=str(cfg.combine_rule),
        peak_mode=str(cfg.peak_mode),
        near_center_max_deg=float(cfg.near_center_max_deg),
        directsum_block_size=int(cfg.directsum_block_size),
    )


def _make_alpha_main_policy_from_sweep(cfg: AlphaMainTableSweepConfig) -> AlphaMainTablePolicy:
    return AlphaMainTablePolicy(
        alpha_main_source=str(cfg.alpha_main_source),
        reduction_rule="min_over_b",
        mainlobe_margin_deg=float(cfg.mainlobe_margin_deg),
        apply_margin_to_alpha_main=True,
        alpha_3db_reduction_rule="min_over_b",
        apply_margin_to_alpha_3db=False,
        apply_margin_to_debug_null=True,
        apply_margin_to_debug_guard_null=True,
    )


def calibrate_alpha_main_entry_for_structure(
    *,
    eco_eval: Callable[[np.ndarray, np.ndarray], np.ndarray],
    K: int,
    d_min: float,
    cfg: AlphaMainTableSweepConfig,
    theta_s_deg: np.ndarray,
    b_values: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    d_eff = float(d_min) if bool(cfg.bind_d_eff_to_d_min) else float(cfg.d_eff_default)

    solver_cfg = _make_mainlobe_solver_config_from_sweep(cfg)
    policy = _make_alpha_main_policy_from_sweep(cfg)
    scan = ScanSpecMainlobe(
        theta_s_deg_list=np.asarray(theta_s_deg, dtype=np.float64),
        phi_s_deg=0.0,
    )

    raw_null_list = []
    raw_guard_list = []
    raw_3db_list = []
    r_list = []
    N_list = []
    stats_list = []

    b_arr = np.asarray(b_values, dtype=np.float64).reshape(-1)
    for ib, b in enumerate(b_arr.tolist()):
        r = init_r_gompertz(
            K=int(K),
            R0=float(cfg.R0),
            d_min=float(d_min),
            b=float(b),
        )
        N = derive_ring_counts(
            r,
            d_eff=float(d_eff),
            outer5_reduce_mult4=bool(cfg.outer5_reduce_mult4),
            outer5_threshold_gt=int(cfg.outer5_threshold_gt),
        )
        geom = ArrayGeometry(
            r_over_lambda=r,
            N_per_ring=N,
            include_center=bool(cfg.include_center),
        )

        if verbose:
            N_total = int(np.sum(N) + (1 if cfg.include_center else 0))
            print(
                f"[alpha-main] K={K} d_min={d_min:.4f} d_eff={d_eff:.4f} "
                f"b={b:.3f} ({ib + 1}/{b_arr.size}) N_total={N_total}"
            )

        res = estimate_mainlobe_domains_from_directsum(
            eco_eval=eco_eval,
            geometry=geom,
            scan=scan,
            cfg=solver_cfg,
        )

        raw_null_list.append(np.asarray(res.alpha_null_deg, dtype=np.float64))
        raw_guard_list.append(np.asarray(res.alpha_guard_null_deg, dtype=np.float64))
        raw_3db_list.append(np.asarray(res.alpha_3db_deg, dtype=np.float64))
        r_list.append(np.asarray(r, dtype=np.float64))
        N_list.append(np.asarray(N, dtype=np.int64))
        stats_list.append(dict(res.stats))

    key = {
        "K": int(K),
        "d_min": float(d_min),
        "d_eff": float(d_eff),
        "R0": float(cfg.R0),
        "include_center": bool(cfg.include_center),
        "outer5_reduce_mult4": bool(cfg.outer5_reduce_mult4),
        "outer5_threshold_gt": int(cfg.outer5_threshold_gt),
    }

    return build_alpha_main_table_entry(
        key=key,
        theta_s_deg=np.asarray(theta_s_deg, dtype=np.float64),
        b_values=b_arr,
        raw_alpha_null_deg_by_b_theta=np.vstack(raw_null_list),
        raw_alpha_guard_null_deg_by_b_theta=np.vstack(raw_guard_list),
        raw_alpha_3db_deg_by_b_theta=np.vstack(raw_3db_list),
        r_over_lambda_by_b=np.vstack(r_list),
        N_per_ring_by_b=np.vstack(N_list),
        raw_stats_by_b=stats_list,
        policy=policy,
    )


def calibrate_alpha_main_table(
    cfg: AlphaMainTableSweepConfig,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    model = _load_iep_model_for_cli(
        cfg.unit_cell_npz_path,
        cfg.unit_cell_meta_json_path,
    )
    eco_eval = model.query_E_co

    theta_s = _arange_inclusive(
        float(cfg.theta_s_min_deg),
        float(cfg.theta_s_max_deg),
        float(cfg.theta_s_step_deg),
        ndigits=6,
    )
    b_values = _arange_inclusive(
        float(cfg.b_min),
        float(cfg.b_max),
        float(cfg.b_step),
        ndigits=6,
    )

    entries: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    total = len(cfg.K_list) * len(cfg.d_min_list)
    idx = 0

    for K in cfg.K_list:
        for d_min in cfg.d_min_list:
            idx += 1
            if verbose:
                print("=" * 80)
                print(f"[alpha-main][structure {idx}/{total}] K={int(K)} d_min={float(d_min):.4f}")
            entry = calibrate_alpha_main_entry_for_structure(
                eco_eval=eco_eval,
                K=int(K),
                d_min=float(d_min),
                cfg=cfg,
                theta_s_deg=theta_s,
                b_values=b_values,
                verbose=verbose,
            )
            entries.append(entry)

    config_payload = asdict(cfg)
    config_payload.update({
        "theta_s_deg": theta_s.astype(float).tolist(),
        "b_values": b_values.astype(float).tolist(),
        "elapsed_sec": float(time.perf_counter() - t0),
        "runtime_fields": ["theta_s_deg", "alpha_main_deg", "alpha_3db_deg"],
        "iep_source": {
            "schema_name": model.schema_name,
            "schema_version": int(model.schema_version),
            "spectral_frame": model.spectral_frame,
            "co_handedness": model.co_handedness,
            "cx_handedness": model.cx_handedness,
            "spin_sign": int(model.spin_sign),
            "cp_basis_sign": int(model.cp_basis_sign),
            "npz_path": model.npz_path,
            "json_path": model.meta_json_path,
            "npz_sha256": model.npz_sha256,
            "source_csv_path": model.source_csv_path,
            "source_csv_sha256": model.source_csv_sha256,
            "normalization_A_ref": float(model.normalization_A_ref),
            "normalization_scale": float(model.normalization_scale),
            "quality_control": dict(model.meta.get("quality_control", {})),
            "fit_checks": dict(model.meta.get("fit_checks", {})),
        },
    })
    return build_alpha_main_table(
        entries=entries,
        config=config_payload,
    )


def save_alpha_main_table_bundle(
    table: Dict[str, Any],
    *,
    out_dir: str,
    basename: str = "a0cra_alpha_main_table",
) -> Tuple[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{basename}.json"
    npz_path = out / f"{basename}.npz"

    save_alpha_main_table_json(table, str(json_path))
    save_alpha_main_table_npz(table, str(npz_path))
    return str(json_path), str(npz_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline calibration of the scan mainlobe/3-dB exclusion table.")

    p.add_argument("--unit-cell-npz", default="")
    p.add_argument("--unit-cell-meta-json", default="")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--basename", default="a0cra_alpha_main_table")

    p.add_argument("--K-list", default="6,7,8")
    p.add_argument("--d-min-list", default="0.50:0.60:0.01")
    p.add_argument("--R0", type=float, default=5.0)

    g = p.add_mutually_exclusive_group()
    g.add_argument("--bind-d-eff-to-d-min", dest="bind_d_eff_to_d_min", action="store_true")
    g.add_argument("--no-bind-d-eff-to-d-min", dest="bind_d_eff_to_d_min", action="store_false")
    p.set_defaults(bind_d_eff_to_d_min=True)
    p.add_argument("--d-eff-default", type=float, default=0.5)

    p.add_argument("--b-min", type=float, default=6.0)
    p.add_argument("--b-max", type=float, default=12.0)
    p.add_argument("--b-step", type=float, default=0.25)

    p.add_argument("--theta-min", type=float, default=0.0)
    p.add_argument("--theta-max", type=float, default=60.0)
    p.add_argument("--theta-step", type=float, default=1.0)

    p.add_argument("--mainlobe-margin-deg", type=float, default=0.5)
    p.add_argument("--alpha-main-source", default="guard_null", choices=["guard_null", "null", "3db"])

    p.add_argument("--alpha-eval-max-deg", type=float, default=30.0)
    p.add_argument("--n-dir", type=int, default=12)
    p.add_argument("--n-alpha-coarse", type=int, default=1200)
    p.add_argument("--n-alpha-refine", type=int, default=400)
    p.add_argument("--valley-rel-max", type=float, default=0.60)
    p.add_argument("--null-guard-scale", type=float, default=0.98)
    p.add_argument("--db-drop", type=float, default=3.0)
    p.add_argument("--combine-rule", default="median", choices=["median", "q75", "max"])
    p.add_argument("--peak-mode", default="center", choices=["center", "local_max_near_center"])
    p.add_argument("--near-center-max-deg", type=float, default=1.0)
    p.add_argument("--directsum-block-size", type=int, default=256)

    p.add_argument("--quiet", action="store_true", default=False)
    return p


def _sweep_config_from_args(args: argparse.Namespace) -> AlphaMainTableSweepConfig:
    return AlphaMainTableSweepConfig(
        unit_cell_npz_path=str(args.unit_cell_npz),
        unit_cell_meta_json_path=str(args.unit_cell_meta_json or ""),
        out_dir=str(args.out_dir),
        K_list=_parse_int_list(args.K_list),
        d_min_list=_parse_float_range_or_list(args.d_min_list),
        R0=float(args.R0),
        bind_d_eff_to_d_min=bool(args.bind_d_eff_to_d_min),
        d_eff_default=float(args.d_eff_default),
        b_min=float(args.b_min),
        b_max=float(args.b_max),
        b_step=float(args.b_step),
        theta_s_min_deg=float(args.theta_min),
        theta_s_max_deg=float(args.theta_max),
        theta_s_step_deg=float(args.theta_step),
        mainlobe_margin_deg=float(args.mainlobe_margin_deg),
        alpha_main_source=str(args.alpha_main_source),
        alpha_eval_max_deg=float(args.alpha_eval_max_deg),
        n_dir=int(args.n_dir),
        n_alpha_coarse=int(args.n_alpha_coarse),
        n_alpha_refine=int(args.n_alpha_refine),
        valley_rel_max=float(args.valley_rel_max),
        null_guard_scale=float(args.null_guard_scale),
        db_drop=float(args.db_drop),
        combine_rule=str(args.combine_rule),
        peak_mode=str(args.peak_mode),
        near_center_max_deg=float(args.near_center_max_deg),
        directsum_block_size=int(args.directsum_block_size),
    )


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        args = _resolve_cli_input_paths(args)
    except Exception as exc:
        parser.error(str(exc))
    cfg = _sweep_config_from_args(args)
    verbose = not bool(args.quiet)

    if verbose:
        print("=== A0CRA ALPHA-MAIN/3DB TABLE CALIBRATION: START ===")
        print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    table = calibrate_alpha_main_table(cfg, verbose=verbose)
    json_path, npz_path = save_alpha_main_table_bundle(
        table,
        out_dir=str(cfg.out_dir),
        basename=str(args.basename),
    )

    if verbose:
        print("=== A0CRA ALPHA-MAIN/3DB TABLE CALIBRATION: DONE ===")
        print(f"[out] json={json_path}")
        print(f"[out] npz ={npz_path}")

    return 0


# =============================================================================
# Debug export helper
# =============================================================================

def save_mainlobe_domain_result_json(res: MainlobeDomainResult, path: str) -> None:
    payload = {
        "theta_s_deg": res.theta_s_deg.tolist(),
        "phi_s_deg": float(res.phi_s_deg),
        "alpha_null_deg": res.alpha_null_deg.tolist(),
        "alpha_guard_null_deg": res.alpha_guard_null_deg.tolist(),
        "alpha_3db_deg": res.alpha_3db_deg.tolist(),
        "alpha_null_dirs_deg": res.alpha_null_dirs_deg.tolist(),
        "alpha_guard_null_dirs_deg": res.alpha_guard_null_dirs_deg.tolist(),
        "alpha_3db_dirs_deg": res.alpha_3db_dirs_deg.tolist(),
        "psi_dirs_rad": res.psi_dirs_rad.tolist(),
        "stats": res.stats,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


__all__ = [
    "ArrayGeometry",
    "ScanSpecMainlobe",
    "MainlobeDomainConfig",
    "MainlobeDomainResult",
    "AlphaMainTablePolicy",
    "AlphaMainTableSweepConfig",
    "init_r_gompertz",
    "derive_ring_counts",
    "build_element_angles",
    "build_array_geometry_from_init_b6",
    "build_directsum_pattern_evaluator",
    "estimate_mainlobe_domains",
    "estimate_mainlobe_domains_from_directsum",
    "build_alpha_main_table_entry",
    "build_alpha_main_table",
    "find_alpha_main_table_entry",
    "slice_alpha_table_entry_by_theta",
    "slice_alpha_main_entry_by_theta",
    "save_alpha_main_table_json",
    "load_alpha_main_table_json",
    "save_alpha_main_table_npz",
    "calibrate_alpha_main_entry_for_structure",
    "calibrate_alpha_main_table",
    "save_alpha_main_table_bundle",
    "cli_main",
    "save_mainlobe_domain_result_json",
]


if __name__ == "__main__":
    raise SystemExit(cli_main())
