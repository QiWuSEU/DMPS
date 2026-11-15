#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the fixed scan-centered u-v domain used by Stage3 optimization.

This module consumes the canonical point ordering created by
``vecaep_io.py``. It never interpolates the fields, never changes the
row order of the Vec-AEP matrices, and never modifies the polarization basis.

The formal V1 domain is a fixed circular exclusion region in the global u-v
plane around the scan direction:

    rho_uv = sqrt((u-u_s)^2 + (v-v_s)^2)
    mainlobe: rho_uv <= rho_main
    sidelobe: rho_uv >  rho_main

For the current 50 deg / 0 deg scan, rho_main defaults to 0.13.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from vecaep_residual_closure.vecaep_io import (
        decompose_circular_components,
        read_hfss_vector_field_csv,
    )
except ImportError as exc:  # pragma: no cover - explicit user-facing failure
    raise ImportError(
        "vecaep_grid_domain.py requires the installed "
        "vecaep_residual_closure.vecaep_io module"
    ) from exc


SCRIPT_VERSION = "1.0.0"
SCHEMA_NAME = "stage3_grid_domain"
SCHEMA_VERSION = 1
ARRAYS_NAME = "grid_domain_arrays.npz"
POINTS_NAME = "grid_domain_points.csv"
SUMMARY_NAME = "grid_domain_summary.json"
COMPLETE_NAME = "domain_complete.json"
INCOMPLETE_NAME = ".domain_build_incomplete"
LOG_NAME = "stage3_grid_domain.log"

DEFAULT_VECAEP_CACHE = Path(
    r"F:\ningzhiwen\rebuild\stage3_vecaep_synthesis\cache\vecaep_20260725_123606"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"F:\ningzhiwen\rebuild\stage3_vecaep_synthesis\cache"
)


@dataclass(frozen=True)
class GridDomain:
    """Fixed point-domain definition in the exact Vec-AEP row order."""

    scan_theta_deg: float
    scan_phi_deg: float
    scan_u: float
    scan_v: float
    scan_w: float

    theta_rad: np.ndarray
    phi_rad: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    delta_u: np.ndarray
    delta_v: np.ndarray
    rho_uv: np.ndarray
    psi_uv: np.ndarray
    angular_distance_rad: np.ndarray

    target_index: int
    main_mask: np.ndarray
    side_mask: np.ndarray
    main_indices: np.ndarray
    side_indices: np.ndarray

    mainlobe_radius_uv: float
    boundary_tolerance: float
    target_tolerance_deg: float

    @property
    def point_count(self) -> int:
        return int(self.theta_rad.size)

    @property
    def main_count(self) -> int:
        return int(self.main_indices.size)

    @property
    def side_count(self) -> int:
        return int(self.side_indices.size)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
        f.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode("ascii"))
        h.update(str(a.dtype).encode("ascii"))
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def setup_logger(log_path: Path | None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("stage3_grid_domain")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(console)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        )
        logger.addHandler(file_handler)
    return logger


def normalize_phi_deg(phi_deg: float) -> float:
    value = float(phi_deg) % 360.0
    if math.isclose(value, 360.0, rel_tol=0.0, abs_tol=1e-12):
        value = 0.0
    return value


def _format_integer_like(value: float, width: int = 3) -> str:
    rounded = round(float(value))
    if math.isclose(float(value), rounded, rel_tol=0.0, abs_tol=1e-9):
        return f"{int(rounded):0{width}d}"
    token = f"{float(value):.{max(1, width)}f}".rstrip("0").rstrip(".")
    return token.replace("-", "m").replace(".", "p")


def _format_radius(value: float) -> str:
    return f"{float(value):.3f}".replace("-", "m").replace(".", "p")


def make_domain_name(scan_theta_deg: float, scan_phi_deg: float, radius_uv: float) -> str:
    return (
        f"grid_domain_theta{_format_integer_like(scan_theta_deg)}_"
        f"phi{_format_integer_like(normalize_phi_deg(scan_phi_deg))}_"
        f"rho{_format_radius(radius_uv)}"
    )


# -----------------------------------------------------------------------------
# Cache-grid loading and geometry
# -----------------------------------------------------------------------------
def load_grid_from_vecaep_cache(
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and validate the canonical point grid from the Vec-AEP cache."""

    cache_dir = Path(cache_dir).expanduser().resolve()
    complete_path = cache_dir / "cache_complete.json"
    metadata_path = cache_dir / "cache_metadata.json"
    theta_points_path = cache_dir / "theta_points_rad.npy"
    phi_points_path = cache_dir / "phi_points_rad.npy"
    theta_axis_path = cache_dir / "theta_axis_rad.npy"
    phi_axis_path = cache_dir / "phi_axis_rad.npy"

    required = [
        complete_path,
        metadata_path,
        theta_points_path,
        phi_points_path,
        theta_axis_path,
        phi_axis_path,
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Vec-AEP cache is incomplete; missing={missing}")

    complete = read_json(complete_path)
    if complete.get("status") != "complete":
        raise ValueError(f"Vec-AEP cache is not complete: {complete_path}")
    metadata = read_json(metadata_path)

    theta_points = np.load(theta_points_path, mmap_mode="r", allow_pickle=False)
    phi_points = np.load(phi_points_path, mmap_mode="r", allow_pickle=False)
    theta_axis = np.load(theta_axis_path, mmap_mode="r", allow_pickle=False)
    phi_axis = np.load(phi_axis_path, mmap_mode="r", allow_pickle=False)

    if theta_points.ndim != 1 or phi_points.ndim != 1:
        raise ValueError("theta_points_rad.npy and phi_points_rad.npy must be 1-D")
    if theta_points.shape != phi_points.shape:
        raise ValueError(
            f"theta/phi point shape mismatch: {theta_points.shape} vs {phi_points.shape}"
        )
    if theta_points.dtype != np.float64 or phi_points.dtype != np.float64:
        raise ValueError(
            f"theta/phi point arrays must be float64, got {theta_points.dtype}, {phi_points.dtype}"
        )
    if not np.all(np.isfinite(theta_points)) or not np.all(np.isfinite(phi_points)):
        raise ValueError("theta/phi point arrays contain NaN or Inf")

    expected_count = int(np.prod(metadata.get("matrix_shape", [theta_points.size, 0])[:1]))
    if theta_points.size != expected_count:
        raise ValueError(
            f"Point count mismatch: angle arrays={theta_points.size}, metadata={expected_count}"
        )

    expected_theta = np.repeat(np.asarray(theta_axis), int(phi_axis.size))
    expected_phi = np.tile(np.asarray(phi_axis), int(theta_axis.size))
    if expected_theta.shape != theta_points.shape or expected_phi.shape != phi_points.shape:
        raise ValueError(
            "The cached point arrays are not compatible with the cached theta/phi axes"
        )
    if not np.array_equal(np.asarray(theta_points), expected_theta):
        raise ValueError("Cached theta point order is not theta-major canonical order")
    if not np.array_equal(np.asarray(phi_points), expected_phi):
        raise ValueError("Cached phi point order is not theta-major canonical order")

    report = {
        "cache_dir": str(cache_dir),
        "cache_script_version": complete.get("script_version"),
        "point_count": int(theta_points.size),
        "theta_count": int(theta_axis.size),
        "phi_count": int(phi_axis.size),
        "theta_points_sha256": sha256_file(theta_points_path),
        "phi_points_sha256": sha256_file(phi_points_path),
        "row_order_digest": sha256_arrays(theta_points, phi_points),
        "cache_metadata": metadata,
    }
    return np.asarray(theta_points), np.asarray(phi_points), report


def spherical_to_direction_cosines(
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.asarray(theta_rad, dtype=np.float64)
    phi = np.asarray(phi_rad, dtype=np.float64)
    if theta.shape != phi.shape:
        raise ValueError(f"theta/phi shape mismatch: {theta.shape} vs {phi.shape}")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(phi)):
        raise ValueError("theta/phi contain NaN or Inf")

    sin_theta = np.sin(theta)
    u = sin_theta * np.cos(phi)
    v = sin_theta * np.sin(phi)
    w = np.cos(theta)
    return (
        np.asarray(u, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
        np.asarray(w, dtype=np.float64),
    )


def build_scan_direction(
    scan_theta_deg: float,
    scan_phi_deg: float,
) -> tuple[float, float, float, float]:
    theta_deg = float(scan_theta_deg)
    if not math.isfinite(theta_deg) or not (0.0 <= theta_deg <= 90.0):
        raise ValueError("scan_theta_deg must be finite and within [0, 90]")
    phi_deg = normalize_phi_deg(scan_phi_deg)
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    u = math.sin(theta) * math.cos(phi)
    v = math.sin(theta) * math.sin(phi)
    w = math.cos(theta)
    return float(u), float(v), float(w), float(phi_deg)


def find_target_grid_index(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    scan_u: float,
    scan_v: float,
    scan_w: float,
    *,
    tolerance_deg: float,
) -> tuple[int, np.ndarray]:
    if float(tolerance_deg) < 0.0:
        raise ValueError("target tolerance must be nonnegative")
    dot = (
        np.asarray(u, dtype=np.float64) * float(scan_u)
        + np.asarray(v, dtype=np.float64) * float(scan_v)
        + np.asarray(w, dtype=np.float64) * float(scan_w)
    )
    distance = np.arccos(np.clip(dot, -1.0, 1.0))
    index = int(np.argmin(distance))
    min_deg = float(np.rad2deg(distance[index]))
    if min_deg > float(tolerance_deg):
        raise ValueError(
            "The scan direction is not represented by an exact grid point: "
            f"nearest angular error={min_deg:.12g} deg, tolerance={tolerance_deg:.12g} deg"
        )
    return index, np.asarray(distance, dtype=np.float64)


def build_fixed_uv_domain(
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    *,
    scan_theta_deg: float = 50.0,
    scan_phi_deg: float = 0.0,
    mainlobe_radius_uv: float = 0.13,
    boundary_tolerance: float = 1e-12,
    target_tolerance_deg: float = 1e-9,
) -> GridDomain:
    """Build a fixed circular mainlobe mask without changing point order."""

    theta = np.asarray(theta_rad, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi_rad, dtype=np.float64).reshape(-1)
    if theta.shape != phi.shape or theta.size == 0:
        raise ValueError("theta/phi must be nonempty 1-D arrays of equal length")
    radius = float(mainlobe_radius_uv)
    tol = float(boundary_tolerance)
    if not math.isfinite(radius) or not (0.0 < radius < 2.0):
        raise ValueError("mainlobe_radius_uv must be finite and within (0, 2)")
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError("boundary_tolerance must be finite and nonnegative")

    u, v, w = spherical_to_direction_cosines(theta, phi)
    scan_u, scan_v, scan_w, normalized_phi_deg = build_scan_direction(
        scan_theta_deg, scan_phi_deg
    )
    target_index, angular_distance = find_target_grid_index(
        u,
        v,
        w,
        scan_u,
        scan_v,
        scan_w,
        tolerance_deg=target_tolerance_deg,
    )

    delta_u = u - scan_u
    delta_v = v - scan_v
    rho_uv = np.hypot(delta_u, delta_v)
    psi_uv = np.arctan2(delta_v, delta_u)
    main_mask = rho_uv <= (radius + tol)
    side_mask = np.logical_not(main_mask)
    main_indices = np.flatnonzero(main_mask).astype(np.int64)
    side_indices = np.flatnonzero(side_mask).astype(np.int64)

    domain = GridDomain(
        scan_theta_deg=float(scan_theta_deg),
        scan_phi_deg=normalized_phi_deg,
        scan_u=scan_u,
        scan_v=scan_v,
        scan_w=scan_w,
        theta_rad=theta,
        phi_rad=phi,
        u=u,
        v=v,
        w=w,
        delta_u=np.asarray(delta_u, dtype=np.float64),
        delta_v=np.asarray(delta_v, dtype=np.float64),
        rho_uv=np.asarray(rho_uv, dtype=np.float64),
        psi_uv=np.asarray(psi_uv, dtype=np.float64),
        angular_distance_rad=angular_distance,
        target_index=target_index,
        main_mask=np.asarray(main_mask, dtype=bool),
        side_mask=np.asarray(side_mask, dtype=bool),
        main_indices=main_indices,
        side_indices=side_indices,
        mainlobe_radius_uv=radius,
        boundary_tolerance=tol,
        target_tolerance_deg=float(target_tolerance_deg),
    )
    validate_grid_domain(domain)
    return domain


def validate_grid_domain(domain: GridDomain) -> dict[str, Any]:
    n = domain.point_count
    point_arrays = {
        "theta_rad": domain.theta_rad,
        "phi_rad": domain.phi_rad,
        "u": domain.u,
        "v": domain.v,
        "w": domain.w,
        "delta_u": domain.delta_u,
        "delta_v": domain.delta_v,
        "rho_uv": domain.rho_uv,
        "psi_uv": domain.psi_uv,
        "angular_distance_rad": domain.angular_distance_rad,
        "main_mask": domain.main_mask,
        "side_mask": domain.side_mask,
    }
    for name, value in point_arrays.items():
        if np.asarray(value).shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {np.asarray(value).shape}")
    for name in [
        "theta_rad",
        "phi_rad",
        "u",
        "v",
        "w",
        "delta_u",
        "delta_v",
        "rho_uv",
        "psi_uv",
        "angular_distance_rad",
    ]:
        if not np.all(np.isfinite(np.asarray(point_arrays[name]))):
            raise ValueError(f"{name} contains NaN or Inf")

    if domain.main_mask.dtype != bool or domain.side_mask.dtype != bool:
        raise ValueError("main_mask and side_mask must be boolean")
    if not np.all(np.logical_or(domain.main_mask, domain.side_mask)):
        raise ValueError("main/side masks do not cover every point")
    if np.any(np.logical_and(domain.main_mask, domain.side_mask)):
        raise ValueError("main/side masks overlap")
    if domain.main_count + domain.side_count != n:
        raise ValueError("main/side index counts do not sum to point_count")
    if not np.array_equal(domain.main_indices, np.flatnonzero(domain.main_mask)):
        raise ValueError("main_indices are inconsistent with main_mask")
    if not np.array_equal(domain.side_indices, np.flatnonzero(domain.side_mask)):
        raise ValueError("side_indices are inconsistent with side_mask")
    if not (0 <= int(domain.target_index) < n):
        raise ValueError("target_index is out of range")
    if not bool(domain.main_mask[domain.target_index]):
        raise ValueError("target grid point is not in the fixed mainlobe domain")

    unit_norm_error = float(
        np.max(np.abs(domain.u * domain.u + domain.v * domain.v + domain.w * domain.w - 1.0))
    )
    if unit_norm_error > 1e-12:
        raise ValueError(f"Direction-cosine unit-norm error is too large: {unit_norm_error}")

    target_rho = float(domain.rho_uv[domain.target_index])
    target_angle_deg = float(np.rad2deg(domain.angular_distance_rad[domain.target_index]))
    if target_rho > 1e-12:
        raise ValueError(f"Target rho_uv is not zero within tolerance: {target_rho}")
    if target_angle_deg > domain.target_tolerance_deg:
        raise ValueError(
            f"Target angular error {target_angle_deg} exceeds {domain.target_tolerance_deg} deg"
        )
    if domain.main_count <= 0 or domain.main_count >= n:
        raise ValueError("Fixed mainlobe mask is empty or covers the entire grid")

    max_main_rho = float(np.max(domain.rho_uv[domain.main_mask]))
    min_side_rho = float(np.min(domain.rho_uv[domain.side_mask]))
    threshold = domain.mainlobe_radius_uv + domain.boundary_tolerance
    if max_main_rho > threshold + 1e-14:
        raise ValueError("Mainlobe mask contains a point beyond the configured threshold")
    if not min_side_rho > threshold:
        raise ValueError("Sidelobe mask contains a point on or inside the configured threshold")

    target_theta_deg = float(np.rad2deg(domain.theta_rad[domain.target_index]))
    target_phi_deg = float(np.rad2deg(domain.phi_rad[domain.target_index]))
    report = {
        "point_count": n,
        "main_count": domain.main_count,
        "side_count": domain.side_count,
        "main_fraction": float(domain.main_count / n),
        "side_fraction": float(domain.side_count / n),
        "target_index": int(domain.target_index),
        "target_theta_deg": target_theta_deg,
        "target_phi_deg": target_phi_deg,
        "target_rho_uv": target_rho,
        "target_angular_error_deg": target_angle_deg,
        "direction_cosine_unit_norm_max_error": unit_norm_error,
        "max_main_rho_uv": max_main_rho,
        "min_side_rho_uv": min_side_rho,
        "mask_digest": sha256_arrays(np.packbits(domain.main_mask), np.packbits(domain.side_mask)),
        "row_order_digest": sha256_arrays(domain.theta_rad, domain.phi_rad),
    }
    return report


# -----------------------------------------------------------------------------
# Optional baseline audit
# -----------------------------------------------------------------------------
def locate_fullwave_farfield_csv(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Full-wave path not found: {path}")
    candidates = sorted(path.glob("farfield*.csv"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one farfield*.csv in {path}, found {len(candidates)}"
        )
    return candidates[0]


def _safe_relative_db(values: np.ndarray, reference: float) -> np.ndarray:
    floor = max(float(reference) * 1e-15, 1e-300)
    return 20.0 * np.log10(np.maximum(np.asarray(values, dtype=np.float64), floor) / reference)


def _nearest_cut_local_minima(
    domain: GridDomain,
    magnitude: np.ndarray,
) -> dict[str, Any]:
    phi_deg = np.rad2deg(domain.phi_rad)
    circular_delta = np.abs(((phi_deg - domain.scan_phi_deg + 180.0) % 360.0) - 180.0)
    selected_phi = float(phi_deg[int(np.argmin(circular_delta))])
    cut_indices = np.flatnonzero(np.isclose(phi_deg, selected_phi, rtol=0.0, atol=1e-10))
    order = np.argsort(domain.theta_rad[cut_indices])
    cut_indices = cut_indices[order]
    theta_deg = np.rad2deg(domain.theta_rad[cut_indices])
    cut_mag = magnitude[cut_indices]
    peak_local = int(np.argmax(cut_mag))

    local_minima = []
    for i in range(1, cut_mag.size - 1):
        if cut_mag[i] <= cut_mag[i - 1] and cut_mag[i] <= cut_mag[i + 1]:
            local_minima.append(i)
    left = [i for i in local_minima if i < peak_local]
    right = [i for i in local_minima if i > peak_local]
    left_i = max(left) if left else None
    right_i = min(right) if right else None

    peak = float(np.max(magnitude))

    def encode(i: int | None) -> dict[str, Any] | None:
        if i is None:
            return None
        global_index = int(cut_indices[i])
        return {
            "point_index": global_index,
            "theta_deg": float(theta_deg[i]),
            "phi_deg": selected_phi,
            "rho_uv": float(domain.rho_uv[global_index]),
            "relative_to_global_peak_db": float(_safe_relative_db(cut_mag[i : i + 1], peak)[0]),
        }

    return {
        "selected_phi_deg": selected_phi,
        "cut_peak_point_index": int(cut_indices[peak_local]),
        "cut_peak_theta_deg": float(theta_deg[peak_local]),
        "left_nearest_local_minimum": encode(left_i),
        "right_nearest_local_minimum": encode(right_i),
    }


def audit_fixed_mainlobe_radius(
    domain: GridDomain,
    baseline_lhcp: np.ndarray,
    *,
    boundary_half_width_uv: float = 0.005,
    source_path: Path | None = None,
) -> dict[str, Any]:
    field = np.asarray(baseline_lhcp, dtype=np.complex128).reshape(-1)
    if field.shape != (domain.point_count,):
        raise ValueError(
            f"Baseline LHCP field must have shape ({domain.point_count},), got {field.shape}"
        )
    if not np.all(np.isfinite(field)):
        raise ValueError("Baseline LHCP field contains NaN or Inf")
    if boundary_half_width_uv <= 0.0:
        raise ValueError("boundary_half_width_uv must be positive")

    magnitude = np.abs(field)
    peak_index = int(np.argmax(magnitude))
    peak = float(magnitude[peak_index])
    target_mag = float(magnitude[domain.target_index])
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("Baseline LHCP peak is invalid")

    boundary_mask = np.abs(domain.rho_uv - domain.mainlobe_radius_uv) <= float(
        boundary_half_width_uv
    )
    if not np.any(boundary_mask):
        raise ValueError("No grid point falls in the requested boundary audit annulus")

    side_peak_index = int(domain.side_indices[np.argmax(magnitude[domain.side_indices])])
    main_peak_index = int(domain.main_indices[np.argmax(magnitude[domain.main_indices])])
    boundary_values = magnitude[boundary_mask]
    peak_theta_deg = float(np.rad2deg(domain.theta_rad[peak_index]))
    peak_phi_deg = float(np.rad2deg(domain.phi_rad[peak_index]))
    beam_shift_deg = float(np.rad2deg(domain.angular_distance_rad[peak_index]))

    audit = {
        "source_path": None if source_path is None else str(Path(source_path).resolve()),
        "mainlobe_radius_uv": float(domain.mainlobe_radius_uv),
        "boundary_half_width_uv": float(boundary_half_width_uv),
        "boundary_point_count": int(np.sum(boundary_mask)),
        "peak_index": peak_index,
        "peak_theta_deg": peak_theta_deg,
        "peak_phi_deg": peak_phi_deg,
        "peak_rho_uv": float(domain.rho_uv[peak_index]),
        "beam_shift_from_scan_deg": beam_shift_deg,
        "peak_inside_fixed_mainlobe": bool(domain.main_mask[peak_index]),
        "target_relative_to_peak_db": float(_safe_relative_db(np.array([target_mag]), peak)[0]),
        "boundary_max_relative_to_peak_db": float(
            _safe_relative_db(np.array([np.max(boundary_values)]), peak)[0]
        ),
        "boundary_median_relative_to_peak_db": float(
            _safe_relative_db(np.array([np.median(boundary_values)]), peak)[0]
        ),
        "main_peak_index": main_peak_index,
        "side_peak_index": side_peak_index,
        "side_peak_theta_deg": float(np.rad2deg(domain.theta_rad[side_peak_index])),
        "side_peak_phi_deg": float(np.rad2deg(domain.phi_rad[side_peak_index])),
        "side_peak_rho_uv": float(domain.rho_uv[side_peak_index]),
        "side_peak_relative_to_global_peak_db": float(
            _safe_relative_db(np.array([magnitude[side_peak_index]]), peak)[0]
        ),
        "side_peak_relative_to_target_db": float(
            20.0 * math.log10(max(magnitude[side_peak_index], 1e-300) / max(target_mag, 1e-300))
        ),
        "scan_plane_local_minima": _nearest_cut_local_minima(domain, magnitude),
        "mask_modified_by_audit": False,
    }
    return audit


def load_fullwave_lhcp_for_audit(
    fullwave_path: Path,
    domain: GridDomain,
    *,
    cp_basis_sign: int,
) -> tuple[np.ndarray, Path]:
    csv_path = locate_fullwave_farfield_csv(fullwave_path)
    field = read_hfss_vector_field_csv(csv_path, strict_baseline=True)
    if not np.array_equal(field.theta_points_rad, domain.theta_rad):
        raise ValueError("Full-wave theta grid does not match the Vec-AEP cache row order")
    if not np.array_equal(field.phi_points_rad, domain.phi_rad):
        raise ValueError("Full-wave phi grid does not match the Vec-AEP cache row order")
    e_lhcp, _ = decompose_circular_components(
        field.etheta,
        field.ephi,
        cp_basis_sign=int(cp_basis_sign),
    )
    return e_lhcp, csv_path


# -----------------------------------------------------------------------------
# Serialization
# -----------------------------------------------------------------------------
def _domain_arrays(domain: GridDomain) -> dict[str, np.ndarray]:
    return {
        "theta_rad": np.asarray(domain.theta_rad, dtype=np.float64),
        "phi_rad": np.asarray(domain.phi_rad, dtype=np.float64),
        "u": np.asarray(domain.u, dtype=np.float64),
        "v": np.asarray(domain.v, dtype=np.float64),
        "w": np.asarray(domain.w, dtype=np.float64),
        "delta_u": np.asarray(domain.delta_u, dtype=np.float64),
        "delta_v": np.asarray(domain.delta_v, dtype=np.float64),
        "rho_uv": np.asarray(domain.rho_uv, dtype=np.float64),
        "psi_uv": np.asarray(domain.psi_uv, dtype=np.float64),
        "angular_distance_rad": np.asarray(domain.angular_distance_rad, dtype=np.float64),
        "main_mask": np.asarray(domain.main_mask, dtype=bool),
        "side_mask": np.asarray(domain.side_mask, dtype=bool),
        "main_indices": np.asarray(domain.main_indices, dtype=np.int64),
        "side_indices": np.asarray(domain.side_indices, dtype=np.int64),
        "target_index": np.asarray(domain.target_index, dtype=np.int64),
    }


def save_grid_domain(
    output_dir: Path,
    domain: GridDomain,
    *,
    source_cache_report: Mapping[str, Any] | None = None,
    baseline_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    incomplete = output_dir / INCOMPLETE_NAME
    complete = output_dir / COMPLETE_NAME
    if complete.exists():
        complete.unlink()
    write_json(
        incomplete,
        {
            "status": "building",
            "script_version": SCRIPT_VERSION,
            "started_unix_time": time.time(),
        },
    )

    report = validate_grid_domain(domain)
    arrays = _domain_arrays(domain)
    arrays_path = output_dir / ARRAYS_NAME
    np.savez(arrays_path, **arrays)

    points = pd.DataFrame(
        {
            "point_index": np.arange(domain.point_count, dtype=np.int64),
            "theta_deg": np.rad2deg(domain.theta_rad),
            "phi_deg": np.rad2deg(domain.phi_rad),
            "u": domain.u,
            "v": domain.v,
            "w": domain.w,
            "delta_u": domain.delta_u,
            "delta_v": domain.delta_v,
            "rho_uv": domain.rho_uv,
            "psi_uv_deg": np.rad2deg(domain.psi_uv),
            "angular_distance_deg": np.rad2deg(domain.angular_distance_rad),
            "is_target": np.arange(domain.point_count, dtype=np.int64) == domain.target_index,
            "is_mainlobe": domain.main_mask,
            "is_sidelobe": domain.side_mask,
        }
    )
    points.to_csv(output_dir / POINTS_NAME, index=False, encoding="utf-8-sig")

    summary: dict[str, Any] = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "script_version": SCRIPT_VERSION,
        "scan": {
            "theta_deg": float(domain.scan_theta_deg),
            "phi_deg": float(domain.scan_phi_deg),
            "u": float(domain.scan_u),
            "v": float(domain.scan_v),
            "w": float(domain.scan_w),
        },
        "domain_definition": {
            "coordinate_definition": {
                "u": "sin(theta)*cos(phi)",
                "v": "sin(theta)*sin(phi)",
                "w": "cos(theta)",
                "rho_uv": "sqrt((u-u_scan)^2+(v-v_scan)^2)",
            },
            "mainlobe_rule": "rho_uv <= mainlobe_radius_uv + boundary_tolerance",
            "mainlobe_radius_uv": float(domain.mainlobe_radius_uv),
            "boundary_tolerance": float(domain.boundary_tolerance),
            "target_tolerance_deg": float(domain.target_tolerance_deg),
            "mask_frozen": True,
            "field_interpolation_applied": False,
            "field_basis_transformation_applied": False,
            "point_reordering_applied": False,
        },
        "validation": report,
        "source_cache": None if source_cache_report is None else dict(source_cache_report),
        "baseline_audit": None if baseline_audit is None else dict(baseline_audit),
        "outputs": {
            "arrays": ARRAYS_NAME,
            "points_csv": POINTS_NAME,
            "summary": SUMMARY_NAME,
            "complete_marker": COMPLETE_NAME,
        },
    }
    write_json(output_dir / SUMMARY_NAME, summary)

    # Reload all output types before declaring completion.
    with np.load(arrays_path, allow_pickle=False) as z:
        if set(z.files) != set(arrays.keys()):
            raise IOError("Saved NPZ key set does not match the expected domain arrays")
        if tuple(z["theta_rad"].shape) != (domain.point_count,):
            raise IOError("Saved domain arrays failed shape verification")
    reloaded_points = pd.read_csv(output_dir / POINTS_NAME, nrows=5)
    if list(reloaded_points.columns) != list(points.columns):
        raise IOError("Saved point-table columns failed verification")
    read_json(output_dir / SUMMARY_NAME)

    write_json(
        complete,
        {
            "status": "complete",
            "script_version": SCRIPT_VERSION,
            "completed_unix_time": time.time(),
            "point_count": domain.point_count,
            "main_count": domain.main_count,
            "side_count": domain.side_count,
            "target_index": int(domain.target_index),
            "mask_digest": report["mask_digest"],
        },
    )
    if incomplete.exists():
        incomplete.unlink()
    return summary


def load_grid_domain(output_dir: Path) -> GridDomain:
    output_dir = Path(output_dir).expanduser().resolve()
    complete_path = output_dir / COMPLETE_NAME
    arrays_path = output_dir / ARRAYS_NAME
    summary_path = output_dir / SUMMARY_NAME
    if (output_dir / INCOMPLETE_NAME).exists():
        raise RuntimeError(f"Grid-domain output is marked incomplete: {output_dir}")
    if not complete_path.is_file() or not arrays_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Grid-domain output is incomplete: {output_dir}")
    complete = read_json(complete_path)
    if complete.get("status") != "complete":
        raise ValueError(f"Grid-domain complete marker is invalid: {complete_path}")
    summary = read_json(summary_path)
    scan = summary["scan"]
    definition = summary["domain_definition"]
    with np.load(arrays_path, allow_pickle=False) as z:
        required = {
            "theta_rad",
            "phi_rad",
            "u",
            "v",
            "w",
            "delta_u",
            "delta_v",
            "rho_uv",
            "psi_uv",
            "angular_distance_rad",
            "main_mask",
            "side_mask",
            "main_indices",
            "side_indices",
            "target_index",
        }
        if set(z.files) != required:
            raise ValueError(f"Unexpected domain-array keys: {sorted(z.files)}")
        arrays = {name: np.array(z[name], copy=True) for name in required}

    domain = GridDomain(
        scan_theta_deg=float(scan["theta_deg"]),
        scan_phi_deg=float(scan["phi_deg"]),
        scan_u=float(scan["u"]),
        scan_v=float(scan["v"]),
        scan_w=float(scan["w"]),
        theta_rad=arrays["theta_rad"],
        phi_rad=arrays["phi_rad"],
        u=arrays["u"],
        v=arrays["v"],
        w=arrays["w"],
        delta_u=arrays["delta_u"],
        delta_v=arrays["delta_v"],
        rho_uv=arrays["rho_uv"],
        psi_uv=arrays["psi_uv"],
        angular_distance_rad=arrays["angular_distance_rad"],
        target_index=int(arrays["target_index"].item()),
        main_mask=arrays["main_mask"].astype(bool),
        side_mask=arrays["side_mask"].astype(bool),
        main_indices=arrays["main_indices"].astype(np.int64),
        side_indices=arrays["side_indices"].astype(np.int64),
        mainlobe_radius_uv=float(definition["mainlobe_radius_uv"]),
        boundary_tolerance=float(definition["boundary_tolerance"]),
        target_tolerance_deg=float(definition["target_tolerance_deg"]),
    )
    validate_grid_domain(domain)
    return domain


# -----------------------------------------------------------------------------
# Self-test and command line
# -----------------------------------------------------------------------------
def run_self_test() -> dict[str, Any]:
    theta_deg = np.array([0.0, 90.0, 90.0, 50.0, 45.0], dtype=np.float64)
    phi_deg = np.array([0.0, 0.0, 90.0, 0.0, 0.0], dtype=np.float64)
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    u, v, w = spherical_to_direction_cosines(theta, phi)
    expected = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    actual = np.column_stack([u[:3], v[:3], w[:3]])
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-14):
        raise AssertionError(f"Direction-cosine test failed: {actual}")

    domain = build_fixed_uv_domain(
        theta,
        phi,
        scan_theta_deg=50.0,
        scan_phi_deg=360.0,
        mainlobe_radius_uv=0.13,
        target_tolerance_deg=1e-9,
    )
    if domain.target_index != 3:
        raise AssertionError(f"Expected target index 3, got {domain.target_index}")
    if not math.isclose(domain.scan_phi_deg, 0.0, abs_tol=1e-14):
        raise AssertionError("Phi normalization failed")
    if not math.isclose(float(domain.rho_uv[3]), 0.0, abs_tol=1e-14):
        raise AssertionError("Target rho_uv is not zero")
    if not np.all(np.logical_xor(domain.main_mask, domain.side_mask)):
        raise AssertionError("Main/side partition test failed")

    with tempfile.TemporaryDirectory(prefix="stage3_grid_domain_selftest_") as tmp:
        output = Path(tmp) / "domain"
        save_grid_domain(output, domain)
        loaded = load_grid_domain(output)
        if not np.array_equal(domain.main_mask, loaded.main_mask):
            raise AssertionError("Serialization round-trip mask mismatch")
        if not np.array_equal(domain.rho_uv, loaded.rho_uv):
            raise AssertionError("Serialization round-trip rho_uv mismatch")

    return {
        "status": "PASS",
        "script_version": SCRIPT_VERSION,
        "direction_cosine_max_error": float(np.max(np.abs(actual - expected))),
        "target_index": int(domain.target_index),
        "main_count": int(domain.main_count),
        "side_count": int(domain.side_count),
        "serialization_roundtrip": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fixed scan-centered u-v mainlobe/sidelobe domain."
    )
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--vecaep-cache", type=Path, default=DEFAULT_VECAEP_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scan-theta-deg", type=float, default=50.0)
    parser.add_argument("--scan-phi-deg", type=float, default=0.0)
    parser.add_argument("--mainlobe-radius-uv", type=float, default=0.13)
    parser.add_argument("--boundary-tolerance", type=float, default=1e-12)
    parser.add_argument("--target-tolerance-deg", type=float, default=1e-9)
    parser.add_argument("--baseline-fullwave-dir", type=Path, default=None)
    parser.add_argument("--boundary-half-width-uv", type=float, default=0.005)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        logger = setup_logger(None, args.log_level)
        try:
            result = run_self_test()
            logger.info("[SELF-TEST] status=PASS")
            logger.info("[SELF-TEST] %s", result)
            return 0
        except Exception as exc:
            logger.error("[SELF-TEST] status=FAILED error=%s", exc)
            logger.debug("[SELF-TEST] traceback:\n%s", traceback.format_exc())
            return 1

    start = time.perf_counter()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / make_domain_name(
        args.scan_theta_deg, args.scan_phi_deg, args.mainlobe_radius_uv
    )
    if output_dir.exists():
        if not args.overwrite:
            print(
                f"ERROR | Output directory already exists; use --overwrite: {output_dir}",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / LOG_NAME, args.log_level)

    try:
        logger.info("[VERSION] stage3_grid_domain.py %s", SCRIPT_VERSION)
        logger.info("[INPUT] vecaep_cache=%s", Path(args.vecaep_cache).resolve())
        logger.info(
            "[CONFIG] scan=(%.6f, %.6f) deg, rho_main=%.12g",
            args.scan_theta_deg,
            normalize_phi_deg(args.scan_phi_deg),
            args.mainlobe_radius_uv,
        )
        theta, phi, source_report = load_grid_from_vecaep_cache(args.vecaep_cache)
        logger.info(
            "[PASS] canonical grid loaded: points=%d, theta=%d, phi=%d",
            source_report["point_count"],
            source_report["theta_count"],
            source_report["phi_count"],
        )
        domain = build_fixed_uv_domain(
            theta,
            phi,
            scan_theta_deg=args.scan_theta_deg,
            scan_phi_deg=args.scan_phi_deg,
            mainlobe_radius_uv=args.mainlobe_radius_uv,
            boundary_tolerance=args.boundary_tolerance,
            target_tolerance_deg=args.target_tolerance_deg,
        )
        validation = validate_grid_domain(domain)
        logger.info(
            "[PASS] target index=%d, theta=%.6f deg, phi=%.6f deg, rho_uv=%.3e",
            domain.target_index,
            validation["target_theta_deg"],
            validation["target_phi_deg"],
            validation["target_rho_uv"],
        )
        logger.info(
            "[PASS] fixed partition: main=%d, side=%d, total=%d",
            domain.main_count,
            domain.side_count,
            domain.point_count,
        )

        baseline_audit = None
        if args.baseline_fullwave_dir is not None:
            cp_basis_sign = int(source_report["cache_metadata"]["cp_basis_sign"])
            baseline_lhcp, csv_path = load_fullwave_lhcp_for_audit(
                args.baseline_fullwave_dir,
                domain,
                cp_basis_sign=cp_basis_sign,
            )
            baseline_audit = audit_fixed_mainlobe_radius(
                domain,
                baseline_lhcp,
                boundary_half_width_uv=args.boundary_half_width_uv,
                source_path=csv_path,
            )
            logger.info(
                "[AUDIT] LHCP peak=(%.3f, %.3f) deg, beam shift=%.3f deg, inside main=%s",
                baseline_audit["peak_theta_deg"],
                baseline_audit["peak_phi_deg"],
                baseline_audit["beam_shift_from_scan_deg"],
                baseline_audit["peak_inside_fixed_mainlobe"],
            )
            logger.info(
                "[AUDIT] side peak/global peak=%.6f dB, boundary max/global peak=%.6f dB",
                baseline_audit["side_peak_relative_to_global_peak_db"],
                baseline_audit["boundary_max_relative_to_peak_db"],
            )

        save_grid_domain(
            output_dir,
            domain,
            source_cache_report=source_report,
            baseline_audit=baseline_audit,
        )
        duration = float(time.perf_counter() - start)
        logger.info("[PASS] domain output committed: %s", output_dir)
        logger.info("[SUMMARY] status=PASS")
        logger.info("[SUMMARY] output_dir=%s", output_dir)
        logger.info("[SUMMARY] duration=%.3f s", duration)
        return 0
    except Exception as exc:
        logger.error("[SUMMARY] status=FAILED error=%s", exc)
        logger.debug("[SUMMARY] traceback:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
