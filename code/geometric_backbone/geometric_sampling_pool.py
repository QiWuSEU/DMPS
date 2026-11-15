#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate static geometry-sampling pools for backbone synthesis.

Responsibilities
----------------
1. Build full/work scan-angle pools.
2. Build geometry-only static pools for:
   - 1D core   : alpha in [0, alpha_core] deg
   - 2D buffer : alpha in (alpha_core, 90] deg
   - 2D outer  : alpha in (90, 90 + theta_s] deg
3. Output a0 interval descriptors for 1D points, but DO NOT evaluate a0.
4. Output dynamic peak-tracking protocol parameters, but DO NOT execute tracking.

This script reads a strict q0cra_stage1_partition_v2 table produced by
calibrate_q0_partition.py for the 1D/2D boundary, unless manual mode is
explicitly enabled. It intentionally does NOT read unit-cell NPZ/meta files
or interpolate the IEP. The downstream dominant_mode_envelope.py module
evaluates a0 using the interval descriptors generated here.

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
import sys
import time

import numpy as np


FORMAT_VERSION = "q0cra_stage1_static_pool_v2"
PARTITION_FORMAT_VERSION = "q0cra_stage1_partition_v2"

ZONE_CORE_1D = 0
ZONE_BUFFER_2D = 1
ZONE_OUTER_2D = 2

LAYER_BASE = 0
LAYER_BACK_BONUS = 1
LAYER_ANCHOR = 2

ZONE_NAME = {
    ZONE_CORE_1D: "core_1d",
    ZONE_BUFFER_2D: "buffer_2d",
    ZONE_OUTER_2D: "outer_2d",
}

LAYER_NAME = {
    LAYER_BASE: "base",
    LAYER_BACK_BONUS: "back_bonus",
    LAYER_ANCHOR: "anchor",
}

DEFAULT_THETA_S_WORK_INIT_DEG = (
    0.0, 10.0, 20.0, 30.0, 40.0, 45.0,
    50.0, 54.0, 56.0, 58.0, 60.0,
)

DEFAULT_BUFFER_RING_LIST_DEG = (
    60.0, 66.0, 72.0, 78.0, 84.0, 90.0,
)

OUTPUT_BASENAMES = (
    "q0cra_stage1_static_pool.npz",
    "q0cra_stage1_static_pool_summary.json",
    "q0cra_stage1_static_pool_counts_full.csv",
    "q0cra_stage1_static_pool_counts_work.csv",
    "q0cra_stage1_static_pool.log",
)


# =============================================================================
# Python-3.6-compatible record classes
# =============================================================================


class TrackingProtocol(object):
    __slots__ = (
        "topk_buffer", "topk_outer",
        "buf_delta_alpha_1_deg", "buf_delta_eta_1_deg",
        "buf_delta_alpha_2_deg", "buf_delta_eta_2_deg",
        "out_delta_alpha_1_deg", "out_delta_eta_1_deg",
        "out_delta_alpha_2_deg", "out_delta_eta_2_deg",
        "update_work_every_n_iters", "reinject_worst_theta_count",
        "work_max_size",
    )

    def __init__(
            self,
            topk_buffer=2,
            topk_outer=3,
            buf_delta_alpha_1_deg=1.5,
            buf_delta_eta_1_deg=8.0,
            buf_delta_alpha_2_deg=0.75,
            buf_delta_eta_2_deg=4.0,
            out_delta_alpha_1_deg=2.0,
            out_delta_eta_1_deg=10.0,
            out_delta_alpha_2_deg=1.0,
            out_delta_eta_2_deg=5.0,
            update_work_every_n_iters=3,
            reinject_worst_theta_count=2,
            work_max_size=15):
        self.topk_buffer = int(topk_buffer)
        self.topk_outer = int(topk_outer)
        self.buf_delta_alpha_1_deg = float(buf_delta_alpha_1_deg)
        self.buf_delta_eta_1_deg = float(buf_delta_eta_1_deg)
        self.buf_delta_alpha_2_deg = float(buf_delta_alpha_2_deg)
        self.buf_delta_eta_2_deg = float(buf_delta_eta_2_deg)
        self.out_delta_alpha_1_deg = float(out_delta_alpha_1_deg)
        self.out_delta_eta_1_deg = float(out_delta_eta_1_deg)
        self.out_delta_alpha_2_deg = float(out_delta_alpha_2_deg)
        self.out_delta_eta_2_deg = float(out_delta_eta_2_deg)
        self.update_work_every_n_iters = int(update_work_every_n_iters)
        self.reinject_worst_theta_count = int(reinject_worst_theta_count)
        self.work_max_size = int(work_max_size)


class PoolConfigStage1(object):
    __slots__ = (
        "phi_s_deg", "alpha_core_max_deg",
        "partition_table_json", "partition_selection_mode",
        "partition_lookup_K", "partition_lookup_d_min",
        "partition_lookup_d_eff", "partition_lookup_R0",
        "partition_lookup_tol", "require_partition_table",
        "allow_manual_partition",
        "auto_buffer_rings", "buffer_first_offset_deg",
        "buffer_ring_step_deg", "alpha_outer_start_deg",
        "alpha_1d_step_deg",
        "theta_s_full_min_deg", "theta_s_full_max_deg",
        "theta_s_full_step_deg", "theta_s_work_init_deg",
        "buffer_ring_list_deg", "outer_ring_step_deg",
        "outer_ring_start_deg", "buf_base_eta_step_min_deg",
        "buf_base_eta_step_max_deg", "buf_back_eta_step_min_deg",
        "buf_back_eta_step_max_deg", "out_base_eta_step_min_deg",
        "out_base_eta_step_max_deg", "back_sector_min_deg",
        "back_sector_max_deg", "outer_anchor_offset_deg",
        "budget_total_per_scan", "a0_mono_test_step_deg",
        "alpha_main_guard_deg", "tracking", "overwrite",
    )

    def __init__(
            self,
            phi_s_deg=0.0,
            alpha_core_max_deg=57.0,
            partition_table_json="",
            partition_selection_mode="global_min",
            partition_lookup_K=8,
            partition_lookup_d_min=0.5,
            partition_lookup_d_eff=0.5,
            partition_lookup_R0=5.0,
            partition_lookup_tol=1e-9,
            require_partition_table=False,
            allow_manual_partition=False,
            auto_buffer_rings=True,
            buffer_first_offset_deg=3.0,
            buffer_ring_step_deg=6.0,
            alpha_outer_start_deg=90.0,
            alpha_1d_step_deg=0.25,
            theta_s_full_min_deg=0.0,
            theta_s_full_max_deg=60.0,
            theta_s_full_step_deg=1.0,
            theta_s_work_init_deg=DEFAULT_THETA_S_WORK_INIT_DEG,
            buffer_ring_list_deg=DEFAULT_BUFFER_RING_LIST_DEG,
            outer_ring_step_deg=5.0,
            outer_ring_start_deg=95.0,
            buf_base_eta_step_min_deg=12.0,
            buf_base_eta_step_max_deg=18.0,
            buf_back_eta_step_min_deg=8.0,
            buf_back_eta_step_max_deg=10.0,
            out_base_eta_step_min_deg=12.0,
            out_base_eta_step_max_deg=16.0,
            back_sector_min_deg=150.0,
            back_sector_max_deg=210.0,
            outer_anchor_offset_deg=12.0,
            budget_total_per_scan=1000,
            a0_mono_test_step_deg=1.0,
            alpha_main_guard_deg=7.5,
            tracking=None,
            overwrite=False):
        self.phi_s_deg = float(phi_s_deg)
        self.alpha_core_max_deg = float(alpha_core_max_deg)
        self.partition_table_json = str(partition_table_json)
        self.partition_selection_mode = str(partition_selection_mode)
        self.partition_lookup_K = int(partition_lookup_K)
        self.partition_lookup_d_min = float(partition_lookup_d_min)
        self.partition_lookup_d_eff = float(partition_lookup_d_eff)
        self.partition_lookup_R0 = float(partition_lookup_R0)
        self.partition_lookup_tol = float(partition_lookup_tol)
        self.require_partition_table = bool(require_partition_table)
        self.allow_manual_partition = bool(allow_manual_partition)
        self.auto_buffer_rings = bool(auto_buffer_rings)
        self.buffer_first_offset_deg = float(buffer_first_offset_deg)
        self.buffer_ring_step_deg = float(buffer_ring_step_deg)
        self.alpha_outer_start_deg = float(alpha_outer_start_deg)
        self.alpha_1d_step_deg = float(alpha_1d_step_deg)
        self.theta_s_full_min_deg = float(theta_s_full_min_deg)
        self.theta_s_full_max_deg = float(theta_s_full_max_deg)
        self.theta_s_full_step_deg = float(theta_s_full_step_deg)
        self.theta_s_work_init_deg = tuple(float(x) for x in theta_s_work_init_deg)
        self.buffer_ring_list_deg = tuple(float(x) for x in buffer_ring_list_deg)
        self.outer_ring_step_deg = float(outer_ring_step_deg)
        self.outer_ring_start_deg = float(outer_ring_start_deg)
        self.buf_base_eta_step_min_deg = float(buf_base_eta_step_min_deg)
        self.buf_base_eta_step_max_deg = float(buf_base_eta_step_max_deg)
        self.buf_back_eta_step_min_deg = float(buf_back_eta_step_min_deg)
        self.buf_back_eta_step_max_deg = float(buf_back_eta_step_max_deg)
        self.out_base_eta_step_min_deg = float(out_base_eta_step_min_deg)
        self.out_base_eta_step_max_deg = float(out_base_eta_step_max_deg)
        self.back_sector_min_deg = float(back_sector_min_deg)
        self.back_sector_max_deg = float(back_sector_max_deg)
        self.outer_anchor_offset_deg = float(outer_anchor_offset_deg)
        self.budget_total_per_scan = int(budget_total_per_scan)
        self.a0_mono_test_step_deg = float(a0_mono_test_step_deg)
        self.alpha_main_guard_deg = float(alpha_main_guard_deg)
        self.tracking = TrackingProtocol() if tracking is None else tracking
        self.overwrite = bool(overwrite)

    def copy_with(self, **updates):
        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return PoolConfigStage1(**values)


class PartitionInfo(object):
    __slots__ = (
        "mode", "selection_mode", "source_json", "source_sha256",
        "table_version", "table_quality_status", "table_warnings",
        "K", "d_min", "d_eff", "R0",
        "alpha_core_calib_deg", "alpha_core_used_deg",
        "rho_core_used", "alpha_outer_start_deg", "worst_b",
        "xi_core", "peak_gate_db", "n_consecutive_bad",
        "persistent_bad_width_deg", "required_bad_sample_count",
        "isolated_bad_policy", "safety_margin_deg",
    )

    def __init__(
            self,
            mode="manual",
            selection_mode="manual",
            source_json="",
            source_sha256="",
            table_version="",
            table_quality_status="MANUAL",
            table_warnings=None,
            K=0,
            d_min=float("nan"),
            d_eff=float("nan"),
            R0=float("nan"),
            alpha_core_calib_deg=float("nan"),
            alpha_core_used_deg=float("nan"),
            rho_core_used=float("nan"),
            alpha_outer_start_deg=90.0,
            worst_b=float("nan"),
            xi_core=float("nan"),
            peak_gate_db=float("nan"),
            n_consecutive_bad=0,
            persistent_bad_width_deg=float("nan"),
            required_bad_sample_count=0,
            isolated_bad_policy="",
            safety_margin_deg=float("nan")):
        self.mode = str(mode)
        self.selection_mode = str(selection_mode)
        self.source_json = str(source_json)
        self.source_sha256 = str(source_sha256)
        self.table_version = str(table_version)
        self.table_quality_status = str(table_quality_status)
        self.table_warnings = list(table_warnings or [])
        self.K = int(K)
        self.d_min = float(d_min)
        self.d_eff = float(d_eff)
        self.R0 = float(R0)
        self.alpha_core_calib_deg = float(alpha_core_calib_deg)
        self.alpha_core_used_deg = float(alpha_core_used_deg)
        self.rho_core_used = float(rho_core_used)
        self.alpha_outer_start_deg = float(alpha_outer_start_deg)
        self.worst_b = float(worst_b)
        self.xi_core = float(xi_core)
        self.peak_gate_db = float(peak_gate_db)
        self.n_consecutive_bad = int(n_consecutive_bad)
        self.persistent_bad_width_deg = float(persistent_bad_width_deg)
        self.required_bad_sample_count = int(required_bad_sample_count)
        self.isolated_bad_policy = str(isolated_bad_policy)
        self.safety_margin_deg = float(safety_margin_deg)


class PointBatch1D(object):
    __slots__ = (
        "scan_idx", "theta_s_deg", "alpha_deg", "rho",
        "theta_min_deg", "theta_max_deg", "zone_code",
        "is_main_guard", "is_side_1d",
    )

    def __init__(self, scan_idx=None, theta_s_deg=None, alpha_deg=None,
                 rho=None, theta_min_deg=None, theta_max_deg=None,
                 zone_code=None, is_main_guard=None, is_side_1d=None):
        self.scan_idx = list(scan_idx or [])
        self.theta_s_deg = list(theta_s_deg or [])
        self.alpha_deg = list(alpha_deg or [])
        self.rho = list(rho or [])
        self.theta_min_deg = list(theta_min_deg or [])
        self.theta_max_deg = list(theta_max_deg or [])
        self.zone_code = list(zone_code or [])
        self.is_main_guard = list(is_main_guard or [])
        self.is_side_1d = list(is_side_1d or [])

    @classmethod
    def empty(cls):
        return cls()

    def append(self, scan_idx, theta_s_deg, alpha_deg, alpha_main_guard_deg):
        theta_min = abs(float(theta_s_deg) - float(alpha_deg))
        theta_max = min(float(theta_s_deg) + float(alpha_deg), 90.0)
        is_main = int(float(alpha_deg) <= float(alpha_main_guard_deg) + 1e-12)
        is_side = int(not bool(is_main))
        self.scan_idx.append(int(scan_idx))
        self.theta_s_deg.append(float(theta_s_deg))
        self.alpha_deg.append(float(alpha_deg))
        self.rho.append(float(math.sin(math.radians(float(alpha_deg)))))
        self.theta_min_deg.append(float(theta_min))
        self.theta_max_deg.append(float(theta_max))
        self.zone_code.append(ZONE_CORE_1D)
        self.is_main_guard.append(is_main)
        self.is_side_1d.append(is_side)

    def to_dict(self):
        return {
            "scan_idx": np.asarray(self.scan_idx, dtype=np.int32),
            "theta_s_deg": np.asarray(self.theta_s_deg, dtype=np.float64),
            "alpha_deg": np.asarray(self.alpha_deg, dtype=np.float64),
            "rho": np.asarray(self.rho, dtype=np.float64),
            "theta_min_deg": np.asarray(self.theta_min_deg, dtype=np.float64),
            "theta_max_deg": np.asarray(self.theta_max_deg, dtype=np.float64),
            "zone_code": np.asarray(self.zone_code, dtype=np.int8),
            "is_main_guard": np.asarray(self.is_main_guard, dtype=np.int8),
            "is_side_1d": np.asarray(self.is_side_1d, dtype=np.int8),
        }


class PointBatch2D(object):
    __slots__ = (
        "scan_idx", "theta_s_deg", "alpha_deg", "eta_deg",
        "theta_deg", "phi_deg", "u", "v", "zone_code",
        "layer_code", "is_side_2d",
    )

    def __init__(self, scan_idx=None, theta_s_deg=None, alpha_deg=None,
                 eta_deg=None, theta_deg=None, phi_deg=None, u=None, v=None,
                 zone_code=None, layer_code=None, is_side_2d=None):
        self.scan_idx = list(scan_idx or [])
        self.theta_s_deg = list(theta_s_deg or [])
        self.alpha_deg = list(alpha_deg or [])
        self.eta_deg = list(eta_deg or [])
        self.theta_deg = list(theta_deg or [])
        self.phi_deg = list(phi_deg or [])
        self.u = list(u or [])
        self.v = list(v or [])
        self.zone_code = list(zone_code or [])
        self.layer_code = list(layer_code or [])
        self.is_side_2d = list(is_side_2d or [])

    @classmethod
    def empty(cls):
        return cls()

    def append(self, scan_idx, theta_s_deg, alpha_deg, eta_deg,
               theta_deg, phi_deg, u, v, zone_code, layer_code,
               is_side_2d=1):
        self.scan_idx.append(int(scan_idx))
        self.theta_s_deg.append(float(theta_s_deg))
        self.alpha_deg.append(float(alpha_deg))
        self.eta_deg.append(float(eta_deg))
        self.theta_deg.append(float(theta_deg))
        self.phi_deg.append(float(phi_deg))
        self.u.append(float(u))
        self.v.append(float(v))
        self.zone_code.append(int(zone_code))
        self.layer_code.append(int(layer_code))
        self.is_side_2d.append(int(is_side_2d))

    def to_dict(self):
        return {
            "scan_idx": np.asarray(self.scan_idx, dtype=np.int32),
            "theta_s_deg": np.asarray(self.theta_s_deg, dtype=np.float64),
            "alpha_deg": np.asarray(self.alpha_deg, dtype=np.float64),
            "eta_deg": np.asarray(self.eta_deg, dtype=np.float64),
            "theta_deg": np.asarray(self.theta_deg, dtype=np.float64),
            "phi_deg": np.asarray(self.phi_deg, dtype=np.float64),
            "u": np.asarray(self.u, dtype=np.float64),
            "v": np.asarray(self.v, dtype=np.float64),
            "zone_code": np.asarray(self.zone_code, dtype=np.int8),
            "layer_code": np.asarray(self.layer_code, dtype=np.int8),
            "is_side_2d": np.asarray(self.is_side_2d, dtype=np.int8),
        }


class PoolPackageStage1(object):
    __slots__ = (
        "config", "partition_info", "theta_s_full_deg",
        "theta_s_work_init_deg", "theta_s_work_full_scan_idx",
        "core_full", "core_work", "buffer_full", "buffer_work",
        "outer_full", "outer_work", "counts_full", "counts_work",
        "validation_summary",
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


def _float_range(start, stop, step):
    if float(step) <= 0.0:
        raise ValueError("step must be > 0")
    if float(stop) < float(start):
        raise ValueError("stop must be >= start")
    n = int(np.floor((float(stop) - float(start)) / float(step) + 1e-12)) + 1
    arr = float(start) + float(step) * np.arange(n, dtype=np.float64)
    arr = arr[arr <= float(stop) + 1e-12]
    return np.round(arr, 12).astype(np.float64)


def _closed_float_grid(start, stop, step):
    grid = _float_range(start, stop, step)
    if grid.size == 0:
        return np.asarray([float(stop)], dtype=np.float64)
    if abs(float(grid[-1]) - float(stop)) > 1e-10:
        grid = np.concatenate((grid, np.asarray([float(stop)], dtype=np.float64)))
    out = []
    seen = set()
    for value in np.round(grid, 12).tolist():
        key = round(float(value), 12)
        if key not in seen:
            seen.add(key)
            out.append(float(value))
    return np.asarray(out, dtype=np.float64)


def _ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


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


def _parse_float_tuple_csv(value):
    if not str(value).strip():
        return ()
    return tuple(float(x.strip()) for x in str(value).split(",") if x.strip())


def _atomic_write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_record_to_dict(payload), f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_counts_csv(path, counts):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scan_idx", "theta_s_deg", "n_core_1d",
            "n_core_main_guard", "n_core_side_1d", "n_buffer_2d",
            "n_buffer_side_2d", "n_outer_2d", "n_outer_side_2d",
            "n_total",
        ])
        for row in counts:
            w.writerow([
                row["scan_idx"], "{0:.2f}".format(row["theta_s_deg"]),
                row["n_core_1d"], row["n_core_main_guard"],
                row["n_core_side_1d"], row["n_buffer_2d"],
                row["n_buffer_side_2d"], row["n_outer_2d"],
                row["n_outer_side_2d"], row["n_total"],
            ])
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_savez(path, payload):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def _setup_logger(path):
    logger = logging.getLogger("pool_q0cra_stage1_static")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    file_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    console_formatter = logging.Formatter("%(levelname)s | %(message)s")
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setFormatter(file_formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(console_formatter)
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
    existing = [path for path in paths if os.path.exists(path)]
    if existing and not bool(overwrite):
        raise IOError(
            "Output files already exist; use --overwrite or a new --out-dir:\n  "
            + "\n  ".join(existing)
        )


# =============================================================================
# Geometry helpers (original policy retained)
# =============================================================================


def _scan_weight(theta_s_deg):
    denom = math.sin(math.radians(60.0))
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(math.sin(math.radians(float(theta_s_deg))) / denom, 0.0, 1.0))


def _visible_eta_interval_deg(alpha_deg, theta_s_deg, tol=1e-12):
    """Return (kind, start_deg, stop_deg), kind in none/single/arc/full."""
    a = float(alpha_deg)
    ts = float(theta_s_deg)

    if ts <= float(tol):
        if a <= 90.0 + float(tol):
            return "full", 0.0, 360.0
        return "none", 0.0, 0.0

    if a > 90.0 + ts + 1e-9:
        return "none", 0.0, 0.0

    alpha_rad = math.radians(a)
    theta_s_rad = math.radians(ts)
    sin_a = math.sin(alpha_rad)
    sin_t = math.sin(theta_s_rad)
    if abs(sin_a) < float(tol) or abs(sin_t) < float(tol):
        return "full", 0.0, 360.0

    c = (math.cos(alpha_rad) / sin_a) * (math.cos(theta_s_rad) / sin_t)
    if c >= 1.0 - 1e-12:
        return "full", 0.0, 360.0
    if c <= -1.0 - 1e-12:
        return "none", 0.0, 0.0
    if abs(c + 1.0) <= 1e-12:
        return "single", 180.0, 180.0

    eta0 = math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))
    return "arc", eta0, 360.0 - eta0


def _sample_arc_deg(kind, start_deg, stop_deg, step_deg):
    if float(step_deg) <= 0.0:
        raise ValueError("eta step must be > 0")
    if kind == "none":
        return np.asarray([], dtype=np.float64)
    if kind == "single":
        return np.asarray([180.0], dtype=np.float64)
    if kind == "full":
        values = np.arange(0.0, 360.0, float(step_deg), dtype=np.float64)
        return np.mod(values, 360.0)
    if kind == "arc":
        if float(stop_deg) < float(start_deg):
            return np.asarray([], dtype=np.float64)
        values = np.arange(
            float(start_deg),
            float(stop_deg) + 0.5 * float(step_deg),
            float(step_deg),
            dtype=np.float64,
        )
        return np.clip(values, float(start_deg), float(stop_deg))
    raise ValueError("Unsupported arc kind: {0}".format(kind))


def _intersect_arcs(a0, a1, b0, b1):
    lo = max(float(a0), float(b0))
    hi = min(float(a1), float(b1))
    if hi < lo - 1e-12:
        return False, 0.0, 0.0
    return True, lo, hi


def _dedupe_angles_deg(items, round_digits=6):
    best = {}
    for eta_deg, layer_code in items:
        key = round(float(eta_deg) % 360.0, int(round_digits))
        if key not in best:
            best[key] = int(layer_code)
        else:
            best[key] = max(best[key], int(layer_code))
    return [(key, best[key]) for key in sorted(best.keys())]


def _local_to_global(theta_s_deg, phi_s_deg, alpha_deg, eta_deg):
    ths = math.radians(float(theta_s_deg))
    phs = math.radians(float(phi_s_deg))
    alpha = math.radians(float(alpha_deg))
    eta = math.radians(float(eta_deg))

    scan = np.array([
        math.sin(ths) * math.cos(phs),
        math.sin(ths) * math.sin(phs),
        math.cos(ths),
    ], dtype=np.float64)
    e_theta = np.array([
        math.cos(ths) * math.cos(phs),
        math.cos(ths) * math.sin(phs),
        -math.sin(ths),
    ], dtype=np.float64)
    e_phi = np.array([
        -math.sin(phs),
        math.cos(phs),
        0.0,
    ], dtype=np.float64)

    direction = (
        math.cos(alpha) * scan
        + math.sin(alpha) * (
            math.cos(eta) * e_theta + math.sin(eta) * e_phi
        )
    )
    rx, ry, rz = float(direction[0]), float(direction[1]), float(direction[2])
    rz = float(np.clip(rz, -1.0, 1.0))
    theta_deg = math.degrees(math.acos(rz))
    phi_deg = math.degrees(math.atan2(ry, rx))
    if phi_deg < 0.0:
        phi_deg += 360.0
    sin_theta = math.sin(math.radians(theta_deg))
    u = sin_theta * math.cos(math.radians(phi_deg))
    v = sin_theta * math.sin(math.radians(phi_deg))
    return float(theta_deg), float(phi_deg), float(u), float(v)


# =============================================================================
# Sampling policies (original policy retained)
# =============================================================================


def _buf_base_step_deg(theta_s_deg, cfg):
    s = _scan_weight(theta_s_deg)
    return (
        cfg.buf_base_eta_step_max_deg
        - (cfg.buf_base_eta_step_max_deg - cfg.buf_base_eta_step_min_deg) * s
    )


def _buf_back_step_deg(theta_s_deg, cfg):
    s = _scan_weight(theta_s_deg)
    return (
        cfg.buf_back_eta_step_max_deg
        - (cfg.buf_back_eta_step_max_deg - cfg.buf_back_eta_step_min_deg) * s
    )


def _out_base_step_deg(theta_s_deg, cfg):
    s = _scan_weight(theta_s_deg)
    return (
        cfg.out_base_eta_step_max_deg
        - (cfg.out_base_eta_step_max_deg - cfg.out_base_eta_step_min_deg) * s
    )


def _generate_buffer_eta_samples(alpha_deg, theta_s_deg, cfg):
    kind, a_deg, b_deg = _visible_eta_interval_deg(alpha_deg, theta_s_deg)
    if kind == "none":
        return []

    points = []
    base = _sample_arc_deg(kind, a_deg, b_deg, _buf_base_step_deg(theta_s_deg, cfg))
    points.extend((float(x), LAYER_BASE) for x in base.tolist())

    if kind in ("full", "arc", "single"):
        interval_start = 0.0 if kind == "full" else a_deg
        interval_stop = 360.0 if kind == "full" else b_deg
        ok, lo, hi = _intersect_arcs(
            interval_start, interval_stop,
            cfg.back_sector_min_deg, cfg.back_sector_max_deg,
        )
        if ok:
            if abs(hi - lo) < 1e-9:
                points.append((float(lo), LAYER_BACK_BONUS))
            else:
                step = _buf_back_step_deg(theta_s_deg, cfg)
                dense = np.arange(lo, hi + 0.5 * step, step, dtype=np.float64)
                dense = np.clip(dense, lo, hi)
                points.extend((float(x), LAYER_BACK_BONUS) for x in dense.tolist())
    return _dedupe_angles_deg(points)


def _generate_outer_ring_list(theta_s_deg, cfg):
    theta_s = float(theta_s_deg)
    if theta_s <= 0.0:
        return np.asarray([], dtype=np.float64)
    alpha_max = float(cfg.alpha_outer_start_deg) + theta_s
    values = list(np.arange(
        cfg.outer_ring_start_deg,
        alpha_max + 0.5 * cfg.outer_ring_step_deg,
        cfg.outer_ring_step_deg,
        dtype=np.float64,
    ))
    values = [float(x) for x in values if x <= alpha_max + 1e-12]

    # Necessary coverage repair: for 0 < theta_s < regular first-ring offset,
    # the physical outer domain is non-empty and must contain alpha_max.
    if len(values) == 0:
        return np.asarray([float(alpha_max)], dtype=np.float64)

    if alpha_max - values[-1] > 0.5 * cfg.outer_ring_step_deg + 1e-12:
        values.append(float(alpha_max))
    elif abs(values[-1] - alpha_max) > 1e-9 and alpha_max - values[-1] > 2.5:
        values.append(float(alpha_max))
    return np.asarray(values, dtype=np.float64)


def _generate_outer_eta_samples(alpha_deg, theta_s_deg, cfg):
    kind, a_deg, b_deg = _visible_eta_interval_deg(alpha_deg, theta_s_deg)
    if kind == "none":
        return []

    points = []
    base = _sample_arc_deg(kind, a_deg, b_deg, _out_base_step_deg(theta_s_deg, cfg))
    points.extend((float(x), LAYER_BASE) for x in base.tolist())

    anchor_candidates = [
        180.0,
        180.0 - cfg.outer_anchor_offset_deg,
        180.0 + cfg.outer_anchor_offset_deg,
    ]
    for candidate in anchor_candidates:
        eta = float(candidate % 360.0)
        if kind == "full":
            points.append((eta, LAYER_ANCHOR))
        elif kind == "single":
            if abs(eta - 180.0) <= 1e-9:
                points.append((eta, LAYER_ANCHOR))
        elif a_deg - 1e-12 <= eta <= b_deg + 1e-12:
            points.append((eta, LAYER_ANCHOR))
    return _dedupe_angles_deg(points)


# =============================================================================
# Partition table loading and validation
# =============================================================================


def _load_partition_table(path):
    if not os.path.isfile(path):
        raise IOError("partition table does not exist: {0}".format(path))
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("partition table must be a JSON object: {0}".format(path))
    return payload


def _require_finite_float(mapping, key, context):
    if key not in mapping:
        raise ValueError("missing field {0}.{1}".format(context, key))
    value = float(mapping[key])
    if not np.isfinite(value):
        raise ValueError("field {0}.{1} must be finite".format(context, key))
    return value


def _validate_partition_table(table, source_path, tol=1e-10):
    if table.get("version") != PARTITION_FORMAT_VERSION:
        raise ValueError(
            "partition table version must be {0}; got {1}".format(
                PARTITION_FORMAT_VERSION, table.get("version")
            )
        )

    quality = table.get("quality_control")
    if not isinstance(quality, dict):
        raise ValueError("partition table quality_control must be an object")
    quality_status = str(quality.get("status", ""))
    if quality_status not in ("PASS", "WARNING"):
        raise ValueError(
            "partition table quality status must be PASS or WARNING; got {0}".format(
                quality_status
            )
        )
    quality_warnings = quality.get("warnings", [])
    if not isinstance(quality_warnings, list):
        raise ValueError("partition table quality_control.warnings must be a list")

    coordinate = table.get("coordinate")
    if not isinstance(coordinate, dict):
        raise ValueError("partition table coordinate must be an object")
    if coordinate.get("name") != "intrinsic_spectral_radius":
        raise ValueError("unsupported partition coordinate.name")
    if coordinate.get("rho_definition") != "rho = sin(alpha_intrinsic)":
        raise ValueError("unsupported partition rho_definition")
    if bool(coordinate.get("scan_angle_included", True)):
        raise ValueError("partition boundary must not include scan angle")

    definition = table.get("definition")
    if not isinstance(definition, dict):
        raise ValueError("partition table definition must be an object")
    alpha_outer_definition = _require_finite_float(
        definition, "alpha_outer_start_deg", "definition"
    )
    if abs(alpha_outer_definition - 90.0) > float(tol):
        raise ValueError("partition alpha_outer_start_deg must equal 90 deg")
    core_text = str(definition.get("core_1d", ""))
    buffer_text = str(definition.get("buffer_2d", ""))
    if "alpha_core_used_deg" not in core_text or "alpha_intrinsic" not in core_text:
        raise ValueError("partition core_1d definition is incompatible")
    if "alpha_core_used_deg" not in buffer_text or "90" not in buffer_text:
        raise ValueError("partition buffer_2d definition is incompatible")

    calibration = table.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("partition table calibration must be an object")
    safety_margin = _require_finite_float(
        calibration, "safety_margin_deg", "calibration"
    )
    alpha_min_allowed = _require_finite_float(
        calibration, "alpha_min_allowed_deg", "calibration"
    )
    xi_core = _require_finite_float(calibration, "xi_core", "calibration")
    peak_gate_db = _require_finite_float(
        calibration, "peak_gate_db", "calibration"
    )
    persistent_rule = calibration.get("persistent_bad_rule")
    if not isinstance(persistent_rule, dict):
        raise ValueError("calibration.persistent_bad_rule must be an object")
    persistent_bad_width = _require_finite_float(
        persistent_rule, "persistent_bad_width_deg", "persistent_bad_rule"
    )
    required_bad_sample_count = int(persistent_rule.get("required_bad_sample_count", 0))
    if persistent_bad_width < 0.0 or required_bad_sample_count < 1:
        raise ValueError("invalid persistent bad rule")
    isolated_bad_policy = str(calibration.get("isolated_bad_policy", ""))
    if isolated_bad_policy not in ("warn", "error", "ignore"):
        raise ValueError("invalid calibration isolated_bad_policy")
    legacy_n_consecutive_bad = int(calibration.get("legacy_n_consecutive_bad", 0))

    entries = table.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("partition table entries must be a non-empty list")

    valid_entries = []
    unique = {}
    for index, entry in enumerate(entries):
        context = "entries[{0}]".format(index)
        if not isinstance(entry, dict):
            raise ValueError("{0} must be an object".format(context))
        if "K" not in entry:
            raise ValueError("missing field {0}.K".format(context))
        K = int(entry["K"])
        if K < 1:
            raise ValueError("{0}.K must be >= 1".format(context))
        d_min = _require_finite_float(entry, "d_min", context)
        d_eff = _require_finite_float(entry, "d_eff", context)
        R0 = _require_finite_float(entry, "R0", context)
        alpha_calib = _require_finite_float(entry, "alpha_core_calib_deg", context)
        alpha_used = _require_finite_float(entry, "alpha_core_used_deg", context)
        rho_used = _require_finite_float(entry, "rho_core_used", context)
        alpha_outer = _require_finite_float(entry, "alpha_outer_start_deg", context)
        worst_b = _require_finite_float(entry, "worst_b", context)

        if d_min <= 0.0 or d_eff <= 0.0 or R0 <= 0.0:
            raise ValueError("{0} contains non-positive geometry".format(context))
        if not (0.0 <= alpha_used <= alpha_calib < 90.0):
            raise ValueError("{0} has invalid calibrated/used boundary".format(context))
        if abs(alpha_outer - alpha_outer_definition) > float(tol):
            raise ValueError("{0} alpha_outer_start_deg is inconsistent".format(context))
        rho_expected = math.sin(math.radians(alpha_used))
        if abs(rho_used - rho_expected) > float(tol):
            raise ValueError("{0} rho_core_used is inconsistent".format(context))
        expected_used = math.floor(alpha_calib) - safety_margin
        expected_used = max(alpha_min_allowed, min(alpha_outer_definition, expected_used))
        if abs(alpha_used - expected_used) > float(tol):
            raise ValueError("{0} does not satisfy partition safety rule".format(context))

        key = (K, round(d_min, 12), round(d_eff, 12), round(R0, 12))
        if key in unique:
            previous = unique[key]
            if (
                    abs(previous["alpha_core_used_deg"] - alpha_used) > float(tol)
                    or abs(previous["alpha_core_calib_deg"] - alpha_calib) > float(tol)):
                raise ValueError("duplicate partition key has conflicting boundaries: {0}".format(key))
            raise ValueError("duplicate partition key is not allowed: {0}".format(key))

        normalized = dict(entry)
        normalized.update({
            "K": K,
            "d_min": d_min,
            "d_eff": d_eff,
            "R0": R0,
            "alpha_core_calib_deg": alpha_calib,
            "alpha_core_used_deg": alpha_used,
            "rho_core_used": rho_used,
            "alpha_outer_start_deg": alpha_outer,
            "worst_b": worst_b,
        })
        unique[key] = normalized
        valid_entries.append(normalized)

    return {
        "source_json": os.path.abspath(source_path),
        "source_sha256": _sha256_file(source_path),
        "table_version": str(table["version"]),
        "quality_status": quality_status,
        "quality_warnings": list(quality_warnings),
        "entries": valid_entries,
        "calibration": {
            "xi_core": xi_core,
            "peak_gate_db": peak_gate_db,
            "persistent_bad_width_deg": persistent_bad_width,
            "required_bad_sample_count": required_bad_sample_count,
            "isolated_bad_policy": isolated_bad_policy,
            "safety_margin_deg": safety_margin,
            "legacy_n_consecutive_bad": legacy_n_consecutive_bad,
        },
    }


def _lookup_partition_entry(entries, K, d_min, d_eff, R0, tol):
    matches = []
    for entry in entries:
        if int(entry["K"]) != int(K):
            continue
        if abs(float(entry["d_min"]) - float(d_min)) > float(tol):
            continue
        if abs(float(entry["d_eff"]) - float(d_eff)) > float(tol):
            continue
        if abs(float(entry["R0"]) - float(R0)) > float(tol):
            continue
        matches.append(entry)
    if len(matches) != 1:
        raise KeyError(
            "Expected one partition entry for K={0}, d_min={1}, d_eff={2}, R0={3}; found {4}. "
            "Lookup is discrete; no interpolation is performed.".format(
                K, d_min, d_eff, R0, len(matches)
            )
        )
    return matches[0]


def _select_partition_entry(validated, cfg):
    mode = str(cfg.partition_selection_mode)
    entries = list(validated["entries"])
    if mode == "global_min":
        ordered = sorted(entries, key=lambda x: (
            float(x["alpha_core_used_deg"]),
            float(x["alpha_core_calib_deg"]),
            int(x["K"]),
            float(x["d_min"]),
            float(x["worst_b"]),
        ))
        return ordered[0]
    if mode == "entry":
        return _lookup_partition_entry(
            entries,
            K=cfg.partition_lookup_K,
            d_min=cfg.partition_lookup_d_min,
            d_eff=cfg.partition_lookup_d_eff,
            R0=cfg.partition_lookup_R0,
            tol=cfg.partition_lookup_tol,
        )
    raise ValueError("partition_selection_mode must be global_min or entry")


def _generate_buffer_ring_list_from_core(
        alpha_core_deg,
        alpha_outer_start_deg=90.0,
        first_offset_deg=3.0,
        step_deg=6.0):
    if float(step_deg) <= 0.0:
        raise ValueError("buffer_ring_step_deg must be > 0")
    if float(first_offset_deg) <= 0.0:
        raise ValueError("buffer_first_offset_deg must be > 0")
    if not float(alpha_core_deg) < float(alpha_outer_start_deg):
        raise ValueError("alpha_core_deg must be below alpha_outer_start_deg")

    rings = []
    value = float(alpha_core_deg) + float(first_offset_deg)
    while value < float(alpha_outer_start_deg) - 1e-12:
        rings.append(float(round(value, 12)))
        value += float(step_deg)
    rings.append(float(alpha_outer_start_deg))
    return tuple(sorted(set(float(round(x, 12)) for x in rings)))


def _validate_partition_config(cfg):
    if cfg.partition_selection_mode not in ("global_min", "entry"):
        raise ValueError("partition_selection_mode must be global_min or entry")
    if cfg.partition_lookup_tol < 0.0:
        raise ValueError("partition_lookup_tol must be >= 0")
    if not (0.0 <= cfg.alpha_core_max_deg < cfg.alpha_outer_start_deg):
        raise ValueError("alpha_core_max_deg must satisfy 0 <= core < outer start")
    if not (0.0 < cfg.alpha_outer_start_deg <= 90.0 + 1e-9):
        raise ValueError("alpha_outer_start_deg must be in (0, 90]")
    if cfg.alpha_main_guard_deg >= cfg.alpha_core_max_deg:
        raise ValueError("alpha_main_guard_deg must be smaller than alpha_core_max_deg")
    if cfg.alpha_1d_step_deg <= 0.0:
        raise ValueError("alpha_1d_step_deg must be > 0")
    if cfg.theta_s_full_step_deg <= 0.0:
        raise ValueError("theta_s_full_step_deg must be > 0")
    if cfg.theta_s_full_max_deg < cfg.theta_s_full_min_deg:
        raise ValueError("invalid full scan range")
    if cfg.theta_s_full_min_deg < 0.0 or cfg.theta_s_full_max_deg > 90.0 + 1e-9:
        raise ValueError("full scan range must be within [0, 90]")
    if cfg.outer_ring_step_deg <= 0.0:
        raise ValueError("outer_ring_step_deg must be > 0")
    if cfg.outer_ring_start_deg <= cfg.alpha_outer_start_deg:
        raise ValueError("outer_ring_start_deg must be above alpha_outer_start_deg")
    if cfg.budget_total_per_scan < 1:
        raise ValueError("budget_total_per_scan must be >= 1")
    if len(cfg.buffer_ring_list_deg) == 0:
        raise ValueError("buffer_ring_list_deg cannot be empty")
    buffer_rings = np.asarray(cfg.buffer_ring_list_deg, dtype=np.float64)
    if np.any(~np.isfinite(buffer_rings)):
        raise ValueError("buffer_ring_list_deg contains non-finite values")
    if np.any(np.diff(buffer_rings) <= 0.0):
        raise ValueError("buffer rings must be strictly increasing and unique")
    if np.any(buffer_rings <= cfg.alpha_core_max_deg + 1e-12):
        raise ValueError("buffer rings must be strictly above alpha_core_max_deg")
    if np.any(buffer_rings > cfg.alpha_outer_start_deg + 1e-9):
        raise ValueError("buffer rings must not exceed alpha_outer_start_deg")
    if abs(float(buffer_rings[-1]) - cfg.alpha_outer_start_deg) > 1e-9:
        raise ValueError("buffer rings must include alpha_outer_start_deg")
    if bool(cfg.auto_buffer_rings):
        expected_first = cfg.alpha_core_max_deg + cfg.buffer_first_offset_deg
        if expected_first < cfg.alpha_outer_start_deg - 1e-12:
            if abs(float(buffer_rings[0]) - expected_first) > 1e-9:
                raise ValueError("first auto buffer ring is inconsistent")


def _resolve_partition_and_buffer_rings(cfg):
    if str(cfg.partition_table_json).strip():
        source = os.path.abspath(cfg.partition_table_json)
        table = _load_partition_table(source)
        validated = _validate_partition_table(table, source)
        entry = _select_partition_entry(validated, cfg)
        calibration = validated["calibration"]
        alpha_core = float(entry["alpha_core_used_deg"])
        alpha_outer = float(entry["alpha_outer_start_deg"])
        partition_info = PartitionInfo(
            mode="table",
            selection_mode=cfg.partition_selection_mode,
            source_json=validated["source_json"],
            source_sha256=validated["source_sha256"],
            table_version=validated["table_version"],
            table_quality_status=validated["quality_status"],
            table_warnings=validated["quality_warnings"],
            K=entry["K"],
            d_min=entry["d_min"],
            d_eff=entry["d_eff"],
            R0=entry["R0"],
            alpha_core_calib_deg=entry["alpha_core_calib_deg"],
            alpha_core_used_deg=alpha_core,
            rho_core_used=entry["rho_core_used"],
            alpha_outer_start_deg=alpha_outer,
            worst_b=entry["worst_b"],
            xi_core=calibration["xi_core"],
            peak_gate_db=calibration["peak_gate_db"],
            n_consecutive_bad=calibration["legacy_n_consecutive_bad"],
            persistent_bad_width_deg=calibration["persistent_bad_width_deg"],
            required_bad_sample_count=calibration["required_bad_sample_count"],
            isolated_bad_policy=calibration["isolated_bad_policy"],
            safety_margin_deg=calibration["safety_margin_deg"],
        )
    else:
        if cfg.require_partition_table or not cfg.allow_manual_partition:
            raise ValueError(
                "No partition table was provided. Use --partition-table-json, or explicitly enable "
                "manual mode with --allow-manual-partition."
            )
        alpha_core = float(cfg.alpha_core_max_deg)
        alpha_outer = float(cfg.alpha_outer_start_deg)
        partition_info = PartitionInfo(
            mode="manual",
            selection_mode="manual",
            alpha_core_calib_deg=alpha_core,
            alpha_core_used_deg=alpha_core,
            rho_core_used=float(math.sin(math.radians(alpha_core))),
            alpha_outer_start_deg=alpha_outer,
        )

    if cfg.auto_buffer_rings:
        buffer_rings = _generate_buffer_ring_list_from_core(
            alpha_core,
            alpha_outer_start_deg=alpha_outer,
            first_offset_deg=cfg.buffer_first_offset_deg,
            step_deg=cfg.buffer_ring_step_deg,
        )
    else:
        buffer_rings = tuple(float(x) for x in cfg.buffer_ring_list_deg)

    effective = cfg.copy_with(
        alpha_core_max_deg=alpha_core,
        alpha_outer_start_deg=alpha_outer,
        buffer_ring_list_deg=buffer_rings,
    )
    _validate_partition_config(effective)
    return effective, partition_info


# =============================================================================
# Pool generation
# =============================================================================


def _generate_core_for_scan(scan_idx, theta_s_deg, cfg, batch):
    alpha_grid = _closed_float_grid(0.0, cfg.alpha_core_max_deg, cfg.alpha_1d_step_deg)
    for alpha_deg in alpha_grid.tolist():
        batch.append(
            scan_idx=scan_idx,
            theta_s_deg=theta_s_deg,
            alpha_deg=float(alpha_deg),
            alpha_main_guard_deg=cfg.alpha_main_guard_deg,
        )


def _generate_buffer_for_scan(scan_idx, theta_s_deg, cfg, batch):
    for alpha_deg in cfg.buffer_ring_list_deg:
        samples = _generate_buffer_eta_samples(float(alpha_deg), float(theta_s_deg), cfg)
        for eta_deg, layer_code in samples:
            theta_deg, phi_deg, u, v = _local_to_global(
                float(theta_s_deg), cfg.phi_s_deg,
                float(alpha_deg), float(eta_deg),
            )
            if theta_deg <= 90.0 + 1e-9:
                batch.append(
                    scan_idx, theta_s_deg, alpha_deg, eta_deg,
                    theta_deg, phi_deg, u, v,
                    ZONE_BUFFER_2D, layer_code,
                )


def _generate_outer_for_scan(scan_idx, theta_s_deg, cfg, batch):
    for alpha_deg in _generate_outer_ring_list(float(theta_s_deg), cfg).tolist():
        samples = _generate_outer_eta_samples(float(alpha_deg), float(theta_s_deg), cfg)
        for eta_deg, layer_code in samples:
            theta_deg, phi_deg, u, v = _local_to_global(
                float(theta_s_deg), cfg.phi_s_deg,
                float(alpha_deg), float(eta_deg),
            )
            if theta_deg <= 90.0 + 1e-9:
                batch.append(
                    scan_idx, theta_s_deg, alpha_deg, eta_deg,
                    theta_deg, phi_deg, u, v,
                    ZONE_OUTER_2D, layer_code,
                )


def _resolve_work_scan_indices(theta_s_full_deg, theta_s_work_deg, tol=1e-9):
    full = np.asarray(theta_s_full_deg, dtype=np.float64).reshape(-1)
    work = np.asarray(theta_s_work_deg, dtype=np.float64).reshape(-1)
    if full.size == 0 or work.size == 0:
        raise ValueError("full and work scan-angle lists must be non-empty")
    if np.any(~np.isfinite(full)) or np.any(~np.isfinite(work)):
        raise ValueError("scan-angle lists contain non-finite values")
    if np.any(np.diff(full) <= 0.0):
        raise ValueError("full scan-angle grid must be strictly increasing")
    if np.unique(np.round(work, 12)).size != work.size:
        raise ValueError("work scan-angle list contains duplicates")

    indices = []
    for value in work.tolist():
        matches = np.flatnonzero(np.abs(full - float(value)) <= float(tol))
        if matches.size != 1:
            raise ValueError(
                "work scan angle {0} deg must match exactly one full-grid angle; found {1}".format(
                    value, matches.size
                )
            )
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int32)


def _filter_1d_by_scan_indices(batch, scan_indices):
    selected = set(int(x) for x in np.asarray(scan_indices, dtype=np.int32).tolist())
    out = PointBatch1D.empty()
    for i in range(len(batch.scan_idx)):
        if int(batch.scan_idx[i]) in selected:
            out.scan_idx.append(batch.scan_idx[i])
            out.theta_s_deg.append(batch.theta_s_deg[i])
            out.alpha_deg.append(batch.alpha_deg[i])
            out.rho.append(batch.rho[i])
            out.theta_min_deg.append(batch.theta_min_deg[i])
            out.theta_max_deg.append(batch.theta_max_deg[i])
            out.zone_code.append(batch.zone_code[i])
            out.is_main_guard.append(batch.is_main_guard[i])
            out.is_side_1d.append(batch.is_side_1d[i])
    return out


def _filter_2d_by_scan_indices(batch, scan_indices):
    selected = set(int(x) for x in np.asarray(scan_indices, dtype=np.int32).tolist())
    out = PointBatch2D.empty()
    for i in range(len(batch.scan_idx)):
        if int(batch.scan_idx[i]) in selected:
            out.scan_idx.append(batch.scan_idx[i])
            out.theta_s_deg.append(batch.theta_s_deg[i])
            out.alpha_deg.append(batch.alpha_deg[i])
            out.eta_deg.append(batch.eta_deg[i])
            out.theta_deg.append(batch.theta_deg[i])
            out.phi_deg.append(batch.phi_deg[i])
            out.u.append(batch.u[i])
            out.v.append(batch.v[i])
            out.zone_code.append(batch.zone_code[i])
            out.layer_code.append(batch.layer_code[i])
            out.is_side_2d.append(batch.is_side_2d[i])
    return out


def _build_counts(theta_s_full_deg, core, buffer_batch, outer):
    out = []
    core_scan = np.asarray(core.scan_idx, dtype=np.int32)
    core_main = np.asarray(core.is_main_guard, dtype=np.int8)
    core_side = np.asarray(core.is_side_1d, dtype=np.int8)
    buffer_scan = np.asarray(buffer_batch.scan_idx, dtype=np.int32)
    buffer_side = np.asarray(buffer_batch.is_side_2d, dtype=np.int8)
    outer_scan = np.asarray(outer.scan_idx, dtype=np.int32)
    outer_side = np.asarray(outer.is_side_2d, dtype=np.int8)
    for scan_idx, theta_s_deg in enumerate(np.asarray(theta_s_full_deg).tolist()):
        n_core = int(np.sum(core_scan == scan_idx))
        n_main = int(np.sum((core_scan == scan_idx) & (core_main == 1)))
        n_side = int(np.sum((core_scan == scan_idx) & (core_side == 1)))
        n_buffer = int(np.sum(buffer_scan == scan_idx))
        n_buffer_side = int(np.sum((buffer_scan == scan_idx) & (buffer_side == 1)))
        n_outer = int(np.sum(outer_scan == scan_idx))
        n_outer_side = int(np.sum((outer_scan == scan_idx) & (outer_side == 1)))
        out.append({
            "scan_idx": int(scan_idx),
            "theta_s_deg": float(theta_s_deg),
            "n_core_1d": n_core,
            "n_core_main_guard": n_main,
            "n_core_side_1d": n_side,
            "n_buffer_2d": n_buffer,
            "n_buffer_side_2d": n_buffer_side,
            "n_outer_2d": n_outer,
            "n_outer_side_2d": n_outer_side,
            "n_total": n_core + n_buffer + n_outer,
        })
    return out


def _validate_judging_masks(core, cfg):
    arrays = core.to_dict()
    alpha = arrays["alpha_deg"]
    is_main = arrays["is_main_guard"]
    is_side = arrays["is_side_1d"]
    if alpha.size == 0:
        raise ValueError("core_1d is empty")
    if not np.all((is_main + is_side) == 1):
        raise ValueError("core main/side masks must be complementary")
    if np.sum(is_main) == 0:
        raise ValueError("alpha_main_guard_deg is too small: no main-guard points")
    if np.sum(is_side) == 0:
        raise ValueError("alpha_main_guard_deg is too large: no side-1d points")
    if np.any(alpha[is_main.astype(bool)] > cfg.alpha_main_guard_deg + 1e-9):
        raise ValueError("core main-guard mask is inconsistent")
    if np.any(alpha[is_side.astype(bool)] <= cfg.alpha_main_guard_deg - 1e-9):
        raise ValueError("core side-1d mask is inconsistent")


def _point_key_1d(arrays, index):
    return (
        int(arrays["scan_idx"][index]),
        round(float(arrays["theta_s_deg"][index]), 10),
        round(float(arrays["alpha_deg"][index]), 10),
        int(arrays["zone_code"][index]),
        int(arrays["is_main_guard"][index]),
        int(arrays["is_side_1d"][index]),
    )


def _point_key_2d(arrays, index):
    return (
        int(arrays["scan_idx"][index]),
        round(float(arrays["theta_s_deg"][index]), 10),
        round(float(arrays["alpha_deg"][index]), 10),
        round(float(arrays["eta_deg"][index]) % 360.0, 10),
        int(arrays["zone_code"][index]),
        int(arrays["layer_code"][index]),
        int(arrays["is_side_2d"][index]),
    )


def _validate_batch_1d(batch, cfg, work_scan_indices=None):
    arrays = batch.to_dict()
    n = arrays["scan_idx"].size
    for value in arrays.values():
        if value.size != n:
            raise ValueError("1D batch arrays have inconsistent lengths")
    if n == 0:
        raise ValueError("1D batch is empty")
    if np.any(~np.isfinite(arrays["theta_s_deg"])) or np.any(~np.isfinite(arrays["alpha_deg"])):
        raise ValueError("1D batch contains non-finite angles")
    if np.any(arrays["alpha_deg"] < -1e-12) or np.any(arrays["alpha_deg"] > cfg.alpha_core_max_deg + 1e-9):
        raise ValueError("1D alpha is outside the core domain")
    if np.any(arrays["zone_code"] != ZONE_CORE_1D):
        raise ValueError("1D zone code is invalid")
    if not np.allclose(arrays["rho"], np.sin(np.deg2rad(arrays["alpha_deg"])), atol=1e-12, rtol=0.0):
        raise ValueError("1D rho is inconsistent")
    theta_min_expected = np.abs(arrays["theta_s_deg"] - arrays["alpha_deg"])
    theta_max_expected = np.minimum(arrays["theta_s_deg"] + arrays["alpha_deg"], 90.0)
    if not np.allclose(arrays["theta_min_deg"], theta_min_expected, atol=1e-12, rtol=0.0):
        raise ValueError("1D theta_min_deg is inconsistent")
    if not np.allclose(arrays["theta_max_deg"], theta_max_expected, atol=1e-12, rtol=0.0):
        raise ValueError("1D theta_max_deg is inconsistent")
    _validate_judging_masks(batch, cfg)
    if work_scan_indices is not None:
        allowed = set(int(x) for x in np.asarray(work_scan_indices, dtype=np.int32).tolist())
        if any(int(x) not in allowed for x in arrays["scan_idx"].tolist()):
            raise ValueError("work 1D batch contains non-work scan index")


def _validate_batch_2d(batch, cfg, expected_zone, work_scan_indices=None):
    arrays = batch.to_dict()
    n = arrays["scan_idx"].size
    for value in arrays.values():
        if value.size != n:
            raise ValueError("2D batch arrays have inconsistent lengths")
    if n == 0 and expected_zone == ZONE_BUFFER_2D:
        raise ValueError("buffer_2d batch is empty")
    if np.any(~np.isfinite(arrays["theta_s_deg"])) or np.any(~np.isfinite(arrays["alpha_deg"])):
        raise ValueError("2D batch contains non-finite local angles")
    if np.any(~np.isfinite(arrays["theta_deg"])) or np.any(~np.isfinite(arrays["phi_deg"])):
        raise ValueError("2D batch contains non-finite global angles")
    if np.any(arrays["zone_code"] != int(expected_zone)):
        raise ValueError("2D zone code is invalid")
    if np.any(arrays["is_side_2d"] != 1):
        raise ValueError("2D side mask is invalid")
    if expected_zone == ZONE_BUFFER_2D:
        if np.any(arrays["alpha_deg"] <= cfg.alpha_core_max_deg + 1e-12):
            raise ValueError("buffer alpha must be strictly above the core boundary")
        if np.any(arrays["alpha_deg"] > cfg.alpha_outer_start_deg + 1e-9):
            raise ValueError("buffer alpha exceeds the outer boundary")
    else:
        if np.any(arrays["alpha_deg"] <= cfg.alpha_outer_start_deg + 1e-12):
            raise ValueError("outer alpha must be strictly above alpha_outer_start_deg")
        if np.any(arrays["alpha_deg"] > cfg.alpha_outer_start_deg + arrays["theta_s_deg"] + 1e-9):
            raise ValueError("outer alpha exceeds visible local-domain limit")
    if np.any(arrays["theta_deg"] < -1e-10) or np.any(arrays["theta_deg"] > 90.0 + 1e-9):
        raise ValueError("2D theta is outside the visible hemisphere")
    if np.any(arrays["phi_deg"] < -1e-10) or np.any(arrays["phi_deg"] >= 360.0 + 1e-9):
        raise ValueError("2D phi is outside [0, 360)")
    u_expected = np.sin(np.deg2rad(arrays["theta_deg"])) * np.cos(np.deg2rad(arrays["phi_deg"]))
    v_expected = np.sin(np.deg2rad(arrays["theta_deg"])) * np.sin(np.deg2rad(arrays["phi_deg"]))
    if not np.allclose(arrays["u"], u_expected, atol=1e-10, rtol=0.0):
        raise ValueError("2D u is inconsistent")
    if not np.allclose(arrays["v"], v_expected, atol=1e-10, rtol=0.0):
        raise ValueError("2D v is inconsistent")
    if np.any(arrays["u"] ** 2 + arrays["v"] ** 2 > 1.0 + 1e-10):
        raise ValueError("2D uv coordinate lies outside the visible unit disk")
    simple_keys = [
        (
            int(arrays["scan_idx"][i]),
            round(float(arrays["alpha_deg"][i]), 8),
            round(float(arrays["eta_deg"][i]) % 360.0, 8),
        )
        for i in range(n)
    ]
    if len(simple_keys) != len(set(simple_keys)):
        raise ValueError("2D batch contains duplicate scan/alpha/eta points")
    if work_scan_indices is not None:
        allowed = set(int(x) for x in np.asarray(work_scan_indices, dtype=np.int32).tolist())
        if any(int(x) not in allowed for x in arrays["scan_idx"].tolist()):
            raise ValueError("work 2D batch contains non-work scan index")


def _validate_work_subset(full_batch, work_batch, kind):
    full_arrays = full_batch.to_dict()
    work_arrays = work_batch.to_dict()
    if kind == "1d":
        full_keys = set(_point_key_1d(full_arrays, i) for i in range(full_arrays["scan_idx"].size))
        work_keys = [_point_key_1d(work_arrays, i) for i in range(work_arrays["scan_idx"].size)]
    else:
        full_keys = set(_point_key_2d(full_arrays, i) for i in range(full_arrays["scan_idx"].size))
        work_keys = [_point_key_2d(work_arrays, i) for i in range(work_arrays["scan_idx"].size)]
    if any(key not in full_keys for key in work_keys):
        raise ValueError("work pool is not an exact subset of full pool")


def _validate_counts(theta_s_full_deg, core, buffer_batch, outer, counts):
    expected = _build_counts(theta_s_full_deg, core, buffer_batch, outer)
    if len(expected) != len(counts):
        raise ValueError("count table length is inconsistent")
    fields = (
        "scan_idx", "theta_s_deg", "n_core_1d", "n_core_main_guard",
        "n_core_side_1d", "n_buffer_2d", "n_buffer_side_2d",
        "n_outer_2d", "n_outer_side_2d", "n_total",
    )
    for left, right in zip(expected, counts):
        for field in fields:
            if field == "theta_s_deg":
                if abs(float(left[field]) - float(right[field])) > 1e-12:
                    raise ValueError("count table theta_s_deg is inconsistent")
            elif int(left[field]) != int(right[field]):
                raise ValueError("count table field {0} is inconsistent".format(field))


def _validate_outer_scan_coverage(package):
    arrays = package.outer_full.to_dict()
    for scan_idx, theta_s in enumerate(package.theta_s_full_deg.tolist()):
        mask = arrays["scan_idx"] == int(scan_idx)
        if float(theta_s) <= 0.0:
            if np.any(mask):
                raise ValueError("outer pool must be empty at theta_s=0")
        else:
            if not np.any(mask):
                raise ValueError("outer pool must be non-empty for theta_s>0")
            alpha_max = package.config.alpha_outer_start_deg + float(theta_s)
            if np.max(arrays["alpha_deg"][mask]) > alpha_max + 1e-9:
                raise ValueError("outer scan coverage exceeds alpha maximum")


def _validate_pool_package(package):
    _validate_batch_1d(package.core_full, package.config)
    _validate_batch_1d(
        package.core_work, package.config,
        work_scan_indices=package.theta_s_work_full_scan_idx,
    )
    _validate_batch_2d(package.buffer_full, package.config, ZONE_BUFFER_2D)
    _validate_batch_2d(
        package.buffer_work, package.config, ZONE_BUFFER_2D,
        work_scan_indices=package.theta_s_work_full_scan_idx,
    )
    _validate_batch_2d(package.outer_full, package.config, ZONE_OUTER_2D)
    _validate_batch_2d(
        package.outer_work, package.config, ZONE_OUTER_2D,
        work_scan_indices=package.theta_s_work_full_scan_idx,
    )
    _validate_work_subset(package.core_full, package.core_work, "1d")
    _validate_work_subset(package.buffer_full, package.buffer_work, "2d")
    _validate_work_subset(package.outer_full, package.outer_work, "2d")
    _validate_counts(
        package.theta_s_full_deg,
        package.core_full, package.buffer_full, package.outer_full,
        package.counts_full,
    )
    selected = set(int(x) for x in package.theta_s_work_full_scan_idx.tolist())
    expected_work_counts = [row for row in package.counts_full if int(row["scan_idx"]) in selected]
    if expected_work_counts != package.counts_work:
        raise ValueError("work count table is inconsistent")
    _validate_outer_scan_coverage(package)

    return {
        "status": "PASS",
        "core_full_points": int(len(package.core_full.scan_idx)),
        "buffer_full_points": int(len(package.buffer_full.scan_idx)),
        "outer_full_points": int(len(package.outer_full.scan_idx)),
        "core_work_points": int(len(package.core_work.scan_idx)),
        "buffer_work_points": int(len(package.buffer_work.scan_idx)),
        "outer_work_points": int(len(package.outer_work.scan_idx)),
        "work_is_exact_subset": True,
        "geometry_consistent": True,
        "counts_consistent": True,
    }


def build_stage1_static_pool(cfg):
    effective_cfg, partition_info = _resolve_partition_and_buffer_rings(cfg)
    theta_s_full_deg = _float_range(
        effective_cfg.theta_s_full_min_deg,
        effective_cfg.theta_s_full_max_deg,
        effective_cfg.theta_s_full_step_deg,
    )
    theta_s_work_init_deg = np.asarray(
        sorted(set(float(x) for x in effective_cfg.theta_s_work_init_deg)),
        dtype=np.float64,
    )
    theta_s_work_full_scan_idx = _resolve_work_scan_indices(
        theta_s_full_deg, theta_s_work_init_deg
    )

    core_full = PointBatch1D.empty()
    buffer_full = PointBatch2D.empty()
    outer_full = PointBatch2D.empty()

    for scan_idx, theta_s_deg in enumerate(theta_s_full_deg.tolist()):
        _generate_core_for_scan(scan_idx, theta_s_deg, effective_cfg, core_full)
        _generate_buffer_for_scan(scan_idx, theta_s_deg, effective_cfg, buffer_full)
        _generate_outer_for_scan(scan_idx, theta_s_deg, effective_cfg, outer_full)

    core_work = _filter_1d_by_scan_indices(core_full, theta_s_work_full_scan_idx)
    buffer_work = _filter_2d_by_scan_indices(buffer_full, theta_s_work_full_scan_idx)
    outer_work = _filter_2d_by_scan_indices(outer_full, theta_s_work_full_scan_idx)

    counts_full = _build_counts(theta_s_full_deg, core_full, buffer_full, outer_full)
    selected = set(int(x) for x in theta_s_work_full_scan_idx.tolist())
    counts_work = [row for row in counts_full if int(row["scan_idx"]) in selected]

    package = PoolPackageStage1(
        config=effective_cfg,
        partition_info=partition_info,
        theta_s_full_deg=theta_s_full_deg,
        theta_s_work_init_deg=theta_s_work_init_deg,
        theta_s_work_full_scan_idx=theta_s_work_full_scan_idx,
        core_full=core_full,
        core_work=core_work,
        buffer_full=buffer_full,
        buffer_work=buffer_work,
        outer_full=outer_full,
        outer_work=outer_work,
        counts_full=counts_full,
        counts_work=counts_work,
        validation_summary={},
    )
    package.validation_summary = _validate_pool_package(package)
    return package


# =============================================================================
# Writers
# =============================================================================


def _summary_stats(counts, budget):
    total = np.asarray([int(x["n_total"]) for x in counts], dtype=np.int64)
    core = np.asarray([int(x["n_core_1d"]) for x in counts], dtype=np.int64)
    buffer_count = np.asarray([int(x["n_buffer_2d"]) for x in counts], dtype=np.int64)
    outer = np.asarray([int(x["n_outer_2d"]) for x in counts], dtype=np.int64)
    main = np.asarray([int(x.get("n_core_main_guard", 0)) for x in counts], dtype=np.int64)
    side1d = np.asarray([int(x.get("n_core_side_1d", 0)) for x in counts], dtype=np.int64)
    buffer_side = np.asarray([int(x.get("n_buffer_side_2d", 0)) for x in counts], dtype=np.int64)
    outer_side = np.asarray([int(x.get("n_outer_side_2d", 0)) for x in counts], dtype=np.int64)
    return {
        "n_total_min": int(np.min(total)) if total.size else 0,
        "n_total_mean": float(np.mean(total)) if total.size else 0.0,
        "n_total_max": int(np.max(total)) if total.size else 0,
        "n_core_min": int(np.min(core)) if core.size else 0,
        "n_core_max": int(np.max(core)) if core.size else 0,
        "n_core_main_guard_min": int(np.min(main)) if main.size else 0,
        "n_core_main_guard_max": int(np.max(main)) if main.size else 0,
        "n_core_side_1d_min": int(np.min(side1d)) if side1d.size else 0,
        "n_core_side_1d_max": int(np.max(side1d)) if side1d.size else 0,
        "n_buffer_min": int(np.min(buffer_count)) if buffer_count.size else 0,
        "n_buffer_max": int(np.max(buffer_count)) if buffer_count.size else 0,
        "n_buffer_side_2d_min": int(np.min(buffer_side)) if buffer_side.size else 0,
        "n_buffer_side_2d_max": int(np.max(buffer_side)) if buffer_side.size else 0,
        "n_outer_min": int(np.min(outer)) if outer.size else 0,
        "n_outer_max": int(np.max(outer)) if outer.size else 0,
        "n_outer_side_2d_min": int(np.min(outer_side)) if outer_side.size else 0,
        "n_outer_side_2d_max": int(np.max(outer_side)) if outer_side.size else 0,
        "budget": int(budget),
        "budget_ok": bool(np.all(total <= int(budget))) if total.size else True,
    }


def _make_npz_payload(package):
    core_full = package.core_full.to_dict()
    core_work = package.core_work.to_dict()
    buffer_full = package.buffer_full.to_dict()
    buffer_work = package.buffer_work.to_dict()
    outer_full = package.outer_full.to_dict()
    outer_work = package.outer_work.to_dict()
    pinfo = package.partition_info

    payload = {
        "format_version": np.asarray(FORMAT_VERSION),
        "theta_s_full_deg": package.theta_s_full_deg.astype(np.float64),
        "theta_s_work_init_deg": package.theta_s_work_init_deg.astype(np.float64),
        "theta_s_work_full_scan_idx": package.theta_s_work_full_scan_idx.astype(np.int32),
        "alpha_core_max_deg": np.asarray(package.config.alpha_core_max_deg, dtype=np.float64),
        "alpha_outer_start_deg": np.asarray(package.config.alpha_outer_start_deg, dtype=np.float64),
        "buffer_ring_list_deg": np.asarray(package.config.buffer_ring_list_deg, dtype=np.float64),
        "partition_table_version": np.asarray(pinfo.table_version),
        "partition_selection_mode": np.asarray(pinfo.selection_mode),
        "partition_source_sha256": np.asarray(pinfo.source_sha256),
        "partition_quality_status": np.asarray(pinfo.table_quality_status),
        "partition_K": np.asarray(pinfo.K, dtype=np.int32),
        "partition_d_min": np.asarray(pinfo.d_min, dtype=np.float64),
        "partition_d_eff": np.asarray(pinfo.d_eff, dtype=np.float64),
        "partition_R0": np.asarray(pinfo.R0, dtype=np.float64),
        "partition_selected_K": np.asarray(pinfo.K, dtype=np.int32),
        "partition_selected_d_min": np.asarray(pinfo.d_min, dtype=np.float64),
        "partition_selected_d_eff": np.asarray(pinfo.d_eff, dtype=np.float64),
        "partition_selected_R0": np.asarray(pinfo.R0, dtype=np.float64),
        "partition_alpha_core_calib_deg": np.asarray(pinfo.alpha_core_calib_deg, dtype=np.float64),
        "partition_alpha_core_used_deg": np.asarray(pinfo.alpha_core_used_deg, dtype=np.float64),
        "partition_rho_core_used": np.asarray(pinfo.rho_core_used, dtype=np.float64),
        "partition_worst_b": np.asarray(pinfo.worst_b, dtype=np.float64),
        "partition_safety_margin_deg": np.asarray(pinfo.safety_margin_deg, dtype=np.float64),
        "partition_persistent_bad_width_deg": np.asarray(pinfo.persistent_bad_width_deg, dtype=np.float64),
        "partition_required_bad_sample_count": np.asarray(pinfo.required_bad_sample_count, dtype=np.int32),
    }

    for prefix, arrays in (
            ("core_full", core_full),
            ("core_work", core_work),
            ("buffer_full", buffer_full),
            ("buffer_work", buffer_work),
            ("outer_full", outer_full),
            ("outer_work", outer_work)):
        for key, value in arrays.items():
            payload["{0}_{1}".format(prefix, key)] = value
    return payload


def _save_pool(package, out_dir, overwrite=False):
    _ensure_dir(out_dir)
    paths = {
        "npz": os.path.join(out_dir, "q0cra_stage1_static_pool.npz"),
        "summary_json": os.path.join(out_dir, "q0cra_stage1_static_pool_summary.json"),
        "counts_full_csv": os.path.join(out_dir, "q0cra_stage1_static_pool_counts_full.csv"),
        "counts_work_csv": os.path.join(out_dir, "q0cra_stage1_static_pool_counts_work.csv"),
        "log": os.path.join(out_dir, "q0cra_stage1_static_pool.log"),
    }
    _check_output_paths([paths["npz"], paths["summary_json"], paths["counts_full_csv"], paths["counts_work_csv"]], overwrite)

    npz_payload = _make_npz_payload(package)
    stats_full = _summary_stats(package.counts_full, package.config.budget_total_per_scan)
    stats_work = _summary_stats(package.counts_work, package.config.budget_total_per_scan)
    quality_status = "PASS"
    quality_warnings = []
    if package.partition_info.table_quality_status == "WARNING":
        quality_status = "WARNING"
        quality_warnings.extend(package.partition_info.table_warnings)
    if not stats_full["budget_ok"]:
        quality_status = "WARNING"
        quality_warnings.append("Some full-scan point totals exceed budget_total_per_scan")

    summary = {
        "version": FORMAT_VERSION,
        "quality_control": {
            "status": quality_status,
            "warnings": quality_warnings,
        },
        "config": _record_to_dict(package.config),
        "partition_info": _record_to_dict(package.partition_info),
        "partition_definition": {
            "coordinate": "local spherical alpha around scan direction",
            "core_1d": "0 <= alpha <= alpha_core_max_deg",
            "buffer_2d": "alpha_core_max_deg < alpha <= alpha_outer_start_deg",
            "outer_2d": "alpha_outer_start_deg < alpha <= alpha_outer_start_deg + theta_s",
            "alpha_core_max_deg": float(package.config.alpha_core_max_deg),
            "alpha_outer_start_deg": float(package.config.alpha_outer_start_deg),
            "buffer_ring_list_deg": list(package.config.buffer_ring_list_deg),
            "buffer_ring_generation": {
                "auto_buffer_rings": bool(package.config.auto_buffer_rings),
                "first_offset_deg": float(package.config.buffer_first_offset_deg),
                "step_deg": float(package.config.buffer_ring_step_deg),
            },
        },
        "tracking_protocol": _record_to_dict(package.config.tracking),
        "zone_name": ZONE_NAME,
        "layer_name": LAYER_NAME,
        "responsibility_boundary": {
            "pool_script_handles": [
                "geometry",
                "full/work scan-angle pools",
                "1D a0 interval descriptors",
                "judging masks (main guard vs side)",
                "tracking protocol metadata",
            ],
            "pool_script_does_not_handle": [
                "IEP interpolation",
                "a0 numerical evaluation",
                "a0 monotonicity test",
                "dynamic peak tracking execution",
            ],
        },
        "a0_policy_interface": {
            "mono_test_step_deg": package.config.a0_mono_test_step_deg,
            "core_interval_fields": ["theta_min_deg", "theta_max_deg"],
            "main_guard_field": "is_main_guard",
            "core_side_field": "is_side_1d",
            "buffer_side_field": "is_side_2d",
            "outer_side_field": "is_side_2d",
            "note": (
                "Subsequent dominant_mode_envelope code should test monotonicity on 1 deg grid and then use "
                "endpoint-max if monotone decreasing; otherwise sampled interval max via iep_model."
            ),
        },
        "theta_s_work_full_scan_idx": package.theta_s_work_full_scan_idx.tolist(),
        "counts_full": package.counts_full,
        "counts_work": package.counts_work,
        "stats_full": stats_full,
        "stats_work": stats_work,
        "validation": package.validation_summary,
        "outputs": dict(paths),
    }

    _atomic_savez(paths["npz"], npz_payload)
    _atomic_write_json(paths["summary_json"], summary)
    _atomic_write_counts_csv(paths["counts_full_csv"], package.counts_full)
    _atomic_write_counts_csv(paths["counts_work_csv"], package.counts_work)
    return paths, summary


# =============================================================================
# CLI
# =============================================================================


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate static full/work geometry-sampling pools for backbone synthesis."
    )
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--phi-s-deg", type=float, default=0.0)
    parser.add_argument("--alpha-core-max-deg", type=float, default=57.0)
    parser.add_argument("--partition-table-json", type=str, default="")
    parser.add_argument(
        "--partition-selection-mode",
        choices=["global_min", "entry"],
        default="global_min",
    )
    parser.add_argument("--partition-K", type=int, default=8)
    parser.add_argument("--partition-d-min", type=float, default=0.5)
    parser.add_argument("--partition-d-eff", type=float, default=0.5)
    parser.add_argument("--partition-R0", type=float, default=5.0)
    parser.add_argument("--partition-lookup-tol", type=float, default=1e-9)
    parser.add_argument("--require-partition-table", action="store_true")
    parser.add_argument("--allow-manual-partition", action="store_true")
    parser.add_argument("--disable-auto-buffer-rings", action="store_true")
    parser.add_argument("--buffer-first-offset-deg", type=float, default=3.0)
    parser.add_argument("--buffer-ring-step-deg", type=float, default=6.0)
    parser.add_argument("--alpha-outer-start-deg", type=float, default=90.0)
    parser.add_argument("--buffer-ring-list-deg", type=str, default="")
    parser.add_argument("--alpha-1d-step-deg", type=float, default=0.25)
    parser.add_argument("--theta-s-full-min-deg", type=float, default=0.0)
    parser.add_argument("--theta-s-full-max-deg", type=float, default=60.0)
    parser.add_argument("--theta-s-full-step-deg", type=float, default=1.0)
    parser.add_argument("--budget-total-per-scan", type=int, default=1000)
    parser.add_argument("--a0-mono-test-step-deg", type=float, default=1.0)
    parser.add_argument("--alpha-main-guard-deg", type=float, default=7.5)
    return parser.parse_args(argv)


def run_main(argv=None):
    args = _parse_args(argv)
    manual_buffer_rings = _parse_float_tuple_csv(args.buffer_ring_list_deg)
    cfg = PoolConfigStage1(
        phi_s_deg=args.phi_s_deg,
        alpha_core_max_deg=args.alpha_core_max_deg,
        partition_table_json=args.partition_table_json,
        partition_selection_mode=args.partition_selection_mode,
        partition_lookup_K=args.partition_K,
        partition_lookup_d_min=args.partition_d_min,
        partition_lookup_d_eff=args.partition_d_eff,
        partition_lookup_R0=args.partition_R0,
        partition_lookup_tol=args.partition_lookup_tol,
        require_partition_table=args.require_partition_table,
        allow_manual_partition=args.allow_manual_partition,
        auto_buffer_rings=not args.disable_auto_buffer_rings,
        buffer_first_offset_deg=args.buffer_first_offset_deg,
        buffer_ring_step_deg=args.buffer_ring_step_deg,
        alpha_outer_start_deg=args.alpha_outer_start_deg,
        buffer_ring_list_deg=(
            manual_buffer_rings if manual_buffer_rings
            else DEFAULT_BUFFER_RING_LIST_DEG
        ),
        alpha_1d_step_deg=args.alpha_1d_step_deg,
        theta_s_full_min_deg=args.theta_s_full_min_deg,
        theta_s_full_max_deg=args.theta_s_full_max_deg,
        theta_s_full_step_deg=args.theta_s_full_step_deg,
        budget_total_per_scan=args.budget_total_per_scan,
        a0_mono_test_step_deg=args.a0_mono_test_step_deg,
        alpha_main_guard_deg=args.alpha_main_guard_deg,
        overwrite=args.overwrite,
    )

    out_dir = os.path.abspath(args.out_dir)
    _ensure_dir(out_dir)
    output_paths = [os.path.join(out_dir, name) for name in OUTPUT_BASENAMES]
    _check_output_paths(output_paths, cfg.overwrite)
    logger = _setup_logger(os.path.join(out_dir, "q0cra_stage1_static_pool.log"))
    start = time.perf_counter()
    start_utc = _utc_now_iso()

    try:
        logger.info("[CONFIG] out_dir=%s", out_dir)
        logger.info(
            "[CONFIG] partition_table=%s selection_mode=%s allow_manual=%s",
            cfg.partition_table_json, cfg.partition_selection_mode,
            cfg.allow_manual_partition,
        )
        logger.info(
            "[CONFIG] full_scan=[%g,%g] step=%g work_count=%d",
            cfg.theta_s_full_min_deg, cfg.theta_s_full_max_deg,
            cfg.theta_s_full_step_deg, len(cfg.theta_s_work_init_deg),
        )
        logger.info(
            "[CONFIG] alpha_1d_step=%g alpha_main_guard=%g budget=%d",
            cfg.alpha_1d_step_deg, cfg.alpha_main_guard_deg,
            cfg.budget_total_per_scan,
        )
        logger.info(
            "[NOTE] Geometry-only: no IEP interpolation, no a0 evaluation, no dynamic tracking execution"
        )

        package = build_stage1_static_pool(cfg)
        pinfo = package.partition_info
        logger.info(
            "[PARTITION] mode=%s selection=%s version=%s status=%s",
            pinfo.mode, pinfo.selection_mode, pinfo.table_version,
            pinfo.table_quality_status,
        )
        logger.info(
            "[PARTITION] selected K=%d d_min=%g d_eff=%g R0=%g worst_b=%g",
            pinfo.K, pinfo.d_min, pinfo.d_eff, pinfo.R0, pinfo.worst_b,
        )
        logger.info(
            "[PARTITION] alpha_calib=%g alpha_used=%g rho_used=%g outer_start=%g",
            pinfo.alpha_core_calib_deg, pinfo.alpha_core_used_deg,
            pinfo.rho_core_used, pinfo.alpha_outer_start_deg,
        )
        logger.info(
            "[POOL] buffer_rings=%s",
            list(package.config.buffer_ring_list_deg),
        )
        logger.info(
            "[VALIDATION] %s", package.validation_summary,
        )

        paths, summary = _save_pool(package, out_dir, overwrite=cfg.overwrite)
        elapsed = time.perf_counter() - start
        stats_full = summary["stats_full"]
        logger.info(
            "[SUMMARY] status=%s full_scan_count=%d work_scan_count=%d "
            "core=%d buffer=%d outer=%d max_per_scan=%d budget_ok=%s elapsed_sec=%.3f",
            summary["quality_control"]["status"],
            package.theta_s_full_deg.size,
            package.theta_s_work_init_deg.size,
            len(package.core_full.scan_idx),
            len(package.buffer_full.scan_idx),
            len(package.outer_full.scan_idx),
            stats_full["n_total_max"],
            stats_full["budget_ok"],
            elapsed,
        )
        logger.info("[OUTPUT] npz=%s", paths["npz"])
        logger.info("[OUTPUT] summary_json=%s", paths["summary_json"])
        logger.info("[OUTPUT] counts_full_csv=%s", paths["counts_full_csv"])
        logger.info("[OUTPUT] counts_work_csv=%s", paths["counts_work_csv"])
        logger.info("[RUN] start_utc=%s end_utc=%s", start_utc, _utc_now_iso())
        return 0
    except Exception as exc:
        logger.exception("[FAILED] %s", exc)
        return 1
    finally:
        _close_logger(logger)


def main():
    return run_main()


if __name__ == "__main__":
    raise SystemExit(main())
