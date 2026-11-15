# -*- coding: utf-8 -*-
"""Persist optional static and dynamic geometric-constraint pool traces.

This module is deliberately isolated from the geometric optimization logic.
When disabled it performs no file-system writes.  When enabled it writes only
under ``<case_output>/stage1_pool_trace`` and never modifies the canonical
geometric backbone NPZ/JSON/CSV outputs. The established trace directory,
format identifiers, and public writer interface remain unchanged.
"""
from __future__ import print_function

import datetime as _dt
import json
import os
from pathlib import Path

import numpy as np


TRACE_FORMAT = "a0cra_stage1_pool_trace_v1"
STATIC_FORMAT = "a0cra_stage1_static_pool_sidecar_v1"
DYNAMIC_FORMAT = "a0cra_stage1_dynamic_pool_sidecar_v1"

_POINT_FIELDS = (
    ("alpha_deg", np.float64, np.nan),
    ("eta_deg", np.float64, np.nan),
    ("theta_deg", np.float64, np.nan),
    ("phi_deg", np.float64, np.nan),
    ("u", np.float64, np.nan),
    ("v", np.float64, np.nan),
    ("layer_code", np.int16, -1),
    ("is_side_2d", np.int8, 1),
    ("is_dynamic", np.int8, 1),
    ("seed_rank", np.int32, -1),
    ("refine_round", np.int16, 0),
)


def _json_safe(obj):
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        if np.isposinf(obj):
            return "inf"
        if np.isneginf(obj):
            return "-inf"
        return obj
    if isinstance(obj, (np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.floating):
        return _json_safe(float(obj))
    if isinstance(obj, np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(str(tmp), "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(str(tmp), str(path))


def _atomic_savez_compressed(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(str(tmp), **arrays)
    os.replace(str(tmp), str(path))


def _as_1d(value, dtype):
    return np.asarray(value, dtype=dtype).reshape(-1)


def _point_count(point_dict):
    if not point_dict:
        return 0
    if "alpha_deg" in point_dict:
        return int(np.asarray(point_dict["alpha_deg"]).size)
    for value in point_dict.values():
        arr = np.asarray(value)
        if arr.ndim >= 1:
            return int(arr.reshape(-1).size)
    return 0


def _extract_dynamic_stores(source, mode):
    """Return ``(buffer_store, outer_store)`` from a pool context or backup."""
    if source is None:
        return {}, {}
    mode = str(mode).lower().strip()
    if isinstance(source, dict):
        return (
            dict(source.get("pool_buffer", source.get("buffer", {})) or {}),
            dict(source.get("pool_outer", source.get("outer", {})) or {}),
        )
    if mode == "work":
        return (
            dict(getattr(source, "dynamic_buffer_work_by_scan", {}) or {}),
            dict(getattr(source, "dynamic_outer_work_by_scan", {}) or {}),
        )
    if mode == "full":
        return (
            dict(getattr(source, "dynamic_buffer_full_by_scan", {}) or {}),
            dict(getattr(source, "dynamic_outer_full_by_scan", {}) or {}),
        )
    raise ValueError("mode must be 'work' or 'full'")


class Stage1PoolTraceWriter(object):
    """Collect and persist point-pool snapshots without affecting optimization."""

    def __init__(self, case_output_dir, enabled=False):
        self.enabled = bool(enabled)
        self.case_output_dir = Path(case_output_dir)
        self.trace_dir = self.case_output_dir / "stage1_pool_trace"
        self.created_at = _dt.datetime.now().isoformat()
        self.errors = []
        self._finalized = False
        self._static_saved = False
        self._events = {"work": [], "full": []}
        self._point_chunks = {"work": {}, "full": {}}
        self._point_counts = {"work": 0, "full": 0}
        for mode in ("work", "full"):
            fields = {
                "event_id": [],
                "theta_s_deg": [],
                "zone_code": [],
            }
            for name, _dtype, _default in _POINT_FIELDS:
                fields[name] = []
            self._point_chunks[mode] = fields
        if self.enabled:
            try:
                self.trace_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self._record_error("init", exc)

    @property
    def finalized(self):
        return bool(self._finalized)

    def _record_error(self, stage, exc):
        self.errors.append({
            "stage": str(stage),
            "type": type(exc).__name__,
            "message": str(exc),
            "time": _dt.datetime.now().isoformat(),
        })

    def save_static_pool(self, static_npz, summary):
        if not self.enabled or self._static_saved:
            return False
        try:
            arrays = {str(k): np.asarray(v) for k, v in dict(static_npz).items()}
            npz_path = self.trace_dir / "static_pool_exact.npz"
            json_path = self.trace_dir / "static_pool_summary.json"
            _atomic_savez_compressed(npz_path, arrays)
            array_info = {
                key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
                for key, value in arrays.items()
            }
            payload = {
                "format": STATIC_FORMAT,
                "created_at": _dt.datetime.now().isoformat(),
                "npz_file": npz_path.name,
                "array_count": len(arrays),
                "arrays": array_info,
                "runtime_pool_summary": summary,
            }
            _atomic_write_json(json_path, payload)
            self._static_saved = True
            return True
        except Exception as exc:
            self._record_error("save_static_pool", exc)
            return False

    def record_dynamic_snapshot(
        self,
        *,
        event_type,
        mode,
        source,
        metadata=None,
        radius_current=None,
        N_current=None,
        radius_trial=None,
        N_trial=None,
    ):
        if not self.enabled or self._finalized:
            return False
        mode = str(mode).lower().strip()
        if mode not in ("work", "full"):
            self._record_error("record_dynamic_snapshot", ValueError("invalid mode: %s" % mode))
            return False
        try:
            buffer_store, outer_store = _extract_dynamic_stores(source, mode)
            event_id = len(self._events[mode])
            point_start = int(self._point_counts[mode])
            counts_by_scan = {}
            zone_totals = {"buffer_2d": 0, "outer_2d": 0}

            for zone_name, zone_code, store in (
                ("buffer_2d", 0, buffer_store),
                ("outer_2d", 1, outer_store),
            ):
                for theta_key in sorted(store.keys(), key=lambda x: float(x)):
                    point_dict = dict(store[theta_key] or {})
                    n = _point_count(point_dict)
                    if n <= 0:
                        continue
                    theta_value = float(theta_key)
                    counts_by_scan.setdefault("%.12g" % theta_value, {"buffer_2d": 0, "outer_2d": 0})
                    counts_by_scan["%.12g" % theta_value][zone_name] += int(n)
                    zone_totals[zone_name] += int(n)

                    self._point_chunks[mode]["event_id"].append(np.full((n,), event_id, dtype=np.int32))
                    self._point_chunks[mode]["theta_s_deg"].append(np.full((n,), theta_value, dtype=np.float64))
                    self._point_chunks[mode]["zone_code"].append(np.full((n,), zone_code, dtype=np.int8))
                    for field_name, dtype, default in _POINT_FIELDS:
                        if field_name in point_dict:
                            arr = _as_1d(point_dict[field_name], dtype)
                            if arr.size != n:
                                raise ValueError(
                                    "%s/%s theta=%s field=%s has %d values, expected %d"
                                    % (mode, zone_name, theta_value, field_name, arr.size, n)
                                )
                        else:
                            arr = np.full((n,), default, dtype=dtype)
                        self._point_chunks[mode][field_name].append(arr)
                    self._point_counts[mode] += int(n)

            point_count = int(self._point_counts[mode] - point_start)
            event = {
                "event_id": int(event_id),
                "event_type": str(event_type),
                "mode": mode,
                "created_at": _dt.datetime.now().isoformat(),
                "point_start": point_start,
                "point_count": point_count,
                "zone_totals": zone_totals,
                "counts_by_scan": counts_by_scan,
                "metadata": dict(metadata or {}),
                "r_current": None if radius_current is None else _as_1d(radius_current, np.float64).tolist(),
                "N_current": None if N_current is None else _as_1d(N_current, np.int64).tolist(),
                "r_trial": None if radius_trial is None else _as_1d(radius_trial, np.float64).tolist(),
                "N_trial": None if N_trial is None else _as_1d(N_trial, np.int64).tolist(),
            }
            self._events[mode].append(event)
            return True
        except Exception as exc:
            self._record_error("record_dynamic_snapshot:%s:%s" % (mode, event_type), exc)
            return False

    def _build_dynamic_arrays(self, mode):
        arrays = {}
        dtype_map = {
            "event_id": np.int32,
            "theta_s_deg": np.float64,
            "zone_code": np.int8,
        }
        for name, dtype, _default in _POINT_FIELDS:
            dtype_map[name] = dtype
        for key, chunks in self._point_chunks[mode].items():
            dtype = dtype_map[key]
            if chunks:
                arrays[key] = np.concatenate(chunks).astype(dtype, copy=False)
            else:
                arrays[key] = np.zeros((0,), dtype=dtype)
        return arrays

    def finalize(self, success=True):
        if not self.enabled:
            self._finalized = True
            return {"enabled": False, "success": bool(success), "errors": []}
        if self._finalized:
            return {
                "enabled": True,
                "success": bool(success) and not self.errors,
                "errors": list(self.errors),
                "trace_dir": str(self.trace_dir),
            }
        try:
            for mode in ("work", "full"):
                npz_name = "%s_dynamic_trace.npz" % mode
                json_name = "%s_dynamic_trace_index.json" % mode
                arrays = self._build_dynamic_arrays(mode)
                _atomic_savez_compressed(self.trace_dir / npz_name, arrays)
                _atomic_write_json(
                    self.trace_dir / json_name,
                    {
                        "format": DYNAMIC_FORMAT,
                        "mode": mode,
                        "created_at": self.created_at,
                        "finalized_at": _dt.datetime.now().isoformat(),
                        "npz_file": npz_name,
                        "zone_code": {"0": "buffer_2d", "1": "outer_2d"},
                        "event_count": len(self._events[mode]),
                        "point_count": int(self._point_counts[mode]),
                        "events": self._events[mode],
                    },
                )
        except Exception as exc:
            self._record_error("finalize_dynamic", exc)
        complete = {
            "format": TRACE_FORMAT,
            "enabled": True,
            "algorithm_success": bool(success),
            "trace_success": len(self.errors) == 0,
            "created_at": self.created_at,
            "finalized_at": _dt.datetime.now().isoformat(),
            "static_saved": bool(self._static_saved),
            "work_event_count": len(self._events["work"]),
            "work_point_count": int(self._point_counts["work"]),
            "full_event_count": len(self._events["full"]),
            "full_point_count": int(self._point_counts["full"]),
            "files": {
                "static_npz": "static_pool_exact.npz",
                "static_summary": "static_pool_summary.json",
                "work_npz": "work_dynamic_trace.npz",
                "work_index": "work_dynamic_trace_index.json",
                "full_npz": "full_dynamic_trace.npz",
                "full_index": "full_dynamic_trace_index.json",
            },
            "errors": list(self.errors),
        }
        try:
            _atomic_write_json(self.trace_dir / "pool_trace_complete.json", complete)
        except Exception as exc:
            self._record_error("finalize_complete_json", exc)
        self._finalized = True
        return {
            "enabled": True,
            "success": bool(success) and not self.errors,
            "errors": list(self.errors),
            "trace_dir": str(self.trace_dir),
            "work_event_count": len(self._events["work"]),
            "full_event_count": len(self._events["full"]),
        }
