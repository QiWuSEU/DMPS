#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load, validate, and query rotation-compatible sparse IEP spectra for DMPS.

This module reconstructs co- and cross-polarized fields from the retained
azimuthal Fourier coefficients and exposes the dominant mode a0 and the
reindexed cross-polarized modes c_q to downstream synthesis algorithms.

All angular query inputs are in radians. The serialized ``hfss_sr_sparse``
version-2 NPZ/JSON contract remains unchanged for numerical reproducibility.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np


SUPPORTED_SCHEMA_NAME = "hfss_sr_sparse"
SUPPORTED_SCHEMA_VERSION = 2
SUPPORTED_SPECTRAL_FRAME = "sr_compatible_sparse_v2"

REQUIRED_JSON_TOP_KEYS = (
    "schema",
    "run",
    "input",
    "resolved_config",
    "grid",
    "polarization",
    "normalization",
    "spectral_model",
    "sparse_selection",
    "fit_checks",
    "quality_control",
    "outputs",
)

REQUIRED_NPZ_KEYS = (
    "schema_name",
    "schema_version",
    "spectral_frame",
    "co_handedness",
    "cx_handedness",
    "spin_sign",
    "cp_basis_sign",
    "normalization_A_ref",
    "normalization_scale",
    "theta_grid_rad",
    "phi_grid_rad",
    "aq_q_list",
    "aq_grid",
    "aq_keep_mask",
    "a0",
    "cq_q_list",
    "cq_grid",
    "cq_keep_mask",
    "c0",
)

_FLOAT_TOL_RAD = 1e-10
_GRID_RTOL = 0.0
_GRID_ATOL = 1e-10
_ANCHOR_RTOL = 1e-12
_ANCHOR_ATOL = 1e-14


class IEPError(Exception):
    """Base exception for IEP IO failures."""


class IEPSchemaError(IEPError):
    """Raised when the NPZ/JSON schema is not the supported v2 schema."""


class IEPDataError(IEPError):
    """Raised when required data are missing or internally inconsistent."""


class IEPQueryError(IEPError):
    """Raised when a field/spectrum query is invalid."""


def _sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def wrap_phi_to_0_2pi(phi_rad):
    return np.mod(np.asarray(phi_rad, dtype=np.float64), 2.0 * math.pi)


def _scalar_text(x):
    arr = np.asarray(x)
    if arr.size != 1:
        raise IEPDataError("Expected scalar text value, got shape %r." % (arr.shape,))
    value = arr.reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _scalar_int(x, name):
    arr = np.asarray(x)
    if arr.size != 1:
        raise IEPDataError("%s must be scalar, got shape %r." % (name, arr.shape))
    return int(arr.reshape(()).item())


def _scalar_float(x, name):
    arr = np.asarray(x)
    if arr.size != 1:
        raise IEPDataError("%s must be scalar, got shape %r." % (name, arr.shape))
    value = float(arr.reshape(()).item())
    if not np.isfinite(value):
        raise IEPDataError("%s must be finite." % name)
    return value


def _all_finite_complex(x):
    arr = np.asarray(x, dtype=np.complex128)
    return bool(np.all(np.isfinite(arr.real)) and np.all(np.isfinite(arr.imag)))


def _restore_shape(values, original_shape, scalar_input):
    arr = np.asarray(values)
    arr = arr.reshape(original_shape)
    if scalar_input:
        return arr.reshape(()).item()
    return arr


def _load_json_strict(path):
    if not os.path.isfile(path):
        raise IEPDataError("Required metadata JSON not found: %s" % path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception as exc:
        raise IEPDataError("Cannot read metadata JSON %s: %s" % (path, exc))
    if not isinstance(report, dict):
        raise IEPDataError("Metadata JSON root must be an object/dict.")
    missing = [key for key in REQUIRED_JSON_TOP_KEYS if key not in report]
    if missing:
        raise IEPDataError("Metadata JSON missing top-level keys: %s" % missing)

    schema = report.get("schema", {})
    if not isinstance(schema, dict):
        raise IEPSchemaError("JSON schema section must be a dict.")
    if schema.get("name") != SUPPORTED_SCHEMA_NAME:
        raise IEPSchemaError(
            "Unsupported JSON schema name %r; expected %r."
            % (schema.get("name"), SUPPORTED_SCHEMA_NAME)
        )
    if int(schema.get("version", -1)) != SUPPORTED_SCHEMA_VERSION:
        raise IEPSchemaError(
            "Unsupported JSON schema version %r; expected %r."
            % (schema.get("version"), SUPPORTED_SCHEMA_VERSION)
        )
    if schema.get("spectral_frame") != SUPPORTED_SPECTRAL_FRAME:
        raise IEPSchemaError(
            "Unsupported JSON spectral frame %r; expected %r."
            % (schema.get("spectral_frame"), SUPPORTED_SPECTRAL_FRAME)
        )

    qc = report.get("quality_control", {})
    if not isinstance(qc, dict) or qc.get("status") != "PASS":
        raise IEPDataError(
            "Metadata quality_control.status must be 'PASS', got %r."
            % (qc.get("status") if isinstance(qc, dict) else None,)
        )
    run = report.get("run", {})
    if not isinstance(run, dict) or run.get("status") != "PASS":
        raise IEPDataError(
            "Metadata run.status must be 'PASS', got %r."
            % (run.get("status") if isinstance(run, dict) else None,)
        )
    return report


def _load_npz_strict(path):
    if not os.path.isfile(path):
        raise IEPDataError("IEP NPZ not found: %s" % path)
    try:
        with np.load(path, allow_pickle=False) as z:
            missing = [key for key in REQUIRED_NPZ_KEYS if key not in z.files]
            if missing:
                raise IEPDataError("IEP NPZ missing required keys: %s" % missing)
            data = {
                "schema_name": _scalar_text(z["schema_name"]),
                "schema_version": _scalar_int(z["schema_version"], "schema_version"),
                "spectral_frame": _scalar_text(z["spectral_frame"]),
                "co_handedness": _scalar_text(z["co_handedness"]).upper(),
                "cx_handedness": _scalar_text(z["cx_handedness"]).upper(),
                "spin_sign": _scalar_int(z["spin_sign"], "spin_sign"),
                "cp_basis_sign": _scalar_int(z["cp_basis_sign"], "cp_basis_sign"),
                "normalization_A_ref": _scalar_float(z["normalization_A_ref"], "normalization_A_ref"),
                "normalization_scale": _scalar_float(z["normalization_scale"], "normalization_scale"),
                "theta_grid_rad": np.asarray(z["theta_grid_rad"], dtype=np.float64).copy(),
                "phi_grid_rad": np.asarray(z["phi_grid_rad"], dtype=np.float64).copy(),
                "aq_q_list": np.asarray(z["aq_q_list"], dtype=np.int64).copy(),
                "aq_grid": np.asarray(z["aq_grid"], dtype=np.complex128).copy(),
                "aq_keep_mask": np.asarray(z["aq_keep_mask"], dtype=bool).copy(),
                "a0": np.asarray(z["a0"], dtype=np.complex128).copy(),
                "cq_q_list": np.asarray(z["cq_q_list"], dtype=np.int64).copy(),
                "cq_grid": np.asarray(z["cq_grid"], dtype=np.complex128).copy(),
                "cq_keep_mask": np.asarray(z["cq_keep_mask"], dtype=bool).copy(),
                "c0": np.asarray(z["c0"], dtype=np.complex128).copy(),
            }
    except IEPError:
        raise
    except Exception as exc:
        raise IEPDataError("Cannot read IEP NPZ %s: %s" % (path, exc))
    return data


def _validate_uniform_periodic_phi(phi):
    if phi.ndim != 1 or phi.size < 3:
        raise IEPDataError("phi_grid_rad must be a 1D array with at least 3 points.")
    if not np.all(np.isfinite(phi)):
        raise IEPDataError("phi_grid_rad contains NaN or Inf.")
    if np.any(np.diff(phi) <= 0.0):
        raise IEPDataError("phi_grid_rad must be strictly increasing.")
    if float(phi[0]) < -_GRID_ATOL or float(phi[-1]) >= 2.0 * math.pi + _GRID_ATOL:
        raise IEPDataError("phi_grid_rad must lie in [0, 2*pi).")
    steps = np.diff(phi)
    wrap_step = 2.0 * math.pi - float(phi[-1]) + float(phi[0])
    all_steps = np.concatenate([steps, np.array([wrap_step], dtype=np.float64)])
    step = float(np.median(all_steps))
    if step <= 0.0:
        raise IEPDataError("Invalid non-positive phi step.")
    if not np.allclose(all_steps, step, rtol=0.0, atol=1e-9):
        raise IEPDataError("phi_grid_rad must be a uniform full-period grid.")
    if not np.isclose(float(phi.size) * step, 2.0 * math.pi, rtol=0.0, atol=1e-8):
        raise IEPDataError("phi_grid_rad does not cover one complete 2*pi period.")


def _validate_npz_data(data):
    if data["schema_name"] != SUPPORTED_SCHEMA_NAME:
        raise IEPSchemaError(
            "Unsupported NPZ schema_name %r; expected %r."
            % (data["schema_name"], SUPPORTED_SCHEMA_NAME)
        )
    if data["schema_version"] != SUPPORTED_SCHEMA_VERSION:
        raise IEPSchemaError(
            "Unsupported NPZ schema_version %r; expected %r."
            % (data["schema_version"], SUPPORTED_SCHEMA_VERSION)
        )
    if data["spectral_frame"] != SUPPORTED_SPECTRAL_FRAME:
        raise IEPSchemaError(
            "Unsupported NPZ spectral_frame %r; expected %r."
            % (data["spectral_frame"], SUPPORTED_SPECTRAL_FRAME)
        )

    theta = data["theta_grid_rad"]
    phi = data["phi_grid_rad"]
    if theta.ndim != 1 or theta.size < 2:
        raise IEPDataError("theta_grid_rad must be a 1D array with at least 2 points.")
    if not np.all(np.isfinite(theta)):
        raise IEPDataError("theta_grid_rad contains NaN or Inf.")
    if np.any(np.diff(theta) <= 0.0):
        raise IEPDataError("theta_grid_rad must be strictly increasing.")
    if abs(float(theta[0])) > _FLOAT_TOL_RAD:
        raise IEPDataError("theta_grid_rad must start at 0 rad within tolerance.")
    if float(theta[-1]) < 0.5 * math.pi - _FLOAT_TOL_RAD:
        raise IEPDataError("theta_grid_rad must cover theta=90 deg.")
    if float(theta[-1]) > 0.5 * math.pi + _FLOAT_TOL_RAD:
        raise IEPDataError("theta_grid_rad extends beyond theta=90 deg by more than tolerance.")
    _validate_uniform_periodic_phi(phi)

    nt = int(theta.size)
    for prefix in ("aq", "cq"):
        q_list = data[prefix + "_q_list"]
        grid = data[prefix + "_grid"]
        mask = data[prefix + "_keep_mask"]
        anchor = data["a0" if prefix == "aq" else "c0"]
        if q_list.ndim != 1 or q_list.size < 1:
            raise IEPDataError("%s_q_list must be non-empty and 1D." % prefix)
        if np.any(np.diff(q_list) <= 0):
            raise IEPDataError("%s_q_list must be strictly increasing with no duplicates." % prefix)
        if grid.shape != (nt, int(q_list.size)):
            raise IEPDataError(
                "%s_grid shape %r does not match (n_theta,n_q)=(%d,%d)."
                % (prefix, grid.shape, nt, q_list.size)
            )
        if mask.shape != grid.shape:
            raise IEPDataError("%s_keep_mask shape mismatch with %s_grid." % (prefix, prefix))
        if anchor.shape != (nt,):
            raise IEPDataError("%s anchor must have shape (%d,), got %r." % (anchor, nt, anchor.shape))
        if not _all_finite_complex(grid):
            raise IEPDataError("%s_grid contains NaN or Inf." % prefix)
        if not _all_finite_complex(anchor):
            raise IEPDataError("%s anchor contains NaN or Inf." % ("a0" if prefix == "aq" else "c0"))
        q0 = np.where(q_list == 0)[0]
        if q0.size != 1:
            raise IEPDataError("%s_q_list must contain q=0 exactly once." % prefix)
        q0_idx = int(q0[0])
        if not np.all(mask[:, q0_idx]):
            raise IEPDataError("%s q=0 keep mask must be True at every theta." % prefix)
        if not np.allclose(anchor, grid[:, q0_idx], rtol=_ANCHOR_RTOL, atol=_ANCHOR_ATOL):
            raise IEPDataError("%s anchor is inconsistent with the q=0 spectral column." % ("a0" if prefix == "aq" else "c0"))

    if data["normalization_A_ref"] <= 0.0:
        raise IEPDataError("normalization_A_ref must be positive.")
    if data["normalization_scale"] <= 0.0:
        raise IEPDataError("normalization_scale must be positive.")
    if data["cp_basis_sign"] not in (-1, 1):
        raise IEPDataError("cp_basis_sign must be +1 or -1.")
    valid_routes = {
        ("LHCP", "RHCP", 1),
        ("RHCP", "LHCP", -1),
    }
    route = (data["co_handedness"], data["cx_handedness"], data["spin_sign"])
    if route not in valid_routes:
        raise IEPDataError(
            "Invalid co/cx/spin combination %r. Expected LHCP/RHCP/+1 or RHCP/LHCP/-1."
            % (route,)
        )


def _json_q_list(report, channel):
    try:
        return [int(v) for v in report["sparse_selection"][channel]["q_list"]]
    except Exception:
        raise IEPDataError("JSON missing sparse_selection.%s.q_list." % channel)


def _validate_json_npz_consistency(report, data, npz_path, verify_hash):
    schema = report["schema"]
    if schema["name"] != data["schema_name"]:
        raise IEPDataError("JSON/NPZ schema name mismatch.")
    if int(schema["version"]) != data["schema_version"]:
        raise IEPDataError("JSON/NPZ schema version mismatch.")
    if schema["spectral_frame"] != data["spectral_frame"]:
        raise IEPDataError("JSON/NPZ spectral frame mismatch.")

    pol = report.get("polarization", {})
    pairs = (
        ("co_handedness", data["co_handedness"]),
        ("cx_handedness", data["cx_handedness"]),
        ("spin_sign", data["spin_sign"]),
        ("cp_basis_sign", data["cp_basis_sign"]),
    )
    for key, expected in pairs:
        actual = pol.get(key)
        if isinstance(expected, str):
            actual = str(actual).upper()
        else:
            try:
                actual = int(actual)
            except Exception:
                pass
        if actual != expected:
            raise IEPDataError("JSON/NPZ polarization mismatch for %s." % key)

    norm = report.get("normalization", {})
    if not np.isclose(float(norm.get("A_ref")), data["normalization_A_ref"], rtol=1e-12, atol=1e-14):
        raise IEPDataError("JSON/NPZ normalization A_ref mismatch.")
    if not np.isclose(float(norm.get("scale")), data["normalization_scale"], rtol=1e-12, atol=1e-14):
        raise IEPDataError("JSON/NPZ normalization scale mismatch.")

    if _json_q_list(report, "aq") != data["aq_q_list"].tolist():
        raise IEPDataError("JSON/NPZ aq q-list mismatch.")
    if _json_q_list(report, "cq") != data["cq_q_list"].tolist():
        raise IEPDataError("JSON/NPZ cq q-list mismatch.")

    if verify_hash:
        try:
            expected_hash = str(report["outputs"]["main_npz"]["sha256"]).strip().lower()
        except Exception:
            raise IEPDataError("JSON outputs.main_npz.sha256 is required for hash verification.")
        actual_hash = _sha256_file(npz_path).lower()
        if expected_hash != actual_hash:
            raise IEPDataError(
                "NPZ SHA256 mismatch: JSON=%s actual=%s" % (expected_hash, actual_hash)
            )
        return actual_hash
    return _sha256_file(npz_path).lower()


class IEPModelConfig(object):
    """Angular query configuration with a backward-compatible public signature."""

    def __init__(self, phi_main_rad=0.0, bounds="error", enable_bilinear_phi=True):
        bounds = str(bounds).strip().lower()
        if bounds not in ("clamp", "error", "nan"):
            raise ValueError("bounds must be one of: clamp, error, nan")
        self.phi_main_rad = float(phi_main_rad)
        self.bounds = bounds
        # Retained for existing callers; phi is evaluated by spectral reconstruction,
        # not by bilinear interpolation.
        self.enable_bilinear_phi = bool(enable_bilinear_phi)


class IEPModel(object):
    """Strict schema-v2 field model for geometry and passive modal synthesis."""

    def __init__(self, npz_path, meta_json_path=None, config=None, verify_hash=True):
        self.npz_path = os.path.abspath(str(npz_path))
        if meta_json_path is None:
            stem, _ = os.path.splitext(self.npz_path)
            meta_json_path = stem + ".json"
        self.meta_json_path = os.path.abspath(str(meta_json_path))
        self.config = config or IEPModelConfig()
        self.verify_hash = bool(verify_hash)

        self.meta = _load_json_strict(self.meta_json_path)
        data = _load_npz_strict(self.npz_path)
        _validate_npz_data(data)
        self.npz_sha256 = _validate_json_npz_consistency(
            self.meta, data, self.npz_path, self.verify_hash
        )

        self.schema_name = data["schema_name"]
        self.schema_version = data["schema_version"]
        self.spectral_frame = data["spectral_frame"]
        self.co_handedness = data["co_handedness"]
        self.cx_handedness = data["cx_handedness"]
        self.spin_sign = data["spin_sign"]
        self.cp_basis_sign = data["cp_basis_sign"]
        self.normalization_A_ref = data["normalization_A_ref"]
        self.normalization_scale = data["normalization_scale"]

        self.theta_grid_rad = data["theta_grid_rad"]
        self.phi_grid_rad = data["phi_grid_rad"]
        self._aq_q_list = data["aq_q_list"]
        self._aq_grid = data["aq_grid"]
        self._aq_keep_mask = data["aq_keep_mask"]
        self._a0_grid = data["a0"]
        self._cq_q_list = data["cq_q_list"]
        self._cq_grid = data["cq_grid"]
        self._cq_keep_mask = data["cq_keep_mask"]
        self._c0_grid = data["c0"]

        self._aq_sparse_grid = np.where(
            self._aq_keep_mask, self._aq_grid, 0.0 + 0.0j
        ).astype(np.complex128)
        self._cq_sparse_grid = np.where(
            self._cq_keep_mask, self._cq_grid, 0.0 + 0.0j
        ).astype(np.complex128)

        self._aq_q_to_idx = {
            int(q): int(i) for i, q in enumerate(self._aq_q_list.tolist())
        }
        self._cq_q_to_idx = {
            int(q): int(i) for i, q in enumerate(self._cq_q_list.tolist())
        }

        # No generalized/legacy data are supported.
        self.q_floor = None
        self._b0_grid = None
        self._b_floor_grid = None
        self._bq_q_list = None
        self._bq_grid = None
        self._bq_keep_mask = None

        inp = self.meta.get("input", {})
        self.source_csv_path = inp.get("csv_path")
        self.source_csv_sha256 = inp.get("csv_sha256")
        self.variation_selected = inp.get("variation_selected")
        self.co_fit_report = self.meta.get("fit_checks", {}).get("co_0to90", {})
        self.cx_fit_report = self.meta.get("fit_checks", {}).get("cx_0to65", {})
        self.cx_fit_validated_theta_max_deg = float(
            self.cx_fit_report.get("theta_max_deg", 65.0)
        )

        # Reconstruct native sparse fields once. These attributes preserve the old public surface.
        self.E_co = self._reconstruct_native_co()
        self.E_cx = self._reconstruct_native_cx()
        self._g_side_grid = np.max(np.abs(self.E_co), axis=1).astype(np.float64)

    # -------------------------
    # Internal query helpers
    # -------------------------
    def _prepare_theta_query(self, theta_rad):
        arr = np.asarray(theta_rad, dtype=np.float64)
        scalar_input = arr.ndim == 0
        shape = arr.shape
        flat = arr.reshape(-1)
        if not np.all(np.isfinite(flat)):
            raise IEPQueryError("theta query contains NaN or Inf.")
        lo = float(self.theta_grid_rad[0])
        hi = float(self.theta_grid_rad[-1])
        too_low = flat < lo - _FLOAT_TOL_RAD
        too_high = flat > hi + _FLOAT_TOL_RAD
        oob = too_low | too_high
        if self.config.bounds == "error" and np.any(oob):
            raise IEPQueryError(
                "theta query out of bounds [%.17g, %.17g] rad." % (lo, hi)
            )
        clipped = np.clip(flat, lo, hi)
        return clipped, shape, scalar_input, oob

    def _interp_complex_grid(self, theta_rad, value_grid):
        """Linearly interpolate real and imaginary spectral parts over theta."""
        theta_flat, shape, scalar_input, oob = self._prepare_theta_query(theta_rad)
        values = np.asarray(value_grid, dtype=np.complex128)
        if values.ndim == 1:
            real = np.interp(theta_flat, self.theta_grid_rad, values.real)
            imag = np.interp(theta_flat, self.theta_grid_rad, values.imag)
            out = real + 1j * imag
            if self.config.bounds == "nan" and np.any(oob):
                out = out.astype(np.complex128, copy=True)
                out[oob] = np.nan + 1j * np.nan
            return _restore_shape(out, shape, scalar_input)
        if values.ndim == 2:
            out = np.empty((theta_flat.size, values.shape[1]), dtype=np.complex128)
            for j in range(values.shape[1]):
                out[:, j] = (
                    np.interp(theta_flat, self.theta_grid_rad, values[:, j].real)
                    + 1j * np.interp(theta_flat, self.theta_grid_rad, values[:, j].imag)
                )
            if self.config.bounds == "nan" and np.any(oob):
                out[oob, :] = np.nan + 1j * np.nan
            return out.reshape(shape + (values.shape[1],))
        raise IEPDataError("Interpolation value_grid must be 1D or 2D.")

    def _reconstruct_native_co(self):
        """Evaluate the retained a_q spectrum using exp(1j*q*phi)."""
        phase = np.exp(
            1j
            * self._aq_q_list.astype(np.float64)[:, None]
            * self.phi_grid_rad[None, :]
        )
        return np.asarray(self._aq_sparse_grid.dot(phase), dtype=np.complex128)

    def _reconstruct_native_cx(self):
        """Evaluate the retained c_q spectrum using exp(1j*(q+2*s)*phi)."""
        orders = self._cq_q_list.astype(np.float64) + 2.0 * float(self.spin_sign)
        phase = np.exp(1j * orders[:, None] * self.phi_grid_rad[None, :])
        return np.asarray(self._cq_sparse_grid.dot(phase), dtype=np.complex128)

    def _query_field(self, theta_rad, phi_rad, q_list, sparse_grid, cx_mode):
        """Broadcast angular queries and reconstruct the sparse complex field."""
        theta_arr = np.asarray(theta_rad, dtype=np.float64)
        phi_arr = np.asarray(phi_rad, dtype=np.float64)
        try:
            theta_b, phi_b = np.broadcast_arrays(theta_arr, phi_arr)
        except ValueError as exc:
            raise IEPQueryError("theta/phi cannot be broadcast together: %s" % exc)
        if not np.all(np.isfinite(phi_b)):
            raise IEPQueryError("phi query contains NaN or Inf.")
        coeff = np.asarray(self._interp_complex_grid(theta_b, sparse_grid), dtype=np.complex128)
        phi_w = wrap_phi_to_0_2pi(phi_b).reshape(-1)
        coeff2 = coeff.reshape((-1, q_list.size))
        orders = q_list.astype(np.float64)
        if cx_mode:
            orders = orders + 2.0 * float(self.spin_sign)
        phase = np.exp(1j * phi_w[:, None] * orders[None, :])
        out = np.sum(coeff2 * phase, axis=1).reshape(theta_b.shape)
        if theta_b.ndim == 0:
            return out.reshape(()).item()
        return out

    # -------------------------
    # Public field queries
    # -------------------------
    def query_E_co(self, theta_rad, phi_rad):
        return self._query_field(
            theta_rad, phi_rad, self._aq_q_list, self._aq_sparse_grid, False
        )

    def query_E_cx(self, theta_rad, phi_rad):
        return self._query_field(
            theta_rad, phi_rad, self._cq_q_list, self._cq_sparse_grid, True
        )

    def get_cut(self, phi_s_rad):
        th = self.theta_grid_rad.copy()
        phi = np.full(th.shape, float(phi_s_rad), dtype=np.float64)
        return th, self.query_E_co(th, phi), self.query_E_cx(th, phi)

    # -------------------------
    # Envelopes / anchors
    # -------------------------
    def g_main(self, theta_rad):
        theta_arr = np.asarray(theta_rad, dtype=np.float64)
        phi = np.full(theta_arr.shape, float(self.config.phi_main_rad), dtype=np.float64)
        return np.abs(self.query_E_co(theta_arr, phi)).astype(np.float64)

    def g_side(self, theta_rad):
        value = self._interp_complex_grid(
            theta_rad, self._g_side_grid.astype(np.complex128)
        )
        return np.asarray(value).real.astype(np.float64) if not np.isscalar(value) else float(np.real(value))

    def a0(self, theta_rad):
        return self._interp_complex_grid(theta_rad, self._a0_grid)

    def a0_abs(self, theta_rad):
        return np.abs(self.a0(theta_rad)).astype(np.float64)

    def a0_grid(self):
        return self._a0_grid.copy()

    def a0_abs_grid(self):
        return np.abs(self._a0_grid).astype(np.float64)

    def c0(self, theta_rad):
        return self._interp_complex_grid(theta_rad, self._c0_grid)

    # Explicit non-support stubs preserve method-name compatibility without loading legacy formats.
    def b0(self, theta_rad):
        raise IEPDataError("b0 is not part of hfss_sr_sparse schema v2.")

    def b_floor(self, theta_rad):
        raise IEPDataError("b_floor is not part of hfss_sr_sparse schema v2.")

    def b_m2(self, theta_rad):
        raise IEPDataError("b_m2 is not part of hfss_sr_sparse schema v2.")

    # -------------------------
    # Spectral accessors
    # -------------------------
    @property
    def aq_q_list(self):
        return self._aq_q_list

    @property
    def cq_q_list(self):
        return self._cq_q_list

    @property
    def bq_q_list(self):
        return None

    def has_aq(self, q):
        return int(q) in self._aq_q_to_idx

    def has_cq(self, q):
        return int(q) in self._cq_q_to_idx

    def aq_index(self, q):
        q = int(q)
        if q not in self._aq_q_to_idx:
            raise KeyError("q=%d not in aq_q_list." % q)
        return self._aq_q_to_idx[q]

    def cq_index(self, q):
        q = int(q)
        if q not in self._cq_q_to_idx:
            raise KeyError("q=%d not in cq_q_list." % q)
        return self._cq_q_to_idx[q]

    def bq_index(self, q):
        raise IEPDataError("bq is not part of hfss_sr_sparse schema v2.")

    def aq(self, theta_rad, q):
        q = int(q)
        if q not in self._aq_q_to_idx:
            theta = np.asarray(theta_rad, dtype=np.float64)
            if not np.all(np.isfinite(theta)):
                raise IEPQueryError("theta query contains NaN or Inf.")
            # Apply bounds policy even when the requested q is absent.
            _, _, _, oob = self._prepare_theta_query(theta)
            zeros = np.zeros(theta.shape, dtype=np.complex128)
            if self.config.bounds == "nan" and np.any(oob):
                zeros = np.where(oob.reshape(theta.shape), np.nan + 1j * np.nan, zeros)
            return zeros.reshape(()).item() if theta.ndim == 0 else zeros
        return self._interp_complex_grid(
            theta_rad, self._aq_sparse_grid[:, self._aq_q_to_idx[q]]
        )

    def cq(self, theta_rad, q):
        q = int(q)
        if q not in self._cq_q_to_idx:
            theta = np.asarray(theta_rad, dtype=np.float64)
            if not np.all(np.isfinite(theta)):
                raise IEPQueryError("theta query contains NaN or Inf.")
            # Missing modes are zero in range but must follow the bounds policy.
            _, _, _, oob = self._prepare_theta_query(theta)
            zeros = np.zeros(theta.shape, dtype=np.complex128)
            if self.config.bounds == "nan" and np.any(oob):
                zeros = np.where(oob.reshape(theta.shape), np.nan + 1j * np.nan, zeros)
            return zeros.reshape(()).item() if theta.ndim == 0 else zeros
        return self._interp_complex_grid(
            theta_rad, self._cq_sparse_grid[:, self._cq_q_to_idx[q]]
        )

    def bq(self, theta_rad, q):
        raise IEPDataError("bq is not part of hfss_sr_sparse schema v2.")

    def aq_vec(self, theta_rad):
        return self._interp_complex_grid(theta_rad, self._aq_sparse_grid)

    def cq_vec(self, theta_rad):
        return self._interp_complex_grid(theta_rad, self._cq_sparse_grid)

    def bq_vec(self, theta_rad):
        raise IEPDataError("bq is not part of hfss_sr_sparse schema v2.")

    def aq_keep_mask_grid(self):
        return self._aq_keep_mask.copy()

    def cq_keep_mask_grid(self):
        return self._cq_keep_mask.copy()

    def bq_keep_mask_grid(self):
        return None

    def _forced_q_list_enabled(self):
        return True

    def build_interp_spectral_assets(
        self,
        theta_key_rad,
        tau_a=0.05,
        tau_b=0.1,
        eps_ref=1e-12,
        verbose=False,
    ):
        """Interpolate the already-selected spectral coefficients at theta keys.

        The sparse mode masks are frozen in the input dataset. The tau_a,
        tau_b, and eps_ref arguments are retained only for caller compatibility
        and do not trigger another modal-selection pass.
        """
        del tau_a, tau_b, eps_ref
        theta_key_rad = np.asarray(theta_key_rad, dtype=np.float64).reshape(-1)
        a0 = np.asarray(self.a0(theta_key_rad), dtype=np.complex128).reshape(-1)
        c0 = np.asarray(self.c0(theta_key_rad), dtype=np.complex128).reshape(-1)
        aq_grid = np.asarray(self.aq_vec(theta_key_rad), dtype=np.complex128)
        cq_grid = np.asarray(self.cq_vec(theta_key_rad), dtype=np.complex128)
        q0a = self.aq_index(0)
        q0c = self.cq_index(0)
        active_a = []
        active_c = []
        for t in range(theta_key_rad.size):
            a_idx = np.where(np.abs(aq_grid[t, :]) > 0.0)[0].astype(np.int64)
            c_idx = np.where(np.abs(cq_grid[t, :]) > 0.0)[0].astype(np.int64)
            a_idx = a_idx[a_idx != q0a]
            c_idx = c_idx[c_idx != q0c]
            active_a.append(a_idx)
            active_c.append(c_idx)
        if verbose:
            print("=== IEP_IO INTERP: SR-COMPATIBLE V2 ASSETS ===")
            print("- forced_q_list=True")
            print("- theta_keys=%d" % theta_key_rad.size)
            print("- q_list_a=%s" % self._aq_q_list.tolist())
            print("- q_list_c=%s" % self._cq_q_list.tolist())
        return {
            "spectral_frame": self.spectral_frame,
            "theta_key_rad": theta_key_rad,
            "a0": a0,
            "c0": c0,
            "aq_q_list": self._aq_q_list.copy(),
            "cq_q_list": self._cq_q_list.copy(),
            "aq_grid": aq_grid,
            "cq_grid": cq_grid,
            "active_a_qidx_by_theta": active_a,
            "active_c_qidx_by_theta": active_c,
            "forced_q_list": True,
        }

    # -------------------------
    # Metadata / readiness / cache
    # -------------------------
    def stage1_usage(self):
        value = self.meta.get("stage1_usage", {})
        return value if isinstance(value, dict) else {}

    def stage1_anchor_info(self):
        theory = self.meta.get("theory", {})
        if not isinstance(theory, dict):
            return {}
        value = theory.get("stage1_anchor", {})
        return value if isinstance(value, dict) else {}

    def downstream_af_backend_note(self):
        theory = self.meta.get("theory", {})
        if not isinstance(theory, dict):
            return {}
        value = theory.get("downstream_af_backend_note", {})
        return value if isinstance(value, dict) else {}

    def validate_stage1_ready(self, require_sr_compatible=False):
        issues = []
        if self._a0_grid is None:
            issues.append("missing a0 grid")
        if require_sr_compatible and self.spectral_frame != SUPPORTED_SPECTRAL_FRAME:
            issues.append(
                "spectral_frame is %r, expected %r"
                % (self.spectral_frame, SUPPORTED_SPECTRAL_FRAME)
            )
        if abs(float(self.theta_grid_rad[0])) > _FLOAT_TOL_RAD:
            issues.append("theta grid does not start near 0 rad")
        if float(self.theta_grid_rad[-1]) < 0.5 * math.pi - _FLOAT_TOL_RAD:
            issues.append("theta grid does not reach 90 deg")
        return {
            "ok": len(issues) == 0,
            "issues": issues,
            "spectral_frame": self.spectral_frame,
            "has_a0": self._a0_grid is not None,
            "theta_deg_range": [
                float(np.rad2deg(self.theta_grid_rad[0])),
                float(np.rad2deg(self.theta_grid_rad[-1])),
            ],
            "stage1_anchor": self.stage1_anchor_info(),
            "downstream_af_backend_note": self.downstream_af_backend_note(),
        }

    def summary(self):
        return {
            "npz_path": self.npz_path,
            "meta_json_path": self.meta_json_path,
            "npz_sha256": self.npz_sha256,
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "spectral_frame": self.spectral_frame,
            "n_theta": int(self.theta_grid_rad.size),
            "n_phi": int(self.phi_grid_rad.size),
            "theta_deg_range": [
                float(np.rad2deg(self.theta_grid_rad[0])),
                float(np.rad2deg(self.theta_grid_rad[-1])),
            ],
            "phi_deg_range": [
                float(np.rad2deg(self.phi_grid_rad[0])),
                float(np.rad2deg(self.phi_grid_rad[-1])),
            ],
            "co_handedness": self.co_handedness,
            "cx_handedness": self.cx_handedness,
            "spin_sign": self.spin_sign,
            "cp_basis_sign": self.cp_basis_sign,
            "normalization_A_ref": self.normalization_A_ref,
            "normalization_scale": self.normalization_scale,
            "source_csv_path": self.source_csv_path,
            "source_csv_sha256": self.source_csv_sha256,
            "variation_selected": self.variation_selected,
            "has_a0": True,
            "has_c0": True,
            "has_aq": True,
            "has_cq": True,
            "has_b0": False,
            "has_b_floor": False,
            "has_b_m2": False,
            "has_bq": False,
            "q_floor": None,
            "nQa": int(self._aq_q_list.size),
            "nQb": 0,
            "nQc": int(self._cq_q_list.size),
            "q_list_a": self._aq_q_list.tolist(),
            "q_list_b": None,
            "q_list_c": self._cq_q_list.tolist(),
            "bounds": self.config.bounds,
            "phi_main_deg": float(np.rad2deg(self.config.phi_main_rad)),
            "enable_bilinear_phi": bool(self.config.enable_bilinear_phi),
            "field_query_backend": "exact_sparse_spectral_reconstruction",
            "cx_fit_validated_theta_max_deg": self.cx_fit_validated_theta_max_deg,
            "stage1_has_usage_meta": bool(self.stage1_usage()),
            "stage1_anchor_field": self.stage1_anchor_info().get("field"),
            "stage1_recommended_2d_backend": self.downstream_af_backend_note().get("recommended_2d_backend"),
            "stage1_recommended_signoff_backend": self.downstream_af_backend_note().get("recommended_signoff_backend"),
        }

    def export_cache(self, out_npz):
        """Export a derived numerical cache, not a canonical schema-v2 dataset.

        This cache omits the paired metadata required by the strict IEP loader.
        Use the original preprocessing NPZ/JSON pair as the canonical input.
        """
        out_npz = os.path.abspath(str(out_npz))
        out_dir = os.path.dirname(out_npz)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        np.savez(
            out_npz,
            schema_name=np.array(self.schema_name),
            schema_version=np.array(self.schema_version, dtype=np.int64),
            spectral_frame=np.array(self.spectral_frame),
            theta_grid_rad=self.theta_grid_rad.astype(np.float64),
            phi_grid_rad=self.phi_grid_rad.astype(np.float64),
            phi_main_rad=np.array([float(self.config.phi_main_rad)], dtype=np.float64),
            g_main_grid=np.asarray(self.g_main(self.theta_grid_rad), dtype=np.float64),
            g_side_grid=self._g_side_grid.astype(np.float64),
            a0_grid=self._a0_grid.astype(np.complex128),
            a0_abs_grid=np.abs(self._a0_grid).astype(np.float64),
            c0_grid=self._c0_grid.astype(np.complex128),
            aq_q_list=self._aq_q_list.astype(np.int64),
            aq_grid=self._aq_sparse_grid.astype(np.complex128),
            aq_keep_mask=self._aq_keep_mask.astype(bool),
            cq_q_list=self._cq_q_list.astype(np.int64),
            cq_grid=self._cq_sparse_grid.astype(np.complex128),
            cq_keep_mask=self._cq_keep_mask.astype(bool),
        )
        return out_npz


def export_cache(model, out_npz):
    return model.export_cache(out_npz)


def _run_self_test(model):
    print("=== IEP MODEL SELF TEST: START ===")
    print("- schema: %s v%d" % (model.schema_name, model.schema_version))
    print("- spectral_frame: %s" % model.spectral_frame)
    print("- polarization: co=%s cx=%s spin_sign=%d cp_basis_sign=%d" % (
        model.co_handedness,
        model.cx_handedness,
        model.spin_sign,
        model.cp_basis_sign,
    ))
    print("- grid: n_theta=%d n_phi=%d theta=[%.6f,%.6f] deg phi=[%.6f,%.6f] deg" % (
        model.theta_grid_rad.size,
        model.phi_grid_rad.size,
        np.rad2deg(model.theta_grid_rad[0]),
        np.rad2deg(model.theta_grid_rad[-1]),
        np.rad2deg(model.phi_grid_rad[0]),
        np.rad2deg(model.phi_grid_rad[-1]),
    ))
    print("- aq_q_list=%s" % model.aq_q_list.tolist())
    print("- cq_q_list=%s" % model.cq_q_list.tolist())

    theta_probe = model.theta_grid_rad[[0, model.theta_grid_rad.size // 2, -1]]
    err_a0 = float(np.max(np.abs(model.aq(theta_probe, 0) - model.a0(theta_probe))))
    err_c0 = float(np.max(np.abs(model.cq(theta_probe, 0) - model.c0(theta_probe))))
    if err_a0 > 1e-12 or err_c0 > 1e-12:
        raise AssertionError("q=0 anchor consistency failed.")
    eco = model.query_E_co(theta_probe, np.zeros(theta_probe.shape))
    ecx = model.query_E_cx(theta_probe, np.zeros(theta_probe.shape))
    if not _all_finite_complex(eco) or not _all_finite_complex(ecx):
        raise AssertionError("Field query returned NaN or Inf.")
    ready = model.validate_stage1_ready(require_sr_compatible=True)
    if not ready["ok"]:
        raise AssertionError("IEP readiness failed: %s" % ready["issues"])
    print("- max|aq(q=0)-a0|=%.3e" % err_a0)
    print("- max|cq(q=0)-c0|=%.3e" % err_c0)
    print("- npz_sha256=%s" % model.npz_sha256)
    print("IEP_MODEL_SELF_TEST: PASS")


def main():
    parser = argparse.ArgumentParser(
        description="Strict IEP field model for the hfss_sr_sparse schema v2."
    )
    parser.add_argument("--npz", required=True, help="Path to hfss_sr_sparse.npz")
    parser.add_argument(
        "--json",
        "--meta",
        dest="json_path",
        default=None,
        help="Path to hfss_sr_sparse.json; defaults to same stem as NPZ.",
    )
    parser.add_argument(
        "--bounds",
        default="error",
        choices=["error", "clamp", "nan"],
        help="Theta out-of-bounds policy.",
    )
    parser.add_argument("--phi-main-deg", type=float, default=0.0)
    parser.add_argument("--no-verify-hash", action="store_true")
    args = parser.parse_args()

    cfg = IEPModelConfig(
        phi_main_rad=float(np.deg2rad(args.phi_main_deg)),
        bounds=args.bounds,
        enable_bilinear_phi=True,
    )
    model = IEPModel(
        npz_path=args.npz,
        meta_json_path=args.json_path,
        config=cfg,
        verify_hash=(not args.no_verify_hash),
    )
    print("=== IEP MODEL SUMMARY ===")
    for key, value in model.summary().items():
        print("- %s: %s" % (key, value))
    _run_self_test(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
