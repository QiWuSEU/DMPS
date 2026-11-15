# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import csv
import datetime as _dt
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np


# -----------------------------------------------------------------------------
# Stable project defaults for the portable DMPS repository
# -----------------------------------------------------------------------------

DEFAULT_MODULE = os.path.join("geometric_backbone", "ring_geometry_optimizer.py")
DEFAULT_IEP_REL_DIR = os.path.join(
    "data",
    "case_i_iep",
)
DEFAULT_IEP_NPZ = "hfss_sr_sparse.npz"
DEFAULT_IEP_JSON = "hfss_sr_sparse.json"
DEFAULT_PARTITION_REL = os.path.join(
    "data",
    "partition_reference",
    "q0cra_stage1_partition_table.json",
)
DEFAULT_OUTPUT_ROOT = "stage1_runs"
DEFAULT_SWEEP_ROOT = "stage1_sweeps"
ALPHA_TABLE_NAMES = (
    "a0cra_alpha_main_table.json",
    "a0cra_mainlobe_runtime_table.json",
    "alpha_main_table.json",
)

# Single-case defaults.
DEFAULT_K = 8
DEFAULT_D_MIN = 0.5
DEFAULT_R0 = 5.0
DEFAULT_B = 6.0
DEFAULT_GAMMA_MAIN = 0.95

# Full-scan defaults.
DEFAULT_THETA_START = 0.0
DEFAULT_THETA_STOP = 60.0
DEFAULT_THETA_STEP = 1.0

# New main-loop defaults. Keep these aligned with Stage1MainConfig.
DEFAULT_MAX_ITERS = 20
DEFAULT_MAX_FULL_SCOUTS = 20
DEFAULT_MAX_TOPOLOGY_EPOCHS = 12
DEFAULT_SIGMA_INIT = 0.15
DEFAULT_SIGMA_MIN = 0.01
DEFAULT_SIGMA_MAX = 0.30
DEFAULT_SIGMA_EXPAND = 1.20
DEFAULT_SIGMA_SHRINK = 0.50
DEFAULT_SIGMA_EXPAND_IMPROVEMENT_DB = 0.05
DEFAULT_WORK_ACCEPT_TOL_DB = 1e-4
DEFAULT_FULL_SCOUT_EVERY_ACCEPTS = 2
DEFAULT_FULL_IMPROVE_TOL_DB = 0.02
DEFAULT_FULL_NO_IMPROVE_LIMIT = 3
DEFAULT_FULL_ROLLBACK_THRESHOLD_DB = 0.50
DEFAULT_CONSECUTIVE_REJECT_LIMIT = 6
DEFAULT_SIGMA_MIN_REJECT_LIMIT = 2
DEFAULT_TOPOLOGY_CYCLE_WINDOW = 4
DEFAULT_TOPOLOGY_CYCLE_FULL_TOL_DB = 0.02

# Requested sweep defaults retained from the earlier auto-run workflow.
DEFAULT_K_LIST = "6,7,8"
DEFAULT_D_MIN_LIST = "0.50:0.60:0.01"
DEFAULT_B_LIST = "6.00:12.00:0.25"


EXIT_OK = 0
EXIT_INPUT = 2
EXIT_TABLE = 3
EXIT_API = 4
EXIT_MOSEK = 5
EXIT_OPTIMIZATION = 6
EXIT_VALIDATION = 7
EXIT_OUTPUT = 8


class RunnerError(Exception):
    def __init__(self, message, exit_code=EXIT_INPUT, detail=None):
        Exception.__init__(self, message)
        self.exit_code = int(exit_code)
        self.detail = dict(detail or {})


class DriverLogger(object):
    def __init__(self, path, echo=True):
        self.path = str(path)
        self.echo = bool(echo)
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        self._file = open(self.path, "w", encoding="utf-8")

    def write(self, message):
        text = str(message)
        if self.echo:
            sys.__stdout__.write(text + "\n")
            sys.__stdout__.flush()
        self._file.write(text + "\n")
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _now_tag():
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _float_tag(value):
    text = ("%.8g" % float(value)).replace("-", "m").replace(".", "p")
    return text


def _case_name(K, d_min, b):
    return "K%d_d%s_b%s" % (int(K), _float_tag(d_min), _float_tag(b))


def _atomic_write_json(path, payload):
    path = str(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
    os.replace(tmp, path)


def _read_json(path):
    with open(str(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _require_file(path, label, exit_code=EXIT_INPUT):
    if not os.path.isfile(str(path)):
        raise RunnerError(
            "%s not found: %s" % (str(label), str(path)),
            exit_code=exit_code,
            detail={"label": str(label), "path": str(path)},
        )


def _as_abs_path(base_dir, value):
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = Path(base_dir) / p
    return p.resolve()


def _is_close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= float(tol)


def _partition_entry_key(entry):
    nested = entry.get("key") if isinstance(entry, dict) else None
    if isinstance(nested, dict):
        return nested
    return entry if isinstance(entry, dict) else {}


def _parse_solver_order(text):
    values = [x.strip().upper() for x in str(text).split(",") if x.strip()]
    if not values:
        raise RunnerError("solver order must contain at least one solver")
    return tuple(values)


def _parse_int_list(text):
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise RunnerError("integer list is empty: %s" % text)
    return values


def _decimal_range(start, stop, step):
    start = float(start)
    stop = float(stop)
    step = float(step)
    if step <= 0.0:
        raise RunnerError("range step must be > 0")
    n = int(round((stop - start) / step))
    if n < 0:
        raise RunnerError("range stop must be >= start")
    values = []
    for i in range(n + 1):
        value = start + i * step
        if value <= stop + 1e-10:
            values.append(float(round(value, 12)))
    return values


def _parse_float_list(text):
    text = str(text).strip()
    if not text:
        raise RunnerError("float list is empty")
    if ":" in text and "," not in text:
        parts = [x.strip() for x in text.split(":")]
        if len(parts) != 3:
            raise RunnerError("range syntax must be start:stop:step: %s" % text)
        return _decimal_range(float(parts[0]), float(parts[1]), float(parts[2]))
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise RunnerError("float list is empty: %s" % text)
    return values


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Run the rewritten A0CRA/Q0CRA Stage-1 main loop with runtime "
            "partition-entry pool generation, compact logging and final audits."
        )
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--single-case", action="store_true", help="Run one case. This is the default mode.")
    mode.add_argument("--sweep", action="store_true", help="Run a K/d_min/b sweep in isolated subprocesses.")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    preset = p.add_mutually_exclusive_group()
    preset.add_argument("--trial", action="store_true", help="Controlled 6-attempt trial preset.")
    preset.add_argument("--full-run", action="store_true", help="Full 20-attempt preset.")

    p.add_argument("--dry-run", action="store_true", help="Perform all preflight checks without starting optimization.")
    p.add_argument("--continue-on-error", action="store_true", help="Continue a sweep after a failed case.")
    p.add_argument("--max-cases", type=int, default=0, help="Optional sweep case limit; 0 means no limit.")

    # Paths.
    p.add_argument("--base-dir", default="", help="Project directory. Defaults to this script directory.")
    p.add_argument("--module", default=DEFAULT_MODULE, help="Main module filename or path.")
    p.add_argument("--iep-dir", default="", help="Directory containing hfss_sr_sparse.npz/json.")
    p.add_argument("--iep-npz", default="", help="Explicit IEP NPZ path.")
    p.add_argument("--iep-json", default="", help="Explicit IEP JSON path.")
    p.add_argument("--partition-table-json", default="", help="Explicit partition table JSON path.")
    p.add_argument("--alpha-main-table-json", default="", help="Explicit alpha-main table JSON path.")
    p.add_argument("--output-root", default="", help="Root directory for timestamped single-case outputs.")
    p.add_argument("--output-dir", default="", help="Exact output directory; primarily used by sweep workers.")

    # Case.
    p.add_argument("--K", type=int, default=DEFAULT_K)
    p.add_argument("--d-min", type=float, default=DEFAULT_D_MIN)
    p.add_argument("--d-eff", type=float, default=None, help="Defaults to d_min.")
    p.add_argument("--b", type=float, default=DEFAULT_B)
    p.add_argument("--R0", type=float, default=DEFAULT_R0)
    p.add_argument("--gamma-main", type=float, default=DEFAULT_GAMMA_MAIN)

    # Sweep lists.
    p.add_argument("--K-list", default=DEFAULT_K_LIST)
    p.add_argument("--d-min-list", default=DEFAULT_D_MIN_LIST)
    p.add_argument("--b-list", default=DEFAULT_B_LIST)

    # Scan grid.
    p.add_argument("--theta-full-start", type=float, default=DEFAULT_THETA_START)
    p.add_argument("--theta-full-stop", type=float, default=DEFAULT_THETA_STOP)
    p.add_argument("--theta-full-step", type=float, default=DEFAULT_THETA_STEP)

    # New loop configuration. None means use preset/default resolution.
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--max-full-scouts", type=int, default=None)
    p.add_argument("--max-topology-epochs", type=int, default=None)
    p.add_argument("--sigma-init", type=float, default=DEFAULT_SIGMA_INIT)
    p.add_argument("--sigma-min", type=float, default=DEFAULT_SIGMA_MIN)
    p.add_argument("--sigma-max", type=float, default=DEFAULT_SIGMA_MAX)
    p.add_argument("--sigma-expand", type=float, default=DEFAULT_SIGMA_EXPAND)
    p.add_argument("--sigma-shrink", type=float, default=DEFAULT_SIGMA_SHRINK)
    p.add_argument("--sigma-expand-improvement-db", type=float, default=DEFAULT_SIGMA_EXPAND_IMPROVEMENT_DB)
    p.add_argument("--work-accept-tol-db", type=float, default=DEFAULT_WORK_ACCEPT_TOL_DB)
    p.add_argument("--full-scout-every-accepts", type=int, default=DEFAULT_FULL_SCOUT_EVERY_ACCEPTS)
    p.add_argument("--full-improve-tol-db", type=float, default=DEFAULT_FULL_IMPROVE_TOL_DB)
    p.add_argument("--full-no-improve-limit", type=int, default=DEFAULT_FULL_NO_IMPROVE_LIMIT)
    p.add_argument("--full-rollback-threshold-db", type=float, default=DEFAULT_FULL_ROLLBACK_THRESHOLD_DB)
    p.add_argument("--consecutive-reject-limit", type=int, default=DEFAULT_CONSECUTIVE_REJECT_LIMIT)
    p.add_argument("--sigma-min-reject-limit", type=int, default=DEFAULT_SIGMA_MIN_REJECT_LIMIT)
    p.add_argument("--topology-cycle-window", type=int, default=DEFAULT_TOPOLOGY_CYCLE_WINDOW)
    p.add_argument("--topology-cycle-full-tol-db", type=float, default=DEFAULT_TOPOLOGY_CYCLE_FULL_TOL_DB)

    # Mainlobe policy.
    p.add_argument("--mainlobe-source", choices=("auto", "table_raw_b", "directsum"), default="auto")
    p.add_argument("--no-directsum-fallback", action="store_true", help="In auto mode, fail rather than fall back to directsum.")

    # Solver policy.
    p.add_argument("--solver-order", default="MOSEK,ECOS,SCS")
    p.add_argument("--solver-verbose", action="store_true")
    p.add_argument("--require-mosek", action="store_true")

    # Logging/output.
    p.add_argument("--log-level", choices=("silent", "compact", "debug"), default="compact")
    p.add_argument("--no-log-file", action="store_true")
    p.add_argument("--no-attempt-csv", action="store_true")
    p.add_argument("--no-full-csv", action="store_true")
    p.add_argument("--no-history-json", action="store_true")

    return p.parse_args(argv)


def _resolve_limits(args):
    if args.trial:
        preset = (6, 10, 6)
    else:
        preset = (DEFAULT_MAX_ITERS, DEFAULT_MAX_FULL_SCOUTS, DEFAULT_MAX_TOPOLOGY_EPOCHS)
    max_iters = preset[0] if args.max_iters is None else int(args.max_iters)
    max_full = preset[1] if args.max_full_scouts is None else int(args.max_full_scouts)
    max_epochs = preset[2] if args.max_topology_epochs is None else int(args.max_topology_epochs)
    return max_iters, max_full, max_epochs


# -----------------------------------------------------------------------------
# Path resolution and table validation
# -----------------------------------------------------------------------------


def _resolve_paths(args):
    script_dir = Path(__file__).resolve().parent
    base_dir = _as_abs_path(script_dir, args.base_dir) if args.base_dir else script_dir

    module_path = _as_abs_path(base_dir, args.module)

    if args.iep_npz:
        iep_npz = _as_abs_path(base_dir, args.iep_npz)
    else:
        iep_dir = _as_abs_path(base_dir, args.iep_dir) if args.iep_dir else (base_dir / DEFAULT_IEP_REL_DIR)
        iep_npz = (iep_dir / DEFAULT_IEP_NPZ).resolve()

    if args.iep_json:
        iep_json = _as_abs_path(base_dir, args.iep_json)
    else:
        iep_json = (iep_npz.parent / DEFAULT_IEP_JSON).resolve()

    if args.partition_table_json:
        partition_json = _as_abs_path(base_dir, args.partition_table_json)
    else:
        partition_json = (base_dir / DEFAULT_PARTITION_REL).resolve()

    output_root = _as_abs_path(base_dir, args.output_root) if args.output_root else (base_dir / DEFAULT_OUTPUT_ROOT).resolve()

    return {
        "base_dir": base_dir,
        "module": module_path,
        "iep_npz": iep_npz,
        "iep_json": iep_json,
        "partition_json": partition_json,
        "output_root": output_root,
    }


def _validate_partition_entry(path, K, d_min, d_eff, R0, tol=1e-9):
    data = _read_json(path)
    entries = list(data.get("entries", []))
    matches = []
    for index, entry in enumerate(entries):
        key = _partition_entry_key(entry)
        try:
            matched = (
                int(key.get("K")) == int(K)
                and _is_close(key.get("d_min"), d_min, tol)
                and _is_close(key.get("d_eff"), d_eff, tol)
                and _is_close(key.get("R0"), R0, tol)
            )
        except Exception:
            matched = False
        if matched:
            matches.append((index, entry))
    if len(matches) != 1:
        available = []
        for entry in entries:
            key = _partition_entry_key(entry)
            available.append({
                "K": key.get("K"),
                "d_min": key.get("d_min"),
                "d_eff": key.get("d_eff"),
                "R0": key.get("R0"),
            })
        raise RunnerError(
            "Expected exactly one partition entry for K=%s d_min=%s d_eff=%s R0=%s; found %d"
            % (K, d_min, d_eff, R0, len(matches)),
            exit_code=EXIT_TABLE,
            detail={"available_entries": available},
        )
    index, entry = matches[0]
    return {"index": int(index), "entry": entry, "table_version": data.get("version", data.get("format", ""))}


def _alpha_entry_matches(data, K, d_min, d_eff, R0, b, theta_required, tol=1e-9):
    if str(data.get("format", "")) != "a0cra_alpha_main_table_v1":
        return None
    for index, entry in enumerate(list(data.get("entries", []))):
        key = _partition_entry_key(entry)
        try:
            matched = (
                int(key.get("K")) == int(K)
                and _is_close(key.get("d_min"), d_min, tol)
                and _is_close(key.get("d_eff"), d_eff, tol)
                and _is_close(key.get("R0"), R0, tol)
                and bool(key.get("include_center", True))
            )
        except Exception:
            matched = False
        if not matched:
            continue

        theta_available = np.asarray(entry.get("theta_s_deg", []), dtype=np.float64).reshape(-1)
        theta_ok = True
        for theta in np.asarray(theta_required, dtype=np.float64).reshape(-1).tolist():
            if not np.any(np.isclose(theta_available, float(theta), atol=tol, rtol=0.0)):
                theta_ok = False
                break
        if not theta_ok:
            continue

        b_values = np.asarray(entry.get("debug", {}).get("b_values", []), dtype=np.float64).reshape(-1)
        if not np.any(np.isclose(b_values, float(b), atol=tol, rtol=0.0)):
            continue

        raw = dict(entry.get("debug", {}).get("raw", {}) or {})
        required_raw = (
            "alpha_null_deg_by_b_theta",
            "alpha_guard_null_deg_by_b_theta",
            "alpha_3db_deg_by_b_theta",
        )
        if not all(name in raw for name in required_raw):
            continue
        return {"index": int(index), "entry": entry}
    return None


def _candidate_alpha_tables(base_dir, explicit_path):
    if explicit_path:
        return [_as_abs_path(base_dir, explicit_path)]

    candidates = []
    seen = set()
    for name in ALPHA_TABLE_NAMES:
        direct = (Path(base_dir) / name).resolve()
        if direct.is_file():
            candidates.append(direct)
            seen.add(str(direct).lower())

    skip_dirs = set((".git", ".idea", "__pycache__", "stage1_runs", "stage1_sweeps"))
    for root, dirs, files in os.walk(str(base_dir)):
        dirs[:] = [name for name in dirs if name not in skip_dirs]
        for name in ALPHA_TABLE_NAMES:
            if name in files:
                path = (Path(root) / name).resolve()
                key = str(path).lower()
                if key not in seen:
                    candidates.append(path)
                    seen.add(key)
    return candidates


def _resolve_mainlobe_source(args, base_dir, K, d_min, d_eff, R0, b, theta_full):
    requested = str(args.mainlobe_source)
    if requested == "directsum":
        return {"source": "directsum", "alpha_table": "", "alpha_match": None, "fallback_used": False}

    candidates = _candidate_alpha_tables(base_dir, args.alpha_main_table_json)
    valid = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = _read_json(path)
            match = _alpha_entry_matches(data, K, d_min, d_eff, R0, b, theta_full)
        except Exception:
            match = None
        if match is not None:
            valid.append((path, match))

    if valid:
        valid.sort(key=lambda item: (len(str(item[0])), str(item[0]).lower()))
        return {
            "source": "table_raw_b",
            "alpha_table": str(valid[0][0]),
            "alpha_match": valid[0][1],
            "fallback_used": False,
        }

    if requested == "table_raw_b":
        raise RunnerError(
            "No alpha-main table entry matches K=%s d_min=%s d_eff=%s R0=%s b=%s and the requested scan grid"
            % (K, d_min, d_eff, R0, b),
            exit_code=EXIT_TABLE,
            detail={"candidates": [str(x) for x in candidates]},
        )

    if bool(args.no_directsum_fallback):
        raise RunnerError(
            "mainlobe-source=auto found no matching table and directsum fallback is disabled",
            exit_code=EXIT_TABLE,
            detail={"candidates": [str(x) for x in candidates]},
        )

    return {"source": "directsum", "alpha_table": "", "alpha_match": None, "fallback_used": True}


# -----------------------------------------------------------------------------
# Main-module and solver preflight
# -----------------------------------------------------------------------------


def _load_module(module_path):
    name = "a0cra_radii_main_auto_%s" % _now_tag()
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    if spec is None or spec.loader is None:
        raise RunnerError("Cannot load main module: %s" % module_path, exit_code=EXIT_API)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validate_main_api(module):
    required_symbols = (
        "RadiusInitSpec",
        "ElementInitConfig",
        "MainlobeDomainConfig",
        "Stage1A0PolicyConfig",
        "Stage1RadiiOptConfig",
        "Stage1MainConfig",
        "run_a0cra_stage1_main",
    )
    missing = [name for name in required_symbols if not hasattr(module, name)]
    if missing:
        raise RunnerError("Main module is missing symbols: %s" % missing, exit_code=EXIT_API)

    signature = inspect.signature(module.run_a0cra_stage1_main)
    required_params = (
        "unit_cell_npz_path",
        "meta_json_path",
        "radius_spec",
        "partition_table_json_path",
        "element_cfg",
        "mainlobe_cfg",
        "a0_cfg",
        "radii_opt_cfg",
        "main_cfg",
        "output_dir",
    )
    missing_params = [name for name in required_params if name not in signature.parameters]
    if missing_params:
        raise RunnerError(
            "Main API version mismatch; run_a0cra_stage1_main lacks: %s" % missing_params,
            exit_code=EXIT_API,
        )

    slots = set(getattr(module.Stage1MainConfig, "__slots__", ()))
    required_slots = set((
        "max_full_scouts",
        "max_topology_epochs",
        "full_rollback_threshold_db",
        "consecutive_reject_limit",
        "sigma_min_reject_limit",
        "log_level",
    ))
    missing_slots = sorted(required_slots.difference(slots))
    if missing_slots:
        raise RunnerError(
            "Stage1MainConfig is not the rewritten loop version; missing fields: %s" % missing_slots,
            exit_code=EXIT_API,
        )

    return {
        "run_signature": str(signature),
        "stage1_main_config_slots": sorted(slots),
    }


def _mosek_smoke_test():
    try:
        import cvxpy as cp
        import mosek
    except Exception as exc:
        raise RunnerError("MOSEK/CVXPY import failed: %s" % exc, exit_code=EXIT_MOSEK)

    installed = [str(x).upper() for x in cp.installed_solvers()]
    if "MOSEK" not in installed:
        raise RunnerError("CVXPY does not list MOSEK; installed=%s" % installed, exit_code=EXIT_MOSEK)

    try:
        x = cp.Variable(2)
        t = cp.Variable()
        problem = cp.Problem(cp.Minimize(t), [cp.norm(x, 2) <= t, x[0] >= 1.0, x[1] >= 2.0])
        problem.solve(solver="MOSEK", verbose=False)
        solver_name = str(getattr(problem.solver_stats, "solver_name", "")).upper()
        if str(problem.status) not in ("optimal", "optimal_inaccurate") or solver_name != "MOSEK":
            raise RuntimeError("status=%s solver=%s" % (problem.status, solver_name))
    except Exception as exc:
        raise RunnerError("MOSEK license/solve smoke test failed: %s" % exc, exit_code=EXIT_MOSEK)

    return {
        "cvxpy_version": str(cp.__version__),
        "mosek_version": list(mosek.Env().getversion()),
        "installed_solvers": installed,
        "smoke_status": str(problem.status),
        "smoke_solver": solver_name,
    }


# -----------------------------------------------------------------------------
# Case construction, run and validation
# -----------------------------------------------------------------------------


def _validate_numeric_config(args, K, d_min, d_eff, R0, max_iters, max_full, max_epochs):
    issues = []
    if int(K) < 1:
        issues.append("K must be >= 1")
    if float(d_min) <= 0.0 or float(d_eff) <= 0.0:
        issues.append("d_min and d_eff must be > 0")
    if float(R0) < int(K) * float(d_min) - 1e-12:
        issues.append("R0 must be >= K*d_min")
    if float(args.theta_full_step) <= 0.0:
        issues.append("theta-full-step must be > 0")
    if float(args.theta_full_stop) < float(args.theta_full_start):
        issues.append("theta-full-stop must be >= theta-full-start")
    if not (float(args.sigma_min) <= float(args.sigma_init) <= float(args.sigma_max)):
        issues.append("sigma_min <= sigma_init <= sigma_max is required")
    if int(max_iters) < 0 or int(max_full) < 1 or int(max_epochs) < 0:
        issues.append("invalid iteration/full/topology limits")
    if issues:
        raise RunnerError("Invalid case configuration: %s" % "; ".join(issues), exit_code=EXIT_INPUT)


def _build_output_dir(args, paths, K, d_min, b):
    if args.output_dir:
        output_dir = _as_abs_path(paths["base_dir"], args.output_dir)
    else:
        root = Path(paths["output_root"])
        output_dir = root / ("%s_%s" % (_case_name(K, d_min, b), _now_tag()))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def _build_configs(module, args, case, mainlobe, max_iters, max_full, max_epochs):
    theta_full = case["theta_full"]
    radius_spec = module.RadiusInitSpec(
        name=case["case_name"],
        method="gompertz",
        K=case["K"],
        R0=case["R0"],
        d_min=case["d_min"],
        b=case["b"],
    )
    element_cfg = module.ElementInitConfig(
        d_elem_default_lambda=case["d_eff"],
        include_center=True,
        apply_outer5_mod4_minus1=True,
        outer5_threshold_gt=16,
    )
    mainlobe_cfg = module.MainlobeDomainConfig(
        theta_s_deg_list=theta_full,
        phi_s_deg=0.0,
        use_domain="null_guard",
        source=mainlobe["source"],
        final_source="directsum",
        freeze_during_optimization=True,
        alpha_table_json_path=mainlobe["alpha_table"],
        alpha_table_tol=1e-9,
        mainlobe_margin_deg=0.5,
    )
    a0_cfg = module.Stage1A0PolicyConfig(build_mode="full", warmup_all_scans=True)
    solver_order = _parse_solver_order(args.solver_order)
    radii_opt_cfg = module.Stage1RadiiOptConfig(
        d_min=case["d_min"],
        sigma=float(args.sigma_init),
        gamma_main=float(args.gamma_main),
        lambda_dr=0.0,
        use_main_anchors=True,
        anchors_per_scan=1,
        anchor_merge_alpha_deg=0.5,
        radius_feasibility_tol=5e-6,
        solver_order=solver_order,
        solver_verbose=bool(args.solver_verbose),
        scipy_maxiter=400,
        scipy_ftol=1e-8,
    )
    main_cfg = module.Stage1MainConfig(
        max_iters=int(max_iters),
        max_full_scouts=int(max_full),
        max_topology_epochs=int(max_epochs),
        sigma_init=float(args.sigma_init),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        sigma_expand=float(args.sigma_expand),
        sigma_shrink=float(args.sigma_shrink),
        sigma_expand_improvement_db=float(args.sigma_expand_improvement_db),
        work_accept_tol_db=float(args.work_accept_tol_db),
        full_scout_every_accepts=int(args.full_scout_every_accepts),
        full_improve_tol_db=float(args.full_improve_tol_db),
        full_no_improve_limit=int(args.full_no_improve_limit),
        full_rollback_threshold_db=float(args.full_rollback_threshold_db),
        consecutive_reject_limit=int(args.consecutive_reject_limit),
        sigma_min_reject_limit=int(args.sigma_min_reject_limit),
        topology_cycle_window=int(args.topology_cycle_window),
        topology_cycle_full_tol_db=float(args.topology_cycle_full_tol_db),
        force_final_full_audit=True,
        log_level=str(args.log_level),
        log_to_file=not bool(args.no_log_file),
        save_attempt_csv=not bool(args.no_attempt_csv),
        save_full_csv=not bool(args.no_full_csv),
        save_history_json=not bool(args.no_history_json),
        verbose=True,
    )
    return {
        "radius_spec": radius_spec,
        "element_cfg": element_cfg,
        "mainlobe_cfg": mainlobe_cfg,
        "a0_cfg": a0_cfg,
        "radii_opt_cfg": radii_opt_cfg,
        "main_cfg": main_cfg,
        "solver_order": solver_order,
    }


def _validate_main_outputs(result, require_mosek, save_attempt_csv, save_full_csv, log_to_file):
    output_json = str(result.output_json_path)
    output_npz = str(result.output_npz_path)
    _require_file(output_json, "main output JSON", EXIT_OUTPUT)
    _require_file(output_npz, "main output NPZ", EXIT_OUTPUT)

    payload = _read_json(output_json)
    if str(payload.get("format", "")) != "a0cra_stage1_main_v3":
        raise RunnerError("Unexpected main output format: %s" % payload.get("format"), exit_code=EXIT_VALIDATION)

    validation = dict(payload.get("validation", {}) or {})
    final_solution = dict(payload.get("final_solution", {}) or {})
    required_true = (
        "canonical_N_match",
        "geometry_N_match",
        "final_full_audit_performed",
        "final_mainlobe_recomputed",
    )
    failed = [name for name in required_true if not bool(validation.get(name, False))]
    if not bool(final_solution.get("full_validated", False)):
        failed.append("final_solution.full_validated")
    if not bool(final_solution.get("canonical_N_match", False)):
        failed.append("final_solution.canonical_N_match")
    if not bool(final_solution.get("geometry_consistent", False)):
        failed.append("final_solution.geometry_consistent")
    if failed:
        raise RunnerError("Final validation failed: %s" % failed, exit_code=EXIT_VALIDATION)

    run_summary = dict(payload.get("run_summary", {}) or {})
    solver_usage = dict(run_summary.get("solver_usage", {}) or {})
    if require_mosek and int(solver_usage.get("MOSEK", 0)) <= 0:
        raise RunnerError("MOSEK was required but main output reports no MOSEK solves: %s" % solver_usage, exit_code=EXIT_MOSEK)

    optional_paths = []
    if save_attempt_csv:
        optional_paths.append((str(result.output_attempt_csv_path), "attempt CSV"))
    if save_full_csv:
        optional_paths.append((str(result.output_full_csv_path), "full CSV"))
    if log_to_file:
        optional_paths.append((str(result.output_log_path), "main run log"))
    for path, label in optional_paths:
        _require_file(path, label, EXIT_OUTPUT)

    return payload


def _case_payload(args, case, paths, mainlobe, partition_info, api_info, limits, mosek_info, output_dir):
    return {
        "format": "a0cra_stage1_auto_manifest_v2",
        "created_at": _dt.datetime.now().isoformat(),
        "python": {"executable": sys.executable, "version": sys.version},
        "case": {
            "name": case["case_name"],
            "K": case["K"],
            "R0": case["R0"],
            "d_min": case["d_min"],
            "d_eff": case["d_eff"],
            "bind_d_eff_to_d_min": bool(case["d_eff_bound"]),
            "gompertz_b": case["b"],
            "theta_full_start": float(args.theta_full_start),
            "theta_full_stop": float(args.theta_full_stop),
            "theta_full_step": float(args.theta_full_step),
        },
        "paths": {
            "base_dir": str(paths["base_dir"]),
            "module": str(paths["module"]),
            "iep_npz": str(paths["iep_npz"]),
            "iep_json": str(paths["iep_json"]),
            "partition_table_json": str(paths["partition_json"]),
            "alpha_main_table_json": str(mainlobe["alpha_table"]),
            "output_dir": str(output_dir),
        },
        "mainlobe": {
            "requested_source": str(args.mainlobe_source),
            "resolved_source": str(mainlobe["source"]),
            "directsum_fallback_used": bool(mainlobe["fallback_used"]),
        },
        "solver": {
            "order": list(_parse_solver_order(args.solver_order)),
            "require_mosek": bool(args.require_mosek),
            "preflight": mosek_info,
        },
        "limits": {
            "max_iters": int(limits[0]),
            "max_full_scouts": int(limits[1]),
            "max_topology_epochs": int(limits[2]),
        },
        "partition_entry": partition_info,
        "main_api": api_info,
        "command": list(sys.argv),
    }


def run_one_case(args):
    paths = _resolve_paths(args)
    K = int(args.K)
    d_min = float(args.d_min)
    d_eff = d_min if args.d_eff is None else float(args.d_eff)
    R0 = float(args.R0)
    b = float(args.b)
    limits = _resolve_limits(args)
    theta_full = np.arange(
        float(args.theta_full_start),
        float(args.theta_full_stop) + 0.5 * float(args.theta_full_step),
        float(args.theta_full_step),
        dtype=np.float64,
    )
    case = {
        "case_name": _case_name(K, d_min, b),
        "K": K,
        "d_min": d_min,
        "d_eff": d_eff,
        "d_eff_bound": args.d_eff is None,
        "R0": R0,
        "b": b,
        "theta_full": theta_full,
    }
    _validate_numeric_config(args, K, d_min, d_eff, R0, limits[0], limits[1], limits[2])

    output_dir = _build_output_dir(args, paths, K, d_min, b)
    driver_log_path = output_dir / "auto_run_driver.log"
    logger = DriverLogger(driver_log_path, echo=(args.log_level != "silent"))
    manifest_path = output_dir / "case_manifest.json"
    summary_path = output_dir / "auto_run_summary.json"
    error_path = output_dir / "auto_run_error.json"

    try:
        logger.write("=== A0CRA STAGE1 AUTO RUN: START ===")
        for key in ("module", "iep_npz", "iep_json", "partition_json"):
            _require_file(paths[key], key, EXIT_INPUT if key != "partition_json" else EXIT_TABLE)

        if str(paths["base_dir"]) not in sys.path:
            sys.path.insert(0, str(paths["base_dir"]))

        module = _load_module(paths["module"])
        api_info = _validate_main_api(module)
        partition_info = _validate_partition_entry(paths["partition_json"], K, d_min, d_eff, R0)
        mainlobe = _resolve_mainlobe_source(args, paths["base_dir"], K, d_min, d_eff, R0, b, theta_full)
        mosek_info = _mosek_smoke_test() if args.require_mosek else {"required": False}

        manifest = _case_payload(
            args, case, paths, mainlobe, partition_info, api_info,
            limits, mosek_info, output_dir,
        )
        _atomic_write_json(manifest_path, manifest)

        logger.write("[PREFLIGHT] Python=%s" % sys.executable)
        logger.write("[PREFLIGHT] module=%s" % paths["module"])
        logger.write("[PREFLIGHT] IEP NPZ=%s" % paths["iep_npz"])
        logger.write("[PREFLIGHT] IEP JSON=%s" % paths["iep_json"])
        logger.write("[PREFLIGHT] partition=%s entry=%d" % (paths["partition_json"], partition_info["index"]))
        logger.write("[PREFLIGHT] mainlobe source=%s" % mainlobe["source"])
        logger.write("[PREFLIGHT] solver order=%s" % ",".join(_parse_solver_order(args.solver_order)))
        if args.require_mosek:
            logger.write("[PREFLIGHT] MOSEK license=OK")
        logger.write("[PREFLIGHT] output=%s" % output_dir)
        logger.write("[PREFLIGHT] READY")

        if args.dry_run:
            summary = {
                "ok": True,
                "dry_run": True,
                "exit_code": EXIT_OK,
                "case": manifest["case"],
                "paths": manifest["paths"],
                "mainlobe": manifest["mainlobe"],
                "solver": manifest["solver"],
                "partition_entry": partition_info,
            }
            _atomic_write_json(summary_path, summary)
            logger.write("=== A0CRA STAGE1 AUTO RUN: DRY-RUN DONE ===")
            return EXIT_OK

        configs = _build_configs(module, args, case, mainlobe, limits[0], limits[1], limits[2])
        result = module.run_a0cra_stage1_main(
            unit_cell_npz_path=str(paths["iep_npz"]),
            meta_json_path=str(paths["iep_json"]),
            radius_spec=configs["radius_spec"],
            partition_table_json_path=str(paths["partition_json"]),
            element_cfg=configs["element_cfg"],
            mainlobe_cfg=configs["mainlobe_cfg"],
            a0_cfg=configs["a0_cfg"],
            radii_opt_cfg=configs["radii_opt_cfg"],
            main_cfg=configs["main_cfg"],
            iep_phi_main_rad=0.0,
            iep_bounds="clamp",
            iep_enable_bilinear_phi=True,
            pool_tol_deg=1e-9,
            output_dir=str(output_dir),
        )

        main_payload = _validate_main_outputs(
            result,
            require_mosek=bool(args.require_mosek),
            save_attempt_csv=not bool(args.no_attempt_csv),
            save_full_csv=not bool(args.no_full_csv),
            log_to_file=not bool(args.no_log_file),
        )

        summary = {
            "ok": True,
            "dry_run": False,
            "exit_code": EXIT_OK,
            "case": manifest["case"],
            "paths": {
                "manifest": str(manifest_path),
                "driver_log": str(driver_log_path),
                "main_json": str(result.output_json_path),
                "main_npz": str(result.output_npz_path),
                "attempt_csv": str(result.output_attempt_csv_path),
                "full_csv": str(result.output_full_csv_path),
                "main_log": str(result.output_log_path),
            },
            "run_summary": main_payload.get("run_summary", {}),
            "final_solution": main_payload.get("final_solution", {}),
            "validation": main_payload.get("validation", {}),
            "mainlobe": manifest["mainlobe"],
            "solver_policy": manifest["solver"],
        }
        _atomic_write_json(summary_path, summary)
        logger.write("[DONE] stop_reason=%s" % summary["run_summary"].get("stop_reason"))
        logger.write("[DONE] full PSLL=%.6f dB theta=%.3f" % (
            float(summary["final_solution"].get("full_worst_psll_db")),
            float(summary["final_solution"].get("full_worst_theta_s_deg")),
        ))
        logger.write("[DONE] main JSON=%s" % result.output_json_path)
        logger.write("=== A0CRA STAGE1 AUTO RUN: DONE ===")
        return EXIT_OK

    except RunnerError as exc:
        error_payload = {
            "ok": False,
            "exit_code": int(exc.exit_code),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "detail": dict(exc.detail),
            "traceback": traceback.format_exc(),
            "case": {"K": K, "d_min": d_min, "d_eff": d_eff, "R0": R0, "b": b},
            "paths": {key: str(value) for key, value in paths.items()},
        }
        _atomic_write_json(error_path, error_payload)
        logger.write("[FAILED] code=%d %s" % (exc.exit_code, exc))
        logger.write(error_payload["traceback"])
        return int(exc.exit_code)
    except Exception as exc:
        error_payload = {
            "ok": False,
            "exit_code": EXIT_OPTIMIZATION,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "case": {"K": K, "d_min": d_min, "d_eff": d_eff, "R0": R0, "b": b},
            "paths": {key: str(value) for key, value in paths.items()},
        }
        _atomic_write_json(error_path, error_payload)
        logger.write("[FAILED] code=%d %s: %s" % (EXIT_OPTIMIZATION, type(exc).__name__, exc))
        logger.write(error_payload["traceback"])
        return EXIT_OPTIMIZATION
    finally:
        logger.close()


# -----------------------------------------------------------------------------
# Sweep manager: one clean Python process per case
# -----------------------------------------------------------------------------


def _append_arg(command, name, value):
    command.extend([str(name), str(value)])


def _worker_command(args, case, output_dir):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--single-case"]
    if args.trial:
        command.append("--trial")
    elif args.full_run:
        command.append("--full-run")
    if args.dry_run:
        command.append("--dry-run")

    scalar_args = (
        ("--base-dir", args.base_dir), ("--module", args.module),
        ("--iep-dir", args.iep_dir), ("--iep-npz", args.iep_npz),
        ("--iep-json", args.iep_json), ("--partition-table-json", args.partition_table_json),
        ("--alpha-main-table-json", args.alpha_main_table_json),
        ("--output-dir", str(output_dir)),
        ("--K", case["K"]), ("--d-min", case["d_min"]),
        ("--d-eff", case["d_eff"]), ("--b", case["b"]), ("--R0", args.R0),
        ("--gamma-main", args.gamma_main),
        ("--theta-full-start", args.theta_full_start),
        ("--theta-full-stop", args.theta_full_stop),
        ("--theta-full-step", args.theta_full_step),
        ("--sigma-init", args.sigma_init), ("--sigma-min", args.sigma_min),
        ("--sigma-max", args.sigma_max), ("--sigma-expand", args.sigma_expand),
        ("--sigma-shrink", args.sigma_shrink),
        ("--sigma-expand-improvement-db", args.sigma_expand_improvement_db),
        ("--work-accept-tol-db", args.work_accept_tol_db),
        ("--full-scout-every-accepts", args.full_scout_every_accepts),
        ("--full-improve-tol-db", args.full_improve_tol_db),
        ("--full-no-improve-limit", args.full_no_improve_limit),
        ("--full-rollback-threshold-db", args.full_rollback_threshold_db),
        ("--consecutive-reject-limit", args.consecutive_reject_limit),
        ("--sigma-min-reject-limit", args.sigma_min_reject_limit),
        ("--topology-cycle-window", args.topology_cycle_window),
        ("--topology-cycle-full-tol-db", args.topology_cycle_full_tol_db),
        ("--mainlobe-source", args.mainlobe_source),
        ("--solver-order", args.solver_order), ("--log-level", args.log_level),
    )
    for name, value in scalar_args:
        if value not in (None, ""):
            _append_arg(command, name, value)

    for name, value in (
        ("--max-iters", args.max_iters),
        ("--max-full-scouts", args.max_full_scouts),
        ("--max-topology-epochs", args.max_topology_epochs),
    ):
        if value is not None:
            _append_arg(command, name, value)

    if args.no_directsum_fallback:
        command.append("--no-directsum-fallback")
    if args.solver_verbose:
        command.append("--solver-verbose")
    if args.require_mosek:
        command.append("--require-mosek")
    if args.no_log_file:
        command.append("--no-log-file")
    if args.no_attempt_csv:
        command.append("--no-attempt-csv")
    if args.no_full_csv:
        command.append("--no-full-csv")
    if args.no_history_json:
        command.append("--no-history-json")
    return command


def _read_case_result(case_dir, return_code):
    summary_path = Path(case_dir) / "auto_run_summary.json"
    error_path = Path(case_dir) / "auto_run_error.json"
    if summary_path.is_file():
        payload = _read_json(summary_path)
    elif error_path.is_file():
        payload = _read_json(error_path)
    else:
        payload = {"ok": False, "error": "worker produced no summary/error JSON"}
    return {
        "return_code": int(return_code),
        "case_dir": str(case_dir),
        "payload": payload,
    }


def run_sweep(args):
    paths = _resolve_paths(args)
    sweep_root_base = (Path(paths["base_dir"]) / DEFAULT_SWEEP_ROOT).resolve()
    if args.output_root:
        sweep_root_base = _as_abs_path(paths["base_dir"], args.output_root)
    sweep_dir = sweep_root_base / ("sweep_%s" % _now_tag())
    sweep_dir.mkdir(parents=True, exist_ok=True)

    K_values = _parse_int_list(args.K_list)
    d_values = _parse_float_list(args.d_min_list)
    b_values = _parse_float_list(args.b_list)
    cases = []
    for K in K_values:
        for d_min in d_values:
            for b in b_values:
                cases.append({"K": int(K), "d_min": float(d_min), "d_eff": float(d_min), "b": float(b)})
    if int(args.max_cases) > 0:
        cases = cases[:int(args.max_cases)]

    manifest = {
        "format": "a0cra_stage1_sweep_manifest_v2",
        "created_at": _dt.datetime.now().isoformat(),
        "case_count": len(cases),
        "cases": cases,
        "command": list(sys.argv),
    }
    _atomic_write_json(sweep_dir / "sweep_manifest.json", manifest)

    results = []
    for index, case in enumerate(cases):
        case_dir = sweep_dir / ("%04d_%s" % (index + 1, _case_name(case["K"], case["d_min"], case["b"])))
        command = _worker_command(args, case, case_dir)
        sys.__stdout__.write("[SWEEP %d/%d] %s\n" % (index + 1, len(cases), case_dir.name))
        sys.__stdout__.flush()
        proc = subprocess.run(command)
        result = _read_case_result(case_dir, proc.returncode)
        result["case"] = case
        results.append(result)
        if proc.returncode != 0 and not bool(args.continue_on_error):
            break

    rows = []
    for item in results:
        payload = item["payload"]
        final = dict(payload.get("final_solution", {}) or {})
        run_summary = dict(payload.get("run_summary", {}) or {})
        rows.append({
            "K": item["case"]["K"],
            "d_min": item["case"]["d_min"],
            "b": item["case"]["b"],
            "ok": bool(payload.get("ok", False)),
            "return_code": int(item["return_code"]),
            "stop_reason": run_summary.get("stop_reason", ""),
            "attempts": run_summary.get("attempts", ""),
            "accepts": run_summary.get("accepts", ""),
            "full_worst_psll_db": final.get("full_worst_psll_db", ""),
            "full_worst_theta_s_deg": final.get("full_worst_theta_s_deg", ""),
            "case_dir": item["case_dir"],
            "error": payload.get("error", ""),
        })

    csv_path = sweep_dir / "sweep_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    if fieldnames:
        with open(str(csv_path), "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "ok": all(bool(row["ok"]) for row in rows) and len(rows) == len(cases),
        "requested_cases": len(cases),
        "completed_cases": len(rows),
        "successful_cases": sum(1 for row in rows if bool(row["ok"])),
        "failed_cases": sum(1 for row in rows if not bool(row["ok"])),
        "results": rows,
        "sweep_dir": str(sweep_dir),
        "summary_csv": str(csv_path),
    }
    _atomic_write_json(sweep_dir / "sweep_summary.json", summary)
    return EXIT_OK if summary["ok"] else EXIT_OPTIMIZATION


def main(argv=None):
    args = parse_args(argv)
    if bool(args.sweep) and not bool(args.worker):
        return run_sweep(args)
    return run_one_case(args)


if __name__ == "__main__":
    sys.exit(main())
