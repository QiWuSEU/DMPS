# -*- coding: utf-8 -*-
"""
build_cutcorner_cra.py

One-click PyCharm runner for the complete 187-element cut-corner circularly
polarized circular-ring array selected by the frozen Stage1/Stage2 Top1 result.

Build policy
------------
- Read all 187 exact element-center coordinates from the frozen Stage1 NPZ.
- Read the Top1 per-ring physical-rotation law from the frozen Stage2 seeds JSON.
- Create the final-size square GND and square FR4 substrate before array groups.
- Build all 14 groups in sequence.  Every group creates a fresh complete MASTER
  element at the global origin, derives every group element only by Copy/Paste,
  then applies the exact Stage2 physical rotation and exact Stage1 translation.
- Immediately cut each completed group's GND apertures and substrate pin holes.
- Check, save, report, and checkpoint after each geometry group and board group.
- Resume only from checkpoints built with this Cut_L=3.60 mm script revision,
  without rebuilding passed groups.
- Immediately after each group geometry and board-cut transaction passes, assign
  that group's Patch Perfect-E boundary, grouped coax outer-conductor Perfect-E
  boundary, and that group's Modal wave ports; save, refresh the AEDT boundary tree,
  canonicalize modal excitation names, and checkpoint before the next group starts.
- After all 14 groups (187 elements) are committed, create one square AirBox, the global GND and
  Radiation boundaries, Setup1, and the 2.3-2.5 GHz discrete Sweep.
- Do not solve, create far-field spheres, or create result reports.
- Leave AEDT open after completion.

The file intentionally preloads Python's standard-library ``copy`` module so a
local recorded file named ``copy.py`` cannot shadow it during PyAEDT import.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import sysconfig
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def preload_stdlib_copy_module() -> None:
    """Prevent a local copy.py from shadowing Python's standard library."""
    existing = sys.modules.get("copy")
    if existing is not None and hasattr(existing, "deepcopy"):
        return

    stdlib_dir = sysconfig.get_paths().get("stdlib")
    if not stdlib_dir:
        raise RuntimeError("Cannot locate the Python standard-library directory.")

    copy_path = os.path.join(stdlib_dir, "copy.py")
    if not os.path.isfile(copy_path):
        raise RuntimeError(f"Cannot locate standard-library copy.py: {copy_path}")

    spec = importlib.util.spec_from_file_location("copy", copy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create import specification for: {copy_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "copy") or not hasattr(module, "deepcopy"):
        raise RuntimeError("Loaded copy module does not provide copy/deepcopy.")


# The user's HFSS project directory may still contain the recorded copy.py.
# Preload the real standard-library module before NumPy or PyAEDT is imported.
preload_stdlib_copy_module()

import numpy as np


# =============================================================================
# User configuration
# =============================================================================

AEDT_VERSION = "242"
DESIGN_NAME = "HFSSDesign1_LargeArray"
NON_GRAPHICAL = False
NEW_DESKTOP = True
CLOSE_ON_EXIT = False
RUN_MODE = "fresh"  # "auto", "fresh", or "resume"
SCRIPT_REVISION = "cutcorner_cra_cut360_v1"
# Old cut365 checkpoint revisions must not be resumed as cut360 geometry.
PREVIOUS_COMPATIBLE_REVISIONS = {SCRIPT_REVISION}

FREQ_GHZ = 2.4
C0_MM_PER_S = 299_792_458_000.0
LAMBDA0_MM = C0_MM_PER_S / (FREQ_GHZ * 1.0e9)

# Frozen Case-I inputs are discovered inside this checkout, regardless of the
# PyCharm working directory. Explicit environment overrides remain supported.
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DATA_ROOT = Path(os.environ.get("DMPS_STAGE_DATA_ROOT", str(REPO_ROOT)))


def discover_frozen_input(label: str, patterns: Sequence[Tuple[Path, str]]) -> Path:
    """Find the frozen Case-I artifact; reject conflicting repeated runs."""
    candidates: List[Path] = []
    for base, pattern in patterns:
        candidates.extend(path for path in base.glob(pattern) if path.is_file())
    candidates = sorted(set(candidates))
    if not candidates:
        # Keep module imports usable for offline smoke tests without Case-I
        # results; the normal load_stage_inputs() preflight will reject this.
        filename = "a0cra_stage1_main_out.npz" if "Stage-1" in label else "stage2_shortlist_seeds.json"
        return REPO_ROOT / "results" / "__missing_case1_input__" / filename
    if len(candidates) > 1:
        fingerprints = {hashlib.sha256(path.read_bytes()).hexdigest() for path in candidates}
        if len(fingerprints) != 1:
            raise RuntimeError(
                f"Conflicting {label} files; specify an explicit DMPS input path: "
                + ", ".join(str(path) for path in candidates)
            )
    # Multiple byte-identical copies are scientifically interchangeable.
    return candidates[-1]


STAGE1_NPZ_PATH = (
    Path(os.environ["DMPS_STAGE1_NPZ"])
    if os.environ.get("DMPS_STAGE1_NPZ")
    else discover_frozen_input("Case-I Stage-1 NPZ", (
        (REPO_ROOT, "results/case1_frozen_reproduction_*/stage1_paper_reproduction/"
                    "K7_d0p54_b11p75_*/a0cra_stage1_main_out.npz"),
        (PROJECT_DATA_ROOT, "stage1_runs/K7_d0p54_b11p75_*/a0cra_stage1_main_out.npz"),
    ))
)
STAGE2_SEEDS_JSON_PATH = (
    Path(os.environ["DMPS_STAGE2_SEEDS_JSON"])
    if os.environ.get("DMPS_STAGE2_SEEDS_JSON")
    else discover_frozen_input("Case-I Stage-2 seeds", (
        (REPO_ROOT, "stage2_K7_allrings_*/stage2_outputs/stage2_shortlist_seeds.json"),
        (PROJECT_DATA_ROOT, "stage2_K7_allrings_*/stage2_outputs/stage2_shortlist_seeds.json"),
        (REPO_ROOT, "results/case1_frozen_reproduction_*/stage2_K7_allrings_*/"
                    "stage2_outputs/stage2_shortlist_seeds.json"),
    ))
)

if (STAGE1_NPZ_PATH.parents[1].name == "stage1_paper_reproduction"
        and STAGE1_NPZ_PATH.parents[2].name.startswith("case1_frozen_reproduction_")):
    DEFAULT_BUILD_ROOT = STAGE1_NPZ_PATH.parents[2] / "hfss_case1_paper_reproduction"
else:
    DEFAULT_BUILD_ROOT = REPO_ROOT / "results" / "hfss_case1_paper_reproduction"

# A newly built cut360 project must never reuse the historical solved cut365 file.
HFSS_BASE_DIR = Path(os.environ.get("DMPS_HFSS_BUILD_ROOT", str(DEFAULT_BUILD_ROOT)))
OUTPUT_DIR = HFSS_BASE_DIR / "outputs"
PROJECT_PATH = OUTPUT_DIR / "cutcorner_large_array_top1_cut360_legacyenv.aedt"
STATE_DIR = OUTPUT_DIR / "cutcorner_large_array_top1_cut360_legacyenv_state"
MANIFEST_PATH = STATE_DIR / "build_manifest.json"
PLAN_CSV_PATH = STATE_DIR / "array_element_plan.csv"
LOG_PATH = STATE_DIR / "build_cutcorner_large_array.log"
SUMMARY_PATH = STATE_DIR / "build_summary.json"
GROUP_REPORT_DIR = STATE_DIR / "group_reports"
ERROR_REPORT_DIR = STATE_DIR / "error_reports"

# Full build: all 14 groups, including the center group, are processed.
BUILD_ALL_GROUPS = True

# Element geometry copied from the validated 2 x 2 script.
SUB_H_MM = 1.6
SUB_W_MM = 80.0
PATCH_L_MM = 28.9
PIN_R_MM = 0.635
TEFLON_R_MM = 2.05
CONN_L_MM = 5.0
FEED_POS_MM = 7.25
CUT_L_MM = 3.60
BOARD_EDGE_MARGIN_MM = 144.6589285745182
GND_NAME = "GND_Array"
SUBSTRATE_NAME = "Substrate_Array"
GND_BACKUP_NAME = "GND_Array_BACKUP"
SUBSTRATE_BACKUP_NAME = "Substrate_Array_BACKUP"
AIRBOX_NAME = "AirBox_Array"
AIR_MARGIN_XY_MM = 250.0
AIR_MARGIN_TOP_MM = 250.0
COMMON_GND_BOUNDARY = "PerfE_GND_Array"
LEGACY_COMMON_PATCH_BOUNDARY = "PerfE_Patches_All"
RADIATION_BOUNDARY = "Rad1"
SETUP_NAME = "Setup1"
SWEEP_NAME = "Sweep"

POSITION_TOL_MM = 2.0e-3
FACE_AREA_REL_TOL = 0.05

VARIABLES: List[Tuple[str, str]] = [
    ("freq_", "2.4GHz"),
    ("Sub_h", "1.6mm"),
    ("Sub_W", "80mm"),
    ("Patch_L", "28.9mm"),
    ("Pin_R", "0.635mm"),
    ("Teflon_R", "2.05mm"),
    ("Conn_L", "5mm"),
    ("Feed_Pos", "7.25mm"),
    ("Cut_L", "3.60mm"),
]

MASTER_OBJECTS: Tuple[str, ...] = (
    "MASTER_Patch",
    "MASTER_Pin",
    "MASTER_Coax_Teflon",
    "MASTER_PortSheet",
    "MASTER_GND_Hole",
)

FINAL_SUFFIXES: Tuple[str, ...] = (
    "Patch",
    "Pin",
    "Coax_Teflon",
    "PortSheet",
    "GND_Hole",
)

# Full ring grouping. Coordinates within every ring retain their exact Stage1
# NPZ order; they are not regenerated or rotated to force an axis alignment.
RING_GROUP_SIZES: Dict[int, List[int]] = {
    1: [6],
    2: [12],
    3: [18],
    4: [13, 12],
    5: [16, 15],
    6: [13, 13, 13],
    7: [19, 18, 18],
}


# =============================================================================
# Logging and JSON helpers
# =============================================================================


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(message: str = "") -> None:
    text = str(message)
    print(text, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")


def header(title: str) -> None:
    log()
    log("=" * 112)
    log(title)
    log("=" * 112)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


# =============================================================================
# Python/PyAEDT import protection
# =============================================================================


def import_pyaedt():
    preload_stdlib_copy_module()
    import ansys.aedt.core as aedt_core
    from ansys.aedt.core import Hfss

    return Hfss, "ansys.aedt.core", getattr(aedt_core, "__version__", "unknown")


# =============================================================================
# Input loading and exact plan construction
# =============================================================================


def normalize_angle_rad(value: float) -> float:
    return float(value) % (2.0 * math.pi)


def load_stage_inputs() -> Dict[str, Any]:
    if not STAGE1_NPZ_PATH.is_file():
        raise FileNotFoundError(f"Stage1 NPZ not found: {STAGE1_NPZ_PATH}")
    if not STAGE2_SEEDS_JSON_PATH.is_file():
        raise FileNotFoundError(f"Stage2 seeds JSON not found: {STAGE2_SEEDS_JSON_PATH}")

    with np.load(STAGE1_NPZ_PATH, allow_pickle=False) as data:
        required = {"r_best", "N_best", "x_elem_best", "y_elem_best"}
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(f"Stage1 NPZ is missing arrays: {missing}")
        r_best = np.asarray(data["r_best"], dtype=float).reshape(-1)
        n_best = np.asarray(data["N_best"], dtype=int).reshape(-1)
        x_best = np.asarray(data["x_elem_best"], dtype=float).reshape(-1)
        y_best = np.asarray(data["y_elem_best"], dtype=float).reshape(-1)

    seeds = read_json(STAGE2_SEEDS_JSON_PATH)
    top3 = seeds.get("top3")
    if not isinstance(top3, list) or not top3:
        raise RuntimeError("Stage2 seeds JSON does not contain a non-empty top3 list.")
    top1 = top3[0]
    if not isinstance(top1, dict):
        raise RuntimeError("Stage2 top3[0] is not a JSON object.")
    if top1.get("feasible") is False:
        raise RuntimeError("Stage2 Top1 is explicitly marked infeasible.")

    m_values = np.asarray(top1.get("m_json"), dtype=int).reshape(-1)
    delta_values = np.asarray(top1.get("delta_json"), dtype=float).reshape(-1)

    return {
        "r_best": r_best,
        "n_best": n_best,
        "x_best": x_best,
        "y_best": y_best,
        "m_values": m_values,
        "delta_values": delta_values,
        "top1": top1,
        "stage1_sha256": sha256_file(STAGE1_NPZ_PATH),
        "stage2_sha256": sha256_file(STAGE2_SEEDS_JSON_PATH),
    }


def validate_stage_inputs(inputs: Dict[str, Any]) -> None:
    r_best = inputs["r_best"]
    n_best = inputs["n_best"]
    x_best = inputs["x_best"]
    y_best = inputs["y_best"]
    m_values = inputs["m_values"]
    delta_values = inputs["delta_values"]

    if r_best.size != 7 or n_best.size != 7:
        raise RuntimeError(f"Expected K=7, got r={r_best.size}, N={n_best.size}")
    if m_values.size != 7 or delta_values.size != 7:
        raise RuntimeError(
            f"Expected seven Stage2 m/delta values, got m={m_values.size}, delta={delta_values.size}"
        )
    expected_elements = 1 + int(np.sum(n_best))
    if x_best.size != expected_elements or y_best.size != expected_elements:
        raise RuntimeError(
            f"Coordinate count mismatch: expected {expected_elements}, "
            f"x={x_best.size}, y={y_best.size}"
        )
    for name, values in (
        ("r_best", r_best),
        ("x_elem_best", x_best),
        ("y_elem_best", y_best),
        ("delta", delta_values),
    ):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Non-finite values found in {name}")
    if int(n_best[0]) != 6 or list(map(int, n_best)) != [6, 12, 18, 25, 31, 39, 55]:
        raise RuntimeError(f"Unexpected N_best: {n_best.tolist()}")

    center_radius = math.hypot(float(x_best[0]), float(y_best[0]))
    if center_radius > 1.0e-9:
        raise RuntimeError(f"Stage1 element index 0 is not the center: radius={center_radius}")

    cursor = 1
    for ring_index, (radius, count) in enumerate(zip(r_best, n_best), start=1):
        count_i = int(count)
        xs = x_best[cursor : cursor + count_i]
        ys = y_best[cursor : cursor + count_i]
        radial = np.hypot(xs, ys)
        max_error = float(np.max(np.abs(radial - float(radius))))
        if max_error > 1.0e-7:
            raise RuntimeError(
                f"Stage1 ring {ring_index} coordinate order/radius check failed: "
                f"max radial error={max_error} lambda"
            )
        cursor += count_i


def build_full_element_plan(inputs: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    r_best = inputs["r_best"]
    n_best = inputs["n_best"]
    x_best = inputs["x_best"]
    y_best = inputs["y_best"]
    m_values = inputs["m_values"]
    delta_values = inputs["delta_values"]

    elements: List[Dict[str, Any]] = []
    elements.append(
        {
            "global_index": 0,
            "unit_name": "U000",
            "ring_index": 0,
            "ring_local_index": 0,
            "x_lambda": float(x_best[0]),
            "y_lambda": float(y_best[0]),
            "x_mm": float(x_best[0] * LAMBDA0_MM),
            "y_mm": float(y_best[0] * LAMBDA0_MM),
            "phi_rad": 0.0,
            "phi_deg": 0.0,
            "rotation_rad": 0.0,
            "rotation_deg": 0.0,
            "port_name": "p001",
            "group_id": "C00G01",
        }
    )

    cursor = 1
    global_index = 1
    ring_elements: Dict[int, List[Dict[str, Any]]] = {}
    for ring_index, count in enumerate(n_best, start=1):
        count_i = int(count)
        current: List[Dict[str, Any]] = []
        for local_index in range(count_i):
            x_lambda = float(x_best[cursor + local_index])
            y_lambda = float(y_best[cursor + local_index])
            phi_rad = normalize_angle_rad(math.atan2(y_lambda, x_lambda))
            rotation_rad = normalize_angle_rad(
                int(m_values[ring_index - 1]) * phi_rad + float(delta_values[ring_index - 1])
            )
            current.append(
                {
                    "global_index": global_index,
                    "unit_name": f"U{global_index:03d}",
                    "ring_index": ring_index,
                    "ring_local_index": local_index,
                    "x_lambda": x_lambda,
                    "y_lambda": y_lambda,
                    "x_mm": x_lambda * LAMBDA0_MM,
                    "y_mm": y_lambda * LAMBDA0_MM,
                    "phi_rad": phi_rad,
                    "phi_deg": math.degrees(phi_rad),
                    "rotation_rad": rotation_rad,
                    "rotation_deg": math.degrees(rotation_rad),
                    "port_name": f"p{global_index + 1:03d}",
                    "group_id": "",
                }
            )
            global_index += 1
        ring_elements[ring_index] = current
        cursor += count_i

    groups: List[Dict[str, Any]] = []
    for ring_index in range(1, 8):
        current = ring_elements[ring_index]
        offset = 0
        sizes = RING_GROUP_SIZES[ring_index]
        if sum(sizes) != len(current):
            raise RuntimeError(
                f"Ring {ring_index} group sizes {sizes} do not sum to {len(current)}"
            )
        for group_number, size in enumerate(sizes, start=1):
            group_id = f"R{ring_index:02d}G{group_number:02d}"
            chunk = current[offset : offset + size]
            for element in chunk:
                element["group_id"] = group_id
                elements.append(element)
            groups.append(
                {
                    "group_id": group_id,
                    "ring_index": ring_index,
                    "group_number": group_number,
                    "element_indices": [int(row["global_index"]) for row in chunk],
                    "element_count": len(chunk),
                }
            )
            offset += size

    groups.append(
        {
            "group_id": "C00G01",
            "ring_index": 0,
            "group_number": 1,
            "element_indices": [0],
            "element_count": 1,
        }
    )

    elements.sort(key=lambda row: int(row["global_index"]))
    if len(elements) != 187:
        raise RuntimeError(f"Full plan should contain 187 elements, got {len(elements)}")
    return elements, groups


def compute_square_board_dimensions(elements: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    max_abs_x = max(abs(float(row["x_mm"])) for row in elements)
    max_abs_y = max(abs(float(row["y_mm"])) for row in elements)
    center_half_extent = max(max_abs_x, max_abs_y)
    board_half = center_half_extent + BOARD_EDGE_MARGIN_MM
    board_width = 2.0 * board_half
    return {
        "max_abs_x_mm": max_abs_x,
        "max_abs_y_mm": max_abs_y,
        "center_half_extent_mm": center_half_extent,
        "edge_margin_mm": BOARD_EDGE_MARGIN_MM,
        "board_half_mm": board_half,
        "board_width_mm": board_width,
    }


def write_plan_csv(elements: List[Dict[str, Any]], selected_group_ids: Sequence[str]) -> None:
    selected = set(selected_group_ids)
    PLAN_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "global_index",
        "unit_name",
        "ring_index",
        "ring_local_index",
        "x_lambda",
        "y_lambda",
        "x_mm",
        "y_mm",
        "phi_deg",
        "rotation_deg",
        "group_id",
        "port_name",
        "selected_in_phase1",
    ]
    with PLAN_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for element in elements:
            writer.writerow(
                {
                    "global_index": element["global_index"],
                    "unit_name": element["unit_name"],
                    "ring_index": element["ring_index"],
                    "ring_local_index": element["ring_local_index"],
                    "x_lambda": f"{element['x_lambda']:.15g}",
                    "y_lambda": f"{element['y_lambda']:.15g}",
                    "x_mm": f"{element['x_mm']:.15g}",
                    "y_mm": f"{element['y_mm']:.15g}",
                    "phi_deg": f"{element['phi_deg']:.12g}",
                    "rotation_deg": f"{element['rotation_deg']:.12g}",
                    "group_id": element["group_id"],
                    "port_name": element["port_name"],
                    "selected_in_phase1": element["group_id"] in selected,
                }
            )


# =============================================================================
# Manifest and run mode
# =============================================================================


def project_sidecar_paths() -> List[Path]:
    return [
        PROJECT_PATH,
        Path(str(PROJECT_PATH) + ".lock"),
        Path(str(PROJECT_PATH) + "results"),
        PROJECT_PATH.with_suffix(".aedb"),
    ]


def remove_path(path: Path) -> None:
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def prepare_fresh_output() -> None:
    for path in project_sidecar_paths():
        try:
            remove_path(path)
        except FileNotFoundError:
            pass
    if STATE_DIR.exists():
        shutil.rmtree(STATE_DIR)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    GROUP_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_state_dirs() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    GROUP_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def determine_resume_mode() -> bool:
    mode = RUN_MODE.strip().lower()
    if mode not in {"auto", "fresh", "resume"}:
        raise RuntimeError(f"Unsupported RUN_MODE: {RUN_MODE!r}")

    project_exists = PROJECT_PATH.is_file()
    manifest_exists = MANIFEST_PATH.is_file()

    if mode == "fresh":
        prepare_fresh_output()
        return False
    if mode == "resume":
        if not project_exists or not manifest_exists:
            raise RuntimeError(
                "RUN_MODE='resume' requires both the AEDT project and build_manifest.json"
            )
        ensure_state_dirs()
        return True

    # auto
    if project_exists and manifest_exists:
        ensure_state_dirs()
        return True
    if project_exists != manifest_exists:
        raise RuntimeError(
            "Auto mode found an inconsistent checkpoint: exactly one of the AEDT project "
            "and manifest exists. Set RUN_MODE='fresh' after backing up the files."
        )
    prepare_fresh_output()
    return False


def new_manifest(
    inputs: Dict[str, Any],
    full_groups: List[Dict[str, Any]],
    selected_groups: List[Dict[str, Any]],
    board_dimensions: Dict[str, float],
) -> Dict[str, Any]:
    selected_ids = {group["group_id"] for group in selected_groups}
    group_states: Dict[str, Any] = {}
    for group in full_groups:
        group_states[group["group_id"]] = {
            "selected_in_phase1": group["group_id"] in selected_ids,
            "ring_index": group["ring_index"],
            "element_indices": group["element_indices"],
            "element_count": group["element_count"],
            "geometry_status": "pending",
            "board_status": "pending",
            "attempts": 0,
            "board_attempts": 0,
            "started_at": None,
            "geometry_completed_at": None,
            "board_completed_at": None,
            "last_error_report": None,
            "group_report": None,
            "board_report": None,
        }

    return {
        "version": 2,
        "script_revision": SCRIPT_REVISION,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "run_mode": RUN_MODE,
        "phase": "boards_first_four_groups",
        "project_path": str(PROJECT_PATH),
        "design_name": DESIGN_NAME,
        "stage1_npz": str(STAGE1_NPZ_PATH),
        "stage1_sha256": inputs["stage1_sha256"],
        "stage2_seeds_json": str(STAGE2_SEEDS_JSON_PATH),
        "stage2_sha256": inputs["stage2_sha256"],
        "cut_l_mm": CUT_L_MM,
        "lambda0_mm": LAMBDA0_MM,
        "full_element_count": 187,
        "selected_group_ids": [group["group_id"] for group in selected_groups],
        "selected_element_count": int(sum(group["element_count"] for group in selected_groups)),
        "board_dimensions": board_dimensions,
        "boards_initialized": False,
        "groups": group_states,
        "design_initialized": False,
        "phase1_complete": False,
    }

def validate_manifest_inputs(manifest: Dict[str, Any], inputs: Dict[str, Any]) -> None:
    checks = {
        "stage1_sha256": inputs["stage1_sha256"],
        "stage2_sha256": inputs["stage2_sha256"],
        "design_name": DESIGN_NAME,
        "cut_l_mm": CUT_L_MM,
    }
    mismatches = []
    for key, expected in checks.items():
        actual = manifest.get(key)
        if actual != expected:
            mismatches.append(f"{key}: manifest={actual!r}, expected={expected!r}")

    revision = str(manifest.get("script_revision", ""))
    if revision not in PREVIOUS_COMPATIBLE_REVISIONS:
        mismatches.append(
            f"script_revision: manifest={revision!r}, compatible={sorted(PREVIOUS_COMPATIBLE_REVISIONS)!r}"
        )
    if mismatches:
        raise RuntimeError(
            "Checkpoint manifest does not match this script/input set:\n"
            + "\n".join(mismatches)
        )


def upgrade_manifest_for_full_run(
    manifest: Dict[str, Any],
    inputs: Dict[str, Any],
    full_groups: List[Dict[str, Any]],
    board_dimensions: Dict[str, float],
) -> None:
    """Migrate a successful four-group checkpoint into the complete build."""
    manifest["version"] = max(int(manifest.get("version", 0)), 4)
    manifest["script_revision"] = SCRIPT_REVISION
    manifest["phase"] = "full_array_build"
    manifest["run_mode"] = RUN_MODE
    manifest["selected_group_ids"] = [group["group_id"] for group in full_groups]
    manifest["selected_element_count"] = int(
        sum(int(group["element_count"]) for group in full_groups)
    )
    manifest["board_dimensions"] = board_dimensions
    manifest["full_element_count"] = 187
    manifest["phase1_complete"] = bool(manifest.get("phase1_complete", False))
    manifest.setdefault("airbox_status", "pending")
    manifest.setdefault("common_boundaries_status", "pending")
    manifest.setdefault("setup_status", "pending")
    manifest.setdefault("sweep_status", "pending")
    manifest.setdefault("full_build_complete", False)

    states = manifest.setdefault("groups", {})
    for group in full_groups:
        group_id = group["group_id"]
        state = states.setdefault(
            group_id,
            {
                "ring_index": group["ring_index"],
                "element_indices": group["element_indices"],
                "element_count": group["element_count"],
                "geometry_status": "pending",
                "board_status": "pending",
                "attempts": 0,
                "board_attempts": 0,
                "started_at": None,
                "geometry_completed_at": None,
                "board_completed_at": None,
                "last_error_report": None,
                "group_report": None,
                "board_report": None,
            },
        )
        state["selected_for_full_run"] = True
        state.setdefault("port_status", "pending")
        state["assignment_status"] = state.get("port_status", "pending")
        state.setdefault("port_attempts", 0)
        state.setdefault("port_completed_at", None)
        state.setdefault("port_report", None)

    # Re-assert immutable input metadata.
    manifest["stage1_sha256"] = inputs["stage1_sha256"]
    manifest["stage2_sha256"] = inputs["stage2_sha256"]


def save_manifest(manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = now_iso()
    atomic_write_json(MANIFEST_PATH, manifest)


# =============================================================================
# AEDT launch/session helpers
# =============================================================================


def launch_hfss(resume: bool):
    Hfss, namespace, version = import_pyaedt()
    log(f"PyAEDT namespace : {namespace}")
    log(f"PyAEDT version   : {version}")
    log(f"Resume existing  : {resume}")

    hfss = Hfss(
        project=str(PROJECT_PATH),
        design=DESIGN_NAME,
        version=AEDT_VERSION,
        solution_type="Modal",
        non_graphical=NON_GRAPHICAL,
        new_desktop=NEW_DESKTOP,
        close_on_exit=CLOSE_ON_EXIT,
    )
    return hfss


def save_project(hfss) -> None:
    try:
        hfss.save_project()
        log(f"[OK] Project saved: {PROJECT_PATH}")
        return
    except Exception:
        log("[WARN] hfss.save_project() failed; using raw SaveAs.")
        log(traceback.format_exc())
    hfss.oproject.SaveAs(str(PROJECT_PATH), True)
    log(f"[OK] Project saved with raw SaveAs: {PROJECT_PATH}")


def refresh_design_handles(hfss):
    o_project = hfss.oproject
    o_design = None
    try:
        o_design = o_project.SetActiveDesign(DESIGN_NAME)
    except Exception:
        pass
    if o_design is None:
        o_design = hfss.odesign
    if o_design is None:
        raise RuntimeError(f"Could not activate design: {DESIGN_NAME}")
    o_editor = o_design.SetActiveEditor("3D Modeler")
    if o_editor is None:
        raise RuntimeError("Could not activate the 3D Modeler editor.")
    return o_project, o_design, o_editor


def get_module(hfss, o_design, module_name: str):
    attr_map = {
        "BoundarySetup": "oboundary",
        "AnalysisSetup": "oanalysis",
        "RadField": "oradfield",
        "ReportSetup": "oreportsetup",
    }
    attr_name = attr_map.get(module_name)
    if attr_name:
        try:
            module = getattr(hfss, attr_name)
            if module is not None:
                return module
        except Exception:
            pass
    try:
        module = hfss.get_module(module_name)
        if module is not None:
            return module
    except Exception:
        pass
    try:
        module = o_design.GetModule(module_name)
        if module is not None:
            return module
    except Exception:
        pass
    raise RuntimeError(f"Could not acquire AEDT module: {module_name}")


# =============================================================================
# Raw AEDT geometry helpers
# =============================================================================


def mm(value: float) -> str:
    return f"{float(value):.12g}mm"


def deg(value: float) -> str:
    return f"{float(value):.12g}deg"


def parse_number(value: Any) -> float:
    text = str(value).strip().lower()
    for unit in ("mm", "deg"):
        text = text.replace(unit, "")
    return float(text.replace(" ", ""))


def set_model_units_mm(o_editor) -> None:
    o_editor.SetModelUnits(
        [
            "NAME:Units Parameter",
            "Units:=",
            "mm",
            "Rescale:=",
            False,
        ]
    )


def attributes(
    name: str,
    material: str = "vacuum",
    color: str = "(143 175 143)",
    transparency: float = 0.0,
    solve_inside: bool = True,
) -> List[Any]:
    return [
        "NAME:Attributes",
        "Name:=",
        name,
        "Flags:=",
        "",
        "Color:=",
        color,
        "Transparency:=",
        transparency,
        "PartCoordinateSystem:=",
        "Global",
        "UDMId:=",
        "",
        "MaterialValue:=",
        f'"{material}"',
        "SurfaceMaterialValue:=",
        '""',
        "SolveInside:=",
        solve_inside,
        "ShellElement:=",
        False,
        "ShellElementThickness:=",
        "0mm",
        "ReferenceTemperature:=",
        "20cel",
        "IsMaterialEditable:=",
        True,
        "IsSurfaceMaterialEditable:=",
        True,
        "UseMaterialAppearance:=",
        False,
        "IsLightweight:=",
        False,
    ]


def create_rectangle(
    o_editor,
    name: str,
    x: str,
    y: str,
    z: str,
    width: str,
    height: str,
    material: str = "vacuum",
    color: str = "(255 128 0)",
    transparency: float = 0.0,
) -> None:
    o_editor.CreateRectangle(
        [
            "NAME:RectangleParameters",
            "IsCovered:=",
            True,
            "XStart:=",
            x,
            "YStart:=",
            y,
            "ZStart:=",
            z,
            "Width:=",
            width,
            "Height:=",
            height,
            "WhichAxis:=",
            "Z",
        ],
        attributes(name, material, color, transparency, solve_inside=True),
    )


def create_box(
    o_editor,
    name: str,
    x: str,
    y: str,
    z: str,
    sx: str,
    sy: str,
    sz: str,
    material: str,
    color: str = "(143 175 143)",
    transparency: float = 0.6,
) -> None:
    o_editor.CreateBox(
        [
            "NAME:BoxParameters",
            "XPosition:=",
            x,
            "YPosition:=",
            y,
            "ZPosition:=",
            z,
            "XSize:=",
            sx,
            "YSize:=",
            sy,
            "ZSize:=",
            sz,
        ],
        attributes(name, material, color, transparency, solve_inside=True),
    )


def create_cylinder(
    o_editor,
    name: str,
    x: str,
    y: str,
    z: str,
    radius: str,
    height: str,
    material: str,
    color: str = "(255 128 0)",
    transparency: float = 0.0,
    solve_inside: bool = True,
) -> None:
    o_editor.CreateCylinder(
        [
            "NAME:CylinderParameters",
            "XCenter:=",
            x,
            "YCenter:=",
            y,
            "ZCenter:=",
            z,
            "Radius:=",
            radius,
            "Height:=",
            height,
            "WhichAxis:=",
            "Z",
            "NumSides:=",
            "0",
        ],
        attributes(name, material, color, transparency, solve_inside=solve_inside),
    )


def create_circle(
    o_editor,
    name: str,
    x: str,
    y: str,
    z: str,
    radius: str,
    material: str = "vacuum",
) -> None:
    o_editor.CreateCircle(
        [
            "NAME:CircleParameters",
            "IsCovered:=",
            True,
            "XCenter:=",
            x,
            "YCenter:=",
            y,
            "ZCenter:=",
            z,
            "Radius:=",
            radius,
            "WhichAxis:=",
            "Z",
            "NumSegments:=",
            "0",
        ],
        attributes(name, material, solve_inside=True),
    )


def subtract(o_editor, blank: str, tools: Sequence[str] | str, keep_originals: bool) -> None:
    tool_string = ",".join(tools) if isinstance(tools, (list, tuple)) else str(tools)
    o_editor.Subtract(
        [
            "NAME:Selections",
            "Blank Parts:=",
            blank,
            "Tool Parts:=",
            tool_string,
        ],
        [
            "NAME:SubtractParameters",
            "KeepOriginals:=",
            bool(keep_originals),
            "TurnOnNBodyBoolean:=",
            True,
        ],
    )


def rotate_objects(o_editor, object_names: Sequence[str], angle_deg: float) -> None:
    if abs(float(angle_deg)) < 1.0e-12:
        return
    o_editor.Rotate(
        [
            "NAME:Selections",
            "Selections:=",
            ",".join(object_names),
            "NewPartsModelFlag:=",
            "Model",
        ],
        [
            "NAME:RotateParameters",
            "RotateAxis:=",
            "Z",
            "RotateAngle:=",
            deg(angle_deg),
        ],
    )


def move_objects(
    o_editor,
    object_names: Sequence[str],
    dx_mm: float,
    dy_mm: float,
    dz_mm: float = 0.0,
) -> None:
    o_editor.Move(
        [
            "NAME:Selections",
            "Selections:=",
            ",".join(object_names),
            "NewPartsModelFlag:=",
            "Model",
        ],
        [
            "NAME:TranslateParameters",
            "TranslateVectorX:=",
            mm(dx_mm),
            "TranslateVectorY:=",
            mm(dy_mm),
            "TranslateVectorZ:=",
            mm(dz_mm),
        ],
    )


def get_object_names(o_editor) -> List[str]:
    names: List[str] = []
    for group in ("Solids", "Sheets", "Unclassified"):
        try:
            names.extend(str(value) for value in list(o_editor.GetObjectsInGroup(group)))
        except Exception:
            pass
    return sorted(set(names))


def object_exists(o_editor, object_name: str) -> bool:
    return object_name in set(get_object_names(o_editor))


def rename_object(o_editor, old_name: str, new_name: str) -> None:
    if old_name == new_name:
        return
    if not object_exists(o_editor, old_name):
        raise RuntimeError(f"Cannot rename missing object: {old_name}")
    if object_exists(o_editor, new_name):
        raise RuntimeError(f"Rename target already exists: {new_name}")

    o_editor.ChangeProperty(
        [
            "NAME:AllTabs",
            [
                "NAME:Geometry3DAttributeTab",
                ["NAME:PropServers", old_name],
                [
                    "NAME:ChangedProps",
                    ["NAME:Name", "Value:=", new_name],
                ],
            ],
        ]
    )
    for _ in range(40):
        if object_exists(o_editor, new_name) and not object_exists(o_editor, old_name):
            return
        time.sleep(0.1)
    raise RuntimeError(f"AEDT did not complete rename: {old_name} -> {new_name}")


def delete_objects(o_editor, object_names: Sequence[str]) -> None:
    existing = [name for name in object_names if object_exists(o_editor, name)]
    if not existing:
        return
    o_editor.Delete(["NAME:Selections", "Selections:=", ",".join(existing)])
    for _ in range(50):
        if all(not object_exists(o_editor, name) for name in existing):
            return
        time.sleep(0.1)
    remaining = [name for name in existing if object_exists(o_editor, name)]
    raise RuntimeError(f"AEDT did not delete objects: {remaining}")


def get_face_ids(o_editor, object_name: str) -> List[int]:
    return [int(value) for value in list(o_editor.GetFaceIDs(object_name))]


def get_face_area(o_editor, face_id: int) -> Optional[float]:
    try:
        return float(o_editor.GetFaceArea(int(face_id)))
    except Exception:
        return None


def find_outer_cylindrical_face(o_editor, object_name: str) -> Dict[str, Any]:
    target_area = 2.0 * math.pi * TEFLON_R_MM * CONN_L_MM
    candidates: List[Tuple[float, int, float]] = []
    for face_id in get_face_ids(o_editor, object_name):
        area = get_face_area(o_editor, face_id)
        if area is None:
            continue
        relative_error = abs(area - target_area) / max(target_area, 1.0e-12)
        candidates.append((relative_error, face_id, area))
    if not candidates:
        raise RuntimeError(f"No measurable faces found on {object_name}")
    candidates.sort(key=lambda row: row[0])
    relative_error, face_id, area = candidates[0]
    if relative_error > FACE_AREA_REL_TOL:
        raise RuntimeError(
            f"Could not identify outer cylindrical face of {object_name}; "
            f"best relative area error={relative_error:.6g}"
        )
    return {
        "face_id": int(face_id),
        "area_mm2": float(area),
        "target_area_mm2": float(target_area),
        "relative_error": float(relative_error),
    }


def get_object_bounding_box(o_editor, object_name: str) -> Tuple[float, float, float, float, float, float]:
    errors: List[str] = []
    for _ in range(20):
        try:
            raw = list(o_editor.GetObjectBoundingBox(object_name))
            if len(raw) != 6:
                raise RuntimeError(f"unexpected bbox length {len(raw)}")
            values = tuple(parse_number(value) for value in raw)
            return values  # type: ignore[return-value]
        except Exception as exc:
            errors.append(repr(exc))
            time.sleep(0.1)
    raise RuntimeError(f"Could not read bounding box for {object_name}: {errors[-3:]}")


def bbox_center(bbox: Sequence[float]) -> Tuple[float, float, float]:
    return (
        0.5 * (float(bbox[0]) + float(bbox[3])),
        0.5 * (float(bbox[1]) + float(bbox[4])),
        0.5 * (float(bbox[2]) + float(bbox[5])),
    )


# =============================================================================
# Recorded MASTER construction and group copying
# =============================================================================


def create_cutcorner_master_patch(o_editor) -> None:
    """Replay the validated recorded Rectangle + symmetric Chamfer workflow."""
    o_editor.CreateRectangle(
        [
            "NAME:RectangleParameters",
            "IsCovered:=",
            True,
            "XStart:=",
            "-70mm",
            "YStart:=",
            "30mm",
            "ZStart:=",
            "0mm",
            "Width:=",
            "50mm",
            "Height:=",
            "-40mm",
            "WhichAxis:=",
            "Z",
        ],
        [
            "NAME:Attributes",
            "Name:=",
            "Rectangle1",
            "Flags:=",
            "",
            "Color:=",
            "(143 175 143)",
            "Transparency:=",
            0,
            "PartCoordinateSystem:=",
            "Global",
            "UDMId:=",
            "",
            "MaterialValue:=",
            '"vacuum"',
            "SurfaceMaterialValue:=",
            '""',
            "SolveInside:=",
            True,
            "ShellElement:=",
            False,
            "ShellElementThickness:=",
            "0mm",
            "ReferenceTemperature:=",
            "20cel",
            "IsMaterialEditable:=",
            True,
            "IsSurfaceMaterialEditable:=",
            True,
            "UseMaterialAppearance:=",
            False,
            "IsLightweight:=",
            False,
        ],
    )

    o_editor.ChangeProperty(
        [
            "NAME:AllTabs",
            [
                "NAME:Geometry3DAttributeTab",
                ["NAME:PropServers", "Rectangle1"],
                [
                    "NAME:ChangedProps",
                    ["NAME:Name", "Value:=", "MASTER_Patch"],
                    ["NAME:Color", "R:=", 255, "G:=", 128, "B:=", 0],
                ],
            ],
        ]
    )

    o_editor.ChangeProperty(
        [
            "NAME:AllTabs",
            [
                "NAME:Geometry3DCmdTab",
                ["NAME:PropServers", "MASTER_Patch:CreateRectangle:1"],
                [
                    "NAME:ChangedProps",
                    [
                        "NAME:Position",
                        "X:=",
                        "-Patch_L/2",
                        "Y:=",
                        "-Patch_L/2",
                        "Z:=",
                        "Sub_h",
                    ],
                    ["NAME:XSize", "Value:=", "Patch_L"],
                    ["NAME:YSize", "Value:=", "Patch_L"],
                ],
            ],
        ]
    )

    vertex_ids: List[int] = []
    query_errors: List[str] = []
    for _ in range(40):
        for method_name in ("GetVertexIDsFromObject", "GetVertexIDs"):
            method = getattr(o_editor, method_name, None)
            if method is None:
                continue
            try:
                ids = [int(value) for value in list(method("MASTER_Patch"))]
                if len(ids) >= 4:
                    vertex_ids = ids
                    break
            except Exception as exc:
                query_errors.append(f"{method_name}: {exc!r}")
        if vertex_ids:
            break
        time.sleep(0.1)
    if not vertex_ids:
        raise RuntimeError(
            "Could not enumerate MASTER_Patch vertices before Chamfer; "
            f"details={query_errors[-4:]}"
        )

    half = 0.5 * PATCH_L_MM
    targets = {
        "lower-left": (-half, -half, SUB_H_MM),
        "upper-right": (+half, +half, SUB_H_MM),
    }
    positions: Dict[int, Tuple[float, float, float]] = {}
    for vertex_id in vertex_ids:
        try:
            raw = list(o_editor.GetVertexPosition(int(vertex_id)))
            xyz = tuple(parse_number(value) for value in raw)
            if len(xyz) == 3:
                positions[int(vertex_id)] = xyz  # type: ignore[assignment]
        except Exception:
            continue
    if len(positions) < 4:
        raise RuntimeError(
            "MASTER_Patch did not return four usable vertex positions: "
            f"vertex_ids={vertex_ids}, positions={positions}"
        )

    selected: Dict[str, int] = {}
    used: set[int] = set()
    for label, target in targets.items():
        candidates = []
        for vertex_id, xyz in positions.items():
            if vertex_id in used:
                continue
            distance = math.sqrt(sum((xyz[i] - target[i]) ** 2 for i in range(3)))
            candidates.append((distance, vertex_id, xyz))
        candidates.sort(key=lambda row: row[0])
        distance, vertex_id, xyz = candidates[0]
        if distance > 1.0e-3:
            raise RuntimeError(
                f"Could not identify {label} MASTER corner; target={target}, "
                f"nearest={vertex_id}/{xyz}, distance={distance}mm"
            )
        selected[label] = int(vertex_id)
        used.add(int(vertex_id))

    chamfer_vertices = [selected["lower-left"], selected["upper-right"]]
    log(
        "MASTER_Patch Chamfer vertices: "
        f"lower-left={chamfer_vertices[0]}, upper-right={chamfer_vertices[1]}"
    )
    o_editor.Chamfer(
        [
            "NAME:Selections",
            "Selections:=",
            "MASTER_Patch",
            "NewPartsModelFlag:=",
            "Model",
        ],
        [
            "NAME:Parameters",
            [
                "NAME:ChamferParameters",
                "Edges:=",
                [],
                "Vertices:=",
                chamfer_vertices,
                "LeftDistance:=",
                "Cut_L",
                "RightDistance:=",
                "Cut_L",
                "ChamferType:=",
                "Symmetric",
            ],
        ],
    )

    for red, green, blue in ((64, 0, 64), (255, 128, 0)):
        o_editor.ChangeProperty(
            [
                "NAME:AllTabs",
                [
                    "NAME:Geometry3DAttributeTab",
                    ["NAME:PropServers", "MASTER_Patch"],
                    [
                        "NAME:ChangedProps",
                        ["NAME:Color", "R:=", red, "G:=", green, "B:=", blue],
                    ],
                ],
            ]
        )
    log("[OK] Created MASTER_Patch by Rectangle + recorded symmetric Chamfer.")


def create_master_element(o_editor) -> None:
    stale = [name for name in get_object_names(o_editor) if name.startswith("MASTER_")]
    stale.extend(name for name in ("Rectangle1",) if object_exists(o_editor, name))
    if stale:
        delete_objects(o_editor, sorted(set(stale)))

    create_cutcorner_master_patch(o_editor)
    create_cylinder(
        o_editor,
        "MASTER_Pin",
        "Feed_Pos",
        "0mm",
        "-Conn_L",
        "Pin_R",
        "Conn_L + Sub_h",
        material="pec",
        color="(255 128 0)",
        transparency=0.0,
        solve_inside=False,
    )
    create_cylinder(
        o_editor,
        "MASTER_Coax_Teflon",
        "Feed_Pos",
        "0mm",
        "-Conn_L",
        "Teflon_R",
        "Conn_L",
        material="teflon_based",
        color="(143 175 143)",
        transparency=0.6,
        solve_inside=True,
    )
    subtract(o_editor, "MASTER_Coax_Teflon", "MASTER_Pin", keep_originals=True)
    create_circle(
        o_editor,
        "MASTER_PortSheet",
        "Feed_Pos",
        "0mm",
        "-Conn_L",
        "Teflon_R",
        material="vacuum",
    )
    create_circle(
        o_editor,
        "MASTER_GND_Hole",
        "Feed_Pos",
        "0mm",
        "0mm",
        "Teflon_R",
        material="vacuum",
    )

    missing = [name for name in MASTER_OBJECTS if not object_exists(o_editor, name)]
    if missing:
        raise RuntimeError(f"MASTER creation incomplete; missing objects: {missing}")
    log("[OK] Fresh group MASTER created at the global origin.")


def copy_master_group(o_editor, prefix: str) -> List[str]:
    for source in MASTER_OBJECTS:
        if not object_exists(o_editor, source):
            raise RuntimeError(f"MASTER source object is missing: {source}")

    before = set(get_object_names(o_editor))
    o_editor.Copy(["NAME:Selections", "Selections:=", ",".join(MASTER_OBJECTS)])
    o_editor.Paste()

    created: set[str] = set()
    for _ in range(60):
        created = set(get_object_names(o_editor)) - before
        if len(created) >= len(MASTER_OBJECTS):
            break
        time.sleep(0.1)
    if len(created) != len(MASTER_OBJECTS):
        raise RuntimeError(
            f"Expected {len(MASTER_OBJECTS)} pasted objects for {prefix}, "
            f"detected {len(created)}: {sorted(created)}"
        )

    renamed: List[str] = []
    unmatched = set(created)
    for source in MASTER_OBJECTS:
        suffix = source.removeprefix("MASTER_")
        target = f"{prefix}_{suffix}"
        matches = sorted(name for name in unmatched if name.startswith(source))
        if len(matches) != 1:
            raise RuntimeError(
                f"Could not uniquely map pasted object for {source}; "
                f"created={sorted(created)}, matches={matches}"
            )
        rename_object(o_editor, matches[0], target)
        unmatched.remove(matches[0])
        renamed.append(target)
    if unmatched:
        raise RuntimeError(f"Unmapped pasted objects for {prefix}: {sorted(unmatched)}")
    return renamed


def transform_feed_xy(element: Dict[str, Any]) -> Tuple[float, float]:
    angle = float(element["rotation_rad"])
    return (
        float(element["x_mm"]) + FEED_POS_MM * math.cos(angle),
        float(element["y_mm"]) + FEED_POS_MM * math.sin(angle),
    )


def copy_place_element(o_editor, element: Dict[str, Any], ordinal: int, total: int) -> None:
    prefix = str(element["unit_name"])
    copied = copy_master_group(o_editor, prefix)
    rotate_objects(o_editor, copied, float(element["rotation_deg"]))
    move_objects(o_editor, copied, float(element["x_mm"]), float(element["y_mm"]), 0.0)
    feed_x, feed_y = transform_feed_xy(element)
    log(
        f"[COPY {ordinal:02d}/{total:02d}] {prefix} "
        f"ring={element['ring_index']} local={element['ring_local_index']} "
        f"center=({element['x_mm']:.6f},{element['y_mm']:.6f})mm "
        f"rotation={element['rotation_deg']:.6f}deg "
        f"feed=({feed_x:.6f},{feed_y:.6f})mm"
    )


# =============================================================================
# Square shared boards and per-group board transactions
# =============================================================================


def copy_single_object(o_editor, source_name: str, target_name: str) -> None:
    if not object_exists(o_editor, source_name):
        raise RuntimeError(f"Cannot copy missing object: {source_name}")
    if object_exists(o_editor, target_name):
        raise RuntimeError(f"Copy target already exists: {target_name}")
    before = set(get_object_names(o_editor))
    o_editor.Copy(["NAME:Selections", "Selections:=", source_name])
    o_editor.Paste()
    created: set[str] = set()
    for _ in range(60):
        created = set(get_object_names(o_editor)) - before
        if len(created) == 1:
            break
        time.sleep(0.1)
    if len(created) != 1:
        raise RuntimeError(
            f"Expected one pasted object for {source_name}, detected {sorted(created)}"
        )
    rename_object(o_editor, next(iter(created)), target_name)


def create_square_shared_boards(
    o_editor,
    board_dimensions: Dict[str, float],
) -> None:
    width = float(board_dimensions["board_width_mm"])
    half = 0.5 * width
    if object_exists(o_editor, GND_NAME) or object_exists(o_editor, SUBSTRATE_NAME):
        raise RuntimeError("Shared boards already exist during fresh board creation.")
    create_rectangle(
        o_editor,
        GND_NAME,
        mm(-half),
        mm(-half),
        "0mm",
        mm(width),
        mm(width),
        material="vacuum",
        color="(255 128 0)",
        transparency=0.0,
    )
    create_box(
        o_editor,
        SUBSTRATE_NAME,
        mm(-half),
        mm(-half),
        "0mm",
        mm(width),
        mm(width),
        "Sub_h",
        material="FR4_epoxy",
        transparency=0.6,
    )
    log(
        f"[OK] Created square shared boards: width={width:.9f}mm, "
        f"edge_margin={board_dimensions['edge_margin_mm']:.6f}mm"
    )


def validate_shared_boards(
    o_editor,
    board_dimensions: Dict[str, float],
) -> Dict[str, Any]:
    missing = [
        name for name in (GND_NAME, SUBSTRATE_NAME) if not object_exists(o_editor, name)
    ]
    if missing:
        raise RuntimeError(f"Missing shared boards: {missing}")
    width = float(board_dimensions["board_width_mm"])
    half = 0.5 * width
    expected_gnd = (-half, -half, 0.0, half, half, 0.0)
    expected_sub = (-half, -half, 0.0, half, half, SUB_H_MM)
    gnd_bbox = get_object_bounding_box(o_editor, GND_NAME)
    sub_bbox = get_object_bounding_box(o_editor, SUBSTRATE_NAME)
    gnd_error = max(abs(gnd_bbox[i] - expected_gnd[i]) for i in range(6))
    sub_error = max(abs(sub_bbox[i] - expected_sub[i]) for i in range(6))
    if gnd_error > POSITION_TOL_MM:
        raise RuntimeError(f"GND square bounding box mismatch: {gnd_bbox}")
    if sub_error > POSITION_TOL_MM:
        raise RuntimeError(f"Substrate square bounding box mismatch: {sub_bbox}")
    return {
        "ok": True,
        "board_width_mm": width,
        "gnd_bbox_mm": list(gnd_bbox),
        "substrate_bbox_mm": list(sub_bbox),
        "gnd_bbox_max_error_mm": gnd_error,
        "substrate_bbox_max_error_mm": sub_error,
    }


def recover_board_backups_if_present(o_editor) -> bool:
    gnd_backup = object_exists(o_editor, GND_BACKUP_NAME)
    sub_backup = object_exists(o_editor, SUBSTRATE_BACKUP_NAME)
    if not gnd_backup and not sub_backup:
        return False
    if gnd_backup != sub_backup:
        raise RuntimeError(
            "Only one board backup exists. Manual inspection is required before resume."
        )
    delete_objects(
        o_editor,
        [name for name in (GND_NAME, SUBSTRATE_NAME) if object_exists(o_editor, name)],
    )
    rename_object(o_editor, GND_BACKUP_NAME, GND_NAME)
    rename_object(o_editor, SUBSTRATE_BACKUP_NAME, SUBSTRATE_NAME)
    log("[RECOVER] Restored GND/Substrate from interrupted group backups.")
    return True


def create_board_backups(o_editor) -> None:
    delete_objects(
        o_editor,
        [
            name
            for name in (GND_BACKUP_NAME, SUBSTRATE_BACKUP_NAME)
            if object_exists(o_editor, name)
        ],
    )
    copy_single_object(o_editor, GND_NAME, GND_BACKUP_NAME)
    copy_single_object(o_editor, SUBSTRATE_NAME, SUBSTRATE_BACKUP_NAME)


def restore_board_backups(o_editor) -> None:
    if not object_exists(o_editor, GND_BACKUP_NAME) or not object_exists(
        o_editor, SUBSTRATE_BACKUP_NAME
    ):
        raise RuntimeError("Board backups are unavailable for rollback.")
    delete_objects(
        o_editor,
        [name for name in (GND_NAME, SUBSTRATE_NAME) if object_exists(o_editor, name)],
    )
    rename_object(o_editor, GND_BACKUP_NAME, GND_NAME)
    rename_object(o_editor, SUBSTRATE_BACKUP_NAME, SUBSTRATE_NAME)


def delete_board_backups(o_editor) -> None:
    delete_objects(o_editor, [GND_BACKUP_NAME, SUBSTRATE_BACKUP_NAME])


def validate_group_board_cut(
    o_editor,
    group: Dict[str, Any],
    group_elements: Sequence[Dict[str, Any]],
    board_dimensions: Dict[str, float],
    require_no_backups: bool = True,
) -> Dict[str, Any]:
    board_report = validate_shared_boards(o_editor, board_dimensions)
    leftover_holes = [
        f"{row['unit_name']}_GND_Hole"
        for row in group_elements
        if object_exists(o_editor, f"{row['unit_name']}_GND_Hole")
    ]
    missing_pins = [
        f"{row['unit_name']}_Pin"
        for row in group_elements
        if not object_exists(o_editor, f"{row['unit_name']}_Pin")
    ]
    if leftover_holes:
        raise RuntimeError(
            f"Group {group['group_id']} GND-hole cutters remain after subtraction: "
            f"{leftover_holes}"
        )
    if missing_pins:
        raise RuntimeError(
            f"Group {group['group_id']} pins were not preserved: {missing_pins}"
        )
    backups_present = object_exists(o_editor, GND_BACKUP_NAME) or object_exists(
        o_editor, SUBSTRATE_BACKUP_NAME
    )
    if require_no_backups and backups_present:
        raise RuntimeError("Board backup objects remain after a committed group cut.")
    return {
        "ok": True,
        "group_id": group["group_id"],
        "element_count": len(group_elements),
        "gnd_holes_cut": len(group_elements),
        "substrate_holes_cut": len(group_elements),
        "leftover_gnd_hole_cutters": leftover_holes,
        "missing_pins": missing_pins,
        "backups_present": backups_present,
        "shared_boards": board_report,
    }


def cut_group_shared_boards(
    hfss,
    o_editor,
    manifest: Dict[str, Any],
    group: Dict[str, Any],
    group_elements: List[Dict[str, Any]],
    board_dimensions: Dict[str, float],
) -> Dict[str, Any]:
    group_id = group["group_id"]
    state = manifest["groups"][group_id]
    state["board_status"] = "building"
    state["board_attempts"] = int(state.get("board_attempts", 0)) + 1
    save_manifest(manifest)
    header(
        f"BOARD CUT START {group_id}: elements={len(group_elements)} "
        f"attempt={state['board_attempts']}"
    )
    before_names = get_object_names(o_editor)
    create_board_backups(o_editor)
    try:
        hole_names = [f"{row['unit_name']}_GND_Hole" for row in group_elements]
        pin_names = [f"{row['unit_name']}_Pin" for row in group_elements]
        missing_holes = [name for name in hole_names if not object_exists(o_editor, name)]
        missing_pins = [name for name in pin_names if not object_exists(o_editor, name)]
        if missing_holes or missing_pins:
            raise RuntimeError(
                f"Group board tools missing: holes={missing_holes}, pins={missing_pins}"
            )
        subtract(o_editor, GND_NAME, hole_names, keep_originals=False)
        subtract(o_editor, SUBSTRATE_NAME, pin_names, keep_originals=True)

        # Validate the cut while backups still exist. Then save both the AEDT
        # project and an explicit pending-commit state. A restart can safely
        # finish this commit instead of guessing whether to roll it back.
        validate_group_board_cut(
            o_editor,
            group,
            group_elements,
            board_dimensions,
            require_no_backups=False,
        )
        save_project(hfss)
        state["board_status"] = "cut_pending_commit"
        save_manifest(manifest)

        delete_board_backups(o_editor)
        report = validate_group_board_cut(
            o_editor,
            group,
            group_elements,
            board_dimensions,
            require_no_backups=True,
        )
        report["completed_at"] = now_iso()
        report_path = GROUP_REPORT_DIR / f"{group_id}_board.json"
        atomic_write_json(report_path, report)
        save_project(hfss)
        state["board_status"] = "passed"
        state["board_completed_at"] = report["completed_at"]
        state["board_report"] = str(report_path)
        state["last_error_report"] = None
        save_manifest(manifest)
        log(
            f"[BOARD PASS] {group_id}: GND holes={len(group_elements)}, "
            f"substrate holes={len(group_elements)}"
        )
        return report
    except Exception:
        exception_text = traceback.format_exc()
        failure_names = get_object_names(o_editor)
        cleanup_info: Dict[str, Any] = {"attempted": True, "stage": "board_cut"}
        try:
            restore_board_backups(o_editor)
            save_project(hfss)
            cleanup_info.update(
                {
                    "ok": True,
                    "restored_from_backups": True,
                    "remaining_names": get_object_names(o_editor),
                }
            )
        except Exception:
            rollback_traceback = traceback.format_exc()
            exception_text += "\nBOARD ROLLBACK FAILED:\n" + rollback_traceback
            cleanup_info.update(
                {
                    "ok": False,
                    "restored_from_backups": False,
                    "cleanup_traceback": rollback_traceback,
                    "remaining_names": get_object_names(o_editor),
                }
            )
        error_group = dict(group)
        error_group["failure_stage"] = "board_cut"
        json_path, txt_path = write_error_report(
            error_group, exception_text, before_names, failure_names, cleanup_info
        )
        state["board_status"] = "failed"
        state["board_completed_at"] = None
        state["last_error_report"] = str(json_path)
        save_manifest(manifest)
        log(f"[BOARD FAILED] {group_id}")
        log(f"Error JSON: {json_path}")
        log(f"Error TXT : {txt_path}")
        log(exception_text)
        raise


# =============================================================================
# Group checks, cleanup, and reports
# =============================================================================


def expected_names_for_element(
    element: Dict[str, Any],
    include_gnd_hole: bool = True,
) -> List[str]:
    prefix = str(element["unit_name"])
    suffixes = list(FINAL_SUFFIXES)
    if not include_gnd_hole:
        suffixes.remove("GND_Hole")
    return [f"{prefix}_{suffix}" for suffix in suffixes]


def expected_names_for_group(
    group_elements: Sequence[Dict[str, Any]],
    include_gnd_holes: bool = True,
) -> List[str]:
    names: List[str] = []
    for element in group_elements:
        names.extend(expected_names_for_element(element, include_gnd_holes))
    return names


def cleanup_orphan_masters(o_editor) -> List[str]:
    names = get_object_names(o_editor)
    stale = [name for name in names if name.startswith("MASTER_") or name == "Rectangle1"]
    if stale:
        delete_objects(o_editor, stale)
    return stale


def cleanup_group_objects(o_editor, group_elements: Sequence[Dict[str, Any]]) -> List[str]:
    prefixes = [str(element["unit_name"]) + "_" for element in group_elements]
    names = get_object_names(o_editor)
    targets = [name for name in names if any(name.startswith(prefix) for prefix in prefixes)]
    if targets:
        delete_objects(o_editor, targets)
    cleanup_orphan_masters(o_editor)
    return targets


def validate_group_geometry(
    o_editor,
    group: Dict[str, Any],
    group_elements: Sequence[Dict[str, Any]],
    include_gnd_holes: bool = True,
) -> Dict[str, Any]:
    names = set(get_object_names(o_editor))
    expected = set(expected_names_for_group(group_elements, include_gnd_holes))
    missing = sorted(expected - names)
    master_leftovers = sorted(name for name in names if name.startswith("MASTER_"))
    if missing:
        raise RuntimeError(f"Group {group['group_id']} missing objects: {missing}")
    if master_leftovers:
        raise RuntimeError(f"MASTER leftovers after {group['group_id']}: {master_leftovers}")

    unexpected_holes: List[str] = []
    if not include_gnd_holes:
        unexpected_holes = [
            f"{row['unit_name']}_GND_Hole"
            for row in group_elements
            if object_exists(o_editor, f"{row['unit_name']}_GND_Hole")
        ]
        if unexpected_holes:
            raise RuntimeError(
                f"Group {group['group_id']} has committed board cuts but GND-hole "
                f"cutters remain: {unexpected_holes}"
            )

    element_reports: List[Dict[str, Any]] = []
    max_patch_center_error = 0.0
    max_port_center_error = 0.0
    max_coax_face_error = 0.0

    for element in group_elements:
        prefix = str(element["unit_name"])
        expected_center = (
            float(element["x_mm"]),
            float(element["y_mm"]),
            SUB_H_MM,
        )
        expected_feed_x, expected_feed_y = transform_feed_xy(element)
        expected_port_center = (expected_feed_x, expected_feed_y, -CONN_L_MM)

        patch_bbox = get_object_bounding_box(o_editor, f"{prefix}_Patch")
        patch_center = bbox_center(patch_bbox)
        patch_error = math.sqrt(
            sum((patch_center[i] - expected_center[i]) ** 2 for i in range(3))
        )
        if patch_error > POSITION_TOL_MM:
            raise RuntimeError(
                f"{prefix}_Patch center mismatch: expected={expected_center}, "
                f"actual={patch_center}, error={patch_error}mm"
            )

        port_bbox = get_object_bounding_box(o_editor, f"{prefix}_PortSheet")
        port_center = bbox_center(port_bbox)
        port_error = math.sqrt(
            sum((port_center[i] - expected_port_center[i]) ** 2 for i in range(3))
        )
        if port_error > POSITION_TOL_MM:
            raise RuntimeError(
                f"{prefix}_PortSheet center mismatch: expected={expected_port_center}, "
                f"actual={port_center}, error={port_error}mm"
            )

        coax_face = find_outer_cylindrical_face(o_editor, f"{prefix}_Coax_Teflon")
        max_patch_center_error = max(max_patch_center_error, patch_error)
        max_port_center_error = max(max_port_center_error, port_error)
        max_coax_face_error = max(max_coax_face_error, float(coax_face["relative_error"]))

        element_reports.append(
            {
                "global_index": int(element["global_index"]),
                "unit_name": prefix,
                "ring_index": int(element["ring_index"]),
                "ring_local_index": int(element["ring_local_index"]),
                "target_center_mm": list(expected_center),
                "actual_patch_center_mm": list(patch_center),
                "patch_center_error_mm": patch_error,
                "target_feed_center_mm": list(expected_port_center),
                "actual_port_center_mm": list(port_center),
                "port_center_error_mm": port_error,
                "rotation_deg": float(element["rotation_deg"]),
                "coax_outer_face": coax_face,
            }
        )

    object_count_per_element = 5 if include_gnd_holes else 4
    return {
        "ok": True,
        "group_id": group["group_id"],
        "ring_index": int(group["ring_index"]),
        "element_count": len(group_elements),
        "gnd_hole_cutters_expected": include_gnd_holes,
        "expected_object_count": object_count_per_element * len(group_elements),
        "missing_objects": missing,
        "unexpected_gnd_hole_cutters": unexpected_holes,
        "master_leftovers": master_leftovers,
        "max_patch_center_error_mm": max_patch_center_error,
        "max_port_center_error_mm": max_port_center_error,
        "max_coax_face_relative_error": max_coax_face_error,
        "elements": element_reports,
    }

def write_group_report(group_id: str, report: Dict[str, Any]) -> Path:
    path = GROUP_REPORT_DIR / f"{group_id}.json"
    atomic_write_json(path, report)
    return path


def write_error_report(
    group: Dict[str, Any],
    exception_text: str,
    before_names: Sequence[str],
    failure_names: Sequence[str],
    cleanup_result: Dict[str, Any],
) -> Tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = ERROR_REPORT_DIR / f"failure_{stamp}_{group['group_id']}"
    json_path = base.with_suffix(".json")
    txt_path = base.with_suffix(".txt")
    payload = {
        "ok": False,
        "timestamp": now_iso(),
        "script_revision": SCRIPT_REVISION,
        "group": group,
        "project_path": str(PROJECT_PATH),
        "stage1_npz": str(STAGE1_NPZ_PATH),
        "stage2_seeds_json": str(STAGE2_SEEDS_JSON_PATH),
        "exception_traceback": exception_text,
        "object_names_before_group": list(before_names),
        "object_names_at_failure": list(failure_names),
        "cleanup_result": cleanup_result,
        "python_executable": sys.executable,
    }
    atomic_write_json(json_path, payload)
    txt_path.write_text(exception_text, encoding="utf-8")
    return json_path, txt_path


# =============================================================================
# Design initialization and resume verification
# =============================================================================


def get_existing_variables(o_design) -> set[str]:
    try:
        return {str(value) for value in list(o_design.GetVariables())}
    except Exception:
        return set()


def ensure_design_variables(o_design) -> None:
    existing = get_existing_variables(o_design)
    missing = [(name, value) for name, value in VARIABLES if name not in existing]
    if not missing:
        log("[OK] All design variables already exist.")
        return

    new_props: List[Any] = ["NAME:NewProps"]
    for name, value in missing:
        new_props.append(
            [
                f"NAME:{name}",
                "PropType:=",
                "VariableProp",
                "UserDef:=",
                True,
                "Value:=",
                value,
            ]
        )
    o_design.ChangeProperty(
        [
            "NAME:AllTabs",
            [
                "NAME:LocalVariableTab",
                ["NAME:PropServers", "LocalVariables"],
                new_props,
            ],
        ]
    )
    for name, value in missing:
        log(f"Variable created: {name} = {value}")


def validate_cut_l_variable(o_design) -> None:
    """Reject a resumed AEDT model whose actual cut differs from the cut360 build."""
    value = str(o_design.GetVariableValue("Cut_L")).strip()
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*mm", value)
    if match is None or not math.isclose(
        float(match.group(1)), CUT_L_MM, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError(
            f"HFSS design variable Cut_L={value!r} is not {CUT_L_MM:.2f}mm. "
            "Do not resume a cut365 project with the cut360 builder."
        )


def group_elements_from_plan(
    group: Dict[str, Any],
    elements_by_index: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [elements_by_index[int(index)] for index in group["element_indices"]]


def reconcile_checkpoint_groups(
    hfss,
    o_editor,
    manifest: Dict[str, Any],
    selected_groups: List[Dict[str, Any]],
    elements_by_index: Dict[int, Dict[str, Any]],
    board_dimensions: Dict[str, float],
) -> None:
    # Complete an explicitly saved board commit before applying normal rollback
    # rules. At most one group can be in this state because groups are serial.
    pending_commit_groups = [
        group
        for group in selected_groups
        if manifest["groups"][group["group_id"]].get("board_status")
        == "cut_pending_commit"
    ]
    if len(pending_commit_groups) > 1:
        raise RuntimeError(
            f"Multiple cut_pending_commit groups found: "
            f"{[g['group_id'] for g in pending_commit_groups]}"
        )
    if pending_commit_groups:
        group = pending_commit_groups[0]
        group_id = group["group_id"]
        group_elements = group_elements_from_plan(group, elements_by_index)
        validate_group_geometry(
            o_editor, group, group_elements, include_gnd_holes=False
        )
        validate_group_board_cut(
            o_editor,
            group,
            group_elements,
            board_dimensions,
            require_no_backups=False,
        )
        delete_board_backups(o_editor)
        report = validate_group_board_cut(
            o_editor,
            group,
            group_elements,
            board_dimensions,
            require_no_backups=True,
        )
        report["completed_at"] = now_iso()
        report_path = GROUP_REPORT_DIR / f"{group_id}_board.json"
        atomic_write_json(report_path, report)
        save_project(hfss)
        state = manifest["groups"][group_id]
        state["geometry_status"] = "passed"
        state["board_status"] = "passed"
        state["board_completed_at"] = report["completed_at"]
        state["board_report"] = str(report_path)
        save_manifest(manifest)
        log(f"[RESUME COMMIT] Finished saved board transaction for {group_id}.")
    elif recover_board_backups_if_present(o_editor):
        save_project(hfss)

    validate_shared_boards(o_editor, board_dimensions)

    removed_masters = cleanup_orphan_masters(o_editor)
    if removed_masters:
        log(f"[RESUME] Removed orphan MASTER objects: {removed_masters}")
        save_project(hfss)

    for group in selected_groups:
        group_id = group["group_id"]
        state = manifest["groups"][group_id]
        group_elements = group_elements_from_plan(group, elements_by_index)
        geometry_status = str(state.get("geometry_status", "pending"))
        board_status = str(state.get("board_status", "pending"))

        if board_status == "passed":
            try:
                validate_group_geometry(
                    o_editor, group, group_elements, include_gnd_holes=False
                )
                validate_group_board_cut(
                    o_editor, group, group_elements, board_dimensions
                )
                state["geometry_status"] = "passed"
                log(f"[RESUME PASS] {group_id} geometry and board cuts verified.")
                continue
            except Exception:
                log(
                    f"[RESUME FATAL] {group_id} was committed but failed verification. "
                    "Automatic rebuild is unsafe because the board holes are already committed."
                )
                log(traceback.format_exc())
                raise

        if geometry_status == "passed":
            try:
                validate_group_geometry(
                    o_editor, group, group_elements, include_gnd_holes=True
                )
                state["board_status"] = "pending"
                save_manifest(manifest)
                log(f"[RESUME] {group_id} geometry verified; board cut will resume.")
                continue
            except Exception:
                log(f"[RESUME INVALID] {group_id} geometry checkpoint is incomplete.")
                log(traceback.format_exc())
                cleanup_group_objects(o_editor, group_elements)
                state["geometry_status"] = "pending"
                state["board_status"] = "pending"
                state["geometry_completed_at"] = None
                state["board_completed_at"] = None
                state["group_report"] = None
                state["board_report"] = None
                save_project(hfss)
                save_manifest(manifest)
                continue

        removed = cleanup_group_objects(o_editor, group_elements)
        if removed:
            log(f"[RESUME CLEAN] {group_id}: removed partial objects {removed}")
            save_project(hfss)
        state["geometry_status"] = "pending"
        state["board_status"] = "pending"
        state["geometry_completed_at"] = None
        state["board_completed_at"] = None
        save_manifest(manifest)


# =============================================================================
# Main group build transaction
# =============================================================================


def build_one_group_geometry(
    hfss,
    o_editor,
    manifest: Dict[str, Any],
    group: Dict[str, Any],
    group_elements: List[Dict[str, Any]],
) -> None:
    group_id = group["group_id"]
    state = manifest["groups"][group_id]
    state["geometry_status"] = "building"
    state["attempts"] = int(state.get("attempts", 0)) + 1
    state["started_at"] = now_iso()
    save_manifest(manifest)

    header(
        f"GROUP GEOMETRY START {group_id}: ring={group['ring_index']} "
        f"elements={len(group_elements)} attempt={state['attempts']}"
    )
    before_names = get_object_names(o_editor)
    start_time = time.perf_counter()

    try:
        create_master_element(o_editor)
        for ordinal, element in enumerate(group_elements, start=1):
            copy_place_element(o_editor, element, ordinal, len(group_elements))
        delete_objects(o_editor, MASTER_OBJECTS)

        report = validate_group_geometry(
            o_editor, group, group_elements, include_gnd_holes=True
        )
        report["started_at"] = state["started_at"]
        report["completed_at"] = now_iso()
        report["elapsed_seconds"] = time.perf_counter() - start_time
        report["input_stage1_sha256"] = manifest["stage1_sha256"]
        report["input_stage2_sha256"] = manifest["stage2_sha256"]
        report_path = write_group_report(group_id, report)

        save_project(hfss)
        state["geometry_status"] = "passed"
        state["board_status"] = "pending"
        state["geometry_completed_at"] = report["completed_at"]
        state["group_report"] = str(report_path)
        state["last_error_report"] = None
        save_manifest(manifest)

        log(
            f"[GEOMETRY PASS] {group_id}: elements={len(group_elements)} "
            f"objects={report['expected_object_count']} "
            f"max_patch_err={report['max_patch_center_error_mm']:.3e}mm "
            f"max_port_err={report['max_port_center_error_mm']:.3e}mm "
            f"elapsed={report['elapsed_seconds']:.2f}s"
        )
    except Exception:
        exception_text = traceback.format_exc()
        failure_names = get_object_names(o_editor)
        cleanup_info: Dict[str, Any] = {"attempted": True}
        try:
            created_since_start = sorted(set(failure_names) - set(before_names))
            if created_since_start:
                delete_objects(o_editor, created_since_start)
            cleanup_orphan_masters(o_editor)
            cleanup_info.update(
                {
                    "ok": True,
                    "deleted_created_since_group_start": created_since_start,
                    "remaining_names": get_object_names(o_editor),
                }
            )
            save_project(hfss)
        except Exception:
            cleanup_info.update(
                {
                    "ok": False,
                    "cleanup_traceback": traceback.format_exc(),
                    "remaining_names": get_object_names(o_editor),
                }
            )

        json_path, txt_path = write_error_report(
            group, exception_text, before_names, failure_names, cleanup_info
        )
        state["geometry_status"] = "failed"
        state["board_status"] = "pending"
        state["last_error_report"] = str(json_path)
        state["geometry_completed_at"] = None
        save_manifest(manifest)
        log(f"[GEOMETRY FAILED] {group_id}")
        log(f"Error JSON: {json_path}")
        log(f"Error TXT : {txt_path}")
        log(exception_text)
        raise


def process_one_group(
    hfss,
    o_editor,
    manifest: Dict[str, Any],
    group: Dict[str, Any],
    group_elements: List[Dict[str, Any]],
    board_dimensions: Dict[str, float],
) -> None:
    state = manifest["groups"][group["group_id"]]
    if state.get("geometry_status") != "passed":
        build_one_group_geometry(hfss, o_editor, manifest, group, group_elements)
    if state.get("board_status") != "passed":
        cut_group_shared_boards(
            hfss, o_editor, manifest, group, group_elements, board_dimensions
        )




# =============================================================================
# AirBox, boundaries, ports, Setup, and final checks
# =============================================================================


def expected_airbox_bbox(board_dimensions: Dict[str, float]) -> Tuple[float, ...]:
    board_half = float(board_dimensions["board_half_mm"])
    xy_half = board_half + AIR_MARGIN_XY_MM
    return (
        -xy_half,
        -xy_half,
        -CONN_L_MM,
        +xy_half,
        +xy_half,
        SUB_H_MM + AIR_MARGIN_TOP_MM,
    )


def validate_airbox(o_editor, board_dimensions: Dict[str, float]) -> Dict[str, Any]:
    if not object_exists(o_editor, AIRBOX_NAME):
        raise RuntimeError(f"Missing shared AirBox: {AIRBOX_NAME}")
    actual = get_object_bounding_box(o_editor, AIRBOX_NAME)
    expected = expected_airbox_bbox(board_dimensions)
    error = max(abs(float(actual[i]) - float(expected[i])) for i in range(6))
    if error > POSITION_TOL_MM:
        raise RuntimeError(
            f"AirBox bounding box mismatch: expected={expected}, actual={actual}, error={error}mm"
        )
    return {
        "ok": True,
        "name": AIRBOX_NAME,
        "expected_bbox_mm": list(expected),
        "actual_bbox_mm": list(actual),
        "max_bbox_error_mm": error,
    }


def ensure_shared_airbox(
    hfss,
    o_editor,
    manifest: Dict[str, Any],
    board_dimensions: Dict[str, float],
) -> Dict[str, Any]:
    try:
        report = validate_airbox(o_editor, board_dimensions)
        manifest["airbox_status"] = "passed"
        save_manifest(manifest)
        log("[SKIP] Shared square AirBox already exists and passed validation.")
        return report
    except Exception:
        if object_exists(o_editor, AIRBOX_NAME):
            delete_objects(o_editor, [AIRBOX_NAME])

    header("Create final-size square AirBox")
    board_width = float(board_dimensions["board_width_mm"])
    air_width = board_width + 2.0 * AIR_MARGIN_XY_MM
    xy_start = -0.5 * air_width
    create_box(
        o_editor,
        AIRBOX_NAME,
        mm(xy_start),
        mm(xy_start),
        mm(-CONN_L_MM),
        mm(air_width),
        mm(air_width),
        mm(CONN_L_MM + SUB_H_MM + AIR_MARGIN_TOP_MM),
        material="vacuum",
        color="(143 175 143)",
        transparency=0.95,
    )
    report = validate_airbox(o_editor, board_dimensions)
    save_project(hfss)
    manifest["airbox_status"] = "passed"
    save_manifest(manifest)
    log(f"[AIRBOX PASS] width={air_width:.9f}mm")
    return report


def get_boundary_names(boundary_module) -> List[str]:
    try:
        return [str(value) for value in list(boundary_module.GetBoundaries())]
    except Exception:
        return []


def normalize_excitation_name(value: Any) -> str:
    """Convert an AEDT excitation/mode token to its boundary assignment name.

    HFSS Driven Modal can expose a one-mode wave port through GetExcitations()
    as ``p002:1`` even though the BoundarySetup assignment itself is named
    ``p002``.  Other AEDT builds return alternating name/type tokens such as
    ``p002, Wave Port``.  Validation and checkpoint recovery must compare the
    canonical assignment name, not the modal ``:1`` reporter name.
    """
    text = str(value).strip()
    # Modal excitation suffix, for example p002:1 or p002:2.
    text = re.sub(r":\d+$", "", text)
    # Defensive support for APIs that append a bracketed mode index.
    text = re.sub(r"\[\d+\]$", "", text)
    return text.strip()


def is_excitation_type_token(value: Any) -> bool:
    text = str(value).strip().lower()
    return any(
        marker in text
        for marker in (
            "wave port",
            "lumped port",
            "circuit port",
            "floquet port",
            "terminal",
            "port",
            "mode",
        )
    )


def get_excitation_inventory(boundary_module) -> Dict[str, Any]:
    """Return raw and canonical excitation inventories from BoundarySetup."""
    try:
        raw = [str(value).strip() for value in list(boundary_module.GetExcitations())]
    except Exception as exc:
        return {
            "raw": [],
            "canonical_names": [],
            "records": [],
            "error": repr(exc),
        }

    records: List[Dict[str, str]] = []
    candidate_names: List[str] = []

    # Common AEDT form: [name, type, name, type, ...].
    if len(raw) >= 2 and len(raw) % 2 == 0 and all(
        is_excitation_type_token(token) for token in raw[1::2]
    ):
        for name, excitation_type in zip(raw[0::2], raw[1::2]):
            canonical = normalize_excitation_name(name)
            records.append(
                {
                    "raw_name": name,
                    "canonical_name": canonical,
                    "type": excitation_type,
                }
            )
            if canonical:
                candidate_names.append(canonical)
    else:
        # Alternate form: a flat list of modal names such as p002:1.
        for token in raw:
            if not token or is_excitation_type_token(token):
                continue
            canonical = normalize_excitation_name(token)
            records.append(
                {
                    "raw_name": token,
                    "canonical_name": canonical,
                    "type": "",
                }
            )
            if canonical:
                candidate_names.append(canonical)

    canonical_names = sorted(set(candidate_names))
    return {
        "raw": raw,
        "canonical_names": canonical_names,
        "records": records,
        "error": None,
    }


def get_excitation_names(boundary_module) -> List[str]:
    return list(get_excitation_inventory(boundary_module)["canonical_names"])


def delete_boundary_assignments(boundary_module, names: Sequence[str]) -> None:
    existing = set(get_boundary_names(boundary_module)) | set(
        get_excitation_names(boundary_module)
    )
    targets = [str(name) for name in names if str(name) in existing]
    if not targets:
        return
    boundary_module.DeleteBoundaries(targets)
    for _ in range(40):
        remaining = set(get_boundary_names(boundary_module)) | set(
            get_excitation_names(boundary_module)
        )
        if all(name not in remaining for name in targets):
            return
        time.sleep(0.1)
    raise RuntimeError(f"AEDT did not delete assignments: {targets}")


def ensure_global_boundaries(
    hfss,
    o_design,
    manifest: Dict[str, Any],
) -> None:
    """Assign only the two truly global boundaries after all board cuts finish.

    Patch and coax boundaries are intentionally created group-by-group immediately
    after each group is placed and drilled.
    """
    boundary = get_module(hfss, o_design, "BoundarySetup")
    required = {COMMON_GND_BOUNDARY, RADIATION_BOUNDARY}
    # Remove the older all-patches assignment if a previous full-build revision
    # created it.  Patch Perfect-E assignments are now strictly group-specific.
    delete_boundary_assignments(boundary, [LEGACY_COMMON_PATCH_BOUNDARY])
    existing = set(get_boundary_names(boundary))
    if required.issubset(existing):
        manifest["common_boundaries_status"] = "passed"
        save_manifest(manifest)
        log("[SKIP] Global GND/Radiation boundaries already exist.")
        return

    header("Assign global GND and Radiation boundaries")
    delete_boundary_assignments(boundary, sorted(required))
    boundary.AssignPerfectE(
        [
            f"NAME:{COMMON_GND_BOUNDARY}",
            "Objects:=",
            [GND_NAME],
            "InfGroundPlane:=",
            False,
        ]
    )
    boundary.AssignRadiation(
        [
            f"NAME:{RADIATION_BOUNDARY}",
            "Objects:=",
            [AIRBOX_NAME],
        ]
    )
    existing = set(get_boundary_names(boundary))
    missing = sorted(required - existing)
    if missing:
        raise RuntimeError(f"Global boundaries were not created: {missing}")
    save_project(hfss)
    manifest["common_boundaries_status"] = "passed"
    save_manifest(manifest)
    log("[GLOBAL BOUNDARY PASS] GND and AirBox assigned.")


def group_patch_boundary_name(group_id: str) -> str:
    return f"PerfE_Patch_{group_id}"

def group_outer_boundary_name(group_id: str) -> str:
    return f"PerfE_Outer_{group_id}"


def assign_one_modal_wave_port(boundary, element: Dict[str, Any]) -> None:
    prefix = str(element["unit_name"])
    angle = float(element["rotation_rad"])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x0 = float(element["x_mm"])
    y0 = float(element["y_mm"])

    start_radius = FEED_POS_MM + PIN_R_MM
    end_radius = FEED_POS_MM + TEFLON_R_MM
    start_x = x0 + start_radius * cos_a
    start_y = y0 + start_radius * sin_a
    end_x = x0 + end_radius * cos_a
    end_y = y0 + end_radius * sin_a

    boundary.AssignWavePort(
        [
            f"NAME:{element['port_name']}",
            "Objects:=",
            [f"{prefix}_PortSheet"],
            "WavePortType:=",
            "Modal",
            "NumModes:=",
            1,
            "UseLineModeAlignment:=",
            False,
            "DoDeembed:=",
            False,
            [
                "NAME:Modes",
                [
                    "NAME:Mode1",
                    "ModeNum:=",
                    1,
                    "UseIntLine:=",
                    True,
                    [
                        "NAME:IntLine",
                        "Coordinate System:=",
                        "Global",
                        "Start:=",
                        [mm(start_x), mm(start_y), mm(-CONN_L_MM)],
                        "End:=",
                        [mm(end_x), mm(end_y), mm(-CONN_L_MM)],
                    ],
                    "AlignmentGroup:=",
                    0,
                    "CharImp:=",
                    "Zpi",
                ],
            ],
            "UseAnalyticAlignment:=",
            False,
            "ShowReporterFilter:=",
            False,
            "ReporterFilter:=",
            [True],
        ]
    )


def validate_group_ports(
    boundary,
    group: Dict[str, Any],
    group_elements: Sequence[Dict[str, Any]],
    retry_count: int = 40,
    retry_delay_seconds: float = 0.15,
) -> Dict[str, Any]:
    """Validate grouped boundaries and modal wave ports with AEDT-aware naming.

    Assignment creation is synchronous at the scripting layer, but the boundary
    tree exposed by GetExcitations() can lag briefly after several consecutive
    AssignWavePort calls.  Polling also avoids treating that propagation delay as
    a failed build transaction.
    """
    group_id = str(group["group_id"])
    patch_name = group_patch_boundary_name(group_id)
    outer_name = group_outer_boundary_name(group_id)
    expected_ports = [str(row["port_name"]) for row in group_elements]

    last_boundaries: List[str] = []
    last_inventory: Dict[str, Any] = {
        "raw": [],
        "canonical_names": [],
        "records": [],
        "error": None,
    }
    missing_ports = list(expected_ports)
    missing_group_boundaries = [patch_name, outer_name]

    for attempt in range(max(1, int(retry_count))):
        last_boundaries = get_boundary_names(boundary)
        last_inventory = get_excitation_inventory(boundary)
        boundary_set = set(last_boundaries)
        excitation_set = set(last_inventory["canonical_names"])
        missing_ports = [name for name in expected_ports if name not in excitation_set]
        missing_group_boundaries = [
            name for name in (patch_name, outer_name) if name not in boundary_set
        ]
        if not missing_group_boundaries and not missing_ports:
            return {
                "ok": True,
                "group_id": group_id,
                "patch_boundary": patch_name,
                "outer_boundary": outer_name,
                "port_count": len(expected_ports),
                "ports": expected_ports,
                "excitation_raw": last_inventory["raw"],
                "excitation_canonical_names": last_inventory["canonical_names"],
                "validation_attempts": attempt + 1,
            }
        if attempt + 1 < max(1, int(retry_count)):
            time.sleep(float(retry_delay_seconds))

    if missing_group_boundaries:
        raise RuntimeError(
            f"Group {group_id} is missing grouped boundaries: {missing_group_boundaries}; "
            f"GetBoundaries={last_boundaries!r}"
        )
    raise RuntimeError(
        f"Group {group_id} is missing wave ports after canonical modal-name parsing: "
        f"{missing_ports}; GetExcitations raw={last_inventory['raw']!r}; "
        f"canonical={last_inventory['canonical_names']!r}; "
        f"records={last_inventory['records']!r}; error={last_inventory['error']!r}"
    )

def write_port_report(group_id: str, report: Dict[str, Any]) -> Path:
    path = GROUP_REPORT_DIR / f"{group_id}_ports.json"
    atomic_write_json(path, report)
    return path


def ensure_group_boundaries_and_ports(
    hfss,
    o_design,
    o_editor,
    manifest: Dict[str, Any],
    group: Dict[str, Any],
    group_elements: List[Dict[str, Any]],
) -> None:
    group_id = str(group["group_id"])
    state = manifest["groups"][group_id]
    boundary = get_module(hfss, o_design, "BoundarySetup")

    if state.get("port_status") == "passed":
        try:
            validate_group_ports(boundary, group, group_elements)
            state["assignment_status"] = "passed"
            save_manifest(manifest)
            log(f"[SKIP] {group_id} grouped Patch/outer boundaries and ports already passed.")
            return
        except Exception:
            log(f"[ASSIGNMENT RESUME INVALID] {group_id}; assignments will be recreated.")
            log(traceback.format_exc())

    patch_name = group_patch_boundary_name(group_id)
    outer_name = group_outer_boundary_name(group_id)
    port_names = [str(row["port_name"]) for row in group_elements]
    delete_boundary_assignments(boundary, [patch_name, outer_name] + port_names)

    state["port_status"] = "building"
    state["assignment_status"] = "building"
    state["port_attempts"] = int(state.get("port_attempts", 0)) + 1
    save_manifest(manifest)
    header(
        f"GROUP ASSIGNMENT START {group_id}: elements={len(group_elements)} "
        f"attempt={state['port_attempts']}"
    )
    start_time = time.perf_counter()
    try:
        outer_faces: List[int] = []
        outer_face_reports: List[Dict[str, Any]] = []
        for element in group_elements:
            coax_name = f"{element['unit_name']}_Coax_Teflon"
            face_report = find_outer_cylindrical_face(o_editor, coax_name)
            outer_faces.append(int(face_report["face_id"]))
            outer_face_reports.append(
                {
                    "unit_name": element["unit_name"],
                    "coax_name": coax_name,
                    **face_report,
                }
            )

        patch_objects = [f"{element['unit_name']}_Patch" for element in group_elements]
        boundary.AssignPerfectE(
            [
                f"NAME:{patch_name}",
                "Objects:=",
                patch_objects,
                "InfGroundPlane:=",
                False,
            ]
        )
        boundary.AssignPerfectE(
            [
                f"NAME:{outer_name}",
                "Faces:=",
                outer_faces,
                "InfGroundPlane:=",
                False,
            ]
        )
        for ordinal, element in enumerate(group_elements, start=1):
            # Keep the raw AssignWavePort parameter block identical to the
            # already validated SR2x2 builder.
            assign_one_modal_wave_port(boundary, element)
            log(
                f"[PORT {ordinal:02d}/{len(group_elements):02d}] "
                f"{element['port_name']} -> {element['unit_name']}_PortSheet"
            )

        # Flush the assignment tree, reacquire live AEDT proxies, and only then
        # inspect GetExcitations().  This addresses both modal :1 name suffixes
        # and a short BoundarySetup tree propagation delay.
        save_project(hfss)
        _, refreshed_design, _ = refresh_design_handles(hfss)
        boundary = get_module(hfss, refreshed_design, "BoundarySetup")
        inventory_before_validation = get_excitation_inventory(boundary)
        log(f"[EXCITATION RAW] {inventory_before_validation['raw']}")
        log(
            "[EXCITATION CANONICAL] "
            f"{inventory_before_validation['canonical_names']}"
        )
        report = validate_group_ports(boundary, group, group_elements)
        report["outer_faces"] = outer_face_reports
        report["started_at"] = now_iso()
        report["completed_at"] = now_iso()
        report["elapsed_seconds"] = time.perf_counter() - start_time
        report_path = write_port_report(group_id, report)
        save_project(hfss)
        state["port_status"] = "passed"
        state["assignment_status"] = "passed"
        state["port_completed_at"] = report["completed_at"]
        state["port_report"] = str(report_path)
        state["last_error_report"] = None
        save_manifest(manifest)
        log(
            f"[GROUP ASSIGNMENT PASS] {group_id}: patches={len(group_elements)} "
            f"outer_faces={len(outer_faces)} ports={len(group_elements)} "
            f"elapsed={report['elapsed_seconds']:.2f}s"
        )
    except Exception:
        exception_text = traceback.format_exc()
        cleanup_result: Dict[str, Any] = {"attempted": True}
        try:
            delete_boundary_assignments(boundary, [patch_name, outer_name] + port_names)
            cleanup_result["ok"] = True
        except Exception:
            cleanup_result["ok"] = False
            cleanup_result["cleanup_traceback"] = traceback.format_exc()
        error_group = dict(group)
        error_group["failure_stage"] = "group_boundaries_and_ports"
        json_path, txt_path = write_error_report(
            error_group,
            exception_text,
            get_object_names(o_editor),
            get_object_names(o_editor),
            cleanup_result,
        )
        state["port_status"] = "failed"
        state["assignment_status"] = "failed"
        state["port_completed_at"] = None
        state["last_error_report"] = str(json_path)
        save_manifest(manifest)
        log(f"[GROUP ASSIGNMENT FAILED] {group_id}")
        log(f"Error JSON: {json_path}")
        log(f"Error TXT : {txt_path}")
        log(exception_text)
        raise


def reconcile_group_port_checkpoints(
    hfss,
    o_design,
    manifest: Dict[str, Any],
    full_groups: Sequence[Dict[str, Any]],
    elements_by_index: Dict[int, Dict[str, Any]],
) -> None:
    boundary = get_module(hfss, o_design, "BoundarySetup")
    for group in full_groups:
        group_id = str(group["group_id"])
        state = manifest["groups"][group_id]
        group_elements = group_elements_from_plan(group, elements_by_index)
        if state.get("port_status") == "passed":
            try:
                validate_group_ports(boundary, group, group_elements)
                log(f"[ASSIGNMENT RESUME PASS] {group_id}")
                continue
            except Exception:
                log(f"[ASSIGNMENT RESUME RESET] {group_id}")
                log(traceback.format_exc())
        patch_name = group_patch_boundary_name(group_id)
        outer_name = group_outer_boundary_name(group_id)
        port_names = [str(row["port_name"]) for row in group_elements]
        delete_boundary_assignments(boundary, [patch_name, outer_name] + port_names)
        state["port_status"] = "pending"
        state["assignment_status"] = "pending"
        state["port_completed_at"] = None
        state["port_report"] = None
    save_project(hfss)
    save_manifest(manifest)


def ensure_setup_and_sweep(hfss, o_design, manifest: Dict[str, Any]) -> None:
    analysis = get_module(hfss, o_design, "AnalysisSetup")
    try:
        setups = [str(value) for value in list(analysis.GetSetups())]
    except Exception:
        setups = []

    if SETUP_NAME not in setups:
        header("Create Setup1")
        analysis.InsertSetup(
            "HfssDriven",
            [
                f"NAME:{SETUP_NAME}",
                "SolveType:=",
                "Single",
                "Frequency:=",
                "2.4GHz",
                "MaxDeltaS:=",
                0.02,
                "UseMatrixConv:=",
                False,
                "MaximumPasses:=",
                20,
                "MinimumPasses:=",
                1,
                "MinimumConvergedPasses:=",
                1,
                "PercentRefinement:=",
                30,
                "IsEnabled:=",
                True,
                ["NAME:MeshLink", "ImportMesh:=", False],
                "BasisOrder:=",
                1,
                "DoLambdaRefine:=",
                True,
                "DoMaterialLambda:=",
                True,
                "SetLambdaTarget:=",
                False,
                "Target:=",
                0.3333,
                "UseMaxTetIncrease:=",
                False,
                "PortAccuracy:=",
                2,
                "UseABCOnPort:=",
                False,
                "SetPortMinMaxTri:=",
                False,
                "DrivenSolverType:=",
                "Direct Solver",
                "EnhancedLowFreqAccuracy:=",
                False,
                "SaveRadFieldsOnly:=",
                False,
                "SaveAnyFields:=",
                True,
                "IESolverType:=",
                "Auto",
                "LambdaTargetForIESolver:=",
                0.15,
                "UseDefaultLambdaTgtForIESolver:=",
                True,
                "IE Solver Accuracy:=",
                "Balanced",
                "InfiniteSphereSetup:=",
                "",
                "MaxPass:=",
                10,
                "MinPass:=",
                1,
                "MinConvPass:=",
                1,
                "PerError:=",
                1,
                "PerRefine:=",
                30,
            ],
        )
        manifest["setup_status"] = "passed"
        save_manifest(manifest)
        save_project(hfss)
    else:
        manifest["setup_status"] = "passed"

    try:
        sweeps = [str(value) for value in list(analysis.GetSweeps(SETUP_NAME))]
    except Exception:
        sweeps = []
    if SWEEP_NAME not in sweeps:
        header("Create discrete Sweep")
        analysis.InsertFrequencySweep(
            SETUP_NAME,
            [
                f"NAME:{SWEEP_NAME}",
                "IsEnabled:=",
                True,
                "RangeType:=",
                "LinearStep",
                "RangeStart:=",
                "2.3GHz",
                "RangeEnd:=",
                "2.5GHz",
                "RangeStep:=",
                "0.01GHz",
                "Type:=",
                "Discrete",
                "SaveFields:=",
                True,
                "SaveRadFields:=",
                False,
            ],
        )
        save_project(hfss)
    manifest["sweep_status"] = "passed"
    save_manifest(manifest)


def create_full_summary(
    manifest: Dict[str, Any],
    full_groups: List[Dict[str, Any]],
    elements: List[Dict[str, Any]],
    elements_by_index: Dict[int, Dict[str, Any]],
    o_design,
    o_editor,
    board_dimensions: Dict[str, float],
) -> Dict[str, Any]:
    geometry_passed = [
        group["group_id"]
        for group in full_groups
        if manifest["groups"][group["group_id"]].get("geometry_status") == "passed"
    ]
    board_passed = [
        group["group_id"]
        for group in full_groups
        if manifest["groups"][group["group_id"]].get("board_status") == "passed"
    ]
    ports_passed = [
        group["group_id"]
        for group in full_groups
        if manifest["groups"][group["group_id"]].get("port_status") == "passed"
    ]

    names = set(get_object_names(o_editor))
    expected_objects: List[str] = []
    for element in elements:
        expected_objects.extend(expected_names_for_element(element, include_gnd_hole=False))
    expected_objects.extend([GND_NAME, SUBSTRATE_NAME, AIRBOX_NAME])
    missing_objects = sorted(set(expected_objects) - names)
    master_leftovers = sorted(name for name in names if name.startswith("MASTER_"))
    gnd_hole_leftovers = sorted(name for name in names if name.endswith("_GND_Hole"))
    backup_leftovers = sorted(
        name for name in names if name in {GND_BACKUP_NAME, SUBSTRATE_BACKUP_NAME}
    )

    board_report = validate_shared_boards(o_editor, board_dimensions)
    airbox_report = validate_airbox(o_editor, board_dimensions)
    boundary = get_module(None if False else object(), o_design, "BoundarySetup") if False else o_design.GetModule("BoundarySetup")
    if boundary is None:
        raise RuntimeError("BoundarySetup module is unavailable during final verification.")
    boundary_names = get_boundary_names(boundary)
    excitation_inventory = get_excitation_inventory(boundary)
    excitation_names = list(excitation_inventory["canonical_names"])

    expected_boundaries = {
        COMMON_GND_BOUNDARY,
        RADIATION_BOUNDARY,
        *[group_patch_boundary_name(group["group_id"]) for group in full_groups],
        *[group_outer_boundary_name(group["group_id"]) for group in full_groups],
    }
    missing_boundaries = sorted(expected_boundaries - set(boundary_names))
    unexpected_boundaries = sorted(set(boundary_names) - expected_boundaries)
    expected_ports = {str(element["port_name"]) for element in elements}
    missing_ports = sorted(expected_ports - set(excitation_names))
    unexpected_ports = sorted(set(excitation_names) - expected_ports)

    analysis = o_design.GetModule("AnalysisSetup")
    setups = [str(value) for value in list(analysis.GetSetups())] if analysis is not None else []
    sweeps = (
        [str(value) for value in list(analysis.GetSweeps(SETUP_NAME))]
        if analysis is not None and SETUP_NAME in setups
        else []
    )

    expected_total_objects = 4 * len(elements) + 3
    ok = (
        len(geometry_passed) == len(full_groups)
        and len(board_passed) == len(full_groups)
        and len(ports_passed) == len(full_groups)
        and not missing_objects
        and not master_leftovers
        and not gnd_hole_leftovers
        and not backup_leftovers
        and not missing_boundaries
        and not unexpected_boundaries
        and not missing_ports
        and not unexpected_ports
        and SETUP_NAME in setups
        and SWEEP_NAME in sweeps
        and len(names) == expected_total_objects
    )

    summary = {
        "ok": ok,
        "timestamp": now_iso(),
        "script_revision": SCRIPT_REVISION,
        "phase": "complete_187_element_grouped_assignment_build",
        "project_path": str(PROJECT_PATH),
        "element_count": len(elements),
        "group_count": len(full_groups),
        "geometry_groups_passed": geometry_passed,
        "board_groups_passed": board_passed,
        "port_groups_passed": ports_passed,
        "expected_total_object_count": expected_total_objects,
        "actual_total_object_count": len(names),
        "missing_objects": missing_objects,
        "master_leftovers": master_leftovers,
        "gnd_hole_leftovers": gnd_hole_leftovers,
        "backup_leftovers": backup_leftovers,
        "boards": board_report,
        "airbox": airbox_report,
        "boundary_count": len(boundary_names),
        "boundaries": boundary_names,
        "expected_boundary_count": len(expected_boundaries),
        "missing_boundaries": missing_boundaries,
        "unexpected_boundaries": unexpected_boundaries,
        "port_count": len(excitation_names),
        "expected_port_count": len(expected_ports),
        "missing_ports": missing_ports,
        "unexpected_ports": unexpected_ports,
        "setups": setups,
        "sweeps": sweeps,
        "analysis_run": False,
        "far_field_created": False,
        "plan_csv": str(PLAN_CSV_PATH),
        "manifest": str(MANIFEST_PATH),
        "log": str(LOG_PATH),
    }
    atomic_write_json(SUMMARY_PATH, summary)
    return summary


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    inputs = load_stage_inputs()
    validate_stage_inputs(inputs)
    elements, full_groups = build_full_element_plan(inputs)
    selected_groups = full_groups
    selected_group_ids = [group["group_id"] for group in selected_groups]
    elements_by_index = {int(row["global_index"]): row for row in elements}
    board_dimensions = compute_square_board_dimensions(elements)

    resume = determine_resume_mode()
    if not resume and LOG_PATH.exists():
        LOG_PATH.unlink()

    header("LARGE ARRAY FULL BUILD: 187 ELEMENTS")
    log(f"Timestamp           : {now_iso()}")
    log(f"Python executable   : {sys.executable}")
    log(f"Working directory   : {os.getcwd()}")
    log(f"Script revision     : {SCRIPT_REVISION}")
    log(f"RUN_MODE            : {RUN_MODE}")
    log(f"Resume              : {resume}")
    log(f"Stage1 NPZ          : {STAGE1_NPZ_PATH}")
    log(f"Stage2 seeds        : {STAGE2_SEEDS_JSON_PATH}")
    log(f"Output project      : {PROJECT_PATH}")
    log(f"State directory     : {STATE_DIR}")
    log(f"lambda0             : {LAMBDA0_MM:.12f} mm")
    log(f"Full array elements : {len(elements)}")
    log(f"Geometry groups     : {selected_group_ids}")
    log(f"Square board width  : {board_dimensions['board_width_mm']:.9f} mm")
    log(f"Board edge margin   : {board_dimensions['edge_margin_mm']:.6f} mm")

    if resume:
        manifest = read_json(MANIFEST_PATH)
        validate_manifest_inputs(manifest, inputs)
    else:
        manifest = new_manifest(inputs, full_groups, selected_groups, board_dimensions)
    write_plan_csv(elements, selected_group_ids)
    log(f"[OK] Full 187-element plan written: {PLAN_CSV_PATH}")
    upgrade_manifest_for_full_run(manifest, inputs, full_groups, board_dimensions)
    save_manifest(manifest)

    header("Launch/open HFSS Modal project")
    hfss = launch_hfss(resume)
    _, o_design, o_editor = refresh_design_handles(hfss)
    if resume:
        validate_cut_l_variable(o_design)
    set_model_units_mm(o_editor)
    ensure_design_variables(o_design)
    validate_cut_l_variable(o_design)
    if not manifest.get("design_initialized"):
        save_project(hfss)
        manifest["design_initialized"] = True
        save_manifest(manifest)

    if not manifest.get("boards_initialized"):
        header("Create final-size square GND and substrate before all groups")
        create_square_shared_boards(o_editor, board_dimensions)
        validate_shared_boards(o_editor, board_dimensions)
        save_project(hfss)
        manifest["boards_initialized"] = True
        save_manifest(manifest)
    else:
        validate_shared_boards(o_editor, board_dimensions)

    if resume:
        header("Reconcile geometry/board checkpoints against actual AEDT model")
        reconcile_checkpoint_groups(
            hfss,
            o_editor,
            manifest,
            selected_groups,
            elements_by_index,
            board_dimensions,
        )

    if resume:
        header("Reconcile grouped boundary/port checkpoints")
        reconcile_group_port_checkpoints(
            hfss, o_design, manifest, selected_groups, elements_by_index
        )

    for group in selected_groups:
        group_id = group["group_id"]
        state = manifest["groups"][group_id]
        group_elements = group_elements_from_plan(group, elements_by_index)
        if (
            state.get("geometry_status") == "passed"
            and state.get("board_status") == "passed"
        ):
            log(f"[SKIP] {group_id} geometry and board cuts already passed.")
        else:
            process_one_group(
                hfss,
                o_editor,
                manifest,
                group,
                group_elements,
                board_dimensions,
            )
            _, o_design, o_editor = refresh_design_handles(hfss)

        # User-required order: finish and commit each group completely before
        # starting the next group.  Patch/outer boundaries and wave ports are
        # assigned here, not deferred until the complete array exists.
        ensure_group_boundaries_and_ports(
            hfss, o_design, o_editor, manifest, group, group_elements
        )
        _, o_design, o_editor = refresh_design_handles(hfss)

    header("Verify all 187 geometries and board cuts")
    for group in selected_groups:
        group_elements = group_elements_from_plan(group, elements_by_index)
        validate_group_geometry(o_editor, group, group_elements, include_gnd_holes=False)
        validate_group_board_cut(o_editor, group, group_elements, board_dimensions)
    validate_shared_boards(o_editor, board_dimensions)
    save_project(hfss)

    ensure_shared_airbox(hfss, o_editor, manifest, board_dimensions)
    _, o_design, o_editor = refresh_design_handles(hfss)
    ensure_global_boundaries(hfss, o_design, manifest)

    ensure_setup_and_sweep(hfss, o_design, manifest)
    save_project(hfss)
    _, o_design, o_editor = refresh_design_handles(hfss)

    header("Complete-array final verification")
    summary = create_full_summary(
        manifest,
        selected_groups,
        elements,
        elements_by_index,
        o_design,
        o_editor,
        board_dimensions,
    )
    if not summary["ok"]:
        raise RuntimeError(f"Complete-array final verification failed: {summary}")

    manifest["full_build_complete"] = True
    save_manifest(manifest)

    header("LARGE ARRAY FULL BUILD: PASS")
    log(f"Geometry groups : {len(summary['geometry_groups_passed'])}/{len(selected_groups)} passed")
    log(f"Board groups    : {len(summary['board_groups_passed'])}/{len(selected_groups)} passed")
    log(f"Assignment groups: {len(summary['port_groups_passed'])}/{len(selected_groups)} passed")
    log(f"Elements        : {summary['element_count']}")
    log(f"Final objects   : {summary['actual_total_object_count']}")
    log(f"Boundaries      : {summary['boundary_count']}")
    log(f"Wave ports      : {summary['port_count']}")
    log(f"Project         : {PROJECT_PATH}")
    log(f"Plan CSV        : {PLAN_CSV_PATH}")
    log(f"Manifest        : {MANIFEST_PATH}")
    log(f"Summary         : {SUMMARY_PATH}")
    log(f"Log             : {LOG_PATH}")
    log(
        f"GND/Substrate  : SQUARE, width={board_dimensions['board_width_mm']:.9f}mm, "
        "187 apertures/pin holes committed"
    )
    log("AirBox          : CREATED")
    log("Setup/Sweep     : CREATED")
    log("Analysis        : NOT RUN")
    log("Far field       : NOT CREATED")
    if not CLOSE_ON_EXIT:
        log("AEDT remains open because CLOSE_ON_EXIT=False.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            log("[FATAL] Complete large-array build failed.")
            log(traceback.format_exc())
        finally:
            raise
