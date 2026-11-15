# -*- coding: utf-8 -*-
"""
export_embedded_patterns.py

One-click PyAEDT post-processing exporter for the solved 187-port Top1
sequential-rotation circular-ring array with Cut_L = 3.60 mm.

Scope
-----
- Open the already solved HFSS project; do not analyze or remesh.
- Read the exact 187-element/port mapping written by the large-array builder.
- Excite one port at a time with 1 W and 0 deg while explicitly setting all
  other 186 ports to 0 W and 0 deg in the same EditSources call.
- Export the full complex vector active-element pattern (Vec-AEP) for every
  port by replaying the already validated RadField.ExportFieldsToFile call.
- Preserve the raw global-coordinate rEtheta/rEphi complex fields.  Do not add
  scan phase, sequential-rotation compensation phase, position de-embedding,
  circular-polarization conversion, or coordinate back-rotation.
- Write stable element/port/file mappings for later Vec-AEP synthesis.

The new Cut_L = 3.60 mm project is identified by its cut360 filename and the
matching build-state directory. The exported variation is fixed below.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import shutil
import sys
import sysconfig
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


# =============================================================================
# User configuration
# =============================================================================

AEDT_VERSION = "242"
DESIGN_NAME = "HFSSDesign1_LargeArray"
NON_GRAPHICAL = False
NEW_DESKTOP = True
CLOSE_DESKTOP_ON_EXIT = False
SCRIPT_REVISION = "top1_sr_cut360_legacyenv_vecaep_allports_v1"

NEW_PROJECT_NAME = "cutcorner_large_array_top1_cut360_legacyenv.aedt"


def discover_new_project() -> Tuple[Path, Path]:
    """Locate the new cut360 project without a drive-specific default path."""
    explicit_project = os.environ.get("DMPS_HFSS_SOLVED_PROJECT")
    if explicit_project:
        project = Path(explicit_project)
        if "cut365" in project.stem.lower():
            raise RuntimeError("The new cut360 exporter must not open a legacy cut365 project.")
        root = project.parent.parent if project.parent.name.lower() == "outputs" else project.parent
        return root, project

    explicit_root = os.environ.get("DMPS_HFSS_SERVER_ROOT")
    if explicit_root:
        roots = [Path(explicit_root)]
    else:
        roots: List[Path] = []
        seen: set[str] = set()
        for anchor in (Path.cwd(), Path(__file__).resolve().parents[2]):
            for parent in (anchor, *anchor.parents):
                if parent == parent.parent:
                    continue
                for root in (parent, parent / "hfssproject", parent / "rebuild" / "hfssproject"):
                    key = os.path.normcase(str(root))
                    if key not in seen:
                        roots.append(root)
                        seen.add(key)

    matches: List[Tuple[Path, Path]] = []
    for root in roots:
        # The transferred and solved project belongs at the server root.
        at_root = root / NEW_PROJECT_NAME
        in_outputs = root / "outputs" / NEW_PROJECT_NAME
        if at_root.is_file():
            matches.append((root, at_root))
        elif in_outputs.is_file():
            matches.append((root, in_outputs))
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple distinct new cut360 projects found; set DMPS_HFSS_SERVER_ROOT: "
            + ", ".join(str(project) for _, project in matches)
        )
    if matches:
        return matches[0]
    root = roots[0] if explicit_root else Path.cwd()
    return root, root / NEW_PROJECT_NAME


SERVER_ROOT, DISCOVERED_PROJECT_PATH = discover_new_project()
PROJECT_CANDIDATES: Tuple[Path, ...] = (DISCOVERED_PROJECT_PATH,)
if os.environ.get("DMPS_HFSS_PLAN_CSV"):
    PLAN_CANDIDATES: Tuple[Path, ...] = (Path(os.environ["DMPS_HFSS_PLAN_CSV"]),)
else:
    PLAN_CANDIDATES = tuple(dict.fromkeys((
        SERVER_ROOT / "outputs" / f"{DISCOVERED_PROJECT_PATH.stem}_state" / "array_element_plan.csv",
        DISCOVERED_PROJECT_PATH.parent / f"{DISCOVERED_PROJECT_PATH.stem}_state" / "array_element_plan.csv",
    )))

EXPORT_PARENT = Path(os.environ.get(
    "DMPS_HFSS_VECAEP_OUTPUT_DIR", str(SERVER_ROOT / "vecaep_exports")
))
EXPORT_CASE_NAME = "top1_sr_cut360_legacyenv_2p4GHz_allports"
FREQ_GHZ = 2.4
C0_MM_PER_S = 299_792_458_000.0
LAMBDA0_MM = C0_MM_PER_S / (FREQ_GHZ * 1.0e9)
EXPECTED_PORT_COUNT = 187

# Vec-AEP source convention: one active port at 1 W, all others at 0 W.
ACTIVE_PORT_POWER_W = 1.0
INACTIVE_PORT_POWER_W = 0.0
SOURCE_PHASE_DEG = 0.0

# Recorded export settings. Keep these strings exactly aligned with the
# successful AEDT recorder output supplied by the user.
FARFIELD_SETUP = "Infinite Sphere1"
SOLUTION_NAME = "Setup1 : LastAdaptive"
INTRINSIC_VARIATION_KEY = "Freq='2.3999999999999999GHz'"
DESIGN_VARIATION_KEY = (
    "Conn_L='5mm' "
    "Cut_L='3.6000000000000001mm' "
    "Feed_Pos='7.25mm' "
    "freq_='2.3999999999999999GHz' "
    "Patch_L='28.899999999999999mm' "
    "Pin_R='0.63500000000000001mm' "
    "Sub_h='1.6000000000000001mm' "
    "Sub_W='80mm' "
    "Teflon_R='2.0499999999999998mm'"
)

OVERWRITE_EXISTING_EXPORT_FILE = False
MIN_EXPORT_BYTES = 256
SOURCE_SYNC_DELAY_S = 1.0


# =============================================================================
# Runtime paths and logging
# =============================================================================

RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = EXPORT_PARENT / EXPORT_CASE_NAME / RUN_STAMP
LOG_PATH = RUN_DIR / "export_embedded_patterns.log"
RUN_CONFIG_PATH = RUN_DIR / "run_config.json"
AEP_MANIFEST_PATH = RUN_DIR / "vecaep_manifest.csv"
PLAN_SNAPSHOT_PATH = RUN_DIR / "array_element_plan_snapshot.csv"
SUMMARY_PATH = RUN_DIR / "run_summary.json"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(message: str = "") -> None:
    text = str(message)
    print(text, flush=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")


def header(title: str) -> None:
    log()
    log("=" * 112)
    log(title)
    log("=" * 112)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temp_path, path)


# =============================================================================
# Import protection and PyAEDT loading
# =============================================================================


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


def import_pyaedt():
    preload_stdlib_copy_module()
    import ansys.aedt.core as aedt_core
    from ansys.aedt.core import Hfss

    return Hfss, "ansys.aedt.core", getattr(aedt_core, "__version__", "unknown")


# =============================================================================
# Input discovery and plan validation
# =============================================================================


def choose_existing_path(candidates: Sequence[Path], label: str) -> Path:
    found = [path for path in candidates if path.is_file()]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        # Prefer the standard outputs layout when both copies exist.
        preferred = [path for path in found if "outputs" in path.parts]
        chosen = preferred[0] if preferred else found[0]
        log(f"[WARN] Multiple {label} candidates found; using: {chosen}")
        return chosen
    lines = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{lines}")


def validate_new_project_plan_pair(project_path: Path, plan_path: Path) -> None:
    """Reject a mismatched plan/state archive before touching HFSS."""
    if not os.environ.get("DMPS_HFSS_PLAN_CSV") and plan_path.parent.name != f"{project_path.stem}_state":
        raise RuntimeError(f"Plan state directory does not match project: {plan_path}")
    manifest_path = plan_path.parent / "build_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if (manifest.get("script_revision") != "cutcorner_cra_cut360_v1"
                or float(manifest.get("cut_l_mm", -1)) != 3.60
                or int(manifest.get("full_element_count", -1)) != EXPECTED_PORT_COUNT):
            raise RuntimeError(f"Plan manifest is not the new 187-element cut360 build: {manifest_path}")


def read_plan_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [dict(row) for row in reader]

    required = {
        "global_index",
        "unit_name",
        "x_mm",
        "y_mm",
        "rotation_deg",
        "port_name",
    }
    if not rows:
        raise RuntimeError(f"Plan CSV is empty: {path}")
    missing_columns = sorted(required - set(rows[0].keys()))
    if missing_columns:
        raise RuntimeError(f"Plan CSV is missing columns: {missing_columns}")

    parsed: List[Dict[str, Any]] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            parsed.append(
                {
                    "global_index": int(row["global_index"]),
                    "unit_name": str(row["unit_name"]).strip(),
                    "x_mm": float(row["x_mm"]),
                    "y_mm": float(row["y_mm"]),
                    "rotation_deg": float(row["rotation_deg"]),
                    "port_name": str(row["port_name"]).strip(),
                    "ring_index": int(row.get("ring_index", 0) or 0),
                    "ring_local_index": int(row.get("ring_local_index", 0) or 0),
                    "group_id": str(row.get("group_id", "")).strip(),
                }
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot parse plan CSV line {line_number}: {row}; error={exc!r}"
            ) from exc

    parsed.sort(key=lambda item: int(item["global_index"]))
    return parsed


def validate_plan(rows: Sequence[Dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_PORT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PORT_COUNT} plan rows, got {len(rows)}"
        )

    indices = [int(row["global_index"]) for row in rows]
    expected_indices = list(range(EXPECTED_PORT_COUNT))
    if indices != expected_indices:
        raise RuntimeError(
            "Plan global_index order is invalid; expected 0..186 exactly."
        )

    ports = [str(row["port_name"]) for row in rows]
    expected_ports = [f"p{index:03d}" for index in range(1, EXPECTED_PORT_COUNT + 1)]
    if ports != expected_ports:
        mismatch = [
            (index, actual, expected)
            for index, (actual, expected) in enumerate(zip(ports, expected_ports))
            if actual != expected
        ][:10]
        raise RuntimeError(f"Plan port order mismatch; first mismatches={mismatch}")

    for row in rows:
        for field in ("x_mm", "y_mm", "rotation_deg"):
            value = float(row[field])
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Non-finite {field} for {row['unit_name']}: {value}"
                )

    center = rows[0]
    if math.hypot(float(center["x_mm"]), float(center["y_mm"])) > 1.0e-6:
        raise RuntimeError(
            f"Plan row 0 is not the center element: x={center['x_mm']}, y={center['y_mm']}"
        )


# =============================================================================
# Single-port Vec-AEP source construction
# =============================================================================


def source_name_from_port(port_name: str) -> str:
    text = str(port_name).strip()
    return text if ":" in text else f"{text}:1"


def build_single_port_source_arguments(
    plan_rows: Sequence[Dict[str, Any]],
    active_global_index: int,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Create one 187-source EditSources transaction for a single Vec-AEP."""
    args: List[Any] = [
        [
            "IncludePortPostProcessing:=",
            True,
            "SpecifySystemPower:=",
            False,
        ]
    ]
    source_rows: List[Dict[str, Any]] = []
    active_count = 0

    for row in plan_rows:
        global_index = int(row["global_index"])
        is_active = global_index == int(active_global_index)
        magnitude_w = ACTIVE_PORT_POWER_W if is_active else INACTIVE_PORT_POWER_W
        if is_active:
            active_count += 1

        source_name = source_name_from_port(str(row["port_name"]))
        args.append(
            [
                "Name:=",
                source_name,
                "Magnitude:=",
                f"{magnitude_w:.16g}W",
                "Phase:=",
                f"{SOURCE_PHASE_DEG:.12g}deg",
            ]
        )
        source_rows.append(
            {
                "global_index": global_index,
                "unit_name": str(row["unit_name"]),
                "ring_index": int(row.get("ring_index", 0)),
                "ring_local_index": int(row.get("ring_local_index", 0)),
                "group_id": str(row.get("group_id", "")),
                "port_name": str(row["port_name"]),
                "source_name": source_name,
                "x_mm": float(row["x_mm"]),
                "y_mm": float(row["y_mm"]),
                "rotation_deg": float(row["rotation_deg"]),
                "is_active": is_active,
                "magnitude_W": magnitude_w,
                "phase_deg": SOURCE_PHASE_DEG,
            }
        )

    if len(args) != EXPECTED_PORT_COUNT + 1:
        raise RuntimeError(f"EditSources argument count is invalid: {len(args)}")
    if active_count != 1:
        raise RuntimeError(
            f"Expected exactly one active source for global_index={active_global_index}, "
            f"got {active_count}"
        )
    return args, source_rows


def write_source_state_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("Cannot write an empty source-state CSV.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# AEDT inventory and raw module helpers
# =============================================================================


def canonical_excitation_names(raw_values: Iterable[Any]) -> List[str]:
    raw = [str(value).strip() for value in raw_values]
    if len(raw) >= 2 and len(raw) % 2 == 0:
        type_slots = raw[1::2]
        if any("port" in value.lower() for value in type_slots):
            raw = raw[0::2]
    names = []
    for value in raw:
        if not value or "port" == value.lower() or "wave port" == value.lower():
            continue
        names.append(source_name_from_port(value))
    return sorted(set(names))


def validate_hfss_port_inventory(o_design, plan_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    boundary = o_design.GetModule("BoundarySetup")
    raw = list(boundary.GetExcitations())
    actual = canonical_excitation_names(raw)
    expected = sorted(source_name_from_port(str(row["port_name"])) for row in plan_rows)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected or len(actual) != EXPECTED_PORT_COUNT:
        raise RuntimeError(
            "HFSS port inventory mismatch: "
            f"count={len(actual)}, missing={missing[:20]}, unexpected={unexpected[:20]}, "
            f"raw_sample={[str(value) for value in raw[:20]]}"
        )
    return {
        "raw_count": len(raw),
        "canonical_count": len(actual),
        "raw_sample": [str(value) for value in raw[:20]],
        "canonical_first": actual[:10],
        "canonical_last": actual[-10:],
    }


def get_analysis_inventory(o_design) -> Dict[str, Any]:
    module = o_design.GetModule("AnalysisSetup")
    setups = [str(value) for value in list(module.GetSetups())]
    sweeps: List[str] = []
    if "Setup1" in setups:
        try:
            sweeps = [str(value) for value in list(module.GetSweeps("Setup1"))]
        except Exception:
            sweeps = []
    if "Setup1" not in setups:
        raise RuntimeError(f"Setup1 is missing. Existing setups={setups}")
    return {"setups": setups, "sweeps": sweeps}


def locate_results_sidecars(project_path: Path) -> List[Path]:
    candidates = [
        Path(str(project_path) + "results"),
        project_path.with_suffix(".aedtresults"),
    ]
    return [path for path in candidates if path.exists()]


# =============================================================================
# Export helpers
# =============================================================================


def vecaep_label(row: Dict[str, Any]) -> str:
    return (
        f"g{int(row['global_index']):03d}_"
        f"{str(row['port_name']).strip()}_"
        f"{str(row['unit_name']).strip()}"
    )


def export_farfield_exact_recorded_call(
    o_radfield,
    export_path: Path,
) -> None:
    """Replay the user's successful AEDT recorder call structure exactly."""
    o_radfield.ExportFieldsToFile(
        [
            "ExportFileName:=",
            str(export_path),
            "SetupName:=",
            FARFIELD_SETUP,
            "IntrinsicVariationKey:=",
            INTRINSIC_VARIATION_KEY,
            "DesignVariationKey:=",
            DESIGN_VARIATION_KEY,
            "SolutionName:=",
            SOLUTION_NAME,
            "Quantity:=",
            "",
        ]
    )


def validate_export_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"AEDT did not create the export file: {path}")
    size = path.stat().st_size
    if size < MIN_EXPORT_BYTES:
        raise RuntimeError(
            f"Export file is unexpectedly small: {path}, size={size} bytes"
        )
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        preview = [stream.readline().rstrip("\r\n") for _ in range(5)]
    header_line = preview[0] if preview else ""
    required_columns = ("Phi(rad)", "Theta(rad)", "rEphi(mag ang)", "rEtheta(mag ang)")
    missing_columns = [name for name in required_columns if name not in header_line]
    if missing_columns:
        raise RuntimeError(
            f"Export header is missing required columns {missing_columns}: {header_line!r}"
        )
    return {"path": str(path), "size_bytes": size, "preview_lines": preview}


def append_vecaep_manifest(row: Dict[str, Any]) -> None:
    fieldnames = [
        "global_index",
        "unit_name",
        "port_name",
        "source_name",
        "ring_index",
        "ring_local_index",
        "group_id",
        "x_mm",
        "y_mm",
        "rotation_deg",
        "excitation_magnitude_W",
        "excitation_phase_deg",
        "status",
        "started_at",
        "completed_at",
        "elapsed_s",
        "vecaep_file",
        "vecaep_size_bytes",
        "source_state_file",
        "metadata_file",
        "error",
    ]
    exists = AEP_MANIFEST_PATH.is_file()
    with AEP_MANIFEST_PATH.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


# =============================================================================
# Main workflow
# =============================================================================


def main() -> None:
    project_path = choose_existing_path(PROJECT_CANDIDATES, "new cut360 HFSS project")
    plan_path = choose_existing_path(PLAN_CANDIDATES, "matching array element plan CSV")
    validate_new_project_plan_pair(project_path, plan_path)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    if AEP_MANIFEST_PATH.exists():
        AEP_MANIFEST_PATH.unlink()

    header("TOP1 SR CUT_L=3.60 MM VECTOR AEP EXPORT: ALL 187 PORTS")
    log(f"Timestamp              : {now_iso()}")
    log(f"Python executable      : {sys.executable}")
    log(f"Working directory      : {os.getcwd()}")
    log(f"Script revision        : {SCRIPT_REVISION}")
    log(f"Server root            : {SERVER_ROOT}")
    log(f"Run directory          : {RUN_DIR}")
    log(f"Frequency              : {FREQ_GHZ} GHz")
    log(f"lambda0                : {LAMBDA0_MM:.12f} mm")
    log(f"Active-port power      : {ACTIVE_PORT_POWER_W:.12g} W")
    log(f"Inactive-port power    : {INACTIVE_PORT_POWER_W:.12g} W")
    log(f"Source phase           : {SOURCE_PHASE_DEG:.12g} deg")
    log("Analysis               : NOT RUN")

    log(f"HFSS project           : {project_path}")
    log(f"Array plan             : {plan_path}")

    header("Read and validate exact 187-element plan")
    plan_rows = read_plan_csv(plan_path)
    validate_plan(plan_rows)
    shutil.copy2(plan_path, PLAN_SNAPSHOT_PATH)
    rotation_values = [float(row["rotation_deg"]) for row in plan_rows]
    log(f"[OK] Plan rows         : {len(plan_rows)}")
    log(f"[OK] Ports             : {plan_rows[0]['port_name']} ... {plan_rows[-1]['port_name']}")
    log(
        "[OK] Rotation range    : "
        f"{min(rotation_values):.6f} ... {max(rotation_values):.6f} deg"
    )

    results_sidecars = locate_results_sidecars(project_path)
    if results_sidecars:
        log(f"[OK] Result sidecar(s) : {[str(path) for path in results_sidecars]}")
    else:
        log(
            "[WARN] No result sidecar was found by the standard path checks. "
            "AEDT will still be opened, but Vec-AEP export requires solved field data."
        )

    config_payload = {
        "script_revision": SCRIPT_REVISION,
        "created_at": now_iso(),
        "project_path": str(project_path),
        "design_name": DESIGN_NAME,
        "plan_path": str(plan_path),
        "run_dir": str(RUN_DIR),
        "frequency_ghz": FREQ_GHZ,
        "lambda0_mm": LAMBDA0_MM,
        "expected_port_count": EXPECTED_PORT_COUNT,
        "active_port_power_w": ACTIVE_PORT_POWER_W,
        "inactive_port_power_w": INACTIVE_PORT_POWER_W,
        "source_phase_deg": SOURCE_PHASE_DEG,
        "farfield_setup": FARFIELD_SETUP,
        "solution_name": SOLUTION_NAME,
        "intrinsic_variation_key": INTRINSIC_VARIATION_KEY,
        "design_variation_key": DESIGN_VARIATION_KEY,
        "field_components": ["rEtheta(mag ang)", "rEphi(mag ang)"],
        "coordinate_system": "HFSS exported global spherical basis",
        "position_phase_deembedded": False,
        "physical_rotation_backrotated": False,
        "circular_components_precomputed": False,
        "export_method": "RadField.ExportFieldsToFile recorded raw call",
        "analysis_run": False,
    }
    write_json(RUN_CONFIG_PATH, config_payload)

    hfss = None
    successful_indices: List[int] = []
    successful_ports: List[str] = []
    failed_indices: List[int] = []
    failed_ports: List[str] = []
    run_started = time.perf_counter()

    try:
        header("Launch AEDT 2024 R2 and open solved project")
        Hfss, namespace, version = import_pyaedt()
        log(f"PyAEDT namespace       : {namespace}")
        log(f"PyAEDT version         : {version}")
        hfss = Hfss(
            project=str(project_path),
            design=DESIGN_NAME,
            version=AEDT_VERSION,
            solution_type="Modal",
            non_graphical=NON_GRAPHICAL,
            new_desktop=NEW_DESKTOP,
            close_on_exit=False,
        )
        o_project = hfss.oproject
        o_design = o_project.SetActiveDesign(DESIGN_NAME)
        if o_design is None:
            o_design = hfss.odesign
        if o_design is None:
            raise RuntimeError(f"Could not activate design: {DESIGN_NAME}")

        project_name = str(o_project.GetName())
        design_name = str(o_design.GetName())
        log(f"[OK] Active project    : {project_name}")
        log(f"[OK] Active design     : {design_name}")

        header("Validate solution and 187-port inventory")
        analysis_inventory = get_analysis_inventory(o_design)
        port_inventory = validate_hfss_port_inventory(o_design, plan_rows)
        log(f"[OK] Setups            : {analysis_inventory['setups']}")
        log(f"[OK] Sweeps            : {analysis_inventory['sweeps']}")
        log(f"[OK] Canonical ports   : {port_inventory['canonical_count']}")
        log(f"[PORT RAW SAMPLE] {port_inventory['raw_sample']}")

        o_solutions = o_design.GetModule("Solutions")
        o_radfield = o_design.GetModule("RadField")
        if o_solutions is None or o_radfield is None:
            raise RuntimeError("Could not acquire Solutions or RadField module.")

        header("Excite one port at a time and export all vector active-element patterns")
        for ordinal, active_row in enumerate(plan_rows, start=1):
            global_index = int(active_row["global_index"])
            unit_name = str(active_row["unit_name"])
            port_name = str(active_row["port_name"])
            source_name = source_name_from_port(port_name)
            label = vecaep_label(active_row)
            aep_dir = RUN_DIR / label
            aep_dir.mkdir(parents=True, exist_ok=True)
            vecaep_path = aep_dir / f"vecaep_{label}.csv"
            source_state_path = aep_dir / f"source_state_{label}.csv"
            metadata_path = aep_dir / "metadata.json"

            started_at = now_iso()
            aep_started = time.perf_counter()
            header(
                f"VEC-AEP {ordinal}/{len(plan_rows)}: "
                f"global_index={global_index}, port={port_name}, unit={unit_name}"
            )

            common_manifest = {
                "global_index": global_index,
                "unit_name": unit_name,
                "port_name": port_name,
                "source_name": source_name,
                "ring_index": int(active_row.get("ring_index", 0)),
                "ring_local_index": int(active_row.get("ring_local_index", 0)),
                "group_id": str(active_row.get("group_id", "")),
                "x_mm": float(active_row["x_mm"]),
                "y_mm": float(active_row["y_mm"]),
                "rotation_deg": float(active_row["rotation_deg"]),
                "excitation_magnitude_W": ACTIVE_PORT_POWER_W,
                "excitation_phase_deg": SOURCE_PHASE_DEG,
            }

            try:
                if vecaep_path.exists() and not OVERWRITE_EXISTING_EXPORT_FILE:
                    raise FileExistsError(
                        f"Vec-AEP file already exists and overwrite is disabled: {vecaep_path}"
                    )
                if vecaep_path.exists():
                    vecaep_path.unlink()

                source_args, source_rows = build_single_port_source_arguments(
                    plan_rows=plan_rows,
                    active_global_index=global_index,
                )
                write_source_state_csv(source_state_path, source_rows)
                active_rows = [row for row in source_rows if bool(row["is_active"])]
                if len(active_rows) != 1:
                    raise RuntimeError(
                        f"Single-port source-state validation failed: active_rows={active_rows}"
                    )
                active_state = active_rows[0]
                log(f"Source-state CSV        : {source_state_path}")
                log(
                    "Active source          : "
                    f"{active_state['source_name']}, "
                    f"magnitude={active_state['magnitude_W']:.12g} W, "
                    f"phase={active_state['phase_deg']:.12g} deg"
                )
                log("Inactive sources       : 186 sources explicitly set to 0 W, 0 deg")

                log("Apply EditSources      : 187 sources in one transaction")
                o_solutions.EditSources(source_args)
                time.sleep(SOURCE_SYNC_DELAY_S)
                log("[OK] EditSources call returned.")

                log(f"Export Vec-AEP         : {vecaep_path}")
                export_farfield_exact_recorded_call(o_radfield, vecaep_path)
                export_info = validate_export_file(vecaep_path)
                log(f"[OK] Export size       : {export_info['size_bytes']} bytes")
                for preview_index, line in enumerate(export_info["preview_lines"], start=1):
                    log(f"[EXPORT PREVIEW {preview_index}] {line}")

                elapsed_s = time.perf_counter() - aep_started
                completed_at = now_iso()
                metadata = {
                    "ok": True,
                    "script_revision": SCRIPT_REVISION,
                    **common_manifest,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "elapsed_s": elapsed_s,
                    "project_path": str(project_path),
                    "design_name": DESIGN_NAME,
                    "solution_name": SOLUTION_NAME,
                    "farfield_setup": FARFIELD_SETUP,
                    "intrinsic_variation_key": INTRINSIC_VARIATION_KEY,
                    "design_variation_key": DESIGN_VARIATION_KEY,
                    "inactive_source_count": EXPECTED_PORT_COUNT - 1,
                    "inactive_source_magnitude_w": INACTIVE_PORT_POWER_W,
                    "field_components": ["rEtheta(mag ang)", "rEphi(mag ang)"],
                    "coordinate_system": "HFSS exported global spherical basis",
                    "vecaep_export": export_info,
                    "source_state_file": str(source_state_path),
                }
                write_json(metadata_path, metadata)
                append_vecaep_manifest(
                    {
                        **common_manifest,
                        "status": "passed",
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "elapsed_s": f"{elapsed_s:.6f}",
                        "vecaep_file": str(vecaep_path),
                        "vecaep_size_bytes": export_info["size_bytes"],
                        "source_state_file": str(source_state_path),
                        "metadata_file": str(metadata_path),
                        "error": "",
                    }
                )
                successful_indices.append(global_index)
                successful_ports.append(port_name)
                log(
                    f"[VEC-AEP PASS] global_index={global_index}, "
                    f"port={port_name}, elapsed={elapsed_s:.2f}s"
                )

            except Exception:
                elapsed_s = time.perf_counter() - aep_started
                completed_at = now_iso()
                error_text = traceback.format_exc()
                write_json(
                    metadata_path,
                    {
                        "ok": False,
                        "script_revision": SCRIPT_REVISION,
                        **common_manifest,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "elapsed_s": elapsed_s,
                        "error_traceback": error_text,
                    },
                )
                append_vecaep_manifest(
                    {
                        **common_manifest,
                        "status": "failed",
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "elapsed_s": f"{elapsed_s:.6f}",
                        "vecaep_file": str(vecaep_path),
                        "vecaep_size_bytes": "",
                        "source_state_file": str(source_state_path),
                        "metadata_file": str(metadata_path),
                        "error": error_text.replace("\r", " ").replace("\n", " | "),
                    }
                )
                failed_indices.append(global_index)
                failed_ports.append(port_name)
                log(f"[VEC-AEP FAILED] global_index={global_index}, port={port_name}")
                log(error_text)
                raise

        total_elapsed_s = time.perf_counter() - run_started
        summary = {
            "ok": len(successful_indices) == EXPECTED_PORT_COUNT and not failed_indices,
            "script_revision": SCRIPT_REVISION,
            "completed_at": now_iso(),
            "elapsed_s": total_elapsed_s,
            "project_path": str(project_path),
            "design_name": DESIGN_NAME,
            "plan_path": str(plan_path),
            "run_dir": str(RUN_DIR),
            "requested_global_indices": [int(row["global_index"]) for row in plan_rows],
            "requested_ports": [str(row["port_name"]) for row in plan_rows],
            "successful_global_indices": successful_indices,
            "successful_ports": successful_ports,
            "failed_global_indices": failed_indices,
            "failed_ports": failed_ports,
            "expected_port_count": EXPECTED_PORT_COUNT,
            "active_port_power_w": ACTIVE_PORT_POWER_W,
            "inactive_port_power_w": INACTIVE_PORT_POWER_W,
            "source_phase_deg": SOURCE_PHASE_DEG,
            "analysis_run": False,
            "project_saved": False,
            "farfield_setup": FARFIELD_SETUP,
            "solution_name": SOLUTION_NAME,
            "vecaep_manifest": str(AEP_MANIFEST_PATH),
            "plan_snapshot": str(PLAN_SNAPSHOT_PATH),
        }
        write_json(SUMMARY_PATH, summary)
        if not summary["ok"]:
            raise RuntimeError(f"Vec-AEP export summary failed: {summary}")

        header("TOP1 SR CUT_L=3.60 MM VECTOR AEP EXPORT: PASS")
        log(f"Run directory          : {RUN_DIR}")
        log(f"Successful Vec-AEPs    : {len(successful_indices)}/{EXPECTED_PORT_COUNT}")
        log(f"Vec-AEP manifest       : {AEP_MANIFEST_PATH}")
        log(f"Run summary            : {SUMMARY_PATH}")
        log(f"Total elapsed          : {total_elapsed_s:.2f} s")
        log("Analysis               : NOT RUN")
        log("Project save           : NOT REQUESTED")
        log("AEDT source state      : left with p187 active at 1 W, 0 deg")

    except Exception:
        total_elapsed_s = time.perf_counter() - run_started
        error_text = traceback.format_exc()
        write_json(
            SUMMARY_PATH,
            {
                "ok": False,
                "script_revision": SCRIPT_REVISION,
                "failed_at": now_iso(),
                "elapsed_s": total_elapsed_s,
                "project_path": str(project_path),
                "design_name": DESIGN_NAME,
                "plan_path": str(plan_path),
                "run_dir": str(RUN_DIR),
                "successful_global_indices": successful_indices,
                "successful_ports": successful_ports,
                "failed_global_indices": failed_indices,
                "failed_ports": failed_ports,
                "error_traceback": error_text,
            },
        )
        log("[FATAL] Top1 SR Cut_L=3.60 mm Vec-AEP export failed.")
        log(error_text)
        raise

    finally:
        if hfss is not None:
            try:
                hfss.release_desktop(
                    close_projects=False,
                    close_desktop=CLOSE_DESKTOP_ON_EXIT,
                )
                log(
                    "[OK] Released PyAEDT connection; "
                    f"close_desktop={CLOSE_DESKTOP_ON_EXIT}."
                )
            except Exception:
                log("[WARN] Could not release the PyAEDT connection cleanly.")
                log(traceback.format_exc())


if __name__ == "__main__":
    main()
