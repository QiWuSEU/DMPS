# -*- coding: utf-8 -*-
"""
export_passive_farfield.py

One-click PyAEDT post-processing exporter for the solved 187-port Top1
sequential-rotation circular-ring array.

First-version scope
-------------------
- Open the already solved HFSS project; do not analyze or remesh.
- Read the exact element/port mapping written by the large-array builder.
- Form thirteen scan states: theta_s = 0, 5, ..., 60 deg; phi_s = 0 deg.
- Apply all 187 post-processing source magnitudes/phases in one EditSources call.
- Export each synthesized far field by replaying the recorded
  RadField.ExportFieldsToFile argument structure.
- Write detailed logs, source-weight CSV files, per-scan metadata, and a run
  manifest in the configured farfield_exports directory.

Important phase convention
--------------------------
For the current LHCP Top1 SR array, the default excitation phase is

    phase_n = +rotation_n - k0 * r_n dot u_scan

in degrees. Therefore:
- SR_PHASE_SIGN   = +1
- SCAN_PHASE_SIGN = -1

This production run exports the complete 0-to-60-deg scan set in 5-deg steps
toward +x (phi=0 deg). The validated phase signs are fixed for reproduction.
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
SCRIPT_REVISION = "top1_sr_cut360_legacyenv_farfield_export_theta000_060_step005_v1"

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
    "DMPS_HFSS_FARFIELD_OUTPUT_DIR", str(SERVER_ROOT / "farfield_exports")
))
EXPORT_CASE_NAME = "top1_sr_cut360_legacyenv_phi000_theta000_060_step005"

SCAN_THETA_DEG: Tuple[float, ...] = tuple(float(value) for value in range(0, 61, 5))
SCAN_PHI_DEG = 0.0
FREQ_GHZ = 2.4
C0_MM_PER_S = 299_792_458_000.0
LAMBDA0_MM = C0_MM_PER_S / (FREQ_GHZ * 1.0e9)
EXPECTED_PORT_COUNT = 187

# Post-processing source convention.
SR_PHASE_SIGN = +1.0
SCAN_PHASE_SIGN = -1.0
TOTAL_INCIDENT_POWER_W = 1.0
POWER_PER_PORT_W = TOTAL_INCIDENT_POWER_W / EXPECTED_PORT_COUNT

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
LOG_PATH = RUN_DIR / "export_passive_farfield.log"
RUN_CONFIG_PATH = RUN_DIR / "run_config.json"
SCAN_MANIFEST_PATH = RUN_DIR / "scan_manifest.csv"
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
# Source phase calculation
# =============================================================================


def wrap_deg_180(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    # Avoid writing -180 when +180 is more readable.
    if abs(wrapped + 180.0) < 1.0e-12:
        return 180.0
    return wrapped


def compute_source_phase_deg(
    row: Dict[str, Any],
    scan_theta_deg: float,
    scan_phi_deg: float,
) -> Dict[str, float]:
    theta = math.radians(float(scan_theta_deg))
    phi = math.radians(float(scan_phi_deg))
    x_mm = float(row["x_mm"])
    y_mm = float(row["y_mm"])
    rotation_deg = float(row["rotation_deg"])

    projection_mm = (
        x_mm * math.sin(theta) * math.cos(phi)
        + y_mm * math.sin(theta) * math.sin(phi)
    )
    sr_phase_deg = SR_PHASE_SIGN * rotation_deg
    scan_phase_deg = SCAN_PHASE_SIGN * 360.0 * projection_mm / LAMBDA0_MM
    total_unwrapped_deg = sr_phase_deg + scan_phase_deg
    total_wrapped_deg = wrap_deg_180(total_unwrapped_deg)

    return {
        "projection_mm": projection_mm,
        "sr_phase_deg": sr_phase_deg,
        "scan_phase_deg": scan_phase_deg,
        "total_unwrapped_deg": total_unwrapped_deg,
        "total_wrapped_deg": total_wrapped_deg,
    }


def source_name_from_port(port_name: str) -> str:
    text = str(port_name).strip()
    return text if ":" in text else f"{text}:1"


def build_source_arguments(
    plan_rows: Sequence[Dict[str, Any]],
    scan_theta_deg: float,
    scan_phi_deg: float,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    args: List[Any] = [
        [
            "IncludePortPostProcessing:=",
            True,
            "SpecifySystemPower:=",
            False,
        ]
    ]
    weight_rows: List[Dict[str, Any]] = []

    for row in plan_rows:
        phase = compute_source_phase_deg(row, scan_theta_deg, scan_phi_deg)
        source_name = source_name_from_port(str(row["port_name"]))
        args.append(
            [
                "Name:=",
                source_name,
                "Magnitude:=",
                f"{POWER_PER_PORT_W:.16g}W",
                "Phase:=",
                f"{phase['total_wrapped_deg']:.12g}deg",
            ]
        )
        weight_rows.append(
            {
                "global_index": int(row["global_index"]),
                "unit_name": str(row["unit_name"]),
                "ring_index": int(row.get("ring_index", 0)),
                "ring_local_index": int(row.get("ring_local_index", 0)),
                "group_id": str(row.get("group_id", "")),
                "port_name": str(row["port_name"]),
                "source_name": source_name,
                "x_mm": float(row["x_mm"]),
                "y_mm": float(row["y_mm"]),
                "rotation_deg": float(row["rotation_deg"]),
                "projection_mm": phase["projection_mm"],
                "sr_phase_deg": phase["sr_phase_deg"],
                "scan_phase_deg": phase["scan_phase_deg"],
                "total_phase_unwrapped_deg": phase["total_unwrapped_deg"],
                "total_phase_wrapped_deg": phase["total_wrapped_deg"],
                "magnitude_W": POWER_PER_PORT_W,
            }
        )

    if len(args) != EXPECTED_PORT_COUNT + 1:
        raise RuntimeError(f"EditSources argument count is invalid: {len(args)}")
    return args, weight_rows


def write_weight_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("Cannot write an empty source-weight CSV.")
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


def scan_label(theta_deg: float, phi_deg: float) -> str:
    theta_i = int(round(theta_deg))
    phi_i = int(round(phi_deg))
    phi_sign = "m" if phi_i < 0 else "p"
    return f"theta_{theta_i:03d}_phi_{phi_sign}{abs(phi_i):03d}"


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
    return {"path": str(path), "size_bytes": size, "preview_lines": preview}


def append_scan_manifest(row: Dict[str, Any]) -> None:
    fieldnames = [
        "scan_theta_deg",
        "scan_phi_deg",
        "status",
        "started_at",
        "completed_at",
        "elapsed_s",
        "farfield_file",
        "farfield_size_bytes",
        "weights_file",
        "metadata_file",
        "error",
    ]
    exists = SCAN_MANIFEST_PATH.is_file()
    with SCAN_MANIFEST_PATH.open("a", encoding="utf-8-sig", newline="") as stream:
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
    if SCAN_MANIFEST_PATH.exists():
        SCAN_MANIFEST_PATH.unlink()

    header("TOP1 SR ARRAY FAR-FIELD EXPORT: THETA 0-60 DEG STEP 5, PHI 0 DEG")
    log(f"Timestamp              : {now_iso()}")
    log(f"Python executable      : {sys.executable}")
    log(f"Working directory      : {os.getcwd()}")
    log(f"Script revision        : {SCRIPT_REVISION}")
    log(f"Server root            : {SERVER_ROOT}")
    log(f"Run directory          : {RUN_DIR}")
    log(f"Scan theta values      : {list(SCAN_THETA_DEG)} deg")
    log(f"Scan phi               : {SCAN_PHI_DEG} deg")
    log(f"Frequency              : {FREQ_GHZ} GHz")
    log(f"lambda0                : {LAMBDA0_MM:.12f} mm")
    log(f"SR phase sign          : {SR_PHASE_SIGN:+.0f}")
    log(f"Scan phase sign        : {SCAN_PHASE_SIGN:+.0f}")
    log(f"Total incident power   : {TOTAL_INCIDENT_POWER_W:.12g} W")
    log(f"Per-port power         : {POWER_PER_PORT_W:.16g} W")
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
            "AEDT will still be opened, but field export requires solved field data."
        )

    config_payload = {
        "script_revision": SCRIPT_REVISION,
        "created_at": now_iso(),
        "project_path": str(project_path),
        "design_name": DESIGN_NAME,
        "plan_path": str(plan_path),
        "run_dir": str(RUN_DIR),
        "scan_theta_deg": list(SCAN_THETA_DEG),
        "scan_phi_deg": SCAN_PHI_DEG,
        "frequency_ghz": FREQ_GHZ,
        "lambda0_mm": LAMBDA0_MM,
        "expected_port_count": EXPECTED_PORT_COUNT,
        "sr_phase_sign": SR_PHASE_SIGN,
        "scan_phase_sign": SCAN_PHASE_SIGN,
        "total_incident_power_w": TOTAL_INCIDENT_POWER_W,
        "power_per_port_w": POWER_PER_PORT_W,
        "farfield_setup": FARFIELD_SETUP,
        "solution_name": SOLUTION_NAME,
        "intrinsic_variation_key": INTRINSIC_VARIATION_KEY,
        "design_variation_key": DESIGN_VARIATION_KEY,
        "export_method": "RadField.ExportFieldsToFile recorded raw call",
        "analysis_run": False,
    }
    write_json(RUN_CONFIG_PATH, config_payload)

    hfss = None
    successful_scans: List[float] = []
    failed_scans: List[float] = []
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

        header("Apply scan source weights and export far fields")
        for scan_number, theta_deg in enumerate(SCAN_THETA_DEG, start=1):
            label = scan_label(theta_deg, SCAN_PHI_DEG)
            scan_dir = RUN_DIR / label
            scan_dir.mkdir(parents=True, exist_ok=True)
            farfield_path = scan_dir / f"farfield_{label}.csv"
            weights_path = scan_dir / f"source_weights_{label}.csv"
            metadata_path = scan_dir / "metadata.json"

            started_at = now_iso()
            scan_started = time.perf_counter()
            header(
                f"SCAN {scan_number}/{len(SCAN_THETA_DEG)}: "
                f"theta_s={theta_deg:.1f} deg, phi_s={SCAN_PHI_DEG:.1f} deg"
            )

            try:
                if farfield_path.exists() and not OVERWRITE_EXISTING_EXPORT_FILE:
                    raise FileExistsError(
                        f"Export file already exists and overwrite is disabled: {farfield_path}"
                    )
                if farfield_path.exists():
                    farfield_path.unlink()

                source_args, weight_rows = build_source_arguments(
                    plan_rows=plan_rows,
                    scan_theta_deg=theta_deg,
                    scan_phi_deg=SCAN_PHI_DEG,
                )
                write_weight_csv(weights_path, weight_rows)

                phases = [float(row["total_phase_wrapped_deg"]) for row in weight_rows]
                scan_phases = [float(row["scan_phase_deg"]) for row in weight_rows]
                log(f"Weights CSV            : {weights_path}")
                log(
                    "Phase range wrapped    : "
                    f"{min(phases):.6f} ... {max(phases):.6f} deg"
                )
                log(
                    "Scan phase range       : "
                    f"{min(scan_phases):.6f} ... {max(scan_phases):.6f} deg"
                )
                log(
                    "Center source          : "
                    f"{weight_rows[0]['source_name']}, "
                    f"phase={weight_rows[0]['total_phase_wrapped_deg']:.6f} deg"
                )
                log(
                    "First ring source      : "
                    f"{weight_rows[1]['source_name']}, "
                    f"rotation={weight_rows[1]['rotation_deg']:.6f} deg, "
                    f"scan={weight_rows[1]['scan_phase_deg']:.6f} deg, "
                    f"total={weight_rows[1]['total_phase_wrapped_deg']:.6f} deg"
                )

                log("Apply EditSources      : 187 sources in one transaction")
                o_solutions.EditSources(source_args)
                time.sleep(SOURCE_SYNC_DELAY_S)
                log("[OK] EditSources call returned.")

                log(f"Export far field       : {farfield_path}")
                export_farfield_exact_recorded_call(o_radfield, farfield_path)
                export_info = validate_export_file(farfield_path)
                log(f"[OK] Export size       : {export_info['size_bytes']} bytes")
                for preview_index, line in enumerate(export_info["preview_lines"], start=1):
                    log(f"[EXPORT PREVIEW {preview_index}] {line}")

                elapsed_s = time.perf_counter() - scan_started
                completed_at = now_iso()
                metadata = {
                    "ok": True,
                    "script_revision": SCRIPT_REVISION,
                    "scan_theta_deg": theta_deg,
                    "scan_phi_deg": SCAN_PHI_DEG,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "elapsed_s": elapsed_s,
                    "project_path": str(project_path),
                    "design_name": DESIGN_NAME,
                    "solution_name": SOLUTION_NAME,
                    "farfield_setup": FARFIELD_SETUP,
                    "intrinsic_variation_key": INTRINSIC_VARIATION_KEY,
                    "design_variation_key": DESIGN_VARIATION_KEY,
                    "farfield_export": export_info,
                    "weights_file": str(weights_path),
                    "source_count": len(weight_rows),
                    "magnitude_per_port_w": POWER_PER_PORT_W,
                    "total_incident_power_w": TOTAL_INCIDENT_POWER_W,
                    "sr_phase_sign": SR_PHASE_SIGN,
                    "scan_phase_sign": SCAN_PHASE_SIGN,
                    "phase_wrapped_min_deg": min(phases),
                    "phase_wrapped_max_deg": max(phases),
                }
                write_json(metadata_path, metadata)
                append_scan_manifest(
                    {
                        "scan_theta_deg": theta_deg,
                        "scan_phi_deg": SCAN_PHI_DEG,
                        "status": "passed",
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "elapsed_s": f"{elapsed_s:.6f}",
                        "farfield_file": str(farfield_path),
                        "farfield_size_bytes": export_info["size_bytes"],
                        "weights_file": str(weights_path),
                        "metadata_file": str(metadata_path),
                        "error": "",
                    }
                )
                successful_scans.append(theta_deg)
                log(f"[SCAN PASS] theta={theta_deg:.1f} deg, elapsed={elapsed_s:.2f}s")

            except Exception:
                elapsed_s = time.perf_counter() - scan_started
                completed_at = now_iso()
                error_text = traceback.format_exc()
                write_json(
                    metadata_path,
                    {
                        "ok": False,
                        "script_revision": SCRIPT_REVISION,
                        "scan_theta_deg": theta_deg,
                        "scan_phi_deg": SCAN_PHI_DEG,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "elapsed_s": elapsed_s,
                        "error_traceback": error_text,
                    },
                )
                append_scan_manifest(
                    {
                        "scan_theta_deg": theta_deg,
                        "scan_phi_deg": SCAN_PHI_DEG,
                        "status": "failed",
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "elapsed_s": f"{elapsed_s:.6f}",
                        "farfield_file": str(farfield_path),
                        "farfield_size_bytes": "",
                        "weights_file": str(weights_path),
                        "metadata_file": str(metadata_path),
                        "error": error_text.replace("\r", " ").replace("\n", " | "),
                    }
                )
                failed_scans.append(theta_deg)
                log(f"[SCAN FAILED] theta={theta_deg:.1f} deg")
                log(error_text)
                raise

        total_elapsed_s = time.perf_counter() - run_started
        summary = {
            "ok": len(successful_scans) == len(SCAN_THETA_DEG) and not failed_scans,
            "script_revision": SCRIPT_REVISION,
            "completed_at": now_iso(),
            "elapsed_s": total_elapsed_s,
            "project_path": str(project_path),
            "design_name": DESIGN_NAME,
            "plan_path": str(plan_path),
            "run_dir": str(RUN_DIR),
            "requested_scan_theta_deg": list(SCAN_THETA_DEG),
            "scan_phi_deg": SCAN_PHI_DEG,
            "successful_scan_theta_deg": successful_scans,
            "failed_scan_theta_deg": failed_scans,
            "source_count": EXPECTED_PORT_COUNT,
            "analysis_run": False,
            "project_saved": False,
            "farfield_setup": FARFIELD_SETUP,
            "solution_name": SOLUTION_NAME,
        }
        write_json(SUMMARY_PATH, summary)
        if not summary["ok"]:
            raise RuntimeError(f"Far-field export summary failed: {summary}")

        header("TOP1 SR ARRAY FAR-FIELD EXPORT: PASS")
        log(f"Run directory          : {RUN_DIR}")
        log(f"Successful scans       : {successful_scans}")
        log(f"Scan manifest          : {SCAN_MANIFEST_PATH}")
        log(f"Run summary            : {SUMMARY_PATH}")
        log(f"Total elapsed          : {total_elapsed_s:.2f} s")
        log("Analysis               : NOT RUN")
        log("Project save           : NOT REQUESTED")
        log(f"AEDT source state      : left at the final {SCAN_THETA_DEG[-1]:g}-degree scan for inspection")

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
                "requested_scan_theta_deg": list(SCAN_THETA_DEG),
                "successful_scan_theta_deg": successful_scans,
                "failed_scan_theta_deg": failed_scans,
                "error_traceback": error_text,
            },
        )
        log("[FATAL] Top1 SR far-field export failed.")
        log(error_text)
        raise

    finally:
        if hfss is not None:
            try:
                hfss.release_desktop(close_projects=False, close_desktop=CLOSE_DESKTOP_ON_EXIT)
                log(
                    "[OK] Released PyAEDT connection; "
                    f"close_desktop={CLOSE_DESKTOP_ON_EXIT}."
                )
            except Exception:
                log("[WARN] Could not release the PyAEDT connection cleanly.")
                log(traceback.format_exc())


if __name__ == "__main__":
    main()
