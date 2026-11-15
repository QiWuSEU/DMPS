from __future__ import annotations

"""
run_passive_preconditioning.py

Executable orchestration script for the current Q0CRA/A0CRA Stage-2 flow.

It mirrors the Stage-1 auto-run pattern:
1. Resolve project files.
2. Create a timestamped output directory.
3. Build or reuse a Stage-2 static point pool.
4. Import passive_preconditioning.passive_screening as a module.
5. Call run_stage2_screen_main(...).
6. Write auto_run_summary.json / auto_run_error.json and a full run_log.txt.

The Stage-2 algorithm module itself is not a direct command-line script.
"""

import argparse
import contextlib
import datetime as _dt
import importlib.util
import io
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

K_RINGS = 8
R0 = 5.0
D_MIN = 0.5
D_EFF = 0.5
STAGE2_TOPJ = 10
FINAL_TOPJ = 3


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# CLI / path helpers
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-run Stage-2 screening: build static pool then call passive_screening.run_stage2_screen_main(...).")
    p.add_argument("--base-dir", type=str, default="", help="Project directory. Defaults to this script directory.")
    p.add_argument("--stage1-out-dir", type=str, default="", help="Stage-1 output directory. Defaults to latest a0cra_stage1_* dir or base dir.")
    p.add_argument("--boundary-json", type=str, default="", help="Stage-2 boundary source JSON. Defaults to the resolved Stage-1 JSON.")
    p.add_argument("--mainlobe-table-json", type=str, default="", help="Deprecated alias of --boundary-json for legacy commands.")
    p.add_argument("--unit-npz", type=str, default="", help="HFSS SR-compatible sparse npz. Defaults to hfss_sr_compatible_sparse.npz in base-dir.")
    p.add_argument("--unit-json", type=str, default="", help="HFSS SR-compatible sparse meta json. Defaults next to unit npz.")
    p.add_argument("--pool-generator", type=str, default="", help="Stage-2 pool generator. Defaults to passive_preconditioning/passive_sampling_pool.py in base-dir.")
    p.add_argument("--stage2-module", type=str, default="", help="Stage-2 module. Defaults to passive_preconditioning/passive_screening.py in base-dir.")

    p.add_argument("--K", type=int, default=0, help="0 means infer K from Stage-1; an explicit value must match Stage-1.")
    p.add_argument("--R0", type=float, default=R0)
    p.add_argument("--d-min", type=float, default=D_MIN)
    p.add_argument("--d-eff", type=float, default=D_EFF)

    p.add_argument("--reuse-existing-static-pool", action="store_true", help="Reuse an existing Stage-2 static pool instead of generating one for this run.")
    p.add_argument("--static-pool-npz", type=str, default="", help="Explicit stage2_pool_static.npz when reusing a pool.")
    p.add_argument("--static-pool-json", type=str, default="", help="Explicit stage2_pool_summary.json when reusing a pool.")

    p.add_argument("--stage2-topJ", type=int, default=STAGE2_TOPJ)
    p.add_argument("--final-topJ", type=int, default=FINAL_TOPJ)
    p.add_argument("--disable-dynamic-final", action="store_true")
    p.add_argument("--no-final-fine", action="store_true", help="Skip fine c0/c1 grid after coarse final delta grid.")
    p.add_argument("--final-seed-only", action="store_true", help="Debug/smoke only: final delta grid only tests c0=c1=0.")
    p.add_argument("--dynamic-cap-total", type=int, default=2000)
    p.add_argument("--S-band", type=float, default=0.01)
    p.add_argument("--AR-band-db", type=float, default=0.1)
    p.add_argument("--active-rings", type=str, default="all", help="all, or 1-based inclusive range like 1:8")
    p.add_argument("--min-m", type=int, default=1)
    p.add_argument("--candidate-mode", type=str, default="two_per_ring_alias_complement", choices=["two_per_ring_alias_complement", "around_quarter_window"], help="Stage-1 m-candidate generation mode.")
    p.add_argument("--stage1-to-stage2", type=int, default=50, help="Maximum Stage-1 survivors sent to Stage-2 PSLL.")
    p.add_argument("--max-designs-stage1", type=int, default=0, help="Debug only. 0 means full enumeration.")
    p.add_argument("--no-batch-backend", action="store_true", help="Use passive_smain_score/passive_ar_psll_score evaluators instead of passive_field_batch.py.")
    p.add_argument("--batch-stage1-chunk", type=int, default=128)
    p.add_argument("--batch-stage2-chunk", type=int, default=8)
    p.add_argument("--batch-final-chunk", type=int, default=8)
    p.add_argument("--batch-verbose", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _find_first(base_dir: Path, patterns: Sequence[str], *, exclude_names: Optional[set[str]] = None) -> Optional[Path]:
    exclude_names = exclude_names or set()
    for pattern in patterns:
        hits = sorted(p for p in base_dir.glob(pattern) if p.name not in exclude_names)
        if hits:
            return hits[0]
    return None


def _load_module(module_path: Path, *, suffix: str = "_auto"):
    spec = importlib.util.spec_from_file_location(module_path.stem + suffix, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_stage1_output(base_dir: Path, explicit_dir: str = "") -> Tuple[Path, Path]:
    if explicit_dir:
        d = Path(explicit_dir)
        if not d.is_absolute():
            d = base_dir / d
        npz = d / "a0cra_stage1_main_out.npz"
        js = d / "a0cra_stage1_main_out.json"
        if not npz.exists() or not js.exists():
            raise FileNotFoundError(f"Explicit stage1 dir missing a0cra_stage1_main_out.npz/json: {d}")
        return npz.resolve(), js.resolve()

    # base-dir direct files first
    direct_npz = base_dir / "a0cra_stage1_main_out.npz"
    direct_json = base_dir / "a0cra_stage1_main_out.json"
    if direct_npz.exists() and direct_json.exists():
        return direct_npz.resolve(), direct_json.resolve()

    patterns = ["a0cra_stage1_K*", "a0cra_stage1_run_out_*", "a0cra_stage1_*"]
    dirs: list[Path] = []
    for pat in patterns:
        dirs.extend([p for p in base_dir.glob(pat) if p.is_dir()])
    dirs = sorted(set(dirs), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        npz = d / "a0cra_stage1_main_out.npz"
        js = d / "a0cra_stage1_main_out.json"
        if npz.exists() and js.exists():
            return npz.resolve(), js.resolve()

    # Shallow recursive fallback.
    hits = sorted(base_dir.glob("**/a0cra_stage1_main_out.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    for npz in hits:
        js = npz.with_suffix(".json")
        if js.exists():
            return npz.resolve(), js.resolve()
    raise FileNotFoundError("Cannot find a0cra_stage1_main_out.npz/json in base-dir or stage1 run dirs.")


def _resolve_unique_path(candidates: Sequence[Path], *, kind: str) -> Path:
    unique = sorted({p.resolve() for p in candidates if p.is_file()})
    if not unique:
        raise FileNotFoundError(f"Cannot find {kind}")
    if len(unique) > 1:
        joined = "\n  ".join(str(p) for p in unique)
        raise RuntimeError(f"Multiple {kind} candidates found; pass an explicit path:\n  {joined}")
    return unique[0]


def _resolve_unit_files(base_dir: Path, unit_npz_arg: str = "", unit_json_arg: str = "") -> Tuple[Path, Optional[Path]]:
    if unit_npz_arg:
        unit_npz = Path(unit_npz_arg)
        if not unit_npz.is_absolute():
            unit_npz = base_dir / unit_npz
        if not unit_npz.is_file():
            raise FileNotFoundError(f"Cannot find unit npz: {unit_npz}")
        unit_npz = unit_npz.resolve()
    else:
        patterns = (
            "iep_output/**/hfss_sr_sparse.npz",
            "iep_output/**/*sr*sparse*.npz",
            "**/hfss_sr_compatible_sparse.npz",
            "**/hfss_sr_sparse.npz",
            "**/*hfss*sr*sparse*.npz",
        )
        candidates: list[Path] = []
        for pattern in patterns:
            candidates.extend(
                p for p in base_dir.glob(pattern)
                if p.is_file() and "diagnostic" not in p.name.lower()
            )
        unit_npz = _resolve_unique_path(candidates, kind="HFSS SR sparse NPZ")

    if unit_json_arg:
        unit_json = Path(unit_json_arg)
        if not unit_json.is_absolute():
            unit_json = base_dir / unit_json
        if not unit_json.is_file():
            raise FileNotFoundError(f"Cannot find unit json: {unit_json}")
        return unit_npz, unit_json.resolve()

    candidates = [
        unit_npz.with_suffix(".json"),
        unit_npz.parent / f"{unit_npz.stem}_meta.json",
    ]
    unit_json = next((c.resolve() for c in candidates if c.is_file()), None)
    return unit_npz, unit_json

def _resolve_boundary_json(
    base_dir: Path,
    *,
    stage1_json: Path,
    boundary_arg: str = "",
    legacy_arg: str = "",
) -> Path:
    if boundary_arg and legacy_arg:
        p1 = Path(boundary_arg)
        p2 = Path(legacy_arg)
        if not p1.is_absolute():
            p1 = base_dir / p1
        if not p2.is_absolute():
            p2 = base_dir / p2
        if p1.resolve() != p2.resolve():
            raise ValueError("--boundary-json and --mainlobe-table-json refer to different files")
    arg = boundary_arg or legacy_arg
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            p = base_dir / p
    else:
        p = stage1_json
    if not p.is_file():
        raise FileNotFoundError(f"Cannot find boundary JSON: {p}")
    return p.resolve()

def _resolve_pool_generator(base_dir: Path, arg: str = "") -> Path:
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            p = base_dir / p
    else:
        p = base_dir / "passive_preconditioning" / "passive_sampling_pool.py"
    if not p.exists():
        raise FileNotFoundError(f"Cannot find Stage-2 pool generator: {p}")
    return p.resolve()


def _resolve_stage2_module(base_dir: Path, arg: str = "") -> Path:
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            p = base_dir / p
    else:
        p = base_dir / "passive_preconditioning" / "passive_screening.py"
    if not p.exists():
        raise FileNotFoundError(f"Cannot find stage2 module: {p}")
    return p.resolve()


def _resolve_existing_pool(base_dir: Path, npz_arg: str = "", json_arg: str = "") -> Tuple[Path, Path]:
    if npz_arg:
        npz = Path(npz_arg)
        if not npz.is_absolute():
            npz = base_dir / npz
    else:
        npz = _find_first(base_dir, ["stage2_pool_static.npz", "**/stage2_pool_static.npz"])
        if npz is None:
            raise FileNotFoundError("Cannot find existing stage2_pool_static.npz")
    if json_arg:
        js = Path(json_arg)
        if not js.is_absolute():
            js = base_dir / js
    else:
        candidates = [npz.with_name("stage2_pool_summary.json"), npz.with_suffix(".json")]
        js = next((c for c in candidates if c.exists()), None)
        if js is None:
            raise FileNotFoundError("Cannot find stage2_pool_summary.json next to static pool")
    if not npz.exists() or not js.exists():
        raise FileNotFoundError(f"Static pool paths invalid: {npz}, {js}")
    return npz.resolve(), js.resolve()


def _build_output_dir(base_dir: Path, *, K: int) -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base_dir / f"stage2_K{int(K)}_allrings_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_stage1_K(stage1_npz: Path) -> int:
    with np.load(stage1_npz, allow_pickle=False) as d:
        for key in ("r_best", "r_final", "r", "N_best", "N_final", "N"):
            if key in d.files:
                n = int(np.asarray(d[key]).reshape(-1).size)
                if n <= 0:
                    raise ValueError(f"Stage-1 key {key} is empty in {stage1_npz}")
                return n
    raise KeyError(f"Cannot infer K from Stage-1 NPZ keys: {stage1_npz}")

# -----------------------------------------------------------------------------
# Mainlobe-table boundary extraction and pool generation
# -----------------------------------------------------------------------------

def _generate_stage2_static_pool_for_run(
    *,
    pool_generator_path: Path,
    output_dir: Path,
    boundary_json: Path,
    K: int,
    d_min: float,
    d_eff: float,
    R0: float,
    stage1_json: Path,
) -> Tuple[Path, Path, Dict[str, str]]:
    pool_mod = _load_module(pool_generator_path, suffix="_pool_gen")
    required = (
        "Stage2PoolConfig",
        "make_default_angle_sets",
        "load_stage2_boundary_tables_from_json",
        "build_stage2_pool_package",
        "save_stage2_pool_package",
    )
    for name in required:
        if not hasattr(pool_mod, name):
            raise AttributeError(f"{pool_generator_path.name} missing required symbol {name}")
    angle_sets = pool_mod.make_default_angle_sets()
    boundaries = pool_mod.load_stage2_boundary_tables_from_json(
        str(boundary_json),
        angle_sets,
        boundary_format="auto",
        rho_from_alpha="sin_alpha",
        K=int(K),
        d_min=float(d_min),
        d_eff=float(d_eff),
        R0=float(R0),
    )
    cfg = pool_mod.Stage2PoolConfig(phi_s_deg=0.0)
    package = pool_mod.build_stage2_pool_package(angle_sets=angle_sets, boundaries=boundaries, cfg=cfg)
    pool_out_dir = output_dir / "static_pool"
    saved = pool_mod.save_stage2_pool_package(package, str(pool_out_dir))
    pool_npz = Path(saved["npz"])
    pool_json = Path(saved["summary_json"])
    if not pool_npz.exists() or not pool_json.exists():
        raise FileNotFoundError(f"Stage-2 pool generation failed: {pool_npz}, {pool_json}")

    summary = _read_json(pool_json)
    summary["boundary_source"] = {
        "mode": "external_json",
        "path": str(boundary_json),
        "resolved_format": "auto",
        "rho_from_alpha": "sin_alpha",
        "stage1_json_default": bool(boundary_json.resolve() == stage1_json.resolve()),
    }
    pool_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return pool_npz.resolve(), pool_json.resolve(), {k: str(v) for k, v in saved.items()}


def _check_required_project_files(base_dir: Path, *, use_batch_backend: bool) -> None:
    required = [
        "element_spectrum/iep_model.py",
        "passive_preconditioning/passive_sampling_pool.py",
        "passive_preconditioning/passive_smain_score.py",
        "passive_preconditioning/passive_ar_psll_score.py",
        "passive_preconditioning/passive_screening.py",
    ]
    if use_batch_backend:
        required.append("passive_preconditioning/passive_field_batch.py")
    missing = [name for name in required if not (base_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required Stage-2 files in {base_dir}: {missing}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parent
    sys.path.insert(0, str(base_dir))

    # Stage-1 is resolved before output directory so K can be inferred.
    stage1_npz, stage1_json = _resolve_stage1_output(base_dir, args.stage1_out_dir)
    stage1_K = _load_stage1_K(stage1_npz)
    if int(args.K) <= 0:
        K = int(stage1_K)
    else:
        K = int(args.K)
        if K != int(stage1_K):
            raise ValueError(f"--K={K} does not match Stage-1 K={stage1_K}")
    output_dir = _build_output_dir(base_dir, K=K)
    log_path = output_dir / "run_log.txt"

    with log_path.open("w", encoding="utf-8") as log_f:
        tee = _Tee(sys.stdout, log_f)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print("=== STAGE2 AUTO RUN: START ===")
            print(f"[base_dir] {base_dir}")
            print(f"[output_dir] {output_dir}")
            print(f"[stage1_npz] {stage1_npz}")
            print(f"[stage1_json] {stage1_json}")
            print(f"[stage1_K] {stage1_K}")
            print(f"[resolved_K] {K}")
            try:
                _check_required_project_files(base_dir, use_batch_backend=not bool(args.no_batch_backend))
                unit_npz, unit_json = _resolve_unit_files(base_dir, args.unit_npz, args.unit_json)
                boundary_json = _resolve_boundary_json(
                    base_dir,
                    stage1_json=stage1_json,
                    boundary_arg=str(args.boundary_json),
                    legacy_arg=str(args.mainlobe_table_json),
                )
                pool_generator = _resolve_pool_generator(base_dir, args.pool_generator)
                stage2_module_path = _resolve_stage2_module(base_dir, args.stage2_module)

                if args.reuse_existing_static_pool:
                    pool_npz, pool_json = _resolve_existing_pool(base_dir, args.static_pool_npz, args.static_pool_json)
                    pool_saved = {"npz": str(pool_npz), "summary_json": str(pool_json)}
                    pool_generation_mode = "reuse_existing"
                else:
                    pool_npz, pool_json, pool_saved = _generate_stage2_static_pool_for_run(
                        pool_generator_path=pool_generator,
                        output_dir=output_dir,
                        boundary_json=boundary_json,
                        K=K,
                        d_min=float(args.d_min),
                        d_eff=float(args.d_eff),
                        R0=float(args.R0),
                        stage1_json=stage1_json,
                    )
                    pool_generation_mode = "generated_for_run"

                print(f"[unit_npz] {unit_npz}")
                print(f"[unit_json] {unit_json}")
                print(f"[boundary_json] {boundary_json}")
                print(f"[boundary_source_mode] {'stage1_default' if boundary_json == stage1_json else 'explicit'}")
                print(f"[pool_generator] {pool_generator}")
                print(f"[pool_generation_mode] {pool_generation_mode}")
                print(f"[pool_npz] {pool_npz}")
                print(f"[pool_json] {pool_json}")
                print(f"[stage2_module] {stage2_module_path}")

                mod = _load_module(stage2_module_path, suffix="_stage2_mod")
                if not hasattr(mod, "run_stage2_screen_main") or not hasattr(mod, "Stage2ScreenConfig"):
                    raise AttributeError(f"{stage2_module_path.name} must define Stage2ScreenConfig and run_stage2_screen_main(...)")

                final_grid = (0.0,) if bool(args.final_seed_only) else None
                cfg_kwargs = {}
                if final_grid is not None:
                    cfg_kwargs["final_c_grid"] = final_grid
                cfg = mod.Stage2ScreenConfig(
                    stage2_topJ=int(args.stage2_topJ),
                    final_topJ=int(args.final_topJ),
                    active_rings=str(args.active_rings),
                    min_m=int(args.min_m),
                    candidate_mode=str(args.candidate_mode),
                    stage1_to_stage2=int(args.stage1_to_stage2),
                    max_designs_stage1=None if int(args.max_designs_stage1) <= 0 else int(args.max_designs_stage1),
                    S_band=float(args.S_band),
                    AR_band_db=float(args.AR_band_db),
                    enable_dynamic_final=not bool(args.disable_dynamic_final),
                    dynamic_cap_total=int(args.dynamic_cap_total),
                    verbose=not bool(args.quiet),
                    final_use_fine=(not bool(args.no_final_fine)) and (not bool(args.final_seed_only)),
                    use_batch_backend=not bool(args.no_batch_backend),
                    batch_stage1_chunk=int(args.batch_stage1_chunk),
                    batch_stage2_chunk=int(args.batch_stage2_chunk),
                    batch_final_chunk=int(args.batch_final_chunk),
                    batch_verbose=bool(args.batch_verbose),
                    **cfg_kwargs,
                )
                cfg_payload = {
                    "K": K,
                    "stage1_K": int(stage1_K),
                    "R0": float(args.R0),
                    "d_min": float(args.d_min),
                    "d_eff": float(args.d_eff),
                    "stage2_topJ": int(args.stage2_topJ),
                    "final_topJ": int(args.final_topJ),
                    "active_rings": str(args.active_rings),
                    "min_m": int(args.min_m),
                    "candidate_mode": str(args.candidate_mode),
                    "stage1_to_stage2": int(args.stage1_to_stage2),
                    "max_designs_stage1": None if int(args.max_designs_stage1) <= 0 else int(args.max_designs_stage1),
                    "enable_dynamic_final": not bool(args.disable_dynamic_final),
                    "final_use_fine": (not bool(args.no_final_fine)) and (not bool(args.final_seed_only)),
                    "final_seed_only": bool(args.final_seed_only),
                    "S_band": float(args.S_band),
                    "AR_band_db": float(args.AR_band_db),
                    "use_batch_backend": not bool(args.no_batch_backend),
                    "batch_stage1_chunk": int(args.batch_stage1_chunk),
                    "batch_stage2_chunk": int(args.batch_stage2_chunk),
                    "batch_final_chunk": int(args.batch_final_chunk),
                    "batch_verbose": bool(args.batch_verbose),
                }
                print(f"[backend] use_batch_backend={not bool(args.no_batch_backend)}")
                print(f"[backend] stage1_chunk={int(args.batch_stage1_chunk)} stage2_chunk={int(args.batch_stage2_chunk)} final_chunk={int(args.batch_final_chunk)} batch_verbose={bool(args.batch_verbose)}")
                print("[cfg]")
                print(json.dumps(cfg_payload, ensure_ascii=False, indent=2))

                stage2_out_dir = output_dir / "stage2_outputs"
                result = mod.run_stage2_screen_main(
                    stage1_npz_path=str(stage1_npz),
                    stage1_json_path=str(stage1_json),
                    unit_npz_path=str(unit_npz),
                    unit_json_path=None if unit_json is None else str(unit_json),
                    mainlobe_table_json_path=str(boundary_json),
                    static_pool_npz_path=str(pool_npz),
                    static_pool_json_path=str(pool_json),
                    output_dir=str(stage2_out_dir),
                    config=cfg,
                )

                summary = {
                    "ok": True,
                    "case": cfg_payload,
                    "paths": {
                        "stage2_module": str(stage2_module_path),
                        "pool_generator": str(pool_generator),
                        "stage1_npz": str(stage1_npz),
                        "stage1_json": str(stage1_json),
                        "unit_npz": str(unit_npz),
                        "unit_json": None if unit_json is None else str(unit_json),
                        "boundary_json": str(boundary_json),
                        "mainlobe_table_json": str(boundary_json),
                        "static_pool_npz": str(pool_npz),
                        "static_pool_json": str(pool_json),
                        "stage2_output_dir": str(stage2_out_dir),
                        "log_path": str(log_path),
                    },
                    "pool_generation_mode": pool_generation_mode,
                    "pool_saved": pool_saved,
                    "stage2_result": {
                        "stage1_pair_count": int(result.stage1_pair_count),
                        "stage1_survivor_count": int(result.stage1_survivor_count),
                        "stage2_scored_count": int(result.stage2_scored_count),
                        "stage2_topJ_count": int(result.stage2_topJ_count),
                        "final_grid_count": int(result.final_grid_count),
                        "final_topJ_count": int(result.final_topJ_count),
                        "best_final_psll_db": result.best_final_psll_db,
                        "best_final_ar_main_db": result.best_final_ar_main_db,
                        "best_final_smain": result.best_final_smain,
                        "seeds_json_path": result.seeds_json_path,
                        "seeds_npz_path": result.seeds_npz_path,
                        "report_json_path": result.report_json_path,
                        "backend_A_summary_json_path": getattr(result, "backend_A_summary_json_path", ""),
                        "backend_C_summary_json_path": getattr(result, "backend_C_summary_json_path", ""),
                        "backend_Final_summary_json_path": getattr(result, "backend_Final_summary_json_path", ""),
                    },
                }
                (output_dir / "auto_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                print("[summary]")
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                print("=== STAGE2 AUTO RUN: DONE ===")
                print(f"[saved] log={log_path}")
                return 0
            except Exception as exc:
                print("=== STAGE2 AUTO RUN: FAILED ===")
                print(f"[error] {type(exc).__name__}: {exc}")
                traceback.print_exc()
                fail_payload = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "stage1_npz": str(stage1_npz),
                    "stage1_json": str(stage1_json),
                    "log_path": str(log_path),
                }
                (output_dir / "auto_run_error.json").write_text(json.dumps(fail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
