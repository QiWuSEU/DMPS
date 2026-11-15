# DMPS Stage-1 ring-geometry optimizer.
# The established Stage1/A0CRA interfaces and output schemas remain unchanged.
import math
import json
from pathlib import Path
from typing import Callable, Optional, Sequence, List, Dict, Any, Tuple

import numpy as np
from scipy import special as _spsp

from element_spectrum import iep_model as iep_io
from geometric_backbone import scan_mainlobe_domain as _mld
from geometric_backbone import dominant_mode_envelope as _a0p
from geometric_backbone import geometric_psll_metrics as _mps
from geometric_backbone import active_constraints as _as
from geometric_backbone import geometric_sampling_pool as _pool


# =============================================================================
# Small logging helper
# =============================================================================


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


# =============================================================================
# Block 1: IEP initialization
# =============================================================================


class Stage1IEPContext(object):
    __slots__ = (
        'model',
        'eco_eval',
        'a0_eval',
        'a0_abs',
        'g_main',
        'g_side',
    )

    def __init__(
        self,
        model,
        eco_eval,
        a0_eval,
        a0_abs,
        g_main,
        g_side,
    ):
        self.model = model
        self.eco_eval = eco_eval
        self.a0_eval = a0_eval
        self.a0_abs = a0_abs
        self.g_main = g_main
        self.g_side = g_side




def init_stage1_iep(
    unit_cell_npz_path: str,
    *,
    meta_json_path: Optional[str] = None,
    phi_main_rad: float = 0.0,
    bounds: str = "clamp",
    enable_bilinear_phi: bool = True,
    verbose: bool = True,
) -> Stage1IEPContext:
    _log(verbose, "=== [A0CRA][BLOCK1] INIT IEP: START ===")
    _log(verbose, f"[IEP] npz={unit_cell_npz_path}")
    _log(verbose, f"[IEP] meta_json={meta_json_path}")
    _log(verbose, f"[IEP] phi_main_rad={phi_main_rad:.6f}, bounds={bounds}, bilinear_phi={enable_bilinear_phi}")

    cfg = iep_io.IEPModelConfig(
        phi_main_rad=float(phi_main_rad),
        bounds=str(bounds),
        enable_bilinear_phi=bool(enable_bilinear_phi),
    )
    model = iep_io.IEPModel(unit_cell_npz_path, meta_json_path=meta_json_path, config=cfg)

    ctx = Stage1IEPContext(
        model=model,
        eco_eval=model.query_E_co,
        a0_eval=model.a0,
        a0_abs=(lambda theta_deg: np.asarray(np.abs(model.a0(np.deg2rad(theta_deg))), dtype=np.float64)),
        g_main=model.g_main,
        g_side=model.g_side,
    )

    th_smoke = np.array([0.0, 30.0, 60.0], dtype=np.float64)
    ph_smoke = np.deg2rad(np.array([0.0, 0.0, 0.0], dtype=np.float64))
    eco_smoke = np.asarray(ctx.eco_eval(np.deg2rad(th_smoke), ph_smoke))
    a0_smoke = np.asarray(ctx.a0_abs(th_smoke), dtype=np.float64)
    if np.any(~np.isfinite(eco_smoke.real)) or np.any(~np.isfinite(eco_smoke.imag)):
        raise ValueError("eco_eval smoke test returned NaN/Inf.")
    if np.any(~np.isfinite(a0_smoke)):
        raise ValueError("a0_abs smoke test returned NaN/Inf.")

    _log(verbose, f"[IEP] theta_grid_size={int(model.theta_grid_rad.size)}, phi_grid_size={int(model.phi_grid_rad.size)}")
    _log(verbose, f"[IEP] smoke |E_co|={np.round(np.abs(eco_smoke), 6)}")
    _log(verbose, f"[IEP] smoke |a0| ={np.round(a0_smoke, 6)}")
    _log(verbose, "=== [A0CRA][BLOCK1] INIT IEP: DONE ===")
    return ctx


# =============================================================================
# Block 2: Radius initialization (supports multiple initial solutions)
# =============================================================================


class RadiusInitSpec(object):
    __slots__ = (
        'name',
        'method',
        'K',
        'R0',
        'd_min',
        'b',
    )

    def __init__(
        self,
        name,
        method="gompertz",
        K=8,
        R0=5.0,
        d_min=0.5,
        b=6.0,
    ):
        self.name = name
        self.method = method
        self.K = K
        self.R0 = R0
        self.d_min = d_min
        self.b = b



class RadiusInitResult(object):
    __slots__ = (
        'name',
        'method',
        'r_over_lambda',
        'spec',
    )

    def __init__(
        self,
        name,
        method,
        r_over_lambda,
        spec,
    ):
        self.name = name
        self.method = method
        self.r_over_lambda = r_over_lambda
        self.spec = spec



class RadiusInitBatch(object):
    __slots__ = (
        'items',
    )

    def __init__(
        self,
        items,
    ):
        self.items = items




def _gompertz_radii(spec: RadiusInitSpec) -> np.ndarray:
    K = int(spec.K)
    R0 = float(spec.R0)
    d_min = float(spec.d_min)
    b = float(spec.b)
    if K < 1:
        raise ValueError("K must be >= 1")
    if d_min <= 0.0 or R0 <= 0.0:
        raise ValueError("d_min and R0 must be > 0")
    dK = K * d_min
    if R0 < dK - 1e-12:
        raise ValueError(f"R0({R0}) < K*d_min({dK}); Gompertz init infeasible.")

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
    return r.astype(np.float64)



def audit_radius_constraints(
    r_over_lambda,
    *,
    K,
    d_min,
    R0,
    feasibility_tol=5e-6,
):
    r = np.asarray(r_over_lambda, dtype=np.float64).reshape(-1)
    tol = float(feasibility_tol)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError('feasibility_tol must be finite and >= 0')
    if r.size != int(K):
        return {
            'ok': False,
            'reason': 'length_mismatch',
            'size': int(r.size),
            'K': int(K),
            'first_slack': float('nan'),
            'min_gap_slack': float('nan'),
            'outer_slack': float('nan'),
            'max_violation': float('inf'),
            'feasibility_tol': tol,
        }
    if np.any(~np.isfinite(r)):
        return {
            'ok': False,
            'reason': 'nonfinite_radius',
            'size': int(r.size),
            'K': int(K),
            'first_slack': float('nan'),
            'min_gap_slack': float('nan'),
            'outer_slack': float('nan'),
            'max_violation': float('inf'),
            'feasibility_tol': tol,
        }
    first_slack = float(r[0] - float(d_min))
    gaps = np.diff(r)
    min_gap_slack = float(np.min(gaps - float(d_min))) if gaps.size else float('inf')
    outer_slack = float(float(R0) - r[-1])
    violations = [max(0.0, -first_slack), max(0.0, -outer_slack)]
    if np.isfinite(min_gap_slack):
        violations.append(max(0.0, -min_gap_slack))
    max_violation = float(max(violations)) if violations else 0.0
    ok = bool(
        first_slack >= -tol
        and outer_slack >= -tol
        and (not np.isfinite(min_gap_slack) or min_gap_slack >= -tol)
        and np.all(np.diff(r) > -tol)
    )
    return {
        'ok': ok,
        'reason': 'ok' if ok else 'radius_constraint_violation',
        'size': int(r.size),
        'K': int(K),
        'first_slack': first_slack,
        'min_gap_slack': min_gap_slack,
        'outer_slack': outer_slack,
        'max_violation': max_violation,
        'feasibility_tol': tol,
    }


def validate_radius_init(
    r_over_lambda,
    *,
    K,
    d_min,
    R0,
    feasibility_tol=5e-6,
):
    r = np.asarray(r_over_lambda, dtype=np.float64).reshape(-1)
    audit = audit_radius_constraints(
        r,
        K=int(K),
        d_min=float(d_min),
        R0=float(R0),
        feasibility_tol=float(feasibility_tol),
    )
    if not bool(audit['ok']):
        raise ValueError('Invalid radius geometry: ' + json.dumps(audit, ensure_ascii=False, sort_keys=True))




def build_radius_init(spec: RadiusInitSpec, *, verbose: bool = True) -> RadiusInitResult:
    _log(verbose, "=== [A0CRA][BLOCK2] BUILD RADIUS INIT: START ===")
    _log(verbose, f"[RADIUS] name={spec.name}, method={spec.method}, K={spec.K}, R0={spec.R0}, d_min={spec.d_min}, b={spec.b}")
    method = str(spec.method).lower().strip()
    if method != "gompertz":
        raise ValueError(f"Unsupported radius init method: {spec.method}")

    r = _gompertz_radii(spec)
    validate_radius_init(r, K=spec.K, d_min=spec.d_min, R0=spec.R0)

    res = RadiusInitResult(name=spec.name, method=method, r_over_lambda=r, spec=spec)
    _log(verbose, f"[RADIUS] r_over_lambda={np.round(r, 6)}")
    _log(verbose, f"[RADIUS] min_gap={float(np.min(np.diff(r))):.6f}, max_gap={float(np.max(np.diff(r))):.6f}")
    _log(verbose, "=== [A0CRA][BLOCK2] BUILD RADIUS INIT: DONE ===")
    return res



def build_radius_init_batch(specs: Sequence[RadiusInitSpec], *, verbose: bool = True) -> RadiusInitBatch:
    specs_list = list(specs)
    if len(specs_list) == 0:
        raise ValueError("specs must be non-empty")
    _log(verbose, "=== [A0CRA][BLOCK2] BUILD RADIUS INIT BATCH: START ===")
    _log(verbose, f"[RADIUS][BATCH] count={len(specs_list)}")
    items: List[RadiusInitResult] = []
    for i, spec in enumerate(specs_list):
        _log(verbose, f"[RADIUS][BATCH] case {i + 1}/{len(specs_list)} -> {spec.name}")
        items.append(build_radius_init(spec, verbose=verbose))
    _log(verbose, "=== [A0CRA][BLOCK2] BUILD RADIUS INIT BATCH: DONE ===")
    return RadiusInitBatch(items=items)


# =============================================================================
# Block 3: Element initialization
# =============================================================================


class ElementInitConfig(object):
    __slots__ = (
        'd_elem_default_lambda',
        'include_center',
        'apply_outer5_mod4_minus1',
        'outer5_threshold_gt',
    )

    def __init__(
        self,
        d_elem_default_lambda=0.5,
        include_center=True,
        apply_outer5_mod4_minus1=True,
        outer5_threshold_gt=16,
    ):
        self.d_elem_default_lambda = float(d_elem_default_lambda)
        self.include_center = bool(include_center)
        self.apply_outer5_mod4_minus1 = bool(apply_outer5_mod4_minus1)
        self.outer5_threshold_gt = int(outer5_threshold_gt)




class RingElementCountResult(object):
    __slots__ = (
        'N_per_ring',
        'config',
        'include_center',
    )

    def __init__(
        self,
        N_per_ring,
        config,
        include_center,
    ):
        self.N_per_ring = N_per_ring
        self.config = config
        self.include_center = include_center



class ElementGeometry(object):
    __slots__ = (
        'x_elem',
        'y_elem',
        'ring_idx',
        'elem_idx_in_ring',
        'N_per_ring',
        'r_over_lambda',
        'include_center',
    )

    def __init__(
        self,
        x_elem,
        y_elem,
        ring_idx,
        elem_idx_in_ring,
        N_per_ring,
        r_over_lambda,
        include_center,
    ):
        self.x_elem = x_elem
        self.y_elem = y_elem
        self.ring_idx = ring_idx
        self.elem_idx_in_ring = elem_idx_in_ring
        self.N_per_ring = N_per_ring
        self.r_over_lambda = r_over_lambda
        self.include_center = include_center




def build_ring_element_counts(
    r_over_lambda,
    cfg=ElementInitConfig(),
    *,
    verbose=True,
):
    _log(verbose, '=== [A0CRA][BLOCK3] BUILD RING ELEMENT COUNTS: START ===')
    r = np.asarray(r_over_lambda, dtype=np.float64).reshape(-1)
    if r.size == 0:
        raise ValueError('r_over_lambda must be non-empty')
    if np.any(~np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError('r_over_lambda must be positive finite values')
    d_eff = float(cfg.d_elem_default_lambda)
    if d_eff <= 0.0:
        raise ValueError('d_elem_default_lambda must be > 0')
    threshold = int(cfg.outer5_threshold_gt)
    if threshold < 0:
        raise ValueError('outer5_threshold_gt must be >= 0')

    N_raw = np.floor((2.0 * math.pi * r) / d_eff + 1e-12).astype(np.int64)
    N = np.maximum(N_raw, 1)

    if bool(cfg.apply_outer5_mod4_minus1) and r.size >= 5:
        start = max(0, r.size - 5)
        for k in range(start, r.size):
            if int(N[k]) > threshold and int(N[k]) % 4 == 0:
                N[k] -= 1
                N[k] = max(int(N[k]), 1)

    _log(verbose, '[ELEMENT] d_elem_default_lambda=%.6f, include_center=%s, outer5_threshold_gt=%d' % (d_eff, cfg.include_center, threshold))
    _log(verbose, '[ELEMENT] raw_counts=%s' % N_raw.tolist())
    _log(verbose, '[ELEMENT] used_counts=%s' % N.tolist())
    _log(verbose, '=== [A0CRA][BLOCK3] BUILD RING ELEMENT COUNTS: DONE ===')
    return RingElementCountResult(N_per_ring=N.astype(np.int64), config=cfg, include_center=bool(cfg.include_center))




def build_element_geometry(
    r_over_lambda: np.ndarray,
    N_per_ring: np.ndarray,
    *,
    include_center: bool = True,
    verbose: bool = True,
) -> ElementGeometry:
    _log(verbose, "=== [A0CRA][BLOCK3] BUILD ELEMENT GEOMETRY: START ===")
    r = np.asarray(r_over_lambda, dtype=np.float64).reshape(-1)
    N = np.asarray(N_per_ring, dtype=np.int64).reshape(-1)
    if r.size != N.size:
        raise ValueError("r_over_lambda and N_per_ring must have same length")
    if np.any(N <= 0):
        raise ValueError("N_per_ring must be positive integers")

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ring_idx: List[np.ndarray] = []
    elem_idx: List[np.ndarray] = []

    if include_center:
        xs.append(np.array([0.0], dtype=np.float64))
        ys.append(np.array([0.0], dtype=np.float64))
        ring_idx.append(np.array([-1], dtype=np.int64))
        elem_idx.append(np.array([0], dtype=np.int64))

    for k, (rk, Nk) in enumerate(zip(r.tolist(), N.tolist())):
        psi = 2.0 * math.pi * np.arange(int(Nk), dtype=np.float64) / float(Nk)
        xs.append(float(rk) * np.cos(psi))
        ys.append(float(rk) * np.sin(psi))
        ring_idx.append(np.full((int(Nk),), int(k), dtype=np.int64))
        elem_idx.append(np.arange(int(Nk), dtype=np.int64))

    x_elem = np.concatenate(xs, axis=0).astype(np.float64)
    y_elem = np.concatenate(ys, axis=0).astype(np.float64)
    ring_idx_arr = np.concatenate(ring_idx, axis=0).astype(np.int64)
    elem_idx_arr = np.concatenate(elem_idx, axis=0).astype(np.int64)

    # validation: the first element in each ring must satisfy y ~= 0
    start = 1 if include_center else 0
    for k, Nk in enumerate(N.tolist()):
        pos = start + int(np.sum(N[:k]))
        if abs(float(y_elem[pos])) > 1e-10:
            raise ValueError(f"Ring {k} first element does not satisfy y=0 rule; got y={y_elem[pos]}")

    total_elements = int(x_elem.size)
    _log(verbose, f"[GEOM] include_center={include_center}, total_elements={total_elements}")
    _log(verbose, f"[GEOM] first_ring_first_xy=({float(x_elem[start]):.6f}, {float(y_elem[start]):.6e})")
    if N.size > 0:
        last_ring_start = start + int(np.sum(N[:-1])) if N.size > 1 else start
        _log(verbose, f"[GEOM] last_ring_first_xy=({float(x_elem[last_ring_start]):.6f}, {float(y_elem[last_ring_start]):.6e})")
    _log(verbose, "=== [A0CRA][BLOCK3] BUILD ELEMENT GEOMETRY: DONE ===")
    return ElementGeometry(
        x_elem=x_elem,
        y_elem=y_elem,
        ring_idx=ring_idx_arr,
        elem_idx_in_ring=elem_idx_arr,
        N_per_ring=N.astype(np.int64),
        r_over_lambda=r.astype(np.float64),
        include_center=bool(include_center),
    )


# =============================================================================
# Block 4: Mainlobe-domain state
# =============================================================================


class MainlobeDomainConfig(object):
    __slots__ = (
        'theta_s_deg_list',
        'phi_s_deg',
        'use_domain',
        'source',
        'final_source',
        'freeze_during_optimization',
        'alpha_table_json_path',
        'alpha_table_tol',
        'mainlobe_margin_deg',
        'solver_cfg',
    )

    def __init__(
        self,
        theta_s_deg_list,
        phi_s_deg=0.0,
        use_domain='null_guard',
        source='table_raw_b',
        final_source='directsum',
        freeze_during_optimization=True,
        alpha_table_json_path='',
        alpha_table_tol=1e-9,
        mainlobe_margin_deg=0.5,
        solver_cfg=_mld.MainlobeDomainConfig(),
    ):
        self.theta_s_deg_list = theta_s_deg_list
        self.phi_s_deg = float(phi_s_deg)
        self.use_domain = str(use_domain)
        self.source = str(source)
        self.final_source = str(final_source)
        self.freeze_during_optimization = bool(freeze_during_optimization)
        self.alpha_table_json_path = str(alpha_table_json_path)
        self.alpha_table_tol = float(alpha_table_tol)
        self.mainlobe_margin_deg = float(mainlobe_margin_deg)
        self.solver_cfg = solver_cfg




class MainlobeDomainState(object):
    __slots__ = (
        'theta_s_deg',
        'phi_s_deg',
        'alpha_null_deg',
        'alpha_guard_null_deg',
        'alpha_3db_deg',
        'use_domain',
        'stats',
    )

    def __init__(
        self,
        theta_s_deg,
        phi_s_deg,
        alpha_null_deg,
        alpha_guard_null_deg,
        alpha_3db_deg,
        use_domain,
        stats,
    ):
        self.theta_s_deg = theta_s_deg
        self.phi_s_deg = phi_s_deg
        self.alpha_null_deg = alpha_null_deg
        self.alpha_guard_null_deg = alpha_guard_null_deg
        self.alpha_3db_deg = alpha_3db_deg
        self.use_domain = use_domain
        self.stats = stats



_ALPHA_MAIN_TABLE_CACHE: Dict[str, Dict[str, Any]] = {}


def _resolve_alpha_main_table_json_path(path_hint: str = "") -> str:
    if str(path_hint).strip():
        p = Path(str(path_hint)).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"alpha-main table JSON not found: {p}")

    names = [
        'a0cra_alpha_main_table.json',
        'a0cra_mainlobe_runtime_table.json',
        'alpha_main_table.json',
    ]
    roots = []
    try:
        roots.append(Path.cwd())
    except Exception:
        pass
    try:
        roots.append(Path(__file__).resolve().parent)
    except Exception:
        pass

    seen = set()
    for root in roots:
        for name in names:
            cand = (root / name).resolve()
            key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            if cand.is_file():
                return str(cand)

    raise FileNotFoundError(
        "Cannot find alpha-main table JSON. Expected a0cra_alpha_main_table.json "
        "in current directory or script directory, or set alpha_table_json_path."
    )


def _load_alpha_main_table_json(path: str) -> Dict[str, Any]:
    p = str(Path(path).expanduser().resolve())
    cached = _ALPHA_MAIN_TABLE_CACHE.get(p)
    if cached is not None:
        return cached
    with open(p, 'r', encoding='utf-8') as f:
        table = json.load(f)
    if str(table.get('format', '')) != 'a0cra_alpha_main_table_v1':
        raise ValueError(f"Unsupported alpha-main table format: {table.get('format')}")
    entries = list(table.get('entries', []))
    if not entries:
        raise ValueError('alpha-main table has no entries')
    _ALPHA_MAIN_TABLE_CACHE[p] = table
    return table


def _find_alpha_main_table_entry(
    table: Dict[str, Any],
    *,
    K: int,
    d_min: float,
    d_eff: float,
    R0: float,
    include_center: bool,
    tol: float = 1e-9,
) -> Dict[str, Any]:
    for entry in list(table.get('entries', [])):
        key = dict(entry.get('key', {}))
        if int(key.get('K')) != int(K):
            continue
        if abs(float(key.get('d_min')) - float(d_min)) > float(tol):
            continue
        if abs(float(key.get('d_eff')) - float(d_eff)) > float(tol):
            continue
        if abs(float(key.get('R0')) - float(R0)) > float(tol):
            continue
        if bool(key.get('include_center', True)) != bool(include_center):
            continue
        return entry
    raise KeyError(
        "No alpha-main table entry for "
        f"K={K}, d_min={d_min}, d_eff={d_eff}, R0={R0}, include_center={include_center}"
    )


def _match_scalar_index(values: np.ndarray, target: float, *, name: str, tol: float = 1e-9) -> int:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    hit = np.where(np.isclose(arr, float(target), atol=float(tol), rtol=0.0))[0]
    if hit.size == 0:
        raise KeyError(f"{name}={float(target)} not found in table values={arr.astype(float).tolist()}")
    return int(hit[0])


def _theta_indices_from_entry(
    entry: Dict[str, Any],
    theta_s_deg_required: np.ndarray,
    *,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    theta_all = np.asarray(entry['theta_s_deg'], dtype=np.float64).reshape(-1)
    req = np.asarray(theta_s_deg_required, dtype=np.float64).reshape(-1)
    if req.size == 0:
        raise ValueError('theta_s_deg_required is empty')
    idx = []
    missing = []
    for th in req.tolist():
        hit = np.where(np.isclose(theta_all, float(th), atol=float(tol), rtol=0.0))[0]
        if hit.size == 0:
            missing.append(float(th))
        else:
            idx.append(int(hit[0]))
    if missing:
        raise KeyError(f"alpha-main table missing theta_s_deg={missing}")
    idx_arr = np.asarray(idx, dtype=np.int64)
    return theta_all[idx_arr].astype(np.float64), idx_arr


def _build_mainlobe_domain_state_from_alpha_table_raw_b(
    radius_init: RadiusInitResult,
    elem_counts: RingElementCountResult,
    ml_cfg: MainlobeDomainConfig,
    theta_s_deg: np.ndarray,
    *,
    verbose: bool = True,
) -> MainlobeDomainState:
    table_path = _resolve_alpha_main_table_json_path(ml_cfg.alpha_table_json_path)
    table = _load_alpha_main_table_json(table_path)

    K = int(np.asarray(radius_init.r_over_lambda, dtype=np.float64).size)
    d_min = float(radius_init.spec.d_min)
    d_eff = float(elem_counts.config.d_elem_default_lambda)
    R0 = float(radius_init.spec.R0)
    b = float(radius_init.spec.b)

    entry = _find_alpha_main_table_entry(
        table,
        K=K,
        d_min=d_min,
        d_eff=d_eff,
        R0=R0,
        include_center=bool(elem_counts.include_center),
        tol=float(ml_cfg.alpha_table_tol),
    )
    b_values = np.asarray(entry.get('debug', {}).get('b_values', []), dtype=np.float64).reshape(-1)
    if b_values.size == 0:
        raise KeyError('alpha-main table entry missing debug.b_values; raw-b lookup is impossible')
    b_idx = _match_scalar_index(b_values, b, name='b', tol=float(ml_cfg.alpha_table_tol))
    theta_used, theta_idx = _theta_indices_from_entry(entry, theta_s_deg, tol=float(ml_cfg.alpha_table_tol))

    raw = dict(entry.get('debug', {}).get('raw', {}))
    required = ['alpha_null_deg_by_b_theta', 'alpha_guard_null_deg_by_b_theta', 'alpha_3db_deg_by_b_theta']
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f'alpha-main table raw block missing fields: {missing}')

    alpha_null_raw = np.asarray(raw['alpha_null_deg_by_b_theta'], dtype=np.float64)
    alpha_guard_raw = np.asarray(raw['alpha_guard_null_deg_by_b_theta'], dtype=np.float64)
    alpha_3db_raw = np.asarray(raw['alpha_3db_deg_by_b_theta'], dtype=np.float64)
    expected_shape = (int(b_values.size), int(np.asarray(entry['theta_s_deg']).size))
    for name, arr in [
        ('alpha_null_deg_by_b_theta', alpha_null_raw),
        ('alpha_guard_null_deg_by_b_theta', alpha_guard_raw),
        ('alpha_3db_deg_by_b_theta', alpha_3db_raw),
    ]:
        if arr.shape != expected_shape:
            raise ValueError(f'{name} shape mismatch: got {arr.shape}, expected {expected_shape}')

    margin = float(ml_cfg.mainlobe_margin_deg)
    alpha_null = np.asarray(alpha_null_raw[b_idx, theta_idx], dtype=np.float64)
    alpha_guard = np.maximum(0.0, alpha_guard_raw[b_idx, theta_idx] - margin).astype(np.float64)
    alpha_3db = np.asarray(alpha_3db_raw[b_idx, theta_idx], dtype=np.float64)

    state = MainlobeDomainState(
        theta_s_deg=theta_used,
        phi_s_deg=float(ml_cfg.phi_s_deg),
        alpha_null_deg=alpha_null,
        alpha_guard_null_deg=alpha_guard,
        alpha_3db_deg=alpha_3db,
        use_domain=str(ml_cfg.use_domain),
        stats={
            'source': 'alpha_main_table_raw_b',
            'table_json_path': str(table_path),
            'table_format': str(table.get('format', '')),
            'table_key': dict(entry.get('key', {})),
            'b': float(b),
            'b_index': int(b_idx),
            'mainlobe_margin_deg': float(margin),
            'raw_fields': required,
            'note': 'alpha_null is read from the per-b raw table without margin; alpha_guard is reduced by margin; alpha_3db is read per-b without margin.',
        },
    )
    _log(verbose, '=== [A0CRA][BLOCK4] BUILD MAINLOBE DOMAIN STATE FROM RAW-B TABLE ===')
    _log(verbose, f'[MAINLOBE][TABLE] path={table_path}')
    _log(verbose, f'[MAINLOBE][TABLE] key={state.stats["table_key"]} b={b:.6g} b_idx={b_idx}')
    _log(verbose, f'[MAINLOBE][TABLE] alpha_guard_null_deg={np.round(state.alpha_guard_null_deg, 6)}')
    _log(verbose, f'[MAINLOBE][TABLE] alpha_3db_deg={np.round(state.alpha_3db_deg, 6)}')
    return state


def _build_mainlobe_domain_state_from_alpha_table_conservative(
    radius_init: RadiusInitResult,
    elem_counts: RingElementCountResult,
    ml_cfg: MainlobeDomainConfig,
    theta_s_deg: np.ndarray,
    *,
    verbose: bool = True,
) -> MainlobeDomainState:
    table_path = _resolve_alpha_main_table_json_path(ml_cfg.alpha_table_json_path)
    table = _load_alpha_main_table_json(table_path)
    entry = _find_alpha_main_table_entry(
        table,
        K=int(np.asarray(radius_init.r_over_lambda, dtype=np.float64).size),
        d_min=float(radius_init.spec.d_min),
        d_eff=float(elem_counts.config.d_elem_default_lambda),
        R0=float(radius_init.spec.R0),
        include_center=bool(elem_counts.include_center),
        tol=float(ml_cfg.alpha_table_tol),
    )
    theta_used, theta_idx = _theta_indices_from_entry(entry, theta_s_deg, tol=float(ml_cfg.alpha_table_tol))
    alpha_main_all = np.asarray(entry['alpha_main_deg'], dtype=np.float64).reshape(-1)
    alpha_3db_all = np.asarray(entry['alpha_3db_deg'], dtype=np.float64).reshape(-1)
    alpha_main = alpha_main_all[theta_idx].astype(np.float64)
    alpha_3db = alpha_3db_all[theta_idx].astype(np.float64)
    state = MainlobeDomainState(
        theta_s_deg=theta_used,
        phi_s_deg=float(ml_cfg.phi_s_deg),
        alpha_null_deg=alpha_main.copy(),
        alpha_guard_null_deg=alpha_main.copy(),
        alpha_3db_deg=alpha_3db,
        use_domain=str(ml_cfg.use_domain),
        stats={
            'source': 'alpha_main_table_conservative',
            'table_json_path': str(table_path),
            'table_format': str(table.get('format', '')),
            'table_key': dict(entry.get('key', {})),
            'note': 'alpha_null/alpha_guard are populated from top-level conservative alpha_main_deg.',
        },
    )
    _log(verbose, '=== [A0CRA][BLOCK4] BUILD MAINLOBE DOMAIN STATE FROM CONSERVATIVE TABLE ===')
    _log(verbose, f'[MAINLOBE][TABLE] path={table_path}')
    _log(verbose, f'[MAINLOBE][TABLE] alpha_main_deg={np.round(state.alpha_guard_null_deg, 6)}')
    _log(verbose, f'[MAINLOBE][TABLE] alpha_3db_deg={np.round(state.alpha_3db_deg, 6)}')
    return state


def _build_mainlobe_domain_state_directsum(
    iep_ctx: Stage1IEPContext,
    radius_init: RadiusInitResult,
    elem_counts: RingElementCountResult,
    ml_cfg: MainlobeDomainConfig,
    theta_s_deg: np.ndarray,
    *,
    verbose: bool = True,
) -> MainlobeDomainState:
    geom = _mld.ArrayGeometry(
        r_over_lambda=np.asarray(radius_init.r_over_lambda, dtype=np.float64),
        N_per_ring=np.asarray(elem_counts.N_per_ring, dtype=np.int64),
        include_center=bool(elem_counts.include_center),
    )
    scan = _mld.ScanSpecMainlobe(theta_s_deg_list=theta_s_deg, phi_s_deg=0.0)

    _log(verbose, f"[MAINLOBE] use_domain={ml_cfg.use_domain}")
    _log(verbose, f"[MAINLOBE] theta_s_deg_list={theta_s_deg.tolist()}")
    _log(verbose, f"[MAINLOBE] geometry.r={np.round(geom.r_over_lambda, 6)}")
    _log(verbose, f"[MAINLOBE] geometry.N={geom.N_per_ring.astype(int).tolist()} include_center={geom.include_center}")

    res = _mld.estimate_mainlobe_domains_from_directsum(
        eco_eval=iep_ctx.eco_eval,
        geometry=geom,
        scan=scan,
        cfg=ml_cfg.solver_cfg,
    )

    return MainlobeDomainState(
        theta_s_deg=np.asarray(res.theta_s_deg, dtype=np.float64),
        phi_s_deg=float(res.phi_s_deg),
        alpha_null_deg=np.asarray(res.alpha_null_deg, dtype=np.float64),
        alpha_guard_null_deg=np.asarray(res.alpha_guard_null_deg, dtype=np.float64),
        alpha_3db_deg=np.asarray(res.alpha_3db_deg, dtype=np.float64),
        use_domain=str(ml_cfg.use_domain),
        stats=dict(res.stats),
    )


def build_mainlobe_domain_state(
    iep_ctx: Stage1IEPContext,
    radius_init: RadiusInitResult,
    elem_counts: RingElementCountResult,
    ml_cfg: MainlobeDomainConfig,
    *,
    verbose: bool = True,
) -> MainlobeDomainState:
    _log(verbose, "=== [A0CRA][BLOCK4] BUILD MAINLOBE DOMAIN STATE: START ===")
    theta_s_deg = np.asarray(ml_cfg.theta_s_deg_list, dtype=np.float64).reshape(-1)
    theta_s_deg = theta_s_deg[np.isfinite(theta_s_deg)]
    theta_s_deg = np.unique(theta_s_deg)
    theta_s_deg.sort()
    if theta_s_deg.size == 0:
        raise ValueError("theta_s_deg_list must be non-empty")
    if abs(float(ml_cfg.phi_s_deg)) > 1e-12:
        raise ValueError("First version fixes phi_s_deg=0.0")

    source = str(getattr(ml_cfg, 'source', 'table_raw_b')).lower().strip()
    if source == 'table_raw_b':
        state = _build_mainlobe_domain_state_from_alpha_table_raw_b(
            radius_init, elem_counts, ml_cfg, theta_s_deg, verbose=verbose,
        )
    elif source == 'table_conservative':
        state = _build_mainlobe_domain_state_from_alpha_table_conservative(
            radius_init, elem_counts, ml_cfg, theta_s_deg, verbose=verbose,
        )
    elif source == 'directsum':
        state = _build_mainlobe_domain_state_directsum(
            iep_ctx, radius_init, elem_counts, ml_cfg, theta_s_deg, verbose=verbose,
        )
    else:
        raise ValueError("MainlobeDomainConfig.source must be 'table_raw_b', 'table_conservative', or 'directsum'")

    _log(verbose, f"[MAINLOBE] source={source}")
    _log(verbose, f"[MAINLOBE] alpha_null_deg={np.round(state.alpha_null_deg, 6)}")
    _log(verbose, f"[MAINLOBE] alpha_guard_null_deg={np.round(state.alpha_guard_null_deg, 6)}")
    _log(verbose, f"[MAINLOBE] alpha_3db_deg={np.round(state.alpha_3db_deg, 6)}")
    if state.alpha_null_deg.size > 0:
        _log(verbose, f"[MAINLOBE] boresight_null_rad={math.radians(float(state.alpha_null_deg[0])):.6f}")
        _log(verbose, f"[MAINLOBE] boresight_3db_rad={math.radians(float(state.alpha_3db_deg[0])):.6f}")
    _log(verbose, "=== [A0CRA][BLOCK4] BUILD MAINLOBE DOMAIN STATE: DONE ===")
    return state


# =============================================================================
# Block 5: Stage-1 static + dynamic pool management
# =============================================================================


class Stage1PoolConfig(object):
    __slots__ = (
        'pool_npz_path',
        'pool_json_path',
        'tol_deg',
    )

    def __init__(
        self,
        pool_npz_path,
        pool_json_path,
        tol_deg=1e-9,
    ):
        self.pool_npz_path = pool_npz_path
        self.pool_json_path = pool_json_path
        self.tol_deg = tol_deg



class Stage1RuntimePoolConfig(object):
    __slots__ = (
        'partition_table_json',
        'lookup_tol',
        'pool_tol_deg',
    )

    def __init__(
        self,
        partition_table_json,
        lookup_tol=1e-9,
        pool_tol_deg=1e-9,
    ):
        self.partition_table_json = str(partition_table_json)
        self.lookup_tol = float(lookup_tol)
        self.pool_tol_deg = float(pool_tol_deg)

class Stage1PoolContext(object):
    __slots__ = (
        'static_npz',
        'summary',
        'theta_s_full_deg',
        'theta_s_work_init_deg',
        'work_anchor_theta_s_deg',
        'current_work_theta_s_deg',
        'tracking_protocol',
        'alpha_main_guard_deg',
        'alpha_core_used_deg',
        'alpha_outer_start_deg',
        'partition_info',
        'tol_deg',
        'canonical_static_by_scan',
        'static_slice_validated',
        'dynamic_buffer_work_by_scan',
        'dynamic_outer_work_by_scan',
        'dynamic_buffer_full_by_scan',
        'dynamic_outer_full_by_scan',
    )

    def __init__(
        self,
        static_npz,
        summary,
        theta_s_full_deg,
        theta_s_work_init_deg,
        work_anchor_theta_s_deg,
        current_work_theta_s_deg,
        tracking_protocol,
        alpha_main_guard_deg,
        alpha_core_used_deg,
        alpha_outer_start_deg,
        partition_info,
        tol_deg,
        canonical_static_by_scan,
        static_slice_validated,
        dynamic_buffer_work_by_scan,
        dynamic_outer_work_by_scan,
        dynamic_buffer_full_by_scan,
        dynamic_outer_full_by_scan,
    ):
        self.static_npz = static_npz
        self.summary = summary
        self.theta_s_full_deg = theta_s_full_deg
        self.theta_s_work_init_deg = theta_s_work_init_deg
        self.work_anchor_theta_s_deg = work_anchor_theta_s_deg
        self.current_work_theta_s_deg = current_work_theta_s_deg
        self.tracking_protocol = tracking_protocol
        self.alpha_main_guard_deg = float(alpha_main_guard_deg)
        self.alpha_core_used_deg = float(alpha_core_used_deg)
        self.alpha_outer_start_deg = float(alpha_outer_start_deg)
        self.partition_info = dict(partition_info)
        self.tol_deg = float(tol_deg)
        self.canonical_static_by_scan = canonical_static_by_scan
        self.static_slice_validated = bool(static_slice_validated)
        self.dynamic_buffer_work_by_scan = dynamic_buffer_work_by_scan
        self.dynamic_outer_work_by_scan = dynamic_outer_work_by_scan
        self.dynamic_buffer_full_by_scan = dynamic_buffer_full_by_scan
        self.dynamic_outer_full_by_scan = dynamic_outer_full_by_scan




def _canonical_theta_deg(theta_s_deg: float, tol_deg: float) -> float:
    scale = max(1, int(round(1.0 / max(tol_deg, 1e-12))))
    return float(round(float(theta_s_deg) * scale) / scale)


def _theta_mask(theta_vec_deg: np.ndarray, theta_s_deg: float, tol_deg: float) -> np.ndarray:
    return np.isclose(np.asarray(theta_vec_deg, dtype=np.float64), float(theta_s_deg), atol=float(tol_deg))


def _require_npz_fields(data: Dict[str, np.ndarray], fields: Sequence[str]) -> None:
    missing = [k for k in fields if k not in data]
    if missing:
        raise KeyError(f"Stage1 pool npz missing fields: {missing}")


def _load_npz_dict(npz_path: str) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def _build_static_core_points_from_prefix(
    static_npz: Dict[str, np.ndarray],
    theta_s_deg: float,
    *,
    prefix: str,
    tol_deg: float,
    alpha_main_guard_deg: float,
) -> Dict[str, np.ndarray]:
    mask = _theta_mask(static_npz[f'{prefix}_theta_s_deg'], theta_s_deg, tol_deg)
    alpha_deg = np.asarray(static_npz[f'{prefix}_alpha_deg'][mask], dtype=np.float64)
    rho = np.asarray(static_npz[f'{prefix}_rho'][mask], dtype=np.float64)
    theta_min_deg = np.asarray(static_npz[f'{prefix}_theta_min_deg'][mask], dtype=np.float64)
    theta_max_deg = np.asarray(static_npz[f'{prefix}_theta_max_deg'][mask], dtype=np.float64)
    is_main_guard = (alpha_deg <= float(alpha_main_guard_deg) + 1e-12).astype(np.int8)
    is_side_1d = (1 - is_main_guard).astype(np.int8)
    return {
        'alpha_deg': alpha_deg,
        'rho': rho,
        'theta_min_deg': theta_min_deg,
        'theta_max_deg': theta_max_deg,
        'is_main_guard': is_main_guard,
        'is_side_1d': is_side_1d,
    }


def _build_static_2d_points_from_prefix(
    static_npz: Dict[str, np.ndarray],
    theta_s_deg: float,
    *,
    prefix: str,
    tol_deg: float,
) -> Dict[str, np.ndarray]:
    mask = _theta_mask(static_npz[f'{prefix}_theta_s_deg'], theta_s_deg, tol_deg)
    alpha_deg = np.asarray(static_npz[f'{prefix}_alpha_deg'][mask], dtype=np.float64)
    eta_deg = np.asarray(static_npz[f'{prefix}_eta_deg'][mask], dtype=np.float64)
    theta_deg = np.asarray(static_npz[f'{prefix}_theta_deg'][mask], dtype=np.float64)
    phi_deg = np.asarray(static_npz[f'{prefix}_phi_deg'][mask], dtype=np.float64)
    u = np.asarray(static_npz[f'{prefix}_u'][mask], dtype=np.float64)
    v = np.asarray(static_npz[f'{prefix}_v'][mask], dtype=np.float64)
    layer_code = np.asarray(static_npz[f'{prefix}_layer_code'][mask], dtype=np.int8)
    n = int(alpha_deg.size)
    return {
        'alpha_deg': alpha_deg,
        'eta_deg': eta_deg,
        'theta_deg': theta_deg,
        'phi_deg': phi_deg,
        'u': u,
        'v': v,
        'layer_code': layer_code,
        'is_side_2d': np.ones((n,), dtype=np.int8),
        'is_dynamic': np.zeros((n,), dtype=np.int8),
        'seed_rank': np.full((n,), -1, dtype=np.int32),
        'refine_round': np.zeros((n,), dtype=np.int8),
    }


def _build_canonical_static_by_scan(
    static_npz: Dict[str, np.ndarray],
    theta_s_full_deg: np.ndarray,
    *,
    tol_deg: float,
    alpha_main_guard_deg: float,
) -> Dict[float, Dict[str, Dict[str, np.ndarray]]]:
    out: Dict[float, Dict[str, Dict[str, np.ndarray]]] = {}
    for theta_s_deg in np.asarray(theta_s_full_deg, dtype=np.float64).tolist():
        key = _canonical_theta_deg(float(theta_s_deg), tol_deg)
        out[key] = {
            'core_1d': _build_static_core_points_from_prefix(
                static_npz,
                key,
                prefix='core_full',
                tol_deg=tol_deg,
                alpha_main_guard_deg=alpha_main_guard_deg,
            ),
            'buffer_2d': _build_static_2d_points_from_prefix(
                static_npz,
                key,
                prefix='buffer_full',
                tol_deg=tol_deg,
            ),
            'outer_2d': _build_static_2d_points_from_prefix(
                static_npz,
                key,
                prefix='outer_full',
                tol_deg=tol_deg,
            ),
        }
    return out


def _assert_same_shape_and_values(name: str, got: np.ndarray, exp: np.ndarray, *, atol: float = 1e-9) -> None:
    g = np.asarray(got)
    e = np.asarray(exp)
    if g.shape != e.shape:
        raise AssertionError(f'{name} shape mismatch: {g.shape} vs {e.shape}')
    if np.issubdtype(g.dtype, np.floating) or np.issubdtype(e.dtype, np.floating):
        if not np.allclose(g.astype(np.float64), e.astype(np.float64), atol=atol, rtol=0.0, equal_nan=True):
            raise AssertionError(f'{name} values mismatch.')
    else:
        if not np.array_equal(g, e):
            raise AssertionError(f'{name} values mismatch.')


def validate_work_is_full_slice(pool_ctx: Stage1PoolContext) -> None:
    static_npz = pool_ctx.static_npz
    for theta_s_deg in pool_ctx.theta_s_work_init_deg.tolist():
        key = _canonical_theta_deg(float(theta_s_deg), pool_ctx.tol_deg)
        if key not in pool_ctx.canonical_static_by_scan:
            raise AssertionError(f'work theta_s={key} not present in canonical full pool')
        canonical = pool_ctx.canonical_static_by_scan[key]

        work_core = _build_static_core_points_from_prefix(
            static_npz,
            key,
            prefix='core_work',
            tol_deg=pool_ctx.tol_deg,
            alpha_main_guard_deg=pool_ctx.alpha_main_guard_deg,
        )
        for field in ('alpha_deg', 'rho', 'theta_min_deg', 'theta_max_deg'):
            _assert_same_shape_and_values(f'core[{key}].{field}', work_core[field], canonical['core_1d'][field])

        work_buf = _build_static_2d_points_from_prefix(
            static_npz,
            key,
            prefix='buffer_work',
            tol_deg=pool_ctx.tol_deg,
        )
        for field in ('alpha_deg', 'eta_deg', 'theta_deg', 'phi_deg', 'u', 'v', 'layer_code'):
            _assert_same_shape_and_values(f'buffer[{key}].{field}', work_buf[field], canonical['buffer_2d'][field])

        work_out = _build_static_2d_points_from_prefix(
            static_npz,
            key,
            prefix='outer_work',
            tol_deg=pool_ctx.tol_deg,
        )
        for field in ('alpha_deg', 'eta_deg', 'theta_deg', 'phi_deg', 'u', 'v', 'layer_code'):
            _assert_same_shape_and_values(f'outer[{key}].{field}', work_out[field], canonical['outer_2d'][field])


def _record_to_plain_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _record_to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_record_to_plain_dict(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    slots = getattr(obj, '__slots__', None)
    if slots is not None:
        return {str(name): _record_to_plain_dict(getattr(obj, name)) for name in slots}
    return obj


def _pool_package_to_static_arrays(package):
    static_npz = {
        'theta_s_full_deg': np.asarray(package.theta_s_full_deg, dtype=np.float64),
        'theta_s_work_init_deg': np.asarray(package.theta_s_work_init_deg, dtype=np.float64),
        'theta_s_work_full_scan_idx': np.asarray(package.theta_s_work_full_scan_idx, dtype=np.int64),
    }
    for prefix, batch in (
        ('core_full', package.core_full),
        ('core_work', package.core_work),
        ('buffer_full', package.buffer_full),
        ('buffer_work', package.buffer_work),
        ('outer_full', package.outer_full),
        ('outer_work', package.outer_work),
    ):
        for key, value in batch.to_dict().items():
            static_npz[prefix + '_' + str(key)] = np.asarray(value)
    return static_npz


def _pool_package_summary(package):
    cfg_dict = _record_to_plain_dict(package.config)
    partition_dict = _record_to_plain_dict(package.partition_info)
    tracking_dict = _record_to_plain_dict(package.config.tracking)
    return {
        'version': 'q0cra_stage1_static_pool_v2_runtime',
        'config': cfg_dict,
        'partition_info': partition_dict,
        'tracking_protocol': tracking_dict,
        'validation': _record_to_plain_dict(package.validation_summary),
        'runtime_generated': True,
    }


def build_runtime_stage1_pool_package(
    radius_spec,
    element_cfg,
    runtime_pool_cfg,
    *,
    verbose=True,
):
    if not str(runtime_pool_cfg.partition_table_json).strip():
        raise ValueError('partition_table_json must be provided for runtime entry pool generation')
    cfg = _pool.PoolConfigStage1(
        partition_table_json=str(runtime_pool_cfg.partition_table_json),
        partition_selection_mode='entry',
        partition_lookup_K=int(radius_spec.K),
        partition_lookup_d_min=float(radius_spec.d_min),
        partition_lookup_d_eff=float(element_cfg.d_elem_default_lambda),
        partition_lookup_R0=float(radius_spec.R0),
        partition_lookup_tol=float(runtime_pool_cfg.lookup_tol),
        require_partition_table=True,
        allow_manual_partition=False,
        auto_buffer_rings=True,
    )
    package = _pool.build_stage1_static_pool(cfg)
    info = package.partition_info
    checks = (
        int(info.K) == int(radius_spec.K),
        abs(float(info.d_min) - float(radius_spec.d_min)) <= float(runtime_pool_cfg.lookup_tol),
        abs(float(info.d_eff) - float(element_cfg.d_elem_default_lambda)) <= float(runtime_pool_cfg.lookup_tol),
        abs(float(info.R0) - float(radius_spec.R0)) <= float(runtime_pool_cfg.lookup_tol),
        str(info.selection_mode) == 'entry',
    )
    if not all(checks):
        raise ValueError('Runtime pool entry does not match the requested geometry')
    _log(verbose, '[POOL][RUNTIME] built entry pool K=%d d_min=%.6f d_eff=%.6f R0=%.6f alpha_core=%.6f' % (
        int(info.K), float(info.d_min), float(info.d_eff), float(info.R0), float(info.alpha_core_used_deg)))
    return package


def _init_stage1_pool_context_from_data(static_npz, summary, tol_deg, verbose=True):
    _require_npz_fields(static_npz, [
        'theta_s_full_deg', 'theta_s_work_init_deg',
        'core_full_theta_s_deg', 'core_full_alpha_deg', 'core_full_rho', 'core_full_theta_min_deg', 'core_full_theta_max_deg',
        'buffer_full_theta_s_deg', 'buffer_full_alpha_deg', 'buffer_full_eta_deg', 'buffer_full_theta_deg', 'buffer_full_phi_deg', 'buffer_full_u', 'buffer_full_v', 'buffer_full_layer_code',
        'outer_full_theta_s_deg', 'outer_full_alpha_deg', 'outer_full_eta_deg', 'outer_full_theta_deg', 'outer_full_phi_deg', 'outer_full_u', 'outer_full_v', 'outer_full_layer_code',
        'core_work_theta_s_deg', 'core_work_alpha_deg', 'core_work_rho', 'core_work_theta_min_deg', 'core_work_theta_max_deg',
        'buffer_work_theta_s_deg', 'buffer_work_alpha_deg', 'buffer_work_eta_deg', 'buffer_work_theta_deg', 'buffer_work_phi_deg', 'buffer_work_u', 'buffer_work_v', 'buffer_work_layer_code',
        'outer_work_theta_s_deg', 'outer_work_alpha_deg', 'outer_work_eta_deg', 'outer_work_theta_deg', 'outer_work_phi_deg', 'outer_work_u', 'outer_work_v', 'outer_work_layer_code',
    ])
    theta_s_full_deg = np.asarray(static_npz['theta_s_full_deg'], dtype=np.float64).reshape(-1)
    theta_s_work_init_deg = np.asarray(static_npz['theta_s_work_init_deg'], dtype=np.float64).reshape(-1)
    theta_s_full_deg = np.unique(theta_s_full_deg[np.isfinite(theta_s_full_deg)])
    theta_s_work_init_deg = np.unique(theta_s_work_init_deg[np.isfinite(theta_s_work_init_deg)])
    theta_s_full_deg.sort()
    theta_s_work_init_deg.sort()
    tracking_protocol = dict(summary.get('tracking_protocol', {}))
    config_dict = dict(summary.get('config', {}) or {})
    partition_info = dict(summary.get('partition_info', {}) or {})
    alpha_main_guard_deg = float(config_dict.get('alpha_main_guard_deg', 7.5))
    alpha_core_used_deg = float(partition_info.get('alpha_core_used_deg', config_dict.get('alpha_core_max_deg', 57.0)))
    alpha_outer_start_deg = float(partition_info.get('alpha_outer_start_deg', config_dict.get('alpha_outer_start_deg', 90.0)))
    canonical_static_by_scan = _build_canonical_static_by_scan(
        static_npz,
        theta_s_full_deg,
        tol_deg=float(tol_deg),
        alpha_main_guard_deg=alpha_main_guard_deg,
    )
    ctx = Stage1PoolContext(
        static_npz=static_npz,
        summary=summary,
        theta_s_full_deg=theta_s_full_deg,
        theta_s_work_init_deg=theta_s_work_init_deg,
        work_anchor_theta_s_deg=theta_s_work_init_deg.copy(),
        current_work_theta_s_deg=theta_s_work_init_deg.copy(),
        tracking_protocol=tracking_protocol,
        alpha_main_guard_deg=alpha_main_guard_deg,
        alpha_core_used_deg=alpha_core_used_deg,
        alpha_outer_start_deg=alpha_outer_start_deg,
        partition_info=partition_info,
        tol_deg=float(tol_deg),
        canonical_static_by_scan=canonical_static_by_scan,
        static_slice_validated=False,
        dynamic_buffer_work_by_scan={},
        dynamic_outer_work_by_scan={},
        dynamic_buffer_full_by_scan={},
        dynamic_outer_full_by_scan={},
    )
    validate_work_is_full_slice(ctx)
    ctx.static_slice_validated = True
    _log(verbose, '[POOL] full scan count=%d, work init count=%d' % (theta_s_full_deg.size, theta_s_work_init_deg.size))
    _log(verbose, '[POOL] alpha_core_used_deg=%.6f alpha_outer_start_deg=%.6f' % (alpha_core_used_deg, alpha_outer_start_deg))
    return ctx


def init_stage1_pool_context_from_package(package, *, tol_deg=1e-9, verbose=True):
    _log(verbose, '=== [A0CRA][BLOCK5] INIT STAGE1 POOL CONTEXT FROM MEMORY: START ===')
    ctx = _init_stage1_pool_context_from_data(
        _pool_package_to_static_arrays(package),
        _pool_package_summary(package),
        float(tol_deg),
        verbose=verbose,
    )
    _log(verbose, '=== [A0CRA][BLOCK5] INIT STAGE1 POOL CONTEXT FROM MEMORY: DONE ===')
    return ctx

def init_stage1_pool_context(cfg, *, verbose=True):
    _log(verbose, '=== [A0CRA][BLOCK5] INIT STAGE1 POOL CONTEXT: START ===')
    _log(verbose, '[POOL] npz=%s' % cfg.pool_npz_path)
    _log(verbose, '[POOL] json=%s' % cfg.pool_json_path)
    static_npz = _load_npz_dict(cfg.pool_npz_path)
    with open(cfg.pool_json_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    ctx = _init_stage1_pool_context_from_data(static_npz, summary, float(cfg.tol_deg), verbose=verbose)
    _log(verbose, '=== [A0CRA][BLOCK5] INIT STAGE1 POOL CONTEXT: DONE ===')
    return ctx



def get_current_work_theta_s_deg(pool_ctx: Stage1PoolContext) -> np.ndarray:
    return np.asarray(pool_ctx.current_work_theta_s_deg, dtype=np.float64).copy()


def get_full_theta_s_deg(pool_ctx: Stage1PoolContext) -> np.ndarray:
    return np.asarray(pool_ctx.theta_s_full_deg, dtype=np.float64).copy()


def build_current_work_eval_plan(pool_ctx: Stage1PoolContext, *, verbose: bool = True) -> Dict[str, Any]:
    work = get_current_work_theta_s_deg(pool_ctx)
    proto = dict(pool_ctx.tracking_protocol)
    plan = {
        'theta_s_work_deg': work,
        'need_dynamic_update': True,
        'tracking_protocol': proto,
        'work_size': int(work.size),
        'work_max_size': int(proto.get('work_max_size', work.size)),
    }
    _log(verbose, f"[POOL][PLAN] work_size={plan['work_size']} work_max_size={plan['work_max_size']} theta_s={work.tolist()}")
    return plan


def should_refresh_work_from_full(pool_ctx: Stage1PoolContext, iter_idx: int) -> bool:
    every = int(pool_ctx.tracking_protocol.get('update_work_every_n_iters', 0))
    if every <= 0:
        return False
    return ((int(iter_idx) + 1) % every) == 0


def update_work_scan_angles_from_full_feedback(
    pool_ctx: Stage1PoolContext,
    worst_theta_s_deg: np.ndarray,
    *,
    verbose: bool = True,
) -> None:
    anchors = np.asarray(pool_ctx.work_anchor_theta_s_deg, dtype=np.float64)
    current = np.asarray(pool_ctx.current_work_theta_s_deg, dtype=np.float64)
    add = np.asarray(worst_theta_s_deg, dtype=np.float64).reshape(-1)
    add = np.unique(add[np.isfinite(add)])
    add.sort()
    work_max_size = int(pool_ctx.tracking_protocol.get('work_max_size', max(current.size, anchors.size)))

    current_extras = [float(x) for x in current.tolist() if not np.any(np.isclose(anchors, x, atol=pool_ctx.tol_deg))]
    new_extras = []
    for x in add.tolist():
        if np.any(np.isclose(anchors, x, atol=pool_ctx.tol_deg)):
            continue
        if x not in new_extras:
            new_extras.append(float(x))
    merged_extras = []
    for x in new_extras + current_extras:
        if x not in merged_extras:
            merged_extras.append(float(x))

    max_extra = max(0, work_max_size - int(anchors.size))
    keep_extras = merged_extras[:max_extra]
    new_work = np.asarray(sorted(set(anchors.tolist() + keep_extras)), dtype=np.float64)
    pool_ctx.current_work_theta_s_deg = new_work
    _log(verbose, f'[POOL][WORK] updated from full feedback -> work theta_s={new_work.tolist()}')


def _get_canonical_static_points(pool_ctx: Stage1PoolContext, theta_s_deg: float) -> Dict[str, Dict[str, np.ndarray]]:
    key = _canonical_theta_deg(theta_s_deg, pool_ctx.tol_deg)
    if key not in pool_ctx.canonical_static_by_scan:
        raise KeyError(f'theta_s_deg={theta_s_deg} not found in canonical static pool')
    src = pool_ctx.canonical_static_by_scan[key]
    return {
        'core_1d': {k: np.asarray(v).copy() for k, v in src['core_1d'].items()},
        'buffer_2d': {k: np.asarray(v).copy() for k, v in src['buffer_2d'].items()},
        'outer_2d': {k: np.asarray(v).copy() for k, v in src['outer_2d'].items()},
    }


def _wrap_eta_deg(eta_deg: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(eta_deg, dtype=np.float64), 360.0).astype(np.float64)


def _local_to_global(theta_s_deg: float, phi_s_deg: float, alpha_deg: float, eta_deg: float) -> Tuple[float, float, float, float]:
    ths = math.radians(theta_s_deg)
    phs = math.radians(phi_s_deg)
    a = math.radians(alpha_deg)
    e = math.radians(eta_deg)
    s = np.array([math.sin(ths) * math.cos(phs), math.sin(ths) * math.sin(phs), math.cos(ths)], dtype=np.float64)
    e_theta = np.array([math.cos(ths) * math.cos(phs), math.cos(ths) * math.sin(phs), -math.sin(ths)], dtype=np.float64)
    e_phi = np.array([-math.sin(phs), math.cos(phs), 0.0], dtype=np.float64)
    r = math.cos(a) * s + math.sin(a) * (math.cos(e) * e_theta + math.sin(e) * e_phi)
    rx, ry, rz = float(r[0]), float(r[1]), float(np.clip(r[2], -1.0, 1.0))
    theta_deg = math.degrees(math.acos(rz))
    phi_deg = math.degrees(math.atan2(ry, rx))
    if phi_deg < 0.0:
        phi_deg += 360.0
    sin_theta = math.sin(math.radians(theta_deg))
    u = sin_theta * math.cos(math.radians(phi_deg))
    v = sin_theta * math.sin(math.radians(phi_deg))
    return theta_deg, phi_deg, u, v


def _dedupe_dynamic_points(alpha_deg: np.ndarray, eta_deg: np.ndarray, *, round_digits: int = 6) -> np.ndarray:
    seen = set()
    keep = []
    for i, (a, e) in enumerate(zip(alpha_deg.tolist(), eta_deg.tolist())):
        key = (round(float(a), round_digits), round(float(e) % 360.0, round_digits))
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def _build_dynamic_2d_point_dict(
    theta_s_deg: float,
    alpha_deg: np.ndarray,
    eta_deg: np.ndarray,
    seed_rank: np.ndarray,
    refine_round: np.ndarray,
) -> Dict[str, np.ndarray]:
    alpha_arr = np.asarray(alpha_deg, dtype=np.float64).reshape(-1)
    eta_arr = _wrap_eta_deg(np.asarray(eta_deg, dtype=np.float64).reshape(-1))
    seed_rank_arr = np.asarray(seed_rank, dtype=np.int32).reshape(-1)
    refine_round_arr = np.asarray(refine_round, dtype=np.int8).reshape(-1)
    if not (alpha_arr.size == eta_arr.size == seed_rank_arr.size == refine_round_arr.size):
        raise ValueError('dynamic point arrays must have the same length')
    keep = _dedupe_dynamic_points(alpha_arr, eta_arr)
    alpha_arr = alpha_arr[keep]
    eta_arr = eta_arr[keep]
    seed_rank_arr = seed_rank_arr[keep]
    refine_round_arr = refine_round_arr[keep]
    th_list = []
    ph_list = []
    u_list = []
    v_list = []
    for a, e in zip(alpha_arr.tolist(), eta_arr.tolist()):
        th, ph, u, v = _local_to_global(float(theta_s_deg), 0.0, float(a), float(e))
        th_list.append(th)
        ph_list.append(ph)
        u_list.append(u)
        v_list.append(v)
    n = int(alpha_arr.size)
    return {
        'alpha_deg': alpha_arr.astype(np.float64),
        'eta_deg': eta_arr.astype(np.float64),
        'theta_deg': np.asarray(th_list, dtype=np.float64),
        'phi_deg': np.asarray(ph_list, dtype=np.float64),
        'u': np.asarray(u_list, dtype=np.float64),
        'v': np.asarray(v_list, dtype=np.float64),
        'layer_code': np.full((n,), -1, dtype=np.int8),
        'is_side_2d': np.ones((n,), dtype=np.int8),
        'is_dynamic': np.ones((n,), dtype=np.int8),
        'seed_rank': seed_rank_arr.astype(np.int32),
        'refine_round': refine_round_arr.astype(np.int8),
    }


def _empty_dynamic_2d_points(theta_s_deg):
    del theta_s_deg
    return _build_dynamic_2d_point_dict(
        0.0,
        np.zeros((0,), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        np.zeros((0,), dtype=np.int32),
        np.zeros((0,), dtype=np.int8),
    )


def _subset_point_dict(points, mask):
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    n = int(np.asarray(points.get('alpha_deg', np.zeros((0,)))).size)
    if mask_arr.size != n:
        raise ValueError('point subset mask length mismatch')
    out = {}
    for key, value in points.items():
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.size == n:
            out[key] = arr[mask_arr].copy()
        else:
            out[key] = arr.copy()
    return out


def _merge_dynamic_point_dicts(first, second):
    if first is None or int(np.asarray(first.get('alpha_deg', np.zeros((0,)))).size) == 0:
        return {k: np.asarray(v).copy() for k, v in (second or {}).items()}
    if second is None or int(np.asarray(second.get('alpha_deg', np.zeros((0,)))).size) == 0:
        return {k: np.asarray(v).copy() for k, v in first.items()}
    merged = _concat_point_dicts(first, second)
    return merged

def _sanitize_buffer_dynamic_points(pool_ctx, theta_s_deg, points):
    alpha = np.asarray(points['alpha_deg'], dtype=np.float64).reshape(-1)
    eta = _wrap_eta_deg(points['eta_deg'])
    theta_global = np.asarray(points['theta_deg'], dtype=np.float64).reshape(-1)
    if not (alpha.size == eta.size == theta_global.size):
        raise ValueError('buffer dynamic coordinate length mismatch')
    alpha_core = float(pool_ctx.alpha_core_used_deg)
    mask = (
        (alpha > alpha_core + 1e-12)
        & (alpha <= float(pool_ctx.alpha_outer_start_deg) + 1e-12)
        & np.isfinite(theta_global)
        & (theta_global >= -1e-10)
        & (theta_global <= 90.0 + 1e-10)
    )
    if alpha.size == 0 or not np.any(mask):
        return _build_dynamic_2d_point_dict(
            float(theta_s_deg), np.zeros((0,)), np.zeros((0,)),
            np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int8),
        )
    return _build_dynamic_2d_point_dict(
        float(theta_s_deg), alpha[mask], eta[mask],
        np.asarray(points['seed_rank'], dtype=np.int32).reshape(-1)[mask],
        np.asarray(points['refine_round'], dtype=np.int8).reshape(-1)[mask],
    )



def _sanitize_outer_dynamic_points(pool_ctx, theta_s_deg, points):
    alpha = np.asarray(points['alpha_deg'], dtype=np.float64).reshape(-1)
    eta = _wrap_eta_deg(points['eta_deg'])
    theta_global = np.asarray(points['theta_deg'], dtype=np.float64).reshape(-1)
    if not (alpha.size == eta.size == theta_global.size):
        raise ValueError('outer dynamic coordinate length mismatch')
    alpha_min = float(pool_ctx.alpha_outer_start_deg)
    alpha_max = 90.0 + float(theta_s_deg)
    mask = (
        (alpha > alpha_min + 1e-12)
        & (alpha <= alpha_max + 1e-12)
        & np.isfinite(theta_global)
        & (theta_global >= -1e-10)
        & (theta_global <= 90.0 + 1e-10)
    )
    if alpha.size == 0 or not np.any(mask):
        return _build_dynamic_2d_point_dict(
            float(theta_s_deg), np.zeros((0,)), np.zeros((0,)),
            np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int8),
        )
    return _build_dynamic_2d_point_dict(
        float(theta_s_deg), alpha[mask], eta[mask],
        np.asarray(points['seed_rank'], dtype=np.int32).reshape(-1)[mask],
        np.asarray(points['refine_round'], dtype=np.int8).reshape(-1)[mask],
    )



def build_buffer_tracking_points_from_seeds(
    pool_ctx,
    theta_s_deg,
    seed_alpha_deg,
    seed_eta_deg,
    tracking_protocol,
    *,
    round_id=None,
    include_center=True,
    verbose=True,
):
    a_seed = np.asarray(seed_alpha_deg, dtype=np.float64).reshape(-1)
    e_seed = _wrap_eta_deg(np.asarray(seed_eta_deg, dtype=np.float64).reshape(-1))
    if a_seed.size != e_seed.size:
        raise ValueError('buffer seed alpha/eta length mismatch')
    rounds = (1, 2) if round_id is None else (int(round_id),)
    for rr in rounds:
        if rr not in (1, 2):
            raise ValueError('round_id must be 1, 2, or None')
    alpha_all = []
    eta_all = []
    seed_rank = []
    refine_round = []
    for i, (a, e) in enumerate(zip(a_seed.tolist(), e_seed.tolist())):
        if include_center:
            alpha_all.append(a); eta_all.append(e); seed_rank.append(i); refine_round.append(0)
        for rr in rounds:
            da = float(tracking_protocol['buf_delta_alpha_%d_deg' % rr])
            de = float(tracking_protocol['buf_delta_eta_%d_deg' % rr])
            alpha_all.extend([a + da, a - da, a, a])
            eta_all.extend([e, e, e + de, e - de])
            seed_rank.extend([i] * 4)
            refine_round.extend([rr] * 4)
    pts = _build_dynamic_2d_point_dict(
        float(theta_s_deg), np.asarray(alpha_all), np.asarray(eta_all),
        np.asarray(seed_rank), np.asarray(refine_round),
    )
    pts = _sanitize_buffer_dynamic_points(pool_ctx, float(theta_s_deg), pts)
    _log(verbose, '[POOL][TRACK][BUFFER] theta_s=%.3f round=%s seeds=%d dynamic_points=%d' % (
        float(theta_s_deg), str(round_id), a_seed.size, pts['alpha_deg'].size))
    return pts



def build_outer_tracking_points_from_seeds(
    pool_ctx,
    theta_s_deg,
    seed_alpha_deg,
    seed_eta_deg,
    tracking_protocol,
    *,
    round_id=None,
    include_center=True,
    verbose=True,
):
    a_seed = np.asarray(seed_alpha_deg, dtype=np.float64).reshape(-1)
    e_seed = _wrap_eta_deg(np.asarray(seed_eta_deg, dtype=np.float64).reshape(-1))
    if a_seed.size != e_seed.size:
        raise ValueError('outer seed alpha/eta length mismatch')
    rounds = (1, 2) if round_id is None else (int(round_id),)
    for rr in rounds:
        if rr not in (1, 2):
            raise ValueError('round_id must be 1, 2, or None')
    alpha_all = []
    eta_all = []
    seed_rank = []
    refine_round = []
    for i, (a, e) in enumerate(zip(a_seed.tolist(), e_seed.tolist())):
        if include_center:
            alpha_all.append(a); eta_all.append(e); seed_rank.append(i); refine_round.append(0)
        for rr in rounds:
            da = float(tracking_protocol['out_delta_alpha_%d_deg' % rr])
            de = float(tracking_protocol['out_delta_eta_%d_deg' % rr])
            alpha_all.extend([a + da, a - da, a, a])
            eta_all.extend([e, e, e + de, e - de])
            seed_rank.extend([i] * 4)
            refine_round.extend([rr] * 4)
    pts = _build_dynamic_2d_point_dict(
        float(theta_s_deg), np.asarray(alpha_all), np.asarray(eta_all),
        np.asarray(seed_rank), np.asarray(refine_round),
    )
    pts = _sanitize_outer_dynamic_points(pool_ctx, float(theta_s_deg), pts)
    _log(verbose, '[POOL][TRACK][OUTER] theta_s=%.3f round=%s seeds=%d dynamic_points=%d' % (
        float(theta_s_deg), str(round_id), a_seed.size, pts['alpha_deg'].size))
    return pts



def clear_work_dynamic_points_for_scan(pool_ctx: Stage1PoolContext, theta_s_deg: float, *, verbose: bool = True) -> None:
    key = _canonical_theta_deg(theta_s_deg, pool_ctx.tol_deg)
    pool_ctx.dynamic_buffer_work_by_scan.pop(key, None)
    pool_ctx.dynamic_outer_work_by_scan.pop(key, None)
    _log(verbose, f'[POOL][WORK][TRACK] cleared theta_s={key:.3f}')


def clear_full_dynamic_points_for_scan(pool_ctx: Stage1PoolContext, theta_s_deg: float, *, verbose: bool = True) -> None:
    key = _canonical_theta_deg(theta_s_deg, pool_ctx.tol_deg)
    pool_ctx.dynamic_buffer_full_by_scan.pop(key, None)
    pool_ctx.dynamic_outer_full_by_scan.pop(key, None)
    _log(verbose, f'[POOL][FULL][TRACK] cleared theta_s={key:.3f}')


def clear_all_work_dynamic_points(pool_ctx: Stage1PoolContext, *, verbose: bool = True) -> None:
    pool_ctx.dynamic_buffer_work_by_scan.clear()
    pool_ctx.dynamic_outer_work_by_scan.clear()
    _log(verbose, '[POOL][WORK][TRACK] cleared all dynamic points')


def clear_all_full_dynamic_points(pool_ctx: Stage1PoolContext, *, verbose: bool = True) -> None:
    pool_ctx.dynamic_buffer_full_by_scan.clear()
    pool_ctx.dynamic_outer_full_by_scan.clear()
    _log(verbose, '[POOL][FULL][TRACK] cleared all dynamic points')


def attach_dynamic_points_for_scan(
    pool_ctx,
    theta_s_deg,
    *,
    mode='work',
    dynamic_buffer_points=None,
    dynamic_outer_points=None,
    verbose=True,
):
    key = _canonical_theta_deg(theta_s_deg, pool_ctx.tol_deg)
    if mode == 'work':
        buf_store = pool_ctx.dynamic_buffer_work_by_scan
        out_store = pool_ctx.dynamic_outer_work_by_scan
    elif mode == 'full':
        buf_store = pool_ctx.dynamic_buffer_full_by_scan
        out_store = pool_ctx.dynamic_outer_full_by_scan
    else:
        raise ValueError("mode must be 'work' or 'full'")
    if dynamic_buffer_points is not None:
        buf_store[key] = _sanitize_buffer_dynamic_points(pool_ctx, key, dynamic_buffer_points)
    if dynamic_outer_points is not None:
        out_store[key] = _sanitize_outer_dynamic_points(pool_ctx, key, dynamic_outer_points)
    bsz = int(np.asarray(buf_store.get(key, {}).get('alpha_deg', np.zeros((0,)))).size) if key in buf_store else 0
    osz = int(np.asarray(out_store.get(key, {}).get('alpha_deg', np.zeros((0,)))).size) if key in out_store else 0
    _log(verbose, '[POOL][%s][TRACK] attached theta_s=%.3f buffer_dynamic=%d outer_dynamic=%d' % (mode.upper(), key, bsz, osz))



def rebuild_work_dynamic_points_for_scan(
    pool_ctx,
    theta_s_deg,
    *,
    buffer_seed_alpha_deg=None,
    buffer_seed_eta_deg=None,
    outer_seed_alpha_deg=None,
    outer_seed_eta_deg=None,
    round_id=None,
    clear_existing=True,
    verbose=True,
):
    key = _canonical_theta_deg(theta_s_deg, pool_ctx.tol_deg)
    if clear_existing:
        clear_work_dynamic_points_for_scan(pool_ctx, key, verbose=verbose)
    buf_pts = None
    out_pts = None
    if buffer_seed_alpha_deg is not None and buffer_seed_eta_deg is not None:
        buf_pts = build_buffer_tracking_points_from_seeds(
            pool_ctx, key, buffer_seed_alpha_deg, buffer_seed_eta_deg,
            pool_ctx.tracking_protocol, round_id=round_id, verbose=verbose,
        )
        if not clear_existing:
            buf_pts = _merge_dynamic_point_dicts(pool_ctx.dynamic_buffer_work_by_scan.get(key), buf_pts)
    if outer_seed_alpha_deg is not None and outer_seed_eta_deg is not None:
        out_pts = build_outer_tracking_points_from_seeds(
            pool_ctx, key, outer_seed_alpha_deg, outer_seed_eta_deg,
            pool_ctx.tracking_protocol, round_id=round_id, verbose=verbose,
        )
        if not clear_existing:
            out_pts = _merge_dynamic_point_dicts(pool_ctx.dynamic_outer_work_by_scan.get(key), out_pts)
    attach_dynamic_points_for_scan(
        pool_ctx, key, mode='work', dynamic_buffer_points=buf_pts,
        dynamic_outer_points=out_pts, verbose=verbose,
    )



def rebuild_full_dynamic_points_for_scan(
    pool_ctx,
    theta_s_deg,
    *,
    buffer_seed_alpha_deg=None,
    buffer_seed_eta_deg=None,
    outer_seed_alpha_deg=None,
    outer_seed_eta_deg=None,
    round_id=None,
    clear_existing=True,
    verbose=True,
):
    key = _canonical_theta_deg(theta_s_deg, pool_ctx.tol_deg)
    if clear_existing:
        clear_full_dynamic_points_for_scan(pool_ctx, key, verbose=verbose)
    buf_pts = None
    out_pts = None
    if buffer_seed_alpha_deg is not None and buffer_seed_eta_deg is not None:
        buf_pts = build_buffer_tracking_points_from_seeds(
            pool_ctx, key, buffer_seed_alpha_deg, buffer_seed_eta_deg,
            pool_ctx.tracking_protocol, round_id=round_id, verbose=verbose,
        )
        if not clear_existing:
            buf_pts = _merge_dynamic_point_dicts(pool_ctx.dynamic_buffer_full_by_scan.get(key), buf_pts)
    if outer_seed_alpha_deg is not None and outer_seed_eta_deg is not None:
        out_pts = build_outer_tracking_points_from_seeds(
            pool_ctx, key, outer_seed_alpha_deg, outer_seed_eta_deg,
            pool_ctx.tracking_protocol, round_id=round_id, verbose=verbose,
        )
        if not clear_existing:
            out_pts = _merge_dynamic_point_dicts(pool_ctx.dynamic_outer_full_by_scan.get(key), out_pts)
    attach_dynamic_points_for_scan(
        pool_ctx, key, mode='full', dynamic_buffer_points=buf_pts,
        dynamic_outer_points=out_pts, verbose=verbose,
    )



def _concat_point_dicts(static_points, dynamic_points):
    if dynamic_points is None or int(np.asarray(dynamic_points.get('alpha_deg', np.zeros((0,)))).size) == 0:
        return {k: np.asarray(v).copy() for k, v in static_points.items()}
    out = {}
    all_keys = set(static_points.keys()) | set(dynamic_points.keys())
    n_static = int(np.asarray(static_points.get('alpha_deg', np.zeros((0,)))).size)
    n_dynamic = int(np.asarray(dynamic_points.get('alpha_deg', np.zeros((0,)))).size)
    for key in sorted(all_keys):
        sv = static_points.get(key)
        dv = dynamic_points.get(key)
        if sv is None:
            darr = np.asarray(dv)
            fill = np.nan if np.issubdtype(darr.dtype, np.floating) else -1
            sv = np.full((n_static,), fill, dtype=darr.dtype)
        if dv is None:
            sarr = np.asarray(sv)
            fill = np.nan if np.issubdtype(sarr.dtype, np.floating) else -1
            dv = np.full((n_dynamic,), fill, dtype=sarr.dtype)
        out[key] = np.concatenate([np.asarray(sv), np.asarray(dv)], axis=0)
    if 'alpha_deg' in out and 'eta_deg' in out:
        keep = _dedupe_dynamic_points(
            np.asarray(out['alpha_deg'], dtype=np.float64),
            np.asarray(out['eta_deg'], dtype=np.float64),
        )
        total = n_static + n_dynamic
        for key, value in list(out.items()):
            arr = np.asarray(value)
            if arr.ndim == 1 and arr.size == total:
                out[key] = arr[keep].copy()
    return out



def get_stage1_points_for_scan(
    pool_ctx: Stage1PoolContext,
    theta_s_deg: float,
    *,
    mode: str = 'work',
    include_dynamic: bool = True,
    verbose: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    if mode not in ('work', 'full'):
        raise ValueError("mode must be 'work' or 'full'")
    key = _canonical_theta_deg(theta_s_deg, pool_ctx.tol_deg)
    static_pts = _get_canonical_static_points(pool_ctx, key)
    core = static_pts['core_1d']
    buffer_static = static_pts['buffer_2d']
    outer_static = static_pts['outer_2d']
    if include_dynamic:
        if mode == 'work':
            buffer_dynamic = pool_ctx.dynamic_buffer_work_by_scan.get(key, None)
            outer_dynamic = pool_ctx.dynamic_outer_work_by_scan.get(key, None)
        else:
            buffer_dynamic = pool_ctx.dynamic_buffer_full_by_scan.get(key, None)
            outer_dynamic = pool_ctx.dynamic_outer_full_by_scan.get(key, None)
        buffer_pts = _concat_point_dicts(buffer_static, buffer_dynamic)
        outer_pts = _concat_point_dicts(outer_static, outer_dynamic)
    else:
        buffer_pts = {k: np.asarray(v).copy() for k, v in buffer_static.items()}
        outer_pts = {k: np.asarray(v).copy() for k, v in outer_static.items()}
    _log(verbose, f'[POOL][MERGE][{mode.upper()}] theta_s={key:.3f} core={core["alpha_deg"].size} buffer={buffer_pts["alpha_deg"].size} outer={outer_pts["alpha_deg"].size}')
    return {'core_1d': core, 'buffer_2d': buffer_pts, 'outer_2d': outer_pts}


def get_all_work_scan_points(
    pool_ctx: Stage1PoolContext,
    *,
    include_dynamic: bool = True,
    verbose: bool = False,
) -> Dict[float, Dict[str, Dict[str, np.ndarray]]]:
    out: Dict[float, Dict[str, Dict[str, np.ndarray]]] = {}
    for theta_s_deg in get_current_work_theta_s_deg(pool_ctx).tolist():
        out[float(theta_s_deg)] = get_stage1_points_for_scan(pool_ctx, float(theta_s_deg), mode='work', include_dynamic=include_dynamic, verbose=verbose)
    return out


# =============================================================================
# Block 6: Mainlobe-mask generation
# =============================================================================


class Stage1MainlobeMaskConfig(object):
    __slots__ = (
        'domain_mode',
        'on_missing_scan',
    )

    def __init__(
        self,
        domain_mode='null_guard',
        on_missing_scan='error',
    ):
        self.domain_mode = domain_mode
        self.on_missing_scan = on_missing_scan



class Stage1CoreMask(object):
    __slots__ = (
        'theta_s_deg',
        'alpha_main_deg',
        'core_is_main_guard',
        'core_is_side_1d',
        'meta',
    )

    def __init__(
        self,
        theta_s_deg,
        alpha_main_deg,
        core_is_main_guard,
        core_is_side_1d,
        meta,
    ):
        self.theta_s_deg = theta_s_deg
        self.alpha_main_deg = alpha_main_deg
        self.core_is_main_guard = core_is_main_guard
        self.core_is_side_1d = core_is_side_1d
        self.meta = meta



def get_alpha_main_for_scan(
    mainlobe_state: MainlobeDomainState,
    theta_s_deg: float,
    cfg: Stage1MainlobeMaskConfig = Stage1MainlobeMaskConfig(),
) -> float:
    theta = float(theta_s_deg)
    scan = np.asarray(mainlobe_state.theta_s_deg, dtype=np.float64).reshape(-1)
    if scan.size == 0:
        raise ValueError('mainlobe_state.theta_s_deg is empty')
    vals = np.asarray(mainlobe_state.alpha_guard_null_deg if cfg.domain_mode == 'null_guard' else mainlobe_state.alpha_3db_deg, dtype=np.float64).reshape(-1)
    if vals.size != scan.size:
        raise ValueError('mainlobe_state arrays have inconsistent lengths')
    if cfg.domain_mode not in ('null_guard', '3db'):
        raise ValueError("domain_mode must be 'null_guard' or '3db'")
    idx = np.where(np.isclose(scan, theta, atol=1e-9))[0]
    if idx.size == 1:
        return float(vals[int(idx[0])])
    if cfg.on_missing_scan == 'nearest':
        j = int(np.argmin(np.abs(scan - theta)))
        return float(vals[j])
    raise KeyError(f'theta_s_deg={theta_s_deg} not found in mainlobe_state')


def build_core_mask_for_scan(
    core_points: Dict[str, np.ndarray],
    mainlobe_state: MainlobeDomainState,
    theta_s_deg: float,
    cfg: Stage1MainlobeMaskConfig = Stage1MainlobeMaskConfig(),
    *,
    verbose: bool = True,
) -> Stage1CoreMask:
    alpha_core = np.asarray(core_points.get('alpha_deg', []), dtype=np.float64).reshape(-1)
    if alpha_core.size == 0:
        raise ValueError('core_points[alpha_deg] is empty')
    alpha_main_deg = float(get_alpha_main_for_scan(mainlobe_state, theta_s_deg, cfg))
    if not np.isfinite(alpha_main_deg) or alpha_main_deg <= 0.0:
        raise ValueError(f'invalid alpha_main_deg={alpha_main_deg}')
    core_is_main_guard = (alpha_core <= alpha_main_deg + 1e-12)
    core_is_side_1d = ~core_is_main_guard
    if not np.any(core_is_main_guard):
        raise ValueError(f'theta_s={theta_s_deg}: core main-guard is empty')
    meta = {
        'domain_mode': str(cfg.domain_mode),
        'main_count': int(np.count_nonzero(core_is_main_guard)),
        'side_count': int(np.count_nonzero(core_is_side_1d)),
    }
    _log(verbose, f'[MASK] theta_s={float(theta_s_deg):.3f} mode={cfg.domain_mode} alpha_main={alpha_main_deg:.6f} deg')
    _log(verbose, f"[MASK] core main count={meta['main_count']} side count={meta['side_count']}")
    return Stage1CoreMask(
        theta_s_deg=float(theta_s_deg),
        alpha_main_deg=alpha_main_deg,
        core_is_main_guard=core_is_main_guard.astype(np.int8),
        core_is_side_1d=core_is_side_1d.astype(np.int8),
        meta=meta,
    )


def build_masked_points_for_scan(
    pool_ctx: Stage1PoolContext,
    mainlobe_state: MainlobeDomainState,
    theta_s_deg: float,
    *,
    mode: str = 'work',
    include_dynamic: bool = True,
    mask_cfg: Stage1MainlobeMaskConfig = Stage1MainlobeMaskConfig(),
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    pts = get_stage1_points_for_scan(pool_ctx, theta_s_deg, mode=mode, include_dynamic=include_dynamic, verbose=verbose)
    core_mask = build_core_mask_for_scan(pts['core_1d'], mainlobe_state, theta_s_deg, mask_cfg, verbose=verbose)
    core = {k: np.asarray(v).copy() for k, v in pts['core_1d'].items()}
    core['is_main_guard'] = np.asarray(core_mask.core_is_main_guard, dtype=np.int8)
    core['is_side_1d'] = np.asarray(core_mask.core_is_side_1d, dtype=np.int8)
    return {
        'core_1d': core,
        'buffer_2d': {k: np.asarray(v).copy() for k, v in pts['buffer_2d'].items()},
        'outer_2d': {k: np.asarray(v).copy() for k, v in pts['outer_2d'].items()},
    }


# =============================================================================
# Block 7: a0-policy integration
# =============================================================================


class Stage1A0PolicyConfig(object):
    __slots__ = (
        'build_mode',
        'warmup_all_scans',
        'policy_config',
    )

    def __init__(
        self,
        build_mode='full',
        warmup_all_scans=True,
        policy_config=_a0p.A0PolicyConfig(),
    ):
        self.build_mode = build_mode
        self.warmup_all_scans = warmup_all_scans
        self.policy_config = policy_config



class Stage1A0PolicyContext(object):
    __slots__ = (
        'policy',
        'build_mode',
        'warmed_scan_deg',
        'meta',
    )

    def __init__(
        self,
        policy,
        build_mode,
        warmed_scan_deg,
        meta,
    ):
        self.policy = policy
        self.build_mode = build_mode
        self.warmed_scan_deg = warmed_scan_deg
        self.meta = meta



def init_stage1_a0_policy(
    iep_ctx: Stage1IEPContext,
    pool_ctx: Stage1PoolContext,
    cfg: Stage1A0PolicyConfig = Stage1A0PolicyConfig(),
    *,
    verbose: bool = True,
) -> Stage1A0PolicyContext:
    _log(verbose, '=== [A0CRA][BLOCK7] INIT STAGE1 A0 POLICY: START ===')
    _log(verbose, f'[A0] build_mode={cfg.build_mode} warmup_all_scans={cfg.warmup_all_scans}')
    static_pool = _a0p.StaticPoolSnapshot(npz_data=pool_ctx.static_npz, summary=pool_ctx.summary)
    policy = _a0p.A0Policy(iep_ctx.a0_abs, static_pool, config=cfg.policy_config)
    policy.build_theta_cache(force_rebuild=False)
    global_mono = policy.test_global_monotone_decreasing(force_retest=False)
    warmed = np.asarray([], dtype=np.float64)
    if cfg.warmup_all_scans:
        warmed = static_pool.get_scan_angles(mode='full')
        policy.build_core_a0_max_cache(scan_angles_deg=warmed, mode='full', force_rebuild=False)
    meta = {
        'global_monotone_decreasing': bool(global_mono),
        'theta_cache_count': 0 if policy.theta_cache_deg is None else int(policy.theta_cache_deg.size),
        'canonical_static_mode': 'full',
    }
    _log(verbose, f'[A0] global_monotone_decreasing={global_mono}')
    _log(verbose, f'[A0] warmed scans={warmed.size}')
    _log(verbose, '=== [A0CRA][BLOCK7] INIT STAGE1 A0 POLICY: DONE ===')
    return Stage1A0PolicyContext(policy=policy, build_mode=str(cfg.build_mode), warmed_scan_deg=np.asarray(warmed, dtype=np.float64), meta=meta)


def ensure_core_a0_cache_for_scan(
    a0_ctx: Stage1A0PolicyContext,
    theta_s_deg: float,
    *,
    mode: str = 'full',
    verbose: bool = True,
) -> None:
    del mode
    key = float(theta_s_deg)
    if key not in a0_ctx.policy.core_a0_max_cache:
        _log(verbose, f'[A0] warm core_a0_max for theta_s={key:.3f} mode=full(canonical)')
        a0_ctx.policy.build_core_a0_max_cache(scan_angles_deg=np.asarray([key], dtype=np.float64), mode='full', force_rebuild=False)


def get_core_a0_max_for_scan(
    a0_ctx: Stage1A0PolicyContext,
    theta_s_deg: float,
    *,
    verbose: bool = True,
) -> np.ndarray:
    ensure_core_a0_cache_for_scan(a0_ctx, theta_s_deg, mode='full', verbose=verbose)
    out = np.asarray(a0_ctx.policy.get_core_a0_max(theta_s_deg), dtype=np.float64)
    _log(verbose, f'[A0] theta_s={float(theta_s_deg):.3f} core_a0_max n={out.size} min/mean/max={out.min():.6g}/{out.mean():.6g}/{out.max():.6g}')
    return out


def eval_pointwise_a0_for_2d_points(
    a0_ctx: Stage1A0PolicyContext,
    points_2d: Dict[str, np.ndarray],
    *,
    verbose: bool = True,
) -> np.ndarray:
    n = int(np.asarray(points_2d.get('theta_deg', np.zeros((0,)))).size)
    if n == 0:
        out = np.zeros((0,), dtype=np.float64)
        _log(verbose, '[A0] 2d pointwise n=0 (empty)')
        return out
    out = np.asarray(a0_ctx.policy.eval_pool_points_a0(points_2d), dtype=np.float64)
    if verbose:
        _log(verbose, f'[A0] 2d pointwise n={out.size} min/mean/max={out.min():.6g}/{out.mean():.6g}/{out.max():.6g}')
    return out


def build_a0_fields_for_scan(
    a0_ctx: Stage1A0PolicyContext,
    masked_points: Dict[str, Dict[str, np.ndarray]],
    theta_s_deg: float,
    *,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    core_a0_max = get_core_a0_max_for_scan(a0_ctx, theta_s_deg, verbose=verbose)
    buffer_a0_abs = eval_pointwise_a0_for_2d_points(a0_ctx, masked_points['buffer_2d'], verbose=verbose)
    outer_a0_abs = eval_pointwise_a0_for_2d_points(a0_ctx, masked_points['outer_2d'], verbose=verbose)
    return {
        'core_a0_max': core_a0_max.astype(np.float64),
        'buffer_a0_abs': buffer_a0_abs.astype(np.float64),
        'outer_a0_abs': outer_a0_abs.astype(np.float64),
    }


# =============================================================================
# Block 8: Bessel-alias field evaluation + precompute
# =============================================================================


class Stage1ElementKernelCache(object):
    __slots__ = (
        'ring_idx_of_elem',
        'elem_idx_in_ring',
        'psi_elem_rad',
        'cos_psi_elem',
        'sin_psi_elem',
        'include_center',
        'N_per_ring',
    )

    def __init__(
        self,
        ring_idx_of_elem,
        elem_idx_in_ring,
        psi_elem_rad,
        cos_psi_elem,
        sin_psi_elem,
        include_center,
        N_per_ring,
    ):
        self.ring_idx_of_elem = ring_idx_of_elem
        self.elem_idx_in_ring = elem_idx_in_ring
        self.psi_elem_rad = psi_elem_rad
        self.cos_psi_elem = cos_psi_elem
        self.sin_psi_elem = sin_psi_elem
        self.include_center = include_center
        self.N_per_ring = N_per_ring



class Stage1ElementRuntimeGeometry(object):
    __slots__ = (
        'x_elem',
        'y_elem',
        'r_elem',
        'ring_idx_of_elem',
        'beta',
        'r_over_lambda',
        'N_per_ring',
        'include_center',
    )

    def __init__(
        self,
        x_elem,
        y_elem,
        r_elem,
        ring_idx_of_elem,
        beta,
        r_over_lambda,
        N_per_ring,
        include_center,
    ):
        self.x_elem = x_elem
        self.y_elem = y_elem
        self.r_elem = r_elem
        self.ring_idx_of_elem = ring_idx_of_elem
        self.beta = beta
        self.r_over_lambda = r_over_lambda
        self.N_per_ring = N_per_ring
        self.include_center = include_center



class Stage1BesselPointKernel(object):
    __slots__ = (
        'theta_s_deg',
        'zone_name',
        'rho',
        'eta_rad',
    )

    def __init__(
        self,
        theta_s_deg,
        zone_name,
        rho,
        eta_rad,
    ):
        self.theta_s_deg = theta_s_deg
        self.zone_name = zone_name
        self.rho = rho
        self.eta_rad = eta_rad



class Stage1ScanFieldEval(object):
    __slots__ = (
        'theta_s_deg',
        'af_core',
        'af_buffer',
        'af_outer',
        'af_abs_core',
        'af_abs_buffer',
        'af_abs_outer',
        'eco_elem_core',
        'eco_elem_buffer',
        'eco_elem_outer',
        'eco_total_core',
        'eco_total_buffer',
        'eco_total_outer',
        'meta',
    )

    def __init__(
        self,
        theta_s_deg,
        af_core,
        af_buffer,
        af_outer,
        af_abs_core,
        af_abs_buffer,
        af_abs_outer,
        eco_elem_core,
        eco_elem_buffer,
        eco_elem_outer,
        eco_total_core,
        eco_total_buffer,
        eco_total_outer,
        meta,
    ):
        self.theta_s_deg = theta_s_deg
        self.af_core = af_core
        self.af_buffer = af_buffer
        self.af_outer = af_outer
        self.af_abs_core = af_abs_core
        self.af_abs_buffer = af_abs_buffer
        self.af_abs_outer = af_abs_outer
        self.eco_elem_core = eco_elem_core
        self.eco_elem_buffer = eco_elem_buffer
        self.eco_elem_outer = eco_elem_outer
        self.eco_total_core = eco_total_core
        self.eco_total_buffer = eco_total_buffer
        self.eco_total_outer = eco_total_outer
        self.meta = meta



class Stage1PrecomputeContext(object):
    __slots__ = (
        'elem_kernel',
        'elem_runtime',
        'alpha_main_by_scan',
        'core_alpha_static_by_scan',
        'core_a0_max_static_by_scan',
        'buffer_a0_static_by_scan',
        'outer_a0_static_by_scan',
        'eco_elem_static_by_scan_zone',
        'af_kernel_static_by_scan_zone',
        'buffer_a0_dynamic_work_by_scan',
        'outer_a0_dynamic_work_by_scan',
        'buffer_a0_dynamic_full_by_scan',
        'outer_a0_dynamic_full_by_scan',
        'eco_elem_dynamic_work_by_scan_zone',
        'eco_elem_dynamic_full_by_scan_zone',
        'af_kernel_dynamic_work_by_scan_zone',
        'af_kernel_dynamic_full_by_scan_zone',
    )

    def __init__(
        self,
        elem_kernel,
        elem_runtime,
        alpha_main_by_scan,
        core_alpha_static_by_scan,
        core_a0_max_static_by_scan,
        buffer_a0_static_by_scan,
        outer_a0_static_by_scan,
        eco_elem_static_by_scan_zone,
        af_kernel_static_by_scan_zone,
        buffer_a0_dynamic_work_by_scan,
        outer_a0_dynamic_work_by_scan,
        buffer_a0_dynamic_full_by_scan,
        outer_a0_dynamic_full_by_scan,
        eco_elem_dynamic_work_by_scan_zone,
        eco_elem_dynamic_full_by_scan_zone,
        af_kernel_dynamic_work_by_scan_zone,
        af_kernel_dynamic_full_by_scan_zone,
    ):
        self.elem_kernel = elem_kernel
        self.elem_runtime = elem_runtime
        self.alpha_main_by_scan = alpha_main_by_scan
        self.core_alpha_static_by_scan = core_alpha_static_by_scan
        self.core_a0_max_static_by_scan = core_a0_max_static_by_scan
        self.buffer_a0_static_by_scan = buffer_a0_static_by_scan
        self.outer_a0_static_by_scan = outer_a0_static_by_scan
        self.eco_elem_static_by_scan_zone = eco_elem_static_by_scan_zone
        self.af_kernel_static_by_scan_zone = af_kernel_static_by_scan_zone
        self.buffer_a0_dynamic_work_by_scan = buffer_a0_dynamic_work_by_scan
        self.outer_a0_dynamic_work_by_scan = outer_a0_dynamic_work_by_scan
        self.buffer_a0_dynamic_full_by_scan = buffer_a0_dynamic_full_by_scan
        self.outer_a0_dynamic_full_by_scan = outer_a0_dynamic_full_by_scan
        self.eco_elem_dynamic_work_by_scan_zone = eco_elem_dynamic_work_by_scan_zone
        self.eco_elem_dynamic_full_by_scan_zone = eco_elem_dynamic_full_by_scan_zone
        self.af_kernel_dynamic_work_by_scan_zone = af_kernel_dynamic_work_by_scan_zone
        self.af_kernel_dynamic_full_by_scan_zone = af_kernel_dynamic_full_by_scan_zone



def build_element_kernel_cache(elem_geom: ElementGeometry, *, verbose: bool = True) -> Stage1ElementKernelCache:
    ring_idx = np.asarray(elem_geom.ring_idx, dtype=np.int64).reshape(-1)
    elem_idx = np.asarray(elem_geom.elem_idx_in_ring, dtype=np.int64).reshape(-1)
    N_per_ring = np.asarray(elem_geom.N_per_ring, dtype=np.int64).reshape(-1)
    psi = np.zeros_like(ring_idx, dtype=np.float64)
    noncenter = ring_idx >= 0
    if np.any(noncenter):
        rk = ring_idx[noncenter]
        psi[noncenter] = 2.0 * np.pi * elem_idx[noncenter] / N_per_ring[rk]
    cache = Stage1ElementKernelCache(
        ring_idx_of_elem=ring_idx,
        elem_idx_in_ring=elem_idx,
        psi_elem_rad=psi,
        cos_psi_elem=np.cos(psi),
        sin_psi_elem=np.sin(psi),
        include_center=bool(elem_geom.include_center),
        N_per_ring=N_per_ring.astype(np.int64),
    )
    _log(verbose, '=== [A0CRA][BLOCK8] BUILD ELEMENT KERNEL CACHE: DONE ===')
    _log(verbose, f'[AFKERNEL] total_elements={ring_idx.size} include_center={elem_geom.include_center}')
    return cache


def build_element_runtime_geometry(
    radius_init: RadiusInitResult,
    elem_kernel: Stage1ElementKernelCache,
    *,
    beta: float = 2.0 * np.pi,
    verbose: bool = True,
) -> Stage1ElementRuntimeGeometry:
    r_ring = np.asarray(radius_init.r_over_lambda, dtype=np.float64).reshape(-1)
    r_elem = np.zeros_like(elem_kernel.psi_elem_rad, dtype=np.float64)
    noncenter = elem_kernel.ring_idx_of_elem >= 0
    if np.any(noncenter):
        r_elem[noncenter] = r_ring[elem_kernel.ring_idx_of_elem[noncenter]]
    x_elem = r_elem * elem_kernel.cos_psi_elem
    y_elem = r_elem * elem_kernel.sin_psi_elem
    geom = Stage1ElementRuntimeGeometry(
        x_elem=x_elem.astype(np.float64),
        y_elem=y_elem.astype(np.float64),
        r_elem=r_elem.astype(np.float64),
        ring_idx_of_elem=np.asarray(elem_kernel.ring_idx_of_elem, dtype=np.int64),
        beta=float(beta),
        r_over_lambda=r_ring.astype(np.float64),
        N_per_ring=np.asarray(elem_kernel.N_per_ring, dtype=np.int64).reshape(-1),
        include_center=bool(elem_kernel.include_center),
    )
    _log(verbose, '=== [A0CRA][BLOCK8] BUILD ELEMENT RUNTIME GEOMETRY: DONE ===')
    _log(verbose, f'[AFRUNTIME] r_elem min/max={geom.r_elem.min():.6f}/{geom.r_elem.max():.6f}')
    return geom


def _jv_int(order: int, x: np.ndarray) -> np.ndarray:
    order = int(order)
    if order >= 0:
        return _spsp.jv(order, x)
    m = -order
    return ((-1) ** m) * _spsp.jv(m, x)


def _jvp_int(order: int, x: np.ndarray) -> np.ndarray:
    order = int(order)
    if order >= 0:
        return _spsp.jvp(order, x, 1)
    m = -order
    return ((-1) ** m) * _spsp.jvp(m, x, 1)


def eval_af_core_j0_collapsed(
    rho: np.ndarray,
    r_over_lambda: np.ndarray,
    N_per_ring: np.ndarray,
    *,
    beta: float = 2.0 * np.pi,
    include_center: bool = True,
) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    r = np.asarray(r_over_lambda, dtype=np.float64).reshape(-1)
    N = np.asarray(N_per_ring, dtype=np.float64).reshape(-1)
    if r.size != N.size:
        raise ValueError('r_over_lambda and N_per_ring must have same length')
    if rho.size == 0:
        return np.zeros((0,), dtype=np.complex128)
    X = float(beta) * rho[:, None] * r[None, :]
    af_real = np.sum(N[None, :] * _spsp.jv(0, X), axis=1)
    if include_center:
        af_real = af_real + 1.0
    return af_real.astype(np.complex128)


def eval_af_core_j0_and_jacobian(
    rho: np.ndarray,
    r_over_lambda: np.ndarray,
    N_per_ring: np.ndarray,
    *,
    beta: float = 2.0 * np.pi,
    include_center: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    r = np.asarray(r_over_lambda, dtype=np.float64).reshape(-1)
    N = np.asarray(N_per_ring, dtype=np.float64).reshape(-1)
    if r.size != N.size:
        raise ValueError('r_over_lambda and N_per_ring must have same length')
    M = int(rho.size)
    K = int(r.size)
    if M == 0:
        return np.zeros((0,), dtype=np.complex128), np.zeros((0, K), dtype=np.complex128)
    X = float(beta) * rho[:, None] * r[None, :]
    af_real = np.sum(N[None, :] * _spsp.jv(0, X), axis=1)
    if include_center:
        af_real = af_real + 1.0
    J = N[None, :] * float(beta) * rho[:, None] * _spsp.jvp(0, X, 1)
    return af_real.astype(np.complex128), J.astype(np.complex128)


def eval_af_bessel_alias_L1(
    rho: np.ndarray,
    eta_rad: np.ndarray,
    r_over_lambda: np.ndarray,
    N_per_ring: np.ndarray,
    *,
    beta: float = 2.0 * np.pi,
    include_center: bool = True,
) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    eta = np.asarray(eta_rad, dtype=np.float64).reshape(-1)
    r = np.asarray(r_over_lambda, dtype=np.float64).reshape(-1)
    N = np.asarray(N_per_ring, dtype=np.int64).reshape(-1)
    if rho.size != eta.size:
        raise ValueError('rho and eta_rad must have same length')
    if r.size != N.size:
        raise ValueError('r_over_lambda and N_per_ring must have same length')
    M = int(rho.size)
    if M == 0:
        return np.zeros((0,), dtype=np.complex128)
    af = np.ones((M,), dtype=np.complex128) if include_center else np.zeros((M,), dtype=np.complex128)
    for rk, Nk in zip(r.tolist(), N.tolist()):
        Nk_i = int(Nk)
        x = float(beta) * float(rk) * rho
        for ell in (-1, 0, 1):
            order = int(ell * Nk_i)
            coeff = (1j) ** order
            phase = np.exp(-1j * order * eta)
            af += float(Nk_i) * coeff * _jv_int(order, x) * phase
    return af.astype(np.complex128)


def eval_af_bessel_alias_L1_and_jacobian(
    rho: np.ndarray,
    eta_rad: np.ndarray,
    r_over_lambda: np.ndarray,
    N_per_ring: np.ndarray,
    *,
    beta: float = 2.0 * np.pi,
    include_center: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    eta = np.asarray(eta_rad, dtype=np.float64).reshape(-1)
    r = np.asarray(r_over_lambda, dtype=np.float64).reshape(-1)
    N = np.asarray(N_per_ring, dtype=np.int64).reshape(-1)
    if rho.size != eta.size:
        raise ValueError('rho and eta_rad must have same length')
    if r.size != N.size:
        raise ValueError('r_over_lambda and N_per_ring must have same length')
    M = int(rho.size)
    K = int(r.size)
    if M == 0:
        return np.zeros((0,), dtype=np.complex128), np.zeros((0, K), dtype=np.complex128)
    af = np.ones((M,), dtype=np.complex128) if include_center else np.zeros((M,), dtype=np.complex128)
    J = np.zeros((M, K), dtype=np.complex128)
    for k, (rk, Nk) in enumerate(zip(r.tolist(), N.tolist())):
        Nk_i = int(Nk)
        x = float(beta) * float(rk) * rho
        for ell in (-1, 0, 1):
            order = int(ell * Nk_i)
            coeff = (1j) ** order
            phase = np.exp(-1j * order * eta)
            B = _jv_int(order, x)
            Bp = _jvp_int(order, x)
            af += float(Nk_i) * coeff * B * phase
            J[:, k] += float(Nk_i) * float(beta) * rho * coeff * Bp * phase
    return af.astype(np.complex128), J.astype(np.complex128)


def build_af_kernel_for_points(
    theta_s_deg: float,
    points: Dict[str, np.ndarray],
    elem_kernel: Stage1ElementKernelCache,
    *,
    zone_name: str,
) -> Stage1BesselPointKernel:
    del elem_kernel
    z = str(zone_name)
    if z == 'core_1d':
        rho = np.asarray(points.get('rho', np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
        eta = np.zeros_like(rho, dtype=np.float64)
    elif z in ('buffer_2d', 'outer_2d'):
        u = np.asarray(points.get('u', np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
        v = np.asarray(points.get('v', np.asarray([], dtype=np.float64)), dtype=np.float64).reshape(-1)
        if u.size != v.size:
            raise ValueError(f'{z}: u and v must have same length')
        u_s = math.sin(math.radians(float(theta_s_deg)))
        du = u - u_s
        dv = v
        rho = np.sqrt(du * du + dv * dv)
        eta = np.arctan2(dv, du)
    else:
        raise ValueError(f'unknown zone_name={zone_name}')
    return Stage1BesselPointKernel(
        theta_s_deg=float(theta_s_deg),
        zone_name=z,
        rho=rho.astype(np.float64),
        eta_rad=eta.astype(np.float64),
    )


def _concat_bessel_point_kernels(
    a: Stage1BesselPointKernel,
    b: Optional[Stage1BesselPointKernel],
) -> Stage1BesselPointKernel:
    if b is None:
        return a
    if str(a.zone_name) != str(b.zone_name):
        raise ValueError('Bessel kernel zone mismatch')
    if abs(float(a.theta_s_deg) - float(b.theta_s_deg)) > 1e-9:
        raise ValueError('Bessel kernel theta_s mismatch')
    return Stage1BesselPointKernel(
        theta_s_deg=float(a.theta_s_deg),
        zone_name=str(a.zone_name),
        rho=np.concatenate([np.asarray(a.rho, dtype=np.float64), np.asarray(b.rho, dtype=np.float64)]),
        eta_rad=np.concatenate([np.asarray(a.eta_rad, dtype=np.float64), np.asarray(b.eta_rad, dtype=np.float64)]),
    )


def _bessel_kernel_len(k: Stage1BesselPointKernel) -> int:
    return int(np.asarray(k.rho).size)


def eval_af_from_kernel(
    af_kernel: Stage1BesselPointKernel,
    elem_runtime: Stage1ElementRuntimeGeometry,
) -> np.ndarray:
    r = np.asarray(elem_runtime.r_over_lambda, dtype=np.float64).reshape(-1)
    N = np.asarray(elem_runtime.N_per_ring, dtype=np.int64).reshape(-1)
    beta = float(elem_runtime.beta)
    include_center = bool(elem_runtime.include_center)
    if af_kernel.zone_name == 'core_1d':
        return eval_af_core_j0_collapsed(af_kernel.rho, r, N, beta=beta, include_center=include_center)
    if af_kernel.zone_name in ('buffer_2d', 'outer_2d'):
        return eval_af_bessel_alias_L1(af_kernel.rho, af_kernel.eta_rad, r, N, beta=beta, include_center=include_center)
    raise ValueError(f'unknown AF kernel zone_name={af_kernel.zone_name}')


def eval_element_co_on_points(
    iep_ctx: Stage1IEPContext,
    points: Dict[str, np.ndarray],
) -> np.ndarray:
    if 'theta_deg' not in points or 'phi_deg' not in points:
        raise ValueError('points must contain theta_deg and phi_deg for eco evaluation')
    th_rad = np.deg2rad(np.asarray(points['theta_deg'], dtype=np.float64).reshape(-1))
    ph_rad = np.deg2rad(np.asarray(points['phi_deg'], dtype=np.float64).reshape(-1))
    eco = np.asarray(iep_ctx.eco_eval(th_rad, ph_rad))
    if eco.shape != th_rad.shape:
        eco = np.asarray(np.broadcast_to(eco, th_rad.shape))
    return np.asarray(eco, dtype=np.complex128)


def _empty_complex() -> np.ndarray:
    return np.asarray([], dtype=np.complex128)


def _build_alpha_main_cache(mainlobe_state: MainlobeDomainState, domain_mode: str) -> Dict[float, float]:
    scan = np.asarray(mainlobe_state.theta_s_deg, dtype=np.float64).reshape(-1)
    vals = np.asarray(mainlobe_state.alpha_guard_null_deg if domain_mode == 'null_guard' else mainlobe_state.alpha_3db_deg, dtype=np.float64).reshape(-1)
    return {float(k): float(v) for k, v in zip(scan.tolist(), vals.tolist())}


def init_stage1_precompute_context(
    iep_ctx: Stage1IEPContext,
    pool_ctx: Stage1PoolContext,
    mainlobe_state: MainlobeDomainState,
    a0_ctx: Stage1A0PolicyContext,
    elem_geom: ElementGeometry,
    *,
    mask_domain_mode: str = 'null_guard',
    verbose: bool = True,
) -> Stage1PrecomputeContext:
    _log(verbose, '=== [A0CRA][BLOCK8] INIT STAGE1 PRECOMPUTE CONTEXT: START ===')
    elem_kernel = build_element_kernel_cache(elem_geom, verbose=verbose)
    alpha_main_by_scan = _build_alpha_main_cache(mainlobe_state, mask_domain_mode)
    core_alpha_static_by_scan: Dict[float, np.ndarray] = {}
    core_a0_max_static_by_scan: Dict[float, np.ndarray] = {}
    buffer_a0_static_by_scan: Dict[float, np.ndarray] = {}
    outer_a0_static_by_scan: Dict[float, np.ndarray] = {}
    eco_elem_static_by_scan_zone: Dict[Tuple[float, str], np.ndarray] = {}
    af_kernel_static_by_scan_zone: Dict[Tuple[float, str], Any] = {}

    for theta_s_deg in pool_ctx.theta_s_full_deg.tolist():
        theta_key = float(theta_s_deg)
        static_pts = _get_canonical_static_points(pool_ctx, theta_key)
        masked_pts = build_masked_points_for_scan(
            pool_ctx,
            mainlobe_state,
            theta_key,
            mode='full',
            include_dynamic=False,
            mask_cfg=Stage1MainlobeMaskConfig(domain_mode=mask_domain_mode),
            verbose=False,
        )
        core_alpha_static_by_scan[theta_key] = np.asarray(masked_pts['core_1d']['alpha_deg'], dtype=np.float64)
        core_a0_max_static_by_scan[theta_key] = get_core_a0_max_for_scan(a0_ctx, theta_key, verbose=False)
        buffer_a0_static_by_scan[theta_key] = eval_pointwise_a0_for_2d_points(a0_ctx, masked_pts['buffer_2d'], verbose=False)
        outer_a0_static_by_scan[theta_key] = eval_pointwise_a0_for_2d_points(a0_ctx, masked_pts['outer_2d'], verbose=False)
        eco_elem_static_by_scan_zone[(theta_key, 'buffer_2d')] = eval_element_co_on_points(iep_ctx, static_pts['buffer_2d']) if static_pts['buffer_2d']['alpha_deg'].size else _empty_complex()
        eco_elem_static_by_scan_zone[(theta_key, 'outer_2d')] = eval_element_co_on_points(iep_ctx, static_pts['outer_2d']) if static_pts['outer_2d']['alpha_deg'].size else _empty_complex()
        af_kernel_static_by_scan_zone[(theta_key, 'core_1d')] = build_af_kernel_for_points(theta_key, masked_pts['core_1d'], elem_kernel, zone_name='core_1d')
        af_kernel_static_by_scan_zone[(theta_key, 'buffer_2d')] = build_af_kernel_for_points(theta_key, static_pts['buffer_2d'], elem_kernel, zone_name='buffer_2d')
        af_kernel_static_by_scan_zone[(theta_key, 'outer_2d')] = build_af_kernel_for_points(theta_key, static_pts['outer_2d'], elem_kernel, zone_name='outer_2d')

    pre = Stage1PrecomputeContext(
        elem_kernel=elem_kernel,
        elem_runtime=None,
        alpha_main_by_scan=alpha_main_by_scan,
        core_alpha_static_by_scan=core_alpha_static_by_scan,
        core_a0_max_static_by_scan=core_a0_max_static_by_scan,
        buffer_a0_static_by_scan=buffer_a0_static_by_scan,
        outer_a0_static_by_scan=outer_a0_static_by_scan,
        eco_elem_static_by_scan_zone=eco_elem_static_by_scan_zone,
        af_kernel_static_by_scan_zone=af_kernel_static_by_scan_zone,
        buffer_a0_dynamic_work_by_scan={},
        outer_a0_dynamic_work_by_scan={},
        buffer_a0_dynamic_full_by_scan={},
        outer_a0_dynamic_full_by_scan={},
        eco_elem_dynamic_work_by_scan_zone={},
        eco_elem_dynamic_full_by_scan_zone={},
        af_kernel_dynamic_work_by_scan_zone={},
        af_kernel_dynamic_full_by_scan_zone={},
    )
    _log(verbose, f'[PRECOMP] static scans cached={len(core_alpha_static_by_scan)}')
    _log(verbose, '=== [A0CRA][BLOCK8] INIT STAGE1 PRECOMPUTE CONTEXT: DONE ===')
    return pre


def update_element_runtime_geometry(
    pre_ctx: Stage1PrecomputeContext,
    radius_init: RadiusInitResult,
    *,
    beta: float = 2.0 * np.pi,
    verbose: bool = True,
) -> None:
    pre_ctx.elem_runtime = build_element_runtime_geometry(radius_init, pre_ctx.elem_kernel, beta=beta, verbose=verbose)


def _dynamic_a0_store(pre_ctx: Stage1PrecomputeContext, mode: str, zone_name: str) -> Dict[float, np.ndarray]:
    if mode == 'work' and zone_name == 'buffer_2d':
        return pre_ctx.buffer_a0_dynamic_work_by_scan
    if mode == 'work' and zone_name == 'outer_2d':
        return pre_ctx.outer_a0_dynamic_work_by_scan
    if mode == 'full' and zone_name == 'buffer_2d':
        return pre_ctx.buffer_a0_dynamic_full_by_scan
    if mode == 'full' and zone_name == 'outer_2d':
        return pre_ctx.outer_a0_dynamic_full_by_scan
    raise ValueError('invalid mode/zone_name for dynamic a0 store')


def _dynamic_eco_store(pre_ctx: Stage1PrecomputeContext, mode: str) -> Dict[Tuple[float, str], np.ndarray]:
    return pre_ctx.eco_elem_dynamic_work_by_scan_zone if mode == 'work' else pre_ctx.eco_elem_dynamic_full_by_scan_zone


def _dynamic_afk_store(pre_ctx: Stage1PrecomputeContext, mode: str) -> Dict[Tuple[float, str], Any]:
    return pre_ctx.af_kernel_dynamic_work_by_scan_zone if mode == 'work' else pre_ctx.af_kernel_dynamic_full_by_scan_zone


def clear_dynamic_precompute_for_scan(
    pre_ctx: Stage1PrecomputeContext,
    theta_s_deg: float,
    *,
    mode: str = 'work',
) -> None:
    key = float(theta_s_deg)
    for zone_name in ('buffer_2d', 'outer_2d'):
        _dynamic_a0_store(pre_ctx, mode, zone_name).pop(key, None)
        _dynamic_eco_store(pre_ctx, mode).pop((key, zone_name), None)
        _dynamic_afk_store(pre_ctx, mode).pop((key, zone_name), None)


def refresh_dynamic_precompute_for_scan(
    pre_ctx: Stage1PrecomputeContext,
    iep_ctx: Stage1IEPContext,
    a0_ctx: Stage1A0PolicyContext,
    pool_ctx: Stage1PoolContext,
    theta_s_deg: float,
    *,
    mode: str = 'work',
    verbose: bool = True,
) -> None:
    key = float(theta_s_deg)
    clear_dynamic_precompute_for_scan(pre_ctx, key, mode=mode)
    pts = get_stage1_points_for_scan(pool_ctx, key, mode=mode, include_dynamic=True, verbose=False)
    static_pts = _get_canonical_static_points(pool_ctx, key)
    n_buf_static = int(static_pts['buffer_2d']['alpha_deg'].size)
    n_out_static = int(static_pts['outer_2d']['alpha_deg'].size)
    dyn_buf = {k: np.asarray(v)[n_buf_static:].copy() for k, v in pts['buffer_2d'].items()}
    dyn_out = {k: np.asarray(v)[n_out_static:].copy() for k, v in pts['outer_2d'].items()}
    if dyn_buf['alpha_deg'].size:
        _dynamic_a0_store(pre_ctx, mode, 'buffer_2d')[key] = eval_pointwise_a0_for_2d_points(a0_ctx, dyn_buf, verbose=False)
        _dynamic_eco_store(pre_ctx, mode)[(key, 'buffer_2d')] = eval_element_co_on_points(iep_ctx, dyn_buf)
        _dynamic_afk_store(pre_ctx, mode)[(key, 'buffer_2d')] = build_af_kernel_for_points(key, dyn_buf, pre_ctx.elem_kernel, zone_name='buffer_2d')
    if dyn_out['alpha_deg'].size:
        _dynamic_a0_store(pre_ctx, mode, 'outer_2d')[key] = eval_pointwise_a0_for_2d_points(a0_ctx, dyn_out, verbose=False)
        _dynamic_eco_store(pre_ctx, mode)[(key, 'outer_2d')] = eval_element_co_on_points(iep_ctx, dyn_out)
        _dynamic_afk_store(pre_ctx, mode)[(key, 'outer_2d')] = build_af_kernel_for_points(key, dyn_out, pre_ctx.elem_kernel, zone_name='outer_2d')
    _log(verbose, f"[PRECOMP][{mode.upper()}] theta_s={key:.3f} dynamic buffer={dyn_buf['alpha_deg'].size} outer={dyn_out['alpha_deg'].size}")


def _concat_maybe(a: np.ndarray, b: Optional[np.ndarray], *, dtype=None) -> np.ndarray:
    a_arr = np.asarray(a if a is not None else np.zeros((0,)), dtype=dtype)
    if b is None:
        return a_arr.copy()
    b_arr = np.asarray(b, dtype=dtype)
    if a_arr.size == 0:
        return b_arr.copy()
    if b_arr.size == 0:
        return a_arr.copy()
    return np.concatenate([a_arr, b_arr], axis=0)


def build_a0_fields_for_current_points(
    a0_ctx: Stage1A0PolicyContext,
    pre_ctx: Stage1PrecomputeContext,
    points: Dict[str, Dict[str, np.ndarray]],
    theta_s_deg: float,
    *,
    mode: str,
) -> Dict[str, np.ndarray]:
    del a0_ctx
    key = float(theta_s_deg)
    core = np.asarray(pre_ctx.core_a0_max_static_by_scan[key], dtype=np.float64)
    buf = _concat_maybe(
        np.asarray(pre_ctx.buffer_a0_static_by_scan[key], dtype=np.float64),
        _dynamic_a0_store(pre_ctx, mode, 'buffer_2d').get(key, None),
        dtype=np.float64,
    )
    out = _concat_maybe(
        np.asarray(pre_ctx.outer_a0_static_by_scan[key], dtype=np.float64),
        _dynamic_a0_store(pre_ctx, mode, 'outer_2d').get(key, None),
        dtype=np.float64,
    )
    n_core = int(np.asarray(points['core_1d']['alpha_deg']).size)
    n_buf = int(np.asarray(points['buffer_2d']['alpha_deg']).size)
    n_out = int(np.asarray(points['outer_2d']['alpha_deg']).size)
    if core.size != n_core:
        raise AssertionError(f'core a0 length mismatch: {core.size} vs {n_core}')
    if buf.size != n_buf:
        raise AssertionError(f'buffer a0 length mismatch: {buf.size} vs {n_buf}')
    if out.size != n_out:
        raise AssertionError(f'outer a0 length mismatch: {out.size} vs {n_out}')
    return {'core_a0_max': core, 'buffer_a0_abs': buf, 'outer_a0_abs': out}


def _get_af_kernel_for_scan_zone(
    pre_ctx: Stage1PrecomputeContext,
    theta_s_deg: float,
    zone_name: str,
    *,
    mode: str,
) -> Stage1BesselPointKernel:
    key = float(theta_s_deg)
    static = pre_ctx.af_kernel_static_by_scan_zone[(key, zone_name)]
    if zone_name == 'core_1d':
        return static
    dyn = _dynamic_afk_store(pre_ctx, mode).get((key, zone_name), None)
    return _concat_bessel_point_kernels(static, dyn)


def _get_eco_elem_for_scan_zone(
    pre_ctx: Stage1PrecomputeContext,
    theta_s_deg: float,
    zone_name: str,
    *,
    mode: str,
) -> Optional[np.ndarray]:
    key = float(theta_s_deg)
    if zone_name == 'core_1d':
        return None
    static = pre_ctx.eco_elem_static_by_scan_zone[(key, zone_name)]
    dyn = _dynamic_eco_store(pre_ctx, mode).get((key, zone_name), None)
    return _concat_maybe(np.asarray(static, dtype=np.complex128), dyn, dtype=np.complex128)


def eval_fields_for_current_points(
    iep_ctx: Stage1IEPContext,
    theta_s_deg: float,
    points: Dict[str, Dict[str, np.ndarray]],
    pre_ctx: Stage1PrecomputeContext,
    *,
    mode: str = 'work',
    need_co_total: bool = False,
    need_element_co: bool = False,
    verbose: bool = True,
) -> Stage1ScanFieldEval:
    del iep_ctx
    if pre_ctx.elem_runtime is None:
        raise ValueError('pre_ctx.elem_runtime is None; call update_element_runtime_geometry first')
    key = float(theta_s_deg)
    k_core = _get_af_kernel_for_scan_zone(pre_ctx, key, 'core_1d', mode=mode)
    k_buf = _get_af_kernel_for_scan_zone(pre_ctx, key, 'buffer_2d', mode=mode)
    k_out = _get_af_kernel_for_scan_zone(pre_ctx, key, 'outer_2d', mode=mode)

    if _bessel_kernel_len(k_core) != int(np.asarray(points['core_1d']['alpha_deg']).size):
        raise AssertionError('core AF kernel length mismatch with current points')
    if _bessel_kernel_len(k_buf) != int(np.asarray(points['buffer_2d']['alpha_deg']).size):
        raise AssertionError('buffer AF kernel length mismatch with current points')
    if _bessel_kernel_len(k_out) != int(np.asarray(points['outer_2d']['alpha_deg']).size):
        raise AssertionError('outer AF kernel length mismatch with current points')

    af_core = eval_af_from_kernel(k_core, pre_ctx.elem_runtime)
    af_buf = eval_af_from_kernel(k_buf, pre_ctx.elem_runtime) if _bessel_kernel_len(k_buf) else _empty_complex()
    af_out = eval_af_from_kernel(k_out, pre_ctx.elem_runtime) if _bessel_kernel_len(k_out) else _empty_complex()

    eco_core = None
    eco_buf = None
    eco_out = None
    eco_total_core = None
    eco_total_buf = None
    eco_total_out = None

    if need_element_co or need_co_total:
        eco_buf = _get_eco_elem_for_scan_zone(pre_ctx, key, 'buffer_2d', mode=mode)
        eco_out = _get_eco_elem_for_scan_zone(pre_ctx, key, 'outer_2d', mode=mode)
        if need_element_co:
            eco_core = None
        if need_co_total:
            eco_total_core = None
            eco_total_buf = None if eco_buf is None else eco_buf * af_buf
            eco_total_out = None if eco_out is None else eco_out * af_out

    meta = {
        'mode': str(mode),
        'n_core': int(af_core.size),
        'n_buffer': int(af_buf.size),
        'n_outer': int(af_out.size),
    }
    _log(verbose, f"[FIELD] theta_s={key:.3f} mode={mode} n_core={meta['n_core']} n_buffer={meta['n_buffer']} n_outer={meta['n_outer']}")
    return Stage1ScanFieldEval(
        theta_s_deg=key,
        af_core=af_core,
        af_buffer=af_buf,
        af_outer=af_out,
        af_abs_core=np.abs(af_core),
        af_abs_buffer=np.abs(af_buf),
        af_abs_outer=np.abs(af_out),
        eco_elem_core=eco_core,
        eco_elem_buffer=eco_buf,
        eco_elem_outer=eco_out,
        eco_total_core=eco_total_core,
        eco_total_buffer=eco_total_buf,
        eco_total_outer=eco_total_out,
        meta=meta,
    )


def _get_a0_fields_for_scan_from_precompute(
    pre_ctx: Stage1PrecomputeContext,
    theta_s_deg: float,
    *,
    mode: str,
) -> Dict[str, np.ndarray]:
    key = float(theta_s_deg)
    core = np.asarray(pre_ctx.core_a0_max_static_by_scan[key], dtype=np.float64)
    buf = _concat_maybe(np.asarray(pre_ctx.buffer_a0_static_by_scan[key], dtype=np.float64), _dynamic_a0_store(pre_ctx, mode, 'buffer_2d').get(key, None), dtype=np.float64)
    out = _concat_maybe(np.asarray(pre_ctx.outer_a0_static_by_scan[key], dtype=np.float64), _dynamic_a0_store(pre_ctx, mode, 'outer_2d').get(key, None), dtype=np.float64)
    return {'core_a0_max': core, 'buffer_a0_abs': buf, 'outer_a0_abs': out}


def eval_fields_for_scan(
    iep_ctx: Stage1IEPContext,
    theta_s_deg: float,
    masked_points: Dict[str, Dict[str, np.ndarray]],
    pre_ctx: Stage1PrecomputeContext,
    *,
    mode: str = 'work',
    need_co_total: bool = False,
    need_element_co: bool = False,
    verbose: bool = True,
) -> Stage1ScanFieldEval:
    return eval_fields_for_current_points(
        iep_ctx,
        theta_s_deg,
        masked_points,
        pre_ctx,
        mode=mode,
        need_co_total=need_co_total,
        need_element_co=need_element_co,
        verbose=verbose,
    )


def build_stage1_values_for_scan(
    a0_fields: Dict[str, np.ndarray],
    field_eval: Stage1ScanFieldEval,
    *,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    core = np.asarray(a0_fields['core_a0_max'], dtype=np.float64) * np.asarray(field_eval.af_abs_core, dtype=np.float64)
    buf = np.asarray(a0_fields['buffer_a0_abs'], dtype=np.float64) * np.asarray(field_eval.af_abs_buffer, dtype=np.float64)
    out = np.asarray(a0_fields['outer_a0_abs'], dtype=np.float64) * np.asarray(field_eval.af_abs_outer, dtype=np.float64)
    _log(verbose, f'[VALUE] theta_s={field_eval.theta_s_deg:.3f} core={core.size} buffer={buf.size} outer={out.size}')
    return {'core_value': core.astype(np.float64), 'buffer_value': buf.astype(np.float64), 'outer_value': out.astype(np.float64)}


# =============================================================================
# Block 9: Single-scan metric bundle assembly
# =============================================================================


class Stage1ScanMetricBundle(object):
    __slots__ = (
        'theta_s_deg',
        'mode',
        'include_dynamic',
        'points',
        'a0_fields',
        'field_eval',
        'values',
        'metric_input',
        'meta',
    )

    def __init__(
        self,
        theta_s_deg,
        mode,
        include_dynamic,
        points,
        a0_fields,
        field_eval,
        values,
        metric_input,
        meta,
    ):
        self.theta_s_deg = theta_s_deg
        self.mode = mode
        self.include_dynamic = include_dynamic
        self.points = points
        self.a0_fields = a0_fields
        self.field_eval = field_eval
        self.values = values
        self.metric_input = metric_input
        self.meta = meta



class Stage1ScanMetricResult(object):
    __slots__ = (
        'theta_s_deg',
        'mode',
        'include_dynamic',
        'bundle',
        'psll_result',
        'ranking_result',
        'meta',
    )

    def __init__(
        self,
        theta_s_deg,
        mode,
        include_dynamic,
        bundle,
        psll_result,
        ranking_result,
        meta,
    ):
        self.theta_s_deg = theta_s_deg
        self.mode = mode
        self.include_dynamic = include_dynamic
        self.bundle = bundle
        self.psll_result = psll_result
        self.ranking_result = ranking_result
        self.meta = meta



class Stage1BatchMetricResult(object):
    __slots__ = (
        'mode',
        'include_dynamic',
        'per_scan',
        'theta_s_deg',
        'psll_db',
        'worst_theta_s_deg',
        'worst_psll_db',
        'meta',
    )

    def __init__(
        self,
        mode,
        include_dynamic,
        per_scan,
        theta_s_deg,
        psll_db,
        worst_theta_s_deg,
        worst_psll_db,
        meta,
    ):
        self.mode = mode
        self.include_dynamic = include_dynamic
        self.per_scan = per_scan
        self.theta_s_deg = theta_s_deg
        self.psll_db = psll_db
        self.worst_theta_s_deg = worst_theta_s_deg
        self.worst_psll_db = worst_psll_db
        self.meta = meta



class Stage1TrackingSeeds(object):
    __slots__ = (
        'theta_s_deg',
        'buffer_seed_alpha_deg',
        'buffer_seed_eta_deg',
        'outer_seed_alpha_deg',
        'outer_seed_eta_deg',
        'meta',
    )

    def __init__(
        self,
        theta_s_deg,
        buffer_seed_alpha_deg,
        buffer_seed_eta_deg,
        outer_seed_alpha_deg,
        outer_seed_eta_deg,
        meta,
    ):
        self.theta_s_deg = theta_s_deg
        self.buffer_seed_alpha_deg = buffer_seed_alpha_deg
        self.buffer_seed_eta_deg = buffer_seed_eta_deg
        self.outer_seed_alpha_deg = outer_seed_alpha_deg
        self.outer_seed_eta_deg = outer_seed_eta_deg
        self.meta = meta



class Stage1ConstraintPoint(object):
    __slots__ = (
        'theta_s_deg',
        'zone_name',
        'idx_local',
        'alpha_deg',
        'eta_deg',
        'theta_deg',
        'phi_deg',
        'value',
        'ratio_db',
    )

    def __init__(
        self,
        theta_s_deg,
        zone_name,
        idx_local,
        alpha_deg,
        eta_deg,
        theta_deg,
        phi_deg,
        value,
        ratio_db,
    ):
        self.theta_s_deg = theta_s_deg
        self.zone_name = zone_name
        self.idx_local = idx_local
        self.alpha_deg = alpha_deg
        self.eta_deg = eta_deg
        self.theta_deg = theta_deg
        self.phi_deg = phi_deg
        self.value = value
        self.ratio_db = ratio_db



class Stage1ConstraintPool(object):
    __slots__ = (
        'per_scan_active',
        'batch_active',
        'constraints',
        'total_M',
        'meta',
    )

    def __init__(
        self,
        per_scan_active,
        batch_active,
        constraints,
        total_M,
        meta,
    ):
        self.per_scan_active = per_scan_active
        self.batch_active = batch_active
        self.constraints = constraints
        self.total_M = total_M
        self.meta = meta



def adapt_bundle_to_metrics_stage1_input(bundle: Stage1ScanMetricBundle) -> _mps.Stage1ScanMetricInput:
    return bundle.metric_input


def _build_metric_input_from_bundle_parts(
    theta_s_deg,
    mode,
    include_dynamic,
    points,
    values,
):
    core = points['core_1d']
    buf = points['buffer_2d']
    out = points['outer_2d']
    meta = {
        'mode': str(mode),
        'include_dynamic': bool(include_dynamic),
        'n_core': int(np.asarray(core['alpha_deg']).size),
        'n_buffer': int(np.asarray(buf['alpha_deg']).size),
        'n_outer': int(np.asarray(out['alpha_deg']).size),
    }
    return _mps.build_stage1_scan_metric_input(
        theta_s_deg=float(theta_s_deg),
        core_points=core,
        core_value=np.asarray(values['core_value'], dtype=np.float64),
        buffer_points=buf,
        buffer_value=np.asarray(values['buffer_value'], dtype=np.float64),
        outer_points=out,
        outer_value=np.asarray(values['outer_value'], dtype=np.float64),
        meta=meta,
    )



def _build_denominator_trace(metric_input: _mps.Stage1ScanMetricInput) -> Dict[str, Any]:
    main_mask = np.asarray(metric_input.core_is_main_guard, dtype=bool)
    vals = np.asarray(metric_input.core_value, dtype=np.float64)
    alpha = np.asarray(metric_input.core_alpha_deg, dtype=np.float64)
    idx = np.flatnonzero(main_mask)
    if idx.size == 0:
        return {
            'n_core_main': 0,
            'D_core': float('nan'),
            'argmax_core_main': -1,
            'alpha_at_argmax_deg': float('nan'),
        }
    sub = vals[idx]
    j = int(np.argmax(sub))
    gidx = int(idx[j])
    return {
        'n_core_main': int(idx.size),
        'D_core': float(sub[j]),
        'argmax_core_main': gidx,
        'alpha_at_argmax_deg': float(alpha[gidx]),
    }


def build_stage1_scan_metric_bundle(
    iep_ctx: Stage1IEPContext,
    pool_ctx: Stage1PoolContext,
    mainlobe_state: MainlobeDomainState,
    a0_ctx: Stage1A0PolicyContext,
    pre_ctx: Stage1PrecomputeContext,
    theta_s_deg: float,
    *,
    mode: str = 'work',
    include_dynamic: bool = True,
    mask_cfg: Stage1MainlobeMaskConfig = Stage1MainlobeMaskConfig(),
    verbose: bool = True,
) -> Stage1ScanMetricBundle:
    key = float(theta_s_deg)
    if include_dynamic and mode in ('work', 'full'):
        refresh_dynamic_precompute_for_scan(pre_ctx, iep_ctx, a0_ctx, pool_ctx, key, mode=mode, verbose=False)
    points = build_masked_points_for_scan(pool_ctx, mainlobe_state, key, mode=mode, include_dynamic=include_dynamic, mask_cfg=mask_cfg, verbose=verbose)
    a0_fields = build_a0_fields_for_current_points(a0_ctx, pre_ctx, points, key, mode=mode)
    field_eval = eval_fields_for_current_points(iep_ctx, key, points, pre_ctx, mode=mode, need_co_total=False, need_element_co=False, verbose=verbose)
    values = build_stage1_values_for_scan(a0_fields, field_eval, verbose=verbose)
    metric_input = _build_metric_input_from_bundle_parts(key, mode, include_dynamic, points, values)
    meta = {
        'mode': str(mode),
        'include_dynamic': bool(include_dynamic),
        'n_core': int(metric_input.core_value.size),
        'n_buffer': int(metric_input.buffer_value.size),
        'n_outer': int(metric_input.outer_value.size),
        'denominator_trace': _build_denominator_trace(metric_input),
    }
    _log(verbose, f'[BUNDLE] theta_s={key:.3f} mode={mode} include_dynamic={include_dynamic}')
    _log(verbose, f"[BUNDLE] core={meta['n_core']} buffer={meta['n_buffer']} outer={meta['n_outer']}")
    return Stage1ScanMetricBundle(
        theta_s_deg=key,
        mode=str(mode),
        include_dynamic=bool(include_dynamic),
        points=points,
        a0_fields=a0_fields,
        field_eval=field_eval,
        values=values,
        metric_input=metric_input,
        meta=meta,
    )


# =============================================================================
# Block 10: PSLL / ranking / tracking-seed extraction
# =============================================================================


def eval_stage1_scan_metrics(
    iep_ctx: Stage1IEPContext,
    pool_ctx: Stage1PoolContext,
    mainlobe_state: MainlobeDomainState,
    a0_ctx: Stage1A0PolicyContext,
    pre_ctx: Stage1PrecomputeContext,
    theta_s_deg: float,
    *,
    mode: str = 'work',
    include_dynamic: bool = True,
    mask_cfg: Stage1MainlobeMaskConfig = Stage1MainlobeMaskConfig(),
    rank_by: str = 'ratio_db',
    verbose: bool = True,
) -> Stage1ScanMetricResult:
    bundle = build_stage1_scan_metric_bundle(
        iep_ctx,
        pool_ctx,
        mainlobe_state,
        a0_ctx,
        pre_ctx,
        theta_s_deg,
        mode=mode,
        include_dynamic=include_dynamic,
        mask_cfg=mask_cfg,
        verbose=verbose,
    )
    psll_result = _mps.eval_psll_stage1_single(bundle.metric_input, verbose=verbose)
    ranking_result = _mps.rank_stage1_side_candidates_single(bundle.metric_input, D_core=psll_result.D_core, rank_by=rank_by, verbose=verbose)
    meta = {'mode': str(mode), 'include_dynamic': bool(include_dynamic), 'rank_by': str(rank_by)}
    _log(verbose, f'[PSLL] theta_s={float(theta_s_deg):.3f} mode={mode} psll_db={float(psll_result.psll_db):.6f}')
    return Stage1ScanMetricResult(
        theta_s_deg=float(theta_s_deg),
        mode=str(mode),
        include_dynamic=bool(include_dynamic),
        bundle=bundle,
        psll_result=psll_result,
        ranking_result=ranking_result,
        meta=meta,
    )


def eval_stage1_scan_set(
    iep_ctx,
    pool_ctx,
    mainlobe_state,
    a0_ctx,
    pre_ctx,
    theta_s_deg_list,
    *,
    mode='work',
    include_dynamic=True,
    mask_cfg=Stage1MainlobeMaskConfig(),
    rank_by='ratio_db',
    verbose=True,
):
    theta_list = np.asarray(theta_s_deg_list, dtype=np.float64).reshape(-1)
    if theta_list.size == 0:
        raise ValueError('theta_s_deg_list must be non-empty')
    per_scan = tuple(
        eval_stage1_scan_metrics(
            iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, float(th),
            mode=mode, include_dynamic=include_dynamic, mask_cfg=mask_cfg,
            rank_by=rank_by, verbose=verbose,
        )
        for th in theta_list.tolist()
    )
    psll_db = np.asarray([r.psll_result.psll_db for r in per_scan], dtype=np.float64)
    valid = ~np.isnan(psll_db)
    if np.any(valid):
        valid_idx = np.flatnonzero(valid)
        local_idx = int(np.argmax(psll_db[valid_idx]))
        idx = int(valid_idx[local_idx])
        worst_theta = float(theta_list[idx])
        worst_psll = float(psll_db[idx])
    else:
        idx = -1
        worst_theta = float('nan')
        worst_psll = float('nan')
    _log(verbose, '[PSLL][BATCH] mode=%s count=%d worst_theta_s=%.3f worst_psll_db=%.6f' % (mode, theta_list.size, worst_theta, worst_psll))
    return Stage1BatchMetricResult(
        mode=str(mode),
        include_dynamic=bool(include_dynamic),
        per_scan=per_scan,
        theta_s_deg=theta_list.astype(np.float64),
        psll_db=psll_db,
        worst_theta_s_deg=worst_theta,
        worst_psll_db=worst_psll,
        meta={'mode': str(mode), 'include_dynamic': bool(include_dynamic), 'n_scans': int(theta_list.size), 'idx_wc': int(idx)},
    )



def extract_tracking_seeds_from_ranking(
    scan_result: Stage1ScanMetricResult,
    tracking_protocol: Dict[str, Any],
    *,
    verbose: bool = True,
) -> Stage1TrackingSeeds:
    rk = scan_result.ranking_result
    topk_buf = int(tracking_protocol.get('topk_buffer', 0))
    topk_out = int(tracking_protocol.get('topk_outer', 0))
    nbuf = min(topk_buf, int(rk.buffer_idx_sorted.size))
    nout = min(topk_out, int(rk.outer_idx_sorted.size))
    buf_alpha = np.asarray(scan_result.bundle.metric_input.buffer_alpha_deg[:0], dtype=np.float64)
    buf_eta = np.asarray(scan_result.bundle.metric_input.buffer_eta_deg[:0], dtype=np.float64)
    out_alpha = np.asarray(scan_result.bundle.metric_input.outer_alpha_deg[:0], dtype=np.float64)
    out_eta = np.asarray(scan_result.bundle.metric_input.outer_eta_deg[:0], dtype=np.float64)
    if nbuf > 0:
        idx = np.asarray(rk.buffer_idx_sorted[:nbuf], dtype=np.int64)
        buf_alpha = np.asarray(scan_result.bundle.metric_input.buffer_alpha_deg[idx], dtype=np.float64)
        buf_eta = np.asarray(scan_result.bundle.metric_input.buffer_eta_deg[idx], dtype=np.float64)
    if nout > 0:
        idx = np.asarray(rk.outer_idx_sorted[:nout], dtype=np.int64)
        out_alpha = np.asarray(scan_result.bundle.metric_input.outer_alpha_deg[idx], dtype=np.float64)
        out_eta = np.asarray(scan_result.bundle.metric_input.outer_eta_deg[idx], dtype=np.float64)
    _log(verbose, f'[SEED] theta_s={scan_result.theta_s_deg:.3f} buffer_topk={buf_alpha.size} outer_topk={out_alpha.size}')
    return Stage1TrackingSeeds(
        theta_s_deg=float(scan_result.theta_s_deg),
        buffer_seed_alpha_deg=buf_alpha,
        buffer_seed_eta_deg=buf_eta,
        outer_seed_alpha_deg=out_alpha,
        outer_seed_eta_deg=out_eta,
        meta={'topk_buffer': topk_buf, 'topk_outer': topk_out},
    )


def extract_work_tracking_seed_map(
    batch_result: Stage1BatchMetricResult,
    tracking_protocol: Dict[str, Any],
    *,
    verbose: bool = True,
) -> Dict[float, Stage1TrackingSeeds]:
    out: Dict[float, Stage1TrackingSeeds] = {}
    for sr in batch_result.per_scan:
        out[float(sr.theta_s_deg)] = extract_tracking_seeds_from_ranking(sr, tracking_protocol, verbose=verbose)
    return out


# =============================================================================
# Block 11: Active-set / constraint pool
# =============================================================================


def adapt_batch_metrics_to_active_set_input(batch_result: Stage1BatchMetricResult) -> List[_as.Stage1ScanRankingInput]:
    out: List[_as.Stage1ScanRankingInput] = []
    for sr in batch_result.per_scan:
        out.append(
            _as.build_stage1_scan_ranking_input_from_metric(
                scan_input=sr.bundle.metric_input,
                psll_result=sr.psll_result,
                ranking_result=sr.ranking_result,
                meta={'mode': sr.mode, 'include_dynamic': sr.include_dynamic},
            )
        )
    return out


def materialize_active_points_for_scan(
    scan_result: Stage1ScanMetricResult,
    scan_active: _as.Stage1ScanActiveSet,
) -> List[Stage1ConstraintPoint]:
    pts: List[Stage1ConstraintPoint] = []
    for i in range(int(scan_active.idx_local.size)):
        pts.append(Stage1ConstraintPoint(
            theta_s_deg=float(scan_result.theta_s_deg),
            zone_name=str(scan_active.zone_name[i]),
            idx_local=int(scan_active.idx_local[i]),
            alpha_deg=float(scan_active.alpha_deg[i]),
            eta_deg=float(scan_active.eta_deg[i]),
            theta_deg=float(scan_active.theta_deg[i]),
            phi_deg=float(scan_active.phi_deg[i]),
            value=float(scan_active.val_side[i]),
            ratio_db=float(scan_active.ratio_db[i]),
        ))
    return pts


def build_stage1_constraint_pool(
    batch_result: Stage1BatchMetricResult,
    *,
    delta_keep_db: float = 5.0,
    min_keep_core_side_per_scan: int = 15,
    min_keep_buffer_per_scan: int = 9,
    min_keep_outer_per_scan: int = 10,
    max_keep_core_side_per_scan: int = 20,
    max_keep_buffer_per_scan: int = 14,
    max_keep_outer_per_scan: int = 14,
    core_merge_alpha_deg: float = 0.5,
    buffer_merge_alpha_deg: float = 3.0,
    buffer_merge_eta_deg: float = 12.0,
    outer_merge_alpha_deg: float = 4.0,
    outer_merge_eta_deg: float = 15.0,
    max_keep_per_scan: int = 48,
    max_total_points: int = 550,
    verbose: bool = True,
) -> Stage1ConstraintPool:
    rankings = adapt_batch_metrics_to_active_set_input(batch_result)
    batch_active = _as.build_active_set_stage1_batch(
        rankings,
        delta_keep_db=delta_keep_db,
        min_keep_core_side_per_scan=min_keep_core_side_per_scan,
        min_keep_buffer_per_scan=min_keep_buffer_per_scan,
        min_keep_outer_per_scan=min_keep_outer_per_scan,
        max_keep_core_side_per_scan=max_keep_core_side_per_scan,
        max_keep_buffer_per_scan=max_keep_buffer_per_scan,
        max_keep_outer_per_scan=max_keep_outer_per_scan,
        core_merge_alpha_deg=core_merge_alpha_deg,
        buffer_merge_alpha_deg=buffer_merge_alpha_deg,
        buffer_merge_eta_deg=buffer_merge_eta_deg,
        outer_merge_alpha_deg=outer_merge_alpha_deg,
        outer_merge_eta_deg=outer_merge_eta_deg,
        max_keep_per_scan=max_keep_per_scan,
        max_total_points=max_total_points,
        verbose=verbose,
    )
    constraints: List[Stage1ConstraintPoint] = []
    for sr, sa in zip(batch_result.per_scan, batch_active.per_scan):
        constraints.extend(materialize_active_points_for_scan(sr, sa))
    _log(verbose, f'[ACTIVE] mode={batch_result.mode} total_M={batch_active.total_M}')
    _log(verbose, f'[ACTIVE] per_scan_counts={batch_active.per_scan_counts.tolist()}')
    return Stage1ConstraintPool(
        per_scan_active=batch_active.per_scan,
        batch_active=batch_active,
        constraints=tuple(constraints),
        total_M=int(batch_active.total_M),
        meta={'mode': batch_result.mode, 'include_dynamic': batch_result.include_dynamic, 'n_constraints': int(batch_active.total_M)},
    )


__all__ = [
    'Stage1IEPContext',
    'RadiusInitSpec',
    'RadiusInitResult',
    'RadiusInitBatch',
    'ElementInitConfig',
    'RingElementCountResult',
    'ElementGeometry',
    'MainlobeDomainConfig',
    'MainlobeDomainState',
    'init_stage1_iep',
    'validate_radius_init',
    'build_radius_init',
    'build_radius_init_batch',
    'build_ring_element_counts',
    'build_element_geometry',
    'build_mainlobe_domain_state',
    'Stage1PoolConfig',
    'Stage1PoolContext',
    'validate_work_is_full_slice',
    'init_stage1_pool_context',
    'get_current_work_theta_s_deg',
    'get_full_theta_s_deg',
    'build_current_work_eval_plan',
    'should_refresh_work_from_full',
    'update_work_scan_angles_from_full_feedback',
    'build_buffer_tracking_points_from_seeds',
    'build_outer_tracking_points_from_seeds',
    'clear_work_dynamic_points_for_scan',
    'clear_full_dynamic_points_for_scan',
    'clear_all_work_dynamic_points',
    'clear_all_full_dynamic_points',
    'attach_dynamic_points_for_scan',
    'rebuild_work_dynamic_points_for_scan',
    'rebuild_full_dynamic_points_for_scan',
    'get_stage1_points_for_scan',
    'get_all_work_scan_points',
    'Stage1MainlobeMaskConfig',
    'Stage1CoreMask',
    'get_alpha_main_for_scan',
    'build_core_mask_for_scan',
    'build_masked_points_for_scan',
    'Stage1A0PolicyConfig',
    'Stage1A0PolicyContext',
    'init_stage1_a0_policy',
    'ensure_core_a0_cache_for_scan',
    'get_core_a0_max_for_scan',
    'eval_pointwise_a0_for_2d_points',
    'build_a0_fields_for_scan',
    'build_a0_fields_for_current_points',
    'Stage1ElementKernelCache',
    'Stage1ElementRuntimeGeometry',
    'Stage1BesselPointKernel',
    'eval_af_core_j0_collapsed',
    'eval_af_core_j0_and_jacobian',
    'eval_af_bessel_alias_L1',
    'eval_af_bessel_alias_L1_and_jacobian',
    'Stage1ScanFieldEval',
    'Stage1PrecomputeContext',
    'build_element_kernel_cache',
    'build_element_runtime_geometry',
    'build_af_kernel_for_points',
    'eval_af_from_kernel',
    'eval_element_co_on_points',
    'init_stage1_precompute_context',
    'refresh_dynamic_precompute_for_scan',
    'eval_fields_for_current_points',
    'eval_fields_for_scan',
    'build_stage1_values_for_scan',
    'Stage1ScanMetricBundle',
    'Stage1ScanMetricResult',
    'Stage1BatchMetricResult',
    'Stage1TrackingSeeds',
    'Stage1ConstraintPoint',
    'Stage1ConstraintPool',
    'adapt_bundle_to_metrics_stage1_input',
    'build_stage1_scan_metric_bundle',
    'eval_stage1_scan_metrics',
    'eval_stage1_scan_set',
    'extract_tracking_seeds_from_ranking',
    'extract_work_tracking_seed_map',
    'adapt_batch_metrics_to_active_set_input',
    'materialize_active_points_for_scan',
    'build_stage1_constraint_pool',
]


# =============================================================================
# Block 12: Radii convex optimization (trust-region SCA)
# =============================================================================


class Stage1RadiiOptConfig(object):
    __slots__ = (
        'd_min', 'sigma', 'gamma_main', 'lambda_dr',
        'use_main_anchors', 'anchors_per_scan', 'anchor_merge_alpha_deg',
        'radius_feasibility_tol', 'solver_order', 'solver_verbose',
        'scipy_maxiter', 'scipy_ftol',
    )

    def __init__(
        self,
        d_min=0.5,
        sigma=0.15,
        gamma_main=0.95,
        lambda_dr=0.0,
        use_main_anchors=True,
        anchors_per_scan=1,
        anchor_merge_alpha_deg=0.5,
        radius_feasibility_tol=5e-6,
        solver_order=('MOSEK', 'ECOS', 'SCS'),
        solver_verbose=False,
        scipy_maxiter=400,
        scipy_ftol=1e-8,
    ):
        self.d_min = float(d_min)
        self.sigma = float(sigma)
        self.gamma_main = float(gamma_main)
        self.lambda_dr = float(lambda_dr)
        self.use_main_anchors = bool(use_main_anchors)
        self.anchors_per_scan = int(anchors_per_scan)
        self.anchor_merge_alpha_deg = float(anchor_merge_alpha_deg)
        self.radius_feasibility_tol = float(radius_feasibility_tol)
        self.solver_order = tuple(solver_order)
        self.solver_verbose = bool(solver_verbose)
        self.scipy_maxiter = int(scipy_maxiter)
        self.scipy_ftol = float(scipy_ftol)




class Stage1RadiiOptAnchor(object):
    __slots__ = (
        'theta_s_deg', 'core_index', 'alpha_deg', 'a0_value',
        'D_core', 'reference_value', 'AF_old', 'q_old',
    )

    def __init__(
        self,
        theta_s_deg,
        core_index,
        alpha_deg,
        a0_value,
        D_core,
        reference_value,
        AF_old,
        q_old,
    ):
        self.theta_s_deg = float(theta_s_deg)
        self.core_index = int(core_index)
        self.alpha_deg = float(alpha_deg)
        self.a0_value = float(a0_value)
        self.D_core = float(D_core)
        self.reference_value = float(reference_value)
        self.AF_old = complex(AF_old)
        self.q_old = complex(q_old)




class Stage1RadiiOptPrecomputeContext(object):
    __slots__ = (
        'theta_work_deg',
        'active_points',
        'n_active',
        'n_rings',
        'theta_s_of_active',
        'zone_of_active',
        'u_active',
        'v_active',
        'a0_active',
        'D_of_active',
        'D_scan_current',
        'c_by_active_ring',
        'AF_old',
        'J_old',
        'rho_active',
        'eta_active',
        'backend_of_active',
        'anchors',
        'AF_anchor_old',
        'J_anchor_old',
        'anchor_D',
        'anchor_a0',
        'q_anchor_old',
        'rho_anchor',
        'ring_elem_indices',
    )

    def __init__(
        self,
        theta_work_deg,
        active_points,
        n_active,
        n_rings,
        theta_s_of_active,
        zone_of_active,
        u_active,
        v_active,
        a0_active,
        D_of_active,
        D_scan_current,
        c_by_active_ring,
        AF_old,
        J_old,
        rho_active,
        eta_active,
        backend_of_active,
        anchors,
        AF_anchor_old,
        J_anchor_old,
        anchor_D,
        anchor_a0,
        q_anchor_old,
        rho_anchor,
        ring_elem_indices,
    ):
        self.theta_work_deg = theta_work_deg
        self.active_points = active_points
        self.n_active = n_active
        self.n_rings = n_rings
        self.theta_s_of_active = theta_s_of_active
        self.zone_of_active = zone_of_active
        self.u_active = u_active
        self.v_active = v_active
        self.a0_active = a0_active
        self.D_of_active = D_of_active
        self.D_scan_current = D_scan_current
        self.c_by_active_ring = c_by_active_ring
        self.AF_old = AF_old
        self.J_old = J_old
        self.rho_active = rho_active
        self.eta_active = eta_active
        self.backend_of_active = backend_of_active
        self.anchors = anchors
        self.AF_anchor_old = AF_anchor_old
        self.J_anchor_old = J_anchor_old
        self.anchor_D = anchor_D
        self.anchor_a0 = anchor_a0
        self.q_anchor_old = q_anchor_old
        self.rho_anchor = rho_anchor
        self.ring_elem_indices = ring_elem_indices



class Stage1RadiiOptProblemData(object):
    __slots__ = (
        'r_old',
        'D_of_active',
        'a0_active',
        'AF_old',
        'J_old',
        'anchor_D',
        'anchor_a0',
        'AF_anchor_old',
        'J_anchor_old',
        'q_anchor_old',
        'R0',
        'd_min',
        'beta',
        'sigma',
        'gamma_main',
        'lambda_dr',
    )

    def __init__(
        self,
        r_old,
        D_of_active,
        a0_active,
        AF_old,
        J_old,
        anchor_D,
        anchor_a0,
        AF_anchor_old,
        J_anchor_old,
        q_anchor_old,
        R0,
        d_min,
        beta,
        sigma,
        gamma_main,
        lambda_dr,
    ):
        self.r_old = r_old
        self.D_of_active = D_of_active
        self.a0_active = a0_active
        self.AF_old = AF_old
        self.J_old = J_old
        self.anchor_D = anchor_D
        self.anchor_a0 = anchor_a0
        self.AF_anchor_old = AF_anchor_old
        self.J_anchor_old = J_anchor_old
        self.q_anchor_old = q_anchor_old
        self.R0 = R0
        self.d_min = d_min
        self.beta = beta
        self.sigma = sigma
        self.gamma_main = gamma_main
        self.lambda_dr = lambda_dr



class Stage1RadiiOptResult(object):
    __slots__ = (
        'status',
        'success',
        'delta_opt',
        'r_old',
        'dr',
        'r_trial',
        'predicted_ratio_upper_bound',
        'max_abs_dr',
        'solver_used',
        'meta',
    )

    def __init__(
        self,
        status,
        success,
        delta_opt,
        r_old,
        dr,
        r_trial,
        predicted_ratio_upper_bound,
        max_abs_dr,
        solver_used,
        meta,
    ):
        self.status = status
        self.success = success
        self.delta_opt = delta_opt
        self.r_old = r_old
        self.dr = dr
        self.r_trial = r_trial
        self.predicted_ratio_upper_bound = predicted_ratio_upper_bound
        self.max_abs_dr = max_abs_dr
        self.solver_used = solver_used
        self.meta = meta



def _build_ring_elem_indices(elem_kernel: Stage1ElementKernelCache, K: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    ring_idx = np.asarray(elem_kernel.ring_idx_of_elem, dtype=np.int64)
    for k in range(int(K)):
        out.append(np.flatnonzero(ring_idx == int(k)).astype(np.int64))
    return out


def _compute_c_ring_for_zone(
    zone_name: str,
    idx_local: int,
    scan_result: Stage1ScanMetricResult,
    elem_kernel: Stage1ElementKernelCache,
    ring_elem_indices: List[np.ndarray],
) -> Tuple[List[np.ndarray], float, float, float, complex]:
    bundle = scan_result.bundle
    key = str(zone_name)
    if key == 'core_side_1d':
        rho = float(np.asarray(bundle.points['core_1d']['rho'], dtype=np.float64)[idx_local])
        a0 = float(np.asarray(bundle.a0_fields['core_a0_max'], dtype=np.float64)[idx_local])
        af_old = complex(np.asarray(bundle.field_eval.af_core, dtype=np.complex128)[idx_local])
        c_by_ring = [rho * np.asarray(elem_kernel.cos_psi_elem[idxs], dtype=np.float64) for idxs in ring_elem_indices]
        return c_by_ring, float('nan'), float('nan'), a0, af_old
    if key == 'buffer_side_2d':
        u = float(np.asarray(bundle.points['buffer_2d']['u'], dtype=np.float64)[idx_local])
        v = float(np.asarray(bundle.points['buffer_2d']['v'], dtype=np.float64)[idx_local])
        theta_s = float(scan_result.theta_s_deg)
        us = math.sin(math.radians(theta_s))
        a0 = float(np.asarray(bundle.a0_fields['buffer_a0_abs'], dtype=np.float64)[idx_local])
        af_old = complex(np.asarray(bundle.field_eval.af_buffer, dtype=np.complex128)[idx_local])
        du = float(u - us)
        dv = float(v)
        c_by_ring = [du * np.asarray(elem_kernel.cos_psi_elem[idxs], dtype=np.float64) + dv * np.asarray(elem_kernel.sin_psi_elem[idxs], dtype=np.float64) for idxs in ring_elem_indices]
        return c_by_ring, u, v, a0, af_old
    if key == 'outer_side_2d':
        u = float(np.asarray(bundle.points['outer_2d']['u'], dtype=np.float64)[idx_local])
        v = float(np.asarray(bundle.points['outer_2d']['v'], dtype=np.float64)[idx_local])
        theta_s = float(scan_result.theta_s_deg)
        us = math.sin(math.radians(theta_s))
        a0 = float(np.asarray(bundle.a0_fields['outer_a0_abs'], dtype=np.float64)[idx_local])
        af_old = complex(np.asarray(bundle.field_eval.af_outer, dtype=np.complex128)[idx_local])
        du = float(u - us)
        dv = float(v)
        c_by_ring = [du * np.asarray(elem_kernel.cos_psi_elem[idxs], dtype=np.float64) + dv * np.asarray(elem_kernel.sin_psi_elem[idxs], dtype=np.float64) for idxs in ring_elem_indices]
        return c_by_ring, u, v, a0, af_old
    raise ValueError(f'unsupported zone_name={zone_name}')


def _compute_J_from_c_by_ring(c_by_ring: List[np.ndarray], r_old: np.ndarray, beta: float) -> np.ndarray:
    K = int(len(c_by_ring))
    J = np.zeros((K,), dtype=np.complex128)
    for k in range(K):
        c = np.asarray(c_by_ring[k], dtype=np.float64)
        if c.size == 0:
            continue
        phase = np.exp(1j * float(beta) * c * float(r_old[k]))
        J[k] = 1j * float(beta) * np.sum(c * phase)
    return J


def _reconstruct_af_from_c_by_ring(c_by_ring: List[np.ndarray], r_now: np.ndarray, beta: float, include_center: bool) -> complex:
    val = 1.0 + 0.0j if include_center else 0.0 + 0.0j
    for k, c in enumerate(c_by_ring):
        carr = np.asarray(c, dtype=np.float64)
        if carr.size == 0:
            continue
        val += np.sum(np.exp(1j * float(beta) * carr * float(r_now[k])))
    return complex(val)

def _collect_active_bessel_coordinates(
    active_points: Sequence[Stage1ConstraintPoint],
    scan_map: Dict[float, Stage1ScanMetricResult],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[float, float]]:
    M = int(len(active_points))
    rho = np.zeros((M,), dtype=np.float64)
    eta = np.zeros((M,), dtype=np.float64)
    backend = np.empty((M,), dtype=object)
    theta_s = np.zeros((M,), dtype=np.float64)
    zone = np.empty((M,), dtype=object)
    u_active = np.full((M,), np.nan, dtype=np.float64)
    v_active = np.full((M,), np.nan, dtype=np.float64)
    a0_active = np.zeros((M,), dtype=np.float64)
    D_scan_current: Dict[float, float] = {}
    core_rows: List[int] = []
    side_rows: List[int] = []

    for j, cp in enumerate(active_points):
        theta_key = float(cp.theta_s_deg)
        sr = scan_map[theta_key]
        theta_s[j] = theta_key
        zone[j] = str(cp.zone_name)
        D_scan_current[theta_key] = float(sr.psll_result.D_core)
        idx = int(cp.idx_local)
        z = str(cp.zone_name)

        if z == 'core_side_1d':
            rho[j] = float(np.asarray(sr.bundle.points['core_1d']['rho'], dtype=np.float64)[idx])
            eta[j] = 0.0
            a0_active[j] = float(np.asarray(sr.bundle.a0_fields['core_a0_max'], dtype=np.float64)[idx])
            backend[j] = 'core_j0'
            core_rows.append(j)
        elif z == 'buffer_side_2d':
            u = float(np.asarray(sr.bundle.points['buffer_2d']['u'], dtype=np.float64)[idx])
            v = float(np.asarray(sr.bundle.points['buffer_2d']['v'], dtype=np.float64)[idx])
            u_s = math.sin(math.radians(theta_key))
            du = u - u_s
            dv = v
            rho[j] = math.sqrt(du * du + dv * dv)
            eta[j] = math.atan2(dv, du)
            u_active[j] = u
            v_active[j] = v
            a0_active[j] = float(np.asarray(sr.bundle.a0_fields['buffer_a0_abs'], dtype=np.float64)[idx])
            backend[j] = 'side_bessel_L1'
            side_rows.append(j)
        elif z == 'outer_side_2d':
            u = float(np.asarray(sr.bundle.points['outer_2d']['u'], dtype=np.float64)[idx])
            v = float(np.asarray(sr.bundle.points['outer_2d']['v'], dtype=np.float64)[idx])
            u_s = math.sin(math.radians(theta_key))
            du = u - u_s
            dv = v
            rho[j] = math.sqrt(du * du + dv * dv)
            eta[j] = math.atan2(dv, du)
            u_active[j] = u
            v_active[j] = v
            a0_active[j] = float(np.asarray(sr.bundle.a0_fields['outer_a0_abs'], dtype=np.float64)[idx])
            backend[j] = 'side_bessel_L1'
            side_rows.append(j)
        else:
            raise ValueError(f'Unknown active point zone_name={z}')

    return (
        rho,
        eta,
        backend,
        np.asarray(core_rows, dtype=np.int64),
        np.asarray(side_rows, dtype=np.int64),
        theta_s,
        zone,
        np.vstack([u_active, v_active, a0_active]).T,
        D_scan_current,
    )


def _eval_active_af_jacobian_bessel_stage1(
    active_points: Sequence[Stage1ConstraintPoint],
    scan_map: Dict[float, Stage1ScanMetricResult],
    elem_runtime: Stage1ElementRuntimeGeometry,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[float, float]]:
    rho, eta, backend, core_rows, side_rows, theta_s, zone, uv_a0, D_scan_current = _collect_active_bessel_coordinates(active_points, scan_map)
    u_active = uv_a0[:, 0]
    v_active = uv_a0[:, 1]
    a0_active = uv_a0[:, 2]

    r = np.asarray(elem_runtime.r_over_lambda, dtype=np.float64).reshape(-1)
    N = np.asarray(elem_runtime.N_per_ring, dtype=np.int64).reshape(-1)
    beta = float(elem_runtime.beta)
    include_center = bool(elem_runtime.include_center)
    M = int(len(active_points))
    K = int(r.size)
    AF = np.zeros((M,), dtype=np.complex128)
    J = np.zeros((M, K), dtype=np.complex128)

    if core_rows.size:
        AF_c, J_c = eval_af_core_j0_and_jacobian(rho[core_rows], r, N, beta=beta, include_center=include_center)
        AF[core_rows] = AF_c
        J[core_rows, :] = J_c
    if side_rows.size:
        AF_s, J_s = eval_af_bessel_alias_L1_and_jacobian(rho[side_rows], eta[side_rows], r, N, beta=beta, include_center=include_center)
        AF[side_rows] = AF_s
        J[side_rows, :] = J_s

    return AF, J, rho, eta, backend, theta_s, zone, u_active, v_active, a0_active, D_scan_current


def _eval_anchor_af_jacobian_bessel_stage1(
    anchors: Sequence[Stage1RadiiOptAnchor],
    elem_runtime: Stage1ElementRuntimeGeometry,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = np.asarray([math.sin(math.radians(float(a.alpha_deg))) for a in anchors], dtype=np.float64)
    r = np.asarray(elem_runtime.r_over_lambda, dtype=np.float64).reshape(-1)
    N = np.asarray(elem_runtime.N_per_ring, dtype=np.int64).reshape(-1)
    AF, J = eval_af_core_j0_and_jacobian(rho, r, N, beta=float(elem_runtime.beta), include_center=bool(elem_runtime.include_center))
    q = np.ones_like(AF, dtype=np.complex128)
    nz = np.abs(AF) > 1e-15
    q[nz] = AF[nz] / np.abs(AF[nz])
    return AF, J, q, rho




def _scan_result_map(batch_result: Stage1BatchMetricResult) -> Dict[float, Stage1ScanMetricResult]:
    return {float(sr.theta_s_deg): sr for sr in batch_result.per_scan}


def _select_main_anchor_indices(scan_result, anchors_per_scan, merge_alpha_deg):
    count = int(anchors_per_scan)
    if count <= 0:
        return np.asarray([], dtype=np.int64)
    metric_input = scan_result.bundle.metric_input
    mask = np.asarray(metric_input.core_is_main_guard, dtype=bool).reshape(-1)
    values = np.asarray(metric_input.core_value, dtype=np.float64).reshape(-1)
    alpha = np.asarray(metric_input.core_alpha_deg, dtype=np.float64).reshape(-1)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.asarray([], dtype=np.int64)
    order = np.lexsort((idx, -values[idx]))
    ranked = idx[order].astype(np.int64)
    selected = []
    merge = max(0.0, float(merge_alpha_deg))
    for ii in ranked.tolist():
        if all(abs(float(alpha[ii]) - float(alpha[jj])) > merge for jj in selected):
            selected.append(int(ii))
        if len(selected) >= count:
            break
    if len(selected) < count:
        used = set(selected)
        for ii in ranked.tolist():
            if int(ii) not in used:
                selected.append(int(ii)); used.add(int(ii))
            if len(selected) >= count:
                break
    return np.asarray(selected, dtype=np.int64)

def build_radii_opt_precompute_context(
    constraint_pool,
    batch_result,
    elem_kernel,
    elem_runtime,
    radius_init,
    *,
    gamma_main=0.95,
    anchors_per_scan=1,
    anchor_merge_alpha_deg=0.5,
    verbose=True,
):
    del gamma_main
    _log(verbose, '=== [A0CRA][BLOCK12] BUILD RADII OPT PRECOMP: START ===')
    r_old = np.asarray(radius_init.r_over_lambda, dtype=np.float64).reshape(-1)
    K = int(r_old.size)
    scan_map = _scan_result_map(batch_result)
    ring_elem_indices = _build_ring_elem_indices(elem_kernel, K)
    constraints = list(constraint_pool.constraints)
    M = int(len(constraints))

    AF_old, J_old, rho_active, eta_active, backend_of_active, theta_s_of_active, zone_of_active, u_active, v_active, a0_active, D_scan_current = _eval_active_af_jacobian_bessel_stage1(
        constraints, scan_map, elem_runtime,
    )
    D_of_active = np.asarray([D_scan_current[float(cp.theta_s_deg)] for cp in constraints], dtype=np.float64)
    c_by_active_ring = [[] for _ in range(M)]

    anchors = []
    if int(anchors_per_scan) > 0:
        for theta_key, sr in scan_map.items():
            anchor_idx = _select_main_anchor_indices(sr, int(anchors_per_scan), float(anchor_merge_alpha_deg))
            for idx in anchor_idx.tolist():
                alpha = float(np.asarray(sr.bundle.points['core_1d']['alpha_deg'], dtype=np.float64)[idx])
                a0_value = float(np.asarray(sr.bundle.a0_fields['core_a0_max'], dtype=np.float64)[idx])
                D_core = float(sr.psll_result.D_core)
                reference_value = float(np.asarray(sr.bundle.values['core_value'], dtype=np.float64)[idx])
                anchors.append(Stage1RadiiOptAnchor(
                    theta_s_deg=float(theta_key), core_index=int(idx), alpha_deg=alpha,
                    a0_value=a0_value, D_core=D_core, reference_value=reference_value,
                    AF_old=1.0 + 0.0j, q_old=1.0 + 0.0j,
                ))

    AF_anchor_old = np.zeros((0,), dtype=np.complex128)
    J_anchor_old = np.zeros((0, K), dtype=np.complex128)
    anchor_D = np.zeros((0,), dtype=np.float64)
    anchor_a0 = np.zeros((0,), dtype=np.float64)
    q_anchor_old = np.zeros((0,), dtype=np.complex128)
    rho_anchor = np.zeros((0,), dtype=np.float64)
    if anchors:
        AF_anchor_old, J_anchor_old, q_anchor_old, rho_anchor = _eval_anchor_af_jacobian_bessel_stage1(anchors, elem_runtime)
        anchors_new = []
        for i, anchor in enumerate(anchors):
            anchors_new.append(Stage1RadiiOptAnchor(
                theta_s_deg=anchor.theta_s_deg,
                core_index=anchor.core_index,
                alpha_deg=anchor.alpha_deg,
                a0_value=anchor.a0_value,
                D_core=anchor.D_core,
                reference_value=anchor.reference_value,
                AF_old=complex(AF_anchor_old[i]),
                q_old=complex(q_anchor_old[i]),
            ))
        anchors = anchors_new
        anchor_D = np.asarray([a.reference_value for a in anchors], dtype=np.float64)
        anchor_a0 = np.asarray([a.a0_value for a in anchors], dtype=np.float64)

    theta_work_deg = np.asarray(batch_result.theta_s_deg, dtype=np.float64).copy()
    ctx = Stage1RadiiOptPrecomputeContext(
        theta_work_deg=theta_work_deg,
        active_points=constraints,
        n_active=M,
        n_rings=K,
        theta_s_of_active=np.asarray(theta_s_of_active, dtype=np.float64),
        zone_of_active=np.asarray(zone_of_active, dtype=object),
        u_active=np.asarray(u_active, dtype=np.float64),
        v_active=np.asarray(v_active, dtype=np.float64),
        a0_active=np.asarray(a0_active, dtype=np.float64),
        D_of_active=np.asarray(D_of_active, dtype=np.float64),
        D_scan_current=D_scan_current,
        c_by_active_ring=c_by_active_ring,
        AF_old=np.asarray(AF_old, dtype=np.complex128),
        J_old=np.asarray(J_old, dtype=np.complex128),
        rho_active=np.asarray(rho_active, dtype=np.float64),
        eta_active=np.asarray(eta_active, dtype=np.float64),
        backend_of_active=np.asarray(backend_of_active, dtype=object),
        anchors=anchors,
        AF_anchor_old=AF_anchor_old,
        J_anchor_old=J_anchor_old,
        anchor_D=anchor_D,
        anchor_a0=anchor_a0,
        q_anchor_old=q_anchor_old,
        rho_anchor=rho_anchor,
        ring_elem_indices=ring_elem_indices,
    )
    _log(verbose, '[RADII][PRECOMP] work_scans=%s' % theta_work_deg.tolist())
    _log(verbose, '[RADII][PRECOMP] active_M=%d rings_K=%d anchors=%d backend=bessel_stage1' % (M, K, len(anchors)))
    if M > 0:
        _log(verbose, '[RADII][PRECOMP] D_active min/mean/max=%.6g/%.6g/%.6g' % (D_of_active.min(), D_of_active.mean(), D_of_active.max()))
        _log(verbose, '[RADII][PRECOMP] a0_active min/mean/max=%.6g/%.6g/%.6g' % (a0_active.min(), a0_active.mean(), a0_active.max()))
    _log(verbose, '=== [A0CRA][BLOCK12] BUILD RADII OPT PRECOMP: DONE ===')
    return ctx


def build_radii_opt_problem_data(
    radius_init: RadiusInitResult,
    precomp_ctx: Stage1RadiiOptPrecomputeContext,
    cfg: Stage1RadiiOptConfig,
    *,
    beta: float = 2.0 * np.pi,
    verbose: bool = True,
) -> Stage1RadiiOptProblemData:
    r_old = np.asarray(radius_init.r_over_lambda, dtype=np.float64).reshape(-1)
    data = Stage1RadiiOptProblemData(
        r_old=r_old,
        D_of_active=np.asarray(precomp_ctx.D_of_active, dtype=np.float64),
        a0_active=np.asarray(precomp_ctx.a0_active, dtype=np.float64),
        AF_old=np.asarray(precomp_ctx.AF_old, dtype=np.complex128),
        J_old=np.asarray(precomp_ctx.J_old, dtype=np.complex128),
        anchor_D=np.asarray(precomp_ctx.anchor_D, dtype=np.float64),
        anchor_a0=np.asarray(precomp_ctx.anchor_a0, dtype=np.float64),
        AF_anchor_old=np.asarray(precomp_ctx.AF_anchor_old, dtype=np.complex128),
        J_anchor_old=np.asarray(precomp_ctx.J_anchor_old, dtype=np.complex128),
        q_anchor_old=np.asarray(precomp_ctx.q_anchor_old, dtype=np.complex128),
        R0=float(radius_init.spec.R0),
        d_min=float(cfg.d_min),
        beta=float(beta),
        sigma=float(cfg.sigma),
        gamma_main=float(cfg.gamma_main),
        lambda_dr=float(cfg.lambda_dr),
    )
    _log(verbose, f'[RADII][PROBLEM] K={r_old.size} M={data.AF_old.size} anchors={data.AF_anchor_old.size} d_min={data.d_min:.6f} sigma={data.sigma:.6f} gamma={data.gamma_main:.6f}')
    return data


def _solve_radii_socp_step_cvxpy(problem_data: Stage1RadiiOptProblemData, cfg: Stage1RadiiOptConfig, *, verbose: bool = True) -> Optional[Stage1RadiiOptResult]:
    try:
        import cvxpy as cp  # type: ignore
    except Exception:
        return None
    K = int(problem_data.r_old.size)
    dr = cp.Variable(K)
    delta = cp.Variable(nonneg=True)
    constraints = []
    for j in range(int(problem_data.AF_old.size)):
        Fj = problem_data.a0_active[j] * (problem_data.AF_old[j] + problem_data.J_old[j, :] @ dr)
        constraints.append(cp.abs(Fj) <= delta * problem_data.D_of_active[j])
    tr = float(problem_data.sigma / problem_data.beta)
    for k in range(K):
        constraints.append(cp.abs(dr[k]) <= tr)
    r_new = problem_data.r_old + dr
    constraints.append(r_new[0] >= problem_data.d_min)
    constraints.append(r_new[-1] <= problem_data.R0)
    for k in range(K - 1):
        constraints.append(r_new[k + 1] - r_new[k] >= problem_data.d_min)
    if cfg.use_main_anchors and problem_data.AF_anchor_old.size > 0:
        for a in range(int(problem_data.AF_anchor_old.size)):
            Fa = problem_data.AF_anchor_old[a] + problem_data.J_anchor_old[a, :] @ dr
            lhs = problem_data.anchor_a0[a] * cp.real(np.conj(problem_data.q_anchor_old[a]) * Fa)
            constraints.append(lhs >= problem_data.gamma_main * problem_data.anchor_D[a])
    obj = cp.Minimize(delta + problem_data.lambda_dr * cp.sum_squares(dr))
    prob = cp.Problem(obj, constraints)
    last_err = None
    for solver_name in tuple(cfg.solver_order):
        solver = getattr(cp, solver_name, None)
        if solver is None:
            continue
        try:
            prob.solve(solver=solver, verbose=bool(cfg.solver_verbose))
        except Exception as e:
            last_err = e
            continue
        if prob.status in ('optimal', 'optimal_inaccurate') and dr.value is not None and delta.value is not None:
            dr_val = np.asarray(dr.value, dtype=np.float64).reshape(-1)
            delta_val = float(delta.value)
            r_trial = np.asarray(problem_data.r_old + dr_val, dtype=np.float64)
            res = Stage1RadiiOptResult(
                status=str(prob.status),
                success=True,
                delta_opt=delta_val,
                r_old=np.asarray(problem_data.r_old, dtype=np.float64),
                dr=dr_val,
                r_trial=r_trial,
                predicted_ratio_upper_bound=delta_val,
                max_abs_dr=float(np.max(np.abs(dr_val))) if dr_val.size else 0.0,
                solver_used=str(solver_name),
                meta={'backend': 'cvxpy', 'objective': float(prob.value) if prob.value is not None else float('nan')},
            )
            _log(verbose, f'[RADII][SOLVE][CVXPY] solver={solver_name} status={res.status} delta={res.delta_opt:.6g} max|dr|={res.max_abs_dr:.6g}')
            return res
    if last_err is not None:
        _log(verbose, f'[RADII][SOLVE][CVXPY] unavailable/failed, fallback to scipy: {last_err}')
    return None


def _solve_radii_socp_step_scipy(problem_data: Stage1RadiiOptProblemData, cfg: Stage1RadiiOptConfig, *, verbose: bool = True) -> Stage1RadiiOptResult:
    from scipy.optimize import minimize
    K = int(problem_data.r_old.size)
    tr = float(problem_data.sigma / problem_data.beta)
    current_ratio = np.abs(problem_data.a0_active * problem_data.AF_old) / np.maximum(problem_data.D_of_active, 1e-15)
    delta0 = float(np.max(current_ratio)) if current_ratio.size else 0.0
    x0 = np.zeros((K + 1,), dtype=np.float64)
    x0[-1] = max(delta0, 1e-6)

    def unpack(x: np.ndarray) -> Tuple[np.ndarray, float]:
        dr = np.asarray(x[:K], dtype=np.float64)
        delta = float(x[K])
        return dr, delta

    def objective(x: np.ndarray) -> float:
        dr, delta = unpack(x)
        return float(delta + problem_data.lambda_dr * np.dot(dr, dr))

    cons = []
    for j in range(int(problem_data.AF_old.size)):
        def _c_side(x, j=j):
            dr, delta = unpack(np.asarray(x, dtype=np.float64))
            Fj = problem_data.a0_active[j] * (problem_data.AF_old[j] + np.dot(problem_data.J_old[j, :], dr))
            return float(delta * problem_data.D_of_active[j] - abs(Fj))
        cons.append({'type': 'ineq', 'fun': _c_side})
    if cfg.use_main_anchors and problem_data.AF_anchor_old.size > 0:
        for a in range(int(problem_data.AF_anchor_old.size)):
            def _c_anchor(x, a=a):
                dr, _ = unpack(np.asarray(x, dtype=np.float64))
                Fa = problem_data.AF_anchor_old[a] + np.dot(problem_data.J_anchor_old[a, :], dr)
                lhs = problem_data.anchor_a0[a] * np.real(np.conj(problem_data.q_anchor_old[a]) * Fa)
                return float(lhs - problem_data.gamma_main * problem_data.anchor_D[a])
            cons.append({'type': 'ineq', 'fun': _c_anchor})
    cons.append({'type': 'ineq', 'fun': lambda x: float(problem_data.r_old[0] + x[0] - problem_data.d_min)})
    cons.append({'type': 'ineq', 'fun': lambda x: float(problem_data.R0 - (problem_data.r_old[-1] + x[K - 1]))})
    for k in range(K - 1):
        cons.append({'type': 'ineq', 'fun': lambda x, k=k: float((problem_data.r_old[k + 1] + x[k + 1]) - (problem_data.r_old[k] + x[k]) - problem_data.d_min)})
    bounds = [(-tr, tr)] * K + [(0.0, None)]
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': int(cfg.scipy_maxiter), 'ftol': float(cfg.scipy_ftol), 'disp': bool(cfg.solver_verbose)})
    dr_val = np.asarray(res.x[:K], dtype=np.float64)
    delta_val = float(res.x[K])
    r_trial = np.asarray(problem_data.r_old + dr_val, dtype=np.float64)
    status = 'optimal' if bool(res.success) else f'scipy_failed:{res.message}'
    out = Stage1RadiiOptResult(
        status=str(status),
        success=bool(res.success),
        delta_opt=delta_val,
        r_old=np.asarray(problem_data.r_old, dtype=np.float64),
        dr=dr_val,
        r_trial=r_trial,
        predicted_ratio_upper_bound=delta_val,
        max_abs_dr=float(np.max(np.abs(dr_val))) if dr_val.size else 0.0,
        solver_used='SCIPY_SLSQP',
        meta={'backend': 'scipy', 'message': str(res.message), 'nit': int(getattr(res, 'nit', -1)), 'fun': float(getattr(res, 'fun', float('nan')))},
    )
    _log(verbose, f'[RADII][SOLVE][SCIPY] status={out.status} delta={out.delta_opt:.6g} max|dr|={out.max_abs_dr:.6g} nit={out.meta.get("nit")}')
    return out


def solve_radii_socp_step(
    problem_data: Stage1RadiiOptProblemData,
    cfg: Stage1RadiiOptConfig,
    *,
    verbose: bool = True,
) -> Stage1RadiiOptResult:
    _log(verbose, '=== [A0CRA][BLOCK12] SOLVE RADII SOCP STEP: START ===')
    cvx_res = _solve_radii_socp_step_cvxpy(problem_data, cfg, verbose=verbose)
    if cvx_res is not None:
        _log(verbose, '=== [A0CRA][BLOCK12] SOLVE RADII SOCP STEP: DONE (cvxpy) ===')
        return cvx_res
    sc_res = _solve_radii_socp_step_scipy(problem_data, cfg, verbose=verbose)
    _log(verbose, '=== [A0CRA][BLOCK12] SOLVE RADII SOCP STEP: DONE (scipy) ===')
    return sc_res


def apply_trial_radii_to_radius_init(
    radius_init,
    r_trial,
    *,
    feasibility_tol=5e-6,
):
    r = np.asarray(r_trial, dtype=np.float64).reshape(-1)
    validate_radius_init(
        r,
        K=radius_init.spec.K,
        d_min=radius_init.spec.d_min,
        R0=radius_init.spec.R0,
        feasibility_tol=float(feasibility_tol),
    )
    return RadiusInitResult(name=radius_init.name, method=radius_init.method, r_over_lambda=r, spec=radius_init.spec)



def evaluate_trial_radii_on_work(
    iep_ctx: Stage1IEPContext,
    pool_ctx: Stage1PoolContext,
    mainlobe_state: MainlobeDomainState,
    a0_ctx: Stage1A0PolicyContext,
    pre_ctx: Stage1PrecomputeContext,
    radius_trial: RadiusInitResult,
    theta_s_deg_list: np.ndarray,
    *,
    include_dynamic: bool = True,
    rank_by: str = 'ratio_db',
    verbose: bool = True,
) -> Stage1BatchMetricResult:
    update_element_runtime_geometry(pre_ctx, radius_trial, beta=2.0 * np.pi, verbose=verbose)
    return eval_stage1_scan_set(iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, np.asarray(theta_s_deg_list, dtype=np.float64), mode='work', include_dynamic=include_dynamic, rank_by=rank_by, verbose=verbose)


def evaluate_trial_radii_on_full(
    iep_ctx: Stage1IEPContext,
    pool_ctx: Stage1PoolContext,
    mainlobe_state: MainlobeDomainState,
    a0_ctx: Stage1A0PolicyContext,
    pre_ctx: Stage1PrecomputeContext,
    radius_trial: RadiusInitResult,
    theta_s_deg_list: np.ndarray,
    *,
    include_dynamic: bool = True,
    rank_by: str = 'ratio_db',
    verbose: bool = True,
) -> Stage1BatchMetricResult:
    update_element_runtime_geometry(pre_ctx, radius_trial, beta=2.0 * np.pi, verbose=verbose)
    return eval_stage1_scan_set(iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, np.asarray(theta_s_deg_list, dtype=np.float64), mode='full', include_dynamic=include_dynamic, rank_by=rank_by, verbose=verbose)


def audit_radii_solver_result(solve_res, problem_data, cfg):
    tol = float(cfg.radius_feasibility_tol)
    out = {
        'ok': False,
        'reason': '',
        'radius_audit': None,
        'trust_region_violation': float('nan'),
        'feasibility_tol': tol,
    }
    if not bool(solve_res.success):
        out['reason'] = 'solver_unsuccessful'
        return out
    dr = np.asarray(solve_res.dr, dtype=np.float64).reshape(-1)
    r_trial = np.asarray(solve_res.r_trial, dtype=np.float64).reshape(-1)
    if dr.size != problem_data.r_old.size or r_trial.size != problem_data.r_old.size:
        out['reason'] = 'solver_result_length_mismatch'
        return out
    if np.any(~np.isfinite(dr)) or np.any(~np.isfinite(r_trial)):
        out['reason'] = 'solver_result_nonfinite'
        return out
    if not np.isfinite(float(solve_res.delta_opt)) or float(solve_res.delta_opt) < -tol:
        out['reason'] = 'invalid_delta_opt'
        return out
    reconstruction_error = float(np.max(np.abs(r_trial - (np.asarray(problem_data.r_old) + dr)))) if dr.size else 0.0
    if reconstruction_error > tol:
        out['reason'] = 'r_trial_dr_inconsistent'
        out['reconstruction_error'] = reconstruction_error
        return out
    tr = float(problem_data.sigma / problem_data.beta)
    trust_violation = max(0.0, float(np.max(np.abs(dr))) - tr) if dr.size else 0.0
    out['trust_region_violation'] = trust_violation
    if trust_violation > tol:
        out['reason'] = 'trust_region_violation'
        return out
    radius_audit = audit_radius_constraints(
        r_trial,
        K=int(r_trial.size),
        d_min=float(problem_data.d_min),
        R0=float(problem_data.R0),
        feasibility_tol=tol,
    )
    out['radius_audit'] = radius_audit
    if not bool(radius_audit['ok']):
        out['reason'] = 'radius_constraint_violation'
        return out
    out['ok'] = True
    out['reason'] = 'ok'
    return out

def run_stage1_radii_opt_iteration(
    iep_ctx,
    pool_ctx,
    mainlobe_state,
    a0_ctx,
    pre_ctx,
    radius_init,
    work_batch,
    constraint_pool,
    cfg,
    *,
    verbose=True,
):
    """Build and solve one SCA subproblem.

    Block 12 deliberately does not evaluate the trial geometry.  Trial-point
    evaluation, topology changes, fair current/trial comparison, and state
    acceptance are owned by Block 13.
    """
    _log(verbose, '=== [A0CRA][BLOCK12] RUN RADII OPT ITERATION: START ===')
    if pre_ctx.elem_runtime is None:
        update_element_runtime_geometry(pre_ctx, radius_init, beta=2.0 * np.pi, verbose=verbose)
    opt_pre = build_radii_opt_precompute_context(
        constraint_pool, work_batch, pre_ctx.elem_kernel, pre_ctx.elem_runtime,
        radius_init, gamma_main=cfg.gamma_main,
        anchors_per_scan=cfg.anchors_per_scan,
        anchor_merge_alpha_deg=cfg.anchor_merge_alpha_deg,
        verbose=verbose,
    )
    problem_data = build_radii_opt_problem_data(
        radius_init, opt_pre, cfg, beta=float(pre_ctx.elem_runtime.beta), verbose=verbose,
    )
    solve_res = solve_radii_socp_step(problem_data, cfg, verbose=verbose)
    audit = audit_radii_solver_result(solve_res, problem_data, cfg)
    if not bool(audit['ok']):
        _log(verbose, '[RADII][ITER] solver/trial audit failed: %s' % audit['reason'])
        _log(verbose, '=== [A0CRA][BLOCK12] RUN RADII OPT ITERATION: DONE (INVALID) ===')
        return {
            'radii_opt_precomp': opt_pre,
            'problem_data': problem_data,
            'solve_result': solve_res,
            'trial_valid': False,
            'trial_audit': audit,
            'failure_reason': str(audit['reason']),
            'radius_trial': None,
            'work_batch_trial': None,
        }
    radius_trial = apply_trial_radii_to_radius_init(
        radius_init, solve_res.r_trial,
        feasibility_tol=float(cfg.radius_feasibility_tol),
    )
    out = {
        'radii_opt_precomp': opt_pre,
        'problem_data': problem_data,
        'solve_result': solve_res,
        'trial_valid': True,
        'trial_audit': audit,
        'failure_reason': '',
        'radius_trial': radius_trial,
        'work_batch_trial': None,
    }
    _log(verbose, '=== [A0CRA][BLOCK12] RUN RADII OPT ITERATION: DONE ===')
    return out



def finite_difference_af_derivative(
    c_by_ring: List[np.ndarray],
    r_old: np.ndarray,
    beta: float,
    k: int,
    *,
    include_center: bool,
    eps: float = 1e-7,
) -> complex:
    r_p = np.asarray(r_old, dtype=np.float64).copy()
    r_m = np.asarray(r_old, dtype=np.float64).copy()
    r_p[int(k)] += float(eps)
    r_m[int(k)] -= float(eps)
    af_p = _reconstruct_af_from_c_by_ring(c_by_ring, r_p, beta, include_center)
    af_m = _reconstruct_af_from_c_by_ring(c_by_ring, r_m, beta, include_center)
    return complex((af_p - af_m) / (2.0 * float(eps)))


def finite_difference_af_derivative_bessel_stage1(
    rho: float,
    eta_rad: float,
    zone_backend: str,
    r_over_lambda: np.ndarray,
    N_per_ring: np.ndarray,
    k: int,
    *,
    beta: float = 2.0 * np.pi,
    include_center: bool = True,
    eps: float = 1e-6,
) -> complex:
    r0 = np.asarray(r_over_lambda, dtype=np.float64).reshape(-1)
    rp = r0.copy()
    rm = r0.copy()
    kk = int(k)
    rp[kk] += float(eps)
    rm[kk] -= float(eps)
    rho_arr = np.asarray([float(rho)], dtype=np.float64)
    eta_arr = np.asarray([float(eta_rad)], dtype=np.float64)
    if zone_backend == 'core_j0':
        af_p = eval_af_core_j0_collapsed(rho_arr, rp, N_per_ring, beta=beta, include_center=include_center)[0]
        af_m = eval_af_core_j0_collapsed(rho_arr, rm, N_per_ring, beta=beta, include_center=include_center)[0]
    elif zone_backend == 'side_bessel_L1':
        af_p = eval_af_bessel_alias_L1(rho_arr, eta_arr, rp, N_per_ring, beta=beta, include_center=include_center)[0]
        af_m = eval_af_bessel_alias_L1(rho_arr, eta_arr, rm, N_per_ring, beta=beta, include_center=include_center)[0]
    else:
        raise ValueError(f'unknown zone_backend={zone_backend}')
    return complex((af_p - af_m) / (2.0 * float(eps)))


__all__ += [
    'Stage1RadiiOptConfig',
    'Stage1RadiiOptAnchor',
    'Stage1RadiiOptPrecomputeContext',
    'Stage1RadiiOptProblemData',
    'Stage1RadiiOptResult',
    'build_radii_opt_precompute_context',
    'build_radii_opt_problem_data',
    'solve_radii_socp_step',
    'apply_trial_radii_to_radius_init',
    'evaluate_trial_radii_on_work',
    'evaluate_trial_radii_on_full',
    'run_stage1_radii_opt_iteration',
    'finite_difference_af_derivative',
    'finite_difference_af_derivative_bessel_stage1',
]


# =============================================================================
# Block 13: Stage-1 main orchestration
# =============================================================================

import csv as _csv
import os as _os
import time as _time
import sys as _sys
import contextlib as _contextlib


class Stage1MainConfig(object):
    __slots__ = (
        'max_iters', 'max_full_scouts', 'max_topology_epochs',
        'sigma_init', 'sigma_min', 'sigma_max', 'sigma_expand', 'sigma_shrink',
        'sigma_expand_improvement_db', 'work_accept_tol_db',
        'full_scout_every_accepts', 'full_improve_tol_db',
        'full_no_improve_limit', 'full_rollback_threshold_db',
        'consecutive_reject_limit', 'sigma_min_reject_limit',
        'topology_cycle_window', 'topology_cycle_full_tol_db',
        'force_final_full_audit',
        'log_level', 'log_to_file', 'save_attempt_csv', 'save_full_csv',
        'save_history_json', 'verbose',
        # Backward-compatible fields retained but not used by the new loop.
        'topology_refresh_every_attempts', 'grace_psll_tol_db',
        'patience_no_improve', 'oscillation_window',
        'oscillation_psll_tol_db', 'oscillation_radius_tol',
        'save_trial_history', 'save_full_history',
    )

    def __init__(
        self,
        max_iters=20,
        max_full_scouts=20,
        max_topology_epochs=12,
        sigma_init=0.15,
        sigma_min=0.01,
        sigma_max=0.30,
        sigma_expand=1.20,
        sigma_shrink=0.50,
        sigma_expand_improvement_db=0.05,
        work_accept_tol_db=1e-4,
        full_scout_every_accepts=2,
        full_improve_tol_db=0.02,
        full_no_improve_limit=3,
        full_rollback_threshold_db=0.50,
        consecutive_reject_limit=6,
        sigma_min_reject_limit=2,
        topology_cycle_window=4,
        topology_cycle_full_tol_db=0.02,
        force_final_full_audit=True,
        log_level='compact',
        log_to_file=True,
        save_attempt_csv=True,
        save_full_csv=True,
        save_history_json=True,
        verbose=True,
        topology_refresh_every_attempts=None,
        grace_psll_tol_db=None,
        patience_no_improve=None,
        oscillation_window=None,
        oscillation_psll_tol_db=None,
        oscillation_radius_tol=None,
        save_trial_history=True,
        save_full_history=True,
    ):
        self.max_iters = int(max_iters)
        self.max_full_scouts = int(max_full_scouts)
        self.max_topology_epochs = int(max_topology_epochs)
        self.sigma_init = float(sigma_init)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_expand = float(sigma_expand)
        self.sigma_shrink = float(sigma_shrink)
        self.sigma_expand_improvement_db = float(sigma_expand_improvement_db)
        self.work_accept_tol_db = float(work_accept_tol_db)
        self.full_scout_every_accepts = int(full_scout_every_accepts)
        self.full_improve_tol_db = float(full_improve_tol_db)
        self.full_no_improve_limit = int(full_no_improve_limit)
        self.full_rollback_threshold_db = float(full_rollback_threshold_db)
        self.consecutive_reject_limit = int(consecutive_reject_limit)
        self.sigma_min_reject_limit = int(sigma_min_reject_limit)
        self.topology_cycle_window = int(topology_cycle_window)
        self.topology_cycle_full_tol_db = float(topology_cycle_full_tol_db)
        self.force_final_full_audit = bool(force_final_full_audit)
        self.log_level = str(log_level).lower().strip()
        self.log_to_file = bool(log_to_file)
        self.save_attempt_csv = bool(save_attempt_csv)
        self.save_full_csv = bool(save_full_csv)
        self.save_history_json = bool(save_history_json)
        self.verbose = bool(verbose)
        self.topology_refresh_every_attempts = topology_refresh_every_attempts
        self.grace_psll_tol_db = grace_psll_tol_db
        self.patience_no_improve = patience_no_improve
        self.oscillation_window = oscillation_window
        self.oscillation_psll_tol_db = oscillation_psll_tol_db
        self.oscillation_radius_tol = oscillation_radius_tol
        self.save_trial_history = bool(save_trial_history)
        self.save_full_history = bool(save_full_history)
        if self.log_level not in ('silent', 'compact', 'debug'):
            raise ValueError("log_level must be 'silent', 'compact', or 'debug'")
        if self.max_iters < 0:
            raise ValueError('max_iters must be >= 0')
        if self.sigma_min <= 0.0 or self.sigma_init <= 0.0 or self.sigma_max <= 0.0:
            raise ValueError('sigma values must be > 0')
        if self.sigma_min > self.sigma_max:
            raise ValueError('sigma_min must be <= sigma_max')


class Stage1FullCheckpoint(object):
    __slots__ = (
        'tag', 'checkpoint_id', 'reason', 'iter_idx', 'attempt_counter',
        'accept_counter', 'topology_epoch', 'radius_result',
        'elem_counts_result', 'elem_geom_result', 'mainlobe_state',
        'work_theta_s_deg', 'batch_result', 'sigma_value',
        'topology_signature', 'work_signature', 'extra',
    )

    def __init__(
        self, tag, checkpoint_id, reason, iter_idx, attempt_counter,
        accept_counter, topology_epoch, radius_result, elem_counts_result,
        elem_geom_result, mainlobe_state, work_theta_s_deg, batch_result,
        sigma_value, topology_signature, work_signature, extra,
    ):
        self.tag = str(tag)
        self.checkpoint_id = int(checkpoint_id)
        self.reason = str(reason)
        self.iter_idx = int(iter_idx)
        self.attempt_counter = int(attempt_counter)
        self.accept_counter = int(accept_counter)
        self.topology_epoch = int(topology_epoch)
        self.radius_result = radius_result
        self.elem_counts_result = elem_counts_result
        self.elem_geom_result = elem_geom_result
        self.mainlobe_state = mainlobe_state
        self.work_theta_s_deg = np.asarray(work_theta_s_deg, dtype=np.float64)
        self.batch_result = batch_result
        self.sigma_value = float(sigma_value)
        self.topology_signature = tuple(int(x) for x in topology_signature)
        self.work_signature = tuple(float(x) for x in work_signature)
        self.extra = dict(extra or {})


# Compatibility alias.  It now means a full-validated checkpoint.
Stage1BestSnapshot = Stage1FullCheckpoint


class Stage1MainState(object):
    __slots__ = (
        'iter_idx', 'attempt_counter', 'accept_counter', 'topology_epoch',
        'full_scout_count', 'accepted_since_full', 'consecutive_rejects',
        'rejects_at_sigma_min', 'full_no_improve_count', 'rollback_count',
        'sigma_current', 'radius_current', 'elem_counts_current',
        'elem_geom_current', 'mainlobe_state_current', 'current_work_batch',
        'current_full_batch', 'current_constraint_pool',
        'current_full_validated', 'last_full_snapshot', 'best_full_snapshot',
        'work_signature', 'topology_signature', 'full_signature_history',
        'topology_signature_history', 'attempt_history', 'full_history',
        'topology_history', 'stop_reason', 'stop_detail',
    )

    def __init__(
        self, iter_idx, attempt_counter, accept_counter, topology_epoch,
        full_scout_count, accepted_since_full, consecutive_rejects,
        rejects_at_sigma_min, full_no_improve_count, rollback_count,
        sigma_current, radius_current, elem_counts_current, elem_geom_current,
        mainlobe_state_current, current_work_batch, current_full_batch,
        current_constraint_pool, current_full_validated, last_full_snapshot,
        best_full_snapshot, work_signature, topology_signature,
        full_signature_history, topology_signature_history, attempt_history,
        full_history, topology_history, stop_reason='', stop_detail=None,
    ):
        self.iter_idx = int(iter_idx)
        self.attempt_counter = int(attempt_counter)
        self.accept_counter = int(accept_counter)
        self.topology_epoch = int(topology_epoch)
        self.full_scout_count = int(full_scout_count)
        self.accepted_since_full = int(accepted_since_full)
        self.consecutive_rejects = int(consecutive_rejects)
        self.rejects_at_sigma_min = int(rejects_at_sigma_min)
        self.full_no_improve_count = int(full_no_improve_count)
        self.rollback_count = int(rollback_count)
        self.sigma_current = float(sigma_current)
        self.radius_current = radius_current
        self.elem_counts_current = elem_counts_current
        self.elem_geom_current = elem_geom_current
        self.mainlobe_state_current = mainlobe_state_current
        self.current_work_batch = current_work_batch
        self.current_full_batch = current_full_batch
        self.current_constraint_pool = current_constraint_pool
        self.current_full_validated = bool(current_full_validated)
        self.last_full_snapshot = last_full_snapshot
        self.best_full_snapshot = best_full_snapshot
        self.work_signature = tuple(work_signature)
        self.topology_signature = tuple(topology_signature)
        self.full_signature_history = list(full_signature_history)
        self.topology_signature_history = list(topology_signature_history)
        self.attempt_history = list(attempt_history)
        self.full_history = list(full_history)
        self.topology_history = list(topology_history)
        self.stop_reason = str(stop_reason)
        self.stop_detail = dict(stop_detail or {})

    @property
    def history(self):
        return self.attempt_history

    @property
    def best_work_snapshot(self):
        return None


class Stage1MainRunResult(object):
    __slots__ = (
        'state_final', 'best_work_snapshot', 'best_full_snapshot',
        'final_output_radius', 'final_output_counts', 'final_output_geom',
        'final_output_mainlobe', 'final_null_mainlobe_deg',
        'final_guard_null_mainlobe_deg', 'final_3db_mainlobe_deg',
        'output_npz_path', 'output_json_path', 'output_attempt_csv_path',
        'output_full_csv_path', 'output_log_path',
    )

    def __init__(
        self, state_final, best_work_snapshot, best_full_snapshot,
        final_output_radius, final_output_counts, final_output_geom,
        final_output_mainlobe, final_null_mainlobe_deg,
        final_guard_null_mainlobe_deg, final_3db_mainlobe_deg,
        output_npz_path, output_json_path, output_attempt_csv_path='',
        output_full_csv_path='', output_log_path='',
    ):
        self.state_final = state_final
        self.best_work_snapshot = best_work_snapshot
        self.best_full_snapshot = best_full_snapshot
        self.final_output_radius = final_output_radius
        self.final_output_counts = final_output_counts
        self.final_output_geom = final_output_geom
        self.final_output_mainlobe = final_output_mainlobe
        self.final_null_mainlobe_deg = final_null_mainlobe_deg
        self.final_guard_null_mainlobe_deg = final_guard_null_mainlobe_deg
        self.final_3db_mainlobe_deg = final_3db_mainlobe_deg
        self.output_npz_path = output_npz_path
        self.output_json_path = output_json_path
        self.output_attempt_csv_path = output_attempt_csv_path
        self.output_full_csv_path = output_full_csv_path
        self.output_log_path = output_log_path


class Stage1CompactLogger(object):
    def __init__(self, path='', enabled=True, echo=True):
        self.path = str(path or '')
        self.enabled = bool(enabled)
        self.echo = bool(echo)
        self._file = None
        if self.enabled and self.path:
            parent = _os.path.dirname(_os.path.abspath(self.path))
            if parent and not _os.path.isdir(parent):
                _os.makedirs(parent)
            self._file = open(self.path, 'w', encoding='utf-8')

    def write(self, msg):
        text = str(msg)
        if self.echo:
            _sys.__stdout__.write(text + '\n')
            _sys.__stdout__.flush()
        if self._file is not None:
            self._file.write(text + '\n')
            self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


# ----------------------------- cloning / summaries ----------------------------


def _copy_ndarray_dict(d):
    return {str(k): np.asarray(v).copy() for k, v in d.items()}


def _copy_nested_ndarray_dict(d):
    return {float(k): _copy_ndarray_dict(v) for k, v in d.items()}


def _clone_radius_result(radius_result):
    spec = radius_result.spec
    return RadiusInitResult(
        name=str(radius_result.name),
        method=str(radius_result.method),
        r_over_lambda=np.asarray(radius_result.r_over_lambda, dtype=np.float64).copy(),
        spec=RadiusInitSpec(
            name=str(spec.name), method=str(spec.method), K=int(spec.K),
            R0=float(spec.R0), d_min=float(spec.d_min), b=float(spec.b),
        ),
    )


def _clone_ring_count_result(elem_counts):
    cfg = elem_counts.config
    return RingElementCountResult(
        N_per_ring=np.asarray(elem_counts.N_per_ring, dtype=np.int64).copy(),
        config=ElementInitConfig(
            d_elem_default_lambda=float(cfg.d_elem_default_lambda),
            include_center=bool(cfg.include_center),
            apply_outer5_mod4_minus1=bool(cfg.apply_outer5_mod4_minus1),
            outer5_threshold_gt=int(cfg.outer5_threshold_gt),
        ),
        include_center=bool(elem_counts.include_center),
    )


def _clone_element_geometry(elem_geom):
    return ElementGeometry(
        x_elem=np.asarray(elem_geom.x_elem, dtype=np.float64).copy(),
        y_elem=np.asarray(elem_geom.y_elem, dtype=np.float64).copy(),
        ring_idx=np.asarray(elem_geom.ring_idx, dtype=np.int64).copy(),
        elem_idx_in_ring=np.asarray(elem_geom.elem_idx_in_ring, dtype=np.int64).copy(),
        N_per_ring=np.asarray(elem_geom.N_per_ring, dtype=np.int64).copy(),
        r_over_lambda=np.asarray(elem_geom.r_over_lambda, dtype=np.float64).copy(),
        include_center=bool(elem_geom.include_center),
    )


def _clone_mainlobe_state(state):
    return MainlobeDomainState(
        theta_s_deg=np.asarray(state.theta_s_deg, dtype=np.float64).copy(),
        phi_s_deg=float(state.phi_s_deg),
        alpha_null_deg=np.asarray(state.alpha_null_deg, dtype=np.float64).copy(),
        alpha_guard_null_deg=np.asarray(state.alpha_guard_null_deg, dtype=np.float64).copy(),
        alpha_3db_deg=np.asarray(state.alpha_3db_deg, dtype=np.float64).copy(),
        use_domain=str(state.use_domain),
        stats=dict(state.stats),
    )


def _batch_summary_dict(batch):
    if batch is None:
        return None
    return {
        'mode': str(batch.mode),
        'include_dynamic': bool(batch.include_dynamic),
        'theta_s_deg': np.asarray(batch.theta_s_deg, dtype=np.float64).tolist(),
        'psll_db': np.asarray(batch.psll_db, dtype=np.float64).tolist(),
        'worst_theta_s_deg': float(batch.worst_theta_s_deg),
        'worst_psll_db': float(batch.worst_psll_db),
        'meta': dict(batch.meta),
    }


def _checkpoint_summary_dict(snap):
    if snap is None:
        return None
    return {
        'tag': str(snap.tag),
        'checkpoint_id': int(snap.checkpoint_id),
        'reason': str(snap.reason),
        'iter_idx': int(snap.iter_idx),
        'attempt_counter': int(snap.attempt_counter),
        'accept_counter': int(snap.accept_counter),
        'topology_epoch': int(snap.topology_epoch),
        'sigma_value': float(snap.sigma_value),
        'radius': np.asarray(snap.radius_result.r_over_lambda, dtype=np.float64).tolist(),
        'N_per_ring': np.asarray(snap.elem_counts_result.N_per_ring, dtype=np.int64).tolist(),
        'work_theta_s_deg': np.asarray(snap.work_theta_s_deg, dtype=np.float64).tolist(),
        'topology_signature': list(snap.topology_signature),
        'work_signature': list(snap.work_signature),
        'batch_result': _batch_summary_dict(snap.batch_result),
        'extra': dict(snap.extra),
    }


def _theta_signature(theta):
    arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    return tuple(round(float(x), 9) for x in arr.tolist())


def _topology_signature(counts):
    return tuple(int(x) for x in np.asarray(counts.N_per_ring, dtype=np.int64).tolist())


def _same_theta_set(a, b, tol=1e-9):
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    return aa.shape == bb.shape and bool(np.allclose(aa, bb, atol=float(tol), rtol=0.0))


def _make_full_checkpoint(state, batch_result, reason, pool_ctx, extra=None):
    geom = build_element_geometry(
        state.radius_current.r_over_lambda,
        state.elem_counts_current.N_per_ring,
        include_center=state.elem_counts_current.include_center,
        verbose=False,
    )
    return Stage1FullCheckpoint(
        tag='full',
        checkpoint_id=int(state.full_scout_count),
        reason=str(reason),
        iter_idx=int(state.iter_idx),
        attempt_counter=int(state.attempt_counter),
        accept_counter=int(state.accept_counter),
        topology_epoch=int(state.topology_epoch),
        radius_result=_clone_radius_result(state.radius_current),
        elem_counts_result=_clone_ring_count_result(state.elem_counts_current),
        elem_geom_result=_clone_element_geometry(geom),
        mainlobe_state=_clone_mainlobe_state(state.mainlobe_state_current),
        work_theta_s_deg=get_current_work_theta_s_deg(pool_ctx),
        batch_result=batch_result,
        sigma_value=float(state.sigma_current),
        topology_signature=_topology_signature(state.elem_counts_current),
        work_signature=_theta_signature(get_current_work_theta_s_deg(pool_ctx)),
        extra=dict(extra or {}),
    )


# ----------------------------- dynamic-point state ----------------------------


def _clear_all_dynamic_precompute_mode(pre_ctx, mode):
    if mode == 'work':
        pre_ctx.buffer_a0_dynamic_work_by_scan.clear()
        pre_ctx.outer_a0_dynamic_work_by_scan.clear()
        pre_ctx.eco_elem_dynamic_work_by_scan_zone.clear()
        pre_ctx.af_kernel_dynamic_work_by_scan_zone.clear()
        return
    if mode == 'full':
        pre_ctx.buffer_a0_dynamic_full_by_scan.clear()
        pre_ctx.outer_a0_dynamic_full_by_scan.clear()
        pre_ctx.eco_elem_dynamic_full_by_scan_zone.clear()
        pre_ctx.af_kernel_dynamic_full_by_scan_zone.clear()
        return
    raise ValueError("mode must be 'work' or 'full'")


def _copy_bessel_point_kernel(k):
    if isinstance(k, Stage1BesselPointKernel):
        return Stage1BesselPointKernel(
            theta_s_deg=float(k.theta_s_deg),
            zone_name=str(k.zone_name),
            rho=np.asarray(k.rho, dtype=np.float64).copy(),
            eta_rad=np.asarray(k.eta_rad, dtype=np.float64).copy(),
        )
    if isinstance(k, np.ndarray):
        return np.asarray(k).copy()
    return k


def _copy_af_kernel_store(d):
    return {(float(k1), str(k2)): _copy_bessel_point_kernel(v) for (k1, k2), v in d.items()}


def _backup_work_dynamic_state(pool_ctx, pre_ctx=None):
    out = {
        'pool_buffer': _copy_nested_ndarray_dict(pool_ctx.dynamic_buffer_work_by_scan),
        'pool_outer': _copy_nested_ndarray_dict(pool_ctx.dynamic_outer_work_by_scan),
    }
    if pre_ctx is not None:
        out.update({
            'a0_buffer': {float(k): np.asarray(v).copy() for k, v in pre_ctx.buffer_a0_dynamic_work_by_scan.items()},
            'a0_outer': {float(k): np.asarray(v).copy() for k, v in pre_ctx.outer_a0_dynamic_work_by_scan.items()},
            'eco': {(float(k1), str(k2)): np.asarray(v).copy() for (k1, k2), v in pre_ctx.eco_elem_dynamic_work_by_scan_zone.items()},
            'afk': _copy_af_kernel_store(pre_ctx.af_kernel_dynamic_work_by_scan_zone),
        })
    return out


def _install_work_dynamic_pool_state(pool_ctx, state_dict):
    pool_ctx.dynamic_buffer_work_by_scan = _copy_nested_ndarray_dict(state_dict.get('pool_buffer', {}))
    pool_ctx.dynamic_outer_work_by_scan = _copy_nested_ndarray_dict(state_dict.get('pool_outer', {}))


def _restore_work_dynamic_state(pool_ctx, pre_ctx, backup):
    _install_work_dynamic_pool_state(pool_ctx, backup)
    if 'a0_buffer' in backup:
        pre_ctx.buffer_a0_dynamic_work_by_scan = {float(k): np.asarray(v).copy() for k, v in backup['a0_buffer'].items()}
        pre_ctx.outer_a0_dynamic_work_by_scan = {float(k): np.asarray(v).copy() for k, v in backup['a0_outer'].items()}
        pre_ctx.eco_elem_dynamic_work_by_scan_zone = {(float(k1), str(k2)): np.asarray(v).copy() for (k1, k2), v in backup['eco'].items()}
        pre_ctx.af_kernel_dynamic_work_by_scan_zone = _copy_af_kernel_store(backup['afk'])
    else:
        _refresh_dynamic_precompute_for_theta_list(
            pre_ctx, None, None, pool_ctx, np.asarray([], dtype=np.float64),
            mode='work', verbose=False,
        )


def _merge_work_dynamic_states(current_state, trial_state, theta_s_deg_list, pool_ctx):
    buf = {}
    out = {}
    for th in np.asarray(theta_s_deg_list, dtype=np.float64).reshape(-1).tolist():
        key = _canonical_theta_deg(float(th), pool_ctx.tol_deg)
        a = current_state.get('pool_buffer', {}).get(key)
        b = trial_state.get('pool_buffer', {}).get(key)
        c = current_state.get('pool_outer', {}).get(key)
        d = trial_state.get('pool_outer', {}).get(key)
        mb = _merge_dynamic_point_dicts(a, b)
        mo = _merge_dynamic_point_dicts(c, d)
        if mb:
            buf[key] = mb
        if mo:
            out[key] = mo
    return {'pool_buffer': buf, 'pool_outer': out}


def _rebuild_dynamic_points_from_seed_map(
    pool_ctx, seed_map, *, mode, round_id, clear_existing, verbose=False,
):
    if clear_existing:
        if mode == 'work':
            clear_all_work_dynamic_points(pool_ctx, verbose=verbose)
        elif mode == 'full':
            clear_all_full_dynamic_points(pool_ctx, verbose=verbose)
        else:
            raise ValueError("mode must be 'work' or 'full'")
    for theta_s_deg, seeds in seed_map.items():
        kwargs = dict(
            buffer_seed_alpha_deg=seeds.buffer_seed_alpha_deg,
            buffer_seed_eta_deg=seeds.buffer_seed_eta_deg,
            outer_seed_alpha_deg=seeds.outer_seed_alpha_deg,
            outer_seed_eta_deg=seeds.outer_seed_eta_deg,
            round_id=int(round_id), clear_existing=False, verbose=verbose,
        )
        if mode == 'work':
            rebuild_work_dynamic_points_for_scan(pool_ctx, float(theta_s_deg), **kwargs)
        else:
            rebuild_full_dynamic_points_for_scan(pool_ctx, float(theta_s_deg), **kwargs)


def _refresh_dynamic_precompute_for_theta_list(
    pre_ctx, iep_ctx, a0_ctx, pool_ctx, theta_s_deg_list, *, mode, verbose=False,
):
    _clear_all_dynamic_precompute_mode(pre_ctx, mode)
    for th in np.asarray(theta_s_deg_list, dtype=np.float64).reshape(-1).tolist():
        refresh_dynamic_precompute_for_scan(
            pre_ctx, iep_ctx, a0_ctx, pool_ctx, float(th), mode=mode, verbose=verbose,
        )


def _bootstrap_dynamic_mode(
    iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx,
    theta_s_deg_list, *, mode, verbose=False,
):
    theta_list = np.asarray(theta_s_deg_list, dtype=np.float64).reshape(-1)
    if theta_list.size == 0:
        raise ValueError('theta_s_deg_list must be non-empty')
    if mode == 'work':
        clear_all_work_dynamic_points(pool_ctx, verbose=verbose)
    elif mode == 'full':
        clear_all_full_dynamic_points(pool_ctx, verbose=verbose)
    else:
        raise ValueError("mode must be 'work' or 'full'")
    _clear_all_dynamic_precompute_mode(pre_ctx, mode)
    static_batch = eval_stage1_scan_set(
        iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, theta_list,
        mode=mode, include_dynamic=False, rank_by='ratio_db', verbose=verbose,
    )
    seed_map_1 = extract_work_tracking_seed_map(static_batch, pool_ctx.tracking_protocol, verbose=verbose)
    _rebuild_dynamic_points_from_seed_map(
        pool_ctx, seed_map_1, mode=mode, round_id=1,
        clear_existing=True, verbose=verbose,
    )
    _refresh_dynamic_precompute_for_theta_list(
        pre_ctx, iep_ctx, a0_ctx, pool_ctx, theta_list, mode=mode, verbose=verbose,
    )
    round1_batch = eval_stage1_scan_set(
        iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, theta_list,
        mode=mode, include_dynamic=True, rank_by='ratio_db', verbose=verbose,
    )
    seed_map_2 = extract_work_tracking_seed_map(round1_batch, pool_ctx.tracking_protocol, verbose=verbose)
    _rebuild_dynamic_points_from_seed_map(
        pool_ctx, seed_map_2, mode=mode, round_id=2,
        clear_existing=False, verbose=verbose,
    )
    _refresh_dynamic_precompute_for_theta_list(
        pre_ctx, iep_ctx, a0_ctx, pool_ctx, theta_list, mode=mode, verbose=verbose,
    )
    dynamic_batch = eval_stage1_scan_set(
        iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, theta_list,
        mode=mode, include_dynamic=True, rank_by='ratio_db', verbose=verbose,
    )
    dynamic_batch.meta['tracking_round1_worst_psll_db'] = float(round1_batch.worst_psll_db)
    return static_batch, seed_map_2, dynamic_batch


def _compact_accepted_work_dynamic_state(
    iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, radius_current,
    comparison_batch, *, verbose=False,
):
    theta_list = np.asarray(comparison_batch.theta_s_deg, dtype=np.float64)
    seed_map_1 = extract_work_tracking_seed_map(
        comparison_batch, pool_ctx.tracking_protocol, verbose=verbose,
    )
    _rebuild_dynamic_points_from_seed_map(
        pool_ctx, seed_map_1, mode='work', round_id=1,
        clear_existing=True, verbose=verbose,
    )
    update_element_runtime_geometry(pre_ctx, radius_current, beta=2.0 * np.pi, verbose=verbose)
    _refresh_dynamic_precompute_for_theta_list(
        pre_ctx, iep_ctx, a0_ctx, pool_ctx, theta_list, mode='work', verbose=verbose,
    )
    round1 = eval_stage1_scan_set(
        iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, theta_list,
        mode='work', include_dynamic=True, rank_by='ratio_db', verbose=verbose,
    )
    seed_map_2 = extract_work_tracking_seed_map(round1, pool_ctx.tracking_protocol, verbose=verbose)
    _rebuild_dynamic_points_from_seed_map(
        pool_ctx, seed_map_2, mode='work', round_id=2,
        clear_existing=False, verbose=verbose,
    )
    _refresh_dynamic_precompute_for_theta_list(
        pre_ctx, iep_ctx, a0_ctx, pool_ctx, theta_list, mode='work', verbose=verbose,
    )
    final = eval_stage1_scan_set(
        iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, theta_list,
        mode='work', include_dynamic=True, rank_by='ratio_db', verbose=verbose,
    )
    final.meta['compacted_after_union'] = True
    final.meta['tracking_round1_worst_psll_db'] = float(round1.worst_psll_db)
    return final


# ----------------------------- work/full management ---------------------------


def _select_full_feedback_theta(batch_result, pool_ctx):
    n = int(pool_ctx.tracking_protocol.get('reinject_worst_theta_count', 2))
    theta = np.asarray(batch_result.theta_s_deg, dtype=np.float64)
    psll = np.asarray(batch_result.psll_db, dtype=np.float64)
    valid = ~np.isnan(psll)
    if not np.any(valid) or n <= 0:
        return np.asarray([], dtype=np.float64)
    valid_idx = np.flatnonzero(valid)
    order = np.lexsort((valid_idx, -psll[valid_idx]))
    # Angle NMS avoids filling work with adjacent 1-degree scans.
    selected = []
    min_sep = float(pool_ctx.tracking_protocol.get('reinject_theta_min_sep_deg', 1.5))
    for idx in valid_idx[order].tolist():
        th = float(theta[int(idx)])
        if not np.isfinite(th):
            continue
        if any(abs(th - old) < min_sep - 1e-12 for old in selected):
            continue
        selected.append(th)
        if len(selected) >= n:
            break
    arr = np.asarray(sorted(set(selected)), dtype=np.float64)
    return arr


def set_work_scan_angles_from_full_feedback(pool_ctx, feedback_theta_s_deg, verbose=False):
    old_work = get_current_work_theta_s_deg(pool_ctx)
    anchors = np.asarray(pool_ctx.work_anchor_theta_s_deg, dtype=np.float64).reshape(-1)
    add = np.asarray(feedback_theta_s_deg, dtype=np.float64).reshape(-1)
    add = np.unique(add[np.isfinite(add)])
    add.sort()
    max_size = int(pool_ctx.tracking_protocol.get('work_max_size', max(old_work.size, anchors.size)))
    extras = []
    for x in add.tolist():
        if np.any(np.isclose(anchors, float(x), atol=pool_ctx.tol_deg, rtol=0.0)):
            continue
        if not any(abs(float(x) - y) <= pool_ctx.tol_deg for y in extras):
            extras.append(float(x))
    extras = extras[:max(0, max_size - int(anchors.size))]
    new_work = np.asarray(sorted(set(anchors.tolist() + extras)), dtype=np.float64)
    pool_ctx.current_work_theta_s_deg = new_work
    changed = not _same_theta_set(old_work, new_work, pool_ctx.tol_deg)
    added = [float(x) for x in new_work.tolist() if not np.any(np.isclose(old_work, x, atol=pool_ctx.tol_deg, rtol=0.0))]
    removed = [float(x) for x in old_work.tolist() if not np.any(np.isclose(new_work, x, atol=pool_ctx.tol_deg, rtol=0.0))]
    _log(verbose, '[POOL][WORK] replaced -> %s' % new_work.tolist())
    return {
        'old_work': old_work, 'new_work': new_work, 'changed': bool(changed),
        'added': added, 'removed': removed,
    }


def bootstrap_stage1_full_then_work(
    iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, *, verbose=False,
):
    full_theta = get_full_theta_s_deg(pool_ctx)
    _, _, full_batch = _bootstrap_dynamic_mode(
        iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx,
        full_theta, mode='full', verbose=verbose,
    )
    feedback = _select_full_feedback_theta(full_batch, pool_ctx)
    work_change = set_work_scan_angles_from_full_feedback(pool_ctx, feedback, verbose=verbose)
    work_theta = get_current_work_theta_s_deg(pool_ctx)
    _, _, work_batch = _bootstrap_dynamic_mode(
        iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx,
        work_theta, mode='work', verbose=verbose,
    )
    return {
        'full_batch': full_batch,
        'work_batch': work_batch,
        'feedback_theta': feedback,
        'work_change': work_change,
        'work_signature': _theta_signature(work_theta),
    }


# Backward-compatible helper name, now full-first.
def bootstrap_stage1_baselines(
    iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, *, verbose=True,
):
    out = bootstrap_stage1_full_then_work(
        iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, verbose=verbose,
    )
    return out['work_batch'], out['full_batch']


def _make_mainlobe_cfg_from_pool(pool_ctx, template_cfg):
    return MainlobeDomainConfig(
        theta_s_deg_list=get_full_theta_s_deg(pool_ctx),
        phi_s_deg=float(template_cfg.phi_s_deg),
        use_domain=str(template_cfg.use_domain),
        source=str(template_cfg.source),
        final_source=str(template_cfg.final_source),
        freeze_during_optimization=bool(template_cfg.freeze_during_optimization),
        alpha_table_json_path=str(template_cfg.alpha_table_json_path),
        alpha_table_tol=float(template_cfg.alpha_table_tol),
        mainlobe_margin_deg=float(template_cfg.mainlobe_margin_deg),
        solver_cfg=template_cfg.solver_cfg,
    )


def validate_stage1_geometry_runtime_config(
    *, radius_spec, element_cfg, radii_opt_cfg,
    static_pool_summary=None, bind_d_eff_to_d_min=True, tol=1e-9,
):
    issues = []
    K = int(radius_spec.K)
    R0 = float(radius_spec.R0)
    d_min = float(radius_spec.d_min)
    d_eff = float(element_cfg.d_elem_default_lambda)
    d_min_opt = float(radii_opt_cfg.d_min)
    b = float(radius_spec.b)
    if K <= 0:
        issues.append('radius_spec.K must be positive')
    if R0 <= 0.0:
        issues.append('radius_spec.R0 must be positive')
    if d_min <= 0.0:
        issues.append('radius_spec.d_min must be positive')
    if d_eff <= 0.0:
        issues.append('element_cfg.d_elem_default_lambda must be positive')
    if R0 < K * d_min - float(tol):
        issues.append('infeasible radii geometry: R0=%s < K*d_min=%s' % (R0, K * d_min))
    if abs(d_min - d_min_opt) > float(tol):
        issues.append('radius_spec.d_min=%s != radii_opt_cfg.d_min=%s' % (d_min, d_min_opt))
    if bool(bind_d_eff_to_d_min) and abs(d_eff - d_min) > float(tol):
        issues.append('d_eff=%s is not bound to d_min=%s' % (d_eff, d_min))
    pool_check = {}
    if static_pool_summary is not None:
        partition_info = dict(static_pool_summary.get('partition_info', {}) or {})
        pool_config = dict(static_pool_summary.get('config', {}) or {})
        pool_K = partition_info.get('K', pool_config.get('partition_lookup_K'))
        pool_d_min = partition_info.get('d_min', pool_config.get('partition_lookup_d_min'))
        pool_d_eff = partition_info.get('d_eff', pool_config.get('partition_lookup_d_eff'))
        pool_R0 = partition_info.get('R0', pool_config.get('partition_lookup_R0'))
        pool_check = {
            'pool_K': pool_K, 'pool_d_min': pool_d_min,
            'pool_d_eff': pool_d_eff, 'pool_R0': pool_R0,
            'pool_alpha_core_used_deg': partition_info.get('alpha_core_used_deg'),
            'pool_alpha_outer_start_deg': partition_info.get('alpha_outer_start_deg'),
            'partition_mode': partition_info.get('selection_mode', partition_info.get('mode', 'missing')),
        }
        if pool_K is not None and int(pool_K) != K:
            issues.append('static pool K=%s != runtime K=%s' % (pool_K, K))
        if pool_d_min is not None and abs(float(pool_d_min) - d_min) > float(tol):
            issues.append('static pool d_min=%s != runtime d_min=%s' % (pool_d_min, d_min))
        if pool_d_eff is not None and abs(float(pool_d_eff) - d_eff) > float(tol):
            issues.append('static pool d_eff=%s != runtime d_eff=%s' % (pool_d_eff, d_eff))
        if pool_R0 is not None and abs(float(pool_R0) - R0) > float(tol):
            issues.append('static pool R0=%s != runtime R0=%s' % (pool_R0, R0))
    return {
        'ok': len(issues) == 0, 'issues': issues, 'K': K, 'R0': R0,
        'd_min': d_min, 'd_eff': d_eff, 'b': b,
        'radii_opt_d_min': d_min_opt,
        'bind_d_eff_to_d_min': bool(bind_d_eff_to_d_min),
        'pool_check': pool_check,
    }


def _init_stage1_contexts(
    *, unit_cell_npz_path, meta_json_path, radius_spec, element_cfg,
    mainlobe_cfg, a0_cfg, partition_table_json_path='',
    static_pool_npz_path=None, static_pool_json_path=None,
    iep_phi_main_rad=0.0, iep_bounds='clamp',
    iep_enable_bilinear_phi=True, pool_tol_deg=1e-9, verbose=False,
):
    iep_ctx = init_stage1_iep(
        unit_cell_npz_path, meta_json_path=meta_json_path,
        phi_main_rad=iep_phi_main_rad, bounds=iep_bounds,
        enable_bilinear_phi=iep_enable_bilinear_phi, verbose=verbose,
    )
    radius_current = build_radius_init(radius_spec, verbose=verbose)
    elem_counts_current = build_ring_element_counts(radius_current.r_over_lambda, element_cfg, verbose=verbose)
    elem_geom_current = build_element_geometry(
        radius_current.r_over_lambda, elem_counts_current.N_per_ring,
        include_center=elem_counts_current.include_center, verbose=verbose,
    )
    if str(partition_table_json_path).strip():
        runtime_cfg = Stage1RuntimePoolConfig(
            partition_table_json=str(partition_table_json_path),
            lookup_tol=1e-9, pool_tol_deg=float(pool_tol_deg),
        )
        package = build_runtime_stage1_pool_package(
            radius_spec, element_cfg, runtime_cfg, verbose=verbose,
        )
        pool_ctx = init_stage1_pool_context_from_package(
            package, tol_deg=float(pool_tol_deg), verbose=verbose,
        )
    else:
        if not static_pool_npz_path or not static_pool_json_path:
            raise ValueError('Provide partition_table_json_path, or both static pool files')
        pool_ctx = init_stage1_pool_context(
            Stage1PoolConfig(str(static_pool_npz_path), str(static_pool_json_path), tol_deg=float(pool_tol_deg)),
            verbose=verbose,
        )
    ml_cfg_use = _make_mainlobe_cfg_from_pool(pool_ctx, mainlobe_cfg)
    mainlobe_state_current = build_mainlobe_domain_state(
        iep_ctx, radius_current, elem_counts_current, ml_cfg_use, verbose=verbose,
    )
    a0_ctx = init_stage1_a0_policy(iep_ctx, pool_ctx, a0_cfg, verbose=verbose)
    pre_ctx = init_stage1_precompute_context(
        iep_ctx, pool_ctx, mainlobe_state_current, a0_ctx, elem_geom_current,
        mask_domain_mode=mainlobe_state_current.use_domain, verbose=verbose,
    )
    update_element_runtime_geometry(pre_ctx, radius_current, beta=2.0 * np.pi, verbose=verbose)
    return {
        'iep_ctx': iep_ctx, 'radius_current': radius_current,
        'elem_counts_current': elem_counts_current,
        'elem_geom_current': elem_geom_current, 'pool_ctx': pool_ctx,
        'mainlobe_state_current': mainlobe_state_current,
        'a0_ctx': a0_ctx, 'pre_ctx': pre_ctx,
        'mainlobe_cfg_use': ml_cfg_use,
    }


def _build_precompute_for_topology(
    iep_ctx, pool_ctx, mainlobe_state, a0_ctx, radius, counts, *, verbose=False,
):
    geom = build_element_geometry(
        radius.r_over_lambda, counts.N_per_ring,
        include_center=counts.include_center, verbose=verbose,
    )
    pre = init_stage1_precompute_context(
        iep_ctx, pool_ctx, mainlobe_state, a0_ctx, geom,
        mask_domain_mode=mainlobe_state.use_domain, verbose=verbose,
    )
    update_element_runtime_geometry(pre, radius, beta=2.0 * np.pi, verbose=verbose)
    return geom, pre


def _copy_precompute_context_into(dst, src):
    for name in Stage1PrecomputeContext.__slots__:
        setattr(dst, name, getattr(src, name))


def _build_current_constraint_pool(work_batch, verbose=False):
    return build_stage1_constraint_pool(work_batch, verbose=verbose)


def _make_radii_cfg_for_sigma(base, sigma):
    return Stage1RadiiOptConfig(
        d_min=float(base.d_min), sigma=float(sigma),
        gamma_main=float(base.gamma_main), lambda_dr=float(base.lambda_dr),
        use_main_anchors=bool(base.use_main_anchors),
        anchors_per_scan=int(base.anchors_per_scan),
        anchor_merge_alpha_deg=float(base.anchor_merge_alpha_deg),
        radius_feasibility_tol=float(base.radius_feasibility_tol),
        solver_order=tuple(base.solver_order), solver_verbose=bool(base.solver_verbose),
        scipy_maxiter=int(base.scipy_maxiter), scipy_ftol=float(base.scipy_ftol),
    )


def _evaluate_radius_on_installed_work_points(
    iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, radius, theta_list,
    *, verbose=False,
):
    update_element_runtime_geometry(pre_ctx, radius, beta=2.0 * np.pi, verbose=verbose)
    return eval_stage1_scan_set(
        iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx,
        np.asarray(theta_list, dtype=np.float64), mode='work',
        include_dynamic=True, rank_by='ratio_db', verbose=verbose,
    )


def _build_trial_topology_info(radius_trial, state, element_cfg):
    counts_trial = build_ring_element_counts(
        radius_trial.r_over_lambda, element_cfg, verbose=False,
    )
    old_sig = _topology_signature(state.elem_counts_current)
    trial_sig = _topology_signature(counts_trial)
    return {
        'counts_trial': counts_trial,
        'old_signature': old_sig,
        'trial_signature': trial_sig,
        'topology_changed': old_sig != trial_sig,
    }


def _register_trial_reject(state, main_cfg, sigma_before, reason):
    was_at_min = float(sigma_before) <= float(main_cfg.sigma_min) + 1e-12
    state.consecutive_rejects += 1
    state.sigma_current = max(
        float(main_cfg.sigma_min), float(sigma_before) * float(main_cfg.sigma_shrink),
    )
    if was_at_min:
        state.rejects_at_sigma_min += 1
    else:
        state.rejects_at_sigma_min = 0
    return {
        'reason': str(reason), 'sigma_after': float(state.sigma_current),
        'consecutive_rejects': int(state.consecutive_rejects),
        'rejects_at_sigma_min': int(state.rejects_at_sigma_min),
    }


def _register_trial_accept(state, main_cfg, improvement_db, topology_changed):
    state.accept_counter += 1
    state.accepted_since_full += 1
    state.consecutive_rejects = 0
    state.rejects_at_sigma_min = 0
    if topology_changed:
        state.sigma_current = min(float(state.sigma_current), float(main_cfg.sigma_init))
    elif float(improvement_db) >= float(main_cfg.sigma_expand_improvement_db):
        state.sigma_current = min(
            float(main_cfg.sigma_max), float(state.sigma_current) * float(main_cfg.sigma_expand),
        )


def _format_N(sig):
    return '[' + ','.join(str(int(x)) for x in sig) + ']'


def _psll_improvement(current_db, trial_db):
    c = float(current_db)
    t = float(trial_db)
    if np.isnan(c) or np.isnan(t):
        return float('nan')
    if np.isneginf(c) and np.isneginf(t):
        return 0.0
    if np.isneginf(t):
        return float('inf')
    if np.isneginf(c):
        return float('-inf')
    return c - t


def _accurate_check_trial_on_trial_work(
    iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx,
    radius_trial, quick_batch, *, verbose=False,
):
    theta_list = np.asarray(quick_batch.theta_s_deg, dtype=np.float64)
    seed_map_1 = extract_work_tracking_seed_map(
        quick_batch, pool_ctx.tracking_protocol, verbose=verbose,
    )
    _rebuild_dynamic_points_from_seed_map(
        pool_ctx, seed_map_1, mode='work', round_id=1,
        clear_existing=True, verbose=verbose,
    )
    update_element_runtime_geometry(
        pre_ctx, radius_trial, beta=2.0 * np.pi, verbose=verbose,
    )
    _refresh_dynamic_precompute_for_theta_list(
        pre_ctx, iep_ctx, a0_ctx, pool_ctx, theta_list,
        mode='work', verbose=verbose,
    )
    round1_batch = eval_stage1_scan_set(
        iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, theta_list,
        mode='work', include_dynamic=True, rank_by='ratio_db', verbose=verbose,
    )
    seed_map_2 = extract_work_tracking_seed_map(
        round1_batch, pool_ctx.tracking_protocol, verbose=verbose,
    )
    _rebuild_dynamic_points_from_seed_map(
        pool_ctx, seed_map_2, mode='work', round_id=2,
        clear_existing=False, verbose=verbose,
    )
    _refresh_dynamic_precompute_for_theta_list(
        pre_ctx, iep_ctx, a0_ctx, pool_ctx, theta_list,
        mode='work', verbose=verbose,
    )
    accurate = eval_stage1_scan_set(
        iep_ctx, pool_ctx, mainlobe_state, a0_ctx, pre_ctx, theta_list,
        mode='work', include_dynamic=True, rank_by='ratio_db', verbose=verbose,
    )
    accurate.meta['tracking_round1_worst_psll_db'] = float(round1_batch.worst_psll_db)
    return accurate


def _evaluate_trial_with_fair_union(
    state, iep_ctx, pool_ctx, a0_ctx, pre_current, pre_trial,
    radius_trial, *, verbose=False,
):
    theta_list = np.asarray(state.current_work_batch.theta_s_deg, dtype=np.float64)
    current_dynamic = _backup_work_dynamic_state(pool_ctx, pre_current)

    # Quick test on the old work points using the trial's physical topology.
    _install_work_dynamic_pool_state(pool_ctx, current_dynamic)
    _refresh_dynamic_precompute_for_theta_list(
        pre_trial, iep_ctx, a0_ctx, pool_ctx, theta_list, mode='work', verbose=verbose,
    )
    quick_batch = _evaluate_radius_on_installed_work_points(
        iep_ctx, pool_ctx, state.mainlobe_state_current, a0_ctx,
        pre_trial, radius_trial, theta_list, verbose=verbose,
    )
    quick_improvement = _psll_improvement(
        state.current_work_batch.worst_psll_db, quick_batch.worst_psll_db,
    )
    if not np.isfinite(quick_improvement) and not np.isposinf(quick_improvement):
        return {
            'quick_batch': quick_batch, 'quick_improvement_db': quick_improvement,
            'quick_reject': True, 'current_dynamic': current_dynamic,
        }
    if float(quick_improvement) <= 0.0:
        return {
            'quick_batch': quick_batch, 'quick_improvement_db': quick_improvement,
            'quick_reject': True, 'current_dynamic': current_dynamic,
        }

    # Trial-specific two-stage tracking.
    accurate_trial = _accurate_check_trial_on_trial_work(
        iep_ctx, pool_ctx, state.mainlobe_state_current, a0_ctx,
        pre_trial, radius_trial, quick_batch, verbose=verbose,
    )
    trial_dynamic = _backup_work_dynamic_state(pool_ctx, pre_trial)
    union_dynamic = _merge_work_dynamic_states(
        current_dynamic, trial_dynamic, theta_list, pool_ctx,
    )
    _install_work_dynamic_pool_state(pool_ctx, union_dynamic)

    # Both arrays are evaluated on exactly the same static+union direction set.
    _refresh_dynamic_precompute_for_theta_list(
        pre_current, iep_ctx, a0_ctx, pool_ctx, theta_list, mode='work', verbose=verbose,
    )
    current_cmp = _evaluate_radius_on_installed_work_points(
        iep_ctx, pool_ctx, state.mainlobe_state_current, a0_ctx,
        pre_current, state.radius_current, theta_list, verbose=verbose,
    )
    _refresh_dynamic_precompute_for_theta_list(
        pre_trial, iep_ctx, a0_ctx, pool_ctx, theta_list, mode='work', verbose=verbose,
    )
    trial_cmp = _evaluate_radius_on_installed_work_points(
        iep_ctx, pool_ctx, state.mainlobe_state_current, a0_ctx,
        pre_trial, radius_trial, theta_list, verbose=verbose,
    )
    improvement = _psll_improvement(current_cmp.worst_psll_db, trial_cmp.worst_psll_db)
    return {
        'quick_batch': quick_batch,
        'quick_improvement_db': quick_improvement,
        'quick_reject': False,
        'accurate_trial_batch': accurate_trial,
        'current_cmp_batch': current_cmp,
        'trial_cmp_batch': trial_cmp,
        'improvement_db': improvement,
        'current_dynamic': current_dynamic,
        'trial_dynamic': trial_dynamic,
        'union_dynamic': union_dynamic,
    }


def _restore_checkpoint_state(
    state, checkpoint, iep_ctx, pool_ctx, a0_ctx, pre_ctx, *, verbose=False,
):
    state.radius_current = _clone_radius_result(checkpoint.radius_result)
    state.elem_counts_current = _clone_ring_count_result(checkpoint.elem_counts_result)
    state.elem_geom_current = _clone_element_geometry(checkpoint.elem_geom_result)
    state.mainlobe_state_current = _clone_mainlobe_state(checkpoint.mainlobe_state)
    geom, pre_new = _build_precompute_for_topology(
        iep_ctx, pool_ctx, state.mainlobe_state_current, a0_ctx,
        state.radius_current, state.elem_counts_current, verbose=verbose,
    )
    state.elem_geom_current = geom
    _copy_precompute_context_into(pre_ctx, pre_new)
    state.current_full_batch = checkpoint.batch_result
    state.current_full_validated = True
    state.topology_signature = _topology_signature(state.elem_counts_current)
    return pre_ctx


def _run_full_scout_event(
    state, iep_ctx, pool_ctx, a0_ctx, pre_ctx, main_cfg, logger,
    *, reason, verbose=False,
):
    old_work = get_current_work_theta_s_deg(pool_ctx)
    state.full_scout_count += 1
    _, _, full_batch = _bootstrap_dynamic_mode(
        iep_ctx, pool_ctx, state.mainlobe_state_current, a0_ctx, pre_ctx,
        get_full_theta_s_deg(pool_ctx), mode='full', verbose=verbose,
    )
    state.current_full_batch = full_batch
    degradation = float('nan')
    if state.last_full_snapshot is not None:
        degradation = float(full_batch.worst_psll_db) - float(state.last_full_snapshot.batch_result.worst_psll_db)
    feedback = _select_full_feedback_theta(full_batch, pool_ctx)
    rolled_back = False
    improved_best = False

    if (
        state.last_full_snapshot is not None
        and np.isfinite(degradation)
        and degradation > float(main_cfg.full_rollback_threshold_db)
    ):
        rolled_back = True
        state.rollback_count += 1
        bad_full = full_batch
        _restore_checkpoint_state(
            state, state.last_full_snapshot, iep_ctx, pool_ctx, a0_ctx,
            pre_ctx, verbose=verbose,
        )
        state.sigma_current = max(
            float(main_cfg.sigma_min),
            min(float(state.sigma_current), float(main_cfg.sigma_init)) * float(main_cfg.sigma_shrink),
        )
        work_change = set_work_scan_angles_from_full_feedback(pool_ctx, feedback, verbose=verbose)
        _, _, state.current_work_batch = _bootstrap_dynamic_mode(
            iep_ctx, pool_ctx, state.mainlobe_state_current, a0_ctx, pre_ctx,
            get_current_work_theta_s_deg(pool_ctx), mode='work', verbose=verbose,
        )
        state.current_full_batch = state.last_full_snapshot.batch_result
        state.current_full_validated = True
        state.work_signature = _theta_signature(get_current_work_theta_s_deg(pool_ctx))
        state.topology_signature = _topology_signature(state.elem_counts_current)
        full_for_log = bad_full
    else:
        work_change = set_work_scan_angles_from_full_feedback(pool_ctx, feedback, verbose=verbose)
        _, _, state.current_work_batch = _bootstrap_dynamic_mode(
            iep_ctx, pool_ctx, state.mainlobe_state_current, a0_ctx, pre_ctx,
            get_current_work_theta_s_deg(pool_ctx), mode='work', verbose=verbose,
        )
        state.current_full_validated = True
        state.work_signature = _theta_signature(get_current_work_theta_s_deg(pool_ctx))
        state.topology_signature = _topology_signature(state.elem_counts_current)
        checkpoint = _make_full_checkpoint(
            state, full_batch, reason, pool_ctx,
            extra={'feedback_theta_s_deg': feedback.tolist()},
        )
        state.last_full_snapshot = checkpoint
        best_improvement_db = float('inf') if state.best_full_snapshot is None else _psll_improvement(
            state.best_full_snapshot.batch_result.worst_psll_db,
            full_batch.worst_psll_db,
        )
        # Always retain the numerically best full-validated checkpoint.
        # full_improve_tol_db controls stagnation counting only; it must not
        # discard a real, albeit small, improvement from the final output.
        if state.best_full_snapshot is None or best_improvement_db > 1e-12:
            state.best_full_snapshot = checkpoint
            improved_best = True
        if best_improvement_db > float(main_cfg.full_improve_tol_db):
            state.full_no_improve_count = 0
        else:
            state.full_no_improve_count += 1
        full_for_log = full_batch

    state.accepted_since_full = 0
    state.consecutive_rejects = 0
    state.rejects_at_sigma_min = 0
    state.full_signature_history.append({
        'full_id': int(state.full_scout_count),
        'topology_signature': list(state.topology_signature),
        'work_signature': list(state.work_signature),
        'worst_psll_db': float(full_for_log.worst_psll_db),
        'reason': str(reason), 'rolled_back': bool(rolled_back),
    })
    state.topology_signature_history.append(tuple(state.topology_signature))
    rec = {
        'full_id': int(state.full_scout_count), 'reason': str(reason),
        'attempt': int(state.attempt_counter), 'topology_epoch': int(state.topology_epoch),
        'worst_psll_db': float(full_for_log.worst_psll_db),
        'worst_theta_s_deg': float(full_for_log.worst_theta_s_deg),
        'best_full_psll_db': float(state.best_full_snapshot.batch_result.worst_psll_db) if state.best_full_snapshot is not None else float('nan'),
        'improved_best': bool(improved_best), 'rolled_back': bool(rolled_back),
        'degradation_from_last_db': float(degradation),
        'work_changed': bool(work_change['changed']),
        'work_theta_s_deg': get_current_work_theta_s_deg(pool_ctx).tolist(),
        'topology_signature': list(state.topology_signature),
    }
    state.full_history.append(rec)
    if logger is not None:
        if rolled_back:
            logger.write(
                '[FULL %03d][%s] current=%.4f last=%.4f degrade=%+.4f ROLLBACK sigma=%.4f work=%s'
                % (
                    state.full_scout_count, reason,
                    float(full_for_log.worst_psll_db),
                    float(state.last_full_snapshot.batch_result.worst_psll_db),
                    float(degradation), float(state.sigma_current),
                    get_current_work_theta_s_deg(pool_ctx).tolist(),
                )
            )
        else:
            logger.write(
                '[FULL %03d][%s] worst=%.4f dB theta=%.1f best=%.4f work=%s'
                % (
                    state.full_scout_count, reason,
                    float(full_batch.worst_psll_db), float(full_batch.worst_theta_s_deg),
                    float(state.best_full_snapshot.batch_result.worst_psll_db),
                    get_current_work_theta_s_deg(pool_ctx).tolist(),
                )
            )
    return {
        'rolled_back': bool(rolled_back),
        'work_changed': bool(work_change['changed']),
        'old_work': old_work,
        'new_work': get_current_work_theta_s_deg(pool_ctx),
        'feedback_theta': feedback,
        'full_batch': full_for_log,
    }


def _topology_abab_stop(state, main_cfg):
    w = int(main_cfg.topology_cycle_window)
    if w < 4 or len(state.full_signature_history) < w:
        return False
    recent = state.full_signature_history[-w:]
    sig = [tuple(x['topology_signature']) for x in recent]
    if not (sig[-4] == sig[-2] and sig[-3] == sig[-1] and sig[-4] != sig[-3]):
        return False
    vals = np.asarray([float(x['worst_psll_db']) for x in recent[-4:]], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return False
    return float(np.max(vals) - np.min(vals)) <= float(main_cfg.topology_cycle_full_tol_db)


def evaluate_stop_request(state, main_cfg):
    if state.attempt_counter >= int(main_cfg.max_iters):
        return {'requested': True, 'hard': True, 'reason': 'max_iters', 'detail': {}}
    if state.full_scout_count >= int(main_cfg.max_full_scouts):
        return {'requested': True, 'hard': True, 'reason': 'max_full_scouts', 'detail': {}}
    if state.topology_epoch >= int(main_cfg.max_topology_epochs):
        return {'requested': True, 'hard': True, 'reason': 'max_topology_epochs', 'detail': {}}
    if state.rejects_at_sigma_min >= int(main_cfg.sigma_min_reject_limit):
        return {'requested': True, 'hard': False, 'reason': 'sigma_min_rejects', 'detail': {'count': state.rejects_at_sigma_min}}
    if state.consecutive_rejects >= int(main_cfg.consecutive_reject_limit):
        return {'requested': True, 'hard': False, 'reason': 'consecutive_rejects', 'detail': {'count': state.consecutive_rejects}}
    if state.full_no_improve_count >= int(main_cfg.full_no_improve_limit):
        stable = False
        if len(state.full_signature_history) >= 2:
            a = state.full_signature_history[-1]
            b = state.full_signature_history[-2]
            stable = (
                tuple(a['topology_signature']) == tuple(b['topology_signature'])
                and tuple(a['work_signature']) == tuple(b['work_signature'])
            )
        if stable:
            return {'requested': True, 'hard': False, 'reason': 'full_no_improve', 'detail': {'count': state.full_no_improve_count}}
    if _topology_abab_stop(state, main_cfg):
        return {'requested': True, 'hard': False, 'reason': 'topology_abab_cycle', 'detail': {}}
    return {'requested': False, 'hard': False, 'reason': '', 'detail': {}}


def _run_final_full_audit(
    state, iep_ctx, pool_ctx, a0_ctx, pre_ctx, element_cfg,
    main_cfg, logger, *, verbose=False,
):
    work_before = get_current_work_theta_s_deg(pool_ctx)
    topology_changed = False
    canonical_counts = build_ring_element_counts(
        state.radius_current.r_over_lambda, element_cfg, verbose=False,
    )
    if _topology_signature(canonical_counts) != _topology_signature(state.elem_counts_current):
        topology_changed = True
        old_sig = _topology_signature(state.elem_counts_current)
        state.elem_counts_current = canonical_counts
        state.elem_geom_current, pre_new = _build_precompute_for_topology(
            iep_ctx, pool_ctx, state.mainlobe_state_current, a0_ctx,
            state.radius_current, state.elem_counts_current, verbose=verbose,
        )
        _copy_precompute_context_into(pre_ctx, pre_new)
        state.topology_epoch += 1
        state.topology_signature = _topology_signature(state.elem_counts_current)
        state.topology_history.append({
            'attempt': int(state.attempt_counter), 'event': 'final_canonicalize',
            'old_signature': list(old_sig), 'new_signature': list(state.topology_signature),
        })
    out = _run_full_scout_event(
        state, iep_ctx, pool_ctx, a0_ctx, pre_ctx, main_cfg, logger,
        reason='final_audit', verbose=verbose,
    )
    work_after = get_current_work_theta_s_deg(pool_ctx)
    continue_required = bool(topology_changed or not _same_theta_set(work_before, work_after, pool_ctx.tol_deg))
    return {
        'topology_changed': topology_changed,
        'work_changed': not _same_theta_set(work_before, work_after, pool_ctx.tol_deg),
        'continue_required': continue_required,
        'full_result': out,
    }


# ----------------------------- final mainlobe/output --------------------------


def _compute_final_mainlobe_domains(
    iep_ctx, pool_ctx, radius_result, elem_counts_result, template_cfg,
    *, verbose=False,
):
    ml_cfg = MainlobeDomainConfig(
        theta_s_deg_list=get_full_theta_s_deg(pool_ctx),
        phi_s_deg=float(template_cfg.phi_s_deg),
        use_domain=str(template_cfg.use_domain),
        source=str(template_cfg.final_source),
        final_source=str(template_cfg.final_source),
        freeze_during_optimization=bool(template_cfg.freeze_during_optimization),
        alpha_table_json_path=str(template_cfg.alpha_table_json_path),
        alpha_table_tol=float(template_cfg.alpha_table_tol),
        mainlobe_margin_deg=float(template_cfg.mainlobe_margin_deg),
        solver_cfg=template_cfg.solver_cfg,
    )
    out = build_mainlobe_domain_state(
        iep_ctx, radius_result, elem_counts_result, ml_cfg, verbose=verbose,
    )
    out.stats['final_recompute'] = True
    out.stats['optimization_mainlobe_frozen'] = bool(template_cfg.freeze_during_optimization)
    return out


def _json_safe(obj):
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        if np.isposinf(obj):
            return 'inf'
        if np.isneginf(obj):
            return '-inf'
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
    slots = getattr(obj, '__slots__', None)
    if slots is not None:
        return {str(k): _json_safe(getattr(obj, k)) for k in slots}
    return str(obj)


def _atomic_write_json(path, payload):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            _os.fsync(f.fileno())
        except Exception:
            pass
    _os.replace(tmp, path)


def _atomic_write_csv(path, rows, fieldnames):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8-sig', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            clean = {}
            for key in fieldnames:
                value = row.get(key, '')
                if isinstance(value, (list, tuple)):
                    value = '|'.join(str(x) for x in value)
                clean[key] = value
            w.writerow(clean)
        f.flush()
        try:
            _os.fsync(f.fileno())
        except Exception:
            pass
    _os.replace(tmp, path)


def _atomic_savez(path, **arrays):
    tmp = str(path) + '.tmp.npz'
    np.savez(tmp, **arrays)
    _os.replace(tmp, path)


def finalize_stage1_outputs(run_result_core, output_dir, *, verbose=True):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = str(out_dir / 'a0cra_stage1_main_out.npz')
    json_path = str(out_dir / 'a0cra_stage1_main_out.json')
    attempts_path = str(out_dir / 'a0cra_stage1_attempts.csv')
    full_path = str(out_dir / 'a0cra_stage1_full_scouts.csv')

    final_radius = run_result_core['final_output_radius']
    final_counts = run_result_core['final_output_counts']
    final_geom = run_result_core['final_output_geom']
    final_mainlobe = run_result_core['final_output_mainlobe']
    best_full = run_result_core['best_full_snapshot']
    state = run_result_core['state_final']
    final_batch = best_full.batch_result

    attempts = list(state.attempt_history)
    full_hist = list(state.full_history)
    _atomic_savez(
        npz_path,
        r_best=np.asarray(final_radius.r_over_lambda, dtype=np.float64),
        N_best=np.asarray(final_counts.N_per_ring, dtype=np.int64),
        x_elem_best=np.asarray(final_geom.x_elem, dtype=np.float64),
        y_elem_best=np.asarray(final_geom.y_elem, dtype=np.float64),
        theta_s_full_deg=np.asarray(run_result_core['theta_s_full_deg'], dtype=np.float64),
        theta_s_work_deg_final=np.asarray(run_result_core['theta_s_work_deg_final'], dtype=np.float64),
        alpha_null_deg=np.asarray(final_mainlobe.alpha_null_deg, dtype=np.float64),
        alpha_guard_null_deg=np.asarray(final_mainlobe.alpha_guard_null_deg, dtype=np.float64),
        alpha_3db_deg=np.asarray(final_mainlobe.alpha_3db_deg, dtype=np.float64),
        best_full_theta_deg=np.asarray(final_batch.theta_s_deg, dtype=np.float64),
        best_full_psll_db=np.asarray(final_batch.psll_db, dtype=np.float64),
        final_full_theta_deg=np.asarray(final_batch.theta_s_deg, dtype=np.float64),
        final_full_psll_db=np.asarray(final_batch.psll_db, dtype=np.float64),
        best_work_theta_deg=np.asarray(state.current_work_batch.theta_s_deg, dtype=np.float64),
        best_work_psll_db=np.asarray(state.current_work_batch.psll_db, dtype=np.float64),
        attempt_sigma_before=np.asarray([x.get('sigma_before', np.nan) for x in attempts], dtype=np.float64),
        attempt_sigma_after=np.asarray([x.get('sigma_after', np.nan) for x in attempts], dtype=np.float64),
        attempt_current_cmp_psll_db=np.asarray([x.get('current_cmp_psll_db', np.nan) for x in attempts], dtype=np.float64),
        attempt_trial_cmp_psll_db=np.asarray([x.get('trial_cmp_psll_db', np.nan) for x in attempts], dtype=np.float64),
        attempt_improvement_db=np.asarray([x.get('improvement_db', np.nan) for x in attempts], dtype=np.float64),
        attempt_accepted=np.asarray([bool(x.get('accepted', False)) for x in attempts], dtype=np.int8),
        full_history_worst_psll_db=np.asarray([x.get('worst_psll_db', np.nan) for x in full_hist], dtype=np.float64),
        full_history_worst_theta_deg=np.asarray([x.get('worst_theta_s_deg', np.nan) for x in full_hist], dtype=np.float64),
        full_history_topology_epoch=np.asarray([x.get('topology_epoch', -1) for x in full_hist], dtype=np.int64),
    )

    canonical_counts = build_ring_element_counts(
        final_radius.r_over_lambda, final_counts.config, verbose=False,
    )
    radius_audit = audit_radius_constraints(
        final_radius.r_over_lambda, K=final_radius.spec.K,
        d_min=final_radius.spec.d_min, R0=final_radius.spec.R0,
        feasibility_tol=5e-6,
    )
    geometry_radius_error = float(np.max(np.abs(
        np.asarray(final_geom.r_over_lambda, dtype=np.float64)
        - np.asarray(final_radius.r_over_lambda, dtype=np.float64)
    )))
    payload = {
        'format': 'a0cra_stage1_main_v3',
        'run_summary': {
            'stop_reason': str(state.stop_reason),
            'stop_detail': dict(state.stop_detail),
            'attempts': int(state.attempt_counter),
            'accepts': int(state.accept_counter),
            'full_scouts': int(state.full_scout_count),
            'topology_epochs': int(state.topology_epoch),
            'rollbacks': int(state.rollback_count),
            'solver_usage': dict(run_result_core.get('solver_usage', {})),
            'runtime_seconds': float(run_result_core.get('runtime_seconds', np.nan)),
        },
        'config': dict(run_result_core.get('config', {})),
        'legacy_unused_config': dict(run_result_core.get('legacy_unused_config', {})),
        'model_policy': {
            'af_backend': {
                'core': 'j0_collapsed', 'side': 'bessel_alias_L1',
                'direct_sum_used_in_stage1_metric': False,
                'direct_sum_used_in_stage1_radii_jacobian': False,
            },
            'mainlobe_policy': dict(run_result_core.get('mainlobe_policy', {})),
        },
        'stage1_geometry_runtime': dict(run_result_core.get('stage1_geometry_runtime', {})),
        'static_pool_partition': dict(run_result_core.get('static_pool_partition', {})),
        'initial_state': {
            'mainlobe': dict(run_result_core.get('initial_mainlobe', {})),
            'full_batch': dict(run_result_core.get('initial_full_batch', {})),
        },
        'final_solution': {
            'radius': np.asarray(final_radius.r_over_lambda, dtype=np.float64).tolist(),
            'N_per_ring': np.asarray(final_counts.N_per_ring, dtype=np.int64).tolist(),
            'total_elements': int(final_geom.x_elem.size),
            'full_worst_psll_db': float(final_batch.worst_psll_db),
            'full_worst_theta_s_deg': float(final_batch.worst_theta_s_deg),
            'theta_s_full_deg': np.asarray(final_batch.theta_s_deg, dtype=np.float64).tolist(),
            'full_psll_db': np.asarray(final_batch.psll_db, dtype=np.float64).tolist(),
            'work_theta_s_deg': np.asarray(run_result_core['theta_s_work_deg_final'], dtype=np.float64).tolist(),
            'topology_signature': list(best_full.topology_signature),
            'full_validated': True,
            'canonical_N_match': bool(np.array_equal(canonical_counts.N_per_ring, final_counts.N_per_ring)),
            'geometry_consistent': bool(geometry_radius_error <= 1e-12 and np.array_equal(final_geom.N_per_ring, final_counts.N_per_ring)),
            'selected_from': 'best_full_checkpoint',
            'mainlobe': {
                'theta_s_deg': np.asarray(final_mainlobe.theta_s_deg, dtype=np.float64).tolist(),
                'alpha_null_deg': np.asarray(final_mainlobe.alpha_null_deg, dtype=np.float64).tolist(),
                'alpha_guard_null_deg': np.asarray(final_mainlobe.alpha_guard_null_deg, dtype=np.float64).tolist(),
                'alpha_3db_deg': np.asarray(final_mainlobe.alpha_3db_deg, dtype=np.float64).tolist(),
                'stats': dict(final_mainlobe.stats),
            },
        },
        'best_full_checkpoint': _checkpoint_summary_dict(best_full),
        'last_full_checkpoint': _checkpoint_summary_dict(state.last_full_snapshot),
        'state_final': {
            'iter_idx': int(state.iter_idx), 'attempt_counter': int(state.attempt_counter),
            'accept_counter': int(state.accept_counter), 'topology_epoch': int(state.topology_epoch),
            'full_scout_count': int(state.full_scout_count),
            'accepted_since_full': int(state.accepted_since_full),
            'consecutive_rejects': int(state.consecutive_rejects),
            'rejects_at_sigma_min': int(state.rejects_at_sigma_min),
            'full_no_improve_count': int(state.full_no_improve_count),
            'rollback_count': int(state.rollback_count),
            'sigma_current': float(state.sigma_current),
            'current_work_batch': _batch_summary_dict(state.current_work_batch),
            'current_full_batch': _batch_summary_dict(state.current_full_batch),
        },
        'validation': {
            'radius_audit': radius_audit,
            'canonical_N_recomputed': np.asarray(canonical_counts.N_per_ring, dtype=np.int64).tolist(),
            'canonical_N_match': bool(np.array_equal(canonical_counts.N_per_ring, final_counts.N_per_ring)),
            'geometry_radius_error_max': geometry_radius_error,
            'geometry_N_match': bool(np.array_equal(final_geom.N_per_ring, final_counts.N_per_ring)),
            'final_full_audit_performed': bool(run_result_core.get('final_full_audit_performed', False)),
            'final_mainlobe_recomputed': True,
        },
        'attempt_history': attempts,
        'full_history': full_hist,
        'topology_history': list(state.topology_history),
        'output_files': {
            'npz': npz_path, 'json': json_path,
            'attempt_csv': attempts_path, 'full_csv': full_path,
            'log': str(run_result_core.get('output_log_path', '')),
        },
    }
    _atomic_write_json(json_path, payload)

    attempt_fields = [
        'attempt', 'topology_epoch', 'sigma_before', 'sigma_after',
        'solver', 'solver_status', 'delta_opt', 'max_abs_dr',
        'N_changed', 'N_old', 'N_trial', 'quick_psll_db',
        'current_cmp_psll_db', 'trial_cmp_psll_db', 'improvement_db',
        'accepted', 'reason', 'consecutive_rejects',
        'rejects_at_sigma_min', 'accepted_since_full',
    ]
    full_fields = [
        'full_id', 'reason', 'attempt', 'topology_epoch',
        'worst_psll_db', 'worst_theta_s_deg', 'best_full_psll_db',
        'improved_best', 'rolled_back', 'degradation_from_last_db',
        'work_changed', 'work_theta_s_deg', 'topology_signature',
    ]
    if bool(run_result_core.get('save_attempt_csv', True)):
        _atomic_write_csv(attempts_path, attempts, attempt_fields)
    else:
        attempts_path = ''
    if bool(run_result_core.get('save_full_csv', True)):
        _atomic_write_csv(full_path, full_hist, full_fields)
    else:
        full_path = ''
    _log(verbose, '[SAVE] %s' % npz_path)
    _log(verbose, '[SAVE] %s' % json_path)
    return npz_path, json_path, attempts_path, full_path


# ----------------------------- main loop --------------------------------------


def run_a0cra_stage1_main(
    *, unit_cell_npz_path, meta_json_path, radius_spec,
    partition_table_json_path='', static_pool_npz_path=None,
    static_pool_json_path=None, element_cfg=None, mainlobe_cfg=None,
    a0_cfg=Stage1A0PolicyConfig(), radii_opt_cfg=Stage1RadiiOptConfig(),
    main_cfg=Stage1MainConfig(), iep_phi_main_rad=0.0,
    iep_bounds='clamp', iep_enable_bilinear_phi=True,
    pool_tol_deg=1e-9, output_dir='.',
):
    if element_cfg is None or mainlobe_cfg is None:
        raise ValueError('element_cfg and mainlobe_cfg are required')
    start_time = _time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(out_dir / 'a0cra_stage1_run.log') if main_cfg.log_to_file else ''
    logger = Stage1CompactLogger(
        log_path, enabled=bool(main_cfg.log_to_file),
        echo=bool(main_cfg.verbose and main_cfg.log_level != 'silent'),
    )
    detail_verbose = bool(main_cfg.verbose and main_cfg.log_level == 'debug')
    solver_usage = {}
    final_audit_performed = False
    _stdout_sink = None
    _stdout_redirect = None
    if not detail_verbose:
        _stdout_sink = open(_os.devnull, 'w')
        _stdout_redirect = _contextlib.redirect_stdout(_stdout_sink)
        _stdout_redirect.__enter__()
    try:
        if abs(float(radius_spec.d_min) - float(radii_opt_cfg.d_min)) > 1e-12:
            radius_spec = RadiusInitSpec(
                name=radius_spec.name, method=radius_spec.method,
                K=radius_spec.K, R0=radius_spec.R0,
                d_min=float(radii_opt_cfg.d_min), b=radius_spec.b,
            )
        ctx = _init_stage1_contexts(
            unit_cell_npz_path=unit_cell_npz_path,
            meta_json_path=meta_json_path, radius_spec=radius_spec,
            partition_table_json_path=partition_table_json_path,
            static_pool_npz_path=static_pool_npz_path,
            static_pool_json_path=static_pool_json_path,
            element_cfg=element_cfg, mainlobe_cfg=mainlobe_cfg,
            a0_cfg=a0_cfg, iep_phi_main_rad=iep_phi_main_rad,
            iep_bounds=iep_bounds,
            iep_enable_bilinear_phi=iep_enable_bilinear_phi,
            pool_tol_deg=pool_tol_deg, verbose=detail_verbose,
        )
        iep_ctx = ctx['iep_ctx']
        pool_ctx = ctx['pool_ctx']
        a0_ctx = ctx['a0_ctx']
        pre_ctx = ctx['pre_ctx']
        geometry_check = validate_stage1_geometry_runtime_config(
            radius_spec=radius_spec, element_cfg=element_cfg,
            radii_opt_cfg=radii_opt_cfg,
            static_pool_summary=pool_ctx.summary,
            bind_d_eff_to_d_min=True,
        )
        if not geometry_check['ok']:
            raise ValueError('Stage-1 geometry/static-pool mismatch:\n' + json.dumps(geometry_check, ensure_ascii=False, indent=2))

        bootstrap = bootstrap_stage1_full_then_work(
            iep_ctx, pool_ctx, ctx['mainlobe_state_current'], a0_ctx,
            pre_ctx, verbose=detail_verbose,
        )
        state = Stage1MainState(
            iter_idx=0, attempt_counter=0, accept_counter=0,
            topology_epoch=0, full_scout_count=1,
            accepted_since_full=0, consecutive_rejects=0,
            rejects_at_sigma_min=0, full_no_improve_count=0,
            rollback_count=0, sigma_current=float(main_cfg.sigma_init),
            radius_current=ctx['radius_current'],
            elem_counts_current=ctx['elem_counts_current'],
            elem_geom_current=ctx['elem_geom_current'],
            mainlobe_state_current=ctx['mainlobe_state_current'],
            current_work_batch=bootstrap['work_batch'],
            current_full_batch=bootstrap['full_batch'],
            current_constraint_pool=None, current_full_validated=True,
            last_full_snapshot=None, best_full_snapshot=None,
            work_signature=_theta_signature(get_current_work_theta_s_deg(pool_ctx)),
            topology_signature=_topology_signature(ctx['elem_counts_current']),
            full_signature_history=[], topology_signature_history=[],
            attempt_history=[], full_history=[], topology_history=[],
        )
        initial_cp = _make_full_checkpoint(
            state, state.current_full_batch, 'initial', pool_ctx,
            extra={'feedback_theta_s_deg': bootstrap['feedback_theta'].tolist()},
        )
        state.last_full_snapshot = initial_cp
        state.best_full_snapshot = initial_cp
        initial_full_summary = _batch_summary_dict(state.current_full_batch)
        initial_mainlobe = {
            'theta_s_deg': np.asarray(ctx['mainlobe_state_current'].theta_s_deg, dtype=np.float64).tolist(),
            'alpha_null_deg': np.asarray(ctx['mainlobe_state_current'].alpha_null_deg, dtype=np.float64).tolist(),
            'alpha_guard_null_deg': np.asarray(ctx['mainlobe_state_current'].alpha_guard_null_deg, dtype=np.float64).tolist(),
            'alpha_3db_deg': np.asarray(ctx['mainlobe_state_current'].alpha_3db_deg, dtype=np.float64).tolist(),
            'stats': dict(ctx['mainlobe_state_current'].stats),
        }
        initial_full_rec = {
            'full_id': 1, 'reason': 'initial', 'attempt': 0,
            'topology_epoch': 0,
            'worst_psll_db': float(state.current_full_batch.worst_psll_db),
            'worst_theta_s_deg': float(state.current_full_batch.worst_theta_s_deg),
            'best_full_psll_db': float(state.current_full_batch.worst_psll_db),
            'improved_best': True, 'rolled_back': False,
            'degradation_from_last_db': float('nan'),
            'work_changed': bool(bootstrap['work_change']['changed']),
            'work_theta_s_deg': get_current_work_theta_s_deg(pool_ctx).tolist(),
            'topology_signature': list(state.topology_signature),
        }
        state.full_history.append(initial_full_rec)
        state.full_signature_history.append({
            'full_id': 1, 'topology_signature': list(state.topology_signature),
            'work_signature': list(state.work_signature),
            'worst_psll_db': float(state.current_full_batch.worst_psll_db),
            'reason': 'initial', 'rolled_back': False,
        })
        state.topology_signature_history.append(tuple(state.topology_signature))
        logger.write(
            '[INIT] K=%d dmin=%.4f R0=%.4f N=%s elements=%d'
            % (
                int(radius_spec.K), float(radius_spec.d_min), float(radius_spec.R0),
                _format_N(state.topology_signature), int(state.elem_geom_current.x_elem.size),
            )
        )
        logger.write(
            '[FULL 001][initial] worst=%.4f dB theta=%.1f best=%.4f'
            % (
                float(state.current_full_batch.worst_psll_db),
                float(state.current_full_batch.worst_theta_s_deg),
                float(state.best_full_snapshot.batch_result.worst_psll_db),
            )
        )
        logger.write(
            '[WORK] theta=%s worst=%.4f dB'
            % (get_current_work_theta_s_deg(pool_ctx).tolist(), float(state.current_work_batch.worst_psll_db))
        )

        pending_stop = None
        while state.attempt_counter < int(main_cfg.max_iters):
            state.iter_idx += 1
            state.attempt_counter += 1
            sigma_before = float(state.sigma_current)
            state.current_constraint_pool = _build_current_constraint_pool(
                state.current_work_batch, verbose=detail_verbose,
            )
            cfg_iter = _make_radii_cfg_for_sigma(radii_opt_cfg, state.sigma_current)
            step = run_stage1_radii_opt_iteration(
                iep_ctx, pool_ctx, state.mainlobe_state_current, a0_ctx,
                pre_ctx, state.radius_current, state.current_work_batch,
                state.current_constraint_pool, cfg_iter, verbose=detail_verbose,
            )
            solve_res = step['solve_result']
            solver_name = str(solve_res.solver_used)
            solver_usage[solver_name] = int(solver_usage.get(solver_name, 0)) + 1
            rec = {
                'attempt': int(state.attempt_counter),
                'topology_epoch': int(state.topology_epoch),
                'sigma_before': sigma_before,
                'sigma_after': sigma_before,
                'solver': solver_name,
                'solver_status': str(solve_res.status),
                'delta_opt': float(solve_res.delta_opt),
                'max_abs_dr': float(solve_res.max_abs_dr),
                'N_changed': False,
                'N_old': list(state.topology_signature),
                'N_trial': list(state.topology_signature),
                'quick_psll_db': float('nan'),
                'current_cmp_psll_db': float('nan'),
                'trial_cmp_psll_db': float('nan'),
                'improvement_db': float('nan'),
                'accepted': False,
                'reason': '',
            }

            if not bool(step.get('trial_valid', False)):
                rej = _register_trial_reject(
                    state, main_cfg, sigma_before,
                    'solver_or_trial_invalid:' + str(step.get('failure_reason', 'unknown')),
                )
                rec.update(rej)
                logger.write(
                    '[IT %03d][E%02d] sigma=%.4f solver=%s/%s INVALID:%s sigma=%.4f reject=%d'
                    % (
                        state.attempt_counter, state.topology_epoch, sigma_before,
                        solver_name, solve_res.status, step.get('failure_reason', 'unknown'),
                        state.sigma_current, state.consecutive_rejects,
                    )
                )
            else:
                radius_trial = step['radius_trial']
                topo = _build_trial_topology_info(radius_trial, state, element_cfg)
                rec['N_changed'] = bool(topo['topology_changed'])
                rec['N_trial'] = list(topo['trial_signature'])
                if topo['topology_changed']:
                    geom_trial, pre_trial = _build_precompute_for_topology(
                        iep_ctx, pool_ctx, state.mainlobe_state_current, a0_ctx,
                        radius_trial, topo['counts_trial'], verbose=detail_verbose,
                    )
                else:
                    geom_trial = build_element_geometry(
                        radius_trial.r_over_lambda, state.elem_counts_current.N_per_ring,
                        include_center=state.elem_counts_current.include_center,
                        verbose=False,
                    )
                    pre_trial = pre_ctx

                eval_out = _evaluate_trial_with_fair_union(
                    state, iep_ctx, pool_ctx, a0_ctx, pre_ctx, pre_trial,
                    radius_trial, verbose=detail_verbose,
                )
                rec['quick_psll_db'] = float(eval_out['quick_batch'].worst_psll_db)
                if eval_out['quick_reject']:
                    _install_work_dynamic_pool_state(pool_ctx, eval_out['current_dynamic'])
                    _refresh_dynamic_precompute_for_theta_list(
                        pre_ctx, iep_ctx, a0_ctx, pool_ctx,
                        state.current_work_batch.theta_s_deg,
                        mode='work', verbose=detail_verbose,
                    )
                    update_element_runtime_geometry(pre_ctx, state.radius_current, beta=2.0 * np.pi, verbose=detail_verbose)
                    rej = _register_trial_reject(state, main_cfg, sigma_before, 'quick_reject')
                    rec.update(rej)
                    logger.write(
                        '[IT %03d][E%02d] sigma=%.4f solver=%s/%s dR=%.5f N=%s quick=%.4f QUICK_REJECT sigma=%.4f reject=%d'
                        % (
                            state.attempt_counter, state.topology_epoch, sigma_before,
                            solver_name, solve_res.status, float(solve_res.max_abs_dr),
                            'change' if topo['topology_changed'] else 'same',
                            float(eval_out['quick_batch'].worst_psll_db),
                            state.sigma_current, state.consecutive_rejects,
                        )
                    )
                else:
                    current_cmp = eval_out['current_cmp_batch']
                    trial_cmp = eval_out['trial_cmp_batch']
                    improvement = float(eval_out['improvement_db'])
                    rec['current_cmp_psll_db'] = float(current_cmp.worst_psll_db)
                    rec['trial_cmp_psll_db'] = float(trial_cmp.worst_psll_db)
                    rec['improvement_db'] = improvement
                    accepted = bool(
                        (np.isposinf(improvement))
                        or (np.isfinite(improvement) and improvement > float(main_cfg.work_accept_tol_db))
                    )
                    if accepted:
                        old_sig = tuple(state.topology_signature)
                        state.radius_current = radius_trial
                        state.elem_counts_current = topo['counts_trial'] if topo['topology_changed'] else state.elem_counts_current
                        state.elem_geom_current = geom_trial
                        if topo['topology_changed']:
                            _copy_precompute_context_into(pre_ctx, pre_trial)
                            state.topology_epoch += 1
                            state.topology_signature = topo['trial_signature']
                            state.topology_history.append({
                                'attempt': int(state.attempt_counter),
                                'event': 'transition_accept',
                                'old_signature': list(old_sig),
                                'new_signature': list(state.topology_signature),
                                'improvement_db': improvement,
                            })
                        else:
                            state.topology_signature = old_sig
                        state.current_work_batch = _compact_accepted_work_dynamic_state(
                            iep_ctx, pool_ctx, state.mainlobe_state_current, a0_ctx,
                            pre_ctx, state.radius_current, trial_cmp,
                            verbose=detail_verbose,
                        )
                        state.current_full_validated = False
                        _register_trial_accept(
                            state, main_cfg, improvement, topo['topology_changed'],
                        )
                        rec['accepted'] = True
                        rec['reason'] = 'topology_accept' if topo['topology_changed'] else 'continuous_accept'
                        rec['sigma_after'] = float(state.sigma_current)
                        rec['consecutive_rejects'] = int(state.consecutive_rejects)
                        rec['rejects_at_sigma_min'] = int(state.rejects_at_sigma_min)
                        rec['accepted_since_full'] = int(state.accepted_since_full)
                        if topo['topology_changed']:
                            logger.write(
                                '[IT %03d][E%02d] sigma=%.4f solver=%s/%s dR=%.5f N:%s->%s cmp=%.4f->%.4f TOPO_ACCEPT sigma=%.4f'
                                % (
                                    state.attempt_counter, state.topology_epoch - 1,
                                    sigma_before, solver_name, solve_res.status,
                                    float(solve_res.max_abs_dr), _format_N(old_sig),
                                    _format_N(state.topology_signature),
                                    float(current_cmp.worst_psll_db), float(trial_cmp.worst_psll_db),
                                    state.sigma_current,
                                )
                            )
                            _run_full_scout_event(
                                state, iep_ctx, pool_ctx, a0_ctx, pre_ctx,
                                main_cfg, logger, reason='topology_transition',
                                verbose=detail_verbose,
                            )
                        else:
                            logger.write(
                                '[IT %03d][E%02d] sigma=%.4f solver=%s/%s dR=%.5f N=same quick=%.4f cmp=%.4f->%.4f ACCEPT sigma=%.4f'
                                % (
                                    state.attempt_counter, state.topology_epoch,
                                    sigma_before, solver_name, solve_res.status,
                                    float(solve_res.max_abs_dr), float(eval_out['quick_batch'].worst_psll_db),
                                    float(current_cmp.worst_psll_db), float(trial_cmp.worst_psll_db),
                                    state.sigma_current,
                                )
                            )
                    else:
                        _install_work_dynamic_pool_state(pool_ctx, eval_out['current_dynamic'])
                        _refresh_dynamic_precompute_for_theta_list(
                            pre_ctx, iep_ctx, a0_ctx, pool_ctx,
                            state.current_work_batch.theta_s_deg,
                            mode='work', verbose=detail_verbose,
                        )
                        update_element_runtime_geometry(pre_ctx, state.radius_current, beta=2.0 * np.pi, verbose=detail_verbose)
                        rej = _register_trial_reject(
                            state, main_cfg, sigma_before,
                            'topology_reject' if topo['topology_changed'] else 'fair_compare_reject',
                        )
                        rec.update(rej)
                        logger.write(
                            '[IT %03d][E%02d] sigma=%.4f solver=%s/%s dR=%.5f N=%s cmp=%.4f->%.4f REJECT sigma=%.4f reject=%d'
                            % (
                                state.attempt_counter, state.topology_epoch,
                                sigma_before, solver_name, solve_res.status,
                                float(solve_res.max_abs_dr),
                                'change' if topo['topology_changed'] else 'same',
                                float(current_cmp.worst_psll_db), float(trial_cmp.worst_psll_db),
                                state.sigma_current, state.consecutive_rejects,
                            )
                        )

            rec.setdefault('sigma_after', float(state.sigma_current))
            rec.setdefault('consecutive_rejects', int(state.consecutive_rejects))
            rec.setdefault('rejects_at_sigma_min', int(state.rejects_at_sigma_min))
            rec.setdefault('accepted_since_full', int(state.accepted_since_full))
            state.attempt_history.append(rec)

            if (
                rec.get('accepted', False)
                and not rec.get('N_changed', False)
                and int(main_cfg.full_scout_every_accepts) > 0
                and state.accepted_since_full >= int(main_cfg.full_scout_every_accepts)
            ):
                _run_full_scout_event(
                    state, iep_ctx, pool_ctx, a0_ctx, pre_ctx,
                    main_cfg, logger, reason='periodic', verbose=detail_verbose,
                )

            stop_req = evaluate_stop_request(state, main_cfg)
            if stop_req['requested']:
                pending_stop = stop_req
                audit = _run_final_full_audit(
                    state, iep_ctx, pool_ctx, a0_ctx, pre_ctx, element_cfg,
                    main_cfg, logger, verbose=detail_verbose,
                )
                final_audit_performed = True
                if stop_req['hard'] or not audit['continue_required']:
                    state.stop_reason = str(stop_req['reason'])
                    state.stop_detail = dict(stop_req['detail'])
                    break
                # A soft stop discovered a new topology/work constraint.
                state.consecutive_rejects = 0
                state.rejects_at_sigma_min = 0
                state.full_no_improve_count = 0
                pending_stop = None

        if not state.stop_reason:
            state.stop_reason = 'max_iters'
            state.stop_detail = {}
        if bool(main_cfg.force_final_full_audit) and not final_audit_performed:
            _run_final_full_audit(
                state, iep_ctx, pool_ctx, a0_ctx, pre_ctx, element_cfg,
                main_cfg, logger, verbose=detail_verbose,
            )
            final_audit_performed = True
        if state.best_full_snapshot is None:
            raise RuntimeError('No full-validated checkpoint is available')

        final_snapshot = state.best_full_snapshot
        final_output_radius = _clone_radius_result(final_snapshot.radius_result)
        final_output_counts = _clone_ring_count_result(final_snapshot.elem_counts_result)
        final_output_geom = _clone_element_geometry(final_snapshot.elem_geom_result)
        final_output_mainlobe = _compute_final_mainlobe_domains(
            iep_ctx, pool_ctx, final_output_radius, final_output_counts,
            mainlobe_cfg, verbose=detail_verbose,
        )
        runtime_seconds = float(_time.time() - start_time)
        logger.write(
            '[STOP] reason=%s attempts=%d accepts=%d full=%d epochs=%d'
            % (
                state.stop_reason, state.attempt_counter, state.accept_counter,
                state.full_scout_count, state.topology_epoch,
            )
        )
        logger.write(
            '[FINAL] full=%.4f dB theta=%.1f r=%s N=%s'
            % (
                float(final_snapshot.batch_result.worst_psll_db),
                float(final_snapshot.batch_result.worst_theta_s_deg),
                np.round(final_output_radius.r_over_lambda, 6).tolist(),
                np.asarray(final_output_counts.N_per_ring, dtype=np.int64).tolist(),
            )
        )
        core = {
            'state_final': state,
            'best_work_snapshot': None,
            'best_full_snapshot': state.best_full_snapshot,
            'final_output_radius': final_output_radius,
            'final_output_counts': final_output_counts,
            'final_output_geom': final_output_geom,
            'final_output_mainlobe': final_output_mainlobe,
            'theta_s_full_deg': get_full_theta_s_deg(pool_ctx),
            'theta_s_work_deg_final': get_current_work_theta_s_deg(pool_ctx),
            'stage1_geometry_runtime': geometry_check,
            'static_pool_partition': dict(pool_ctx.summary.get('partition_info', {}) or {}),
            'mainlobe_policy': {
                'optimization_mainlobe_frozen': bool(mainlobe_cfg.freeze_during_optimization),
                'initial_source': str(mainlobe_cfg.source),
                'final_source': str(mainlobe_cfg.final_source),
            },
            'initial_mainlobe': initial_mainlobe,
            'initial_full_batch': initial_full_summary,
            'solver_usage': solver_usage,
            'runtime_seconds': runtime_seconds,
            'final_full_audit_performed': final_audit_performed,
            'output_log_path': log_path,
            'save_attempt_csv': bool(main_cfg.save_attempt_csv),
            'save_full_csv': bool(main_cfg.save_full_csv),
            'config': _record_to_plain_dict(main_cfg),
            'legacy_unused_config': {
                'topology_refresh_every_attempts': main_cfg.topology_refresh_every_attempts,
                'grace_psll_tol_db': main_cfg.grace_psll_tol_db,
                'patience_no_improve': main_cfg.patience_no_improve,
                'oscillation_window': main_cfg.oscillation_window,
                'oscillation_psll_tol_db': main_cfg.oscillation_psll_tol_db,
                'oscillation_radius_tol': main_cfg.oscillation_radius_tol,
            },
        }
        npz_path, json_path, attempt_path, full_path = finalize_stage1_outputs(
            core, output_dir, verbose=detail_verbose,
        )
        logger.write('[SAVE] %s' % npz_path)
        logger.write('[SAVE] %s' % json_path)
        return Stage1MainRunResult(
            state_final=state, best_work_snapshot=None,
            best_full_snapshot=state.best_full_snapshot,
            final_output_radius=final_output_radius,
            final_output_counts=final_output_counts,
            final_output_geom=final_output_geom,
            final_output_mainlobe=final_output_mainlobe,
            final_null_mainlobe_deg=np.asarray(final_output_mainlobe.alpha_null_deg, dtype=np.float64),
            final_guard_null_mainlobe_deg=np.asarray(final_output_mainlobe.alpha_guard_null_deg, dtype=np.float64),
            final_3db_mainlobe_deg=np.asarray(final_output_mainlobe.alpha_3db_deg, dtype=np.float64),
            output_npz_path=npz_path, output_json_path=json_path,
            output_attempt_csv_path=attempt_path,
            output_full_csv_path=full_path, output_log_path=log_path,
        )
    finally:
        if _stdout_redirect is not None:
            _stdout_redirect.__exit__(None, None, None)
        if _stdout_sink is not None:
            _stdout_sink.close()
        logger.close()


__all__ += [
    'Stage1MainConfig', 'Stage1FullCheckpoint', 'Stage1BestSnapshot',
    'Stage1MainState', 'Stage1MainRunResult', 'Stage1CompactLogger',
    'bootstrap_stage1_full_then_work', 'set_work_scan_angles_from_full_feedback',
    'evaluate_stop_request', 'finalize_stage1_outputs',
    'run_a0cra_stage1_main',
]
