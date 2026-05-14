"""
main.py — FastAPI HTTP 레이어 (Phase 4.1)
"""
from __future__ import annotations

import functools
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

import phase4 as _phase4
from core import (
    CuttingError,
    Dims,
    EngineSettings,
    InvalidCutError,
    Node,
    OptimizationGoal,
    Part,
    Stock,
    TrimmingMargins,
)
from packer import StripAdapter
from virtual_strip import VirtualStrip

_PHASE4_TIME_BUDGET: float = 30.0

app = FastAPI(
    title="3D Guillotine Cut Optimizer",
    version="4.1.0",
    description="판재 최적 재단 API — Phase 4.1 Strip Engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrimmingIn(BaseModel):
    x: float = Field(0.0, ge=0, description="X축 양단 여백 합산 (mm)")
    y: float = Field(0.0, ge=0, description="Y축 양단 여백 합산 (mm)")
    z: float = Field(0.0, ge=0, description="Z축 양단 여백 합산 (mm)")

class SettingsIn(BaseModel):
    kerf: float = Field(3.0, ge=0, le=50, description="톱날 두께 손실 (mm)")
    trimming: TrimmingIn = Field(default_factory=TrimmingIn)
    optimization_goal: str = Field("MINIMIZE_WASTE")
    machine_speed_mm_per_sec: float = Field(50.0, gt=0, description="기계 절단 속도 (mm/s)")
    setup_time_sec: float = Field(10.0, ge=0, description="1회 절단당 셋팅 시간 (s)")

class StockIn(BaseModel):
    id: str = Field(..., min_length=1)
    l: float = Field(..., gt=0, description="길이 (mm)")
    w: float = Field(..., gt=0, description="너비 (mm)")
    t: float = Field(..., gt=0, description="두께 (mm)")
    qty: int = Field(..., ge=1, le=1000)

class PartIn(BaseModel):
    id: str = Field(..., min_length=1)
    l: float = Field(..., gt=0)
    w: float = Field(..., gt=0)
    t: float = Field(..., gt=0)
    qty: int = Field(..., ge=1, le=10000)
    lock_z: bool = True
    allow_xy_rotation: bool = True
    priority: int = Field(0, ge=0, le=100)
    color: str = Field("#4f8ef7")

class OptimizeRequest(BaseModel):
    settings: SettingsIn
    stocks: List[StockIn] = Field(..., min_length=1)
    parts: List[PartIn] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_parts_fit_in_stocks(self) -> "OptimizeRequest":
        kerf = self.settings.kerf
        trim = self.settings.trimming

        for part in self.parts:
            part_orientations = [(part.l, part.w, part.t)]
            if part.allow_xy_rotation:
                part_orientations.append((part.w, part.l, part.t))
            if not part.lock_z:
                from itertools import permutations
                part_orientations = list(set(permutations([part.l, part.w, part.t])))

            fits = False
            for stock in self.stocks:
                ul = stock.l - trim.x
                uw = stock.w - trim.y
                ut = stock.t - trim.z
                if ul <= 0 or uw <= 0 or ut <= 0:
                    continue
                for pl, pw, pt in part_orientations:
                    if pl <= ul and pw <= uw and pt <= ut:
                        fits = True
                        break
                if fits:
                    break

            if not fits:
                raise ValueError(
                    f"부품 '{part.id}' ({part.l}x{part.w}x{part.t}mm)은 "
                    f"어떤 원장에도 들어가지 않습니다."
                )
        return self

class DimsOut(BaseModel):
    l: float
    w: float
    t: float
    volume: float

class OriginOut(BaseModel):
    x: float
    y: float
    z: float

class OffcutOut(BaseModel):
    node_id: str
    stock_id: str
    dims: DimsOut
    origin: OriginOut

class CutRecordOut(BaseModel):
    cut_id: str
    axis: str
    position: float
    kerf: float
    parent_node_id: str

class PlacedPartOut(BaseModel):
    node_id: str
    stock_id: str
    part_id: str
    color: str
    placed_dims: DimsOut
    origin: OriginOut
    cut_history: List[CutRecordOut]
    depth: int
    from_strip: bool = False
    strip_id: Optional[str] = None

class StockSummaryOut(BaseModel):
    stock_id: str
    original_dims: DimsOut
    usable_dims: DimsOut
    placed_count: int
    placed_volume: float
    usable_volume: float
    efficiency_pct: float

class OptimizeResponse(BaseModel):
    placements: List[PlacedPartOut]
    offcuts: List[OffcutOut]
    unplaced: Dict[str, int]
    stock_summaries: List[StockSummaryOut]
    failures: List[str]
    stats: Dict[str, Any]

def _build_engine_settings(s: SettingsIn) -> EngineSettings:
    return EngineSettings(
        kerf=s.kerf,
        trimming=TrimmingMargins(x=s.trimming.x, y=s.trimming.y, z=s.trimming.z),
        optimization_goal=OptimizationGoal.MINIMIZE_WASTE,
    )

def _build_stocks(raw: List[StockIn], trim: TrimmingIn) -> List[Stock]:
    return [
        Stock(
            id=s.id,
            dims=Dims(l=s.l, w=s.w, t=s.t),
            qty=s.qty,
            trimming=TrimmingMargins(x=trim.x, y=trim.y, z=trim.z),
        )
        for s in raw
    ]

def _build_parts(raw: List[PartIn]) -> List[Part]:
    return [
        Part(
            id=p.id,
            dims=Dims(l=p.l, w=p.w, t=p.t),
            qty=p.qty,
            lock_z=p.lock_z,
            allow_xy_rotation=p.allow_xy_rotation,
            priority=p.priority,
            color=p.color,
        )
        for p in raw
    ]

def _make_cut_records(node: Node) -> List[CutRecordOut]:
    return [
        CutRecordOut(
            cut_id=c.cut_id,
            axis=c.axis.value,
            position=c.position,
            kerf=c.kerf,
            parent_node_id=c.parent_node_id,
        )
        for c in node.collect_cut_history()
    ]

def _explode_strip_node(node: Node, strip: VirtualStrip, kerf: float) -> List[PlacedPartOut]:
    shared_cut_records = _make_cut_records(node)
    base_x = node.origin.x
    base_y = node.origin.y
    base_z = node.origin.z
    current_x = base_x
    part_seq = 0
    results: List[PlacedPartOut] = []

    for p_obj, p_dims, qty in strip.internal_parts:
        for _ in range(qty):
            results.append(PlacedPartOut(
                node_id=f"{node.node_id}_x{part_seq:03d}",
                stock_id=node.stock_id or "",
                part_id=p_obj.id,
                color=p_obj.color,
                placed_dims=DimsOut(
                    l=p_dims.l, w=p_dims.w, t=p_dims.t,
                    volume=p_dims.volume,
                ),
                origin=OriginOut(x=current_x, y=base_y, z=base_z),
                cut_history=shared_cut_records,
                depth=node.depth,
                from_strip=True,
                strip_id=strip.strip_id,
            ))
            current_x += p_dims.l + kerf
            part_seq += 1

    last_end_x = current_x - kerf
    expected_end_x = base_x + strip.dims.l
    if abs(last_end_x - expected_end_x) > 0.5:
        import logging
        logging.warning(
            f"Strip {strip.strip_id} 끝점 불일치: "
            f"계산={last_end_x:.2f} 기대={expected_end_x:.2f} "
            f"diff={abs(last_end_x - expected_end_x):.2f}"
        )
    return results

def _node_to_placed_out(node: Node) -> PlacedPartOut:
    dims = node.placed_part_dims
    origin = node.origin
    return PlacedPartOut(
        node_id=node.node_id,
        stock_id=node.stock_id or "",
        part_id=node.placed_part.id,
        color=node.placed_part.color,
        placed_dims=DimsOut(l=dims.l, w=dims.w, t=dims.t, volume=dims.volume),
        origin=OriginOut(x=origin.x, y=origin.y, z=origin.z),
        cut_history=_make_cut_records(node),
        depth=node.depth,
        from_strip=False,
        strip_id=None,
    )

def _build_placements(result: "_phase4.Phase4Result", kerf: float) -> List[PlacedPartOut]:
    strip_map: Dict[str, VirtualStrip] = {s.strip_id: s for s in result.strips}
    placements: List[PlacedPartOut] = []

    for node in result.occupied_nodes:
        placed = node.placed_part
        if placed is None:
            continue

        if StripAdapter.is_strip_part(placed):
            strip_id = StripAdapter.extract_strip_id(placed)
            strip = strip_map.get(strip_id)
            if strip is None:
                placements.append(_node_to_placed_out(node))
            else:
                placements.extend(_explode_strip_node(node, strip, kerf))
        else:
            placements.append(_node_to_placed_out(node))
    return placements

def _build_stock_summaries_phase4(
    placements: List[PlacedPartOut],
    result: "_phase4.Phase4Result",
    stocks: List[Stock],
) -> List[StockSummaryOut]:
    stocks_map: Dict[str, Stock] = {s.id: s for s in stocks}
    counts: Dict[str, int] = defaultdict(int)
    volumes: Dict[str, float] = defaultdict(float)

    for p in placements:
        sid = p.stock_id or "unknown"
        counts[sid] += 1
        volumes[sid] += p.placed_dims.volume

    strip_slot_counts: Counter = Counter(s.stock.id for s in result.eval_result.slots_used)
    fallback_slot_counts: Dict[str, int] = {}
    nodes_by_stock: Dict[str, List[Node]] = defaultdict(list)

    for node in result.occupied_nodes:
        if node.stock_id:
            nodes_by_stock[node.stock_id].append(node)

    for sid, nodes in nodes_by_stock.items():
        stock = stocks_map.get(sid)
        if stock is None:
            continue
        try:
            usable_l = stock.usable_dims.l
        except ValueError:
            usable_l = stock.dims.l

        max_slot_idx = max((int(n.origin.x // usable_l) for n in nodes)) if nodes else 0
        fallback_slot_counts[sid] = max_slot_idx + 1

    all_stock_ids = set(counts.keys())
    slot_counts: Dict[str, int] = {}
    for sid in all_stock_ids:
        from_strip = strip_slot_counts.get(sid, 0)
        from_fallback = fallback_slot_counts.get(sid, 1)
        slot_counts[sid] = max(from_strip, from_fallback, 1)

    summaries: List[StockSummaryOut] = []
    for sid in counts:
        stock = stocks_map.get(sid)
        if stock is None:
            continue

        orig = stock.dims
        try:
            usable = stock.usable_dims
        except ValueError:
            usable = orig

        n_slots = slot_counts.get(sid, 1)
        placed_vol = volumes[sid]
        usable_vol = usable.volume * n_slots
        eff = round((placed_vol / usable_vol * 100) if usable_vol > 0 else 0.0, 2)

        summaries.append(StockSummaryOut(
            stock_id=sid,
            original_dims=DimsOut(l=orig.l, w=orig.w, t=orig.t, volume=orig.volume),
            usable_dims=DimsOut(l=usable.l, w=usable.w, t=usable.t, volume=usable.volume),
            placed_count=counts[sid],
            placed_volume=placed_vol,
            usable_volume=usable_vol,
            efficiency_pct=eff,
        ))
    return summaries

def _collect_cuts_with_dims(occupied_nodes: List[Node]) -> Dict[str, tuple]:
    cuts_map: Dict[str, tuple] = {}
    for node in occupied_nodes:
        ancestor = node
        while ancestor is not None:
            if ancestor.cut is not None and ancestor.cut.cut_id not in cuts_map:
                cut = ancestor.cut
                parent_dims = ancestor.parent.dims if ancestor.parent is not None else ancestor.dims
                cuts_map[cut.cut_id] = (cut, parent_dims)
            ancestor = ancestor.parent
    return cuts_map

def _cut_travel_mm(cut, parent_dims: Dims) -> float:
    ax = cut.axis.value
    if ax == "X":
        return parent_dims.w
    else:
        return parent_dims.l

def _estimate_work_time(
    occupied_nodes: List[Node],
    machine_speed_mm_per_sec: float,
    setup_time_sec: float,
) -> Dict[str, Any]:
    cuts_map = _collect_cuts_with_dims(occupied_nodes)
    setup_count = len(cuts_map)
    total_travel = sum(_cut_travel_mm(c, d) for c, d in cuts_map.values())
    pure_cut_time = round(total_travel / machine_speed_mm_per_sec if machine_speed_mm_per_sec > 0 else 0.0, 2)
    total_time = round(pure_cut_time + setup_count * setup_time_sec, 2)

    return {
        "setup_count": setup_count,
        "pure_cut_time_sec": pure_cut_time,
        "total_estimated_time_sec": total_time,
    }

def _run_packing(
    engine_settings: EngineSettings,
    stocks: List[Stock],
    parts: List[Part],
) -> "_phase4.Phase4Result":
    return _phase4.pack_parts_phase4(
        settings=engine_settings,
        stocks=stocks,
        parts=parts,
        time_budget=_PHASE4_TIME_BUDGET,
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "4.1.0",
        "engine": "Phase 4.1 Strip Engine",
    }

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(body: OptimizeRequest) -> OptimizeResponse:
    engine_settings = _build_engine_settings(body.settings)
    stocks = _build_stocks(body.stocks, body.settings.trimming)
    parts = _build_parts(body.parts)
    kerf = engine_settings.kerf
    machine_speed_mm_per_sec = body.settings.machine_speed_mm_per_sec
    setup_time_sec = body.settings.setup_time_sec

    try:
        result: _phase4.Phase4Result = await run_in_threadpool(
            functools.partial(_run_packing, engine_settings, stocks, parts)
        )
    except InvalidCutError as e:
        raise HTTPException(status_code=400, detail={"error": "물리 제약 위반", "detail": str(e), "error_code": "INVALID_CUT"})
    except CuttingError as e:
        raise HTTPException(status_code=422, detail={"error": "절단 엔진 오류", "detail": str(e), "error_code": "CUTTING_ERROR"})
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "입력값 오류", "detail": str(e), "error_code": "VALUE_ERROR"})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "서버 오류", "detail": str(e), "error_code": "INTERNAL_ERROR"})

    placements = _build_placements(result, kerf)
    summaries = _build_stock_summaries_phase4(placements, result, stocks)

    offcuts = [
        OffcutOut(
            node_id=n.node_id,
            stock_id=n.stock_id or "",
            dims=DimsOut(l=n.dims.l, w=n.dims.w, t=n.dims.t, volume=n.dims.volume),
            origin=OriginOut(x=n.origin.x, y=n.origin.y, z=n.origin.z),
        )
        for n in result.free_nodes
    ]

    failures = [
        f"부품 '{pid}' {qty}개를 배치하지 못했습니다. 원장 공간이 부족합니다."
        for pid, qty in result.unplaced.items()
    ]

    placed_vol = sum(s.placed_volume for s in summaries)
    efficiency_vol = sum(s.usable_volume for s in summaries)
    yield_vol = sum(
        (s.usable_dims.volume if hasattr(s, 'usable_dims') else s.dims.volume) * s.qty
        for s in stocks
    )

    overall_eff = round((placed_vol / efficiency_vol * 100) if efficiency_vol > 0 else 0.0, 2)
    yield_rate_pct = round((placed_vol / yield_vol * 100) if yield_vol > 0 else 0.0, 2)

    s = result.stats
    time_est = _estimate_work_time(result.occupied_nodes, machine_speed_mm_per_sec, setup_time_sec)

    stats: Dict[str, Any] = {
        "total_placed": len(placements),
        "total_unplaced_types": len(result.unplaced),
        "total_placed_volume": placed_vol,
        "total_usable_volume": efficiency_vol,
        "overall_efficiency_pct": overall_eff,
        "stocks_used": result.stocks_used,
        "processing_time_sec": round(result.processing_time, 4),
        "yield_rate_pct": yield_rate_pct,
        "n_groups": s.n_groups,
        "n_strips": s.n_strips,
        "strip_assigned": s.n_assigned,
        "strip_unassigned": s.n_unassigned_strips,
        "strip_assignment_rate": round(s.strip_assignment_rate * 100, 1) if hasattr(s, 'strip_assignment_rate') else 0.0,
        "fallback_placed": s.n_fallback_placed,
        "step_times": {
            "step1_dp_sec": round(s.step1_sec, 4),
            "step2_strip_sec": round(s.step2_sec, 4),
            "step3_assign_sec": round(s.step3_sec, 4),
            "step4_place_sec": round(s.step4_sec, 4),
        },
        "setup_count": time_est["setup_count"],
        "pure_cut_time_sec": time_est["pure_cut_time_sec"],
        "total_estimated_time_sec": time_est["total_estimated_time_sec"],
        "machine_speed_mm_per_sec": machine_speed_mm_per_sec,
        "setup_time_sec": setup_time_sec,
    }

    return OptimizeResponse(
        placements=placements,
        offcuts=offcuts,
        unplaced=result.unplaced,
        stock_summaries=summaries,
        failures=failures,
        stats=stats,
    )