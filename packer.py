"""
packer.py — 탐색 & 배치 엔진 (Corner-First & 최대 단일 잔재 보존 알고리즘 - 오타 수정 완료)
"""
from __future__ import annotations

import heapq
import time
import random
import copy
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Tuple

from core import (
    CutAxis, Dims, EngineSettings, InvalidCutError, Node, NodeState, Part, Stock,
    _get_axis, _new_id, create_root_node, split_node,
)

_EPSILON = 0.5

class NodeHeap:
    def __init__(self):
        self._heap: list = []
        self._removed: set = set()

    def push(self, node: Node):
        # 정렬 키: (-depth, -volume)
        # -depth: 절단 깊이가 깊을수록(자투리일수록) 먼저 꺼냄 → 자투리 우선 소진
        # -volume: 같은 깊이면 큰 공간 우선 → Best-Fit 효율 보존
        # 두께(Z축) 보존: _offcut_score_for_order의 rem_t 가중치가
        # 좌표 기반 외부 강제 없이 순수하게 Z절단 순서를 결정하도록 위임
        heapq.heappush(self._heap, (-node.depth, -node.volume, node.node_id, node))

    def pop(self) -> Optional[Node]:
        while self._heap:
            neg_depth, neg_vol, node_id, node = heapq.heappop(self._heap)
            if node_id in self._removed:
                continue
            if node.state != NodeState.FREE:
                self._removed.add(node_id)
                continue
            return node
        return None

    def invalidate(self, node_id: str):
        self._removed.add(node_id)

    def __len__(self) -> int:
        return len(self._heap)

@dataclass(order=True)
class PlacementCandidate:
    neg_estimated_volume: float
    linear_waste: float
    part_idx: int
    rotation_penalty: int
    neg_max_offcut: float
    node_id: str = field(compare=False)
    node: Node = field(compare=False)
    part: Part = field(compare=False)
    orientation: Dims = field(compare=False)
    cut_order: Tuple[CutAxis, ...] = field(compare=False)

_ALL_AXES = [CutAxis.X, CutAxis.Y, CutAxis.Z]
_ALL_ORDERS = list(permutations(_ALL_AXES))

def _get_lwt(dims_obj) -> Tuple[float, float, float]:
    """튜플과 Dims 객체를 모두 지원하는 안전한 치수 추출"""
    l = dims_obj.l if hasattr(dims_obj, 'l') else dims_obj[0]
    w = dims_obj.w if hasattr(dims_obj, 'w') else dims_obj[1]
    t = dims_obj.t if hasattr(dims_obj, 't') else dims_obj[2]
    return l, w, t

def _offcut_score_for_order(
    node: Node, part_dims: Dims, cut_order: Tuple[CutAxis, ...], kerf: float
) -> Optional[float]:
    n_l, n_w, n_t = _get_lwt(node.dims)
    p_l, p_w, p_t = _get_lwt(part_dims)
    
    remaining = {CutAxis.X: n_l, CutAxis.Y: n_w, CutAxis.Z: n_t}
    part_size = {CutAxis.X: p_l, CutAxis.Y: p_w, CutAxis.Z: p_t}
    max_score = -1.0

    for axis in cut_order:
        pos = part_size[axis]
        total = remaining[axis]

        if abs(total - pos) <= _EPSILON:
            remaining[axis] = pos
            continue

        remainder = total - pos - kerf
        if remainder < -_EPSILON:
            return None  

        if remainder <= _EPSILON:
            remaining[axis] = pos
            continue

        rem_l = remaining[CutAxis.X] if axis != CutAxis.X else remainder
        rem_w = remaining[CutAxis.Y] if axis != CutAxis.Y else remainder
        rem_t = remaining[CutAxis.Z] if axis != CutAxis.Z else remainder
        
        short_edge = min(rem_l, rem_w)
        
        if short_edge < 30:
            score = 0.0
        else:
            normalized_l = rem_l / 1000.0
            normalized_w = rem_w / 1000.0
            normalized_t = rem_t / 1000.0
            normalized_short = short_edge / 1000.0
            score = (normalized_short ** 2) * normalized_t * (normalized_l * normalized_w * normalized_t)
            
        if score > max_score:
            max_score = score
            
        remaining[axis] = pos

    return max_score

def _best_cut_order(node: Node, part_dims: Dims, kerf: float) -> Optional[Tuple[Tuple[CutAxis, ...], float]]:
    best_order, best_score = None, -1.0
    for order in _ALL_ORDERS:
        score = _offcut_score_for_order(node, part_dims, order, kerf)
        if score is None: continue
        if score > best_score:
            best_score = score
            best_order = order
    return (best_order, best_score) if best_order else None

def _fit_count(total: float, pdim: float, kerf: float) -> int:
    if pdim > total + _EPSILON: return 0
    return int((total + kerf + _EPSILON) // (pdim + kerf))

def _axis_waste(total: float, pdim: float, kerf: float) -> float:
    count = _fit_count(total, pdim, kerf)
    if count <= 0: return total
    return max(0.0, total - (count * pdim) - (count - 1) * kerf)

def _find_best_candidate(
    node: Node, remaining_parts: Dict[str, int], parts_by_id: Dict[str, Part], kerf: float
) -> Optional[PlacementCandidate]:
    best: Optional[PlacementCandidate] = None
    part_keys = list(remaining_parts.keys())

    for part_id, qty in remaining_parts.items():
        if qty <= 0: continue
        part = parts_by_id[part_id]
        p_idx = part_keys.index(part_id)

        for orientation in part.allowed_orientations():
            n_l, n_w, n_t = _get_lwt(node.dims)
            p_l, p_w, p_t = _get_lwt(orientation)
            
            if hasattr(orientation, 'fits_in'):
                if not orientation.fits_in(node.dims): continue
            else:
                if p_l > n_l + _EPSILON or p_w > n_w + _EPSILON or p_t > n_t + _EPSILON: continue

            cx = _fit_count(n_l, p_l, kerf)
            cy = _fit_count(n_w, p_w, kerf)
            cz = _fit_count(n_t, p_t, kerf)
            est_count = cx * cy * cz  
            
            est_vol = est_count * (p_l * p_w * p_t)

            lw_x = _axis_waste(n_l, p_l, kerf)
            lw_y = _axis_waste(n_w, p_w, kerf)
            lw_z = _axis_waste(n_t, p_t, kerf)
            total_linear_waste = lw_x + lw_y + lw_z

            part_l, part_w, part_t = _get_lwt(part.dims)
            rot_penalty = 0 if (p_l == part_l and p_w == part_w and p_t == part_t) else (1 if p_t == part_t else 2)

            order_result = _best_cut_order(node, orientation, kerf)
            if order_result is None: continue

            best_order, max_offcut = order_result

            candidate = PlacementCandidate(
                neg_estimated_volume=-est_vol,
                linear_waste=total_linear_waste,
                part_idx=p_idx,                 
                rotation_penalty=rot_penalty,   
                neg_max_offcut=-max_offcut,     
                node_id=node.node_id, node=node, part=part,
                orientation=orientation, cut_order=best_order,
            )

            if best is None or candidate < best:
                best = candidate

    return best

def _place_part_on_node(
    node: Node, part: Part, orientation: Dims, cut_order: Tuple[CutAxis, ...], kerf: float,
) -> Tuple[Node, List[Node]]:
    p_l, p_w, p_t = _get_lwt(orientation)
    part_size = { CutAxis.X: p_l, CutAxis.Y: p_w, CutAxis.Z: p_t }
    current = node
    new_free_nodes = []

    for axis in cut_order:
        pos = part_size[axis]
        total = _get_axis(current.dims, axis)
        if abs(total - pos) <= _EPSILON: continue
        child_a, child_b = split_node(current, axis, pos, kerf)
        new_free_nodes.append(child_b)
        current = child_a

    current.state = NodeState.OCCUPIED
    current.placed_part = part
    current.placed_part_dims = orientation
    return current, new_free_nodes

@dataclass
class PackResult:
    occupied_nodes: List[Node]
    unplaced: Dict[str, int]
    processing_time: float
    stocks_used: int
    free_nodes: List[Node] = field(default_factory=list)

def _pack_parts_single(
    settings: EngineSettings, stocks: List[Stock], parts: List[Part]
) -> PackResult:
    start = time.perf_counter()
    kerf = settings.kerf
    parts_by_id = {p.id: p for p in parts}
    remaining = {p.id: p.qty for p in parts}
    occupied_nodes = []
    heap = NodeHeap()
    stocks_used = 0

    stock_pool = [stock for stock in stocks for _ in range(stock.qty)]
    stock_index = 0

    def _open_next_stock() -> bool:
        nonlocal stock_index, stocks_used
        if stock_index >= len(stock_pool): return False

        # ── Best-Fit Bin 선택 ──────────────────────────────────────────
        # 미배치 부품들의 모든 허용 방향을 수집하고,
        # 그 중 가장 큰 부품(최대 footprint 기준)이 들어갈 수 있는
        # 가장 작은(usable_volume 최소) 원장을 우선 선택한다.
        #
        # 폴백: 조건을 만족하는 원장이 없으면 pool 순서대로 꺼낸다.
        remaining_dims: List[Dims] = []
        for pid, qty in remaining.items():
            if qty > 0:
                for orient in parts_by_id[pid].allowed_orientations():
                    remaining_dims.append(orient)

        best_local_idx: Optional[int] = None
        best_local_vol: float = float('inf')

        if remaining_dims:
            # 가장 큰 부품 기준으로 내림차순 정렬 (footprint = l*w 최대)
            sorted_dims = sorted(remaining_dims, key=lambda d: d.l * d.w * d.t, reverse=True)

            for candidate_dims in sorted_dims:
                # 이 치수가 들어갈 수 있는 원장 중 usable_volume이 최소인 것 찾기
                found_for_this_dim = False
                for i in range(stock_index, len(stock_pool)):
                    s = stock_pool[i]
                    ud = s.usable_dims  # trimming 반영
                    if (candidate_dims.l <= ud.l + _EPSILON and
                        candidate_dims.w <= ud.w + _EPSILON and
                        candidate_dims.t <= ud.t + _EPSILON):
                        uv = ud.volume
                        if uv < best_local_vol:
                            best_local_vol = uv
                            best_local_idx = i
                        found_for_this_dim = True
                # 가장 큰 부품이 들어가는 원장을 찾았으면 확정
                if found_for_this_dim and best_local_idx is not None:
                    break

        # 선택된 원장이 있으면 현재 위치와 swap 후 꺼냄
        if best_local_idx is not None and best_local_idx != stock_index:
            stock_pool[stock_index], stock_pool[best_local_idx] = (
                stock_pool[best_local_idx], stock_pool[stock_index]
            )

        stock = stock_pool[stock_index]
        stock_index += 1
        stocks_used += 1
        heap.push(create_root_node(stock))
        return True

    if not _open_next_stock():
        return PackResult([], remaining, 0.0, 0)

    while any(v > 0 for v in remaining.values()):
        node = heap.pop()
        if node is None:
            if not _open_next_stock(): break
            node = heap.pop()
            if node is None: break

        candidate = _find_best_candidate(node, remaining, parts_by_id, kerf)
        if candidate is None:
            node.state = NodeState.DISCARDED
            continue

        occupied, new_free = _place_part_on_node(
            candidate.node, candidate.part, candidate.orientation, candidate.cut_order, kerf
        )
        occupied_nodes.append(occupied)
        remaining[candidate.part.id] -= 1

        for free_node in new_free:
            heap.push(free_node)

    # 튜플 구조: (-depth, -volume, node_id, node) → node는 인덱스 3
    free_nodes = [item[3] for item in heap._heap
                  if item[3].node_id not in heap._removed
                  and item[3].state == NodeState.FREE]
    
    return PackResult(
        occupied_nodes=occupied_nodes,
        unplaced={k: v for k, v in remaining.items() if v > 0},
        processing_time=time.perf_counter() - start,
        stocks_used=stocks_used,
        free_nodes=free_nodes,
    )

def pack_parts(
    settings: EngineSettings, stocks: List[Stock], parts: List[Part],
) -> PackResult:
    """
    5초 GRASP 최적화 엔진
      - NodeHeap: (-depth, -volume) 정렬 → 자투리 우선 소진, Z축 두께 보존
      - GRASP 평가: 1순위 미배치 최소, 2순위 총 선형 로스 최소
      - Best-Fit Bin: 미배치 부품 중 최대 부품이 들어가는 최소 원장 우선 선택
    """
    start_total = time.perf_counter()
    TIME_LIMIT = 5.0  
    
    best_result = None
    best_unplaced = float('inf')
    best_linear_waste = float('inf')  # 2순위: 선형 로스 합산이 작을수록 좋음
    
    # 기본 모드
    best_result = _pack_parts_single(settings, stocks, parts)
    best_unplaced = sum(best_result.unplaced.values())
    # 총 선형 로스 = 각 free_node 부피의 합 (작을수록 빈틈없이 채운 것)
    best_linear_waste = sum(n.volume for n in best_result.free_nodes)

    # GRASP 다중 패스
    while True:
        if time.perf_counter() - start_total > TIME_LIMIT:
            break
            
        test_parts = copy.deepcopy(parts)
        strategy = random.random()
        
        # ✨ 버그 수정 완료: p.l -> p.dims.l 로 접근하도록 모두 교체!
        if strategy < 0.2:
            test_parts.sort(key=lambda p: -(p.dims.l * p.dims.w * p.dims.t))
        elif strategy < 0.4:
            test_parts.sort(key=lambda p: -max(p.dims.l, p.dims.w))
        elif strategy < 0.6:
            test_parts.sort(key=lambda p: -min(p.dims.l, p.dims.w))
        else:
            random.shuffle(test_parts)
            
        test_stocks = copy.deepcopy(stocks)
        
        result = _pack_parts_single(settings, test_stocks, test_parts)
        unplaced = sum(result.unplaced.values())
        # 총 선형 로스: 남은 FREE 노드 부피 합 (미배치 공간 총량)
        linear_waste = sum(n.volume for n in result.free_nodes)

        # 1순위: 미배치 부품 수 최소
        # 2순위: 총 선형 로스 최소 (자투리 공간을 알뜰하게 활용한 도면 선택)
        if unplaced < best_unplaced or (
            unplaced == best_unplaced and linear_waste < best_linear_waste
        ):
            best_unplaced = unplaced
            best_linear_waste = linear_waste
            best_result = result
            
    best_result.processing_time = time.perf_counter() - start_total
    return best_result
