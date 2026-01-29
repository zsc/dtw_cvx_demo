from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


class NoPathError(RuntimeError):
    pass


@dataclass
class FlowResult:
    path: List[Tuple[int, int]]
    total_cost: float


def _load_min_cost_flow():
    try:
        from ortools.graph import pywrapgraph
        return pywrapgraph.SimpleMinCostFlow
    except Exception:
        try:
            from ortools.graph.python import min_cost_flow
            return min_cost_flow.SimpleMinCostFlow
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import OR-Tools SimpleMinCostFlow. Install ortools in the active environment."
            ) from exc
    return None


def _call_mcf(mcf, camel: str, snake: str, *args):
    if hasattr(mcf, camel):
        return getattr(mcf, camel)(*args)
    return getattr(mcf, snake)(*args)


def solve_dtw_mincost_flow(
    cost_map: Dict[Tuple[int, int], float],
    allowed_js: List[List[int]],
    T1: int,
    T2: int,
    step_penalty: Dict[str, float],
    cost_scale: float = 1_000_000,
) -> FlowResult:
    if T1 == 0 or T2 == 0:
        raise NoPathError("Empty feature sequence")

    if (0, 0) not in cost_map or (T1 - 1, T2 - 1) not in cost_map:
        raise NoPathError("Band excludes start or end node")

    SimpleMinCostFlow = _load_min_cost_flow()
    mcf = SimpleMinCostFlow()

    node_id: Dict[Tuple[int, int], int] = {}
    nodes: List[Tuple[int, int]] = []

    for i in range(T1):
        for j in allowed_js[i]:
            node_id[(i, j)] = len(nodes)
            nodes.append((i, j))

    def add_arc(u: Tuple[int, int], v: Tuple[int, int], penalty: float):
        if v not in node_id:
            return
        cost = cost_map[v] + penalty
        u_id = node_id[u]
        v_id = node_id[v]
        _call_mcf(
            mcf,
            "AddArcWithCapacityAndUnitCost",
            "add_arc_with_capacity_and_unit_cost",
            u_id,
            v_id,
            1,
            int(round(cost * cost_scale)),
        )

    for i in range(T1):
        for j in allowed_js[i]:
            u = (i, j)
            # horiz
            if j + 1 < T2:
                add_arc(u, (i, j + 1), step_penalty.get("horiz", 0.0))
            # vert
            if i + 1 < T1:
                add_arc(u, (i + 1, j), step_penalty.get("vert", 0.0))
            # diag
            if i + 1 < T1 and j + 1 < T2:
                add_arc(u, (i + 1, j + 1), step_penalty.get("diag", 0.0))

    source_id = node_id[(0, 0)]
    sink_id = node_id[(T1 - 1, T2 - 1)]
    _call_mcf(mcf, "SetNodeSupply", "set_node_supply", source_id, 1)
    _call_mcf(mcf, "SetNodeSupply", "set_node_supply", sink_id, -1)

    status = _call_mcf(mcf, "Solve", "solve")
    if status != mcf.OPTIMAL:
        raise NoPathError("Min-cost flow failed to find an optimal path")

    next_node: Dict[int, int] = {}
    total_cost = 0.0
    for arc in range(_call_mcf(mcf, "NumArcs", "num_arcs")):
        if _call_mcf(mcf, "Flow", "flow", arc) == 1:
            u_id = _call_mcf(mcf, "Tail", "tail", arc)
            v_id = _call_mcf(mcf, "Head", "head", arc)
            next_node[u_id] = v_id
            total_cost += _call_mcf(mcf, "UnitCost", "unit_cost", arc) / cost_scale

    if source_id not in next_node:
        raise NoPathError("No outgoing flow from source")

    path_nodes: List[Tuple[int, int]] = []
    current = source_id
    visited = set()
    while True:
        if current in visited:
            raise NoPathError("Cycle detected in flow")
        visited.add(current)
        path_nodes.append(nodes[current])
        if current == sink_id:
            break
        if current not in next_node:
            raise NoPathError("Broken flow path")
        current = next_node[current]

    if path_nodes[0] != (0, 0) or path_nodes[-1] != (T1 - 1, T2 - 1):
        raise NoPathError("Recovered path does not reach target")

    return FlowResult(path=path_nodes, total_cost=total_cost)
