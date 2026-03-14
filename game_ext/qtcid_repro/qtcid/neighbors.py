from __future__ import annotations

from typing import Dict, List

from game_ext.qtcid_repro.types import Node
from game_ext.qtcid_repro.utils import euclidean_distance


def build_neighbor_map(nodes: list[Node], communication_range: float) -> dict[int, list[int]]:
    neighbors: dict[int, list[int]] = {node.node_id: [] for node in nodes}

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ni = nodes[i]
            nj = nodes[j]

            d = euclidean_distance(ni.pos.x, ni.pos.y, nj.pos.x, nj.pos.y)
            if d <= communication_range:
                neighbors[ni.node_id].append(nj.node_id)
                neighbors[nj.node_id].append(ni.node_id)

    return neighbors


def active_neighbors_of(
    node_id: int,
    nodes: list[Node],
    neighbor_map: dict[int, list[int]],
    emin: float,
) -> list[int]:
    node_by_id = {n.node_id: n for n in nodes}
    result: list[int] = []

    for nid in neighbor_map.get(node_id, []):
        n = node_by_id[nid]
        if n.is_active(emin):
            result.append(nid)

    return result