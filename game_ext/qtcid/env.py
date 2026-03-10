from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import math
import random


class NodeState(str, Enum):
    GOOD = "GOOD"
    BAD = "BAD"
    EVICTED = "EVICTED"


@dataclass
class Node:
    node_id: int
    x: float
    y: float
    energy: float
    state: NodeState = NodeState.GOOD
    alive: bool = True

    def distance_to(self, other: "Node") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class IoDEnvironment:
    n_nodes: int
    area_size: float
    comm_radius: float
    initial_energy: float
    capture_probability: float
    malicious_vote_probability: float
    tx_energy_cost: float
    audit_energy_cost: float
    seed: Optional[int] = None
    nodes: List[Node] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        if not self.nodes:
            self.reset()

    def reset(self) -> None:
        self.nodes = []
        for i in range(self.n_nodes):
            self.nodes.append(
                Node(
                    node_id=i,
                    x=self.rng.uniform(0.0, self.area_size),
                    y=self.rng.uniform(0.0, self.area_size),
                    energy=self.initial_energy,
                    state=NodeState.GOOD,
                    alive=True,
                )
            )

    def active_nodes(self) -> List[Node]:
        return [node for node in self.nodes if node.state != NodeState.EVICTED and node.alive]

    def good_nodes(self) -> List[Node]:
        return [node for node in self.nodes if node.state == NodeState.GOOD and node.alive]

    def bad_nodes(self) -> List[Node]:
        return [node for node in self.nodes if node.state == NodeState.BAD and node.alive]

    def evicted_nodes(self) -> List[Node]:
        return [node for node in self.nodes if node.state == NodeState.EVICTED]

    def get_neighbors(self, node_id: int) -> List[Node]:
        source = self.nodes[node_id]
        if source.state == NodeState.EVICTED or not source.alive:
            return []

        neighbors: List[Node] = []
        for node in self.nodes:
            if node.node_id == node_id:
                continue
            if node.state == NodeState.EVICTED or not node.alive:
                continue
            if source.distance_to(node) <= self.comm_radius:
                neighbors.append(node)
        return neighbors

    def maybe_capture_nodes(self) -> int:
        newly_captured = 0
        for node in self.nodes:
            if node.state == NodeState.GOOD and node.alive:
                if self.rng.random() < self.capture_probability:
                    node.state = NodeState.BAD
                    newly_captured += 1
        return newly_captured

    def consume_energy(self, node_id: int, amount: float) -> None:
        node = self.nodes[node_id]
        if not node.alive:
            return
        node.energy -= amount
        if node.energy <= 0.0:
            node.energy = 0.0
            node.alive = False
            node.state = NodeState.EVICTED

    def evict_node(self, node_id: int) -> None:
        node = self.nodes[node_id]
        node.state = NodeState.EVICTED
        node.alive = False

    def network_is_failed(self) -> bool:
        active = self.active_nodes()
        if not active:
            return True

        n_good = sum(1 for node in active if node.state == NodeState.GOOD)
        n_bad = sum(1 for node in active if node.state == NodeState.BAD)

        return n_bad >= n_good

    def snapshot_counts(self) -> dict:
        return {
            "GOOD": sum(1 for node in self.nodes if node.state == NodeState.GOOD),
            "BAD": sum(1 for node in self.nodes if node.state == NodeState.BAD),
            "EVICTED": sum(1 for node in self.nodes if node.state == NodeState.EVICTED),
            "ALIVE": sum(1 for node in self.nodes if node.alive),
        }