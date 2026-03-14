from __future__ import annotations

import random
from typing import List

from game_ext.qtcid_repro.types import Node, Position2D, QTCIDConfig


def random_deploy_nodes(cfg: QTCIDConfig, rng: random.Random) -> list[Node]:
    nodes: list[Node] = []

    for node_id in range(cfg.n_nodes):
        x = rng.uniform(0.0, cfg.region_length)
        y = rng.uniform(0.0, cfg.region_width)
        nodes.append(
            Node(
                node_id=node_id,
                pos=Position2D(x=x, y=y),
                energy=cfg.ein,
            )
        )

    return nodes