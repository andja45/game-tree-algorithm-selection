from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class NodeType(Enum):
    MAX = "max"
    MIN = "min"
    CHANCE = "chance"
    LEAF = "leaf"

@dataclass
class Node:
    node_type: NodeType
    value: Optional[float] = None  # only leaves have values
    children: list[Node] = field(default_factory=list)   # =[] shares one list across all instances

@dataclass
class TreeParams:
    branching_factor: int
    depth: int
    chance_node_density: float
    deceptiveness_score: float
    evaluation_noise: float
    reward_concentration: float
    tree_balance: float

@dataclass
class Tree:
    root: Node
    params: TreeParams
