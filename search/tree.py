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
    value: Optional[float] = None            
    heuristic_value: Optional[float] = None  
    children: list[Node] = field(default_factory=list) 

@dataclass
class TreeParams:
    branching_factor: int
    depth: int
    chance_node_density: float
    evaluation_noise: float
    tree_balance: float

@dataclass
class Tree:
    root: Node
    params: TreeParams