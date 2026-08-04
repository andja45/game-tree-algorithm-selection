import time
from dataclasses import dataclass
from .tree import Node, NodeType, Tree

@dataclass
class ExpectimaxResult:
    best_move: int
    best_value: float
    move_values: list[float]
    nodes_visited: int
    chance_nodes_visited: int
    depth_reached: int
    search_time: float

class Expectimax:
    def __init__(self, depth_limit: int):
        self.depth_limit = depth_limit

    def solve(self, tree: Tree) -> ExpectimaxResult:
        self._nodes_visited = 0
        self._chance_nodes_visited = 0
        self._max_depth_reached = 0
        start = time.time()

        move_values = [
            self._evaluate(child, self.depth_limit - 1)
            for child in tree.root.children
        ]

        best_move = move_values.index(max(move_values))

        return ExpectimaxResult(
            best_move=best_move,
            best_value=move_values[best_move],
            move_values=move_values,
            nodes_visited=self._nodes_visited,
            chance_nodes_visited=self._chance_nodes_visited,
            depth_reached=self._max_depth_reached,
            search_time=time.time() - start,
        )

    def _evaluate(self, node: Node, depth: int) -> float:
        self._nodes_visited += 1
        self._max_depth_reached = max(self._max_depth_reached, self.depth_limit - depth)

        if node.node_type == NodeType.LEAF or depth == 0:
            return node.value if node.value is not None else 0.0

        child_values = [self._evaluate(child, depth - 1) for child in node.children]

        if node.node_type == NodeType.MAX:
            return max(child_values)
        if node.node_type == NodeType.MIN:
            return min(child_values)
        if node.node_type == NodeType.CHANCE:
            self._chance_nodes_visited += 1
            return sum(child_values) / len(child_values)  # assuming uniform probability