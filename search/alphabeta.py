import time
from dataclasses import dataclass
from .tree import Node, NodeType, Tree

@dataclass
class AlphaBetaResult:
    best_move: int
    best_value: float
    nodes_visited: int
    pruning_count: int
    pruning_rate: float # pruning_count / nodes_visited 
    depth_reached: int
    search_time: float

class AlphaBeta:
    def __init__(self, depth_limit: int):
        self.depth_limit = depth_limit

    def solve(self, tree: Tree) -> AlphaBetaResult:
        self._nodes_visited = 0
        self._pruning_count = 0
        self._max_depth_reached = 0
        start = time.time()

        move_values = [
            self._max(child, self.depth_limit - 1, float("-inf"), float("inf"))
            for child in tree.root.children
        ]

        best_move = move_values.index(max(move_values))

        return AlphaBetaResult(
            best_move=best_move,
            best_value=move_values[best_move],
            nodes_visited=self._nodes_visited,
            pruning_count=self._pruning_count,
            pruning_rate=self._pruning_count / max(self._nodes_visited, 1),
            depth_reached=self._max_depth_reached,
            search_time=time.time() - start,
        )

    def _max(self, node: Node, depth: int, alpha: float, beta: float) -> float:
        self._nodes_visited += 1
        self._max_depth_reached = max(self._max_depth_reached, self.depth_limit - depth)

        if node.node_type == NodeType.LEAF or depth == 0:
            return node.value if node.value is not None else 0.0

        if node.node_type == NodeType.CHANCE:
            # we must visit all children to compute mean - pruning isnt possible
            child_values = [self._max(child, depth - 1, alpha, beta) for child in node.children]
            return sum(child_values) / len(child_values)

        a = float("-inf")
        for child in node.children:
            b = self._min(child, depth - 1, alpha, beta)
            if b > a:
                a = b
            if a >= beta:
                self._pruning_count += 1
                return a
            if a > alpha:
                alpha = a
        return a

    def _min(self, node: Node, depth: int, alpha: float, beta: float) -> float:
        self._nodes_visited += 1
        self._max_depth_reached = max(self._max_depth_reached, self.depth_limit - depth)

        if node.node_type == NodeType.LEAF or depth == 0:
            return node.value if node.value is not None else 0.0

        if node.node_type == NodeType.CHANCE:
            child_values = [self._min(child, depth - 1, alpha, beta) for child in node.children]
            return sum(child_values) / len(child_values)

        b = float("inf")
        for child in node.children:
            a = self._max(child, depth - 1, alpha, beta)
            if a < b:
                b = a
            if b <= alpha:
                self._pruning_count += 1
                return b
            if b < beta:
                beta = b
        return b
