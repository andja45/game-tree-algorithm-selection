from dataclasses import dataclass, field
from .tree import Node, NodeType, Tree

@dataclass
class MinimaxResult:
    best_move: int
    best_value: float
    second_best_value: float
    worst_move_value: float
    move_values: list[float]
    nodes_visited: int

class Minimax:
    def solve(self, tree: Tree) -> MinimaxResult:
        self._nodes_visited = 0
        root = tree.root

        move_values = [self._evaluate(child) for child in root.children]
        best_move = move_values.index(max(move_values))
        best_value = move_values[best_move]

        sorted_vals = sorted(move_values, reverse=True)
        second_best_value = sorted_vals[1] if len(sorted_vals) > 1 else best_value
        worst_move_value = sorted_vals[-1]

        return MinimaxResult(
            best_move=best_move,
            best_value=best_value,
            second_best_value=second_best_value,
            worst_move_value=worst_move_value,
            move_values=move_values,
            nodes_visited=self._nodes_visited,
        )

    def _evaluate(self, node: Node) -> float:
        self._nodes_visited += 1

        if node.node_type == NodeType.LEAF:
            return node.value

        child_values = [self._evaluate(child) for child in node.children]

        if node.node_type == NodeType.MAX:
            return max(child_values)
        elif node.node_type == NodeType.MIN:
            return min(child_values)
        else: # CHANCE
            return sum(child_values) / len(child_values) # pretpostavljamo da je svaki potez jednako verovatan
