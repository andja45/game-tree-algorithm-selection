from dataclasses import dataclass, field
from .tree import Node, NodeType, Tree

@dataclass
class MinimaxResult:
    best_move: int
    best_value: float
    second_best_value: float
    move_values: list[float]
    nodes_visited: int

class Minimax:
    def solve(self, tree: Tree) -> MinimaxResult:
        self._nodes_visited = 0
        self._cache = {} # cashed values used by optimal_path_stats()
        root = tree.root

        move_values = [self._evaluate(child) for child in root.children]
        best_move = move_values.index(max(move_values))
        best_value = move_values[best_move]

        sorted_vals = sorted(move_values, reverse=True)
        second_best_value = sorted_vals[1] if len(sorted_vals) > 1 else best_value

        return MinimaxResult(
            best_move=best_move,
            best_value=best_value,
            second_best_value=second_best_value,
            move_values=move_values,
            nodes_visited=self._nodes_visited,
        )

    def _evaluate(self, node: Node) -> float:
        self._nodes_visited += 1

        if node.node_type == NodeType.LEAF:
            self._cache[id(node)] = node.value
            return node.value

        child_values = [self._evaluate(child) for child in node.children]

        if node.node_type == NodeType.MAX:
            val = max(child_values)
        elif node.node_type == NodeType.MIN:
            val = min(child_values)
        else: # CHANCE
            val = sum(child_values) / len(child_values)

        self._cache[id(node)] = val
        return val

    def optimal_path_stats(self, node: Node) -> tuple[int, int, int]:
        path_length = 0
        chance_count = 0
        min_count = 0

        current = node
        while current.node_type != NodeType.LEAF and current.children:
            child_vals = [self._cache[id(c)] for c in current.children]

            if current.node_type == NodeType.MAX:
                current = current.children[child_vals.index(max(child_vals))]
            elif current.node_type == NodeType.MIN:
                min_count += 1
                current = current.children[child_vals.index(min(child_vals))]
            else:  # CHANCE — trace toward child closest to expected value
                chance_count += 1
                expected = sum(child_vals) / len(child_vals)
                distances = [abs(v - expected) for v in child_vals]
                best_idx = distances.index(min(distances))
                current = current.children[best_idx]

            path_length += 1

        return path_length, chance_count, min_count
