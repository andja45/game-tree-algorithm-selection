from dataclasses import dataclass, field
from .tree import Node, NodeType, Tree


class _BudgetExhausted(Exception):
    pass


@dataclass
class ABResult:
    best_move: int
    move_values: list[float]
    nodes_visited: int
    pruning_rate: float
    depth_reached: int 
    instability: float # koliko puta se najbolji potez promenio / broj iteracija
    root_eval_std: float


class AlphaBetaBudget:
    #                                    ne uracunava CHANCE cvorove (u suprotnom se verzija algoritma zove Expectimax)
    def __init__(self, node_budget: int, assume_deterministic: bool = True):
        self.node_budget = node_budget
        self.assume_deterministic = assume_deterministic

    def solve(self, tree: Tree) -> ABResult:
        self._total_nodes = 0
        self._total_prunes = 0
        n_moves = len(tree.root.children)

        best_by_depth: list[int] = []
        last_values = [0.0] * n_moves
        depth_done = 0

        depth = 1
        while True:
            self._budget_left = self.node_budget - self._total_nodes
            if self._budget_left <= n_moves:
                break

            self._iter_nodes = 0
            self._iter_prunes = 0
            try:
                vals = [
                    self._search(child, depth - 1, float("-inf"), float("inf"), is_max=False)
                    for child in tree.root.children
                ]
            except _BudgetExhausted:
                # ako se istrosi budzet pre zavrsetka iteracije, ne uzimamo rezultat
                self._total_nodes += self._iter_nodes
                self._total_prunes += self._iter_prunes
                break

            self._total_nodes += self._iter_nodes
            self._total_prunes += self._iter_prunes
            last_values = vals
            depth_done = depth
            best_by_depth.append(vals.index(max(vals)))
            depth += 1
            if depth > 64:
                break

        if not best_by_depth: # nijedna iteracija nije zavrsena, AB gleda hauristike
            last_values = [c.heuristic_value for c in tree.root.children]
            best_by_depth = [last_values.index(max(last_values))]
            depth_done = 0

        best_move = best_by_depth[-1]
        changes = sum(1 for i in range(1, len(best_by_depth)) if best_by_depth[i] != best_by_depth[i - 1])
        instability = changes / max(len(best_by_depth) - 1, 1)

        mean_v = sum(last_values) / len(last_values)
        var = sum((v - mean_v) ** 2 for v in last_values) / max(len(last_values) - 1, 1)

        return ABResult(
            best_move=best_move,
            move_values=last_values,
            nodes_visited=self._total_nodes,
            pruning_rate=self._total_prunes / max(self._total_nodes, 1),
            depth_reached=depth_done,
            instability=instability,
            root_eval_std=var ** 0.5,
        )

    def _search(self, node: Node, depth: int, alpha: float, beta: float, is_max: bool) -> float:
        self._iter_nodes += 1
        if self._iter_nodes > self._budget_left:
            raise _BudgetExhausted()

        if node.node_type == NodeType.LEAF:
            return node.value
        if depth == 0 or not node.children:
            return node.heuristic_value

        if node.node_type == NodeType.CHANCE and not self.assume_deterministic:
            vals = [self._search(c, depth - 1, alpha, beta, not is_max) for c in node.children]
            return sum(vals) / len(vals)

        if is_max:
            best = float("-inf")
            for child in node.children:
                v = self._search(child, depth - 1, alpha, beta, False)
                if v > best:
                    best = v
                if best >= beta:
                    self._iter_prunes += 1
                    return best # beta odsecanje
                if best > alpha:
                    alpha = best
            return best
        else: # min
            best = float("inf")
            for child in node.children:
                v = self._search(child, depth - 1, alpha, beta, True)
                if v < best:
                    best = v
                if best <= alpha:
                    self._iter_prunes += 1
                    return best # alfa odsecanje
                if best < beta:
                    beta = best
            return best
