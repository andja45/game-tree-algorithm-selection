import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional
from .tree import Node, NodeType, Tree

@dataclass
class _MCTSNode:
    game_node: Node
    parent: Optional[_MCTSNode] = None
    visits: int = 0
    total_value: float = 0.0
    children: list[_MCTSNode] = field(default_factory=list)

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.game_node.children)

@dataclass
class MCTSResult:
    best_move: int
    visit_counts: list[int]
    visit_entropy: float # how spread the visits are across moves
    search_time: float

class MCTS:

    def __init__(self, iterations: int, c: float = math.sqrt(2)):
        self.iterations = iterations
        self.c = c # exploration constant in UCB1, sqrt(2) is theoretically optimal

    def solve(self, tree: Tree) -> MCTSResult:
        start = time.time()
        root = _MCTSNode(game_node=tree.root)

        for _ in range(self.iterations):
            node = self._select(root)
            node = self._expand(node)
            value = self._rollout(node)
            self._backprop(node, value)

        # most visits, not highest average - visit count is more robust
        visit_counts = [child.visits for child in root.children]
        best_move = visit_counts.index(max(visit_counts))

        return MCTSResult(
            best_move=best_move,
            visit_counts=visit_counts,
            visit_entropy=self._entropy(visit_counts),
            search_time=time.time() - start,
        )

    def _select(self, node: _MCTSNode) -> _MCTSNode:
        while node.is_fully_expanded() and node.children:
            node = max(node.children, key=self._ucb1)
        return node

    def _expand(self, node: _MCTSNode) -> _MCTSNode:
        if node.game_node.node_type == NodeType.LEAF:
            return node

        expanded_count = len(node.children)
        if expanded_count < len(node.game_node.children):
            next_game_node = node.game_node.children[expanded_count]
            child = _MCTSNode(game_node=next_game_node, parent=node)
            node.children.append(child)
            return child

        return node

    def _rollout(self, node: _MCTSNode) -> float:
        current = node.game_node
        while current.node_type != NodeType.LEAF:
            if not current.children:
                break
            current = random.choice(current.children)
        return current.value if current.value is not None else 0.0

    def _backprop(self, node: _MCTSNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    def _ucb1(self, node: _MCTSNode) -> float:
        if node.visits == 0:
            return float("inf")
        parent_visits = node.parent.visits if node.parent else node.visits
        exploitation = node.total_value / node.visits
        exploration = self.c * math.sqrt(math.log(parent_visits) / node.visits)
        return exploitation + exploration

    def _entropy(self, counts: list[int]) -> float:
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts if c > 0]
        return -sum(p * math.log(p) for p in probs)
