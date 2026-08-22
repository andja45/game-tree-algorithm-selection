from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional
from .tree import Node, NodeType, Tree


@dataclass
class _MCTSNode:
    game_node: Node
    parent: Optional["_MCTSNode"] = None
    visits: int = 0
    total_value: float = 0.0
    children: list["_MCTSNode"] = field(default_factory=list)

    def is_fully_expanded(self) -> bool: # da li smo vec posetili svu decu ovog cvora
        return len(self.children) == len(self.game_node.children)


@dataclass
class MCTSResult:
    best_move: int
    vote_margin: float
    best_value_estimate: float
    visit_entropy: float


class MCTSBudget:
    def __init__(self, node_budget: int, seed: int, c: float = math.sqrt(2),
                 charge_rollout: bool = True):
        self.node_budget = node_budget
        self.c = c
        self.charge_rollout = charge_rollout
        self.rng = random.Random(seed)

    def solve(self, tree: Tree) -> MCTSResult:
        self._touched = 0
        root = _MCTSNode(game_node=tree.root)
        n_moves = len(tree.root.children)
        iterations = 0

        while self._touched < self.node_budget:
            node = self._select(root)
            node = self._expand(node)
            value = self._rollout(node)
            self._backprop(node, value)
            iterations += 1
            if iterations > 200_000:
                break

        visit_counts = [child.visits for child in root.children]
        visit_counts += [0] * (n_moves - len(visit_counts))

        best_move = visit_counts.index(max(visit_counts)) if any(visit_counts) else 0

        s = sorted(visit_counts, reverse=True)
        total = sum(visit_counts)
        vote_margin = (s[0] - s[1]) / total if total > 0 and len(s) > 1 else 1.0

        if root.children and best_move < len(root.children):
            bc = root.children[best_move]
            bve = bc.total_value / max(bc.visits, 1)
        else:
            bve = 0.0

        return MCTSResult(
            best_move=best_move,
            vote_margin=vote_margin,
            best_value_estimate=bve,
            visit_entropy=self._entropy(visit_counts),
        )

    def _select(self, node: _MCTSNode) -> _MCTSNode: # bira sledeci cvor preko UCB1 formule
        while node.is_fully_expanded() and node.children:
            self._touched += 1
            if node.game_node.node_type == NodeType.CHANCE:
                node = self.rng.choice(node.children) # biramo slucajan potez
            else:
                node = self._best_child(node)
        return node

    def _expand(self, node: _MCTSNode) -> _MCTSNode: # otvori novi cvor
        if node.game_node.node_type == NodeType.LEAF or not node.game_node.children:
            return node
        k = len(node.children)
        if k < len(node.game_node.children):
            self._touched += 1
            child = _MCTSNode(game_node=node.game_node.children[k], parent=node)
            node.children.append(child)
            return child
        return node

    def _rollout(self, node: _MCTSNode) -> float: # nasumicno dodji do lista (odigravanje partije)
        current = node.game_node
        while current.node_type != NodeType.LEAF and current.children:
            if self.charge_rollout:
                self._touched += 1
            current = self.rng.choice(current.children)
        return current.value if current.value is not None else 0.0

    def _backprop(self, node: Optional[_MCTSNode], value: float) -> None: # propagiranje rezultate rollouta
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    def _best_child(self, node: _MCTSNode) -> _MCTSNode: # UCB1 formula
        kids = node.children
        for k in kids:
            if k.visits == 0:         
                return k

        qs = [k.total_value / k.visits for k in kids] # prosecna vrednost dece
        if node.game_node.node_type == NodeType.MIN:
            qs = [-q for q in qs]  # min trazi najmanje Q

        log_n = math.log(max(node.visits, 2))
        best, best_score = kids[0], float("-inf")
        for k, q in zip(kids, qs):
            # exploit + explore (vece sto je dete manje poseceno)
            score = q + self.c * math.sqrt(log_n / k.visits)
            if score > best_score:
                best, best_score = k, score
        return best

    def _entropy(self, counts: list[int]) -> float: # koliko su posete rasute po deci
        total = sum(counts)
        if total == 0:
            return 0.0
        return -sum((c / total) * math.log(c / total) for c in counts if c > 0)
