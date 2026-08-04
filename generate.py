import csv
import math
import random
from pathlib import Path
from tqdm import tqdm

from search.tree import Node, NodeType, TreeParams, Tree
from search.exact_solver import Minimax
from search.alphabeta import AlphaBeta
from search.mcts import MCTS
from search.expectimax import Expectimax

BRANCHING_FACTORS = [2, 3, 4, 5, 6, 8, 12, 16]
CHANCE_DENSITIES = [round(x * 0.1, 1) for x in range(11)]
TREES_PER_COMBO = 57
FIXED_DEPTH = 6

AB_DEPTH_LIMIT = 5
MCTS_ITERATIONS = 500
MCTS_C = math.sqrt(2) # exploration constant in UCB1, sqrt(2) is theoretically optimal
EXP_DEPTH_LIMIT = 5

OUTPUT_PATH = Path("data/trees.csv")

def generate_tree(params: TreeParams, rng: random.Random) -> Tree:
    root = _build_node(params, depth=0, rng=rng)
    return Tree(root=root, params=params)


def _build_node(params: TreeParams, depth: int, rng: random.Random) -> Node:
    if depth == params.depth:
        value = rng.uniform(-1.0, 1.0)
        value += rng.gauss(0, params.evaluation_noise * 0.3)
        return Node(node_type=NodeType.LEAF, value=value)

    # tree_balance = 1.0 -> no early cuts | lower -> some branches end before max depth
    if depth > 0 and rng.random() < (1.0 - params.tree_balance) * 0.15:
        return Node(node_type=NodeType.LEAF, value=rng.uniform(-1.0, 1.0))

    node_type = _pick_node_type(params, rng)
    children = [_build_node(params, depth + 1, rng) for _ in range(params.branching_factor)]
    return Node(node_type=node_type, children=children)


def _pick_node_type(params: TreeParams, rng: random.Random) -> NodeType:
    if rng.random() < params.chance_node_density:
        return NodeType.CHANCE
    return NodeType.MAX if rng.random() < 0.5 else NodeType.MIN


def _collect_leaves(node: Node) -> list[float]:
    if node.node_type == NodeType.LEAF:
        return [node.value]
    values = []
    for child in node.children:
        values.extend(_collect_leaves(child))
    return values


def _count_nodes(node: Node) -> dict:
    counts = {"total": 0, "leaves": 0, "internal": 0, "chance": 0, "max_nodes": 0, "min_nodes": 0}

    def _walk(n):
        counts["total"] += 1
        if n.node_type == NodeType.LEAF:
            counts["leaves"] += 1
        else:
            counts["internal"] += 1
            if n.node_type == NodeType.CHANCE:
                counts["chance"] += 1
            elif n.node_type == NodeType.MAX:
                counts["max_nodes"] += 1
            elif n.node_type == NodeType.MIN:
                counts["min_nodes"] += 1
            for child in n.children:
                _walk(child)

    _walk(node)
    return counts


def _nodes_at_depth(node: Node, target: int, current: int = 0) -> list[Node]:
    if current == target:
        return [node]
    result = []
    for child in node.children:
        result.extend(_nodes_at_depth(child, target, current + 1))
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float], mean: float) -> float:
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 if values else 0.0


def extract_features(tree: Tree, exact, ab, mcts, exp) -> dict:
    row = {}

    row["branching_factor"] = tree.params.branching_factor
    row["depth"] = tree.params.depth
    row["chance_node_density"] = tree.params.chance_node_density
    row["deceptiveness_score"] = tree.params.deceptiveness_score
    row["evaluation_noise"] = tree.params.evaluation_noise
    row["reward_concentration"] = tree.params.reward_concentration
    row["tree_balance"] = tree.params.tree_balance

    counts = _count_nodes(tree.root)
    row["total_nodes"] = counts["total"]
    row["total_leaves"] = counts["leaves"]
    row["total_internal_nodes"] = counts["internal"]
    row["total_chance_nodes"] = counts["chance"]
    row["total_max_nodes"] = counts["max_nodes"]
    row["total_min_nodes"] = counts["min_nodes"]
    row["leaf_to_internal_ratio"] = counts["leaves"] / max(counts["internal"], 1)
    row["chance_node_fraction"] = counts["chance"] / max(counts["internal"], 1)  # measured vs target param

    leaf_values = _collect_leaves(tree.root)
    mean = _mean(leaf_values)
    row["leaf_value_mean"] = mean
    row["leaf_value_std"] = _stddev(leaf_values, mean)
    row["leaf_value_min"] = min(leaf_values)
    row["leaf_value_max"] = max(leaf_values)
    row["leaf_fraction_positive"] = sum(1 for v in leaf_values if v > 0) / len(leaf_values)
    row["leaf_fraction_negative"] = sum(1 for v in leaf_values if v < 0) / len(leaf_values)

    row["exact_best_value"] = exact.best_value
    row["exact_second_best_value"] = exact.second_best_value
    row["best_move_margin"] = exact.best_value - exact.second_best_value
    row["exact_nodes_visited"] = exact.nodes_visited

    for d in range(1, 11):
        nodes_d = _nodes_at_depth(tree.root, d)
        leaves_d = [n for n in nodes_d if n.node_type == NodeType.LEAF]
        chance_d = [n for n in nodes_d if n.node_type == NodeType.CHANCE]
        vals_d = [n.value for n in leaves_d]
        mean_d = _mean(vals_d)

        row[f"node_count_d{d}"] = len(nodes_d)
        row[f"leaf_count_d{d}"] = len(leaves_d)
        row[f"chance_node_count_d{d}"] = len(chance_d)
        row[f"leaf_value_mean_d{d}"] = mean_d
        row[f"leaf_value_std_d{d}"] = _stddev(vals_d, mean_d)
        row[f"leaf_fraction_d{d}"] = len(leaves_d) / max(len(nodes_d), 1)

    row["ab_depth_limit"] = AB_DEPTH_LIMIT
    row["ab_nodes_visited"] = ab.nodes_visited
    row["ab_pruning_count"] = ab.pruning_count
    row["ab_pruning_rate"] = ab.pruning_rate
    row["ab_depth_reached"] = ab.depth_reached
    row["ab_search_time"] = ab.search_time

    row["mcts_iterations"] = MCTS_ITERATIONS
    row["mcts_exploration_constant"] = MCTS_C
    row["mcts_visit_entropy"] = mcts.visit_entropy
    row["mcts_search_time"] = mcts.search_time

    row["exp_depth_limit"] = EXP_DEPTH_LIMIT
    row["exp_nodes_visited"] = exp.nodes_visited
    row["exp_chance_nodes_visited"] = exp.chance_nodes_visited
    row["exp_depth_reached"] = exp.depth_reached
    row["exp_search_time"] = exp.search_time

    return row


def compute_targets(exact, ab, mcts, exp) -> dict:
    ab_correct = int(ab.best_move == exact.best_move)
    mcts_correct = int(mcts.best_move == exact.best_move)
    exp_correct = int(exp.best_move == exact.best_move)

    winners = []
    if ab_correct: winners.append("AB")
    if mcts_correct: winners.append("MCTS")
    if exp_correct: winners.append("Expectimax")
    if len(winners) == 0:
        which_algo = "None"
    elif len(winners) == 1:
        which_algo = winners[0]
    else:
        which_algo = "Tie"

    margin = exact.best_value - exact.second_best_value
    if margin < 0.1:
        margin_category = "low"
    elif margin < 0.4:
        margin_category = "medium"
    else:
        margin_category = "high"

    return {
        "which_algo": which_algo,
        "ab_correct": ab_correct,
        "mcts_correct": mcts_correct,
        "exp_correct": exp_correct,
        "is_contested": int(margin < 0.1),
        "margin_category": margin_category,
    }


def main():
    exact_solver = Minimax()
    ab_solver = AlphaBeta(depth_limit=AB_DEPTH_LIMIT)
    mcts_solver = MCTS(iterations=MCTS_ITERATIONS, c=MCTS_C)
    exp_solver = Expectimax(depth_limit=EXP_DEPTH_LIMIT)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    combos = [(bf, cd) for bf in BRANCHING_FACTORS for cd in CHANCE_DENSITIES]
    rows = []

    with tqdm(total=len(combos) * TREES_PER_COMBO, desc="generating trees") as pbar:
        for bf, cd in combos:
            for seed in range(TREES_PER_COMBO):
                rng = random.Random(seed)  # fixed seed per (bf, cd, seed) → reproducible dataset

                params = TreeParams(
                    branching_factor=bf,
                    depth=FIXED_DEPTH,
                    chance_node_density=cd,
                    deceptiveness_score=rng.uniform(0, 1),
                    evaluation_noise=rng.uniform(0, 0.5),
                    reward_concentration=rng.uniform(0, 1),
                    tree_balance=rng.uniform(0.5, 1.0),
                )

                tree = generate_tree(params, rng)
                exact = exact_solver.solve(tree)
                ab = ab_solver.solve(tree)
                mcts = mcts_solver.solve(tree)
                exp = exp_solver.solve(tree)

                row = extract_features(tree, exact, ab, mcts, exp)
                row.update(compute_targets(exact, ab, mcts, exp))
                rows.append(row)
                pbar.update(1)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {len(rows)} rows → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()