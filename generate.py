import csv
import math
import random
import statistics
from pathlib import Path
from tqdm import tqdm

from search.tree import Node, NodeType, TreeParams, Tree
from search.exact_solver import Minimax
from search.alphabeta import AlphaBeta
from search.mcts import MCTS

BRANCHING_FACTORS = [2, 3, 4, 5, 6, 8, 12, 16]
CHANCE_DENSITIES = [round(x * 0.1, 1) for x in range(11)]
TREES_PER_COMB = 57

TARGET_NODES = 50_000

def adaptive_depth(branching_factor: int) -> int:
    return max(2, int(math.log(TARGET_NODES, branching_factor)))

MAX_DEPTH = max(adaptive_depth(bf) for bf in BRANCHING_FACTORS)

MCTS_ITERATIONS = 500
MCTS_C = math.sqrt(2)

OUTPUT_PATH = Path("data/trees.csv")


def generate_tree(params: TreeParams, rng: random.Random) -> Tree:
    root = _build_node(params, depth=0, rng=rng)
    return Tree(root=root, params=params)


def _build_node(params: TreeParams, depth: int, rng: random.Random) -> Node:
    if depth == params.depth:
        value = rng.uniform(-1.0, 1.0)
        value += rng.gauss(0, params.evaluation_noise * 0.3)
        return Node(node_type=NodeType.LEAF, value=value)

    # tree_balance < 1.0 causes some branches to end early - creates irregular trees
    early_termination_prob = (1.0 - params.tree_balance) * 0.15
    if depth > 0 and rng.random() < early_termination_prob:
        return Node(node_type=NodeType.LEAF, value=rng.uniform(-1.0, 1.0))

    node_type = _pick_node_type(params, rng, depth)
    children = [_build_node(params, depth + 1, rng) for _ in range(params.branching_factor)]
    return Node(node_type=node_type, children=children)


def _pick_node_type(params: TreeParams, rng: random.Random, depth: int) -> NodeType:
    # game trees alternate MAX/MIN by depth - CHANCE can substitute at any non-root level
    if depth > 0 and rng.random() < params.chance_node_density:
        return NodeType.CHANCE
    return NodeType.MAX if depth % 2 == 0 else NodeType.MIN


def _scan_tree(root: Node, max_depth: int) -> dict:
    counts = {"total": 0, "leaves": 0, "internal": 0, "chance": 0, "max_nodes": 0, "min_nodes": 0}
    leaf_values = []
    nodes_by_depth = {d: [] for d in range(1, max_depth + 1)}

    def _walk(node, depth):
        counts["total"] += 1
        if 1 <= depth <= max_depth:
            nodes_by_depth[depth].append(node)

        if node.node_type == NodeType.LEAF:
            counts["leaves"] += 1
            leaf_values.append(node.value)
        else:
            counts["internal"] += 1
            if node.node_type == NodeType.CHANCE:
                counts["chance"] += 1
            elif node.node_type == NodeType.MAX:
                counts["max_nodes"] += 1
            elif node.node_type == NodeType.MIN:
                counts["min_nodes"] += 1
            for child in node.children:
                _walk(child, depth + 1)

    _walk(root, 0)
    return {"counts": counts, "leaf_values": leaf_values, "nodes_by_depth": nodes_by_depth}


def _scan_subtree(node: Node) -> tuple[float, float]:
    total_internal = 0
    chance_count = 0
    min_count = 0

    def _walk(n: Node):
        nonlocal total_internal, chance_count, min_count
        if n.node_type == NodeType.LEAF:
            return
        total_internal += 1
        if n.node_type == NodeType.CHANCE:
            chance_count += 1
        elif n.node_type == NodeType.MIN:
            min_count += 1
        for child in n.children:
            _walk(child)

    _walk(node)
    if total_internal == 0:
        return 0.0, 0.0

    # returns (chance_fraction, min_fraction) of internal nodes in this subtree
    return chance_count / total_internal, min_count / total_internal


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float], mean: float) -> float:
    if not values:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def extract_features(tree: Tree, exact, ab, mcts, exact_solver: Minimax) -> dict:
    row = {}

    row["branching_factor"] = tree.params.branching_factor
    row["depth"] = tree.params.depth
    row["chance_node_density"] = tree.params.chance_node_density
    row["evaluation_noise"] = tree.params.evaluation_noise
    row["tree_balance"] = tree.params.tree_balance

    scan = _scan_tree(tree.root, MAX_DEPTH)
    counts = scan["counts"]
    leaf_values = scan["leaf_values"]
    nodes_by_depth = scan["nodes_by_depth"]

    row["total_nodes"] = counts["total"]
    row["total_leaves"] = counts["leaves"]
    row["total_internal_nodes"] = counts["internal"]
    row["total_chance_nodes"] = counts["chance"]
    row["total_max_nodes"] = counts["max_nodes"]
    row["total_min_nodes"] = counts["min_nodes"]
    row["leaf_to_internal_ratio"] = counts["leaves"] / max(counts["internal"], 1)
    row["chance_node_fraction"] = counts["chance"] / max(counts["internal"], 1)

    mean = _mean(leaf_values)
    row["leaf_value_mean"] = mean
    row["leaf_value_std"] = _stddev(leaf_values, mean)
    row["leaf_value_min"] = min(leaf_values)
    row["leaf_value_max"] = max(leaf_values)
    row["leaf_value_range"] = max(leaf_values) - min(leaf_values)
    row["leaf_fraction_positive"] = sum(1 for v in leaf_values if v > 0) / len(leaf_values)
    row["leaf_fraction_negative"] = sum(1 for v in leaf_values if v < 0) / len(leaf_values)

    row["exact_best_value"] = exact.best_value
    row["exact_second_best_value"] = exact.second_best_value
    margin = exact.best_value - exact.second_best_value
    row["best_move_margin"] = margin
    # normalized margin introduces one scale independent from concrete values
    row["best_move_margin_normalized"] = margin / max(row["leaf_value_range"], 1e-9)
    row["exact_nodes_visited"] = exact.nodes_visited

    best_child = tree.root.children[exact.best_move]
    path_length, path_chance, path_min = exact_solver.optimal_path_stats(best_child)
    row["optimal_path_length"] = path_length
    row["optimal_path_chance_count"] = path_chance
    row["optimal_path_min_count"] = path_min

    best_subtree_chance_frac, best_subtree_min_frac = _scan_subtree(best_child)
    row["best_subtree_chance_fraction"] = best_subtree_chance_frac
    row["best_subtree_min_fraction"] = best_subtree_min_frac

    # per-depth stats - shallower trees get 0s for depths they don't reach
    for d in range(1, MAX_DEPTH + 1):
        nodes_d = nodes_by_depth[d]
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

    row["ab_nodes_visited"] = ab.nodes_visited
    row["ab_pruning_count"] = ab.pruning_count
    row["ab_pruning_rate"] = ab.pruning_rate
    # how spread AB's evaluations are across root moves - low means AB sees all moves as similar
    row["ab_root_eval_std"] = statistics.stdev(ab.move_values) if len(ab.move_values) > 1 else 0.0
    row["ab_search_time"] = ab.search_time
    row["exact_root_eval_std"] = statistics.stdev(exact.move_values) if len(exact.move_values) > 1 else 0.0

    visits = sorted(mcts.visit_counts, reverse=True)
    total_visits = sum(visits)
    # how much more did we visit first place move than the second best
    mcts_vote_margin = (visits[0] - visits[1]) / total_visits if total_visits > 0 and len(visits) > 1 else 1.0

    row["mcts_iterations"] = MCTS_ITERATIONS
    row["mcts_exploration_constant"] = MCTS_C
    row["mcts_visit_entropy"] = mcts.visit_entropy
    row["mcts_vote_margin"] = mcts_vote_margin
    row["mcts_search_time"] = mcts.search_time

    return row


def compute_targets(exact, ab, mcts) -> dict:
    ab_correct = int(ab.best_move == exact.best_move)
    mcts_correct = int(mcts.best_move == exact.best_move)

    if ab_correct and mcts_correct:
        which_algo = "Both"    
    elif ab_correct:
        which_algo = "AB"      
    elif mcts_correct:
        which_algo = "MCTS" 
    else:
        which_algo = "Neither" 

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
        "is_contested": int(margin < 0.1),
        "margin_category": margin_category,
    }


def main():
    exact_solver = Minimax()
    mcts_solver = MCTS(iterations=MCTS_ITERATIONS, c=MCTS_C)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    combos = [(bf, cd) for bf in BRANCHING_FACTORS for cd in CHANCE_DENSITIES]
    rows = []

    with tqdm(total=len(combos) * TREES_PER_COMB, desc="generating trees") as pbar:
        for bf, cd in combos:
            for seed in range(TREES_PER_COMB):
                tree_seed = hash((bf, cd, seed)) & 0xFFFFFFFF
                rng = random.Random(tree_seed)

                params = TreeParams(
                    branching_factor=bf,
                    depth=adaptive_depth(bf),
                    chance_node_density=cd,
                    evaluation_noise=rng.uniform(0, 0.5),
                    tree_balance=rng.uniform(0.5, 1.0),
                )

                ab_solver = AlphaBeta(depth_limit=params.depth)

                tree = generate_tree(params, rng)
                exact = exact_solver.solve(tree)
                ab = ab_solver.solve(tree)
                mcts = mcts_solver.solve(tree)

                row = extract_features(tree, exact, ab, mcts, exact_solver)
                row.update(compute_targets(exact, ab, mcts))
                rows.append(row)
                pbar.update(1)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {len(rows)} rows → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()