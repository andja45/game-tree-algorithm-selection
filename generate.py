import csv
import math
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from search.tree import Node, NodeType, TreeParams, Tree
from search.exact_solver import Minimax
from search.alphabeta import AlphaBetaBudget
from search.mcts import MCTSBudget

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "trees10.csv"
NUMBER_OF_TREES = 20_000
WORKERS = None  

DEPTH_FEATS = 5    # broj nivoa za per-depth atribute
MIN_NODES = 80     # odbacuje degenerisana stabla
EARLY_LEAF = 0.08  # verovatnoca da cvor postane list pre max dubine
STEP_BASE = 1.0    # bazna skala koraka slucajnog hoda 
MCTS_REPEATS = 5   # mcts je nasumican, uzimamo majority vote
MCTS_C = math.sqrt(2) 
CHARGE_ROLLOUT = True # da li je rollout uracunat u budzet

CFG = dict(
    bf_range=(2.0, 14.0),
    depth_cap_nodes=30_000,        # gornja granica broja cvorova
    depth_dmax=10,
    spread_range=(0.6, 2.2),       # korak slucajnog hoda, kontrolise marginu
    heur_noise_range=(0.0, 1.5),   # koliko je AB-ova heuristika losa
    chance_zero_prob=0.15,         # udeo stabala sa cd=0.0 
    chance_range=(0.0, 0.9),       # kad cd nije tacno 0
    budget_cov_range=(0.02, 0.7),  # budzet kao udeo total_nodes
)

# generisanje strukture stabla

def _max_depth_for(b_mean, cap, dmax): # dubina odredjena po bf tako da se ne predje cap
    eb = b_mean * (1.0 - EARLY_LEAF) # efektivno grananje
    if eb <= 1.0:
        return dmax
    level = total = 1.0    
    depth = 0
    while depth < dmax:
        level *= eb
        total += level
        if total > cap:
            break
        depth += 1
    return max(2, depth)


def sample_params(rng, cfg=CFG):
    bf = rng.uniform(*cfg["bf_range"])
    hi = _max_depth_for(bf, cfg["depth_cap_nodes"], cfg["depth_dmax"])
    lo = min(2, hi)
    depth = rng.randint(lo, hi)

    if rng.random() < cfg["chance_zero_prob"]:
        cd = 0.0
    else:
        cd = rng.uniform(*cfg["chance_range"])

    spread = math.exp(rng.uniform(*(math.log(x) for x in cfg["spread_range"])))
    heur_noise = rng.uniform(*cfg["heur_noise_range"])
    budget_cov = math.exp(rng.uniform(*(math.log(x) for x in cfg["budget_cov_range"])))

    return dict(bf=bf, max_depth=depth, chance_density=cd, spread=spread,
                heur_noise=heur_noise, budget_cov=budget_cov)


def _build(p, depth, parent_latent, rng, state):
    state["n"] += 1

    step_std = p["spread"] * STEP_BASE / math.sqrt(max(p["max_depth"], 1))
    latent = parent_latent + rng.gauss(0, step_std)

    is_leaf = depth >= p["max_depth"] or (depth > 0 and rng.random() < EARLY_LEAF)
    if is_leaf:
        v = math.tanh(latent)
        return Node(node_type=NodeType.LEAF, value=v, heuristic_value=v)

    remaining_frac = (p["max_depth"] - depth) / max(p["max_depth"], 1)
    heur_latent = latent + rng.gauss(0, p["heur_noise"] * remaining_frac) if p["heur_noise"] > 0 else latent
    heuristic = math.tanh(heur_latent)

    if depth > 0 and rng.random() < p["chance_density"]:
        ntype = NodeType.CHANCE
    else:
        ntype = NodeType.MAX if depth % 2 == 0 else NodeType.MIN

    n_children = max(1, round(rng.gauss(p["bf"], p["bf"] * 0.15)))
    if depth == 0:
        n_children = max(n_children, 2)

    node = Node(node_type=ntype, heuristic_value=heuristic, children=[])
    for _ in range(n_children):
        node.children.append(_build(p, depth + 1, latent, rng, state))
    return node


def build_tree(p, rng):
    state = {"n": 0}
    root = _build(p, 0, 0.0, rng, state)
    return Tree(root=root, params=TreeParams(
        branching_factor=round(p["bf"]),
        depth=p["max_depth"],
        chance_node_density=p["chance_density"],
        evaluation_noise=p["heur_noise"],
        tree_balance=1.0 - EARLY_LEAF,
    )), state["n"]


#  pomocne statisticke funkcije

def _mean(xs):
    return statistics.mean(xs) if xs else 0.0


def _std(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def _skew(xs): # racuna asimetriju raspodele
    n = len(xs)
    if n < 3:
        return 0.0
    m = _mean(xs)
    s = _std(xs)
    if s == 0:
        return 0.0
    return sum((x - m) ** 3 for x in xs) / n / (s ** 3)


def _scan(root, depth_feats):
    counts = dict(n=0, leaves=0, internal=0, chance=0, mx=0, mn=0)
    leaf_vals = []
    leaf_depths = []
    branch_counts = []
    by_depth = {d: dict(n=0, chance=0, leaf_vals=[], branch_counts=[]) for d in range(1, depth_feats + 1)}

    stack = [(root, 0)] # (cvor, dubina)
    while stack:
        node, d = stack.pop()
        counts["n"] += 1

        if 1 <= d <= depth_feats:
            by_depth[d]["n"] += 1
            if node.node_type == NodeType.CHANCE:
                by_depth[d]["chance"] += 1

        if node.node_type == NodeType.LEAF:
            counts["leaves"] += 1
            leaf_vals.append(node.value)
            leaf_depths.append(d)
            if 1 <= d <= depth_feats:
                by_depth[d]["leaf_vals"].append(node.value)
        else:
            counts["internal"] += 1
            branch_counts.append(len(node.children))
            if 1 <= d <= depth_feats:
                by_depth[d]["branch_counts"].append(len(node.children))
            if node.node_type == NodeType.CHANCE:
                counts["chance"] += 1
            elif node.node_type == NodeType.MAX:
                counts["mx"] += 1
            elif node.node_type == NodeType.MIN:
                counts["mn"] += 1
            for c in node.children:
                stack.append((c, d + 1))

    return counts, leaf_vals, by_depth, leaf_depths, branch_counts


def _branch_top2(root):
    # strukturni atribut za marginu: razlika prosecne vrednosti listova izmedju
    # dve najbolje grane
    means, sizes, maxdepths, chance_fracs = [], [], [], []
    for c in root.children:
        cnts, lv, _, ldepths, _ = _scan(c, 0)
        means.append(_mean(lv))
        sizes.append(cnts["n"])
        maxdepths.append(max(ldepths) if ldepths else 0)
        chance_fracs.append(cnts["chance"] / max(cnts["internal"], 1))

    order_idx = sorted(range(len(means)), key=lambda i: -means[i])
    gap = means[order_idx[0]] - means[order_idx[1]] if len(order_idx) > 1 else 0.0
    spread_ = _std(means) 
    skew_ = _skew(means)

    if len(order_idx) > 1:
        i1, i2 = order_idx[0], order_idx[1]
        size_ratio = sizes[i1] / max(sizes[i2], 1)
        depth_diff = maxdepths[i1] - maxdepths[i2]
        cf1, cf2 = chance_fracs[i1], chance_fracs[i2]
    else:
        size_ratio, depth_diff, cf1, cf2 = 1.0, 0, 0.0, 0.0

    return gap, spread_, skew_, size_ratio, depth_diff, cf1, cf2


def extract_features(tree, budget, total_nodes):
    root = tree.root
    counts, leaf_vals, by_depth, leaf_depths, branch_counts = _scan(root, DEPTH_FEATS)
    r = {}

    # globalni atributi generatora
    r["branching_factor"] = tree.params.branching_factor
    r["depth"] = tree.params.depth
    r["chance_node_density"] = tree.params.chance_node_density
    r["heur_noise_param"] = tree.params.evaluation_noise

    # globalni atributi stabla
    r["total_nodes"] = counts["n"]
    r["total_leaves"] = counts["leaves"]
    r["total_internal_nodes"] = counts["internal"]
    r["total_chance_nodes"] = counts["chance"]
    r["total_max_nodes"] = counts["mx"]
    r["total_min_nodes"] = counts["mn"]
    r["leaf_to_internal_ratio"] = counts["leaves"] / max(counts["internal"], 1)
    r["chance_node_fraction"] = counts["chance"] / max(counts["internal"], 1)
    r["leaf_density"] = counts["leaves"] / max(counts["n"], 1)
    
    r["branching_mean"] = _mean(branch_counts) # prosecan broj dece po cvoru
    r["branching_std"] = _std(branch_counts)
    
    r["realized_max_depth"] = max(leaf_depths) if leaf_depths else 0
    r["leaf_depth_mean"] = _mean(leaf_depths)
    r["leaf_depth_std"] = _std(leaf_depths)

    # globalni atributi listova
    m = _mean(leaf_vals)
    r["leaf_value_mean"] = m
    r["leaf_value_std"] = _std(leaf_vals)
    r["leaf_value_min"] = min(leaf_vals) if leaf_vals else 0.0
    r["leaf_value_max"] = max(leaf_vals) if leaf_vals else 0.0
    r["leaf_value_range"] = r["leaf_value_max"] - r["leaf_value_min"]
    r["leaf_fraction_positive"] = sum(1 for v in leaf_vals if v > 0) / max(len(leaf_vals), 1)
    r["leaf_skew"] = _skew(leaf_vals)

    sorted_lv = sorted(leaf_vals)
    n_lv = len(sorted_lv)
    if n_lv > 0:
        r["leaf_value_median"] = sorted_lv[n_lv // 2]
        r["leaf_iqr"] = sorted_lv[int(n_lv * 0.75)] - sorted_lv[int(n_lv * 0.25)]
    else:
        r["leaf_value_median"] = 0.0
        r["leaf_iqr"] = 0.0
    # udeo listova cija je vrednost blizu maksimalne vrednosti lista u stablu
    r["leaf_frac_near_max"] = sum(1 for v in leaf_vals if v >= r["leaf_value_max"] - 0.05) / max(len(leaf_vals), 1)

    # atributi po dubinama
    for d in range(1, DEPTH_FEATS + 1):
        bd = by_depth[d]
        lvd = bd["leaf_vals"]
        r[f"node_count_d{d}"] = bd["n"]
        r[f"chance_node_count_d{d}"] = bd["chance"]
        r[f"leaf_value_mean_d{d}"] = _mean(lvd)
        r[f"leaf_value_std_d{d}"] = _std(lvd)
        r[f"branching_mean_d{d}"] = _mean(bd["branch_counts"])

    # atributi budzeta algoritama
    r["node_budget"] = budget
    r["budget_coverage"] = budget / max(total_nodes, 1) # koliko % stabla pokriva budzet
    r["ab_reach_depth"] = math.log(max(budget, 2)) / math.log(max(tree.params.branching_factor, 1.05))
    r["depth_deficit"] = max(tree.params.depth - r["ab_reach_depth"], 0.0) # koliko dubine AB ne pretrazi
    r["depth_deficit_rel"] = r["depth_deficit"] / max(tree.params.depth, 1)

    # atributi vezani za grane iz korena
    gap, bspread, bskew, top2_size_ratio, top2_depth_diff, chance_frac_top1, chance_frac_top2 = _branch_top2(root)
    r["branch_top2_gap"] = gap 
    r["branch_mean_spread"] = bspread
    r["branch_value_skew"] = bskew
    r["top2_size_ratio"] = top2_size_ratio
    r["top2_depth_diff"] = top2_depth_diff
    r["chance_frac_top1"] = chance_frac_top1
    r["chance_frac_top2"] = chance_frac_top2

    # ocena heuristike, da li heuristika dobro predvidja MIN, MAX, CHANCE
    res = []
    stack = [root]
    while stack:
        nd = stack.pop()
        if nd.node_type == NodeType.LEAF or not nd.children:
            continue
        stack.extend(nd.children)
        ch = [c.heuristic_value for c in nd.children if c.heuristic_value is not None]
        if not ch or nd.heuristic_value is None:
            continue
        back = (sum(ch) / len(ch) if nd.node_type == NodeType.CHANCE
                else (max(ch) if nd.node_type == NodeType.MAX else min(ch)))
        res.append(abs(nd.heuristic_value - back))
    r["heur_incoherence"] = _mean(res) # koliko se razlikuje heuristika roditelja i dece, koliko je kozistentna sama sa sobom
    r["heur_incoherence_std"] = _std(res)
    r["heur_noise_to_signal"] = tree.params.evaluation_noise / max(r["leaf_value_std"], 1e-9) # odnos suma heuristike i vrednosti listova
    
    return r


# spirmanov koeficijent korelacije - koliko je rangiranje poteza tacno
def _spearman(a, b):
    n = len(a)
    if n <= 1:
        return 1.0
    def ranks(v):
        order = sorted(range(n), key=lambda i: -v[i])
        rr = [0] * n
        for rank, idx in enumerate(order):
            rr[idx] = rank
        return rr
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    sa = math.sqrt(sum((x - ma) ** 2 for x in ra))
    sb = math.sqrt(sum((x - mb) ** 2 for x in rb))
    return cov / (sa * sb) if sa > 0 and sb > 0 else 1.0


#  generisanje jednog stabla

def generate_row(seed, cfg=CFG):
    rng = random.Random(seed)
    p = sample_params(rng, cfg)
    tree, total_nodes = build_tree(p, rng)
    if len(tree.root.children) < 2 or total_nodes < MIN_NODES:
        return None

    # odbacujemo degenerisana stabla
    budget = min(total_nodes, max(20, int(total_nodes * p["budget_cov"])))

    exact = Minimax().solve(tree)
    ab = AlphaBetaBudget(node_budget=budget).solve(tree)
    mcts_runs = [MCTSBudget(node_budget=budget, seed=seed * 31 + i, c=MCTS_C,
                            charge_rollout=CHARGE_ROLLOUT).solve(tree)
                for i in range(MCTS_REPEATS)]

    row = extract_features(tree, budget, total_nodes)

    row["ab_nodes_visited"] = ab.nodes_visited
    row["ab_pruning_rate"] = ab.pruning_rate
    row["ab_root_eval_std"] = ab.root_eval_std
    row["ab_instability"] = ab.instability # koliko puta se predomisljao
    row["ab_depth_reached"] = ab.depth_reached
    row["ab_ranking_correlation"] = _spearman(ab.move_values, exact.move_values)

    row["mcts_vote_margin"] = _mean([m.vote_margin for m in mcts_runs])
    row["mcts_visit_entropy"] = _mean([m.visit_entropy for m in mcts_runs]) # koliko siroko mcts posecuje cvorove
    row["mcts_best_value_estimate"] = _mean([m.best_value_estimate for m in mcts_runs])
    mcts_moves = [m.best_move for m in mcts_runs]

    row["exact_best_value"] = exact.best_value
    row["exact_second_best_value"] = exact.second_best_value
    margin = exact.best_value - exact.second_best_value
    row["best_move_margin"] = margin
    row["best_move_margin_normalized"] = margin / max(row["leaf_value_range"], 1e-9)

    ab_correct_each = int(ab.best_move == exact.best_move)
    mcts_correct_each = [int(mv == exact.best_move) for mv in mcts_moves] # tacnost svakog ponavljanja
    mcts_correct_rate = _mean(mcts_correct_each) # udeo ponavljanja koja su tacno pogodila
    mcts_correct = int(mcts_correct_rate > 0.5) # tacan ako je vecina pogodila

    row["ab_correct"] = ab_correct_each
    row["mcts_correct"] = mcts_correct # tacan ako je vecina nasla
    row["mcts_correct_rate"] = mcts_correct_rate # u koliko simulacija je nadjen

    if ab_correct_each and mcts_correct:
        which_algo = "Both"
    elif ab_correct_each:
        which_algo = "AB"
    elif mcts_correct:
        which_algo = "MCTS"
    else:
        which_algo = "Neither"
    row["which_algo"] = which_algo

    if margin < 0.1:
        margin_category = "low"
    elif margin < 0.4:
        margin_category = "medium"
    else:
        margin_category = "high"
    row["margin_category"] = margin_category
    row["is_contested"] = int(margin < 0.1)

    ab_value = exact.move_values[ab.best_move] # ab best move values
    mcts_value = _mean([exact.move_values[mv] for mv in mcts_moves])
    delta = ab_value - mcts_value
    row["which_better"] = "Tie" if abs(delta) < 1e-9 else ("AB" if delta > 0 else "MCTS")
    row["delta"] = delta # razlika nadjenih poteza ab i mcts
    # koliko su gori u odnosu na pravi optimum
    row["ab_regret"] = exact.best_value - ab_value 
    row["mcts_regret"] = exact.best_value - mcts_value

    return row


def _generate_row_star(args):
    return generate_row(*args)


def generate(n_trees, seed0=0, workers=WORKERS, cfg=CFG):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_generate_row_star, [(s, cfg) for s in range(seed0, seed0 + n_trees)], chunksize=16))
    return [r for r in rows if r is not None]


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    rows = generate(NUMBER_OF_TREES)
    
    with open(OUTPUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    algo = {k: sum(1 for r in rows if r["which_algo"] == k) for k in ("AB", "MCTS", "Both", "Neither")}
    better = {k: sum(1 for r in rows if r["which_better"] == k) for k in ("AB", "MCTS", "Tie")}
    print(f"sacuvano {len(rows)} redova (od {NUMBER_OF_TREES} pokusaja) -> {OUTPUT_PATH}")
    print(f"which_algo:   {algo}")
    print(f"which_better: {better}")


if __name__ == "__main__":
    main()
