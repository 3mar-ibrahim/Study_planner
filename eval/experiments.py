"""
P1 Assessment-Aware Study Planner — Evaluation Experiments

Deliverables ( ≥5 seeds; mean ± std tables + PNG plots in eval/ ):

  1. UCS vs Greedy vs A*          — nodes expanded & plan cost
  2. BT vs BT+FC vs BT+FC+AC-3    — backtracks & assignments at 2/3/4 slots/week
  3. NN vs average-mark predictor — both used inside the planner
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "processed_student_profiles.csv")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nn.predictor import MarkPredictor
from search.search_planner import run_search
from csp.scheduler import CSPScheduler
from integration.planner_system import PlannerSystem


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEEDS: List[int] = [0, 1, 2, 3, 4]  # ≥ 5 runs/seeds
SEARCH_ALGOS = ("ucs", "greedy", "astar")
CSP_METHODS = ("bt", "fc", "ac3")  # BT, BT+FC, BT+FC+AC-3
CAPACITIES = (2, 3, 4)
MAX_STUDENTS_SEARCH = 12
MAX_STUDENTS_PLANNER = 10


# ---------------------------------------------------------------------------
# Average-mark baseline predictor (same interface as MarkPredictor)
# ---------------------------------------------------------------------------

class AverageMarkPredictor:
    """
    Simple baseline: predicted mark = mean(prev_score_1..3).
    Extra study sessions do not change the prediction (classic average baseline).
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.is_trained = False
        self._mse: Optional[float] = None
        self._mae: Optional[float] = None

    def train(self, df: pd.DataFrame):
        preds = df[["prev_score_1", "prev_score_2", "prev_score_3"]].mean(axis=1).values
        y = df["target_mark"].values
        self._mse = float(np.mean((preds - y) ** 2))
        self._mae = float(np.mean(np.abs(preds - y)))
        self.is_trained = True
        return self._mse, self._mae

    def predict_student_mark(self, student_row, extra_sessions: int = 0) -> float:
        if isinstance(student_row, dict):
            p1, p2, p3 = (
                float(student_row["prev_score_1"]),
                float(student_row["prev_score_2"]),
                float(student_row["prev_score_3"]),
            )
        else:
            p1 = float(student_row["prev_score_1"])
            p2 = float(student_row["prev_score_2"])
            p3 = float(student_row["prev_score_3"])
        # Baseline ignores effort; clip to valid mark range
        return float(np.clip((p1 + p2 + p3) / 3.0, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Data / helpers
# ---------------------------------------------------------------------------

def load_profiles() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if not df.empty:
            return df
    # On-the-fly mock if CSV missing
    rng = np.random.default_rng(42)
    rows = []
    for i in range(1, 31):
        p1, p2, p3 = rng.integers(40, 90, size=3).astype(float)
        rows.append(
            {
                "student_id": 3000 + i,
                "prev_score_1": p1,
                "prev_score_2": p2,
                "prev_score_3": p3,
                "clicks_group_1": int(rng.integers(80, 300)),
                "clicks_group_2": int(rng.integers(60, 260)),
                "clicks_group_3": int(rng.integers(40, 220)),
                "target_mark": round(min(100.0, float(np.mean([p1, p2, p3]) * 1.10)), 2),
                "weekly_capacity": int(rng.choice([2, 3, 4])),
                "clicks_per_session": 45,
            }
        )
    return pd.DataFrame(rows)


@contextlib.contextmanager
def suppress_stdout():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def print_table(title: str, headers: List[str], rows: List[List[Any]]) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    widths = [len(h) for h in headers]
    str_rows = [[f"{c}" for c in row] for row in rows]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in str_rows:
        print(fmt.format(*row))


def fmt_ms(mean: float, std: float, digits: int = 2) -> str:
    if np.isnan(mean):
        return "n/a"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def sample_students(df: pd.DataFrame, seed: int, n: int) -> pd.DataFrame:
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def train_nn(df: pd.DataFrame, seed: int) -> MarkPredictor:
    model = MarkPredictor(random_state=seed)
    with suppress_stdout():
        model.train(df)
    return model


# ---------------------------------------------------------------------------
# Experiment 1 — UCS vs Greedy vs A*
# ---------------------------------------------------------------------------

def experiment_1_search(df: pd.DataFrame, seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    """Compare search algorithms on nodes expanded and plan cost."""
    # per algo → list of per-seed means
    expanded_by_seed: Dict[str, List[float]] = {a: [] for a in SEARCH_ALGOS}
    cost_by_seed: Dict[str, List[float]] = {a: [] for a in SEARCH_ALGOS}
    success_by_seed: Dict[str, List[float]] = {a: [] for a in SEARCH_ALGOS}

    for seed in seeds:
        predictor = train_nn(df, seed)
        students = sample_students(df, seed, MAX_STUDENTS_SEARCH)

        for algo in SEARCH_ALGOS:
            expanded_vals, cost_vals, successes = [], [], []
            for _, student in students.iterrows():
                res = run_search(
                    predictor,
                    student,
                    algo=algo,
                    max_total_sessions=int(student["weekly_capacity"]) * 12,
                )
                expanded_vals.append(float(res.get("expanded_nodes", 0)))
                if res.get("status") == "Success":
                    successes.append(1.0)
                    cost_vals.append(float(res.get("total_sessions", 0)))
                else:
                    successes.append(0.0)
            expanded_by_seed[algo].append(float(np.mean(expanded_vals)))
            cost_by_seed[algo].append(float(np.mean(cost_vals)) if cost_vals else float("nan"))
            success_by_seed[algo].append(float(np.mean(successes)))

    summary = {}
    rows = []
    for algo in SEARCH_ALGOS:
        e_m, e_s = mean_std(expanded_by_seed[algo])
        c_m, c_s = mean_std([v for v in cost_by_seed[algo] if not np.isnan(v)])
        s_m, s_s = mean_std(success_by_seed[algo])
        summary[algo] = {
            "expanded_mean": e_m,
            "expanded_std": e_s,
            "cost_mean": c_m,
            "cost_std": c_s,
            "success_mean": s_m,
            "success_std": s_s,
            "expanded_per_seed": expanded_by_seed[algo],
            "cost_per_seed": cost_by_seed[algo],
        }
        rows.append(
            [
                algo.upper(),
                fmt_ms(e_m, e_s),
                fmt_ms(c_m, c_s),
                fmt_ms(s_m, s_s, digits=3),
            ]
        )

    print_table(
        f"Experiment 1 — UCS vs Greedy vs A*  (n_seeds={len(seeds)})",
        ["Algorithm", "Nodes Expanded (mean±std)", "Plan Cost / sessions", "Success Rate"],
        rows,
    )
    return summary


def plot_experiment_1(summary: Dict[str, Any], out_path: str) -> None:
    algos = list(SEARCH_ALGOS)
    labels = [a.upper() for a in algos]
    exp_means = [summary[a]["expanded_mean"] for a in algos]
    exp_stds = [summary[a]["expanded_std"] for a in algos]
    cost_means = [summary[a]["cost_mean"] for a in algos]
    cost_stds = [summary[a]["cost_std"] for a in algos]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(algos))
    colors = ["#2E86AB", "#E94F37", "#44AF69"]

    axes[0].bar(x, exp_means, yerr=exp_stds, capsize=5, color=colors, edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Nodes Expanded")
    axes[0].set_title("Search Effort (mean ± std)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(x, cost_means, yerr=cost_stds, capsize=5, color=colors, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Plan Cost (total sessions)")
    axes[1].set_title("Solution Cost (mean ± std)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Experiment 1: UCS vs Greedy vs A*", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {out_path}")


# ---------------------------------------------------------------------------
# Experiment 2 — BT vs BT+FC vs BT+FC+AC-3 × capacities
# ---------------------------------------------------------------------------

def _csp_method_key(method: str) -> str:
    # CSPScheduler.solve: 'none'/'bt' → plain BT, 'fc' → BT+FC, 'ac3' → BT+FC+AC-3
    return {"bt": "none", "fc": "fc", "ac3": "ac3"}[method]


def _challenge_allocations(seed: int, capacity: int) -> List[Tuple[int, int, int]]:
    """
    Allocations that pack near deadline / capacity limits so inference methods differ.
    Includes tight A1 loads (deadline week 4) and mixed multi-assignment packs.
    """
    rng = np.random.default_rng(seed + 17 * capacity)
    # A1 can use at most 4*capacity slots; push close to that limit
    a1_max = 4 * capacity
    base = [
        (min(a1_max, capacity + 2), 0, 0),
        (min(a1_max - 1, capacity + 1), capacity, 0),
        (capacity, capacity, capacity),
        (min(a1_max, 2 * capacity), capacity, 1),
        (0, min(8 * capacity - 1, 3 * capacity), capacity),
        (capacity + 1, capacity + 1, capacity + 1),
        (min(a1_max, 3), min(4, 2 * capacity), min(5, 2 * capacity)),
        (2, 3, 4) if capacity >= 2 else (1, 1, 2),
    ]
    extra = []
    for _ in range(4):
        a1 = int(rng.integers(0, a1_max + 1))
        a2 = int(rng.integers(0, min(8 * capacity, 6) + 1))
        a3 = int(rng.integers(0, min(12 * capacity, 6) + 1))
        if a1 + a2 + a3 == 0:
            a3 = capacity
        if a1 + a2 + a3 <= 12 * capacity:
            extra.append((a1, a2, a3))
    # Deduplicate while preserving order
    seen = set()
    out = []
    for alloc in base + extra:
        if alloc not in seen and sum(alloc) <= 12 * capacity:
            seen.add(alloc)
            out.append(alloc)
    return out


def _prepare_scheduler(
    allocation: Tuple[int, int, int],
    weekly_capacity: int,
    seed: int,
    reverse_domains: bool = True,
) -> CSPScheduler:
    """
    Build a CSP instance and scramble domain order so plain BT must search.
    (Natural ascending domains make earliest-slot greedy almost backtrack-free.)
    """
    scheduler = CSPScheduler(allocation, weekly_capacity=weekly_capacity, weeks=12)
    rng = np.random.default_rng(seed)
    for var in scheduler.variables:
        domain = list(scheduler.domains[var])
        if reverse_domains:
            domain = list(reversed(domain))
        # Light shuffle on top of reverse to vary runs across seeds
        if len(domain) > 2:
            # swap a few pairs
            for _ in range(min(3, len(domain) // 2)):
                i, j = int(rng.integers(0, len(domain))), int(rng.integers(0, len(domain)))
                domain[i], domain[j] = domain[j], domain[i]
        scheduler.domains[var] = domain
    return scheduler


def experiment_2_csp(df: pd.DataFrame, seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    """Compare CSP inference methods across weekly capacities 2/3/4."""
    # key: (method, capacity) → per-seed means of backtracks / assignments
    backtracks: Dict[Tuple[str, int], List[float]] = {}
    assignments: Dict[Tuple[str, int], List[float]] = {}
    success_rates: Dict[Tuple[str, int], List[float]] = {}

    for method in CSP_METHODS:
        for cap in CAPACITIES:
            backtracks[(method, cap)] = []
            assignments[(method, cap)] = []
            success_rates[(method, cap)] = []

    for seed in seeds:
        for method in CSP_METHODS:
            solve_method = _csp_method_key(method)
            for cap in CAPACITIES:
                allocations = _challenge_allocations(seed, cap)
                bt_vals, asg_vals, ok_vals = [], [], []
                for idx, alloc in enumerate(allocations):
                    if sum(alloc) > 12 * cap or sum(alloc) == 0:
                        continue
                    scheduler = _prepare_scheduler(
                        alloc, weekly_capacity=cap, seed=seed * 1009 + idx * 13 + cap
                    )
                    # Domains are scrambled so BT must search; FC / AC-3 prune more
                    res = scheduler.solve(method=solve_method)
                    bt_vals.append(float(res.get("backtracks", 0)))
                    asg_vals.append(float(res.get("assignments", 0)))
                    ok_vals.append(1.0 if res.get("status") == "Success" else 0.0)
                backtracks[(method, cap)].append(float(np.mean(bt_vals)) if bt_vals else 0.0)
                assignments[(method, cap)].append(float(np.mean(asg_vals)) if asg_vals else 0.0)
                success_rates[(method, cap)].append(float(np.mean(ok_vals)) if ok_vals else 0.0)

    summary = {"backtracks": {}, "assignments": {}, "success": {}}
    rows = []
    method_labels = {"bt": "BT", "fc": "BT+FC", "ac3": "BT+FC+AC-3"}
    for method in CSP_METHODS:
        for cap in CAPACITIES:
            b_m, b_s = mean_std(backtracks[(method, cap)])
            a_m, a_s = mean_std(assignments[(method, cap)])
            s_m, s_s = mean_std(success_rates[(method, cap)])
            summary["backtracks"][(method, cap)] = (b_m, b_s)
            summary["assignments"][(method, cap)] = (a_m, a_s)
            summary["success"][(method, cap)] = (s_m, s_s)
            rows.append(
                [
                    method_labels[method],
                    str(cap),
                    fmt_ms(b_m, b_s),
                    fmt_ms(a_m, a_s),
                    fmt_ms(s_m, s_s, digits=3),
                ]
            )

    print_table(
        f"Experiment 2 — BT vs BT+FC vs BT+FC+AC-3  (n_seeds={len(seeds)})",
        ["Method", "Slots/week", "Backtracks", "Assignments", "Success Rate"],
        rows,
    )
    # Keep raw series for optional debugging / richer plots
    summary["_raw_backtracks"] = backtracks
    summary["_raw_assignments"] = assignments
    return summary


def plot_experiment_2(summary: Dict[str, Any], out_path: str) -> None:
    method_labels = {"bt": "BT", "fc": "BT+FC", "ac3": "BT+FC+AC-3"}
    methods = list(CSP_METHODS)
    capacities = list(CAPACITIES)
    x = np.arange(len(capacities))
    width = 0.25
    colors = ["#6C757D", "#F4A261", "#2A9D8F"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, metric, ylabel, title in [
        (axes[0], "backtracks", "Backtracks", "Backtracks by Capacity"),
        (axes[1], "assignments", "Assignments Tried", "Assignments by Capacity"),
    ]:
        for i, method in enumerate(methods):
            means = [summary[metric][(method, cap)][0] for cap in capacities]
            stds = [summary[metric][(method, cap)][1] for cap in capacities]
            ax.bar(
                x + (i - 1) * width,
                means,
                width,
                yerr=stds,
                capsize=4,
                label=method_labels[method],
                color=colors[i],
                edgecolor="black",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in capacities])
        ax.set_xlabel("Weekly Capacity (slots)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Experiment 2: CSP Inference Comparison", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {out_path}")


# ---------------------------------------------------------------------------
# Experiment 3 — NN vs Average predictor inside the planner
# ---------------------------------------------------------------------------

def _prediction_errors(
    predictor, students: pd.DataFrame
) -> Tuple[float, float]:
    """MSE / MAE of predictor mark (0 extra sessions) vs target_mark."""
    errs = []
    for _, row in students.iterrows():
        pred = predictor.predict_student_mark(row, extra_sessions=0)
        errs.append(pred - float(row["target_mark"]))
    errs = np.asarray(errs, dtype=float)
    mse = float(np.mean(errs ** 2))
    mae = float(np.mean(np.abs(errs)))
    return mse, mae


def experiment_3_nn_vs_average(
    df: pd.DataFrame, seeds: Sequence[int] = SEEDS
) -> Dict[str, Any]:
    """
    Plug NN and AverageMarkPredictor into PlannerSystem and compare:
      - prediction MSE/MAE
      - planner success rate
      - sessions used when successful
      - final predicted mark (when successful)
    """
    metrics = {
        "nn": {"mse": [], "mae": [], "success": [], "sessions": [], "final_mark": []},
        "avg": {"mse": [], "mae": [], "success": [], "sessions": [], "final_mark": []},
    }

    for seed in seeds:
        students = sample_students(df, seed, MAX_STUDENTS_PLANNER)
        nn = train_nn(df, seed)
        avg = AverageMarkPredictor(random_state=seed)
        with suppress_stdout():
            avg.train(df)

        for key, model in (("nn", nn), ("avg", avg)):
            mse, mae = _prediction_errors(model, students)
            metrics[key]["mse"].append(mse)
            metrics[key]["mae"].append(mae)

            successes, sessions, finals = [], [], []
            planner = PlannerSystem(
                predictor=model, search_algo="astar", csp_method="ac3", weeks=12
            )
            for _, student in students.iterrows():
                # Keep student's own capacity; average may often be Unreachable
                result = planner.plan(student)
                if result["status"] == "Success":
                    successes.append(1.0)
                    sessions.append(float(result["total_sessions"]))
                    finals.append(float(result["final_predicted_mark"]))
                else:
                    successes.append(0.0)
            metrics[key]["success"].append(float(np.mean(successes)))
            metrics[key]["sessions"].append(
                float(np.mean(sessions)) if sessions else float("nan")
            )
            metrics[key]["final_mark"].append(
                float(np.mean(finals)) if finals else float("nan")
            )

    summary = {}
    rows = []
    for key, label in (("nn", "Neural Network"), ("avg", "Average Mark")):
        block = {}
        for name in ("mse", "mae", "success", "sessions", "final_mark"):
            vals = [v for v in metrics[key][name] if not (isinstance(v, float) and np.isnan(v))]
            m, s = mean_std(vals)
            block[f"{name}_mean"] = m
            block[f"{name}_std"] = s
            block[f"{name}_per_seed"] = metrics[key][name]
        summary[key] = block
        rows.append(
            [
                label,
                fmt_ms(block["mse_mean"], block["mse_std"]),
                fmt_ms(block["mae_mean"], block["mae_std"]),
                fmt_ms(block["success_mean"], block["success_std"], digits=3),
                fmt_ms(block["sessions_mean"], block["sessions_std"]),
                fmt_ms(block["final_mark_mean"], block["final_mark_std"]),
            ]
        )

    print_table(
        f"Experiment 3 — NN vs Average Mark Predictor (inside planner)  (n_seeds={len(seeds)})",
        [
            "Predictor",
            "Pred MSE",
            "Pred MAE",
            "Planner Success",
            "Sessions (ok)",
            "Final Mark (ok)",
        ],
        rows,
    )
    return summary


def plot_experiment_3(summary: Dict[str, Any], out_path: str) -> None:
    labels = ["Neural Network", "Average Mark"]
    keys = ["nn", "avg"]
    colors = ["#264653", "#E76F51"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    # 1) Prediction error
    mse_m = [summary[k]["mse_mean"] for k in keys]
    mse_s = [summary[k]["mse_std"] for k in keys]
    mae_m = [summary[k]["mae_mean"] for k in keys]
    mae_s = [summary[k]["mae_std"] for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    axes[0].bar(x - w / 2, mse_m, w, yerr=mse_s, capsize=4, label="MSE", color="#264653")
    axes[0].bar(x + w / 2, mae_m, w, yerr=mae_s, capsize=4, label="MAE", color="#E76F51")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("Error")
    axes[0].set_title("Prediction Error vs Target")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # 2) Planner success rate
    suc_m = [summary[k]["success_mean"] for k in keys]
    suc_s = [summary[k]["success_std"] for k in keys]
    axes[1].bar(x, suc_m, yerr=suc_s, capsize=5, color=colors, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Planner Success (inside loop)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    # 3) Sessions when successful
    ses_m = [summary[k]["sessions_mean"] for k in keys]
    ses_s = [summary[k]["sessions_std"] for k in keys]
    axes[2].bar(x, ses_m, yerr=ses_s, capsize=5, color=colors, edgecolor="black")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("Sessions")
    axes[2].set_title("Sessions Used (successful plans)")
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Experiment 3: NN vs Average Mark Predictor in Planner",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_experiments(seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    os.makedirs(EVAL_DIR, exist_ok=True)
    df = load_profiles()

    print("\n" + "#" * 78)
    print("# Assessment-Aware Study Planner — Evaluation Suite")
    print(f"# Profiles: {len(df)} students | Seeds: {list(seeds)}")
    print("#" * 78)

    s1 = experiment_1_search(df, seeds)
    p1 = os.path.join(EVAL_DIR, "exp1_search_ucs_greedy_astar.png")
    plot_experiment_1(s1, p1)

    s2 = experiment_2_csp(df, seeds)
    p2 = os.path.join(EVAL_DIR, "exp2_csp_bt_fc_ac3.png")
    plot_experiment_2(s2, p2)

    s3 = experiment_3_nn_vs_average(df, seeds)
    p3 = os.path.join(EVAL_DIR, "exp3_nn_vs_average.png")
    plot_experiment_3(s3, p3)

    print("\n" + "#" * 78)
    print("# All experiments finished. Plots written to eval/:")
    print(f"#   {os.path.basename(p1)}")
    print(f"#   {os.path.basename(p2)}")
    print(f"#   {os.path.basename(p3)}")
    print("#" * 78)

    return {"exp1": s1, "exp2": s2, "exp3": s3}


if __name__ == "__main__":
    run_all_experiments()
