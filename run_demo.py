"""
P1 Assessment-Aware Study Planner — End-to-End Demo

Runs: data load → NN train → Search → NN mark check → CSP schedule
and prints the search trace, CSP pruning stats, week-by-week timetable,
and predicted marks for a selected student.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nn.predictor import MarkPredictor
from integration.planner_system import PlannerSystem, format_timetable_text


DATA_PATH = os.path.join(ROOT, "data", "processed_student_profiles.csv")


def load_profiles(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Processed profiles not found at {path}.")
        print("Generating data via data/make_data.py ...")
        from data.make_data import process_oulad_data

        process_oulad_data()
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("Student profile CSV is empty.")
    return df


def select_student(df: pd.DataFrame, student_id: int | None, index: int) -> pd.Series:
    if student_id is not None:
        matches = df[df["student_id"] == student_id]
        if matches.empty:
            available = ", ".join(str(s) for s in df["student_id"].tolist())
            raise SystemExit(
                f"Student id {student_id} not found. Available ids: {available}"
            )
        return matches.iloc[0]
    if index < 0 or index >= len(df):
        raise SystemExit(f"Student index {index} out of range (0..{len(df) - 1}).")
    return df.iloc[index]


def print_student_header(student: pd.Series) -> None:
    print("=" * 72)
    print(" P1 Assessment-Aware Study Planner — DEMO")
    print("=" * 72)
    print(f" Student ID        : {int(student['student_id'])}")
    print(
        f" Prev scores       : "
        f"{student['prev_score_1']:.1f}, "
        f"{student['prev_score_2']:.1f}, "
        f"{student['prev_score_3']:.1f}"
    )
    print(
        f" VLE click groups  : "
        f"{int(student['clicks_group_1'])}, "
        f"{int(student['clicks_group_2'])}, "
        f"{int(student['clicks_group_3'])}"
    )
    print(f" Target mark       : {student['target_mark']:.2f}")
    print(f" Weekly capacity   : {int(student['weekly_capacity'])} slot(s)/week")
    print(f" Clicks / session  : {int(student['clicks_per_session'])}")
    print("=" * 72)


def print_search_trace(result: dict, max_rows: int = 40) -> None:
    trace = result.get("search_trace", [])
    print("\n--- Search Trace ---")
    print(
        f"Algorithm: {result['search_algo'].upper()} | "
        f"Generated: {result['generated_nodes']} | "
        f"Expanded: {result['expanded_nodes']} | "
        f"NN-goal plans checked: {result['plans_checked']}"
    )
    if not trace:
        print("  (empty)")
        return

    header = (
        f"  {'#':>4}  {'State(A1,A2,A3)':<16}  {'g':>3}  {'h':>3}  {'f':>3}  "
        f"{'Pred':>6}  {'Goal?':<5}  CSP"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    shown = trace[:max_rows]
    for i, step in enumerate(shown, start=1):
        csp = step.get("csp_status", "-")
        print(
            f"  {i:4d}  {str(step['state']):<16}  {step['g']:3d}  "
            f"{int(step['h']):3d}  {int(step['f']):3d}  "
            f"{step['predicted_mark']:6.2f}  "
            f"{'YES' if step['meets_target'] else 'no':<5}  {csp}"
        )
    if len(trace) > max_rows:
        print(f"  ... ({len(trace) - max_rows} more steps omitted)")


def print_csp_pruning(result: dict) -> None:
    print("\n--- CSP Pruning / Solve Stats ---")
    print(f"Method: {result['csp_method']}")

    rejected = result.get("rejected_plans", [])
    if rejected:
        print(f"Rejected plans (NN ok, CSP failed): {len(rejected)}")
        for i, plan in enumerate(rejected, start=1):
            print(
                f"  [{i}] allocation={plan['allocation']}  "
                f"sessions={plan['total_sessions']}  "
                f"pred={plan['predicted_mark']:.2f}  "
                f"reason={plan['csp_status']}  "
                f"backtracks={plan.get('csp_backtracks', 0)}"
            )
            pruning = plan.get("pruning")
            if pruning and pruning.get("method") == "AC-3":
                print(
                    f"       AC-3 pruned {pruning.get('total_values_pruned', 0)} "
                    f"domain value(s); consistent={pruning.get('consistent')}"
                )

    csp_trace = result.get("csp_trace", [])
    if not csp_trace:
        print("  No CSP attempts were made.")
        return

    last = csp_trace[-1]
    pruning = last.get("pruning") or result.get("csp_pruning")
    print(
        f"Last CSP attempt on allocation={last.get('allocation')} → {last.get('status')}"
    )
    if result["status"] == "Success":
        print(
            f"  Assignments tried : {result.get('csp_assignments', last.get('assignments', 0))}"
        )
        print(
            f"  Backtracks        : {result.get('csp_backtracks', last.get('backtracks', 0))}"
        )

    if pruning:
        method = pruning.get("method", "?")
        print(f"  Inference         : {method}")
        if method == "AC-3":
            print(f"  AC-3 consistent   : {pruning.get('consistent')}")
            print(f"  Total values pruned: {pruning.get('total_values_pruned', 0)}")
            removed = pruning.get("values_pruned_per_var") or {}
            if removed:
                # Show a compact sample of pruned variables
                sample = list(removed.items())[:8]
                pretty = ", ".join(f"{v}:{n}" for v, n in sample)
                more = "" if len(removed) <= 8 else f" ... (+{len(removed) - 8} vars)"
                print(f"  Pruned per var    : {pretty}{more}")
            before = pruning.get("domain_sizes_before") or {}
            after = pruning.get("domain_sizes_after") or {}
            if before and after:
                avg_before = sum(before.values()) / max(len(before), 1)
                avg_after = sum(after.values()) / max(len(after), 1)
                print(
                    f"  Avg domain size   : {avg_before:.1f} → {avg_after:.1f} after AC-3"
                )
        elif "note" in pruning:
            print(f"  Note              : {pruning['note']}")


def print_success(result: dict) -> None:
    print("\n--- Final Plan ---")
    print(f"Status            : SUCCESS")
    print(f"Allocation (A1,A2,A3): {result['allocation']}")
    print(f"Total sessions    : {result['total_sessions']}")
    print(f"Plan sequence     : {result['plan_sequence']}")
    print(
        f"Predicted mark    : {result['baseline_predicted_mark']:.2f} (baseline / 0 extra)"
        f"  →  {result['final_predicted_mark']:.2f} (with plan)"
    )
    print(f"Target mark       : {result['target_mark']:.2f}")
    gap = result["final_predicted_mark"] - result["target_mark"]
    print(f"Margin vs target  : {gap:+.2f}")

    print("\n--- Week-by-Week Timetable ---")
    print(
        format_timetable_text(
            result["timetable"], weekly_capacity=int(result["weekly_capacity"])
        )
    )

    print("\n--- Slot → Session Map (raw CSP assignment) ---")
    schedule = result["schedule"]
    for var in sorted(schedule.keys(), key=lambda v: schedule[v]):
        slot = schedule[var]
        cap = int(result["weekly_capacity"])
        week = slot // cap + 1
        print(f"  {var:<10} → global slot {slot:3d}  (week {week}, slot {slot % cap + 1})")


def print_unreachable(result: dict) -> None:
    print("\n--- Final Plan ---")
    print(f"Status            : {result['message']}")
    print(f"Target mark       : {result['target_mark']:.2f}")
    print(f"Weekly capacity   : {result['weekly_capacity']} slot(s)/week "
          f"× {result.get('max_capacity_sessions', 0) // max(result['weekly_capacity'], 1)} weeks "
          f"= {result.get('max_capacity_sessions', 0)} total slots")
    print(f"Baseline predicted mark      : {result['baseline_predicted_mark']:.2f}")
    print(f"Best possible predicted mark : {result['best_possible_mark']:.2f}")
    print(
        f"  (seen at allocation {result.get('best_allocation_seen')} "
        f"/ {result.get('best_total_sessions')} sessions)"
    )
    print(
        f"Mark at full capacity ({result.get('max_capacity_sessions')} sessions): "
        f"{result.get('max_capacity_predicted_mark'):.2f}"
    )
    if result.get("rejected_plans"):
        print(
            f"NN-feasible plans that CSP could not schedule: "
            f"{len(result['rejected_plans'])}"
        )


def run_demo(
    student_id: int | None = None,
    index: int = 0,
    search_algo: str = "astar",
    csp_method: str = "ac3",
    skip_mono: bool = False,
) -> dict:
    df = load_profiles()
    student = select_student(df, student_id=student_id, index=index)
    print_student_header(student)

    print("\n--- Training Neural Network Mark Predictor ---")
    predictor = MarkPredictor(random_state=42)
    mse_nn, mse_base = predictor.train(df)
    print(f"NN beats baseline on MSE? {'YES' if mse_nn < mse_base else 'NO / close'}")

    if not skip_mono:
        print("\n--- Monotonicity Check (selected student) ---")
        violations = predictor.test_monotonicity(student, max_sessions=10)
        print(f"Violations: {violations}")

    print(
        f"\n--- Running Integrated Planner "
        f"(Search={search_algo.upper()}, CSP={csp_method}) ---"
    )
    planner = PlannerSystem(
        predictor=predictor,
        search_algo=search_algo,
        csp_method=csp_method,
        weeks=12,
    )
    result = planner.plan(student)

    print_search_trace(result)
    print_csp_pruning(result)

    if result["status"] == "Success":
        print_success(result)
    else:
        print_unreachable(result)

    print("\n" + "=" * 72)
    print(" Demo finished.")
    print("=" * 72)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assessment-Aware Study Planner end-to-end demo"
    )
    parser.add_argument(
        "--student-id",
        type=int,
        default=None,
        help="Select student by student_id from the processed CSV",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Select student by row index (default: 0)",
    )
    parser.add_argument(
        "--search",
        choices=["ucs", "greedy", "astar"],
        default="astar",
        help="Search algorithm (default: astar)",
    )
    parser.add_argument(
        "--csp",
        choices=["bt", "fc", "ac3"],
        default="ac3",
        help="CSP method: bt | fc (BT+FC) | ac3 (AC-3 + BT+FC)",
    )
    parser.add_argument(
        "--skip-mono",
        action="store_true",
        help="Skip the monotonicity probe for faster demo runs",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        student_id=args.student_id,
        index=args.index,
        search_algo=args.search,
        csp_method=args.csp,
        skip_mono=args.skip_mono,
    )
