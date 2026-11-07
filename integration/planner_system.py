"""
P1 Assessment-Aware Study Planner — Integration Layer

Workflow:
  1. Search proposes a session allocation plan (A1, A2, A3).
  2. NN predictor checks whether the predicted mark meets the target.
  3. If yes, the CSP scheduler tries to place those sessions into weekly slots
     (AC-3 and/or BT+FC).
  4. If CSP fails, the plan is discarded and Search continues to the next plan.
  5. If nothing works, report that the target is unreachable at this capacity
     and show the best achievable predicted marks.
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional, Tuple

from search.search_planner import SearchNode
from csp.scheduler import CSPScheduler


class PlannerSystem:
    """Orchestrates Search + NN + CSP into one assessment-aware planner."""

    def __init__(
        self,
        predictor,
        search_algo: str = "astar",
        csp_method: str = "ac3",
        weeks: int = 12,
    ):
        if search_algo not in ("ucs", "greedy", "astar"):
            raise ValueError(f"Unsupported search algorithm: {search_algo}")
        if csp_method not in ("none", "bt", "fc", "ac3"):
            raise ValueError(f"Unsupported CSP method: {csp_method}")

        self.predictor = predictor
        self.search_algo = search_algo
        self.csp_method = "none" if csp_method == "bt" else csp_method
        self.weeks = weeks

        # Filled during the last plan() call for demo / reporting
        self.search_trace: List[Dict[str, Any]] = []
        self.csp_trace: List[Dict[str, Any]] = []
        self.rejected_plans: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, student_row) -> Dict[str, Any]:
        """
        Run the full Search → NN → CSP loop for one student.

        Returns a result dict with status, timetable, predicted marks, and traces.
        """
        student = self._as_dict(student_row)
        target = float(student["target_mark"])
        weekly_capacity = int(student["weekly_capacity"])
        max_total_sessions = self.weeks * weekly_capacity

        self.search_trace = []
        self.csp_trace = []
        self.rejected_plans = []

        baseline_mark = self.predictor.predict_student_mark(student, extra_sessions=0)
        best_mark = baseline_mark
        best_allocation = (0, 0, 0)
        best_total = 0

        # --- Search setup (same state space as search_planner.run_search) ---
        frontier: List[Tuple[float, int, SearchNode]] = []
        nodes_generated = 1
        nodes_expanded = 0
        explored = set()
        plans_checked = 0

        def heuristic(state: Tuple[int, int, int]) -> float:
            if self.search_algo == "ucs":
                return 0.0
            pred = self.predictor.predict_student_mark(student, extra_sessions=sum(state))
            return 0.0 if pred >= target else 1.0

        start = SearchNode((0, 0, 0), path_cost=0, heuristic=heuristic((0, 0, 0)))
        self._push(frontier, start, nodes_generated)

        while frontier:
            _, _, node = heapq.heappop(frontier)
            state = node.state
            total_sessions = sum(state)

            if state in explored:
                continue
            if total_sessions > max_total_sessions:
                continue

            predicted = self.predictor.predict_student_mark(
                student, extra_sessions=total_sessions
            )

            # Track best achievable mark seen so far
            if predicted > best_mark + 1e-9 or (
                abs(predicted - best_mark) <= 1e-9 and total_sessions < best_total
            ):
                best_mark = predicted
                best_allocation = state
                best_total = total_sessions

            step = {
                "expanded": nodes_expanded + 1,
                "state": state,
                "total_sessions": total_sessions,
                "g": node.path_cost,
                "h": node.heuristic,
                "f": node.path_cost + node.heuristic,
                "predicted_mark": round(predicted, 2),
                "meets_target": predicted >= target,
                "action": node.action,
            }

            # Goal candidate for Search/NN: predicted mark reaches target
            if predicted >= target:
                plans_checked += 1
                csp_result = self._try_schedule(state, weekly_capacity, plans_checked)

                step["csp_status"] = csp_result["status"]
                self.search_trace.append(step)

                if csp_result["status"] == "Success":
                    nodes_expanded += 1
                    timetable = self.build_timetable(
                        csp_result["schedule"], weekly_capacity
                    )
                    return {
                        "status": "Success",
                        "student_id": student["student_id"],
                        "target_mark": target,
                        "weekly_capacity": weekly_capacity,
                        "search_algo": self.search_algo,
                        "csp_method": self.csp_method,
                        "allocation": state,
                        "total_sessions": total_sessions,
                        "plan_sequence": self._extract_plan_sequence(node),
                        "baseline_predicted_mark": round(baseline_mark, 2),
                        "final_predicted_mark": round(predicted, 2),
                        "schedule": csp_result["schedule"],
                        "timetable": timetable,
                        "csp_assignments": csp_result.get("assignments", 0),
                        "csp_backtracks": csp_result.get("backtracks", 0),
                        "csp_pruning": csp_result.get("pruning"),
                        "generated_nodes": nodes_generated,
                        "expanded_nodes": nodes_expanded,
                        "plans_checked": plans_checked,
                        "rejected_plans": list(self.rejected_plans),
                        "search_trace": list(self.search_trace),
                        "csp_trace": list(self.csp_trace),
                    }

                # CSP failed → discard plan, keep searching
                self.rejected_plans.append(
                    {
                        "allocation": state,
                        "total_sessions": total_sessions,
                        "predicted_mark": round(predicted, 2),
                        "csp_status": csp_result["status"],
                        "csp_backtracks": csp_result.get("backtracks", 0),
                        "pruning": csp_result.get("pruning"),
                    }
                )
                # Fall through: expand this node so other allocations are explored
            else:
                self.search_trace.append(step)

            explored.add(state)
            nodes_expanded += 1

            if total_sessions >= max_total_sessions:
                continue

            for i in range(3):
                new_state_list = list(state)
                new_state_list[i] += 1
                new_state = tuple(new_state_list)
                if new_state in explored:
                    continue
                # Soft capacity guard: never propose more sessions than total slots
                if sum(new_state) > max_total_sessions:
                    continue

                nodes_generated += 1
                h = heuristic(new_state)
                child = SearchNode(
                    new_state,
                    parent=node,
                    action=f"Assign {i + 1}",
                    path_cost=node.path_cost + 1,
                    heuristic=h,
                )
                self._push(frontier, child, nodes_generated)

        # Also evaluate the absolute max-effort mark under capacity (for the report)
        max_effort_mark = self.predictor.predict_student_mark(
            student, extra_sessions=max_total_sessions
        )
        if max_effort_mark > best_mark:
            best_mark = max_effort_mark
            best_total = max_total_sessions
            best_allocation = (
                max_total_sessions,
                0,
                0,
            )  # illustrative; mark depends on total only

        return {
            "status": "Unreachable",
            "message": "Target unreachable at this capacity",
            "student_id": student["student_id"],
            "target_mark": target,
            "weekly_capacity": weekly_capacity,
            "search_algo": self.search_algo,
            "csp_method": self.csp_method,
            "baseline_predicted_mark": round(baseline_mark, 2),
            "best_possible_mark": round(best_mark, 2),
            "best_allocation_seen": best_allocation,
            "best_total_sessions": best_total,
            "max_capacity_sessions": max_total_sessions,
            "max_capacity_predicted_mark": round(max_effort_mark, 2),
            "generated_nodes": nodes_generated,
            "expanded_nodes": nodes_expanded,
            "plans_checked": plans_checked,
            "rejected_plans": list(self.rejected_plans),
            "search_trace": list(self.search_trace),
            "csp_trace": list(self.csp_trace),
        }

    # ------------------------------------------------------------------
    # CSP + timetable helpers
    # ------------------------------------------------------------------

    def _try_schedule(
        self, allocation: Tuple[int, int, int], weekly_capacity: int, plan_id: int
    ) -> Dict[str, Any]:
        """Run CSP on a candidate allocation and record pruning / search stats."""
        scheduler = CSPScheduler(
            allocation=allocation,
            weekly_capacity=weekly_capacity,
            weeks=self.weeks,
        )

        # 0 sessions → empty schedule is a valid solution
        if not scheduler.variables:
            result = {
                "status": "Success",
                "schedule": {},
                "assignments": 0,
                "backtracks": 0,
                "pruning": {"method": self.csp_method, "note": "No variables to schedule."},
            }
            self.csp_trace.append({"plan_id": plan_id, "allocation": allocation, **result})
            return result

        domain_before = {v: len(scheduler.domains[v]) for v in scheduler.variables}
        pruning_info: Optional[Dict[str, Any]] = None

        if self.csp_method == "ac3":
            ok = scheduler.ac3()
            domain_after = {v: len(scheduler.domains[v]) for v in scheduler.variables}
            values_removed = {
                v: domain_before[v] - domain_after[v]
                for v in scheduler.variables
                if domain_before[v] != domain_after[v]
            }
            pruning_info = {
                "method": "AC-3",
                "consistent": ok,
                "domain_sizes_before": domain_before,
                "domain_sizes_after": domain_after,
                "values_pruned_per_var": values_removed,
                "total_values_pruned": sum(values_removed.values()),
            }
            if not ok:
                result = {
                    "status": "Failed by AC-3 pre-processing",
                    "pruning": pruning_info,
                    "backtracks": 0,
                    "assignments": 0,
                }
                self.csp_trace.append({"plan_id": plan_id, "allocation": allocation, **result})
                return result

            # AC-3 succeeded as pre-process; finish with BT + FC
            assignment = scheduler.backtrack({}, inference="ac3")
        elif self.csp_method == "fc":
            assignment = scheduler.backtrack({}, inference="fc")
            pruning_info = {
                "method": "BT+FC",
                "note": "Forward checking removes the taken slot from remaining domains at each step.",
                "domain_sizes_initial": domain_before,
            }
        else:
            # Plain backtracking
            assignment = scheduler.backtrack({}, inference="none")
            pruning_info = {
                "method": "BT",
                "note": "No forward checking / AC-3; consistency checked on assignment only.",
                "domain_sizes_initial": domain_before,
            }

        # `is not None` so empty {} schedules are accepted
        if assignment is not None:
            result = {
                "status": "Success",
                "schedule": assignment,
                "assignments": scheduler.assignments_count,
                "backtracks": scheduler.backtracks_count,
                "pruning": pruning_info,
            }
        else:
            result = {
                "status": "Failed to schedule",
                "assignments": scheduler.assignments_count,
                "backtracks": scheduler.backtracks_count,
                "pruning": pruning_info,
            }

        self.csp_trace.append({"plan_id": plan_id, "allocation": allocation, **result})
        return result

    def build_timetable(
        self, schedule: Dict[str, int], weekly_capacity: int
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Convert flat slot assignment {var: slot_index} into a week-by-week timetable.

        Slot indexing matches CSPScheduler:
          week = slot // weekly_capacity + 1   (1..weeks)
          slot_in_week = slot % weekly_capacity + 1
        """
        weeks: Dict[int, List[Dict[str, Any]]] = {
            w: [] for w in range(1, self.weeks + 1)
        }

        for var, slot in sorted(schedule.items(), key=lambda kv: kv[1]):
            week = slot // weekly_capacity + 1
            slot_in_week = slot % weekly_capacity + 1
            assignment_id = var.split("_")[0]  # e.g. A1
            session_id = var.split("_")[1]  # e.g. S2
            entry = {
                "variable": var,
                "assignment": assignment_id,
                "session": session_id,
                "global_slot": slot,
                "week_slot": slot_in_week,
            }
            if 1 <= week <= self.weeks:
                weeks[week].append(entry)

        return weeks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _push(self, frontier, node: SearchNode, generated_id: int) -> None:
        if self.search_algo == "greedy":
            key = node.heuristic
        else:
            key = node.path_cost + node.heuristic
        heapq.heappush(frontier, (key, generated_id, node))

    @staticmethod
    def _extract_plan_sequence(node: SearchNode) -> List[str]:
        plan = []
        curr = node
        while curr.parent is not None:
            plan.append(curr.action)
            curr = curr.parent
        plan.reverse()
        return plan

    @staticmethod
    def _as_dict(student_row) -> Dict[str, Any]:
        if isinstance(student_row, dict):
            return dict(student_row)
        return student_row.to_dict()


def format_timetable_text(
    timetable: Dict[int, List[Dict[str, Any]]], weekly_capacity: int
) -> str:
    """Pretty-print a week-by-week timetable for the demo console."""
    lines = []
    for week in sorted(timetable.keys()):
        sessions = timetable[week]
        if not sessions:
            lines.append(f"  Week {week:2d}: (free — {weekly_capacity} slot(s) unused)")
            continue
        parts = [
            f"{s['assignment']}-{s['session']}@slot{s['week_slot']}" for s in sessions
        ]
        used = len(sessions)
        free = weekly_capacity - used
        lines.append(
            f"  Week {week:2d}: {', '.join(parts)}"
            + (f"  [{free} free]" if free > 0 else "")
        )
    return "\n".join(lines)
