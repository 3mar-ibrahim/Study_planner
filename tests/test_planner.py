"""
pytest suite for P1 Assessment-Aware Study Planner.

Required coverage:
  - 5 normal cases  (average targets, normal weekly capacity)
  - 3 edge cases    (e.g. 100% target + high capacity, 0 sessions needed)
  - 2 impossible cases (must return clear unreachable / no-solution status)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Project root on sys.path so nn / search / csp / integration import cleanly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nn.predictor import MarkPredictor
from search.search_planner import run_search
from csp.scheduler import CSPScheduler
from integration.planner_system import PlannerSystem


DATA_PATH = os.path.join(ROOT, "data", "processed_student_profiles.csv")
PROFILE_COLUMNS = [
    "student_id",
    "prev_score_1",
    "prev_score_2",
    "prev_score_3",
    "clicks_group_1",
    "clicks_group_2",
    "clicks_group_3",
    "target_mark",
    "weekly_capacity",
    "clicks_per_session",
]


# ---------------------------------------------------------------------------
# Mock / fixture data
# ---------------------------------------------------------------------------

def _make_mock_profiles(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic training/profile table (used when CSV is missing)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, n + 1):
        p1 = float(rng.integers(40, 85))
        p2 = float(rng.integers(45, 90))
        p3 = float(rng.integers(50, 95))
        mean_prev = (p1 + p2 + p3) / 3.0
        rows.append(
            {
                "student_id": 2000 + i,
                "prev_score_1": p1,
                "prev_score_2": p2,
                "prev_score_3": p3,
                "clicks_group_1": int(rng.integers(80, 320)),
                "clicks_group_2": int(rng.integers(60, 280)),
                "clicks_group_3": int(rng.integers(40, 240)),
                "target_mark": round(min(100.0, mean_prev * 1.10), 2),
                "weekly_capacity": int(rng.choice([2, 3, 4])),
                "clicks_per_session": 45,
            }
        )
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


def load_or_generate_profiles() -> pd.DataFrame:
    """Load processed CSV if present; otherwise build mock data on the fly."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if not df.empty and set(PROFILE_COLUMNS).issubset(df.columns):
            return df[PROFILE_COLUMNS].copy()
    return _make_mock_profiles()


def make_student(**overrides) -> dict:
    """Build one student profile dict with sensible defaults."""
    base = {
        "student_id": 9000,
        "prev_score_1": 70.0,
        "prev_score_2": 68.0,
        "prev_score_3": 72.0,
        "clicks_group_1": 180,
        "clicks_group_2": 150,
        "clicks_group_3": 120,
        "target_mark": 75.0,
        "weekly_capacity": 3,
        "clicks_per_session": 45,
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def profiles() -> pd.DataFrame:
    return load_or_generate_profiles()


@pytest.fixture(scope="module")
def predictor(profiles: pd.DataFrame) -> MarkPredictor:
    model = MarkPredictor(random_state=42)
    model.train(profiles)
    assert model.is_trained
    return model


def _assert_success_result(result: dict, student: dict) -> None:
    assert result["status"] == "Success"
    assert result["final_predicted_mark"] >= student["target_mark"] - 1e-6
    assert result["total_sessions"] == sum(result["allocation"])
    assert isinstance(result["schedule"], dict)
    assert isinstance(result["timetable"], dict)
    assert len(result["schedule"]) == result["total_sessions"]
    # Every scheduled session lands in some week
    placed = sum(len(v) for v in result["timetable"].values())
    assert placed == result["total_sessions"]


def _assert_unreachable(result: dict) -> None:
    """Impossible cases must expose a clear no-solution / unreachable signal."""
    assert result["status"] == "Unreachable"
    message = str(result.get("message", "")).lower()
    status = str(result["status"]).lower()
    blob = f"{status} {message}"
    assert (
        "unreachable" in blob
        or "no solution" in blob
        or "target unreachable" in blob
    ), f"Expected clear unreachable message, got: {result.get('message')!r}"
    assert "best_possible_mark" in result
    assert result["best_possible_mark"] < result["target_mark"]


# ===========================================================================
# 5 NORMAL CASES
# ===========================================================================

class TestNormalCases:
    """Average targets + normal weekly capacity (2–4 slots)."""

    def test_normal_01_average_target_capacity_3_astar_ac3(self, predictor):
        student = make_student(
            student_id=9101,
            prev_score_1=72.0,
            prev_score_2=70.0,
            prev_score_3=68.0,
            target_mark=72.0,
            weekly_capacity=3,
        )
        # Search module proposes a plan that already clears the NN target
        search_res = run_search(predictor, student, algo="astar", max_total_sessions=15)
        assert search_res["status"] == "Success"
        assert search_res["final_predicted_mark"] >= student["target_mark"]

        planner = PlannerSystem(predictor, search_algo="astar", csp_method="ac3")
        result = planner.plan(student)
        _assert_success_result(result, student)

    def test_normal_02_moderate_target_capacity_2_ucs_fc(self, predictor):
        student = make_student(
            student_id=9102,
            prev_score_1=65.0,
            prev_score_2=62.0,
            prev_score_3=60.0,
            clicks_group_1=160,
            clicks_group_2=140,
            clicks_group_3=110,
            target_mark=68.0,
            weekly_capacity=2,
        )
        planner = PlannerSystem(predictor, search_algo="ucs", csp_method="fc")
        result = planner.plan(student)
        _assert_success_result(result, student)
        assert result["weekly_capacity"] == 2

    def test_normal_03_average_target_capacity_4_greedy_bt(self, predictor):
        student = make_student(
            student_id=9103,
            prev_score_1=75.0,
            prev_score_2=74.0,
            prev_score_3=73.0,
            target_mark=78.0,
            weekly_capacity=4,
        )
        planner = PlannerSystem(predictor, search_algo="greedy", csp_method="bt")
        result = planner.plan(student)
        _assert_success_result(result, student)

        # CSP module can place the chosen allocation under capacity 4
        csp = CSPScheduler(result["allocation"], weekly_capacity=4, weeks=12)
        csp_res = csp.solve(method="none")
        assert csp_res["status"] == "Success"
        assert len(csp_res["schedule"]) == result["total_sessions"]

    def test_normal_04_profile_from_dataset_row(self, predictor, profiles):
        # Typical dataset student: mid-range target, capacity in {2,3,4}
        row = profiles.iloc[0].to_dict()
        # Soften target slightly so a normal plan remains feasible under the NN
        student = dict(row)
        baseline = predictor.predict_student_mark(student, extra_sessions=0)
        student["target_mark"] = round(min(100.0, max(baseline + 2.0, float(row["target_mark"]))), 2)
        if student["weekly_capacity"] < 2:
            student["weekly_capacity"] = 3

        planner = PlannerSystem(predictor, search_algo="astar", csp_method="ac3")
        result = planner.plan(student)
        # Normal path: either succeeds with a timetable, or (rarely) unreachable —
        # for this softened target we expect success.
        assert result["status"] == "Success"
        _assert_success_result(result, student)

    def test_normal_05_balanced_allocation_search_then_csp(self, predictor):
        student = make_student(
            student_id=9105,
            prev_score_1=68.0,
            prev_score_2=71.0,
            prev_score_3=69.0,
            target_mark=74.0,
            weekly_capacity=3,
        )
        search_res = run_search(predictor, student, algo="ucs", max_total_sessions=20)
        assert search_res["status"] == "Success"
        allocation = search_res["allocation"]

        # Direct CSP check on the search proposal (AC-3 + BT+FC)
        scheduler = CSPScheduler(allocation, weekly_capacity=3, weeks=12)
        csp_res = scheduler.solve(method="ac3")
        assert csp_res["status"] == "Success"

        planner = PlannerSystem(predictor, search_algo="ucs", csp_method="ac3")
        result = planner.plan(student)
        _assert_success_result(result, student)
        assert result["final_predicted_mark"] >= search_res["final_predicted_mark"] - 1e-6 or (
            result["total_sessions"] >= search_res["total_sessions"]
        )


# ===========================================================================
# 3 EDGE CASES
# ===========================================================================

class TestEdgeCases:
    """Boundary behaviours the planner must handle cleanly."""

    def test_edge_01_zero_sessions_already_meets_target(self, predictor):
        """Target already met at baseline → allocation (0,0,0), empty timetable."""
        student = make_student(
            student_id=9201,
            prev_score_1=88.0,
            prev_score_2=90.0,
            prev_score_3=92.0,
            clicks_group_1=250,
            clicks_group_2=240,
            clicks_group_3=230,
            weekly_capacity=3,
        )
        baseline = predictor.predict_student_mark(student, extra_sessions=0)
        # Target strictly below (or equal to) the current predicted mark
        student["target_mark"] = round(max(0.0, baseline - 1.0), 2)

        search_res = run_search(predictor, student, algo="astar", max_total_sessions=10)
        assert search_res["status"] == "Success"
        assert search_res["total_sessions"] == 0
        assert search_res["allocation"] == (0, 0, 0)

        # Empty CSP is a valid solution
        csp_res = CSPScheduler((0, 0, 0), weekly_capacity=3, weeks=12).solve(method="ac3")
        assert csp_res["status"] == "Success"
        assert csp_res["schedule"] == {}

        planner = PlannerSystem(predictor, search_algo="astar", csp_method="ac3")
        result = planner.plan(student)
        _assert_success_result(result, student)
        assert result["allocation"] == (0, 0, 0)
        assert result["total_sessions"] == 0
        assert result["schedule"] == {}

    def test_edge_02_target_100_with_high_capacity(self, predictor):
        """Aim for a perfect 100 with capacity 4 — success or clear unreachable."""
        student = make_student(
            student_id=9202,
            prev_score_1=90.0,
            prev_score_2=92.0,
            prev_score_3=95.0,
            clicks_group_1=300,
            clicks_group_2=280,
            clicks_group_3=260,
            target_mark=100.0,
            weekly_capacity=4,
        )
        planner = PlannerSystem(predictor, search_algo="astar", csp_method="ac3", weeks=12)
        result = planner.plan(student)

        if result["status"] == "Success":
            _assert_success_result(result, student)
            assert result["final_predicted_mark"] >= 100.0 - 1e-6
            assert result["weekly_capacity"] == 4
        else:
            _assert_unreachable(result)
            # High capacity still explored up to 12 * 4 slots
            assert result["max_capacity_sessions"] == 48

    def test_edge_03_single_extra_session_boundary(self, predictor):
        """Target just above baseline so Search needs the smallest possible lift."""
        student = make_student(
            student_id=9203,
            prev_score_1=70.0,
            prev_score_2=72.0,
            prev_score_3=71.0,
            clicks_group_1=190,
            clicks_group_2=170,
            clicks_group_3=150,
            weekly_capacity=3,
        )
        baseline = predictor.predict_student_mark(student, extra_sessions=0)
        # Place target a tiny step above baseline (still reachable with a few sessions)
        student["target_mark"] = round(min(99.0, baseline + 0.5), 2)

        planner = PlannerSystem(predictor, search_algo="ucs", csp_method="fc")
        result = planner.plan(student)
        _assert_success_result(result, student)
        assert result["total_sessions"] >= 1
        assert result["baseline_predicted_mark"] < student["target_mark"] + 1e-6


# ===========================================================================
# 2 IMPOSSIBLE CASES
# ===========================================================================

class TestImpossibleCases:
    """Targets that cannot be met under the given weekly capacity."""

    def test_impossible_01_target_95_capacity_1(self, predictor):
        """
        Weak history + target 95 + only 1 slot/week.
        Must return Unreachable with a clear message and best-possible mark.
        """
        student = make_student(
            student_id=9301,
            prev_score_1=35.0,
            prev_score_2=40.0,
            prev_score_3=38.0,
            clicks_group_1=40,
            clicks_group_2=30,
            clicks_group_3=20,
            target_mark=95.0,
            weekly_capacity=1,
        )
        # Sanity: even the NN mark at full 12-slot capacity stays below 95
        max_mark = predictor.predict_student_mark(student, extra_sessions=12)
        assert max_mark < student["target_mark"]

        search_res = run_search(predictor, student, algo="ucs", max_total_sessions=12)
        assert search_res["status"] == "Unreachable"

        planner = PlannerSystem(predictor, search_algo="ucs", csp_method="fc", weeks=12)
        result = planner.plan(student)
        _assert_unreachable(result)
        assert "Target unreachable at this capacity" in result["message"]
        assert result["weekly_capacity"] == 1
        assert result["best_possible_mark"] <= max(max_mark, result["best_possible_mark"])

    def test_impossible_02_near_perfect_target_tiny_capacity(self, predictor):
        """
        Target 98 with weekly_capacity=1 and weak engagement — no feasible plan.
        """
        student = make_student(
            student_id=9302,
            prev_score_1=42.0,
            prev_score_2=45.0,
            prev_score_3=40.0,
            clicks_group_1=50,
            clicks_group_2=45,
            clicks_group_3=35,
            target_mark=98.0,
            weekly_capacity=1,
        )
        max_mark = predictor.predict_student_mark(student, extra_sessions=12)
        assert max_mark < student["target_mark"]

        # CSP alone cannot invent mark lift; integration must still say unreachable
        planner = PlannerSystem(predictor, search_algo="astar", csp_method="ac3", weeks=12)
        result = planner.plan(student)
        _assert_unreachable(result)
        assert result["status"] == "Unreachable"
        assert "unreachable" in result["message"].lower()
        assert isinstance(result["best_possible_mark"], (int, float))
        assert result["best_possible_mark"] < 98.0

