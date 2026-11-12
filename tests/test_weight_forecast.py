from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from weight_forecast import estimate_goal_date, recent_logged_calorie_days


AS_OF = date(2026, 11, 12)
OBJECTIVE = "Perdre du poids progressivement"


class GoalDateEstimateTests(unittest.TestCase):
    def test_calorie_days_allow_variable_meal_counts(self) -> None:
        daily = pd.DataFrame({
            "Date": pd.to_datetime([f"2026-11-{day:02d}" for day in range(2, 13)]),
            "Calories estimées": [1800] * 11,
            "Repas estimés": [1, 2, 3, 1, 2, 1, 3, 2, 1, 2, 1],
        })
        selected = recent_logged_calorie_days(daily, as_of=AS_OF)
        self.assertEqual(len(selected), 10)
        self.assertIn(1, selected["Repas estimés"].tolist())
        self.assertIn(3, selected["Repas estimés"].tolist())

    def test_estimates_date_from_recent_weight_trend(self) -> None:
        weights = [
            {"measured_on": "2026-10-15", "weight_kg": 80.0},
            {"measured_on": "2026-10-29", "weight_kg": 79.5},
            {"measured_on": "2026-11-12", "weight_kg": 79.0},
        ]
        result = estimate_goal_date(weights, 77.0, OBJECTIVE, as_of=AS_OF)
        self.assertEqual(result.status, "estimated")
        self.assertEqual(result.estimated_date, date(2027, 1, 7))
        self.assertAlmostEqual(result.weekly_change_kg, -0.25)

    def test_needs_enough_pesees_over_time(self) -> None:
        weights = [
            {"measured_on": "2026-11-01", "weight_kg": 80.0},
            {"measured_on": "2026-11-12", "weight_kg": 79.8},
        ]
        result = estimate_goal_date(weights, 77.0, OBJECTIVE, as_of=AS_OF)
        self.assertEqual(result.status, "insufficient")
        self.assertIsNone(result.estimated_date)

    def test_does_not_predict_when_trend_moves_away(self) -> None:
        weights = [
            {"measured_on": "2026-10-15", "weight_kg": 79.0},
            {"measured_on": "2026-10-29", "weight_kg": 79.5},
            {"measured_on": "2026-11-12", "weight_kg": 80.0},
        ]
        result = estimate_goal_date(weights, 77.0, OBJECTIVE, as_of=AS_OF)
        self.assertEqual(result.status, "opposite")
        self.assertIsNone(result.estimated_date)

    def test_uses_latest_pesee_for_reached_target(self) -> None:
        weights = [{"measured_on": "2026-11-12", "weight_kg": 76.8}]
        result = estimate_goal_date(weights, 77.0, OBJECTIVE, as_of=AS_OF)
        self.assertEqual(result.status, "reached")
        self.assertEqual(result.estimated_date, AS_OF)

    def test_requires_recent_pesee(self) -> None:
        weights = [
            {"measured_on": "2026-08-01", "weight_kg": 80.0},
            {"measured_on": "2026-08-15", "weight_kg": 79.5},
            {"measured_on": "2026-09-01", "weight_kg": 79.0},
        ]
        result = estimate_goal_date(weights, 77.0, OBJECTIVE, as_of=AS_OF)
        self.assertEqual(result.status, "stale")


if __name__ == "__main__":
    unittest.main()
