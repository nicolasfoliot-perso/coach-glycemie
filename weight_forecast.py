"""Projection prudente d'une date cible à partir des pesées observées."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class GoalDateEstimate:
    status: str
    estimated_date: date | None = None
    last_weight_kg: float | None = None
    weekly_change_kg: float | None = None
    measurement_count: int = 0


def recent_logged_calorie_days(
    daily: pd.DataFrame, *, as_of: date, window_days: int = 14
) -> pd.DataFrame:
    """Retient les jours renseignés terminés, indépendamment du nombre de repas."""
    if daily.empty:
        return daily.copy()
    today = pd.Timestamp(as_of)
    return daily[
        (daily["Date"] < today)
        & (daily["Date"] >= today - pd.Timedelta(days=window_days))
    ].copy()


def estimate_goal_date(
    weights: list[dict[str, Any]],
    target_weight_kg: float,
    objective: str,
    *,
    as_of: date,
) -> GoalDateEstimate:
    """Extrapole une tendance robuste uniquement si les pesées sont suffisantes.

    Les doublons d'une journée sont remplacés par la dernière mesure. La date
    résulte d'une pente médiane entre paires de pesées espacées d'au moins une
    semaine ; elle ne repose pas sur les calories du journal.
    """
    if objective not in {
        "Perdre du poids progressivement",
        "Prendre du poids progressivement",
    }:
        return GoalDateEstimate("not_applicable")

    by_day: dict[date, float] = {}
    for item in weights:
        try:
            measured_on = date.fromisoformat(str(item["measured_on"])[:10])
            weight_kg = float(item["weight_kg"])
        except (KeyError, TypeError, ValueError):
            continue
        if measured_on <= as_of and 20 <= weight_kg <= 400:
            by_day[measured_on] = weight_kg

    if not by_day:
        return GoalDateEstimate("insufficient")

    ordered = sorted(by_day.items())
    latest_day, latest_weight = ordered[-1]
    base = GoalDateEstimate(
        "insufficient", last_weight_kg=latest_weight,
        measurement_count=len(ordered),
    )
    wants_loss = objective == "Perdre du poids progressivement"
    if (wants_loss and latest_weight <= target_weight_kg) or (
        not wants_loss and latest_weight >= target_weight_kg
    ):
        return GoalDateEstimate(
            "reached", latest_day, latest_weight, measurement_count=len(ordered)
        )
    if (as_of - latest_day).days > 30:
        return GoalDateEstimate(
            "stale", last_weight_kg=latest_weight,
            measurement_count=len(ordered),
        )

    recent = [
        (day, value) for day, value in ordered
        if day >= as_of - timedelta(days=120)
    ]
    if len(recent) < 3 or (recent[-1][0] - recent[0][0]).days < 21:
        return base

    slopes = [
        (second_weight - first_weight) / (second_day - first_day).days
        for index, (first_day, first_weight) in enumerate(recent)
        for second_day, second_weight in recent[index + 1:]
        if (second_day - first_day).days >= 7
    ]
    if not slopes:
        return base

    slope = median(slopes)
    weekly_change = slope * 7
    if (wants_loss and slope >= 0) or (not wants_loss and slope <= 0):
        return GoalDateEstimate(
            "opposite", last_weight_kg=latest_weight,
            weekly_change_kg=weekly_change, measurement_count=len(recent),
        )
    if abs(weekly_change) < 0.10 or abs(weekly_change) > 1.0:
        return GoalDateEstimate(
            "unstable", last_weight_kg=latest_weight,
            weekly_change_kg=weekly_change, measurement_count=len(recent),
        )

    days_to_goal = math.ceil((target_weight_kg - latest_weight) / slope)
    if days_to_goal < 1 or days_to_goal > 365:
        return GoalDateEstimate(
            "beyond_horizon", last_weight_kg=latest_weight,
            weekly_change_kg=weekly_change, measurement_count=len(recent),
        )
    return GoalDateEstimate(
        "estimated", as_of + timedelta(days=days_to_goal),
        latest_weight, weekly_change, len(recent),
    )
