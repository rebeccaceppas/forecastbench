"""Extreme market filtering and clipped log score."""

import numpy as np
import pandas as pd
import pytest


def _forecast_rows():
    return pd.DataFrame(
        [
            {
                "id": "low-market",
                "source": "manifold",
                "resolved": True,
                "resolution_date": "2026-05-10",
                "imputed": False,
                "market_value_on_due_date": 0.09,
            },
            {
                "id": "low-boundary",
                "source": "polymarket",
                "resolved": True,
                "resolution_date": "2026-05-10",
                "imputed": False,
                "market_value_on_due_date": 0.10,
            },
            {
                "id": "mid-market",
                "source": "metaculus",
                "resolved": True,
                "resolution_date": "2026-05-10",
                "imputed": False,
                "market_value_on_due_date": 0.50,
            },
            {
                "id": "high-boundary",
                "source": "manifold",
                "resolved": True,
                "resolution_date": "2026-05-10",
                "imputed": False,
                "market_value_on_due_date": 0.90,
            },
            {
                "id": "high-market",
                "source": "polymarket",
                "resolved": True,
                "resolution_date": "2026-05-10",
                "imputed": False,
                "market_value_on_due_date": 0.91,
            },
            {
                "id": "dataset-question",
                "source": "fred",
                "resolved": True,
                "resolution_date": "2026-05-10",
                "imputed": False,
                "market_value_on_due_date": 254.23,
            },
        ]
    )


def _org_and_model():
    return {
        "organization": "ForecastBench",
        "model": "Always 0.5",
        "model_organization": "ForecastBench",
    }


def test_extreme_mask_keeps_both_tails_and_excludes_the_boundaries(leaderboard_main):
    mask = leaderboard_main.get_extreme_market_mask(_forecast_rows())
    assert mask.tolist() == [True, False, False, False, True, False]


def test_get_df_info_keeps_both_tails_and_dataset_rows(leaderboard_main):
    processed = leaderboard_main.get_df_info(
        df=_forecast_rows(),
        org_and_model=_org_and_model(),
        forecast_due_date="2026-05-07",
        filter_extreme_markets=True,
    )
    assert processed is not None
    assert sorted(processed["id"].tolist()) == [
        "dataset-question",
        "high-market",
        "low-market",
    ]


def test_extreme_variants_write_to_distinct_stems(leaderboard_main):
    stems = leaderboard_main.LEADERBOARD_FILE_STEMS
    assert stems[leaderboard_main.LeaderboardType.EXTREME] == "extreme_leaderboard"
    assert (
        stems[leaderboard_main.LeaderboardType.EXTREME_TOURNAMENT]
        == "extreme_tournament_leaderboard"
    )
    assert (
        stems[leaderboard_main.LeaderboardType.EXTREME]
        != stems[leaderboard_main.LeaderboardType.LOW_PROBABILITY]
    )


def test_clipped_log_skill_anchors_always_half_at_zero_and_perfect_at_one(leaderboard_main):
    scored = leaderboard_main.clipped_log_skill_bits(
        pd.DataFrame(
            {
                "forecast": [0.5, 0.5, 1.0, 0.0],
                "resolved_to": [0.0, 1.0, 1.0, 0.0],
            }
        )
    )
    at_clip = 1 - (-np.log2(1 - leaderboard_main.LOG_SCORE_CLIP))
    assert scored["clipped_log_skill_bits"].tolist() == pytest.approx([0.0, 0.0, at_clip, at_clip])


def test_clipped_log_skill_penalizes_zero_more_than_one_percent_on_a_hit(leaderboard_main):
    scored = leaderboard_main.clipped_log_skill_bits(
        pd.DataFrame({"forecast": [0.0, 0.01], "resolved_to": [1.0, 1.0]})
    )
    clipped_zero = 1 - (-np.log2(leaderboard_main.LOG_SCORE_CLIP))
    one_percent = 1 - (-np.log2(0.01))
    assert scored["clipped_log_skill_bits"].iloc[0] == pytest.approx(clipped_zero)
    assert scored["clipped_log_skill_bits"].iloc[1] == pytest.approx(one_percent)
    assert one_percent > clipped_zero
