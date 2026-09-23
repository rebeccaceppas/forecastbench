"""Paired human–LLM low-probability Brier analysis."""

import pandas as pd
import pytest

from tests.leaderboard._leaderboard_import import (
    import_leaderboard_main,
    patched_import_environment,
)


def _human_rows():
    return pd.DataFrame(
        [
            {
                "id": "low-a",
                "source": "manifold",
                "resolved": True,
                "resolution_date": "2024-12-01",
                "imputed": False,
                "forecast": 0.01,
                "resolved_to": 0.0,
                "market_value_on_due_date": 0.02,
            },
            {
                "id": "low-b",
                "source": "polymarket",
                "resolved": True,
                "resolution_date": "2024-12-01",
                "imputed": False,
                "forecast": 0.04,
                "resolved_to": 0.0,
                "market_value_on_due_date": 0.005,
            },
            {
                "id": "high-c",
                "source": "metaculus",
                "resolved": True,
                "resolution_date": "2024-12-01",
                "imputed": False,
                "forecast": 0.40,
                "resolved_to": 1.0,
                "market_value_on_due_date": 0.50,
            },
        ]
    )


def test_human_low_probability_questions_keep_only_strictly_low_markets():
    import_leaderboard_main()
    with patched_import_environment():
        import intermediate_analysis

        df = intermediate_analysis.human_low_probability_questions(_human_rows())

    assert sorted(df["id"].tolist()) == ["low-a", "low-b"]
    assert set(df["question_key"]) == {"manifold::low-a", "polymarket::low-b"}


def test_summarize_model_brier_marks_complete_coverage_and_ranks_by_mean_brier():
    import_leaderboard_main()
    with patched_import_environment():
        import intermediate_analysis

        df = pd.DataFrame(
            [
                {
                    "organization": "ForecastBench",
                    "model_organization": "ForecastBench",
                    "model": "Superforecaster median forecast",
                    "model_pk": "super",
                    "brier_score": 0.0001,
                },
                {
                    "organization": "ForecastBench",
                    "model_organization": "ForecastBench",
                    "model": "Superforecaster median forecast",
                    "model_pk": "super",
                    "brier_score": 0.0003,
                },
                {
                    "organization": "ForecastBench",
                    "model_organization": "Anthropic",
                    "model": "partial-model",
                    "model_pk": "partial",
                    "brier_score": 0.00001,
                },
            ]
        )
        summary = intermediate_analysis.summarize_model_brier(df, n_questions=2)

    assert summary.loc[summary["model_pk"] == "super", "complete"].iloc[0]
    assert not summary.loc[summary["model_pk"] == "partial", "complete"].iloc[0]
    assert summary.iloc[0]["model_pk"] == "super"
    assert summary.loc[summary["model_pk"] == "super", "mean_brier"].iloc[0] == pytest.approx(0.0002)
