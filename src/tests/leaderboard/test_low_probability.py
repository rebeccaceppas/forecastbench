"""Low-probability market filtering on processed forecast rows."""

import numpy as np
import pandas as pd


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
                "id": "boundary-market",
                "source": "polymarket",
                "resolved": True,
                "resolution_date": "2026-05-10",
                "imputed": False,
                "market_value_on_due_date": 0.10,
            },
            {
                "id": "high-market",
                "source": "metaculus",
                "resolved": True,
                "resolution_date": "2026-05-10",
                "imputed": False,
                "market_value_on_due_date": 0.50,
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


def test_low_probability_mask_is_market_only_and_strictly_below_cutoff(leaderboard_main):
    df = _forecast_rows()
    mask = leaderboard_main.get_low_probability_market_mask(df)
    assert mask.tolist() == [True, False, False, False]


def test_get_df_info_keeps_all_resolved_markets_without_low_prob_filter(leaderboard_main):
    processed = leaderboard_main.get_df_info(
        df=_forecast_rows(),
        org_and_model=_org_and_model(),
        forecast_due_date="2026-05-07",
    )
    assert processed is not None
    assert sorted(processed["id"].tolist()) == [
        "boundary-market",
        "dataset-question",
        "high-market",
        "low-market",
    ]


def test_get_df_info_filters_markets_but_keeps_dataset_rows(leaderboard_main):
    processed = leaderboard_main.get_df_info(
        df=_forecast_rows(),
        org_and_model=_org_and_model(),
        forecast_due_date="2026-05-07",
        filter_low_probability_markets=True,
    )
    assert processed is not None
    assert sorted(processed["id"].tolist()) == ["dataset-question", "low-market"]


def test_parity_summary_reports_empty_string_when_no_intersection(leaderboard_main):
    # The SOTA trend need not cross the superforecaster median on a question subset;
    # that must be reported as a blank date rather than crashing the leaderboard.
    ordinal = float(pd.Timestamp("2027-01-15").toordinal())
    summary = leaderboard_main.summarize_parity_dates(
        {"market": {"low_probability_leaderboard": [ordinal, ordinal + 30]}},
        {"market": {"low_probability_leaderboard": pd.NaT}},
    )

    assert summary["market"]["low_probability_leaderboard"]["intersection"] == ""
    assert summary["market"]["low_probability_leaderboard"]["lower"] == "January 2027"


def test_low_probability_variants_write_to_distinct_stems(leaderboard_main):
    stems = leaderboard_main.LEADERBOARD_FILE_STEMS
    assert (
        stems[leaderboard_main.LeaderboardType.LOW_PROBABILITY]
        != stems[leaderboard_main.LeaderboardType.LOW_PROBABILITY_TOURNAMENT]
    )


def test_parity_summary_formats_a_real_intersection_date(leaderboard_main):
    ordinal = float(pd.Timestamp("2027-01-15").toordinal())
    summary = leaderboard_main.summarize_parity_dates(
        {"market": {"low_probability_leaderboard": [ordinal]}},
        {"market": {"low_probability_leaderboard": np.float64(ordinal)}},
    )

    assert summary["market"]["low_probability_leaderboard"]["intersection"] == "January 2027"
