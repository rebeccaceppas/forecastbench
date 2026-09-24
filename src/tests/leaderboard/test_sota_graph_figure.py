"""SOTA graph figure rendering from the leaderboard's SOTA graph CSV."""

import pandas as pd

from tests.leaderboard._leaderboard_import import patched_import_environment


def _sota_graph_csv():
    return pd.DataFrame(
        [
            {
                "Team": "Lab A",
                "Model Organization": "Lab A",
                "Model": "model-early",
                "Dataset": 60.0,
                "N dataset": 720,
                "Dataset 95% CI": "[59.0, 61.0]",
                "Market": 70.0,
                "N market": 30,
                "Market 95% CI": "[68.0, 72.0]",
                "Overall": 65.0,
                "N": 750,
                "Overall 95% CI": "[64.0, 66.0]",
                "Model release date": "2024-01-01",
            },
            {
                "Team": "Lab B",
                "Model Organization": "Lab B",
                "Model": "model-worse-but-later",
                "Dataset": 55.0,
                "N dataset": 720,
                "Dataset 95% CI": "[54.0, 56.0]",
                "Market": 60.0,
                "N market": 30,
                "Market 95% CI": "[58.0, 62.0]",
                "Overall": 57.0,
                "N": 750,
                "Overall 95% CI": "[56.0, 58.0]",
                "Model release date": "2025-01-01",
            },
            {
                "Team": "Lab B",
                "Model Organization": "Lab B",
                "Model": "model-best",
                "Dataset": 70.0,
                "N dataset": 720,
                "Dataset 95% CI": "[69.0, 71.0]",
                "Market": 80.0,
                "N market": 30,
                "Market 95% CI": "[78.0, 82.0]",
                "Overall": 74.0,
                "N": 750,
                "Overall 95% CI": "[73.0, 75.0]",
                "Model release date": "2026-01-01",
            },
            {
                "Team": "ForecastBench",
                "Model Organization": "ForecastBench",
                "Model": "Superforecaster median forecast",
                "Dataset": 68.0,
                "N dataset": 521,
                "Dataset 95% CI": "[67.0, 69.0]",
                "Market": 75.0,
                "N market": 27,
                "Market 95% CI": "[73.0, 77.0]",
                "Overall": 71.0,
                "N": 548,
                "Overall 95% CI": "[70.0, 72.0]",
                "Model release date": None,
            },
            {
                "Team": "ForecastBench",
                "Model Organization": "ForecastBench",
                "Model": "Always 1",
                "Dataset": 12.0,
                "N dataset": 37294,
                "Dataset 95% CI": "[11.0, 13.0]",
                "Market": 0.5,
                "N market": 1724,
                "Market 95% CI": "[0.4, 0.6]",
                "Overall": 10.6,
                "N": 37294,
                "Overall 95% CI": "[10.0, 11.0]",
                "Model release date": None,
            },
        ]
    )


def test_comparison_models_become_reference_lines_not_points():
    with patched_import_environment():
        import sota_graph_figure

        df_scores = sota_graph_figure.select_score(_sota_graph_csv(), "market")
        references = sota_graph_figure.comparison_scores(df_scores)
        points = sota_graph_figure.model_points(df_scores)

    assert references == {"Superforecaster": 75.0, "Always 1": 0.5}
    assert sorted(points["model"]) == ["model-best", "model-early", "model-worse-but-later"]


def test_sota_requires_beating_every_earlier_model():
    with patched_import_environment():
        import sota_graph_figure

        df_scores = sota_graph_figure.select_score(_sota_graph_csv(), "overall")
        points = sota_graph_figure.mark_sota(sota_graph_figure.model_points(df_scores))

    sota = set(points.loc[points["is_sota"], "model"])
    assert sota == {"model-early", "model-best"}


def test_parity_is_reported_from_the_first_model_to_match_supers():
    with patched_import_environment():
        import sota_graph_figure

        df_scores = sota_graph_figure.select_score(_sota_graph_csv(), "overall")
        points = sota_graph_figure.model_points(df_scores)
        achieved = sota_graph_figure.parity_achieved_date(points, superforecaster_score=71.0)

    assert achieved == "January 01, 2026"


def test_score_type_selects_the_matching_columns_and_sample_sizes():
    with patched_import_environment():
        import sota_graph_figure

        df_market = sota_graph_figure.select_score(_sota_graph_csv(), "market")

    row = df_market[df_market["model"] == "model-best"].iloc[0]
    assert (row["score"], row["ci_lower"], row["ci_upper"], row["n"]) == (80.0, 78.0, 82.0, 30)
