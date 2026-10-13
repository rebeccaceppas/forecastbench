"""Regression check: reproduce the official leaderboard's point scores from the cache.

Scores every question with the production settings (`score_models` with the two-way fixed
effects; market Brier for the tournament, market fixed effects for the baseline) and, when a
published leaderboard CSV is given, compares the Brier Index columns model by model.

Usage:
  python check_official.py [--config config.toml] [--leaderboard tournament]
                           [--official-csv path/to/leaderboard_tournament.csv]
"""

import argparse
from pathlib import Path

import pandas as pd
from _common import leaderboard, load_config, logger, make_output_dir

SCORE_COLUMNS = {
    "brier_index_overall": "Overall",
    "brier_index_dataset": "Dataset",
    "brier_index_market": "Market",
}


def main(config_path: str, leaderboard_type: str, official_csv: str | None) -> None:
    """Score all questions with official settings and compare to a published leaderboard.

    Args:
      config_path (str): Config file.
      leaderboard_type (str): "tournament" or "baseline".
      official_csv (str | None): Published leaderboard CSV to compare against.
    """
    lb = leaderboard()
    config = load_config(Path(config_path))
    df = pd.read_parquet(Path(config["data"]["cache_dir"]) / "forecasts.parquet")
    df = df[df[f"{leaderboard_type}_model"].astype(bool)].reset_index(drop=True)
    # `get_masks` reads `resolved`; every cached row already passed the resolved filters.
    df["resolved"] = True
    adjustment = (
        lb.MarketQuestionAdjustment.MARKET_BRIER
        if leaderboard_type == "tournament"
        else lb.MarketQuestionAdjustment.TWO_WAY_FIXED_EFFECTS
    )
    scores, _ = lb.score_models(
        df=df,
        scoring_funcs=[lb.two_way_fixed_effects],
        market_question_adjustment=adjustment,
    )
    out = make_output_dir(config, f"check_official_{leaderboard_type}")
    keep = ["organization", "model", "n_dataset", "n_market", *SCORE_COLUMNS]
    scores = scores[keep].sort_values("brier_index_overall", ascending=False)
    scores.to_csv(out / "tables" / "reproduced_leaderboard.csv", index=False)
    logger.info(f"Reproduced {len(scores)} leaderboard rows.")

    if not official_csv:
        logger.info("No --official-csv given; skipping the comparison.")
        return
    official = pd.read_csv(official_csv)
    merged = scores.merge(
        official, left_on=["organization", "model"], right_on=["Team", "Model"], how="inner"
    )
    for ours, theirs in SCORE_COLUMNS.items():
        merged[f"diff_{theirs}"] = (merged[ours] - pd.to_numeric(merged[theirs])).abs()
    merged.to_csv(out / "tables" / "comparison_with_official.csv", index=False)
    worst = merged[[f"diff_{c}" for c in SCORE_COLUMNS.values()]].max()
    logger.info(
        f"Matched {len(merged)} of {len(official)} published rows. Largest absolute "
        f"differences:\n{worst}"
    )
    # Published values are rounded to 0.1, so anything within 0.05 matches.
    n_match = int((merged[[f"diff_{c}" for c in SCORE_COLUMNS.values()]] <= 0.05).all(axis=1).sum())
    logger.info(f"{n_match} models match on all three columns within rounding.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.toml"))
    parser.add_argument("--leaderboard", default="tournament", choices=["tournament", "baseline"])
    parser.add_argument("--official-csv", default=None)
    args = parser.parse_args()
    main(args.config, args.leaderboard, args.official_csv)
