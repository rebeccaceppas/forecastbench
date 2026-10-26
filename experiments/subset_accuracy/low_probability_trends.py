"""Release-date trends for low-probability dataset and market questions.

Dataset questions are tagged with the shrunk historical base rate from
`dataset_extreme_tags.py`. A question is low-probability when that rate is at
most the configured threshold. Market questions use the due-date price, at most
the same threshold. The shrunk rate is the tag only.

The zero line is the shrunk base rate on a dataset question and the due-date
market price on a market question. Positive means the model beat that
reference. Difficulty adjustment is the two-way question effect fit on every
resolved question of that type, not only the low-probability ones.

Usage:
  python low_probability_trends.py [--config config.toml]
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import plots
import scoring
from _common import HERE, load_config, logger, make_output_dir
from comparisons import HUMAN_COMPARISONS, is_comparison
from dataset_extreme_tags import load_dataset_questions, shrunk_series_rates
from tags import market_low_probability_tags

DATASET_REFERENCE = "Shrunk base rate"
COMBINED_REFERENCE = "Shrunk base rate or market price"
LOAD_COLUMNS = [
    "question_pk",
    "question_type",
    "forecast_due_date",
    "source",
    "id",
    "resolution_date",
    "resolved_to",
    "forecast",
    "market_value_on_due_date",
    "model_pk",
    "organization",
    "model_organization",
    "model",
    "external_submission",
    "forecastbench_llm",
    "first_forecast_due_date",
    "model_release_date",
    "model_age_at_due_date",
    "model_run_key",
    "tournament_model",
    "baseline_model",
]
FE_COLUMNS = [
    "question_pk",
    "model_pk",
    "model",
    "model_organization",
    "organization",
    "external_submission",
    "model_age_at_due_date",
]


def load_leaderboard_rows(cache_dir: str, leaderboard: str) -> pd.DataFrame:
    """Return resolved leaderboard forecasts for market and dataset questions.

    Args:
      cache_dir (str): Directory containing `forecasts.parquet`.
      leaderboard (str): "tournament" or "baseline".
    """
    if leaderboard not in ("tournament", "baseline"):
        raise ValueError("analysis.leaderboard must be 'tournament' or 'baseline'.")
    df = pd.read_parquet(Path(cache_dir) / "forecasts.parquet", columns=LOAD_COLUMNS)
    df = df[df[f"{leaderboard}_model"].astype(bool)].drop(
        columns=["tournament_model", "baseline_model"]
    )
    df = df[df["resolved_to"].isin([0, 1])].reset_index(drop=True)
    return df


def dataset_low_table(cache_dir: str, threshold: float) -> pd.DataFrame:
    """Return dataset questions with the shrunk base rate and a low-probability flag.

    Args:
      cache_dir (str): Directory containing `forecasts.parquet`.
      threshold (float): Low-probability cutoff.
    """
    questions = load_dataset_questions(cache_dir)
    stats = shrunk_series_rates(questions)
    questions["n_earlier"] = stats["n_earlier"].to_numpy()
    questions["base_rate"] = stats["shrunk_base_rate"].to_numpy()
    questions["shrinkage_m"] = stats["shrinkage_m"].to_numpy()
    questions["low"] = questions["base_rate"].notna() & (questions["base_rate"] <= threshold)
    return questions


def fit_effects(df: pd.DataFrame, question_type: str) -> Dict[str, pd.Series]:
    """Fit question effects for Brier and log on one question type.

    Args:
      df (pd.DataFrame): Scored rows with `brier`, `log`, and `question_type`.
      question_type (str): "dataset" or "market".
    """
    part = df.loc[df["question_type"] == question_type, FE_COLUMNS + ["brier", "log"]]
    logger.info(
        f"Fitting question effects on {part['question_pk'].nunique()} {question_type} questions "
        f"({len(part)} forecasts)."
    )
    return {
        rule: scoring.question_effects(part, rule, question_type=question_type)
        for rule in scoring.SCORE_RULES
    }


def apply_effects(df: pd.DataFrame, effects: Dict[str, Dict[str, pd.Series]]) -> pd.DataFrame:
    """Subtract the question effect for the row's question type.

    Args:
      df (pd.DataFrame): Rows with `brier`, `log`, and `question_type`.
      effects (Dict[str, Dict[str, pd.Series]]): Question type to score rule to effects.
    """
    df = df.copy()
    for rule in scoring.SCORE_RULES:
        adjusted = pd.Series(np.nan, index=df.index, dtype=float)
        for question_type, by_rule in effects.items():
            mask = df["question_type"].eq(question_type)
            adjusted.loc[mask] = df.loc[mask, rule] - df.loc[mask, "question_pk"].map(by_rule[rule])
        df[f"{rule}_adj"] = adjusted
    return df


def reference_rows(subset: pd.DataFrame, dataset_forecast: pd.Series, name: str) -> pd.DataFrame:
    """One reference forecast per question.

    Market questions use the due-date price. Dataset questions use the shrunk
    base rate passed in `dataset_forecast`.

    Args:
      subset (pd.DataFrame): Low-probability forecasts. One question may appear many times.
      dataset_forecast (pd.Series): Shrunk base rate indexed by `question_pk`.
      name (str): Reference forecaster name.
    """
    questions = subset.drop_duplicates("question_pk").copy()
    market = questions["question_type"].eq("market")
    naive = questions["question_pk"].map(dataset_forecast)
    questions["forecast"] = questions["market_value_on_due_date"].where(market, naive)
    missing = int(questions["forecast"].isna().sum())
    if missing:
        raise ValueError(f"{missing} reference forecasts are missing for {name}.")
    questions["model_pk"] = name
    questions["model"] = name
    questions["organization"] = "Comparison"
    questions["model_organization"] = "Comparison"
    questions["external_submission"] = False
    questions["forecastbench_llm"] = False
    questions["model_release_date"] = pd.NaT
    questions["first_forecast_due_date"] = pd.NaT
    return questions


def score_against_reference(
    subset: pd.DataFrame,
    effects: Dict[str, Dict[str, pd.Series]],
    dataset_forecast: pd.Series,
    reference: str,
    log_clip: float,
) -> pd.DataFrame:
    """Score every forecaster on `subset` against the reference.

    Args:
      subset (pd.DataFrame): Low-probability rows, already carrying `brier` and `log`.
      effects (Dict[str, Dict[str, pd.Series]]): Effects from `fit_effects`, by question type.
      dataset_forecast (pd.Series): Shrunk base rate indexed by `question_pk`.
      reference (str): Reference forecaster name. Its mean is the zero line.
      log_clip (float): Log-score clip.
    """
    reference_frame = scoring.add_row_scores(
        reference_rows(subset, dataset_forecast, reference), log_clip
    )
    scored = pd.concat([subset, reference_frame], ignore_index=True)
    scored = apply_effects(scored, effects)
    questions = scored.drop_duplicates("question_pk").set_index("question_pk")
    tables = []
    for rule in scoring.SCORE_RULES:
        table, _ = scoring.score_table(
            scored,
            f"{rule}_adj",
            [reference],
            None,
            questions.index,
            paired=False,
            label=f"{rule}_all_question_fit",
        )
        tables.append(table)
    meta = scored.groupby("model_pk")[
        [
            "model",
            "model_release_date",
            "first_forecast_due_date",
            "external_submission",
            "forecastbench_llm",
        ]
    ].first()
    meta["n_subset"] = scored.groupby("model_pk")["question_pk"].nunique()
    results = meta.join(pd.concat(tables, axis=1))
    results.index.name = "model_pk"
    names = results["model"]
    results["kind"] = np.select(
        [names.eq(reference) | is_comparison(names), results["external_submission"].astype(bool)],
        ["Comparison", "External"],
        default="LLM",
    )
    return results


def comparison_lines(
    results: pd.DataFrame,
    reference: str,
    col: str,
    min_n_human: int,
    names: Iterable[str],
) -> Dict[str, float]:
    """Reference-line values for the configured comparison forecasters.

    The zero line is `reference` and is drawn by the plot, so it is not returned here.
    Only names from the market trend's primary list are included, which keeps constant
    forecasts and the naive forecaster out of the figure.

    Args:
      results (pd.DataFrame): Output of `score_against_reference`.
      reference (str): Zero-line forecaster.
      col (str): Score-difference column.
      min_n_human (int): Human lines below this count are labeled as a small sample.
      names (Iterable[str]): Comparison names to draw when they are on the subset.
    """
    lines = {}
    wanted = set(names) - {reference}
    comparisons = results[results["model"].isin(wanted)]
    for _, row in comparisons.iterrows():
        if not np.isfinite(row[col]):
            continue
        label = str(row["model"])
        n = int(row["n_subset"])
        if label in HUMAN_COMPARISONS:
            caveat = "" if n >= min_n_human else ", caveat: small N"
            label = f"{label} (N={n}{caveat})"
        lines[label] = float(row[col])
    return lines


def draw_trends(
    results: pd.DataFrame,
    reference: str,
    population: str,
    title_prefix: str,
    y_note: str,
    config: dict,
    figures: Path,
    tables: Path,
) -> None:
    """Write one Brier figure and one log figure in the market-trend style.

    Args:
      results (pd.DataFrame): Scores against `reference`.
      reference (str): Zero-line forecaster.
      population (str): File-name stem, "dataset" or "combined".
      title_prefix (str): Figure title before the score name.
      y_note (str): Second line of the y-axis label.
      config (dict): Loaded config.
      figures (Path): Figure directory.
      tables (Path): Table directory.
    """
    a_cfg, p_cfg = config["analysis"], config["plots"]
    keep = results["n_subset"] >= a_cfg["min_n"]
    plotted = results[keep | (results["kind"] == "Comparison")]
    llm_pks = plotted.index[plotted["kind"] == "LLM"].tolist()
    ext_pks = plotted.index[plotted["kind"] == "External"].tolist()
    for rule in scoring.SCORE_RULES:
        col = f"{rule}_all_question_fit_vs_{reference}"
        plots.monthly_llm_trend(
            results.loc[llm_pks], col, p_cfg.get("min_release_date") or None
        ).to_csv(tables / f"{population}_monthly_llm_trend_{rule}.csv", index=False)
        plots.trend_plot(
            llms=results.loc[llm_pks],
            externals=results.loc[ext_pks],
            references=comparison_lines(
                plotted, reference, col, a_cfg["min_n_human"], config["comparisons"]["primary"]
            ),
            col=col,
            reference_name=reference,
            rule=rule,
            title=f"{title_prefix}: {plots.RULE_LABELS[rule]}",
            path=figures / f"{population}_trend_release_date_{rule}.png",
            min_date=p_cfg.get("min_release_date") or None,
            y_min=p_cfg.get(f"{rule}_y_min"),
            y_max=p_cfg.get(f"{rule}_y_max"),
            y_caption=(
                f"Difficulty-adjusted {plots.RULE_LABELS[rule]}: {reference} minus model\n"
                f"({y_note})"
            ),
        )


def write_summary(
    path: Path,
    threshold: float,
    dataset_questions: pd.DataFrame,
    n_market_low: int,
    n_market: int,
    populations: Dict[str, pd.DataFrame],
    min_n: int,
) -> None:
    """Write what was tagged and where the figures are.

    Args:
      path (Path): `summary.md`.
      threshold (float): Low-probability cutoff.
      dataset_questions (pd.DataFrame): Output of `dataset_low_table`.
      n_market_low (int): Market questions priced at or below the cutoff.
      n_market (int): Resolved market questions.
      populations (Dict[str, pd.DataFrame]): Score tables by population name.
      min_n (int): Minimum subset forecasts for a model to be counted as plotted.
    """
    pct = int(round(threshold * 100))
    n = len(dataset_questions)
    n_low = int(dataset_questions["low"].sum())
    lines = [
        f"# Low-probability trends, {pct}% cutoff",
        "",
        "Dataset questions are tagged with the shrunk historical base rate from",
        "`dataset_extreme_tags.py` (series family, resolutions before the due date,",
        "pulled toward the source yes rate). Market questions are tagged from",
        "the due-date price. Scores use the same two-way question effects and the same",
        "release-date trend as the market-only figures. The zero line is the shrunk",
        "base rate on a dataset question and the due-date price on a market question.",
        "Positive means better than that reference.",
        "",
        "## Questions",
        "",
        f"- Dataset questions: {n}. Low-probability: {n_low} "
        f"({n_low / n:.1%} of dataset questions).",
        f"- Shrunk rate defined: {int(dataset_questions['base_rate'].notna().sum())}.",
        f"- Market questions: {n_market}. Priced at or below {pct}%: {n_market_low}.",
        "",
        "## Figures",
        "",
        "- `figures/dataset_trend_release_date_brier.png` and `_log.png`: dataset",
        "  questions only. The zero line is the shrunk base rate.",
        "- `figures/combined_trend_release_date_brier.png` and `_log.png`: those",
        "  dataset questions plus the low-probability market questions. The zero line",
        "  is the shrunk base rate on a dataset question and the market price on",
        "  a market question.",
        "",
    ]
    for name, results in populations.items():
        llms = results[(results["kind"] == "LLM") & (results["n_subset"] >= min_n)]
        lines.append(
            f"- {name}: {len(llms)} ForecastBench LLMs with at least {min_n} subset forecasts."
        )
    path.write_text("\n".join(lines) + "\n")


def main(config_path: Optional[str] = None) -> Path:
    """Tag the low-probability questions and write the trend figures.

    Args:
      config_path (Optional[str]): Config file. Defaults to `config.toml`.
    """
    config = load_config(Path(config_path) if config_path else HERE / "config.toml")
    threshold = float(config["analysis"]["threshold"])
    pct = int(round(threshold * 100))
    out = make_output_dir(config, f"low{pct}_trends")
    tables, figures = out / "tables", out / "figures"

    dataset_questions = dataset_low_table(config["data"]["cache_dir"], threshold)
    dataset_low = set(dataset_questions.loc[dataset_questions["low"], "question_pk"])
    base_rate = dataset_questions.set_index("question_pk")["base_rate"]
    logger.info(
        f"{dataset_questions['low'].sum()} dataset questions have a shrunk base rate "
        f"at or below {threshold:.0%}."
    )

    forecasts = load_leaderboard_rows(
        config["data"]["cache_dir"], config["analysis"]["leaderboard"]
    )
    forecasts = scoring.add_row_scores(forecasts, config["scoring"]["log_clip"])
    market = forecasts["question_type"].eq("market")
    market_questions = forecasts.loc[market, ["question_pk", "market_value_on_due_date"]]
    market_tags = market_low_probability_tags(market_questions.drop_duplicates("question_pk"))
    market_low = set(market_tags.loc[market_tags[f"low_{pct}"], "question_pk"])
    logger.info(f"{len(market_low)} market questions are priced at or below {threshold:.0%}.")

    effects = {
        question_type: fit_effects(forecasts, question_type)
        for question_type in ("dataset", "market")
    }
    populations = {
        "dataset": (dataset_low, DATASET_REFERENCE),
        "combined": (dataset_low | market_low, COMBINED_REFERENCE),
    }
    scored = {}
    titles = {
        "dataset": (
            f"Dataset questions with shrunk base rate ≤{pct}%",
            "positive = better than the shrunk base rate",
        ),
        "combined": (
            f"Low-probability dataset and market questions (≤{pct}%)",
            "positive = better than the shrunk base rate or the market",
        ),
    }
    for name, (question_ids, reference) in populations.items():
        subset = forecasts[forecasts["question_pk"].isin(question_ids)].copy()
        logger.info(f"Scoring {subset['question_pk'].nunique()} {name} questions.")
        results = score_against_reference(
            subset, effects, base_rate, reference, config["scoring"]["log_clip"]
        )
        results.to_csv(tables / f"{name}_scores.csv")
        prefix, note = titles[name]
        draw_trends(
            results,
            reference,
            name,
            f"{config['analysis']['leaderboard'].title()} leaderboard, {prefix}",
            note,
            config,
            figures,
            tables,
        )
        scored[name] = results

    write_summary(
        out / "summary.md",
        threshold,
        dataset_questions,
        len(market_low),
        int(market_questions["question_pk"].nunique()),
        scored,
        config["analysis"]["min_n"],
    )
    logger.info(f"Done. Outputs in {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
