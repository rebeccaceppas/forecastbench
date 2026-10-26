"""Extreme-question performance against a comparison set and against ECI.

The market comparison set is every market question. The combined comparison
set is every ForecastBench question with a reference.

A dataset question is extreme when its shrunk base rate is in either tail. A market
question is extreme when its due-date price is in either tail. The score on every
set is the difficulty-adjusted difference from the shrunk base rate on a dataset
question and from the due-date price on a market question. Positive means the
model beat that reference.

Usage:
  python extreme_performance.py [--config config.toml]
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plots
import scoring
from _common import HERE, load_config, logger, make_output_dir
from dataset_extreme_tags import load_dataset_questions, shrunk_series_rates, tail_side
from downward_curve import canonical_runs, join_eci
from low_probability_trends import (
    fit_effects,
    load_leaderboard_rows,
    score_against_reference,
)
from tags import market_low_probability_tags

ECI_PATH = HERE / "ECI_data" / "epoch_capabilities_index" / "eci_scores.csv"
MAP_PATH = HERE / "eci_model_map.csv"
ALL_REFERENCE = "Shrunk base rate or market price"
MARKET_REFERENCE = "Market price"


def _score_column(rule: str, reference: str) -> str:
    return f"{rule}_all_question_fit_vs_{reference}"


def _population_frame(
    forecasts: pd.DataFrame,
    effects: Dict[str, Dict[str, pd.Series]],
    base_rate: pd.Series,
    question_ids: set,
    reference: str,
    log_clip: float,
) -> pd.DataFrame:
    """Score one question set and attach the run key.

    Args:
      forecasts (pd.DataFrame): Resolved leaderboard rows with row scores.
      effects (Dict[str, Dict[str, pd.Series]]): Question effects by question type.
      base_rate (pd.Series): Shrunk base rate indexed by `question_pk`.
      question_ids (set): Questions in this set.
      reference (str): Reference forecaster name.
      log_clip (float): Log-score clip.
    """
    subset = forecasts[forecasts["question_pk"].isin(question_ids)].copy()
    logger.info(f"Scoring {subset['question_pk'].nunique()} questions against {reference}.")
    results = score_against_reference(subset, effects, base_rate, reference, log_clip)
    extra = subset.groupby("model_pk")[["model_run_key", "model_organization"]].first()
    return results.join(extra)


def _pair(
    overall: pd.DataFrame,
    extreme: pd.DataFrame,
    rule: str,
    overall_reference: str,
    extreme_reference: str,
) -> pd.DataFrame:
    """One row per model with its all-question score and its extreme-question score.

    Args:
      overall (pd.DataFrame): Scores on all ForecastBench questions.
      extreme (pd.DataFrame): Scores on one extreme set.
      rule (str): "brier" or "log".
      overall_reference (str): Reference name used for `overall`.
      extreme_reference (str): Reference name used for `extreme`.
    """
    overall_col = _score_column(rule, overall_reference)
    extreme_col = _score_column(rule, extreme_reference)
    paired = pd.DataFrame(
        {
            "all_questions": overall[overall_col],
            "extreme": extreme[extreme_col],
            "n_all": overall["n_subset"],
            "n_extreme": extreme["n_subset"],
            "model": extreme["model"],
            "kind": extreme["kind"],
            "model_run_key": extreme["model_run_key"],
            "model_organization": extreme["model_organization"],
            "model_release_date": extreme["model_release_date"],
        }
    )
    paired = paired[paired["kind"].isin(["LLM", "External"])]
    return paired.dropna(subset=["all_questions", "extreme"])


def _scatter_styles() -> Dict[str, dict]:
    return {
        "LLM": dict(color=plots.SERIES[0], marker="o", s=26),
        "External": dict(facecolor="white", edgecolor=plots.SERIES[1], marker="o", s=34, lw=1.4),
    }


def extreme_vs_all_plot(
    paired: pd.DataFrame,
    rule: str,
    population: str,
    path: Path,
    x_label: str,
) -> None:
    """Scatter of comparison-set performance against extreme-question performance.

    Args:
      paired (pd.DataFrame): Rows with `all_questions`, `extreme`, `model`, and `kind`.
      rule (str): "brier" or "log".
      population (str): "market" or "combined", used in the title.
      path (Path): Output file.
      x_label (str): Axis label for the comparison set.
    """
    fig, ax = plots.plt.subplots(figsize=(6.8, 6.2))
    for kind, style in _scatter_styles().items():
        rows = paired[paired["kind"] == kind]
        if rows.empty:
            continue
        ax.scatter(rows["all_questions"], rows["extreme"], label=kind, zorder=3, alpha=0.9, **style)
    values = np.concatenate([paired["all_questions"].to_numpy(), paired["extreme"].to_numpy()])
    finite = values[np.isfinite(values)]
    pad = (finite.max() - finite.min()) * 0.06 or 0.01
    lims = [float(finite.min()) - pad, float(finite.max()) + pad]
    ax.plot(lims, lims, color=plots.INK_MUTED, lw=1, ls="--", zorder=1)
    ax.annotate(
        "equal on both", xy=(lims[1], lims[1]), fontsize=7, color=plots.INK_MUTED, ha="right"
    )
    ax.axhline(0, color=plots.INK, lw=0.9)
    ax.axvline(0, color=plots.INK, lw=0.9)
    scope = (
        "Market extreme questions"
        if population == "market"
        else "Dataset and market extreme questions"
    )
    comparison = "all market questions" if population == "market" else "all questions"
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{scope}\n(positive = better than the reference)")
    ax.set_title(
        f"Extreme vs. {comparison}, {plots.RULE_LABELS[rule]}",
        loc="left",
    )
    ax.legend(loc="upper left", fontsize=7.5)
    plots._save(fig, path)


def extreme_vs_eci_plot(
    matched: pd.DataFrame,
    rule: str,
    population: str,
    path: Path,
) -> None:
    """Scatter of Epoch ECI against extreme-question performance.

    Args:
      matched (pd.DataFrame): One row per matched run with `eci`, `extreme`, and `model`.
      rule (str): "brier" or "log".
      population (str): "market" or "combined", used in the title.
      path (Path): Output file.
    """
    fig, ax = plots.plt.subplots(figsize=(7.4, 5.4))
    ax.scatter(
        matched["eci"],
        matched["extreme"],
        s=28,
        color=plots.SERIES[0],
        zorder=3,
        label="Matched ForecastBench model",
    )
    ax.axhline(0, color=plots.INK, lw=1)
    scope = "market" if population == "market" else "dataset and market"
    ax.set_xlabel("Epoch Capabilities Index")
    ax.set_ylabel("Extreme questions: reference minus model\n(positive = better)")
    ax.set_title(
        f"Extreme {scope} questions vs. ECI, {plots.RULE_LABELS[rule]}",
        loc="left",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5)
    plots._save(fig, path)


def _load_eci() -> pd.DataFrame:
    """Return the local Epoch table with the column names `join_eci` expects."""
    eci = pd.read_csv(ECI_PATH)
    return eci.rename(
        columns={
            "eci": "ECI",
            "eci_ci_low": "ECI CI low",
            "eci_ci_high": "ECI CI high",
            "date": "Date",
        }
    )


def _corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 3:
        return float("nan")
    return float(np.corrcoef(pair["left"], pair["right"])[0, 1])


def main(config_path: Optional[str] = None) -> Path:
    """Write the extreme-question scatters.

    Args:
      config_path (Optional[str]): Config file. Defaults to `config.toml`.
    """
    config = load_config(Path(config_path) if config_path else HERE / "config.toml")
    threshold = float(config["analysis"]["threshold"])
    pct = int(round(threshold * 100))
    min_n = int(config["analysis"]["min_n"])
    out = make_output_dir(config, "extreme_performance")
    tables, figures = out / "tables", out / "figures"

    questions = load_dataset_questions(config["data"]["cache_dir"])
    questions["base_rate"] = shrunk_series_rates(questions)["shrunk_base_rate"].to_numpy()
    dataset_side = tail_side(questions["base_rate"], threshold)
    dataset_extreme = set(questions.loc[dataset_side.isin(["low", "high"]), "question_pk"])
    dataset_with_rate = set(questions.loc[questions["base_rate"].notna(), "question_pk"])
    base_rate = questions.set_index("question_pk")["base_rate"]

    forecasts = load_leaderboard_rows(
        config["data"]["cache_dir"], config["analysis"]["leaderboard"]
    )
    forecasts = scoring.add_row_scores(forecasts, config["scoring"]["log_clip"])
    market = forecasts["question_type"].eq("market")
    market_questions = forecasts.loc[market, ["question_pk", "market_value_on_due_date"]]
    market_tags = market_low_probability_tags(
        market_questions.drop_duplicates("question_pk"), thresholds=(threshold,)
    )
    market_extreme = set(market_tags.loc[market_tags[f"extreme_{pct}"], "question_pk"])
    market_ids = set(market_questions["question_pk"])
    all_ids = dataset_with_rate | market_ids
    combined_extreme = dataset_extreme | market_extreme
    logger.info(
        f"{len(dataset_extreme)} dataset and {len(market_extreme)} market questions are extreme. "
        f"{len(market_ids)} market questions in total. "
        f"{len(all_ids)} questions have a reference."
    )

    effects = {
        question_type: fit_effects(forecasts, question_type)
        for question_type in ("dataset", "market")
    }
    log_clip = config["scoring"]["log_clip"]
    overall = _population_frame(forecasts, effects, base_rate, all_ids, ALL_REFERENCE, log_clip)
    market_all = _population_frame(
        forecasts, effects, base_rate, market_ids, MARKET_REFERENCE, log_clip
    )
    populations = {
        "combined": _population_frame(
            forecasts, effects, base_rate, combined_extreme, ALL_REFERENCE, log_clip
        ),
        "market": _population_frame(
            forecasts, effects, base_rate, market_extreme, MARKET_REFERENCE, log_clip
        ),
    }
    baselines = {
        "combined": (
            overall,
            ALL_REFERENCE,
            "All ForecastBench questions\n(positive = better than the reference)",
        ),
        "market": (
            market_all,
            MARKET_REFERENCE,
            "All market questions\n(positive = better than the market)",
        ),
    }
    mapping = pd.read_csv(MAP_PATH)
    eci = _load_eci()
    lines = [
        f"# Extreme-question performance, {pct}% tails",
        "",
        "A dataset question is extreme when the shrunk base rate is in either tail.",
        "A market question is extreme when the due-date price is in either tail.",
        "Positive scores are better than the shrunk base rate on a dataset question",
        "and better than the due-date price on a market question.",
        "",
        f"- Dataset extreme questions: {len(dataset_extreme)}.",
        f"- Market extreme questions: {len(market_extreme)}.",
        f"- All market questions: {len(market_ids)}.",
        f"- All questions with a reference: {len(all_ids)}.",
        "The market scatter compares extreme market questions with every market",
        "question. The combined scatter compares extreme dataset and market",
        "questions with every question that has a reference.",
        f"- Models need at least {min_n} questions on each axis.",
        "",
    ]
    for population, extreme in populations.items():
        lines.append(f"## {population}")
        lines.append("")
        reference = ALL_REFERENCE if population == "combined" else MARKET_REFERENCE
        baseline, baseline_reference, x_label = baselines[population]
        comparison_name = "all-market" if population == "market" else "all-question"
        for rule in scoring.SCORE_RULES:
            paired = _pair(baseline, extreme, rule, baseline_reference, reference)
            paired = paired[(paired["n_extreme"] >= min_n) & (paired["n_all"] >= min_n)]
            paired.to_csv(tables / f"extreme_vs_all_{population}_{rule}.csv")
            extreme_vs_all_plot(
                paired,
                rule,
                population,
                figures / f"extreme_vs_all_{population}_{rule}.png",
                x_label,
            )
            runs = canonical_runs(paired.rename(columns={"n_extreme": "n_subset"}))
            matched, unmatched, _, _ = join_eci(runs, eci, mapping)
            matched = matched.dropna(subset=["eci"])
            matched.to_csv(tables / f"extreme_vs_eci_{population}_{rule}.csv", index=False)
            extreme_vs_eci_plot(
                matched,
                rule,
                population,
                figures / f"extreme_vs_eci_{population}_{rule}.png",
            )
            lines.append(
                f"- {plots.RULE_LABELS[rule]}: {len(paired)} models. "
                f"Correlation of {comparison_name} and extreme scores: "
                f"{_corr(paired['all_questions'], paired['extreme']):+.2f}. "
                f"{len(matched)} matched to ECI "
                f"({len(unmatched)} ForecastBench runs unmatched). "
                f"Correlation with ECI: {_corr(matched['eci'], matched['extreme']):+.2f}."
            )
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    logger.info(f"Done. Outputs in {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
