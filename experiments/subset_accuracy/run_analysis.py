"""Score the market low-probability subset and draw the initial figures.

Reads the cache from `build_cache.py` and writes tables and figures to
`outputs/<timestamp>_<leaderboard>_<subset><pct>/`. Dataset questions are not analyzed here.

Usage:
  python run_analysis.py [--config config.toml] [--n-bootstrap N]
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plots
import scoring
from _common import load_config, logger, make_output_dir
from comparisons import (
    HUMAN_COMPARISONS,
    MARKET_PRICE,
    base_rate_table,
    build_comparison_rows,
    is_comparison,
    resolve_comparison_names,
    template_rows,
)
from tags import (
    TAG_THRESHOLDS,
    flip_onto_low_side,
    market_low_probability_tags,
    subset_columns,
)

META_COLS = [
    "model",
    "organization",
    "model_organization",
    "external_submission",
    "forecastbench_llm",
    "model_release_date",
    "first_forecast_due_date",
]
QUESTION_COLS = [
    "question_pk",
    "forecast_due_date",
    "source",
    "id",
    "side",
    "market_value_on_due_date",
    "resolved_to",
    "resolution_date",
    "question",
    "url",
    "freeze_datetime_value",
    "in_human_question_set",
]


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------


def load_market_rows(config: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Return scored market rows for the chosen leaderboard, plus the all-rounds market table.

    Args:
      config (Dict[str, Any]): Loaded config.
    """
    cache = Path(config["data"]["cache_dir"])
    df = pd.read_parquet(cache / "forecasts.parquet", filters=[("question_type", "==", "market")])
    markets = pd.read_parquet(cache / "market_questions.parquet")
    notes = {}

    leaderboard = config["analysis"]["leaderboard"]
    if leaderboard not in ("tournament", "baseline"):
        raise ValueError("analysis.leaderboard must be 'tournament' or 'baseline'.")
    df = df[df[f"{leaderboard}_model"].astype(bool)].copy()

    dates = pd.to_datetime(df["forecast_due_date"])
    if config["analysis"].get("min_forecast_due_date"):
        df = df[dates >= pd.Timestamp(config["analysis"]["min_forecast_due_date"])]
    if config["analysis"].get("max_forecast_due_date"):
        df = df[dates <= pd.Timestamp(config["analysis"]["max_forecast_due_date"])]

    voided = df["resolved_to"].isna()
    notes["rows_without_outcome_dropped"] = int(voided.sum())
    notes["questions_without_outcome_dropped"] = int(df.loc[voided, "question_pk"].nunique())
    df = df[~voided].reset_index(drop=True)
    if "question" not in df.columns:
        df["question"] = pd.NA
        df["url"] = pd.NA
        df["freeze_datetime_value"] = np.nan
        df["in_human_question_set"] = False
    df = scoring.add_row_scores(df, config["scoring"]["log_clip"])
    return df, markets, notes


# ----------------------------------------------------------------------------------------------
# Scoring the subset
# ----------------------------------------------------------------------------------------------


def evaluate(
    subset: pd.DataFrame,
    fit_rows: pd.DataFrame,
    comparisons: List[str],
    config: Dict[str, Any],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Score every forecaster on the subset under each rule and difficulty method.

    Args:
      subset (pd.DataFrame): Subset rows with `brier`, `log`, `brier_adj` and `log_adj` (from
        the all-question fit), including comparison rows.
      fit_rows (pd.DataFrame): Subset rows of leaderboard models, used for the subset-only fit.
      comparisons (List[str]): Comparison `model_pk`s.
      config (Dict[str, Any]): Loaded config.
      rng (np.random.Generator): Random generator.
      n_bootstrap (int): Bootstrap replicates (0 for point estimates only).

    Returns:
      tuple: One row per forecaster with every score column, and the bootstrap draws of each
        rule's all-question-fit mean score.
    """
    questions = (
        subset[["question_pk", "forecast_due_date"]]
        .drop_duplicates("question_pk")
        .set_index("question_pk")
    )
    weights = (
        scoring.bootstrap_weights(questions["forecast_due_date"].astype(str), n_bootstrap, rng)
        if n_bootstrap
        else None
    )
    subset = subset.copy()
    tables, draws = [], {}
    for rule in scoring.SCORE_RULES:
        table, rule_draws = scoring.score_table(
            subset,
            f"{rule}_adj",
            comparisons,
            weights,
            questions.index,
            paired=False,
            label=f"{rule}_all_question_fit",
        )
        tables.append(table)
        draws[rule] = rule_draws

        fe_subset = scoring.question_effects(fit_rows, rule)
        subset[f"{rule}_adj_subset"] = subset[rule] - subset["question_pk"].map(fe_subset)
        table, _ = scoring.score_table(
            subset,
            f"{rule}_adj_subset",
            comparisons,
            None,
            questions.index,
            paired=False,
            label=f"{rule}_subset_fit",
        )
        if config["scoring"]["bootstrap_subset_fit"] and n_bootstrap:
            ci = scoring.subset_fit_bootstrap(subset, fit_rows, rule, comparisons, n_bootstrap, rng)
            table = table.drop(columns=[c for c in ci.columns if c in table.columns]).join(ci)
        tables.append(table)

        table, _ = scoring.score_table(
            subset,
            rule,
            comparisons,
            weights,
            questions.index,
            paired=True,
            label=f"{rule}_none",
        )
        tables.append(table)
    return pd.concat(tables, axis=1), draws


def describe_forecasters(subset: pd.DataFrame) -> pd.DataFrame:
    """Return per-forecaster counts, forecast extremity, realized yes rate and AUC.

    Args:
      subset (pd.DataFrame): Subset rows with flipped columns.
    """
    meta = subset.groupby("model_pk")[META_COLS].first()
    stats = subset.groupby("model_pk").agg(
        n_subset=("question_pk", "nunique"),
        n_low=("side", lambda s: int((s == "low").sum())),
        n_high=("side", lambda s: int((s == "high").sum())),
        mean_forecast_f=("forecast_f", "mean"),
        median_forecast_f=("forecast_f", "median"),
        realized_yes_rate_f=("outcome_f", "mean"),
        n_rounds=("forecast_due_date", "nunique"),
    )
    stats["auc"] = subset.groupby("model_pk").apply(
        lambda g: scoring.auc(g["forecast_f"], g["outcome_f"]), include_groups=False
    )
    out = meta.join(stats)
    out["kind"] = np.select(
        [is_comparison(out["model"]), out["external_submission"].astype(bool)],
        ["Comparison", "External"],
        default="LLM",
    )
    return out


def frontier_with_band(
    results: pd.DataFrame,
    draws: pd.DataFrame,
    llm_pks: List[str],
    reference: str,
    col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the SOTA frontier (point estimate) and its bootstrap band over release dates.

    Args:
      results (pd.DataFrame): Score table with `model`, `model_release_date` and `col`.
      draws (pd.DataFrame): Bootstrap draws of mean scores (replicates x forecasters).
      llm_pks (List[str]): LLMs eligible for the frontier.
      reference (str): Reference forecaster `model_pk`.
      col (str): Difference column (positive = better than the reference).
    """
    from _common import leaderboard

    llms = results.loc[llm_pks].dropna(subset=["model_release_date", col])
    llms = llms.sort_values("model_release_date")
    frontier_input = llms.reset_index().rename(columns={"index": "model_pk"})
    frontier_input = frontier_input[["model_pk", "model", "model_release_date", col]]
    frontier = leaderboard().find_sota_models(frontier_input, col)

    band = pd.DataFrame()
    if draws is not None and reference in draws.columns and len(llms):
        diff = draws[[reference]].to_numpy() - draws[llms.index].to_numpy()
        running_best = np.fmax.accumulate(diff, axis=1)
        band = pd.DataFrame(
            {
                "model_release_date": llms["model_release_date"].to_numpy(),
                "lower": np.nanpercentile(running_best, 2.5, axis=0),
                "upper": np.nanpercentile(running_best, 97.5, axis=0),
            }
        ).drop_duplicates("model_release_date", keep="last")
    return frontier, band


# ----------------------------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------------------------


def round_table(
    questions: pd.DataFrame,
    markets: pd.DataFrame,
    threshold: float,
    subset_name: str,
    leaderboard_rounds: List,
) -> pd.DataFrame:
    """Per-round counts, surprise rate and resolved share for the subset.

    Args:
      questions (pd.DataFrame): Scored subset questions (flipped `outcome_f`).
      markets (pd.DataFrame): All market questions per round, resolved or not.
      threshold (float): Tail cutoff.
      subset_name (str): "extreme", "low" or "high".
      leaderboard_rounds (List): Rounds that pass the leaderboard filters.
    """
    markets = markets.copy()
    markets["question_pk"] = (
        markets["forecast_due_date"].astype(str) + "_" + markets["source"] + "_" + markets["id"]
    )
    tags = market_low_probability_tags(markets, thresholds=TAG_THRESHOLDS)
    columns = subset_columns(subset_name, threshold)
    markets = markets.merge(tags[["question_pk", columns["mask"]]], on="question_pk")
    tagged = markets[markets[columns["mask"]]]
    resolved_before_due = pd.to_datetime(tagged["resolution_date"]) <= pd.to_datetime(
        tagged["forecast_due_date"]
    )
    voided = tagged["resolved"].astype(bool) & resolved_before_due
    scorable = tagged[~voided]
    resolved = scorable["resolved"].astype(bool) & scorable["resolved_to"].notna()
    per_round = (
        scorable.assign(_resolved=resolved)
        .groupby("forecast_due_date")
        .agg(n_tagged=("question_pk", "nunique"), n_resolved_any=("_resolved", "sum"))
    )
    per_round["resolved_share"] = per_round["n_resolved_any"] / per_round["n_tagged"]
    per_round["n_voided"] = voided.groupby(tagged["forecast_due_date"]).sum()

    scored = questions.groupby("forecast_due_date").agg(
        n_resolved=("question_pk", "nunique"),
        n_low=("side", lambda s: int((s == "low").sum())),
        n_high=("side", lambda s: int((s == "high").sum())),
        n_surprises=("outcome_f", "sum"),
    )
    out = per_round.join(scored, how="outer").reset_index()
    out[["n_resolved", "n_surprises"]] = out[["n_resolved", "n_surprises"]].fillna(0)
    out["yes_rate"] = out["n_surprises"] / out["n_resolved"].replace(0, np.nan)
    intervals = [
        scoring.wilson_interval(s, n) for s, n in zip(out["n_surprises"], out["n_resolved"])
    ]
    out["ci_lower"] = [lo for lo, _ in intervals]
    out["ci_upper"] = [hi for _, hi in intervals]
    out["in_leaderboard"] = out["forecast_due_date"].isin(leaderboard_rounds)
    return out.sort_values("forecast_due_date")


def calibration_table(subset: pd.DataFrame, forecasters: Dict[str, List[str]]) -> pd.DataFrame:
    """Binned tail calibration on the flipped scale.

    Args:
      subset (pd.DataFrame): Subset rows with `forecast_f` and `outcome_f`.
      forecasters (Dict[str, List[str]]): Display name to the `model_pk`s pooled under it.
    """
    edges = [0, 0.01, 0.02, 0.05, 0.10, 0.25, 1.0001]
    labels = ["<1%", "1-2%", "2-5%", "5-10%", "10-25%", "25%+"]
    rows = []
    for name, pks in forecasters.items():
        g = subset[subset["model_pk"].isin(pks)].copy()
        if g.empty:
            continue
        g["bin"] = pd.cut(g["forecast_f"], edges, labels=labels, right=False)
        for label, b in g.groupby("bin", observed=True):
            lo, hi = scoring.wilson_interval(b["outcome_f"].sum(), len(b))
            rows.append(
                {
                    "forecaster": name,
                    "bin": label,
                    "n": len(b),
                    "mean_forecast": b["forecast_f"].mean(),
                    "yes_rate": b["outcome_f"].mean(),
                    "ci_lower": lo,
                    "ci_upper": hi,
                }
            )
    return pd.DataFrame(rows)


def surprises_table(
    subset: pd.DataFrame,
    questions: pd.DataFrame,
    best_llms: List[str],
    llm_pks: List[str],
    human_pks: Dict[str, str],
) -> pd.DataFrame:
    """Subset questions that resolved against the market's direction, most surprising first.

    Probabilities are on the flipped scale: the probability each forecaster gave the outcome
    that happened.

    Args:
      subset (pd.DataFrame): Subset rows with `forecast_f`.
      questions (pd.DataFrame): One row per subset question.
      best_llms (List[str]): `model_pk`s of the best LLMs by headline score.
      llm_pks (List[str]): All evaluated LLMs.
      human_pks (Dict[str, str]): Human comparison name to `model_pk`.
    """
    surprised = questions[questions["outcome_f"] == 1].copy()
    forecasts = subset[subset["question_pk"].isin(surprised["question_pk"])]
    wide = forecasts.pivot_table(
        index="question_pk", columns="model_pk", values="forecast_f", aggfunc="mean"
    )
    out = surprised.set_index("question_pk")[
        [
            "forecast_due_date",
            "source",
            "side",
            "price_f",
            "freeze_value_f",
            "resolution_date",
            "question",
            "url",
        ]
    ].rename(
        columns={
            "price_f": "market_p_outcome",
            "freeze_value_f": "freeze_price_shown_to_llms_p_outcome",
        }
    )
    for name, pk in human_pks.items():
        if pk in wide.columns:
            out[f"{name} p_outcome"] = wide[pk]
    llm_cols = [pk for pk in llm_pks if pk in wide.columns]
    out["llm_median_p_outcome"] = wide[llm_cols].median(axis=1)
    out["llm_max_p_outcome"] = wide[llm_cols].max(axis=1)
    out["n_llms"] = wide[llm_cols].notna().sum(axis=1)
    for pk in best_llms:
        if pk in wide.columns:
            out[f"{subset.loc[subset['model_pk'] == pk, 'model'].iloc[0]} p_outcome"] = wide[pk]
    return out.sort_values("market_p_outcome").reset_index()


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------


def main(config_path: str, n_bootstrap_override: int | None) -> Path:
    """Run the market low-probability analysis end to end.

    Args:
      config_path (str): Config file.
      n_bootstrap_override (int | None): Replicates, overriding the config.
    """
    config = load_config(Path(config_path))
    a_cfg, s_cfg, c_cfg, p_cfg = (
        config["analysis"],
        config["scoring"],
        config["comparisons"],
        config["plots"],
    )
    n_bootstrap = n_bootstrap_override if n_bootstrap_override is not None else s_cfg["n_bootstrap"]
    rng = np.random.default_rng(s_cfg["seed"])
    pct = int(round(a_cfg["threshold"] * 100))
    out = make_output_dir(config, f"{a_cfg['leaderboard']}_{a_cfg['subset']}{pct}")
    tables, figures = out / "tables", out / "figures"
    summary: Dict[str, Any] = {"n_bootstrap": n_bootstrap}

    # 1. Load, tag, and fit question effects on all market questions.
    df, markets, notes = load_market_rows(config)
    summary.update(notes)
    leaderboard_rounds = sorted(df["forecast_due_date"].unique())
    tags = market_low_probability_tags(df)
    tags.to_csv(tables / "market_tags.csv", index=False)
    columns = subset_columns(a_cfg["subset"], a_cfg["threshold"])
    tag_cols = list(dict.fromkeys([columns["mask"], columns["side"], f"extreme_{pct}"]))
    df = df.merge(tags[["question_pk", *tag_cols]], on="question_pk", how="left")
    df["in_subset"] = df[columns["mask"]].astype(bool)
    df["side"] = df[columns["side"]]
    df = df.drop(columns=tag_cols)

    effects = {rule: scoring.question_effects(df, rule) for rule in scoring.SCORE_RULES}
    df = scoring.adjust(df, effects)
    logger.info(f"Fitted question effects on {df['question_pk'].nunique()} market questions.")

    # 2. Subset rows and the comparison forecasters.
    subset = df[df["in_subset"]].copy()
    questions = subset[QUESTION_COLS].drop_duplicates("question_pk").reset_index(drop=True)
    questions = flip_onto_low_side(questions.assign(forecast=np.nan))
    high = questions["side"] == "high"
    freeze = questions["freeze_datetime_value"]
    questions["freeze_value_f"] = freeze.where(~high, 1 - freeze)
    base_rates = base_rate_table(questions, c_cfg["base_rate_min_n"])
    base_rates.to_csv(tables / "base_rates.csv", index=False)

    synthetic = build_comparison_rows(questions, base_rates)
    synthetic = scoring.adjust(scoring.add_row_scores(synthetic, s_cfg["log_clip"]), effects)
    fit_rows = subset.copy()
    subset = pd.concat([subset.assign(synthetic_comparison=False), synthetic], ignore_index=True)
    subset = flip_onto_low_side(subset)

    available = subset.groupby("model_pk")["model"].first()
    primary = resolve_comparison_names(c_cfg["primary"], available)
    secondary = resolve_comparison_names(c_cfg["secondary"], available)
    comparison_pks = list(dict.fromkeys([*primary.values(), *secondary.values()]))
    # The trend plots are drawn relative to this forecaster; columns are keyed by `model_pk`.
    reference = {**secondary, **primary}.get(c_cfg["reference"], c_cfg["reference"])
    missing = [n for n in c_cfg["primary"] if n not in primary]
    if missing:
        logger.warning(f"Primary comparisons not found on the subset: {missing}")
    summary["comparisons_missing"] = missing

    # 3. Scores.
    results, draws = evaluate(subset, fit_rows, comparison_pks, config, rng, n_bootstrap)
    results = describe_forecasters(subset).join(results)
    headline = s_cfg["headline_difficulty"]
    sort_col = f"brier_{headline}_vs_{reference}"
    results = results.sort_values(sort_col, ascending=False)
    results.index.name = "model_pk"
    results.to_csv(tables / "subset_scores.csv")

    keep = results["n_subset"] >= a_cfg["min_n"]
    dropped = results.loc[~keep & (results["kind"] != "Comparison"), ["model", "n_subset"]]
    dropped.to_csv(tables / "models_below_min_n.csv")
    plotted = results[keep | (results["kind"] == "Comparison")]
    llm_pks = plotted.index[plotted["kind"] == "LLM"].tolist()
    ext_pks = plotted.index[plotted["kind"] == "External"].tolist()

    # 4. Counts report.
    rounds = round_table(
        questions, markets, a_cfg["threshold"], a_cfg["subset"], leaderboard_rounds
    )
    rounds.to_csv(tables / "counts_by_round.csv", index=False)
    by_source = questions.groupby(["source", "side"]).agg(
        n_questions=("question_pk", "nunique"), surprise_rate=("outcome_f", "mean")
    )
    by_source.to_csv(tables / "counts_by_source.csv")
    models_per_question = (
        subset[~subset["synthetic_comparison"]].groupby("question_pk")["model_pk"].nunique()
    )
    summary["models_per_question"] = models_per_question.describe().round(1).to_dict()
    summary["n_subset_questions"] = int(questions["question_pk"].nunique())
    summary["n_subset_low"] = int((questions["side"] == "low").sum())
    summary["n_subset_high"] = int((questions["side"] == "high").sum())
    summary["n_surprises"] = int(questions["outcome_f"].sum())
    summary["surprise_rate"] = float(questions["outcome_f"].mean())
    summary["n_llms_plotted"] = len(llm_pks)
    summary["n_models_below_min_n"] = len(dropped)

    # 5. Human comparisons: headline line or caveated point.
    human_pks = resolve_comparison_names(list(HUMAN_COMPARISONS), available)
    human = {}
    for name in HUMAN_COMPARISONS:
        pk = human_pks.get(name)
        n = int(results.loc[pk, "n_subset"]) if pk in results.index else 0
        human[name] = {
            "n_subset_questions": n,
            "decision": "headline line" if n >= a_cfg["min_n_human"] else "caveated point",
        }
    summary["human_comparisons"] = human

    # 6. Figures for both rules.
    for rule in scoring.SCORE_RULES:
        col = f"{rule}_{headline}_vs_{reference}"
        frontier, _ = frontier_with_band(results, draws.get(rule), llm_pks, reference, col)
        frontier.to_csv(tables / f"sota_frontier_{rule}.csv", index=False)
        plots.monthly_llm_trend(
            results.loc[llm_pks], col, p_cfg.get("min_release_date") or None
        ).to_csv(tables / f"monthly_llm_trend_{rule}.csv", index=False)
        references = {}
        # Primary comparisons are drawn as reference lines; secondary ones are in the tables.
        for name, pk in primary.items():
            if pk == reference or pk not in results.index:
                continue
            label = name
            if name in human:
                label = f"{name} (N={human[name]['n_subset_questions']})" + (
                    "" if human[name]["decision"] == "headline line" else ", caveat: small N"
                )
            references[label] = results.loc[pk, col]
        plots.trend_plot(
            llms=results.loc[llm_pks],
            externals=results.loc[ext_pks],
            references=references,
            col=col,
            reference_name=reference,
            rule=rule,
            title=(
                f"{a_cfg['leaderboard'].title()} leaderboard, market questions priced "
                f"{'≤' if a_cfg['subset'] != 'high' else '≥'}{pct}%"
                f"{' or ≥' + str(100 - pct) + '% (flipped)' if a_cfg['subset'] == 'extreme' else ''}"
                f": {plots.RULE_LABELS[rule]}"
            ),
            path=figures / f"trend_release_date_{rule}.png",
            min_date=p_cfg.get("min_release_date") or None,
            y_min=p_cfg.get(f"{rule}_y_min", p_cfg.get("y_min")),
            y_max=p_cfg.get(f"{rule}_y_max", p_cfg.get("y_max")),
        )

    # 7. Extreme vs. non-extreme (both use the all-question fit).
    all_questions = df.drop_duplicates("question_pk")[
        ["question_pk", "forecast_due_date", "market_value_on_due_date", "resolved_to"]
    ]
    market_rows = template_rows(all_questions, MARKET_PRICE)
    market_rows["forecast"] = market_rows["market_value_on_due_date"]
    market_rows = scoring.adjust(scoring.add_row_scores(market_rows, s_cfg["log_clip"]), effects)
    with_market = pd.concat([df, market_rows], ignore_index=True)
    is_extreme = with_market["question_pk"].isin(tags.loc[tags[f"extreme_{pct}"], "question_pk"])
    for rule in scoring.SCORE_RULES:
        means = with_market.groupby([is_extreme.rename("extreme"), "model_pk"])[
            f"{rule}_adj"
        ].mean()
        ext_means, non_means = means.loc[True], means.loc[False]
        scatter = pd.DataFrame(
            {
                "non_extreme": non_means.get(MARKET_PRICE, np.nan) - non_means,
                "extreme": ext_means.get(MARKET_PRICE, np.nan) - ext_means,
            }
        ).join(results[["model", "kind", "n_subset"]], how="inner")
        scatter = scatter[scatter.index.isin([*llm_pks, *ext_pks, *comparison_pks])]
        scatter.to_csv(tables / f"extreme_vs_nonextreme_{rule}.csv")
        plots.extreme_vs_nonextreme_plot(
            scatter.dropna(subset=["non_extreme", "extreme"]),
            "non_extreme",
            "extreme",
            rule,
            MARKET_PRICE,
            p_cfg["n_labels"],
            figures / f"extreme_vs_nonextreme_{rule}.png",
        )

    # 8. Calibration and the surprises table.
    best_llms = (
        results.loc[llm_pks]
        .sort_values(sort_col, ascending=False)
        .head(p_cfg["n_calibration_models"])
        .index.tolist()
    )
    calibration_groups = {MARKET_PRICE: [MARKET_PRICE]}
    for name, pk in human_pks.items():
        if pk in results.index:
            calibration_groups[name] = [pk]
    calibration_groups["All ForecastBench LLMs (pooled)"] = llm_pks
    for pk in best_llms:
        calibration_groups[results.loc[pk, "model"]] = [pk]
    calibration = calibration_table(subset, calibration_groups)
    calibration.to_csv(tables / "calibration.csv", index=False)
    plots.calibration_plot(calibration, list(calibration_groups), figures / "calibration.png")

    surprises = surprises_table(subset, questions, best_llms[:3], llm_pks, human_pks)
    surprises.head(p_cfg["n_surprises"]).to_csv(tables / "surprises.csv", index=False)
    surprises.head(p_cfg["n_surprises"]).to_html(
        tables / "surprises.html",
        index=False,
        float_format=lambda x: f"{x:.3f}",
        render_links=True,
    )

    # 9. Downward-curve checks.
    min_share = config["checks"]["min_resolved_share"]
    good_rounds = rounds.loc[
        rounds["in_leaderboard"] & (rounds["resolved_share"] >= min_share), "forecast_due_date"
    ].tolist()
    summary["rounds_meeting_resolved_share"] = [str(d) for d in good_rounds]
    headline_col = f"brier_{headline}_vs_{reference}"
    frontier_all, _ = frontier_with_band(results, None, llm_pks, reference, headline_col)
    restricted = {"All rounds": frontier_all}
    sub_good = subset[subset["forecast_due_date"].isin(good_rounds)]
    if not sub_good.empty and sub_good["model_pk"].isin(llm_pks).any():
        fit_good = fit_rows[fit_rows["forecast_due_date"].isin(good_rounds)]
        res_good, _ = evaluate(sub_good, fit_good, comparison_pks, config, rng, 0)
        res_good = describe_forecasters(sub_good).join(res_good)
        res_good.to_csv(tables / "subset_scores_well_resolved_rounds.csv")
        good_llms = [
            pk for pk in llm_pks if pk in res_good.index and res_good.loc[pk, "n_subset"] >= 1
        ]
        frontier_good, _ = frontier_with_band(res_good, None, good_llms, reference, headline_col)
        restricted[f"Rounds ≥{min_share:.0%} resolved ({len(good_rounds)})"] = frontier_good
    extremity = results.loc[llm_pks, ["model", "model_release_date", "median_forecast_f"]]
    extremity.to_csv(tables / "llm_forecast_extremity.csv")
    plots.downward_curve_checks_plot(
        rounds=rounds,
        extremity=extremity.dropna(subset=["model_release_date"]),
        market_median=float(questions["price_f"].median()),
        restricted=restricted,
        col=headline_col,
        min_resolved_share=min_share,
        path=figures / "downward_curve_checks.png",
    )
    summary["downward_curve"] = downward_curve_notes(rounds, extremity, restricted, headline_col)

    write_summary(out, config, summary, results, llm_pks, reference, headline)
    logger.info(f"Done. Outputs in {out}")
    return out


def downward_curve_notes(
    rounds: pd.DataFrame,
    extremity: pd.DataFrame,
    restricted: Dict[str, pd.DataFrame],
    col: str,
) -> Dict[str, Any]:
    """Return the numbers behind the four downward-curve checks.

    Args:
      rounds (pd.DataFrame): Output of `round_table`.
      extremity (pd.DataFrame): Median flipped forecast per LLM.
      restricted (Dict[str, pd.DataFrame]): Frontiers by round filter.
      col (str): Frontier column.
    """
    scored = rounds[rounds["in_leaderboard"] & (rounds["n_resolved"] > 0)].copy()
    order = np.arange(len(scored))
    notes: Dict[str, Any] = {}
    if len(scored) >= 3:
        notes["corr_round_order_vs_surprise_rate"] = float(
            np.corrcoef(order, scored["yes_rate"])[0, 1]
        )
        notes["corr_resolved_share_vs_surprise_rate"] = float(
            np.corrcoef(scored["resolved_share"], scored["yes_rate"])[0, 1]
        )
        half = len(scored) // 2
        early, late = scored.iloc[:half], scored.iloc[half:]
        notes["surprise_rate_early_rounds"] = float(
            early["n_surprises"].sum() / early["n_resolved"].sum()
        )
        notes["surprise_rate_late_rounds"] = float(
            late["n_surprises"].sum() / late["n_resolved"].sum()
        )
        notes["resolved_share_early_rounds"] = float(early["resolved_share"].mean())
        notes["resolved_share_late_rounds"] = float(late["resolved_share"].mean())
    ext = extremity.dropna(subset=["model_release_date", "median_forecast_f"])
    if len(ext) >= 3:
        days = pd.to_datetime(ext["model_release_date"]).map(pd.Timestamp.toordinal)
        notes["corr_release_date_vs_median_forecast_f"] = float(
            np.corrcoef(days, ext["median_forecast_f"])[0, 1]
        )
    for label, frontier in restricted.items():
        if len(frontier):
            notes[f"frontier_first_to_last[{label}]"] = [
                float(frontier[col].iloc[0]),
                float(frontier[col].iloc[-1]),
            ]
    return notes


def write_summary(
    out: Path,
    config: Dict[str, Any],
    summary: Dict[str, Any],
    results: pd.DataFrame,
    llm_pks: List[str],
    reference: str,
    headline: str,
) -> None:
    """Write `summary.md` with the headline numbers and a pointer to each output.

    Args:
      out (Path): Output folder.
      config (Dict[str, Any]): Loaded config.
      summary (Dict[str, Any]): Collected numbers.
      results (pd.DataFrame): Score table.
      llm_pks (List[str]): Plotted LLMs.
      reference (str): Reference comparison.
      headline (str): Headline difficulty method.
    """
    a_cfg = config["analysis"]
    lines = [
        f"# Market low-probability analysis: {a_cfg['leaderboard']} leaderboard",
        "",
        f"Subset: `{a_cfg['subset']}` at {a_cfg['threshold']:.0%}. Scores are difficulty-adjusted "
        f"with `{headline}` and shown as comparison minus model (positive = model better).",
        "",
        "## Counts",
        f"- Subset questions: {summary['n_subset_questions']} "
        f"(low side {summary['n_subset_low']}, high side {summary['n_subset_high']})",
        f"- Surprises (resolved against the price): {summary['n_surprises']} "
        f"({summary['surprise_rate']:.1%})",
        f"- Rows with no outcome dropped (voided or non-binary): "
        f"{summary['rows_without_outcome_dropped']} "
        f"({summary['questions_without_outcome_dropped']} questions)",
        f"- LLMs plotted (N ≥ {a_cfg['min_n']}): {summary['n_llms_plotted']}; "
        f"below the minimum: {summary['n_models_below_min_n']} (see `tables/models_below_min_n.csv`)",
        f"- Models per subset question: {summary['models_per_question']}",
        f"- Bootstrap replicates: {summary['n_bootstrap']}",
        "",
        "## Human comparisons",
    ]
    for name, info in summary["human_comparisons"].items():
        lines.append(f"- {name}: N = {info['n_subset_questions']} → {info['decision']}")
    if summary["comparisons_missing"]:
        lines.append(f"- Missing comparisons: {summary['comparisons_missing']}")

    lines += ["", f"## Top LLMs vs. {reference}", ""]
    for rule in scoring.SCORE_RULES:
        col = f"{rule}_{headline}_vs_{reference}"
        top = results.loc[llm_pks].sort_values(col, ascending=False).head(10)
        lines.append(f"**{plots.RULE_LABELS[rule]}**")
        lines.append("")
        lines.append("| Model | N | Diff | 95% CI | P(better) | AUC |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in top.iterrows():
            lines.append(
                f"| {row['model']} | {int(row['n_subset'])} | {row[col]:.4f} | "
                f"[{row.get(col + '_ci_lower', np.nan):.4f}, "
                f"{row.get(col + '_ci_upper', np.nan):.4f}] | "
                f"{row.get(col + '_p_better', np.nan):.2f} | {row['auc']:.2f} |"
            )
        lines.append("")

    lines += ["## Downward-curve checks", ""]
    for key, value in summary["downward_curve"].items():
        lines.append(f"- {key}: {value}")
    lines.append(
        f"- Rounds with ≥{config['checks']['min_resolved_share']:.0%} of the subset resolved: "
        f"{summary['rounds_meeting_resolved_share']}"
    )
    lines += [
        "",
        "Reading: if late rounds show a higher surprise rate and a lower resolved share than "
        "early rounds, and the frontier flattens when restricted to well-resolved rounds, the "
        "downward curve is at least partly the resolution-timing effect.",
        "",
        "## Outputs",
        "- `tables/subset_scores.csv`: every forecaster, both rules, all three difficulty methods",
        "- `tables/counts_by_round.csv`, `counts_by_source.csv`: counts report",
        "- `tables/base_rates.csv`: out-of-sample base rates (with fallback flag)",
        "- `tables/surprises.csv|html`: questions that resolved against the price",
        "- `tables/calibration.csv`, `figures/calibration.png`",
        "- `tables/monthly_llm_trend_{brier,log}.csv`: ForecastBench median by release month",
        "- `figures/trend_release_date_{brier,log}.png`",
        "- `figures/extreme_vs_nonextreme_{brier,log}.png`",
        "- `figures/downward_curve_checks.png`",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.toml"))
    parser.add_argument("--n-bootstrap", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.n_bootstrap)
