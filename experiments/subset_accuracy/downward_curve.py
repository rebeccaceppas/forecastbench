"""SOTA frontier and downward-curve checks on low-probability market questions.

Plan section 4.4. The subset is market questions whose price on the due date is at or
below the configured threshold, whatever `analysis.subset` says. Scores are
difficulty-adjusted differences versus the market, so positive means the model did better.

The running-best frontier reuses `find_sota_models`. It can only stay flat or step up on
a fixed question set, so the figures also show every ForecastBench LLM. A downward curve
is a decline in those points, not in the step itself.

Epoch's published ECI file is downloaded on each run and joined through `eci_model_map.csv`.

Usage:
  python downward_curve.py [--config config.toml]
"""

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plots
import scoring
from _common import HERE, load_config, logger, make_output_dir
from comparisons import (
    MARKET_PRICE,
    base_rate_table,
    build_comparison_rows,
    resolve_comparison_names,
    template_rows,
)
from run_analysis import (
    QUESTION_COLS,
    describe_forecasters,
    downward_curve_notes,
    evaluate,
    frontier_with_band,
    load_market_rows,
    round_table,
)
from tags import flip_onto_low_side, market_low_probability_tags, subset_columns

ECI_URL = "https://epoch.ai/data/charts/ai-capabilities-progress-has-sped-up/eci_scores.csv"
MAP_PATH = HERE / "eci_model_map.csv"
ECI_CACHE = HERE / "data" / "eci_scores.csv"


def _fetch_eci_bytes() -> bytes:
    """Return the Epoch ECI CSV, using curl when Python's certificate store rejects the site.

    Returns:
      bytes: File contents.
    """
    request = urllib.request.Request(
        ECI_URL, headers={"User-Agent": "forecastbench-subset-accuracy"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            logger.info(f"Downloaded Epoch ECI scores from {ECI_URL}")
            return response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.warning(f"urllib could not download Epoch ECI scores ({exc}); trying curl")
    try:
        payload = subprocess.check_output(
            ["curl", "-fsSL", "-A", "forecastbench-subset-accuracy", ECI_URL],
            timeout=60,
        )
        logger.info(f"Downloaded Epoch ECI scores with curl from {ECI_URL}")
        return payload
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        if ECI_CACHE.exists():
            logger.warning(f"ECI download failed ({exc}); using {ECI_CACHE}")
            return ECI_CACHE.read_bytes()
        raise RuntimeError(f"Could not download Epoch ECI scores from {ECI_URL}") from exc


def download_eci(dest: Path) -> pd.DataFrame:
    """Download Epoch's ECI table, falling back to `data/eci_scores.csv` if the request fails.

    Args:
      dest (Path): Where to write the downloaded file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = _fetch_eci_bytes()
    dest.write_bytes(payload)
    ECI_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ECI_CACHE.write_bytes(payload)
    table = pd.read_csv(dest)
    if "Model" not in table.columns or "ECI" not in table.columns:
        raise RuntimeError(f"Epoch ECI file at {dest} is missing Model or ECI.")
    return table


def canonical_runs(results: pd.DataFrame) -> pd.DataFrame:
    """Return one ForecastBench LLM row per `model_run_key`.

    Prompt variants share a run key and an ECI score. The dagger-marked variant is kept
    when it is present; otherwise the variant with the most subset questions is kept.

    Args:
      results (pd.DataFrame): Score table indexed by `model_pk`, with `kind` and `model_run_key`.
    """
    llms = results[results["kind"] == "LLM"].dropna(subset=["model_run_key"]).copy()
    llms["_prefer"] = llms["model"].astype(str).str.endswith("†").astype(int)
    llms = llms.sort_values(["_prefer", "n_subset"], ascending=[False, False])
    return llms.groupby("model_run_key", as_index=False).head(1).drop(columns="_prefer")


def join_eci(
    runs: pd.DataFrame, eci: pd.DataFrame, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach ECI scores and return unmatched ForecastBench runs and stale map rows.

    Args:
      runs (pd.DataFrame): One row per ForecastBench `model_run_key`.
      eci (pd.DataFrame): Epoch's published table.
      mapping (pd.DataFrame): Hand-maintained `model_run_key` to `eci_model` map.

    Returns:
      tuple: Runs with ECI columns, ForecastBench runs with no usable match, map rows
        whose `model_run_key` is absent from the forecasts, and map rows whose Epoch
        name is absent from the downloaded file.
    """
    mapping = mapping.copy()
    mapping["model_run_key"] = mapping["model_run_key"].astype(str).str.strip()
    mapping["eci_model"] = mapping["eci_model"].astype(str).str.strip()
    mapping = mapping.drop_duplicates("model_run_key")
    published = eci.rename(
        columns={
            "Model": "eci_model",
            "Display name": "eci_display_name",
            "ECI": "eci",
            "ECI CI low": "eci_ci_low",
            "ECI CI high": "eci_ci_high",
            "Date": "eci_date",
        }
    )
    keep = ["eci_model", "eci_display_name", "eci", "eci_ci_low", "eci_ci_high", "eci_date"]
    published = published[keep].drop_duplicates("eci_model")
    joined = mapping.merge(published, on="eci_model", how="left")
    stale = joined[joined["eci"].isna()].copy()
    found = joined[joined["eci"].notna()]
    matched = runs.merge(found, on="model_run_key", how="left")
    unmatched = matched[matched["eci"].isna()][
        ["model_run_key", "model", "model_organization", "model_release_date"]
    ].sort_values("model_release_date")
    absent = mapping.loc[
        ~mapping["model_run_key"].isin(runs["model_run_key"]), ["model_run_key", "eci_model"]
    ]
    return matched, unmatched, absent, stale


def _corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan")
    return float(np.corrcoef(pair["x"], pair["y"])[0, 1])


def _pct(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.1%}"


def _num(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:+.3f}"


def _year_sentence(llms: pd.DataFrame, col: str) -> str:
    """Median score and share beating the market, by release year.

    Args:
      llms (pd.DataFrame): ForecastBench LLMs with `model_release_date` and `col`.
      col (str): Score difference column.
    """
    dated = llms.dropna(subset=["model_release_date", col]).copy()
    dated["year"] = pd.to_datetime(dated["model_release_date"]).dt.year
    bits = []
    for year, group in dated.groupby("year"):
        bits.append(
            f"{int(year)} median {_num(float(group[col].median()))} "
            f"({(group[col] > 0).mean():.0%} beat the market, best {_num(float(group[col].max()))})"
        )
    return "By release year: " + "; ".join(bits) + "."


def _record_sentence(frontier: pd.DataFrame, results: pd.DataFrame, col: str) -> str:
    """Describe the running-best record, including how many rounds it rests on.

    Args:
      frontier (pd.DataFrame): `find_sota_models` output.
      results (pd.DataFrame): Score table indexed by `model_pk`.
      col (str): Score difference column.
    """
    if frontier.empty:
        return "No model was eligible for the running-best frontier."
    bits = []
    for _, row in frontier.iterrows():
        info = results.loc[row["model_pk"]]
        bits.append(
            f"{info['model']} at {_num(float(info[col]))} "
            f"({int(info['n_rounds'])} round, {int(info['n_subset'])} questions, "
            f"realized yes rate {_pct(float(info['realized_yes_rate_f']))})"
            if int(info["n_rounds"]) == 1
            else (
                f"{info['model']} at {_num(float(info[col]))} "
                f"({int(info['n_rounds'])} rounds, {int(info['n_subset'])} questions, "
                f"realized yes rate {_pct(float(info['realized_yes_rate_f']))})"
            )
        )
    if len(frontier) == 1:
        return (
            "The running-best Brier record is "
            + bits[0]
            + ". No later ForecastBench LLM has a higher score."
        )
    return "The running-best Brier records are " + "; ".join(bits) + "."


def _curve_paragraph(
    llms: pd.DataFrame,
    rounds: pd.DataFrame,
    frontier: pd.DataFrame,
    results: pd.DataFrame,
    col: str,
    cloud_corr: float,
    early_median: float,
    late_median: float,
    notes: Dict[str, Any],
    good_label: str,
) -> str:
    """Describe whether scores fall with release date and whether timing can explain it.

    Args:
      llms (pd.DataFrame): Plotted ForecastBench LLMs.
      rounds (pd.DataFrame): Per-round yes rate and resolved share.
      frontier (pd.DataFrame): Running-best Brier frontier.
      results (pd.DataFrame): Score table indexed by `model_pk`.
      col (str): Brier score difference column.
      cloud_corr (float): Correlation of release date with the score difference.
      early_median (float): Median score of models in the first half of the release span.
      late_median (float): Median score of models in the second half.
      notes (Dict[str, Any]): Output of `downward_curve_notes`.
      good_label (str): Key of the well-resolved frontier in `notes`.
    """
    scored = rounds[rounds["in_leaderboard"] & (rounds["n_resolved"] > 0)]
    n_yes = float(scored["n_surprises"].sum())
    n_resolved = float(scored["n_resolved"].sum())
    peak = scored.loc[scored["yes_rate"].idxmax()] if len(scored) else None
    direction = "up" if cloud_corr > 0.05 else "down" if cloud_corr < -0.05 else "flat"
    lines = [
        (
            f"The cloud of ForecastBench LLMs slopes {direction}: the correlation between "
            f"release date and the Brier difference is {cloud_corr:+.2f}. Median score in the "
            f"first half of models is {_num(early_median)}, and {_num(late_median)} in the "
            f"second half. {_year_sentence(llms, col)}"
        ),
        _record_sentence(frontier, results, col),
    ]
    early_yes = notes.get("surprise_rate_early_rounds")
    late_yes = notes.get("surprise_rate_late_rounds")
    early_share = notes.get("resolved_share_early_rounds")
    late_share = notes.get("resolved_share_late_rounds")
    if early_yes is not None and n_resolved:
        peak_text = ""
        if peak is not None:
            peak_text = (
                f" The highest round is {_pct(float(peak['yes_rate']))} "
                f"({int(peak['n_surprises'])} of {int(peak['n_resolved'])} on "
                f"{pd.Timestamp(peak['forecast_due_date']).date()})."
            )
        lines.append(
            f"Resolved low-probability questions came in yes {int(n_yes)} times out of "
            f"{int(n_resolved)} ({_pct(n_yes / n_resolved)}): {_pct(early_yes)} in the first "
            f"half of leaderboard rounds and {_pct(late_yes)} in the second.{peak_text} "
            f"Mean resolved share was {_pct(early_share)} then {_pct(late_share)}."
        )
        yes_up = late_yes > early_yes + 0.02
        share_down = late_share + 0.05 < early_share
        if yes_up and share_down:
            lines.append(
                "Checks 1 and 2 show recent rounds loaded with yes outcomes and less fully "
                "resolved, so a downward curve would be at least partly the resolution-timing "
                "effect."
            )
        elif yes_up:
            lines.append(
                "Check 1 shows recent rounds loaded with yes outcomes. The share resolved is "
                "similar in the two halves."
            )
        else:
            lines.append(
                "The yes rate stays near the low single digits, so checks 1 and 2 do not show "
                "recent rounds loaded with yes outcomes."
            )
    forecast_corr = notes.get("corr_release_date_vs_median_forecast_f")
    if forecast_corr is not None and not np.isnan(forecast_corr):
        lines.append(
            f"Median forecast on the subset correlates {forecast_corr:+.2f} with release date."
        )
    good_key = f"frontier_first_to_last[{good_label}]"
    if good_key in notes:
        first, last = notes[good_key]
        lines.append(
            f"On rounds with most of the subset resolved, the running best goes from "
            f"{_num(first)} to {_num(last)}."
        )
    return " ".join(lines)


def write_summary(
    path: Path,
    threshold: float,
    n_questions: int,
    n_bootstrap: int,
    reference: str,
    headline: str,
    frontiers: Dict[str, pd.DataFrame],
    notes: Dict[str, Any],
    cloud: Dict[str, Dict[str, float]],
    n_matched: int,
    n_unmatched: int,
    unmatched: pd.DataFrame,
    eci_corr: Dict[str, float],
    good_label: str,
    curve_text: str,
) -> None:
    """Write `summary.md` for the low-probability downward-curve run.

    Args:
      path (Path): Output file.
      threshold (float): Low-probability cutoff.
      n_questions (int): Resolved subset questions.
      n_bootstrap (int): Bootstrap replicates.
      reference (str): Reference forecaster.
      headline (str): Difficulty method.
      frontiers (Dict[str, pd.DataFrame]): Running-best table per scoring rule.
      notes (Dict[str, Any]): Four-check numbers.
      cloud (Dict[str, Dict[str, float]]): Release-date correlations and half medians.
      n_matched (int): ForecastBench runs with an ECI score.
      n_unmatched (int): ForecastBench runs without one.
      unmatched (pd.DataFrame): Those runs.
      eci_corr (Dict[str, float]): Correlation of ECI with each rule's score.
      good_label (str): Label of the well-resolved frontier.
      curve_text (str): Paragraph on the curve and the timing checks.
    """
    pct = int(round(threshold * 100))
    lines = [
        f"# SOTA on market questions priced at {pct}% or below",
        "",
        f"Tournament leaderboard. {n_questions} resolved questions. "
        f"Scores are {headline.replace('_', ' ')} {reference} minus model "
        f"({n_bootstrap} question bootstrap replicates). Positive means better than the market.",
        "",
        "## Takeaway",
        "",
        curve_text,
        "",
        f"Epoch ECI is attached for {n_matched} ForecastBench model runs. "
        f"{n_unmatched} runs have no row in `eci_model_map.csv` or no published score "
        "(listed below). Among plotted models, "
        + "; ".join(
            f"ECI and the {rule} difference correlate at {eci_corr[rule]:+.2f}"
            for rule in scoring.SCORE_RULES
            if rule in eci_corr
        )
        + ".",
        "",
        "## Running-best frontier",
        "",
    ]
    for rule, frontier in frontiers.items():
        lines.append(f"### {plots.RULE_LABELS[rule]}")
        lines.append("")
        if frontier.empty:
            lines.append("No eligible model.")
            lines.append("")
            continue
        lines.append("| Release date | Model | Score vs market |")
        lines.append("| --- | --- | --- |")
        col = f"{rule}_{headline}_vs_{reference}"
        for _, row in frontier.iterrows():
            lines.append(
                f"| {pd.Timestamp(row['model_release_date']).date()} | {row['model']} | "
                f"{row[col]:+.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Models without an ECI match",
            "",
            "One row per `model_run_key`. Prompt variants of the same run are not repeated. "
            "A blank Epoch row usually means the model is newer than the published ECI file, "
            "or the checkpoint dates do not line up (for example DeepSeek-V3's December 2024 "
            "ForecastBench release against Epoch's March 2025 score).",
            "",
        ]
    )
    if unmatched.empty:
        lines.append("Every ForecastBench run in this subset is mapped.")
    else:
        lines.append("| Release date | Organization | Model | `model_run_key` |")
        lines.append("| --- | --- | --- | --- |")
        for _, row in unmatched.iterrows():
            released = pd.Timestamp(row["model_release_date"])
            released_text = "" if pd.isna(released) else str(released.date())
            lines.append(
                f"| {released_text} | {row['model_organization']} | {row['model']} | "
                f"`{row['model_run_key']}` |"
            )
    lines.extend(["", "## Files", ""])
    for name in (
        "figures/sota_frontier_brier.png",
        "figures/sota_frontier_log.png",
        "figures/score_vs_eci_brier.png",
        "figures/score_vs_eci_log.png",
        "figures/extreme_vs_nonextreme_brier.png",
        "figures/extreme_vs_nonextreme_log.png",
        "figures/downward_curve_checks.png",
        "tables/eci_unmatched_forecastbench.csv",
        "tables/eci_matched.csv",
        "tables/counts_by_round.csv",
    ):
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append(f"Check 4 uses `{good_label}`.")
    path.write_text("\n".join(lines) + "\n")


def _half_medians(llms: pd.DataFrame, col: str) -> tuple[float, float, float]:
    dated = llms.dropna(subset=["model_release_date", col]).copy()
    dated["model_release_date"] = pd.to_datetime(dated["model_release_date"])
    days = dated["model_release_date"].map(pd.Timestamp.toordinal)
    corr = _corr(days, dated[col])
    dated = dated.sort_values("model_release_date")
    if len(dated) < 2:
        return corr, float("nan"), float("nan")
    half = len(dated) // 2
    early = float(dated.iloc[:half][col].median())
    late = float(dated.iloc[half:][col].median())
    return corr, early, late


def main(config_path: Optional[str] = None) -> Path:
    """Score the low-probability market subset and write the section 4.4 figures.

    Args:
      config_path (Optional[str]): Config file. Defaults to `config.toml`.
    """
    config = load_config(Path(config_path) if config_path else HERE / "config.toml")
    config["analysis"]["subset"] = "low"
    a_cfg, s_cfg, c_cfg = config["analysis"], config["scoring"], config["comparisons"]
    threshold = float(a_cfg["threshold"])
    pct = int(round(threshold * 100))
    n_bootstrap = int(s_cfg["n_bootstrap"])
    rng = np.random.default_rng(s_cfg["seed"])
    headline = s_cfg["headline_difficulty"]
    out = make_output_dir(config, f"low{pct}_downward_curve")
    tables, figures = out / "tables", out / "figures"
    info_path = out / "run_info.json"
    info = json.loads(info_path.read_text())
    info.update({"subset": "low", "threshold": threshold, "eci_url": ECI_URL})
    info_path.write_text(json.dumps(info, indent=2))

    df, markets, _ = load_market_rows(config)
    leaderboard_rounds = sorted(df["forecast_due_date"].unique())
    tags = market_low_probability_tags(df)
    columns = subset_columns("low", threshold)
    df = df.merge(tags[["question_pk", columns["mask"], columns["side"]]], on="question_pk")
    df["in_subset"] = df[columns["mask"]].astype(bool)
    df["side"] = df[columns["side"]]
    effects = {rule: scoring.question_effects(df, rule) for rule in scoring.SCORE_RULES}
    df = scoring.adjust(df, effects)
    logger.info(f"Fitted question effects on {df['question_pk'].nunique()} market questions.")

    subset = df[df["in_subset"]].copy()
    questions = subset[QUESTION_COLS].drop_duplicates("question_pk").reset_index(drop=True)
    questions = flip_onto_low_side(questions.assign(forecast=np.nan))
    base_rates = base_rate_table(questions, c_cfg["base_rate_min_n"])
    synthetic = build_comparison_rows(questions, base_rates)
    synthetic = scoring.adjust(scoring.add_row_scores(synthetic, s_cfg["log_clip"]), effects)
    fit_rows = subset.copy()
    subset = pd.concat([subset, synthetic], ignore_index=True)
    subset = flip_onto_low_side(subset)

    available = subset.groupby("model_pk")["model"].first()
    primary = resolve_comparison_names(c_cfg["primary"], available)
    reference = primary.get(c_cfg["reference"], c_cfg["reference"])
    if reference not in set(available.index):
        raise RuntimeError(f"Reference forecaster {c_cfg['reference']!r} is not on the subset.")
    comparison_pks = list(primary.values())

    results, draws = evaluate(subset, fit_rows, comparison_pks, config, rng, n_bootstrap)
    results = describe_forecasters(subset).join(results)
    run_key = (
        subset.loc[subset["model_run_key"].notna()].groupby("model_pk")["model_run_key"].first()
    )
    results = results.join(run_key.rename("model_run_key"))
    results.index.name = "model_pk"
    results.to_csv(tables / "subset_scores.csv")

    keep = results["n_subset"] >= a_cfg["min_n"]
    plotted = results[keep | (results["kind"] == "Comparison")]
    llm_pks = plotted.index[plotted["kind"] == "LLM"].tolist()
    ext_pks = plotted.index[plotted["kind"] == "External"].tolist()
    logger.info(
        f"{questions['question_pk'].nunique()} low-probability questions; "
        f"{len(llm_pks)} LLMs with at least {a_cfg['min_n']} forecasts."
    )

    rounds = round_table(questions, markets, threshold, "low", leaderboard_rounds)
    rounds.to_csv(tables / "counts_by_round.csv", index=False)

    frontiers: Dict[str, pd.DataFrame] = {}
    cloud: Dict[str, Dict[str, float]] = {}
    for rule in scoring.SCORE_RULES:
        col = f"{rule}_{headline}_vs_{reference}"
        frontier, band = frontier_with_band(results, draws.get(rule), llm_pks, reference, col)
        frontier.to_csv(tables / f"sota_frontier_{rule}.csv", index=False)
        band.to_csv(tables / f"sota_frontier_band_{rule}.csv", index=False)
        frontiers[rule] = frontier
        corr, early, late = _half_medians(results.loc[llm_pks], col)
        cloud[rule] = {"corr": corr, "early_median": early, "late_median": late}
        plots.sota_frontier_plot(
            llms=results.loc[llm_pks].dropna(subset=["model_release_date", col]),
            frontier=frontier,
            band=band,
            col=col,
            rule=rule,
            reference_name=reference,
            title=(
                f"SOTA frontier on market questions priced at {pct}% or below "
                f"({plots.RULE_LABELS[rule]})"
            ),
            path=figures / f"sota_frontier_{rule}.png",
        )

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
            config["plots"]["n_labels"],
            figures / f"extreme_vs_nonextreme_{rule}.png",
        )

    eci = download_eci(tables / "eci_scores.csv")
    mapping = pd.read_csv(MAP_PATH)
    runs = canonical_runs(results)
    matched, unmatched, absent, stale = join_eci(runs, eci, mapping)
    matched.to_csv(tables / "eci_matched.csv", index=False)
    unmatched.to_csv(tables / "eci_unmatched_forecastbench.csv", index=False)
    absent.to_csv(tables / "eci_map_run_missing.csv", index=False)
    stale.to_csv(tables / "eci_map_name_missing.csv", index=False)
    frontier_runs = set()
    for frontier in frontiers.values():
        if "model_pk" in frontier.columns:
            frontier_runs.update(
                results.loc[results.index.isin(frontier["model_pk"]), "model_run_key"].dropna()
            )
    eligible = matched[matched["n_subset"] >= a_cfg["min_n"]].dropna(subset=["eci"])
    eci_corr = {}
    for rule in scoring.SCORE_RULES:
        col = f"{rule}_{headline}_vs_{reference}"
        eci_corr[rule] = _corr(eligible["eci"], eligible[col])
        plotted_eci = eligible.copy()
        plotted_eci["on_frontier"] = plotted_eci["model_run_key"].isin(frontier_runs)
        plots.eci_score_plot(plotted_eci, col, rule, figures / f"score_vs_eci_{rule}.png")

    min_share = config["checks"]["min_resolved_share"]
    good_rounds = rounds.loc[
        rounds["in_leaderboard"] & (rounds["resolved_share"] >= min_share), "forecast_due_date"
    ].tolist()
    good_label = f"Rounds ≥{min_share:.0%} resolved ({len(good_rounds)})"
    headline_col = f"brier_{headline}_vs_{reference}"
    restricted = {"All rounds": frontiers["brier"]}
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
        restricted[good_label] = frontier_good
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
        suptitle="Checks for a downward curve on low-probability market questions",
        panel1_title="1. Yes rate of resolved low-probability questions",
        yes_ylabel="Share resolving yes",
        extremity_ylabel="Median forecast on the low-probability subset",
    )
    notes = downward_curve_notes(rounds, extremity, restricted, headline_col)
    curve_text = _curve_paragraph(
        results.loc[llm_pks],
        rounds,
        frontiers["brier"],
        results,
        headline_col,
        cloud["brier"]["corr"],
        cloud["brier"]["early_median"],
        cloud["brier"]["late_median"],
        notes,
        good_label,
    )
    log_text = (
        f" On log score the release-date correlation is {cloud['log']['corr']:+.2f} "
        f"(median {_num(cloud['log']['early_median'])} early, {_num(cloud['log']['late_median'])} late)."
    )
    write_summary(
        out / "summary.md",
        threshold,
        int(questions["question_pk"].nunique()),
        n_bootstrap,
        reference,
        headline,
        frontiers,
        notes,
        cloud,
        int(matched["eci"].notna().sum()),
        int(len(unmatched)),
        unmatched,
        eci_corr,
        good_label,
        curve_text + log_text,
    )
    logger.info(f"Done. Outputs in {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
