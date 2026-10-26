"""Scoring rules, difficulty adjustment and the question-level bootstrap.

All scores here are losses (lower is better). Comparisons are reported as
`comparison score - model score`, so a positive difference means the model did better.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from _common import leaderboard, logger

SCORE_RULES = ("brier", "log")
DIFFICULTY_METHODS = ("all_question_fit", "subset_fit", "none")


def add_row_scores(df: pd.DataFrame, log_clip: float) -> pd.DataFrame:
    """Add per-forecast `brier` and `log` (natural log, clipped forecasts) columns.

    Args:
      df (pd.DataFrame): Rows with `forecast` and `resolved_to`.
      log_clip (float): Forecasts are clipped to [log_clip, 1 - log_clip] for the log score.
    """
    df = df.copy()
    forecast = df["forecast"].astype(float)
    outcome = df["resolved_to"].astype(float)
    df["brier"] = (forecast - outcome) ** 2
    clipped = forecast.clip(log_clip, 1 - log_clip)
    df["log"] = -np.log(clipped * outcome + (1 - clipped) * (1 - outcome))
    return df


def question_effects(df: pd.DataFrame, score_col: str, question_type: str = "market") -> pd.Series:
    """Fit score = model effect + question effect and return the question effects.

    Uses `two_way_fixed_effects` from the production leaderboard, so the fit drops the same
    models (dummy forecasters, Imputed Forecaster, LLM crowds, external submissions, and models
    released more than a year before the due date).

    Args:
      df (pd.DataFrame): Leaderboard rows with `score_col`.
      score_col (str): "brier" or "log".
      question_type (str): "market" or "dataset". Both use the two-way fixed effects fit.
    """
    lb = leaderboard()
    df_fit = df.copy()
    df_fit["brier_score"] = df_fit[score_col]
    fitted = lb.two_way_fixed_effects(
        df_fit,
        question_type=question_type,
        market_question_adjustment=lb.MarketQuestionAdjustment.TWO_WAY_FIXED_EFFECTS,
    )
    return fitted.drop_duplicates("question_pk").set_index("question_pk")["question_fixed_effect"]


def adjust(df: pd.DataFrame, effects: Dict[str, pd.Series]) -> pd.DataFrame:
    """Add `<rule>_adj = <rule> - question effect` for each rule in `effects`.

    Args:
      df (pd.DataFrame): Scored rows.
      effects (Dict[str, pd.Series]): Question effects by score rule, indexed by question_pk.
    """
    df = df.copy()
    for rule, fe in effects.items():
        df[f"{rule}_adj"] = df[rule] - df["question_pk"].map(fe)
    return df


def _pivot(df: pd.DataFrame, value_col: str, questions: pd.Index) -> pd.DataFrame:
    """Return a question x forecaster matrix of `value_col` (NaN where not forecast)."""
    return (
        df.pivot_table(index="question_pk", columns="model_pk", values=value_col, aggfunc="mean")
        .reindex(questions)
        .astype(float)
    )


def bootstrap_weights(strata: pd.Series, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return an (n, Q) matrix of how often each question is drawn in each replicate.

    Questions are resampled with replacement within each stratum (round), as in the
    production leaderboard's question-level bootstrap.

    Args:
      strata (pd.Series): Stratum label per question, in matrix order.
      n (int): Number of replicates.
      rng (np.random.Generator): Random generator.
    """
    weights = np.zeros((n, len(strata)))
    for _, positions in pd.Series(np.arange(len(strata))).groupby(strata.values):
        idx = positions.to_numpy()
        draws = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n)
        weights[:, idx] = draws
    return weights


def _weighted_means(values: pd.DataFrame, weights: Optional[np.ndarray]) -> np.ndarray:
    """Return column means of `values` ignoring NaN, for each weight row (or unweighted)."""
    present = values.notna().to_numpy(dtype=float)
    filled = values.fillna(0).to_numpy()
    if weights is None:
        with np.errstate(invalid="ignore", divide="ignore"):
            return filled.sum(axis=0) / present.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (weights @ filled) / (weights @ present)


def _summarize_draws(point: np.ndarray, draws: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    if draws is None:
        nan = np.full_like(point, np.nan)
        return {"": point, "_ci_lower": nan, "_ci_upper": nan, "_p_better": nan}
    return {
        "": point,
        "_ci_lower": np.nanpercentile(draws, 2.5, axis=0),
        "_ci_upper": np.nanpercentile(draws, 97.5, axis=0),
        "_p_better": np.nanmean(draws > 0, axis=0),
    }


def score_table(
    df: pd.DataFrame,
    value_col: str,
    comparisons: List[str],
    weights: Optional[np.ndarray],
    questions: pd.Index,
    paired: bool,
    label: str,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Return mean scores and differences from each comparison forecaster, with bootstrap CIs.

    With `paired=False` (difficulty-adjusted scores) each forecaster's mean is taken over its
    own subset questions and differences are taken between those means. With `paired=True`
    (unadjusted scores) each difference is the mean over the questions both forecasters answered.

    Args:
      df (pd.DataFrame): Subset rows with `value_col`.
      value_col (str): Column to average, e.g. "brier_adj".
      comparisons (List[str]): `model_pk`s of comparison forecasters present in `df`.
      weights (Optional[np.ndarray]): Bootstrap weights from `bootstrap_weights`, or None.
      questions (pd.Index): Question order matching the columns of `weights`.
      paired (bool): Use common questions for each difference.
      label (str): Prefix for output columns, e.g. "brier_all_question_fit".

    Returns:
      tuple: The score table indexed by `model_pk`, and the bootstrap draws of the mean score
        (replicates x forecasters) when `weights` is given.
    """
    matrix = _pivot(df, value_col, questions)
    point = _weighted_means(matrix, None)
    draws = _weighted_means(matrix, weights) if weights is not None else None
    out = pd.DataFrame(index=matrix.columns)
    out[label] = point
    if draws is not None:
        out[f"{label}_ci_lower"] = np.nanpercentile(draws, 2.5, axis=0)
        out[f"{label}_ci_upper"] = np.nanpercentile(draws, 97.5, axis=0)

    for comparison in comparisons:
        if comparison not in matrix.columns:
            continue
        if paired:
            diff_matrix = matrix[[comparison]].to_numpy() - matrix
            diff_matrix = diff_matrix.where(matrix[comparison].notna(), axis=0)
            diff_point = _weighted_means(diff_matrix, None)
            diff_draws = _weighted_means(diff_matrix, weights) if weights is not None else None
        else:
            idx = matrix.columns.get_loc(comparison)
            diff_point = point[idx] - point
            diff_draws = draws[:, [idx]] - draws if draws is not None else None
        for suffix, values in _summarize_draws(diff_point, diff_draws).items():
            out[f"{label}_vs_{comparison}{suffix}"] = values

    draws_df = None
    if draws is not None:
        draws_df = pd.DataFrame(draws, columns=matrix.columns)
    return out, draws_df


def subset_fit_bootstrap(
    df_subset: pd.DataFrame,
    df_fit_rows: pd.DataFrame,
    rule: str,
    comparisons: List[str],
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Bootstrap the subset-only fit by refitting the fixed effects in every replicate.

    Slow: one fixed-effects fit per replicate. Only used when `bootstrap_subset_fit` is on.

    Args:
      df_subset (pd.DataFrame): All subset rows (leaderboard models and comparisons).
      df_fit_rows (pd.DataFrame): Subset rows used for the fit.
      rule (str): "brier" or "log".
      comparisons (List[str]): Comparison `model_pk`s.
      n (int): Replicates.
      rng (np.random.Generator): Random generator.
    """
    questions = df_subset[["question_pk", "forecast_due_date"]].drop_duplicates("question_pk")
    draws = []
    for i in range(n):
        sampled = questions.groupby("forecast_due_date", group_keys=False).apply(
            lambda g: g.sample(len(g), replace=True, random_state=rng.integers(1 << 31))
        )
        sampled = sampled.assign(draw=sampled.groupby("question_pk").cumcount())
        sampled["new_pk"] = sampled["question_pk"] + "_sim_" + sampled["draw"].astype(str)
        mapping = sampled[["question_pk", "new_pk"]]
        boot_fit = df_fit_rows.merge(mapping, on="question_pk").assign(
            question_pk=lambda d: d["new_pk"]
        )
        boot_all = df_subset.merge(mapping, on="question_pk").assign(
            question_pk=lambda d: d["new_pk"]
        )
        fe = question_effects(boot_fit, rule)
        boot_all[f"{rule}_adj"] = boot_all[rule] - boot_all["question_pk"].map(fe)
        draws.append(boot_all.groupby("model_pk")[f"{rule}_adj"].mean())
        if (i + 1) % 20 == 0:
            logger.info(f"Subset-fit bootstrap {i + 1}/{n}")
    draws_df = pd.DataFrame(draws).reset_index(drop=True)
    out = pd.DataFrame(index=draws_df.columns)
    label = f"{rule}_subset_fit"
    for comparison in comparisons:
        if comparison not in draws_df.columns:
            continue
        diff = draws_df[[comparison]].to_numpy() - draws_df
        out[f"{label}_vs_{comparison}_ci_lower"] = np.nanpercentile(diff, 2.5, axis=0)
        out[f"{label}_vs_{comparison}_ci_upper"] = np.nanpercentile(diff, 97.5, axis=0)
        out[f"{label}_vs_{comparison}_p_better"] = np.nanmean(diff > 0, axis=0)
    return out


def auc(forecast: pd.Series, outcome: pd.Series) -> float:
    """Return the area under the ROC curve (ties count half), or NaN without both outcomes.

    Args:
      forecast (pd.Series): Forecasts.
      outcome (pd.Series): 0/1 outcomes.
    """
    from scipy.stats import rankdata

    mask = forecast.notna() & outcome.notna()
    forecast, outcome = forecast[mask].to_numpy(), outcome[mask].to_numpy()
    n_pos, n_neg = int((outcome == 1).sum()), int((outcome == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(forecast)
    return float((ranks[outcome == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def wilson_interval(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion.

    Args:
      successes (float): Number of yes outcomes.
      n (float): Number of trials.
      z (float): Normal quantile.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)
