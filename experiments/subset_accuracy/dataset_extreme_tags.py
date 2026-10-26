"""Compare two ways of tagging dataset questions as extreme.

Method 1 uses the Naive Forecaster's probability. Method 2, the default, uses
the shrunk historical yes rate of the question's family. The raw historical
rate is still reported. Both use only resolutions dated before that question's
forecast due date.

The family is the dataset series (`source` and `id`), pooled across horizons.
The same series at one horizon is also recorded. That narrower family usually
has only one or two earlier resolutions, which is too few to estimate a 10%
tail, so it is not the tagging rule. This script does not score forecasts.

Usage:
  python dataset_extreme_tags.py [--config config.toml]
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plots
from _common import load_config, logger, make_output_dir

NAIVE_MODEL = "Naive Forecaster"
# Same tail rule as market questions in `tags.py`: <= t, or >= 1 - t.
THRESHOLDS = (0.10, 0.05)
# A base rate needs this many earlier resolutions before it can tag a question.
MIN_NS = (5, 20)
HEADLINE_THRESHOLD = 0.10
SERIES_FAMILY = ("source", "id")
HORIZON_FAMILY = ("source", "id", "horizon")
SIDES = ("low", "middle", "high")


def load_dataset_questions(cache_dir: str) -> pd.DataFrame:
    """Return one Naive Forecaster row per dataset question.

    Args:
      cache_dir (str): Directory containing `forecasts.parquet`.
    """
    path = Path(cache_dir) / "forecasts.parquet"
    df = pd.read_parquet(
        path,
        columns=[
            "question_pk",
            "forecast_due_date",
            "source",
            "id",
            "horizon",
            "resolution_date",
            "resolved_to",
            "forecast",
            "imputed",
            "model",
            "question_type",
        ],
        filters=[("question_type", "==", "dataset"), ("model", "==", NAIVE_MODEL)],
    )
    df = df.drop(columns=["question_type", "model"]).rename(columns={"forecast": "naive_forecast"})
    df["forecast_due_date"] = pd.to_datetime(df["forecast_due_date"])
    df["resolution_date"] = pd.to_datetime(df["resolution_date"])
    df["naive_forecast"] = pd.to_numeric(df["naive_forecast"], errors="coerce")
    df["resolved_to"] = pd.to_numeric(df["resolved_to"], errors="coerce")
    df["id"] = df["id"].astype(str)
    if df["question_pk"].duplicated().any():
        raise ValueError("Expected one Naive Forecaster row per dataset question.")
    return df.reset_index(drop=True)


def earlier_resolution_stats(questions: pd.DataFrame, family_cols: Iterable[str]) -> pd.DataFrame:
    """Count earlier resolved family members and their yes rate.

    A resolution counts only when its resolution date is strictly before the
    question's forecast due date. The question's own outcome is therefore
    excluded: dataset questions resolve after the due date.

    Args:
      questions (pd.DataFrame): One row per dataset question.
      family_cols (Iterable[str]): Columns that define the family.

    Returns:
      pd.DataFrame: `n_earlier` and `base_rate`, aligned to `questions`.
    """
    family_cols = list(family_cols)
    n_earlier = pd.Series(0, index=questions.index, dtype=int)
    base_rate = pd.Series(np.nan, index=questions.index, dtype=float)
    resolved = questions["resolved_to"].isin([0, 1])
    for _, group in questions.groupby(family_cols, sort=False):
        hist = (
            group.loc[resolved.loc[group.index]]
            .dropna(subset=["resolution_date"])
            .sort_values("resolution_date")
        )
        dates = hist["resolution_date"].to_numpy()
        outcomes = hist["resolved_to"].to_numpy(dtype=float)
        dues = group["forecast_due_date"].to_numpy()
        if len(dates):
            n = np.searchsorted(dates, dues, side="left")
        else:
            n = np.zeros(len(group), dtype=int)
        rate = np.full(len(group), np.nan)
        known = n > 0
        rate[known] = np.cumsum(outcomes)[n[known] - 1] / n[known]
        n_earlier.loc[group.index] = n
        base_rate.loc[group.index] = rate
    return pd.DataFrame({"n_earlier": n_earlier, "base_rate": base_rate})


def shrinkage_strength(n_by_series: np.ndarray, k_by_series: np.ndarray) -> tuple[float, float]:
    """Estimate the source yes rate and the shrinkage strength from series counts.

    The strength m is the empirical-Bayes prior sample size. It is small when
    series yes rates in the source differ by more than sampling noise, and
    infinite when they do not, which shrinks every series all the way to the
    source rate.

    Args:
      n_by_series (np.ndarray): Earlier resolutions per series.
      k_by_series (np.ndarray): Earlier yes outcomes per series.

    Returns:
      tuple: Source yes rate, and shrinkage strength. The strength is infinite
        when the series rates do not differ beyond sampling noise.
    """
    total_n = float(np.sum(n_by_series))
    if total_n <= 0:
        return float("nan"), float("nan")
    mu = float(np.sum(k_by_series) / total_n)
    active = n_by_series > 0
    n_active = n_by_series[active].astype(float)
    n_series = int(active.sum())
    if n_series < 2 or mu <= 0 or mu >= 1:
        return mu, float("inf")
    rates = k_by_series[active].astype(float) / n_active
    binomial_var = mu * (1 - mu)
    weights = n_active / binomial_var
    weight_sum = float(weights.sum())
    q_stat = float(np.sum(weights * (rates - mu) ** 2))
    scale = weight_sum - float(np.sum(weights**2) / weight_sum)
    if scale <= 0:
        return mu, float("inf")
    tau2 = (q_stat - (n_series - 1)) / scale
    if tau2 <= 0:
        return mu, float("inf")
    if tau2 >= binomial_var:
        return mu, 0.0
    return mu, binomial_var / tau2 - 1


def shrunk_series_rates(questions: pd.DataFrame) -> pd.DataFrame:
    """Shrunk series yes rate, using only resolutions before each due date.

    The raw rate is k / n for the series (`source`, `id`). The shrunk rate is
    (k + m * mu) / (n + m), where mu is the source's earlier yes rate and m
    is estimated from how much series rates vary inside that source. With no
    earlier resolutions in the series, the rate is the source rate. With no
    earlier resolutions in the source, the rate is missing.

    Args:
      questions (pd.DataFrame): One row per dataset question, with `source`,
        `id`, `forecast_due_date`, `resolution_date`, and `resolved_to`.

    Returns:
      pd.DataFrame: `n_earlier`, `k_earlier`, `raw_base_rate`, `source_n`,
        `source_yes_rate`, `shrinkage_m`, and `shrunk_base_rate`, aligned to
        `questions`.
    """
    n_earlier = pd.Series(0, index=questions.index, dtype=int)
    k_earlier = pd.Series(0.0, index=questions.index, dtype=float)
    source_n = pd.Series(0, index=questions.index, dtype=int)
    source_yes_rate = pd.Series(np.nan, index=questions.index, dtype=float)
    shrinkage_m = pd.Series(np.nan, index=questions.index, dtype=float)
    shrunk = pd.Series(np.nan, index=questions.index, dtype=float)
    resolved = questions["resolved_to"].isin([0, 1])

    for _, group in questions.groupby("source", sort=False):
        hist = group.loc[resolved.loc[group.index]].dropna(subset=["resolution_date"])
        series_hist = {}
        for series_id, series in hist.groupby("id", sort=False):
            series = series.sort_values("resolution_date")
            outcomes = series["resolved_to"].to_numpy(dtype=float)
            series_hist[series_id] = (
                series["resolution_date"].to_numpy(),
                np.cumsum(outcomes),
                np.arange(1, len(series) + 1, dtype=float),
            )
        due_dates = np.sort(group["forecast_due_date"].unique())
        series_ids = list(series_hist)
        k_at = np.zeros((len(series_ids), len(due_dates)))
        n_at = np.zeros((len(series_ids), len(due_dates)))
        for i, series_id in enumerate(series_ids):
            dates, cum_k, cum_n = series_hist[series_id]
            idx = np.searchsorted(dates, due_dates, side="left") - 1
            valid = idx >= 0
            k_at[i, valid] = cum_k[idx[valid]]
            n_at[i, valid] = cum_n[idx[valid]]
        mu_at = np.full(len(due_dates), np.nan)
        m_at = np.full(len(due_dates), np.nan)
        source_n_at = n_at.sum(axis=0)
        for j in range(len(due_dates)):
            mu_at[j], m_at[j] = shrinkage_strength(n_at[:, j], k_at[:, j])
        series_index = {series_id: i for i, series_id in enumerate(series_ids)}
        rows = group.index.to_numpy()
        due_j = np.searchsorted(due_dates, group["forecast_due_date"].to_numpy())
        series_i = group["id"].map(series_index)
        known_series = series_i.notna().to_numpy()
        series_i = series_i.fillna(0).astype(int).to_numpy()
        k = np.zeros(len(group))
        n = np.zeros(len(group))
        k[known_series] = k_at[series_i[known_series], due_j[known_series]]
        n[known_series] = n_at[series_i[known_series], due_j[known_series]]
        mu = mu_at[due_j]
        m = m_at[due_j]
        p = np.full(len(group), np.nan)
        finite = np.isfinite(m) & (n + m > 0)
        p[finite] = (k[finite] + m[finite] * mu[finite]) / (n[finite] + m[finite])
        full = np.isinf(m) & np.isfinite(mu)
        p[full] = mu[full]
        n_earlier.loc[rows] = n.astype(int)
        k_earlier.loc[rows] = k
        source_n.loc[rows] = source_n_at[due_j].astype(int)
        source_yes_rate.loc[rows] = mu
        shrinkage_m.loc[rows] = m
        shrunk.loc[rows] = p

    raw = pd.Series(np.nan, index=questions.index, dtype=float)
    has_history = n_earlier > 0
    raw.loc[has_history] = k_earlier.loc[has_history] / n_earlier.loc[has_history]
    return pd.DataFrame(
        {
            "n_earlier": n_earlier,
            "k_earlier": k_earlier,
            "raw_base_rate": raw,
            "source_n": source_n,
            "source_yes_rate": source_yes_rate,
            "shrinkage_m": shrinkage_m,
            "shrunk_base_rate": shrunk,
        }
    )


def tail_side(probability: pd.Series, threshold: float) -> pd.Series:
    """Return "low", "middle", "high", or missing for a probability.

    Args:
      probability (pd.Series): Forecast or base rate.
      threshold (float): Tail cutoff. Low is <= threshold; high is >= 1 - threshold.
    """
    p = pd.to_numeric(probability, errors="coerce")
    side = pd.Series(pd.NA, index=p.index, dtype="object")
    known = p.notna()
    side.loc[known] = "middle"
    side.loc[known & (p <= threshold)] = "low"
    side.loc[known & (p >= round(1 - threshold, 10))] = "high"
    return side


def _kappa(left: pd.Series, right: pd.Series) -> float:
    """Cohen's kappa for two boolean series of equal length."""
    if len(left) == 0:
        return float("nan")
    observed = float((left.to_numpy() == right.to_numpy()).mean())
    p_left = float(left.mean())
    p_right = float(right.mean())
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    if np.isclose(expected, 1):
        return float("nan")
    return (observed - expected) / (1 - expected)


def agreement_row(
    questions: pd.DataFrame,
    naive_side: pd.Series,
    base_rate: pd.Series,
    n_earlier: pd.Series,
    threshold: float,
    min_n: int,
    family: str,
) -> dict:
    """Summarize agreement between the naive tag and a base-rate tag.

    Args:
      questions (pd.DataFrame): Dataset questions. Used only for the row count.
      naive_side (pd.Series): "low", "middle", or "high" from the naive forecast.
      base_rate (pd.Series): Historical yes rate.
      n_earlier (pd.Series): Earlier resolutions behind `base_rate`.
      threshold (float): Tail cutoff.
      min_n (int): Minimum earlier resolutions required to tag with the base rate.
      family (str): Family name, for the output table.
    """
    base_side = tail_side(base_rate, threshold).where(n_earlier >= min_n)
    comparable = naive_side.notna() & base_side.notna()
    naive = naive_side[comparable]
    base = base_side[comparable]
    naive_extreme = naive.isin(["low", "high"])
    base_extreme = base.isin(["low", "high"])
    both = naive_extreme & base_extreme
    n_both = int(both.sum())
    same_side = int((naive[both] == base[both]).sum()) if n_both else 0
    n_comp = int(comparable.sum())
    return {
        "family": family,
        "threshold": threshold,
        "min_n": min_n,
        "n_questions": len(questions),
        "n_comparable": n_comp,
        "n_naive_extreme": int(naive_extreme.sum()),
        "n_base_extreme": int(base_extreme.sum()),
        "both_extreme": n_both,
        "naive_only": int((naive_extreme & ~base_extreme).sum()),
        "base_only": int((~naive_extreme & base_extreme).sum()),
        "neither": int((~naive_extreme & ~base_extreme).sum()),
        "agreement_rate": (naive_extreme == base_extreme).mean() if n_comp else np.nan,
        "kappa": _kappa(naive_extreme, base_extreme),
        "both_extreme_same_side": same_side,
        "both_extreme_opposite_side": n_both - same_side,
    }


def side_crosstab(naive_side: pd.Series, base_side: pd.Series) -> pd.DataFrame:
    """Count naive side against base-rate side.

    Args:
      naive_side (pd.Series): Naive side on the comparable questions.
      base_side (pd.Series): Base-rate side on the same questions.
    """
    table = pd.crosstab(naive_side, base_side, dropna=False).reindex(
        index=list(SIDES), columns=list(SIDES), fill_value=0
    )
    table.index.name = "naive_side"
    table.columns.name = "base_rate_side"
    return table


def agreement_plot(
    questions: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    path: Path,
    min_n: int = 0,
    n_col: str = "n_earlier_series",
) -> None:
    """Hexbin of the naive forecast against a base rate.

    Args:
      questions (pd.DataFrame): Rows with `naive_forecast` and `y_col`.
      y_col (str): Base-rate column.
      y_label (str): Y-axis label.
      title (str): Figure title.
      path (Path): Output file.
      min_n (int): Minimum earlier series resolutions to include. Zero keeps
        every question with a base rate.
      n_col (str): Column holding that resolution count.
    """
    kept = questions[questions[y_col].notna() & questions["naive_forecast"].notna()]
    if min_n:
        kept = kept[kept[n_col] >= min_n]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    hexbin = ax.hexbin(
        kept["naive_forecast"],
        kept[y_col],
        gridsize=35,
        cmap="Blues",
        mincnt=1,
        extent=(0, 1, 0, 1),
    )
    for value in (HEADLINE_THRESHOLD, round(1 - HEADLINE_THRESHOLD, 10)):
        ax.axvline(value, color=plots.INK_SECONDARY, lw=0.8, ls="--")
        ax.axhline(value, color=plots.INK_SECONDARY, lw=0.8, ls="--")
    fig.colorbar(hexbin, ax=ax, label="Questions")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Naive forecast")
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left")
    plots._save(fig, path)


# Naive forecast against the shrunk rate. The raw rate is not in this chart.
NAIVE_SHRUNK_BARS = (
    ("agree_not_extreme", "Agree:\nnot extreme", "#8aa0b4"),
    ("agree_extreme_same_side", "Agree:\nextreme, same side", "#1baf7a"),
    ("opposite_tail", "Both extreme,\nopposite tails", "#eb6834"),
    ("naive_only", "Only the naive forecast", "#2a78d6"),
    ("shrunk_only", "Only the shrunk rate", "#4a3aa7"),
    ("other", "Other", "#0b0b0b"),
)


def naive_shrunk_counts(questions: pd.DataFrame) -> pd.DataFrame:
    """Count agreement between the naive forecast and the shrunk rate.

    A question is included when both the naive forecast and the shrunk rate
    are defined. The raw series rate is not used.

    Args:
      questions (pd.DataFrame): Rows with `naive_side_10` and `shrunk_side_10`.
    """
    naive = questions["naive_side_10"]
    shrunk = questions["shrunk_side_10"]
    kept = naive.notna() & shrunk.notna()
    naive, shrunk = naive[kept], shrunk[kept]
    naive_extreme = naive.isin(["low", "high"])
    shrunk_extreme = shrunk.isin(["low", "high"])
    category = np.select(
        [
            ~naive_extreme & ~shrunk_extreme,
            naive_extreme & shrunk_extreme & (naive == shrunk),
            naive_extreme & shrunk_extreme & (naive != shrunk),
            naive_extreme & ~shrunk_extreme,
            ~naive_extreme & shrunk_extreme,
        ],
        [key for key, _, _ in NAIVE_SHRUNK_BARS[:-1]],
        default="other",
    )
    counts = pd.Series(category).value_counts()
    notes = {
        "agree_extreme_same_side": _side_note(naive, shrunk, (("low", "low"), ("high", "high"))),
        "opposite_tail": _side_note(naive, shrunk, (("high", "low"), ("low", "high"))),
        "naive_only": _side_note(naive, shrunk, (("low", "middle"), ("high", "middle"))),
        "shrunk_only": _side_note(naive, shrunk, (("middle", "low"), ("middle", "high"))),
    }
    rows = []
    for key, label, color in NAIVE_SHRUNK_BARS:
        n = int(counts.get(key, 0))
        if n == 0:
            continue
        rows.append(
            {"category": key, "label": label, "color": color, "n": n, "note": notes.get(key, "")}
        )
    table = pd.DataFrame(rows)
    table["share"] = table["n"] / int(kept.sum())
    return table


def _side_note(naive: pd.Series, shrunk: pd.Series, pairs: tuple) -> str:
    """Describe the largest side pairs inside one agreement category.

    Args:
      naive (pd.Series): Naive side.
      shrunk (pd.Series): Shrunk-rate side.
      pairs (tuple): `(naive side, shrunk side)` pairs, largest first in the note.
    """
    counted = []
    for naive_side, shrunk_side in pairs:
        n = int(((naive == naive_side) & (shrunk == shrunk_side)).sum())
        if n:
            counted.append((n, naive_side, shrunk_side))
    counted.sort(reverse=True)
    phrases = []
    for n, left, right in counted:
        if n < 100:
            continue
        if left == right:
            phrases.append(f"{n:,} both {left}")
        else:
            phrases.append(f"{n:,} naive {left}, shrunk {right}")
    return "; ".join(phrases[:2])


def naive_shrunk_plot(questions: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Bar chart of the naive forecast against the shrunk 10% tag.

    Args:
      questions (pd.DataFrame): Rows with `naive_side_10` and `shrunk_side_10`.
      path (Path): Output file.
    """
    table = naive_shrunk_counts(questions)
    n_questions = int(table["n"].sum())
    y = np.arange(len(table))[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.barh(y, table["n"], color=table["color"], height=0.68)
    labels = [f"{row.label}\n({row.note})" if row.note else row.label for row in table.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    pad = n_questions * 0.012
    for yi, row in zip(y, table.itertuples(index=False)):
        ax.text(
            row.n + pad,
            yi,
            f"{row.n:,}  ({row.share:.1%})",
            va="center",
            ha="left",
            color=plots.INK,
            fontsize=8,
        )
    ax.set_xlim(0, table["n"].max() * 1.32)
    ax.set_xlabel("Dataset questions")
    ax.grid(axis="y", visible=False)
    ax.set_title("Naive forecast vs. shrunk base rate", loc="left")
    fig.text(
        0.01,
        0.01,
        f"10% tails. {n_questions:,} questions with both a naive forecast and a shrunk rate.",
        color=plots.INK_SECONDARY,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plots._save(fig, path)
    return table


# A low tag claims the question resolves no. A high tag claims it resolves yes.
TAIL_CALLS = (("low", "resolves no", 0), ("high", "resolves yes", 1))


def tail_call_accuracy(questions: pd.DataFrame) -> pd.DataFrame:
    """Score each 10% tail call against the resolution.

    The first rows are the naive forecast and the shrunk rate on every question
    where both are defined. The later rows add the raw series rate, which needs
    at least five earlier resolutions, and repeat the other two methods on that
    same set of questions.

    Args:
      questions (pd.DataFrame): Rows with the 10% side columns and `resolved_to`.
    """
    resolved = questions["resolved_to"].isin([0, 1])
    sets = (
        (
            "Naive forecast and shrunk rate both defined",
            (
                ("Naive forecast", "naive_side_10"),
                ("Shrunk base rate", "shrunk_side_10"),
            ),
            questions["naive_side_10"].notna() & questions["shrunk_side_10"].notna() & resolved,
        ),
        (
            "All three defined; raw rate has at least 5 earlier resolutions",
            (
                ("Naive forecast", "naive_side_10"),
                ("Shrunk base rate", "shrunk_side_10"),
                ("Raw base rate", "series_side_10_n5"),
            ),
            questions["naive_side_10"].notna()
            & questions["shrunk_side_10"].notna()
            & questions["series_side_10_n5"].notna()
            & resolved,
        ),
    )
    rows = []
    for question_set, methods, mask in sets:
        kept = questions.loc[mask]
        n_questions = int(mask.sum())
        for call, claim, correct_value in TAIL_CALLS:
            for method, column in methods:
                called = kept[column] == call
                n_calls = int(called.sum())
                came_true = int((kept.loc[called, "resolved_to"] == correct_value).sum())
                precision = came_true / n_calls if n_calls else float("nan")
                rows.append(
                    {
                        "question_set": question_set,
                        "n_questions": n_questions,
                        "call": call,
                        "claim": claim,
                        "method": method,
                        "calls": n_calls,
                        "came_true": came_true,
                        "precision": precision,
                        "error_rate": 1 - precision if n_calls else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _kappa_text(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.2f}"


def _shrinkage_lines(questions: pd.DataFrame) -> str:
    """Describe the fitted shrinkage strength by source.

    Args:
      questions (pd.DataFrame): Rows with `source`, `shrinkage_m`, and `source_yes_rate`.
    """
    lines = []
    for source, group in questions.groupby("source", sort=True):
        known = group["shrinkage_m"].notna()
        m = group.loc[known, "shrinkage_m"]
        mu = group.loc[known, "source_yes_rate"]
        full = int(np.isinf(m).sum())
        finite = m[np.isfinite(m)]
        median_m = "—" if finite.empty else f"{float(finite.median()):.1f}"
        median_mu = "—" if mu.empty else f"{float(mu.median()):.1%}"
        lines.append(
            f"- {source}: median source yes rate {median_mu}. "
            f"Median finite shrinkage strength {median_m}. "
            f"{full} questions are shrunk all the way to the source rate."
        )
    return "\n".join(lines)


def _pct(n: int, d: int) -> str:
    if d == 0:
        return "0"
    return f"{n} ({n / d:.1%})"


def write_summary(
    path: Path,
    questions: pd.DataFrame,
    agreement: pd.DataFrame,
    by_source: pd.DataFrame,
) -> None:
    """Write the methodology comparison.

    Args:
      path (Path): `summary.md`.
      questions (pd.DataFrame): Tagged dataset questions.
      agreement (pd.DataFrame): One row per family, threshold, and minimum history.
      by_source (pd.DataFrame): Series-family agreement by source at the 10% tail.
    """
    n = len(questions)
    capped_low = int((questions["naive_forecast"] <= 0.05).sum())
    capped_high = int((questions["naive_forecast"] >= 0.95).sum())
    naive_side = tail_side(questions["naive_forecast"], HEADLINE_THRESHOLD)
    lines = [
        "# Dataset extreme tags: naive forecast vs. historical base rate",
        "",
        "This comparison does not score any forecaster. It asks whether two tagging",
        "rules pick the same dataset questions as extreme.",
        "",
        "## Definitions",
        "",
        "- A question is extreme when the probability is at most 10%, or at least 90%.",
        "  The 5% tail is reported alongside it.",
        "- Method 1 is the Naive Forecaster's probability on that question. Those",
        "  forecasts are capped at 5% and 95% when they are produced.",
        "- Method 2, the default, is the shrunk yes rate of the same dataset series",
        "  (`source` and `id`). Horizons are pooled. Only resolutions dated before",
        "  the question's forecast due date count. The raw rate k/n is pulled",
        "  toward the source's earlier yes rate mu:",
        "  (k + m × mu) / (n + m). The strength m is estimated from how much",
        "  series rates vary inside the source. Small m means the series differ,",
        "  so a few resolutions already dominate. Large m pulls a thin series",
        "  toward the source rate.",
        "- The raw series rate is still compared below. It tags a question only",
        "  with at least 5 earlier resolutions, or, in the stricter rows, at least",
        "  20. The shrunk rate is defined whenever the source has any earlier",
        "  resolution, including when the series itself has none.",
        "",
        "## Population",
        "",
        f"- {n} dataset questions, from the leaderboard rounds in the forecast cache.",
        f"- Forecast due dates {questions['forecast_due_date'].min().date()} to "
        f"{questions['forecast_due_date'].max().date()}.",
        f"- Imputed naive forecasts: {int(questions['imputed'].fillna(False).sum())}.",
        "",
        "## Naive forecast (method 1), 10% tail",
        "",
        f"- Low: {(naive_side == 'low').sum()}. High: {(naive_side == 'high').sum()}. "
        f"Middle: {(naive_side == 'middle').sum()}.",
        f"- Forecasts sitting on the 5% cap: {capped_low}. On the 95% cap: {capped_high}. "
        "Most naive extreme tags are these capped forecasts, not probabilities between "
        "5% and 10% or between 90% and 95%.",
        "",
        "## How much history method 2 has",
        "",
        f"- Series family (`source`, `id`): "
        f"{_pct(int((questions['n_earlier_series'] == 0).sum()), n)} have no earlier "
        f"resolution, {_pct(int((questions['n_earlier_series'] >= 5).sum()), n)} have at "
        f"least 5, and {_pct(int((questions['n_earlier_series'] >= 20).sum()), n)} have "
        "at least 20.",
        f"- Horizon family (`source`, `id`, `horizon`): "
        f"{_pct(int((questions['n_earlier_horizon'] >= 5).sum()), n)} have at least 5 "
        f"and {_pct(int((questions['n_earlier_horizon'] >= 20).sum()), n)} have at least 20.",
        f"- Shrunk rate defined: {_pct(int(questions['shrunk_base_rate'].notna().sum()), n)}.",
        "",
        "## Shrinkage",
        "",
        _shrinkage_lines(questions),
        "",
        "## Agreement",
        "",
        "Comparable questions are those with a naive forecast and a base rate.",
        "For `series shrunk`, min history 1 means the shrunk rate is defined.",
        "`both_extreme` means both rules call the",
        "question extreme. `opposite_side` means both do, but one says low and the",
        "other says high. Kappa is Cohen's kappa on the extreme / not-extreme label.",
        "",
        "| Family | Tail | Min history | Comparable | Both extreme | Naive only | "
        "Base rate only | Neither | Opposite side | Agreement | Kappa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in agreement.iterrows():
        lines.append(
            f"| {row['family']} | {row['threshold']:.0%} | {int(row['min_n'])} | "
            f"{int(row['n_comparable'])} | {int(row['both_extreme'])} | "
            f"{int(row['naive_only'])} | {int(row['base_only'])} | {int(row['neither'])} | "
            f"{int(row['both_extreme_opposite_side'])} | {row['agreement_rate']:.1%} | "
            f"{_kappa_text(row['kappa'])} |"
        )
    lines += [
        "",
        "## Agreement by source",
        "",
        "10% tail. `series shrunk` is the default tag. `series raw` is the unshrunk rate.",
        "",
        "| Source | Family | Min history | Comparable | Both extreme | Naive only | "
        "Base rate only | Neither | Kappa |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_source.iterrows():
        lines.append(
            f"| {row['source']} | {row['family']} | {int(row['min_n'])} | "
            f"{int(row['n_comparable'])} | {int(row['both_extreme'])} | "
            f"{int(row['naive_only'])} | {int(row['base_only'])} | "
            f"{int(row['neither'])} | {_kappa_text(row['kappa'])} |"
        )
    lines += [
        "",
        "Side-by-side counts are in `tables/side_crosstab_shrunk_10.csv`,",
        "`tables/side_crosstab_series_10_n5.csv`, and",
        "`tables/side_crosstab_series_10_n20.csv`. Naive versus shrunk is in",
        "`tables/naive_vs_shrunk_agreement.csv` and `figures/naive_vs_shrunk_agreement.png`.",
        "How often each tail call resolved as claimed is in `tables/tail_call_accuracy.csv`.",
        "One row per question is in `tables/question_tags.csv`.",
        "",
    ]
    path.write_text("\n".join(lines))


def _source_agreement(
    questions: pd.DataFrame,
    naive_side: pd.Series,
    rate_col: str,
    n_col: str,
    threshold: float,
    min_ns: Iterable[int],
    family: str,
) -> pd.DataFrame:
    rows = []
    for source, group in questions.groupby("source", sort=True):
        for min_n in min_ns:
            row = agreement_row(
                group,
                naive_side.loc[group.index],
                group[rate_col],
                group[n_col],
                threshold,
                min_n,
                family,
            )
            row["source"] = source
            rows.append(row)
    return pd.DataFrame(rows)


def main(config_path: Optional[str] = None) -> Path:
    """Tag dataset questions and write the comparison.

    Args:
      config_path (Optional[str]): Config file. Defaults to `config.toml`.
    """
    config = load_config(
        Path(config_path) if config_path else Path(__file__).parent / "config.toml"
    )
    out = make_output_dir(config, "dataset_tags")
    tables, figures = out / "tables", out / "figures"
    questions = load_dataset_questions(config["data"]["cache_dir"])
    logger.info(f"Loaded {len(questions)} dataset questions.")

    shrunk = shrunk_series_rates(questions)
    horizon = earlier_resolution_stats(questions, HORIZON_FAMILY)
    questions["n_earlier_series"] = shrunk["n_earlier"]
    questions["k_earlier_series"] = shrunk["k_earlier"]
    questions["base_rate_series"] = shrunk["raw_base_rate"]
    questions["source_n"] = shrunk["source_n"]
    questions["source_yes_rate"] = shrunk["source_yes_rate"]
    questions["shrinkage_m"] = shrunk["shrinkage_m"]
    questions["shrunk_base_rate"] = shrunk["shrunk_base_rate"]
    questions["n_earlier_horizon"] = horizon["n_earlier"]
    questions["base_rate_horizon"] = horizon["base_rate"]
    # The shrunk rate is defined from the source even when the series has no history.
    questions["shrunk_history"] = questions["shrunk_base_rate"].notna().astype(int)

    agreement_rows: List[dict] = []
    for family, rate_col, n_col, min_ns in (
        ("series shrunk", "shrunk_base_rate", "shrunk_history", (1,)),
        ("series raw", "base_rate_series", "n_earlier_series", MIN_NS),
        ("horizon raw", "base_rate_horizon", "n_earlier_horizon", MIN_NS),
    ):
        for threshold in THRESHOLDS:
            naive_side = tail_side(questions["naive_forecast"], threshold)
            for min_n in min_ns:
                agreement_rows.append(
                    agreement_row(
                        questions,
                        naive_side,
                        questions[rate_col],
                        questions[n_col],
                        threshold,
                        min_n,
                        family,
                    )
                )
    agreement = pd.DataFrame(agreement_rows)
    naive_side_10 = tail_side(questions["naive_forecast"], HEADLINE_THRESHOLD)
    by_source = pd.concat(
        [
            _source_agreement(
                questions,
                naive_side_10,
                "shrunk_base_rate",
                "shrunk_history",
                HEADLINE_THRESHOLD,
                (1,),
                "series shrunk",
            ),
            _source_agreement(
                questions,
                naive_side_10,
                "base_rate_series",
                "n_earlier_series",
                HEADLINE_THRESHOLD,
                MIN_NS,
                "series raw",
            ),
        ],
        ignore_index=True,
    )

    questions["naive_side_10"] = naive_side_10
    questions["naive_extreme_10"] = naive_side_10.isin(["low", "high"])
    questions["shrunk_side_10"] = tail_side(questions["shrunk_base_rate"], HEADLINE_THRESHOLD)
    questions["shrunk_extreme_10"] = questions["shrunk_side_10"].isin(["low", "high"])
    for min_n in MIN_NS:
        side = tail_side(questions["base_rate_series"], HEADLINE_THRESHOLD).where(
            questions["n_earlier_series"] >= min_n
        )
        questions[f"series_side_10_n{min_n}"] = side
        extreme = side.isin(["low", "high"]).astype("boolean")
        questions[f"series_extreme_10_n{min_n}"] = extreme.mask(side.isna())

    questions.to_csv(tables / "question_tags.csv", index=False)
    agreement.to_csv(tables / "agreement.csv", index=False)
    by_source.to_csv(tables / "agreement_by_source.csv", index=False)
    for min_n in MIN_NS:
        base_side = questions[f"series_side_10_n{min_n}"]
        comparable = questions["naive_side_10"].notna() & base_side.notna()
        side_crosstab(questions.loc[comparable, "naive_side_10"], base_side[comparable]).to_csv(
            tables / f"side_crosstab_series_10_n{min_n}.csv"
        )
    shrunk_side = questions["shrunk_side_10"]
    comparable = questions["naive_side_10"].notna() & shrunk_side.notna()
    side_crosstab(questions.loc[comparable, "naive_side_10"], shrunk_side[comparable]).to_csv(
        tables / "side_crosstab_shrunk_10.csv"
    )
    agreement_plot(
        questions,
        "shrunk_base_rate",
        "Shrunk series base rate",
        f"Naive forecast vs. shrunk base rate\nDashed lines are the {HEADLINE_THRESHOLD:.0%} tails",
        figures / "naive_vs_shrunk_base_rate.png",
    )
    agreement_plot(
        questions,
        "base_rate_series",
        "Raw series base rate, from earlier resolutions",
        f"Dataset questions with at least 5 earlier resolutions\n"
        f"Dashed lines are the {HEADLINE_THRESHOLD:.0%} tails",
        figures / "naive_vs_base_rate.png",
        min_n=5,
    )
    naive_shrunk_plot(questions, figures / "naive_vs_shrunk_agreement.png").to_csv(
        tables / "naive_vs_shrunk_agreement.csv", index=False
    )
    tail_call_accuracy(questions).to_csv(tables / "tail_call_accuracy.csv", index=False)
    write_summary(out / "summary.md", questions, agreement, by_source)
    logger.info(f"Done. Outputs in {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.toml"))
    main(parser.parse_args().config)
