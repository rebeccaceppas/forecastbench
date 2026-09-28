"""Render the SOTA-over-time figure from a leaderboard's SOTA graph CSV.

The Explore page on the website draws this chart with d3 from `sota_graph_<type>.csv` and
`parity_dates.<stem>.json`. Only the baseline and tournament leaderboards are published
there, so this module redraws the same chart locally with matplotlib for any leaderboard
type, including the low-probability and extreme boards.
"""

import json
import logging
import os
import sys
from datetime import timedelta
from itertools import cycle
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from termcolor import colored  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from helpers import data_utils, decorator, env  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_FOLDER = "sota_graphs"

SCORE_COLUMNS = {
    "overall": {"score": "Overall", "ci": "Overall 95% CI", "n": "N"},
    "market": {"score": "Market", "ci": "Market 95% CI", "n": "N market"},
    "dataset": {"score": "Dataset", "ci": "Dataset 95% CI", "n": "N dataset"},
}

# Reference-line models, in the order they are labelled on the chart. Keys are the model
# names as they appear in the `Model` column of the SOTA graph CSV.
COMPARISON_MODELS = {
    "Superforecaster median forecast": "Superforecaster",
    "Public median forecast": "Public",
    "Imputed Forecaster": "Imputed Forecaster",
    "Naive Forecaster": "Naive Forecaster",
    "Always 0": "Always 0",
    "Always 0.5": "Always 0.5",
    "Always 1": "Always 1",
    "Random Uniform": "Random Uniform",
}

SOTA_COLOR = "#e8743b"
MODEL_COLOR = "#9fb0c4"
REFERENCE_COLOR = "#8a8a8a"
COMPARISON_STYLES = {
    "Superforecaster": ("#1f4e79", "--"),
    "Public": ("#4a7ebb", "--"),
    "Imputed Forecaster": ("#3f7d4f", (0, (4, 2))),
    "Naive Forecaster": ("#7fa87f", (0, (1, 2))),
    "Always 0": ("#b5838d", "-."),
    "Always 0.5": ("#8a8a8a", "-."),
    "Always 1": ("#c1121f", (0, (3, 4))),
    "Random Uniform": ("#9c6644", (0, (1, 3))),
}
DAYS_PER_MONTH = 30.0
MIN_SOTA_MODELS_FOR_TREND = 3


def parse_confidence_interval(value: object) -> Tuple[float, float]:
    """Parse a `[lower, upper]` confidence interval string.

    Args:
        value (object): Cell from a 95% CI column.

    Returns:
        Tuple[float, float]: Lower and upper bound, NaN when unparseable.
    """
    if not isinstance(value, str):
        return (np.nan, np.nan)
    match = pd.Series([value]).str.extract(r"\[\s*([^,]+),\s*([^\]]+)\]")
    try:
        return (float(match.iloc[0, 0]), float(match.iloc[0, 1]))
    except (TypeError, ValueError):
        return (np.nan, np.nan)


def select_score(df: pd.DataFrame, score_type: str) -> pd.DataFrame:
    """Reshape the SOTA graph CSV to one row per model for a single score column.

    Args:
        df (pd.DataFrame): Contents of `sota_graph_<type>.csv`.
        score_type (str): One of "overall", "market" or "dataset".

    Returns:
        pd.DataFrame: Columns model, score, ci_lower, ci_upper, n, release_date.
    """
    if score_type not in SCORE_COLUMNS:
        raise ValueError(f"score_type must be one of {sorted(SCORE_COLUMNS)}, got {score_type}.")

    cols = SCORE_COLUMNS[score_type]
    bounds = [parse_confidence_interval(value) for value in df[cols["ci"]]]
    df_scores = pd.DataFrame(
        {
            "model": df["Model"],
            "score": pd.to_numeric(df[cols["score"]], errors="coerce"),
            "ci_lower": [bound[0] for bound in bounds],
            "ci_upper": [bound[1] for bound in bounds],
            "n": pd.to_numeric(df[cols["n"]], errors="coerce"),
            "release_date": pd.to_datetime(df["Model release date"], errors="coerce"),
        }
    )
    return df_scores.dropna(subset=["score"]).reset_index(drop=True)


def comparison_scores(df_scores: pd.DataFrame) -> Dict[str, float]:
    """Pull the reference-line scores out of the reshaped scores.

    Args:
        df_scores (pd.DataFrame): Output of `select_score`.

    Returns:
        Dict[str, float]: Reference label to score, for the comparison models present.
    """
    by_model = df_scores.drop_duplicates(subset=["model"]).set_index("model")["score"]
    return {
        label: float(by_model[model])
        for model, label in COMPARISON_MODELS.items()
        if model in by_model.index and pd.notna(by_model[model])
    }


def model_points(df_scores: pd.DataFrame) -> pd.DataFrame:
    """Keep the dated forecaster models that are drawn as points.

    Comparison models have no release date and are drawn as reference lines instead.

    Args:
        df_scores (pd.DataFrame): Output of `select_score`.

    Returns:
        pd.DataFrame: Dated model rows sorted by release date.
    """
    df = df_scores[df_scores["release_date"].notna()].copy()
    df = df[~df["model"].isin(COMPARISON_MODELS)]
    return df.sort_values("release_date", ignore_index=True)


def mark_sota(df_points: pd.DataFrame) -> pd.DataFrame:
    """Flag the models that were state of the art on their release date.

    A model is SOTA when it is the best-scoring model released that day and it beats every
    model released earlier, which matches the rule the website uses.

    Args:
        df_points (pd.DataFrame): Output of `model_points`.

    Returns:
        pd.DataFrame: Copy with an `is_sota` column.
    """
    df = df_points.copy()
    df["is_sota"] = False
    best = -np.inf
    for _, group in df.groupby(df["release_date"].dt.date, sort=True):
        group = group.sort_values(["score", "model"], ascending=[False, True])
        index = group.index[0]
        if df.loc[index, "score"] > best:
            df.loc[index, "is_sota"] = True
            best = df.loc[index, "score"]
    return df


def fit_sota_trend(df_sota: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Fit the linear SOTA trend in Brier Index points per month.

    Args:
        df_sota (pd.DataFrame): SOTA rows with `release_date` and `score`.

    Returns:
        Optional[Dict[str, float]]: Intercept, slope and trend origin, None when there are
            too few SOTA models to fit a line.
    """
    if len(df_sota) < MIN_SOTA_MODELS_FOR_TREND:
        return None

    df = df_sota.sort_values("release_date")
    first_date = df["release_date"].min()
    months = (df["release_date"] - first_date).dt.days / DAYS_PER_MONTH
    if months.nunique() < 2:
        return None

    slope, intercept = np.polyfit(months, df["score"], 1)
    return {"intercept": float(intercept), "slope": float(slope), "first_date": first_date}


def parity_achieved_date(df_points: pd.DataFrame, superforecaster_score: float) -> Optional[str]:
    """Return the first release date at which a model matched the superforecasters.

    Args:
        df_points (pd.DataFrame): Output of `model_points`.
        superforecaster_score (float): Superforecaster Brier Index.

    Returns:
        Optional[str]: Formatted date, None when no model has reached parity.
    """
    reached = df_points[df_points["score"] >= superforecaster_score]
    if reached.empty:
        return None
    return reached["release_date"].min().strftime("%B %d, %Y")


def parity_label(
    df_points: pd.DataFrame,
    references: Dict[str, float],
    projected: Optional[dict],
) -> Optional[str]:
    """Build the parity annotation shown in the bottom-right of the chart.

    Args:
        df_points (pd.DataFrame): Output of `model_points`.
        references (Dict[str, float]): Reference-line scores.
        projected (Optional[dict]): Parity date summary for this score type, may be None.

    Returns:
        Optional[str]: Annotation text, None when there is nothing to report.
    """
    if "Superforecaster" not in references:
        return None

    achieved = parity_achieved_date(df_points, references["Superforecaster"])
    if achieved:
        return f"LLM-superforecaster parity achieved: {achieved}"
    if projected:
        return (
            f"Projected LLM-superforecaster parity: {projected['intersection']}\n"
            f"(95% CI: {projected['lower']} – {projected['upper']})"
        )
    return None


def label_positions(df_sota: pd.DataFrame, x_span_days: float) -> List[bool]:
    """Decide which SOTA points get a text label, dropping ones that would overlap.

    Args:
        df_sota (pd.DataFrame): SOTA rows sorted by release date.
        x_span_days (float): Width of the x axis in days.

    Returns:
        List[bool]: One flag per row.
    """
    min_gap = x_span_days * 0.09
    keep = []
    last = None
    for release_date in df_sota["release_date"]:
        if last is None or (release_date - last).days >= min_gap:
            keep.append(True)
            last = release_date
        else:
            keep.append(False)
    return keep


def short_label(model: str) -> str:
    """Shorten a model name for the in-chart SOTA label.

    Args:
        model (str): Full model name.

    Returns:
        str: Truncated name without the trailing parenthetical.
    """
    base = model.split(" (")[0].strip()
    return base if len(base) <= 22 else f"{base[:21]}…"


def render_figure(
    df_scores: pd.DataFrame,
    score_type: str,
    title: str,
    projected_parity: Optional[dict],
    ylabel: str = "Brier Index (%)",
    score_decimals: int = 1,
    xmin: Optional[pd.Timestamp] = None,
    ymin: Optional[float] = None,
) -> plt.Figure:
    """Draw the SOTA figure for one score type.

    Args:
        df_scores (pd.DataFrame): Output of `select_score`.
        score_type (str): One of "overall", "market" or "dataset".
        title (str): Figure title.
        projected_parity (Optional[dict]): Parity date summary for this score type.
        ylabel (str): Vertical axis label.
        score_decimals (int): Decimal places for reference-line labels.
        xmin (Optional[pd.Timestamp]): Left edge of the release-date axis.
        ymin (Optional[float]): Lower edge of the score axis.

    Returns:
        plt.Figure: The rendered figure.
    """
    references = comparison_scores(df_scores)
    df_points = mark_sota(model_points(df_scores))
    df_sota = df_points[df_points["is_sota"]].reset_index(drop=True)
    df_rest = df_points[~df_points["is_sota"]]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.scatter(df_rest["release_date"], df_rest["score"], s=24, color=MODEL_COLOR, zorder=2)

    trend = fit_sota_trend(df_sota)
    if trend is not None:
        span = [df_sota["release_date"].min(), df_points["release_date"].max()]
        months = [(date - trend["first_date"]).days / DAYS_PER_MONTH for date in span]
        ax.plot(
            span,
            [trend["intercept"] + trend["slope"] * month for month in months],
            linestyle="--",
            color=SOTA_COLOR,
            linewidth=1.5,
            zorder=3,
            label=f"Linear trend ({trend['slope']:+.2f} pts/month)",
        )

    ax.errorbar(
        df_sota["release_date"],
        df_sota["score"],
        yerr=_error_bar_lengths(df_sota),
        fmt="o",
        markersize=5,
        color=SOTA_COLOR,
        ecolor=SOTA_COLOR,
        elinewidth=1,
        capsize=3,
        zorder=4,
        label="SOTA model",
    )
    ax.scatter([], [], s=24, color=MODEL_COLOR, label="Non-SOTA model")

    visible_points = df_points if xmin is None else df_points[df_points["release_date"] >= xmin]
    if visible_points.empty:
        visible_points = df_points
    keep_labels = label_positions(df_sota, _x_span_days(visible_points))
    for (_, row), keep, offset in zip(df_sota.iterrows(), keep_labels, cycle([9, 21])):
        if not keep:
            continue
        top = row["ci_upper"] if pd.notna(row["ci_upper"]) else row["score"]
        ax.annotate(
            short_label(row["model"]),
            (row["release_date"], top),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            fontsize=8,
        )

    # The comparison models sit within a point or two of each other on low-probability
    # questions, so they are named in a side legend rather than labelled on their lines.
    reference_handles = []
    for label, value in references.items():
        color, dashes = COMPARISON_STYLES.get(label, (REFERENCE_COLOR, "--"))
        line = ax.axhline(
            value,
            color=color,
            linestyle=dashes,
            linewidth=1.0,
            zorder=1,
            label=f"{label} ({value:.{score_decimals}f})",
        )
        reference_handles.append(line)

    y_low, y_high = _y_limits(df_points, references)
    if ymin is not None:
        y_low = ymin
    ax.set_ylim(y_low, y_high)
    if xmin is not None:
        right = pd.to_datetime(df_points["release_date"].max()).to_pydatetime() + timedelta(
            days=31
        )
        ax.set_xlim(pd.to_datetime(xmin).to_pydatetime(), right)
    ax.set_xlabel("Model release date")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — {score_type} questions")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(False)

    model_handles = [handle for handle in ax.get_legend_handles_labels()[0]]
    model_handles = [handle for handle in model_handles if handle not in reference_handles]
    ax.add_artist(ax.legend(handles=model_handles, loc="lower left", fontsize=8, framealpha=0.9))
    ax.legend(
        handles=reference_handles,
        title="Comparisons",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        title_fontsize=8,
        framealpha=0.9,
    )

    annotation = parity_label(df_points, references, projected_parity)
    if annotation:
        ax.annotate(
            annotation,
            (0.985, 0.03),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc"},
        )

    fig.tight_layout()
    return fig


def _error_bar_lengths(df_sota: pd.DataFrame) -> np.ndarray:
    """Return the lower and upper error bar lengths for the SOTA points.

    A bootstrap interval can sit entirely on one side of the point estimate, and a model
    that forecast on a single round has no interval at all, so lengths are floored at zero.

    Args:
        df_sota (pd.DataFrame): SOTA rows with `score`, `ci_lower` and `ci_upper`.

    Returns:
        np.ndarray: Two rows of error bar lengths, lower then upper.
    """
    lower = (df_sota["score"] - df_sota["ci_lower"]).fillna(0).clip(lower=0)
    upper = (df_sota["ci_upper"] - df_sota["score"]).fillna(0).clip(lower=0)
    return np.vstack([lower, upper])


def _x_span_days(df_points: pd.DataFrame) -> float:
    """Return the number of days covered by the dated models.

    Args:
        df_points (pd.DataFrame): Output of `model_points`.

    Returns:
        float: Span in days, at least 1.
    """
    span = (df_points["release_date"].max() - df_points["release_date"].min()).days
    return float(max(span, 1))


def _y_limits(df_points: pd.DataFrame, references: Dict[str, float]) -> Tuple[float, float]:
    """Return y limits that fit the points, their CIs and the reference lines.

    Args:
        df_points (pd.DataFrame): Output of `model_points`.
        references (Dict[str, float]): Reference-line scores.

    Returns:
        Tuple[float, float]: Lower and upper limit.
    """
    values = pd.concat([df_points["score"], df_points["ci_lower"], df_points["ci_upper"]]).dropna()
    values = pd.concat([values, pd.Series(list(references.values()))])
    low, high = float(values.min()), float(values.max())
    pad = max((high - low) * 0.08, 1.0)
    return (low - pad, high + pad)


def read_projected_parity_dates(directory: str, leaderboard_type: str) -> Dict[str, Optional[dict]]:
    """Read the projected parity dates written by the leaderboard run.

    Args:
        directory (str): Public release mount directory.
        leaderboard_type (str): Leaderboard type value, e.g. `low_probability_tournament`.

    Returns:
        Dict[str, Optional[dict]]: Score type to parity summary, empty when the file is absent.
    """
    path = f"{directory}/simulated_llm_parity/parity_dates.{leaderboard_type}_leaderboard.json"
    if not os.path.exists(path):
        logger.warning(f"No parity dates at {path}; drawing the figure without them.")
        return {}

    with open(path, "r", encoding="utf-8") as file:
        parity_dates = json.load(file)
    return {
        score_type: (leaderboards or {}).get(leaderboard_type)
        for score_type, leaderboards in parity_dates.items()
    }


@decorator.log_runtime
def driver() -> None:
    """Render SOTA figures for the leaderboard named by `LEADERBOARD_TYPE`.

    Returns:
        None.
    """
    leaderboard_type = os.environ.get("LEADERBOARD_TYPE", "low_probability_tournament")
    score_types = [
        score_type.strip()
        for score_type in os.environ.get("SCORE_TYPES", "overall,market,dataset").split(",")
        if score_type.strip()
    ]

    score_kind = os.environ.get("SCORE_KIND", "brier_index")
    directory = data_utils.get_mounted_bucket(bucket=env.PUBLIC_RELEASE_BUCKET)
    source_name = f"sota_graph_{leaderboard_type}"
    ylabel = "Brier Index (%)"
    score_decimals = 1
    file_label = leaderboard_type
    xmin = None
    ymin = None
    if score_kind == "log_score":
        source_name = f"{source_name}_log_score"
        ylabel = "Log score (bits vs Always 0.5)"
        score_decimals = 3
        file_label = f"{leaderboard_type}_log_score"
        if leaderboard_type == "extreme_tournament":
            xmin = pd.Timestamp("2024-04-01")
            ymin = -2.0
    elif score_kind != "brier_index":
        raise ValueError("SCORE_KIND must be 'brier_index' or 'log_score'.")

    source = f"{directory}/leaderboards/csv/{source_name}.csv"
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"{source} not found. Run `LEADERBOARD_TYPE={leaderboard_type} python main.py` first."
        )

    df = pd.read_csv(source)
    projected_parity = {}
    if score_kind == "brier_index":
        projected_parity = read_projected_parity_dates(directory, leaderboard_type)
    title = leaderboard_type.replace("_", " ").title()
    if score_kind == "log_score":
        title = f"{title} log score"

    output_dir = f"{directory}/{OUTPUT_FOLDER}"
    os.makedirs(output_dir, exist_ok=True)
    for score_type in score_types:
        figure = render_figure(
            df_scores=select_score(df, score_type),
            score_type=score_type,
            title=title,
            projected_parity=projected_parity.get(score_type),
            ylabel=ylabel,
            score_decimals=score_decimals,
            xmin=xmin,
            ymin=ymin,
        )
        destination = f"{output_dir}/sota_graph_{file_label}_{score_type}.png"
        figure.savefig(destination, dpi=200)
        plt.close(figure)
        logger.info(colored(f"Wrote {destination}", "red"))


if __name__ == "__main__":
    driver()
