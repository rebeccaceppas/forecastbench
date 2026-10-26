"""Figures for the market low-probability analysis.

Colors follow a fixed categorical order (never cycled); reference lines are direct-labeled so
identity never rests on color alone.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"
MODEL_POINT = "#b9c3cf"
REFERENCE_STYLES = ["--", "-.", ":", (0, (5, 2, 1, 2)), (0, (1, 1)), (0, (8, 3))]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "legend.frameon": False,
    }
)

RULE_LABELS = {"brier": "Brier", "log": "log score"}
# ForecastBench points use `errorbar`, whose size is a diameter in points. Scatter sizes are
# areas, so external submissions use the squared diameter to match those points.
LLM_MARKER_PT = 4.5


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _spread(values: List[float], min_gap: float) -> List[float]:
    """Return label positions near `values` that are at least `min_gap` apart, in order."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    shift = (placed - np.array(values, dtype=float)[order]).mean() / 2
    out = np.empty_like(placed)
    out[order] = placed - shift
    return out.tolist()


def _reference_lines(
    ax: plt.Axes, values: Dict[str, float], x_right: float, emphasis: Optional[str] = None
) -> None:
    """Draw labeled horizontal reference lines, with labels spread apart at the right edge.

    Args:
      ax (plt.Axes): Axes.
      values (Dict[str, float]): Label to y value.
      x_right (float): x position (data units) the labels start from.
      emphasis (Optional[str]): Label drawn as a solid, bold reference (the zero line).
    """
    values = {k: v for k, v in values.items() if v is not None and np.isfinite(v)}
    if not values:
        return
    ax.figure.canvas.draw()
    to_px = ax.transData.transform
    pixels = [to_px((x_right, v))[1] for v in values.values()]
    min_gap = 10 * ax.figure.dpi / 72
    label_px = _spread(pixels, min_gap)
    inverse = ax.transData.inverted()
    style_index = 0
    for (name, value), y_px in zip(values.items(), label_px):
        if name == emphasis:
            ax.axhline(value, color=INK, lw=1.2, zorder=1)
        else:
            ls = REFERENCE_STYLES[style_index % len(REFERENCE_STYLES)]
            style_index += 1
            ax.axhline(value, color=INK_SECONDARY, lw=1.1, ls=ls, zorder=1)
        y_label = inverse.transform((0, y_px))[1]
        ax.annotate(
            name,
            xy=(x_right, value),
            xytext=(x_right, y_label),
            textcoords="data",
            va="center",
            fontsize=7.5,
            color=INK if name == emphasis else INK_SECONDARY,
            fontweight="bold" if name == emphasis else "normal",
            annotation_clip=False,
        )


def _date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _release_window(
    df: pd.DataFrame, min_date: Optional[str]
) -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    """Return rows whose release date is on or after `min_date`.

    Args:
      df (pd.DataFrame): Rows with `model_release_date`.
      min_date (Optional[str]): Inclusive lower bound, YYYY-MM-DD, or None for no bound.
    """
    min_ts = pd.Timestamp(min_date) if min_date else None
    if df.empty or min_ts is None:
        return df, min_ts
    keep = pd.to_datetime(df["model_release_date"]) >= min_ts
    return df.loc[keep].copy(), min_ts


def monthly_llm_trend(llms: pd.DataFrame, col: str, min_date: Optional[str] = None) -> pd.DataFrame:
    """Median and interquartile range of ForecastBench LLMs by release month.

    External submissions are not included. Months with fewer than four models get a median
    and a count, and no interquartile range.

    Args:
      llms (pd.DataFrame): ForecastBench LLM rows with `model_release_date` and `col`.
      col (str): Score difference. Positive means better than the reference.
      min_date (Optional[str]): Inclusive release-date bound, YYYY-MM-DD.
    """
    empty = pd.DataFrame(columns=["month", "date", "n", "median", "q25", "q75"])
    window, _ = _release_window(llms, min_date)
    df = window.dropna(subset=["model_release_date", col]).copy()
    if df.empty:
        return empty
    df["month"] = pd.to_datetime(df["model_release_date"]).dt.to_period("M")
    rows = []
    for month, group in df.groupby("month", sort=True):
        values = group[col].astype(float)
        n = int(values.notna().sum())
        q25 = q75 = np.nan
        if n >= 4:
            q25 = float(values.quantile(0.25))
            q75 = float(values.quantile(0.75))
        rows.append(
            {
                "month": str(month),
                "date": month.to_timestamp() + pd.Timedelta(days=14),
                "n": n,
                "median": float(values.median()),
                "q25": q25,
                "q75": q75,
            }
        )
    return pd.DataFrame(rows)


def _month_segments(trend: pd.DataFrame) -> List[pd.DataFrame]:
    """Split a monthly trend where consecutive rows skip a calendar month."""
    segments: List[pd.DataFrame] = []
    current: List[dict] = []
    previous = None
    for _, row in trend.iterrows():
        period = pd.Period(row["month"], freq="M")
        if previous is not None and period != previous + 1 and current:
            segments.append(pd.DataFrame(current))
            current = []
        current.append(row.to_dict())
        previous = period
    if current:
        segments.append(pd.DataFrame(current))
    return segments


def trend_plot(
    llms: pd.DataFrame,
    externals: pd.DataFrame,
    references: Dict[str, float],
    col: str,
    reference_name: str,
    rule: str,
    title: str,
    path: Path,
    min_date: Optional[str] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    y_caption: Optional[str] = None,
) -> None:
    """Score difference vs. the reference forecaster against model release date.

    ForecastBench LLMs are drawn individually and summarized by a monthly median. External
    submissions are drawn at their first due date and are not part of that median.

    Args:
      llms (pd.DataFrame): ForecastBench LLMs with `model_release_date` and `col`.
      externals (pd.DataFrame): External submissions, placed at their first due date.
      references (Dict[str, float]): Comparison forecasters on the same scale.
      col (str): Column holding the difference vs. the reference (positive = better).
      reference_name (str): Name of the reference forecaster (the zero line).
      rule (str): "brier" or "log".
      title (str): Figure title.
      path (Path): Output file.
      min_date (Optional[str]): Inclusive left edge of the release-date axis, YYYY-MM-DD.
      y_min (Optional[float]): Lower y limit. Used only together with `y_max`.
      y_max (Optional[float]): Upper y limit. Used only together with `y_min`.
      y_caption (Optional[str]): Y-axis label. The market wording is used when this is omitted.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2))
    llms_draw, min_ts = _release_window(llms, min_date)
    externals_draw, _ = _release_window(externals, min_date)
    trend = monthly_llm_trend(llms_draw, col)

    plotted_dates = []
    if not llms_draw.empty:
        dates = pd.to_datetime(llms_draw["model_release_date"])
        plotted_dates.append(dates)
        ax.scatter(
            dates,
            llms_draw[col],
            s=LLM_MARKER_PT**2,
            color=MODEL_POINT,
            edgecolor="white",
            linewidths=0.4,
            zorder=2,
            label="ForecastBench LLM",
        )
    if not externals_draw.empty:
        plotted_dates.append(pd.to_datetime(externals_draw["model_release_date"]))
        ax.scatter(
            pd.to_datetime(externals_draw["model_release_date"]),
            externals_draw[col],
            s=LLM_MARKER_PT**2,
            facecolor="white",
            edgecolor=SERIES[1],
            linewidths=0.8,
            zorder=3,
            label="External submission (not in the trend)",
        )
    labeled_band = False
    labeled_line = False
    for segment in _month_segments(trend):
        if segment["q25"].notna().any():
            ax.fill_between(
                pd.to_datetime(segment["date"]),
                segment["q25"],
                segment["q75"],
                color=SERIES[0],
                alpha=0.18,
                lw=0,
                zorder=1,
                label="Interquartile range" if not labeled_band else None,
            )
            labeled_band = True
        ax.plot(
            pd.to_datetime(segment["date"]),
            segment["median"],
            color=SERIES[0],
            lw=2,
            marker="o",
            ms=5.5,
            zorder=4,
            label="Monthly median, ForecastBench LLMs" if not labeled_line else None,
        )
        labeled_line = True
    label_offset = 8
    previous_date = None
    for _, row in trend.iterrows():
        date = pd.to_datetime(row["date"])
        if previous_date is not None and date - previous_date < pd.Timedelta(days=50):
            label_offset = 20 if label_offset == 8 else 8
        else:
            label_offset = 8
        previous_date = date
        ax.annotate(
            f"n={int(row['n'])}",
            xy=(date, row["median"]),
            xytext=(0, label_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
            color=INK_MUTED,
            zorder=6,
        )

    all_dates = (
        pd.concat(plotted_dates, ignore_index=True)
        if plotted_dates
        else pd.Series(dtype="datetime64[ns]")
    )
    if all_dates.empty:
        right = min_ts if min_ts is not None else pd.Timestamp("2025-07-01")
        left = right
    else:
        right = all_dates.max()
        left = min_ts if min_ts is not None else all_dates.min()
    span = right - left
    if not isinstance(span, pd.Timedelta) or span <= pd.Timedelta(0):
        span = pd.Timedelta(days=30)
    left_pad = pd.Timedelta(0) if min_ts is not None else span * 0.03
    ax.set_xlim(left - left_pad, right + span * 0.04)
    x_right = mdates.date2num(right + span * 0.055)
    _date_axis(ax)
    _reference_lines(ax, {reference_name: 0.0, **references}, x_right, emphasis=reference_name)
    ax.set_xlabel("Model release date")
    ax.set_ylabel(
        y_caption
        or (
            f"Difficulty-adjusted {RULE_LABELS[rule]}: {reference_name} minus model\n"
            "(positive = better than the market)"
        )
    )
    ax.set_title(title, loc="left")
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5)
    _save(fig, path)


def extreme_vs_nonextreme_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    rule: str,
    reference_name: str,
    n_labels: int,
    path: Path,
) -> None:
    """Each model's score on non-extreme vs. extreme market questions.

    Args:
      df (pd.DataFrame): One row per model with `x_col`, `y_col`, `model`, `kind`
        ("LLM", "External" or "Comparison").
      x_col (str): Difference vs. reference on non-extreme questions.
      y_col (str): Difference vs. reference on extreme questions.
      rule (str): "brier" or "log".
      reference_name (str): Reference forecaster.
      n_labels (int): Points to label.
      path (Path): Output file.
    """
    fig, ax = plt.subplots(figsize=(6.6, 6))
    styles = {
        "LLM": dict(color=SERIES[0], marker="o", s=26),
        "External": dict(facecolor="white", edgecolor=SERIES[1], marker="o", s=34, lw=1.4),
        "Comparison": dict(color=SERIES[2], marker="D", s=34),
    }
    for kind, style in styles.items():
        rows = df[df["kind"] == kind]
        if rows.empty:
            continue
        ax.scatter(rows[x_col], rows[y_col], label=kind, zorder=3, alpha=0.9, **style)
    lims = [
        np.nanmin([df[x_col].min(), df[y_col].min()]),
        np.nanmax([df[x_col].max(), df[y_col].max()]),
    ]
    pad = (lims[1] - lims[0]) * 0.05 or 0.01
    lims = [lims[0] - pad, lims[1] + pad]
    ax.plot(lims, lims, color=INK_MUTED, lw=1, ls="--", zorder=1)
    ax.annotate("equal on both", xy=(lims[1], lims[1]), fontsize=7, color=INK_MUTED, ha="right")
    ax.axhline(0, color=INK, lw=0.9)
    ax.axvline(0, color=INK, lw=0.9)
    to_label = pd.concat(
        [df[df["kind"] == "Comparison"], df[df["kind"] != "Comparison"].nlargest(n_labels, y_col)]
    )
    for _, row in to_label.iterrows():
        ax.annotate(
            row["model"],
            xy=(row[x_col], row[y_col]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
            color=INK_SECONDARY,
        )
    ax.set_xlabel(f"Non-extreme market questions: {reference_name} minus model")
    ax.set_ylabel(f"Extreme market questions: {reference_name} minus model")
    ax.set_title(
        f"Extreme vs. non-extreme market questions (difficulty-adjusted {RULE_LABELS[rule]})",
        loc="left",
    )
    ax.legend(loc="upper left", fontsize=7.5)
    _save(fig, path)


def calibration_plot(
    calibration: pd.DataFrame, order: List[str], path: Path, min_bin_n: int = 10
) -> None:
    """Tail calibration on the flipped scale, both axes in log-odds.

    Bins with fewer than `min_bin_n` forecasts are left off the plot (they stay in the table).

    Args:
      calibration (pd.DataFrame): Rows per forecaster x bin with `forecaster`, `mean_forecast`,
        `yes_rate`, `ci_lower`, `ci_upper`, `n`.
      order (List[str]): Forecasters in color order.
      path (Path): Output file.
      min_bin_n (int): Smallest bin drawn.
    """
    calibration = calibration[calibration["n"] >= min_bin_n]
    floor = 1e-3
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([floor, 0.6], [floor, 0.6], color=INK_MUTED, lw=1, ls="--", zorder=1)
    ax.annotate("perfect calibration", xy=(0.3, 0.36), fontsize=7, color=INK_MUTED, rotation=38)
    for i, name in enumerate(order):
        rows = calibration[calibration["forecaster"] == name].sort_values("mean_forecast")
        if rows.empty:
            continue
        color = SERIES[i % len(SERIES)]
        x = rows["mean_forecast"].clip(floor, 1 - floor)
        y = rows["yes_rate"].clip(floor, 1 - floor)
        yerr = np.vstack(
            [
                (y - rows["ci_lower"].clip(floor, 1 - floor)).clip(lower=0).to_numpy(),
                (rows["ci_upper"].clip(floor, 1 - floor) - y).clip(lower=0).to_numpy(),
            ]
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            lw=2,
            marker="o",
            ms=5,
            mec="white",
            mew=1,
            elinewidth=0.8,
            capsize=0,
            label=name,
            zorder=3,
        )
    ax.set_xscale("logit")
    ax.set_yscale("logit")
    ax.set_xlim(floor * 0.8, 0.7)
    ax.set_ylim(floor * 0.8, 0.7)
    ax.set_xlabel("Mean forecast in bin (flipped onto the low side)")
    ax.set_ylabel(f"Realized yes rate (Wilson 95% CI; 0 shown at {floor:.1%})")
    ax.set_title(f"Tail calibration on the market subset (bins with N ≥ {min_bin_n})", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    _save(fig, path)


def downward_curve_checks_plot(
    rounds: pd.DataFrame,
    extremity: pd.DataFrame,
    market_median: float,
    restricted: Optional[Dict[str, pd.DataFrame]],
    col: str,
    min_resolved_share: float,
    path: Path,
    suptitle: str = "Why does the SOTA curve slope down on extreme market questions?",
    panel1_title: str = "1. Surprise rate of resolved subset questions",
    yes_ylabel: str = "Share resolving against the price (flipped yes)",
    extremity_ylabel: str = "Median forecast on subset (flipped)",
) -> None:
    """Four diagnostics for the downward SOTA curve (plan section 4.4).

    Args:
      rounds (pd.DataFrame): Per round: `forecast_due_date`, `yes_rate`, `ci_lower`,
        `ci_upper`, `n_resolved`, `resolved_share`, `n_tagged`, `in_leaderboard`.
      extremity (pd.DataFrame): Per LLM: `model_release_date`, `median_forecast_f`.
      market_median (float): Median flipped market price on the subset.
      restricted (Optional[Dict[str, pd.DataFrame]]): Frontiers keyed by label, each with
        `model_release_date` and `col`.
      col (str): Frontier score column.
      min_resolved_share (float): Threshold used for check 4.
      path (Path): Output file.
      suptitle (str): Figure title.
      panel1_title (str): Title of the yes-rate panel.
      yes_ylabel (str): Y-axis label of the yes-rate panel.
      extremity_ylabel (str): Y-axis label of the forecast-extremity panel.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    dates = pd.to_datetime(rounds["forecast_due_date"])

    ax = axes[0, 0]
    scored = rounds[rounds["n_resolved"] > 0]
    ax.errorbar(
        pd.to_datetime(scored["forecast_due_date"]),
        scored["yes_rate"],
        yerr=np.vstack(
            [
                (scored["yes_rate"] - scored["ci_lower"]).clip(lower=0).to_numpy(),
                (scored["ci_upper"] - scored["yes_rate"]).clip(lower=0).to_numpy(),
            ]
        ),
        fmt="o-",
        color=SERIES[0],
        lw=2,
        ms=5,
        elinewidth=0.8,
        mec="white",
    )
    for _, row in scored.iterrows():
        ax.annotate(
            f"n={int(row['n_resolved'])}",
            xy=(pd.to_datetime(row["forecast_due_date"]), row["yes_rate"]),
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=6,
            color=INK_MUTED,
            ha="center",
        )
    ax.set_title(panel1_title, loc="left", fontsize=10)
    ax.set_ylabel(yes_ylabel)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))

    ax = axes[0, 1]
    colors = np.where(rounds["in_leaderboard"], SERIES[0], MODEL_POINT)
    ax.bar(dates, rounds["resolved_share"], width=9, color=colors, edgecolor="white", lw=0.5)
    ax.axhline(min_resolved_share, color=INK_SECONDARY, ls="--", lw=1)
    ax.annotate(
        f"check-4 threshold ({min_resolved_share:.0%})",
        xy=(dates.min(), min_resolved_share),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=7,
        color=INK_SECONDARY,
    )
    ax.set_title(
        "2. Share of each round's subset resolved so far\n(gray = round not on the leaderboard)",
        loc="left",
        fontsize=10,
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_ylim(0, 1.05)

    ax = axes[1, 0]
    ax.scatter(
        pd.to_datetime(extremity["model_release_date"]),
        extremity["median_forecast_f"],
        color=SERIES[0],
        s=22,
        edgecolor="white",
        lw=0.5,
    )
    ax.axhline(market_median, color=INK, lw=1.1, ls="--")
    ax.annotate(
        "median market price",
        xy=(pd.to_datetime(extremity["model_release_date"]).min(), market_median),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=7,
        color=INK,
    )
    ax.set_title("3. How extreme each LLM's forecasts are", loc="left", fontsize=10)
    ax.set_ylabel(extremity_ylabel)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=1))
    ax.set_xlabel("Model release date")

    ax = axes[1, 1]
    if restricted:
        for i, (label, frontier) in enumerate(restricted.items()):
            if frontier.empty:
                continue
            ax.step(
                pd.to_datetime(frontier["model_release_date"]),
                frontier[col],
                where="post",
                color=SERIES[i],
                lw=2,
                marker="o",
                ms=4,
                label=label,
            )
        ax.axhline(0, color=INK, lw=1)
        ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("4. SOTA frontier, all rounds vs. well-resolved rounds", loc="left", fontsize=10)
    ax.set_ylabel("Reference minus model (positive = better)")
    ax.set_xlabel("Model release date")

    for ax in axes.flat:
        _date_axis(ax)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def _display_name(name: str) -> str:
    return str(name).removesuffix("†")


def sota_frontier_plot(
    llms: pd.DataFrame,
    frontier: pd.DataFrame,
    band: pd.DataFrame,
    col: str,
    rule: str,
    reference_name: str,
    title: str,
    path: Path,
) -> None:
    """Running-best ForecastBench LLM, with the bootstrap band of that running maximum.

    Grey points are every eligible LLM. The step holds each record until the next one and
    continues to the last release date in `llms`.

    Args:
      llms (pd.DataFrame): ForecastBench LLMs with `model_release_date` and `col`.
      frontier (pd.DataFrame): Output of `find_sota_models`, with `model` and `col`.
      band (pd.DataFrame): Bootstrap 2.5/97.5 of the running maximum (`lower`, `upper`).
      col (str): Score column (positive = better than the reference).
      rule (str): "brier" or "log".
      reference_name (str): Reference forecaster.
      title (str): Figure title.
      path (Path): Output file.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    dates = pd.to_datetime(llms["model_release_date"])
    ax.scatter(
        dates,
        llms[col],
        s=LLM_MARKER_PT**2,
        color=MODEL_POINT,
        zorder=2,
        label="ForecastBench LLM",
    )
    if not band.empty:
        band_dates = pd.to_datetime(band["model_release_date"])
        ax.fill_between(
            band_dates,
            band["lower"],
            band["upper"],
            color=SERIES[0],
            alpha=0.18,
            lw=0,
            zorder=1,
            label="Bootstrap 95% of the running best",
        )
    if not frontier.empty:
        line = frontier.sort_values("model_release_date")
        line_dates = pd.to_datetime(line["model_release_date"])
        line_scores = line[col].to_numpy()
        if len(dates) and line_dates.iloc[-1] < dates.max():
            line_dates = pd.concat([line_dates, pd.Series([dates.max()])], ignore_index=True)
            line_scores = np.append(line_scores, line_scores[-1])
        ax.step(
            line_dates,
            line_scores,
            where="post",
            color=SERIES[0],
            lw=2,
            zorder=3,
            label="Best model released so far",
        )
        ax.scatter(
            pd.to_datetime(frontier["model_release_date"]),
            frontier[col],
            color=SERIES[0],
            s=36,
            zorder=4,
            edgecolor="white",
            lw=0.6,
        )
        for _, row in frontier.iterrows():
            ax.annotate(
                _display_name(row["model"]),
                xy=(pd.to_datetime(row["model_release_date"]), row[col]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=7,
                color=INK,
            )
    ax.axhline(0, color=INK, lw=1)
    ax.set_ylabel(f"{reference_name} minus model (positive = better)")
    ax.set_xlabel("Model release date")
    ax.set_title(title, loc="left")
    ax.annotate(
        RULE_LABELS[rule],
        xy=(0.99, 0.02),
        xycoords="axes fraction",
        ha="right",
        fontsize=8,
        color=INK_MUTED,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5)
    _date_axis(ax)
    _save(fig, path)


def eci_score_plot(matched: pd.DataFrame, col: str, rule: str, path: Path) -> None:
    """Difficulty-adjusted score against Epoch's ECI, one point per matched model.

    Args:
      matched (pd.DataFrame): One row per model run with `eci`, `col`, `model` and `on_frontier`.
      col (str): Score column (positive = better than the market).
      rule (str): "brier" or "log".
      path (Path): Output file.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    rest = matched[~matched["on_frontier"]]
    frontier = matched[matched["on_frontier"]]
    ax.scatter(rest["eci"], rest[col], s=28, color=SERIES[0], zorder=3, label="Matched model")
    if not frontier.empty:
        ax.scatter(
            frontier["eci"],
            frontier[col],
            s=46,
            color=SERIES[1],
            zorder=4,
            label="On the SOTA frontier",
        )
        for _, row in frontier.iterrows():
            ax.annotate(
                _display_name(row["model"]),
                xy=(row["eci"], row[col]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.5,
                color=INK_SECONDARY,
            )
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("Epoch Capabilities Index")
    ax.set_ylabel("Market minus model (positive = better)")
    ax.set_title(
        f"Low-probability market questions vs. Epoch ECI ({RULE_LABELS[rule]})",
        loc="left",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5)
    _save(fig, path)
