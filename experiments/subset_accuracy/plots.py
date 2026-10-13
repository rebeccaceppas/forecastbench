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


def trend_plot(
    llms: pd.DataFrame,
    externals: pd.DataFrame,
    frontier: pd.DataFrame,
    band: pd.DataFrame,
    references: Dict[str, float],
    col: str,
    reference_name: str,
    rule: str,
    title: str,
    n_labels: int,
    path: Path,
) -> None:
    """Score difference vs. the reference forecaster against model release date.

    Args:
      llms (pd.DataFrame): ForecastBench LLMs with `model_release_date`, `col`,
        `<col>_ci_lower`, `<col>_ci_upper` and `model`.
      externals (pd.DataFrame): External submissions, placed at their first due date.
      frontier (pd.DataFrame): SOTA models (point estimate) with `model_release_date`, `col`.
      band (pd.DataFrame): `model_release_date`, `lower`, `upper` of the bootstrapped frontier.
      references (Dict[str, float]): Comparison forecasters on the same scale.
      col (str): Column holding the difference vs. the reference (positive = better).
      reference_name (str): Name of the reference forecaster (the zero line).
      rule (str): "brier" or "log".
      title (str): Figure title.
      n_labels (int): Number of points to label by name.
      path (Path): Output file.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2))
    dates = pd.to_datetime(llms["model_release_date"])
    lower = llms[col] - llms[f"{col}_ci_lower"]
    upper = llms[f"{col}_ci_upper"] - llms[col]
    ax.errorbar(
        dates,
        llms[col],
        yerr=np.vstack([lower.clip(lower=0).to_numpy(), upper.clip(lower=0).to_numpy()]),
        fmt="o",
        ms=4.5,
        color=MODEL_POINT,
        ecolor=GRID,
        elinewidth=0.8,
        mec="white",
        mew=0.5,
        zorder=2,
        label="ForecastBench LLM (95% CI)",
    )
    if not band.empty:
        ax.fill_between(
            pd.to_datetime(band["model_release_date"]),
            band["lower"],
            band["upper"],
            step="post",
            color=SERIES[0],
            alpha=0.15,
            lw=0,
            zorder=1,
            label="SOTA frontier, bootstrap 95% band",
        )
    if not frontier.empty:
        x_end = max(dates.max(), pd.to_datetime(frontier["model_release_date"]).max())
        steps = pd.concat(
            [frontier, frontier.tail(1).assign(model_release_date=x_end)], ignore_index=True
        )
        ax.step(
            pd.to_datetime(steps["model_release_date"]),
            steps[col],
            where="post",
            color=SERIES[0],
            lw=2,
            zorder=3,
            label="SOTA frontier",
        )
        ax.scatter(
            pd.to_datetime(frontier["model_release_date"]),
            frontier[col],
            s=40,
            color=SERIES[0],
            edgecolor="white",
            lw=1,
            zorder=4,
        )
    if not externals.empty:
        ax.scatter(
            pd.to_datetime(externals["model_release_date"]),
            externals[col],
            s=42,
            facecolor="white",
            edgecolor=SERIES[1],
            lw=1.6,
            zorder=4,
            label="External submission (at first due date)",
        )

    labeled = pd.concat(
        [frontier.assign(_kind="llm"), externals.nlargest(n_labels // 2, col).assign(_kind="ext")]
    )
    labeled = labeled.nlargest(n_labels, col) if len(labeled) > n_labels else labeled
    for _, row in labeled.iterrows():
        ax.annotate(
            row["model"],
            xy=(pd.to_datetime(row["model_release_date"]), row[col]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=6.5,
            color=INK_SECONDARY,
        )

    all_dates = pd.concat([dates, pd.to_datetime(externals.get("model_release_date", pd.Series()))])
    span = all_dates.max() - all_dates.min()
    ax.set_xlim(all_dates.min() - span * 0.03, all_dates.max() + span * 0.03)
    x_right = mdates.date2num(all_dates.max() + span * 0.05)
    _date_axis(ax)
    _reference_lines(ax, {reference_name: 0.0, **references}, x_right, emphasis=reference_name)
    ax.set_xlabel("Model release date")
    ax.set_ylabel(
        f"Difficulty-adjusted {RULE_LABELS[rule]}: {reference_name} minus model\n"
        "(positive = better than the reference)"
    )
    ax.set_title(title, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=7.5)
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
    ax.set_title("1. Surprise rate of resolved subset questions", loc="left", fontsize=10)
    ax.set_ylabel("Share resolving against the price (flipped yes)")
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
    ax.set_ylabel("Median forecast on subset (flipped)")
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
    fig.suptitle(
        "Why does the SOTA curve slope down on extreme market questions?",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, path)
