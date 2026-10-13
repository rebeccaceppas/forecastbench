"""Low-probability tags for market questions.

A market question is tagged per round from the market price on that round's due date. The
question key already includes the due date, so a question can be low-probability in one round
and not in another. Only market rows are tagged: on dataset rows `market_value_on_due_date` is
the series level, not a probability.
"""

from typing import Dict, Iterable

import numpy as np
import pandas as pd

TAG_THRESHOLDS = (0.10, 0.05)


def _pct(threshold: float) -> int:
    return int(round(threshold * 100))


def market_low_probability_tags(
    questions: pd.DataFrame,
    thresholds: Iterable[float] = TAG_THRESHOLDS,
) -> pd.DataFrame:
    """Return one tag row per market `question_pk`.

    For each threshold t, the columns are `low_<t>` (price <= t), `high_<t>` (price >= 1 - t),
    `extreme_<t>` (either) and `side_<t>` ("low", "high" or None).

    Args:
      questions (pd.DataFrame): Market rows with `question_pk` and `market_value_on_due_date`.
      thresholds (Iterable[float]): Tail cutoffs, e.g. 0.10 for 10% or less / 90% or more.
    """
    tags = (
        questions[["question_pk", "market_value_on_due_date"]]
        .drop_duplicates("question_pk")
        .rename(columns={"market_value_on_due_date": "price"})
        .reset_index(drop=True)
    )
    price = pd.to_numeric(tags["price"], errors="coerce")
    for t in thresholds:
        pct = _pct(t)
        low = price <= t
        # Rounded so that 1 - 0.1 compares equal to a price of exactly 0.9.
        high = price >= round(1 - t, 10)
        tags[f"low_{pct}"] = low
        tags[f"high_{pct}"] = high
        tags[f"extreme_{pct}"] = low | high
        tags[f"side_{pct}"] = np.select([low, high], ["low", "high"], default=None)
    return tags


def subset_columns(subset: str, threshold: float) -> Dict[str, str]:
    """Return the tag column that selects `subset` and the column holding each row's side.

    Args:
      subset (str): "extreme", "low" or "high".
      threshold (float): Tail cutoff.
    """
    if subset not in ("extreme", "low", "high"):
        raise ValueError(f"Unknown subset {subset!r}; use 'extreme', 'low' or 'high'.")
    pct = _pct(threshold)
    if threshold not in TAG_THRESHOLDS:
        raise ValueError(f"Threshold {threshold} is not tagged; add it to TAG_THRESHOLDS.")
    return {"mask": f"{subset}_{pct}", "side": f"side_{pct}"}


def flip_onto_low_side(df: pd.DataFrame, side_col: str = "side") -> pd.DataFrame:
    """Add flipped columns so every subset row reads as a low-probability question.

    On the high side the forecast, outcome and price become 1 - x. Brier and log scores are
    unchanged by this, so scores can be computed on either version; the flipped columns are for
    calibration, base rates, constant forecasters and discrimination.

    Args:
      df (pd.DataFrame): Rows with `forecast`, `resolved_to`, `market_value_on_due_date` and a
        side column.
      side_col (str): Column holding "low" or "high".
    """
    df = df.copy()
    high = df[side_col] == "high"
    for src, dst in (
        ("forecast", "forecast_f"),
        ("resolved_to", "outcome_f"),
        ("market_value_on_due_date", "price_f"),
    ):
        df[dst] = df[src].where(~high, 1 - df[src])
    return df
