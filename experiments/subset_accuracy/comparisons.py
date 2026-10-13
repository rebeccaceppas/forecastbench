"""Comparison forecasters for the market low-probability subset.

Forecasters built here are added as extra rows on the subset questions. They are never part of
the difficulty fit; their scores are adjusted with the same question effects as everyone else.
Forecasters that already exist in the processed files (Naive Forecaster, the human medians and
the LLM crowds) are looked up by name.
"""

from typing import Dict, List

import pandas as pd

MARKET_PRICE = "Market price (due date)"
ALWAYS_BASE_RATE = "Always base rate"
CONSTANT_FORECASTERS = {"Always 1%": 0.01, "Always 5%": 0.05}
LLM_CROWD = "LLM crowd"
LLM_CROWD_PREFIX = "LLM Crowd"
HUMAN_COMPARISONS = ("Superforecaster median forecast", "Public median forecast")
SYNTHETIC = (MARKET_PRICE, ALWAYS_BASE_RATE, *CONSTANT_FORECASTERS)


def template_rows(questions: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return one row per subset question for a forecaster built in this script."""
    rows = questions.copy()
    rows["model_pk"] = name
    rows["model"] = name
    rows["organization"] = "Comparison"
    rows["model_organization"] = "Comparison"
    for col in (
        "external_submission",
        "forecastbench_llm",
        "forecastbench_comparison_model",
        "imputed",
    ):
        rows[col] = False
    rows["synthetic_comparison"] = True
    rows["model_release_date"] = pd.NaT
    return rows


def base_rate_table(questions: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """Return the out-of-sample yes rate (on the flipped scale) per round and side.

    For round R, the rate uses only subset questions from earlier rounds that had resolved
    before R's due date. With fewer than `min_n` such questions, it falls back to the rate over
    all other rounds (leave-one-round-out) and flags it.

    Args:
      questions (pd.DataFrame): One row per subset question with `forecast_due_date`, `side`,
        `outcome_f` and `resolution_date`.
      min_n (int): Minimum earlier questions for an out-of-sample rate.
    """
    rows = []
    due = pd.to_datetime(questions["forecast_due_date"])
    resolved_on = pd.to_datetime(questions["resolution_date"])
    for (round_date, side), _ in questions.groupby(["forecast_due_date", "side"]):
        round_ts = pd.Timestamp(round_date)
        same_side = questions["side"] == side
        prior = questions[same_side & (due < round_ts) & (resolved_on < round_ts)]
        if len(prior) >= min_n:
            rate, n, method = prior["outcome_f"].mean(), len(prior), "earlier rounds"
        else:
            others = questions[same_side & (due != round_ts)]
            rate, n, method = others["outcome_f"].mean(), len(others), "leave-one-round-out"
        rows.append(
            {
                "forecast_due_date": round_date,
                "side": side,
                "base_rate_f": rate,
                "base_rate_n": n,
                "base_rate_method": method,
            }
        )
    return pd.DataFrame(rows)


def build_comparison_rows(
    questions: pd.DataFrame,
    base_rates: pd.DataFrame,
) -> pd.DataFrame:
    """Return rows for the market price, always-base-rate and constant forecasters.

    Forecasts are set on the flipped scale and mapped back, so "Always 1%" forecasts 99% on a
    question priced at 90% or more.

    Args:
      questions (pd.DataFrame): One row per subset question with `side` and
        `market_value_on_due_date`.
      base_rates (pd.DataFrame): Output of `base_rate_table`.
    """
    high = questions["side"] == "high"

    def unflip(values: pd.Series) -> pd.Series:
        return values.where(~high, 1 - values)

    frames = []
    market = template_rows(questions, MARKET_PRICE)
    market["forecast"] = questions["market_value_on_due_date"]
    frames.append(market)

    for name, value in CONSTANT_FORECASTERS.items():
        rows = template_rows(questions, name)
        rows["forecast"] = unflip(pd.Series(value, index=questions.index))
        frames.append(rows)

    rates = questions[["forecast_due_date", "side"]].merge(
        base_rates, on=["forecast_due_date", "side"], how="left"
    )
    rates.index = questions.index
    rows = template_rows(questions, ALWAYS_BASE_RATE)
    rows["forecast"] = unflip(rates["base_rate_f"])
    rows["base_rate_method"] = rates["base_rate_method"]
    frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def resolve_comparison_names(names: List[str], available: pd.Series) -> Dict[str, str]:
    """Map configured comparison names to `model_pk`s present in the data.

    "LLM crowd" expands to every LLM crowd ensemble in the data.

    Args:
      names (List[str]): Names from the config.
      available (pd.Series): `model` by `model_pk` for forecasters on the subset.
    """
    mapping = {}
    by_name = {model: pk for pk, model in available.items()}
    for name in names:
        if name in SYNTHETIC:
            mapping[name] = name
        elif name == LLM_CROWD:
            for pk, model in available.items():
                if str(model).startswith(LLM_CROWD_PREFIX):
                    mapping[str(model)] = pk
        elif name in by_name:
            mapping[name] = by_name[name]
    return mapping


def is_comparison(model: pd.Series) -> pd.Series:
    """Return True for comparison forecasters rather than models being evaluated.

    Args:
      model (pd.Series): Model display names.
    """
    names = model.astype(str)
    return (
        names.isin(SYNTHETIC)
        | names.isin(HUMAN_COMPARISONS)
        | names.isin(["Naive Forecaster", "Imputed Forecaster"])
        | names.str.startswith(LLM_CROWD_PREFIX)
        | names.str.match(r"^(Always|Random Uniform)")
    )
