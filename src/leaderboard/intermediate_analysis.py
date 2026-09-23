"""Paired human–LLM comparison on human low-probability market questions.

This is a diagnostic, not a production leaderboard. It scores every processed forecast
file on the exact resolved market questions that superforecasters and the public answered
and that had a due-date crowd price below `LOW_PROBABILITY_MARKET_THRESHOLD`. The metric
is plain Brier score, not the difficulty-adjusted Brier Index.
"""

import json
import logging
import os
import sys
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
from termcolor import colored

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import llm_identities  # noqa: E402
import main as leaderboard  # noqa: E402

from helpers import data_utils, decorator, env, resolution  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_FOLDER = "intermediate_analysis"
HUMAN_QUESTION_SET_URL = (
    "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/"
    "datasets/question_sets/{due_date}-human.json"
)
CROWD_MODEL_NAME = "Crowd (market value on due date)"
BRIER_DECIMALS = 4


def question_key(source: str, question_id: object) -> str:
    """Return a stable join key for a standard (non-combo) question.

    Args:
        source (str): Question source name.
        question_id (object): Question id.

    Returns:
        str: `source::id` key.
    """
    return f"{source}::{question_id}"


def add_question_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `question_key` column.

    Args:
        df (pd.DataFrame): Forecast rows with `source` and `id`.

    Returns:
        pd.DataFrame: Copy with `question_key`.
    """
    df = df.copy()
    df["question_key"] = [question_key(source, qid) for source, qid in zip(df["source"], df["id"])]
    return df


def human_low_probability_questions(df_human: pd.DataFrame) -> pd.DataFrame:
    """Return resolved low-probability market rows from a human forecast file.

    Args:
        df_human (pd.DataFrame): Processed human forecasts.

    Returns:
        pd.DataFrame: One row per standard market question below the threshold.
    """
    df = resolution.make_columns_hashable(df_human)
    df = df[~df["id"].apply(resolution.is_combo)].copy()
    masks = leaderboard.get_masks(df)
    low_prob = leaderboard.get_low_probability_market_mask(df)
    df = df.loc[masks["market_resolved"] & low_prob].copy()
    return add_question_key(df).drop_duplicates(subset=["question_key"], ignore_index=True)


def filter_to_human_questions(
    df: pd.DataFrame,
    question_keys: Iterable[str],
    forecast_due_date: str,
) -> pd.DataFrame:
    """Keep rows for the human low-probability questions in the human survey round.

    Args:
        df (pd.DataFrame): Processed forecasts.
        question_keys (Iterable[str]): Keys from `human_low_probability_questions`.
        forecast_due_date (str): Human survey forecast due date.

    Returns:
        pd.DataFrame: Restricted forecast rows.
    """
    df = add_question_key(df)
    due_date = pd.to_datetime(forecast_due_date).date()
    df["forecast_due_date"] = pd.to_datetime(df["forecast_due_date"]).dt.date
    return df[
        df["question_key"].isin(set(question_keys)) & (df["forecast_due_date"] == due_date)
    ].copy()


def add_plain_brier(df: pd.DataFrame) -> pd.DataFrame:
    """Add a plain Brier score column.

    Args:
        df (pd.DataFrame): Rows with `forecast` and `resolved_to`.

    Returns:
        pd.DataFrame: Copy with `brier_score`.
    """
    df = df.copy()
    df["brier_score"] = (df["forecast"] - df["resolved_to"]) ** 2
    return df


def summarize_model_brier(df: pd.DataFrame, n_questions: int) -> pd.DataFrame:
    """Aggregate plain Brier scores by model.

    Args:
        df (pd.DataFrame): Question-level rows with `brier_score`.
        n_questions (int): Size of the human low-probability set.

    Returns:
        pd.DataFrame: One row per model, sorted by mean Brier (lower is better).
    """
    grouped = (
        df.groupby(
            ["organization", "model_organization", "model", "model_pk"],
            dropna=False,
        )
        .agg(
            n=("brier_score", "count"),
            mean_brier=("brier_score", "mean"),
        )
        .reset_index()
    )
    grouped["rmse"] = grouped["mean_brier"] ** 0.5
    grouped["complete"] = grouped["n"] == n_questions
    grouped = grouped.sort_values(
        by=["complete", "mean_brier", "n"],
        ascending=[False, True, False],
        ignore_index=True,
    )
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    return grouped


def attach_human_comparisons(df_scores: pd.DataFrame) -> pd.DataFrame:
    """Add mean-Brier differences versus superforecasters and the public.

    Args:
        df_scores (pd.DataFrame): Model-level scores.

    Returns:
        pd.DataFrame: Copy with comparison columns. Missing if a human model is absent.
    """
    df_scores = df_scores.copy()
    for label, human in [
        ("vs_superforecaster", leaderboard.HUMAN_SUPERFORECASTER),
        ("vs_public", leaderboard.HUMAN_PUBLIC),
    ]:
        match = df_scores["model"] == human["model"]
        if match.sum() != 1:
            df_scores[label] = pd.NA
            continue
        human_brier = df_scores.loc[match, "mean_brier"].iloc[0]
        df_scores[label] = df_scores["mean_brier"] - human_brier
    return df_scores


def crowd_rows_from_human_questions(df_questions: pd.DataFrame) -> pd.DataFrame:
    """Build crowd-forecast rows from due-date market prices.

    Args:
        df_questions (pd.DataFrame): Human low-probability question rows.

    Returns:
        pd.DataFrame: One row per question using `market_value_on_due_date` as the forecast.
    """
    df = df_questions.copy()
    df["organization"] = leaderboard.constants.BENCHMARK_NAME
    df["model_organization"] = leaderboard.constants.BENCHMARK_NAME
    df["model"] = CROWD_MODEL_NAME
    df["model_pk"] = (
        f"{leaderboard.constants.BENCHMARK_NAME}_"
        f"{leaderboard.constants.BENCHMARK_NAME}_{CROWD_MODEL_NAME}"
    )
    df["forecast"] = pd.to_numeric(df["market_value_on_due_date"], errors="coerce")
    return df


def load_question_text(forecast_due_date: str) -> pd.DataFrame:
    """Load question text for the human question set, if available.

    Args:
        forecast_due_date (str): Human survey forecast due date.

    Returns:
        pd.DataFrame: Columns `question_key` and `question`, empty if unavailable.
    """
    empty = pd.DataFrame(columns=["question_key", "question"])
    candidates = []
    if env.QUESTION_SETS_BUCKET:
        local_dir = data_utils.get_local_file_dir(bucket=env.QUESTION_SETS_BUCKET)
        candidates.append(os.path.join(local_dir, f"{forecast_due_date}-human.json"))

    questions = None
    for path in candidates:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                questions = json.load(f).get("questions", [])
            break

    if questions is None:
        url = HUMAN_QUESTION_SET_URL.format(due_date=forecast_due_date)
        try:
            with urlopen(url, timeout=30) as response:
                questions = json.loads(response.read().decode("utf-8")).get("questions", [])
        except (URLError, json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Could not load human question text: {exc}")
            return empty

    rows = [
        {
            "question_key": question_key(item["source"], item["id"]),
            "question": item.get("question", ""),
        }
        for item in questions
    ]
    return pd.DataFrame(rows)


def find_human_forecast_files(local_forecast_set_dir: str) -> tuple[str, dict[str, str]]:
    """Locate processed human comparison files and their forecast due date.

    Args:
        local_forecast_set_dir (str): Local processed-forecast directory.

    Returns:
        tuple[str, dict[str, str]]: Due date and mapping of model name to file path.

    Raises:
        FileNotFoundError: If either human comparison file is missing.
    """
    wanted = {human["model"] for human in leaderboard.HUMAN_MODELS}
    filename_hints = {
        leaderboard.HUMAN_SUPERFORECASTER["model"]: "human_super",
        leaderboard.HUMAN_PUBLIC["model"]: "human_public",
    }
    found: dict[str, str] = {}
    due_dates = set()
    for root, _, filenames in os.walk(local_forecast_set_dir):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            if not any(hint in filename for hint in filename_hints.values()):
                continue
            path = os.path.join(root, filename)
            data = resolution.read_forecast_file(path)
            if data is None:
                continue
            model = data.get("model")
            if model in wanted:
                found[model] = path
                due_dates.add(str(data.get("forecast_due_date")))

    missing = wanted - set(found)
    if missing:
        raise FileNotFoundError(f"Missing human forecast files for: {sorted(missing)}")
    if len(due_dates) != 1:
        raise ValueError(f"Expected one human survey date, found {sorted(due_dates)}")
    return due_dates.pop(), found


def load_round_forecasts(local_forecast_set_dir: str, forecast_due_date: str) -> pd.DataFrame:
    """Load every eligible processed forecast file for one due date.

    Args:
        local_forecast_set_dir (str): Local processed-forecast directory.
        forecast_due_date (str): Forecast due date.

    Returns:
        pd.DataFrame: Combined processed rows for that round.
    """
    round_dir = os.path.join(local_forecast_set_dir, forecast_due_date)
    entries: list[pd.DataFrame] = []
    for filename in sorted(os.listdir(round_dir)):
        if not filename.endswith(".json"):
            continue
        data = resolution.read_forecast_file(os.path.join(round_dir, filename))
        if data is None or not data.get("leaderboard_eligible"):
            continue
        org_and_model = llm_identities.normalize_llm_identity(
            {
                "organization": data.get("organization"),
                "model": data.get("model"),
                "model_organization": data.get("model_organization"),
                "model_run_key": data.get("model_run_key"),
                "forecast_variant_key": data.get("forecast_variant_key"),
            }
        )
        leaderboard.process_forecast_file(
            leaderboard_entries=entries,
            org_and_model=org_and_model,
            df=data.get("df"),
            forecast_due_date=str(data.get("forecast_due_date")),
            filter_low_probability_markets=False,
        )
    if not entries:
        raise ValueError(f"No processed forecasts found for {forecast_due_date}.")
    return pd.concat(entries, ignore_index=True)


def build_question_table(
    df_super: pd.DataFrame,
    df_public: pd.DataFrame,
    df_question_text: pd.DataFrame,
) -> pd.DataFrame:
    """Build the per-question diagnostic table.

    Args:
        df_super (pd.DataFrame): Superforecaster low-probability rows.
        df_public (pd.DataFrame): Public low-probability rows.
        df_question_text (pd.DataFrame): Optional question text.

    Returns:
        pd.DataFrame: One row per human low-probability question.
    """
    df_super = add_plain_brier(df_super)
    df_public = add_plain_brier(df_public)
    table = df_super[
        [
            "question_key",
            "source",
            "id",
            "market_value_on_due_date",
            "resolved_to",
            "forecast",
            "brier_score",
        ]
    ].rename(
        columns={
            "forecast": "superforecaster_forecast",
            "brier_score": "superforecaster_brier",
        }
    )
    public = df_public[["question_key", "forecast", "brier_score"]].rename(
        columns={
            "forecast": "public_forecast",
            "brier_score": "public_brier",
        }
    )
    table = table.merge(public, on="question_key", how="inner")
    if not df_question_text.empty:
        table = table.merge(df_question_text, on="question_key", how="left")
    else:
        table["question"] = ""
    table["crowd_forecast"] = pd.to_numeric(table["market_value_on_due_date"], errors="coerce")
    table["crowd_brier"] = (table["crowd_forecast"] - table["resolved_to"]) ** 2
    table = table.sort_values(
        by=["market_value_on_due_date", "source", "id"],
        ignore_index=True,
    )
    display_cols = [
        "source",
        "id",
        "question",
        "market_value_on_due_date",
        "resolved_to",
        "crowd_forecast",
        "superforecaster_forecast",
        "public_forecast",
        "crowd_brier",
        "superforecaster_brier",
        "public_brier",
    ]
    return table[display_cols]


def format_score_table(df_scores: pd.DataFrame) -> pd.DataFrame:
    """Format model scores for CSV/HTML display.

    Args:
        df_scores (pd.DataFrame): Model-level scores.

    Returns:
        pd.DataFrame: Display columns.
    """
    df = df_scores.copy()
    numeric_cols = ["mean_brier", "rmse", "vs_superforecaster", "vs_public"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(BRIER_DECIMALS)
    return df.rename(
        columns={
            "rank": "Rank",
            "organization": "Team",
            "model_organization": "Model Organization",
            "model": "Model",
            "n": "N",
            "mean_brier": "Mean Brier",
            "rmse": "RMSE",
            "complete": "Complete",
            "vs_superforecaster": "Mean Brier − Super",
            "vs_public": "Mean Brier − Public",
        }
    )[
        [
            "Rank",
            "Team",
            "Model Organization",
            "Model",
            "N",
            "Complete",
            "Mean Brier",
            "RMSE",
            "Mean Brier − Super",
            "Mean Brier − Public",
        ]
    ]


def write_tables(tables: dict[str, pd.DataFrame], output_dir: str) -> None:
    """Write CSV and HTML tables.

    Args:
        tables (dict[str, pd.DataFrame]): Mapping of file stem to table.
        output_dir (str): Output directory.

    Returns:
        None.
    """
    os.makedirs(output_dir, exist_ok=True)
    for stem, df in tables.items():
        csv_path = os.path.join(output_dir, f"{stem}.csv")
        html_path = os.path.join(output_dir, f"{stem}.html")
        df.to_csv(csv_path, index=False)
        title = stem.replace("_", " ").title()
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title>"
            "<style>body{font-family:Arial,sans-serif;padding:20px}"
            "table{border-collapse:collapse}th,td{border:1px solid #ccc;"
            "padding:6px 10px;text-align:left}th{background:#f4f4f4}</style>"
            f"</head><body><h1>{title}</h1>{df.to_html(index=False, na_rep='')}"
            "</body></html>"
        )
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Wrote {csv_path}")
        logger.info(f"Wrote {html_path}")


def print_table(title: str, df: pd.DataFrame) -> None:
    """Print a table to stdout.

    Args:
        title (str): Section title.
        df (pd.DataFrame): Table to print.

    Returns:
        None.
    """
    print(f"\n{title}")
    print("=" * len(title))
    print(df.to_string(index=False))


@decorator.log_runtime
def driver() -> None:
    """Run the paired human–LLM low-probability analysis.

    Returns:
        None.
    """
    local_dir = data_utils.get_local_file_dir(bucket=env.PROCESSED_FORECAST_SETS_BUCKET)
    forecast_due_date, human_files = find_human_forecast_files(local_dir)
    logger.info(colored(f"Human survey round: {forecast_due_date}", "red"))

    human_rows = {}
    for human in leaderboard.HUMAN_MODELS:
        data = resolution.read_forecast_file(human_files[human["model"]])
        processed = leaderboard.get_df_info(
            df=data["df"],
            org_and_model={
                "organization": data["organization"],
                "model": data["model"],
                "model_organization": data["model_organization"],
            },
            forecast_due_date=str(data["forecast_due_date"]),
        )
        if processed is None:
            raise ValueError(f"Could not process {human['model']}.")
        human_rows[human["model"]] = human_low_probability_questions(processed)

    df_super = human_rows[leaderboard.HUMAN_SUPERFORECASTER["model"]]
    df_public = human_rows[leaderboard.HUMAN_PUBLIC["model"]]
    super_keys = set(df_super["question_key"])
    public_keys = set(df_public["question_key"])
    if super_keys != public_keys:
        raise ValueError("Superforecaster and public low-probability question sets differ.")

    question_keys = sorted(super_keys)
    logger.info(
        f"Scoring {len(question_keys)} human low-probability market questions "
        f"(threshold {leaderboard.LOW_PROBABILITY_MARKET_THRESHOLD})."
    )

    df_round = load_round_forecasts(local_dir, forecast_due_date)
    df_paired = filter_to_human_questions(df_round, question_keys, forecast_due_date)
    df_paired = add_plain_brier(df_paired)

    df_crowd = crowd_rows_from_human_questions(df_super)
    df_crowd = add_plain_brier(df_crowd)
    score_rows = pd.concat([df_paired, df_crowd], ignore_index=True, sort=False)
    df_scores = attach_human_comparisons(
        summarize_model_brier(score_rows, n_questions=len(question_keys))
    )
    df_complete = df_scores[df_scores["complete"]].reset_index(drop=True)
    df_complete["rank"] = range(1, len(df_complete) + 1)

    df_question_text = load_question_text(forecast_due_date)
    df_questions = build_question_table(df_super, df_public, df_question_text)
    display_all = format_score_table(df_scores)
    display_complete = format_score_table(df_complete)

    output_dir = os.path.join(
        data_utils.get_mounted_bucket(bucket=env.PUBLIC_RELEASE_BUCKET),
        OUTPUT_FOLDER,
    )
    write_tables(
        {
            "human_low_probability_questions": df_questions,
            "paired_brier_complete": display_complete,
            "paired_brier_all_models": display_all,
        },
        output_dir,
    )
    print_table(
        f"Human low-probability questions (N={len(df_questions)}, round {forecast_due_date})",
        df_questions,
    )
    print_table(
        "Plain Brier on the exact human low-probability questions (complete coverage)",
        display_complete,
    )
    n_incomplete = int((~df_scores["complete"]).sum())
    if n_incomplete:
        print_table(
            f"Models missing some of the {len(question_keys)} questions",
            display_all[~display_all["Complete"]],
        )
    logger.info(colored(f"Wrote tables to {output_dir}", "red"))


if __name__ == "__main__":
    driver()
