"""Build the cached forecast table the analysis reads.

Writes two parquet files to `cache_dir`:

* `forecasts.parquet`: one row per scored forecast, after the same per-file filters as the
  production leaderboard (`get_df_info`: imputation cutoff, combos dropped, unresolved markets
  dropped). Includes question text and the human-set flag.
* `market_questions.parquet`: one row per market question per round, resolved or not, taken
  from the Naive Forecaster file. It covers every round on disk, including recent ones the
  leaderboard skips, and feeds the resolved-share checks.

Usage:
  python build_cache.py [--config config.toml]
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional
from urllib.error import URLError

import pandas as pd
from _common import leaderboard, load_config, logger

FORECAST_COLUMNS = [
    "question_pk",
    "forecast_due_date",
    "source",
    "id",
    "horizon",
    "resolution_date",
    "resolved_to",
    "forecast",
    "imputed",
    "market_value_on_due_date",
    "market_value_on_due_date_minus_one",
    "model_pk",
    "organization",
    "model_organization",
    "model",
    "model_key",
    "model_run_key",
    "forecast_variant_key",
    "uses_freeze_values",
    "uses_tools",
    "external_submission",
    "forecastbench_llm",
    "forecastbench_comparison_model",
    "baseline_model",
    "tournament_model",
    "first_forecast_due_date",
    "model_release_date",
    "model_age_at_due_date",
]


def list_local_forecast_files(directory: str) -> tuple[List[str], List[str]]:
    """Return processed forecast files under `directory` as `<date>/<file>.json` paths.

    Args:
      directory (str): Local copy of the processed forecast sets bucket.
    """
    leaderboard()
    from helpers.run_mode import RunMode

    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"{directory} does not exist. Sync the processed forecast sets first (see README)."
        )
    files, dates = [], []
    for date_folder in sorted(os.listdir(directory)):
        folder = Path(directory) / date_folder
        if not folder.is_dir():
            continue
        try:
            pd.to_datetime(date_folder, format="%Y-%m-%d")
        except ValueError:
            continue
        dates.append(date_folder)
        for name in sorted(os.listdir(folder)):
            if name.endswith(".json") and not name.startswith(RunMode.TEST.output_file_prefix):
                files.append(f"{date_folder}/{name}")
    return files, dates


def read_question_set(forecast_due_date: str, kind: str, directory: str) -> Optional[dict]:
    """Return a published question set, downloading it once into `directory`.

    Args:
      forecast_due_date (str): Round due date.
      kind (str): "llm" or "human".
      directory (str): Local question set folder.
    """
    from orchestration import _io

    os.makedirs(directory, exist_ok=True)
    path = Path(directory) / f"{forecast_due_date}-{kind}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    try:
        data = _io.read_question_set_json(path.name)
    except (URLError, ValueError, OSError) as e:
        logger.warning(f"Could not download question set {path.name}: {e}")
        return None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def question_text_table(dates: List[str], directory: str) -> pd.DataFrame:
    """Return question text, URL and human-set membership for market and dataset questions.

    Args:
      dates (List[str]): Round due dates.
      directory (str): Local question set folder.
    """
    rows = []
    for forecast_due_date in dates:
        llm = read_question_set(forecast_due_date, "llm", directory)
        human = read_question_set(forecast_due_date, "human", directory)
        if llm is None:
            continue
        human_ids = {
            (q["source"], str(q["id"]))
            for q in (human or {}).get("questions", [])
            if not isinstance(q["id"], list)
        }
        for q in llm["questions"]:
            if isinstance(q["id"], list):
                continue
            rows.append(
                {
                    "forecast_due_date": pd.to_datetime(forecast_due_date).date(),
                    "source": q["source"],
                    "id": str(q["id"]),
                    "question": q.get("question"),
                    "background": q.get("background"),
                    "resolution_criteria": q.get("resolution_criteria"),
                    "url": q.get("url"),
                    "freeze_datetime_value": pd.to_numeric(
                        q.get("freeze_datetime_value"), errors="coerce"
                    ),
                    "in_human_question_set": (q["source"], str(q["id"])) in human_ids,
                }
            )
    return pd.DataFrame(rows)


def market_question_table(files: List[str], directory: str) -> pd.DataFrame:
    """Return every market question per round, resolved or not, from the Naive Forecaster files.

    Args:
      files (List[str]): All forecast files on disk.
      directory (str): Local processed forecast folder.
    """
    lb = leaderboard()
    naive_name = lb.BASELINE_ORG_NAIVE_MODEL["model"]
    frames = []
    for f in files:
        forecast_due_date = f.split("/")[0]
        if Path(f).name != lb.data_utils.get_forecast_filename(forecast_due_date, naive_name):
            continue
        data = lb.resolution.read_forecast_file(filename=f"{directory}/{f}")
        if data is None:
            continue
        df = lb.resolution.make_columns_hashable(data["df"])
        df = df[lb.get_market_mask(df) & ~df["id"].apply(lb.resolution.is_combo)].copy()
        df["forecast_due_date"] = pd.to_datetime(forecast_due_date).date()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["id"] = df["id"].astype(str)
    df["resolution_date"] = pd.to_datetime(df["resolution_date"]).dt.date
    cols = [
        "forecast_due_date",
        "source",
        "id",
        "market_value_on_due_date",
        "market_value_on_due_date_minus_one",
        "resolved",
        "resolved_to",
        "resolution_date",
    ]
    return df[[c for c in cols if c in df.columns]]


def compile_forecasts(files: List[str], directory: str) -> pd.DataFrame:
    """Read forecast files and apply the leaderboard's per-file processing.

    Args:
      files (List[str]): Forecast files that pass the leaderboard's round filters.
      directory (str): Local processed forecast folder.
    """
    lb = leaderboard()
    entries: List[pd.DataFrame] = []
    for f in files:
        data = lb.resolution.read_forecast_file(filename=f"{directory}/{f}")
        if data is None or not data.get("leaderboard_eligible"):
            continue
        org_and_model = lb.llm_identities.normalize_llm_identity(
            {
                "organization": data.get("organization"),
                "model": data.get("model"),
                "model_organization": data.get("model_organization"),
                "model_run_key": data.get("model_run_key"),
                "forecast_variant_key": data.get("forecast_variant_key"),
            }
        )
        lb.process_forecast_file(
            leaderboard_entries=entries,
            org_and_model=org_and_model,
            df=data["df"],
            forecast_due_date=data.get("forecast_due_date") or f.split("/")[0],
        )
    df = lb.combine_forecasting_rounds(entries)
    df = lb.get_model_release_date_info(
        df=df, add_model_age_at_due_date=True, add_model_release_date=True
    )
    df["id"] = df["id"].astype(str)
    df["question_type"] = "dataset"
    df.loc[lb.get_market_mask(df), "question_type"] = "market"
    for col in FORECAST_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[FORECAST_COLUMNS + ["question_type"]]
    for col in ("forecast", "resolved_to", "market_value_on_due_date"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["market_value_on_due_date_minus_one"] = pd.to_numeric(
        df["market_value_on_due_date_minus_one"], errors="coerce"
    )
    # Dataset `market_value_on_due_date` is the series level, not a probability. Blank it out
    # so it can never be filtered on by mistake; the raw value stays in the processed files.
    dataset_rows = df["question_type"] == "dataset"
    df.loc[dataset_rows, ["market_value_on_due_date", "market_value_on_due_date_minus_one"]] = (
        float("nan")
    )
    for col in ("model_key", "model_run_key", "forecast_variant_key"):
        df[col] = df[col].astype("string")
    for col in ("uses_freeze_values", "uses_tools"):
        df[col] = df[col].astype("boolean")
    return df


def main(config_path: str) -> None:
    """Build both cache tables.

    Args:
      config_path (str): Config file.
    """
    config = load_config(Path(config_path))
    lb = leaderboard()
    data_cfg = config["data"]
    directory = data_cfg["processed_forecasts_dir"]
    os.makedirs(data_cfg["cache_dir"], exist_ok=True)

    all_files, all_dates = list_local_forecast_files(directory)
    logger.info(f"Found {len(all_files)} forecast files across {len(all_dates)} rounds.")

    markets = market_question_table(all_files, directory)
    markets.to_parquet(Path(data_cfg["cache_dir"]) / "market_questions.parquet", index=False)
    logger.info(f"Wrote {len(markets)} market question rows (all rounds, resolved or not).")

    files, dates = lb.filter_forecast_files_by_forecast_due_date(
        forecast_files=all_files, valid_dates=all_dates, min_days=data_cfg["min_days"]
    )
    files, dates = lb.filter_forecast_files_by_min_resolved_questions(
        forecast_files=files,
        valid_dates=dates,
        local_forecast_set_dir=directory,
        min_counts={
            "market_resolved": data_cfg["min_num_market_questions"],
            "dataset": lb.MIN_NUM_DATASET_QUESTIONS,
        },
    )
    logger.info(f"Rounds used by the leaderboard: {dates}")

    df = compile_forecasts(files, directory)
    text = question_text_table(sorted(set(all_dates)), data_cfg["question_sets_dir"])
    if not text.empty:
        df = df.merge(text, on=["forecast_due_date", "source", "id"], how="left")
        markets_missing_text = df.loc[df["question_type"] == "market", "question"].isna().mean()
        logger.info(f"Market rows without question text: {markets_missing_text:.1%}")
    df.to_parquet(Path(data_cfg["cache_dir"]) / "forecasts.parquet", index=False)

    counts = df.groupby(["forecast_due_date", "question_type"]).size().unstack(fill_value=0)
    logger.info(f"Wrote {len(df)} forecast rows. Rows per round:\n{counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.toml"))
    main(parser.parse_args().config)
