# Sub-benchmark accuracy: market low-probability questions

Analysis code for the low-probability sub-benchmark (plan: *ForecastBench Sub-Benchmarks:
Low-Probability and AI-Policy Accuracy*, October 8, 2026). This folder covers **market
questions only**. Dataset questions and AI-policy tagging come later.

`src/leaderboard/main.py` is never edited. It is imported (`_common.leaderboard()`), and its
functions are reused: file filters, `get_df_info`, `two_way_fixed_effects`, `score_models` and
`find_sota_models`. The Slack and git helpers are replaced by stand-ins at import time, so
importing it needs no secrets.

## Setup

Use the repository virtual environment (Python 3.13). The scripts need nothing beyond what it
already has. The config is TOML (read with the standard library), because PyYAML isn't installed.

```bash
source .venv/bin/activate
gcloud auth login            # needed for the bucket sync only
gcloud storage rsync --recursive \
  gs://forecastbench-processed-forecast-sets-rebecca \
  experiments/subset_accuracy/data/processed_forecast_sets
```

`data/` and `outputs/` are git-ignored. Question sets are downloaded from the public datasets
repo on the first `build_cache.py` run and cached in `data/question_sets/`.

## Run

```bash
cd experiments/subset_accuracy
python build_cache.py                          # once, after each data sync
python check_official.py --official-csv <published leaderboard_tournament.csv>
python run_analysis.py                         # uses config.toml
python run_analysis.py --n-bootstrap 1999      # final numbers
```

Each run writes `outputs/<timestamp>_<leaderboard>_<subset><pct>/` containing a copy of the
config, `run_info.json` (git commit, branch, dirty flag), `summary.md`, `tables/` and
`figures/`.

## What `run_analysis.py` does

1. **Loads** the cached market rows for the chosen leaderboard (`tournament` or `baseline`).
   It drops rows without an outcome (voided markets that resolved before the due date, or
   non-binary resolutions) and reports how many.
2. **Tags** each question per round from the market price on the due date: `low` (≤ t), `high`
   (≥ 1 − t), `extreme` (both, with the high side flipped to 1 − p and the outcome reversed).
   Tags are stored for t = 10% and 5% (`tables/market_tags.csv`).
3. **Fits difficulty** (score = model effect + question effect) on **all** market questions,
   separately for Brier and log score. It uses the production model filters. A forecaster's
   subset score is the mean of (score − question effect) over its subset questions.
4. **Comparison forecasters**:
   - Built in the script: market price on the due date, always the base rate, always 1%, and
     always 5%.
   - Taken from the forecast files: the Naive Forecaster, the superforecaster and public
     medians, and the LLM crowds.

   The base rate is out of sample. For round R it uses only subset questions from earlier
   rounds that had resolved before R's due date. With fewer than `base_rate_min_n` such
   questions, it falls back to leave-one-round-out and is flagged in `tables/base_rates.csv`.
5. **Scores** every forecaster with both rules and three difficulty methods:
   - `all_question_fit`: the headline method.
   - `subset_fit`: for contrast; a point estimate unless `bootstrap_subset_fit = true`.
   - `none`: unadjusted, paired on the questions both forecasters answered.

   Results are reported as differences from each comparison forecaster (positive = model
   better). Bootstrap CIs resample questions within each round. All of this is in
   `tables/subset_scores.csv`, which also has N, low/high counts, mean and median flipped
   forecast, realized yes rate, and AUC.
6. **Figures**:
   - `trend_release_date_{brier,log}.png`: score vs. the market against release date. Shows
     the SOTA frontier with a bootstrap band, reference lines for the primary comparisons, and
     external submissions as hollow points at their first due date.
   - `extreme_vs_nonextreme_{brier,log}.png`: each model's score on extreme vs. non-extreme
     market questions.
   - `calibration.png`: tail calibration on log-odds axes. Bins are <1%, 1–2%, 2–5%, 5–10%,
     10–25% and 25%+.
   - `downward_curve_checks.png`: the four checks in section 4.4. They are (1) surprise rate
     by round, (2) share of each round's subset resolved so far, (3) median forecast per LLM
     against release date, and (4) the frontier again on rounds with at least
     `min_resolved_share` resolved.
7. **Tables**:
   - `surprises.csv|html`: questions that resolved against the price, most surprising first.
   - `counts_by_round.csv`, `counts_by_source.csv`: the counts report.
   - `models_below_min_n.csv`: models dropped from the plots.
   - `summary.md`: records the human-comparison decision (headline line vs. caveated point,
     against `min_n_human`).

## Settings worth knowing (`config.toml`)

| Setting | Default | Notes |
|---|---|---|
| `analysis.subset` / `threshold` | `extreme` / `0.10` | `low`, `high`, `extreme`; `0.05` for the sensitivity check |
| `scoring.log_clip` | `0.005` | applied to every forecaster, including market and humans |
| `scoring.n_bootstrap` | `200` | use 1999 for final numbers |
| `analysis.min_n` | `30` | minimum subset questions for a model to be plotted |
| `comparisons.reference` | market price | the zero line on the trend plots |

## Known gaps

- Question categories aren't in the cache yet. The counts report breaks down by round and
  source only.
- No ECI mapping yet (Phase 6). External submissions have no release date, so they are placed
  at their first forecast due date and kept out of the frontier.
- In tournament runs, LLMs saw the market price on the freeze date. `surprises.csv` shows that
  price next to the due-date price.
