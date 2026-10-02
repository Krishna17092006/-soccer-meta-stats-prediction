# Gaining a Statistical Edge in Soccer Prediction using Machine Learning
## Role of Meta-Statistics in Match Prediction

**Team 20** — Kenche Srujana (PES1UG24CS227), Krishna Keerthan Reddy D R (PES1UG24CS237)
**Course:** UE24CS352A — Machine Learning, PES University

---

### 1. Problem Statement

Predicting the outcome of a football match (Home win, Draw, or Away win) is a classic
three-class classification problem. Most naive approaches use raw per-match statistics — goals,
shots, cards — which are themselves outcomes of the match being predicted, so naive averages of
them carry limited forward-looking signal. This project asks: does replacing raw statistics with
**meta-statistics** — engineered features that summarize a team's broader context (strength
rating, recent form, head-to-head history, rest, and the betting market's own prediction) — give
a measurably better predictive edge?

### 2. Dataset

We use English Premier League match data from [football-data.co.uk](https://www.football-data.co.uk/),
covering six seasons (2019/20 to 2024/25, ~2,280 matches). Each record includes the match date,
teams, full-time score and result, shot counts, and average closing betting odds across
bookmakers. The data is fetched directly inside the notebook at runtime, so no manual download or
Kaggle credentials are required.

### 3. Approach

**Leakage prevention.** The central methodological constraint: every feature used to predict a
match must be computable using only information available strictly *before* kickoff. We process
matches in strict chronological order, maintaining running per-team state (Elo rating, recent
results, goal difference, head-to-head record), and update that state only *after* a match's
features have been recorded — never before.

**Two feature sets** are built and compared head-to-head, using identical models:

- **Baseline (raw statistics).** Rolling averages (last 5 matches) of each team's own goals
  scored, goal difference, and shots.
- **Meta-statistics.**
  - *Elo rating difference* — a dynamically updated strength rating (K=20, +100 home advantage)
    that compounds results over a team's entire history, not just a fixed window.
  - *Form difference* — points-per-game (3/1/0) over the last 5 matches.
  - *Goal-difference trend* — same rolling window, phrased as a differential between the two
    teams rather than each team's absolute number.
  - *Rest-day difference* — proxy for fixture congestion/fatigue.
  - *Head-to-head home win rate* — historical record between this specific pair of teams.
  - *Market-implied probability* — closing bookmaker odds, de-vigorized (overround removed) into
    a proper probability distribution over Home/Draw/Away. This treats the betting market as an
    aggregator of information (injuries, news, public sentiment) not otherwise in our data.

**Train/test split.** Chronological, not random: the first five seasons are the training set,
and the most recent season (2024/25) is held out entirely as the test set. This mirrors the
real-world task (predicting future matches from past data) and avoids the optimistic bias a
random split would introduce.

**Models.** Logistic Regression and Random Forest, trained identically on each feature set
(median imputation for early-season missing history, standard scaling). Using the same models
across feature sets isolates the effect of the *features* rather than the *model*.

### 4. Implementation Overview

The full pipeline lives in `notebooks/Soccer_Match_Prediction.ipynb`:

1. Download and concatenate 6 seasons of CSVs.
2. Clean columns, parse dates, compute de-vigorized market probabilities.
3. EDA: result distribution, goals distribution, missingness check.
4. Chronological, leakage-free feature engineering (Elo loop as described above).
5. Chronological train/test split.
6. Train Logistic Regression and Random Forest on baseline, meta-statistic, and combined feature
   sets; evaluate accuracy, macro-F1, and log-loss.
7. Confusion matrix and feature-importance analysis for the best-performing feature set.

### 5. Results

| Feature Set            | Model      | Accuracy | F1 (macro) | Log-loss |
|--------------------------|------------|----------|------------|----------|
| Majority-class baseline  | –          | 0.408    | –          | –        |
| Baseline (raw stats)     | LogReg     | 0.505    | 0.378      | 1.022    |
| Baseline (raw stats)     | RandForest | 0.500    | 0.372      | 1.049    |
| **Meta-statistics**      | **LogReg**     | **0.545**    | **0.409**      | **0.977**    |
| **Meta-statistics**      | **RandForest** | **0.542**    | **0.409**      | **0.966**    |
| Combined                 | LogReg     | 0.542    | 0.406      | 0.981    |
| Combined                 | RandForest | 0.537    | 0.405      | 0.977    |

By feature importance (see `results/feature_importance.png`), **market-implied probability** and
**Elo rating difference** are the two strongest individual predictors.

### 6. Conclusions

- A majority-class guess ("always predict Home win") reaches ~41% accuracy — the floor any useful
  model must beat.
- Raw per-match statistics, averaged over a rolling window, barely clear that floor (~50%).
- Meta-statistics give a clear, consistent improvement over the raw-stats baseline — roughly a
  4-point accuracy gain and a meaningful log-loss reduction — for both a linear model and a
  tree-based model, confirming the effect is about the features, not the model choice.
- Combining both feature sets does not beat meta-statistics alone: the raw rolling stats add
  little information beyond what Elo, form, and market odds already encode.
- This supports the project's thesis: **aggregated, context-aware "meta" statistics carry a
  genuine statistical edge over raw per-match numbers** for predicting soccer match outcomes.

**Limitations & future work.** No player-level data (injuries, lineups, transfers) is used;
results are specific to one league. Future work could add gradient boosting with tuned
hyperparameters, multi-league generalization, and in-season recency weighting of Elo updates.
