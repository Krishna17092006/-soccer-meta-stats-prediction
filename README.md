# Gaining a Statistical Edge in Soccer Prediction using Machine Learning
### Role of Meta-Statistics in Match Prediction

**Team 20** — Kenche Srujana, Krishna Keerthan Reddy D R
**Course:** UE24CS352A — Machine Learning (PES University)

## Problem

Predict English Premier League match outcomes (Home win / Draw / Away win) and test whether
engineered **meta-statistics** (Elo ratings, recent form, head-to-head history, rest days,
betting-market implied probability) give a measurable predictive edge over a baseline of simple
rolling averages of each team's raw match stats (goals, shots).

## Dataset

Match-by-match CSVs from [football-data.co.uk](https://www.football-data.co.uk/), English Premier
League (`E0`), seasons 2019/20–2024/25 (~2280 matches). Downloaded live inside the notebook — no
manual download needed. Includes full-time results, shot counts, and closing betting odds.

## Approach

1. Download and clean 6 seasons of match data.
2. Engineer two feature sets, both leakage-free (every feature for a match uses only information
   available strictly before kickoff):
   - **Baseline**: rolling averages (last 5 games) of each team's own raw goals/shots.
   - **Meta-statistics**: Elo rating difference, weighted recent form, goal-difference trend,
     rest-day difference, head-to-head home win rate, market-implied win/draw/loss probability.
3. Chronological train/test split — train on the first 5 seasons, test on the most recent
   (2024/25) season held out.
4. Train Logistic Regression and Random Forest on each feature set; compare accuracy, macro-F1,
   and log-loss against a majority-class baseline.

## Results (test season 2024/25)

| Feature Set            | Model      | Accuracy | F1 (macro) | Log-loss |
|-------------------------|------------|----------|------------|----------|
| Majority-class baseline | –          | 0.408    | –          | –        |
| Baseline (raw stats)    | LogReg     | 0.505    | 0.378      | 1.022    |
| Baseline (raw stats)    | RandForest | 0.500    | 0.372      | 1.049    |
| **Meta-statistics**     | **LogReg**     | **0.545**    | **0.409**      | **0.977**    |
| **Meta-statistics**     | **RandForest** | **0.542**    | **0.409**      | **0.966**    |
| Combined                | LogReg     | 0.542    | 0.406      | 0.981    |
| Combined                | RandForest | 0.537    | 0.405      | 0.977    |

Meta-statistics consistently beat the raw-stats baseline and the majority-class floor, across both
models and all three metrics. See `writeup/writeup.md` for full discussion.

## Repository structure

```
notebooks/Soccer_Match_Prediction.ipynb   # full pipeline: data -> features -> models -> results
writeup/writeup.md                        # project write-up (convert to PDF for submission)
results/                                  # saved plots (EDA, model comparison, confusion matrix, feature importance)
requirements.txt
```

## How to run

**Option A — Google Colab (recommended)**
1. Open `notebooks/Soccer_Match_Prediction.ipynb` in Colab.
2. Run all cells top to bottom. Data is pulled live; no setup needed.

**Option B — Locally**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/Soccer_Match_Prediction.ipynb
```

## Authors

- Kenche Srujana — PES1UG24CS227
- Krishna Keerthan Reddy D R — PES1UG24CS237
