# ML Project Phase 3 - Food Price Crisis Prediction

**Name:** Jawad Ali
**Roll Number:** BCSF23M541
**Dataset:** Global Food Price Inflation
**Kaggle:** https://www.kaggle.com/code/bcsf23m541jawadali/csf23m-ml-project-phase3-food-price-crisis-predict

---

## What This Phase Covers

Phase 3 is about feature engineering. The goal was to find which features matter, build new ones, and check if they improve the model.

---

## Dataset

- **Rows:** 4434
- **Target column:** `crisis_next_3m` (binary - will there be a food crisis in the next 3 months)
- **Input:** Preprocessed CSV from Phase 2

---

## Steps Done

### 1. Feature Importance - 5 Methods

- **Random Forest** - FCAI and Inflation came out on top
- **Gradient Boosting** - FCAI dominated heavily (0.79 importance score)
- **Extra Trees** - More balanced, lag features ranked higher here
- **SHAP** - Most reliable method; confirmed FCAI and Inflation as top 2
- **Permutation Importance** - Tested by shuffling features and measuring AUC drop
- **LIME** - Used for one sample to explain a single prediction locally

Combined all 5 into a rank aggregation table. Top features: FCAI, Inflation, rolling_avg_3m, lag_2

---

### 2. New Features Created (10 total)

| Feature | Why |
|---|---|
| inflation_sq | Inflation squared - captures non-linear price acceleration |
| price_spread | High minus Low - measures daily market instability |
| close_open_ratio | Close / Open - shows price direction within a session |
| inflation_volatility | Inflation x Volatility - both being high together is worse than either alone |
| lag_diff | lag_1 minus lag_2 - rate of change between months |
| price_vs_rolling | Current price minus 3-month average - detects sudden spikes |
| fcai_inflation | FCAI x Inflation - combines the two strongest signals |
| month_sin / month_cos | Seasonal encoding - harvest and lean seasons follow a cycle |
| inflation_vel_sq | Velocity squared - rapid acceleration is more dangerous than steady inflation |
| market_coverage_ratio | Markets modeled / covered - low ratio means missing data = hidden risk |

---

### 3. LightGBM Validation (default settings)

| Model | Features | ROC-AUC |
|---|---|---|
| Baseline | 25 | 0.9957 |
| + Feature Engineering | 36 | 0.9950 |
| Pruned FE | 28 | 0.9950 |
| Pruned FE + KMeans | 29 | 0.9956 |

---

### 4. Dropped Features

Bottom 20% by LightGBM importance removed: number_of_markets_modeled, number_of_markets_covered, number_of_food_items, data_coverage_food, average_annualized_food_inflation, average_annualized_food_volatility, index_confidence_score, market_coverage_ratio

---

### 5. KMeans Cluster Feature

4 clusters. Cluster 3 had 100% crisis rate - every single row in it was a food crisis.

| Cluster | Crisis Rate |
|---|---|
| 0 | 5.1% |
| 1 | 53.7% |
| 2 | 42.0% |
| 3 | 100% |

---

## Files

| File | Description |
|---|---|
| Phase3_Jawad_Ali_BCSF23M541.py | Main Phase 3 script |
| global_food_inflation_preprocessed.csv | Input data from Phase 2 |
| global_food_inflation_phase3_FE.csv | Output - final dataset with new features |
