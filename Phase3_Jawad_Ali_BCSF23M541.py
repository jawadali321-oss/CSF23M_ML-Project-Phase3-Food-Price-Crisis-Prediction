# =============================================================================
# ML Project - Phase 3: Feature Engineering
# Name        : Jawad Ali
# Roll Number : BCSF23M541
# Dataset     : Global Food Price Inflation
# Task        : Food Crisis Prediction in Conflict-Affected Countries
# =============================================================================
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb

plt.rcParams.update({
    'figure.facecolor': '#0f0f23', 'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#444', 'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#aaa', 'ytick.color': '#aaa',
    'text.color': '#e0e0e0', 'grid.color': '#2a2a3e',
    'grid.alpha': 0.5, 'font.size': 11,
})
print("All imports OK.")

DATA_PATH = '/kaggle/input/datasets/bcsf23m541jawadali/food-price-crisis-phase2-preprocessed/global_food_inflation_preprocessed.csv'

df = pd.read_csv(DATA_PATH)
print(f"Loaded real dataset: {df.shape}")
print(df.columns.tolist())

TARGET = 'crisis_next_3m'

DROP_COLS = ['ISO3', 'date', 'region', 'price_range']
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

if 'country' in df.columns:
    le = LabelEncoder()
    df['country'] = le.fit_transform(df['country'].astype(str))

df = df.dropna(subset=[TARGET])
df[TARGET] = df[TARGET].astype(int)

print(f"Clean dataset: {df.shape}")
print(f"Target distribution:\n{df[TARGET].value_counts()}")

FEATURES = [c for c in df.columns if c != TARGET]
X = df[FEATURES].copy()
y = df[TARGET].copy()
X = X.fillna(X.median(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# A1 - Random Forest
print("\n[A1] Random Forest ...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(rf_imp)

# A2 - Gradient Boosting
print("\n[A2] Gradient Boosting ...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_imp = pd.Series(gb.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(gb_imp)

# A3 - Extra Trees
print("\n[A3] Extra Trees ...")
et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
et.fit(X_train, y_train)
et_imp = pd.Series(et.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(et_imp)

# A4 - SHAP
print("\n[A4] SHAP values ...")
lgb_base = lgb.LGBMClassifier(random_state=42, verbose=-1)
lgb_base.fit(X_train, y_train)
explainer = shap.TreeExplainer(lgb_base)
shap_vals = explainer.shap_values(X_test)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]
shap_imp = pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURES).sort_values(ascending=False)
print("SHAP:\n", shap_imp)

# A5 - Permutation Importance
print("\n[A5] Permutation Importance ...")
perm = permutation_importance(lgb_base, X_test, y_test, n_repeats=10, random_state=42)
perm_imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
print(perm_imp)

# LIME
print("\n[LIME] Local explanation ...")
lime_exp = lime_tabular.LimeTabularExplainer(
    X_train.values, feature_names=FEATURES,
    class_names=['No Crisis','Crisis'], mode='classification', random_state=42)
expl = lime_exp.explain_instance(X_test.iloc[0].values,
                                  lgb_base.predict_proba, num_features=8)
fig = expl.as_pyplot_figure()
plt.tight_layout()
plt.savefig('fig_LIME.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n-- Rank Aggregation --")
methods = {'RF': rf_imp, 'GB': gb_imp, 'ET': et_imp,
           'SHAP': shap_imp, 'Perm': perm_imp}
rank_df = pd.DataFrame({k: v.rank(ascending=False) for k, v in methods.items()})
rank_df['avg_rank'] = rank_df.mean(axis=1)
rank_df = rank_df.sort_values('avg_rank')
print(rank_df)

# NEW FEATURES
df_fe = df.copy()

# B1 - Inflation Squared
if 'Inflation' in df_fe.columns:
    df_fe['inflation_sq'] = df_fe['Inflation'] ** 2

# B2 - Price Spread
if 'High' in df_fe.columns and 'Low' in df_fe.columns:
    df_fe['price_spread'] = df_fe['High'] - df_fe['Low']

# B3 - Close/Open Ratio
if 'Close' in df_fe.columns and 'Open' in df_fe.columns:
    df_fe['close_open_ratio'] = df_fe['Close'] / df_fe['Open'].clip(lower=0.01)

# B4 - Inflation x Volatility
if 'Inflation' in df_fe.columns and 'average_annualized_food_volatility' in df_fe.columns:
    df_fe['inflation_volatility'] = df_fe['Inflation'] * df_fe['average_annualized_food_volatility']

# B5 - Lag Difference
if 'lag_1' in df_fe.columns and 'lag_2' in df_fe.columns:
    df_fe['lag_diff'] = df_fe['lag_1'] - df_fe['lag_2']

# B6 - Price vs Rolling Average
if 'rolling_avg_3m' in df_fe.columns and 'Close' in df_fe.columns:
    df_fe['price_vs_rolling'] = df_fe['Close'] - df_fe['rolling_avg_3m']

# B7 - FCAI x Inflation
if 'FCAI' in df_fe.columns and 'Inflation' in df_fe.columns:
    df_fe['fcai_inflation'] = df_fe['FCAI'] * df_fe['Inflation']

# B8 - Month Sin/Cos
if 'month' in df_fe.columns:
    df_fe['month_sin'] = np.sin(2 * np.pi * df_fe['month'] / 12)
    df_fe['month_cos'] = np.cos(2 * np.pi * df_fe['month'] / 12)

# B9 - Inflation Velocity Squared
if 'inflation_velocity' in df_fe.columns:
    df_fe['inflation_vel_sq'] = df_fe['inflation_velocity'] ** 2

# B10 - Market Coverage Ratio
if 'number_of_markets_modeled' in df_fe.columns and 'number_of_markets_covered' in df_fe.columns:
    df_fe['market_coverage_ratio'] = (df_fe['number_of_markets_modeled'] /
                                       df_fe['number_of_markets_covered'].clip(lower=1))

new_feats = ['inflation_sq','price_spread','close_open_ratio','inflation_volatility',
             'lag_diff','price_vs_rolling','fcai_inflation','month_sin','month_cos',
             'inflation_vel_sq','market_coverage_ratio']
new_feats = [f for f in new_feats if f in df_fe.columns]
print(f"\nNew features created: {new_feats}")

FEATURES2 = [c for c in df_fe.columns if c != TARGET]
X2 = df_fe[FEATURES2].fillna(df_fe[FEATURES2].median(numeric_only=True))
y2 = df_fe[TARGET].astype(int)

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2)

lgb_base2 = lgb.LGBMClassifier(random_state=42, verbose=-1)
lgb_base2.fit(X_train, y_train)
base_auc = roc_auc_score(y_test, lgb_base2.predict_proba(X_test)[:,1])

lgb_fe = lgb.LGBMClassifier(random_state=42, verbose=-1)
lgb_fe.fit(X2_train, y2_train)
fe_auc = roc_auc_score(y2_test, lgb_fe.predict_proba(X2_test)[:,1])

print(f"\nBaseline AUC : {base_auc:.4f}")
print(f"With FE AUC  : {fe_auc:.4f}  (delta {fe_auc-base_auc:+.4f})")
print(classification_report(y2_test, lgb_fe.predict(X2_test),
                             target_names=['No Crisis','Crisis']))

feat_imp2 = pd.Series(lgb_fe.feature_importances_, index=FEATURES2)
threshold = np.percentile(feat_imp2, 20)
drop_feats = feat_imp2[feat_imp2 <= threshold].index.tolist()
print(f"\nDropping (bottom 20%, threshold={threshold:.2f}): {drop_feats}")

X3 = X2.drop(columns=drop_feats)
X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y2, test_size=0.2, random_state=42, stratify=y2)

lgb_pruned = lgb.LGBMClassifier(random_state=42, verbose=-1)
lgb_pruned.fit(X3_train, y3_train)
pruned_auc = roc_auc_score(y3_test, lgb_pruned.predict_proba(X3_test)[:,1])
print(f"Pruned AUC   : {pruned_auc:.4f}")

print("\n[E] StandardScaler ...")
scaler = StandardScaler()
X3_scaled = scaler.fit_transform(X3)
print(f"Scaled shape: {X3_scaled.shape}")

print("\n[F] K-Means ...")
km = KMeans(n_clusters=4, random_state=42, n_init=10)
df_fe = df_fe.drop(columns=drop_feats, errors='ignore')
df_fe[TARGET] = y2.values
cluster_input = df_fe.drop(columns=[TARGET]).fillna(0)
df_fe['vulnerability_cluster'] = km.fit_predict(cluster_input)

print("Cluster distribution:\n", df_fe['vulnerability_cluster'].value_counts())
print("Crisis rate per cluster:\n",
      df_fe.groupby('vulnerability_cluster')[TARGET].mean().round(3))

X4 = df_fe.drop(columns=[TARGET]).fillna(0)
y4 = df_fe[TARGET].astype(int)
X4_train, X4_test, y4_train, y4_test = train_test_split(
    X4, y4, test_size=0.2, random_state=42, stratify=y4)

lgb_final = lgb.LGBMClassifier(random_state=42, verbose=-1)
lgb_final.fit(X4_train, y4_train)
final_auc = roc_auc_score(y4_test, lgb_final.predict_proba(X4_test)[:,1])
print(f"\nFinal AUC (+ K-Means cluster): {final_auc:.4f}")
print(classification_report(y4_test, lgb_final.predict(X4_test),
                             target_names=['No Crisis','Crisis']))

print("\n" + "="*50)
print("     PHASE 3 FINAL RESULTS")
print("="*50)
results = pd.DataFrame({
    'Model':      ['Baseline', '+ FE', 'Pruned FE', 'Pruned FE + KMeans'],
    'N_Features': [len(FEATURES), len(FEATURES2), len(X3.columns), len(X4.columns)],
    'ROC_AUC':    [base_auc, fe_auc, pruned_auc, final_auc]
})
print(results.to_string(index=False))
print("="*50)

out_path = 'global_food_inflation_phase3_FE.csv'
df_fe.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} -- {df_fe.shape[0]} rows x {df_fe.shape[1]} cols")
print("Phase 3 complete!")
