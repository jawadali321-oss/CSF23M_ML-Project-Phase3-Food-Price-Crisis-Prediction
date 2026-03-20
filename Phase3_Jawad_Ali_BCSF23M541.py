#!/usr/bin/env python
# coding: utf-8

# ML Project - Phase 2: Visualization & Preprocessing
# Student  : Jawad Ali
# Roll No  : BCSF23M541
# Dataset  : Global Food Price Inflation
# Source   : https://www.kaggle.com/datasets/anshtanwar/global-food-price-inflation


# ===========================================================================
# SECTION 0 - IMPORTS
# ===========================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor' : '#0d0d1a',
    'axes.facecolor'   : '#13132b',
    'text.color'       : 'white',
    'axes.labelcolor'  : 'white',
    'xtick.color'      : 'white',
    'ytick.color'      : 'white',
    'axes.edgecolor'   : '#2a2a50',
    'grid.color'       : '#1e1e3a',
})

CRISIS_THRESHOLD = 50
FORECAST_MONTHS  = 3

print("All libraries imported.")


# ===========================================================================
# SECTION 1 - LOAD DATA
# ===========================================================================

df_monthly = pd.read_csv(
    '/kaggle/input/datasets/anshtanwar/monthly-food-price-estimates/WLD_RTFP_country_2023-10-02.csv'
)

df_details = pd.read_csv(
    '/kaggle/input/datasets/anshtanwar/monthly-food-price-estimates/WLD_RTP_details_2023-10-02.csv'
)

print("File 1 shape:", df_monthly.shape)
print("File 2 shape:", df_details.shape)

print("\nFile 1 - Monthly Price Data (first 5 rows):")
print(df_monthly.head())

print("\nFile 2 - Country Details (selected columns):")
print(df_details[['country','number_of_markets_modeled',
                   'average_annualized_food_inflation',
                   'index_confidence_score']].head(3))


# ===========================================================================
# SECTION 2 - MERGE BOTH FILES
# ===========================================================================

# File 2 mein har mulk ki background information hai jaise volatility,
# data coverage aur confidence score. Hum dono files ko country name pe
# merge karte hain taake har monthly row ke saath us mulk ki info bhi aaye.

df_details = df_details.rename(columns={'iso3': 'ISO3'})
df = df_monthly.merge(df_details, on='country', how='left')

print("Shape after merge:", df.shape)


# ===========================================================================
# SECTION 3 - DROP USELESS COLUMNS
# ===========================================================================

# Neeche diye gaye columns drop kiye hain kyunke ye machine learning mein
# use nahi ho sakte. Kuch plain text hain, kuch duplicate hain, kuch
# sirf metadata hain jo food prices ke baare mein kuch nahi batate.

# ISO3_y duplicate hai
# components sirf text hai jaise Bread Rice Wheat
# currency ki zaroorat nahi kyunke index already normalized hai
# start aur end date columns mein jo info hai woh date column mein already hai
# number_of_observations columns text format mein hain numbers nahi
# Rsquared columns model ki quality describe karte hain food prices nahi
# imputation_model sirf describe karta hai ke data gaps kaise fill kiye

drop_cols = [
    'ISO3_y',
    'components',
    'currency',
    'start_date_observations',
    'end_date_observations',
    'number_of_observations_food',
    'number_of_observations_other',
    'Rsquared_individual_food_items',
    'Rsquared_individual_other_items',
    'imputation_model',
]

df = df.drop(columns=drop_cols)
df = df.rename(columns={'ISO3_x': 'ISO3'})

print("Shape after dropping useless columns:", df.shape)
print("Remaining columns:", df.columns.tolist())


# ===========================================================================
# SECTION 4 - FIX DATA TYPES
# ===========================================================================

# Kuch columns mein numbers percent sign ke saath text ki form mein hain
# jaise 6.06%. Hum percent sign hata ke inhe proper number mein convert
# karte hain taake calculations ho sakein.

pct_columns = [
    'data_coverage_food',
    'data_coverage_previous_12_months_food',
    'total_food_price_increase_since_start_date',
    'average_annualized_food_inflation',
    'maximum_food_drawdown',
    'average_annualized_food_volatility',
]

for col in pct_columns:
    df[col] = df[col].str.replace('%', '').astype(float)

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['country', 'date']).reset_index(drop=True)

print("Data types after fixing:")
print(df.dtypes)


# ===========================================================================
# SECTION 5A - STATISTICAL ANALYSIS
# ===========================================================================

print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(df.describe().round(2).to_string())

print("\nMISSING VALUES PER COLUMN")
print("=" * 60)
miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
miss_df = pd.DataFrame({'Missing Count': miss, 'Missing %': miss_pct})
print(miss_df[miss_df['Missing Count'] > 0])

print("\nDATASET OVERVIEW")
print("=" * 60)
print("Total rows      :", len(df))
print("Countries       :", df['country'].nunique())
print("Date range      :", df['date'].min().date(), "to", df['date'].max().date())
print("Avg rows/country:", round(len(df) / df['country'].nunique(), 1))
print("Inflation Min   :", round(df['Inflation'].min(), 1), "%")
print("Inflation Max   :", round(df['Inflation'].max(), 1), "%")
print("Inflation Mean  :", round(df['Inflation'].mean(), 1), "%")
print("Rows above 50%  :", (df['Inflation'] > 50).sum())


# ===========================================================================
# SECTION 5B - FIGURE 1: Statistical Analysis
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('FIGURE 1 - Statistical Analysis - Global Food Price Inflation',
             fontsize=15, fontweight='bold', y=0.99)


# Panel A - Descriptive stats table
ax = axes[0, 0]
ax.axis('off')
num_cols_stats = ['Open', 'High', 'Low', 'Close', 'Inflation']
stats = df[num_cols_stats].describe().round(2)
tbl = ax.table(cellText=stats.values,
               rowLabels=stats.index,
               colLabels=stats.columns,
               cellLoc='center', loc='center',
               bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor('#1e1e40' if r == 0 else '#13132b')
    cell.set_edgecolor('#2a2a50')
    cell.set_text_props(color='white')
ax.set_title('Descriptive Statistics', color='white', fontsize=11, pad=10)

# Inflation ka minimum -31 percent aur maximum 363 percent hai.
# Mean 14.7 percent hai lekin median sirf 5.4 percent hai.
# Itna farq isliye hai kyunke Lebanon aur South Sudan jaise mulkon ne
# itni zyada mehngai dekhi ke poora average upar chala gaya.


# Panel B - Missing values
ax = axes[0, 1]
miss_nonzero = df.isnull().sum()
miss_nonzero = miss_nonzero[miss_nonzero > 0]
colors_miss = ['#e94560' if v > 200 else '#f5a623' for v in miss_nonzero.values]
bars = ax.barh(miss_nonzero.index, miss_nonzero.values, color=colors_miss)
for bar, val in zip(bars, miss_nonzero.values):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            f'{val}  ({val/len(df)*100:.1f}%)',
            va='center', color='white', fontsize=9)
ax.set_xlabel('Number of Missing Values')
ax.set_title('Missing Values Per Column', fontsize=11)

# Inflation mein 364 values missing hain yani 7.6 percent.
# Price columns mein 64 missing hain yani 1.3 percent.
# Ye missing values har mulk ke shuru ke mahino mein hain
# jab WFP ne data collect karna abhi start nahi kiya tha.


# Panel C - Inflation distribution
ax = axes[1, 0]
ax.hist(df['Inflation'].dropna(), bins=70, color='#4ecdc4', edgecolor='none', alpha=0.85)
ax.axvline(CRISIS_THRESHOLD, color='#e94560', lw=2.5, ls='--',
           label=f'{CRISIS_THRESHOLD}% Crisis Threshold')
ax.set_xlabel('Inflation (%)')
ax.set_ylabel('Count')
ax.set_title('Inflation Distribution - All Countries All Months', fontsize=11)
ax.legend(facecolor='#13132b', labelcolor='white')

# Zyada tar mahine normal hain jahan mehngai 0 se 30 percent ke beech hai.
# 50 percent se upar bahut kam mahine hain.
# Red line 50 percent pe hai jo normal aur crisis months ko alag karti hai.
# Jo months is line ke baad hain wahi hamare model ka target hain.


# Panel D - Confidence score per country
ax = axes[1, 1]
conf = df.groupby('country')['index_confidence_score'].first().sort_values()
colors_conf = ['#e94560' if v < 0.85 else '#4ecdc4' for v in conf.values]
ax.barh(conf.index, conf.values, color=colors_conf)
ax.axvline(0.85, color='#ffd700', lw=2, ls='--', label='0.85 reliability cutoff')
ax.set_xlabel('Index Confidence Score')
ax.set_title('Data Reliability Score Per Country', fontsize=11)
ax.legend(facecolor='#13132b', labelcolor='white', fontsize=8)

# Zyada tar mulkon ka score 0.85 se upar hai matlab WFP ka data trustworthy hai.
# Kuch chhote mulk jaise Guinea-Bissau ka score thoda kam hai.
# Kam score wale mulkon mein inflation numbers real nahi balke estimated hain.

plt.tight_layout()
plt.savefig('fig1_statistical_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


# ===========================================================================
# SECTION 6 - REGION ASSIGNMENT
# ===========================================================================

# Dataset mein 3 alag regions ke mulk hain.
# Region-wise analysis se pata chalega ke crisis ke signals
# har region mein alag hain ya same.
# Ye hamare research ka ek novel contribution hai.

region_mapping = {
    'Sub-Saharan Africa': [
        'Burkina Faso', 'Burundi', 'Cameroon', 'Central African Republic',
        'Chad', 'Congo, Dem. Rep.', 'Congo, Rep.', 'Gambia, The',
        'Guinea-Bissau', 'Liberia', 'Mali', 'Mozambique', 'Niger',
        'Nigeria', 'Somalia', 'South Sudan', 'Sudan'
    ],
    'Middle East & N.Africa': [
        'Iraq', 'Lebanon', 'Syrian Arab Republic', 'Yemen, Rep.'
    ],
    'South Asia': [
        'Afghanistan', 'Lao PDR', 'Myanmar'
    ],
    'Other': ['Haiti']
}

country_to_region = {}
for region, countries in region_mapping.items():
    for c in countries:
        country_to_region[c] = region

df['region'] = df['country'].map(country_to_region)
df['month']  = df['date'].dt.month

print("Region distribution:")
print(df['region'].value_counts())


# ===========================================================================
# SECTION 7 - FIGURE 2: Visual EDA Analysis
# ===========================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle('FIGURE 2 - Visual EDA - Patterns Distributions and Relationships',
             fontsize=15, fontweight='bold', y=0.99)

REGION_COLORS = {
    'Sub-Saharan Africa'    : '#e94560',
    'Middle East & N.Africa': '#ffd700',
    'South Asia'            : '#4ecdc4',
    'Other'                 : '#9b5de5',
}


# Panel A - Time series by region
ax = axes[0, 0]
for region, grp in df.groupby('region'):
    ts = grp.groupby('date')['Inflation'].mean().dropna()
    ax.plot(ts.index, ts.values, label=region,
            color=REGION_COLORS[region], lw=2)
ax.axhline(CRISIS_THRESHOLD, color='white', lw=1.5, ls='--', alpha=0.5)
ax.text(df['date'].min(), CRISIS_THRESHOLD + 4,
        f'{CRISIS_THRESHOLD}% crisis line', color='white', fontsize=8)
ax.set_title('Average Inflation Over Time by Region', fontsize=11)
ax.set_xlabel('Year')
ax.set_ylabel('Avg Inflation (%)')
ax.legend(facecolor='#13132b', labelcolor='white', fontsize=7)

# 2019 ke baad Middle East mein mehngai asman ko chhu gayi.
# Ye Lebanon ki wajah se hua jahan 2020 mein 400 percent se zyada mehngai thi.
# Sub Saharan Africa mein mehngai consistently thodi thodi rehti hai.
# Afghanistan mein 2021-22 mein spike aya jab Taliban ka takeover hua.


# Panel B - Top 10 worst countries
ax = axes[0, 1]
top10 = df.groupby('country')['Inflation'].mean().nlargest(10).sort_values()
bar_colors = ['#e94560' if v > 50 else '#f5a623' if v > 20 else '#4ecdc4'
              for v in top10.values]
ax.barh(top10.index, top10.values, color=bar_colors)
ax.axvline(CRISIS_THRESHOLD, color='#ffd700', lw=2, ls='--',
           label=f'{CRISIS_THRESHOLD}% threshold')
ax.set_xlabel('Average Inflation (%) across all months')
ax.set_title('Top 10 Countries by Average Inflation', fontsize=11)
ax.legend(facecolor='#13132b', labelcolor='white')

# Lebanon, South Sudan aur Sudan ka average inflation hi 50 percent se upar hai.
# Matlab ye mulk sirf kabhi kabhi crisis mein nahi aate balke hamesha crisis mein rehte hain.
# Red color wale mulk sabse zyada affected hain.


# Panel C - Correlation heatmap
ax = axes[0, 2]
heatmap_cols = [
    'Open', 'High', 'Low', 'Close', 'Inflation',
    'number_of_markets_modeled', 'number_of_food_items',
    'data_coverage_food', 'average_annualized_food_inflation',
    'average_annualized_food_volatility', 'index_confidence_score'
]
corr = df[heatmap_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap='RdBu_r', center=0,
            annot=True, fmt='.1f', annot_kws={'size': 6},
            cbar_kws={'shrink': 0.7}, linewidths=0.3,
            xticklabels=[c.replace('_', '\n') for c in heatmap_cols],
            yticklabels=[c.replace('_', '\n') for c in heatmap_cols])
ax.set_title('Correlation Between All Numeric Features', fontsize=11)
ax.tick_params(axis='x', rotation=45, labelsize=6)
ax.tick_params(axis='y', rotation=0, labelsize=6)

# Open High Low Close charon almost 1.0 correlated hain kyunke
# ye sab ek hi food index ke alag versions hain.
# Inflation ka number_of_markets ya data_coverage se koi zyada connection nahi.
# Iska matlab ye hai ke background columns inflation se alag information dete hain.


# Panel D - Boxplot by region
ax = axes[1, 0]
region_order = list(REGION_COLORS.keys())
bdata = [df[df['region'] == r]['Inflation'].dropna().values for r in region_order]
bp = ax.boxplot(bdata, patch_artist=True,
                labels=['SSA', 'MENA', 'S.Asia', 'Other'])
for patch, color in zip(bp['boxes'], REGION_COLORS.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for element in ['whiskers', 'fliers', 'medians', 'caps']:
    plt.setp(bp[element], color='white')
ax.axhline(CRISIS_THRESHOLD, color='#ffd700', lw=2, ls='--',
           label=f'{CRISIS_THRESHOLD}% threshold')
ax.set_ylabel('Inflation (%)')
ax.set_title('Inflation Distribution by Region', fontsize=11)
ax.legend(facecolor='#13132b', labelcolor='white')

# Middle East mein dots 300 percent se bhi upar hain jo extreme cases hain.
# Sub Saharan Africa ka box sabse chora hai matlab ek hi region mein
# kuch mulk bahut zyada affected hain aur kuch bilkul normal hain.
# South Asia ka spread sabse chhota hai.


# Panel E - Data coverage vs inflation
ax = axes[1, 1]
sub_scatter = df[['data_coverage_food', 'Inflation', 'region']].dropna()
for region in REGION_COLORS:
    r_sub = sub_scatter[sub_scatter['region'] == region]
    ax.scatter(r_sub['data_coverage_food'], r_sub['Inflation'],
               alpha=0.4, s=8, color=REGION_COLORS[region], label=region)
ax.axhline(CRISIS_THRESHOLD, color='white', lw=1, ls='--', alpha=0.5)
ax.set_xlabel('Data Coverage Food (%)')
ax.set_ylabel('Inflation (%)')
ax.set_title('Data Coverage vs Inflation', fontsize=11)
ax.legend(facecolor='#13132b', labelcolor='white', fontsize=7)

# Kam coverage wale mulkon mein bhi high aur low dono inflation values hain.
# Iska matlab data ki kami ka mehngai se koi seedha taalluq nahi hai.
# Data mein koi obvious bias nazar nahi ata.


# Panel F - Seasonal pattern
ax = axes[1, 2]
monthly_avg = df.groupby('month')['Inflation'].mean()
month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
ax.bar(range(1, 13), monthly_avg.values, color='#4ecdc4', alpha=0.75)
ax.plot(range(1, 13), monthly_avg.values, marker='o',
        color='#ffd700', lw=2.5, ms=7)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels, fontsize=8)
ax.set_xlabel('Month')
ax.set_ylabel('Average Inflation (%)')
ax.set_title('Seasonal Pattern - Average Inflation by Month', fontsize=11)

# Saal ke shuru mein January se April tak mehngai thodi zyada hoti hai.
# Baad mein thodi kam ho jaati hai.
# Ye pattern har saal repeat hota hai isliye month ko feature mein shamil kiya hai.

plt.tight_layout()
plt.savefig('fig2_eda_visual.png', dpi=150, bbox_inches='tight')
plt.show()


# ===========================================================================
# SECTION 8 - PREPROCESSING
# ===========================================================================

# Note: Sir ki instruction ke mutabiq Phase 2 mein scaling ya normalization
# nahi karni. Ye Phase 3 mein feature engineering ke baad hogi.


# Step 1: Forward fill missing values per country
# Food prices ek mahine mein gayab nahi hoti. Agar kisi mahine ka data
# missing hai toh matlab pichle mahine ki price hi rehti hai.
# Isliye missing values ko us mulk ke pichle available value se fill kiya.
# Ek mulk ki missing value doosre mulk ki value se fill nahi karni
# isliye groupby country use kiya.

for col in ['Open', 'High', 'Low', 'Close', 'Inflation']:
    df[col] = df.groupby('country')[col].transform(lambda x: x.ffill())

print("Missing after forward fill:")
print(df[['Open', 'High', 'Low', 'Close', 'Inflation']].isnull().sum())


# Step 2: Drop rows that still have missing inflation
# Kuch mulkon ke bilkul pehle mahine mein koi pichli value nahi thi
# isliye forward fill kaam nahi kar saki. Ye rows bahut kam hain
# aur inhe drop karna theek hai kyunke koi better option nahi.

rows_before = len(df)
df = df.dropna(subset=['Inflation'])
rows_after = len(df)
print(f"\nDropped {rows_before - rows_after} rows.")
print(f"Remaining rows: {rows_after}")


# ===========================================================================
# SECTION 9 - FEATURE ENGINEERING
# ===========================================================================

# Price Range
# Agar ek mahine mein highest aur lowest price ka farq zyada ho toh market
# unstable hai. Ye instability baad mein inflation spike la sakti hai.
df['price_range'] = df['High'] - df['Low']


# Inflation Velocity
# Ye feature batata hai ke mehngai pichle mahine se kitni change hui.
# Agar kisi mulk mein mehngai 20 se 35 percent ho gayi toh velocity +15 hai.
# Tezi se barh rahi mehngai ek strong warning sign hai chahe abhi 50% na ho.
df['inflation_velocity'] = df.groupby('country')['Inflation'].diff()


# 3 Month Rolling Average
# Ek mahine ka spike misleading ho sakta hai. 3 mahine ka average
# zyada reliable trend dikhata hai. Agar 3 mahine lagataar inflation
# high hai toh ye serious concern hai aur crisis ka signal ho sakta hai.
df['rolling_avg_3m'] = df.groupby('country')['Inflation'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)


# Lag Features
# Model ko pichle mahino ki values dene se use yaad rehta hai ke pehle kya tha.
# Food crisis suddenly nahi aati, pehle se signals hote hain.
# Lag 1 pichle mahine ki value hai, lag 2 do mahine pehle ki, lag 3 teen mahine pehle ki.
df['lag_1'] = df.groupby('country')['Inflation'].shift(1)
df['lag_2'] = df.groupby('country')['Inflation'].shift(2)
df['lag_3'] = df.groupby('country')['Inflation'].shift(3)


# FCAI - Food Crisis Alarm Index
# Ye hamare taraf se banaya gaya ek naya composite score hai.
# Teen cheezein milake ek alarm score banaya:
#   50% weight: abhi ki inflation kitni hai
#   30% weight: mehngai kitni tezi se barh rahi hai
#   20% weight: pichle 3 mahine ka average kya tha
# Is dataset pe Kaggle mein sirf EDA notebooks hain.
# Aisa composite index pehle kisine nahi banaya.
df['FCAI'] = (df['Inflation'] * 0.5 +
              df['inflation_velocity'].fillna(0) * 0.3 +
              df['rolling_avg_3m'] * 0.2)


# Target Variable: crisis_next_3m
# Agar agle 3 mahino mein se kisi bhi mahine inflation 50% se upar gayi
# toh 1 hai warna 0 hai. Yahi wo cheez hai jo ML model predict karega.

def create_crisis_label(group):
    values = group['Inflation'].values
    labels = []
    for i in range(len(values)):
        next_3 = values[i + 1 : i + FORECAST_MONTHS + 1]
        labels.append(1 if len(next_3) > 0 and any(v > CRISIS_THRESHOLD for v in next_3) else 0)
    return pd.Series(labels, index=group.index)

df['crisis_next_3m'] = df.groupby('country').apply(create_crisis_label).values

print("\nTarget Variable Distribution:")
print(df['crisis_next_3m'].value_counts())
print(f"Crisis rate: {df['crisis_next_3m'].mean()*100:.1f}%")

# Sirf 9 percent rows crisis labeled hain.
# Matlab data mein crisis months bahut kam hain normal months ke muqable mein.
# Ye class imbalance hai. Isko Phase 3 mein SMOTE se theek karein ge.
# Phase 2 mein iske saath kuch nahi karna.

print("\nFinal dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# ===========================================================================
# SECTION 10 - FIGURE 3: Preprocessing Results
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('FIGURE 3 - Preprocessing Results and Engineered Features',
             fontsize=15, fontweight='bold', y=0.98)


# Panel A - Before vs after missing values
ax = axes[0, 0]
before_miss = {'Open': 64, 'High': 64, 'Low': 64, 'Close': 64, 'Inflation': 364}
after_miss  = {col: int(df[col].isnull().sum()) for col in before_miss}
x = np.arange(len(before_miss))
ax.bar(x - 0.2, list(before_miss.values()), 0.35,
       label='Before Forward-Fill', color='#e94560', alpha=0.85)
ax.bar(x + 0.2, list(after_miss.values()),  0.35,
       label='After Forward-Fill',  color='#4ecdc4', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(list(before_miss.keys()))
ax.set_ylabel('Missing Count')
ax.set_title('Missing Values: Before vs After Forward-Fill')
ax.legend(facecolor='#13132b', labelcolor='white')

# Forward fill ke baad price columns mein koi missing value nahi bachi.
# Inflation column mein jo missing values thi woh bhi fill ho gayi.
# Sirf woh rows reh gayi jahan bilkul shuru ka data tha
# aur koi pichli value fill karne ke liye nahi thi.


# Panel B - Engineered features distribution
ax = axes[0, 1]
ax.hist(df['inflation_velocity'].dropna(), bins=60,
        color='#9b5de5', alpha=0.7, label='Inflation Velocity', density=True)
ax.hist(df['rolling_avg_3m'].dropna(), bins=60,
        color='#ffd700', alpha=0.5, label='Rolling Avg 3m', density=True)
ax.set_xlim(-80, 150)
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('Engineered Feature Distributions')
ax.legend(facecolor='#13132b', labelcolor='white')

# Inflation velocity 0 ke aas paas zyada concentrated hai.
# Matlab zyada tar mahino mein mehngai zyada change nahi hoti.
# Jab velocity zyada hoti hai woh crisis ka early signal hota hai.


# Panel C - FCAI by region
ax = axes[1, 0]
for region in REGION_COLORS:
    sub_r = df[df['region'] == region]['FCAI'].dropna()
    ax.hist(sub_r, bins=40, alpha=0.6,
            color=REGION_COLORS[region], label=region, density=True)
ax.set_xlabel('FCAI Score')
ax.set_ylabel('Density')
ax.set_title('FCAI Score Distribution by Region')
ax.legend(facecolor='#13132b', labelcolor='white', fontsize=7)

# Middle East ka FCAI score sabse zyada door tak jaata hai.
# Lebanon aur Yemen ki extreme mehngai ki wajah se unka alarm score
# baaki sab mulkon se kaafi upar hai.
# Har region ka FCAI pattern alag hai jo Phase 3 mein SHAP analysis mein kaam aega.


# Panel D - Target class balance
ax = axes[1, 1]
vals = df['crisis_next_3m'].value_counts().sort_index()
bars = ax.bar(['No Crisis (0)', 'Crisis (1)'], vals.values,
              color=['#4ecdc4', '#e94560'], alpha=0.85, width=0.5)
for bar, val in zip(bars, vals.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f'{val}\n({val/len(df)*100:.1f}%)',
            ha='center', color='white', fontweight='bold', fontsize=11)
ax.set_ylabel('Number of Rows')
ax.set_title('Target Variable Distribution (crisis_next_3m)')

# 91 percent rows normal hain aur sirf 9 percent crisis hain.
# Agar abhi model train karein toh model hamesha no crisis bolega
# aur phir bhi 91 percent accurate lagega lekin asli crisis pakad nahi payega.
# Phase 3 mein SMOTE se dono classes ko balance karein ge.

plt.tight_layout()
plt.savefig('fig3_preprocessing.png', dpi=150, bbox_inches='tight')
plt.show()


# ===========================================================================
# SECTION 11 - SAVE PREPROCESSED DATASET
# ===========================================================================

save_cols = [
    'country', 'ISO3', 'date', 'Open', 'High', 'Low', 'Close', 'Inflation',
    'number_of_markets_modeled', 'number_of_markets_covered',
    'number_of_food_items', 'data_coverage_food',
    'data_coverage_previous_12_months_food',
    'total_food_price_increase_since_start_date',
    'average_annualized_food_inflation', 'maximum_food_drawdown',
    'average_annualized_food_volatility',
    'average_monthly_food_price_correlation_between_markets',
    'average_annual_food_price_correlation_between_markets',
    'index_confidence_score',
    'region', 'month', 'price_range', 'inflation_velocity',
    'rolling_avg_3m', 'lag_1', 'lag_2', 'lag_3', 'FCAI',
    'crisis_next_3m'
]

df[save_cols].to_csv('global_food_inflation_preprocessed.csv', index=False)

print("Preprocessed CSV saved: global_food_inflation_preprocessed.csv")
print("Shape:", df[save_cols].shape)
print("Phase 2 complete.")
