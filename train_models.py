#!/usr/bin/env python3
"""
Train RandomForest models for each stat and WAR.
Includes historical FanGraphs data (2015-2025) and 2026 Steamer projections
as features for the 3 focus players and all players.
Saves models as .pkl files for use in Dash app.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# =============================
# CONFIG
# =============================
HISTORICAL_CSV = "historical_fg_stats_2015_2025.csv"  # Historical FG data
STEAMER_CSV = "steamer_2026.csv"                       # Steamer projections for 2026
FOCUS_PLAYERS = ["TJ Friedl", "Oneil Cruz", "Colt Keith"]

# Stats
AVG_STATS = ['AVG','OBP','SLG','wOBA']
DISCIPLINE_STATS = ['K%','BB%']
TOTAL_STATS = ['HR','2B','3B','R','RBI','SB']
ALL_STATS = AVG_STATS + DISCIPLINE_STATS + TOTAL_STATS

FEATURES_BASE = ['PA','HR','BB','SO','AVG','OBP','SLG','wOBA',
                 'SB','2B','3B','R','RBI']

# =============================
# LOAD HISTORICAL DATA
# =============================
df = pd.read_csv(HISTORICAL_CSV)
df.columns = df.columns.str.strip()
df.fillna(0, inplace=True)

# Compute K% and BB% if missing
if 'K%' not in df.columns:
    df['K%'] = df['SO'] / df['PA']
if 'BB%' not in df.columns:
    df['BB%'] = df['BB'] / df['PA']

# =============================
# LOAD STEAMER PROJECTIONS
# =============================
steamer = pd.read_csv(STEAMER_CSV)
steamer.columns = steamer.columns.str.strip()
# Keep only columns that match ALL_STATS
for stat in ALL_STATS:
    if stat not in steamer.columns:
        steamer[stat] = 0
# Compute K%/BB% if missing
if 'K%' not in steamer.columns:
    steamer['K%'] = steamer['SO'] / steamer['PA']
if 'BB%' not in steamer.columns:
    steamer['BB%'] = steamer['BB'] / steamer['PA']

# Add suffix "_steamer" to avoid collisions
steamer = steamer[['Name'] + ALL_STATS].copy()
steamer = steamer.rename(columns={stat: f"{stat}_steamer" for stat in ALL_STATS})

# Merge Steamer into historical data for 2025
df = df.merge(steamer, on='Name', how='left')
df.fillna(0, inplace=True)  # Fill NaNs for players not in Steamer

# =============================
# ADD WEIGHTS
# =============================
df['weight'] = 1.0 + 0.1 * (df['Season'] - 2015)
df.loc[df['Name'].isin(FOCUS_PLAYERS), 'weight'] *= 2.0  # extra emphasis

# =============================
# TRAIN STAT MODELS
# =============================
stat_models = {}
FEATURES = FEATURES_BASE + [f"{s}_steamer" for s in ALL_STATS]  # include Steamer projections

for stat in ALL_STATS:
    df[f"{stat}_next"] = df.groupby('Name')[stat].shift(-1)
    train_df = df.dropna(subset=[f"{stat}_next"]).copy()
    
    X_train = train_df[FEATURES]
    y_train = train_df[f"{stat}_next"]
    sample_weight = train_df['weight']
    
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    stat_models[stat] = model
    joblib.dump(model, f"{stat}_model.pkl")
    print(f"Saved {stat}_model.pkl")

# =============================
# TRAIN WAR MODEL
# =============================
df_war = df.dropna(subset=['WAR']).copy()
X_war = df_war[FEATURES]
y_war = df_war['WAR']
sample_weight_war = df_war['weight']

war_model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42)
war_model.fit(X_war, y_war, sample_weight=sample_weight_war)
joblib.dump(war_model, "war_model.pkl")
print("Saved war_model.pkl")

# =============================
# SAVE FEATURES
# =============================
joblib.dump(FEATURES, "model_features.pkl")
print("Saved model_features.pkl")

