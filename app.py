#!/usr/bin/env python3
"""
Dash app showing 2025 stats and predicted 2026 stats/WAR
for TJ Friedl, Oneil Cruz, Colt Keith using historical data
and Steamer projections in the model.
"""

import pandas as pd
import joblib
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

# =============================
# CONFIG
# =============================
PLAYERS = ["TJ Friedl", "Oneil Cruz", "Colt Keith"]
HISTORICAL_CSV = "historical_fg_stats_2015_2025.csv"
STEAMER_CSV = "steamer_2026.csv"

AVG_STATS = ['AVG','OBP','SLG','wOBA']
DISCIPLINE_STATS = ['K%','BB%']
TOTAL_STATS = ['HR','2B','3B','R','RBI','SB']
ALL_STATS = AVG_STATS + DISCIPLINE_STATS + TOTAL_STATS

# =============================
# LOAD 2025 DATA
# =============================
fg_stats = pd.read_csv(HISTORICAL_CSV)
fg_stats.columns = fg_stats.columns.str.strip()
fg_stats.fillna(0, inplace=True)
fg_stats = fg_stats[(fg_stats['Season']==2025) & (fg_stats['Name'].isin(PLAYERS))]

# Compute K% and BB% if missing
for col in ['K%', 'BB%']:
    if col not in fg_stats.columns:
        if col=='K%':
            fg_stats[col] = fg_stats['SO']/fg_stats['PA']
        else:
            fg_stats[col] = fg_stats['BB']/fg_stats['PA']

# =============================
# LOAD STEAMER PROJECTIONS
# =============================
steamer = pd.read_csv(STEAMER_CSV)
steamer.columns = steamer.columns.str.strip()
steamer = steamer[['Name'] + ALL_STATS].copy()
steamer = steamer.rename(columns={s:f"{s}_steamer" for s in ALL_STATS})

# Merge Steamer with 2025 stats
fg_stats = fg_stats.merge(steamer, on='Name', how='left')
fg_stats.fillna(0, inplace=True)

# =============================
# LOAD MODELS
# =============================
war_model = joblib.load("war_model.pkl")
stat_models = {stat: joblib.load(f"{stat}_model.pkl") for stat in ALL_STATS}
FEATURES = joblib.load("model_features.pkl")

# Ensure all features exist
for f in FEATURES:
    if f not in fg_stats.columns:
        fg_stats[f] = 0

# =============================
# PREDICTIONS
# =============================
# Predict 2026 WAR
fg_stats['WAR_2026_pred'] = war_model.predict(fg_stats[FEATURES]).round(3)

# Predict each stat
for stat, model in stat_models.items():
    fg_stats[f'{stat}_2026'] = model.predict(fg_stats[FEATURES]).round(3)

# =============================
# CREATE CHARTS
# =============================
# 2025 Charts
df_avg_2025 = fg_stats.melt(id_vars=['Name'], value_vars=AVG_STATS)
df_disc_2025 = fg_stats.melt(id_vars=['Name'], value_vars=DISCIPLINE_STATS)
df_total_2025 = fg_stats.melt(id_vars=['Name'], value_vars=TOTAL_STATS)

fig_avg_2025 = px.bar(df_avg_2025, x='variable', y='value', color='Name', barmode='group', title="2025 AVG/Rate Stats")
fig_disc_2025 = px.bar(df_disc_2025, x='variable', y='value', color='Name', barmode='group', title="2025 Plate Discipline Stats")
fig_total_2025 = px.bar(df_total_2025, x='variable', y='value', color='Name', barmode='group', title="2025 Counting Stats")

# 2026 Predicted Charts
df_avg_2026 = fg_stats.melt(id_vars=['Name'], value_vars=[f'{s}_2026' for s in AVG_STATS])
df_disc_2026 = fg_stats.melt(id_vars=['Name'], value_vars=[f'{s}_2026' for s in DISCIPLINE_STATS])
df_total_2026 = fg_stats.melt(id_vars=['Name'], value_vars=[f'{s}_2026' for s in TOTAL_STATS])

fig_avg_2026 = px.bar(df_avg_2026, x='variable', y='value', color='Name', barmode='group', title="2026 Predicted AVG/Rate Stats")
fig_disc_2026 = px.bar(df_disc_2026, x='variable', y='value', color='Name', barmode='group', title="2026 Predicted Plate Discipline Stats")
fig_total_2026 = px.bar(df_total_2026, x='variable', y='value', color='Name', barmode='group', title="2026 Predicted Counting Stats")

# =============================
# DASH APP
# =============================
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

war_table = dbc.Table.from_dataframe(
    fg_stats[['Name','WAR_2026_pred']].rename(columns={'Name':'Player','WAR_2026_pred':'Projected 2026 Offensive WAR'})
             .sort_values(by='Projected 2026 Offensive WAR', ascending=False),
    striped=True, bordered=True, hover=True
)

app.layout = dbc.Container([

    html.H2("Colt Keith, TJ Friedl, Oneil Cruz – 2025 Stats & 2026 Predictions"),
    html.Br(),

    html.H4("Projected 2026 Offensive WAR"),
    war_table,
    html.Br(),

    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_avg_2025), width=6),
        dbc.Col(dcc.Graph(figure=fig_avg_2026), width=6)
    ]),
    html.Br(),

        dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_disc_2025), width=6),
        dbc.Col(dcc.Graph(figure=fig_disc_2026), width=6)
        
    ]),
    html.Br(),

    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_total_2025), width=6),
        dbc.Col(dcc.Graph(figure=fig_total_2026), width=6)
    ]),

], fluid=True)

# =============================
# RUN SERVER
# =============================
if __name__ == "__main__":
    app.run(debug=True)
