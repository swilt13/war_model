#!/usr/bin/env python3

"""
Script to download historical FanGraphs batting stats (2015–2025)
and save them to a CSV for local use in a Dash app.
"""

from pybaseball import batting_stats
import pandas as pd

# =============================
# CONFIG
# =============================

START_YEAR = 2015
END_YEAR = 2025
OUTPUT_CSV = "historical_fg_stats_2015_2025.csv"

# Columns to keep (add more if needed)
COLUMNS_NEEDED = [
    'Name', 'PA', 'HR', 'BB', 'SO', 'AVG', 'OBP', 'SLG', 'wOBA',
    'SB', '2B', '3B', 'R', 'RBI', 'WAR'
]

# =============================
# DOWNLOAD DATA
# =============================

dfs = []

print(f"Downloading FanGraphs batting stats from {START_YEAR} to {END_YEAR}...")

for year in range(START_YEAR, END_YEAR + 1):
    print(f"Fetching {year}...")
    df = batting_stats(year, qual=0)  # include all players
    df['Season'] = year  # add season column
    dfs.append(df)

# Combine all years into one DataFrame
historical_stats = pd.concat(dfs, ignore_index=True)

# =============================
# CLEAN DATA
# =============================

historical_stats.fillna(0, inplace=True)

# Keep only relevant columns + Season
columns_to_save = COLUMNS_NEEDED + ['Season']
historical_stats = historical_stats[columns_to_save]

# Compute K% and BB% for convenience
historical_stats['K%'] = historical_stats['SO'] / historical_stats['PA']
historical_stats['BB%'] = historical_stats['BB'] / historical_stats['PA']

# =============================
# SAVE TO CSV
# =============================

historical_stats.to_csv(OUTPUT_CSV, index=False)
print(f"CSV saved successfully to {OUTPUT_CSV}!")
