#!/usr/bin/env python3
"""
Monte Carlo Tournament Simulator for March Madness 2025
Using a KenPom-based logistic model with calibrated coefficients.
Author: Your Name
Date: 2025-03-XX
"""

import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Setup directories
# -----------------------------------------------------------------------------
MONTE_DIR = "./montecarlo"
os.makedirs(MONTE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Load Data
# -----------------------------------------------------------------------------
# Load tournament bracket file (with empty rounds for later stages)
tourney_file = "./data/ncaa_2025_tournament.csv"
tourney_df = pd.read_csv(tourney_file)

# Load KenPom data and restrict to 2025.
# IMPORTANT: KenPom file headers have spaces. We immediately replace spaces with underscores.
kenpom_file = "./kenpom/kenpom_data.csv"
kenpom_df = pd.read_csv(kenpom_file)
# Replace spaces in column names with underscores so that we can reference them as variables.
kenpom_df.columns = kenpom_df.columns.str.replace(" ", "_")
kenpom_df = kenpom_df[kenpom_df["Year"] == 2025].copy()

# -----------------------------------------------------------------------------
# 3. Build KenPom mapping (normalized team names to metrics)
# -----------------------------------------------------------------------------
# Define the list of KenPom metrics we will use.
kenpom_metrics = [
    "NetRtg", "ORtg", "DRtg", "AdjT", "Luck", 
    "Strength_of_Schedule_NetRtg", "NCSOS_NetRtg"
]

# Function to normalize team names (remove trailing seed numbers, lowercase, etc.)
def normalize_team(name):
    if pd.isna(name):
        return None
    # Remove any trailing digits (e.g., "Duke 1" -> "Duke")
    name = re.sub(r'\s+\d+$', '', name)
    return name.strip().lower()

# Build a dictionary mapping normalized KenPom team names to their metric values.
kenpom_dict = {}
for _, row in kenpom_df.iterrows():
    team_norm = normalize_team(row["Team"])
    if team_norm:
        kenpom_dict[team_norm] = {metric: row[metric] for metric in kenpom_metrics}

# -----------------------------------------------------------------------------
# 4. Normalize tournament team names
# -----------------------------------------------------------------------------
# In the tournament bracket, we add new columns for normalized team names.
tourney_df["Team_A_norm"] = tourney_df["Team_A"].apply(normalize_team)
tourney_df["Team_B_norm"] = tourney_df["Team_B"].apply(normalize_team)

# -----------------------------------------------------------------------------
# 5. Define the Logistic Model Function
# -----------------------------------------------------------------------------
# Calibrated logistic regression coefficients (from prior calibration)
# Order: [const, diff_NetRtg, diff_ORtg, diff_DRtg, diff_AdjT, diff_Luck, diff_Strength_of_Schedule_NetRtg, diff_NCSOS_NetRtg]
model_coefs = {
    "const": -0.1183,
    "NetRtg": -0.1771,
    "ORtg": 0.3491,
    "DRtg": -0.3348,
    "AdjT": -0.0098,
    "Luck": 8.7579,
    "SoS_NetRtg": -1.9778,
    "NCSOS_NetRtg": 0.0036
}

def logistic_probability(team_A_name, team_B_name):
    """
    Given two normalized team names, look up their KenPom metrics and compute
    the probability that team A beats team B using the logistic model.
    """
    A = kenpom_dict.get(team_A_name)
    B = kenpom_dict.get(team_B_name)
    if A is None or B is None:
        return None
    # Compute differences for each metric.
    diff_NetRtg = A["NetRtg"] - B["NetRtg"]
    diff_ORtg   = A["ORtg"] - B["ORtg"]
    diff_DRtg   = A["DRtg"] - B["DRtg"]
    diff_AdjT   = A["AdjT"] - B["AdjT"]
    diff_Luck   = A["Luck"] - B["Luck"]
    diff_SoS_NetRtg = A["Strength_of_Schedule_NetRtg"] - B["Strength_of_Schedule_NetRtg"]
    diff_NCSOS_NetRtg = A["NCSOS_NetRtg"] - B["NCSOS_NetRtg"]

    # Linear predictor (η)
    eta = (model_coefs["const"] +
           model_coefs["NetRtg"] * diff_NetRtg +
           model_coefs["ORtg"]   * diff_ORtg +
           model_coefs["DRtg"]   * diff_DRtg +
           model_coefs["AdjT"]   * diff_AdjT +
           model_coefs["Luck"]   * diff_Luck +
           model_coefs["SoS_NetRtg"] * diff_SoS_NetRtg +
           model_coefs["NCSOS_NetRtg"] * diff_NCSOS_NetRtg)
    # Apply logistic function.
    prob = 1.0 / (1.0 + np.exp(-eta))
    return prob

# -----------------------------------------------------------------------------
# 6. Game Simulation Functions
# -----------------------------------------------------------------------------
def simulate_game(team_A, team_B, override_first_game=False):
    """
    Simulate one game between team_A and team_B.
    If override_first_game is True (e.g. for forcing Duke to win the first game), return team_A.
    """
    if team_A is None:
        return team_B
    if team_B is None:
        return team_A
    # For an override condition (e.g. Duke must win the first game)
    if override_first_game and (team_A == "duke"):
        return team_A
    # Compute the probability that team_A wins.
    p = logistic_probability(team_A, team_B)
    if p is None:
        p = 0.5  # default to coin flip if missing metrics.
    return team_A if random.random() < p else team_B

def simulate_region(region_games_df, override_first=False):
    """
    Simulate the bracket for one region.
    region_games_df should be the subset of tourney_df for a given region and Round 1.
    override_first is used to force a preset outcome (e.g. Duke win in East).
    """
    # Sort games by 'Game' order.
    round_games = region_games_df.sort_values("Game")
    winners = []
    for _, game in round_games.iterrows():
        override = override_first and (game["Team_A_norm"] == "duke")
        winner = simulate_game(game["Team_A_norm"], game["Team_B_norm"], override_first_game=override)
        winners.append(winner)
    # Simulate subsequent rounds by pairing winners.
    current_teams = winners
    while len(current_teams) > 1:
        next_round = []
        for i in range(0, len(current_teams), 2):
            team1 = current_teams[i]
            team2 = current_teams[i+1]
            winner = simulate_game(team1, team2)
            next_round.append(winner)
        current_teams = next_round
    return current_teams[0]  # Regional champion

def simulate_national(east, midwest, south, west):
    """
    Simulate the national rounds.
    Pairings:
      Semifinals: East vs. West, Midwest vs. South.
      Final: winners of the semifinals.
    Returns:
      final champion and runner-up.
    """
    semi1 = simulate_game(east, west)
    semi2 = simulate_game(midwest, south)
    final = simulate_game(semi1, semi2)
    runner_up = semi1 if final != semi1 else semi2
    return final, runner_up

# -----------------------------------------------------------------------------
# 7. Monte Carlo Simulation of the Full Bracket
# -----------------------------------------------------------------------------
def run_simulations(n_sims=10000):
    results = {"east": {}, "midwest": {}, "south": {}, "west": {}, "FinalChampion": {}}
    regions = ["east", "midwest", "south", "west"]

    for sim in range(n_sims):
        regional_champions = {}
        for region in regions:
            reg_df = tourney_df[tourney_df["Region"].str.lower() == region]
            round1 = reg_df[reg_df["Round"] == 1]
            override_first = (region == "east")
            champ = simulate_region(round1, override_first=override_first)
            regional_champions[region] = champ
            results[region][champ] = results[region].get(champ, 0) + 1

        final_champ, _ = simulate_national(regional_champions["east"],
                                           regional_champions["midwest"],
                                           regional_champions["south"],
                                           regional_champions["west"])
        results["FinalChampion"][final_champ] = results["FinalChampion"].get(final_champ, 0) + 1

    # Convert counts to probabilities.
    prob_results = {}
    for region in regions:
        prob_results[region] = {team: count / n_sims for team, count in results[region].items()}
    prob_results["FinalChampion"] = {team: count / n_sims for team, count in results["FinalChampion"].items()}
    return prob_results

# Run Monte Carlo simulation.
n_simulations = 10000
mc_results = run_simulations(n_sims=n_simulations)

# Save regional and national simulation results.
for region in ["East", "Midwest", "South", "West"]:
    region_key = region.lower()
    df_region = pd.DataFrame(list(mc_results[region_key].items()), columns=["Team", "Advancement_Probability"])
    df_region["Region"] = region
    region_file = os.path.join(MONTE_DIR, f"{region.lower()}_advancement_probabilities.csv")
    df_region.to_csv(region_file, index=False)
    print(f"✅ Saved {region} region results to {region_file}")

df_final = pd.DataFrame(list(mc_results["FinalChampion"].items()), columns=["Team", "Championship_Probability"])
df_final.to_csv(os.path.join(MONTE_DIR, "final_champion_probabilities.csv"), index=False)
print("✅ Saved final champion probabilities.")

# -----------------------------------------------------------------------------
# 8. Deterministic (Optimal) Bracket Construction
# -----------------------------------------------------------------------------
def deterministic_winner(team_A, team_B, override_first=False):
    """
    Return the team with the higher predicted win probability (deterministic pick).
    """
    if team_A is None:
        return team_B
    if team_B is None:
        return team_A
    if override_first and (team_A == "duke"):
        return team_A
    p = logistic_probability(team_A, team_B)
    if p is None:
        p = 0.5
    return team_A if p >= 0.5 else team_B

def build_deterministic_bracket(region_games_df, override_first=False):
    """
    Construct a deterministic bracket for one region by always picking the team
    with higher win probability.
    """
    round_games = region_games_df.sort_values("Game")
    winners = []
    for _, game in round_games.iterrows():
        override = override_first and (game["Team_A_norm"] == "duke")
        winner = deterministic_winner(game["Team_A_norm"], game["Team_B_norm"], override_first=override)
        winners.append(winner)
    current_teams = winners
    while len(current_teams) > 1:
        next_round = []
        for i in range(0, len(current_teams), 2):
            team1 = current_teams[i]
            team2 = current_teams[i+1]
            winner = deterministic_winner(team1, team2)
            next_round.append(winner)
        current_teams = next_round
    return current_teams[0]

deterministic_results = {}
for region in ["east", "midwest", "south", "west"]:
    reg_df = tourney_df[tourney_df["Region"].str.lower() == region]
    round1 = reg_df[reg_df["Round"] == 1]
    override_first = (region == "east")
    champ = build_deterministic_bracket(round1, override_first=override_first)
    deterministic_results[region] = champ

semi1 = deterministic_winner(deterministic_results["east"], deterministic_results["west"])
semi2 = deterministic_winner(deterministic_results["midwest"], deterministic_results["south"])
final_det = deterministic_winner(semi1, semi2)

optimal_bracket = {
    "Region_Champion_East": deterministic_results["east"],
    "Region_Champion_Midwest": deterministic_results["midwest"],
    "Region_Champion_South": deterministic_results["south"],
    "Region_Champion_West": deterministic_results["west"],
    "Semifinal_1": semi1,
    "Semifinal_2": semi2,
    "Final_Champion": final_det
}
df_optimal = pd.DataFrame([optimal_bracket])
df_optimal.to_csv(os.path.join(MONTE_DIR, "optimal_bracket.csv"), index=False)
print("✅ Saved deterministic (optimal) bracket to ./montecarlo/optimal_bracket.csv")

# -----------------------------------------------------------------------------
# 9. Save Calibrated Model Parameters and Mapping
# -----------------------------------------------------------------------------
coef_df = pd.DataFrame({
    "Feature": ["const", "diff_NetRtg", "diff_ORtg", "diff_DRtg", "diff_AdjT", "diff_Luck", "diff_Strength_of_Schedule_NetRtg", "diff_NCSOS_NetRtg"],
    "Coefficient": [
        model_coefs["const"],
        model_coefs["NetRtg"],
        model_coefs["ORtg"],
        model_coefs["DRtg"],
        model_coefs["AdjT"],
        model_coefs["Luck"],
        model_coefs["SoS_NetRtg"],
        model_coefs["NCSOS_NetRtg"]
    ]
})
coef_file = os.path.join(MONTE_DIR, "calibrated_model_coefficients.csv")
coef_df.to_csv(coef_file, index=False)
print(f"✅ Saved calibrated model coefficients to {coef_file}")

# Save the mapping from normalized team names to KenPom metrics.
mapping = []
for team, metrics in kenpom_dict.items():
    row = {"Team": team}
    row.update(metrics)
    mapping.append(row)
df_mapping = pd.DataFrame(mapping)
mapping_file = os.path.join(MONTE_DIR, "team_to_kenpom_mapping.csv")
df_mapping.to_csv(mapping_file, index=False)
print(f"✅ Saved team-to-KenPom mapping to {mapping_file}")

print("\n✅ Monte Carlo simulation, deterministic bracket construction, and mapping complete!")
