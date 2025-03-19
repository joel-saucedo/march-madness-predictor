#!/usr/bin/env python3
import os, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from numba import njit, prange

# =============================================================================
# 0. Setup Directories
# =============================================================================
os.makedirs("./sensitivity", exist_ok=True)

# =============================================================================
# 1. Read and Preprocess KenPom Data (for 2025)
# =============================================================================
kenpom_file = "./kenpom/kenpom_data.csv"
kenpom_df = pd.read_csv(kenpom_file)
# Replace spaces in column names with underscores
kenpom_df.columns = [c.replace(" ", "_") for c in kenpom_df.columns]
# Filter for 2025
kenpom_2025 = kenpom_df[kenpom_df["Year"]==2025].copy()

def split_team_seed(team_str):
    tokens = team_str.strip().split()
    if tokens and tokens[-1].isdigit():
        return " ".join(tokens[:-1]), int(tokens[-1])
    else:
        return team_str.strip(), None

team_names = []
seed_list = []
for t in kenpom_2025["Team"]:
    name, seed = split_team_seed(t)
    team_names.append(name)
    seed_list.append(seed)
kenpom_2025["Team_Name"] = team_names
kenpom_2025["Seed"] = seed_list

metrics = ["NetRtg", "ORtg", "DRtg", "AdjT", "Luck",
           "Strength_of_Schedule_NetRtg", "NCSOS_NetRtg"]
teams_stats = kenpom_2025[["Team_Name", "Seed"] + metrics].reset_index(drop=True)
stats_array = teams_stats[metrics].to_numpy(dtype=np.float64)
# For any missing seed, default to 16
seeds_array = teams_stats["Seed"].fillna(16).to_numpy(dtype=np.int64)

# Build dictionary: KenPom team name -> index
team_to_index = { row["Team_Name"].strip(): idx for idx, row in teams_stats.iterrows() }

# =============================================================================
# 2. Define Manual Mapping (Bracket Name -> KenPom Name) and Seed Lookup
# =============================================================================
mapping_dict = {
    # East region:
    "Duke": "Duke", "American": "American", "Mississippi State": "Mississippi St.",
    "Baylor": "Baylor", "Oregon": "Oregon", "Liberty": "Liberty",
    "Arizona": "Arizona", "Akron": "Akron", "BYU": "BYU",
    "VCU": "VCU", "Wisconsin": "Wisconsin", "Montana": "Montana",
    "Saint Mary's": "Saint Mary's", "Vanderbilt": "Vanderbilt",
    "Alabama": "Alabama", "Robert Morris": "Robert Morris",
    # Midwest region:
    "Houston": "Houston", "SIU-Edwardsville": "SIUE", "Gonzaga": "Gonzaga",
    "Georgia": "Georgia", "Clemson": "Clemson", "McNeese State": "McNeese",
    "Purdue": "Purdue", "High Point": "High Point", "Illinois": "Illinois",
    "Texas": "Texas", "Kentucky": "Kentucky", "Troy": "Troy",
    "UCLA": "UCLA", "Utah State": "Utah St.", "Tennessee": "Tennessee",
    "Wofford": "Wofford",
    # South region:
    "Auburn": "Auburn", "Alabama State": "Alabama St.", "Louisville": "Louisville",
    "Creighton": "Creighton", "Michigan": "Michigan", "UC-San Diego": "UC San Diego",
    "Texas A&M": "Texas A&M", "Yale": "Yale", "Ole Miss": "Mississippi",
    "UNC": "North Carolina", "Iowa State": "Iowa St.", "Lipscomb": "Lipscomb",
    "Marquette": "Marquette", "New Mexico": "New Mexico", "Michigan State": "Michigan St.",
    "Bryant": "Bryant",
    # West region:
    "Florida": "Florida", "Norfolk State": "Norfolk St.", "UConn": "Connecticut",
    "Oklahoma": "Oklahoma", "Memphis": "Memphis", "Colorado State": "Colorado St.",
    "Maryland": "Maryland", "Grand Canyon": "Grand Canyon", "Missouri": "Missouri",
    "Drake": "Drake", "Texas Tech": "Texas Tech", "UNC Wilmington": "UNC Wilmington",
    "Kansas": "Kansas", "Arkansas": "Arkansas", "St. John's": "St. John's",
    "Omaha": "Nebraska Omaha"
}

def get_seed_from_kenpom_name(team_name):
    if not team_name:
        return 16
    mapped = mapping_dict.get(team_name.strip(), None)
    if not mapped:
        return 16
    idx = team_to_index.get(mapped, -1)
    if idx == -1:
        return 16
    s = teams_stats.loc[idx, "Seed"]
    if pd.isna(s):
        return 16
    return int(s)

# =============================================================================
# 3. Define and Reorder the Bracket (Round 1)
# =============================================================================
initial_bracket = {
    "East": [
        ("Duke", "American"),
        ("Mississippi State", "Baylor"),
        ("Oregon", "Liberty"),
        ("Arizona", "Akron"),
        ("BYU", "VCU"),
        ("Wisconsin", "Montana"),
        ("Saint Mary's", "Vanderbilt"),
        ("Alabama", "Robert Morris")
    ],
    "Midwest": [
        ("Houston", "SIU-Edwardsville"),
        ("Gonzaga", "Georgia"),
        ("Clemson", "McNeese State"),
        ("Purdue", "High Point"),
        ("Illinois", "Texas"),
        ("Kentucky", "Troy"),
        ("UCLA", "Utah State"),
        ("Tennessee", "Wofford")
    ],
    "South": [
        ("Auburn", "Alabama State"),
        ("Louisville", "Creighton"),
        ("Michigan", "UC-San Diego"),
        ("Texas A&M", "Yale"),
        ("Ole Miss", "UNC"),
        ("Iowa State", "Lipscomb"),
        ("Marquette", "New Mexico"),
        ("Michigan State", "Bryant")
    ],
    "West": [
        ("Florida", "Norfolk State"),
        ("UConn", "Oklahoma"),
        ("Memphis", "Colorado State"),
        ("Maryland", "Grand Canyon"),
        ("Missouri", "Drake"),
        ("Texas Tech", "UNC Wilmington"),
        ("Kansas", "Arkansas"),
        ("St. John's", "Omaha")
    ]
}

def reorder_game_by_seed(teamA, teamB):
    seedA = get_seed_from_kenpom_name(teamA)
    seedB = get_seed_from_kenpom_name(teamB)
    return (teamA, teamB) if seedA <= seedB else (teamB, teamA)

bracket_reordered = {}
for region, games in initial_bracket.items():
    reordered = []
    for (ta, tb) in games:
        reordered.append(reorder_game_by_seed(ta, tb))
    bracket_reordered[region] = reordered

# Save the reordered Round1 bracket for debugging
df_bracket = []
for region, games in bracket_reordered.items():
    for i, (ta, tb) in enumerate(games):
        df_bracket.append({
            "Region": region,
            "Round1_Game": i+1,
            "Team_A": ta,
            "Team_B": tb,
            "SeedA": get_seed_from_kenpom_name(ta),
            "SeedB": get_seed_from_kenpom_name(tb)
        })
pd.DataFrame(df_bracket).to_csv("./sensitivity/bracket_round1.csv", index=False)

def map_team_to_index(team_name):
    if team_name is None:
        return -1
    mapped = mapping_dict.get(team_name.strip(), None)
    if mapped is None:
        raise KeyError(f"No mapping for {team_name}")
    idx = team_to_index.get(mapped.strip(), -1)
    if idx == -1:
        raise KeyError(f"No KenPom stats for {mapped}")
    return idx

# Build region_order: for each region, a list of 16 KenPom indices in bracket order.
region_order = {}
for region, games in bracket_reordered.items():
    indices = []
    for (ta, tb) in games:
        indices.append(map_team_to_index(ta))
        indices.append(map_team_to_index(tb))
    region_order[region] = np.array(indices, dtype=np.int64)

# =============================================================================
# 4. Mixture Measure Setup: Normalize Betas & Define Seed-Based Advancement Rates
# =============================================================================
const_ = -0.1183
beta_raw = np.array([-0.1771, 0.3491, -0.3348, -0.0098, 8.7579, -1.9778, 0.0036], dtype=np.float64)
beta = beta_raw / np.sum(np.abs(beta_raw))  # normalized coefficients

# Historical seed-based R32 probabilities (indices 1 to 16; index 0 unused)
seed_R32 = np.array([0.0, 0.993, 0.932, 0.841, 0.784, 0.649, 0.630, 0.600, 0.480,
                     0.520, 0.390, 0.370, 0.350, 0.210, 0.150, 0.070, 0.015], dtype=np.float64)

# Mixture weights (λs for seed, λt for team metrics)
# These will be varied in the sensitivity analysis.
# For now, default values:
default_lambda_s = 0.3
default_lambda_t = 0.7

# =============================================================================
# 5. Define Monte Carlo Simulation Functions (sensitivity Chain with Mixture Measure)
# =============================================================================
@njit
def simulate_matchup_numba(teamA_idx, teamB_idx, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta):
    # Get seeds; if missing or <=0, default to 16.
    seedA = seeds_array[teamA_idx] if seeds_array[teamA_idx] > 0 else 16
    seedB = seeds_array[teamB_idx] if seeds_array[teamB_idx] > 0 else 16
    P_s = seed_R32[int(seedA)] / (seed_R32[int(seedA)] + seed_R32[int(seedB)])
    diff =  stats_array[teamB_idx] - stats_array[teamA_idx] 
    z = const_
    for i in range(beta.shape[0]):
        z += beta[i] * diff[i]
    P_t = 1.0 / (1.0 + math.exp(-z))
    P_mix = lambda_s * P_s + lambda_t * P_t
    if np.random.rand() < P_mix:
        return teamA_idx
    else:
        return teamB_idx

@njit
def simulate_region_tournament(region_team_indices, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta):
    current = region_team_indices.copy()
    n = current.shape[0]
    while n > 1:
        new_round = np.empty(n//2, dtype=np.int64)
        for i in range(n//2):
            tA = current[2*i]
            tB = current[2*i+1]
            new_round[i] = simulate_matchup_numba(tA, tB, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
        current = new_round
        n = current.shape[0]
    return current[0]

@njit
def simulate_final_four(east_idx, midwest_idx, south_idx, west_idx, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta):
    semi1 = simulate_matchup_numba(east_idx, midwest_idx, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
    semi2 = simulate_matchup_numba(south_idx, west_idx, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
    champion = simulate_matchup_numba(semi1, semi2, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
    return champion

def simulate_tournament(M, region_order, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta):
    num_teams = stats_array.shape[0]
    champ_counts = np.zeros(num_teams, dtype=np.int64)
    regions = list(region_order.keys())
    for m in range(M):
        region_champs = {}
        for region in regions:
            reg_indices = region_order[region]
            champ = simulate_region_tournament(reg_indices, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
            region_champs[region] = champ
        final_champ = simulate_final_four(region_champs["East"], region_champs["Midwest"],
                                          region_champs["South"], region_champs["West"],
                                          seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
        champ_counts[final_champ] += 1
    return champ_counts

# =============================================================================
# 6. Sensitivity Analysis
# =============================================================================
def sensitivity_analysis(lambda_s_values, M, actual_champion=None):
    # actual_champion: string name of the actual tournament champion (for log loss)
    results = []
    for lambda_s in lambda_s_values:
        lambda_t = 1.0 - lambda_s
        champ_counts = simulate_tournament(M, region_order, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
        champ_prob = champ_counts / M
        # Create dictionary: team_name -> probability (only include teams with prob > 0)
        prob_dict = { teams_stats.loc[i, "Team_Name"]: champ_prob[i] for i in range(len(champ_prob)) if champ_prob[i] > 0 }
        # Compute variance across teams (using the probabilities of teams that have > 0 probability)
        prob_vals = np.array(list(prob_dict.values()))
        variance = np.var(prob_vals)
        # Compute entropy
        entropy = -np.sum(prob_vals * np.log(prob_vals + 1e-12))
        # Compute log loss if actual champion is provided: -log(prob(actual champion))
        log_loss = -np.log(prob_dict.get(actual_champion, 1e-12)) if actual_champion is not None else np.nan
        # For convenience, also capture the top team and its probability:
        top_team = max(prob_dict, key=prob_dict.get)
        top_prob = prob_dict[top_team]
        results.append({
            "lambda_s": lambda_s,
            "lambda_t": lambda_t,
            "Variance": variance,
            "Entropy": entropy,
            "LogLoss": log_loss,
            "Top_Team": top_team,
            "Top_Prob": top_prob
        })
        print(f"λs={lambda_s:.2f} | Variance: {variance:.4e} | Entropy: {entropy:.4e} | LogLoss: {log_loss:.4e} | Top: {top_team} ({top_prob:.4e})")
    return pd.DataFrame(results)

# Define lambda_s values to test (from 0 to 1 in steps of 0.1)
lambda_s_values = np.arange(0, 1.01, 0.1)
M = 100000  # number of tournament simulations
# For retrospective calibration, assume the actual champion was "Alabama" (modify as needed)
actual_champion = "Alabama"

sensitivity_df = sensitivity_analysis(lambda_s_values, M, actual_champion)
sensitivity_df.to_csv("./sensitivity/sensitivity_analysis.csv", index=False)

# =============================================================================
# 7. Plot Sensitivity Metrics
# =============================================================================
plt.figure(figsize=(10, 6))
plt.plot(sensitivity_df["lambda_s"].to_numpy(), sensitivity_df["Variance"].to_numpy(), marker="o", label="Variance")
plt.plot(sensitivity_df["lambda_s"].to_numpy(), sensitivity_df["Entropy"].to_numpy(), marker="o", label="Entropy")
plt.plot(sensitivity_df["lambda_s"].to_numpy(), sensitivity_df["LogLoss"].to_numpy(), marker="o", label="Log Loss")
plt.xlabel("λₛ (Seed Weight)")
plt.ylabel("Metric Value")
plt.title("Sensitivity Analysis of Mixture Weights")
plt.legend()
plt.tight_layout()
plt.savefig("./sensitivity/sensitivity_analysis.png")
plt.close()

print("✅ Sensitivity analysis complete. Results saved in ./sensitivity/sensitivity_analysis.csv and plot in ./sensitivity/")
