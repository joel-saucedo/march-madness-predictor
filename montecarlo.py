#!/usr/bin/env python3
import os, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from numba import njit, prange

# =============================================================================
# 0. Setup Directories (output to ./markov)
# =============================================================================
os.makedirs("./markov", exist_ok=True)

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

metrics = ["NetRtg", "ORtg", "DRtg", "AdjT", "Luck", "Strength_of_Schedule_NetRtg", "NCSOS_NetRtg"]
teams_stats = kenpom_2025[["Team_Name", "Seed"] + metrics].reset_index(drop=True)
# Convert metrics to NumPy array
stats_array = teams_stats[metrics].to_numpy(dtype=np.float64)
# Create seeds array (fill missing with 16)
seeds_array = teams_stats["Seed"].fillna(16).to_numpy(dtype=np.int64)

# Build a dictionary from KenPom team name to index
team_to_index = { row["Team_Name"].strip(): idx for idx, row in teams_stats.iterrows() }

# =============================================================================
# 2. Define Manual Mapping (Bracket Name -> KenPom Name)
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
    if seedB < seedA:
        return (teamB, teamA)
    else:
        return (teamA, teamB)

bracket_reordered = {}
for region, games in initial_bracket.items():
    reordered_games = []
    for (ta, tb) in games:
        newA, newB = reorder_game_by_seed(ta, tb)
        reordered_games.append((newA, newB))
    bracket_reordered[region] = reordered_games

# Save bracket structure for debugging
df_bracket = []
for region, games in bracket_reordered.items():
    for i, (ta, tb) in enumerate(games):
        df_bracket.append({
            "Region": region,
            "Round1_Game": i+1,
            "Team_A": ta,
            "Team_B": tb,
            "SeedA": get_seed_from_kenpom_name(ta),
            "SeedB": get_seed_from_kenpom_name(tb),
        })
pd.DataFrame(df_bracket).to_csv("./markov/bracket_round1.csv", index=False)

def map_team_to_index(team_name):
    if team_name is None:
        return -1
    mapped = mapping_dict.get(team_name.strip(), None)
    if mapped is None:
        raise KeyError(f"No mapping for {team_name}")
    idx = team_to_index.get(mapped.strip(), -1)
    if idx == -1:
        raise KeyError(f"No KenPom stats for mapped team {mapped}")
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
# 4. Mixture Measure Setup: Normalize Betas and Define Seed-Based Probabilities
# =============================================================================
const_ = -0.1183
beta_raw = np.array([-0.1771, 0.3491, -0.3348, -0.0098, 8.7579, -1.9778, 0.0036], dtype=np.float64)
abs_sum = np.sum(np.abs(beta_raw))
beta = beta_raw / abs_sum  # normalized coefficients

# Seed-based Round-of-32 probabilities (R32) from historical data:
# (Indices 1-16; index 0 unused)
seed_R32 = np.array([0.0, 0.993, 0.932, 0.841, 0.784, 0.649, 0.630, 0.600, 0.480,
                     0.520, 0.390, 0.370, 0.350, 0.210, 0.150, 0.070, 0.015], dtype=np.float64)

# Weight parameters for mixture measure:
lambda_s = 0.7  # seed weight
lambda_t = 0.3  # team efficiency weight

# Define team-specific (efficiency-based) probability using logistic function.
# We'll use a mixture of the two measures for every game.
@njit
def simulate_matchup_numba(teamA_idx, teamB_idx, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta):
    # Get seeds (if 0 or missing, default to 16)
    seedA = seeds_array[teamA_idx]
    seedB = seeds_array[teamB_idx]
    if seedA <= 0:
        seedA = 16
    if seedB <= 0:
        seedB = 16
    P_s = seed_R32[int(seedA)] / (seed_R32[int(seedA)] + seed_R32[int(seedB)])
    diff = stats_array[teamB_idx] - stats_array[teamA_idx]
    z = const_
    for i in range(beta.shape[0]):
        z += beta[i] * diff[i]
    P_t = 1.0 / (1.0 + math.exp(-z))
    P_mix = lambda_s * P_s + lambda_t * P_t
    # Draw uniform random number:
    u = np.random.rand()
    if u < P_mix:
        return teamA_idx
    else:
        return teamB_idx

# =============================================================================
# 5. Monte Carlo Simulation Functions using Branching Markov Chain
# =============================================================================
@njit
def simulate_region_tournament(region_team_indices, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta):
    """Simulate a single-elimination tournament for one region (16 teams)."""
    current = region_team_indices.copy()
    n = current.shape[0]
    while n > 1:
        new_round = np.empty(n // 2, dtype=np.int64)
        for i in range(n // 2):
            teamA = current[2 * i]
            teamB = current[2 * i + 1]
            new_round[i] = simulate_matchup_numba(teamA, teamB, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
        current = new_round
        n = current.shape[0]
    return current[0]  # champion index

@njit
def simulate_final_four(east_idx, midwest_idx, south_idx, west_idx, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta):
    """Simulate the Final Four bracket: East vs Midwest, South vs West, then final."""
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
            region_team_indices = region_order[region]
            champ = simulate_region_tournament(region_team_indices, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
            region_champs[region] = champ
        final_four_champ = simulate_final_four(region_champs["East"], region_champs["Midwest"],
                                               region_champs["South"], region_champs["West"],
                                               seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
        champ_counts[final_four_champ] += 1
    return champ_counts

# Set simulation parameters
M = 100000  # number of tournaments to simulate

# Run the simulation
championship_counts = simulate_tournament(M, region_order, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
championship_prob = championship_counts / M

# =============================================================================
# 6. Output Results
# =============================================================================
# Create a DataFrame mapping team index to team name and championship probability.
results = []
for team_idx in range(len(championship_prob)):
    prob = championship_prob[team_idx]
    if prob > 0:
        team_name = teams_stats.loc[team_idx, "Team_Name"]
        results.append((team_name, prob))
df_results = pd.DataFrame(results, columns=["Team", "Championship_Prob"])
df_results.sort_values("Championship_Prob", ascending=False, inplace=True)
df_results.reset_index(drop=True, inplace=True)
df_results.to_csv("./markov/final_championship_probabilities.csv", index=False)

plt.figure(figsize=(10, 6))
plt.bar(df_results["Team"], df_results["Championship_Prob"])
plt.xlabel("Team")
plt.ylabel("Championship Probability")
plt.title("Championship Probabilities (Mixture Measure, Monte Carlo Simulation)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("./markov/final_championship_probabilities.png")
plt.close()

# =============================================================================
# 7. Build an "Optimal" Bracket by Maximum Expected Value (Using Mixture Measure)
# =============================================================================
def pick_winner(teamA, teamB, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta):
    # Compute P_s and P_t (as in simulate_matchup_numba) but deterministically choose max
    seedA = seeds_array[map_team_to_index(teamA)]
    seedB = seeds_array[map_team_to_index(teamB)]
    if seedA <= 0:
        seedA = 16
    if seedB <= 0:
        seedB = 16
    P_s = seed_R32[int(seedA)] / (seed_R32[int(seedA)] + seed_R32[int(seedB)])
    diff = stats_array[map_team_to_index(teamA)] - stats_array[map_team_to_index(teamB)]
    z = const_ + np.dot(beta, diff)
    P_t = 1.0 / (1.0 + math.exp(-z))
    P_mix = lambda_s * P_s + lambda_t * P_t
    return teamA if P_mix >= 0.5 else teamB

def build_optimal_bracket(bracket, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta):
    region_picks = {}
    for region, games in bracket.items():
        # For each round, pick the winner deterministically.
        r1 = [pick_winner(ta, tb, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
              for (ta, tb) in games]
        # Round2: pair winners in order.
        r2 = [pick_winner(r1[i], r1[i+1], seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
              for i in range(0, len(r1), 2)]
        r3 = [pick_winner(r2[i], r2[i+1], seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
              for i in range(0, len(r2), 2)]
        region_champ = pick_winner(r3[0], r3[1], seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
        region_picks[region] = {"Round1": r1, "Round2": r2, "Round3": r3, "Champion": region_champ}
    # Final Four: assume East vs Midwest, South vs West, then final.
    ff1 = pick_winner(region_picks["East"]["Champion"], region_picks["Midwest"]["Champion"],
                      seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
    ff2 = pick_winner(region_picks["South"]["Champion"], region_picks["West"]["Champion"],
                      seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
    final_champ = pick_winner(ff1, ff2, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
    return region_picks, {"Semifinal1": (region_picks["East"]["Champion"], region_picks["Midwest"]["Champion"], ff1),
                            "Semifinal2": (region_picks["South"]["Champion"], region_picks["West"]["Champion"], ff2),
                            "Champion": final_champ}

optimal_region_picks, optimal_final = build_optimal_bracket(bracket_reordered, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
df_optimal = pd.DataFrame([
    {"Region": r, "Round1_Winners": picks["Round1"],
     "Round2_Winners": picks["Round2"],
     "Round3_Winners": picks["Round3"],
     "Region_Champion": picks["Champion"]}
    for r, picks in optimal_region_picks.items()
])
df_optimal.to_csv("./markov/optimal_bracket_by_region.csv", index=False)
df_ff = pd.DataFrame([
    {"Semifinal": "East vs Midwest", "Teams": [optimal_final["Semifinal1"][0], optimal_final["Semifinal1"][1]], "Winner": optimal_final["Semifinal1"][2]},
    {"Semifinal": "South vs West", "Teams": [optimal_final["Semifinal2"][0], optimal_final["Semifinal2"][1]], "Winner": optimal_final["Semifinal2"][2]},
    {"Final": f"{optimal_final['Semifinal1'][2]} vs {optimal_final['Semifinal2'][2]}", "Champion": optimal_final["Champion"]}
])
df_ff.to_csv("./markov/optimal_bracket_final_four.csv", index=False)

# =============================================================================
# 8. Save Full Mapping for Debugging (Bracket Team -> KenPom Index)
# =============================================================================
full_map = []
for region, games in bracket_reordered.items():
    for (ta, tb) in games:
        try:
            idxA = map_team_to_index(ta)
        except:
            idxA = -1
        try:
            idxB = map_team_to_index(tb)
        except:
            idxB = -1
        full_map.append({
            "Region": region,
            "Team_A": ta,
            "SeedA": get_seed_from_kenpom_name(ta),
            "Team_A_Index": idxA,
            "Team_B": tb,
            "SeedB": get_seed_from_kenpom_name(tb),
            "Team_B_Index": idxB
        })
pd.DataFrame(full_map).to_csv("./markov/full_bracket_mapping.csv", index=False)

print("✅ Markov Chain Monte Carlo simulation complete with mixture measure, normalized betas, seeded reordering, and Round1 capping.")
