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

metrics = ["NetRtg", "ORtg", "DRtg", "AdjT", "Luck", 
           "Strength_of_Schedule_NetRtg", "NCSOS_NetRtg"]
teams_stats = kenpom_2025[["Team_Name", "Seed"] + metrics].reset_index(drop=True)

# Convert these stats to NumPy arrays
stats_array = teams_stats[metrics].to_numpy(dtype=np.float64)
seeds_array = teams_stats["Seed"].fillna(16).to_numpy(dtype=np.int64)

# Build dict from KenPom (Team_Name) -> row index
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

def map_team_to_index(team_name):
    if team_name is None:
        return -1
    mapped_name = mapping_dict.get(team_name.strip(), None)
    if mapped_name is None:
        raise KeyError(f"No mapping for bracket team: '{team_name}'")
    idx = team_to_index.get(mapped_name.strip(), -1)
    if idx == -1:
        raise KeyError(f"No KenPom stats for mapped team '{mapped_name}'")
    return idx

# =============================================================================
# 3. Reorder Round 1 Bracket
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
    sA = get_seed_from_kenpom_name(teamA)
    sB = get_seed_from_kenpom_name(teamB)
    return (teamA, teamB) if sA <= sB else (teamB, teamA)

bracket_reordered = {}
for reg, gms in initial_bracket.items():
    new_gms = []
    for (ta, tb) in gms:
        newA, newB = reorder_game_by_seed(ta, tb)
        new_gms.append((newA, newB))
    bracket_reordered[reg] = new_gms

# Build region_order for use in the simulation, if needed
region_order = {}
for region, games in bracket_reordered.items():
    indices = []
    for (ta, tb) in games:
        indices.append(map_team_to_index(ta))
        indices.append(map_team_to_index(tb))
    region_order[region] = np.array(indices, dtype=np.int64)

# =============================================================================
# 4. Mixture Measure Setup
# =============================================================================
const_ = -0.1183
beta_raw = np.array([-0.1771, 0.3491, -0.3348, -0.0098, 8.7579, -1.9778, 0.0036])
abs_sum = np.sum(np.abs(beta_raw))
beta = beta_raw / abs_sum  # normalized

# Historical seed-based probabilities for R32 (Index 1..16; index 0 unused)
seed_R32 = np.array([0.0, 0.993, 0.932, 0.841, 0.784, 0.649, 0.630, 0.600,
                     0.480, 0.520, 0.390, 0.370, 0.350, 0.210, 0.150, 0.070, 0.015])

# Mixture weights (70% seeds, 30% team metrics)
lambda_s = 0.7
lambda_t = 0.3

# =============================================================================
# 5. Probability Function for "Team A beats Team B"
# =============================================================================
def matchup_probability(teamA, teamB, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta):
    """
    Return the mixture probability that 'teamA' beats 'teamB' 
    (both are bracket team names, not KenPom names).
    """
    idxA = map_team_to_index(teamA)
    idxB = map_team_to_index(teamB)
    # handle seeds
    sA = seeds_array[idxA] if seeds_array[idxA] > 0 else 16
    sB = seeds_array[idxB] if seeds_array[idxB] > 0 else 16
    p_s = seed_R32[int(sA)] / (seed_R32[int(sA)] + seed_R32[int(sB)])
    # handle team-based logistic
    diff = stats_array[idxB] - stats_array[idxA]
    z = const_ + np.sum(beta * diff)
    p_t = 1.0 / (1.0 + math.exp(-z))
    return lambda_s * p_s + lambda_t * p_t

# =============================================================================
# 6. Construct the Full Bracket Path (All Rounds) and Output Matchup Probabilities
# =============================================================================
def build_full_bracket_probabilities(bracket, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta):
    """
    We'll produce a CSV of matchup probabilities for every round (1..4 in region, then Final4 + Championship).
    We do so by repeatedly picking winners in the "deterministic" sense (the higher mixture probability).
    At each game, we record the probability that "Team A" (the bracket's listing) beats "Team B."
    """
    rows = []
    # For convenience, define a function to pick the deterministic winner:
    def pick_winner(teamA, teamB):
        pA = matchup_probability(teamA, teamB, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
        return teamA if pA >= 0.5 else teamB
    
    # We'll do region by region, round by round:
    # Round1: bracket[region] has 8 games => 16 teams
    # Round2: 4 games => 8 teams
    # Round3: 2 games => 4 teams
    # Round4: 1 game => region champion
    all_region_winners = {}
    
    for region, games in bracket.items():
        # Round1
        r1_winners = []
        for i, (ta, tb) in enumerate(games):
            pA = matchup_probability(ta, tb, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
            # store row
            rows.append({
                "Region": region,
                "Round": 1,
                "Matchup": f"Game {i+1}",
                "Team_A": ta,
                "Team_B": tb,
                "Prob_A_Wins": pA,
                "Prob_B_Wins": 1.0 - pA
            })
            winner = ta if pA >= 0.5 else tb
            r1_winners.append(winner)
        
        # Round2
        r2_winners = []
        for i in range(0, 8, 2):
            tA = r1_winners[i]
            tB = r1_winners[i+1]
            pA = matchup_probability(tA, tB, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
            rows.append({
                "Region": region,
                "Round": 2,
                "Matchup": f"Game {i//2+1}",
                "Team_A": tA,
                "Team_B": tB,
                "Prob_A_Wins": pA,
                "Prob_B_Wins": 1.0 - pA
            })
            w = tA if pA >= 0.5 else tB
            r2_winners.append(w)
        
        # Round3 (region semifinal)
        r3_winners = []
        for i in range(0, 4, 2):
            tA = r2_winners[i]
            tB = r2_winners[i+1]
            pA = matchup_probability(tA, tB, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
            rows.append({
                "Region": region,
                "Round": 3,
                "Matchup": f"Game {i//2+1}",
                "Team_A": tA,
                "Team_B": tB,
                "Prob_A_Wins": pA,
                "Prob_B_Wins": 1.0 - pA
            })
            w = tA if pA >= 0.5 else tB
            r3_winners.append(w)
        
        # Round4 (region final)
        tA = r3_winners[0]
        tB = r3_winners[1]
        pA = matchup_probability(tA, tB, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
        rows.append({
            "Region": region,
            "Round": 4,
            "Matchup": "Region Final",
            "Team_A": tA,
            "Team_B": tB,
            "Prob_A_Wins": pA,
            "Prob_B_Wins": 1.0 - pA
        })
        region_champ = tA if pA >= 0.5 else tB
        all_region_winners[region] = region_champ
    
    # Now the Final Four:
    # Semifinal1 = East vs Midwest
    # Semifinal2 = South vs West
    # Then Championship
    # We'll label these Round=5 for semifinal, Round=6 for championship
    # Semifinal1
    tE = all_region_winners["East"]
    tM = all_region_winners["Midwest"]
    pA = matchup_probability(tE, tM, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
    rows.append({
        "Region": "FinalFour",
        "Round": 5,
        "Matchup": "Semifinal1 (East vs Midwest)",
        "Team_A": tE,
        "Team_B": tM,
        "Prob_A_Wins": pA,
        "Prob_B_Wins": 1.0 - pA
    })
    ff1_winner = tE if pA >= 0.5 else tM
    
    # Semifinal2
    tS = all_region_winners["South"]
    tW = all_region_winners["West"]
    pA = matchup_probability(tS, tW, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
    rows.append({
        "Region": "FinalFour",
        "Round": 5,
        "Matchup": "Semifinal2 (South vs West)",
        "Team_A": tS,
        "Team_B": tW,
        "Prob_A_Wins": pA,
        "Prob_B_Wins": 1.0 - pA
    })
    ff2_winner = tS if pA >= 0.5 else tW
    
    # Championship
    pA = matchup_probability(ff1_winner, ff2_winner, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta)
    rows.append({
        "Region": "FinalFour",
        "Round": 6,
        "Matchup": "Championship",
        "Team_A": ff1_winner,
        "Team_B": ff2_winner,
        "Prob_A_Wins": pA,
        "Prob_B_Wins": 1.0 - pA
    })
    return pd.DataFrame(rows)

# Actually produce the bracket matchup probabilities
df_all_matchups = build_full_bracket_probabilities(
    bracket_reordered, seeds_array, seed_R32, lambda_s, lambda_t, stats_array, const_, beta
)
df_all_matchups.to_csv("./markov/all_matchup_probabilities.csv", index=False)

print("✅ Finished producing per-round matchup probabilities in all_matchup_probabilities.csv!")
