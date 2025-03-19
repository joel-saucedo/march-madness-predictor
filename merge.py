import os
import re
import pandas as pd
from thefuzz import process, fuzz  # pip install thefuzz[speedup]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##########################################
# 1. Helper Functions for Standardization
##########################################

def standardize_team_name(name):
    """
    Standardize team names: lower-case, remove punctuation and trailing numbers.
    """
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)  # Remove punctuation
    name = re.sub(r"\s*\d+$", "", name)  # Remove trailing numbers
    name = re.sub(r"\s+", " ", name).strip()
    return name

def get_best_match(team_name, candidate_names, threshold=90):
    """
    Use fuzzy matching to get the best match for a team name.
    Returns None if no match meets the threshold.
    """
    best_match, score = process.extractOne(team_name, candidate_names, scorer=fuzz.token_sort_ratio)
    return best_match if score >= threshold else None

##########################################
# 2. Loading and Preparing the Data
##########################################

# File paths
tourney_file = "./data/ncaa_2002_2024_tournament.csv"
kenpom_file = "./kenpom/kenpom_data.csv"

# Load datasets
tourney_df = pd.read_csv(tourney_file)
kenpom_df = pd.read_csv(kenpom_file)

# Standardize KenPom team names
kenpom_df['Team_std'] = kenpom_df['Team'].apply(standardize_team_name)

##########################################
# 3. Fuzzy Matching and Data Cleaning
##########################################

def match_team(row_team, kenpom_year_df, threshold=90):
    """
    Matches a tournament team to the closest KenPom team.
    If no match meets the threshold, returns None.
    """
    std_name = standardize_team_name(row_team)
    candidates = kenpom_year_df["Team_std"].tolist()
    return get_best_match(std_name, candidates, threshold)

# Create lists for matched data
valid_games = []
unmatched_teams = []

# Iterate through tournament games
for _, game in tourney_df.iterrows():
    year = game["Year"]
    kenpom_year_df = kenpom_df[kenpom_df["Year"] == year]
    
    if kenpom_year_df.empty:
        continue  # Skip years without KenPom data

    teamA_match = match_team(game["Team_A"], kenpom_year_df)
    teamB_match = match_team(game["Team_B"], kenpom_year_df)
    
    if teamA_match and teamB_match:
        # Fetch matching KenPom rows
        teamA_data = kenpom_year_df[kenpom_year_df["Team_std"] == teamA_match].iloc[0]
        teamB_data = kenpom_year_df[kenpom_year_df["Team_std"] == teamB_match].iloc[0]

        # Append cleaned row
        valid_games.append({
            "Year": year,
            "Region": game["Region"],
            "Round": game["Round"],
            "Game": game["Game"],
            "Team_A": game["Team_A"],
            "Team_A_Seed": game["Team_A_Seed"],
            "Team_A_Score": game["Team_A_Score"],
            "Team_B": game["Team_B"],
            "Team_B_Seed": game["Team_B_Seed"],
            "Team_B_Score": game["Team_B_Score"],
            "Winner": game["Winner"],
            **{f"Team_A_{col}": teamA_data[col] for col in kenpom_df.columns if col not in ["Year", "Team", "Team_std"]},
            **{f"Team_B_{col}": teamB_data[col] for col in kenpom_df.columns if col not in ["Year", "Team", "Team_std"]}
        })
    else:
        # Store unmatched teams
        unmatched_teams.append((year, game["Team_A"], teamA_match, game["Team_B"], teamB_match))

# Convert valid games to DataFrame
cleaned_df = pd.DataFrame(valid_games)

##########################################
# 4. Feature Engineering
##########################################

# Compute differences between key KenPom stats
metrics = ["NetRtg", "ORtg", "DRtg", "AdjT", "Luck", "Strength of Schedule NetRtg"]
for metric in metrics:
    cleaned_df[f"{metric}_diff"] = cleaned_df[f"Team_A_{metric}"] - cleaned_df[f"Team_B_{metric}"]

# Drop rows with missing values
cleaned_df.dropna(inplace=True)

##########################################
# 5. Model Training - Logistic Regression
##########################################

X = cleaned_df[[f"{metric}_diff" for metric in metrics]]
y = cleaned_df["Winner"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Logistic Regression Test Accuracy: {accuracy:.3f}")

# Save cleaned dataset
output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)
cleaned_output_file = os.path.join(output_dir, "ncaa_2002_2024_cleaned.csv")
cleaned_df.to_csv(cleaned_output_file, index=False)

# Save unmatched teams for review
unmatched_output_file = os.path.join(output_dir, "unmatched_teams.csv")
pd.DataFrame(unmatched_teams, columns=["Year", "Team_A", "Matched_A", "Team_B", "Matched_B"]).to_csv(unmatched_output_file, index=False)

print(f"✅ Cleaned data saved to {cleaned_output_file}")
print(f"⚠️ Unmatched teams saved to {unmatched_output_file}")
