import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL for the 2002 NCAA Tournament Summary page
url = "https://www.sports-reference.com/cbb/postseason/men/2002-ncaa.html"
response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"Failed to load page: {response.status_code}")

# Parse the page HTML
soup = BeautifulSoup(response.text, "html.parser")

# Define the regions to iterate over
regions = ["east", "midwest", "south", "west", "national"]
games = []

# Helper function: parse a team block to extract seed, name, score, and winner flag
def parse_team(team_div):
    # Get the seed (assumed to be in the first <span>)
    seed_tag = team_div.find("span")
    seed = seed_tag.get_text(strip=True) if seed_tag else ""
    
    # Get all <a> tags in this div
    a_tags = team_div.find_all("a")
    team_name = a_tags[0].get_text(strip=True) if len(a_tags) >= 1 else ""
    score = a_tags[1].get_text(strip=True) if len(a_tags) >= 2 else ""
    
    # Determine if this team is marked as the winner (its div has class "winner")
    is_winner = "winner" in team_div.get("class", [])
    return seed, team_name, score, is_winner

# Loop through each region
for region in regions:
    region_div = soup.find("div", id=region)
    if not region_div:
        continue

    # Find the bracket container in this region (the structured games are inside this div)
    bracket_div = region_div.find("div", id="bracket")
    if not bracket_div:
        continue

    # Get all rounds (each round is a direct child <div class="round">)
    rounds = bracket_div.find_all("div", class_="round", recursive=False)
    for round_index, round_div in enumerate(rounds, start=1):
        # Each game is assumed to be a direct child <div> of the round div
        game_divs = round_div.find_all("div", recursive=False)
        for game_index, game_div in enumerate(game_divs, start=1):
            # Within a game, get the team blocks (direct child <div> elements)
            team_divs = game_div.find_all("div", recursive=False)
            if len(team_divs) < 2:
                # If there are fewer than 2 team divs, skip this game (e.g. a bye or placeholder)
                continue

            # Parse the two teams
            teamA = parse_team(team_divs[0])
            teamB = parse_team(team_divs[1])

            # Label the winner: 1 if Team A is marked as winner; if not and Team B is, label 0.
            if teamA[3]:
                winner_label = 1
            elif teamB[3]:
                winner_label = 0
            else:
                winner_label = None

            # Get the location string.
            # We assume the location is in a direct child <span> of the game_div (outside the team divs)
            direct_spans = game_div.find_all("span", recursive=False)
            location = direct_spans[0].get_text(strip=True) if direct_spans else ""

            games.append({
                "Region": region.capitalize(),
                "Round": round_index,
                "Game": game_index,
                "Team_A_Seed": teamA[0],
                "Team_A": teamA[1],
                "Team_A_Score": teamA[2],
                "Team_B_Seed": teamB[0],
                "Team_B": teamB[1],
                "Team_B_Score": teamB[2],
                "Winner": winner_label,
                "Location": location
            })

# Create output directory if it doesn't exist
output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)

# Save the games data to CSV
df = pd.DataFrame(games)
output_file = os.path.join(output_dir, "ncaa_2002_tournament.csv")
df.to_csv(output_file, index=False)
print(f"✅ Data saved to {output_file}")
