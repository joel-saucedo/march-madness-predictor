import os 
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Set the target year to 2025 and initialize the list for game data
year = 2025
games = []

# Helper function: parse a team block to extract seed, team name, score, and winner flag
def parse_team(team_div):
    seed_tag = team_div.find("span")
    seed = seed_tag.get_text(strip=True) if seed_tag else ""
    
    # Get all <a> tags in the team block
    a_tags = team_div.find_all("a")
    team_name = a_tags[0].get_text(strip=True) if len(a_tags) >= 1 else ""
    score = a_tags[1].get_text(strip=True) if len(a_tags) >= 2 else ""
    
    # Check if this team div has a class "winner"
    is_winner = "winner" in team_div.get("class", [])
    return seed, team_name, score, is_winner

# Build the URL for 2025
url = f"https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html"
print(f"Processing year {year}...")
response = requests.get(url)
if response.status_code != 200:
    print(f"⚠️ Failed to load page for year {year}: {response.status_code}")
else:
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Get the region labels from the "switcher filter"
    switcher = soup.find("div", class_="switcher filter")
    if not switcher:
        print(f"⚠️ No region switcher found for year {year}")
    else:
        region_labels = [div.find("a").get_text(strip=True) for div in switcher.find_all("div")]
        
        # Get the container that holds all region brackets
        brackets_container = soup.find("div", id="brackets")
        if not brackets_container:
            print(f"⚠️ No brackets container found for year {year}")
        else:
            # Get the region containers (assumed to be direct children)
            region_containers = brackets_container.find_all("div", recursive=False)
            if len(region_labels) < 4 or len(region_containers) < 4:
                print(f"⚠️ Unexpected number of regions for year {year} (labels: {len(region_labels)}, containers: {len(region_containers)})")
            else:
                # Loop over the regions in order using zip (assumes same order)
                for region_label, region_div in zip(region_labels, region_containers):
                    # The actual bracket games are in a child div with id="bracket"
                    bracket_div = region_div.find("div", id="bracket")
                    if not bracket_div:
                        continue

                    # Each round is assumed to be a direct child div with class "round"
                    rounds = bracket_div.find_all("div", class_="round", recursive=False)
                    for round_index, round_div in enumerate(rounds, start=1):
                        # Each game is assumed to be a direct child div of the round div
                        game_divs = round_div.find_all("div", recursive=False)
                        for game_index, game_div in enumerate(game_divs, start=1):
                            # Each game should have two team blocks
                            team_divs = game_div.find_all("div", recursive=False)
                            if len(team_divs) < 2:
                                # Skip games with fewer than 2 teams (e.g., byes/placeholders)
                                continue

                            teamA = parse_team(team_divs[0])
                            teamB = parse_team(team_divs[1])

                            # Label the winner: 1 if Team A is marked as winner, else 0 if Team B is marked, else None
                            if teamA[3]:
                                winner_label = 1
                            elif teamB[3]:
                                winner_label = 0
                            else:
                                winner_label = None

                            # Get the location string, if available (assumed in a direct child <span> of game_div)
                            direct_spans = game_div.find_all("span", recursive=False)
                            location = direct_spans[0].get_text(strip=True) if direct_spans else ""

                            games.append({
                                "Year": year,
                                "Region": region_label,
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

# Create the output directory if it doesn't exist
output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)

# Save the aggregated game data to a CSV file for 2025
df = pd.DataFrame(games)
output_file = os.path.join(output_dir, "ncaa_2025_tournament.csv")
df.to_csv(output_file, index=False)
print(f"✅ Data saved to {output_file}")
