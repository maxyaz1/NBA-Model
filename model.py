import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import string
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

def scrape_player_stats(letter):
    """
    Scrape all players whose last name starts with a specific letter
    """
    url = f"https://www.basketball-reference.com/players/{letter.lower()}/"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve data for letter {letter}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # The player table has id 'players'
    player_table = soup.find('table', {'id': 'players'})
    
    if not player_table:
        print(f"No player table found for letter {letter}")
        return None
    
    # Extract the data into a pandas DataFrame
    df = pd.read_html(str(player_table))[0]
    
    return df

def scrape_active_players():
    """
    Scrape player data for active players
    """
    all_players = []
    
    # Go through each letter of the alphabet
    for letter in string.ascii_lowercase:
        print(f"Scraping players with last names starting with: {letter.upper()}")
        letter_df = scrape_player_stats(letter)
        
        if letter_df is not None:
            # Filter to keep only active players (last season is 2024-25)
            if 'To' in letter_df.columns:
                active_players = letter_df[letter_df['To'] >= 2023]
                all_players.append(active_players)
        
        # Be nice to the server
        time.sleep(1)
    
    # Combine all the data
    combined_df = pd.concat(all_players, ignore_index=True)
    
    # Clean and format player names
    combined_df['Player'] = combined_df['Player'].str.replace('*', '', regex=False)
    
    # Save raw data
    combined_df.to_csv('basketball_reference_players_raw.csv', index=False)
    
    return combined_df

def get_detailed_player_stats(player_url):
    """
    Get detailed stats for a player from their individual page
    """
    base_url = "https://www.basketball-reference.com"
    full_url = base_url + player_url
    
    response = requests.get(full_url)
    if response.status_code != 200:
        print(f"Failed to retrieve data for {full_url}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Get the advanced stats table
    advanced_table = soup.find('table', {'id': 'advanced'})
    
    if not advanced_table:
        print(f"No advanced stats found for {full_url}")
        return None
    
    # Extract the advanced stats
    advanced_df = pd.read_html(str(advanced_table))[0]
    
    # Use the most recent season's data
    # Filter rows that are not headers or totals
    recent_season = advanced_df[(advanced_df['Season'] != 'Season') & 
                              (advanced_df['Season'] != 'Career') & 
                              (advanced_df['Season'] != 'Totals')]
    
    if len(recent_season) > 0:
        return recent_season.iloc[-1]  # Get the most recent season
    else:
        return None

def process_and_merge_data():
    """
    Process scraped data and merge with existing salary data
    """
    # Scrape player data if it doesn't exist
    if not os.path.exists('basketball_reference_players_raw.csv'):
        players_df = scrape_active_players()
    else:
        players_df = pd.read_csv('basketball_reference_players_raw.csv')
    
    # Load existing salary data
    if os.path.exists('nba_salary_data.csv'):
        salary_df = pd.read_csv('nba_salary_data.csv')
    else:
        print("Error: nba_salary_data.csv not found")
        return None
    
    # Create a processed dataframe with required stats
    processed_data = []
    
    # Extract player URLs if available, otherwise just use basic stats
    if 'url' in players_df.columns:
        for idx, player in players_df.iterrows():
            detailed_stats = get_detailed_player_stats(player['url'])
            if detailed_stats is not None:
                player_data = {
                    'Player': player['Player'],
                    'PER': detailed_stats.get('PER', 0),
                    'TS_Percentage': detailed_stats.get('TS%', 0),
                    'points': player.get('PTS', 0),
                    'assists': player.get('AST', 0),
                    'reboundsTotal': player.get('TRB', 0),
                    'Simple_PER': float(detailed_stats.get('PER', 0)) if not pd.isna(detailed_stats.get('PER', 0)) else 0
                }
                processed_data.append(player_data)
            time.sleep(0.5)  # Be nice to the server
    else:
        # Use only basic stats from the player listing
        for idx, player in players_df.iterrows():
            player_data = {
                'Player': player['Player'],
                'points': player.get('PTS', 0),
                'assists': player.get('AST', 0),
                'reboundsTotal': player.get('TRB', 0),
                'TS_Percentage': 0,  # Placeholder
                'Simple_PER': 0  # Placeholder
            }
            processed_data.append(player_data)
    
    stats_df = pd.DataFrame(processed_data)
    
    # Merge with salary data
    merged_df = pd.merge(stats_df, salary_df, on='Player', how='inner')
    
    # Save the merged dataset
    merged_df.to_csv('player_stats_and_salary_fixed (1).csv', index=False)
    
    return merged_df

def train_model():
    """Train and return the salary prediction model"""
    try:
        # Try different possible paths to find the data file
        possible_paths = [
            'data/nba_salary_stats_merged.csv',
            'nba_salary_stats_merged.csv',
            './data/nba_salary_stats_merged.csv',
            '../data/nba_salary_stats_merged.csv'
        ]
        
        data = None
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    print(f"Found data file at: {path}")
                    data = pd.read_csv(path)
                    break
            except Exception as e:
                print(f"Error trying path {path}: {e}")
        
        if data is None:
            # As a backup, try to create a simple mock dataset
            print("Creating mock dataset for model training")
            mock_data = {
                'points': [25.7, 26.4, 27.1, 30.4, 26.4, 33.9, 34.7, 23.7, 26.9, 24.5],
                'assists': [8.3, 5.1, 5.3, 6.5, 9.0, 9.8, 5.6, 3.6, 4.9, 7.0],
                'reboundsTotal': [7.3, 4.5, 6.9, 11.5, 12.4, 9.2, 11.2, 6.1, 8.1, 4.4],
                'TS_Percentage': [0.61, 0.63, 0.64, 0.61, 0.65, 0.62, 0.65, 0.59, 0.58, 0.61],
                'Simple_PER': [22.5, 24.3, 25.6, 31.8, 31.3, 28.4, 31.6, 24.4, 23.6, 22.2],
                'TeamSalaryCommitment': [192057940, 178316619, 220708856, 185971982, 185864258, 178812859, 174059777, 174124752, 195610488, 185971982],
                'Salary': [47600000, 55760130, 51207168, 48787676, 48016920, 44290000, 53763753, 45640084, 37845020, 45650000]
            }
            data = pd.DataFrame(mock_data)
            # Save the mock data to file for future use
            os.makedirs('data', exist_ok=True)
            data.to_csv('data/mock_salary_data.csv', index=False)
            print("Created and saved mock dataset")
        
        # Select features and target
        features = ['points', 'assists', 'reboundsTotal', 'TS_Percentage', 'Simple_PER']
        if 'TeamSalaryCommitment' in data.columns:
            features.append('TeamSalaryCommitment')
        
        target = 'Salary'
        
        # Make sure all features are numeric
        for feature in features:
            if feature in data.columns:
                if data[feature].dtype == 'object':
                    data[feature] = pd.to_numeric(data[feature], errors='coerce')
                    data[feature].fillna(data[feature].median(), inplace=True)
        
        # Drop rows with NaN
        data = data.dropna(subset=features + [target])
        
        print(f"Training model with {len(data)} samples and features: {features}")
        
        # Train-test split
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        # Ensure the 'models' directory exists
        os.makedirs('models', exist_ok=True)
        
        # Save model
        with open('models/salary_prediction_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print("Model trained and saved successfully")
        return model
    
    except Exception as e:
        print(f"Error in train_model: {e}")
        import traceback
        traceback.print_exc()
        raise

def predict_salary_with_cap_and_mle(player_features, model):
    """
    Predict player salary using the trained model,
    with a cap for top 3% players,
    adjustment for MLE-range salaries,
    and MAX salary for players with rating > 35
    """
    if model is None:
        return None

    # NBA salary constants
    NBA_SALARY_CAP = 140588000
    MAX_SALARY_PERCENTAGE = 0.35
    MAX_SALARY = NBA_SALARY_CAP * MAX_SALARY_PERCENTAGE
    MID_LEVEL_EXCEPTION = 12822000
    MLE_THRESHOLD = 3000000  # $3 million threshold around the MLE

    # Step 1: Check for high player rating
    if 'rating' in player_features.columns:
        rating_value = player_features['rating'].iloc[0]
        if rating_value > 35:
            return MAX_SALARY

    # Step 2: Predict with model
    expected_features = ['points', 'assists', 'reboundsTotal', 'TS_Percentage',
                         'Simple_PER', 'TeamSalaryCommitment']
    prediction_df = pd.DataFrame()

    for feature in expected_features:
        if feature in player_features.columns:
            prediction_df[feature] = player_features[feature]
        else:
            prediction_df[feature] = 0

    predicted_salary = model.predict(prediction_df)[0]

    # Step 3: Apply 97th percentile override
    try:
        with open('models/salary_percentile_threshold.pkl', 'rb') as f:
            percentile_97 = pickle.load(f)
        if predicted_salary >= percentile_97:
            return MAX_SALARY
    except FileNotFoundError:
        pass

    # Step 4: MLE adjustment
    if abs(predicted_salary - MID_LEVEL_EXCEPTION) <= MLE_THRESHOLD:
        return MID_LEVEL_EXCEPTION

    # Step 5: Return regular prediction
    return predicted_salary

# Streamlit app integration
def prepare_player_features(player_stats, player_info):
    # Team salary data for the 2024/25 season
    team_salary_data = {
        "Phoenix": 220708856,
        "Minnesota": 204780898,
        "Boston": 195610488,
        "New York": 193588886,
        "LA Lakers": 192057940,
        "Milwaukee": 185971982,
        "Denver": 185864258,
        "Dallas": 178812859,
        "Golden State": 178316619,
        "Miami": 176102077,
        "New Orleans": 175581168,
        "LA Clippers": 174124752,
        "Philadelphia": 174059777,
        "Washington": 173873325,
        "Toronto": 173621417,
        "Sacramento": 172815356,
        "Cleveland": 172471107,
        "Charlotte": 171952448,
        "Brooklyn": 171804859,
        "Atlanta": 170056977,
        "Houston": 170038023,
        "Indiana": 169846170,
        "Portland": 169031747,
        "Chicago": 168147899,
        "Oklahoma City": 167471133,
        "Memphis": 165903638,
        "San Antonio": 164872330,
        "Utah": 156874018,
        "Orlando": 151728562,
        "Detroit": 140746162,
    }

    # Get the player's team name
    team_name = player_info['TEAM_NAME'].values[0]

    # Fetch the team salary commitment, default to 120M if not found
    team_salary_commitment = team_salary_data.get(team_name, 120000000)

    # Prepare features
    features = {
        'points': player_stats['PTS'].astype(float).mean(),
        'assists': player_stats['AST'].astype(float).mean(),
        'reboundsTotal': player_stats['REB'].astype(float).mean(),
        'TS_Percentage': player_stats['PTS'].sum() / (2 * (player_stats['FGA'].sum() + 0.44 * player_stats['FTA'].sum())),
        'Simple_PER': (
            player_stats['PTS'].astype(float).mean() +
            player_stats['REB'].astype(float).mean() * 1.2 +
            player_stats['AST'].astype(float).mean() * 1.5 +
            player_stats['STL'].astype(float).mean() * 2 +
            player_stats['BLK'].astype(float).mean() * 2 -
            player_stats['TOV'].astype(float).mean() * 1.2
        ) / player_stats['MIN'].astype(float).mean(),
        'TeamSalaryCommitment': team_salary_commitment,
    }

    # Return only the features used during training
    return pd.DataFrame([features])

# Modified version of predict_salary function to use in the Streamlit app
def predict_salary(player_features, model):
    """
    Wrapper function to use the enhanced salary prediction with cap and MLE
    """
    return predict_salary_with_cap_and_mle(player_features, model)

# If this file is run directly, run the process
if __name__ == "__main__":
    train_model()
