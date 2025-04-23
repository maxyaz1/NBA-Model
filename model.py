import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
import pickle

def train_model():
    """Train and return the salary prediction model using embedded data"""
    try:
        # Create a mock dataset directly in the code
        print("Creating embedded dataset for model training")
        mock_data = {
            'points': [25.7, 26.4, 27.1, 30.4, 26.4, 33.9, 34.7, 23.7, 26.9, 24.5, 28.3, 22.1, 20.5, 19.8, 18.3],
            'assists': [8.3, 5.1, 5.3, 6.5, 9.0, 9.8, 5.6, 3.6, 4.9, 7.0, 6.3, 7.5, 4.2, 3.5, 5.8],
            'reboundsTotal': [7.3, 4.5, 6.9, 11.5, 12.4, 9.2, 11.2, 6.1, 8.1, 4.4, 5.2, 4.1, 7.8, 9.5, 3.2],
            'TS_Percentage': [0.61, 0.63, 0.64, 0.61, 0.65, 0.62, 0.65, 0.59, 0.58, 0.61, 0.57, 0.60, 0.58, 0.56, 0.59],
            'Simple_PER': [22.5, 24.3, 25.6, 31.8, 31.3, 28.4, 31.6, 24.4, 23.6, 22.2, 21.5, 20.8, 19.2, 18.8, 17.5],
            'TeamSalaryCommitment': [192057940, 178316619, 220708856, 185971982, 185864258, 178812859, 174059777, 
                                     174124752, 195610488, 185971982, 169846170, 167471133, 176102077, 151728562, 140746162],
            'Salary': [47600000, 55760130, 51207168, 48787676, 48016920, 44290000, 53763753, 45640084, 37845020, 
                       45650000, 30500000, 25500000, 22000000, 17800000, 12500000]
        }
        
        # Create DataFrame
        data = pd.DataFrame(mock_data)
        
        # Select features and target
        features = ['points', 'assists', 'reboundsTotal', 'TS_Percentage', 'Simple_PER', 'TeamSalaryCommitment']
        target = 'Salary'
        
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

def predict_salary_with_cap_and_tiers(player_features, model, info_df, player_name=None):
    """
    Predict player salary using percentage of cap method with appropriate tiers
    """
    if model is None:
        return None
   
    # Current NBA salary cap
    NBA_SALARY_CAP = 140588000  # 2024-25 season
    
    # NBA salary constants for tiers
    MIN_SALARY_PCT = 2.5  # Tier 1: Minimum salary (approximate percentage)
    BAE_PCT = 3.32  # Tier 2: Bi-annual exception
    ROOM_PCT = 5.678  # Tier 3: Room exception
    MLE_PCT = 9.0  # Tier 4: Mid-Level Exception
    
    # Check if player name was provided
    player_is_all_nba = False
    if player_name:
        # All-NBA Players (2023-24 and 2022-23 seasons)
        all_nba_players = [
            # 2023-24 First Team
            "Giannis Antetokounmpo", "Luka Dončić", "Shai Gilgeous-Alexander", 
            "Nikola Jokić", "Jayson Tatum",
            # 2023-24 Second Team
            "Jalen Brunson", "Anthony Davis", "Kevin Durant", 
            "Anthony Edwards", "Kawhi Leonard",
            # 2023-24 Third Team
            "Devin Booker", "Stephen Curry", "Tyrese Haliburton", 
            "LeBron James", "Domantas Sabonis",
            # 2022-23 First Team
            "Joel Embiid",
            # 2022-23 Second Team
            "Jimmy Butler", "Jaylen Brown", "Donovan Mitchell",
            # 2022-23 Third Team
            "Julius Randle", "De'Aaron Fox", "Damian Lillard"
        ]
        
        # Standardize player name for comparison
        for i, p in enumerate(all_nba_players):
            # Remove diacritics for more reliable comparison
            all_nba_players[i] = p.replace("č", "c").replace("ć", "c")
        
        # Check if player is in All-NBA list (with standardized name)
        standardized_player_name = player_name.replace("č", "c").replace("ć", "c")
        player_is_all_nba = any(name.lower() in standardized_player_name.lower() for name in all_nba_players)
    
    # List of features the model was trained with
    expected_features = ['points', 'assists', 'reboundsTotal', 'TS_Percentage', 'Simple_PER', 'TeamSalaryCommitment']
   
    # Create a DataFrame with only the expected features
    prediction_df = pd.DataFrame()
   
    # Add only the features the model expects
    for feature in expected_features:
        if feature in player_features.columns:
            prediction_df[feature] = player_features[feature]
        else:
            # If a required feature is missing, add it with a default value
            prediction_df[feature] = 0
    
    # Get the player's experience
    experience = int(info_df['SEASON_EXP'].values[0])
    
    # Determine max eligible percentage based on experience
    if experience <= 6:
        max_eligible_pct = 25.0  # 0-6 years: 25% of cap
    elif experience <= 9:
        max_eligible_pct = 30.0  # 7-9 years: 30% of cap  
    else:
        max_eligible_pct = 35.0  # 10+ years: 35% of cap
    
    # If player is All-NBA, automatically assign max contract
    if player_is_all_nba:
        salary_pct = max_eligible_pct
        tier = 9  # Max tier
        tier_name = f"Maximum Contract ({max_eligible_pct}% of Cap) - All-NBA Player"
        # Set a high player rating for consistency
        player_rating = 100.0
    else:
        # Make initial prediction
        initial_predicted_salary = model.predict(prediction_df)[0]
        
        # Convert to percentage of salary cap
        predicted_pct = (initial_predicted_salary / NBA_SALARY_CAP) * 100
        
        # Define a player rating based on key features to identify top players
        player_rating = (
            player_features['points'].values[0] * 1.0 +
            player_features['assists'].values[0] * 0.7 +
            player_features['reboundsTotal'].values[0] * 0.5 +
            player_features['Simple_PER'].values[0] * 3.0
        )
        
        # Thresholds for top players
        TOP_PLAYER_THRESHOLD = 50.0  # Players above this rating get max contract
        HIGH_TIER_THRESHOLD = 40.0   # Players above this get 25-30% tier
        
        # Determine the salary tier and adjusted percentage based on rating and prediction
        if player_rating >= TOP_PLAYER_THRESHOLD:
            # Top players always get max eligible percentage
            salary_pct = max_eligible_pct
            tier = 9  # Max tier
            tier_name = f"Maximum Contract ({max_eligible_pct}% of Cap)"
        elif player_rating >= HIGH_TIER_THRESHOLD or predicted_pct >= 25.0:
            # High tier players get 25-30% (capped at their max eligible)
            salary_pct = min(27.5, max_eligible_pct)
            tier = 8  # 25-30% tier
            tier_name = "25-30% of Cap"
        elif predicted_pct >= 20.0:
            salary_pct = 22.5  # Middle of Tier 7
            tier = 7  # 20-25% tier
            tier_name = "20-25% of Cap"
        elif predicted_pct >= 15.0:
            salary_pct = 17.5  # Middle of Tier 6
            tier = 6  # 15-20% tier
            tier_name = "15-20% of Cap"
        elif predicted_pct >= 10.0:
            salary_pct = 12.5  # Middle of Tier 5
            tier = 5  # 10-15% tier
            tier_name = "10-15% of Cap"
        elif abs(predicted_pct - MLE_PCT) <= 1.5:  # Within 1.5% of MLE
            salary_pct = MLE_PCT
            tier = 4  # MLE tier
            tier_name = "Mid-Level Exception"
        elif abs(predicted_pct - ROOM_PCT) <= 1.0:  # Within 1.0% of Room Exception
            salary_pct = ROOM_PCT
            tier = 3  # Room exception tier
            tier_name = "Room Exception"
        elif abs(predicted_pct - BAE_PCT) <= 0.7:  # Within 0.7% of BAE
            salary_pct = BAE_PCT
            tier = 2  # BAE tier
            tier_name = "Bi-Annual Exception"
        elif predicted_pct <= 3.0:
            salary_pct = MIN_SALARY_PCT
            tier = 1  # Minimum salary tier
            tier_name = "Minimum Salary"
        else:
            salary_pct = predicted_pct
            tier = 0  # No specific tier
            tier_name = "Role Player"
    
    # Calculate final salary
    final_salary = NBA_SALARY_CAP * (salary_pct / 100)
    
    result = {
        'Salary': final_salary,
        'Percentage': salary_pct,
        'Tier': tier,
        'TierName': tier_name,
        'Experience': experience,
        'MaxEligible': max_eligible_pct,
        'PlayerRating': player_rating,
        'IsAllNBA': player_is_all_nba
    }
    
    return result

def prepare_player_features(player_stats, player_info):
    # Team salary data
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

    # Return the features
    return pd.DataFrame([features])

# If this file is run directly, train the model
if __name__ == "__main__":
    train_model()
