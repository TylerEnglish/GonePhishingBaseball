import os
import time
import pandas as pd
import numpy as np
import logging
import gc
from joblib import dump, load

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

import zipfile
# Setup logging
logging.basicConfig(level=logging.INFO)

# Expanded feature list: additional metrics from the dataset
MODEL_FEATURES = [
    'balls', 'strikes', 'outs', 'inning', 'is_first_pitch',
    'relspeed', 'spinrate', 'zonetime', 'avg_pitch_speed', 'avg_spin_rate',
    'strike_percentage', 'ball_percentage', 'pitcher_throws_num', 'batterside_num',
    'vertrelangle', 'horzrelangle', 'vertbreak', 'horzbreak', 'platelocheight', 'platelocside',
    'effectivevelo', 'speeddrop', 'extension'
]

# ------ Recommendation Sampling Config ------
EPSILON = 0.1        # Exploration chance
TEMPERATURE = 1.0    # Softmax temperature scaling

# Global cache for pitcher arsenal
_pitcher_arsenal_cache = {}

# ---------------- Memory Reduction Helper ----------------
def reduce_memory_usage(df):
    """
    Downcasts numeric columns and converts object columns to categorical where beneficial.
    """
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_object_dtype(col_type):
            num_unique_values = df[col].nunique()
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
    return df

# ---------------- Data Ingestion Module ----------------
def load_data(file_path):
    """
    Loads the data from a parquet file and applies memory reduction.
    """
    logging.info(f"Loading data from {file_path}")
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"Data loaded with columns: {df.columns.tolist()}")
        df = reduce_memory_usage(df)
    except Exception as e:
        logging.error(f"Error loading parquet file: {e}")
        raise
    return df

# ---------------- Data Preprocessing and Feature Computation ----------------
def compute_features(df):
    """
    Cleans, preprocesses, and computes new features for the pitch sequencing dataset.
    """
    logging.info("Computing and preprocessing features")
    df = df.copy(deep=True)
    df.columns = df.columns.str.lower()

    # Rename common columns if needed
    if 'batterid' in df.columns and 'batter_id' not in df.columns:
        df.rename(columns={'batterid': 'batter_id'}, inplace=True)
    if 'pitcherid' in df.columns and 'pitcher_id' not in df.columns:
        df.rename(columns={'pitcherid': 'pitcher_id'}, inplace=True)
    
    df = reduce_memory_usage(df)
    
    # Create 'count' if not available
    if 'count' not in df.columns and 'balls' in df.columns and 'strikes' in df.columns:
        df['count'] = df['balls'].astype(str) + '-' + df['strikes'].astype(str)
        df['count'] = df['count'].astype('category')
    
    # Ensure 'last_pitch' exists
    if 'last_pitch' not in df.columns:
        df['last_pitch'] = np.nan

    # Process 'count' for numeric conversion if needed
    if 'count' in df.columns and ((df['balls'].dtype == 'O') or (df['strikes'].dtype == 'O')):
        try:
            df[['balls', 'strikes']] = df['count'].str.split('-', expand=True).astype(int)
        except Exception as e:
            logging.error(f"Error processing 'count' column: {e}")
    
    # Create is_first_pitch flag
    df['is_first_pitch'] = df['last_pitch'].isna().astype(int)
    
    # Label success using vectorized operation:
    if 'cleanpitchcall' in df.columns and 'outsonplay' in df.columns:
        df['successful'] = np.where(df['cleanpitchcall'].str.lower() == "strike", 1,
                                    np.where(df['outsonplay'] > 0, 1, 0))
    else:
        logging.error("Missing 'cleanpitchcall' and/or 'outsonplay'")
        df['successful'] = 0
    
    # Compute next_count using at_bat_id if available; otherwise, fallback on pitchofpa and pitchuid.
    if 'at_bat_id' in df.columns and 'count' in df.columns:
        df['next_count'] = df.groupby('at_bat_id')['count'].shift(-1)
    elif all(col in df.columns for col in ['pitchofpa', 'pitchuid', 'count']):
        logging.info("at_bat_id not found; using pitchofpa and pitchuid to compute next_count")
        df = df.sort_values(by=['pitchofpa', 'pitchuid'])
        df['next_count'] = df.groupby('pitchofpa')['count'].shift(-1)
    else:
        logging.warning("Not enough columns for next_count. Setting as NaN.")
        df['next_count'] = np.nan

    df['next_count'] = df.apply(
        lambda row: ('terminal_out' if row['successful'] == 1 else 'terminal_hit')
        if pd.isna(row['next_count']) else row['next_count'], axis=1
    )
    
    # Compute previous pitch using groupby; fallback if necessary.
    if 'at_bat_id' in df.columns and 'cleanpitchtype' in df.columns:
        df['prev_pitch'] = df.groupby('at_bat_id')['cleanpitchtype'].shift(1)
    elif all(col in df.columns for col in ['pitchofpa', 'cleanpitchtype']):
        df = df.sort_values(by=['pitchofpa'])
        df['prev_pitch'] = df.groupby('pitchofpa')['cleanpitchtype'].shift(1)
    else:
        df['prev_pitch'] = np.nan

    # Fix for categorical fillna for prev_pitch.
    if isinstance(df['prev_pitch'].dtype, pd.CategoricalDtype):
        df['prev_pitch'] = df['prev_pitch'].cat.add_categories("None").fillna("None")
    else:
        df['prev_pitch'] = df['prev_pitch'].fillna("None").astype(str)

    # Build the 'state' column by concatenating 'count' and 'prev_pitch'
    df['state'] = df['count'].astype(str) + "_" + df['prev_pitch'].astype(str)
    df['state'] = df['state'].astype('category')
    df['next_state'] = df.apply(
        lambda row: row['next_count'] if isinstance(row['next_count'], str) and row['next_count'].startswith("terminal")
        else (str(row['next_count']) + "_" + str(row['cleanpitchtype'])), axis=1
    )
    df['next_state'] = df['next_state'].astype('category')
    
    # Encode pitcher and batter handedness numerically.
    if 'pitcherthrows' in df.columns:
        df['pitcher_throws_num'] = df['pitcherthrows'].apply(lambda x: 1 if str(x).upper()=='R' else 0 if str(x).upper()=='L' else -1)
    else:
        df['pitcher_throws_num'] = 0.0
    if 'batterside' in df.columns:
        df['batterside_num'] = df['batterside'].apply(lambda x: 1 if str(x).upper()=='R' else 0 if str(x).upper()=='L' else -1)
    else:
        df['batterside_num'] = 0.0
        
    # Ensure extra features exist and are numeric.
    extra_features = ['zonetime', 'avg_pitch_speed', 'avg_spin_rate', 'strike_percentage', 'ball_percentage',
                      'vertrelangle', 'horzrelangle', 'vertbreak', 'horzbreak', 'platelocheight', 'platelocside',
                      'effectivevelo', 'speeddrop', 'extension']
    for col in extra_features:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    df.drop_duplicates(inplace=True)
    df.dropna(subset=MODEL_FEATURES + ['cleanpitchtype'], inplace=True)
    
    # Only keep successful pitches for training.
    df = df[df['successful'] == 1].copy()
    
    # Fix for "cleanpitchtype": if already categorical, add "Unknown" if missing before fillna.
    if 'cleanpitchtype' in df.columns:
        if isinstance(df['cleanpitchtype'].dtype, pd.CategoricalDtype):
            if "Unknown" not in df['cleanpitchtype'].cat.categories:
                df['cleanpitchtype'] = df['cleanpitchtype'].cat.add_categories("Unknown")
            df['cleanpitchtype'] = df['cleanpitchtype'].fillna("Unknown")
        else:
            df['cleanpitchtype'] = df['cleanpitchtype'].fillna("Unknown").astype('category')
        df['target'] = df['cleanpitchtype']
    else:
        logging.error("'cleanpitchtype' not found for target.")
    
    df = df[df['target'] != "Unknown"].copy()
    df = reduce_memory_usage(df)
    return df

def get_pitcher_arsenal(df, pitcher_id):
    """
    Returns the unique pitch types for the given pitcher, caching results.
    """
    global _pitcher_arsenal_cache
    if pitcher_id in _pitcher_arsenal_cache:
        return _pitcher_arsenal_cache[pitcher_id]
    arsenal = df[df['pitcher_id'] == pitcher_id]['cleanpitchtype'].unique().tolist()
    _pitcher_arsenal_cache[pitcher_id] = arsenal
    return arsenal

def train_supervised_model(features, labels):
    """
    Trains a RandomForestClassifier using a pipeline with scaling and polynomial feature expansion.
    Uses grid search over a compact hyperparameter grid.
    """
    logging.info("Starting supervised model training...")
    start_time = time.time()
    label_counts = labels.value_counts()
    stratify_val = labels if label_counts.min() >= 2 else None
    if stratify_val is None:
        logging.warning("Some classes have fewer than 2 samples.")
        
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=stratify_val)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    param_grid = {
        'rf__n_estimators': [100, 120],
        'rf__max_depth': [None, 10],
        'rf__max_features': ['sqrt', 'log2']
    }
    
    logging.info("Starting grid search for hyperparameter tuning...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    train_acc = best_model.score(X_train, y_train)
    test_acc = best_model.score(X_test, y_test)
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    elapsed = time.time() - start_time
    logging.info(f"Grid search complete in {elapsed:.2f} seconds")
    logging.info(f"Best Params: {grid_search.best_params_}")
    logging.info(f"Training complete. Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
    logging.info(f"Classification Report:\n{report}")
    logging.info(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    del X_train, X_test, y_train, y_test, grid_search
    gc.collect()
    logging.info("Garbage collection complete after training.")
    return best_model

class PitchSequencingMDP:
    """
    Class to estimate transition probabilities and solve the Markov Decision Process (MDP)
    for pitch sequencing.
    """
    def __init__(self, df):
        self.state_space = df['state'].cat.categories.tolist() if 'state' in df.columns else []
        self.action_space = df['cleanpitchtype'].cat.categories.tolist() if 'cleanpitchtype' in df.columns else []
        self.transition_model = {}
        self.policy = {}
        self._df = df

    def estimate_transition_probabilities(self):
        """
        Estimates transition probabilities using groupby aggregation.
        """
        logging.info("Estimating transition probabilities for MDP")
        required_cols = ['state', 'cleanpitchtype', 'next_state']
        if not all(col in self._df.columns for col in required_cols):
            logging.warning("Missing columns for transition estimation.")
            return
        trans_counts = self._df.groupby(['state', 'cleanpitchtype', 'next_state']).size().reset_index(name='count')
        for (state, action), group in trans_counts.groupby(['state', 'cleanpitchtype']):
            total = group['count'].sum()
            self.transition_model[(state, action)] = dict(zip(group['next_state'], group['count'] / total))

    def solve_mdp(self, method='value_iteration'):
        """
        Solves the MDP using value iteration and returns a policy mapping state to best action.
        """
        logging.info(f"Solving MDP using {method}")
        if not self.transition_model:
            for state in self.state_space:
                self.policy[state] = self.action_space[0] if self.action_space else None
            return self.policy
        
        gamma = 0.9
        theta = 1e-4
        max_iterations = 1000
        V = {state: 0 for state in self.state_space}
        iteration = 0
        while iteration < max_iterations:
            delta = 0
            for state in self.state_space:
                q_values = {}
                for action in self.action_space:
                    key = (state, action)
                    if key not in self.transition_model:
                        continue
                    q = 0
                    for next_state, prob in self.transition_model[key].items():
                        if isinstance(next_state, str) and next_state.startswith('terminal'):
                            reward = 1 if 'out' in next_state else (-1 if 'hit' in next_state else 0)
                            next_val = 0
                        else:
                            reward = 0
                            next_val = V.get(next_state, 0)
                        q += prob * (reward + gamma * next_val)
                    q_values[action] = q
                if q_values:
                    best_action_value = max(q_values.values())
                    delta = max(delta, abs(best_action_value - V[state]))
                    V[state] = best_action_value
                else:
                    V[state] = 0
            if delta < theta:
                break
            iteration += 1
        
        for state in self.state_space:
            best_action = None
            best_q = -np.inf
            for action in self.action_space:
                key = (state, action)
                if key not in self.transition_model:
                    continue
                q = 0
                for next_state, prob in self.transition_model[key].items():
                    if isinstance(next_state, str) and next_state.startswith('terminal'):
                        reward = 1 if 'out' in next_state else (-1 if 'hit' in next_state else 0)
                        next_val = 0
                    else:
                        reward = 0
                        next_val = V.get(next_state, 0)
                    q += prob * (reward + gamma * next_val)
                if q > best_q:
                    best_q = q
                    best_action = action
            self.policy[state] = best_action if best_action is not None else (self.action_space[0] if self.action_space else None)
        return self.policy

def prepare_game_state(count: str, outs: int, inning: int, batter_id: int, pitcher_id: int,
                       relspeed: float = None, spinrate: float = None, last_pitch=None,
                       hist_df: pd.DataFrame = None) -> dict:
    """
    Prepares a game state dictionary with all necessary fields for pitch recommendation.
    """
    try:
        balls, strikes = map(int, count.split('-'))
    except Exception as e:
        logging.error(f"Invalid count format: {e}")
        balls, strikes = 0, 0
    
    if relspeed is None or relspeed == 0.0:
        if hist_df is not None:
            avg_relspeed = hist_df.loc[hist_df['pitcher_id'] == pitcher_id, 'relspeed'].mean()
            relspeed = avg_relspeed if not np.isnan(avg_relspeed) else 0.0
        else:
            relspeed = 0.0

    if spinrate is None or spinrate == 0.0:
        if hist_df is not None:
            avg_spinrate = hist_df.loc[hist_df['pitcher_id'] == pitcher_id, 'spinrate'].mean()
            spinrate = avg_spinrate if not np.isnan(avg_spinrate) else 0.0
        else:
            spinrate = 0.0

    state = count + "_" + (str(last_pitch) if last_pitch is not None else "None")
    
    return {
        'count': count,
        'outs': outs,
        'inning': inning,
        'batter_id': batter_id,
        'pitcher_id': pitcher_id,
        'last_pitch': last_pitch,
        'relspeed': relspeed,
        'spinrate': spinrate,
        'state': state,
        'balls': balls,
        'strikes': strikes
    }

def softmax_probabilities(probs, temperature=TEMPERATURE):
    """
    Computes softmax probabilities with temperature scaling.
    """
    scaled = np.power(probs, 1/temperature)
    total = np.sum(scaled)
    return scaled / total if total != 0 else np.ones_like(probs) / len(probs)

def update_game_state(current_state, pitch, mdp):
    """
    Updates the current game state based on the pitch thrown using the MDP transition model.
    """
    current_state['last_pitch'] = pitch
    new_state = current_state['count'] + "_" + str(pitch)
    key = (current_state['state'], pitch)
    if key in mdp.transition_model:
        next_states = list(mdp.transition_model[key].keys())
        probs = list(mdp.transition_model[key].values())
        # Filter out any NaN probabilities and their corresponding next_states.
        valid = [(ns, p) for ns, p in zip(next_states, probs) if not np.isnan(p)]
        if valid:
            next_states, probs = zip(*valid)
            probs = np.array(probs)
            total = probs.sum()
            if total == 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / total
            next_state = np.random.choice(next_states, p=probs)
        else:
            logging.warning("No valid transition probabilities found. Using fallback.")
            available = [s for s in mdp.state_space if s != current_state['state']]
            if available:
                next_state = np.random.choice(available)
            else:
                next_state = current_state['state']
        if isinstance(next_state, str) and next_state.startswith('terminal'):
            logging.info(f"Reached terminal state: {next_state}")
            available = [s for s in mdp.state_space if s != current_state['state']]
            if available:
                new_state = np.random.choice(available)
                logging.info(f"Fallback: updating state from {current_state['state']} to {new_state}")
            else:
                new_state = next_state
        else:
            new_state = next_state
    else:
        logging.info(f"No transition for key ({current_state.get('state')}, {pitch}). Using fallback state update.")
        available = [s for s in mdp.state_space if s != current_state.get('state')]
        if available:
            new_state = np.random.choice(available)
            logging.info(f"Fallback: updating state from {current_state.get('state')} to {new_state}")
    current_state['state'] = new_state
    try:
        cnt, _ = new_state.split('_')
        b, s = cnt.split('-')
        b_val = int(float(b))
        s_val = int(float(s))
        current_state['balls'] = b_val
        current_state['strikes'] = s_val
        current_state['count'] = f"{b_val}-{s_val}"
    except Exception as e:
        logging.error(f"Error updating state: {e}")
    return current_state


def recommend_pitch(game_state, supervised_model, mdp_policy, df_features):
    """
    Recommends the next pitch based on the current game state using the supervised model and MDP policy.
    """
    logging.info(f"Recommending pitch for game state: {game_state}")
    feature_row = pd.DataFrame([game_state])
    try:
        feature_row[['balls', 'strikes']] = feature_row['count'].str.split('-', expand=True).astype(int)
    except Exception as e:
        logging.error(f"Error processing count in game_state: {e}")
    feature_row['last_pitch'] = feature_row.get('last_pitch', np.nan)
    feature_row['is_first_pitch'] = feature_row['last_pitch'].apply(lambda x: 1 if pd.isna(x) else 0)
    for feat in ['relspeed', 'spinrate']:
        if feat not in feature_row.columns:
            feature_row[feat] = 0.0

    # Ensure the feature row has the same columns as used for training.
    if hasattr(supervised_model.named_steps['rf'], "feature_names_in_"):
        feature_names = supervised_model.named_steps['rf'].feature_names_in_
    else:
        feature_names = MODEL_FEATURES
    feature_row = feature_row.reindex(columns=feature_names, fill_value=0)
    
    try:
        probs = supervised_model.predict_proba(feature_row)
        pitch_classes = supervised_model.named_steps['rf'].classes_
        prob_dict = dict(zip(pitch_classes, probs[0]))
    except Exception as e:
        logging.error(f"Error predicting with supervised model: {e}")
        prob_dict = {}
    
    if "Unknown" in prob_dict and len(prob_dict) > 1:
        prob_dict.pop("Unknown")
    
    pitcher_id = game_state.get('pitcher_id')
    pitcher_arsenal = get_pitcher_arsenal(df_features, pitcher_id)
    logging.info(f"Pitcher {pitcher_id} arsenal: {pitcher_arsenal}")

    # Filter to only include pitches in the pitcher's arsenal.
    prob_dict = {p: prob for p, prob in prob_dict.items() if p in pitcher_arsenal}
    if not prob_dict:
        logging.warning("No predictions within pitcher arsenal. Falling back to full predictions.")
        pitcher_arsenal = supervised_model.named_steps['rf'].classes_
        prob_dict = dict(zip(pitch_classes, probs[0]))
        if "Unknown" in prob_dict:
            prob_dict.pop("Unknown")
    
    state_key = game_state.get('state')
    mdp_recommendation = mdp_policy.get(state_key)
    if mdp_recommendation and mdp_recommendation in prob_dict:
        prob_dict[mdp_recommendation] += 0.1

    total_prob = sum(prob_dict.values())
    if total_prob > 0:
        for k in prob_dict:
            prob_dict[k] /= total_prob
    else:
        keys = list(prob_dict.keys())
        prob_dict = {k: 1/len(keys) for k in keys}
    
    pitches = list(prob_dict.keys())
    pitch_probs = np.array([prob_dict[p] for p in pitches])
    
    if np.random.rand() < EPSILON:
        recommended_pitch = np.random.choice(pitches)
    else:
        scaled_probs = softmax_probabilities(pitch_probs, temperature=TEMPERATURE)
        recommended_pitch = np.random.choice(pitches, p=scaled_probs)
    
    return recommended_pitch

def simulate_next_pitches(initial_state, supervised_model, mdp_policy, mdp, df_features, n=3):
    """
    Simulates the next n pitches starting from the given game state.
    """
    sequence = []
    current_state = initial_state.copy()
    for _ in range(n):
        pitch = recommend_pitch(current_state, supervised_model, mdp_policy, df_features)
        sequence.append(pitch)
        current_state = update_game_state(current_state, pitch, mdp)
    return sequence

def run_pretrained_example(model_path, hist_df, mdp_policy, mdp):
    """
    Loads a pretrained model and runs an example pitch recommendation.
    """
    logging.info(f"Loading pretrained model from {model_path}")
    pretrained_model = load(model_path)
    game_state = prepare_game_state(
        count='1-1',
        outs=1,
        inning=6,
        batter_id=1000032366,
        pitcher_id=1000066910,
        hist_df=hist_df
    )
    recommended_pitch = recommend_pitch(game_state, pretrained_model, mdp_policy, hist_df)
    logging.info(f"Pretrained model recommends: {recommended_pitch}")
    return recommended_pitch

def main():
    """
    Main function to load data, compute features, train the model, solve the MDP, and simulate recommendations.
    """
    file_path = "Derived_Data/feature/feature_20250301_105232.parquet"
    df_raw = load_data(file_path)
    df_features = compute_features(df_raw)
    
    if 'target' in df_features.columns:
        labels = df_features['target']
        features = df_features[MODEL_FEATURES].copy()
    else:
        logging.error("Target column not found in features")
        return

    logging.info("Starting supervised model training process...")
    supervised_model = train_supervised_model(features, labels)
    
    model_save_path = "models/recommend/model.pkl"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    dump(supervised_model, model_save_path)
    logging.info(f"Supervised model saved to {model_save_path}")

    zip_path = "models/recommend/model.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(model_save_path, arcname=os.path.basename(model_save_path))
    logging.info(f"Model zipped to {zip_path}")
    
    mdp = PitchSequencingMDP(df_features)
    mdp.estimate_transition_probabilities()
    mdp_policy = mdp.solve_mdp()
    
    game_state = prepare_game_state(
        count='1-1',
        outs=1,
        inning=6,
        batter_id=1000032366,
        pitcher_id=1000066910,
        hist_df=df_features
    )
    
    rec_pitch = recommend_pitch(game_state, supervised_model, mdp_policy, df_features)
    logging.info(f"Batter Recommended Pitch: {rec_pitch}")
    
    next_pitches = simulate_next_pitches(game_state, supervised_model, mdp_policy, mdp, df_features, n=10)
    logging.info(f"Simulated next pitches: {next_pitches}")

if __name__ == '__main__':
    main()
    # To run without training (using the saved model), uncomment the following lines:
    # pretrained_model_path = "models/recommend/model.pkl"
    # df_raw = load_data("Derived_Data/feature/feature_20250301_105232.parquet")
    # df_features = compute_features(df_raw)
    # mdp = PitchSequencingMDP(df_features)
    # mdp.estimate_transition_probabilities()
    # mdp_policy = mdp.solve_mdp()
    # run_pretrained_example(pretrained_model_path, df_features, mdp_policy, mdp)
