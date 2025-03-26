import os
import time
import pandas as pd
import numpy as np
import json
import logging
import gc
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
# AutoGluon
from autogluon.tabular import TabularDataset, TabularPredictor

import zipfile

# Setup logging
logging.basicConfig(level=logging.INFO)

# ------ Recommendation Sampling Config ------
EPSILON = 0.1        # Exploration chance
TEMPERATURE = 1.0    # Softmax temperature scaling
MODEL_FEATURES = []
# Global cache for pitcher arsenal
_pitcher_arsenal_cache = {}

def fill_missing_features(game_state, full_feature_list, model_defaults):
    """
    Given a game state and the full feature list used during training,
    return a DataFrame row that includes all required features.
    Missing features are filled using model_defaults.
    """
    full_data = {col: model_defaults.get(col, 0) for col in full_feature_list}
    full_data.update(game_state)
    return pd.DataFrame([full_data], columns=full_feature_list)

# ---------------- Memory Reduction Helper ----------------
def reduce_memory_usage(df):
    """
    Downcasts numeric columns and converts object columns to categorical where beneficial.
    Skips columns with zero rows.
    """
    for col in df.columns:
        num_total_values = len(df[col])
        if num_total_values == 0:
            continue
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_object_dtype(col_type):
            num_unique_values = df[col].nunique()
            if num_total_values > 0 and (num_unique_values / num_total_values) < 0.5:
                df[col] = df[col].astype('category')
    return df

# ---------------- Data Ingestion Module ----------------
def load_data(file_path):
    """
    Loads the full dataset from a parquet file and applies memory reduction.
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
    # Drop datetime columns from the computed features
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if datetime_cols:
        logging.info(f"Dropping datetime columns in compute_features: {datetime_cols}")
        df = df.drop(columns=datetime_cols)

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

    # Fill missing prev_pitch values.
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
    
    # Remove duplicate rows.
    df.drop_duplicates(inplace=True)
    # Instead of dropping all rows with any NA, drop only rows missing essential columns.
    essential_cols = ['cleanpitchtype']  # You can add more essential columns here if needed.
    df.dropna(subset=essential_cols, inplace=True)
    
    # Only keep successful pitches for training.
    df = df[df['successful'] == 1].copy()
    
    # Fix for "cleanpitchtype": fill missing values and filter out "Unknown"
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
    logging.info(f"Features computed: {df.shape[0]} rows, {df.shape[1]} columns")
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

#====================
# AutoGluon Model
#====================
class AutoGluonModel(BaseEstimator, ClassifierMixin):
    """
    A lightweight scikit-learn wrapper for the trained AutoGluon predictor.
    """
    def __init__(self, predictor=None, label_col='target'):
        self.predictor = predictor
        self.label_col = label_col
        self._classes = None
        # This attribute will store the list of feature names.
        self.feature_names_in_ = []

    def fit(self, X, y):
        self._classes = np.unique(y)
        return self

    def predict(self, X):
        df = TabularDataset(pd.DataFrame(X))
        preds = self.predictor.predict(df)
        return preds.values if hasattr(preds, 'values') else preds

    def predict_proba(self, X):
        df = TabularDataset(pd.DataFrame(X))
        raw_proba = self.predictor.predict_proba(df)
        proba_df = pd.DataFrame(raw_proba)
        col_order = sorted(proba_df.columns)
        proba_df = proba_df[col_order]
        class_list = list(self._classes)
        if sorted(class_list) != col_order:
            logging.warning("Mismatch in class ordering between self._classes and AutoGluon output. Attempting alignment.")
            class_list = col_order  # override
            self._classes = np.array(col_order)
        proba_array = proba_df.values
        return proba_array

    @property
    def classes_(self):
        return self._classes if self._classes is not None else []

def train_supervised_model(features, labels):
    logging.info("Starting advanced AutoGluon-based training (better backend).")
    start_time = time.time()

    df_ag = pd.DataFrame(features.copy())
    df_ag['target'] = labels.values

    label_counts = labels.value_counts()
    stratify_val = labels if label_counts.min() >= 2 else None
    X_train, X_test = train_test_split(
        df_ag,
        test_size=0.2,
        random_state=42,
        stratify=stratify_val
    )
    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    train_data = TabularDataset(X_train)
    test_data = TabularDataset(X_test)
    label_col = 'target'

    save_dir = 'autogluon_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    predictor = TabularPredictor(
        label=label_col,
        eval_metric='f1_weighted',
        path=save_dir
    ).fit(
        train_data=train_data,
        time_limit=1200,
        presets='medium_quality_faster_train',
        verbosity=2
    )

    leaderboard = predictor.leaderboard(test_data, silent=True)
    logging.info(f"AutoGluon leaderboard:\n{leaderboard}")

    y_test_true = test_data[label_col]
    y_test_pred = predictor.predict(test_data)
    test_acc = accuracy_score(y_test_true, y_test_pred)
    p_ = precision_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
    r_ = recall_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
    f_ = f1_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
    logging.info(f"AutoGluon test accuracy: {test_acc:.3f}")
    logging.info(f"Precision: {p_:.3f}, Recall: {r_:.3f}, F1: {f_:.3f}")
    logging.info(f"Classification Report:\n{classification_report(y_test_true, y_test_pred)}")

    elapsed = time.time() - start_time
    logging.info(f"AutoGluon ensemble training finished in {elapsed:.2f} sec.")

    # Set the global MODEL_FEATURES and MODEL_DEFAULTS
    global MODEL_FEATURES  # (You can remove MODEL_DEFAULTS from the global scope)
    MODEL_FEATURES = list(features.columns)
    MODEL_DEFAULTS = {}
    for col in MODEL_FEATURES:
        if pd.api.types.is_numeric_dtype(features[col]):
            MODEL_DEFAULTS[col] = features[col].median()
        else:
            mode_val = features[col].mode()
            if not mode_val.empty:
                MODEL_DEFAULTS[col] = mode_val.iloc[0]
            else:
                MODEL_DEFAULTS[col] = "Unknown"

    final_pipeline = Pipeline([
        ('rf', AutoGluonModel(predictor=predictor))
    ])

    final_pipeline.fit(features, labels)
    
    # Save the feature list in the model
    feature_list = list(features.columns)
    final_pipeline.named_steps['rf'].feature_names_in_ = feature_list
    logging.info(f"Feature list stored in model: {feature_list}")
    
    # Save the model defaults inside the model (so they are available at prediction time)
    final_pipeline.named_steps['rf'].model_defaults_ = MODEL_DEFAULTS

    # Optionally, also save the feature list to a JSON file.
    feature_list_path = "models/recommend/feature_list.json"
    os.makedirs(os.path.dirname(feature_list_path), exist_ok=True)
    with open(feature_list_path, "w") as f:
        json.dump(feature_list, f)
    logging.info(f"Feature list saved to {feature_list_path}")

    return final_pipeline

def testing_phase(features, labels, model):
    """
    Splits the data into a new training and holdout set,
    evaluates the model on both, and logs performance metrics.
    """
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Evaluate on the training set
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    train_report = classification_report(y_train, train_preds)
    logging.info("=== Training Set Performance ===")
    logging.info(f"Accuracy: {train_acc:.3f}")
    logging.info(f"Classification Report:\n{train_report}")
    
    # Evaluate on the holdout test set
    holdout_preds = model.predict(X_holdout)
    holdout_acc = accuracy_score(y_holdout, holdout_preds)
    holdout_report = classification_report(y_holdout, holdout_preds)
    logging.info("=== Holdout Test Set Performance ===")
    logging.info(f"Accuracy: {holdout_acc:.3f}")
    logging.info(f"Classification Report:\n{holdout_report}")

class PitchSequencingMDP:
    """
    Estimates transition probabilities and solves the Markov Decision Process (MDP) for pitch sequencing.
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
            next_state = np.random.choice(available) if available else current_state['state']
        if isinstance(next_state, str) and next_state.startswith('terminal'):
            logging.info(f"Reached terminal state: {next_state}")
            available = [s for s in mdp.state_space if s != current_state['state']]
            new_state = np.random.choice(available) if available else next_state
        else:
            new_state = next_state
    else:
        logging.info(f"No transition for key ({current_state.get('state')}, {pitch}). Using fallback state update.")
        available = [s for s in mdp.state_space if s != current_state.get('state')]
        new_state = np.random.choice(available) if available else current_state.get('state')
    current_state['state'] = new_state
    try:
        cnt, _ = new_state.split('_')
        b, s = cnt.split('-')
        current_state['balls'] = int(float(b))
        current_state['strikes'] = int(float(s))
        current_state['count'] = f"{b}-{s}"
    except Exception as e:
        logging.error(f"Error updating state: {e}")
    return current_state

def recommend_pitch(game_state, supervised_model, mdp_policy, df_features):
    logging.info(f"Recommending pitch for game state: {game_state}")
    
    # Retrieve the full feature list from the saved model if available.
    if hasattr(supervised_model.named_steps['rf'], "feature_names_in_"):
        full_feature_list = list(supervised_model.named_steps['rf'].feature_names_in_)
    else:
        full_feature_list = MODEL_FEATURES

    # Retrieve the model defaults from the saved model if available.
    if hasattr(supervised_model.named_steps['rf'], "model_defaults_"):
        model_defaults = supervised_model.named_steps['rf'].model_defaults_
    else:
        model_defaults = {}
    
    # Build a feature row with all required features.
    feature_row = fill_missing_features(game_state, full_feature_list, model_defaults)
    
    # Get the pitch classes (avoid ambiguous truth value checks)
    pitch_classes = []
    if hasattr(supervised_model.named_steps['rf'], "classes_"):
        pitch_classes = supervised_model.named_steps['rf'].classes_
        # Convert to list if it's an ndarray.
        pitch_classes = list(pitch_classes)
    if not pitch_classes:
        logging.error("No classes available in the model.")
    
    try:
        probs = supervised_model.predict_proba(feature_row)
    except Exception as e:
        logging.error(f"Error predicting with supervised model: {e}")
        # Fallback: assign equal probability if prediction fails.
        if pitch_classes:
            probs = np.array([[1/len(pitch_classes)] * len(pitch_classes)])
        else:
            probs = np.array([[0]])
    
    prob_dict = dict(zip(pitch_classes, probs[0]))
    
    # Remove the "Unknown" prediction if there are other options.
    if "Unknown" in prob_dict and len(prob_dict) > 1:
        prob_dict.pop("Unknown")
    
    pitcher_id = game_state.get('pitcher_id')
    pitcher_arsenal = get_pitcher_arsenal(df_features, pitcher_id)
    logging.info(f"Pitcher {pitcher_id} arsenal: {pitcher_arsenal}")
    
    # Filter predictions to only include those in the pitcher's arsenal.
    filtered_prob_dict = {p: prob for p, prob in prob_dict.items() if p in pitcher_arsenal}
    if not filtered_prob_dict:
        logging.warning("No predictions within pitcher arsenal. Falling back to full predictions.")
        filtered_prob_dict = prob_dict.copy()
        if "Unknown" in filtered_prob_dict:
            filtered_prob_dict.pop("Unknown")
    
    # Optionally adjust probabilities with the MDP recommendation.
    state_key = game_state.get('state')
    mdp_recommendation = mdp_policy.get(state_key)
    if mdp_recommendation and mdp_recommendation in filtered_prob_dict:
        filtered_prob_dict[mdp_recommendation] += 0.1

    total_prob = sum(filtered_prob_dict.values())
    if total_prob > 0:
        for k in filtered_prob_dict:
            filtered_prob_dict[k] /= total_prob
    else:
        keys = list(filtered_prob_dict.keys())
        filtered_prob_dict = {k: 1/len(keys) for k in keys}
    
    pitches = list(filtered_prob_dict.keys())
    pitch_probs = np.array([filtered_prob_dict[p] for p in pitches])
    
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
    Main function: loads the full dataset, computes features, trains the supervised model,
    solves the MDP, and simulates pitch recommendations.
    """
    file_path = "Derived_Data/feature/feature_20250301_105232.parquet"
    df_raw = load_data(file_path)
    df_features = compute_features(df_raw)
    
    # Drop datetime columns (like 'date' or 'time') from features as they cause issues with StandardScaler.
    drop_cols = [col for col in df_features.columns if pd.api.types.is_datetime64_any_dtype(df_features[col])]
    if drop_cols:
        logging.info(f"Dropping datetime columns: {drop_cols}")
        df_features = df_features.drop(columns=drop_cols)
    
    if 'target' in df_features.columns:
        labels = df_features['target']
        # Use all columns except the target as features
        features = df_features.drop(columns=['target']).copy()
        # Optionally, store the dynamic feature list
        MODEL_FEATURES = list(features.columns)
    else:
        logging.error("Target column not found in features")
        return

    logging.info("Starting supervised model training process...")
    supervised_model = train_supervised_model(features, labels)
    logging.info("Starting testing phase to check for overfitting...")
    testing_phase(features, labels, supervised_model)
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
    # main()
    # Or, to load a saved model and run a prediction example:
    df_raw = load_data("Derived_Data/feature/feature_20250301_105232.parquet")
    df_features = compute_features(df_raw)

    # Drop datetime columns if any remain
    drop_cols = [col for col in df_features.columns if pd.api.types.is_datetime64_any_dtype(df_features[col])]
    if drop_cols:
        logging.info(f"Dropping datetime columns: {drop_cols}")
        df_features = df_features.drop(columns=drop_cols)

    # Load the saved supervised model
    model_path = "models/recommend/model.pkl"
    saved_model = load(model_path)
    logging.info("Saved model loaded.")

    # Recreate the MDP from the computed features
    mdp = PitchSequencingMDP(df_features)
    mdp.estimate_transition_probabilities()
    mdp_policy = mdp.solve_mdp()

    # Prepare a sample game state for a given count, outs, inning, batter_id, and pitcher_id.
    game_state = prepare_game_state(
        count="1-1",
        outs=1,
        inning=6,
        batter_id=1000032366,
        pitcher_id=1000066910,
        hist_df=df_features
    )

    # Get a single recommended pitch using the loaded model.
    recommended_pitch = recommend_pitch(game_state, saved_model, mdp_policy, df_features)
    print("Recommended pitch:", recommended_pitch)

    # Simulate the next 10 pitches starting from the game state.
    pitch_sequence = simulate_next_pitches(game_state, saved_model, mdp_policy, mdp, df_features, n=10)
    print("Simulated pitch sequence:", pitch_sequence)
