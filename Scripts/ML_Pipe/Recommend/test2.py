import os
import time
import pandas as pd
import numpy as np
import json
import logging
import gc
import zipfile
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
# AutoGluon
from autogluon.tabular import TabularDataset, TabularPredictor

import re
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass as dc_dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from functools import lru_cache, wraps
from pydantic import BaseModel, field_validator, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

# -----------------------------------------------------------------------------
# Logging Configuration and Helpers
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(funcName)s: %(message)s'
)

@contextmanager
def log_exceptions(context: str = ""):
    """Context manager that logs exceptions with additional context."""
    try:
        yield
    except Exception as e:
        logging.error(f"Exception in {context}: {e}", exc_info=True)
        raise

def log_function(func: Callable) -> Callable:
    """Decorator that logs function entry, exit (with elapsed time), parameters, and exceptions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logging.debug(f"Exiting {func.__name__} with result={result} (elapsed {elapsed:.4f} sec)")
            return result
        except Exception as e:
            logging.error(f"Exception in {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper

def retry(attempts: int = 3, delay: float = 0.1, backoff: float = 2.0, circuit_breaker: int = 5):
    """
    Retry decorator with exponential backoff and a circuit breaker.
    After `circuit_breaker` consecutive failures, stops retrying.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            _attempts = attempts
            _delay = delay
            failures = 0
            while _attempts > 0:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    failures += 1
                    logging.warning(f"Retrying {func.__name__} (attempts left: {_attempts-1}, failures: {failures}) due to error: {e}")
                    if failures >= circuit_breaker:
                        logging.error(f"Circuit breaker triggered for {func.__name__} after {failures} failures")
                        break
                    time.sleep(_delay)
                    _attempts -= 1
                    _delay *= backoff
            logging.error(f"All retries failed for {func.__name__}")
            raise Exception(f"Function {func.__name__} failed after {attempts} attempts with {failures} failures")
        return wrapper
    return decorator

def timeout(seconds: float):
    """Timeout decorator using ThreadPoolExecutor. Best suited for I/O–bound functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except FuturesTimeoutError:
                    logging.error(f"Timeout reached in function {func.__name__} after {seconds} seconds")
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
        return wrapper
    return decorator

def safe_normalize(probabilities: np.ndarray) -> np.ndarray:
    """Safely normalize an array of probabilities; if invalid, return a uniform distribution."""
    total = probabilities.sum()
    if total <= 0 or not np.isfinite(total):
        logging.warning("Probabilities invalid (sum <= 0 or non-finite). Returning uniform distribution.")
        return np.ones_like(probabilities) / len(probabilities)
    return probabilities / total

def min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array."""
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
        return np.ones_like(arr)
    return (arr - min_val) / (max_val - min_val)

def safe_choice(options: List[Any], probabilities: np.ndarray, fallback: Any) -> Any:
    """Safely choose an element from options using normalized probabilities."""
    try:
        norm_probs = safe_normalize(probabilities)
        return np.random.choice(options, p=norm_probs)
    except Exception as e:
        logging.error("Error in safe_choice; returning fallback.", exc_info=True)
        return fallback

def softmax_probabilities(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax probabilities with temperature scaling."""
    scaled = np.power(probs, 1/temperature)
    total = np.sum(scaled)
    if total == 0:
        return np.ones_like(probs) / len(probs)
    return scaled / total
    
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
    # game_state is now a GameState instance; convert it to dict
    full_data.update(game_state.dict())
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
        time_limit=3600*2,
        presets='best_quality',
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

# -----------------------------------------------------------------------------
# Configuration Dataclass with Advanced Data Management Parameters
# -----------------------------------------------------------------------------

@dc_dataclass(frozen=True)
class MDPConfig:
    gamma: float = 0.9
    theta: float = 1e-4
    max_iterations: int = 1000
    fallback_action: Optional[str] = None
    laplace_smoothing: float = 1.0  # Smoothing constant for transition probabilities
    polynomial_degree: int = 1      # If > 1, Q-values are transformed (e.g. squared)

# -----------------------------------------------------------------------------
# Advanced GameState Using Pydantic for Validation
# -----------------------------------------------------------------------------
class GameState(BaseModel):
    count: str
    outs: int
    inning: int
    batter_id: int
    pitcher_id: int
    last_pitch: Optional[str] = None
    relspeed: float = 0.0
    spinrate: float = 0.0
    state: str = None  # Will be set in parse_count
    balls: int = None
    strikes: int = None

    @field_validator('count')
    def validate_count_format(cls, v: str) -> str:
        if not re.match(r'^\d+-\d+$', v):
            raise ValueError("Count must be in format 'balls-strikes'")
        return v

    @model_validator(mode='after')
    def parse_count(self) -> "GameState":
        count_str = self.count if self.count is not None else "0-0"
        try:
            balls_str, strikes_str = count_str.split('-')
            self.balls = int(balls_str)
            self.strikes = int(strikes_str)
        except Exception as e:
            logging.error(f"Error parsing count '{count_str}': {e}")
            self.balls, self.strikes = 0, 0
            self.count = "0-0"
        last_pitch = self.last_pitch
        self.state = f"{self.count}_{last_pitch if last_pitch is not None else 'None'}"
        return self

    def update_from_state_string(self, state_str: str) -> None:
        match = re.match(r'(\d+)-(\d+)', state_str)
        if match:
            self.__dict__['balls'], self.__dict__['strikes'] = map(int, match.groups())
            self.__dict__['count'] = f"{self.__dict__['balls']}-{self.__dict__['strikes']}"
            logging.debug(f"GameState updated: count set to {self.__dict__['count']} from '{state_str}'")
        else:
            logging.warning(f"Unable to extract count from state string '{state_str}'.")

# -----------------------------------------------------------------------------
# Abstract Base Class for MDP Solvers and Implementations
# -----------------------------------------------------------------------------

class MDPSolver(ABC):
    def __init__(self, mdp: "PitchSequencingMDP", config: MDPConfig):
        self.mdp = mdp
        self.config = config

    @abstractmethod
    def solve(self) -> Dict[Any, Optional[Any]]:
        pass

class ValueIterationSolver(MDPSolver):
    @log_function
    @retry(attempts=3, delay=0.05, circuit_breaker=5)
    @timeout(5)
    def solve(self) -> Dict[Any, Optional[Any]]:
        V = {state: 0.0 for state in self.mdp.state_space}
        # Perform value iteration with error checking
        for iteration in range(self.config.max_iterations):
            delta = 0.0
            for state in self.mdp.state_space:
                q_values = {}
                for action in self.mdp.action_space:
                    key = (state, action)
                    if key not in self.mdp.transition_model:
                        continue
                    try:
                        # Sum over transitions with Laplace smoothing applied
                        transitions = self.mdp.transition_model[key]
                        q_value = 0.0
                        for ns, prob in transitions.items():
                            q_value += prob * (self.mdp._get_reward(ns) + self.config.gamma * V.get(ns, 0.0))
                        # Apply optional polynomial transformation
                        if self.config.polynomial_degree > 1:
                            q_value = q_value ** self.config.polynomial_degree
                        q_values[action] = q_value
                    except Exception as e:
                        logging.error(f"Error computing Q-value for ({state}, {action}): {e}", exc_info=True)
                if q_values:
                    best_q = max(q_values.values())
                    delta = max(delta, abs(best_q - V[state]))
                    V[state] = best_q
                else:
                    V[state] = 0.0
            logging.debug(f"Value iteration iteration {iteration}, delta={delta}")
            if delta < self.config.theta:
                logging.info(f"Value iteration converged after {iteration} iterations.")
                break

        policy = {}
        for state in self.mdp.state_space:
            best_action = None
            best_q = -np.inf
            for action in self.mdp.action_space:
                key = (state, action)
                if key not in self.mdp.transition_model:
                    continue
                try:
                    q_val = sum(
                        prob * (self.mdp._get_reward(ns) + self.config.gamma * V.get(ns, 0.0))
                        for ns, prob in self.mdp.transition_model[key].items()
                    )
                    if self.config.polynomial_degree > 1:
                        q_val = q_val ** self.config.polynomial_degree
                    if q_val > best_q:
                        best_q = q_val
                        best_action = action
                except Exception as e:
                    logging.error(f"Error computing policy Q-value for state {state}, action {action}: {e}", exc_info=True)
            if best_action is None and self.mdp.action_space:
                best_action = self.config.fallback_action or self.mdp.action_space[0]
            policy[state] = best_action
            logging.debug(f"Derived policy for state {state}: {policy[state]}")
        self.mdp.policy = policy
        return policy

class PolicyIterationSolver(MDPSolver):
    @log_function
    @retry(attempts=3, delay=0.05, circuit_breaker=5)
    @timeout(5)
    def solve(self) -> Dict[Any, Optional[Any]]:
        policy = {state: (self.mdp.action_space[0] if self.mdp.action_space else None)
                  for state in self.mdp.state_space}
        V = {state: 0.0 for state in self.mdp.state_space}

        def policy_evaluation(policy: Dict[Any, Any], V: Dict[Any, float]) -> Dict[Any, float]:
            while True:
                delta = 0.0
                for state in self.mdp.state_space:
                    action = policy.get(state)
                    if action is None:
                        continue
                    key = (state, action)
                    if key not in self.mdp.transition_model:
                        continue
                    try:
                        new_value = sum(
                            prob * (self.mdp._get_reward(ns) + self.config.gamma * V.get(ns, 0.0))
                            for ns, prob in self.mdp.transition_model[key].items()
                        )
                    except Exception as e:
                        logging.error(f"Error in policy evaluation for state {state}: {e}", exc_info=True)
                        new_value = 0.0
                    delta = max(delta, abs(new_value - V[state]))
                    V[state] = new_value
                if delta < self.config.theta:
                    break
            return V

        for iteration in range(self.config.max_iterations):
            V = policy_evaluation(policy, V)
            policy_stable = True
            for state in self.mdp.state_space:
                old_action = policy.get(state)
                action_values = {}
                for action in self.mdp.action_space:
                    key = (state, action)
                    if key not in self.mdp.transition_model:
                        continue
                    try:
                        value = sum(
                            prob * (self.mdp._get_reward(ns) + self.config.gamma * V.get(ns, 0.0))
                            for ns, prob in self.mdp.transition_model[key].items()
                        )
                        if self.config.polynomial_degree > 1:
                            value = value ** self.config.polynomial_degree
                        action_values[action] = value
                    except Exception as e:
                        logging.error(f"Error computing action value for state {state}, action {action}: {e}", exc_info=True)
                if action_values:
                    best_action = max(action_values, key=action_values.get)
                    if best_action != old_action:
                        policy[state] = best_action
                        policy_stable = False
            logging.debug(f"Policy iteration {iteration}, policy stable: {policy_stable}")
            if policy_stable:
                logging.info(f"Policy iteration converged after {iteration} iterations.")
                break

        self.mdp.policy = policy
        return policy

# -----------------------------------------------------------------------------
# Main MDP Class with Dependency Injection for Solver Strategy and Smoothing
# -----------------------------------------------------------------------------

class PitchSequencingMDP:
    """
    Advanced MDP for pitch sequencing with robust error handling,
    defensive programming, dependency injection for solver strategy,
    timeout protection, and advanced data management (smoothing, normalization, polynomial transformation).
    """
    def __init__(self, df):
        self.state_space: List[str] = []
        self.action_space: List[str] = []
        try:
            if 'state' in df.columns:
                self.state_space = (df['state'].cat.categories.tolist() 
                                    if hasattr(df['state'], 'cat')
                                    else df['state'].unique().tolist())
        except Exception as e:
            logging.error(f"Error initializing state space: {e}", exc_info=True)
        try:
            if 'cleanpitchtype' in df.columns:
                self.action_space = (df['cleanpitchtype'].cat.categories.tolist() 
                                     if hasattr(df['cleanpitchtype'], 'cat')
                                     else df['cleanpitchtype'].unique().tolist())
        except Exception as e:
            logging.error(f"Error initializing action space: {e}", exc_info=True)

        self.transition_model: Dict[Tuple[Any, Any], Dict[Any, float]] = {}
        self.policy: Dict[Any, Optional[Any]] = {}
        self._df = df.copy()
        logging.debug(f"MDP initialized with {len(self.state_space)} states and {len(self.action_space)} actions.")

    @lru_cache(maxsize=None)
    @log_function
    def _get_reward(self, next_state: str) -> float:
        """Cached reward function."""
        if isinstance(next_state, str) and next_state.startswith('terminal'):
            if 'out' in next_state:
                return 1.0
            elif 'hit' in next_state:
                return -1.0
        return 0.0

    @log_function
    def estimate_transition_probabilities(self) -> None:
        """
        Estimate transition probabilities from the dataframe using Laplace smoothing.
        For each (state, action) group, adds a smoothing constant to counts.
        """
        required_cols = ['state', 'cleanpitchtype', 'next_state']
        if not all(col in self._df.columns for col in required_cols):
            msg = f"Missing required columns: {', '.join(required_cols)}"
            logging.error(msg)
            raise Exception(msg)
        try:
            trans_counts = self._df.groupby(['state', 'cleanpitchtype', 'next_state']).size().reset_index(name='count')
        except Exception as e:
            logging.error(f"Error grouping transitions: {e}", exc_info=True)
            raise Exception("Transition grouping failed") from e

        # Use Laplace smoothing: (count + alpha) / (total + alpha * N)
        for (state, action), group in trans_counts.groupby(['state', 'cleanpitchtype']):
            try:
                alpha = 1.0  # Default smoothing constant
                # Use the smoothing constant from configuration if needed
                num_possible = len(group['next_state'].unique())
                total = group['count'].sum()
                smoothed_total = total + alpha * num_possible
                transitions = {row['next_state']: (row['count'] + alpha) / smoothed_total 
                               for _, row in group.iterrows()}
                norm_sum = sum(transitions.values())
                if not np.isclose(norm_sum, 1.0, atol=1e-5):
                    logging.debug(f"Normalizing probabilities for ({state}, {action}). Sum: {norm_sum}")
                    transitions = {ns: p / norm_sum for ns, p in transitions.items()}
                self.transition_model[(state, action)] = transitions
                logging.debug(f"Set transition for ({state}, {action}): {transitions}")
            except Exception as e:
                logging.error(f"Error processing transition for ({state}, {action}): {e}", exc_info=True)

    @log_function
    def solve_mdp(self, solver: str = 'value_iteration', config: Optional[MDPConfig] = None) -> Dict[Any, Optional[Any]]:
        """Solve the MDP using the specified solver strategy."""
        if config is None:
            config = MDPConfig()
        if not self.state_space:
            logging.error("Empty state space; cannot solve MDP.")
            return {}
        if not self.transition_model:
            logging.warning("Empty transition model; returning fallback policy.")
            for state in self.state_space:
                self.policy[state] = config.fallback_action or (self.action_space[0] if self.action_space else None)
            return self.policy

        if solver == 'value_iteration':
            solver_instance = ValueIterationSolver(self, config)
        elif solver == 'policy_iteration':
            solver_instance = PolicyIterationSolver(self, config)
        else:
            msg = f"Unknown solver: {solver}"
            logging.error(msg)
            raise Exception(msg)

        return solver_instance.solve()

# -----------------------------------------------------------------------------
# Robust and Smooth Game State Update Function with Advanced Data Techniques
# -----------------------------------------------------------------------------
def prepare_game_state(count: str, outs: int, inning: int, batter_id: int, pitcher_id: int,
                       relspeed: Optional[float] = None, spinrate: Optional[float] = None, 
                       last_pitch: Optional[str] = None, hist_df: Optional[pd.DataFrame] = None) -> GameState:
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
    return GameState(count=count, outs=outs, inning=inning, batter_id=batter_id, pitcher_id=pitcher_id,
                     last_pitch=last_pitch, relspeed=relspeed, spinrate=spinrate)

@log_function
def update_game_state(current_state: GameState, pitch: str, mdp: PitchSequencingMDP) -> GameState:
    """
    Update the GameState based on the given pitch using the MDP transition model.
    Uses multiple fallback strategies and safe selection methods.
    """
    current_state.last_pitch = pitch
    transition_key = (current_state.state, pitch)
    chosen_state: Optional[str] = None

    try:
        if transition_key in mdp.transition_model:
            transitions = mdp.transition_model[transition_key]
            next_states = list(transitions.keys())
            probabilities = np.array(list(transitions.values()))
            if probabilities.sum() <= 0:
                logging.warning(f"Zero total probability for key {transition_key}. Using uniform probabilities.")
                probabilities = np.ones(len(next_states))
            chosen_state = safe_choice(next_states, probabilities, fallback=current_state.state)
            logging.debug(f"Chosen state for {transition_key}: {chosen_state}")
        else:
            logging.info(f"No transition found for key {transition_key}. Using fallback state update.")
            fallback_states = [s for s in mdp.state_space if s != current_state.state]
            if fallback_states:
                chosen_state = safe_choice(fallback_states, np.ones(len(fallback_states)), fallback=current_state.state)
            else:
                chosen_state = current_state.state

        if isinstance(chosen_state, str) and chosen_state.startswith('terminal'):
            logging.info(f"Terminal state '{chosen_state}' encountered. Selecting fallback state.")
            fallback_states = [s for s in mdp.state_space if s != current_state.state]
            if fallback_states:
                chosen_state = safe_choice(fallback_states, np.ones(len(fallback_states)), fallback=chosen_state)
    except Exception as e:
        logging.error(f"Error during state update: {e}", exc_info=True)
        chosen_state = current_state.state

    current_state.state = chosen_state
    try:
        current_state.update_from_state_string(chosen_state)
    except Exception as e:
        logging.error(f"Error updating GameState from '{chosen_state}': {e}", exc_info=True)
    return current_state

def recommend_pitch(game_state: GameState, supervised_model, mdp_policy, df_features):
    logging.info(f"Recommending pitch for game state: {game_state}")
    
    if hasattr(supervised_model.named_steps['rf'], "feature_names_in_"):
        full_feature_list = list(supervised_model.named_steps['rf'].feature_names_in_)
    else:
        full_feature_list = MODEL_FEATURES

    if hasattr(supervised_model.named_steps['rf'], "model_defaults_"):
        model_defaults = supervised_model.named_steps['rf'].model_defaults_
    else:
        model_defaults = {}
    
    feature_row = fill_missing_features(game_state, full_feature_list, model_defaults)
    
    pitch_classes = []
    if hasattr(supervised_model.named_steps['rf'], "classes_"):
        pitch_classes = list(supervised_model.named_steps['rf'].classes_)
    if not pitch_classes:
        logging.error("No classes available in the model.")
    
    try:
        probs = supervised_model.predict_proba(feature_row)
    except Exception as e:
        logging.error(f"Error predicting with supervised model: {e}")
        if pitch_classes:
            probs = np.array([[1/len(pitch_classes)] * len(pitch_classes)])
        else:
            probs = np.array([[0]])
    
    prob_dict = dict(zip(pitch_classes, probs[0]))
    if "Unknown" in prob_dict and len(prob_dict) > 1:
        prob_dict.pop("Unknown")
    
    pitcher_id = game_state.pitcher_id
    pitcher_arsenal = get_pitcher_arsenal(df_features, pitcher_id)
    logging.info(f"Pitcher {pitcher_id} arsenal: {pitcher_arsenal}")
    
    filtered_prob_dict = {p: prob for p, prob in prob_dict.items() if p in pitcher_arsenal}
    if not filtered_prob_dict:
        logging.warning("No predictions within pitcher arsenal. Falling back to full predictions.")
        filtered_prob_dict = prob_dict.copy()
        if "Unknown" in filtered_prob_dict:
            filtered_prob_dict.pop("Unknown")
    
    state_key = game_state.state
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

from copy import deepcopy
def simulate_next_pitches(initial_state: GameState, supervised_model, mdp_policy, mdp, df_features, n=3):
    sequence = []
    current_state = deepcopy(initial_state)
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
    file_path = "Derived_Data/feature/feature_20250301_105232.parquet"
    df_raw = load_data(file_path)
    df_features = compute_features(df_raw)
    
    drop_cols = [col for col in df_features.columns if pd.api.types.is_datetime64_any_dtype(df_features[col])]
    if drop_cols:
        logging.info(f"Dropping datetime columns: {drop_cols}")
        df_features = df_features.drop(columns=drop_cols)
    
    if 'target' in df_features.columns:
        labels = df_features['target']
        features = df_features.drop(columns=['target']).copy()
        global MODEL_FEATURES
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
    df_raw = load_data("Derived_Data/feature/feature_20250301_105232.parquet")
    df_features = compute_features(df_raw)
    drop_cols = [col for col in df_features.columns if pd.api.types.is_datetime64_any_dtype(df_features[col])]
    if drop_cols:
        logging.info(f"Dropping datetime columns: {drop_cols}")
        df_features = df_features.drop(columns=drop_cols)

    model_path = "models/recommend/model.pkl"
    saved_model = load(model_path)
    logging.info("Saved model loaded.")

    mdp = PitchSequencingMDP(df_features)
    mdp.estimate_transition_probabilities()
    mdp_policy = mdp.solve_mdp()

    game_state = prepare_game_state(
        count="1-1",
        outs=1,
        inning=6,
        batter_id=1000032366,
        pitcher_id=1000066910,
        hist_df=df_features
    )

    recommended_pitch = recommend_pitch(game_state, saved_model, mdp_policy, df_features)
    print("Recommended pitch:", recommended_pitch)

    pitch_sequence = simulate_next_pitches(game_state, saved_model, mdp_policy, mdp, df_features, n=10)
    print("Simulated pitch sequence:", pitch_sequence)
