import os
import zipfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
# from imblearn.under_sampling import TomekLinks

import warnings
warnings.simplefilter("ignore", category=UserWarning)  
warnings.simplefilter("ignore", category=RuntimeWarning)  

###############################################################################
# UTILITY FUNCTIONS
###############################################################################
def compute_class_weights(y_train, device='cpu'):
    """
    Given an array of class labels, compute simple
    inverse-frequency weights for CrossEntropyLoss.
    """
    counts = np.bincount(y_train)        
    total = sum(counts)
    num_classes = len(counts)
    weights = [total / (num_classes * c) if c > 0 else 0.0 for c in counts]

    # Convert to torch tensor
    return torch.tensor(weights, dtype=torch.float32).to(device)

def advanced_data_cleaning(df, target_col, categorical_cols):
    df_clean = df.copy(deep=True)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]
    imputer = SimpleImputer(strategy="median")
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("Missing")
    return df_clean.dropna(subset=[target_col])

def encode_categorical_columns(df, categorical_cols, encoders=None):
    df_encoded = df.copy(deep=True)
    if encoders is None:
        encoders = {}
        fit_encoders = True
    else:
        fit_encoders = False
    for col in categorical_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(str)
            if fit_encoders:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                encoders[col] = le
            else:
                df_encoded[col] = encoders[col].transform(df_encoded[col])
    return df_encoded, encoders

def mutual_info_feature_selection(X, y, feature_cols, top_k=None):
    mi_scores = mutual_info_classif(X, y, random_state=42)
    features = sorted(zip(feature_cols, mi_scores), key=lambda x: x[1], reverse=True)
    if top_k is None:
        return [f[0] for f in features]
    else:
        return [f[0] for f in features[:top_k]]

def augment_data_with_noise(X, noise_level=0.01):
    std_vec = np.std(X, axis=0)
    noise = np.random.randn(*X.shape) * (std_vec * noise_level)
    return X + noise

def save_zipped_state(state, save_path):
    temp_path = save_path.replace(".zip", ".pth")
    torch.save(state, temp_path)
    with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(temp_path, arcname=os.path.basename(temp_path))
    os.remove(temp_path)

def load_zipped_state(load_path, map_location=None):
    temp_dir = "temp_model_dir"
    os.makedirs(temp_dir, exist_ok=True)
    with zipfile.ZipFile(load_path, "r") as zf:
        zf.extractall(temp_dir)
    file_list = os.listdir(temp_dir)
    state_path = os.path.join(temp_dir, file_list[0])
    state = torch.load(state_path, map_location=map_location)
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)
    return state

###############################################################################
# CUSTOM DATASET
###############################################################################

class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

###############################################################################
# TRANSFORMER MODEL
###############################################################################

class TabTransformer(nn.Module):
    def __init__(self, num_numeric, num_categories_list, model_dim=32,
                 num_heads=4, num_layers=2, num_classes=2):
        super().__init__()
        self.num_numeric = num_numeric
        self.num_cats = len(num_categories_list)
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_size, model_dim) for cat_size in num_categories_list
        ])
        self.numeric_linear = nn.Linear(num_numeric, model_dim) if num_numeric > 0 else None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=model_dim*4, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, num_classes)
    def forward(self, x_numeric, x_categorical):
        cat_embs = []
        for i, emb in enumerate(self.cat_embeddings):
            # Clamp indices so they are within the valid range [0, emb.weight.size(0)-1]
            idx = x_categorical[:, i].long()
            idx = torch.clamp(idx, 0, emb.weight.size(0) - 1)
            cat_embs.append(emb(idx))
        cat_embs = torch.stack(cat_embs, dim=1) if len(cat_embs) > 0 else None

        if self.numeric_linear is not None:
            num_emb = self.numeric_linear(x_numeric).unsqueeze(1)
        else:
            num_emb = None

        if num_emb is not None and cat_embs is not None:
            x_seq = torch.cat([num_emb, cat_embs], dim=1)
        elif num_emb is not None:
            x_seq = num_emb
        else:
            x_seq = cat_embs

        x_enc = self.transformer_encoder(x_seq)
        x_pooled = x_enc.mean(dim=1)
        logits = self.fc_out(x_pooled)
        return logits

###############################################################################
# TRAINING WRAPPER
###############################################################################

class TabTransformerTrainer:
    def __init__(self, numeric_dims, cat_dims, num_classes=2,
                 model_dim=64,       
                 num_heads=4,
                 num_layers=3,   
                 lr=1e-4,        
                 batch_size=256,
                 epochs=25,         
                 device='cpu',
                 class_weights=None  
                 ):
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = TabTransformer(
            num_numeric=numeric_dims,
            num_categories_list=cat_dims,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # If class_weights is not None, use weighted CrossEntropy
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.numeric_dims = numeric_dims
        self.cat_dims = cat_dims

    def _split_features(self, X):
        x_num = X[:, :self.numeric_dims]
        x_cat = X[:, self.numeric_dims:].long()
        return x_num, x_cat

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        ds_train = TabularDataset(X_train, y_train)
        dl_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        dl_valid = None
        if X_valid is not None and y_valid is not None:
            ds_valid = TabularDataset(X_valid, y_valid)
            dl_valid = DataLoader(ds_valid, batch_size=self.batch_size, shuffle=False)
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for Xb, yb in dl_train:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                x_num, x_cat = self._split_features(Xb)
                logits = self.model(x_num, x_cat)
                loss = self.criterion(logits, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dl_train)
            if dl_valid is not None:
                acc, f1, prec = self.evaluate(dl_valid)
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} - ValAcc: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

    def evaluate(self, dataloader):
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for Xb, yb in dataloader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                x_num, x_cat = self._split_features(Xb)
                logits = self.model(x_num, x_cat)
                pred = torch.argmax(logits, dim=1)
                preds.append(pred.cpu().numpy())
                labels.append(yb.cpu().numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        prec = precision_score(labels, preds, average="macro", zero_division=1)
        return acc, f1, prec

    def predict(self, X):
        self.model.eval()
        ds = TabularDataset(X, None)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for Xb in dl:
                Xb = Xb.to(self.device)
                x_num, x_cat = self._split_features(Xb)
                logits = self.model(x_num, x_cat)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
        return np.concatenate(all_preds)

    def predict_proba(self, X):
        self.model.eval()
        ds = TabularDataset(X, None)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for Xb in dl:
                Xb = Xb.to(self.device)
                x_num, x_cat = self._split_features(Xb)
                logits = self.model(x_num, x_cat)
                probs = nn.Softmax(dim=1)(logits)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    def predict_cumulative(self, X_sequence, alpha=0.5):
        """
        For a given candidate pitch type, X_sequence (n_pitches x n_features)
        is updated recursively: the "PrevStrike" feature (assumed to be the last column)
        is updated with:
            cum_strike = alpha * current_strike + (1 - alpha) * previous_cum_strike
        Returns a list of dictionaries for each pitch event.
        """
        n_pitches = X_sequence.shape[0]
        cumulative_results = []
        cum_strike = 0.0
        for i in range(n_pitches):
            # Update the "PrevStrike" feature (assumed to be the last column)
            X_sequence[i, -1] = cum_strike
            current_probs = self.predict_proba(X_sequence[i:i+1])[0]
            cum_strike = alpha * current_probs[0] + (1 - alpha) * cum_strike
            rec = {"PitchNumber": i + 1, "CumulativePrediction": int(np.argmax(current_probs))}
            # Assume first two classes are Strike and Ball, additional as Class_2, Class_3, ...
            classes = ["Strike", "Ball"] + [f"Class_{j}" for j in range(2, self.model.num_classes)]
            for cls, p_val in zip(classes, current_probs):
                rec[cls] = round(p_val * 100, 2)
            cumulative_results.append(rec)
        return cumulative_results

    def save(self, save_dir="Derived_data/ad_model_params/"):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "transformer_model.zip")
        state = self.model.state_dict()
        save_zipped_state(state, save_path)
        print(f"Model saved to {save_path}")

    def load(self, load_dir="Derived_data/ad_model_params/"):
        load_path = os.path.join(load_dir, "transformer_model.zip")
        state = load_zipped_state(load_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")

###############################################################################
# CROSS-VALIDATION
###############################################################################

def cross_validate_transformer(df, feature_cols, target_col,
                               numeric_cols, cat_cols, cat_cardinalities,
                               num_classes, n_splits=5, top_k_features=None,
                               device='cpu'):
    X = df[feature_cols].values
    y = df[target_col].values

    # If top-k selection, filter the columns
    if top_k_features is not None:
        X_full = df[feature_cols].values
        y_full = df[target_col].values
        sel = mutual_info_feature_selection(X_full, y_full, feature_cols, top_k=top_k_features)
        df_sel = df[sel + [target_col]].copy()
        numeric_cols = [c for c in numeric_cols if c in sel]
        cat_cols = [c for c in cat_cols if c in sel]
        feature_cols = numeric_cols + cat_cols
    else:
        df_sel = df.copy(deep=True)

    # Recompute local cat cardinalities on df_sel
    cat_cardinalities_local = [cat_cardinalities[i] for i, c in enumerate(cat_cols)]

    X = df_sel[feature_cols].values
    y = df_sel[target_col].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold_idx+1}/{n_splits} ===")
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        # Tomek Undersampling
        # tl = TomekLinks()
        X_train_tl, y_train_tl = X_train, y_train

        # Optionally add noise
        X_train_noisy = augment_data_with_noise(X_train_tl, noise_level=0.01)

        # Scale numeric only
        num_idx = [feature_cols.index(c) for c in numeric_cols]
        scaler = StandardScaler()
        X_train_num = X_train_noisy[:, num_idx]
        X_valid_num = X_valid[:, num_idx]
        X_train_num_sc = scaler.fit_transform(X_train_num)
        X_valid_num_sc = scaler.transform(X_valid_num)
        X_train_noisy[:, num_idx] = X_train_num_sc
        X_valid[:, num_idx] = X_valid_num_sc

        # Prepare cat dims (embedding sizes)
        cat_dims_local = [cat_cardinalities_local[cat_cols.index(c)] for c in cat_cols]

        # Compute class weights *after* Tomek
        class_weights = compute_class_weights( y_train_tl, device=device)

        # Create bigger model with weighting
        trainer = TabTransformerTrainer(
            numeric_dims=len(numeric_cols),       # must match how many numeric cols remain
            cat_dims=cat_dims_local,
            num_classes=num_classes,
            model_dim=64,
            num_heads=4,
            num_layers=5,
            lr=1e-4,           # smaller LR
            batch_size=512,
            epochs=25,         # more epochs
            device=device,
            class_weights=class_weights
        )
        # for idx, c in enumerate(cat_cols):
        #     # Get the training values for column c
        #     col_index = feature_cols.index(c)
        #     max_train = X_train_sm[:, col_index].max()
        #     emb_size = cat_cardinalities_local[idx]
        #     print(f"Fold {fold_idx+1} - Column '{c}': max index in training = {max_train}, embedding size = {emb_size}")
        #     if max_train >= emb_size:
        #         print(f"WARNING: For column '{c}', max index {max_train} >= embedding size {emb_size}")
        # Fit and evaluate
        trainer.fit(X_train_noisy, y_train_tl, X_valid, y_valid)
        ds_valid = TabularDataset(X_valid, y_valid)
        dl_valid = DataLoader(ds_valid, batch_size=512, shuffle=False)
        acc, f1, prec = trainer.evaluate(dl_valid)
        print(f"Fold {fold_idx+1} - ACC: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}")
        results.append((acc, f1, prec))

    # Summaries
    accs = [r[0] for r in results]
    f1s = [r[1] for r in results]
    precs = [r[2] for r in results]
    print("\n=== Cross-Validation Summary ===")
    print(f"Mean ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Mean Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")

###############################################################################
# MAIN PIPELINE (TRAINING)
###############################################################################

def main_transformer_pipeline(data_path="data.parquet", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None

    print(f"Reading data from: {data_path}")
    df = pq.read_table(source=data_path).to_pandas()

    # --- Basic Setup ---
    target_col = "CleanPitchCall"
    all_objs = df.select_dtypes(include=["object"]).columns.tolist()
    if target_col in all_objs:
        all_objs.remove(target_col)
    categorical_cols = all_objs

    df_clean = advanced_data_cleaning(df, target_col, categorical_cols)

    # Encode target if needed
    le_target = None
    if df_clean[target_col].dtype == "object":
        le_target = LabelEncoder()
        df_clean[target_col] = le_target.fit_transform(df_clean[target_col])

    # Keep IDs for lookup only (but exclude from model features)
    if "PitcherId" in df.columns:
        df_clean["PitcherId"] = df["PitcherId"]
    if "BatterId" in df.columns:
        df_clean["BatterId"] = df["BatterId"]

    # --- Encode Categorical Columns ---
    df_encoded, encoders = encode_categorical_columns(df_clean, categorical_cols, encoders=None)

    # --- Define Numeric vs. Categorical Features ---
    numeric_cols = [c for c in df_encoded.select_dtypes(include=[np.number]).columns 
                    if c not in categorical_cols and c != target_col]
    # Ensure "PrevStrike" exists and is numeric.
    if "PrevStrike" not in df_encoded.columns:
        df_encoded["PrevStrike"] = 0.0
    if "PrevStrike" not in numeric_cols:
        numeric_cols.append("PrevStrike")
        
    # Exclude ID columns from categorical features.
    cat_cols = [c for c in categorical_cols if c not in ["PitcherId", "BatterId"]]

    # Build feature columns: numeric features first, then categorical.
    feature_cols = numeric_cols + cat_cols

    # Convert categorical columns (used in the model) to int.
    for c in cat_cols:
        df_encoded[c] = df_encoded[c].astype(int)

    num_classes = df_encoded[target_col].nunique()

    # Compute global categorical cardinalities with a larger buffer (+10)
    cat_cardinalities_global = [int(df_encoded[c].max() + 10) for c in cat_cols]

    # --- CROSS-VALIDATION ---
    print("\n===== CROSS VALIDATION STAGE =====")
    cross_validate_transformer(
        df=df_encoded,
        feature_cols=feature_cols,
        target_col=target_col,
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
        cat_cardinalities=cat_cardinalities_global,
        num_classes=num_classes,
        n_splits=3,
        top_k_features=None,  # or set to 20 if desired
        device=device
    )

    # --- FINAL TRAINING ---
    print("\n===== FINAL TRAINING STAGE =====")
    X_all = df_encoded[feature_cols].values
    y_all = df_encoded[target_col].values

    # Apply mutual information feature selection (top_k = 20 here)
    sel = mutual_info_feature_selection(X_all, y_all, feature_cols, top_k=None)
    numeric_cols_final = [c for c in numeric_cols if c in sel]
    cat_cols_final = [c for c in cat_cols if c in sel]
    feature_cols_final = numeric_cols_final + cat_cols_final

    df_final = df_encoded[feature_cols_final + [target_col]].copy()
    # Recompute final categorical cardinalities with the same buffer
    cat_cardinalities_final = [int(df_final[c].max() + 10) for c in cat_cols_final]

    X = df_final[feature_cols_final].values
    y = df_final[target_col].values
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Tomek Undersampling
    # tl = TomekLinks()
    X_train_tl, y_train_tl = X_train, y_train
    X_train_noisy = augment_data_with_noise(X_train_tl, noise_level=0.01)

    # Scale only numeric features
    num_idx = [feature_cols_final.index(c) for c in numeric_cols_final]
    scaler = StandardScaler()
    X_train_num = X_train_noisy[:, num_idx]
    X_val_num = X_val[:, num_idx]
    X_train_num_sc = scaler.fit_transform(X_train_num)
    X_val_num_sc = scaler.transform(X_val_num)
    X_train_noisy[:, num_idx] = X_train_num_sc
    X_val[:, num_idx] = X_val_num_sc

    # Compute class weights from Tomek labels
    class_weights = compute_class_weights(y_train_tl, device=device)

    # IMPORTANT: pass len(numeric_cols_final) as numeric_dims.
    final_trainer = TabTransformerTrainer(
        numeric_dims=len(numeric_cols_final),
        cat_dims=cat_cardinalities_final,
        num_classes=num_classes,
        model_dim=64,
        num_heads=4,
        num_layers=5,
        lr=1e-4,
        batch_size=512,
        epochs=25,
        device=device,
        class_weights=class_weights
    )

    final_trainer.fit(X_train_noisy, y_train_tl, X_val, y_val)
    ds_val = TabularDataset(X_val, y_val)
    dl_val = DataLoader(ds_val, batch_size=512, shuffle=False)
    acc, f1, prec = final_trainer.evaluate(dl_val)
    print(f"\nFinal Validation => ACC: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}")

    final_trainer.save(save_dir="Derived_data/ad_model_params/")

    return {
        "model_trainer": final_trainer,
        "scaler": scaler,
        "numeric_cols": numeric_cols_final,
        "cat_cols": cat_cols_final,
        "feature_cols": feature_cols_final,
        "target_encoder": le_target,
        "encoders": encoders,
        "df_processed": df_final,
    }

###############################################################################
# PREDICTION FUNCTION (CUMULATIVE RECOMMENDATIONS)
###############################################################################

def prediction(pitcher, batter, model, scaler, encoders, df, feature_cols, target_col,
               pitch_type_col="CleanPitchType", n_pitches=3, alpha=0.5):
    """
    For a given pitcher and batter, simulate cumulative predictions for each candidate pitch type
    over n pitch events. For each candidate, start with a fresh baseline row (matchup data if available,
    else pitcher data). For each pitch event, update the "PrevStrike" feature recursively:
         cum_strike = alpha * current_strike + (1 - alpha) * previous_cum_strike.
    After simulation, determine the best candidate for each pitch event (i.e. the candidate with the highest Strike chance)
    and add that as a new column "BestStrikePotential" to all rows of that pitch event.
    
    Returns a DataFrame with columns:
      "PitchNumber", "CandidatePitchType", [target class columns], "BestStrikePotential".
    """
    pitcher_data = df[df["PitcherId"] == pitcher]
    if pitcher_data.empty:
        print(f"No data found for Pitcher {pitcher}.")
        return pd.DataFrame()
    candidate_types = pitcher_data[pitch_type_col].unique()
    if len(candidate_types) == 0:
        print(f"No candidate pitch types for Pitcher {pitcher}.")
        return pd.DataFrame()
    matchup = df[(df["PitcherId"] == pitcher) & (df["BatterId"] == batter)]
    if not matchup.empty:
        baseline = matchup.iloc[0].copy(deep=True)
    else:
        baseline = pitcher_data.iloc[0].copy(deep=True)
        print(f"No matchup data for Pitcher {pitcher} vs Batter {batter}; using general pitcher data.")
    if "PrevStrike" not in baseline:
        baseline["PrevStrike"] = 0.0
    target_encoder = encoders.get(target_col, None)
    if target_encoder is not None:
        target_classes = target_encoder.classes_
    else:
        target_classes = [f"Class_{i}" for i in range(model.model.num_classes)]
    records = []
    for candidate in candidate_types:
        sim_row = baseline.copy(deep=True)
        sim_row[pitch_type_col] = candidate
        cum_strike = 0.0
        candidate_records = []
        for pitch_num in range(1, n_pitches + 1):
            sim_row["PrevStrike"] = cum_strike
            sim_df = pd.DataFrame([sim_row])
            for col, encoder in encoders.items():
                if col in sim_df.columns:
                    sim_df[col] = encoder.transform(sim_df[col])
            X_sim = sim_df[feature_cols].values
            X_sim = scaler.transform(X_sim)
            probs = model.predict_proba(X_sim)[0]
            cum_strike = alpha * probs[0] + (1 - alpha) * cum_strike
            rec = {"PitchNumber": pitch_num, "CandidatePitchType": candidate}
            for cls, p_val in zip(target_classes, probs):
                rec[cls] = round(p_val * 100, 2)
            candidate_records.append(rec)
        records.extend(candidate_records)
    df_rec = pd.DataFrame(records)
    # For each pitch number, determine which candidate had the highest Strike percentage.
    best_dict = df_rec.groupby("PitchNumber").apply(lambda d: d.loc[d["Strike"].idxmax(), "CandidatePitchType"]).to_dict()
    df_rec["BestStrikePotential"] = df_rec["PitchNumber"].map(best_dict)
    return df_rec

###############################################################################
# EXAMPLE USAGE
###############################################################################

if __name__ == "__main__":
    data_path = "Derived_Data/feature/nDate_feature.parquet"  # Adapt path as needed
    pipeline_objs = main_transformer_pipeline(data_path=data_path)
    if pipeline_objs is not None:
        final_trainer = pipeline_objs["model_trainer"]
        scaler = pipeline_objs["scaler"]
        numeric_cols = pipeline_objs["numeric_cols"]
        cat_cols = pipeline_objs["cat_cols"]
        feature_cols = pipeline_objs["feature_cols"]
        df_proc = pipeline_objs["df_processed"]
        encoders = pipeline_objs["encoders"]
        target_col = "CleanPitchCall"
        
        # Sample individual predictions (for sanity check)
        sample = df_proc.iloc[:5].copy(deep=True)
        X_sample = sample[feature_cols].values
        num_idxs = [feature_cols.index(c) for c in numeric_cols]
        X_sample[:, num_idxs] = scaler.transform(X_sample[:, num_idxs])
        preds = final_trainer.predict(X_sample)
        print("\nSample Predictions:", preds)
        probs = final_trainer.predict_proba(X_sample)
        print("Sample Probabilities:\n", probs)
        
        # Run cumulative prediction for a given pitcher and batter.
        pitcher_id = 1000066910.0
        batter_id = 1000032366.0
        cum_df = prediction(
            pitcher=pitcher_id,
            batter=batter_id,
            model=final_trainer,
            scaler=scaler,
            encoders=encoders,
            df=df_proc,
            feature_cols=feature_cols,
            target_col=target_col,
            pitch_type_col="CleanPitchType",
            n_pitches=3,
            alpha=0.5
        )
        if cum_df.empty:
            print(f"No cumulative prediction data for (Pitcher={pitcher_id}, Batter={batter_id}).")
        else:
            cum_df["PitcherId"] = pitcher_id
            cum_df["BatterId"] = batter_id
            print("\nCumulative Prediction Results:")
            print(cum_df)
            out_dir = "Derived_data/ad_pred"
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"cls_prediction_report_{int(float(pitcher_id))}_{int(float(batter_id))}.csv")
            cum_df.to_csv(save_path, index=False)
            print(f"\nSaved cumulative classification results to {save_path}\n")
