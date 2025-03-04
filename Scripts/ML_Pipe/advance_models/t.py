import os
import zipfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import pickle

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  # Scheduler
from torch.utils.data import Dataset, DataLoader
from torch_optimizer import Ranger

from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

EPS = 1e-8

###############################################################################
# UTILITY FUNCTIONS (unchanged or enhanced)
###############################################################################
def compute_class_weights(y_train, device='cpu'):
    counts = np.bincount(y_train)
    total = sum(counts)
    num_classes = len(counts)
    weights = [total / (num_classes * c) if c > 0 else 0.0 for c in counts]
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

def variance_based_target_encoding(df, col, target_col, smoothing=10):
    """
    Computes a variance-based (shrinkage) target encoding for a categorical column.
    For each category, the encoding is computed as:
        weight = count / (count + smoothing)
        encoding = weight * (mean outcome for the category) + (1-weight) * (global mean)
    """
    global_mean = df[target_col].mean()
    stats = df.groupby(col)[target_col].agg(['mean', 'count'])
    stats['weight'] = stats['count'] / (stats['count'] + smoothing)
    stats['encoded'] = stats['weight'] * stats['mean'] + (1 - stats['weight']) * global_mean
    return stats['encoded'].to_dict()

def encode_categorical_columns(df, categorical_cols, target_col, encoders=None):
    """
    Applies variance-based target encoding to categorical columns.
    If encoders are provided, uses them for transforming new data.
    """
    df_encoded = df.copy(deep=True)
    if encoders is None:
        encoders = {}
        global_mean = df[target_col].mean()
        for col in categorical_cols:
            mapping = variance_based_target_encoding(df, col, target_col, smoothing=10)
            df_encoded[col] = df_encoded[col].map(mapping).fillna(global_mean)
            encoders[col] = mapping
    else:
        for col in categorical_cols:
            global_mean = df[target_col].mean()
            df_encoded[col] = df_encoded[col].map(encoders.get(col, {})).fillna(global_mean)
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

def save_pipeline_extras(extras_dict, save_dir="Derived_data/ad_model_extra/"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "pipeline_extras.pkl.zip")
    temp_pkl_path = save_path.replace(".zip", "")
    with open(temp_pkl_path, "wb") as f:
        pickle.dump(extras_dict, f)
    with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(temp_pkl_path, arcname="pipeline_extras.pkl")
    os.remove(temp_pkl_path)
    print(f"Pipeline extras saved to {save_path}")

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
# CUSTOM SEQUENCE DATASET
###############################################################################
class SequenceDataset(Dataset):
    """
    Dataset for sequential pitch data.
    Expects X with shape: (num_sequences, seq_len, feature_dim)
    """
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

###############################################################################
# LABEL-SMOOTHING CROSS-ENTROPY LOSS (unchanged)
###############################################################################
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1, weight=None, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, logits, target):
        log_probs = self.log_softmax(logits)
        nll_loss = -log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        if self.weight is not None:
            loss = loss * self.weight[target]
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

###############################################################################
# SQUEEZE-AND-EXCITATION (SE) BLOCK (unchanged)
###############################################################################
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        se = self.fc1(x)
        se = self.relu(se)
        se = self.dropout(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return x * se

###############################################################################
# GATED RESIDUAL NETWORK (GRN) BLOCK (unchanged)
###############################################################################
class GRNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.gate = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.layer_norm = nn.LayerNorm(input_dim)
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.elu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        gate = torch.sigmoid(self.gate(residual))
        out = gate * out + (1 - gate) * residual
        return self.layer_norm(out)

###############################################################################
# HYBRID CNN-LSTM-ATTENTION-GRN MODEL FOR PITCH SELECTION
###############################################################################
class HybridPitchSelectionModel(nn.Module):
    """
    This model implements:
      - A 1D CNN for extracting local/spatial features (e.g., pitch location patterns)
      - An LSTM to capture sequential dependencies in pitch sequences
      - An attention mechanism over LSTM outputs for context-aware focus
      - A Gated Residual Network (GRN) for adaptive fusion of features
      - A final MLP for multi-class classification (e.g., Ball, Strike, Hit, Walk, Undefined)
    """
    def __init__(self, input_dim, cnn_channels=32, lstm_hidden_dim=64, num_classes=5, dropout=0.4):
        super(HybridPitchSelectionModel, self).__init__()
        # CNN to extract spatial/locational patterns
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # LSTM for sequential modeling (e.g., evolving pitch sequence)
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden_dim, batch_first=True)
        # Attention: compute a weight for each time step
        self.attention_fc = nn.Linear(lstm_hidden_dim, 1)
        # GRN block to fuse and refine the context vector
        self.grn = GRNBlock(lstm_hidden_dim, lstm_hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # Final output layer
        self.fc_out = nn.Linear(lstm_hidden_dim, num_classes)
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # Apply CNN: first transpose to (batch, input_dim, seq_len)
        x_cnn = self.cnn(x.transpose(1, 2))
        x_cnn = self.relu(x_cnn)
        # Transpose back: (batch, seq_len, cnn_channels)
        x_cnn = x_cnn.transpose(1, 2)
        # Process through LSTM
        lstm_out, _ = self.lstm(x_cnn)  # lstm_out: (batch, seq_len, lstm_hidden_dim)
        # Compute attention weights and context vector over the sequence
        attn_weights = torch.softmax(self.attention_fc(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, lstm_hidden_dim)
        # Fuse with GRN and drop out before final prediction
        grn_out = self.grn(context)
        out = self.dropout(grn_out)
        logits = self.fc_out(out)
        return logits

###############################################################################
# HYBRID PITCH TRAINER FOR SEQUENTIAL DATA
###############################################################################
class HybridPitchTrainer:
    def __init__(self, input_dim, num_classes=5, cnn_channels=32, lstm_hidden_dim=64,
                 lr=1e-4, batch_size=256, epochs=25, device='cpu',
                 class_weights=None, weight_decay=1e-4, grad_clip=1.0,
                 warmup_epochs=3, early_stopping_patience=25, dropout=0.4,
                 use_mixup=True, mixup_alpha=0.2):
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.early_stopping_patience = early_stopping_patience
        self.current_epoch = 0
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        self.model = HybridPitchSelectionModel(input_dim=input_dim,
                                               cnn_channels=cnn_channels,
                                               lstm_hidden_dim=lstm_hidden_dim,
                                               num_classes=num_classes,
                                               dropout=dropout).to(device)
        self.optimizer = Ranger(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=1e-6)
        self.criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.1, weight=class_weights)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        ds_train = SequenceDataset(X_train, y_train)
        dl_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        dl_valid = None
        if X_valid is not None and y_valid is not None:
            ds_valid = SequenceDataset(X_valid, y_valid)
            dl_valid = DataLoader(ds_valid, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.current_epoch += 1
            self.model.train()
            total_loss = 0
            if self.current_epoch <= self.warmup_epochs:
                warmup_lr = (self.current_epoch / self.warmup_epochs) * self.optimizer.defaults['lr']
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            for Xb, yb in dl_train:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                if self.use_mixup:
                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    index = torch.randperm(Xb.size(0)).to(self.device)
                    Xb = lam * Xb + (1 - lam) * Xb[index]
                    y_a, y_b = yb, yb[index]
                    logits = self.model(Xb)
                    loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
                else:
                    logits = self.model(Xb)
                    loss = self.criterion(logits, yb)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dl_train)
            self.scheduler.step()

            if dl_valid is not None:
                val_loss = self._validate_loss(dl_valid)
                acc, f1, prec, rec = self.evaluate(dl_valid)
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} - ValLoss: {val_loss:.4f} - ValAcc: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print("Early stopping triggered.")
                        break
            else:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

    def _validate_loss(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for Xb, yb in dataloader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                logits = self.model(Xb)
                loss = self.criterion(logits, yb)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for Xb, yb in dataloader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                logits = self.model(Xb)
                pred = torch.argmax(logits, dim=1)
                preds.append(pred.cpu().numpy())
                labels.append(yb.cpu().numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        prec = precision_score(labels, preds, average="macro", zero_division=1)
        rec = recall_score(labels, preds, average="macro", zero_division=1)
        return acc, f1, prec, rec

    def predict(self, X):
        self.model.eval()
        ds = SequenceDataset(X, None)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for Xb in dl:
                Xb = Xb.to(self.device)
                logits = self.model(Xb)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
        return np.concatenate(all_preds)

    def predict_proba(self, X):
        self.model.eval()
        ds = SequenceDataset(X, None)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for Xb in dl:
                Xb = Xb.to(self.device)
                logits = self.model(Xb)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    def predict_cumulative(self, X_sequence, alpha=0.5):
        """
        For a simulated sequence of pitches (each row is one pitch as a feature vector),
        update the cumulative strike probability.
        Expects X_sequence as a 2D array (n_pitches, feature_dim).
        """
        n_pitches = X_sequence.shape[0]
        cumulative_results = []
        cum_strike = 0.0
        for i in range(n_pitches):
            # Update "PrevStrike" feature (assumed to be the last column)
            X_sequence[i, -1] = cum_strike
            # Reshape to (1, 1, feature_dim) for a single-pitch sequence
            current_input = X_sequence[i:i+1].reshape(1, 1, -1)
            current_probs = self.predict_proba(current_input)[0]
            cum_strike = alpha * current_probs[0] + (1 - alpha) * cum_strike
            rec = {"PitchNumber": i + 1, "CumulativePrediction": int(np.argmax(current_probs))}
            for cls, p_val in zip([f"Class_{j}" for j in range(self.model.fc_out.out_features)], current_probs):
                rec[cls] = round(p_val * 100, 2)
            cumulative_results.append(rec)
        return cumulative_results

    def save(self, save_dir="Derived_data/ad_model_params/"):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "hybrid_pitch_model.zip")
        state = self.model.state_dict()
        save_zipped_state(state, save_path)
        print(f"Model saved to {save_path}")

    def load(self, load_dir="Derived_data/ad_model_params/"):
        load_path = os.path.join(load_dir, "hybrid_pitch_model.zip")
        state = load_zipped_state(load_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")

###############################################################################
# MAIN PIPELINE FOR THE HYBRID MODEL
###############################################################################
def main_hybrid_pipeline(data_path="data.parquet", top_k_features=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None
    print(f"Reading data from: {data_path}")
    df = pq.read_table(source=data_path).to_pandas()
    target_col = "CleanPitchCall"
    
    # Use PlateAppearanceId if available; otherwise create one.
    if "PlateAppearanceId" in df.columns:
        seq_id_col = "PlateAppearanceId"
    else:
        df["PlateAppearanceId"] = np.arange(len(df))
        seq_id_col = "PlateAppearanceId"
    
    # Data cleaning and encoding
    all_objs = df.select_dtypes(include=["object"]).columns.tolist()
    if target_col in all_objs:
        all_objs.remove(target_col)
    categorical_cols = all_objs
    df_clean = advanced_data_cleaning(df, target_col, categorical_cols)
    
    # Encode target using a simple LabelEncoder (kept for target only)
    from sklearn.preprocessing import LabelEncoder
    le_target = None
    if df_clean[target_col].dtype == "object":
        le_target = LabelEncoder()
        df_clean[target_col] = le_target.fit_transform(df_clean[target_col])
    
    if "PitcherId" in df.columns:
        df_clean["PitcherId"] = df["PitcherId"]
    if "BatterId" in df.columns:
        df_clean["BatterId"] = df["BatterId"]
        
    # Use variance-based encoding for categorical features
    df_encoded, encoders = encode_categorical_columns(df_clean, categorical_cols, target_col, encoders=None)
    
    # Determine numeric and categorical columns (exclude target and sequence id)
    numeric_cols = [c for c in df_encoded.select_dtypes(include=[np.number]).columns 
                    if c not in categorical_cols and c != target_col and c != seq_id_col]
    if "PrevStrike" not in df_encoded.columns:
        df_encoded["PrevStrike"] = 0.0
    if "PrevStrike" not in numeric_cols:
        numeric_cols.append("PrevStrike")
    # Exclude PitcherId and BatterId from categorical encoding (they might be used later as is)
    cat_cols = [c for c in categorical_cols if c not in ["PitcherId", "BatterId"]]
    feature_cols = numeric_cols + cat_cols

    # If feature selection is desired, use mutual information to choose top features
    if top_k_features is not None:
        selected_features = mutual_info_feature_selection(df_encoded[feature_cols], df_encoded[target_col], feature_cols, top_k=top_k_features)
        feature_cols = selected_features
        numeric_cols = [c for c in numeric_cols if c in feature_cols]
        cat_cols = [c for c in cat_cols if c in feature_cols]
    
    num_classes = df_encoded[target_col].nunique()
    
    # Create sequences by grouping on PlateAppearanceId
    sequences = []
    sequence_labels = []
    for pa_id, group in df_encoded.groupby(seq_id_col):
        # Sort by PitchNumber if available
        if "PitchNumber" in group.columns:
            group = group.sort_values("PitchNumber")
        seq_features = group[feature_cols].values  # (seq_len, feature_dim)
        label = group[target_col].iloc[-1]  # use last pitch outcome as sequence label
        sequences.append(seq_features)
        sequence_labels.append(label)
    
    # Pad sequences to equal length
    from torch.nn.utils.rnn import pad_sequence
    seq_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    padded_seqs = pad_sequence(seq_tensors, batch_first=True)
    X = padded_seqs.numpy()  # (num_sequences, max_seq_len, feature_dim)
    y = np.array(sequence_labels)
    
    # Scale numeric features (apply scaler on flattened numeric columns)
    num_idx = [feature_cols.index(c) for c in numeric_cols]
    X_reshaped = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler()
    X_reshaped[:, num_idx] = scaler.fit_transform(X_reshaped[:, num_idx])
    X = X_reshaped.reshape(X.shape)
    
    class_weights = compute_class_weights(y, device=device)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    final_trainer = HybridPitchTrainer(
        input_dim=X_train.shape[-1],
        num_classes=num_classes,
        cnn_channels=32,
        lstm_hidden_dim=64,
        lr=1e-4,
        batch_size=64,
        epochs=100,
        early_stopping_patience=10,
        device=device,
        class_weights=class_weights,
        weight_decay=1e-4,
        dropout=0.4
    )
    final_trainer.fit(X_train, y_train, X_val, y_val)
    ds_val = SequenceDataset(X_val, y_val)
    dl_val = DataLoader(ds_val, batch_size=64, shuffle=False)
    acc, f1, prec, rec = final_trainer.evaluate(dl_val)
    print(f"\nFinal Validation => ACC: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    final_trainer.save(save_dir="Derived_data/ad_model_params/")

    pipeline_extras = {
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "feature_cols": feature_cols,
        "target_encoder": le_target,
        "encoders": encoders,
        "df_processed": df_encoded,
        "sequence_ids": df_encoded[seq_id_col].unique()
    }
    save_pipeline_extras(pipeline_extras)

    return {
        "model_trainer": final_trainer,
        **pipeline_extras
    }

###############################################################################
# PREDICTION FUNCTION FOR REAL-TIME PITCH SIMULATION
###############################################################################
def prediction(pitcher, batter, model, scaler, encoders, target_encoder, df, feature_cols, numeric_cols, target_col,
               pitch_type_col="CleanPitchType", n_pitches=3, alpha=0.5):
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

    if target_encoder is not None:
        target_classes = list(target_encoder.classes_)
    else:
        target_classes = [f"Class_{i}" for i in range(model.model.fc_out.out_features)]

    # Use the stored encoder for the pitch type if available
    pitch_type_encoder = encoders.get(pitch_type_col, None)

    records = []
    for candidate in candidate_types:
        sim_row = baseline.copy(deep=True)
        if pitch_type_encoder is not None:
            # Reverse mapping: in target encoding we may not invert easily.
            # Here we assume the candidate value is already in an appropriate format.
            candidate_label = candidate
        else:
            candidate_label = candidate
        sim_row[pitch_type_col] = candidate_label
        cum_strike = 0.0
        candidate_records = []
        for pitch_num in range(1, n_pitches + 1):
            sim_row["PrevStrike"] = cum_strike
            sim_df = pd.DataFrame([sim_row])
            # Apply the same variance-based encoding to simulation data
            for col, mapping in encoders.items():
                if col in sim_df.columns:
                    sim_df[col] = sim_df[col].apply(lambda x: mapping.get(x, np.mean(list(mapping.values()))))
            X_sim = sim_df[feature_cols].values
            num_idx = [feature_cols.index(c) for c in numeric_cols]
            X_sim[:, num_idx] = scaler.transform(X_sim[:, num_idx])
            # Reshape to (1, 1, feature_dim) for the sequence model
            current_input = X_sim.reshape(1, 1, -1)
            probs = model.predict_proba(current_input)[0]
            cum_strike = alpha * probs[0] + (1 - alpha) * cum_strike
            rec = {"PitchNumber": pitch_num, "CandidatePitchType": candidate_label}
            for cls, p_val in zip(target_classes, probs):
                rec[cls] = round(p_val * 100, 2)
            candidate_records.append(rec)
        records.extend(candidate_records)
    df_rec = pd.DataFrame(records)
    if "Strike" in target_classes:
        strike_col = "Strike"
    else:
        strike_col = target_classes[0]
    best_dict = df_rec.groupby("PitchNumber").apply(lambda d: d.loc[d[strike_col].idxmax(), "CandidatePitchType"]).to_dict()
    df_rec["BestStrikePotential"] = df_rec["PitchNumber"].map(best_dict)
    df_rec = df_rec.sort_values("PitchNumber").reset_index(drop=True)
    return df_rec

###############################################################################
# MAIN PIPELINE EXECUTION
###############################################################################
if __name__ == "__main__":
    data_path = "Derived_Data/feature/nDate_feature.parquet"  # Adapt path as needed
    # Optionally, you can set top_k_features (e.g., 80% of features) if desired:
    pipeline_objs = main_hybrid_pipeline(data_path=data_path, top_k_features=None)
    if pipeline_objs is not None:
        final_trainer = pipeline_objs["model_trainer"]
        scaler = pipeline_objs["scaler"]
        numeric_cols = pipeline_objs["numeric_cols"]
        cat_cols = pipeline_objs["cat_cols"]
        feature_cols = pipeline_objs["feature_cols"]
        df_proc = pipeline_objs["df_processed"]
        encoders = pipeline_objs["encoders"]
        target_col = "CleanPitchCall"
        # For sample predictions, select one plate appearance (sequence)
        seq_id = df_proc["PlateAppearanceId"].iloc[0]
        sample_seq = df_proc[df_proc["PlateAppearanceId"] == seq_id].copy(deep=True)
        X_sample = sample_seq[feature_cols].values
        num_idxs = [feature_cols.index(c) for c in numeric_cols]
        X_sample[:, num_idxs] = scaler.transform(X_sample[:, num_idxs])
        # Reshape to (1, seq_len, feature_dim)
        X_sample = X_sample.reshape(1, -1, X_sample.shape[1])
        preds = final_trainer.predict(X_sample)
        print("\nSample Predictions:", preds)
        probs = final_trainer.predict_proba(X_sample)
        print("Sample Probabilities:\n", probs)
        pitcher_id = 1000066910.0
        batter_id = 1000032366.0
        cum_df = prediction(
            pitcher=pitcher_id,
            batter=batter_id,
            model=final_trainer,
            scaler=scaler,
            encoders=encoders,
            target_encoder=pipeline_objs["target_encoder"],
            df=df_proc,
            feature_cols=feature_cols,
            numeric_cols=numeric_cols,
            target_col=target_col,
            pitch_type_col="CleanPitchType",
            n_pitches=10,
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
