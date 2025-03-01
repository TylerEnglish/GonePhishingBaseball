import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

# For data augmentation / balancing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

def advanced_data_cleaning(df, target_col, categorical_cols):
    """
    1. Impute numeric columns by median.
    2. Impute categorical columns with "Missing".
    3. Drop rows where target_col is missing.
    """
    df_clean = df.copy(deep=True)
    
    # Numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]
    numeric_imputer = SimpleImputer(strategy='median')
    df_clean[numeric_cols] = numeric_imputer.fit_transform(df_clean[numeric_cols])
    
    # Categorical columns
    for cat_col in categorical_cols:
        if cat_col not in df_clean.columns:
            continue
        df_clean[cat_col] = df_clean[cat_col].fillna('Missing')
    
    # Drop rows where target is missing
    df_clean = df_clean.dropna(subset=[target_col])
    return df_clean

def encode_categorical_columns(df, categorical_cols, encoders=None):
    """
    Label-encode each categorical column. If encoders is None, new encoders are fit.
    Returns (df_encoded, encoders).
    """
    df_encoded = df.copy()
    
    if encoders is None:
        encoders = {}
        fit_encoders = True
    else:
        fit_encoders = False
    
    for col in categorical_cols:
        if col not in df_encoded.columns:
            continue
        
        df_encoded[col] = df_encoded[col].astype(str)
        
        if fit_encoders:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df_encoded[col] = le.transform(df_encoded[col])
    
    return df_encoded, encoders

def mutual_info_feature_selection(X, y, feature_cols, top_k=20):
    """
    Select top_k features using mutual information.
    Returns list of selected feature names.
    """
    mi_scores = mutual_info_classif(X, y, random_state=42)
    feature_mi = sorted(zip(feature_cols, mi_scores), key=lambda x: x[1], reverse=True)
    selected_features = [f[0] for f in feature_mi[:top_k]]
    return selected_features

def augment_data_with_noise(X, noise_level=0.01):
    """
    Add Gaussian noise to numeric features for data augmentation.
    noise_level is relative, e.g. 0.01 = 1% of std.
    """
    X_noisy = X.copy()
    std_vec = np.std(X_noisy, axis=0)
    noise = np.random.randn(*X_noisy.shape) * (std_vec * noise_level)
    X_noisy += noise
    return X_noisy

###############################################################################
# CUSTOM DATASET FOR PYTORCH
###############################################################################

class TabularDataset(Dataset):
    """
    A simple Dataset for tabular data.
    """
    def __init__(self, X, y=None):
        """
        X: np.array of shape (num_samples, num_features)
        y: np.array of shape (num_samples,) or None if inference
        """
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None
    
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

###############################################################################
# A SIMPLE TRANSFORMER MODEL FOR TABULAR DATA
###############################################################################

class TabTransformer(nn.Module):
    """
    A minimal Transformer-based model for tabular data.
    - Numeric features are passed through a linear layer (optionally).
    - Categorical features are turned into embeddings.
    - Concatenate or sum these embeddings to form a sequence, then pass through
      a standard Transformer encoder.
    - Finally, average pool or CLS token to produce a classification logit.
    
    Note: The architecture can be adapted in many ways. This is just an example.
    
    Usage:
      - Provide a dimension for embeddings (model_dim).
      - Provide how many heads, layers, etc.
      - Provide # of numeric features vs # of categorical features.
    """
    def __init__(
        self,
        num_numeric,
        num_categories_list,  # list of unique cat counts for each categorical col
        model_dim=32,
        num_heads=4,
        num_layers=2,
        num_classes=2
    ):
        super().__init__()
        
        self.num_numeric = num_numeric
        self.num_cats = len(num_categories_list)
        self.model_dim = model_dim
        self.num_classes = num_classes
        
        # Embeddings for each categorical feature
        # Each cat feature gets an embedding of size model_dim
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_cats, model_dim)
            for num_cats in num_categories_list
        ])
        
        # Linear embedding for numeric features (optional)
        # We can embed numeric features into model_dim dimension for each numeric feature,
        # or handle them differently. Here we do a single linear to model_dim:
        if self.num_numeric > 0:
            self.numeric_linear = nn.Linear(num_numeric, model_dim)
        else:
            self.numeric_linear = None
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        # We'll use a simple pooling approach (mean pool across the sequence).
        self.fc_out = nn.Linear(model_dim, num_classes)
    
    def forward(self, x_numeric, x_categorical):
        """
        x_numeric: [batch_size, num_numeric]
        x_categorical: list of [batch_size] for each cat col OR [batch_size, num_cats]
        """
        batch_size = x_numeric.size(0)
        
        # Embeddings for categorical features
        cat_embs = []
        for i, emb_layer in enumerate(self.cat_embeddings):
            # x_categorical[:, i] => shape [batch_size]
            # After embedding => [batch_size, model_dim]
            emb = emb_layer(x_categorical[:, i])
            cat_embs.append(emb)
        if len(cat_embs) > 0:
            cat_embs = torch.stack(cat_embs, dim=1)  # [batch_size, num_cats, model_dim]
        else:
            cat_embs = None
        
        # Embedding for numeric features
        if self.numeric_linear is not None:
            # [batch_size, num_numeric] -> [batch_size, model_dim]
            num_emb = self.numeric_linear(x_numeric)
            # We'll treat each numeric row as a single "token" (just an example approach).
            num_emb = num_emb.unsqueeze(1)  # [batch_size, 1, model_dim]
        else:
            num_emb = None
        
        # Combine numeric "token" with the cat embeddings to form a sequence
        if cat_embs is not None and num_emb is not None:
            seq = torch.cat([num_emb, cat_embs], dim=1)  # shape [batch_size, (1 + num_cats), model_dim]
        elif cat_embs is None:
            seq = num_emb
        else:
            seq = cat_embs
        
        # Pass through Transformer
        # shape remains [batch_size, seq_len, model_dim]
        x_enc = self.transformer_encoder(seq)  
        
        # Pooling: mean pool across seq_len dimension
        # x_enc => [batch_size, seq_len, model_dim]
        x_pooled = x_enc.mean(dim=1)  # [batch_size, model_dim]
        
        # Classification
        logits = self.fc_out(x_pooled)  # [batch_size, num_classes]
        
        return logits

###############################################################################
# TRAINING & EVALUATION FUNCTIONS FOR OUR TRANSFORMER MODEL
###############################################################################

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        # X_batch => shape [batch_size, num_features]
        # We need to split numeric vs cat features
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Suppose numeric features come first, cat features come last
        # We'll define a split scheme externally or dynamically
        # We'll guess: numeric_cols + cat_cols = total features
        # For example, let's store that info in the dataloader via closure or something:
        # but let's do a simpler approach:
        
        # We'll assume cat indices are integer-coded in last columns,
        # numeric in first columns. We need the actual split index:
        # Because we might not know them in code, let's say we store them globally or pass them in.
        # We'll assume we stored them in the dataloader. For brevity, let's define placeholders:
        
        # We can't do this inside the function without that info. We'll pass in a global or add as param
        raise NotImplementedError("You need to split numeric vs cat inside the loop. See code below for an example.")


def evaluate_model(model, dataloader, device):
    model.eval()
    preds_all = []
    labels_all = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # Similar numeric/categorical split needed here.
            raise NotImplementedError("Split numeric vs cat for inference. Return predictions.")
    
    # Once we've done the inference, compute accuracy/f1.
    # ...
    return accuracy, f1


###############################################################################
# A FULL (WORKING) EXAMPLE, WITH SPLIT OF NUMERIC AND CATS
###############################################################################

class TabTransformerTrainer:
    """
    A small helper class that:
     - Splits features into numeric vs cat columns
     - Builds the TabTransformer
     - Provides fit/predict methods
    """
    def __init__(
        self,
        numeric_dims,
        cat_dims,
        num_classes=2,
        model_dim=32,
        num_heads=4,
        num_layers=2,
        lr=1e-3,
        batch_size=256,
        epochs=10,
        device='cpu'
    ):
        """
        numeric_dims: number of numeric columns
        cat_dims: list of cardinalities for each categorical column
        """
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
        self.criterion = nn.CrossEntropyLoss()
        
        # Remember how many numeric vs cat columns
        self.numeric_dims = numeric_dims
        self.cat_dims = cat_dims
    
    def _split_features(self, X):
        """
        X is shape [batch_size, total_features].
        The first self.numeric_dims are numeric, the rest are cat.
        We return x_numeric, x_cat (as int64).
        """
        x_numeric = X[:, :self.numeric_dims]
        x_cat = X[:, self.numeric_dims:].long()
        return x_numeric, x_cat
    
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        # Create datasets
        train_ds = TabularDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        if X_valid is not None and y_valid is not None:
            valid_ds = TabularDataset(X_valid, y_valid)
            valid_dl = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False)
        else:
            valid_dl = None
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for Xb, yb in train_dl:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                
                # Split numeric vs cat
                x_num, x_cat = self._split_features(Xb)
                
                logits = self.model(x_num, x_cat)
                loss = self.criterion(logits, yb)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dl)
            
            if valid_dl is not None:
                acc_val, f1_val = self.evaluate(valid_dl)
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} - ValAcc: {acc_val:.4f}, ValF1: {f1_val:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")
    
    def evaluate(self, dataloader):
        self.model.eval()
        preds_all = []
        labels_all = []
        
        with torch.no_grad():
            for Xb, yb in dataloader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                
                x_num, x_cat = self._split_features(Xb)
                logits = self.model(x_num, x_cat)
                preds = torch.argmax(logits, dim=1)
                
                preds_all.append(preds.cpu().numpy())
                labels_all.append(yb.cpu().numpy())
        
        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)
        
        acc = accuracy_score(labels_all, preds_all)
        f1 = f1_score(labels_all, preds_all, average='macro')
        return acc, f1
    
    def predict(self, X):
        """
        Inference on new data X (numpy array).
        Returns predicted labels.
        """
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
        
        all_preds = np.concatenate(all_preds)
        return all_preds
    
    def predict_proba(self, X):
        """
        Returns class probabilities for X.
        """
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
        
        all_probs = np.concatenate(all_probs, axis=0)
        return all_probs

###############################################################################
# CROSS-VALIDATION WITH THE TRANSFORMER MODEL
###############################################################################

def cross_validate_transformer(
    df,
    feature_cols,
    target_col,
    numeric_cols,
    cat_cols,
    cat_cardinalities,
    num_classes,
    n_splits=5,
    top_k_features=None,
    device='cpu'
):
    """
    Demonstrates cross-validation with the custom TabTransformerTrainer.
    Also includes SMOTE, noise injection, etc.
    """
    X = df[feature_cols].values
    y = df[target_col].values
    
    # If top_k_features is specified, do mutual_info selection
    if top_k_features is not None:
        mi_selected_features = mutual_info_feature_selection(X, y, feature_cols, top_k=top_k_features)
        df_selected = df[mi_selected_features + [target_col]].copy()
        
        # We also need to update numeric_cols, cat_cols based on selected features
        numeric_cols = [c for c in numeric_cols if c in mi_selected_features]
        cat_cols = [c for c in cat_cols if c in mi_selected_features]
        
        feature_cols = mi_selected_features
    else:
        df_selected = df.copy()
    
    X = df_selected[feature_cols].values
    y = df_selected[target_col].values
    
    # Create folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold_idx+1}/{n_splits} ===")
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        # SMOTE (optional)
        sm = SMOTE(random_state=42)
        X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
        
        # Add noise (optional)
        X_train_sm_noisy = augment_data_with_noise(X_train_sm, noise_level=0.01)
        
        # Next, we might scale numeric columns only
        # We'll do that in-place. We must do it carefully (only numeric portion).
        # numeric_indices, cat_indices
        numeric_indices = [feature_cols.index(c) for c in numeric_cols]
        cat_indices = [feature_cols.index(c) for c in cat_cols]
        
        # Scale numeric columns in X_train_sm_noisy, X_valid
        scaler = StandardScaler()
        
        # Extract numeric part
        X_train_num = X_train_sm_noisy[:, numeric_indices]
        X_valid_num = X_valid[:, numeric_indices]
        
        X_train_num_scaled = scaler.fit_transform(X_train_num)
        X_valid_num_scaled = scaler.transform(X_valid_num)
        
        # Put scaled values back
        X_train_sm_noisy[:, numeric_indices] = X_train_num_scaled
        X_valid[:, numeric_indices] = X_valid_num_scaled
        
        # Now define the model
        # We need cat cardinalities for each cat col, or pass them in
        # We'll assume cat_cardinalities is in the same order as cat_cols
        cat_dims_local = []
        for cat_col in cat_cols:
            # Find index of cat_col in cat_cols
            i = cat_cols.index(cat_col)
            cat_dims_local.append(cat_cardinalities[i])
        
        trainer = TabTransformerTrainer(
            numeric_dims=len(numeric_cols),
            cat_dims=cat_dims_local,
            num_classes=num_classes,
            model_dim=32,
            num_heads=4,
            num_layers=2,
            lr=1e-3,
            batch_size=512,
            epochs=10,
            device=device
        )
        
        # Train
        trainer.fit(X_train_sm_noisy, y_train_sm, X_valid, y_valid)
        
        # Evaluate
        valid_ds = TabularDataset(X_valid, y_valid)
        valid_dl = DataLoader(valid_ds, batch_size=512, shuffle=False)
        acc_val, f1_val = trainer.evaluate(valid_dl)
        print(f"Fold {fold_idx+1} - ACC: {acc_val:.4f}, F1: {f1_val:.4f}")
        results.append((acc_val, f1_val))
    
    # Print overall CV results
    accs = [r[0] for r in results]
    f1s = [r[1] for r in results]
    print("\n=== Cross-Validation Summary ===")
    print(f"Mean ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Mean F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

###############################################################################
# MAIN SCRIPT-LIKE FUNCTION
###############################################################################

def main_transformer_pipeline(data_path="data.parquet", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None
    
    print(f"Reading data from: {data_path}")
    df = pq.read_table(source=data_path).to_pandas()
    
    # Define target column
    target_col = "PitchCall"
    
    # Identify categorical columns
    all_object_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # If PitcherId, BatterId are numeric but should be categorical, convert them to str
    # df["PitcherId"] = df["PitcherId"].astype(str)
    # df["BatterId"] = df["BatterId"].astype(str)
    
    # Build final list of categorical columns
    if target_col in all_object_cols:
        # remove target col from cat list
        all_object_cols.remove(target_col)
    categorical_cols = all_object_cols
    
    # Basic cleaning
    df_clean = advanced_data_cleaning(df, target_col, categorical_cols)
    
    # If target is object, encode it
    if df_clean[target_col].dtype == 'object':
        le_target = LabelEncoder()
        df_clean[target_col] = le_target.fit_transform(df_clean[target_col])
    else:
        le_target = None
    
    # Now encode cat features
    df_encoded, encoders = encode_categorical_columns(df_clean, categorical_cols, encoders=None)
    
    # Build feature list
    feature_cols = [c for c in df_encoded.columns if c != target_col]
    
    # Identify which are numeric vs cat
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    # But we have also encoded the cat columns => they're numeric now.
    # We must separate them carefully. Because we label-encoded them, we should keep track.
    # We'll define:
    cat_cols = list(categorical_cols)  # the original cat columns
    # But now they're also numeric. Let's remove them from numeric_cols to avoid duplication:
    numeric_cols = [c for c in numeric_cols if c not in cat_cols and c != target_col]
    
    # We also need the cardinalities for each cat column
    cat_cardinalities = []
    for col in cat_cols:
        cat_cardinalities.append(df_encoded[col].max() + 1)  # since it's label-encoded from 0..max
    
    # Type cast cat columns to int if not already
    for c in cat_cols:
        df_encoded[c] = df_encoded[c].astype(int)
    
    # Number of classes
    num_classes = len(df_clean[target_col].unique())  # or len(np.unique(df_clean[target_col]))
    
    print("\n===== CROSS VALIDATION STAGE =====")
    cross_validate_transformer(
        df=df_encoded,
        feature_cols=feature_cols,
        target_col=target_col,
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
        cat_cardinalities=cat_cardinalities,
        num_classes=num_classes,
        n_splits=3,
        top_k_features=20,  # mutual info feature selection
        device=device
    )
    
    print("\n===== FINAL TRAINING STAGE =====")
    # A simplified approach to final training:
    # 1. Possibly re-do feature selection or skip
    # 2. Train on entire data (or train/val split)
    # Here we do a quick train/val split:
    
    X = df_encoded[feature_cols].values
    y = df_encoded[target_col].values
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    X_train_sm_noisy = augment_data_with_noise(X_train_sm, noise_level=0.01)
    
    # Scale numeric
    numeric_indices = [feature_cols.index(c) for c in numeric_cols]
    scaler = StandardScaler()
    
    X_train_num = X_train_sm_noisy[:, numeric_indices]
    X_valid_num = X_valid[:, numeric_indices]
    
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_valid_num_scaled = scaler.transform(X_valid_num)
    
    X_train_sm_noisy[:, numeric_indices] = X_train_num_scaled
    X_valid[:, numeric_indices] = X_valid_num_scaled
    
    final_trainer = TabTransformerTrainer(
        numeric_dims=len(numeric_cols),
        cat_dims=cat_cardinalities,
        num_classes=num_classes,
        model_dim=32,
        num_heads=4,
        num_layers=2,
        lr=1e-3,
        batch_size=512,
        epochs=10,
        device=device
    )
    
    final_trainer.fit(X_train_sm_noisy, y_train_sm, X_valid, y_valid)
    # Evaluate final
    valid_ds = TabularDataset(X_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=512, shuffle=False)
    acc_val, f1_val = final_trainer.evaluate(valid_dl)
    print(f"\nFinal Validation - ACC: {acc_val:.4f}, F1: {f1_val:.4f}")
    
    # Return objects needed for inference
    return {
        "model_trainer": final_trainer,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "feature_cols": feature_cols,
        "target_encoder": le_target,
        "df_processed": df_encoded
    }


###############################################################################
# EXAMPLE USAGE
###############################################################################

if __name__ == "__main__":
    data_path = "Derived_Data/feature/nDate_feature.parquet"
    
    pipeline_objects = main_transformer_pipeline(data_path=data_path)
    
    # For inference:
    # 1. Prepare new data the same way (clean, encode cats, scale numeric, etc.).
    # 2. Then call final_trainer.predict(...) or predict_proba(...).
    if pipeline_objects is not None:
        final_trainer = pipeline_objects["model_trainer"]
        scaler = pipeline_objects["scaler"]
        numeric_cols = pipeline_objects["numeric_cols"]
        cat_cols = pipeline_objects["cat_cols"]
        feature_cols = pipeline_objects["feature_cols"]
        
        # Example: Predict on a small subset of the original data
        df_proc = pipeline_objects["df_processed"]
        sample = df_proc.iloc[:5].copy()
        X_sample = sample[feature_cols].values
        
        # Scale numeric
        numeric_idxs = [feature_cols.index(c) for c in numeric_cols]
        X_sample_num = X_sample[:, numeric_idxs]
        X_sample_num_scaled = scaler.transform(X_sample_num)
        X_sample[:, numeric_idxs] = X_sample_num_scaled
        
        preds = final_trainer.predict(X_sample)
        print("\nSample Predictions:", preds)
        
        probs = final_trainer.predict_proba(X_sample)
        print("Sample Probabilities:\n", probs)
