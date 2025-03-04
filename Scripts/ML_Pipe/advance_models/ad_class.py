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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

import warnings
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

EPS = 1e-8

###############################################################################
# UTILITY FUNCTIONS (unchanged)
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
# LABEL-SMOOTHING CROSS-ENTROPY LOSS
###############################################################################
class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Label-Smoothing Cross-Entropy Loss.
    
    Instead of one-hot targets, the ground-truth label is smoothed.
    """
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
# SQUEEZE-AND-EXCITATION (SE) BLOCK 
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
# FiLM BLOCK & MultiHead Latent Attention BLOCK
###############################################################################
class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) block.
    Modulates the input features using a conditioning vector.
    """
    def __init__(self, feature_dim, condition_dim, hidden_dim=64):
        super().__init__()
        self.gamma_fc = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.beta_fc = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    def forward(self, x, condition):
        gamma = self.gamma_fc(condition)
        beta = self.beta_fc(condition)
        return gamma * x + beta

class MultiHeadLatentAttention(nn.Module):
    """
    MultiHead Latent Attention block.
    Uses a learnable latent query to attend over the input features.
    """
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.latent_query = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
    def forward(self, x):
        x_unsq = x.unsqueeze(1)  # [batch, 1, feature_dim]
        batch_size = x.size(0)
        query = self.latent_query.expand(batch_size, -1, -1)
        attn_out, _ = self.attention(query, x_unsq, x_unsq)
        return attn_out.squeeze(1)

###############################################################################
# Gated Residual Network (GRN) BLOCK
###############################################################################
class GRNBlock(nn.Module):
    """
    A Gated Residual Network block to capture higher-order feature interactions.
    """
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
# MULTI-LEVEL TRANSFORMER MODEL WITH POSITIONAL ENCODING
###############################################################################
class MultiLevelTabTransformer(nn.Module):
    def __init__(self, num_numeric, num_categories_list, model_dim=64,
                 num_heads=4, num_layers_cat=2, num_layers_num=2, num_layers_fusion=3,
                 num_classes=2, dropout=0.4):
        super().__init__()
        self.num_numeric = num_numeric
        self.num_cats = len(num_categories_list)
        self.model_dim = model_dim
        self.num_classes = num_classes

        # Categorical embeddings and transformer
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_size, model_dim) for cat_size in num_categories_list
        ])
        encoder_layer_cat = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.cat_transformer = nn.TransformerEncoder(encoder_layer_cat, num_layers=num_layers_cat)
        self.cat_layernorm = nn.LayerNorm(model_dim)

        # Numeric projection with positional encoding
        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim)
        )
        # Learnable positional encoding for numeric features
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, model_dim))
        encoder_layer_num = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.num_transformer = nn.TransformerEncoder(encoder_layer_num, num_layers=num_layers_num)
        self.num_layernorm = nn.LayerNorm(model_dim)

        # Fusion block with residual connection and SE block
        self.fusion_linear = nn.Linear(model_dim*2, model_dim)
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_layernorm = nn.LayerNorm(model_dim)
        self.fusion_residual = nn.Sequential(
            nn.Linear(model_dim*2, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        encoder_layer_fusion = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer_fusion, num_layers=num_layers_fusion)
        self.se_block = SEBlock(model_dim, reduction=4, dropout_rate=dropout)

        # FiLM block and latent attention
        self.film_block = FiLMBlock(model_dim, model_dim, hidden_dim=model_dim)
        self.latent_attention = MultiHeadLatentAttention(model_dim, num_heads=num_heads)

        # GRN block
        self.grn_block = GRNBlock(model_dim, model_dim, dropout=dropout)

        # Additional MLP head with residual connection
        self.mlp_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim)
        )
        self.mlp_layernorm = nn.LayerNorm(model_dim)

        self.fc_out = nn.Linear(model_dim, num_classes)

    def forward(self, x_numeric, x_categorical):
        # Process categorical features
        cat_embs = []
        for i, emb in enumerate(self.cat_embeddings):
            idx = x_categorical[:, i].long()
            idx = torch.clamp(idx, 0, emb.num_embeddings - 1)
            cat_embs.append(emb(idx))
        cat_seq = torch.stack(cat_embs, dim=1)
        cat_enc = self.cat_transformer(cat_seq)
        cat_enc = self.cat_layernorm(cat_enc)
        cat_pool = cat_enc.mean(dim=1)

        # Process numeric features + add positional encoding
        num_proj = self.numeric_proj(x_numeric)
        num_proj = num_proj.unsqueeze(1) + self.positional_encoding  
        num_enc = self.num_transformer(num_proj)
        num_enc = self.num_layernorm(num_enc.squeeze(1))

        # Fusion of numeric and categorical features
        fusion_input = torch.cat([num_enc, cat_pool], dim=1)
        fusion_proj = F.relu(self.fusion_linear(fusion_input))
        fusion_proj = self.fusion_dropout(fusion_proj)
        residual_proj = self.fusion_residual(fusion_input)
        fusion_res = fusion_proj + residual_proj
        fusion_res = self.fusion_layernorm(fusion_res)
        fusion_res = self.se_block(fusion_res)
        fusion_seq = fusion_res.unsqueeze(1)
        fusion_enc = self.fusion_transformer(fusion_seq)
        fusion_pool = fusion_enc.squeeze(1)
        
        # FiLM modulation and latent attention
        modulated = self.film_block(fusion_pool, cat_pool)
        latent_att = self.latent_attention(modulated)
        grn_out = self.grn_block(latent_att)
        
        # MLP head with residual connection
        mlp_input = grn_out + latent_att
        mlp_out = self.mlp_head(mlp_input)
        mlp_out = mlp_out + mlp_input
        mlp_out = self.mlp_layernorm(mlp_out)
        
        logits = self.fc_out(mlp_out)
        return logits

###############################################################################
# TRAINING WRAPPER WITH LR WARMUP, SCHEDULER, EARLY STOPPING & GRADIENT CLIPPING
###############################################################################
class TabTransformerTrainer:
    def __init__(self, numeric_dims, cat_dims, num_classes=2,
                 model_dim=64, num_heads=4, num_layers_cat=2, num_layers_num=2,
                 num_layers_fusion=3, lr=1e-4, batch_size=256, epochs=25,
                 device='cpu', class_weights=None, weight_decay=1e-4,
                 grad_clip=1.0, warmup_epochs=3, early_stopping_patience=25, dropout=0.4,
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

        self.model = MultiLevelTabTransformer(
            num_numeric=numeric_dims,
            num_categories_list=cat_dims,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers_cat=num_layers_cat,
            num_layers_num=num_layers_num,
            num_layers_fusion=num_layers_fusion,
            num_classes=num_classes,
            dropout=dropout
        ).to(device)

        self.optimizer = Ranger(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=1e-6)
        self.criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.1, weight=class_weights)
        self.numeric_dims = numeric_dims

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
                x_num, x_cat = self._split_features(Xb)
                if self.use_mixup:
                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    index = torch.randperm(x_num.size(0)).to(self.device)
                    x_num = lam * x_num + (1 - lam) * x_num[index]
                    y_a, y_b = yb, yb[index]
                    logits = self.model(x_num, x_cat)
                    loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
                else:
                    logits = self.model(x_num, x_cat)
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
                x_num, x_cat = self._split_features(Xb)
                logits = self.model(x_num, x_cat)
                loss = self.criterion(logits, yb)
                total_loss += loss.item()
        return total_loss / len(dataloader)

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
        rec = recall_score(labels, preds, average="macro", zero_division=1)
        return acc, f1, prec, rec

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
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    def predict_cumulative(self, X_sequence, alpha=0.5):
        n_pitches = X_sequence.shape[0]
        cumulative_results = []
        cum_strike = 0.0
        for i in range(n_pitches):
            X_sequence[i, -1] = cum_strike
            current_probs = self.predict_proba(X_sequence[i:i+1])[0]
            cum_strike = alpha * current_probs[0] + (1 - alpha) * cum_strike
            rec = {"PitchNumber": i + 1, "CumulativePrediction": int(np.argmax(current_probs))}
            classes = [f"Class_{j}" for j in range(2, self.model.num_classes)]
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
# CROSS-VALIDATION & MAIN PIPELINE
###############################################################################
def cross_validate_transformer(df, feature_cols, target_col,
                               numeric_cols, cat_cols, cat_cardinalities,
                               num_classes, n_splits=5, top_k_features=None,
                               device='cpu'):
    X = df[feature_cols].values
    y = df[target_col].values
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
    cat_cardinalities_local = [cat_cardinalities[i] for i, c in enumerate(cat_cols)]
    X = df_sel[feature_cols].values
    y = df_sel[target_col].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold_idx+1}/{n_splits} ===")
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        X_train_noisy = augment_data_with_noise(X_train, noise_level=0.01)
        num_idx = [feature_cols.index(c) for c in numeric_cols]
        scaler = StandardScaler()
        X_train_num = X_train_noisy[:, num_idx]
        X_valid_num = X_valid[:, num_idx]
        X_train_num_sc = scaler.fit_transform(X_train_num)
        X_valid_num_sc = scaler.transform(X_valid_num)
        X_train_noisy[:, num_idx] = X_train_num_sc
        X_valid[:, num_idx] = X_valid_num_sc
        cat_dims_local = [cat_cardinalities_local[cat_cols.index(c)] for c in cat_cols]
        class_weights = compute_class_weights(y_train, device=device)
        trainer = TabTransformerTrainer(
            numeric_dims=len(numeric_cols),
            cat_dims=cat_dims_local,
            num_classes=num_classes,
            model_dim=64,
            num_heads=4,
            num_layers_cat=2,
            num_layers_num=2,
            num_layers_fusion=3,
            lr=1e-4,
            batch_size=512,
            epochs=100,
            early_stopping_patience=10,
            device=device,
            class_weights=class_weights,
            weight_decay=1e-4,
            dropout=0.4
        )
        trainer.fit(X_train_noisy, y_train, X_valid, y_valid)
        ds_valid = TabularDataset(X_valid, y_valid)
        dl_valid = DataLoader(ds_valid, batch_size=512, shuffle=False)
        acc, f1, prec, rec = trainer.evaluate(dl_valid)
        print(f"Fold {fold_idx+1} - ACC: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
        results.append((acc, f1, prec, rec))
    accs = [r[0] for r in results]
    f1s = [r[1] for r in results]
    precs = [r[2] for r in results]
    recs = [r[3] for r in results]
    print("\n=== Cross-Validation Summary ===")
    print(f"Mean ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Mean Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"Mean Recall: {np.mean(recs):.4f} ± {np.std(recs):.4f}")

def main_transformer_pipeline(data_path="data.parquet", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None
    print(f"Reading data from: {data_path}")
    df = pq.read_table(source=data_path).to_pandas()
    target_col = "CleanPitchCall"
    all_objs = df.select_dtypes(include=["object"]).columns.tolist()
    if target_col in all_objs:
        all_objs.remove(target_col)
    categorical_cols = all_objs
    df_clean = advanced_data_cleaning(df, target_col, categorical_cols)
    le_target = None
    if df_clean[target_col].dtype == "object":
        le_target = LabelEncoder()
        df_clean[target_col] = le_target.fit_transform(df_clean[target_col])
    if "PitcherId" in df.columns:
        df_clean["PitcherId"] = df["PitcherId"]
    if "BatterId" in df.columns:
        df_clean["BatterId"] = df["BatterId"]
    df_encoded, encoders = encode_categorical_columns(df_clean, categorical_cols, encoders=None)
    numeric_cols = [c for c in df_encoded.select_dtypes(include=[np.number]).columns
                    if c not in categorical_cols and c != target_col]
    if "PrevStrike" not in df_encoded.columns:
        df_encoded["PrevStrike"] = 0.0
    if "PrevStrike" not in numeric_cols:
        numeric_cols.append("PrevStrike")
    cat_cols = [c for c in categorical_cols if c not in ["PitcherId", "BatterId"]]
    feature_cols = numeric_cols + cat_cols
    for c in cat_cols:
        df_encoded[c] = df_encoded[c].astype(int)
    num_classes = df_encoded[target_col].nunique()
    cat_cardinalities_global = [int(df_encoded[c].max() + 10) for c in cat_cols]
    print("\n===== CROSS VALIDATION STAGE =====")
    # Uncomment cross-validation call if needed.
    # cross_validate_transformer(
    #     df=df_encoded,
    #     feature_cols=feature_cols,
    #     target_col=target_col,
    #     numeric_cols=numeric_cols,
    #     cat_cols=cat_cols,
    #     cat_cardinalities=cat_cardinalities_global,
    #     num_classes=num_classes,
    #     n_splits=3,
    #     top_k_features=None,
    #     device=device
    # )
    print("\n===== FINAL TRAINING STAGE =====")
    X_all = df_encoded[feature_cols].values
    y_all = df_encoded[target_col].values
    sel = mutual_info_feature_selection(X_all, y_all, feature_cols, top_k=None)
    numeric_cols_final = [c for c in numeric_cols if c in sel]
    cat_cols_final = [c for c in cat_cols if c in sel]
    feature_cols_final = numeric_cols_final + cat_cols_final
    df_final = df_encoded[feature_cols_final + [target_col]].copy()
    cat_cardinalities_final = [int(df_final[c].max() + 10) for c in cat_cols_final]
    X = df_final[feature_cols_final].values
    y = df_final[target_col].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train_noisy = augment_data_with_noise(X_train, noise_level=0.01)
    num_idx = [feature_cols_final.index(c) for c in numeric_cols_final]
    scaler = StandardScaler()
    X_train_num = X_train_noisy[:, num_idx]
    X_val_num = X_val[:, num_idx]
    X_train_num_sc = scaler.fit_transform(X_train_num)
    X_val_num_sc = scaler.transform(X_val_num)
    X_train_noisy[:, num_idx] = X_train_num_sc
    X_val[:, num_idx] = X_val_num_sc
    class_weights = compute_class_weights(y_train, device=device)
    final_trainer = TabTransformerTrainer(
        numeric_dims=len(numeric_cols_final),
        cat_dims=cat_cardinalities_final,
        num_classes=num_classes,
        model_dim=64,
        num_heads=4,
        num_layers_cat=2,
        num_layers_num=2,
        num_layers_fusion=3,
        lr=1e-4,
        batch_size=512,
        epochs=100,
        early_stopping_patience=10,
        device=device,
        class_weights=class_weights,
        weight_decay=1e-4,
        dropout=0.4
    )
    final_trainer.fit(X_train_noisy, y_train, X_val, y_val)
    ds_val = TabularDataset(X_val, y_val)
    dl_val = DataLoader(ds_val, batch_size=512, shuffle=False)
    acc, f1, prec, rec = final_trainer.evaluate(dl_val)
    print(f"\nFinal Validation => ACC: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    final_trainer.save(save_dir="Derived_data/ad_model_params/")

    # Save pipeline extras
    pipeline_extras = {
        "scaler": scaler,
        "numeric_cols": numeric_cols_final,
        "cat_cols": cat_cols_final,
        "feature_cols": feature_cols_final,
        "target_encoder": le_target,
        "encoders": encoders,
        "df_processed": df_final,
    }
    save_pipeline_extras(pipeline_extras)

    return {
        "model_trainer": final_trainer,
        **pipeline_extras
    }

###############################################################################
# PREDICTION FUNCTION WITH SAFE TRANSFORM FOR CATEGORICAL VALUES
# Note: The prediction function now accepts "numeric_cols" as a parameter.
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

    # Use the provided target_encoder to get the actual target class names.
    if target_encoder is not None:
        target_classes = list(target_encoder.classes_)
    else:
        target_classes = [f"Class_{i}" for i in range(model.model.num_classes)]

    # Retrieve the pitch type encoder (if available) for CleanPitchType.
    pitch_type_encoder = encoders.get(pitch_type_col, None)

    records = []
    for candidate in candidate_types:
        sim_row = baseline.copy(deep=True)
        # Convert candidate to its actual pitch label if a pitch type encoder exists.
        if pitch_type_encoder is not None:
            candidate_label = pitch_type_encoder.inverse_transform([candidate])[0]
        else:
            candidate_label = candidate
        sim_row[pitch_type_col] = candidate_label
        cum_strike = 0.0
        candidate_records = []
        for pitch_num in range(1, n_pitches + 1):
            sim_row["PrevStrike"] = cum_strike
            sim_df = pd.DataFrame([sim_row])
            # For each categorical column, apply a safe transform.
            for col, encoder in encoders.items():
                if col in sim_df.columns:
                    sim_df[col] = sim_df[col].astype(str)
                    def safe_transform_value(x):
                        return x if x in encoder.classes_ else encoder.classes_[0]
                    sim_df[col] = sim_df[col].apply(safe_transform_value)
                    sim_df[col] = encoder.transform(sim_df[col])
            X_sim = sim_df[feature_cols].values
            # Scale only the numeric features.
            num_idx = [feature_cols.index(c) for c in numeric_cols]
            X_sim[:, num_idx] = scaler.transform(X_sim[:, num_idx])
            probs = model.predict_proba(X_sim)[0]
            cum_strike = alpha * probs[0] + (1 - alpha) * cum_strike
            rec = {"PitchNumber": pitch_num, "CandidatePitchType": candidate_label}
            for cls, p_val in zip(target_classes, probs):
                rec[cls] = round(p_val * 100, 2)
            candidate_records.append(rec)
        records.extend(candidate_records)
    df_rec = pd.DataFrame(records)
    # Use the "Strike" column if available; otherwise, default to the first target class.
    if "Strike" in target_classes:
        strike_col = "Strike"
    else:
        strike_col = target_classes[0]
    # For each pitch number, pick the candidate with the highest strike probability.
    best_dict = df_rec.groupby("PitchNumber").apply(lambda d: d.loc[d[strike_col].idxmax(), "CandidatePitchType"]).to_dict()
    df_rec["BestStrikePotential"] = df_rec["PitchNumber"].map(best_dict)
    # Sort the results by pitch number.
    df_rec = df_rec.sort_values("PitchNumber").reset_index(drop=True)
    return df_rec

###############################################################################
# MAIN PIPELINE
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
        sample = df_proc.iloc[:5].copy(deep=True)
        X_sample = sample[feature_cols].values
        num_idxs = [feature_cols.index(c) for c in numeric_cols]
        X_sample[:, num_idxs] = scaler.transform(X_sample[:, num_idxs])
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
            numeric_cols=numeric_cols,  # Pass the numeric columns list
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
