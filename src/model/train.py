"""
Model training for AcciMap accident prediction.
Includes both XGBoost and deep learning models.
"""
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, brier_score_loss, 
    precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt

# Configuration constants
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.06,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
)

# Deep learning parameters
W = 168  # sequence window (hours)
BATCH_SIZE = 256
EPOCHS = 18
LR = 1e-3
BETA = 2.0  # F-beta for threshold picking

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from pathlib import Path


class XGBoostTrainer:
    """XGBoost model trainer."""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or XGB_PARAMS
        self.pipeline = None
        self.calibrator = None
        
    def prepare_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple:
        """Prepare features for XGBoost training."""
        # Define feature columns
        infra_cols = [
            "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
            "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"
        ]
        
        num_cols = ["lag_1h", "lag_3h", "lag_24h"]
        infra_cols_present = [c for c in infra_cols if c in train_df.columns]
        num_cols += infra_cols_present
        
        cat_cols = ["hour", "dow", "month", "is_weekend", "is_holiday"]
        cat_cols = [c for c in cat_cols if c in train_df.columns]
        
        def make_xyw(df):
            X = df[num_cols + cat_cols].copy()
            y = df["y"].astype(int).values
            w = df["weight"].astype(float).values
            return X, y, w
        
        X_tr, y_tr, w_tr = make_xyw(train_df)
        X_va, y_va, w_va = make_xyw(val_df)
        X_te, y_te, w_te = make_xyw(test_df)
        
        return (X_tr, y_tr, w_tr), (X_va, y_va, w_va), (X_te, y_te, w_te)
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """Train XGBoost model with calibration."""
        # Prepare features
        (X_tr, y_tr, w_tr), (X_va, y_va, w_va), (X_te, y_te, w_te) = self.prepare_features(train_df, val_df, test_df)
        
        # Create preprocessing pipeline
        cat_cols = ["hour", "dow", "month", "is_weekend", "is_holiday"]
        cat_cols = [c for c in cat_cols if c in train_df.columns]
        
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ], remainder="passthrough")
        
        # Create classifier
        clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=RANDOM_SEED,
            n_jobs=4,
            **self.params
        )
        
        # Create pipeline
        self.pipeline = Pipeline([("pre", pre), ("model", clf)])
        
        # Train
        self.pipeline.fit(X_tr, y_tr, model__sample_weight=w_tr)
        
        # Calibrate on validation set
        p_va_raw = self.pipeline.predict_proba(X_va)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(p_va_raw, y_va, sample_weight=w_va)
        
        # Evaluate on test set
        p_te_raw = self.pipeline.predict_proba(X_te)[:, 1]
        p_te_cal = self.calibrator.predict(p_te_raw)
        
        return {
            "test_predictions": p_te_cal,
            "test_labels": y_te,
            "test_weights": w_te,
            "raw_predictions": p_te_raw
        }
    
    def evaluate_model(self, results: Dict[str, Any], threshold: float = 0.01) -> None:
        """Evaluate XGBoost model results."""
        y_te = results["test_labels"]
        p_te_cal = results["test_predictions"]
        w_te = results["test_weights"]
        
        # Classification metrics
        y_pred = (p_te_cal >= threshold).astype(int)
        
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        cm = confusion_matrix(y_te, y_pred)
        
        print("\n--- XGBoost Classification Metrics ---")
        print(f"Threshold        : {threshold}")
        print(f"Accuracy         : {acc:.6f}")
        print(f"Precision        : {prec:.6f}")
        print(f"Recall           : {rec:.6f}")
        print(f"F1-score         : {f1:.6f}")
        print("Confusion Matrix :")
        print(pd.DataFrame(cm,
                          index=["Actual 0", "Actual 1"],
                          columns=["Pred 0", "Pred 1"]))
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_te, p_te_cal, sample_weight=w_te)
        print(f"ROC–AUC: {roc_auc:.6f}")
        
        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_te, p_te_cal, sample_weight=w_te)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title("XGBoost ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


class CatEmb(nn.Module):
    """Categorical embedding layer."""
    
    def __init__(self, cardinals: list, emb_dim: int = 8):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(c, min(emb_dim, max(2, c))) for c in cardinals
        ])
        self.out_dim = sum(e.embedding_dim for e in self.embs)
    
    def forward(self, Xc):
        if Xc is None:
            return None
        embs = [emb(Xc[:, :, i]) for i, emb in enumerate(self.embs)]
        return torch.cat(embs, dim=-1)


class RNNSeq(nn.Module):
    """RNN-based sequence model (LSTM/GRU)."""
    
    def __init__(self, kind: str, num_in: int, cat_card: list = None, 
                 static_in: int = 0, hid: int = 128, layers: int = 1):
        super().__init__()
        self.cat = CatEmb(cat_card) if cat_card else None
        in_dim = num_in + (self.cat.out_dim if self.cat else 0)
        
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = rnn(in_dim, hid, num_layers=layers, batch_first=True)
        
        head_in = hid + static_in
        self.head = nn.Sequential(
            nn.Linear(head_in, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.25),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, Xn, Xc=None, Xs=None):
        if self.cat:
            Ec = self.cat(Xc)
            x = torch.cat([Xn, Ec], dim=-1)
        else:
            x = Xn
        
        seq, _ = self.rnn(x)
        h = seq[:, -1, :]
        
        if Xs is not None:
            h = torch.cat([h, Xs], dim=1)
        
        logit = self.head(h).squeeze(1)
        return torch.sigmoid(logit)


class Chomp1d(nn.Module):
    """Chomp layer for TCN."""
    
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block."""
    
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1, p: float = 0.2):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d),
            Chomp1d(pad), nn.ReLU(), nn.BatchNorm1d(out_ch), nn.Dropout(p),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d),
            Chomp1d(pad), nn.ReLU(), nn.BatchNorm1d(out_ch), nn.Dropout(p),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        out = self.net(x)
        return out + self.down(x)


class TCNSeq(nn.Module):
    """Temporal Convolutional Network."""
    
    def __init__(self, num_in: int, cat_card: list = None, static_in: int = 0, 
                 hid: int = 128, levels: int = 4, k: int = 3):
        super().__init__()
        self.cat = CatEmb(cat_card) if cat_card else None
        in_dim = num_in + (self.cat.out_dim if self.cat else 0)
        
        chs = [in_dim] + [hid] * levels
        blocks = []
        for i in range(levels):
            blocks.append(TCNBlock(chs[i], chs[i + 1], k=k, d=2**i, p=0.2))
        
        self.tcn = nn.Sequential(*blocks)
        
        head_in = hid + static_in
        self.head = nn.Sequential(
            nn.Linear(head_in, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.25),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, Xn, Xc=None, Xs=None):
        if self.cat:
            Ec = self.cat(Xc)
            x = torch.cat([Xn, Ec], dim=-1)
        else:
            x = Xn
        
        x = x.transpose(1, 2)
        y = self.tcn(x)
        h = y[:, :, -1]
        
        if Xs is not None:
            h = torch.cat([h, Xs], dim=1)
        
        logit = self.head(h).squeeze(1)
        return torch.sigmoid(logit)


class CellSeqDataset(Dataset):
    """Dataset for sequence models."""
    
    def __init__(self, groups: dict, split: str, W: int, num_cols: list, 
                 cat_cols: list, static_cols: list):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.static_cols = static_cols
        self.W = W
        self.samples = []
        
        for h3_idx, g in groups.items():
            g = g[g["split"] == split]
            if len(g) < W:
                continue
            
            arr_num = g[num_cols].to_numpy(np.float32)
            arr_cat = g[cat_cols].to_numpy(np.int64) if cat_cols else None
            arr_static = g[static_cols].to_numpy(np.float32) if static_cols else None
            y = g["y"].astype("int64").to_numpy()
            
            for t in range(W - 1, len(g)):
                Xn = arr_num[t - W + 1:t + 1]
                Xc = arr_cat[t - W + 1:t + 1] if arr_cat is not None else None
                Xs = arr_static[t] if arr_static is not None else None
                self.samples.append((h3_idx, Xn, Xc, Xs, y[t]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        h3_idx, Xn, Xc, Xs, y = self.samples[i]
        return (
            torch.tensor(h3_idx, dtype=torch.long),
            torch.tensor(Xn, dtype=torch.float32),
            (torch.tensor(Xc, dtype=torch.long) if Xc is not None else torch.empty(0, dtype=torch.long)),
            (torch.tensor(Xs, dtype=torch.float32) if Xs is not None else torch.empty(0)),
            torch.tensor(y, dtype=torch.float32),
        )


def collate_fn(batch):
    """Collate function for DataLoader."""
    h_idx, Xn, Xc, Xs, y = zip(*batch)
    Xn = torch.stack(Xn)
    
    if Xc[0].numel() == 0:
        Xc = None
    else:
        Xc = torch.stack(Xc)
    
    if Xs[0].numel() == 0:
        Xs = None
    else:
        Xs = torch.stack(Xs)
    
    y = torch.stack(y)
    return (torch.tensor(h_idx, dtype=torch.long), Xn, Xc, Xs, y)


class DeepLearningTrainer:
    """Deep learning model trainer."""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_name = None
        
    def prepare_sequence_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                            test_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data for sequence models."""
        # Combine all data
        all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        # Define features
        num_feats = [c for c in [
            "lag_1h", "lag_3h", "lag_24h", "lag_7d_sum", "lag_30d_sum"
        ] if c in all_data.columns]
        
        cat_feats = ["hour", "dow", "month", "is_weekend", "is_holiday"]
        cat_feats = [c for c in cat_feats if c in all_data.columns]
        
        static_feats = [c for c in [
            "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
            "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"
        ] if c in all_data.columns]
        
        # Normalize numeric features
        train_mask = all_data["split"] == "train"
        num_stats = {}
        
        for c in num_feats:
            m = pd.to_numeric(all_data.loc[train_mask, c], errors="coerce").astype("float32")
            mu, sd = float(m.mean()), float(m.std() if m.std() > 1e-6 else 1.0)
            num_stats[c] = (mu, sd)
            all_data[c] = ((pd.to_numeric(all_data[c], errors="coerce").astype("float32") - mu) / sd).fillna(0.0)
        
        # Prepare categorical and static features
        for c in cat_feats:
            all_data[c] = pd.to_numeric(all_data[c], errors="coerce").fillna(0).astype("int64")
        for c in static_feats:
            all_data[c] = pd.to_numeric(all_data[c], errors="coerce").fillna(0).astype("float32")
        
        # Create groups
        all_data = all_data.sort_values(["h3_id", "ts_utc"]).reset_index(drop=True)
        h3_list = all_data["h3_id"].drop_duplicates().tolist()
        h3_to_idx = {h: i for i, h in enumerate(h3_list)}
        all_data["h3_idx"] = all_data["h3_id"].map(h3_to_idx).astype("int32")
        
        groups = {gid: g for gid, g in all_data.groupby("h3_idx", sort=True)}
        
        # Create datasets
        train_ds = CellSeqDataset(groups, "train", W, num_feats, cat_feats, static_feats)
        val_ds = CellSeqDataset(groups, "val", W, num_feats, cat_feats, static_feats)
        test_ds = CellSeqDataset(groups, "test", W, num_feats, cat_feats, static_feats)
        
        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, 
                                 shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, 
                               shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, 
                                shuffle=False, collate_fn=collate_fn)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader) -> Tuple[nn.Module, float]:
        """Train a single model."""
        model = model.to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.BCELoss()
        
        best_pr = -1
        best_state = None
        
        for epoch in range(1, EPOCHS + 1):
            # Training
            model.train()
            for _, Xn, Xc, Xs, y in train_loader:
                Xn, y = Xn.to(DEVICE), y.to(DEVICE)
                if Xc is not None:
                    Xc = Xc.to(DEVICE)
                if Xs is not None:
                    Xs = Xs.to(DEVICE)
                
                optimizer.zero_grad()
                pred = model(Xn, Xc, Xs)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for _, Xn, Xc, Xs, y in val_loader:
                    Xn = Xn.to(DEVICE)
                    if Xc is not None:
                        Xc = Xc.to(DEVICE)
                    if Xs is not None:
                        Xs = Xs.to(DEVICE)
                    
                    pred = model(Xn, Xc, Xs).detach().cpu().numpy()
                    val_preds.append(pred)
                    val_labels.append(y.numpy())
            
            val_preds = np.concatenate(val_preds) if val_preds else np.array([])
            val_labels = np.concatenate(val_labels) if val_labels else np.array([])
            
            pr_auc = average_precision_score(val_labels, val_preds) if (val_labels.size and len(np.unique(val_labels)) > 1) else 0.0
            
            if pr_auc > best_pr:
                best_pr = pr_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            print(f"[{model.__class__.__name__}] epoch {epoch:02d} val PR-AUC={pr_auc:.6f}")
        
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        return model, best_pr
    
    def train_all_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                        test_df: pd.DataFrame) -> Dict:
        """Train all deep learning models and select the best one."""
        train_loader, val_loader, test_loader = self.prepare_sequence_data(train_df, val_df, test_df)
        
        # Define models
        num_feats = [c for c in [
            "lag_1h", "lag_3h", "lag_24h", "lag_7d_sum", "lag_30d_sum"
        ] if c in train_df.columns]
        
        cat_feats = ["hour", "dow", "month", "is_weekend", "is_holiday"]
        cat_feats = [c for c in cat_feats if c in train_df.columns]
        
        static_feats = [c for c in [
            "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
            "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"
        ] if c in train_df.columns]
        
        cat_card = [24, 7, 12, 2, 2] if cat_feats else None
        c_static = len(static_feats)
        
        candidates = {
            "LSTM": RNNSeq(kind="lstm", num_in=len(num_feats), cat_card=cat_card, 
                          static_in=c_static, hid=160, layers=1),
            "GRU": RNNSeq(kind="gru", num_in=len(num_feats), cat_card=cat_card, 
                         static_in=c_static, hid=160, layers=1),
            "TCN": TCNSeq(num_in=len(num_feats), cat_card=cat_card, 
                         static_in=c_static, hid=192, levels=4, k=3),
        }
        
        results = {}
        best_val = -1
        
        for name, model in candidates.items():
            print(f"\n==== Train {name} ====")
            trained_model, val_pr = self.train_model(model, train_loader, val_loader)
            results[name] = val_pr
            self.models[name] = trained_model
            
            if val_pr > best_val:
                best_val = val_pr
                self.best_model = trained_model
                self.best_name = name
        
        print(f"\nValidation PR-AUC per model: {results}")
        print(f"→ Selected: {self.best_name} with PR-AUC {best_val:.6f}")
        
        # Evaluate on test set
        self.best_model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for _, Xn, Xc, Xs, y in test_loader:
                Xn = Xn.to(DEVICE)
                if Xc is not None:
                    Xc = Xc.to(DEVICE)
                if Xs is not None:
                    Xs = Xs.to(DEVICE)
                
                pred = self.best_model(Xn, Xc, Xs).detach().cpu().numpy()
                test_preds.append(pred)
                test_labels.append(y.numpy())
        
        test_preds = np.concatenate(test_preds) if test_preds else np.array([])
        test_labels = np.concatenate(test_labels) if test_labels else np.array([])
        
        return {
            "test_predictions": test_preds,
            "test_labels": test_labels,
            "best_model_name": self.best_name,
            "all_results": results
        }
    
    def evaluate_model(self, results: Dict[str, Any]) -> None:
        """Evaluate deep learning model results."""
        yt = results["test_labels"]
        pt = results["test_predictions"]
        
        # Find optimal threshold on validation
        self.best_model.eval()
        pv, yv = [], []
        
        # We need to get validation predictions again
        train_loader, val_loader, test_loader = self.prepare_sequence_data(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()  # Dummy data, we just need the loaders
        )
        
        with torch.no_grad():
            for _, Xn, Xc, Xs, y in val_loader:
                Xn = Xn.to(DEVICE)
                if Xc is not None:
                    Xc = Xc.to(DEVICE)
                if Xs is not None:
                    Xs = Xs.to(DEVICE)
                pv.append(self.best_model(Xn, Xc, Xs).detach().cpu().numpy())
                yv.append(y.numpy())
        
        pv = np.concatenate(pv) if pv else np.array([])
        yv = np.concatenate(yv) if yv else np.array([])
        
        prec, rec, thr = precision_recall_curve(yv, pv)
        fb = (1 + BETA**2) * prec * rec / (BETA**2 * prec + rec + 1e-12)
        best_t = float(thr[np.nanargmax(fb[:-1])]) if thr.size else 0.5
        
        print("Chosen threshold (val, Fβ):", best_t)
        
        # Test evaluation
        y_pred = (pt >= best_t).astype(int)
        print("Test precision:", round(float(precision_score(yt, y_pred, zero_division=0)), 6))
        print("Test recall   :", round(float(recall_score(yt, y_pred, zero_division=0)), 6))
        print("Test F1       :", round(float(f1_score(yt, y_pred, zero_division=0)), 6))
        print("Brier score   :", round(float(brier_score_loss(yt, np.clip(pt, 1e-6, 1-1e-6))), 6))
        print("Confusion:\n", pd.DataFrame(confusion_matrix(yt, y_pred),
                                         index=["Actual 0", "Actual 1"],
                                         columns=["Pred 0", "Pred 1"]))


def train_xgboost_model(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """Train XGBoost model."""
    trainer = XGBoostTrainer()
    results = trainer.train(train_df, val_df, test_df)
    trainer.evaluate_model(results)
    return results


def train_deep_learning_models(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """Train deep learning models."""
    trainer = DeepLearningTrainer()
    results = trainer.train_all_models(train_df, val_df, test_df)
    trainer.evaluate_model(results)
    return results


def load_processed_data_from_gcs():
    """Load processed training data from GCS."""
    from google.cloud import storage
    from pathlib import Path
    from tempfile import TemporaryDirectory
    
    BUCKET = "accimap-data"
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    
    # Look for processed parquet files
    processed_blobs = list(client.list_blobs(BUCKET, prefix="accidents/processed/"))
    
    if not processed_blobs:
        print("No processed data found. Please run the datapipeline first.")
        return None, None, None
    
    # Use the first processed file (you might want to modify this logic)
    blob = processed_blobs[0]
    print(f"Loading processed data from {blob.name}")
    
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "processed_data.parquet"
        blob.download_to_filename(str(tmp_path))
        
        # Load data
        data = pd.read_parquet(tmp_path)
        
        # Split back into train/val/test
        train_df = data[data["split"] == "train"].copy()
        val_df = data[data["split"] == "val"].copy()
        test_df = data[data["split"] == "test"].copy()
        
        print(f"Loaded data - Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
        
        return train_df, val_df, test_df


def main():
    """Main entry point for model training."""
    print("Model training service started!")
    
    # Load processed data from GCS
    train_df, val_df, test_df = load_processed_data_from_gcs()
    
    if train_df is None:
        print("No data available for training. Exiting.")
        return
    
    print("\n" + "="*50)
    print("TRAINING XGBOOST MODEL")
    print("="*50)
    
    # Train XGBoost model
    xgb_results = train_xgboost_model(train_df, val_df, test_df)
    
    print("\n" + "="*50)
    print("TRAINING DEEP LEARNING MODELS")
    print("="*50)
    
    # Train Deep Learning models
    #dl_results = train_deep_learning_models(train_df, val_df, test_df)
    
    #print("\n" + "="*50)
    #print("TRAINING COMPLETED")
    #print("="*50)
    #print("Both XGBoost and Deep Learning models have been trained and evaluated!")
    
    return xgb_results


if __name__ == "__main__":
    main()
