"""
TradeMaster Cup 2025 - Single Asset Solution (Updated v2)
Based on EDA: 
- Single Stock ID (0) -> Removed Cross-Sectional features
- Feature 27 Dropped
- Feature 11 treated as Price Level -> Diff
- Feature 4 treated as Volume/Vol -> Interaction with F11
- Categorical Embeddings for F1, F8, F12
"""

import os
import gc
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
class Config:
    seed = 2025
    device = torch.device('cpu')  # Force CPU due to CUDA issues
    
    # Paths
    train_path = 'trademaster25/train_v2.csv'
    test_path = 'trademaster25/test_v2.csv'
    submission_path = 'submission.csv'
    
    # Feature Engineering Config
    drop_cols = ['feature_27']  # Explicitly drop F27
    cat_cols = ['feature_1', 'feature_8', 'feature_12'] # Discrete features
    
    # Categorical Stats (Unique values count based on EDA)
    # F1: 2, F8: 5, F12: 5. We add +1 for padding/unknown handling.
    cat_dims = {
        'feature_1': 3,   
        'feature_8': 6,
        'feature_12': 6
    }
    
    # Model Architecture
    emb_dim = 4             # Dimension for categorical embeddings
    hidden_dim = 512
    dropout_rate = 0.2
    num_layers = 4
    
    # Training
    epochs = 20
    batch_size = 4096       # Increased batch size for single GPU efficiency
    learning_rate = 1e-3
    weight_decay = 1e-4
    early_stopping = 6
    
    # WMAE Weights
    w_short = 0.5
    w_medium = 0.3
    w_long = 0.2

def seed_everything(seed):
    random.seed(seed)
    os.environ = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(Config.seed)

# ==========================================
# 2. 特征工程 (Specific to User Analysis)
# ==========================================

def perform_feature_engineering(df):
    """
    Implements specific logic:
    1. Drop F27
    2. F11 Diff
    3. F11/F4 Interaction
    4. Handle F6 (Time)
    """
    # 1. Drop Feature 27
    if 'feature_27' in df.columns:
        df = df.drop(columns=['feature_27'])
    
    # 2. Feature 11 Diff (Price -> Return)
    # Group by date_id to prevent diffing across days (last min of day T vs first min of day T+1)
    # FillNA with 0 for the first minute of the day
    df['f11_diff'] = df.groupby('date_id')['feature_11'].diff().fillna(0).astype(np.float32)
    
    # 3. Interaction: F11_diff / F4 (Proxy for Price Impact / Amihud)
    # Add epsilon to avoid division by zero
    df['vol_proxy'] = df['f11_diff'] / (df['feature_4'] + 1e-6)
    
    # 4. Feature 6 (Time of Day)
    # Normalize to  for Neural Net stability
    # Assuming F6 ranges 0-239 roughly
    if 'feature_6' in df.columns:
        df['f6_norm'] = (df['feature_6'] - df['feature_6'].min()) / \
                        (df['feature_6'].max() - df['feature_6'].min() + 1e-6)
    
    # 5. Handle Categorical Columns for Embedding
    # Ensure they are integers and start from 0
    # Map raw values to 0..N indices if necessary. Here we assume they are already low-cardinality floats/ints.
    # We round them to be safe (in case they are 1.0, 2.0)
    for col in Config.cat_cols:
        # Simple label encoding logic if values aren't 0,1,2...
        # For this competition, if values are 0,1 and -0.2 etc, we might need robust mapping.
        # Given EDA says "exact 5 unique values", let's map them dynamically.
        unique_vals = sorted(df[col].unique())
        val_map = {v: i for i, v in enumerate(unique_vals)}
        df[f'{col}_cat'] = df[col].map(val_map).fillna(0).astype(int)
    
    return df, val_map  # Return map to apply to test set

# ==========================================
# 3. 数据集与加载器
# ==========================================

class TradeDataset(Dataset):
    def __init__(self, df, num_cols, cat_cols, targets=None, mode='train'):
        self.mode = mode
        self.num_data = df[num_cols].values.astype(np.float32)
        self.cat_data = df[cat_cols].values.astype(np.int64)
        
        if mode!= 'test':
            self.targets = targets.astype(np.float32)
        else:
            self.ids = df['id'].values
            
    def __len__(self):
        return len(self.num_data)
    
    def __getitem__(self, idx):
        num_x = torch.tensor(self.num_data[idx])
        cat_x = torch.tensor(self.cat_data[idx])
        
        if self.mode!= 'test':
            y = torch.tensor(self.targets[idx])
            return num_x, cat_x, y
        else:
            row_id = self.ids[idx]
            return num_x, cat_x, row_id

# ==========================================
# 4. 模型定义 (ResNet with Embeddings)
# ==========================================

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.block(x)

class TradeNet(nn.Module):
    def __init__(self, num_feat_dim, cat_dims, emb_dim, hidden_dim, dropout, num_layers):
        super().__init__()
        
        # Define Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dims, embedding_dim=emb_dim)
            for col, dims in cat_dims.items()
        ])
        
        # Calculate total input dimension
        # Continuous features + (number of cat features * emb_dim)
        total_input_dim = num_feat_dim + (len(cat_dims) * emb_dim)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        
        # ResNet Backbone
        self.backbone = nn.ModuleList()
        
        # Multi-Heads for 3 Targets
        self.head_short = nn.Linear(hidden_dim, 1)
        self.head_medium = nn.Linear(hidden_dim, 1)
        self.head_long = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_num, x_cat):
        # x_cat shape:
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_list.append(emb_layer(x_cat[:, i]))
        
        x_emb = torch.cat(emb_list, dim=1)
        x = torch.cat([x_num, x_emb], dim=1)
        
        x = self.encoder(x)
        
        for layer in self.backbone:
            x = layer(x)
            
        return self.head_short(x), self.head_medium(x), self.head_long(x)

# ==========================================
# 5. 训练与验证流程
# ==========================================

def train_and_predict():
    print(">>> Loading Data...")
    df = pd.read_csv(Config.train_path)
    
    # ---------------------------
    # Feature Engineering (Train)
    # ---------------------------
    print(">>> Processing Features...")
    df, cat_maps = perform_feature_engineering(df)
    
    # Define Column Groups
    exclude_cols = ['id', 'date_id', 'minute_id', 'stock_id', 
                    'target_short', 'target_medium', 'target_long'] + \
                   Config.cat_cols + [f'{c}_cat' for c in Config.cat_cols]
    
    num_cols = [c for c in df.columns if c not in exclude_cols]
    cat_cols_final = [f'{c}_cat' for c in Config.cat_cols]
    
    print(f"Num Features: {len(num_cols)} | Cat Features: {len(cat_cols_final)}")
    
    # Normalize Continuous Features
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # ---------------------------
    # Time Series Split
    # ---------------------------
    dates = df['date_id'].unique()
    split_idx = int(len(dates) * 0.9)
    train_dates = dates[:split_idx]
    valid_dates = dates[split_idx:]
    
    train_df = df[df['date_id'].isin(train_dates)].reset_index(drop=True)
    valid_df = df[df['date_id'].isin(valid_dates)].reset_index(drop=True)
    
    # Targets
    target_cols = ['target_short', 'target_medium', 'target_long']
    y_train = train_df[target_cols].values
    y_valid = valid_df[target_cols].values
    
    # Datasets
    train_ds = TradeDataset(train_df, num_cols, cat_cols_final, y_train, mode='train')
    valid_ds = TradeDataset(valid_df, num_cols, cat_cols_final, y_valid, mode='train')
    
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=Config.batch_size, shuffle=False, num_workers=4)
    
    # ---------------------------
    # Model Initialization
    # ---------------------------
    model = TradeNet(
        num_feat_dim=len(num_cols),
        cat_dims=Config.cat_dims,
        emb_dim=Config.emb_dim,
        hidden_dim=Config.hidden_dim,
        dropout=Config.dropout_rate,
        num_layers=Config.num_layers
    ).to(Config.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=Config.learning_rate, 
                           steps_per_epoch=len(train_loader), epochs=Config.epochs)
    scaler_amp = GradScaler()
    
    # Custom WMAE Loss
    def wmae_loss(pred_s, pred_m, pred_l, targets):
        loss_s = torch.abs(pred_s - targets[:, 0:1]).mean()
        loss_m = torch.abs(pred_m - targets[:, 1:2]).mean()
        loss_l = torch.abs(pred_l - targets[:, 2:3]).mean()
        return 0.5*loss_s + 0.3*loss_m + 0.2*loss_l
    
    # ---------------------------
    # Training Loop
    # ---------------------------
    print(">>> Starting Training...")
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(Config.epochs):
        model.train()
        t_loss = 0
        
        for num_x, cat_x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            num_x, cat_x, y = num_x.to(Config.device), cat_x.to(Config.device), y.to(Config.device)
            
            optimizer.zero_grad()
            with autocast():
                ps, pm, pl = model(num_x, cat_x)
                loss = wmae_loss(ps, pm, pl, y)
            
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()
            
            t_loss += loss.item()
            
        avg_t_loss = t_loss / len(train_loader)
        
        # Validation
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for num_x, cat_x, y in valid_loader:
                num_x, cat_x, y = num_x.to(Config.device), cat_x.to(Config.device), y.to(Config.device)
                ps, pm, pl = model(num_x, cat_x)
                loss = wmae_loss(ps, pm, pl, y)
                v_loss += loss.item()
                
        avg_v_loss = v_loss / len(valid_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_t_loss:.5f} | Valid Loss: {avg_v_loss:.5f}")
        
        if avg_v_loss < best_loss:
            best_loss = avg_v_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience = 0
        else:
            patience += 1
            if patience >= Config.early_stopping:
                print("Early stopping triggered.")
                break
                
    # ---------------------------
    # Inference / Submission
    # ---------------------------
    if os.path.exists(Config.test_path):
        print(">>> Generating Submission...")
        test_df = pd.read_csv(Config.test_path)
        
        # Apply same FE logic
        # Note: Apply val_map logic for categoricals
        # 1. Drop F27
        if 'feature_27' in test_df.columns:
            test_df = test_df.drop(columns=['feature_27'])
            
        # 2. F11 Diff (Need care here for test set structure)
        # Assuming test set is chronological continuation
        # For simple submission script, we might lose first diff point or fill 0
        test_df['f11_diff'] = test_df.groupby('date_id')['feature_11'].diff().fillna(0).astype(np.float32)
        test_df['vol_proxy'] = test_df['f11_diff'] / (test_df['feature_4'] + 1e-6)
        if 'feature_6' in test_df.columns:
            test_df['f6_norm'] = (test_df['feature_6'] - test_df['feature_6'].min()) / \
                                 (test_df['feature_6'].max() - test_df['feature_6'].min() + 1e-6)

        # Map categoricals using saved maps
        for col in Config.cat_cols:
            if col in cat_maps:
                # Use map, fill unknown with 0
                test_df[f'{col}_cat'] = test_df[col].map(cat_maps[col]).fillna(0).astype(int)
            else:
                 test_df[f'{col}_cat'] = 0
                 
        # Normalize
        test_df[num_cols] = scaler.transform(test_df[num_cols])
        
        test_ds = TradeDataset(test_df, num_cols, cat_cols_final, mode='test')
        test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False, num_workers=4)
        
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        
        ids, p_s, p_m, p_l = [], [], [], []
        with torch.no_grad():
            for num_x, cat_x, row_id in tqdm(test_loader, desc="Predicting"):
                num_x, cat_x = num_x.to(Config.device), cat_x.to(Config.device)
                ps, pm, pl = model(num_x, cat_x)
                
                ids.extend(row_id.numpy())
                p_s.extend(ps.cpu().numpy().flatten())
                p_m.extend(pm.cpu().numpy().flatten())
                p_l.extend(pl.cpu().numpy().flatten())
                
        sub = pd.DataFrame({
            'id': ids,
            'target_short': p_s,
            'target_medium': p_m,
            'target_long': p_l
        })
        sub.to_csv(Config.submission_path, index=False)
        print("Done!")

if __name__ == "__main__":
    train_and_predict()