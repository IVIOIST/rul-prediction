#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
import math
import copy
from sklearn.utils import resample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
#%%
train_df = pd.read_csv('data/processed/train_FD001_processed.csv')
test_df = pd.read_csv('data/processed/test_FD001_processed.csv')

print(train_df.head())
#%%
index_cols = ["engine", "cycle"]
setting_cols = ["setting1", "setting2", "setting3"]

sensor_cols = [
    "Fan Inlet Temperature (°K)",
    "LPC Outlet Temperature (°K)",
    "HPC Outlet Temperature (°K)",
    "LPT Outlet Temperature (°K)",
    "Fan Inlet Pressure (kPa)",
    "Bypass-Duct Pressure (kPa)",
    "HPC Outlet Pressure (kPa)",
    "Physical Fan Speed (rpm)",
    "Physical Core Speed (rpm)",
    "Engine Pressure Ratio (P50/P2)",
    "HPC Outlet Static Pressure (kPa)",
    "Ratio of Fuel Flow to Ps30 (m²/s)",
    "Corrected Fan Speed (rpm)",
    "Corrected Core Speed (rpm)",
    "Bypass Ratio",
    "Burner Fuel-Air Ratio",
    "Bleed Enthalpy",
    "Required Fan Speed",
    "Required Fan Conversion Speed",
    "High-Pressure Turbines Cool Air Flow",
    "Low-Pressure Turbines Cool Air Flow"
]
#%%
window_size = 5

for sensor in sensor_cols:
    train_df[f'{sensor}_rolling_mean'] = (
        train_df.groupby('engine')[sensor].transform(lambda x: x.rolling(window_size, min_periods=1).mean())
    )
    train_df[f'{sensor}_rolling_std'] = (
        train_df.groupby('engine')[sensor].transform(lambda x: x.rolling(window_size, min_periods=1).std())
    )

    test_df[f'{sensor}_rolling_mean'] = (
        test_df.groupby('engine')[sensor].transform(lambda x: x.rolling(window_size, min_periods=1).mean())
    )
    test_df[f'{sensor}_rolling_std'] = (
        test_df.groupby('engine')[sensor].transform(lambda x: x.rolling(window_size, min_periods=1).std())
    )

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

#%%
# Use ONLY raw sensor columns for baseline
feature_cols_initial = sensor_cols

train_baseline = (
    train_df.groupby('engine')[feature_cols_initial]
    .head(5)
    .groupby(train_df['engine'])
    .mean()
)
train_baseline.columns = [c + "_baseline" for c in train_baseline.columns]

train_df = train_df.join(train_baseline, on='engine', how='left')

test_baseline = (
    test_df.groupby('engine')[feature_cols_initial]
    .head(5)
    .groupby(test_df['engine'])
    .mean()
)
test_baseline.columns = [c + "_baseline" for c in test_baseline.columns]

test_df = test_df.join(test_baseline, on='engine', how='left')

# Now create delta features correctly
for c in feature_cols_initial:
    train_df[c + "_delta"] = train_df[c] - train_df[c + "_baseline"]
    test_df[c + "_delta"]  = test_df[c] - test_df[c + "_baseline"]

#%%
np.random.seed(42)
all_engines = train_df['engine'].unique()
np.random.shuffle(all_engines)
split = int(len(all_engines) * 0.8)

train_engines = all_engines[:split]
val_engines   = all_engines[split:]

train_subset_df = train_df[train_df['engine'].isin(train_engines)]
val_subset_df   = train_df[train_df['engine'].isin(val_engines)]

feature_cols = [c for c in train_df.columns if c not in ['engine','cycle','RUL']]

scaler = StandardScaler()
scaler.fit(train_subset_df[feature_cols])

train_subset_df[feature_cols] = scaler.transform(train_subset_df[feature_cols])
val_subset_df[feature_cols]   = scaler.transform(val_subset_df[feature_cols])
test_df[feature_cols]         = scaler.transform(test_df[feature_cols])
#%%
LOOKBACK = 50

def create_windows(df, lookback, features):
    X, y = [], []
    for eng in df['engine'].unique():
        d = df[df.engine == eng]
        for i in range(len(d)-lookback+1):
            X.append(d.iloc[i:i+lookback][features].values)
            y.append(d.iloc[i+lookback-1].RUL)
    return np.array(X), np.array(y).reshape(-1,1)

def get_test_windows(df, lookback, features):
    X, y = [], []
    for eng in df.engine.unique():
        d = df[df.engine == eng]
        if len(d) >= lookback:
            X.append(d.iloc[-lookback:][features].values)
            y.append(d.iloc[-1].RUL)
    return np.array(X), np.array(y).reshape(-1,1)

X_train_np, y_train_np = create_windows(train_subset_df, LOOKBACK, feature_cols)
X_val_np,   y_val_np   = create_windows(val_subset_df, LOOKBACK, feature_cols)
X_test_np,  y_test_np  = get_test_windows(test_df, LOOKBACK, feature_cols)
#%%
def rebalance_by_rul_bins(X, y, bin_width=20, target_per_bin=2000):
    Xb, yb = [], []
    bins = np.arange(0, 400, bin_width)
    ids = np.digitize(y.flatten(), bins)
    for b in range(1, len(bins)):
        idx = np.where(ids == b)[0]
        if len(idx) == 0:
            continue
        if len(idx) > target_per_bin:
            idx = resample(idx, replace=False, n_samples=target_per_bin, random_state=42)
        else:
            idx = resample(idx, replace=True, n_samples=target_per_bin, random_state=42)
        Xb.append(X[idx])
        yb.append(y[idx])
    return np.concatenate(Xb), np.concatenate(yb)

X_train_np_bal, y_train_np_bal = rebalance_by_rul_bins(X_train_np, y_train_np)
#%%
X_train = torch.from_numpy(X_train_np_bal).float()
y_train = torch.from_numpy(y_train_np_bal).float()
X_val   = torch.from_numpy(X_val_np).float()
y_val   = torch.from_numpy(y_val_np).float()
X_test  = torch.from_numpy(X_test_np).float()
y_test  = torch.from_numpy(y_test_np).float()

batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
#%%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class CNNTransformerRUL(nn.Module):
    def __init__(self, input_dim, d_model, nhead, d_hid, nlayers, dropout):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pos = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, nlayers)
        self.post_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 1))

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.cnn(x)
        x = x.permute(0,2,1)
        x = self.pos(x)
        x = self.transformer(x)
        x = self.post_norm(x)
        x = torch.mean(x, dim=1)
        return self.head(x)

model = CNNTransformerRUL(X_train.shape[2], 128, 8, 256, 4, 0.2).to(device)
#%%
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5)

best_val = float('inf')
patience = 15
no_improve = 0
history = {'train':[], 'val':[]}

for epoch in range(1,150):
    model.train()
    total = 0
    for Xb,yb in train_loader:
        Xb,yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        total += loss.item()
    train_loss = total/len(train_loader)

    model.eval()
    total=0
    with torch.no_grad():
        for Xb,yb in val_loader:
            Xb,yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            total += criterion(pred,yb).item()
    val_loss = total/len(val_loader)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    print(f"Epoch {epoch:03d} | Train {train_loss:.2f} | Val {val_loss:.2f}")

    scheduler.step(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        best = copy.deepcopy(model.state_dict())
        no_improve = 0
        print("✅ Saving Best Model")
    else:
        no_improve += 1

    if no_improve >= patience:
        print("❌ Early Stopping")
        break

model.load_state_dict(best)
#%%
model.eval()
preds=[]
with torch.no_grad():
    for X,_ in val_loader:
        preds.append(model(X.to(device)).cpu().numpy())
y_val_pred = np.concatenate(preds).flatten()
y_val_true = y_val_np.flatten()

mask = y_val_true >= 150
slope,_ = np.polyfit(y_val_true[mask], y_val_pred[mask],1)
print("Slope ≥150:", round(slope,3))
#%%
model.eval()
y_pred_list = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)
        output = model(batch_X)
        y_pred_list.append(output.cpu().numpy())

y_pred_test = np.concatenate(y_pred_list, axis=0).flatten()
y_test_np_flat = y_test_np.flatten()

# Calculate final metrics on the test set
mae_test = mean_absolute_error(y_test_np_flat, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test_np_flat, y_pred_test))

print("\n--- Test Set Performance ---")
print(f"Test MAE: {mae_test:.2f}")
print(f"Test RMSE: {rmse_test:.2f}\n")
#%%
plt.figure(figsize=(18, 8))

# 1. True RUL vs. Predicted RUL
plt.subplot(1, 2, 1)
plt.scatter(y_test_np_flat, y_pred_test, alpha=0.6, edgecolors='k')
plt.plot([min(y_test_np_flat), max(y_test_np_flat)], [min(y_test_np_flat), max(y_test_np_flat)], color='red', linestyle='--', lw=2, label='Ideal Line')
plt.title('True RUL vs. Predicted RUL on Test Set', fontsize=14)
plt.xlabel('True RUL (Cycles)', fontsize=12)
plt.ylabel('Predicted RUL (Cycles)', fontsize=12)
plt.grid(True)
plt.legend()

# 2. Distribution of Prediction Errors
errors = y_test_np_flat - y_pred_test
plt.subplot(1, 2, 2)
sns.histplot(errors, bins=30, kde=True)
plt.title('Distribution of Prediction Errors (True - Predicted)', fontsize=14)
plt.xlabel('Prediction Error (Cycles)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(x=0, color='red', linestyle='--', lw=2)
plt.grid(True)

plt.tight_layout()
plt.show()
#%%
model.eval()
y_val_pred_list = []
with torch.no_grad():
    for batch_X, _ in val_loader:
        batch_X = batch_X.to(device)
        output = model(batch_X)
        y_val_pred_list.append(output.cpu().numpy())
y_val_pred = np.concatenate(y_val_pred_list, axis=0).flatten()
y_val_np_flat = y_val_np.flatten()
mae_val = mean_absolute_error(y_val_np_flat, y_val_pred)
rmse_val = np.sqrt(mean_squared_error(y_val_np_flat, y_val_pred))
print("\n--- Validation Set Performance ---")
print(f"Validation MAE: {mae_val:.2f}")
print(f"Validation RMSE: {rmse_val:.2f}\n")
#%%
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_val_np_flat, y_val_pred, alpha=0.6, edgecolors='k')
plt.plot([min(y_val_np_flat), max(y_val_np_flat)], [min(y_val_np_flat), max(y_val_np_flat)], color='red',
         linestyle='--', lw=2)
plt.title('True RUL vs. Predicted RUL on Validation Set', fontsize=14)
plt.xlabel('True RUL (Cycles)', fontsize=12)
plt.ylabel('Predicted RUL (Cycles)', fontsize=12)
plt.grid(True)
errors_val = y_val_np_flat - y_val_pred
plt.subplot(1, 2, 2)
sns.histplot(errors_val, bins=30, kde=True)
plt.title('Distribution of Prediction Errors (True - Predicted) on Validation Set', fontsize=14)
plt.xlabel('Prediction Error (Cycles)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(x=0, color='red', linestyle='--', lw=2)
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
val_df = val_subset_df.copy()
val_df["pred"] = np.nan

index = 0
for eng in val_df.engine.unique():
    d = val_df[val_df.engine == eng]
    seq_len = len(d)
    if seq_len >= LOOKBACK:
        val_df.loc[d.index[LOOKBACK-1:], "pred"] = y_val_pred[index:index+(seq_len-LOOKBACK+1)]
        index += (seq_len - LOOKBACK + 1)

plt.figure(figsize=(7,7))
for eng in val_df.engine.unique():
    d = val_df[val_df.engine == eng]
    plt.plot(d["RUL"].values, d["pred"].values, alpha=0.6)

plt.plot([0, 320], [0, 320], 'r--', label="Ideal")
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title("True vs Predicted RUL (Trajectory by Engine)")
plt.legend()
plt.grid(True)
plt.show()