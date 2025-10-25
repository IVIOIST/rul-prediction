#%%
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns
#%%
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)
#%%
train_df = pd.read_csv('data/processed/train_FD001_processed.csv')
test_df = pd.read_csv('data/processed/test_FD001_processed.csv')
#%%
np.random.seed(42)
all_engine_ids = train_df['engine'].unique()
np.random.shuffle(all_engine_ids)
#%%
split_percentage = 0.8
split_point = int(len(all_engine_ids) * split_percentage)
train_subset_ids = all_engine_ids[:split_point]
validation_ids = all_engine_ids[split_point:]
#%%
train_subset_df =train_df[train_df['engine'].isin(train_subset_ids)]
val_subset_df = train_df[train_df['engine'].isin(validation_ids)]
#%%
def create_sliding_windows(df, lookback=30):
    X, y = [], []
    for engine_id in df["engine"].unique():
        engine_data = df[df["engine"] == engine_id].reset_index(drop=True)
        for i in range(len(engine_data) - lookback + 1):
            window = engine_data.iloc[i:i + lookback]
            X.append(window.drop(columns=["engine", "cycle", "RUL"]).values.flatten())
            y.append(window["RUL"].iloc[-1])
    return np.array(X), np.array(y)

def get_last_window_per_engine(df, lookback=30):
    X_test, y_test = [], []
    for engine_id in df["engine"].unique():
        engine_data = df[df["engine"] == engine_id].reset_index(drop=True)
        if len(engine_data) >= lookback:
            window = engine_data.iloc[-lookback:]
            X_test.append(window.drop(columns=["engine", "cycle", "RUL"]).values.flatten())
            y_test.append(window["RUL"].iloc[-1])
    return np.array(X_test), np.array(y_test)
#%%
LOOKBACK = 30
X_train, y_train = create_sliding_windows(train_subset_df, lookback=LOOKBACK)
X_val, y_val = create_sliding_windows(val_subset_df, lookback=LOOKBACK)
X_test, y_test = get_last_window_per_engine(test_df, lookback=LOOKBACK)

print("Data shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}\n")
#%%
lgbm = lgb.LGBMRegressor(
    n_estimators=250,
    max_depth=15,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

lgbm.fit(X_train, y_train)
#%% md
# ### Validation Set Performance
#%%
y_pred_val = lgbm.predict(X_val)
#%%
mae_val = mean_absolute_error(y_val, y_pred_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

print("--- Validation Set Performance ---")
print(f"Validation MAE: {mae_val:.2f}")
print(f"Validation RMSE: {rmse_val:.2f}\n")

print("--- Detailed Validation Predictions ---")
# true RUL is 0 since we are predicting the last cycle
for i, (true, pred) in enumerate(zip(y_val, y_pred_val)):
    print(f"Engine {i+1:03d} => True RUL = {true}, Predicted = {pred:.1f}")
#%%
plt.subplot(1, 2, 1)
plt.scatter(y_val, y_pred_val, alpha=0.6, edgecolors='k')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--', lw=2)
plt.title('True RUL vs. Predicted RUL on Validation Set', fontsize=14)
plt.xlabel('True RUL (Cycles)', fontsize=12)
plt.ylabel('Predicted RUL (Cycles)', fontsize=12)
plt.xlim(0)
plt.ylim(0)
plt.grid(True)

errors = y_val - y_pred_val
plt.subplot(1, 2, 2)
sns.histplot(errors, bins=30, kde=True)
plt.title('Distribution of Prediction Errors (True - Predicted)', fontsize=14)
plt.xlabel('Prediction Error (Cycles)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(x=0, color='red', linestyle='--', lw=2)
plt.grid(True)
#%% md
# ### Test Set Performance
#%%
y_pred_test = lgbm.predict(X_test)
#%%
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
#%%
print("--- Test Set Performance ---")
print(f"Test MAE: {mae_test:.2f}")
print(f"Test RMSE: {rmse_test:.2f}\n")

print("--- Detailed Test Predictions ---")
for i, (true, pred) in enumerate(zip(y_test, y_pred_test)):
    print(f"Engine {i+1:03d} => True RUL = {true}, Predicted = {pred:.1f}")
#%%
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
plt.title('True RUL vs. Predicted RUL on Test Set', fontsize=14)
plt.xlabel('True RUL (Cycles)', fontsize=12)
plt.ylabel('Predicted RUL (Cycles)', fontsize=12)
plt.xlim(0)
plt.ylim(0)
plt.grid(True)

errors = y_test - y_pred_test
plt.subplot(1, 2, 2)
sns.histplot(errors, bins=30, kde=True)
plt.title('Distribution of Prediction Errors (True - Predicted)', fontsize=14)
plt.xlabel('Prediction Error (Cycles)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(x=0, color='red', linestyle='--', lw=2)
plt.grid(True)