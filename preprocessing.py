#%% md
# # Please Don't Change Anything in this file unless you know what you're doing
#%%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
#%% md
# # training data preporcessing
#%%
df = pd.read_csv('data/raw/train_FD001.txt', sep=" ", header=None)
#%%
columns = [
    # Index Names
    "engine",   # Engine No.
    "cycle",    # Time, In Cycles

    # Setting Names
    "setting1",  # Operation Setting 1
    "setting2",  # Operation Setting 2
    "setting3",  # Operation Setting 3

    # Sensor Names (full names)
    "Fan Inlet Temperature (°K)", #1
    "LPC Outlet Temperature (°K)", #2
    "HPC Outlet Temperature (°K)", #3
    "LPT Outlet Temperature (°K)", #4
    "Fan Inlet Pressure (kPa)", #5
    "Bypass-Duct Pressure (kPa)", #6
    "HPC Outlet Pressure (kPa)", #7
    "Physical Fan Speed (rpm)", #8
    "Physical Core Speed (rpm)", #9
    "Engine Pressure Ratio (P50/P2)", #10
    "HPC Outlet Static Pressure (kPa)", #11
    "Ratio of Fuel Flow to Ps30 (m²/s)", #12
    "Corrected Fan Speed (rpm)", #13
    "Corrected Core Speed (rpm)", #14
    "Bypass Ratio", #15
    "Burner Fuel-Air Ratio", #16
    "Bleed Enthalpy", #17
    "Required Fan Speed", #18
    "Required Fan Conversion Speed", #19
    "High-Pressure Turbines Cool Air Flow", #20
    "Low-Pressure Turbines Cool Air Flow" #21
]

#%%
## Changing to SI units
df.iloc[:, 5] = df.iloc[:, 5] * (5/9)   # Fan Inlet Temperature Rankine to Kelvin
df.iloc[:, 6] = df.iloc[:, 6] * (5/9)   # LPC Outlet Temp Rankine to Kelvin
df.iloc[:, 7] = df.iloc[:, 7] * (5/9)   # HPC Outlet Temp Rankine to Kelvin
df.iloc[:, 8] = df.iloc[:, 8] * (5/9)   # LPT Outlet Temp Rankine to Kelvin
df.iloc[:, 9] = df.iloc[:, 9] * 6.89476     # Fan Inlet Pressure psia to kPa
df.iloc[:, 10] = df.iloc[:, 10] * 6.89476     # Bypass-Duct Pressure psia to kPa
df.iloc[:, 11] = df.iloc[:, 11] * 6.89476     # HPC Outlet Pressure psia to kPa
df.iloc[:, 15] = df.iloc[:, 15] * 6.89476     # HPC Outlet Static Pressure psia to kPa
df.iloc[:, 16] = df.iloc[:, 16] * 0.00064516  # Mass flow rate to LPC outlet converted from pps/psia to m²/s

#%%
# df.head()
#%%
# drop the last two columns which are empty
df = df.drop(columns=[26, 27])
df.columns = columns
#%%
# df.head()
#%%
# df.columns = columns
#%%
df.head()
#%%
# plot all cycles for engine 1 for sensor "Fan Inlet Temperature (°R)", "Fan Inlet Pressure (psia)", "Physical Fan Speed (rpm)" in 3 subplots
engine_1 = df[df['engine'] == 7]

fig, axs = plt.subplots(3, 1, figsize=(10, 10))
axs[0].plot(engine_1['cycle'], engine_1['Fan Inlet Temperature (°K)'])
axs[0].set_title('Fan Inlet Temperature (°K)')
axs[0].set_xlabel('Cycle')
axs[0].set_ylabel('Fan Inlet Temperature (°K)')
axs[1].plot(engine_1['cycle'], engine_1['Fan Inlet Pressure (kPa)'])
axs[1].set_title('Fan Inlet Pressure (kPa)')
axs[1].set_xlabel('Cycle')
axs[1].set_ylabel('Fan Inlet Pressure (kPa)')
axs[2].plot(engine_1['cycle'], engine_1['Physical Fan Speed (rpm)'])
axs[2].set_title('Physical Fan Speed (rpm)')
axs[2].set_xlabel('Cycle')
axs[2].set_ylabel('Physical Fan Speed (rpm)')
plt.tight_layout()
plt.show()
#%%
## Second lot of plotting to determine relationships
f, ax = plt.subplots(5,2, figsize = (10,15))
ax = ax.flatten()
ax[0].plot(engine_1['cycle'], engine_1['LPC Outlet Temperature (°K)'])
ax[0].set_title('LPC Outlet Temperature (°K)')
ax[0].set_xlabel('Cycle')
ax[0].set_ylabel('LPC Outlet Temperature (°K)')

ax[1].plot(engine_1['cycle'], engine_1['HPC Outlet Temperature (°K)'])
ax[1].set_title('HPC Outlet Temperature (°K)')
ax[1].set_xlabel('Cycle')
ax[1].set_ylabel('HPC Outlet Temperature (°K)')

ax[2].plot(engine_1['cycle'], engine_1['LPT Outlet Temperature (°K)'])
ax[2].set_title('LPT Outlet Temperature (°K)')
ax[2].set_xlabel('Cycle')
ax[2].set_ylabel('LPT Outlet Temperature (°K)')

ax[3].plot(engine_1['cycle'], engine_1['Bypass-Duct Pressure (kPa)'])
ax[3].set_title('Bypass-Duct Pressure (kPa)')
ax[3].set_xlabel('Cycle')
ax[3].set_ylabel('Bypass-Duct Pressure (kPa)')

ax[4].plot(engine_1['cycle'], engine_1['HPC Outlet Pressure (kPa)'])
ax[4].set_title('HPC Outlet Pressure (kPa)')
ax[4].set_xlabel('Cycle')
ax[4].set_ylabel('HPC Outlet Pressure (kPa)')

ax[5].plot(engine_1['cycle'], engine_1['Physical Core Speed (rpm)'])
ax[5].set_title('Physical Core Speed (rpm)')
ax[5].set_xlabel('Cycle')
ax[5].set_ylabel('Physical Core Speed (rpm)')

ax[6].plot(engine_1['cycle'], engine_1['Engine Pressure Ratio (P50/P2)'])
ax[6].set_title('Engine Pressure Ratio (P50/P2)')
ax[6].set_xlabel('Cycle')
ax[6].set_ylabel('Engine Pressure Ratio (P50/P2)')

ax[7].plot(engine_1['cycle'], engine_1['Ratio of Fuel Flow to Ps30 (m²/s)'])
ax[7].set_title('Ratio of Fuel Flow to Ps30 (m²/s)')
ax[7].set_xlabel('Cycle')
ax[7].set_ylabel('Ratio of Fuel Flow to Ps30 (m²/s)')

ax[8].plot(engine_1['cycle'], engine_1['Burner Fuel-Air Ratio'])
ax[8].set_title('Burner Fuel-Air Ratio')
ax[8].set_xlabel('Cycle')
ax[8].set_ylabel('Burner Fuel-Air Ratio')

ax[9].plot(engine_1['cycle'], engine_1['Required Fan Speed'])
ax[9].set_title('Required Fan Conversion Speed')
ax[9].set_xlabel('Cycle')
ax[9].set_ylabel('Required Fan Speed')

plt.tight_layout()
plt.show()

#%%
# add rul column to df
for engine in df['engine'].unique():
    engine_df = df[df['engine'] == engine]
    max_cycle = engine_df['cycle'].max()
    df.loc[df['engine'] == engine, 'RUL'] = max_cycle - engine_df['cycle']
#%%
df
#%%
# export to csv
df.to_csv('data/processed/train_FD001_processed.csv', index=False)
#%% md
# # testing data preprocessing
#%%
df = pd.read_csv('data/raw/test_FD001.txt', sep=" ", header=None)
#%%
columns = [
    # Index Names
    "engine",   # Engine No.
    "cycle",    # Time, In Cycles

    # Setting Names
    "setting1",  # Operation Setting 1
    "setting2",  # Operation Setting 2
    "setting3",  # Operation Setting 3

    # Sensor Names (full names)
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
## Changing to SI units
df.iloc[:, 5] = df.iloc[:, 5] * (5/9)   # Fan Inlet Temperature Rankine to Kelvin
df.iloc[:, 6] = df.iloc[:, 6] * (5/9)   # LPC Outlet Temp Rankine to Kelvin
df.iloc[:, 7] = df.iloc[:, 7] * (5/9)   # HPC Outlet Temp Rankine to Kelvin
df.iloc[:, 8] = df.iloc[:, 8] * (5/9)   # LPT Outlet Temp Rankine to Kelvin
df.iloc[:, 9] = df.iloc[:, 9] * 6.89476     # Fan Inlet Pressure psia to kPa
df.iloc[:, 10] = df.iloc[:, 10] * 6.89476     # Bypass-Duct Pressure psia to kPa
df.iloc[:, 11] = df.iloc[:, 11] * 6.89476     # HPC Outlet Pressure psia to kPa
df.iloc[:, 15] = df.iloc[:, 15] * 6.89476     # HPC Outlet Static Pressure psia to kPa
df.iloc[:, 16] = df.iloc[:, 16] * 0.00064516  # Mass flow rate to LPC outlet converted from pps/psia to m²/s

#%%
df = df.drop(columns=[26, 27])
df.columns = columns
#%%
df
#%%
df = pd.read_csv("data/processed/test_FD001_processed.csv")

rul_df = pd.read_csv("data/raw/RUL_FD001.txt",
                     header=None, names=["final_rul"])
rul_df["engine"] = rul_df.index + 1

last_cycle = df.groupby("engine")["cycle"].max().rename("last_cycle").reset_index()

tmp = df.merge(last_cycle, on="engine", how="left").merge(rul_df, on="engine", how="left")

tmp["RUL"] = tmp["final_rul"] + (tmp["last_cycle"] - tmp["cycle"])

train_with_rul = tmp.drop(columns=["last_cycle", "final_rul"])
#%%
train_with_rul
#%%
train_with_rul.to_csv('data/processed/test_FD001_processed.csv', index=False)
#%%

#%%

#%%
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
#%%
# Calculate and visualize correlations between sensors
sensor_cols = [col for col in df.columns if col not in ['engine', 'cycle']]
correlation_matrix = df[sensor_cols].corr()

plt.figure(figsize=(20,16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Sensor Measurements Correlation Matrix')
plt.show()

# Normalize sensor data
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[sensor_cols] = scaler.fit_transform(df[sensor_cols])

def create_sequences(data, sequence_length):
    sequences = []
    targets = []

    for engine in data['engine'].unique():
        engine_data = data[data['engine'] == engine]
        max_cycles = engine_data['cycle'].max()

        for i in range(len(engine_data) - sequence_length):
            sequence = engine_data.iloc[i:(i + sequence_length)]
            target = max_cycles - (i + sequence_length)  # RUL
            sequences.append(sequence[sensor_cols].values)
            targets.append(target)

    return np.array(sequences), np.array(targets)

# Create sequences with 30-cycle lookback
sequence_length = 30
X, y = create_sequences(df_normalized, sequence_length)

# Print dataset statistics
print("Dataset Summary:")
print("-" * 50)
print(f"Number of engines: {len(df['engine'].unique())}")
print(f"Average cycles per engine: {df.groupby('engine')['cycle'].max().mean():.2f}")
print(f"Sequence shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Min RUL: {y.min():.2f}")
print(f"Max RUL: {y.max():.2f}")

# Check for data quality
print("\nData Quality Check:")
print("-" * 50)
print("Missing values:")
print(df_normalized.isna().sum().sum())

# Save processed data
np.save('data/processed/X_sequences.npy', X)
np.save('data/processed/y_sequences.npy', y)
df_normalized.to_csv('data/processed/train_FD001_processed_scaled.csv', index=False)
#%%
