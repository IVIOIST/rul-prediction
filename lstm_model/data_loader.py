"""
Data loading and preprocessing utilities for LSTM RUL prediction
Author: Tony
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class RULDataset(Dataset):
    """PyTorch Dataset for RUL prediction sequences"""
    
    def __init__(self, sequences, targets):
        """
        Args:
            sequences: numpy array of shape (num_samples, sequence_length, num_features)
            targets: numpy array of shape (num_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def prepare_sequences(df, sequence_length, feature_cols, scaler=None, is_train=True):
    """
    Prepare sequences from dataframe for LSTM training/inference
    
    Args:
        df: pandas DataFrame with columns ['engine', 'cycle', features..., 'RUL']
        sequence_length: number of time steps in each sequence
        feature_cols: list of feature column names to use
        scaler: fitted StandardScaler object (if None and is_train=True, will create and fit)
        is_train: whether this is training data (for fitting scaler)
    
    Returns:
        X: numpy array of sequences (num_samples, sequence_length, num_features)
        y: numpy array of targets (num_samples,)
        scaler: fitted StandardScaler object
    """
    sequences = []
    targets = []
    
    # Fit scaler on training data only
    if is_train and scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    elif scaler is not None:
        df[feature_cols] = scaler.transform(df[feature_cols])
    else:
        raise ValueError("Must provide scaler for test data or set is_train=True")
    
    # Create sequences for each engine
    for engine_id in df['engine'].unique():
        engine_data = df[df['engine'] == engine_id].sort_values('cycle')
        
        # Extract features and RUL
        features = engine_data[feature_cols].values
        rul_values = engine_data['RUL'].values
        
        # Create sequences with sliding window
        for i in range(len(features) - sequence_length + 1):
            # Get sequence of features
            seq = features[i:i + sequence_length]
            # Target is the RUL at the last time step of the sequence
            target = rul_values[i + sequence_length - 1]
            
            sequences.append(seq)
            targets.append(target)
    
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f"Created {len(X)} sequences of shape {X.shape}")
    print(f"RUL range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, scaler


def load_and_prepare_data(train_path, test_path, sequence_length, exclude_cols, 
                          batch_size=64, save_scaler=True):
    """
    Load CSV files and prepare train/test DataLoaders
    
    Args:
        train_path: path to training CSV file
        test_path: path to testing CSV file
        sequence_length: number of time steps in each sequence
        exclude_cols: columns to exclude from features (e.g., ['engine', 'cycle', 'RUL'])
        batch_size: batch size for DataLoader
        save_scaler: whether to save the fitted scaler
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        scaler: fitted StandardScaler object
        feature_cols: list of feature column names
    """
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv(train_path)
    print("Loading testing data...")
    test_df = pd.read_csv(test_path)
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    
    # Prepare sequences
    print("\nPreparing training sequences...")
    X_train, y_train, scaler = prepare_sequences(
        train_df.copy(), sequence_length, feature_cols, is_train=True
    )
    
    print("\nPreparing testing sequences...")
    X_test, y_test, _ = prepare_sequences(
        test_df.copy(), sequence_length, feature_cols, scaler=scaler, is_train=False
    )
    
    # Save scaler for future use
    if save_scaler:
        scaler_path = Path('lstm_model/checkpoints/scaler.pkl')
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"\nScaler saved to {scaler_path}")
    
    # Create datasets
    train_dataset = RULDataset(X_train, y_train)
    test_dataset = RULDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Testing batches: {len(test_loader)}")
    
    return train_loader, test_loader, scaler, feature_cols


def load_scaler(scaler_path='lstm_model/checkpoints/scaler.pkl'):
    """Load saved scaler"""
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)
