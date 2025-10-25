"""
Configuration file for LSTM RUL Prediction Model
Author: Tony
"""

import torch

class Config:
    """Configuration parameters for the LSTM model"""
    
    # Data parameters
    SEQUENCE_LENGTH = 30  # Number of time steps to look back
    TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% validation
    MAX_RUL = 125  # Clip RUL values above this (standard in NASA C-MAPSS literature)
    
    # Model architecture
    INPUT_SIZE = 24  # Number of features (will be set dynamically)
    HIDDEN_SIZE = 128  # LSTM hidden dimension (increased back for better capacity)
    NUM_LAYERS = 2  # Number of LSTM layers
    DROPOUT = 0.2  # Dropout rate (reduced for better fitting)
    BIDIRECTIONAL = True  # Use bidirectional LSTM (better pattern recognition)
    
    # Training parameters
    BATCH_SIZE = 256  # Larger batches for more stable training
    LEARNING_RATE = 0.001  # Increased for faster initial learning
    NUM_EPOCHS = 100  # Increased to allow more training time
    WEIGHT_DECAY = 1e-5  # Very light regularization
    
    # Early stopping
    PATIENCE = 15  # Increased patience for better convergence
    MIN_DELTA = 0.0  # Accept any improvement
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data paths
    TRAIN_DATA_PATH = 'data/processed/train_FD001_processed.csv'
    TEST_DATA_PATH = 'data/processed/test_FD001_processed.csv'
    
    # Model save path
    MODEL_SAVE_PATH = 'lstm_model/checkpoints/'
    BEST_MODEL_NAME = 'best_lstm_model.pt'
    
    # Features to exclude from input (non-sensor columns)
    EXCLUDE_FEATURES = ['engine', 'cycle', 'RUL']
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("LSTM RUL Prediction Configuration")
        print("=" * 60)
        print(f"Sequence Length: {cls.SEQUENCE_LENGTH}")
        print(f"Max RUL (clipping): {cls.MAX_RUL}")
        print(f"Hidden Size: {cls.HIDDEN_SIZE}")
        print(f"Number of Layers: {cls.NUM_LAYERS}")
        print(f"Dropout: {cls.DROPOUT}")
        print(f"Bidirectional: {cls.BIDIRECTIONAL}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Weight Decay: {cls.WEIGHT_DECAY}")
        print(f"Device: {cls.DEVICE}")
        print("=" * 60)
