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
    
    # Model architecture
    INPUT_SIZE = 24  # Number of features (will be set dynamically)
    HIDDEN_SIZE = 128  # LSTM hidden dimension
    NUM_LAYERS = 3  # Number of LSTM layers
    DROPOUT = 0.3  # Dropout rate for regularization
    BIDIRECTIONAL = True  # Use bidirectional LSTM
    
    # Training parameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-5  # L2 regularization
    
    # Early stopping
    PATIENCE = 30 # Number of epochs to wait before early stopping
    MIN_DELTA = 0.001  # Minimum change to qualify as an improvement
    
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
        print(f"Hidden Size: {cls.HIDDEN_SIZE}")
        print(f"Number of Layers: {cls.NUM_LAYERS}")
        print(f"Dropout: {cls.DROPOUT}")
        print(f"Bidirectional: {cls.BIDIRECTIONAL}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Device: {cls.DEVICE}")
        print("=" * 60)
