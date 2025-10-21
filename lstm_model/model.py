"""
LSTM Model Architecture for RUL Prediction
Author: Tony
"""

import torch
import torch.nn as nn


class LSTMRULPredictor(nn.Module):
    """
    LSTM-based model for Remaining Useful Life prediction
    
    Architecture:
        - Multi-layer LSTM (optionally bidirectional)
        - Dropout for regularization
        - Fully connected layers for final prediction
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, 
                 dropout=0.3, bidirectional=True):
        """
        Args:
            input_size: number of input features
            hidden_size: number of hidden units in LSTM
            num_layers: number of LSTM layers
            dropout: dropout probability
            bidirectional: whether to use bidirectional LSTM
        """
        super(LSTMRULPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Determine the size of LSTM output
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            output: predicted RUL of shape (batch_size, 1)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size * num_directions)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the output from the last time step
        # Shape: (batch_size, hidden_size * num_directions)
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc_layers(last_output)
        
        return output
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
        
        return info
    
    def print_model_info(self):
        """Print model architecture information"""
        info = self.get_model_info()
        print("=" * 60)
        print("LSTM RUL Predictor Model Architecture")
        print("=" * 60)
        print(f"Input Size: {info['input_size']}")
        print(f"Hidden Size: {info['hidden_size']}")
        print(f"Number of Layers: {info['num_layers']}")
        print(f"Bidirectional: {info['bidirectional']}")
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print("=" * 60)


class SimpleLSTMRULPredictor(nn.Module):
    """
    Simplified LSTM model for RUL prediction
    Lighter architecture for faster training
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Args:
            input_size: number of input features
            hidden_size: number of hidden units in LSTM
            num_layers: number of LSTM layers
            dropout: dropout probability
        """
        super(SimpleLSTMRULPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Single fully connected layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            output: predicted RUL of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        output = self.fc(last_output)
        
        return output
