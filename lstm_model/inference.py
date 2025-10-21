"""
Inference script for LSTM RUL Prediction
Author: Tony

Usage:
    python lstm_model/inference.py
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from model import LSTMRULPredictor
from data_loader import prepare_sequences
from evaluator import LSTMEvaluator


class RULPredictor:
    """Class for making RUL predictions with a trained LSTM model"""
    
    def __init__(self, model_path, scaler_path=None):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: path to saved model checkpoint
            scaler_path: path to saved scaler (if None, will look in checkpoints/)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get configuration
        self.config = checkpoint['config']
        
        # Initialize model
        self.model = LSTMRULPredictor(
            input_size=self.config.INPUT_SIZE,
            hidden_size=self.config.HIDDEN_SIZE,
            num_layers=self.config.NUM_LAYERS,
            dropout=self.config.DROPOUT,
            bidirectional=self.config.BIDIRECTIONAL
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        self.model.print_model_info()
        
        # Load scaler
        if scaler_path is None:
            scaler_path = Path(model_path).parent / 'scaler.pkl'
        
        print(f"\nLoading scaler from {scaler_path}...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Scaler loaded successfully!")
        
        self.test_data = None
        self.predictions = None
        self.actuals = None
    
    def load_test_data(self, test_path):
        """
        Load test data from CSV
        
        Args:
            test_path: path to test CSV file
        """
        print(f"\nLoading test data from {test_path}...")
        self.test_data = pd.read_csv(test_path)
        print(f"Loaded {len(self.test_data)} records from {len(self.test_data['engine'].unique())} engines")
    
    def predict(self, data=None):
        """
        Make predictions on test data
        
        Args:
            data: pandas DataFrame (if None, uses loaded test data)
        
        Returns:
            predictions: numpy array of predicted RUL values
        """
        if data is None:
            if self.test_data is None:
                raise ValueError("No test data loaded. Use load_test_data() first.")
            data = self.test_data
        
        # Get feature columns
        feature_cols = [col for col in data.columns if col not in self.config.EXCLUDE_FEATURES]
        
        print(f"\nPreparing sequences with length {self.config.SEQUENCE_LENGTH}...")
        X, y, _ = prepare_sequences(
            data.copy(),
            self.config.SEQUENCE_LENGTH,
            feature_cols,
            scaler=self.scaler,
            is_train=False
        )
        
        print("Making predictions...")
        self.actuals = y
        self.predictions = []
        
        # Make predictions in batches
        with torch.no_grad():
            for i in range(0, len(X), self.config.BATCH_SIZE):
                batch = X[i:i + self.config.BATCH_SIZE]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                pred = self.model(batch_tensor)
                self.predictions.extend(pred.cpu().numpy())
        
        self.predictions = np.array(self.predictions).flatten()
        
        print(f"Predictions complete! Generated {len(self.predictions)} predictions.")
        
        return self.predictions
    
    def predict_single_engine(self, engine_id):
        """
        Make predictions for a single engine
        
        Args:
            engine_id: ID of the engine to predict
        
        Returns:
            predictions: numpy array of predictions for this engine
            actuals: numpy array of actual RUL values
        """
        if self.test_data is None:
            raise ValueError("No test data loaded. Use load_test_data() first.")
        
        # Filter data for this engine
        engine_data = self.test_data[self.test_data['engine'] == engine_id].copy()
        
        if len(engine_data) == 0:
            raise ValueError(f"Engine {engine_id} not found in test data")
        
        print(f"\nPredicting for Engine {engine_id} ({len(engine_data)} cycles)...")
        
        # Get feature columns
        feature_cols = [col for col in engine_data.columns if col not in self.config.EXCLUDE_FEATURES]
        
        # Prepare sequences
        X, y, _ = prepare_sequences(
            engine_data,
            self.config.SEQUENCE_LENGTH,
            feature_cols,
            scaler=self.scaler,
            is_train=False
        )
        
        # Make predictions
        predictions = []
        with torch.no_grad():
            for i in range(len(X)):
                x_tensor = torch.FloatTensor(X[i:i+1]).to(self.device)
                pred = self.model(x_tensor)
                predictions.append(pred.cpu().numpy()[0, 0])
        
        predictions = np.array(predictions)
        
        return predictions, y
    
    def evaluate(self):
        """Evaluate predictions and calculate metrics"""
        if self.predictions is None or self.actuals is None:
            raise ValueError("No predictions available. Run predict() first.")
        
        evaluator = LSTMEvaluator(self.model, self.device)
        metrics = evaluator.calculate_metrics(self.predictions, self.actuals)
        evaluator.print_metrics(metrics)
        
        return metrics
    
    def visualize_predictions(self, save_path=None):
        """Visualize predictions vs actuals"""
        if self.predictions is None or self.actuals is None:
            raise ValueError("No predictions available. Run predict() first.")
        
        evaluator = LSTMEvaluator(self.model, self.device)
        evaluator.plot_predictions(self.predictions, self.actuals, save_path=save_path)
        evaluator.plot_error_distribution(self.predictions, self.actuals)
    
    def plot_engine_trajectory(self, engine_id, save_path=None):
        """
        Plot RUL trajectory for a specific engine
        
        Args:
            engine_id: ID of the engine to plot
            save_path: path to save the plot (optional)
        """
        predictions, actuals = self.predict_single_engine(engine_id)
        
        # Get the cycle numbers for this engine (after sequence_length offset)
        engine_data = self.test_data[self.test_data['engine'] == engine_id]
        cycles = engine_data['cycle'].values[self.config.SEQUENCE_LENGTH - 1:]
        
        plt.figure(figsize=(12, 6))
        plt.plot(cycles, actuals, label='Actual RUL', marker='o', markersize=4, linewidth=2)
        plt.plot(cycles, predictions, label='Predicted RUL', marker='s', markersize=4, linewidth=2)
        plt.xlabel('Cycle', fontsize=12)
        plt.ylabel('RUL (cycles)', fontsize=12)
        plt.title(f'RUL Trajectory for Engine {engine_id}', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_predictions(self, save_path='lstm_model/results/inference_predictions.csv'):
        """Save predictions to CSV"""
        if self.predictions is None:
            raise ValueError("No predictions available. Run predict() first.")
        
        df = pd.DataFrame({
            'Actual_RUL': self.actuals,
            'Predicted_RUL': self.predictions,
            'Error': self.predictions - self.actuals,
            'Absolute_Error': np.abs(self.predictions - self.actuals)
        })
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        
        print(f"\nPredictions saved to {save_path}")
        return df


def main():
    """Main inference function"""
    
    # Paths
    model_path = 'lstm_model/checkpoints/best_lstm_model.pt'
    test_path = 'data/processed/test_FD001_processed.csv'
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python lstm_model/main.py")
        return
    
    # Initialize predictor
    predictor = RULPredictor(model_path)
    
    # Load test data
    predictor.load_test_data(test_path)
    
    # Make predictions
    predictions = predictor.predict()
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    metrics = predictor.evaluate()
    
    # Visualize
    print("\nGenerating visualizations...")
    predictor.visualize_predictions(save_path='lstm_model/results/inference_predictions_plot.png')
    
    # Save predictions
    predictor.save_predictions()
    
    # Example: Plot trajectory for first engine
    print("\nPlotting trajectory for Engine 1...")
    predictor.plot_engine_trajectory(1, save_path='lstm_model/results/engine_1_trajectory.png')
    
    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
