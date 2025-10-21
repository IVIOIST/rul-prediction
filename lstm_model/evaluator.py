"""
Evaluation utilities for LSTM RUL Prediction
Author: Tony
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path


class LSTMEvaluator:
    """Evaluator class for LSTM RUL prediction model"""
    
    def __init__(self, model, device):
        """
        Args:
            model: trained LSTM model
            device: torch device (cuda or cpu)
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(self, dataloader):
        """
        Make predictions on a dataset
        
        Args:
            dataloader: DataLoader for the dataset
        
        Returns:
            predictions: numpy array of predictions
            actuals: numpy array of actual values
        """
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.numpy())
        
        return np.array(predictions).flatten(), np.array(actuals).flatten()
    
    def calculate_metrics(self, predictions, actuals):
        """
        Calculate evaluation metrics
        
        Args:
            predictions: predicted values
            actuals: actual values
        
        Returns:
            metrics: dictionary of evaluation metrics
        """
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
            'MAE': mean_absolute_error(actuals, predictions),
            'R2': r2_score(actuals, predictions),
            'MAPE': np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
        }
        
        # Custom score function (lower is better)
        # Penalizes late predictions more than early predictions
        errors = predictions - actuals
        late_predictions = errors[errors > 0]
        early_predictions = errors[errors <= 0]
        
        if len(late_predictions) > 0:
            late_penalty = np.sum(late_predictions ** 2)
        else:
            late_penalty = 0
        
        if len(early_predictions) > 0:
            early_penalty = np.sum(early_predictions ** 2)
        else:
            early_penalty = 0
        
        metrics['Custom_Score'] = late_penalty + early_penalty
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print evaluation metrics"""
        print("=" * 60)
        print("Model Evaluation Metrics")
        print("=" * 60)
        print(f"RMSE (Root Mean Squared Error): {metrics['RMSE']:.4f}")
        print(f"MAE (Mean Absolute Error): {metrics['MAE']:.4f}")
        print(f"R² Score: {metrics['R2']:.4f}")
        print(f"MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.2f}%")
        print(f"Custom Score: {metrics['Custom_Score']:.4f}")
        print("=" * 60)
    
    def plot_predictions(self, predictions, actuals, title="RUL Predictions vs Actual", 
                         save_path=None):
        """
        Plot predictions vs actual values
        
        Args:
            predictions: predicted values
            actuals: actual values
            title: plot title
            save_path: path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot
        axes[0, 0].scatter(actuals, predictions, alpha=0.5, s=10)
        axes[0, 0].plot([actuals.min(), actuals.max()], 
                        [actuals.min(), actuals.max()], 
                        'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual RUL', fontsize=12)
        axes[0, 0].set_ylabel('Predicted RUL', fontsize=12)
        axes[0, 0].set_title('Predicted vs Actual RUL', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = predictions - actuals
        axes[0, 1].scatter(actuals, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Actual RUL', fontsize=12)
        axes[0, 1].set_ylabel('Residual (Predicted - Actual)', fontsize=12)
        axes[0, 1].set_title('Residual Plot', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residual', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Distribution of Residuals', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Time series comparison (first 500 samples)
        n_samples = min(500, len(predictions))
        x_axis = np.arange(n_samples)
        axes[1, 1].plot(x_axis, actuals[:n_samples], label='Actual', alpha=0.7, linewidth=1.5)
        axes[1, 1].plot(x_axis, predictions[:n_samples], label='Predicted', alpha=0.7, linewidth=1.5)
        axes[1, 1].set_xlabel('Sample Index', fontsize=12)
        axes[1, 1].set_ylabel('RUL', fontsize=12)
        axes[1, 1].set_title(f'RUL Comparison (First {n_samples} samples)', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_error_distribution(self, predictions, actuals, save_path=None):
        """
        Plot error distribution analysis
        
        Args:
            predictions: predicted values
            actuals: actual values
            save_path: path to save the plot (optional)
        """
        errors = predictions - actuals
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Error by RUL range
        rul_ranges = [(0, 50), (50, 100), (100, 150), (150, 300)]
        range_errors = []
        range_labels = []
        
        for low, high in rul_ranges:
            mask = (actuals >= low) & (actuals < high)
            if mask.sum() > 0:
                range_errors.append(errors[mask])
                range_labels.append(f'{low}-{high}')
        
        axes[0].boxplot(range_errors, labels=range_labels)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('RUL Range', fontsize=12)
        axes[0].set_ylabel('Prediction Error', fontsize=12)
        axes[0].set_title('Error Distribution by RUL Range', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Absolute error by actual RUL
        abs_errors = np.abs(errors)
        axes[1].scatter(actuals, abs_errors, alpha=0.5, s=10)
        axes[1].set_xlabel('Actual RUL', fontsize=12)
        axes[1].set_ylabel('Absolute Error', fontsize=12)
        axes[1].set_title('Absolute Error vs Actual RUL', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        # Percentage error distribution
        pct_errors = (errors / (actuals + 1e-8)) * 100
        # Filter out extreme outliers for better visualization
        pct_errors_filtered = pct_errors[(pct_errors > -100) & (pct_errors < 100)]
        axes[2].hist(pct_errors_filtered, bins=50, edgecolor='black', alpha=0.7)
        axes[2].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[2].set_xlabel('Percentage Error (%)', fontsize=12)
        axes[2].set_ylabel('Frequency', fontsize=12)
        axes[2].set_title('Distribution of Percentage Errors', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error distribution plot saved to {save_path}")
        
        plt.show()
    
    def evaluate(self, dataloader, plot=True, save_plots=False, save_dir='lstm_model/results'):
        """
        Complete evaluation of the model
        
        Args:
            dataloader: DataLoader for the dataset
            plot: whether to generate plots
            save_plots: whether to save plots
            save_dir: directory to save plots
        
        Returns:
            metrics: dictionary of evaluation metrics
        """
        print("\nEvaluating model...")
        predictions, actuals = self.predict(dataloader)
        metrics = self.calculate_metrics(predictions, actuals)
        self.print_metrics(metrics)
        
        if plot:
            if save_plots:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                pred_plot_path = save_dir / 'predictions_plot.png'
                error_plot_path = save_dir / 'error_distribution.png'
            else:
                pred_plot_path = None
                error_plot_path = None
            
            self.plot_predictions(predictions, actuals, save_path=pred_plot_path)
            self.plot_error_distribution(predictions, actuals, save_path=error_plot_path)
        
        return metrics
    
    def save_predictions(self, dataloader, save_path='lstm_model/results/predictions.csv'):
        """
        Save predictions to CSV file
        
        Args:
            dataloader: DataLoader for the dataset
            save_path: path to save the CSV file
        """
        predictions, actuals = self.predict(dataloader)
        
        df = pd.DataFrame({
            'Actual_RUL': actuals,
            'Predicted_RUL': predictions,
            'Error': predictions - actuals,
            'Absolute_Error': np.abs(predictions - actuals)
        })
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        
        print(f"\nPredictions saved to {save_path}")
        return df
