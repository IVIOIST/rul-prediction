"""
Visualization Helper for LSTM RUL Prediction Results
Author: Tony

This script provides additional visualization tools for analyzing
LSTM model performance and results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_training_history_advanced(history_dict, save_path=None):
    """
    Create advanced training history visualization
    
    Args:
        history_dict: Dictionary with training history
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs = range(1, len(history_dict['train_loss']) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, history_dict['train_loss'], 'b-', linewidth=2, label='Training Loss')
    axes[0, 0].plot(epochs, history_dict['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMSE plot
    axes[0, 1].plot(epochs, history_dict['train_rmse'], 'b-', linewidth=2, label='Training RMSE')
    axes[0, 1].plot(epochs, history_dict['val_rmse'], 'r-', linewidth=2, label='Validation RMSE')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('RMSE (cycles)', fontsize=12)
    axes[0, 1].set_title('Training and Validation RMSE Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 0].plot(epochs, history_dict['learning_rates'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Generalization gap
    gap = np.array(history_dict['val_loss']) - np.array(history_dict['train_loss'])
    axes[1, 1].plot(epochs, gap, 'purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Validation Loss - Training Loss', fontsize=12)
    axes[1, 1].set_title('Generalization Gap (Overfitting Indicator)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Advanced training history saved to {save_path}")
    
    plt.show()


def compare_models_performance(predictions_dict, actuals, save_path=None):
    """
    Compare performance of multiple models
    
    Args:
        predictions_dict: Dictionary of {model_name: predictions_array}
        actuals: Array of actual values
        save_path: Optional path to save the figure
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Metrics comparison
    metrics_data = []
    for model_name, predictions in predictions_dict.items():
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        metrics_data.append([model_name, rmse, mae, r2])
    
    metrics_df = pd.DataFrame(metrics_data, columns=['Model', 'RMSE', 'MAE', 'R²'])
    
    # RMSE comparison
    axes[0, 0].bar(metrics_df['Model'], metrics_df['RMSE'], color='steelblue', alpha=0.7)
    axes[0, 0].set_ylabel('RMSE (cycles)', fontsize=12)
    axes[0, 0].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # MAE comparison
    axes[0, 1].bar(metrics_df['Model'], metrics_df['MAE'], color='coral', alpha=0.7)
    axes[0, 1].set_ylabel('MAE (cycles)', fontsize=12)
    axes[0, 1].set_title('MAE Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # R² comparison
    axes[1, 0].bar(metrics_df['Model'], metrics_df['R²'], color='forestgreen', alpha=0.7)
    axes[1, 0].set_ylabel('R² Score', fontsize=12)
    axes[1, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].axhline(y=1.0, color='red', linestyle='--', linewidth=1)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Scatter plot comparison
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))
    for (model_name, predictions), color in zip(predictions_dict.items(), colors):
        axes[1, 1].scatter(actuals, predictions, alpha=0.4, s=20, label=model_name, color=color)
    
    axes[1, 1].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_xlabel('Actual RUL', fontsize=12)
    axes[1, 1].set_ylabel('Predicted RUL', fontsize=12)
    axes[1, 1].set_title('Predictions Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    plt.show()
    
    return metrics_df


def plot_engine_degradation_patterns(test_data_path, predictions, n_engines=4, save_path=None):
    """
    Plot degradation patterns for multiple engines
    
    Args:
        test_data_path: Path to test data CSV
        predictions: Array of predictions
        n_engines: Number of engines to plot
        save_path: Optional path to save the figure
    """
    df = pd.read_csv(test_data_path)
    
    # Select engines with different characteristics
    engine_ids = df['engine'].unique()[:n_engines]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, engine_id in enumerate(engine_ids):
        engine_data = df[df['engine'] == engine_id].sort_values('cycle')
        cycles = engine_data['cycle'].values
        actual_rul = engine_data['RUL'].values
        
        # Note: This is simplified - in real usage, you'd align predictions with engine data
        # For demonstration, we'll use the first len(cycles) predictions
        
        axes[idx].plot(cycles, actual_rul, 'b-', linewidth=2, marker='o', 
                      markersize=4, label='Actual RUL', alpha=0.7)
        axes[idx].set_xlabel('Cycle', fontsize=11)
        axes[idx].set_ylabel('RUL (cycles)', fontsize=11)
        axes[idx].set_title(f'Engine {engine_id} - Degradation Pattern', 
                           fontsize=13, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        
        # Add shaded regions for different health states
        max_rul = actual_rul.max()
        axes[idx].axhspan(0, max_rul * 0.3, alpha=0.2, color='red', label='Critical')
        axes[idx].axhspan(max_rul * 0.3, max_rul * 0.6, alpha=0.2, color='yellow', label='Warning')
        axes[idx].axhspan(max_rul * 0.6, max_rul, alpha=0.2, color='green', label='Healthy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Engine degradation patterns saved to {save_path}")
    
    plt.show()


def create_model_report(metrics, save_path='lstm_model/results/model_report.txt'):
    """
    Create a text report summarizing model performance
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the report
    """
    report = []
    report.append("=" * 70)
    report.append("LSTM RUL PREDICTION MODEL - PERFORMANCE REPORT")
    report.append("Author: Tony")
    report.append("=" * 70)
    report.append("")
    
    report.append("EVALUATION METRICS")
    report.append("-" * 70)
    report.append(f"Root Mean Squared Error (RMSE):        {metrics['RMSE']:.4f} cycles")
    report.append(f"Mean Absolute Error (MAE):             {metrics['MAE']:.4f} cycles")
    report.append(f"R² Score:                              {metrics['R2']:.4f}")
    report.append(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
    report.append(f"Custom Score:                          {metrics['Custom_Score']:.4f}")
    report.append("")
    
    report.append("INTERPRETATION")
    report.append("-" * 70)
    
    # RMSE interpretation
    if metrics['RMSE'] < 15:
        rmse_rating = "Excellent"
    elif metrics['RMSE'] < 20:
        rmse_rating = "Good"
    elif metrics['RMSE'] < 25:
        rmse_rating = "Fair"
    else:
        rmse_rating = "Needs Improvement"
    report.append(f"RMSE Rating: {rmse_rating}")
    
    # R² interpretation
    if metrics['R2'] > 0.9:
        r2_rating = "Excellent fit"
    elif metrics['R2'] > 0.85:
        r2_rating = "Good fit"
    elif metrics['R2'] > 0.75:
        r2_rating = "Acceptable fit"
    else:
        r2_rating = "Poor fit"
    report.append(f"R² Rating: {r2_rating}")
    
    report.append("")
    report.append("SUMMARY")
    report.append("-" * 70)
    report.append(f"On average, predictions are off by ±{metrics['MAE']:.2f} cycles.")
    report.append(f"The model explains {metrics['R2']*100:.2f}% of the variance in RUL.")
    report.append("")
    
    report.append("=" * 70)
    
    # Save to file
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    # Also print to console
    print('\n'.join(report))
    print(f"\nReport saved to {save_path}")


def main():
    """Demonstration of visualization tools"""
    print("=" * 60)
    print("LSTM Visualization Helper")
    print("Author: Tony")
    print("=" * 60)
    print("\nThis script provides additional visualization functions.")
    print("\nAvailable functions:")
    print("  - plot_training_history_advanced()")
    print("  - compare_models_performance()")
    print("  - plot_engine_degradation_patterns()")
    print("  - create_model_report()")
    print("\nImport this module to use these functions in your analysis.")
    print("=" * 60)


if __name__ == "__main__":
    main()
