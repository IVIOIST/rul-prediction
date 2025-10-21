"""
Main script to train and evaluate LSTM RUL Prediction model
Author: Tony

Usage:
    python lstm_model/main.py
"""

import torch
import numpy as np
import random
from pathlib import Path

from config import Config
from model import LSTMRULPredictor
from data_loader import load_and_prepare_data
from trainer import LSTMTrainer
from evaluator import LSTMEvaluator


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main function to run the entire pipeline"""
    
    # Set random seed
    set_seed(Config.RANDOM_SEED)
    
    # Print configuration
    Config.print_config()
    
    # Load and prepare data
    print("\n" + "=" * 60)
    print("Loading and Preparing Data")
    print("=" * 60)
    
    train_loader, test_loader, scaler, feature_cols = load_and_prepare_data(
        train_path=Config.TRAIN_DATA_PATH,
        test_path=Config.TEST_DATA_PATH,
        sequence_length=Config.SEQUENCE_LENGTH,
        exclude_cols=Config.EXCLUDE_FEATURES,
        batch_size=Config.BATCH_SIZE,
        save_scaler=True
    )
    
    # Update input size in config
    Config.INPUT_SIZE = len(feature_cols)
    
    # Initialize model
    print("\n" + "=" * 60)
    print("Initializing Model")
    print("=" * 60)
    
    model = LSTMRULPredictor(
        input_size=Config.INPUT_SIZE,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        bidirectional=Config.BIDIRECTIONAL
    )
    
    model.print_model_info()
    
    # Initialize trainer
    trainer = LSTMTrainer(model, Config.DEVICE, Config)
    
    # Train model
    print("\n")
    trainer.train(train_loader, test_loader, num_epochs=Config.NUM_EPOCHS)
    
    # Save model
    trainer.save_model()
    
    # Plot training history
    plot_training_history(trainer.get_history())
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    evaluator = LSTMEvaluator(model, Config.DEVICE)
    metrics = evaluator.evaluate(
        test_loader, 
        plot=True, 
        save_plots=True,
        save_dir='lstm_model/results'
    )
    
    # Save predictions
    evaluator.save_predictions(test_loader)
    
    print("\n" + "=" * 60)
    print("Training and Evaluation Complete!")
    print("=" * 60)
    print(f"\nModel saved to: {Config.MODEL_SAVE_PATH}{Config.BEST_MODEL_NAME}")
    print(f"Results saved to: lstm_model/results/")
    print("\nBest Test Metrics:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  R²: {metrics['R2']:.4f}")
    
    return model, trainer, evaluator, metrics


def plot_training_history(history):
    """Plot training history"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RMSE plot
    axes[1].plot(epochs, history['train_rmse'], label='Train RMSE', linewidth=2)
    axes[1].plot(epochs, history['val_rmse'], label='Validation RMSE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('Training and Validation RMSE', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path('lstm_model/results/training_history.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    model, trainer, evaluator, metrics = main()
