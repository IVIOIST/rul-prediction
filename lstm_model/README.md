# LSTM RUL Prediction Model
**Author: Tony**  
**Project: ENGG2112 - RUL Prediction on NASA C-MAPSS Dataset**

## Overview
This directory contains a complete LSTM (Long Short-Term Memory) model implementation for predicting the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset.

## Directory Structure
```
lstm_model/
├── __init__.py          # Package initialization
├── config.py            # Configuration parameters
├── model.py             # LSTM model architecture
├── data_loader.py       # Data loading and preprocessing
├── trainer.py           # Training utilities
├── evaluator.py         # Evaluation and visualization
├── main.py              # Main training script
├── inference.py         # Inference script for predictions
├── README.md            # This file
├── checkpoints/         # Saved models and scalers
└── results/             # Evaluation results and plots
```

## Model Architecture

### LSTMRULPredictor
- **Input**: Sequences of sensor readings (shape: `[batch_size, sequence_length, num_features]`)
- **LSTM Layers**: 3 bidirectional LSTM layers with 128 hidden units
- **Dropout**: 0.3 for regularization
- **Output Layers**: Fully connected layers reducing to single RUL prediction
- **Total Parameters**: ~500K trainable parameters

### Key Features
- Bidirectional LSTM for better temporal pattern recognition
- Gradient clipping to prevent exploding gradients
- Early stopping to prevent overfitting
- Learning rate scheduling with ReduceLROnPlateau
- Standardized input features using StandardScaler

## Configuration

Default configuration in `config.py`:
- **Sequence Length**: 30 cycles
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Epochs**: 100 (with early stopping)
- **Hidden Size**: 128
- **Number of Layers**: 3
- **Dropout**: 0.3

## Usage

### Training the Model

Run the main training script:
```bash
python lstm_model/main.py
```

This will:
1. Load and preprocess data from `data/processed/`
2. Create train/validation splits
3. Train the LSTM model
4. Save the best model to `lstm_model/checkpoints/`
5. Generate evaluation plots in `lstm_model/results/`

### Making Predictions

Use the inference script:
```bash
python lstm_model/inference.py
```

Or in Python:
```python
from lstm_model.inference import RULPredictor

# Initialize predictor
predictor = RULPredictor(model_path='lstm_model/checkpoints/best_lstm_model.pt')

# Load test data
predictor.load_test_data('data/processed/test_FD001_processed.csv')

# Make predictions
predictions = predictor.predict()

# Visualize results
predictor.visualize_predictions()
```

## Data Requirements

The model expects processed CSV files with the following structure:
- **engine**: Engine ID
- **cycle**: Cycle number
- **sensor columns**: 24 sensor readings
- **RUL**: Target remaining useful life

Files should be located at:
- Training: `data/processed/train_FD001_processed.csv`
- Testing: `data/processed/test_FD001_processed.csv`

## Output Files

After training, the following files are generated:

### Model Checkpoints
- `lstm_model/checkpoints/best_lstm_model.pt` - Best model weights
- `lstm_model/checkpoints/scaler.pkl` - Fitted StandardScaler

### Results
- `lstm_model/results/training_history.png` - Loss and RMSE curves
- `lstm_model/results/predictions_plot.png` - Prediction vs actual plots
- `lstm_model/results/error_distribution.png` - Error analysis
- `lstm_model/results/predictions.csv` - Detailed predictions

## Evaluation Metrics

The model is evaluated using:
- **RMSE** (Root Mean Squared Error): Overall prediction accuracy
- **MAE** (Mean Absolute Error): Average prediction error
- **R² Score**: Goodness of fit
- **MAPE** (Mean Absolute Percentage Error): Percentage error
- **Custom Score**: Penalty function (late predictions penalized more)

## Dependencies

Required packages:
```
torch
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
```

Install with:
```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn tqdm
```

## Performance Tips

### For Limited Memory (CPU Training)
- Reduce `BATCH_SIZE` to 32 or 16 in `config.py`
- Reduce `NUM_EPOCHS` to 50
- Use `SimpleLSTMRULPredictor` instead (lighter architecture)

### For GPU Training
- Ensure CUDA is available
- Set `pin_memory=True` in data loaders
- Increase `BATCH_SIZE` to 128 or 256

## Customization

### Modify Hyperparameters
Edit `lstm_model/config.py`:
```python
class Config:
    SEQUENCE_LENGTH = 50  # Increase lookback window
    HIDDEN_SIZE = 256     # Increase model capacity
    NUM_LAYERS = 4        # Deeper network
    LEARNING_RATE = 0.0005  # Lower learning rate
```

### Use Simpler Model
In `main.py`, replace:
```python
from model import SimpleLSTMRULPredictor as LSTMRULPredictor
```

## Results Visualization

The evaluation generates comprehensive plots:
1. **Predicted vs Actual**: Scatter plot showing prediction accuracy
2. **Residual Plot**: Shows systematic errors
3. **Residual Distribution**: Histogram of prediction errors
4. **Time Series Comparison**: Sample predictions over time
5. **Error by RUL Range**: Box plots showing performance across RUL ranges

## Troubleshooting

### Out of Memory Error
- Reduce `BATCH_SIZE` in config
- Reduce `SEQUENCE_LENGTH`
- Use `SimpleLSTMRULPredictor`

### Poor Performance
- Increase `NUM_EPOCHS`
- Adjust `LEARNING_RATE`
- Check data preprocessing in `data_loader.py`
- Ensure features are properly standardized

### Training Too Slow
- Reduce `SEQUENCE_LENGTH`
- Reduce `HIDDEN_SIZE` or `NUM_LAYERS`
- Use GPU if available

## Contact
For questions or issues, contact Tony.

## License
This project is for educational purposes as part of ENGG2112.
