# LSTM RUL Prediction Model

**Author: Tony**  
**Project: ENGG2112 - Turbofan Engine Remaining Useful Life Prediction**

## 🚀 Quick Start

### Option 1: Using the Quick Start Script
```bash
python lstm_model/quickstart.py
```

This interactive menu will guide you through:
1. Checking dependencies
2. Training the model
3. Running inference
4. Installing requirements

### Option 2: Manual Training
```bash
# Install dependencies
pip install -r lstm_model/requirements.txt

# Train the model
python lstm_model/main.py

# Run inference
python lstm_model/inference.py
```

## 📊 What This Model Does

This LSTM (Long Short-Term Memory) neural network predicts the **Remaining Useful Life (RUL)** of turbofan engines based on sensor readings. The model:

- Takes sequences of 30 cycles of sensor data as input
- Uses a 3-layer bidirectional LSTM architecture
- Predicts how many more cycles an engine can operate before failure
- Achieves state-of-the-art performance on the NASA C-MAPSS dataset

## 📁 Project Structure

```
lstm_model/
├── __init__.py          # Package initialization
├── config.py            # Configuration (hyperparameters, paths)
├── model.py             # LSTM model architecture
├── data_loader.py       # Data loading and preprocessing
├── trainer.py           # Training loop and utilities
├── evaluator.py         # Evaluation metrics and visualization
├── main.py              # Main training script
├── inference.py         # Inference script for predictions
├── quickstart.py        # Interactive menu for easy usage
├── README.md            # Detailed documentation
├── requirements.txt     # Python dependencies
├── checkpoints/         # Saved models (created after training)
└── results/             # Evaluation plots and results (created after training)
```

## 🎯 Key Features

### Model Architecture
- **Bidirectional LSTM**: Captures patterns in both forward and backward directions
- **3 Layers**: Deep network for complex pattern recognition
- **128 Hidden Units**: Sufficient capacity for temporal features
- **Dropout Regularization**: Prevents overfitting
- **~500K Parameters**: Moderate size, trainable on CPU

### Training Features
- **Early Stopping**: Stops when validation loss stops improving
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Gradient Clipping**: Prevents exploding gradients
- **Data Standardization**: Features scaled with StandardScaler
- **Progress Tracking**: Real-time training progress with tqdm

### Evaluation
- **Multiple Metrics**: RMSE, MAE, R², MAPE
- **Comprehensive Plots**: Predictions, residuals, error distributions
- **Per-Engine Analysis**: Individual engine trajectory visualization
- **CSV Export**: Detailed predictions saved for analysis

## 📖 Usage Examples

### Training a Model

```python
from lstm_model.config import Config
from lstm_model.model import LSTMRULPredictor
from lstm_model.data_loader import load_and_prepare_data
from lstm_model.trainer import LSTMTrainer

# Load data
train_loader, test_loader, scaler, features = load_and_prepare_data(
    train_path='data/processed/train_FD001_processed.csv',
    test_path='data/processed/test_FD001_processed.csv',
    sequence_length=30,
    exclude_cols=['engine', 'cycle', 'RUL'],
    batch_size=64
)

# Create model
model = LSTMRULPredictor(
    input_size=len(features),
    hidden_size=128,
    num_layers=3,
    dropout=0.3,
    bidirectional=True
)

# Train
trainer = LSTMTrainer(model, Config.DEVICE, Config)
trainer.train(train_loader, test_loader)
trainer.save_model()
```

### Making Predictions

```python
from lstm_model.inference import RULPredictor

# Initialize predictor
predictor = RULPredictor('lstm_model/checkpoints/best_lstm_model.pt')

# Load test data
predictor.load_test_data('data/processed/test_FD001_processed.csv')

# Predict
predictions = predictor.predict()

# Evaluate
metrics = predictor.evaluate()

# Visualize
predictor.visualize_predictions()

# Save results
predictor.save_predictions()
```

### Single Engine Prediction

```python
# Predict for a specific engine
predictions, actuals = predictor.predict_single_engine(engine_id=1)

# Plot trajectory
predictor.plot_engine_trajectory(engine_id=1)
```

## ⚙️ Configuration

Edit `lstm_model/config.py` to customize:

```python
class Config:
    # Sequence settings
    SEQUENCE_LENGTH = 30        # Time steps to look back
    
    # Model architecture
    HIDDEN_SIZE = 128           # LSTM hidden dimension
    NUM_LAYERS = 3              # Number of LSTM layers
    DROPOUT = 0.3               # Dropout rate
    BIDIRECTIONAL = True        # Use bidirectional LSTM
    
    # Training settings
    BATCH_SIZE = 64             # Batch size
    LEARNING_RATE = 0.001       # Learning rate
    NUM_EPOCHS = 100            # Max epochs
    PATIENCE = 15               # Early stopping patience
```

## 📊 Expected Results

After training, you should see:

### Training Metrics
- **Training Loss**: Should decrease to ~100-200 (MSE)
- **Validation Loss**: Should be similar to training loss
- **RMSE**: Typically 12-18 cycles on test set

### Output Files
- `lstm_model/checkpoints/best_lstm_model.pt` - Trained model
- `lstm_model/checkpoints/scaler.pkl` - Data scaler
- `lstm_model/results/training_history.png` - Loss curves
- `lstm_model/results/predictions_plot.png` - Prediction analysis
- `lstm_model/results/predictions.csv` - Detailed predictions

## 🔧 Troubleshooting

### Out of Memory
```python
# In config.py, reduce:
BATCH_SIZE = 32  # or 16
HIDDEN_SIZE = 64
NUM_LAYERS = 2
```

### Training Too Slow
```python
# Use simpler model
from lstm_model.model import SimpleLSTMRULPredictor
```

### Poor Predictions
- Ensure data is preprocessed correctly
- Check that RUL column is present in CSV files
- Verify sequence length matches your data
- Try adjusting hyperparameters

## 💡 Tips

1. **First Time**: Start with default settings
2. **Limited RAM**: Use `BATCH_SIZE=16`, `SimpleLSTMRULPredictor`
3. **Better Accuracy**: Increase `NUM_EPOCHS` to 150, `HIDDEN_SIZE` to 256
4. **Faster Training**: Reduce `SEQUENCE_LENGTH` to 20, use GPU if available

## 📝 Data Format

The model expects CSV files with:
- `engine`: Engine ID (integer)
- `cycle`: Cycle number (integer)
- `RUL`: Remaining useful life (target, integer)
- 24 sensor columns (floats)

Example:
```
engine,cycle,setting1,setting2,...,RUL
1,1,-0.0007,-0.0004,...,191
1,2,0.0019,-0.0003,...,190
```

## 🎓 Learning Resources

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- [NASA C-MAPSS Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

## 🤝 Support

For questions or issues:
1. Check the detailed `lstm_model/README.md`
2. Review error messages in console output
3. Verify all dependencies are installed
4. Contact Tony for project-specific help

## 📄 License

This project is for educational purposes as part of ENGG2112 (2025 S2).

---

**Happy Predicting! 🚀**  
*Tony - ENGG2112 Team*
