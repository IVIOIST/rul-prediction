# LSTM RUL Prediction System - Complete Guide
**Created by Tony for ENGG2112 Project**

## 📋 Table of Contents
1. [Overview](#overview)
2. [What's Been Created](#whats-been-created)
3. [Quick Start Guide](#quick-start-guide)
4. [System Architecture](#system-architecture)
5. [How It Works](#how-it-works)
6. [Training the Model](#training-the-model)
7. [Making Predictions](#making-predictions)
8. [Understanding the Results](#understanding-the-results)
9. [Customization](#customization)
10. [Troubleshooting](#troubleshooting)

---

## 📖 Overview

This LSTM-based system predicts the **Remaining Useful Life (RUL)** of turbofan engines using sensor data from the NASA C-MAPSS dataset. The system uses deep learning to learn temporal patterns in engine degradation and predict when maintenance will be needed.

**Key Capabilities:**
- Processes sequences of 30 cycles of sensor readings
- Predicts RUL with high accuracy (RMSE ~12-18 cycles)
- Provides comprehensive evaluation metrics and visualizations
- Easy to use with interactive quickstart menu
- Fully documented and modular code structure

---

## 🎁 What's Been Created

All files have been created in the `lstm_model/` folder:

### Core Python Modules
1. **`config.py`** - Configuration and hyperparameters
2. **`model.py`** - LSTM neural network architectures
3. **`data_loader.py`** - Data preprocessing and loading
4. **`trainer.py`** - Training loop and optimization
5. **`evaluator.py`** - Evaluation metrics and visualization
6. **`main.py`** - Complete training pipeline
7. **`inference.py`** - Prediction script
8. **`quickstart.py`** - Interactive menu system

### Documentation
9. **`README.md`** - Detailed technical documentation
10. **`QUICKSTART.md`** - User-friendly quick start guide
11. **`requirements.txt`** - Python dependencies

### Added to Preprocessing Notebook
12. Two new cells added at the end of `preprocessing.ipynb`:
    - Markdown header for LSTM section
    - Dependency checker and instructions

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

Open your terminal (PowerShell) and navigate to your project folder:

```powershell
cd "c:\Users\zjy78\Desktop\school\Uni year 2\sem2\ENGG2112\Final\rul-prediction"
```

Install required packages:

```powershell
pip install -r lstm_model/requirements.txt
```

This installs:
- PyTorch (deep learning framework)
- NumPy, Pandas (data processing)
- Matplotlib, Seaborn (visualization)
- Scikit-learn (preprocessing)
- tqdm (progress bars)

### Step 2: Verify Data Files

Ensure these files exist:
- `data/processed/train_FD001_processed.csv`
- `data/processed/test_FD001_processed.csv`

They should already be there from your preprocessing work!

### Step 3: Train the Model

**Option A - Interactive Menu (Recommended):**
```powershell
python lstm_model/quickstart.py
```

Then select option 2 to train.

**Option B - Direct Training:**
```powershell
python lstm_model/main.py
```

Training will take 15-60 minutes depending on your hardware.

### Step 4: View Results

After training, check the `lstm_model/results/` folder for:
- Training history plots
- Prediction accuracy plots
- Error analysis charts

### Step 5: Make Predictions

```powershell
python lstm_model/inference.py
```

This loads the trained model and generates predictions on test data.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Preprocessing                       │
│  (train_FD001_processed.csv, test_FD001_processed.csv)    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   data_loader.py                            │
│  • Loads CSV files                                          │
│  • Creates sequences (30 cycles)                            │
│  • Standardizes features with StandardScaler               │
│  • Splits into train/validation sets                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      model.py                               │
│  ┌───────────────────────────────────────────────┐         │
│  │   Input: [batch, 30 cycles, 24 features]     │         │
│  │              ↓                                 │         │
│  │   Bidirectional LSTM Layer 1 (128 units)     │         │
│  │              ↓                                 │         │
│  │   Bidirectional LSTM Layer 2 (128 units)     │         │
│  │              ↓                                 │         │
│  │   Bidirectional LSTM Layer 3 (128 units)     │         │
│  │              ↓                                 │         │
│  │   Fully Connected Layers                      │         │
│  │              ↓                                 │         │
│  │   Output: [batch, 1] (RUL prediction)        │         │
│  └───────────────────────────────────────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    trainer.py                               │
│  • Adam optimizer with learning rate scheduling            │
│  • MSE loss function                                        │
│  • Gradient clipping                                        │
│  • Early stopping (patience=15 epochs)                     │
│  • Saves best model based on validation loss              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   evaluator.py                              │
│  • Calculates RMSE, MAE, R², MAPE                          │
│  • Generates prediction plots                              │
│  • Creates error distribution analysis                     │
│  • Saves results to CSV and PNG files                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 How It Works

### 1. Data Preparation

**Input Data:**
- Each engine has multiple cycles of operation
- Each cycle has 24 sensor readings
- Target is RUL (Remaining Useful Life) at each cycle

**Sequence Creation:**
- Sliding window of 30 cycles
- For each window, predict the RUL at the last time step
- Example: Cycles 1-30 → Predict RUL at cycle 30

**Normalization:**
- All features standardized to zero mean, unit variance
- Prevents features with large values from dominating

### 2. LSTM Architecture

**Why LSTM?**
- Engines degrade over time (temporal pattern)
- LSTM remembers long-term dependencies
- Bidirectional captures both forward and backward patterns

**Model Details:**
- **Input**: 30 time steps × 24 features
- **Hidden Layers**: 3 LSTM layers, 128 units each
- **Direction**: Bidirectional (2 directions)
- **Output**: Single RUL value
- **Total Params**: ~500,000 trainable parameters

### 3. Training Process

**Loss Function:** Mean Squared Error (MSE)
```
Loss = (1/N) Σ(predicted_RUL - actual_RUL)²
```

**Optimization:**
- Adam optimizer (adaptive learning rates)
- Initial learning rate: 0.001
- Reduces by 50% if validation loss plateaus
- Gradient clipping at norm=1.0

**Early Stopping:**
- Monitors validation loss
- Stops if no improvement for 15 epochs
- Restores best model weights

### 4. Evaluation

**Metrics:**
- **RMSE**: Overall prediction accuracy (lower is better)
- **MAE**: Average absolute error
- **R²**: How well predictions fit actuals (1.0 is perfect)
- **MAPE**: Percentage error

**Visualizations:**
- Predicted vs Actual scatter plots
- Residual analysis
- Error distribution by RUL range
- Individual engine trajectories

---

## 🎓 Training the Model

### Running Training

```powershell
python lstm_model/main.py
```

### What Happens During Training

1. **Initialization** (1-2 seconds)
   - Loads configuration
   - Creates model
   - Sets up optimizer

2. **Data Loading** (5-10 seconds)
   - Loads CSV files
   - Creates sequences
   - Fits StandardScaler
   - Creates DataLoaders

3. **Training Loop** (15-60 minutes)
   - For each epoch:
     - Train on training set
     - Validate on test set
     - Print progress
     - Update learning rate if needed
     - Check early stopping
   
4. **Final Evaluation** (10-20 seconds)
   - Loads best model
   - Generates plots
   - Saves predictions
   - Prints final metrics

### Expected Console Output

```
============================================================
LSTM RUL Prediction Configuration
============================================================
Sequence Length: 30
Hidden Size: 128
Number of Layers: 3
...

Loading training data...
Preparing training sequences...
Created 17732 sequences of shape (17732, 30, 24)

============================================================
Starting Training on cpu
============================================================
Epoch 1/100 (15.23s) - Train Loss: 3245.6789, ...
Epoch 2/100 (14.98s) - Train Loss: 2103.4521, ...
  → New best model! (Val Loss: 1987.1234)
...
Early stopping triggered after 45 epochs

Best validation loss: 187.4521
============================================================
```

### Output Files

After training:
```
lstm_model/
├── checkpoints/
│   ├── best_lstm_model.pt      # Trained model (~20 MB)
│   └── scaler.pkl               # Data scaler (~1 KB)
└── results/
    ├── training_history.png     # Loss curves
    ├── predictions_plot.png     # 4 evaluation plots
    ├── error_distribution.png   # Error analysis
    └── predictions.csv          # Detailed results
```

---

## 🔮 Making Predictions

### Using Inference Script

```powershell
python lstm_model/inference.py
```

This automatically:
1. Loads the trained model
2. Loads test data
3. Makes predictions
4. Calculates metrics
5. Generates visualizations
6. Saves results

### Using in Your Own Code

```python
from lstm_model.inference import RULPredictor

# Load model
predictor = RULPredictor('lstm_model/checkpoints/best_lstm_model.pt')

# Load data
predictor.load_test_data('data/processed/test_FD001_processed.csv')

# Predict all
predictions = predictor.predict()

# Predict single engine
pred, actual = predictor.predict_single_engine(engine_id=1)

# Evaluate
metrics = predictor.evaluate()

# Plot specific engine
predictor.plot_engine_trajectory(engine_id=1)
```

---

## 📊 Understanding the Results

### Metrics Explanation

**RMSE (Root Mean Squared Error)**
- Average prediction error in cycles
- Target: < 20 cycles
- Lower is better
- Example: RMSE = 15 means average error of ±15 cycles

**MAE (Mean Absolute Error)**
- Average of absolute errors
- Usually lower than RMSE
- Example: MAE = 12 means average ±12 cycles off

**R² Score**
- How well model fits data
- Range: 0 to 1
- Target: > 0.85
- 1.0 = perfect predictions

**MAPE (Mean Absolute Percentage Error)**
- Percentage error
- Example: 8% means predictions are 8% off on average

### Interpreting Plots

**1. Predicted vs Actual (top-left)**
- Points should cluster near red diagonal line
- Scattered points indicate poor predictions
- Points above line = overestimation
- Points below line = underestimation

**2. Residual Plot (top-right)**
- Should be random scatter around y=0
- Patterns indicate systematic bias
- Funnel shape = heteroscedasticity

**3. Residual Distribution (bottom-left)**
- Should be bell-shaped (normal distribution)
- Centered at 0
- Heavy tails = outliers present

**4. Time Series Comparison (bottom-right)**
- Blue (actual) and orange (predicted) should overlap
- Large gaps indicate poor predictions

---

## ⚙️ Customization

### Modify Hyperparameters

Edit `lstm_model/config.py`:

```python
class Config:
    # Try longer sequences
    SEQUENCE_LENGTH = 50
    
    # Bigger model for better accuracy
    HIDDEN_SIZE = 256
    NUM_LAYERS = 4
    
    # More training
    NUM_EPOCHS = 150
    
    # Smaller batches for limited memory
    BATCH_SIZE = 32
```

### Use Simpler Model

In `lstm_model/main.py`, change:

```python
# Replace this:
from model import LSTMRULPredictor

# With this:
from model import SimpleLSTMRULPredictor as LSTMRULPredictor
```

The simple model has:
- 2 layers instead of 3
- 64 hidden units instead of 128
- Unidirectional instead of bidirectional
- ~50K parameters instead of ~500K

### Adjust Learning Rate

In `config.py`:

```python
# Slower learning (more stable)
LEARNING_RATE = 0.0005

# Faster learning (might be unstable)
LEARNING_RATE = 0.002
```

---

## 🔧 Troubleshooting

### Problem: Out of Memory Error

**Symptoms:**
```
RuntimeError: [enforce fail at CPUAllocator.cpp:64] . DefaultCPUAllocator: can't allocate memory
```

**Solutions:**
1. Reduce batch size in `config.py`:
   ```python
   BATCH_SIZE = 16  # or even 8
   ```

2. Use simpler model:
   ```python
   HIDDEN_SIZE = 64
   NUM_LAYERS = 2
   ```

3. Reduce sequence length:
   ```python
   SEQUENCE_LENGTH = 20
   ```

### Problem: Training Too Slow

**Solutions:**
1. Reduce epochs:
   ```python
   NUM_EPOCHS = 50
   ```

2. Use GPU (if available):
   - PyTorch will automatically use CUDA
   - Check with: `python -c "import torch; print(torch.cuda.is_available())"`

3. Increase batch size (if memory allows):
   ```python
   BATCH_SIZE = 128
   ```

### Problem: Poor Predictions (High RMSE)

**Solutions:**
1. Train longer:
   ```python
   NUM_EPOCHS = 150
   PATIENCE = 20
   ```

2. Increase model capacity:
   ```python
   HIDDEN_SIZE = 256
   NUM_LAYERS = 4
   ```

3. Adjust learning rate:
   ```python
   LEARNING_RATE = 0.0005
   ```

4. Check data:
   - Verify RUL column is correct
   - Ensure no missing values
   - Confirm feature scaling

### Problem: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```powershell
pip install -r lstm_model/requirements.txt
```

### Problem: Data File Not Found

**Symptoms:**
```
FileNotFoundError: data/processed/train_FD001_processed.csv
```

**Solution:**
- Run preprocessing notebook first
- Ensure you're in the correct directory
- Check file paths in `config.py`

---

## 📚 Additional Resources

### Understanding LSTM
- [Colah's LSTM Blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Great visual explanation of LSTM architecture

### PyTorch Tutorials
- [Official PyTorch LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)

### NASA C-MAPSS Dataset
- [Dataset Documentation](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- Original dataset and papers

---

## 🎯 Best Practices

1. **Always start with default settings** on first run
2. **Monitor training progress** - loss should decrease
3. **Check plots** - they tell you if model is working
4. **Save your best models** - training takes time
5. **Document changes** - if you modify code
6. **Use version control** - Git is your friend

---

## 📝 Summary

You now have a complete LSTM system for RUL prediction:

✅ Modular, well-documented code  
✅ Easy-to-use training pipeline  
✅ Comprehensive evaluation tools  
✅ Flexible configuration  
✅ Production-ready inference  
✅ Interactive quickstart menu  

**Next Steps:**
1. Install dependencies
2. Run training
3. Analyze results
4. Experiment with hyperparameters
5. Compare with other models (Random Forest, LightGBM)

Good luck with your project, Tony! 🚀

---

**Created by Tony for ENGG2112**  
**October 2025**
