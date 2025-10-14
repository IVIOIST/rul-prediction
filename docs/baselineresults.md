### Baseline Model Performance Summary (30-Cycle Lookback $\to$ Predict Last-Cycle RUL)

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
|--------|----------------|----------------|----------|-----------|
| **Random Forest Regressor** | 27.72 | 38.40 | 19.02 | 25.43 |
| **LightGBM Regressor** | **24.77** | **35.21** | **16.94** | **23.27** |

---

### Key Observations 🫣🔑

- **LightGBM consistently outperforms Random Forest**, achieving lower error across both MAE and RMSE.
- Both models **systematically underpredict RUL at higher values**, flattening around ~150–180 cycles regardless of the true RUL.
- **Error distributions are slightly left-skewed**, meaning the models tend to predict *shorter* RUL than actual (safer but conservative).
- **Test performance is better than validation**, likely due to more homogeneous degradation patterns in test trajectories.

---

### Proposed Next Steps ⏭️🪜

**Model Improvements**

- Progress from tree ensembles to **sequence-aware models**:
- LSTM / GRU (recurrent models for temporal dependencies)
- 1D CNN / Temporal Convolutional Networks
- Transformer-based regressors

**Feature Engineering**
- Add **trend-based features** (rolling means, deltas, exponential decay)
- Introduce **per-unit normalization or condition labels**
- Experiment with **RUL clipping (e.g. max 125 cycles)** to stabilise the tail

**Training & Evaluation Enhancements**
- Track **error as a function of engine life / cycle position**

