# Stock Return Prediction — LSTM vs. Simple Baselines

Predicting next-day stock returns for 442 companies from 12 years of daily return data, built as the final project for my MSc Deep Learning course. The project pairs a tuned PyTorch LSTM against a non-ML baseline and uses interpretability tooling to explain the result.

**Key finding:** a weighted moving average (MSE **2.78**) outperformed the Optuna-tuned LSTM (MSE **2.98**) — a result consistent with the efficient-market intuition that daily returns are dominated by noise. The analysis explains *why* the simpler model won rather than just reporting the numbers.

## What it does

- **Preprocessing** — per-company `StandardScaler` normalisation to put differing volatility levels on a comparable scale; sliding-window sequence generation with a configurable lookback.
- **Model** — stacked LSTM with dropout regularisation and a linear output head (PyTorch).
- **Hyperparameter tuning** — Optuna search over hidden size, layers, dropout, learning rate, lookback, and batch size (20 trials).
- **Baseline comparison** — a recency-weighted moving average as a deliberately simple, non-ML benchmark.
- **Interpretability** — Captum Integrated Gradients to attribute each prediction across the input window, with a near-zero convergence delta confirming attribution accuracy.

## Results

| Model | MSE |
|---|---|
| Optuna-tuned LSTM | 2.98 |
| Weighted moving average | **2.78** |

The LSTM collapsed toward near-zero predictions for most companies — the expected failure mode when a model cannot find exploitable structure in noisy return data. Integrated Gradients showed no single day dominating the prediction (attributions ~0.001–0.01), reinforcing the same conclusion.

## Stack

`PyTorch` · `Optuna` · `Captum` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib`

## Notes

The dataset was provided as part of the course and is not redistributed here, so the notebook is intended to be read rather than re-run end-to-end.
