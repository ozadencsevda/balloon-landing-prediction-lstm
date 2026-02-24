# 🎈 Balloon Landing Point Prediction with LSTM

A deep learning project that predicts the landing coordinates of hot air balloons using time-series flight data. Built with TensorFlow/Keras, the model processes the first 10 steps of each training flight to forecast the final latitude and longitude at landing.

---

## Overview

Hot air balloon flights are highly sensitive to atmospheric conditions — wind speed, direction, temperature, and pressure all influence the balloon's trajectory. This project frames landing point prediction as a **sequence-to-point regression problem**: given the first 10 recorded positions and sensor readings of a flight, predict where the balloon will land.

The model uses a stacked LSTM architecture to capture temporal dependencies in the flight trajectory, trained on real flight data collected from training flights.

---

## Problem Definition

- **Input:** First 10 timesteps of a flight (position, altitude, atmospheric readings, derived motion features)
- **Output:** Final landing coordinates (latitude, longitude)
- **Task:** Regression — predicting continuous geographic coordinates

---

## Feature Engineering

Raw flight data is enriched with derived features that capture the balloon's motion dynamics:

| Feature | Description |
|---|---|
| `latitude`, `longitude`, `altitude` | GPS position at each timestep |
| `temperature_6am`, `pressure_6am` | Atmospheric conditions |
| `wind_speed_6am`, `wind_dir_6am` | Wind measurements |
| `delta_time` | Time elapsed between consecutive readings |
| `horizontal_speed` | Computed via Haversine formula from consecutive GPS points |
| `vertical_speed` | Altitude change rate (m/s) |
| `wind_u`, `wind_v` | Wind decomposed into U (east-west) and V (north-south) components |

Haversine formula is used for horizontal speed — the spherical Earth approximation is necessary because standard Euclidean distance is inaccurate at geographic scales.

---

## Model Architecture

```
Input → LSTM(256, return_sequences=True) → Dropout(0.1)
      → LSTM(128, return_sequences=False) → Dropout(0.1)
      → Dense(64, relu)
      → Dense(2)  ← [latitude, longitude]
```

**Training configuration:**
- Optimizer: Adam (lr=0.0005)
- Loss: Mean Squared Error
- Callbacks: EarlyStopping (patience=10), ModelCheckpoint (best val_loss)
- Both input features and output coordinates are standardized with separate `StandardScaler` instances

---

## Results & Reflections

The model converged but produced higher-than-expected prediction error on validation data. Key factors that likely contributed:

- **Limited sequence length:** Only the first 10 steps are used — early trajectory may not capture enough directional momentum
- **Weather data granularity:** Wind readings are recorded at 6am and do not reflect real-time atmospheric changes during flight
- **Dataset size:** The number of unique flights constrains the model's ability to generalize across varied wind conditions
- **Inherent unpredictability:** Hot air balloon trajectories are highly sensitive to micro-weather variations that sensor data alone cannot fully capture

Despite the results, this project demonstrates a complete deep learning pipeline on a real-world time-series problem: domain-aware feature engineering, sequence modeling, proper train/validation splitting, and output normalization.

---

## Dataset

Real hot air balloon training flight data (`enriched_train_flights.csv`). Update the file path in the notebook to match your environment:

```python
# Local environment
df = pd.read_csv('enriched_train_flights.csv')

# Google Colab
df = pd.read_csv('/content/drive/MyDrive/enriched_train_flights.csv')
```

---

## Requirements

```
tensorflow
pandas
numpy
scikit-learn
```

Install with:

```bash
pip install tensorflow pandas numpy scikit-learn
```

---

## Tech Stack

| | |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Data Processing | pandas, numpy |
| Preprocessing | scikit-learn (StandardScaler) |
| Environment | Google Colab / Jupyter Notebook |

---

## License

MIT
