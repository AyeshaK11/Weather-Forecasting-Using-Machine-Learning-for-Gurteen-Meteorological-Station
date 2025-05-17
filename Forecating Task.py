import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

# Load data
df = pd.read_csv('C:/Users/Ayesha/Downloads/09052025/dly1475.csv')
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df = df.sort_values('date')

# Convert columns to numeric
cols_to_use = ['maxtp', 'mintp', 'gmin', 'rain', 'cbl', 'wdsp', 'hm', 'ddhm', 'hg',
               'soil', 'pe', 'evap', 'smd_wd', 'smd_md', 'smd_pd', 'glorad']
for col in cols_to_use:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)

# Detailed EDA
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df[cols_to_use].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Time-series plot
plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['maxtp'], label="Max Temperature")
plt.title("Max Temperature Over Time")
plt.xlabel("Date")
plt.ylabel("Max Temperature")
plt.legend()
plt.show()

# Histogram for each feature
df[cols_to_use].hist(bins=20, figsize=(12, 8), layout=(4, 4))
plt.suptitle("Histograms of Features")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Boxplot for each feature
plt.figure(figsize=(12, 8))
df[cols_to_use].plot(kind='box', subplots=True, layout=(4, 4), figsize=(16, 12), sharex=False, sharey=False)
plt.suptitle("Boxplots of Features")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Pairplot for selected features (to avoid too many plots)
selected_features = ['maxtp', 'mintp', 'rain', 'wdsp', 'soil']
sns.pairplot(df[selected_features])
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# Prepare data
X = df[cols_to_use].drop(columns=['maxtp'])
y = df['maxtp']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Use time-based splitting
train_size = int(len(df) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train and evaluate models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "R2": r2_score(y_test, y_pred)
    }

    # Forecasts
    last_7 = X_scaled[-7:]
    forecast_1 = model.predict([last_7[-1]])[0]
    forecast_7 = model.predict(last_7).mean()
    results[name]["1-Day Forecast"] = forecast_1
    results[name]["7-Day Forecast"] = forecast_7

# ANN Model
ann = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
ann.compile(optimizer='adam', loss='mse')
ann.fit(X_train, y_train, epochs=50, verbose=0)
y_pred_ann = ann.predict(X_test).flatten()

results['ANN'] = {
    "MAE": mean_absolute_error(y_test, y_pred_ann),
    "RMSE": mean_squared_error(y_test, y_pred_ann, squared=False),
    "R2": r2_score(y_test, y_pred_ann),
    "1-Day Forecast": ann.predict(last_7[-1].reshape(1, -1))[0][0],
    "7-Day Forecast": ann.predict(last_7).mean()
}

# Reshape for RNNs (samples, timesteps, features)
X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
X_last_7_rnn = last_7.reshape((7, 1, X_train.shape[1]))

# LSTM Model
lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
    Dense(1)
])
lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train_rnn, y_train, epochs=50, verbose=0)
y_pred_lstm = lstm.predict(X_test_rnn).flatten()

results['LSTM'] = {
    "MAE": mean_absolute_error(y_test, y_pred_lstm),
    "RMSE": mean_squared_error(y_test, y_pred_lstm, squared=False),
    "R2": r2_score(y_test, y_pred_lstm),
    "1-Day Forecast": lstm.predict(X_last_7_rnn[-1:])[0][0],
    "7-Day Forecast": lstm.predict(X_last_7_rnn).mean()
}

# RNN Model
rnn = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(1, X_train.shape[1])),
    Dense(1)
])
rnn.compile(optimizer='adam', loss='mse')
rnn.fit(X_train_rnn, y_train, epochs=50, verbose=0)
y_pred_rnn = rnn.predict(X_test_rnn).flatten()

results['RNN'] = {
    "MAE": mean_absolute_error(y_test, y_pred_rnn),
    "RMSE": mean_squared_error(y_test, y_pred_rnn, squared=False),
    "R2": r2_score(y_test, y_pred_rnn),
    "1-Day Forecast": rnn.predict(X_last_7_rnn[-1:])[0][0],
    "7-Day Forecast": rnn.predict(X_last_7_rnn).mean()
}

# Visualize results
for model, metrics in results.items():
    print(f"\n{model} Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# Plot all graphs at the end
plt.figure(figsize=(10, 6))
plt.plot(df['date'][train_size:], y_test.values, label="Actual")
plt.plot(df['date'][train_size:], y_pred_lstm, label="Predicted (LSTM)")
plt.plot(df['date'][train_size:], y_pred_ann, label="Predicted (ANN)")
plt.plot(df['date'][train_size:], y_pred_rnn, label="Predicted (RNN)")
plt.title("Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Max Temperature")
plt.legend()
plt.show()