import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

data = pd.read_csv(r"C:\Users\HP\Documents\GitHub\FULL-STACK-B-6-MON-to-TUE-7pm-to-9pm\FinalAssessment\AI-Impact-on-Jobs-2030\ai_financial_market_daily_realistic_synthetic.csv")
print(data.head(7))
print(data.shape)
print(data.dtypes)
print(data.isnull().sum())
print(data.describe())
print(data['Company'].value_counts())

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date').reset_index(drop=True)

plt.subplots(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.histplot(data['Stock_Impact_%'], bins=40, kde=True, color='steelblue')
plt.title("Stock Impact % Distribution")
plt.subplot(1, 2, 2)
sns.histplot(data['AI_Revenue_USD_Mn'], bins=40, kde=True, color='green')
plt.title("AI Revenue Distribution")
plt.show()

plt.figure(figsize=(14, 5))
for company, color in zip(['OpenAI', 'Google', 'Meta'], ['blue', 'red', 'green']):
    subset = data[data['Company'] == company]
    plt.plot(subset['Date'], subset['AI_Revenue_USD_Mn'], label=company, color=color, alpha=0.7)
plt.title("AI Revenue Over Time by Company")
plt.xlabel("Date")
plt.ylabel("AI Revenue (USD Mn)")
plt.legend()
plt.show()

plt.subplots(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=data, x='Company', y='Stock_Impact_%', palette='Set2')
plt.title("Stock Impact % by Company")

plt.subplot(1, 2, 2)
sns.scatterplot(data=data, x='R&D_Spending_USD_Mn', y='AI_Revenue_USD_Mn',
                hue='Company', palette='viridis', alpha=0.5)
plt.title("R&D Spending vs AI Revenue")
plt.show()

corr_cols = ['R&D_Spending_USD_Mn', 'AI_Revenue_USD_Mn', 'AI_Revenue_Growth_%', 'Stock_Impact_%']
plt.figure(figsize=(7, 5))
sns.heatmap(data[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

le = LabelEncoder()
data['Company_Encoded'] = le.fit_transform(data['Company'])
data = data.drop(columns=['Event', 'Date', 'Company'])

feature_cols = ['R&D_Spending_USD_Mn', 'AI_Revenue_USD_Mn',
                'AI_Revenue_Growth_%', 'Company_Encoded']
target_col = 'Stock_Impact_%'

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(data[feature_cols])
y_scaled = scaler_y.fit_transform(data[[target_col]])

print("Feature Matrix Shape:", X_scaled.shape)
print("Target Shape:", y_scaled.shape)

SEQUENCE_LENGTH = 30

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQUENCE_LENGTH)

print("Sequence Shape (samples, timesteps, features):", X_seq.shape)
print("Target Shape:", y_seq.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

print("Train Samples:", X_train.shape[0])
print("Test  Samples:", X_test.shape[0])

print("RNN")
rnn_model = Sequential()
rnn_model.add(SimpleRNN(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(feature_cols))))
rnn_model.add(Dropout(0.2))
rnn_model.add(BatchNormalization())
rnn_model.add(SimpleRNN(32))
rnn_model.add(Dropout(0.2))
rnn_model.add(Dense(16, activation='relu'))
rnn_model.add(Dense(1))

rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print(rnn_model.summary())

rnn_history = rnn_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

rnn_pred = rnn_model.predict(X_test)
rnn_pred_inv = scaler_y.inverse_transform(rnn_pred)
y_test_inv   = scaler_y.inverse_transform(y_test)

rnn_rmse = np.sqrt(mean_squared_error(y_test_inv, rnn_pred_inv))
rnn_mae  = mean_absolute_error(y_test_inv, rnn_pred_inv)
rnn_r2   = r2_score(y_test_inv, rnn_pred_inv)

print("RNN Results:")
print(f"  RMSE : {rnn_rmse:.4f}")
print(f"  MAE  : {rnn_mae:.4f}")
print(f"  R²   : {rnn_r2:.4f}")

print("LSTM")

lstm_model = Sequential()
lstm_model.add(LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(feature_cols))))
lstm_model.add(Dropout(0.2))
lstm_model.add(BatchNormalization())
lstm_model.add(LSTM(32))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(16, activation='relu'))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print(lstm_model.summary())

lstm_history = lstm_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

lstm_pred = lstm_model.predict(X_test)
lstm_pred_inv = scaler_y.inverse_transform(lstm_pred)

lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, lstm_pred_inv))
lstm_mae  = mean_absolute_error(y_test_inv, lstm_pred_inv)
lstm_r2   = r2_score(y_test_inv, lstm_pred_inv)

print("LSTM Results:")
print(f"  RMSE : {lstm_rmse:.4f}")
print(f"  MAE  : {lstm_mae:.4f}")
print(f"  R²   : {lstm_r2:.4f}")

print("Part 3 GRU")

gru_model = Sequential()
gru_model.add(GRU(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(feature_cols))))
gru_model.add(Dropout(0.2))
gru_model.add(BatchNormalization())
gru_model.add(GRU(32))
gru_model.add(Dropout(0.2))
gru_model.add(Dense(16, activation='relu'))
gru_model.add(Dense(1))

gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print(gru_model.summary())

gru_history = gru_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

gru_pred = gru_model.predict(X_test)
gru_pred_inv = scaler_y.inverse_transform(gru_pred)

gru_rmse = np.sqrt(mean_squared_error(y_test_inv, gru_pred_inv))
gru_mae  = mean_absolute_error(y_test_inv, gru_pred_inv)
gru_r2   = r2_score(y_test_inv, gru_pred_inv)

print("GRU Results:")
print(f"  RMSE : {gru_rmse:.4f}")
print(f"  MAE  : {gru_mae:.4f}")
print(f"  R²   : {gru_r2:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Training vs Validation Loss", fontsize=14, fontweight='bold')

for ax, history, name, color in zip(
    axes,
    [rnn_history, lstm_history, gru_history],
    ['RNN', 'LSTM', 'GRU'],
    ['#e74c3c', '#3498db', '#2ecc71']
):
    ax.plot(history.history['loss'], label='Train Loss', color=color)
    ax.plot(history.history['val_loss'], label='Val Loss', color=color, linestyle='--')
    ax.set_title(f"{name} Loss Curve")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()

plt.tight_layout()
plt.show()

x_idx = np.arange(len(y_test_inv))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Actual vs Predicted — Stock Impact %", fontsize=14, fontweight='bold')

for ax, preds, name, color in zip(
    axes,
    [rnn_pred_inv, lstm_pred_inv, gru_pred_inv],
    ['RNN', 'LSTM', 'GRU'],
    ['#e74c3c', '#3498db', '#2ecc71']
):
    ax.plot(x_idx[:100], y_test_inv[:100], label='Actual', color='black', linewidth=2)
    ax.plot(x_idx[:100], preds[:100], label=f'{name} Predicted', color=color, linestyle='--')
    ax.set_title(f"{name} — Actual vs Predicted")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Stock Impact %")
    ax.legend()

plt.tight_layout()
plt.show()

model_names = ['RNN', 'LSTM', 'GRU']
rmse_vals = [rnn_rmse, lstm_rmse, gru_rmse]
mae_vals  = [rnn_mae,  lstm_mae,  gru_mae]
r2_vals   = [rnn_r2,   lstm_r2,   gru_r2]
colors    = ['#e74c3c', '#3498db', '#2ecc71']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Metric Comparison", fontsize=14, fontweight='bold')

for ax, vals, title in zip(
    axes,
    [rmse_vals, mae_vals, r2_vals],
    ['RMSE (lower is better)', 'MAE (lower is better)', 'R² Score (higher is better)']
):
    sns.barplot(x=model_names, y=vals, palette=colors, ax=ax)
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.001, f"{v:.4f}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()

summary_df = pd.DataFrame({
    'Model': model_names,
    'RMSE':  [f"{v:.4f}" for v in rmse_vals],
    'MAE':   [f"{v:.4f}" for v in mae_vals],
    'R²':    [f"{v:.4f}" for v in r2_vals]
})
print(summary_df.to_string(index=False))
