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


data = pd.read_csv(r"C:\Users\HP\Documents\GitHub\FULL-STACK-B-6-MON-to-TUE-7pm-to-9pm\FinalAssessment\AI-Impact-on-Jobs-2030\AI_Impact_on_Jobs_2030.csv")
print(data.head(7))
print(data.shape)
print(data.dtypes)
print(data.isnull().sum())
print(data.describe())
print(data['Risk_Category'].value_counts())


"""
EDA — Exploratory Data Analysis
We visualize the distribution of key columns to understand the data better before building any model.
"""

plt.subplots(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.histplot(data['Automation_Probability_2030'], bins=30, kde=True, color='steelblue')
plt.title("Automation Probability Distribution")

plt.subplot(1, 2, 2)
sns.countplot(data=data, x='Risk_Category', palette='Set2')
plt.title("Risk Category Counts")
plt.show()

plt.subplots(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=data, x='AI_Exposure_Index', y='Automation_Probability_2030',
                hue='Risk_Category', palette='viridis', alpha=0.6)
plt.title("AI Exposure vs Automation Probability")

plt.subplot(1, 2, 2)
sns.boxplot(data=data, x='Risk_Category', y='Average_Salary', palette='pastel')
plt.title("Salary by Risk Category")
plt.show()

corr_cols = ['Average_Salary', 'AI_Exposure_Index', 'Tech_Growth_Factor',
             'Automation_Probability_2030', 'Skill_1', 'Skill_2', 'Skill_3']
plt.figure(figsize=(8, 6))
sns.heatmap(data[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


"""
Data Preprocessing
Categorical columns are encoded using LabelEncoder so the model can work with numerical values.
The target column is Automation_Probability_2030.
We select key feature columns and scale everything to the [0, 1] range using MinMaxScaler.
"""

le = LabelEncoder()
data['Risk_Category_Encoded'] = le.fit_transform(data['Risk_Category'])
data['Education_Level_Encoded'] = le.fit_transform(data['Education_Level'])
data['Job_Title_Encoded'] = le.fit_transform(data['Job_Title'])

feature_cols = ['AI_Exposure_Index', 'Tech_Growth_Factor', 'Average_Salary',
                'Years_Experience', 'Skill_1', 'Skill_2', 'Skill_3',
                'Skill_4', 'Skill_5', 'Risk_Category_Encoded']
target_col = 'Automation_Probability_2030'

df_sorted = data.sort_values('Years_Experience').reset_index(drop=True)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df_sorted[feature_cols])
y_scaled = scaler_y.fit_transform(df_sorted[[target_col]])

print("Feature Matrix Shape:", X_scaled.shape)
print("Target Shape:", y_scaled.shape)


"""
Creating Time-Series Sequences
Since RNN, LSTM and GRU are sequential models they expect data in the form of (samples, timesteps, features).
We use a sliding window of size 10 to create input sequences from the sorted dataset.
"""

SEQUENCE_LENGTH = 10

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


"""
Model 1 — Simple RNN
SimpleRNN is the most basic recurrent model. It reads the sequence one step at a time and passes a hidden state forward.
The problem with SimpleRNN is that it tends to forget information from earlier time steps in long sequences.
We stack two RNN layers followed by Dense layers with Dropout to reduce overfitting.
"""

rnn_model = Sequential()
rnn_model.add(SimpleRNN(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(feature_cols))))
rnn_model.add(Dropout(0.2))
rnn_model.add(BatchNormalization())
rnn_model.add(SimpleRNN(32))
rnn_model.add(Dropout(0.2))
rnn_model.add(Dense(16, activation='relu'))
rnn_model.add(Dense(1, activation='sigmoid'))

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
y_test_inv  = scaler_y.inverse_transform(y_test)

rnn_rmse = np.sqrt(mean_squared_error(y_test_inv, rnn_pred_inv))
rnn_mae  = mean_absolute_error(y_test_inv, rnn_pred_inv)
rnn_r2   = r2_score(y_test_inv, rnn_pred_inv)

print("RNN Results:")
print(f"  RMSE : {rnn_rmse:.4f}")
print(f"  MAE  : {rnn_mae:.4f}")
print(f"  R²   : {rnn_r2:.4f}")


"""
Model 2 — LSTM (Long Short-Term Memory)
LSTM solves the vanishing gradient problem that SimpleRNN faces.
It has three gates — forget gate, input gate and output gate — that control what information to keep or discard.
This makes LSTM much better at learning long-range dependencies in sequential data.
"""

lstm_model = Sequential()
lstm_model.add(LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(feature_cols))))
lstm_model.add(Dropout(0.2))
lstm_model.add(BatchNormalization())
lstm_model.add(LSTM(32))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(16, activation='relu'))
lstm_model.add(Dense(1, activation='sigmoid'))

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


"""
Model 3 — GRU (Gated Recurrent Unit)
GRU is a lighter and faster alternative to LSTM.
It uses only two gates — reset gate and update gate — which means fewer parameters and faster training.
On smaller datasets GRU often matches or outperforms LSTM because it is less prone to overfitting.
"""

gru_model = Sequential()
gru_model.add(GRU(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(feature_cols))))
gru_model.add(Dropout(0.2))
gru_model.add(BatchNormalization())
gru_model.add(GRU(32))
gru_model.add(Dropout(0.2))
gru_model.add(Dense(16, activation='relu'))
gru_model.add(Dense(1, activation='sigmoid'))

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


"""
Training Loss Curves
We plot the training and validation loss for all three models side by side.
This helps us understand how well each model learned over epochs and whether it overfit or converged properly.
"""

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


"""
Actual vs Predicted Plots
We compare the real Automation_Probability_2030 values with what each model predicted on the test set.
A close match between the two lines means the model learned the pattern well.
"""

x_idx = np.arange(len(y_test_inv))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Actual vs Predicted — Automation Probability 2030", fontsize=14, fontweight='bold')

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
    ax.set_ylabel("Automation Probability")
    ax.legend()

plt.tight_layout()
plt.show()


"""
Metric Comparison — RNN vs LSTM vs GRU
We compare all three models using RMSE, MAE and R² score.
RMSE and MAE measure how far the predictions are from actual values. Lower is better.
R² tells us how much variance the model explains. Closer to 1.0 is better.
"""

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