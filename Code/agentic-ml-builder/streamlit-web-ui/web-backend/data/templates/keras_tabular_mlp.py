
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load Data
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 2. Preprocessing
X = df.drop(columns=['target'])
y = df['target']

# Scale features (Critical for Neural Networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Define Model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dense(32, activation='relu'),
    layers.Dense(1) # Linear output for regression
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# 4. Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=50,
    verbose=1
)

# 5. Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.4f}")

# 6. Visualize
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.title('Training Loss')
plt.show()
