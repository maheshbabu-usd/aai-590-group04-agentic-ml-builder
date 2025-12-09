
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. Load Data
# Generating synthetic data with outliers
np.random.seed(42)
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X_inliers, X_outliers]

df = pd.DataFrame(X, columns=['x1', 'x2'])

# 2. Train Isolation Forest
# contamination: proportion of outliers in the data set
model = IsolationForest(max_samples=100, contamination=0.1, random_state=42)
model.fit(df)

# 3. Predict (1: Inlier, -1: Outlier)
df['anomaly_score'] = model.decision_function(df)
df['prediction'] = model.predict(df)

print(df['prediction'].value_counts())

# 4. Visualize
plt.figure(figsize=(10, 6))
plt.scatter(df.loc[df['prediction'] == 1, 'x1'], df.loc[df['prediction'] == 1, 'x2'], c='blue', label='Inliers')
plt.scatter(df.loc[df['prediction'] == -1, 'x1'], df.loc[df['prediction'] == -1, 'x2'], c='red', label='Outliers')
plt.title('Isolation Forest Anomaly Detection')
plt.legend()
plt.show()
