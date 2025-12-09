
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load Data
# df = pd.read_csv('data.csv')
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
df = pd.DataFrame(X, columns=['x1', 'x2'])
df['target'] = y

target_col = 'target'
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 3. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
