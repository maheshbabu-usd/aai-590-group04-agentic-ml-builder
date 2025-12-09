
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import numpy as np

def train_xgboost(data_path, target_col, task='classification'):
    # Load data
    try:
        df = pd.read_csv(data_path)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables (simple encoding)
        X = pd.get_dummies(X)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task == 'classification':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.4f}")
    else:
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"RMSE: {rmse:.4f}")

    # Feature Importance
    importance = model.feature_importances_
    features = X.columns
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    print("\nTop 5 Features:")
    print(feat_imp.head())

    model.save_model("xgboost_model.json")
    print("Model saved to xgboost_model.json")

if __name__ == "__main__":
    # train_xgboost('data.csv', 'target')
    pass
