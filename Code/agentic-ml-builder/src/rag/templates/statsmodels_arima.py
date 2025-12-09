
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# 1. Load Data
# df = pd.read_csv('your_timeseries.csv', parse_dates=['date_col'], index_col='date_col')
# Synthetic data
date_rng = pd.date_range(start='1/1/2023', end='1/04/2023', freq='H')
ts = pd.DataFrame(date_rng, columns=['date'])
ts['data'] = np.random.randint(0,100,size=(len(date_rng)))
ts = ts.set_index('date')
ts['data'] += range(len(ts)) # Add trend

# 2. Split (Time Series Split)
train_size = int(len(ts) * 0.8)
train, test = ts[0:train_size], ts[train_size:len(ts)]

# 3. Train ARIMA
# Order (p,d,q) needs to be tuned or use auto_arima from pmdarima
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()
print(model_fit.summary())

# 4. Forecast
forecast = model_fit.forecast(steps=len(test))

# 5. Evaluate
mse = mean_squared_error(test, forecast)
print(f'MSE: {mse:.4f}')

# 6. Visualize
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, forecast, label='Forecast')
plt.legend()
plt.title('ARIMA Forecast')
plt.show()
