
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1. Load Data
# Prophet requires columns 'ds' (date) and 'y' (value)
# df = pd.read_csv('data.csv')
date_rng = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
df = pd.DataFrame({'ds': date_rng, 'y': np.random.randn(len(date_rng)) + np.arange(len(date_rng))*0.1})

# 2. Train
m = Prophet()
m.fit(df)

# 3. Make Future Dataframe
future = m.make_future_dataframe(periods=365) # Forecast 1 year

# 4. Predict
forecast = m.predict(future)

# 5. Visualize
fig1 = m.plot(forecast)
plt.title("Prophet Forecast")
plt.show()

fig2 = m.plot_components(forecast)
plt.show()
