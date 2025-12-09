
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load example data
df = sns.load_dataset("penguins")

# 1. Pair Plot (Scatter matrix)
sns.pairplot(df, hue="species")
plt.title("Pair Plot")
plt.show()

# 2. Violin Plot (Distribution)
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="species", y="flipper_length_mm", inner="quart")
plt.title("Violin Plot")
plt.show()

# 3. Heatmap (Correlation)
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 4. Joint Plot (Bivariate + Univariate)
sns.jointplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species", kind="kde")
plt.show()
