
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 1. Load Data
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
# No target needed for unsupervised, but keeping for visualization
labels = data.target 

# 2. Preprocessing
# PCA requires scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Apply PCA
n_components = 2
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(X_scaled)

# Creating a DataFrame for visualization
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['label'] = labels

# 4. Explained Variance
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.4f}")

# 5. Visualize
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='label', palette='viridis', data=pca_df, alpha=0.7)
plt.title(f'PCA: {sum(pca.explained_variance_ratio_)*100:.2f}% Variance Explained')
plt.show()
