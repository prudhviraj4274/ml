import numpy as np
from sklearn.mixture import GaussianMixture

# Dataset
X = np.array([
    [1], [2], [3], [4], [5],
    [10], [11], [12], [13], [14]
])

# EM algorithm using Gaussian Mixture Model
model = GaussianMixture(n_components=2, random_state=42)

# Train
model.fit(X)

# Predict clusters
clusters = model.predict(X)

print("Data:")
print(X.ravel())

print("\nCluster Labels:")
print(clusters)

print("\nCluster Means:")
print(model.means_.ravel())

print("\nCluster Probabilities:")
print(model.predict_proba(X))
