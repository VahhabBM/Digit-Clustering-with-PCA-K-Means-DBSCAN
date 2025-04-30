#loading data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

digits = load_digits()
data = digits.data
labels = digits.target
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
# k_means inertia plot
inertia = []
k_values = range(1, 21)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='-')
plt.title("Elbow")
plt.xlabel("Number of Clusters or number of (k)")
plt.ylabel("Inertia")
plt.grid()
plt.show()
#applying pca on k_means

pca_2d = PCA(n_components=2)
data_2d = pca_2d.fit_transform(data_scaled)
pca_3d = PCA(n_components=3)
data_3d = pca_3d.fit_transform(data_scaled)
best_k = 10
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)
# 2d plot of k_means:
plt.figure(figsize=(8, 8))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=kmeans_labels, cmap='viridis', s=10)
plt.title("K-means(2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")
plt.grid()
plt.show()
# 3d plot of k_means:
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=kmeans_labels, cmap='viridis', s=10)
ax.set_title("K-means(3D)")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
fig.colorbar(scatter, ax=ax, label="Cluster Label")
plt.show()
# and now DBSCAN!:
# finding best eps & minpts
eps_values = [0.5, 1.0, 1.5]
min_samples_values = [5, 10, 20]
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(data_scaled)
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
#DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(data_scaled)
# plotting the result in 2d plot:
plt.figure(figsize=(8, 8))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=dbscan_labels, cmap='viridis', s=10)
plt.title("DBSCAN(2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")
plt.grid()
plt.show()

# 3d plot of PCA:
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=dbscan_labels, cmap='viridis', s=10)
ax.set_title("DBSCAN(3D)")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
fig.colorbar(scatter, ax=ax, label="Cluster Label")
plt.show()


