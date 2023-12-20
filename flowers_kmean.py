import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load labeled images and labels from a .mat file
jpg_folder = r'C:\Users\dell\Downloads\jpg'

# Load and resize images
def load_and_resize_images(folder_path, target_size=(32, 32), num_images=None):
    images = []
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            images.append(img)

            count += 1
            if num_images is not None and count >= num_images:
                break
    return images

images = load_and_resize_images(jpg_folder, target_size=(32, 32))  # Remove num_images to load all available images

# Flatten and normalize the pixel values of images
X = np.array(images)
X_flat = X.reshape(len(images), -1)  # Flatten the array
X_flat = X_flat / 255.0  # Normalize pixel values to the range [0, 1]

# Adjust the range and step for exploring different cluster numbers
num_clusters_range = range(2, 102, 20)

# Record silhouette scores for different numbers of clusters
silhouette_scores = []

# Record cluster labels for the best silhouette score
best_cluster_labels = None
best_n_clusters = None

for n_clusters in num_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(X_flat)
    silhouette_avg = silhouette_score(X_flat, cluster_labels)
    silhouette_scores.append(silhouette_avg)

    # Record the cluster labels for the best silhouette score
    if best_cluster_labels is None or silhouette_avg > max(silhouette_scores[:-1]):
        best_cluster_labels = cluster_labels
        best_n_clusters = n_clusters

# Visualize the silhouette scores
plt.figure(figsize=(12, 5))
print(max(silhouette_scores[:-1]))
# Silhouette Score Plot
plt.subplot(1, 2, 1)
plt.plot(num_clusters_range, silhouette_scores, marker='o')
plt.title('K-Means Silhouette Analysis')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

# Visualize the clusters with labels for the best silhouette score
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_flat)

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_cluster_labels, s=50, cmap='viridis', label='K-Means Clusters')
plt.title(f'K-Means Clustering (Best Silhouette Score with {best_n_clusters} clusters)')
plt.legend()

plt.tight_layout()
plt.show()
