from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import numpy as np
from sklearn.datasets import make_blobs

def gmm(X_aniso):
    # データの標準化
    X_norm = StandardScaler().fit_transform(X=X_aniso)

    # k-means cluster適用
    kmeans = KMeans(n_clusters=3, random_state=5)
    kmeans.fit(X_norm)
    kmeans_y_pred = kmeans.predict(X_norm)

    # ガウス混合モデルの適用(混合要素数は３、共分散の形式はfull)
    gmm = GaussianMixture(
        n_components=3,
        covariance_type='full',
        random_state=5
    )
    gmm.fit(X_norm)
    gmm_y_pred = gmm.predict(X_norm)

    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.title("(a) true cluster")
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y)
    plt.subplot(132)
    plt.title("(b) k-mean cluster")
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=kmeans_y_pred)
    plt.scatter(
        kmeans.cluster_centers_[:,0],
        kmeans.cluster_centers_[:,1],
        s=250, marker="*", c='red'
    )
    plt.subplot(133)
    plt.title("(c) GMM cluster")
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=gmm_y_pred)
    plt.show()

if __name__ == "__main__":
    X,y = make_blobs(n_samples=1500,
                     random_state=170)
    transformation = [[0.5, -0.6], [-0.3, 0.8]]
    X_aniso = np.dot(X, transformation)
    print(type(X_aniso))
    print(np.shape(X_aniso))
    # gmm(X_aniso)
