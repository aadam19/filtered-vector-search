import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd

MAX_RANGE = 200

def import_dataset():
    
    def read_ivecs(fname: str) -> np.ndarray:
        with open(fname, "rb") as f:
            data = np.fromfile(f, dtype=np.int32)

        dim = data[0]
        data = data.reshape(-1, dim + 1)

        return data[:, 1:]

    def read_fvecs(fname: str) -> np.ndarray:
        with open(fname, "rb") as f:
            data = np.fromfile(f, dtype=np.int32)

        dim = data[0]
        data = data.reshape(-1, dim + 1)

        # reinterpret everything except first column as float32
        vectors = data[:, 1:].view(np.float32)

        return vectors
    
    BASE = "/home/adamm/Desktop/filtered-vector-search/siftsmall/siftsmall_base.fvecs"
    QUERY = "/home/adamm/Desktop/filtered-vector-search/siftsmall/siftsmall_query.fvecs"
    TRUTH = "/home/adamm/Desktop/filtered-vector-search/siftsmall/siftsmall_groundtruth.ivecs"

    base_vectors = read_fvecs(BASE)
    query_vectors = read_fvecs(QUERY)
    truth_vectors = read_ivecs(TRUTH)
    
    return base_vectors, query_vectors, truth_vectors

def generate_correlated_attribute(vecs, k=10, seed=0):
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(vecs)
    centroids = kmeans.cluster_centers_

    cluster_values = np.linspace(0, 1, k)

    dists = np.linalg.norm(vecs[:, None, :] - centroids[None, :, :], axis=2)

    sigma = np.median(dists)

    weights = np.exp(-(dists**2) / sigma**2)

    attr = (weights @ cluster_values) / weights.sum(axis=1)
    attr = (attr - attr.min()) / (attr.max() - attr.min()) * MAX_RANGE

    return attr, kmeans.labels_

def generate_random_attribute(vecs, seed=0):
    rng = np.random.default_rng(seed)
    attr = rng.uniform(0, MAX_RANGE, size=len(vecs))

    return attr, None

def compute_neighbor_stats(vecs, attr, n_neighbors=10):
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(vecs)

    dist, idx = nn.kneighbors(vecs)

    neighbor_attr = attr[idx]

    diff = np.abs(attr[:, None] - neighbor_attr)

    neighbor_mean = neighbor_attr.mean(axis=1)

    return {
        "neighbor_mean": neighbor_mean,
        "neighbor_diff_mean": diff.mean()
    }
    
    
def plot_attribute_analysis(vecs_2d, attr, neighbor_mean, clusters=None):
    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    # --- PCA colored by attribute ---
    sc = axes[0].scatter(vecs_2d[:,0], vecs_2d[:,1], c=attr, s=5, cmap='viridis')
    axes[0].set_title("Vectors colored by attribute")
    axes[0].set_xlabel("PCA1")
    axes[0].set_ylabel("PCA2")

    plt.colorbar(sc, ax=axes[0], label="Attribute")

    # --- Cluster distribution if clusters exist ---
    if clusters is not None:

        df = pd.DataFrame({
            "cluster": clusters,
            "attr": attr
        })

        cluster_means = df.groupby("cluster")["attr"].mean()
        cluster_std = df.groupby("cluster")["attr"].std()

        axes[1].errorbar(range(len(cluster_means)), cluster_means,
                         yerr=cluster_std, fmt='o')
        axes[1].set_title("Cluster attribute distribution")
        axes[1].set_xlabel("Cluster")
        axes[1].set_ylabel("Attribute value")

    else:
        axes[1].hist(attr, bins=30)
        axes[1].set_title("Attribute distribution")
        axes[1].set_xlabel("Attribute")
        axes[1].set_ylabel("Count")

    # --- neighbor correlation ---
    axes[2].scatter(attr, neighbor_mean, s=5)
    axes[2].set_title("Attribute vs neighbor mean")
    axes[2].set_xlabel("Point attribute")
    axes[2].set_ylabel("neighbor mean attribute")

    plt.tight_layout()
    plt.show()
    
def generate_query_ranges(vecs, attr, queries, k=50, margin=10, seed=0):
    rng = np.random.default_rng(seed)
    
    # Compute domain
    domain_min = attr.min()
    domain_max = attr.max()
    
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(vecs)
    
    dist, idx = nn.kneighbors(queries)
    
    pos_ranges = []
    neg_ranges = []
    
    for neighbors in idx:
        vals = attr[neighbors]
        
        # Making sure we stay in bounds
        a_min = max(domain_min, vals.min() - margin)
        a_max = min(domain_max, vals.max() + margin)
        
        pos_ranges.append((a_min, a_max))
        
        # Construct negative ranges that lie completely outside the positive range
        # while staying within the dataset's attribute domain
        left = (domain_min, a_min - margin)
        right = (a_max + margin, domain_max)

        candidates = []

        # Ensure a valid interval
        if left[1] > left[0]:
            candidates.append(left)

        if right[1] > right[0]:
            candidates.append(right)

        # Pick randomly between candidates
        if candidates:
            neg_ranges.append(candidates[rng.integers(len(candidates))])
        else:
            # Fallback (rare): no valid outside range
            neg_ranges.append((domain_min, domain_max))
    
    return pos_ranges, neg_ranges

def compute_cluster_stats(fitted_vecs, vecs, attr, td):
    centroids = fitted_vecs.cluster_centers_
    histograms, cdfs = [], []
    
    for cluster_id, centroid in enumerate(centroids):
        # 1. Get points assigned to this cluster
        cluster_idx = np.where(fitted_vecs.labels_ == cluster_id)[0]
        cluster_points = vecs[cluster_idx]
        cluster_attrs = attr[cluster_idx]  # your attribute values
        
        # 2. Compute distances from centroid
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        
        # 3. Determine cluster radius and include neighbors slightly outside
        cluster_radius = distances.max()
        threshold = cluster_radius * td
        selected_idx = cluster_idx[distances <= threshold]
        
        selected_attrs = attr[selected_idx]

        # 4. Compute histogram
        hist, bin_edges = np.histogram(selected_attrs, bins='auto')        
        
        # 5. Compute CDF
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]
        
        histograms.append((hist, bin_edges))
        cdfs.append(cdf)        
    return histograms, cdfs
            
