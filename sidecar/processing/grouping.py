"""Numpy-only helpers that turn patch labels and attention into contract outputs.

Nothing here imports torch, so CI can test it. scikit-learn is imported lazily
inside cluster_patches.
"""

import base64
import io

import numpy as np
from PIL import Image


def normalize_attention(attention: np.ndarray) -> np.ndarray:
    """Divide by the largest group value; all zeros if that value is not positive."""
    values = np.asarray(attention, dtype=np.float64)
    peak = values.max() if values.size else 0.0
    if peak <= 0:
        return np.zeros_like(values)
    return values / peak


def encode_label_png(labels: np.ndarray) -> str:
    """8-bit grayscale PNG of the label map, base64 (ASCII)."""
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(labels, dtype=np.uint8), mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_label_png(b64: str) -> np.ndarray:
    image = Image.open(io.BytesIO(base64.b64decode(b64)))
    return np.asarray(image, dtype=np.uint8)


def relabel_by_height(patch_labels: np.ndarray, k: int = 8) -> np.ndarray:
    """Renumber groups by image-space height, as the paper does.

    New ID 0 is the group whose patches have the smallest mean row; ties go to
    the smaller mean column, then the old ID. Empty groups take the highest IDs.
    """
    labels = np.asarray(patch_labels)
    rows, cols = np.indices(labels.shape)
    present = []
    for old in range(k):
        mask = labels == old
        if mask.any():
            present.append((rows[mask].mean(), cols[mask].mean(), old))
    lookup = np.zeros(k, dtype=np.uint8)
    for new, (_, _, old) in enumerate(sorted(present)):
        lookup[old] = new
    return lookup[labels]


def group_attention(patch_labels: np.ndarray, patch_attention: np.ndarray, k: int = 8) -> np.ndarray:
    """Mean attention of each group's patches; 0.0 for an empty group."""
    labels = np.asarray(patch_labels).ravel()
    attention = np.asarray(patch_attention, dtype=np.float64).ravel()
    sums = np.bincount(labels, weights=attention, minlength=k)[:k]
    counts = np.bincount(labels, minlength=k)[:k]
    return np.divide(sums, counts, out=np.zeros(k), where=counts > 0)


def upsample_nearest(patch_labels: np.ndarray, size: int = 224) -> np.ndarray:
    """Repeat each patch label over its square block of pixels."""
    labels = np.asarray(patch_labels, dtype=np.uint8)
    block = size // labels.shape[0]
    return np.repeat(np.repeat(labels, block, axis=0), block, axis=1)


def assign_nearest(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Label each feature row (N, D) by its nearest centroid (K, D), squared Euclidean;
    ties go to the lowest index. Used for FeatUp pixel features (paper: nearest patch-level centroid)."""
    f = np.asarray(features, dtype=np.float32)
    c = np.asarray(centroids, dtype=np.float32)
    distances = (c * c).sum(1)[None, :] - 2.0 * (f @ c.T)  # |f|^2 is the same for every centroid
    return distances.argmin(1).astype(np.uint8)


def order_centroids(raw_labels: np.ndarray, patch_labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Reorder K-means centroids (k, D) from raw IDs to the height-ordered patch IDs.
    raw_labels and patch_labels are the same patches before/after relabel_by_height;
    empty groups get a zero row."""
    raw = np.asarray(raw_labels).ravel()
    new = np.asarray(patch_labels).ravel()
    ordered = np.zeros_like(centroids)
    for old in range(len(centroids)):
        mask = raw == old
        if mask.any():
            ordered[new[mask][0]] = centroids[old]
    return ordered


def pixel_labels(pixels: np.ndarray, patch_labels: np.ndarray, ordered: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """FeatUp pixel features (H*W, D) -> (H, W) uint8 IDs: each pixel takes the nearest
    height-ordered centroid among the groups present in the patches (empty groups have none)."""
    present = np.unique(patch_labels)
    return present[assign_nearest(pixels, ordered[present])].astype(np.uint8).reshape(shape)


def cluster_patches(features: np.ndarray, k: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """K-means on patch features. Returns (labels (N,), centroids (k, D))."""
    import warnings

    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    # One thread: the data is tiny, and OpenBLAS's default of one thread per core
    # made this ~110 ms instead of ~10 ms on the 24-core lab machine.
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        # Fewer distinct features than k (e.g. a blank image) is expected, not an error.
        warnings.simplefilter("ignore", ConvergenceWarning)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(features)
    return kmeans.labels_.astype(np.int64), kmeans.cluster_centers_.astype(np.float32)
