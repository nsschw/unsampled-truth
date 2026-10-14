import numpy as np
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import pdist, cdist
from typing import List, Dict, Optional
import itertools

def generalized_jensen_shannon_divergence(
    distributions: List[Dict[str, float]], 
    weights: Optional[List[float]] = None,
    base: float = 2.0
) -> float:
    """
    Calculates the Generalized Jensen-Shannon Divergence for N distributions.
    
    Args:
        distributions: A list of dictionaries, where each dict maps an anchor (str) 
                       to its probability mass (float). 
                       e.g., [{"A": 0.5, "B": 0.5}, {"A": 0.1, "B": 0.9}]
        weights: Optional list of floats representing the weight of each distribution.
                 If None, uniform weights (1/N) are applied.
        base: The logarithmic base to use. Default is 2.0 (returns bits). 
              Use np.e for nats.
              
    Returns:
        The Generalized JSD (float). Bounded between 0 and log(N) when using base 2.
    """
    if not distributions or len(distributions) < 2:
        return 0.0

    # 1. Extract universal support (vocabulary) across all distributions
    vocab = set()
    for dist in distributions:
        vocab.update(dist.keys())
    vocab = sorted(list(vocab))
    
    N = len(distributions)
    V = len(vocab)
    
    # 2. Vectorize the probability distributions into a matrix (N x V)
    dist_matrix = np.zeros((N, V))
    for i, dist in enumerate(distributions):
        for j, anchor in enumerate(vocab):
            dist_matrix[i, j] = dist.get(anchor, 0.0)
            
    # Normalize rows to strictly enforce the sum-to-1 constraint
    # (Mitigates rounding errors from upstream raw logits)
    row_sums = dist_matrix.sum(axis=1, keepdims=True)
    dist_matrix = np.divide(
        dist_matrix, 
        row_sums, 
        out=np.zeros_like(dist_matrix), 
        where=row_sums != 0
    )
    
    # 3. Formulate the weights vector (\pi)
    if weights is None:
        weights = np.ones(N) / N
    else:
        if len(weights) != N:
            raise ValueError("Length of weights must match number of distributions.")
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Ensure weights sum to 1.0
        
    # 4. Calculate the Mixture Distribution (M = \sum \pi_i P_i)
    M = np.average(dist_matrix, axis=0, weights=weights)
    
    # 5. Calculate Generalized JSD: \sum \pi_i KL(P_i || M)
    jsd = 0.0
    for i in range(N):
        # scipy.stats.entropy handles 0*log(0) limits gracefully
        kl_div = entropy(pk=dist_matrix[i], qk=M, base=base)
        # Handle potential NaNs from extreme floating point anomalies
        if np.isfinite(kl_div):
            jsd += weights[i] * kl_div
            
    return float(jsd)


def average_pairwise_wasserstein_distance(
    distributions: List[Dict[str, float]],
    ordinal_mapping: Dict[str, float]
) -> float:
    """
    Calculates the average 1D Wasserstein Distance (Earth Mover's Distance) 
    across all pairs of distributions using vectorized CDF representations.
    
    Args:
        distributions: A list of dictionaries representing probability masses.
        ordinal_mapping: A dictionary projecting nominal anchors onto a metric space.
                         e.g., {"Strongly Disagree": 1, "Disagree": 2, ...}
                         
    Returns:
        The average pairwise Wasserstein distance.
    """
    if not distributions or len(distributions) < 2:
        return 0.0

    # 1. Sort the vocabulary by its metric mapping to ensure a valid CDF sequence
    sorted_items = sorted(ordinal_mapping.items(), key=lambda x: x[1])
    vocab = [item[0] for item in sorted_items]
    sorted_values = np.array([item[1] for item in sorted_items])
    
    N = len(distributions)
    V = len(vocab)
    
    # 2. Vectorize and normalize distributions into a dense matrix
    dist_matrix = np.zeros((N, V))
    for i, dist in enumerate(distributions):
        for j, anchor in enumerate(vocab):
            dist_matrix[i, j] = dist.get(anchor, 0.0)
            
        row_sum = np.sum(dist_matrix[i])
        if row_sum > 0:
            dist_matrix[i] /= row_sum
            
    # 3. Calculate intervals between adjacent points in the metric space (\Delta x)
    deltas = np.diff(sorted_values)
    
    # 4. Compute Cumulative Distribution Functions
    # We slice [:, :-1] to drop the final column, as the CDF is always 1.0 there
    cdfs = np.cumsum(dist_matrix, axis=1)[:, :-1]
    
    # 5. Weight the CDFs by the metric space intervals
    weighted_cdfs = cdfs * deltas
    
    # 6. Compute all pairwise L1 (cityblock) distances simultaneously
    distances = pdist(weighted_cdfs, metric='cityblock')
    
    return float(np.mean(distances))




def cross_group_wasserstein_distance(
    distributions_a: List[Dict[str, float]],
    distributions_b: List[Dict[str, float]],
    ordinal_mapping: Dict[str, float]
) -> float:
    """
    Calculates the average 1D Wasserstein Distance strictly between two distinct 
    sets of distributions (A vs B), ignoring within-group agreement.
    """
    # 1. Lock the ordinal space sorting for consistent matrix projection
    sorted_items = sorted(ordinal_mapping.items(), key=lambda x: x[1])
    vocab = [item[0] for item in sorted_items]
    sorted_values = np.array([item[1] for item in sorted_items])
    
    def transform_to_cdfs(dists: List[Dict[str, float]]) -> np.ndarray:
        """Filters invalid distributions and returns normalized, truncated CDFs."""
        valid_dists = [
            d for d in dists 
            if sum(d.get(anchor, 0.0) for anchor in vocab) > 0
        ]
        
        if not valid_dists:
            return np.empty((0, len(vocab) - 1))
            
        N = len(valid_dists)
        V = len(vocab)
        matrix = np.zeros((N, V))
        
        for i, dist in enumerate(valid_dists):
            for j, anchor in enumerate(vocab):
                matrix[i, j] = dist.get(anchor, 0.0)
            matrix[i] /= np.sum(matrix[i]) # Safe division due to filtering
            
        return np.cumsum(matrix, axis=1)[:, :-1]

    # 2. Transform both groups into CDF matrices
    cdfs_a = transform_to_cdfs(distributions_a)
    cdfs_b = transform_to_cdfs(distributions_b)
    
    if cdfs_a.size == 0 or cdfs_b.size == 0:
        return 0.0

    # 3. Calculate metric intervals and apply weights
    deltas = np.diff(sorted_values)
    weighted_a = cdfs_a * deltas
    weighted_b = cdfs_b * deltas
    
    # 4. Compute the cross-distance matrix (A vs B strictly)
    # Output shape: (len(distributions_a), len(distributions_b))
    distance_matrix = cdist(weighted_a, weighted_b, metric='cityblock')
    
    # 5. Return the global mean of the cross-distances
    return float(np.mean(distance_matrix))