"""
prdc 
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import logging
import numpy as np
import sklearn.metrics

__all__ = ['compute_prdc']

logger = logging.getLogger(__name__)


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(X_real, X_synth, nearest_k=5):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        X_real: numpy.ndarray([N, feature_dim], dtype=np.float32)
        X_synth: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    logger.info('Num real: {} Num synth: {}'
          .format(X_real.shape[0], X_synth.shape[0]))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        X_real, nearest_k)
    synth_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        X_synth, nearest_k)
    distance_real_synth = compute_pairwise_distance(
        X_real, X_synth)

    precision = (
            distance_real_synth <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean().round(6)

    recall = (
            distance_real_synth <
            np.expand_dims(synth_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean().round(6)

    density = (1. / float(nearest_k)) * (
            distance_real_synth <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean().round(6)

    coverage = (
            distance_real_synth.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean().round(6)

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)