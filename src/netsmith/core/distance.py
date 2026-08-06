"""
Distance measures on node feature vectors.

These operate on features attached to nodes rather than on the graph itself,
and are the usual first step in building a graph from a feature table: compute
distances, then threshold or take k nearest neighbours to get edges.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..exceptions import ValidationError

__all__ = ["mahalanobis"]


def _as_samples(array, name: str) -> NDArray[np.float64]:
    """Coerce input to a (n_samples, n_features) float matrix."""
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2:
        raise ValidationError(
            f"{name} must be 2-D (n_samples, n_features), got {values.ndim}-D with "
            f"shape {values.shape}. Reshape explicitly: a single point is "
            f"{name}.reshape(1, -1), a single feature is {name}.reshape(-1, 1)."
        )
    if not np.isfinite(values).all():
        raise ValidationError(f"{name} must be finite; found NaN or infinity")
    return values


def _whitening_transform(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Return L from the Cholesky factorization ``covariance = L @ L.T``.

    Mahalanobis distance is Euclidean distance in the coordinates whitened by
    this factor, which is both faster and better conditioned than forming the
    inverse covariance and evaluating the quadratic form.

    Rank is checked before factorizing. Exactly redundant features leave the
    covariance singular in theory but a hair positive-definite in floating
    point, so Cholesky succeeds and whitening then divides by that hair —
    returning enormous distances that look like results.
    """
    n_features = covariance.shape[0]
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = float(eigenvalues[-1])
    # The same tolerance numpy.linalg.matrix_rank uses to call a value zero.
    tolerance = n_features * np.finfo(np.float64).eps * max(largest, 0.0)

    if largest <= 0.0 or float(eigenvalues[0]) <= tolerance:
        rank = int((eigenvalues > tolerance).sum())
        raise ValidationError(
            f"the covariance matrix is not positive definite (rank {rank} of "
            f"{n_features}), so Mahalanobis distance is undefined. This usually "
            f"means fewer samples than features, or features that are exact "
            f"linear combinations of each other. Drop the redundant features, "
            f"reduce dimension first, or pass a regularized `covariance`."
        )

    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - rank check catches this
        raise ValidationError(f"the covariance matrix could not be factorized: {exc}") from exc


def mahalanobis(
    x,
    y=None,
    covariance: Optional[NDArray] = None,
) -> NDArray[np.float64]:
    """
    Compute Mahalanobis distance between feature vectors.

    Euclidean distance treats every feature as equally important and mutually
    independent. Mahalanobis first rescales by the covariance, so correlated
    features stop counting twice and a step along a low-variance direction
    counts for more than a step along a high-variance one.

    Parameters
    ----------
    x : array_like, shape (n_samples, n_features)
        Feature vectors. Must be 2-D; the reshape is left to the caller because
        guessing which axis is which silently returns plausible nonsense.
    y : array_like, optional
        Points to measure against, shape (n_points, n_features) or a single
        point of shape (n_features,). When omitted, distances are measured
        from the mean of `x` — the usual outlier score.
    covariance : array_like, optional
        Covariance matrix, shape (n_features, n_features). Estimated from `x`
        with one delta degree of freedom when omitted. Supply it when `x` is a
        sample of a distribution you have already characterized, or when you
        need to regularize a near-singular estimate.

    Returns
    -------
    distances : NDArray[np.float64]
        Shape (n_samples,) when `y` is omitted, giving each row's distance
        from the mean. Shape (n_samples, n_points) otherwise, giving every
        pairwise distance.

    Raises
    ------
    ValidationError
        If the inputs are not 2-D, contain non-finite values, disagree on the
        number of features, or if the covariance is not positive definite —
        in which case the distance genuinely does not exist, rather than being
        quietly replaced by a pseudo-inverse.

    Notes
    -----
    Computed by whitening with the Cholesky factor of the covariance and taking
    Euclidean distances there, which is equivalent to the quadratic form
    ``sqrt((u - v) @ inv(S) @ (u - v))`` but better conditioned.

    Matches ``scipy.spatial.distance.mahalanobis`` and ``cdist(..., 'mahalanobis')``
    to floating-point tolerance, without requiring SciPy.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> features = rng.normal(size=(100, 3))
    >>> scores = mahalanobis(features)          # distance from the mean
    >>> scores.shape
    (100,)
    >>> pairs = mahalanobis(features, features[:5])
    >>> pairs.shape
    (100, 5)
    """
    samples = _as_samples(x, "x")
    n_samples, n_features = samples.shape

    if covariance is None:
        if n_samples < 2:
            raise ValidationError(
                "at least 2 samples are needed to estimate a covariance; pass "
                "`covariance` explicitly to score a single point"
            )
        estimated = np.cov(samples, rowvar=False, ddof=1)
        # np.cov collapses to a scalar for a single feature.
        covariance = np.atleast_2d(estimated)
    else:
        covariance = np.asarray(covariance, dtype=np.float64)
        if covariance.shape != (n_features, n_features):
            raise ValidationError(
                f"covariance must be ({n_features}, {n_features}) to match x's "
                f"features, got {covariance.shape}"
            )
        if not np.isfinite(covariance).all():
            raise ValidationError("covariance must be finite; found NaN or infinity")

    factor = _whitening_transform(covariance)

    if y is None:
        targets = samples.mean(axis=0, keepdims=True)
        single = True
    else:
        targets = np.asarray(y, dtype=np.float64)
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)
        targets = _as_samples(targets, "y")
        if targets.shape[1] != n_features:
            raise ValidationError(f"y has {targets.shape[1]} features but x has {n_features}")
        single = False

    # Whiten both sets, after which Mahalanobis is plain Euclidean distance.
    whitened_samples = np.linalg.solve(factor, samples.T).T
    whitened_targets = np.linalg.solve(factor, targets.T).T

    # Expand ||a - b||^2 into ||a||^2 + ||b||^2 - 2 a.b so the work is one BLAS
    # matmul with an (n, m) result, rather than an (n, m, n_features) temporary
    # that would run to gigabytes on any interesting number of points.
    square_samples = np.einsum("ij,ij->i", whitened_samples, whitened_samples)
    square_targets = np.einsum("ij,ij->i", whitened_targets, whitened_targets)
    squared = (
        square_samples[:, None]
        + square_targets[None, :]
        - 2.0 * whitened_samples @ whitened_targets.T
    )
    # Cancellation can push near-zero distances a hair below zero.
    np.maximum(squared, 0.0, out=squared)
    distances = np.sqrt(squared)

    return distances[:, 0] if single else distances
