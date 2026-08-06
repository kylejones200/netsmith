"""
Tests for Mahalanobis distance.

The defining property is invariance under linear transformation of the
features: rescaling or rotating the feature space must not change the
distances, which is exactly what Euclidean distance fails to do.
"""

import numpy as np
import pytest

from netsmith.core.distance import mahalanobis
from netsmith.exceptions import ValidationError


@pytest.fixture
def correlated():
    """Samples with correlated, differently scaled features."""
    rng = np.random.default_rng(0)
    mixing = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 3.0]])
    return rng.normal(size=(200, 3)) @ mixing


class TestDefiningProperties:
    """What makes this Mahalanobis rather than some other distance."""

    def test_invariant_under_linear_transformation(self, correlated):
        """Rescaling or rotating the features must not move the distances."""
        transform = np.array([[3.0, 0.0, 1.0], [0.0, 0.5, 0.0], [0.0, 0.0, 7.0]])

        before = mahalanobis(correlated)
        after = mahalanobis(correlated @ transform)

        np.testing.assert_allclose(before, after, rtol=1e-9)

    def test_euclidean_when_the_covariance_is_the_identity(self):
        """With uncorrelated unit-variance features it reduces to Euclidean."""
        points = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
        identity = np.eye(2)

        distances = mahalanobis(points, points, covariance=identity)

        assert distances[0, 1] == pytest.approx(5.0)
        assert distances[0, 2] == pytest.approx(np.sqrt(2.0))

    def test_single_feature_is_the_absolute_z_score(self):
        """In one dimension it collapses to distance in standard deviations."""
        rng = np.random.default_rng(3)
        values = rng.normal(size=(50, 1))

        expected = np.abs(values[:, 0] - values.mean()) / values.std(ddof=1)

        np.testing.assert_allclose(mahalanobis(values), expected, rtol=1e-9)

    def test_rescaling_a_feature_moves_euclidean_but_not_mahalanobis(self):
        """The contrast that motivates the metric."""
        rng = np.random.default_rng(5)
        points = rng.normal(size=(100, 2))
        # Same data, one feature measured in different units.
        rescaled = points * np.array([1.0, 1000.0])

        centred = points - points.mean(axis=0)
        euclidean = np.sqrt((centred**2).sum(axis=1))
        centred_rescaled = rescaled - rescaled.mean(axis=0)
        euclidean_rescaled = np.sqrt((centred_rescaled**2).sum(axis=1))

        # Euclidean is now dominated by whichever feature has the bigger unit.
        assert np.corrcoef(euclidean, euclidean_rescaled)[0, 1] < 0.9
        # Mahalanobis does not notice the change of units at all.
        np.testing.assert_allclose(mahalanobis(points), mahalanobis(rescaled), rtol=1e-9)

    def test_a_point_is_zero_distance_from_itself(self, correlated):
        distances = mahalanobis(correlated, correlated)

        np.testing.assert_array_equal(np.diag(distances), 0.0)

    def test_symmetric(self, correlated):
        distances = mahalanobis(correlated[:20], correlated[:20])

        np.testing.assert_allclose(distances, distances.T, atol=1e-12)

    def test_triangle_inequality_holds(self, correlated):
        """It is a metric, so detours cannot be shorter."""
        points = correlated[:30]
        distances = mahalanobis(points, points)

        direct = distances[:, None, :]  # d[i, k]
        detour = distances[:, :, None] + distances[None, :, :]  # d[i, j] + d[j, k]
        assert (direct <= detour + 1e-9).all()


class TestShapes:
    """The API says what it returns."""

    def test_distance_to_the_mean_is_one_per_sample(self, correlated):
        assert mahalanobis(correlated).shape == (200,)

    def test_pairwise_is_one_row_per_sample(self, correlated):
        assert mahalanobis(correlated, correlated[:7]).shape == (200, 7)

    def test_a_single_target_point_may_be_one_dimensional(self, correlated):
        """A 1-D y is unambiguous: it has to be one point."""
        assert mahalanobis(correlated, correlated[0]).shape == (200, 1)

    def test_supplied_covariance_is_used(self, correlated):
        """A different covariance must give different distances."""
        estimated = np.cov(correlated, rowvar=False, ddof=1)

        default = mahalanobis(correlated)
        widened = mahalanobis(correlated, covariance=estimated * 4.0)

        # Quadrupling the covariance halves every distance.
        np.testing.assert_allclose(widened, default / 2.0, rtol=1e-9)


class TestFailures:
    """Undefined inputs say so instead of returning a plausible number."""

    def test_singular_covariance_is_rejected(self):
        """Collinear features leave the distance genuinely undefined."""
        base = np.random.default_rng(7).normal(size=(50, 1))
        collinear = np.hstack([base, base * 2.0])  # exactly redundant

        with pytest.raises(ValidationError, match="positive definite"):
            mahalanobis(collinear)

    def test_more_features_than_samples_is_rejected(self):
        """The covariance cannot have full rank, so it is not silently patched."""
        data = np.random.default_rng(8).normal(size=(3, 10))

        with pytest.raises(ValidationError, match="rank"):
            mahalanobis(data)

    def test_one_dimensional_x_is_rejected_with_the_fix(self):
        """Guessing the axis would silently answer a different question."""
        with pytest.raises(ValidationError, match="reshape"):
            mahalanobis(np.arange(10.0))

    def test_feature_counts_must_agree(self, correlated):
        with pytest.raises(ValidationError, match="features"):
            mahalanobis(correlated, np.zeros((4, 5)))

    def test_covariance_shape_must_match(self, correlated):
        with pytest.raises(ValidationError, match="covariance must be"):
            mahalanobis(correlated, covariance=np.eye(5))

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_values_are_rejected(self, correlated, bad):
        broken = correlated.copy()
        broken[0, 0] = bad

        with pytest.raises(ValidationError, match="finite"):
            mahalanobis(broken)

    def test_a_single_sample_needs_an_explicit_covariance(self):
        """One point cannot describe a distribution."""
        point = np.array([[1.0, 2.0]])

        with pytest.raises(ValidationError, match="at least 2 samples"):
            mahalanobis(point)

        # ...but works when the covariance is supplied.
        assert mahalanobis(point, covariance=np.eye(2)).shape == (1,)


class TestAgainstSciPy:
    """Match the reference implementation, without depending on it."""

    def test_pairwise_matches_cdist(self, correlated):
        spatial = pytest.importorskip("scipy.spatial.distance")

        targets = correlated[:5]
        precision = np.linalg.inv(np.cov(correlated, rowvar=False, ddof=1))
        expected = spatial.cdist(correlated, targets, metric="mahalanobis", VI=precision)

        np.testing.assert_allclose(mahalanobis(correlated, targets), expected, atol=1e-10)

    def test_distance_to_mean_matches_scipy(self, correlated):
        spatial = pytest.importorskip("scipy.spatial.distance")

        precision = np.linalg.inv(np.cov(correlated, rowvar=False, ddof=1))
        centre = correlated.mean(axis=0)
        expected = [spatial.mahalanobis(row, centre, precision) for row in correlated]

        np.testing.assert_allclose(mahalanobis(correlated), expected, atol=1e-10)
