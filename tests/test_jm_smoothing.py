import numpy as np
import pytest
from scipy import sparse

from src.jm_smoothing import SmoothedTWCNB, make_smoothed_variant
from src.twcnb import VARIANTS, make_variant

RAW = dict(log_tf=False, idf=False, length_norm=False)


@pytest.fixture
def data():
    """300 documents, 50 words, 4 deliberately imbalanced classes (60/25/10/5 %)."""
    rng = np.random.default_rng(0)
    n_docs, n_words = 300, 50
    y = rng.choice(4, size=n_docs, p=[0.6, 0.25, 0.1, 0.05])
    X = rng.poisson(1.0, size=(n_docs, n_words))
    # Give class k extra occurrences of word k so there is something to learn.
    X[:, :4] += rng.poisson(3.0, size=(n_docs, 4)) * (y[:, None] == np.arange(4))
    return sparse.csr_matrix(X), y


def raw_counts(X, y, complement):
    """N_ci from raw counts, computed by hand."""
    X = X.toarray().astype(float)
    per_class = np.vstack([X[y == c].sum(axis=0) for c in np.unique(y)])
    return per_class.sum(axis=0, keepdims=True) - per_class if complement else per_class


@pytest.mark.parametrize("name", list(VARIANTS))
def test_laplace_is_the_parent_exactly(data, name):
    X, y = data
    ours = make_smoothed_variant(name, smoothing="laplace").fit(X, y)
    ref = make_variant(name).fit(X, y)
    np.testing.assert_array_equal(ours.w_, ref.w_)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


@pytest.mark.parametrize("name", list(VARIANTS))
def test_jm_theta_is_a_distribution(data, name):
    X, y = data
    model = make_smoothed_variant(name, smoothing="jm", lam=0.1, weight_norm=False).fit(X, y)
    np.testing.assert_allclose(np.exp(model.w_).sum(axis=1), 1.0, atol=1e-12)
    assert np.all(model.w_ < 0)


def test_jm_is_scale_invariant_laplace_is_not(data):
    """Length normalisation rescales the class totals; JM ignores that, Laplace does not."""
    X, y = data
    Xs = X.astype(float) * 7.3
    for complement in [False, True]:
        params = dict(RAW, complement=complement, weight_norm=False)
        jm = SmoothedTWCNB(smoothing="jm", lam=0.1, **params)
        np.testing.assert_allclose(jm.fit(X, y).w_, jm.fit(Xs, y).w_)
        lap = SmoothedTWCNB(smoothing="laplace", alpha=1.0, **params)
        assert not np.allclose(lap.fit(X, y).w_, lap.fit(Xs, y).w_)


@pytest.mark.parametrize("complement", [False, True])
def test_jm_equals_additive_with_per_class_alpha(data, complement):
    X, y = data
    lam = 0.2
    model = SmoothedTWCNB(smoothing="jm", lam=lam, complement=complement,
                          weight_norm=False, **RAW).fit(X, y)
    counts = raw_counts(X, y, complement)
    n_words = counts.shape[1]
    mass = counts.sum(axis=1, keepdims=True)
    alpha_c = lam * mass / ((1 - lam) * n_words)
    additive = np.log((counts + alpha_c) / (mass + alpha_c * n_words))
    np.testing.assert_allclose(model.w_, additive, rtol=1e-12)


def test_smoothing_share(data):
    X, y = data
    jm = SmoothedTWCNB(smoothing="jm", lam=0.05, complement=False, **RAW).fit(X, y)
    np.testing.assert_array_equal(jm.smoothing_share_, 0.05)
    lap = SmoothedTWCNB(smoothing="laplace", complement=False, **RAW).fit(X, y)
    by_size = np.argsort(lap.class_count_)
    assert np.all(np.diff(lap.smoothing_share_[by_size]) < 0)


@pytest.mark.parametrize("lam", [0.0, 1.0, -0.1, 1.5])
def test_lam_outside_unit_interval_raises(data, lam):
    X, y = data
    with pytest.raises(ValueError):
        SmoothedTWCNB(smoothing="jm", lam=lam).fit(X, y)
