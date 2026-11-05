import numpy as np
import pytest
from scipy import sparse
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from src.twcnb import RennieTransformer, make_variant


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


def test_mnb_matches_sklearn(data):
    X, y = data
    ours = make_variant("MNB").fit(X, y)
    ref = MultinomialNB(alpha=1.0, fit_prior=False).fit(X, y)
    np.testing.assert_allclose(ours.w_, ref.feature_log_prob_)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


@pytest.mark.parametrize("variant, norm", [("CNB", False), ("WCNB", True)])
def test_complement_matches_sklearn(data, variant, norm):
    X, y = data
    ours = make_variant(variant).fit(X, y)
    ref = ComplementNB(alpha=1.0, norm=norm).fit(X, y)
    # scikit-learn stores negated weights so that it can use arg max.
    np.testing.assert_allclose(ours.w_, -ref.feature_log_prob_)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


def test_log_tf_keeps_zero_and_one():
    X = sparse.csr_matrix(np.array([[0, 1, 3]]))
    out = RennieTransformer(log_tf=True, idf=False, length_norm=False).fit(X).transform(X)
    np.testing.assert_allclose(out.toarray(), [[0.0, 1.0, 2.0]])


def test_idf_matches_paper_formula():
    X = sparse.csr_matrix(np.array([[1, 0], [1, 1], [1, 0], [1, 0]]))
    t = RennieTransformer(log_tf=False, idf=True, length_norm=False).fit(X)
    np.testing.assert_allclose(t.idf_, [np.log(4 / 4), np.log(4 / 1)])


def test_length_norm_gives_unit_rows(data):
    X, _ = data
    out = RennieTransformer().fit(X).transform(X)
    norms = np.sqrt(np.asarray(out.multiply(out).sum(axis=1)).ravel())
    np.testing.assert_allclose(norms, 1.0)


def test_weight_normalisation_keeps_weights_negative(data):
    """Guards against the missing |.| in Table 4, step 6."""
    X, y = data
    model = make_variant("TWCNB").fit(X, y)
    assert np.all(model.w_ < 0)


def test_literal_table4_step6_flips_sign(data):
    """Table 4, step 6 as printed (no |.|) makes every weight positive, so arg min
    picks the class that fits worst."""
    from src.fidelity_checks import literal_step6_predict
    X, y = data
    plain = make_variant("TWCNB", weight_norm=False).fit(X, y)
    assert np.all(plain.w_ / plain.w_.sum(axis=1, keepdims=True) > 0)
    proper = make_variant("TWCNB").fit(X, y)
    assert (literal_step6_predict(plain, X) == y).mean() < (proper.predict(X) == y).mean()


def test_test_doc_length_norm_never_changes_predictions(data):
    """Scaling a test document rescales all its class scores alike, so arg min is unchanged."""
    X, y = data
    model = make_variant("TWCNB").fit(X, y)
    with_norm = model.predict(X)
    model.transformer_.length_norm = False
    np.testing.assert_array_equal(model.predict(X), with_norm)


def test_log_base_is_irrelevant_after_length_norm(data):
    """log2(1+f) and ln(1+f) differ by a constant factor, which length normalisation removes."""
    X, _ = data
    t = RennieTransformer().fit(X)
    ours = t.transform(X).toarray()
    ln = np.log1p(X.toarray()) * t.idf_
    ln /= np.linalg.norm(ln, axis=1, keepdims=True)
    np.testing.assert_allclose(ours, ln, atol=1e-12)


def test_cv_tuning_returns_grid_value_with_singleton_class(data):
    """Cross-validated tuning must cope with a class that has a single document (Reuters)."""
    from src.experiments import pick_best
    X, y = data
    y = y.copy()
    y[0] = 9                                   # a one-document class
    grid = [1.0, 0.1, 0.01]
    best = pick_best(lambda a: make_variant("MNB", alpha=a), grid, X, y, seed=0,
                     metric="macro_f1", n_jobs=1)
    assert best in grid
