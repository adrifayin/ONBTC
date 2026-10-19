"""
From-scratch implementation of

    Rennie, Shih, Teevan & Karger (2003).
    "Tackling the Poor Assumptions of Naive Bayes Text Classifiers." ICML.

RennieTransformer : Table 4, steps 1-3 (log TF, IDF, length normalisation)
TWCNB             : Table 4, steps 4-8 (complement estimates, weight
                    normalisation, classification)

Every correction can be switched on or off, so this one class produces all
six variants used in the ablation study (see VARIANTS at the bottom).
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class RennieTransformer(TransformerMixin, BaseEstimator):
    """Steps 1-3 of Table 4. Input: a (documents x words) count matrix."""

    def __init__(self, log_tf=True, idf=True, length_norm=True):
        self.log_tf = log_tf
        self.idf = idf
        self.length_norm = length_norm

    def fit(self, X, y=None):
        X = sparse.csr_matrix(X)
        n_docs = X.shape[0]
        # Document frequency: in how many training documents does word i occur?
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        # Section 4.2: idf_i = log(total documents / documents containing word i).
        # A word never seen in training (df = 0) gets weight 0.
        safe_df = np.maximum(df, 1)
        self.idf_ = np.where(df > 0, np.log(n_docs / safe_df), 0.0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = sparse.csr_matrix(X, dtype=np.float64, copy=True)
        if self.log_tf:
            # Step 1: f' = log2(1 + f). Base 2 keeps counts 0 and 1 unchanged,
            # which is the property the paper describes in Section 4.1.
            X.data = np.log2(1.0 + X.data)
        if self.idf:
            # Step 2: multiply every column (word) by its IDF.
            X = (X @ sparse.diags(self.idf_)).tocsr()
        if self.length_norm:
            # Step 3: divide every row (document) by its Euclidean length.
            norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
            norms[norms == 0] = 1.0
            X = (sparse.diags(1.0 / norms) @ X).tocsr()
        return X


class TWCNB(ClassifierMixin, BaseEstimator):
    """Transformed Weight-normalised Complement Naive Bayes (Table 4).

    Parameters
    ----------
    alpha : float, default 1.0
        Smoothing added to every word count (alpha_i in the paper).
    log_tf, idf, length_norm : bool
        The text transforms of Section 4 (steps 1-3).
    complement : bool
        Estimate each class from all *other* classes (Section 3.1, step 4).
    weight_norm : bool
        Normalise each class's weight vector (Section 3.2, step 6).
    transform_test : bool
        Apply steps 1-3 to test documents as well. Table 4, step 7 literally
        uses raw counts; the report discusses this ambiguity.
    """

    def __init__(self, alpha=1.0, log_tf=True, idf=True, length_norm=True,
                 complement=True, weight_norm=True, transform_test=True):
        self.alpha = alpha
        self.log_tf = log_tf
        self.idf = idf
        self.length_norm = length_norm
        self.complement = complement
        self.weight_norm = weight_norm
        self.transform_test = transform_test

    def fit(self, X, y):
        X = sparse.csr_matrix(X)
        y = np.asarray(y)
        if X.shape[0] != len(y):
            raise ValueError("X and y must contain the same number of documents")
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("Need at least two classes")
        n_words = X.shape[1]

        # Steps 1-3: learn IDF from the training documents, then transform them.
        self.transformer_ = RennieTransformer(
            log_tf=self.log_tf, idf=self.idf, length_norm=self.length_norm).fit(X)
        D = self.transformer_.transform(X)

        # N_ci: total (transformed) count of word i in the documents of class c.
        per_class = np.vstack([
            np.asarray(D[y == c].sum(axis=0)).ravel() for c in self.classes_
        ])

        # Step 4: complement counts = counts in every class except c (Eq. 6).
        if self.complement:
            counts = per_class.sum(axis=0, keepdims=True) - per_class
        else:
            counts = per_class

        # Smoothed estimate (Eq. 4 / Eq. 6). Total alpha = alpha * vocabulary size.
        theta = (counts + self.alpha) / (
            counts.sum(axis=1, keepdims=True) + self.alpha * n_words)

        # Step 5: the weights are log-parameters.
        w = np.log(theta)

        # Step 6, written as Eq. 8 with absolute values. Table 4 leaves out the |.|;
        # without it every (negative) weight flips sign and arg min picks the worst class.
        if self.weight_norm:
            w = w / np.abs(w).sum(axis=1, keepdims=True)

        self.w_ = w
        self.class_count_ = np.array([np.sum(y == c) for c in self.classes_])
        self.n_features_in_ = n_words
        return self

    def decision_function(self, X):
        """Score of every class for every document: t . w_c (steps 7-8)."""
        X = sparse.csr_matrix(X, dtype=np.float64)
        T = self.transformer_.transform(X) if self.transform_test else X
        return np.asarray(T @ self.w_.T)

    def predict(self, X):
        scores = self.decision_function(X)
        if self.complement:
            # Complement weights describe the *other* classes, so the best class
            # is the one whose complement fits the document worst (step 8).
            idx = scores.argmin(axis=1)
        else:
            idx = scores.argmax(axis=1)
        return self.classes_[idx]


VARIANTS = {
    "MNB":   dict(log_tf=False, idf=False, length_norm=False, complement=False, weight_norm=False),
    "CNB":   dict(log_tf=False, idf=False, length_norm=False, complement=True,  weight_norm=False),
    "WCNB":  dict(log_tf=False, idf=False, length_norm=False, complement=True,  weight_norm=True),
    "T-MNB": dict(log_tf=True,  idf=True,  length_norm=True,  complement=False, weight_norm=False),
    "TCNB":  dict(log_tf=True,  idf=True,  length_norm=True,  complement=True,  weight_norm=False),
    "TWCNB": dict(log_tf=True,  idf=True,  length_norm=True,  complement=True,  weight_norm=True),
}


def make_variant(name, **overrides):
    """Return an unfitted TWCNB configured as one of the named variants."""
    params = dict(VARIANTS[name])
    params.update(overrides)
    return TWCNB(**params)
