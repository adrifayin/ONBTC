"""
Jelinek-Mercer smoothing for the Rennie et al. (2003) classifier.

Rennie et al. smooth every class with the same additive pseudo-count alpha (Eq. 4 / Eq. 6):

    theta_ci = (N_ci + alpha) / (N_c + alpha * V)

That adds alpha * V pseudo-counts to every class whatever its size, so the share of a
class's estimate that comes from pseudo-counts,

    s_c = alpha * V / (N_c + alpha * V),

is large for small classes and small for large ones. Complement NB evens this out as a side
effect, because every complement has roughly the same mass. This module replaces the
additive estimate with Jelinek-Mercer (JM) interpolation toward the uniform distribution:

    theta_ci = (1 - lam) * N_ci / N_c + lam / V

JM gives every class the same smoothing share lam, does not change when all counts are
multiplied by a constant (length normalisation does exactly that to the class totals), and
equals additive smoothing with a per-class alpha_c = lam * N_c / ((1 - lam) * V).

SmoothedTWCNB subclasses TWCNB, so src/twcnb.py is used unchanged; smoothing="laplace" is the
parent class exactly.
"""
import numpy as np
from scipy import sparse

from src.twcnb import VARIANTS, TWCNB


class SmoothedTWCNB(TWCNB):
    """TWCNB with a choice of smoothing.

    Parameters
    ----------
    smoothing : {"laplace", "jm"}, default "laplace"
        "laplace" is the paper's additive smoothing with pseudo-count alpha (the parent
        class, unchanged). "jm" is Jelinek-Mercer interpolation toward uniform with weight lam.
    lam : float, default 0.1
        Weight of the uniform distribution under JM; must lie strictly between 0 and 1.
        Ignored when smoothing="laplace".
    alpha, log_tf, idf, length_norm, complement, weight_norm, transform_test
        As in TWCNB. alpha is ignored when smoothing="jm".

    Fitted attributes (besides those of TWCNB)
    -------------------------------------------
    class_mass_ : total (transformed, possibly complement) count N_c behind each class's
                  estimate.
    smoothing_share_ : fraction of each class's estimate that comes from the smoothing:
                  lam under JM, alpha * V / (N_c + alpha * V) under Laplace.
    """

    def __init__(self, alpha=1.0, log_tf=True, idf=True, length_norm=True,
                 complement=True, weight_norm=True, transform_test=True,
                 smoothing="laplace", lam=0.1):
        super().__init__(alpha=alpha, log_tf=log_tf, idf=idf, length_norm=length_norm,
                         complement=complement, weight_norm=weight_norm,
                         transform_test=transform_test)
        self.smoothing = smoothing
        self.lam = lam

    def _estimation_counts(self, X, y):
        """Steps 1-4 again: the counts N_ci the estimate is built from (one row per class)."""
        D = self.transformer_.transform(sparse.csr_matrix(X))
        per_class = np.vstack([
            np.asarray(D[y == c].sum(axis=0)).ravel() for c in self.classes_
        ])
        if self.complement:
            return per_class.sum(axis=0, keepdims=True) - per_class
        return per_class

    def fit(self, X, y):
        if self.smoothing not in ("laplace", "jm"):
            raise ValueError("smoothing must be 'laplace' or 'jm'")
        if self.smoothing == "jm" and not 0.0 < self.lam < 1.0:
            raise ValueError("lam must lie strictly between 0 and 1")

        # The parent fit learns the transforms, the classes and the Laplace weights.
        super().fit(X, y)
        y = np.asarray(y)
        counts = self._estimation_counts(X, y)
        n_words = counts.shape[1]
        mass = counts.sum(axis=1)
        self.class_mass_ = mass

        if self.smoothing == "laplace":
            # Parent weights are kept exactly; only the diagnostic is added.
            self.smoothing_share_ = self.alpha * n_words / (mass + self.alpha * n_words)
            return self

        # JM: interpolate the maximum-likelihood estimate with the uniform distribution.
        # A class with no mass at all (cannot happen with real data) falls back to uniform.
        safe_mass = np.where(mass > 0, mass, 1.0)[:, None]
        ml = np.where(mass[:, None] > 0, counts / safe_mass, 1.0 / n_words)
        theta = (1.0 - self.lam) * ml + self.lam / n_words

        # Step 5: log-parameters. Step 6: weight normalisation with |.|, as in the parent.
        w = np.log(theta)
        if self.weight_norm:
            w = w / np.abs(w).sum(axis=1, keepdims=True)
        self.w_ = w
        self.smoothing_share_ = np.full(len(self.classes_), float(self.lam))
        return self


def make_smoothed_variant(name, smoothing="laplace", lam=0.1, **overrides):
    """Return an unfitted SmoothedTWCNB configured as one of the named variants."""
    params = dict(VARIANTS[name])
    params.update(overrides)
    return SmoothedTWCNB(smoothing=smoothing, lam=lam, **params)
