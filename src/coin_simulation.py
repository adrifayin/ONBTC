"""
Exact reproduction of Table 1 in Rennie et al. (2003): skewed data bias.

Class 1's coin lands heads with probability 0.25, class 2's with 0.20.
Each coin's heads rate is estimated (maximum likelihood) from n1 and n2 training
flips, and a single 'heads' is labelled with the class whose estimate is higher.
Instead of simulating, we enumerate every possible training outcome exactly.
Run from the repository root:  python -m src.coin_simulation
"""
from fractions import Fraction
from itertools import product
from math import comb


def binomial_pmf(n, p, k):
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def win_probabilities(n1, n2, p1=0.25, p2=0.20):
    """P(heads labelled class 1), P(heads labelled class 2); ties are labelled neither."""
    win1 = win2 = 0.0
    for k1, k2 in product(range(n1 + 1), range(n2 + 1)):
        prob = binomial_pmf(n1, p1, k1) * binomial_pmf(n2, p2, k2)
        est1, est2 = Fraction(k1, n1), Fraction(k2, n2)   # exact, no rounding ties
        if est1 > est2:
            win1 += prob
        elif est2 > est1:
            win2 += prob
    return win1, win2


def main():
    print("Training flips (class 1, class 2) -> P(labelled class 1), P(labelled class 2)")
    for n1, n2 in [(1, 2), (2, 2), (5, 10), (10, 10), (10, 100), (100, 100)]:
        w1, w2 = win_probabilities(n1, n2)
        flag = "  <- biased toward class 2" if w2 > w1 else ""
        print(f"  ({n1:>3}, {n2:>3}) -> {w1:.4f}, {w2:.4f}{flag}")
    print("Paper, Table 1: (1, 2) -> 0.24, 0.27")


if __name__ == "__main__":
    main()
