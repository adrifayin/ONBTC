"""
Dataset loaders. Each returns (train_texts, test_texts, y_train, y_test).
"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


def load_20ng(setting="random", seed=0, test_size=0.2):
    """20 Newsgroups.

    setting="random" : the paper's protocol (Appendix of Rennie et al.):
                       all documents, headers kept, stratified random 80/20 split.
    setting="bydate" : scikit-learn's standard chronological split, with headers,
                       signatures and quoted replies removed (harder, more realistic).
    """
    if setting == "random":
        data = fetch_20newsgroups(subset="all")
        X_tr, X_te, y_tr, y_te = train_test_split(
            data.data, data.target, test_size=test_size,
            stratify=data.target, random_state=seed)
        return X_tr, X_te, np.asarray(y_tr), np.asarray(y_te)
    if setting == "bydate":
        remove = ("headers", "footers", "quotes")
        train = fetch_20newsgroups(subset="train", remove=remove)
        test = fetch_20newsgroups(subset="test", remove=remove)
        return train.data, test.data, train.target, test.target
    raise ValueError("setting must be 'random' or 'bydate'")


def _reuters():
    """Return NLTK's Reuters corpus, downloading it the first time."""
    import nltk
    from nltk.corpus import reuters
    try:
        reuters.fileids()
    except LookupError:
        nltk.download("reuters", quiet=True)
    return reuters


def load_reuters_single_label():
    """Reuters-21578, ModApte split (NLTK copy), single-label documents only.

    Keeps documents with exactly one topic, then keeps only topics that appear
    in both the training and the test set. The result is close to R52.
    """
    reuters = _reuters()
    train_ids, test_ids = [], []
    for fid in reuters.fileids():
        if len(reuters.categories(fid)) != 1:
            continue
        if fid.startswith("training/"):
            train_ids.append(fid)
        else:
            test_ids.append(fid)

    def label(fid):
        return reuters.categories(fid)[0]

    shared = {label(f) for f in train_ids} & {label(f) for f in test_ids}
    train_ids = [f for f in train_ids if label(f) in shared]
    test_ids = [f for f in test_ids if label(f) in shared]

    X_tr = [reuters.raw(f) for f in train_ids]
    X_te = [reuters.raw(f) for f in test_ids]
    y_tr = np.array([label(f) for f in train_ids])
    y_te = np.array([label(f) for f in test_ids])
    return X_tr, X_te, y_tr, y_te


def describe(name, y_train, y_test):
    """Print the dataset statistics needed for the report."""
    classes, counts = np.unique(y_train, return_counts=True)
    print(f"{name}: {len(y_train)} train / {len(y_test)} test documents, "
          f"{len(classes)} classes; training docs per class: "
          f"largest {counts.max()}, smallest {counts.min()}")
