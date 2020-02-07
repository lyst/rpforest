import six
from six.moves import cPickle as pickle
import os

import numpy as np

from sklearn.datasets import load_digits

from rpforest import RPForest


def _get_mnist_data(seed=None):

    digits = load_digits()["images"]

    if seed is not None:
        rnd = np.random.RandomState(seed=seed)
    else:
        rnd = np.random.RandomState()

    no_img, rows, cols = digits.shape
    X = digits.reshape((no_img, rows * cols))
    X = np.ascontiguousarray(X)
    rnd.shuffle(X)

    X_test = X[:100]
    X_train = X[100:]

    return X_train, X_test


def test_find_self():

    X_train, X_test = _get_mnist_data()

    for no_trees, expected_precision in ((1, 0.05), (5, 0.3), (10, 0.5), (50, 0.9)):

        tree = RPForest(leaf_size=10, no_trees=no_trees)
        tree.fit(X_train)

        nodes = {k: set(v) for k, v in tree.get_leaf_nodes()}
        for i, x_train in enumerate(X_train):
            nns = tree.query(x_train, 10)[:10]
            assert nns[0] == i

            point_codes = tree.encode(x_train)

            for code in point_codes:
                assert i in nodes[code]

        tree = pickle.loads(pickle.dumps(tree))

        nodes = {k: set(v) for k, v in tree.get_leaf_nodes()}
        for i, x_train in enumerate(X_train):
            nns = tree.query(x_train, 10)[:10]
            assert nns[0] == i

            point_codes = tree.encode(x_train)

            for code in point_codes:
                assert i in nodes[code]


def test_clear():

    X_train, X_test = _get_mnist_data()

    tree = RPForest(leaf_size=10, no_trees=10)
    tree.fit(X_train)

    for leaf_code, leaf_indices in tree.get_leaf_nodes():
        assert leaf_indices

    tree.clear()

    for leaf_code, leaf_indices in tree.get_leaf_nodes():
        assert not leaf_indices


def test_max_size():

    X_train, X_test = _get_mnist_data()

    tree = RPForest(leaf_size=10, no_trees=10)
    tree.fit(X_train)

    for leaf_code, leaf_indices in tree.get_leaf_nodes():
        assert len(leaf_indices) < 10


def test_encoding_mnist():

    X_train, X_test = _get_mnist_data()

    for no_trees, expected_precision in ((1, 0.05), (5, 0.3), (10, 0.5), (50, 0.9)):

        tree = RPForest(leaf_size=10, no_trees=no_trees)
        tree.fit(X_train)

        for x_train in X_train:
            encodings_0 = tree.encode(x_train)
            encodings_1 = tree.encode(x_train)
            assert encodings_0 == encodings_1

        tree = pickle.loads(pickle.dumps(tree))

        for x_train in X_train:
            encodings_0 = tree.encode(x_train)
            encodings_1 = tree.encode(x_train)
            assert encodings_0 == encodings_1


def test_serialization_mnist():

    X_train, X_test = _get_mnist_data()

    for no_trees, expected_precision in ((1, 0.05), (5, 0.3), (10, 0.5), (50, 0.9)):

        tree = RPForest(leaf_size=10, no_trees=no_trees)
        tree.fit(X_train)

        # Serialize and deserialize
        tree = pickle.loads(pickle.dumps(tree))

        precision = 0.0
        X_train /= np.linalg.norm(X_train, axis=1)[:, np.newaxis]
        for x_test in X_test:
            true_nns = np.argsort(-np.dot(X_train, x_test))[:10]
            nns = tree.query(x_test, 10)[:10]
            assert (nns < X_train.shape[0]).all()

            precision += len(set(nns) & set(true_nns)) / 10.0

        precision /= X_test.shape[0]

        assert precision >= expected_precision


def test_mnist():

    X_train, X_test = _get_mnist_data()

    for no_trees, expected_precision in ((1, 0.05), (5, 0.3), (10, 0.5), (50, 0.9)):

        tree = RPForest(leaf_size=10, no_trees=no_trees)
        tree.fit(X_train)

        precision = 0.0
        X_train /= np.linalg.norm(X_train, axis=1)[:, np.newaxis]
        for x_test in X_test:
            true_nns = np.argsort(-np.dot(X_train, x_test))[:10]
            nns = tree.query(x_test, 10)[:10]
            assert (nns < X_train.shape[0]).all()

            precision += len(set(nns) & set(true_nns)) / 10.0

        precision /= X_test.shape[0]

        assert precision >= expected_precision


def test_candidates_mnist():

    X_train, X_test = _get_mnist_data()

    for no_trees, expected_precision in ((1, 0.05), (5, 0.12), (10, 0.2), (50, 0.5), (80, 0.6)):

        tree = RPForest(leaf_size=10, no_trees=no_trees)
        tree.fit(X_train)

        precision = 0.0
        X_train /= np.linalg.norm(X_train, axis=1)[:, np.newaxis]
        for x_test in X_test:
            true_nns = np.argsort(-np.dot(X_train, x_test))[:10]
            check_nns = tree.get_candidates(x_test, 100000)
            assert len(check_nns) == len(set(check_nns))
            assert -1 not in check_nns
            assert (check_nns < X_train.shape[0]).all()
            nns = tree.get_candidates(x_test, 10)[:10]
            assert (nns < X_train.shape[0]).all()

            precision += len(set(nns) & set(true_nns)) / 10.0

        precision /= X_test.shape[0]

        assert precision >= expected_precision


def test_sample_training():

    X_train, X_test = _get_mnist_data()

    for no_trees, expected_precision in ((1, 0.05), (5, 0.3), (10, 0.5), (50, 0.9)):

        tree = RPForest(leaf_size=10, no_trees=no_trees)
        # Fit on quarter of data
        slice_size = int(X_train.shape[0] / 4)
        X_sample = X_train[:slice_size]
        tree.fit(X_sample)
        # Clear and index everything
        tree.clear()
        for i, x in enumerate(X_train):
            tree.index(i, x)
        tree._X = X_train

        precision = 0.0
        X_train /= np.linalg.norm(X_train, axis=1)[:, np.newaxis]
        for x_test in X_test:
            true_nns = np.argsort(-np.dot(X_train, x_test))[:10]
            nns = tree.query(x_test, 10)[:10]

            precision += len(set(nns) & set(true_nns)) / 10.0

        precision /= X_test.shape[0]

        assert precision >= expected_precision


def test_multiple_fit_calls():

    X_train, X_test = _get_mnist_data()

    tree = RPForest(leaf_size=10, no_trees=10)
    tree.fit(X_train)

    assert len(tree.trees) == 10

    tree.fit(X_train)

    assert len(tree.trees) == 10


def test_load_v1_model():
    """
    Make sure that models serialized using older versions deserialize correctly
    """

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rpforest_v1.pickle")

    with open(path, "rb") as fl:
        data = fl.read()
        if six.PY2:
            tree = pickle.loads(data)
        elif six.PY3:
            tree = pickle.loads(data, encoding="iso-8859-1")
        else:
            assert False

    X_train, X_test = _get_mnist_data(seed=10)

    nodes = {k: set(v) for k, v in tree.get_leaf_nodes()}
    for i, x_train in enumerate(X_train):
        nns = tree.query(x_train, 10)[:10]
        assert nns[0] == i

        point_codes = tree.encode(x_train)

        for code in point_codes:
            assert i in nodes[code]

    tree = pickle.loads(pickle.dumps(tree))

    nodes = {k: set(v) for k, v in tree.get_leaf_nodes()}
    for i, x_train in enumerate(X_train):
        nns = tree.query(x_train, 10)[:10]
        assert nns[0] == i

        point_codes = tree.encode(x_train)

        for code in point_codes:
            assert i in nodes[code]

    # Pickle and unpickle again
    tree = pickle.loads(pickle.dumps(tree))

    nodes = {k: set(v) for k, v in tree.get_leaf_nodes()}
    for i, x_train in enumerate(X_train):
        nns = tree.query(x_train, 10)[:10]
        assert nns[0] == i

        point_codes = tree.encode(x_train)

        for code in point_codes:
            assert i in nodes[code]

    tree = pickle.loads(pickle.dumps(tree))

    nodes = {k: set(v) for k, v in tree.get_leaf_nodes()}
    for i, x_train in enumerate(X_train):
        nns = tree.query(x_train, 10)[:10]
        assert nns[0] == i

        point_codes = tree.encode(x_train)

        for code in point_codes:
            assert i in nodes[code]
