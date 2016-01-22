from __future__ import absolute_import

import numpy as np

from rpforest.rpforest_fast import (Tree, query_all, encode_all,
                                    get_candidates_all)


SERIALIZATION_VERSION = 2


class RPForest(object):
    """
    Constructs approximate nearest neighbour lookup structures
    via random projection trees [1, 2] and performs approximate
    nearest neighbour queries.
    """

    def __init__(self, leaf_size=1000, no_trees=10):
        """
        Initialise RPForest model.

        Arguments:
        - integer leaf_size: maximum size of a leaf node,
                             used as a stopping criterion
                             in model fitting.
        - no_trees: number of random projection trees to
                    fit.

        At query time, the model will evaluate at most
        leaf_size * no_trees candidates.
        """

        self.leaf_size = leaf_size
        self.no_trees = no_trees

        self.trees = []
        self.dim = None
        self._X = None

        self.serialization_version = SERIALIZATION_VERSION

    def _is_constructed(self):

        return (self.dim is not None
                and self.trees)

    def _has_vectors(self):

        return self._X is not None

    def fit(self, X, normalise=True):
        """
        Construct the random projection forest for points in X.

        Arguments:
        - np.float64 array X [n_points, dim]
        - optional boolean normalise: whether to normalise X. If True,
                                      a copy of X will be made and
                                      normalised.

        Returns:
        - object self
        """

        if X.shape[0] < 1 or X.shape[1] < 1:
            raise Exception('You must supply a valid 2D array.')

        self.dim = X.shape[1]

        if normalise:
            self._X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        else:
            self._X = X

        # Reset the trees list in case of repeated calls to fit
        self.trees = []

        for _ in range(self.no_trees):
            tree = Tree(self.leaf_size, self.dim)
            tree.make_tree(self._X)
            self.trees.append(tree)

        return self

    def clear(self):
        """
        Remove all indexed points and their vectors from the forest,
        but preserve the forest structure for further indexing and
        leaf node lookup.
        """

        if not self._is_constructed():
            raise Exception('Tree has not been fit')

        for tree in self.trees:
            tree.clear()

        self._X = None

    def index(self, point_id, x, normalise=True):
        """
        Index a vector denoted by point_id into the forest.

        Arguments:
        - int point_id
        - np.float64 vector x [dim]
        """

        if not self._is_constructed():
            raise Exception('Tree has not been fit')

        if self._has_vectors():
            raise Exception('Indexing not supported '
                            'without calling .clear() '
                            'first.')

        assert len(x) == self.dim

        if normalise:
            x = x / np.linalg.norm(x)

        for tree in self.trees:
            tree.index(point_id, x)

    def query(self, x, number=10, normalise=True):
        """
        Return nearest neighbours for vector x by first retrieving
        candidate NNs from x's leaf nodes, then merging them
        and sorting by cosine similarity with x.

        Vectors for each point must be available.
        
        At most no_trees * leaf_size NNs will can be returned.

        Arguments:
        - np.float64 vector x [dim]
        - optional int number: number of candidates to return.
        """

        if not self._is_constructed():
            raise Exception('Tree has not been fit')

        if not self._has_vectors():
            raise Exception('No point vectors found.')

        assert self._X.shape[1] == self.dim
        assert len(x) == self.dim

        if normalise:
            x = x / np.linalg.norm(x)

        return query_all(x, self._X, self.trees, number)

    def get_candidates(self, x, number=10, normalise=True):
        """
        Returns candidates for nearest neighbours for x by first
        retrieving NNs from x's leaf nodes, then merging them
        and sorting by the number of leaf nodes they share with x.

        Does not require storage of vectors for each point.

        At most no_trees * leaf_size NNs will can be returned.
        """

        if not self._is_constructed():
            raise Exception('Tree has not been fit')

        assert len(x) == self.dim

        if normalise:
            x = x / np.linalg.norm(x)

        return get_candidates_all(x, self.trees,
                                  self.dim, number)

    def encode(self, x, normalise=True):
        """
        Return a list of names of leaf nodes x belongs to.
        """

        if not self._is_constructed():
            raise Exception('Tree has not been fit')

        assert len(x) == self.dim

        if normalise:
            x = x / np.linalg.norm(x)

        return encode_all(x, self.trees)

    def get_leaf_nodes(self):
        """
        Yield pairs of (string leaf node name, ndarray point indices)
        for all trees in the model.
        """

        for i, tree in enumerate(self.trees):
            for leaf_code, indices in tree.get_leaf_nodes():
                yield '%s:' % i + leaf_code, indices

    def _get_size(self):

        size = 0

        for tree in self.trees:
            size += tree.get_size()

        return size

    def __getstate__(self):

        state = {'dim': self.dim,
                 'leaf_size': self.leaf_size,
                 'no_trees': self.no_trees,
                 'serialization_version': self.serialization_version,
                 'X': self._X}

        tree_states = []

        for tree in self.trees:
            tree_states.append(tree.serialize())

        state['trees'] = tree_states

        return state

    def __setstate__(self, state):

        self.dim = state['dim']
        self.leaf_size = state['leaf_size']
        self.no_trees = state['no_trees']
        self._X = state['X']

        self.trees = []

        for tree_state in state['trees']:
            tree = Tree(self.leaf_size, self.dim)
            tree.deserialize(tree_state, state.get('serialization_version', 1))
            self.trees.append(tree)

        # Make sure that when serialized again it gets the right serialization version
        self.serialization_version = SERIALIZATION_VERSION
