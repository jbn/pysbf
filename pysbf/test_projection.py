import networkx as nx
import numpy as np
from numpy.testing import assert_allclose

from pysbf.projection import _build_bipartite_graph, _iter_empty_nodes, _clear_all_empty_nodes, _trim_to_top_k_R_nodes, \
    _prune_L_nodes_with_degree_of_1_or_above_threshold_and_singleton_R_nodes, _build_similarity_matrix


def test__clear_all_empty_nodes():
    B, R_nodes = _build_bipartite_graph(
        [(1, 2), (1, 3), (4, 5)],
        [1]
    )

    empties = list('xy')
    B.add_nodes_from(empties)
    assert all(n in B.nodes for n in empties)
    assert list(_iter_empty_nodes(B)) == empties

    _clear_all_empty_nodes(B)

    assert all(n not in B.nodes for n in empties)


def test__trim_to_top_k_r_nodes():
    B, R_nodes = _build_bipartite_graph(
        [
            # order here important for so it's not just
            # accidentally passing on insertion order artifact
            ('top_3', 5),
            ('top_1', 2), ('top_1', 3),
            ('top_2', 5), ('top_2', 3),
        ],
        ['top_3', 'top_1', 'top_2']
    )

    assert _trim_to_top_k_R_nodes(B, R_nodes, 2) == 1

    assert 'top_3' not in B.nodes


def test__prune_l_nodes_with_degree_of_1_or_above_threshold_and_singleton_R_nodes():
    B, R_nodes = _build_bipartite_graph(
        [
            (1, 'a'),
            (2, 'a'),
            (2, 'b'),
            (3, 'a'),
            (3, 'b'),
            (3, 'c'),
            (4, 'a'),
            (4, 'b'),
            (4, 'c'),
            (4, 'd'),
            (5, 'c'),
            (9, 'x')
        ],
        list('abcd')
    )

    n_removed_L, n_removed_R = _prune_L_nodes_with_degree_of_1_or_above_threshold_and_singleton_R_nodes(
        B, R_nodes, degree_upper_bound=3,
    )

    assert n_removed_L == 5
    assert n_removed_R == 2
    assert set(B.nodes) == {'a', 'b', 2, 3}


def test__build_similarity_matrix():
    B, R_nodes = _build_bipartite_graph(
        [
            ('a', 1), ('a', 2), ('a', 3),
            ('b', 1), ('b', 3),
            ('c', 1),
        ],
        list('abc')
    )

    G = _build_similarity_matrix(
        B,
        R_nodes,
        {1: 0, 2: 1, 3: 2},
        [1, 2, 3],
        {'a': 0, 'b': 1, 'c': 2},
        list('abc'),
        prune_percentile=False
    )

    expected = np.array(
        [[1., 0.81649655, 0.57735026],
         [0.81649655, 1., 0.70710677],
         [0.57735026, 0.70710677, 1.]]
    )

    assert_allclose(G.todense(), expected)
