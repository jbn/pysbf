from typing import Optional

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


def build_R_projection(edges, R_nodes, top_k: int = 100_000, prune_percentile: Optional[float] = None):
    B, R_nodes = _build_bipartite_graph(edges, R_nodes)

    # There shouldn't be any initial empty nodes
    _clear_all_empty_nodes(B)

    _trim_to_top_k_R_nodes(B, R_nodes, top_k)

    _prune_B_by_degree(B, R_nodes)

    _clear_all_empty_nodes(B)

    R_nodes = {n for n in B.nodes if n in R_nodes}
    L_nodes = {n for n in B.nodes if n not in R_nodes}

    R_i_to_v = list(R_nodes)
    R_v_to_i = {v: i for i, v in enumerate(R_i_to_v)}

    L_i_to_v = list(L_nodes)
    L_v_to_i = {v: i for i, v in enumerate(L_i_to_v)}

    G = _build_csr_matrix(B, R_nodes, L_v_to_i, L_i_to_v, R_v_to_i, R_i_to_v, prune_percentile)

    return G, L_i_to_v, L_v_to_i, R_i_to_v, R_v_to_i


def _build_bipartite_graph(edges, R_nodes):
    R_nodes = set(R_nodes)

    B = nx.Graph()
    B.add_edges_from(edges)

    return B, R_nodes


def _iter_empty_nodes(B):
    return (n for n, degree in B.degree() if degree == 0)


def _clear_all_empty_nodes(B):
    empties = list(_iter_empty_nodes(B))
    B.remove_nodes_from(empties)
    return len(empties)


def _trim_to_top_k_R_nodes(B, R_nodes, top_k=int(1e5)):
    remaining_R_nodes = {n for n in B.nodes if n in R_nodes}

    top_pairs = sorted(
        [pair for pair in B.degree if pair[0] in remaining_R_nodes],
        key=lambda p: -p[1]
    )[:int(top_k)]

    top_R_nodes = {pair[0] for pair in top_pairs}
    to_remove = remaining_R_nodes - top_R_nodes
    B.remove_nodes_from(to_remove)

    return len(to_remove)


def _prune_1_and_above_threshold(B, R_nodes, upper_bound=500):
    # Calculate L nodes
    L_nodes = {n for n in B.nodes() if n not in R_nodes}

    # Grab L nodes to remove
    L_nodes_to_remove = set()
    for n, k_degree in B.degree():
        if n in L_nodes and (k_degree == 1 or k_degree > upper_bound):
            L_nodes_to_remove.add(n)

    # Remove them
    B.remove_nodes_from(L_nodes_to_remove)

    # Find R_nodes with no edges
    R_nodes_to_remove = set()
    for n, k_degree in B.degree():
        if n not in L_nodes and k_degree == 1:
            R_nodes_to_remove.add(n)

    # Remove them
    B.remove_nodes_from(R_nodes_to_remove)

    return len(L_nodes_to_remove), len(R_nodes_to_remove)


def _prune_B_by_degree(B, R_nodes, degree_upper_bound=5_000, max_iter=100, debug=True) -> bool:
    for i in range(max_iter):
        n_L_removed, n_R_removed = _prune_1_and_above_threshold(B, R_nodes, upper_bound=degree_upper_bound)

        if debug:
            print(f"{i} {n_L_removed=} {n_R_removed=} n_edges={B.number_of_edges()}")

        if n_L_removed == 0 and n_R_removed == 0:
            return True

    return False


def _build_csr_matrix(B, R_nodes, L_v_to_i, L_i_to_v, R_v_to_i, R_i_to_v, prune_percentile: Optional[float]):
    row_ind, col_ind = [], []

    for l, r in B.edges():

        if r not in R_nodes:
            # I am using a unidirected graph as a bipartite graph by maintaining
            # the R nodes. However, networkx is under no obligation to return my
            # R nodes on the right and the L nodes on the left.
            l, r = r, l

        row_ind.append(R_v_to_i[r])
        col_ind.append(L_v_to_i[l])

    M = csr_matrix(
        (
            np.ones(len(row_ind), dtype=np.float32),
            (row_ind, col_ind),
        ),
        shape=(len(R_i_to_v), len(L_i_to_v))
    )

    M = M.multiply(csr_matrix(1 / np.sqrt(M.multiply(M).sum(1))))
    M = M.tocsr()

    G = M @ M.T

    if prune_percentile is not None:
        idx = G.data < np.percentile(G.data, prune_percentile)
        G.data[idx] = 0
        G.eliminate_zeros()

    return G

