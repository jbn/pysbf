from typing import Optional, Tuple, Any, Iterable, Set, Generator, Dict, Sequence
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


def build_R_projection(
    edges: Iterable[Tuple[Any, Any]],
    R_nodes: Iterable[Any],
    top_k_R_nodes: Optional[int] = 100_000,
    L_node_degree_upper_bound: Optional[int] = 5_000,
    L_node_prune_max_iter: int = 100,
    normalize_rows_before_computing_similarity: bool=True,
    prune_similarity_to_percentile: Optional[float] = None,
    prune_similarity_below: Optional[float] = None,
):
    """

    :param prune_similarity_below:
    :param edges: edges to add to the bipartate (item, user) graph
    :param R_nodes: nodes in edges that are item (right) nodes
    :param top_k_R_nodes: keep top k R nodes (or don't trim if None)
    :param L_node_degree_upper_bound: maximum number of edges per L node
    :param L_node_prune_max_iter: maximum number of prune iterations to run
    :param prune_similarity_to_percentile: prune final similarity matrix to a percentile
    :return:
    """
    B, R_nodes = _build_bipartite_graph(edges, R_nodes)

    if top_k_R_nodes is not None:
        _trim_to_top_k_R_nodes(B, R_nodes, top_k_R_nodes)

    # Prune all L nodes that have only 1 degree or are above the degree upper bound
    _prune_L_nodes_by_degree(B, R_nodes, degree_upper_bound=L_node_degree_upper_bound, max_iter=L_node_prune_max_iter)

    _clear_all_empty_nodes(B)

    R_nodes = {n for n in B.nodes if n in R_nodes}
    L_nodes = {n for n in B.nodes if n not in R_nodes}

    R_i_to_v = list(R_nodes)
    R_v_to_i = {v: i for i, v in enumerate(R_i_to_v)}

    L_i_to_v = list(L_nodes)
    L_v_to_i = {v: i for i, v in enumerate(L_i_to_v)}

    G = _build_similarity_matrix(
        B, R_nodes, L_v_to_i, L_i_to_v, R_v_to_i, R_i_to_v,
        normalize_rows=normalize_rows_before_computing_similarity
    )

    pruned_G = False
    if prune_similarity_to_percentile is not None:
        idx = G.data < np.percentile(G.data, prune_similarity_to_percentile)
        G.data[idx] = 0
        pruned_G = True

    if prune_similarity_below:
        idx = G.data < prune_similarity_below
        G.data[idx] = 0
        pruned_G = True

    if pruned_G:
        G.eliminate_zeros()

    return G, L_i_to_v, L_v_to_i, R_i_to_v, R_v_to_i


def _build_bipartite_graph(edges: Iterable[Tuple[Any, Any]], R_nodes: Iterable[Any]) -> Tuple[nx.Graph, Set[Any]]:
    R_nodes = set(R_nodes)

    B = nx.Graph()
    B.add_edges_from(edges)

    return B, R_nodes


def _iter_empty_nodes(B: nx.Graph) -> Generator[Any, None, None]:
    return (n for n, degree in B.degree() if degree == 0)


def _clear_all_empty_nodes(B):
    empties = list(_iter_empty_nodes(B))
    B.remove_nodes_from(empties)
    return len(empties)


def _trim_to_top_k_R_nodes(B: nx.Graph, R_nodes: Set[Any], top_k: int) -> int:
    remaining_R_nodes = {n for n in B.nodes if n in R_nodes}

    top_pairs = sorted(
        (pair for pair in B.degree if pair[0] in remaining_R_nodes),
        key=lambda p: -p[1]
    )[:int(top_k)]

    to_remove = remaining_R_nodes - {pair[0] for pair in top_pairs}
    B.remove_nodes_from(to_remove)

    return len(to_remove)


def _prune_L_nodes_with_degree_of_1_or_above_threshold_and_singleton_R_nodes(B: nx.Graph, R_nodes: Set[Any], degree_upper_bound: int) -> Tuple[int, int]:
    # This function is confusing because order matters. The L nodes get removed first
    # But this can create a situation where R nodes that previously had degree > 1
    # no longer do.

    # Calculate L nodes
    L_nodes = {n for n in B.nodes() if n not in R_nodes}

    # Grab L nodes to remove
    L_nodes_to_remove = {
        n
        for n, k_degree in B.degree()
        if n in L_nodes and (k_degree == 1 or k_degree > degree_upper_bound)
    }

    # Remove them
    B.remove_nodes_from(L_nodes_to_remove)

    # Find R_nodes with degree one
    R_nodes_to_remove = {
        n
        for n, k_degree in B.degree()
        if n not in L_nodes and k_degree <= 1
    }

    # Remove them
    B.remove_nodes_from(R_nodes_to_remove)

    return len(L_nodes_to_remove), len(R_nodes_to_remove)


def _prune_L_nodes_by_degree(
    B: nx.Graph,
    R_nodes: Set[Any],
    degree_upper_bound: int,
    max_iter: int = 100,
    debug: bool = True
) -> bool:
    if debug:
        print("pruning_B_by_degree")

    for i in range(max_iter):
        n_L_removed, n_R_removed = _prune_L_nodes_with_degree_of_1_or_above_threshold_and_singleton_R_nodes(B, R_nodes, degree_upper_bound=degree_upper_bound)

        if debug:
            print(f"\t{i} {n_L_removed=} {n_R_removed=} n_edges={B.number_of_edges()}")

        if n_L_removed == 0 and n_R_removed == 0:
            # Keep iteratively pruning until pruning can stop
            return True

    return False


def _build_similarity_matrix(
    B: nx.Graph,
    R_nodes: Set[Any],
    L_v_to_i: Dict[Any, int],
    L_i_to_v: Sequence[Any],
    R_v_to_i: Dict[Any, int],
    R_i_to_v: Sequence[Any],
    normalize_rows: bool = True

) -> csr_matrix:
    row_ind, col_ind = [], []

    for l, r in B.edges():

        if r not in R_nodes:
            # I am using an undirected graph as a bipartite graph by maintaining
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

    if normalize_rows:
        M = M.multiply(csr_matrix(1 / np.sqrt(M.multiply(M).sum(1))))
        M = M.tocsr()

    # WARNING: Computing the cosine similarity can be very expensive at scale.
    # Twitter spent considerable effort engineering a map-reduce to do it instead.
    G = M @ M.T

    return G
