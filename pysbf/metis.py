from pathlib import Path
from typing import Iterable, Union, Tuple, Iterator
import numpy as np
from scipy.sparse import csr_matrix


def read_metis(path: Union[Path, str], run_checks: bool = True, dtype: str = 'f4') -> csr_matrix:
    """
    Read a metis file as a sparse csr matrix.

    :param path: the path to the metis file
    :param run_checks: if True, run checks for expectations (e.g. is symmetric)
    :param dtype: the datatype of the sparse matrix (default 32 byte float)

    :return: the represented csr matrix
    """
    # Metis files are `%-prefix` commented, ignore those comments
    lines = iter(_read_uncommented_lines(path))

    # The header line is enforced.
    n_vertices, n_edges, is_weighted = _pop_metis_header(lines)

    if is_weighted:
        M = _read_weighted_edges(lines, dtype=dtype)
    else:
        M = _read_unweighted_edges(lines, dtype=dtype)

    if run_checks:
        # The edges are unweighted so the symmetric matrix doubles them.
        half_nonzero = M.nnz // 2
        if half_nonzero != n_edges:
            raise ValueError(f"Expected {n_edges=} got {half_nonzero}")

    return M


def write_metis(G: csr_matrix, output_path: Union[Path, str], *, run_checks: bool = True, is_weighted: bool = False):
    """
    Write a sparse matrix in metis format.

    :param G: symmetric csr matrix, weighted if weighted is True
    :param output_path: the output path
    :param run_checks: checks that the matrix is symmetric and zero-indexed
    :param is_weighted: if True assume data in G is weighted, otherwise assume 1
    """
    i, j = G.nonzero()

    if run_checks:

        # I am going to create the adjacency list by iterating over
        # the (i, j) pairs. For this to work, `i` must be sorted already.
        # this is a property of csr_matrix but doesn't seem guaranteed
        # anywhere
        assert sorted(i) == list(i)

        # And the first element must be the 0 index
        assert i[0] == 0

        # The graph should be symmetric
        assert np.all(np.abs((G - G.T).data) < 1e-10)

    # noinspection PyTypeChecker
    self_edges: np.ndarray = i == j

    n_vertices = i.max() + 1             # its zero-indexed
    n_edges = len(i) - self_edges.sum()  # no self-edges
    n_edges_adj = n_edges // 2           # symmetric

    with Path(output_path).open("w") as fp:
        fp.write(_sbf_header(n_vertices, n_edges_adj, is_weighted))

        for line_no, bunch in _iter_metis_rows(G, i, j, is_weighted):
            if bunch:
                fp.write(f"{' '.join((str(n) for n in bunch))}\n")
            else:
                # the metis file infers node id from the line number
                # you must emit empty lines for nodes with degree 0
                fp.write("\n")

            if line_no % 1_000 == 0:  # TODO: make extensible
                print(f"...finished emitting {line_no}")


def _pop_metis_header(lines: Iterator[str]) -> Tuple[int, int, bool]:
    header = [int(s) for s in next(lines).split()]

    if (n := len(header)) == 2:
        n_vertices, n_edges = header
        is_weighted = 0
    elif n == 3:
        n_vertices, n_edges, is_weighted = header
    else:
        raise ValueError(f"Expected header to have 2 or 3 elements got: {header}")

    return n_vertices, n_edges, is_weighted == 1


def _sbf_header(n_vertices: int, n_edges_adj: int, is_weighted: bool):
    if is_weighted:
        return f"{n_vertices} {n_edges_adj} 1\n"
    else:
        return f"{n_vertices} {n_edges_adj}\n"


def _iter_metis_rows(G: csr_matrix, i: np.ndarray, j: np.ndarray, is_weighted: bool):
    if is_weighted:
        return _iter_with_values(G, _iter_adj(i, j))
    else:
        return _iter_adj(i, j)


def _iter_with_values(G, items):
    for line_no, bunch in items:
        expanded_bunch = []

        for j in bunch:
            # Expand from just the node to the `node weight` pairs
            expanded_bunch.append(j)
            expanded_bunch.append(G[line_no, j-1])

        yield line_no, expanded_bunch


def _iter_adj(left, right):
    last_i, bunch = -1, None

    for i, j in zip(left, right):

        if i != last_i:

            # There were intervening empty nodes.
            for missing_i in range(last_i + 1, i):
                print(f"{last_i=} {i=}")
                yield missing_i, []

            # Ignore initial bunch.
            if bunch is not None:
                yield last_i, sorted(bunch)

            # Reset accumulator
            bunch, last_i = [], i

        # Ignore self-loops
        if i != j:
            bunch.append(j + 1)

    if bunch:
        # TODO: subtle error possible if last node or nodes have 0 degree
        yield last_i, sorted(bunch)


def _read_uncommented_lines(path: Path):
    with path.open() as fp:
        # if you return the generator itself it will fail on closed file
        yield from (line.strip() for line in fp if not line.startswith("%"))


def _read_unweighted_edges(lines: Iterable[str], dtype='f4'):
    i, j = [], []

    for u, line in enumerate(lines):

        for v in line.split():
            i.append(u)
            j.append(int(v) - 1)

    return csr_matrix((np.ones(len(i), dtype=dtype), (i, j)))


def _read_weighted_edges(lines: Iterable[str], dtype='f4'):
    i, j, data = [], [], []

    for u, line in enumerate(lines):
        parts = line.split()
        if len(parts) % 2 != 0:
            raise ValueError(f"line invalid (odd number of elements): {line}")

        for v, w in zip(parts[0::2], parts[1::2]):
            i.append(u)
            j.append(int(v) - 1)
            data.append(float(w))

    return csr_matrix((data, (i, j)), dtype=dtype)
