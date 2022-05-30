from pathlib import Path
from typing import Iterable
import numpy as np
from scipy.sparse import csr_matrix


def write_metis(G: csr_matrix, output_path: Path, *, run_checks: bool = True, weighted: bool = False):
    i, j = G.nonzero()

    if run_checks:

        # I am going to create the adjacency list by iterating over
        # the (i, j) pairs. For this to work, i must be sorted already.
        assert sorted(i) == list(i)

        # And the first element must be the 0 index
        assert i[0] == 0

        # There should not be unconnected nodes
        assert i.max() == j.max()

        # The graph should be symmetric
        assert np.all(np.abs((G - G.T).data) < 1e-10)

    n_vertices = i.max() + 1           # its zero-indexed
    n_edges = len(i) - (i == j).sum()  # no self-edges
    n_edges_adj = n_edges // 2         # symmetric

    with output_path.open("w") as fp:

        # Write the header line (sbf checks this)
        if weighted:
            fp.write(f"{n_vertices} {n_edges_adj} 1\n")
        else:
            fp.write(f"{n_vertices} {n_edges_adj}\n")

        items = _iter_adj(i, j)
        if weighted:
            items = _iter_with_values(G, items)

        for line_no, bunch in items:
            if bunch:
                fp.write(f"{' '.join((str(n) for n in bunch))}\n")
            else:
                fp.write("\n")

            if line_no % 1_000 == 0:
                print(f"finished emiting {line_no}...")


def _iter_with_values(G, items):
    for line_no, bunch in items:
        expanded_bunch = []

        for j in bunch:
            expanded_bunch.append(j)
            expanded_bunch.append(G[line_no, j-1])

        yield line_no, expanded_bunch


def _iter_adj(left, right):
    last_i, bunch = -1, None

    for i, j in zip(left, right):

        if i != last_i:

            # There were interveining empty nodes.
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

    yield last_i, sorted(bunch)


def _read_uncommented_lines(path: Path):
    with path.open() as fp:
        for line in fp:
            if line.startswith('%'):
                continue

            yield line.strip()


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


def read_metis(path: Path, skip_checks: bool = False, dtype='f4') -> csr_matrix:
    lines = iter(_read_uncommented_lines(path))
    header = [int(s) for s in next(lines).split()]

    if (n := len(header)) == 2:
        n_vertices, n_edges = header
        is_weighted = 0
    elif n == 3:
        n_vertices, n_edges, is_weighted = header
    else:
        raise ValueError("Expected header to have 2 or 3 elements got: {header}")

    if is_weighted != 1:
        M = _read_unweighted_edges(lines, dtype=dtype)
    else:
        M = _read_weighted_edges(lines, dtype=dtype)

    if not skip_checks:
        if M.nnz // 2 != n_edges:
            raise ValueError(f"Expected {n_edges=} got {M.nnz // 2}")

    return M


