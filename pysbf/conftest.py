import pytest
from scipy.sparse import dok_matrix, csr_matrix


@pytest.fixture()
def G_unweighted() -> csr_matrix:
    """
    :return: unweighted 8x8 matrix as per twitter/sbf examples
    """
    G = dok_matrix((8, 8))

    for i, j, v in EXPECTED_COORDS:
        G[i, j] = 1

    return G.tocsr()


@pytest.fixture()
def G_weighted() -> csr_matrix:
    """
    :return: weighted 8x8 matrix with weights as per twitter/sbf examples
    """
    G = dok_matrix((8, 8))

    for i, j, v in EXPECTED_COORDS:
        G[i, j] = v

    return G.tocsr()


EXPECTED_COORDS = [
    (0, 1, 1.0),
    (0, 2, 1.0),
    (0, 3, 1.0),
    (0, 4, 0.5),
    (1, 0, 1.0),
    (1, 2, 1.0),
    (1, 3, 1.0),
    (2, 0, 1.0),
    (2, 1, 1.0),
    (2, 3, 1.0),
    (3, 0, 1.0),
    (3, 1, 1.0),
    (3, 2, 1.0),
    (4, 0, 0.5),
    (4, 5, 1.0),
    (4, 6, 1.0),
    (4, 7, 1.0),
    (5, 4, 1.0),
    (5, 6, 1.0),
    (5, 7, 1.0),
    (6, 4, 1.0),
    (6, 5, 1.0),
    (6, 7, 1.0),
    (7, 4, 1.0),
    (7, 5, 1.0),
    (7, 6, 1.0)
]
