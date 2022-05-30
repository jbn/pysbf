from pathlib import Path

import numpy as np

from pysbf.metis import write_metis, read_metis


def test_metis_unweighted(G_unweighted, tmp_path: Path):
    metis_path = tmp_path / "metis.txt"
    write_metis(G_unweighted, metis_path, weighted=False)
    assert np.all(G_unweighted.todense() == read_metis(metis_path).todense())


def test_metis_weighted(G_weighted, tmp_path: Path):
    metis_path = tmp_path / "metis.txt"
    write_metis(G_weighted, metis_path, weighted=True)
    assert np.all(G_weighted.todense() == read_metis(metis_path).todense())


