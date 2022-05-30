from pathlib import Path

from pysbf.reference import Experiment
from pysbf.metis import write_metis


def test_experiment(tmp_path: Path, G_unweighted):
    metis_path = tmp_path / "unweighted_metis.txt"
    write_metis(G_unweighted, metis_path)

    experiment = Experiment(
        name='my_experiment',
        metis_file=metis_path,
        output_dir=tmp_path
    )

    got = experiment.run(Path("/Users/generativist/Projects/pysbf/sbf-1.0.0.jar"))
    assert len(got) > 0
