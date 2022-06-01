from pathlib import Path
from pysbf.reference import Experiment, sbf_jar_path
from pysbf.metis import write_metis


def test_experiment(tmp_path: Path, G_unweighted):
    metis_path = tmp_path / "unweighted_metis.txt"
    write_metis(G_unweighted, metis_path)

    experiment = Experiment(
        name='my_experiment',
        metis_file=metis_path,
        output_dir=tmp_path
    )

    got = experiment.run(sbf_jar_path())
    assert len(got) > 0
