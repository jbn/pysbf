from pathlib import Path
from pysbf.reference import Experiment, sbf_jar_path, load_assignments
from pysbf.metis import write_metis


def test_experiment(tmp_path: Path, G_unweighted):
    metis_path = tmp_path / "unweighted_metis.txt"
    write_metis(G_unweighted, metis_path)

    experiment = Experiment(
        name='my_experiment',
        metis_file=metis_path,
        output_dir=tmp_path
    )

    experiment.run(sbf_jar_path())

    R_i_to_v = list('abcdefghi')
    clusters = load_assignments(experiment.assignments_path, R_i_to_v)

    assert set(clusters) == {0, 1, 2}

    # TODO: is this being property random seeded?
    # assert set(tuple(sorted(assignments)) for assignments in clusters.values()) == {
    #     ('a',),
    #     ('e', 'f', 'g', 'h'),
    #     ('a', 'b', 'c', 'd'),
    # }
    # assert {
    #     0: {'a'},
    #     1: {'e', 'f', 'g', 'h'},
    #     2: {'a', 'b', 'c', 'd'},
    # } == clusters
