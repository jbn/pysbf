import os
import subprocess
import shlex
from collections import defaultdict
from enum import Enum

from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List, Sequence, Any, Dict, Set


class ProposalStrategy(str, Enum):
    FractionOfNeighborhoods = 'FractionOfNeighborhoods'
    FractionOfNeighborsAndCluster = 'FractionOfNeighborsAndCluster'
    SingleOrZeroMembershipLikelihood = 'SingleOrZeroMembershipLikelihood'
    MultipleMembershipLikelihood = 'MultipleMembershipLikelihood'


class Experiment(BaseModel):
    name: str
    metis_file: Path
    output_dir: Path

    cpu: Optional[int] = 10
    k: Optional[int] = 10_000
    scale_coeff: Optional[float] = None
    max_epoch: Optional[int] = 10
    eps: Optional[float] = None
    eval_ratio: Optional[float] = 0.1
    eval_every: Optional[int] = None
    update_immediately: Optional[bool] = None
    no_locking: Optional[bool] = None
    min_cluster_size: Optional[int] = None
    random_seed: Optional[int] = 42
    proposal_strategy: Optional[ProposalStrategy] = ProposalStrategy.MultipleMembershipLikelihood
    max_memberships_per_vertex: Optional[int] = 5
    init_from_nonoverlapping_neighborhood: Optional[bool] = None
    init_from_random_neighborhood: Optional[bool] = None
    init_from_best_neighborhood: Optional[bool] = None
    run_all_epochs: Optional[bool] = None
    use_weight_coeff_for_proposal: Optional[bool] = None
    divide_result_into_connected_components: Optional[bool] = None
    remove_weak_links_for_connected_components: Optional[bool] = None
    wt_coeff: Optional[float] = None
    use_temperature_schedule: Optional[bool] = None
    max_temperature: Optional[float] = None

    def run(self, jar_path: Path):
        self.write_config()

        # Run the experiment
        command = f"java -jar {jar_path} {self.config_path}"
        process = subprocess.Popen(
            shlex.split(command),
            stderr=subprocess.PIPE,
            encoding='utf8'
        )

        lines = []
        while True:
            line = process.stderr.readline()
            if line == '' and process.poll() is not None:
                break

            if line:
                lines.append(line)

        rc = process.poll()
        if rc != 0:
            err_msg = "\n".join(lines)
            raise RuntimeError(f"An error occurred {rc}: {err_msg}")

        # Save the standard error (results)
        with self.outputs_path.open('w') as fp:
            for line in lines:
                fp.write(line)

    @property
    def config_path(self) -> Path:
        return self.output_dir / f"{self.name}.config"

    @property
    def assignments_path(self) -> Path:
        return self.output_dir / f"{self.name}_assignments.txt"

    @property
    def outputs_path(self) -> Path:
        return self.output_dir / f"{self.name}.output"

    def write_config(self) -> Path:
        output_path = self.config_path
        assert not output_path.exists(), f"{output_path} exists!"

        lines = [f'outputByRowsFile {str(self.assignments_path)}']

        for k, v in self.__dict__.items():
            if k in _NON_PARAM_NAMES:
                continue
            elif k in _SPECIAL_CASES:
                k = _SPECIAL_CASES.get(k)
            else:
                k = k.replace('_', ' ').title().replace(' ', '')
                k = k[0].lower() + k[1:]

            if v is not None:
                lines.append(f"{k} {v}")

        with output_path.open("w") as fp:
            fp.write("\n".join(lines) + "\n")

        return output_path


def sbf_jar_path() -> Path:
    if (jar_path := os.environ.get("SBF_JAR_PATH")) is not None:
        return Path(jar_path)
    return Path(__file__).parent.parent / "sbf-1.0.0.jar"


def load_assignments(path: Path, R_i_to_v: Sequence[Any]) -> Dict[Any, Set[int]]:
    clusters = defaultdict(set)

    with path.open() as fp:
        for claim_i, cluster_assignments in enumerate(fp):
            claim_id = R_i_to_v[claim_i]

            for cluster in (int(part) for part in cluster_assignments.split()):
                clusters[cluster-1].add(claim_id)

    return dict(clusters)


_NON_PARAM_NAMES = {'name', 'output_dir'}
_SPECIAL_CASES = {'k': 'K'}


