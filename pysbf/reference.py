import os
import subprocess
import shlex

from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List



class Experiment(BaseModel):
    name: str
    metis_file: Path
    output_dir: Path

    k: Optional[int] = 10_000
    eval_ratio: Optional[float] = 0.1
    random_seed: Optional[int] = 42
    max_memberships_per_vertex: Optional[int] = 5
    proposal_strategy: Optional[str] = 'MultipleMembershipLikelihood'
    max_epoch: Optional[int] = 10

    def run(self, jar_path: Path) -> List[str]:
        config_path = self.write_config(self.name)

        # Run the experiment
        command = f"java -jar {jar_path} {config_path}"
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
            raise RuntimeError(f"An error occurred {rc}")

        # Save the standard error (results)
        with (self.output_dir / f"{self.name}.output").open('w') as fp:
            for line in lines:
                fp.write(line)

        return lines

    def write_config(self, experiment_name: str) -> Path:
        output_path = self.output_dir / f"{experiment_name}.config"
        assert not output_path.exists(), f"{output_path} exists!"

        lines = [f'outputByRowsFile {str(self.output_dir / (self.name + "_assignments.txt"))}']

        for k, v in self.__dict__.items():
            if k in _NON_PARAM_NAMES:
                continue
            elif k in _SPECIAL_CASES:
                k = _SPECIAL_CASES.get(k)
            else:
                k = k.replace('_', ' ').title().replace(' ', '')
                k = k[0].lower() + k[1:]

            lines.append(f"{k} {v}")

        with output_path.open("w") as fp:
            fp.write("\n".join(lines) + "\n")

        return output_path


def sbf_jar_path() -> Path:
    if (jar_path := os.environ.get("SBF_JAR_PATH")) is not None:
        return Path(jar_path)
    return Path(__file__).parent.parent / "sbf-1.0.0.jar"


_NON_PARAM_NAMES = {'name', 'output_dir'}
_SPECIAL_CASES = {'k': 'K'}


