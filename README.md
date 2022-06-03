# What is this?

- Right now: a python interface to twitter's `sbf` java package
- Eventually: a python port of twitter's `sbf` algorithm

# But, why?

I want to understand it, and I don't want to run a JVM.

# Usage

```python
import pandas as pd
from pathlib import Path
from pysbf.projection import build_R_projection
from pysbf.metis import write_metis
from pysbf.reference import Experiment, sbf_jar_path, load_assignments


# Load your user items
df = pd.read_csv("user_items.csv")

# Build the R projection from bipartite similarities
G, L_i_to_v, L_v_to_i, R_i_to_v, R_v_to_i = build_R_projection(
    df[['users', 'items']], 
    df['items'],
    top_k_R_nodes=100_000,
    L_node_degree_upper_bound=5_000,
    prune_similarity_below=0.01,
)

# Write this projection to a format sbf understands
output_path = Path("output")
metis_path = output_path / "unweighted_metis.txt"
write_metis(G, metis_path)

# Create an experiment
experiment = Experiment(
    name='my_experiment',
    metis_file=metis_path,
    output_dir=output_path
)

# Run it
experiment.run(sbf_jar_path())

# Retrieve the Cluster -> {R_Node, ...} map
assignments = load_assignments(experiment.assignments_path, R_i_to_v)
```

then,

```bash
head output/my_experiment_assignments.txt
head output/my_experiment_output.txt
```

# Should I use this?

Almost certainly not, no.