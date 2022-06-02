# Data used in unit tests

## Clustermatch data

The `clustermatch-example-*.pkl` files were generated using the original clustermatch
code (https://github.com/sinc-lab/clustermatch - Commit 8b66b3d7) plus the patch below:

```patch
$ git diff
diff --git a/clustermatch/cluster.py b/clustermatch/cluster.py
index 9f7d06c..07e8192 100644
--- a/clustermatch/cluster.py
+++ b/clustermatch/cluster.py
@@ -160,7 +160,7 @@ def _get_range_n_clusters(n_common_features, **kwargs):
     if internal_n_clusters is None:
         estimated_k = int(np.floor(np.sqrt(n_common_features)))
         estimated_k = np.min((estimated_k, 10))
-        range_n_clusters = range(2, np.max((estimated_k, 3)))
+        range_n_clusters = range(2, np.max((estimated_k, 3))+1)
     elif isinstance(internal_n_clusters, (tuple, list, range)):
         range_n_clusters = internal_n_clusters
     elif isinstance(internal_n_clusters, int):
@@ -211,7 +211,7 @@ def row_col_from_condensed_index(d,i):
 
 
 def _compute_ari(part1, part2):
-    if np.isnan(part1).any() or len(part1) == 0:
+    if np.isnan(part1).any() or np.isnan(part2).any() or len(part1) == 0 or len(part2) == 0:
         return 0.0
 
     return ari(part1, part2)
```

Then I moved to the git root directory and executed the following commands in ipython:

### Random data without NaN
```python
from pathlib import Path

import numpy as np
import pandas as pd

from clustermatch.cluster import calculate_simmatrix

np.random.seed(0)
random_data = pd.DataFrame(np.random.rand(20, 100))

OUTPUT_DIR = Path("/home/miltondp/projects/ccc/ccc/tests/data/")

random_data.to_pickle(OUTPUT_DIR / "ccc-random_data-data.pkl")

int_n_clusters = range(2, 10+1)
cm_sim_matrix = calculate_simmatrix(random_data, internal_n_clusters=int_n_clusters, n_jobs=3)
cm_sim_matrix.to_pickle(OUTPUT_DIR / "ccc-random_data-coef.pkl")
```


THIS IS WITH THE ORIGINAL DATA WITH NANS
### Tomato dataset used in the original clustermatch implementation (contains NaN)
```python
from pathlib import Path

import pandas as pd

from clustermatch.cluster import calculate_simmatrix
from clustermatch.utils.data import merge_sources

data_files = ['experiments/tomato/data/real_sample.xlsx']
merged_sources, feature_names, sources_names = merge_sources(data_files)

OUTPUT_DIR = Path("/home/miltondp/projects/ccc/ccc/tests/data/")

merged_sources_final = merged_sources.apply(lambda x: pd.to_numeric(x, errors="coerce"), axis=1)
merged_sources_final = merged_sources_final.dropna(how="all")
merged_sources_final.to_pickle(OUTPUT_DIR / "ccc-example-data.pkl")

int_n_clusters = range(2, 5)
cm_sim_matrix = calculate_simmatrix(merged_sources_final, internal_n_clusters=int_n_clusters, n_jobs=3)
cm_sim_matrix.to_pickle(OUTPUT_DIR / "ccc-example-coef.pkl")
```
