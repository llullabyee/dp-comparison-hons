# %%
import sys

import pandas as pd

sys.path.insert(0, "/Users/lullaby/Projects/University/DP-ML-Hons")

# %%

df = pd.read_parquet(
    "/Volumes/lullaby-nas/IDEAL/preprocess/eps=1/sens=1500/ldp_energy/2018-04-01.parquet"
)
df

# %%
df = pd.read_parquet("../data/eco/preprocess/appliances/sm_01.parquet")
df

# %%
df = pd.read_parquet("../results/perturbed/sm_01.parquet")
df
