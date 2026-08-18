#%%
import pandas as pd

#%%
df = pd.read_parquet('../data/eco/preprocess/sm_01.parquet')
df
df = pd.read_parquet('../results/perturbed/sm_01.parquet')
df
