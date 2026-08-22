#%%
import pandas as pd
import sys
sys.path.insert(0, '/Users/lullaby/Projects/University/DP-ML-Hons')

from src.data.load_data import load_data
from src.utils.config import PREPROCESS_DIR

#%%
gen = load_data(PREPROCESS_DIR / 'mains')
next(gen)
#%%

df = pd.read_parquet('../data/eco/preprocess/mains/sm_01.parquet')
df

#%%
df = pd.read_parquet('../data/eco/preprocess/appliances/sm_01.parquet')
df

#%%
df = pd.read_parquet('../results/perturbed/sm_01.parquet')
df
