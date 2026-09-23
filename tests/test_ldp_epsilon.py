# %%
import sys

sys.path.insert(0, "/Users/lullaby/Projects/University/dp-comparison-hons")


from src.mechanisms.ldp_energy import ldp_probability

# %%
for eps in [0.1, 0.5, 1, 2, 5, 10]:
    q = ldp_probability(eps)
    print(f"eps={eps}: q={q:.4f}, p-q={0.5 - q:.4f}")
