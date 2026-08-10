import random
import math
import pandas as pd
from src.utils.functions import find_bounds

def gai_discretise(u: int,v: int,x: int) -> int: ### change u to lower and v to upper, use more verbose code?
    if u == v:
        return u
    intervals = [u,v]

    u_prob = (v-x)/(v-u)
    v_prob = (x-u)/(v-u)
    probs = [u_prob,v_prob]

    result = random.choices(intervals,weights=probs,k=1)
    return result[0]


def gai_perturb(x_i: int, epsilon: float, interval_list: list) -> int:
    len_list = len(interval_list)
    x_position = interval_list.index(x_i)

    p_prob = (math.exp(epsilon))/(len_list-1+math.exp(epsilon))
    q_prob = (1)/(len_list-1+math.exp(epsilon))
    probs = [q_prob] * len_list
    probs[x_position] = p_prob

    result = random.choices(interval_list,weights=probs,k=1)
    return result[0]


def gai_mechanism(df: pd.DataFrame, interval_list: list, epsilon: float) -> pd.DataFrame:
    df[['lower', 'upper']] = df['power'].apply(lambda x: pd.Series(find_bounds(x, interval_list)))
    df['discretised'] = df.apply(
        lambda row: gai_discretise(row['lower'], row['upper'], row['power']),
        axis=1
    )
    df['perturbed'] = df.apply(
        lambda row: gai_perturb(row['discretised'], epsilon, interval_list),
        axis=1
    )
    return df['perturbed']
