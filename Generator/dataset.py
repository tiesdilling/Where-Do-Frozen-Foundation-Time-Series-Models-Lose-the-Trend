import sys
from pathlib import Path
import pandas as pd
import numpy as np

gen_dir = "/Users/tiesdilling/Desktop/Python_Study/studying_python/Thesis/Environment_Setup"
sys.path.append(gen_dir)

from data_generator import drifted_random_walk_gaussian
from sweep import full_sweep



def create_dataset(SNR_values, horizon_values, seed_values, x0, sigma, context=512, sign_values=(1.0,)):

    rows = full_sweep(SNR_values, horizon_values, seed_values, x0, sigma,
                      context=context, sign_values=sign_values)
    df = pd.DataFrame(rows)

    df["series_id"] = (
        np.where(df["sign"] < 0, "-", "+")
        + df["SNR"].map(lambda v: f"{v:g}") + "_"
        + df["horizon"].astype(str) + "_"
        + df["seed"].astype(str)
    )
    
    if not df["series_id"].is_unique:
        dup = df.loc[df["series_id"].duplicated(), "series_id"].unique()[:5]
        raise ValueError(f"series_id collision, e.g. {list(dup)}")

    series = {r.series_id: {"context": r.context_x, "target": r.target_x}
              for r in df.itertuples()}
    meta = df.drop(columns=["context_x", "target_x"])
    return meta, series