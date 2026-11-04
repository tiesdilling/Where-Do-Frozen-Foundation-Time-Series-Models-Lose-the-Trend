from pathlib import Path

import numpy as np
import pandas as pd

from data_generator import make_walk


def build_dataset(snr_values, seeds, origin, max_horizon, sigma, data_dir):
    data_dir = Path(data_dir)
    (data_dir / "walks").mkdir(parents=True, exist_ok=True)


    rows = []


    for snr in snr_values:
        for seed in seeds:

            walk_id = f"snr{snr:.4g}_seed{seed}"

            walk = make_walk(snr, sigma, origin + max_horizon, seed)

            np.save(data_dir / "walks" / (walk_id + ".npy"), walk)

            rows.append({
                "walk_id": walk_id,
                "snr": snr,
                "seed": seed,
                "sigma": sigma,
                "mu": snr * sigma,   
                "origin": origin,
                "max_horizon": max_horizon,
            })

    overview = pd.DataFrame(rows)
    overview.to_csv(data_dir / "overview.csv", index=False)

    return overview



def load_overview(data_dir):
    overview = pd.read_csv(Path(data_dir) / "overview.csv")

    return overview


def load_window(walk_id, context_length, origin, horizon, data_dir):
    walk = np.load(Path(data_dir) / "walks" / (walk_id + ".npy"))

    context = walk[origin - context_length:origin]
    future  = walk[origin:origin + horizon]

    offset = context[-1]

    return context - offset, future - offset