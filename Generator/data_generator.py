import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum


def drifted_random_walk_gaussian(x0, T, SNR, sigma, seed, sign=1.0):
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0; SNR = |mu|/sigma is undefined at sigma = 0.")
    if SNR < 0:
        raise ValueError("SNR is |mu|/sigma and must be >= 0; use sign=-1 for downward drift.")

    rng = np.random.default_rng(int(seed))
    mu  = sign * SNR * sigma
    eps = rng.normal(0.0, sigma, T)
    x   = np.r_[x0, x0 + np.cumsum(mu + eps)]

    return {"time": np.arange(T + 1, dtype=float), 
            "x": x, 
            "noise": eps,
            "mu": mu, 
            "sigma": sigma, 
            "SNR": SNR, 
            "T": int(T)
            }