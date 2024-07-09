# This is a sample Python script.

import torch
import numpy as np
from evaluation_func import generate_initial_data
from Bay_Opt import BO_feed

tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def Optimize():
    seeds = np.array([149763])
    Seeds = seeds.tolist()
    _, y = generate_initial_data(50)
    print("Initial target data:")
    print(y)
    Exp1 = BO_feed(Seeds=Seeds, Init_samples=50, num_exp=1)
    print("Candidate solutions found by BO:")
    print(Exp1)
    return


if __name__ == "__main__":
    print("Starting optimization...")
    Optimize()
