# This is a sample Python script.

import torch
import numpy as np
from botorch.sampling import SobolQMCNormalSampler
from evaluation_func import generate_initial_data, initialize_model
from Bay_Opt import BO_feed

tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

if __name__ == "__main__":

# seeds = np.random.randint(1, 300000, 1)
seeds = np.array([149763])
Seeds = seeds.tolist()
print(Seeds)
x, y = generate_initial_data(50)
print(y)
mll, model = initialize_model(x, y)
sampler = SobolQMCNormalSampler(512)
# optimizer=optimize_qNehvi_and_get_observation(model=model, sampler=sampler, train_x=x)
Exp1 = BO_feed(Seeds=Seeds, Init_samples=50, num_exp=1)
print(Exp1)
