import os
import torch
import numpy as np

import time
import warnings

from evaluation_func import (generate_initial_data,
                             initialize_model
                             )
from Acq_funct import (optimize_qNehvi_and_get_observation,
                       optimize_qehvi_and_get_observation
                       )

from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch import fit_gpytorch_model, fit_gpytorch_mll
from botorch.sampling import SobolQMCNormalSampler

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

ref_point = torch.tensor([-170, 0, 0], dtype=torch.float64, device='cuda:0')

bound = np.array([0.4, 0.4, 0.4, 0.05, 0.22, 1, 0.04, 0.08, 1, 0.0065, 0.06, 0.04, 0.05, 0.1, 0.15, 0.2, 1, ])
lu, ub = torch.zeros(17), torch.as_tensor(bound)
bounds = torch.stack((lu, ub), dim=0)

def BO_feed(Seeds, Init_samples,num_exp):
    # Seeds: Set of seeds use in each experiment
    # num_rp: Number of repetitions of Bayeain optimiation
    Data=[]

    for i in range(num_exp):
        N_BATCH = 50 if not SMOKE_TEST else 10
        Mc_samples = 256 if not SMOKE_TEST else 16
        seed_index = i % len(Seeds)
        S = Seeds[seed_index]
        np.random.seed(S)
        torch.manual_seed(S)

        verbose = True

        hvs_ehvi = [] #Hypervolume in each iteration
        card = [] # Number of samples in every approximation Pareto front in each iteration

        # call helper functions to generate initial training data and initialize model
        train_x_ehvi, train_obj_ehvi = generate_initial_data(n=Init_samples)
        mll_ehvi, model_ehvi = initialize_model(train_x_ehvi, train_obj_ehvi)

        # compute hypervolume
        bd = FastNondominatedPartitioning(ref_point=ref_point, Y=train_obj_ehvi)
        volume = bd.compute_hypervolume().item()
        card.append(bd.pareto_Y.shape[0])

        hvs_ehvi.append(volume)

        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):

            t0 = time.monotonic()

            # fit the models
            fit_gpytorch_mll(mll_ehvi)

            # define the qEI and qNEI acquisition modules using a QMC sampler

            ehvi_sampler = SobolQMCNormalSampler(Mc_samples)

            # optimize acquisition functions and get new observations

            new_x_ehvi, new_obj_ehvi = optimize_qehvi_and_get_observation(
                model_ehvi, train_x_ehvi, ehvi_sampler
            )

            train_x_ehvi = torch.cat([train_x_ehvi, new_x_ehvi])
            train_obj_ehvi = torch.cat([train_obj_ehvi, new_obj_ehvi])

            # compute hypervolume
            bd = FastNondominatedPartitioning(ref_point=ref_point, Y=train_obj_ehvi)
            volume = bd.compute_hypervolume().item()
            hvs_ehvi.append(volume)
            card.append(bd.pareto_Y.shape[0])

            mll_ehvi, model_ehvi = initialize_model(train_x_ehvi, train_obj_ehvi)

            t1 = time.monotonic()

        Data.append([Init_samples, S, volume, bd.pareto_Y.shape[0], bd.pareto_Y, train_x_ehvi, train_obj_ehvi])


    return Data


