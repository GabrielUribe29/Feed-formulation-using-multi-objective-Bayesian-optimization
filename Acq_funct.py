import torch
import numpy as np

from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list

from constraints import Pena_constant_constraints
from pena_function import Pena_func

tkwargs = {"dtype": torch.float64,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

ref_point=torch.tensor([-170,0,0], dtype=torch.float64)

bound = np.array(
            [
                0.4,
                0.4,
                0.4,
                0.05,
                0.22,
                1,
                0.04,
                0.08,
                1,
                0.0065,
                0.06,
                0.04,
                0.05,
                0.1,
                0.15,
                0.2,
                1,
            ]
        )
lu, ub = torch.zeros(17), torch.as_tensor(bound)
bounds = torch.stack((lu, ub), dim=0)

inc, eqc=Pena_constant_constraints()


def optimize_qNehvi_and_get_observation(model, train_x, sampler):
    """Optimizes the qNEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, bounds)).mean

    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        X_baseline=normalize(train_x, bounds=bounds),
        sampler=sampler,

    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=256,
        inequality_constraints=inc,
        equality_constraints=eqc,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    # observe new values
    new_x = candidates
    new_obj = Pena_func(new_x)
    return new_x, new_obj


def optimize_qehvi_and_get_observation(model, train_x, sampler):
    """Optimizes the qNEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, bounds)).mean
    partitioning = FastNondominatedPartitioning(ref_point=ref_point,Y=pred)

    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
        sampler=sampler,

    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=256,
        inequality_constraints=inc,
        equality_constraints=eqc,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    # observe new values
    new_x = candidates
    new_obj = Pena_func(new_x)
    return new_x, new_obj