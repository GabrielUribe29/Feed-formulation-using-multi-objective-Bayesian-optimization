import torch
import numpy as np

from botorch.utils.transforms import normalize
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.optim.optimize import optimize_acqf

from pena_function import Pena_func

tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

ref_point = torch.tensor([-170, 0, 0], dtype=torch.float64)

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


def Pena_constant_constraints():
    supbd = 10
    Ene = torch.as_tensor(
        [
            14.7446,
            15.4048,
            14.9983,
            5.7664,
            10.9456,
            15.3092,
            14.4188,
            12.5758,
            0,
            16.4095,
            9.9224,
            17.37,
            7.9859,
            14.363,
            14.4828,
            15.2728,
            0,
        ],
        dtype=torch.float64,
    )
    Ly = torch.as_tensor(
        [
            0.4,
            0.33,
            0.22,
            0.73,
            0.09,
            2.88,
            4.75,
            0.62,
            0,
            78,
            1.06,
            0,
            0.59,
            1.46,
            1.55,
            0.34,
            0,
        ],
        dtype=torch.float64,
    )
    CF = torch.as_tensor(
        [4.5, 2.8, 2.5, 24.7, 6.1, 5.6, 1, 8, 0, 0, 22.5, 0, 17.8, 14.5, 5.7, 2.3, 0]
    ).to(torch.float64)
    Ca = torch.as_tensor(
        [
            0.06,
            0.04,
            0.02,
            1.75,
            0.24,
            0.29,
            4.50,
            0.16,
            38.3,
            0.04,
            0.35,
            0,
            0.98,
            0.23,
            0.10,
            0.05,
            32,
        ],
        dtype=torch.float64,
    )
    AP = torch.as_tensor(
        [
            0.13,
            0.18,
            0.05,
            0.26,
            0.03,
            0.19,
            2.45,
            0.22,
            0,
            0,
            0.17,
            0,
            0.04,
            0.13,
            0.15,
            0.15,
            0,
        ],
        dtype=torch.float64,
    )
    DM = torch.as_tensor(
        [
            90.2,
            89.4,
            86.3,
            91.2,
            88.8,
            88,
            92,
            88.6,
            98,
            98.5,
            89.3,
            0,
            89.7,
            90.8,
            86.7,
            89.4,
            99.4,
        ],
        dtype=torch.float64,
    )
    CP = torch.as_tensor(
        [
            11.3,
            11.6,
            7.7,
            16.7,
            2.5,
            44,
            62.4,
            19,
            0,
            95,
            30.5,
            0,
            10.1,
            30.7,
            21.5,
            8.9,
            0,
        ],
        dtype=torch.float64,
    )
    MC = torch.as_tensor(
        [
            0.43,
            0.46,
            0.33,
            0.45,
            0.06,
            1.28,
            2.36,
            0.83,
            0,
            0,
            1.25,
            0,
            0.22,
            0.66,
            0.56,
            0.37,
            0,
        ],
        dtype=torch.float64,
    )
    T = torch.as_tensor(
        [
            0.37,
            0.34,
            0.27,
            0.7,
            0.07,
            1.75,
            2.65,
            0.74,
            0,
            0,
            1.06,
            0,
            0.47,
            0.99,
            0.82,
            0.3,
            0,
        ],
        dtype=torch.float64,
    )
    Tp = torch.as_tensor(
        [
            0.13,
            0.13,
            0.06,
            0.31,
            0.02,
            0.59,
            0.65,
            0.13,
            0,
            0,
            0.43,
            0,
            0.1,
            0.25,
            0.19,
            0.1,
            0,
        ],
        dtype=torch.float64,
    )
    ones = torch.ones(17, dtype=torch.float64)
    Co = torch.as_tensor(
        [
            141.24,
            144.24,
            147.25,
            132.22,
            132.22,
            195.32,
            342.57,
            120.2,
            30.05,
            1803.03,
            111.18,
            336.56,
            152.65,
            150.25,
            159.27,
            136.73,
            300.5,
        ],
        dtype=torch.float64,
    )
    ineqconstr = [
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            -CF,
            -6,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            Ca,
            0.6,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            -Ca,
            -supbd,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            AP,
            0.15,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            -AP,
            -supbd,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            DM,
            87,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            -DM,
            -95,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            CP,
            18,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            -CP,
            -21,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            MC,
            0.475,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            Tp,
            0.171,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            T,
            0.627,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            -MC,
            -supbd,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            -Tp,
            -supbd,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            -T,
            -supbd,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            Ene,
            14.001,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            -Ene,
            -20,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            Ly,
            0.9501,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            -Ly,
            -2,
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            Co,
            143,
        ),
    ]

    eqconstr = [
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            ones,
            1,
        ),
    ]

    return ineqconstr, eqconstr


inc, eqc = Pena_constant_constraints()


def optimize_qNehvi_and_get_observation(model, train_x, sampler):
    """Optimizes the qNEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, bounds)).mean

    acq_func = qLogNoisyExpectedHypervolumeImprovement(
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
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pred)

    acq_func = qLogExpectedHypervolumeImprovement(
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
