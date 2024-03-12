from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import normalize

import torch
import numpy as np

tkwargs = {"dtype": torch.float64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

from pena_function import Pena_func
from GeneratorSampling import gensamples

NOISE_SE = torch.tensor([0.1, 0.1, 0.1], **tkwargs)

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


def generate_initial_data(n):
    # generate training data
    train_x = gensamples(n)
    train_obj_true = Pena_func(train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * NOISE_SE

    return train_x, train_obj


def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    train_x = normalize(train_x, bounds=bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i:i + 1]
        train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
        models.append(
            FixedNoiseGP(train_x, train_y, train_yvar, outcome_transform=Standardize(m=1))
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model
