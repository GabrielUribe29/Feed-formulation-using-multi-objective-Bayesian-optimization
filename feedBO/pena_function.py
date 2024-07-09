import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")


def Pena_func(X):
    X = X.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    W_Costs = torch.tensor(
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
        device=tkwargs["device"],
    )
    W_Energy = torch.tensor(
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
        device=tkwargs["device"],
    )
    W_Lysine = torch.tensor(
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
        device=tkwargs["device"],
    )
    Costs = W_Costs.to(X.dtype)
    Energy = W_Energy.to(X.dtype)
    Lysine = W_Lysine.to(X.dtype)

    f1 = -torch.matmul(X, Costs.unsqueeze(1)).squeeze(1)
    f2 = torch.matmul(X, Lysine.unsqueeze(1)).squeeze(1)
    f3 = torch.matmul(X, Energy.unsqueeze(1)).squeeze(1)

    problem = torch.stack([f1, f2, f3], dim=-1)

    return problem
