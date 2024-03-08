import numpy as np
import cvxpy as cp
import os
import torch

# Dataset: Macronutrients and micronutrients and others.

# Costs of raw materials
Costs = np.array([141.24,144.24,147.25,132.22,132.22,195.32,342.57,120.2,30.05,1803.03,111.18,336.56,152.65,150.25,159.27,136.73,300.5])

# CF
CF = np.array([4.5,2.8,2.5,24.7,6.1,5.6,1,8,0,0,22.5,0,17.8,14.5,5.7,2.3,0])

# Ca
Ca = np.array([0.06,0.04, 0.02, 1.75, 0.24,0.29,4.50,0.16,38.3,0.04, 0.35, 0, 0.98, 0.23, 0.10, 0.05, 32])

# AP
AP = np.array([0.13, 0.18, 0.05, 0.26, 0.03, 0.19, 2.45, 0.22, 0,0,0.17, 0, 0.04, 0.13, 0.15, 0.15,0])

# DM
DM = np.array([90.2,89.4, 86.3,91.2, 88.8, 88, 92, 88.6, 98, 98.5, 89.3, 0, 89.7, 90.8, 86.7, 89.4, 99.4])

# CP
CP = np.array([11.3, 11.6, 7.7, 16.7, 2.5, 44, 62.4, 19, 0, 95, 30.5, 0, 10.1,30.7, 21.5, 8.9, 0])

# Energy
Energy = np.array([14.7446, 15.4048, 14.9983,5.7664, 10.9456,15.3092,14.4188,12.5758,0,16.4095,9.9224,17.37,7.9859,  14.363 ,  14.4828,  15.2728, 0])

# L
L = np.array([0.4,0.33,0.22, 0.73, 0.09, 2.88, 4.75, 0.62, 0, 78, 1.06, 0, 0.59, 1.46, 1.55, 0.34, 0])

# MC
MC = np.array([0.43,0.46,0.33,0.45,0.06,1.28,2.36,0.83,0,0,1.25,0,0.22,0.66,0.56,0.37,0])

# T
T = np.array([0.37,0.34,0.27,0.7,0.07,1.75,2.65,0.74,0,0,1.06,0,0.47,0.99,0.82,0.3,0])

# Tp
Tp = np.array([0.13,0.13,0.06,0.31,0.02,0.59,0.65,0.13,0,0,0.43,0,0.1,0.25,0.19,0.1,0])

# Bounds of raw materials and macronutrients
supbd = 10
bound = np.array([0.4, 0.4, 0.4, 0.05, 0.22, 1,0.04, 0.08,1,0.0065, 0.06, 0.04, 0.05, 0.1, 0.15, 0.2, 1])
Q = np.concatenate([CF, Ca, AP, DM, CP, MC, T, Tp, Energy, L]).reshape(10,17)
lower = np.array([0, 0.6, 0.15, 87, 18, 0.475, 0.627, 0.171, 14.1, 0.9501])
upper = np.array([6, supbd, supbd, 95, 21, supbd,supbd, supbd, 20,2])

# Number of variables and standard deviation, and variances
n = 17
ones = np.ones(n)

stdl = np.array([0.03962323, 0.0484768 , 0.01870829, 0.08031189, 0.02408319,0.13823892, 0.54625086, 0.09300538, 0., 0.,0.06356099, 0. , 0.08848729, 0.08757854, 0.06511528,0.03 , 0. ])
stde = np.array([0.3197, 0.4066, 0.2644, 1.0968, 0.5168, 0.3607, 1.4353, 0.5676,0., 0., 0.7056, 0., 1.4691, 0.5844, 0.3746, 0.2462,0.])

vare = np.diag(stde**2)
varl = np.diag(stdl**2)

def gensamples(nn):
    # lista=np.random.seed(1234)
    lista = []
    for i in range(nn):
        w = cp.Variable(n, pos=True)
        cons = cp.Constant(np.random.normal(155, 2))

        objective_fct = cp.Maximize(1)

        constraints = [ones @ w == 1, Q @ w >= lower, Q @ w <= upper,
                       w[0] <= bound[0],
                       w[1] <= bound[1],
                       w[2] <= bound[2],
                       w[3] <= bound[3],
                       w[4] <= bound[4],
                       w[5] <= bound[5],
                       w[6] <= bound[6],
                       w[7] <= bound[7],
                       w[8] <= bound[8],
                       w[9] <= bound[9],
                       w[10] <= bound[10],
                       w[11] <= bound[11],
                       w[12] <= bound[12],
                       w[13] <= bound[13],
                       w[14] <= bound[14],
                       w[15] <= bound[15],
                       w[16] <= bound[16],
                       Costs @ w.T - cons <= 0]
        prob = cp.Problem(objective_fct, constraints)
        assert prob.is_dqcp()
        prob.solve(qcp=True, solver='ECOS')

        lista.append(w.value)
    return torch.tensor(np.array(lista), dtype=torch.float64)
