# Feed Formulation using multi-objective Bayesian optimization

This project focuses on feed formulation using Bayesian optimization techniques, specifically utilizing q-Expected Hypervolume Improvement (qEHVI) and q-Noisy Expected Hypervolume Improvement (qNEHVI) acquisition functions.

## Installation

* Clone the repo with
```
git clone https://github.com/GabrielUribe29/MOBO-feed.git

```
## Usage

* Import each function
```
import numpy as np
from feedBO.Bay_Opt import BO_feed
```
* Generate random seeds
```
seeds = np.random.randint(1, 300000, 1)
Seeds = seeds.tolist()
```
* Get results for one experiment with
```
BO_feed(Seeds=Seeds, Init_samples=50, num_exp=1)
```
or run 
```
main.py
```

If you find our work helpful, please consider citing our paper using:

```
@article{uribe2024feed,
  title={Feed formulation using multi-objective Bayesian optimization},
  author={Uribe-Guerra, Gabriel D and M{\'u}nera-Ram{\'\i}rez, Danny A and Arias-Londo{\~n}o, Juli{\'a}n D},
  journal={Computers and Electronics in Agriculture},
  volume={224},
  pages={109173},
  year={2024},
  publisher={Elsevier}
}
```
