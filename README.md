# Feed Formulation using multi-objective Bayesian optimization

This project focuses on feed formulation using Bayesian optimization techniques, specifically utilizing q-LogExpected Hypervolume Improvement (qEHVI) and q-LogNoisy Expected Hypervolume Improvement (qNEHVI) acquisition functions.

This branch applies the Dimensionality-scaled lengthscale prior to the GP kernel using a Log-Normal distribution to increase the lengthscale at a rate proportional to the dimensionality of the problem. See: Hvarfner, C., Hellsten, E. O., & Nardi, L. (2024). Vanilla Bayesian Optimization Performs Great in High Dimension. arXiv preprint arXiv:2402.02229

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
