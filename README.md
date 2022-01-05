# HUROT: An Homogeneous formulation of Unbalanced Regularized Optimal Transport.

This repository provides code related to this preprint (link to appear soon). 

This is an alpha version and is likely to be modified in the future. 
Any suggestion or feedback is welcome!

We refer to [the tutorial](https://github.com/tlacombe/homogeneousUROT/blob/master/tutorial_HUROT.ipynb) for a
presentation of the mathematical concepts behind this implementation.

## Dependencies

- `numpy`
- `PythonOptimalTransport` (will probably be removed or changed to `scipy` in the future).

## Quick start

```python
import numpy as np
from utils import sk_div, hurot

np.random.seed(42)
# Define the measures as weights + locations.
n, m = 5, 7
a = np.random.rand(n)
b = np.random.rand(m)
x = np.random.randn(n, 2)
y = np.random.randn(m, 2) + np.array([.5, .5])

# Set the parameter for the OT cost and the Sinkhorn divergence:
mode_divergence = "TV"  # To use the total variation as the marginal divergence.
mode_homogeneity = "harmonic"  # To use the harmonic 
eps = 1  # the entropic regularization parameter

# The following returns the Sinkhorn divergence (positive).
value = sk_div(x, y, a, b,
               mode_divergence=mode_divergence,
               mode_homogeneity=mode_homogeneity,
               corrected_marginals=False,
               eps=eps,
               verbose=0, init="unif",
               nb_step=1000, crit=0., stab=True)

# The following returns :
# - P: The optimal transport plan between alpha and beta
# - f,g: the couple of optimal dual potentials
# - ot_value: the value of OT (less relevant than the Sinkhorn divergence though).
P, f, g, ot_value = hurot(x, y, a, b,
                          mode_divergence=mode_divergence,
                          mode_homogeneity=mode_homogeneity,
                          corrected_marginals=False,
                          eps=eps,
                          verbose=0, init="unif",
                          nb_step=1000, crit=0., stab=True)
```
