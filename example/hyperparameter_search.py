import numpy as np
import sys
sys.path.insert(1, '../')
from layers import *
from optimizers import *
from deqnn import *

def search_Ns_alpha(nn, tmin, tmax, tol,
        Ns_values=np.array([10,50,100,150,200,300,500]),
        alpha_values = [0.001,0.005,0.01,0.05,0.1],
        Nepochs_max=20000):
    """Search hyperparameter space of number of samples per epoch Ns
        and learning rate alpha to find minimal training epochs Nepochs
    """
    Ns_best = 0
    alpha_best = 0
    Nepochs_min = Nepochs_max
    for Ns in Ns_values:
        for alpha in alpha_values:
            print("Ns={:>4}, alpha={:.3f}".format(Ns, alpha), end="")

            # Reset random seed and reinitialisize weights to have
            # equal start points
            np.random.seed(1)
            nn.init_weights()
            loss = nn.train(
                Nepochs=Nepochs_max, Ns=Ns, tmin=tmin, tmax=tmax,
                optimizer=OptimizerAdam(alpha=alpha),
                tol=tol,
                show_progress=None
            )
            if loss[-1][0] < Nepochs_min:
                Nepochs_min = loss[-1][0]
                Ns_best = Ns
                alpha_best = alpha
            print(", epochs={:>5.0f}, loss={:.5f}".format(
                loss[-1][0], loss[-1][1]), end="")
            print(" - BEST: Ns={:>4}, alpha={:.3f}, epochs={:>5.0f}".format(
                Ns_best, alpha_best, Nepochs_min))

    print("Optimal parameters: Ns={}, alpha={} -> {:.0f} epochs".format(
        Ns_best, alpha_best, Nepochs_min))
    return Ns_best,alpha_best
