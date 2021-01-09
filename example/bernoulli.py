import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../')
from layers import *
from optimizers import *
from deqnn import *
from hyperparameter_search import *

np.random.seed(1)


class DeqNNBernoulli(DeqNN):
    """Bernoulli differential equation

    x'(t) + P(t)*x(t) = Q(t)*x^n

    https://en.wikipedia.org/wiki/Bernoulli_differential_equation
    """
    def __init__(self, layers, t0, x0,
            n=2, P=lambda t: -2/(t+1e-10), Q=lambda t: -t**2):
        """Set parameters n, P and Q.

        Default values for n, P and Q correspond to Riccati's equation
        """
        super().__init__(layers, t0, x0)
        self.n = n
        self.P = P
        self.Q = Q


    def G(self, x, dx, t):
        return dx + self.P(t) * x - self.Q(t) * x**self.n


    def dGdx(self, x, dx, t):
        deriv = np.zeros((x.shape[0],1,1))
        deriv[:,0,0] = self.P(t[:,0]) \
            - self.n * self.Q(t[:,0]) * x[:,0]**(self.n-1)
        return deriv


    def dGddx(self, x, dx, t):
        deriv = np.zeros((x.shape[0],1,1))
        deriv[:,0,0] = 1
        return deriv


if __name__ == "__main__":
    tmin=0.1
    tmax=10
    t = np.arange(tmin,tmax,0.1)

    # Neural network solution
    nn = DeqNNBernoulli(
        layers=[
            TanhLayer(1, 10),
            LinearLayer(10, 1)
        ],
        t0=1,
        x0=np.array([1])
    )

    #search_Ns_alpha(nn, tmin, tmax, tol=1e-6)
    #exit(0)

    loss = nn.train(
        Nepochs=20000, Ns=50, tmin=tmin, tmax=tmax,
        optimizer=OptimizerAdam(alpha=0.005),
        tol=1e-6
    )

    x_nn,dx_nn,__ = nn.predict(t.reshape(t.shape[0],1))
    G = nn.G(x_nn, dx_nn, t.reshape(t.shape[0],1))

    # Analytic solution
    c = (5*nn.t0**2 - nn.t0**5*nn.x0[0])/(5*nn.x0[0])
    x_ana = 5*t**2/(5*c + t**5)
    dx_ana = (-15*t**6+50*t*c)/(5*c + t**5)**2

    # Plot
    fig = plt.figure(figsize=(6,4))
    fig.suptitle("Bernoulli differential equation")
    plt.subplots_adjust(left=0.12, right=0.95, wspace=0.4, hspace=0.5)
    c1 = "#1F77B4"
    c2 = "#E25140"

    ax = plt.subplot(2, 2, 1)
    ax.annotate('(a)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("$x(t)$")
    plt.plot(t, x_ana, color=c1, label="Analytic")
    plt.plot(t, x_nn[:,0], color=c2, linestyle="dotted", label="DeqNN")
    plt.legend(loc='upper right', frameon=False)

    ax = plt.subplot(2, 2, 2)
    ax.annotate('(b)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("Epochs")
    plt.ylabel("Loss $L$")
    plt.semilogy(loss[:,0], loss[:,1], color=c1)

    ax = plt.subplot(2, 2, 3)
    ax.annotate('(c)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("$|x -\hat x|$")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(t, np.abs(x_ana-x_nn[:,0]), color=c1)

    ax = plt.subplot(2, 2, 4)
    ax.annotate('(d)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("DEQ $|G|$")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(t, np.abs(G), color=c1)

    plt.savefig("bernoulli.png", dpi=200)
