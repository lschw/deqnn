import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys
sys.path.insert(1, '../')
from layers import *
from optimizers import *
from deqnn import *
from hyperparameter_search import *

np.random.seed(1)


class DeqNNHarmonicOscillator(DeqNN):
    """Harmonic oscillator

    https://en.wikipedia.org/wiki/Harmonic_oscillator
    """
    def __init__(self, layers, t0, x0, w0, g0):
        """Set parameters

        Args:
            w0: Frequency
            g0: Damping
        """
        super().__init__(layers, t0, x0)
        self.w0 = w0
        self.g0 = g0


    def G(self, x, dx, t):
        x1 = x[:,0]
        x2 = x[:,1]
        dx1 = dx[:,0]
        dx2 = dx[:,1]
        return np.array([
            dx1 - x2,
            dx2 + self.w0**2 * x1 + self.g0*x2
        ]).T


    def dGdx(self, x, dx, t):
        deriv = np.zeros((x.shape[0],2,2))
        deriv[:,0,0] = 0          # dG_{s1} / dx_1 = dG_{s1} / dA^(L)_{s1}
        deriv[:,0,1] = -1         # dG_{s1} / dx_2 = dG_{s1} / dA^(L)_{s2}
        deriv[:,1,0] = self.w0**2 # dG_{s2} / dx_1 = dG_{s2} / dA^(L)_{s1}
        deriv[:,1,1] = self.g0    # dG_{s2} / dx_2 = dG_{s2} / dA^(L)_{s2}
        return deriv


    def dGddx(self, x, dx, t):
        deriv = np.zeros((x.shape[0],2,2))
        deriv[:,0,0] = 1 # dG_{s1} / d dx_1 = dG_{s1} / d dA^(L)_{s1}
        deriv[:,0,1] = 0 # dG_{s1} / d dx_2 = dG_{s1} / d dA^(L)_{s2}
        deriv[:,1,0] = 0 # dG_{s2} / d dx_1 = dG_{s2} / d dA^(L)_{s1}
        deriv[:,1,1] = 1 # dG_{s2} / d dx_2 = dG_{s2} / d dA^(L)_{s2}
        return deriv



if __name__ == "__main__":

    ###
    # Solution with sin activation function
    ###
    tmin=0
    tmax=30
    t = np.arange(tmin,tmax,0.1)

    # Neural network solution
    nn = DeqNNHarmonicOscillator(
        layers=[
            SinLayer(1, 100),
            LinearLayer(100, 2)
        ],
        t0=0,
        x0=np.array([0,1]),
        w0=1,
        g0=0.4
    )

    #search_Ns_alpha(nn, tmin, tmax, tol=1e-5, Nepochs_max=10000)
    #exit(0)

    loss = nn.train(
        Nepochs=20000, Ns=150, tmin=tmin, tmax=tmax,
        optimizer=OptimizerAdam(alpha=0.05),
        tol=1e-5
    )
    x_nn,dx_nn,__ = nn.predict(t.reshape(t.shape[0],1))
    G = nn.G(x_nn, dx_nn, t.reshape(t.shape[0],1))

    # Runge-Kutta solution
    x_rk = solve_ivp(
        lambda t,x: -nn.G(np.array([x]),np.array([np.zeros(x.shape)]), t)[0],
        t_span=(tmin, tmax), y0=nn.x0, t_eval=t
    )

    # Plot
    fig = plt.figure(figsize=(6,5))
    fig.suptitle("Harmonic oscillator (sin activation)")
    plt.subplots_adjust(left=0.12, right=0.95, wspace=0.4, hspace=0.7)
    c1 = "#1F77B4"
    c2 = "#E25140"

    ax = plt.subplot(3, 2, 1)
    ax.annotate('(a)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("$x(t)$")
    plt.plot(t, x_rk.y[0], color=c1, label="Runge-Kutta")
    plt.plot(t, x_nn[:,0], color=c2, linestyle="dotted", label="DeqNN")
    plt.legend(loc='upper right', frameon=False)

    ax = plt.subplot(3, 2, 2)
    ax.annotate('(b)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$x_1(t)$")
    plt.ylabel("$x_2(t)$")
    plt.plot(x_rk.y[0], x_rk.y[1], color=c1, label="Runge-Kutta")
    plt.plot(x_nn[:,0], x_nn[:,1], color=c2, linestyle="dotted", label="DeqNN")

    ax = plt.subplot(3, 2, 3)
    ax.annotate('(c)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("$|x-\hat x|$")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(t, np.abs(x_rk.y[0]-x_nn[:,0]), color=c1, label="$x_1$")
    plt.plot(t, np.abs(x_rk.y[1]-x_nn[:,1]), color=c2, label="$x_2$")
    plt.legend(loc='upper right', frameon=False)

    ax = plt.subplot(3, 2, 4)
    ax.annotate('(d)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("DEQ $|G|$")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(t, np.abs(G[:,0]), color=c1, label="$G_1$")
    plt.plot(t, np.abs(G[:,1]), color=c2, label="$G_2$")
    plt.legend(loc='upper right', frameon=False)

    ax = plt.subplot(3, 1, 3)
    ax.annotate('(e)', xycoords="axes fraction", xy=(-0.13,0.9))
    plt.xlabel("Epochs")
    plt.ylabel("Loss $L$")
    plt.semilogy(loss[:,0], loss[:,1], color=c1)

    plt.savefig("harmonic_oscillator.png", dpi=200)


    #####
    # Out-of-range solution
    #####
    tmin=-10
    tmax=40
    t = np.arange(tmin,tmax,0.1)
    x_nn,dx_nn,__ = nn.predict(t.reshape(t.shape[0],1))
    G = nn.G(x_nn, dx_nn, t.reshape(t.shape[0],1))
    x_ana = 1.02062*np.exp(-0.2*t)*np.sin(0.979796*t)

    fig = plt.figure(figsize=(6,4))
    fig.suptitle("Harmonic oscillator (sin activation, beyond training range)")
    plt.subplots_adjust(left=0.12, right=0.95, wspace=0.4, hspace=0.5)
    ax = plt.subplot(2, 1, 1)
    ax.annotate('(a)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("$x(t)$")
    plt.plot(t, x_ana, color=c1, label="Analytic")
    plt.plot(t, x_nn[:,0], color=c2, linestyle="dotted", label="DeqNN")
    plt.axvspan(0, 30, facecolor='k', alpha=0.1)
    plt.legend(loc='lower right', frameon=False)

    ax = plt.subplot(2, 1, 2)
    ax.annotate('(b)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("DEQ $|G|$")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(t, np.abs(G[:,0]), color=c1, label="$G_1$")
    plt.plot(t, np.abs(G[:,1]), color=c2, label="$G_2$")
    plt.axvspan(0, 30, facecolor='k', alpha=0.1)
    plt.legend(loc='upper center', frameon=False, ncol=2)

    plt.savefig("harmonic_oscillator_oor.png", dpi=200)


    ###
    # Solution with tanh activation function
    ###
    tmin=0
    tmax=30
    t = np.arange(tmin,tmax,0.1)

    # Neural network solution
    nn = DeqNNHarmonicOscillator(
        layers=[
            TanhLayer(1, 100),
            TanhLayer(100, 100),
            LinearLayer(100, 2)
        ],
        t0=0,
        x0=np.array([0,1]),
        w0=1,
        g0=0.4
    )

    #search_Ns_alpha(nn, tmin, tmax, tol=1e-4, Nepochs_max=20000)
    #exit(0)

    loss = nn.train(
        Nepochs=1660, Ns=200, tmin=tmin, tmax=tmax,
        optimizer=OptimizerAdam(alpha=0.01),
        tol=1e-4
    )

    x_nn,dx_nn,__ = nn.predict(t.reshape(t.shape[0],1))
    G = nn.G(x_nn, dx_nn, t.reshape(t.shape[0],1))

    # Runge-Kutta solution
    x_rk = solve_ivp(
        lambda t,x: -nn.G(np.array([x]),np.array([np.zeros(x.shape)]), t)[0],
        t_span=(tmin, tmax), y0=nn.x0, t_eval=t
    )

    # Plot
    fig = plt.figure(figsize=(6,5))
    fig.suptitle("Harmonic oscillator (tanh activation)")
    plt.subplots_adjust(left=0.12, right=0.95, wspace=0.4, hspace=0.6)
    c1 = "#1F77B4"
    c2 = "#E25140"

    ax = plt.subplot(3, 2, 1)
    ax.annotate('(a)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("$x_1(t)$")
    plt.plot(t, x_rk.y[0], color=c1, label="Runge-Kutta")
    plt.plot(t, x_nn[:,0], color=c2, linestyle="dotted", label="DeqNN")
    plt.legend(loc='upper right', frameon=False)

    ax = plt.subplot(3, 2, 2)
    ax.annotate('(b)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("$x_1(t)$")
    plt.plot(t, x_rk.y[1], color=c1, label="Runge-Kutta")
    plt.plot(t, x_nn[:,1], color=c2, linestyle="dotted", label="DeqNN")

    ax = plt.subplot(3, 2, 3)
    ax.annotate('(c)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("$|x-\hat x|$")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(t, np.abs(x_rk.y[0]-x_nn[:,0]), color=c1, label="$x_1$")
    plt.plot(t, np.abs(x_rk.y[1]-x_nn[:,1]), color=c2, label="$x_2$")
    plt.legend(loc='upper right', frameon=False)

    ax = plt.subplot(3, 2, 4)
    ax.annotate('(d)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("DEQ $|G|$")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(t, np.abs(G[:,0]), color=c1, label="$G_1$")
    plt.plot(t, np.abs(G[:,1]), color=c2, label="$G_2$")
    plt.legend(loc='upper right', frameon=False)

    ax = plt.subplot(3, 1, 3)
    ax.annotate('(e)', xycoords="axes fraction", xy=(-0.13,0.9))
    plt.xlabel("Epochs")
    plt.ylabel("Loss $L$")
    plt.semilogy(loss[:,0], loss[:,1], color=c1)

    plt.savefig("harmonic_oscillator_tanh.png", dpi=200)


    ###
    # Solution with pretrained weights
    ###
    nn.layers[0].W = np.load("W1.npy")
    nn.layers[0].b = np.load("b1.npy")
    nn.layers[1].W = np.load("W2.npy")
    nn.layers[1].b = np.load("b2.npy")
    nn.layers[2].W = np.load("W3.npy")
    nn.layers[2].b = np.load("b3.npy")

    x_nn,dx_nn,__ = nn.predict(t.reshape(t.shape[0],1))
    G = nn.G(x_nn, dx_nn, t.reshape(t.shape[0],1))

    # Plot
    fig = plt.figure(figsize=(6,4))
    fig.suptitle("Harmonic oscillator (tanh activation, pretrained)")
    plt.subplots_adjust(left=0.12, right=0.95, wspace=0.4, hspace=0.6)
    c1 = "#1F77B4"
    c2 = "#E25140"

    ax = plt.subplot(2, 2, 1)
    ax.annotate('(a)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("$x_1(t)$")
    plt.plot(t, x_rk.y[0], color=c1, label="Runge-Kutta")
    plt.plot(t, x_nn[:,0], color=c2, linestyle="dotted", label="DeqNN")
    plt.legend(loc='upper right', frameon=False)

    ax = plt.subplot(2, 2, 2)
    ax.annotate('(b)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("$x_1(t)$")
    plt.plot(t, x_rk.y[1], color=c1, label="Runge-Kutta")
    plt.plot(t, x_nn[:,1], color=c2, linestyle="dotted", label="DeqNN")

    ax = plt.subplot(2, 2, 3)
    ax.annotate('(c)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("$|x-\hat x|$")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(t, np.abs(x_rk.y[0]-x_nn[:,0]), color=c1, label="$x_1$")
    plt.plot(t, np.abs(x_rk.y[1]-x_nn[:,1]), color=c2, label="$x_2$")
    plt.legend(loc='upper right', frameon=False)

    ax = plt.subplot(2, 2, 4)
    ax.annotate('(d)', xycoords="axes fraction", xy=(-0.3,0.9))
    plt.xlabel("$t$")
    plt.ylabel("DEQ $|G|$")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(t, np.abs(G[:,0]), color=c1, label="$G_1$")
    plt.plot(t, np.abs(G[:,1]), color=c2, label="$G_2$")
    plt.legend(loc='upper right', frameon=False)


    plt.savefig("harmonic_oscillator_tanh_pretrained.png", dpi=200)

