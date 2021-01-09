import sys
import copy
import numpy as np

import sys
sys.path.insert(1, '../')
from layers import *
from optimizers import *
from deqnn import *
from bernoulli import *
from harmonic_oscillator import *

np.random.seed(1)


def diff(a, b):
    """Calculates normalized sum of absolute squared element differences
        between two matrices
    """
    ndiff = np.sum(np.abs(a-b)**2)
    return ndiff/np.prod(a.shape)


def test_dxdt(nn, t):
    """Checks implementation of nn.predict()
        by comparing result with numerical derivative
    """
    print("Test dx/dt: ", end="")
    x,dx,caches = nn.predict(t)
    eps = 1e-4
    x1,_,__ = nn.predict(t+eps)
    x2,_,__ = nn.predict(t-eps)
    dx_numerical = (x1 - x2)/(2*eps)

    #print(dx)
    #print(dx_numerical)
    assert(diff(dx, dx_numerical) < 1e-10)
    print("OK", format(diff(dx, dx_numerical)))


def test_dGdx(nn, t):
    """Checks implementation of nn.dGdx()
        by comparing result with numerical derivative
    """
    print("Test dG/dx: ", end="")
    x,dx,caches = nn.predict(t)
    eps = 1e-4
    dGdx = nn.dGdx(x, dx, t)
    dGdx_numerical = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[1]):
        x1 = x.copy()
        x2 = x.copy()
        x1[:,i] += eps
        x2[:,i] -= eps
        G1 = nn.G(x1, dx, t)
        G2 = nn.G(x2, dx, t)
        dGdx_numerical[:,:,i] = (G1 - G2)/(2*eps)

    #print(dGdx)
    #print(dGdx_numerical)
    assert(diff(dGdx, dGdx_numerical) < 1e-10)
    print("OK", diff(dGdx, dGdx_numerical))


def test_dGddx(nn, t):
    """Checks implementation of nn.dGddx()
        by comparing result with numerical derivative
    """
    print("Test dG/dx': ", end="")
    x,dx,caches = nn.predict(t)
    G = nn.G(x, dx, t)
    eps = 1e-4
    dGdx = nn.dGdx(x, dx, t)
    dGddx = nn.dGddx(x, dx, t)
    dGddx_numerical = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[1]):
        dx1 = dx.copy()
        dx2 = dx.copy()
        dx1[:,i] += eps
        dx2[:,i] -= eps
        G1 = nn.G(x, dx1, t)
        G2 = nn.G(x, dx2, t)
        dGddx_numerical[:,:,i] = (G1 - G2)/(2*eps)

    #print(dGddx)
    #print(dGddx_numerical)
    assert(diff(dGddx, dGddx_numerical) < 1e-10)
    print("OK", diff(dGddx, dGddx_numerical))


def test_delta_and_eta(nn, t):
    """Check correct calculation of cache["delta"] and cache["eta"]
        by comparing result with numerical derivative
    """
    print("Test delta_and_eta(nn, t): ", end="")
    x,dx,caches = nn.predict(t)
    G = nn.G(x, dx, t)
    dGdx = nn.dGdx(x, dx, t)
    dGddx = nn.dGddx(x, dx, t)
    for il in reversed(range(len(nn.layers))):
        ic = il+1
        dLdW, dLdb = nn.dLdWdb(caches, G, dGdx, dGddx, il)

        eps = 1e-4
        delta_eta_numerical = np.zeros(caches[ic]["delta"].shape)

        for i in range(delta_eta_numerical.shape[0]):
            for j in range(delta_eta_numerical.shape[1]):

                cache1 = copy.deepcopy(caches[ic])
                cache2 = copy.deepcopy(caches[ic])

                cache1["Z"][i,j] += eps
                cache2["Z"][i,j] -= eps
                cache1["A"] = nn.layers[il].a(cache1["Z"])
                cache2["A"] = nn.layers[il].a(cache2["Z"])
                cache1["dA"] = np.multiply(
                    np.dot(caches[ic-1]["dA"], nn.layers[il].W),
                    nn.layers[il].da(cache1["Z"])
                )
                cache2["dA"] = np.multiply(
                    np.dot(caches[ic-1]["dA"], nn.layers[il].W),
                    nn.layers[il].da(cache2["Z"])
                )

                for il2 in range(il+1, len(nn.layers)):
                    cache1 = nn.layers[il2].forward(cache1)
                    cache2 = nn.layers[il2].forward(cache2)

                x1 = cache1["A"]
                x2 = cache2["A"]
                dx1 = cache1["dA"]
                dx2 = cache2["dA"]
                G1 = nn.G(x1, dx1, t)
                G2 = nn.G(x2, dx2, t)
                L1 = nn.L(G1, x1[0])
                L2 = nn.L(G2, x2[0])

                delta_eta_numerical[i,j] = (L1 - L2)/(2*eps)

        delta_and_eta = copy.deepcopy(caches[ic]["delta"])
        for i in range(delta_and_eta.shape[1]):
            delta_and_eta[0,i] += caches[ic]["eta"][0,i]

        #print(delta_and_eta)
        #print(delta_eta_numerical)
        assert(diff(delta_eta_numerical, delta_and_eta) < 1e-10)
    print("OK", end=" ")
    print()


def test_gamma(nn, t):
    """Check correct calculation of cache["gamma"]
        by comparing result with numerical derivative
    """
    print("Test gamma(nn, t): ", end="")
    x,dx,caches = nn.predict(t)
    G = nn.G(x, dx, t)
    dGdx = nn.dGdx(x, dx, t)
    dGddx = nn.dGddx(x, dx, t)
    for il in reversed(range(len(nn.layers))):
        ic = il+1
        dLdW, dLdb = nn.dLdWdb(caches, G, dGdx, dGddx, il)

        eps = 1e-4
        gamma_numerical = np.zeros(caches[ic]["gamma"].shape)

        for i in range(gamma_numerical.shape[0]):
            for j in range(gamma_numerical.shape[1]):

                cache1 = copy.deepcopy(caches[ic])
                cache2 = copy.deepcopy(caches[ic])

                cache1["dA"][i,j] += eps
                cache2["dA"][i,j] -= eps

                for il2 in range(il+1, len(nn.layers)):
                    cache1 = nn.layers[il2].forward(cache1)
                    cache2 = nn.layers[il2].forward(cache2)

                x1 = cache1["A"]
                x2 = cache2["A"]
                dx1 = cache1["dA"]
                dx2 = cache2["dA"]
                G1 = nn.G(x1, dx1, t)
                G2 = nn.G(x2, dx2, t)
                L1 = nn.L(G1, x1[0])
                L2 = nn.L(G2, x2[0])

                gamma_numerical[i,j] = (L1 - L2)/(2*eps)

        #print(caches[ic]["gamma"])
        #print(gamma_numerical)
        assert(diff(gamma_numerical, caches[ic]["gamma"]) < 1e-10)
    print("OK", end=" ")
    print()


def test_dLdW(nn, t):
    """Check implementation of nn.dLdWdb()
        by comparing result with numerical derivative
    """
    print("Test dLdW(nn, t): ", end="")

    x,dx,caches = nn.predict(t)
    G = nn.G(x, dx, t)
    dGdx = nn.dGdx(x, dx, t)
    dGddx = nn.dGddx(x, dx, t)

    for il in reversed(range(len(nn.layers))):
        dLdW, dLdb = nn.dLdWdb(caches, G, dGdx, dGddx, il)

        eps = 1e-4
        dLdW_numerical = np.zeros(nn.layers[il].W.shape)

        layers = nn.layers
        for i in range(dLdW.shape[0]):
            for j in range(dLdW.shape[1]):

                nn.layers = copy.deepcopy(layers)
                nn.layers[il].W[i,j] += eps
                x1,dx1,caches1 = nn.predict(t)
                G1 = nn.G(x1, dx1, t)
                L1 = nn.L(G1, x1[0])

                nn.layers = copy.deepcopy(layers)
                nn.layers[il].W[i,j] -= eps
                x2,dx2,caches2 = nn.predict(t)
                G2 = nn.G(x2, dx2, t)
                L2 = nn.L(G2, x2[0])

                dLdW_numerical[i,j] = (L1 - L2)/(2*eps)

        nn.layers = layers

        #print(dLdW)
        #print(dLdW_numerical)
        assert(diff(dLdW_numerical, dLdW) < 1e-10)
    print("OK", end=" ")
    print()


def test_dLdb(nn, t):
    """Check implementation of nn.dLdWdb()
        by comparing result with numerical derivative
    """
    print("Test dLdb(nn, t): ", end="")

    x,dx,caches = nn.predict(t)
    G = nn.G(x, dx, t)
    dGdx = nn.dGdx(x, dx, t)
    dGddx = nn.dGddx(x, dx, t)
    for il in reversed(range(len(nn.layers))):
        dLdW, dLdb = nn.dLdWdb(caches, G, dGdx, dGddx, il)

        eps = 1e-4
        dLdb_numerical = np.zeros(nn.layers[il].b.shape)

        layers = nn.layers

        for i in range(dLdb.shape[0]):
            nn.layers = copy.deepcopy(layers)
            nn.layers[il].b[i] += eps
            x1,dx1,caches1 = nn.predict(t)
            G1 = nn.G(x1, dx1, t)
            L1 = nn.L(G1, x1[0])

            nn.layers = copy.deepcopy(layers)
            nn.layers[il].b[i] -= eps
            x2,dx2,caches2 = nn.predict(t)
            G2 = nn.G(x2, dx2, t)
            L2 = nn.L(G2, x2[0])

            dLdb_numerical[i] = (L1 - L2)/(2*eps)

        nn.layers = layers

        #print(dLdb)
        #print(dLdb_numerical)
        assert(diff(dLdb_numerical, dLdb) < 1e-10)
    print("OK", end=" ")
    print()


def test_deqnn_bernoulli():
    print("Test DeqNNBernoulli")
    t = np.vstack([
        [1],
        0.1 + np.random.random([10,1])*2
    ])
    nn = DeqNNBernoulli(
        layers=[
            SinLayer(1, 5),
            TanhLayer(5, 5),
            ReluLayer(5, 5),
            SigmoidLayer(5, 5),
            LinearLayer(5, 1)
        ],
        t0=t[0,0],
        x0=np.array([1])
    )
    test_dxdt(nn, t)
    test_dGdx(nn, t)
    test_dGddx(nn, t)
    test_delta_and_eta(nn, t)
    test_gamma(nn, t)
    test_dLdW(nn, t)
    test_dLdb(nn, t)
    print()


def test_deqnn_harmonic_oscillator():
    print("Test DeqNNHarmonicOscillator")
    t = np.vstack([
        [0],
        np.random.random([10,1])*10
    ])
    nn = DeqNNHarmonicOscillator(
        layers=[
            SinLayer(1, 5),
            ReluLayer(5, 5),
            SinLayer(5, 5),
            ReluLayer(5, 5),
            SinLayer(5, 5),
            LinearLayer(5, 2)
        ],
        t0=t[0,0],
        x0=np.array([0,1]),
        w0=1,
        g0=0.2
    )
    test_dxdt(nn, t)
    test_dGdx(nn, t)
    test_dGddx(nn, t)
    test_delta_and_eta(nn, t)
    test_gamma(nn, t)
    test_dLdW(nn, t)
    test_dLdb(nn, t)
    print()


if __name__ == "__main__":
    test_deqnn_bernoulli()
    test_deqnn_harmonic_oscillator()
