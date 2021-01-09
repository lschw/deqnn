import numpy as np
from layers import *
from optimizers import *


class DeqNN:

    def __init__(self, layers, t0, x0):
        """Setup network

        Args:
            layers: Layers of network
            t0: Initial time
            x0: Initial values, shape (N)
        """
        self.layers = layers
        self.t0 = t0
        self.x0 = x0
        self.N = len(x0)


    def G(self, x, dx, t):
        """Evaluates differential equation G(x,x',t) = 0

        Args:
            x: shape (S,N)
            dx: shape (S,N)
            t: shape (S,1)

        Returns:
            G: shape (S,N)
        """
        raise NotImplemented("G() not implemented")


    def dGdx(self, x, dx, t):
        """Evaluates derivative of differential equation G(x,x',t)
        with respect to x, i.e. dG/dx

        dG_sn / dx_m = dG_snm

        Args:
            x: shape (S,N)
            dx: shape (S,N)
            t: shape(S,1)

        Returns:
            3d tensor deriv[s,n,m]
        """
        raise NotImplemented("dGdx() not implemented")


    def dGddx(self, x, dx, t):
        """Evaluates derivative of differential equation G(x,x',t)
        with respect to x', i.e. dG/dx'

        dG_sn / ddx_m = dG_snm

        Args:
            x: shape (S,N)
            dx: shape (S,N)
            t: shape(S,1)

        Returns:
            3d tensor deriv[s,n,m]
        """
        raise NotImplemented("dGddx() not implemented")


    def init_weights(self):
        """Initialisizes all layers"""
        for l in self.layers:
            l.init_weights()


    def L(self, G, x0):
        """Calculates loss function,
        i.e. sum of differential equation loss and initial condition loss

        Args:
            G: Differential equation G(x,x',t) for predicted values x and x'
            x0: Predicted initial value
        """
        return 1./G.shape[0] * np.sum(np.square(G)) \
            + np.sum(np.square(self.x0 - x0))


    def predict(self, t):
        """Predicts x(t) and x'(t) by performing forward pass of network

        Args:
            t: Input of network, times to predict x(t), shape (S,1)
                Example:
                t = np.array([
                    [t1],
                    [t2],
                    ...,
                    [tS],
                ])

        Returns:
            A: Function x(t), shape (S, D)
                Example:
                A = np.array([
                    [x1(t1), x2(t1), ..., xD(t1)],
                    [x1(t2), x2(t2), ..., xD(t2)],
                    ...,
                    [x1(tS), x2(tS), ..., xD(tS)],
                ])
            dA: Derivative dx/dt, shape (S, D)
                Example:
                dA = np.array([
                    [dx1/dt1, dx2/dt1, ..., dxD/dt1],
                    [dx1/dt2, dx2/dt2, ..., dxD/dt2],
                    ...,
                    [dx1/dtS, dx2/dtS, ..., dxD/dtS],
                ])
            caches: Cached intermediate values of forward pass, list of dict
        """
        caches = [{"A": t, "Z": t, "dA": np.ones((t.shape[0],1))}]
        for layer in self.layers:
            caches.append(layer.forward(caches[-1]))
        return caches[-1]["A"], caches[-1]["dA"], caches


    def dLdWdb(self, caches, G, dGdx, dGddx, l):
        """Calculates derivative of loss function
            with respect to network weights
            for layer l by doing one backward step

        Args:
            caches: Cached forward pass, list of dict
            G: Differential equation G(x,x',t) for predicted values x and x'
            dGdx: Derivative of differential equation with respect to x
            dGddx: Derivative of differential equation with respect to x'
            l: Index of layer

        Returns:
            dLdW: Derivative of loss with respect to weights W
            dLdb: Derivative of loss with respect to biases b
        """
        cache_prev = caches[l]
        cache = caches[l+1]
        S = G.shape[0]

        # Handle last layer to compute initial values for backward pass
        if l == len(self.layers)-1:
            cache["delta"] = np.zeros(G.shape)
            cache["gamma"] = np.zeros(G.shape)
            for s in range(G.shape[0]):
                for j in range(G.shape[1]):
                    for n in range(G.shape[1]):
                        cache["delta"][s,j] += G[s,n]*dGdx[s,n,j]
                        cache["gamma"][s,j] += G[s,n]*dGddx[s,n,j]
            cache["delta"] *= 2/S
            cache["gamma"] *= 2/S
            cache["eta"] = 2 * (cache["A"][None,0,:] - self.x0)
            return self.layers[l].backward(cache, cache_prev, None, None)

        # Usual backward step for all other layers
        else:
            cache_next = caches[l+2]
            return self.layers[l].backward(
                cache, cache_prev, cache_next, self.layers[l+1])


    def train(self, Nepochs, Ns, tmin, tmax, optimizer, tol=1e-10,
            show_progress=10):
        """Trains network

        Args:
            Nepochs: Number of epochs
            Ns: Number of samples per epoch
            tmin: Training time is sampled between tmin and tmax
            tmax: Training time is sampled between tmin and tmax
            optimizer: Optimizer object
            tol: Stop training if loss is smaller than this value
            show_progress: None or epoch interval in which progress is shown

        Returns:
            loss: List of loss values of each epoch
        """
        loss = []
        optimizer.init_params(self.layers)
        for epoch in range(Nepochs):
            t = np.vstack([
                [self.t0], # Always add initial value time
                np.random.random([Ns-1,1])*(tmax-tmin) + tmin
            ])

            # Forward pass
            x,dx,caches = self.predict(t)
            G = self.G(x, dx, t)
            dGdx = self.dGdx(x, dx, t)
            dGddx = self.dGddx(x, dx, t)
            loss.append((epoch, self.L(G, x[0])))

            # Show progress
            if show_progress != None and epoch % show_progress == 0:
                print("Epoch {:05d}: Loss: {:.6f}".format(
                    epoch, loss[epoch][1]))

            # Early stopping if loss is smaller than tolerance
            if loss[epoch][1] < tol:
                if show_progress:
                    print("Solution converged after {}".format(epoch)
                        + " epochs with tolerance {}".format(tol))
                break

            # Backward pass and update
            for l in reversed(range(len(self.layers))):

                # One backward step
                dLdW, dLdb = self.dLdWdb(caches, G, dGdx, dGddx, l)

                # Parameter update
                optimizer.update(self.layers, dLdW, dLdb, l, epoch)

        return np.array(loss)
