import numpy as np


class Layer:

    def __init__(self, Ninput, Noutput):
        """Setup layer

        Args:
            Ninput: Number of inputs
            Noutput: Number of outputs
        """
        self.Ninput = Ninput
        self.Noutput = Noutput
        self.init_weights()


    def init_weights(self):
        """Initialisizes weights with random values using He initialization"""
        self.W = np.random.randn(self.Ninput, self.Noutput) \
            * np.sqrt(2./self.Ninput)
        self.b = np.random.rand(self.Noutput)


    def a(self, Z):
        """Calculates activation function"""
        raise NotImplemented("a() not implemented")


    def da(self, Z):
        """Calculates first derivative of activation function"""
        raise NotImplemented("da() not implemented")


    def da2(self, Z):
        """Calculates second derivative of activation function"""
        raise NotImplemented("da2() not implemented")


    def forward(self, cache_prev):
        """Performs forward pass

        Args:
            cache_prev: Cached values of previous layer

        Returns:
            Cached values of this layer
        """
        Z = np.dot(cache_prev["A"], self.W) + self.b
        A = self.a(Z)
        dA = np.multiply(np.dot(cache_prev["dA"], self.W), self.da(Z))
        return {"A": A, "Z": Z, "dA": dA}


    def backward(self, cache, cache_prev, cache_next, layer_next):
        """Performs backward pass

        Args:
            cache: Cached values of this layer
            cache_prev: Cached values of previous layer
            cache_next: Cached values of next layer
            layer_next: Next layer

        Returns:
            Cached values of this layer
        """
        if cache_next:
            cache["gamma"] = np.dot(
                np.multiply(
                    cache_next["gamma"],
                    layer_next.da(cache_next["Z"])
                ),
                layer_next.W.T
            )
            cache["delta"] = (
                np.multiply(
                    np.dot(cache_next["delta"], layer_next.W.T),
                    self.da(cache["Z"])
                )
                + np.multiply(
                    cache["gamma"],
                    np.multiply(
                        np.dot(cache_prev["dA"], self.W),
                        self.da2(cache["Z"])
                    )
                )
            )
            cache["eta"] = np.multiply(
                np.dot(cache_next["eta"], layer_next.W.T),
                self.da(cache["Z"][None,0,:])
            )

        S = cache["A"].shape[0]

        dLdW = (
            np.dot(cache_prev["A"].T, cache["delta"])
            + np.dot(
                cache_prev["dA"].T,
                np.multiply(
                    cache["gamma"],
                    self.da(cache["Z"])
                )
            )
            + np.dot(cache_prev["A"][None,0,:].T, cache["eta"])
        )

        dLdb = (
            np.dot(
                np.ones((S,1)).T,
                cache["delta"]
            )
            + np.dot(np.ones((1,1)).T, cache["eta"])
        )
        dLdb = dLdb.reshape(self.b.shape)

        return dLdW,dLdb



class LinearLayer(Layer):

    def a(self, Z):
        return Z

    def da(self, Z):
        return np.ones(Z.shape)

    def da2(self, Z):
        return np.zeros(Z.shape)



class ReluLayer(Layer):

    def a(self, Z):
        return np.maximum(0, Z)

    def da(self, Z):
        return (Z > 0)*1

    def da2(self, Z):
        return np.zeros(Z.shape)



class TanhLayer(Layer):

    def a(self, Z):
        return np.tanh(Z)

    def da(self, Z):
        return 1/np.cosh(Z)**2

    def da2(self, Z):
        return -2*np.sinh(Z)/np.cosh(Z)**3



class SigmoidLayer(Layer):

    def a(self, Z):
        return 1/(1+np.exp(-Z))

    def da(self, Z):
        s = self.a(Z)
        return s*(1-s)

    def da2(self, Z):
        s = self.a(Z)
        return s*(1-s)*(1-2*s)



class SinLayer(Layer):

    def a(self, Z):
        return np.sin(Z)

    def da(self, Z):
        return np.cos(Z)

    def da2(self, Z):
        return -np.sin(Z)


