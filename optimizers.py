import numpy as np


class Optimizer:

    def __init__(self, alpha):
        """Initialisizes optimizer

        Args:
            alpha: Learning rate
        """
        self.alpha = alpha


    def init_params(self, layers):
        """Initialisizes parameters of optimizer

        Args:
            layers: List of layers
        """
        pass


    def update(self, layers, dLdW, dLdb, l, epoch):
        """Perfoms parameter update

        Args:
            layers: List of layers
            dLdW: Derivative of loss with respect to weights W of layer l
            dLdb: Derivative of loss with respect to biases b of layer l
            l: Index of layer to update
            epoch: Training epoch
        """
        raise NotImplemented("update() not implemented")



class OptimizerGradientDescent(Optimizer):

    def __init__(self, alpha):
        super().__init__(alpha)


    def update(self, layers, dLdW, dLdb, l, epoch):
        layers[l].W -= self.alpha * dLdW
        layers[l].b -= self.alpha * dLdb



class OptimizerAdam(Optimizer):

    def __init__(self, alpha, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(alpha)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.params = None


    def init_params(self, layers):
        self.params = [{
            "mW": np.zeros(l.W.shape),
            "vW": np.zeros(l.W.shape),
            "mb": np.zeros(l.b.shape),
            "vb": np.zeros(l.b.shape)
        } for l in layers]


    def update(self, layers, dLdW, dLdb, l, epoch):
        self.params[l]["mW"] *= self.b1
        self.params[l]["mW"] += (1 - self.b1) * dLdW
        self.params[l]["vW"] *= self.b2
        self.params[l]["vW"] += (1 - self.b2) * dLdW**2
        m2W = self.params[l]["mW"] / (1 - self.b1**(epoch+1))
        v2W = self.params[l]["vW"] / (1 - self.b2**(epoch+1))
        layers[l].W -= self.alpha * m2W / (np.sqrt(v2W) + self.eps)

        self.params[l]["mb"] *= self.b1
        self.params[l]["mb"] += (1 - self.b1) * dLdb
        self.params[l]["vb"] *= self.b2
        self.params[l]["vb"] += (1 - self.b2) * dLdb**2
        m2b = self.params[l]["mb"] / (1 - self.b1**(epoch+1))
        v2b = self.params[l]["vb"] / (1 - self.b2**(epoch+1))
        layers[l].b -= self.alpha * m2b/(np.sqrt(v2b) + self.eps)
