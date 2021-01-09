import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../')
from deqnn_tf import *

np.random.seed(1)
tf.random.set_seed(1)


class DeqNNBernoulli(DeqNN):

    @tf.function
    def G(self, x, dx, t):
        return dx - 2/(t+1e-10) * x + t**2 * x**2


if __name__ == "__main__":
    tmin=0.1
    tmax=10
    t = np.arange(tmin,tmax,0.1)

    # Neural network solution
    nn = DeqNNBernoulli(
        layers=[
            tf.keras.layers.Dense(10, activation=tf.nn.tanh, input_shape=(1,),
                kernel_initializer="he_normal",
                bias_initializer=tf.keras.initializers.RandomUniform(
                    minval=0, maxval=1
                )
            ),
            tf.keras.layers.Dense(1,
                kernel_initializer="he_normal",
                bias_initializer=tf.keras.initializers.RandomUniform(
                    minval=0, maxval=1
                )
            )
        ],
        t0=1,
        x0=np.array([1])
    )
    loss = nn.train(
        Nepochs=20000, Ns=50, tmin=tmin, tmax=tmax,
        optimizer=tf.keras.optimizers.Adam(0.005),
        tol=1e-6
    )
    x_nn,dx_nn = nn.predict(
        tf.convert_to_tensor(t.reshape(t.shape[0],1), dtype=tf.float32))
    G = nn.G(x_nn, dx_nn,
        tf.convert_to_tensor(t.reshape(t.shape[0],1), dtype=tf.float32))

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

    plt.savefig("bernoulli_tf.png", dpi=200)
