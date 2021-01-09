import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys
sys.path.insert(1, '../')
from deqnn_tf import *

np.random.seed(1)
tf.random.set_seed(1)


class DeqNNHarmonicOscillator(DeqNN):
    def __init__(self, layers, t0, x0, w0, g0):
        super().__init__(layers, t0, x0)
        self.w0 = w0
        self.g0 = g0


    @tf.function
    def G(self, x, dx, t):
        x1 = x[:,0,None]
        x2 = x[:,1,None]
        dx1 = dx[:,0,None]
        dx2 = dx[:,1,None]
        return tf.concat([
            dx1 - x2,
            dx2 + self.w0**2 * x1 + self.g0*x2
        ], axis=1)



if __name__ == "__main__":

    tmin=0
    tmax=30
    t = np.arange(tmin,tmax,0.1)

    # Neural network solution
    nn = DeqNNHarmonicOscillator(
        layers=[
            tf.keras.layers.Dense(100, activation=tf.nn.tanh, input_shape=(1,),
                kernel_initializer="he_normal",
                bias_initializer=tf.keras.initializers.RandomUniform(
                    minval=0, maxval=1
                )
            ),
            tf.keras.layers.Dense(100, activation=tf.nn.tanh,
                kernel_initializer="he_normal",
                bias_initializer=tf.keras.initializers.RandomUniform(
                    minval=0, maxval=1
                )
            ),
            tf.keras.layers.Dense(2,
                kernel_initializer="he_normal",
                bias_initializer=tf.keras.initializers.RandomUniform(
                    minval=0, maxval=1
                )
            )
        ],
        t0=0,
        x0=np.array([0,1]),
        w0=1,
        g0=0.4
    )

    loss = nn.train(
        Nepochs=1660, Ns=200, tmin=tmin, tmax=tmax,
        optimizer=tf.keras.optimizers.Adam(0.01),
        tol=1e-4
    )

    x_nn,dx_nn = nn.predict(
        tf.convert_to_tensor(t.reshape(t.shape[0],1), dtype=tf.float32))
    G = nn.G(x_nn, dx_nn,
        tf.convert_to_tensor(t.reshape(t.shape[0],1), dtype=tf.float32))

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

    plt.savefig("harmonic_oscillator_tanh_tf.png", dpi=200)
