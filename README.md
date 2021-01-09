# Differential equation neural network (DeqNN)

This is an implementation of an artificial neural network
to solve ordinary differential equations,
where the neural network represents the solution function itself.
In contrast to a vanilla neural network,
the network has two outputs,
one corresponding to the solution function x(t) and one to its derivative x'(t).
Both quantities are calculated in a single forward pass.

The general network is implemented with numpy in the files
 * [deqnn.py](deqnn.py)
 * [layers.py](layers.py)
 * [optimizers.py](optimizers.py)

and with tensorflow in
 * [deqnn_tf.py](deqnn_tf.py)

In the folder [example/](example/),
a concrete implementation for the Bernoulli differential equation
and a damped harmonic oscillator can be found.

More details about the theory behind the architecture are described at
[https://lukasschwarz.de/deqnn](https://lukasschwarz.de/deqnn).
