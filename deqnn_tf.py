import numpy as np
import tensorflow as tf

class DeqNN:

    def __init__(self, layers, t0, x0):
        self.t0 = t0
        self.x0 = x0
        self.model = tf.keras.Sequential(layers)


    @tf.function
    def G(self, x, dx, t):
        raise NotImplemented("G() not implemented")


    @tf.function
    def L(self, x, dx, t):
        return (1./x.shape[0]
            * tf.math.reduce_sum(tf.math.square(self.G(x, dx, t)))
            + tf.math.reduce_sum(tf.math.square(self.x0 - x[0]))
        )


    @tf.function
    def predict(self, t):
        with tf.GradientTape() as tape_dx:
            tape_dx.watch(t)
            x = self.model(t)

        return x, tape_dx.batch_jacobian(x, t)[:,:,0]


    @tf.function
    def train_step(self, t, optimizer):
        # Forward pass
        with tf.GradientTape() as tape:
            x, dx = self.predict(t)
            loss = self.L(x, dx, t)

        # Backward pass and update
        grads = tape.gradient(
            loss, self.model.trainable_variables
        )
        optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights)
        )
        return loss


    def train(self, Nepochs, Ns, tmin, tmax, optimizer, tol=1e-10,
            show_progress=10):
        loss = []
        for epoch in range(Nepochs):
            t = np.vstack([
                [self.t0], # Always add initial value time
                np.random.random([Ns-1,1])*(tmax-tmin) + tmin
            ])
            t = tf.convert_to_tensor(t, dtype=tf.float32)

            # Train step
            loss.append((epoch,self.train_step(t, optimizer)))

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

        return np.array(loss)
