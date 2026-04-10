import tensorflow as tf

@tf.custom_gradient
def surrogate_spike(v_mem, threshold):
    """
    Spike emission with Surrogate Gradient.
    Forward pass: Heaviside step function - 1 if v_mem > threshold else 0.
    Backward pass: Fast Sigmoid derivative approximation to permit gradient flow through spikes.
    """
    # Forward calculation
    spikes = tf.where(v_mem >= threshold, tf.ones_like(v_mem), tf.zeros_like(v_mem))

    def grad(dy):
        # Surrogate gradient (fast sigmoid derivative):
        # f(x) = x / (1 + |x|)
        # f'(x) = 1 / (1 + |x|)^2
        gamma = 10.0
        x = v_mem - threshold
        grad_v_mem = dy * (1.0 / tf.square(1.0 + gamma * tf.abs(x)))
        return grad_v_mem, None  # No gradient update for the threshold itself

    return spikes, grad
