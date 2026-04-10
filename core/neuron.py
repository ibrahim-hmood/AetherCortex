import tensorflow as tf
from core.functions import surrogate_spike

class LIFCortexLayer:
    """
    Leaky Integrate-and-Fire (LIF) Cortex Layer.
    Processes inputs over discrete biological time steps instead of instantly.
    Maintains membrane potential (Vm) internally.
    """
    def __init__(self, input_size, num_neurons, beta=0.9, threshold=1.0):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.beta = beta  # Membrane decay constant (leakage)
        self.threshold = threshold
        
        # Synaptic strengths
        init_w = tf.random.normal(shape=(input_size, num_neurons), mean=0.0, stddev=0.1)
        self.weights = tf.Variable(initial_value=init_w, trainable=True, name="synaptic_weights")
        
        # Resting potentials
        init_b = tf.zeros(shape=(num_neurons,))
        self.biases = tf.Variable(initial_value=init_b, trainable=True, name="resting_potentials")

    def forward(self, inputs):
        """
        Forward pass over a time sequence.
        inputs shape: [batch_size, time_steps, input_size]
        Returns spikes over time: [batch_size, time_steps, num_neurons]
        """
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        # Initial membrane potential: Vm(0) = 0
        v_mem = tf.zeros((batch_size, self.num_neurons))
        
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)

        # Iterate over discrete biological time steps
        for t in tf.range(time_steps):
            x_t = inputs[:, t, :]
            
            # Dendritic integration of excitatory/inhibitory input
            current_input = tf.matmul(x_t, self.weights) + self.biases
            
            # LIF Dynamics Update: Leakage + New Current
            v_mem = self.beta * v_mem + current_input
            
            # Spike generation using surrogate gradient function
            spikes = surrogate_spike(v_mem, self.threshold)
            
            # Soft Refractory Reset: subtract threshold if spike occurred
            v_mem = v_mem - (self.threshold * spikes)
            
            spike_trains = spike_trains.write(t, spikes)

        # Stack spikes back into shape [time_steps, batch_size, num_neurons]
        stacked_spikes = spike_trains.stack()
        # Transpose to target shape [batch_size, time_steps, num_neurons]
        return tf.transpose(stacked_spikes, perm=[1, 0, 2])

    def get_variables(self):
        return [self.weights, self.biases]

class SubCortexNetwork:
    """
    A sequence of LIFCortexLayers modeling a specific regional processing center over time.
    """
    def __init__(self, name="SubCortex"):
        self.name = name
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        """
        Passes spike train sequentially through each depth layer of the region.
        Ensures temporal consistency remains intact.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def get_variables(self):
        vars = []
        for layer in self.layers:
            vars.extend(layer.get_variables())
        return vars

class ConvLIFCortexLayer:
    """
    Biological approximation of Retinotopy via restricted spatial receptive fields.
    Instead of mapping all pixels instantly to all neurons, this layer locally groups
    synapses over shapes (like V1 extracting localized edges).
    """
    def __init__(self, input_shape, filters, kernel_size, stride=1, beta=0.9, threshold=1.0):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.beta = beta
        self.threshold = threshold
        
        init_w = tf.random.normal(shape=(kernel_size, kernel_size, input_shape[-1], filters), mean=0.0, stddev=0.1)
        self.weights = tf.Variable(initial_value=init_w, trainable=True, name="conv_weights")
        self.biases = tf.Variable(initial_value=tf.zeros(filters), trainable=True, name="conv_biases")

    def forward(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        spatial_inputs = tf.reshape(inputs, [batch_size, time_steps, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        
        out_h = self.input_shape[0] // self.stride
        out_w = self.input_shape[1] // self.stride
        
        v_mem = tf.zeros((batch_size, out_h, out_w, self.filters))
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        
        for t in tf.range(time_steps):
            x_t = spatial_inputs[:, t, :, :, :]
            current_input = tf.nn.conv2d(x_t, self.weights, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.biases
            
            v_mem = self.beta * v_mem + current_input
            spikes = surrogate_spike(v_mem, self.threshold)
            v_mem = v_mem - (self.threshold * spikes)
            
            spike_trains = spike_trains.write(t, spikes)
            
        stacked_spikes = spike_trains.stack()
        spikes_time_batch = tf.transpose(stacked_spikes, perm=[1, 0, 2, 3, 4])
        
        return tf.reshape(spikes_time_batch, [batch_size, time_steps, out_h * out_w * self.filters])

    def get_variables(self):
        return [self.weights, self.biases]

class RecurrentLIFCortexLayer(LIFCortexLayer):
    """
    Biological Top-Down Attention mechanism.
    Layer leverages its own past spikes at T-1 to establish sustained contexts
    or focal oscillations across time (similar to persistent memory).
    """
    def __init__(self, input_size, num_neurons, beta=0.9, threshold=1.0):
        super().__init__(input_size, num_neurons, beta, threshold)
        
        init_rw = tf.random.normal(shape=(num_neurons, num_neurons), mean=0.0, stddev=0.1)
        self.recurrent_weights = tf.Variable(initial_value=init_rw, trainable=True, name="recurrent_weights")

    def forward(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        v_mem = tf.zeros((batch_size, self.num_neurons))
        prev_spikes = tf.zeros((batch_size, self.num_neurons))
        
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)

        for t in tf.range(time_steps):
            x_t = inputs[:, t, :]
            
            feedforward_input = tf.matmul(x_t, self.weights)
            recurrent_input = tf.matmul(prev_spikes, self.recurrent_weights)
            current_input = feedforward_input + recurrent_input + self.biases
            
            v_mem = self.beta * v_mem + current_input
            spikes = surrogate_spike(v_mem, self.threshold)
            v_mem = v_mem - (self.threshold * spikes)
            
            prev_spikes = spikes
            spike_trains = spike_trains.write(t, spikes)

        stacked_spikes = spike_trains.stack()
        return tf.transpose(stacked_spikes, perm=[1, 0, 2])

    def get_variables(self):
        return [self.weights, self.recurrent_weights, self.biases]
