import tensorflow as tf
from core.functions import surrogate_spike

class LIFCortexLayer:
    """
    Leaky Integrate-and-Fire (LIF) Cortex Layer.
    Processes inputs over discrete biological time steps instead of instantly.
    """
    def __init__(self, input_size, num_neurons, beta=0.9, threshold=1.0):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.beta = beta  
        self.threshold = threshold
        
        init_w = tf.random.normal(shape=(input_size, num_neurons), mean=0.0, stddev=0.1)
        self.weights = tf.Variable(initial_value=init_w, trainable=True, name="synaptic_weights")
        self.biases = tf.Variable(initial_value=tf.zeros(shape=(num_neurons,)), trainable=True, name="resting_potentials")
        
        # STRUCTURAL NEUROPLASTICITY VARIABLES
        self.synaptic_mask = tf.Variable(initial_value=tf.ones_like(self.weights), trainable=False, name="synaptic_mask")
        
        # Eligibility trace accumulation for Sleep Spawning (By-passes TF graph scope tracing limitations)
        self.hebbian_trace = tf.Variable(initial_value=tf.zeros_like(self.weights), trainable=False, name="hebbian_trace")

    def forward(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        v_mem = tf.zeros((batch_size, self.num_neurons))
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        
        active_weights = self.weights * self.synaptic_mask

        for t in tf.range(time_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(v_mem, tf.TensorShape([None, self.num_neurons]))]
            )
            x_t = inputs[:, t, :]
            current_input = tf.matmul(x_t, active_weights) + self.biases
            v_mem = self.beta * v_mem + current_input
            
            # --- LATERAL INHIBITION ---
            spikes_t_minus_1 = spike_trains.read(t-1) if t > 0 else tf.zeros((batch_size, self.num_neurons))
            spikes_t_minus_1 = tf.reshape(spikes_t_minus_1, [batch_size, self.num_neurons])
            inhibition = tf.reduce_mean(spikes_t_minus_1, axis=1, keepdims=True) * 0.1 
            v_mem = v_mem - inhibition
            
            spikes = surrogate_spike(v_mem, self.threshold)
            v_mem = v_mem - (self.threshold * spikes)
            spike_trains = spike_trains.write(t, spikes)

        stacked_spikes = spike_trains.stack()
        final_spikes = tf.transpose(stacked_spikes, perm=[1, 0, 2])
        
        # Store physical synaptic memory rates DURING inference for Deep Sleep tracing
        self.last_input_rate = tf.reduce_mean(inputs, axis=[0, 1]) 
        self.last_output_rate = tf.reduce_mean(final_spikes, axis=[0, 1]) 
        
        return final_spikes

    def update_hebbian_trace(self):
        if hasattr(self, 'last_input_rate') and hasattr(self, 'last_output_rate'):
            demand = tf.expand_dims(self.last_input_rate, 1) * tf.expand_dims(self.last_output_rate, 0)
            self.hebbian_trace.assign_add(demand)

    def prune(self, threshold=0.005):
        # Sever weak synaptic connections entirely
        weak_synapses = tf.cast(tf.abs(self.weights) < threshold, tf.float32)
        survival_mask = 1.0 - weak_synapses
        new_mask = self.synaptic_mask * survival_mask
        pruned_count = tf.reduce_sum(self.synaptic_mask - new_mask)
        self.synaptic_mask.assign(new_mask)
        return pruned_count

    def grow(self, threshold=0.1):
        # Hebbian Demand: "Cells that fire together, wire together"
        spawning_candidates = tf.cast(self.hebbian_trace > threshold, tf.float32)
        pruned_spaces = 1.0 - self.synaptic_mask
        
        new_growth = spawning_candidates * pruned_spaces
        grown_count = tf.reduce_sum(new_growth)
        
        self.synaptic_mask.assign(self.synaptic_mask + new_growth)
        self.hebbian_trace.assign(tf.zeros_like(self.hebbian_trace)) # Flush trace after deep sleep
        return grown_count

    def get_variables(self):
        return [self.weights, self.biases]

class ConvLIFCortexLayer:
    """ Biological approximation of Retinotopy via restricted spatial receptive fields. """
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
        
        self.synaptic_mask = tf.Variable(initial_value=tf.ones_like(self.weights), trainable=False, name="conv_mask")
        self.hebbian_trace = tf.Variable(initial_value=tf.zeros_like(self.weights), trainable=False, name="hebbian_trace")

    def forward(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        spatial_inputs = tf.reshape(inputs, [batch_size, time_steps, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        
        out_h = self.input_shape[0] // self.stride
        out_w = self.input_shape[1] // self.stride
        
        v_mem = tf.zeros((batch_size, out_h, out_w, self.filters))
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        active_weights = self.weights * self.synaptic_mask
        
        for t in tf.range(time_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(v_mem, tf.TensorShape([None, out_h, out_w, self.filters]))]
            )
            x_t = spatial_inputs[:, t, :, :, :]
            current_input = tf.nn.conv2d(x_t, active_weights, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.biases
            v_mem = self.beta * v_mem + current_input
            
            # --- SPATIAL LATERAL INHIBITION ---
            if t > 0:
                spikes_t_minus_1 = spike_trains.read(t-1)
                spikes_t_minus_1 = tf.reshape(spikes_t_minus_1, [batch_size, out_h, out_w, self.filters])
                # Geometric smear suppressors (Contrast Sharpening)
                blur = tf.nn.avg_pool2d(spikes_t_minus_1, ksize=3, strides=1, padding="SAME")
                v_mem = v_mem - (blur * 0.1)
            
            spikes = surrogate_spike(v_mem, self.threshold)
            v_mem = v_mem - (self.threshold * spikes)
            spike_trains = spike_trains.write(t, spikes)
            
        stacked_spikes = spike_trains.stack()
        spikes_time_batch = tf.transpose(stacked_spikes, perm=[1, 0, 2, 3, 4])
        
        self.last_in_rate = tf.reduce_mean(spatial_inputs, axis=[0, 1, 2, 3]) 
        self.last_out_rate = tf.reduce_mean(spikes_time_batch, axis=[0, 1, 2, 3]) 
        
        return tf.reshape(spikes_time_batch, [batch_size, time_steps, out_h * out_w * self.filters])

    def update_hebbian_trace(self):
        if hasattr(self, 'last_in_rate') and hasattr(self, 'last_out_rate'):
            demand_matrix = tf.expand_dims(self.last_in_rate, 1) * tf.expand_dims(self.last_out_rate, 0)
            demand_block = tf.reshape(demand_matrix, [1, 1, self.input_shape[-1], self.filters])
            demand_block = tf.tile(demand_block, [self.kernel_size, self.kernel_size, 1, 1])
            self.hebbian_trace.assign_add(demand_block)

    def prune(self, threshold=0.005):
        weak_synapses = tf.cast(tf.abs(self.weights) < threshold, tf.float32)
        survival_mask = 1.0 - weak_synapses
        new_mask = self.synaptic_mask * survival_mask
        pruned_count = tf.reduce_sum(self.synaptic_mask - new_mask)
        self.synaptic_mask.assign(new_mask)
        return pruned_count

    def grow(self, threshold=0.1):
        spawning_candidates = tf.cast(self.hebbian_trace > threshold, tf.float32)
        pruned_spaces = 1.0 - self.synaptic_mask
        new_growth = spawning_candidates * pruned_spaces
        
        grown_count = tf.reduce_sum(new_growth)
        self.synaptic_mask.assign(self.synaptic_mask + new_growth)
        self.hebbian_trace.assign(tf.zeros_like(self.hebbian_trace))
        return grown_count

    def get_variables(self):
        return [self.weights, self.biases]

class RecurrentLIFCortexLayer(LIFCortexLayer):
    """ Biological Top-Down Attention mechanism """
    def __init__(self, input_size, num_neurons, beta=0.9, threshold=1.0):
        super().__init__(input_size, num_neurons, beta, threshold)
        
        init_rw = tf.random.normal(shape=(num_neurons, num_neurons), mean=0.0, stddev=0.1)
        self.recurrent_weights = tf.Variable(initial_value=init_rw, trainable=True, name="recurrent_weights")
        self.recurrent_synaptic_mask = tf.Variable(initial_value=tf.ones_like(self.recurrent_weights), trainable=False, name="recurrent_mask")
        
        self.recurrent_hebbian_trace = tf.Variable(initial_value=tf.zeros_like(self.recurrent_weights), trainable=False, name="recurrent_trace")

    def forward(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        v_mem = tf.zeros((batch_size, self.num_neurons))
        prev_spikes = tf.zeros((batch_size, self.num_neurons))
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        
        active_weights = self.weights * self.synaptic_mask
        active_recurrent_weights = self.recurrent_weights * self.recurrent_synaptic_mask

        for t in tf.range(time_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (v_mem, tf.TensorShape([None, self.num_neurons])),
                    (prev_spikes, tf.TensorShape([None, self.num_neurons]))
                ]
            )
            x_t = inputs[:, t, :]
            feedforward_input = tf.matmul(x_t, active_weights)
            recurrent_input = tf.matmul(prev_spikes, active_recurrent_weights)
            current_input = feedforward_input + recurrent_input + self.biases
            
            v_mem = self.beta * v_mem + current_input
            
            # Lateral Inhibition
            prev_spikes = tf.reshape(prev_spikes, [batch_size, self.num_neurons])
            inhibition = tf.reduce_mean(prev_spikes, axis=1, keepdims=True) * 0.1 
            v_mem = v_mem - inhibition
            
            spikes = surrogate_spike(v_mem, self.threshold)
            v_mem = v_mem - (self.threshold * spikes)
            prev_spikes = spikes
            spike_trains = spike_trains.write(t, spikes)

        stacked_spikes = spike_trains.stack()
        final_spikes = tf.transpose(stacked_spikes, perm=[1, 0, 2])
        
        # Store Demand Trace averages for decoupled sleep phase plasticity
        self.last_input_rate = tf.reduce_mean(inputs, axis=[0, 1]) 
        self.last_output_rate = tf.reduce_mean(final_spikes, axis=[0, 1]) 
        
        return final_spikes

    def update_hebbian_trace(self):
        super().update_hebbian_trace()
        if hasattr(self, 'last_output_rate'):
            demand_rw = tf.expand_dims(self.last_output_rate, 1) * tf.expand_dims(self.last_output_rate, 0)
            self.recurrent_hebbian_trace.assign_add(demand_rw)

    def prune(self, threshold=0.005):
        pruned_fw = super().prune(threshold)
        weak_rw = tf.cast(tf.abs(self.recurrent_weights) < threshold, tf.float32)
        surv_mask = 1.0 - weak_rw
        new_mask = self.recurrent_synaptic_mask * surv_mask
        pruned_rw = tf.reduce_sum(self.recurrent_synaptic_mask - new_mask)
        self.recurrent_synaptic_mask.assign(new_mask)
        return pruned_fw + pruned_rw

    def grow(self, threshold=0.1):
        grown_fw = super().grow(threshold)
        
        spawn = tf.cast(self.recurrent_hebbian_trace > threshold, tf.float32)
        empty = 1.0 - self.recurrent_synaptic_mask
        new_rw_growth = spawn * empty
        
        self.recurrent_synaptic_mask.assign(self.recurrent_synaptic_mask + new_rw_growth)
        self.recurrent_hebbian_trace.assign(tf.zeros_like(self.recurrent_hebbian_trace))
        return grown_fw + tf.reduce_sum(new_rw_growth)

    def get_variables(self):
        return [self.weights, self.recurrent_weights, self.biases]

class DeconvLIFCortexLayer:
    """ Biological Reverse-Retinotopy via Spatial Extrapolation. """
    def __init__(self, input_shape, filters, kernel_size, stride=2, beta=0.9, threshold=1.0):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.beta = beta
        self.threshold = threshold
        
        # Deconvolution geometrically dilates matrices outward mathematically.
        # Kernel Shape: [height, width, output_channels, input_channels]
        init_w = tf.random.normal(shape=(kernel_size, kernel_size, filters, input_shape[-1]), mean=0.0, stddev=0.1)
        self.weights = tf.Variable(initial_value=init_w, trainable=True, name="deconv_weights")
        self.biases = tf.Variable(initial_value=tf.zeros(filters), trainable=True, name="deconv_biases")
        
        self.synaptic_mask = tf.Variable(initial_value=tf.ones_like(self.weights), trainable=False, name="conv_mask")
        
    def forward(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        spatial_inputs = tf.reshape(inputs, [batch_size, time_steps, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        
        out_h = self.input_shape[0] * self.stride
        out_w = self.input_shape[1] * self.stride
        
        v_mem = tf.zeros((batch_size, out_h, out_w, self.filters))
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        active_weights = self.weights * self.synaptic_mask
        
        for t in tf.range(time_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(v_mem, tf.TensorShape([None, out_h, out_w, self.filters]))]
            )
            x_t = spatial_inputs[:, t, :, :, :]
            output_shape = [batch_size, out_h, out_w, self.filters]
            
            # Cortical Extrapolation: Firing backwards mechanically extrapolates data outward structurally grids.
            current_input = tf.nn.conv2d_transpose(x_t, active_weights, output_shape=output_shape, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.biases
            v_mem = self.beta * v_mem + current_input
            
            # --- SPATIAL LATERAL INHIBITION (Contrast Sharpening) ---
            if t > 0:
                spikes_t_minus_1 = spike_trains.read(t-1)
                spikes_t_minus_1 = tf.reshape(spikes_t_minus_1, [batch_size, out_h, out_w, self.filters])
                blur = tf.nn.avg_pool2d(spikes_t_minus_1, ksize=3, strides=1, padding="SAME")
                v_mem = v_mem - (blur * 0.1)
            
            spikes = surrogate_spike(v_mem, self.threshold)
            v_mem = v_mem - (self.threshold * spikes)
            spike_trains = spike_trains.write(t, spikes)
            
        stacked_spikes = spike_trains.stack()
        spikes_time_batch = tf.transpose(stacked_spikes, perm=[1, 0, 2, 3, 4])
        
        return tf.reshape(spikes_time_batch, [batch_size, time_steps, out_h * out_w * self.filters])

    def get_variables(self):
        return [self.weights, self.biases]

class SubCortexNetwork:
    """ Regional processing center routing over time. """
    def __init__(self, name="SubCortex"):
        self.name = name
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def update_hebbian_trace(self):
        for layer in self.layers:
            if hasattr(layer, 'update_hebbian_trace'):
                layer.update_hebbian_trace()

    def prune(self, threshold=0.005):
        return sum([layer.prune(threshold) for layer in self.layers if hasattr(layer, 'prune')])

    def grow(self, threshold=0.1):
        return sum([layer.grow(threshold) for layer in self.layers if hasattr(layer, 'grow')])

    def get_variables(self):
        vars = []
        for layer in self.layers:
            vars.extend(layer.get_variables())
        return vars
