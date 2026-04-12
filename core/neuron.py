import tensorflow as tf
from core.functions import surrogate_spike

class LIFCortexLayer:
    """
    Leaky Integrate-and-Fire (LIF) Cortex Layer.
    Processes inputs over discrete biological time steps instead of instantly.
    """
    def __init__(self, input_size, num_neurons, beta=0.9, threshold=1.0, noise_std=0.01, init_stddev=0.1):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.beta = beta  
        self.threshold = threshold
        self.noise_std = noise_std
        
        init_w = tf.random.normal(shape=(input_size, num_neurons), mean=0.0, stddev=init_stddev)
        self.weights = tf.Variable(initial_value=init_w, trainable=True, name="synaptic_weights")
        self.biases = tf.Variable(initial_value=tf.zeros(shape=(num_neurons,)), trainable=True, name="resting_potentials")
        
        # STRUCTURAL NEUROPLASTICITY VARIABLES
        self.synaptic_mask = tf.Variable(initial_value=tf.ones_like(self.weights), trainable=False, name="synaptic_mask")
        
        # Eligibility trace accumulation for Sleep Spawning (By-passes TF graph scope tracing limitations)
        self.hebbian_trace = tf.Variable(initial_value=tf.zeros_like(self.weights), trainable=False, name="hebbian_trace")
        
        # BIOLOGICAL PERSISTENCE: State variables for temporal continuity
        self.v_mem = tf.Variable(initial_value=tf.zeros((1, num_neurons)), trainable=False, name="membrane_state")

    def reset_state(self):
        self.v_mem.assign(tf.zeros_like(self.v_mem))

    def forward(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        # Load persistent state and expand to batch size
        v_mem = tf.tile(self.v_mem, [batch_size, 1])
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        
        active_weights = self.weights * self.synaptic_mask

        # TEMPORAL VECTORIZATION: Pre-calculate the entire sensory stream projection at once.
        # Projecting [Batch, Time, Input] -> [Batch, Time, Neurons] via flattened matmul.
        flat_inputs = tf.reshape(inputs, [-1, self.input_size])
        all_projections = tf.matmul(flat_inputs, active_weights) + self.biases
        all_projections = tf.reshape(all_projections, [batch_size, time_steps, self.num_neurons])

        for t in tf.range(time_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(v_mem, tf.TensorShape([None, self.num_neurons]))]
            )
            current_input = all_projections[:, t, :]
            
            # --- BIOLOGICAL SYNAPTIC JITTER ---
            # Prevents neural deadlock by vibrating potentials closer to the threshold.
            if self.noise_std > 0:
                noise = tf.random.normal(shape=tf.shape(v_mem), mean=0.0, stddev=self.noise_std)
                current_input = current_input + noise
            
            v_mem = self.beta * v_mem + current_input
            
            # --- LATERAL INHIBITION ---
            spikes_t_minus_1 = spike_trains.read(t-1) if t > 0 else tf.zeros((batch_size, self.num_neurons))
            spikes_t_minus_1 = tf.reshape(spikes_t_minus_1, [batch_size, self.num_neurons])
            inhibition = tf.reduce_mean(spikes_t_minus_1, axis=1, keepdims=True) * 0.25 
            v_mem = v_mem - inhibition
            
            spikes = surrogate_spike(v_mem, self.threshold)
            v_mem = v_mem - (self.threshold * spikes)
            spike_trains = spike_trains.write(t, spikes)

        stacked_spikes = spike_trains.stack()
        final_spikes = tf.transpose(stacked_spikes, perm=[1, 0, 2])
        
        # Save persistent state (Biological Memory) for the next cycle
        self.v_mem.assign(v_mem[:1, :])
        
        # Store physical synaptic memory rates DURING inference for Deep Sleep tracing
        self.last_input_rate = tf.reduce_mean(inputs, axis=[0, 1]) 
        self.last_output_rate = tf.reduce_mean(final_spikes, axis=[0, 1]) 
        
        return final_spikes

    def update_hebbian_trace(self):
        if hasattr(self, 'last_input_rate') and hasattr(self, 'last_output_rate'):
            demand = tf.expand_dims(self.last_input_rate, 1) * tf.expand_dims(self.last_output_rate, 0)
            self.hebbian_trace.assign_add(demand)

    def apply_stdp(self, learning_rate=1e-4, decay=1e-5):
        """ Strictly decoupled, biologically plausible local synaptic update. """
        delta = (learning_rate * self.hebbian_trace) - (decay * self.weights)
        self.weights.assign_add(delta * self.synaptic_mask)

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
    def __init__(self, input_shape, filters, kernel_size, stride=1, beta=0.9, threshold=1.0, noise_std=0.01, init_stddev=0.1):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.beta = beta
        self.threshold = threshold
        self.noise_std = noise_std
        
        init_w = tf.random.normal(shape=(kernel_size, kernel_size, input_shape[-1], filters), mean=0.0, stddev=init_stddev)
        self.weights = tf.Variable(initial_value=init_w, trainable=True, name="conv_weights")
        self.biases = tf.Variable(initial_value=tf.zeros(filters), trainable=True, name="conv_biases")
        
        self.synaptic_mask = tf.Variable(initial_value=tf.ones_like(self.weights), trainable=False, name="conv_mask")
        self.hebbian_trace = tf.Variable(initial_value=tf.zeros_like(self.weights), trainable=False, name="hebbian_trace")
        
        out_h = input_shape[0] // stride
        out_w = input_shape[1] // stride
        self.v_mem = tf.Variable(initial_value=tf.zeros((1, out_h, out_w, filters)), trainable=False, name="conv_membrane")

    def reset_state(self):
        self.v_mem.assign(tf.zeros_like(self.v_mem))

    def forward(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        spatial_inputs = tf.reshape(inputs, [batch_size, time_steps, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        
        out_h = self.input_shape[0] // self.stride
        out_w = self.input_shape[1] // self.stride
        
        v_mem = tf.tile(self.v_mem, [batch_size, 1, 1, 1])
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        active_weights = self.weights * self.synaptic_mask
        
        # TEMPORAL CONVOLUTIONAL VECTORIZATION: Pre-calculate the entire sensory stream.
        # Reshape [Batch, Time, H, W, C] -> [Batch * Time, H, W, C] for one large conv pass.
        flat_spatial = tf.reshape(spatial_inputs, [-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        all_conv_currents = tf.nn.conv2d(flat_spatial, active_weights, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.biases
        all_conv_currents = tf.reshape(all_conv_currents, [batch_size, time_steps, out_h, out_w, self.filters])

        for t in tf.range(time_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(v_mem, tf.TensorShape([None, out_h, out_w, self.filters]))]
            )
            current_input = all_conv_currents[:, t, :, :, :]
            
            if self.noise_std > 0:
                noise = tf.random.normal(shape=tf.shape(v_mem), mean=0.0, stddev=self.noise_std)
                current_input = current_input + noise

            v_mem = self.beta * v_mem + current_input
            
            # --- SPATIAL LATERAL INHIBITION ---
            if t > 0:
                spikes_t_minus_1 = spike_trains.read(t-1)
                spikes_t_minus_1 = tf.reshape(spikes_t_minus_1, [batch_size, out_h, out_w, self.filters])
                # --- SPATIAL LATERAL INHIBITION (Contrast Sharpening) ---
                blur = tf.nn.avg_pool2d(spikes_t_minus_1, ksize=3, strides=1, padding="SAME")
                v_mem = v_mem - (blur * 0.25)
            
            spikes = surrogate_spike(v_mem, self.threshold)
            v_mem = v_mem - (self.threshold * spikes)
            spike_trains = spike_trains.write(t, spikes)
            
        stacked_spikes = spike_trains.stack()
        spikes_time_batch = tf.transpose(stacked_spikes, perm=[1, 0, 2, 3, 4])
        
        # Save state
        self.v_mem.assign(v_mem[:1, :, :, :])
        
        self.last_in_rate = tf.reduce_mean(spatial_inputs, axis=[0, 1, 2, 3]) 
        self.last_out_rate = tf.reduce_mean(spikes_time_batch, axis=[0, 1, 2, 3]) 
        
        return tf.reshape(spikes_time_batch, [batch_size, time_steps, out_h * out_w * self.filters])

    def update_hebbian_trace(self):
        if hasattr(self, 'last_in_rate') and hasattr(self, 'last_out_rate'):
            demand_matrix = tf.expand_dims(self.last_in_rate, 1) * tf.expand_dims(self.last_out_rate, 0)
            demand_block = tf.reshape(demand_matrix, [1, 1, self.input_shape[-1], self.filters])
            demand_block = tf.tile(demand_block, [self.kernel_size, self.kernel_size, 1, 1])
            self.hebbian_trace.assign_add(demand_block)

    def apply_stdp(self, learning_rate=1e-4, decay=1e-5):
        delta = (learning_rate * self.hebbian_trace) - (decay * self.weights)
        self.weights.assign_add(delta * self.synaptic_mask)

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
    def __init__(self, input_size, num_neurons, beta=0.9, threshold=1.0, noise_std=0.01, init_stddev=0.1):
        super().__init__(input_size, num_neurons, beta, threshold, noise_std, init_stddev)
        
        init_rw = tf.random.normal(shape=(num_neurons, num_neurons), mean=0.0, stddev=init_stddev)
        self.recurrent_weights = tf.Variable(initial_value=init_rw, trainable=True, name="recurrent_weights")
        self.recurrent_synaptic_mask = tf.Variable(initial_value=tf.ones_like(self.recurrent_weights), trainable=False, name="recurrent_mask")
        
        self.recurrent_hebbian_trace = tf.Variable(initial_value=tf.zeros_like(self.recurrent_weights), trainable=False, name="recurrent_trace")
        
        # Recurrent state persistence
        self.prev_spikes = tf.Variable(initial_value=tf.zeros((1, num_neurons)), trainable=False, name="prev_spikes_state")

    def reset_state(self):
        super().reset_state()
        self.prev_spikes.assign(tf.zeros_like(self.prev_spikes))

    def forward(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        v_mem = tf.tile(self.v_mem, [batch_size, 1])
        prev_spikes = tf.tile(self.prev_spikes, [batch_size, 1])
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        
        active_weights = self.weights * self.synaptic_mask
        active_recurrent_weights = self.recurrent_weights * self.recurrent_synaptic_mask

        # TEMPORAL SENSORY VECTORIZATION: Pre-project the feedforward stream.
        flat_inputs = tf.reshape(inputs, [-1, self.input_size])
        all_ff_projections = tf.matmul(flat_inputs, active_weights)
        all_ff_projections = tf.reshape(all_ff_projections, [batch_size, time_steps, self.num_neurons])

        for t in tf.range(time_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (v_mem, tf.TensorShape([None, self.num_neurons])),
                    (prev_spikes, tf.TensorShape([None, self.num_neurons]))
                ]
            )
            feedforward_input = all_ff_projections[:, t, :]
            recurrent_input = tf.matmul(prev_spikes, active_recurrent_weights)
            current_input = feedforward_input + recurrent_input + self.biases
            
            if self.noise_std > 0:
                noise = tf.random.normal(shape=tf.shape(v_mem), mean=0.0, stddev=self.noise_std)
                current_input = current_input + noise

            v_mem = self.beta * v_mem + current_input
            
            # Lateral Inhibition
            prev_spikes = tf.reshape(prev_spikes, [batch_size, self.num_neurons])
            inhibition = tf.reduce_mean(prev_spikes, axis=1, keepdims=True) * 0.25 
            v_mem = v_mem - inhibition
            
            spikes = surrogate_spike(v_mem, self.threshold)
            v_mem = v_mem - (self.threshold * spikes)
            prev_spikes = spikes
            spike_trains = spike_trains.write(t, spikes)

        stacked_spikes = spike_trains.stack()
        final_spikes = tf.transpose(stacked_spikes, perm=[1, 0, 2])
        
        # Save persistent states
        self.v_mem.assign(v_mem[:1, :])
        self.prev_spikes.assign(prev_spikes[:1, :])
        
        # Store Demand Trace averages for decoupled sleep phase plasticity
        self.last_input_rate = tf.reduce_mean(inputs, axis=[0, 1]) 
        self.last_output_rate = tf.reduce_mean(final_spikes, axis=[0, 1]) 
        
        return final_spikes

    def update_hebbian_trace(self):
        super().update_hebbian_trace()
        if hasattr(self, 'last_output_rate'):
            demand_rw = tf.expand_dims(self.last_output_rate, 1) * tf.expand_dims(self.last_output_rate, 0)
            self.recurrent_hebbian_trace.assign_add(demand_rw)

    def apply_stdp(self, learning_rate=1e-4, decay=1e-5):
        super().apply_stdp(learning_rate, decay)
        delta_rw = (learning_rate * self.recurrent_hebbian_trace) - (decay * self.recurrent_weights)
        self.recurrent_weights.assign_add(delta_rw * self.recurrent_synaptic_mask)

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
    def __init__(self, input_shape, filters, kernel_size, stride=2, beta=0.9, threshold=1.0, noise_std=0.01, init_stddev=0.1):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.beta = beta
        self.threshold = threshold
        self.noise_std = noise_std
        
        # Deconvolution geometrically dilates matrices outward mathematically.
        # Kernel Shape: [height, width, output_channels, input_channels]
        init_w = tf.random.normal(shape=(kernel_size, kernel_size, filters, input_shape[-1]), mean=0.0, stddev=init_stddev)
        self.weights = tf.Variable(initial_value=init_w, trainable=True, name="deconv_weights")
        self.biases = tf.Variable(initial_value=tf.zeros(filters), trainable=True, name="deconv_biases")
        
        self.synaptic_mask = tf.Variable(initial_value=tf.ones_like(self.weights), trainable=False, name="conv_mask")
        
        out_h = input_shape[0] * stride
        out_w = input_shape[1] * stride
        self.v_mem = tf.Variable(initial_value=tf.zeros((1, out_h, out_w, filters)), trainable=False, name="deconv_membrane")

    def reset_state(self):
        self.v_mem.assign(tf.zeros_like(self.v_mem))

    def forward(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        spatial_inputs = tf.reshape(inputs, [batch_size, time_steps, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        
        out_h = self.input_shape[0] * self.stride
        out_w = self.input_shape[1] * self.stride
        
        v_mem = tf.tile(self.v_mem, [batch_size, 1, 1, 1])
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        active_weights = self.weights * self.synaptic_mask
        
        # TEMPORAL DECONVOLUTIONAL VECTORIZATION: Pre-calculate the entire motor stream.
        flat_p_inputs = tf.reshape(spatial_inputs, [-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        output_full_shape = [batch_size * time_steps, out_h, out_w, self.filters]
        
        all_deconv_currents = tf.nn.conv2d_transpose(flat_p_inputs, active_weights, output_shape=output_full_shape, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.biases
        all_deconv_currents = tf.reshape(all_deconv_currents, [batch_size, time_steps, out_h, out_w, self.filters])

        for t in tf.range(time_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(v_mem, tf.TensorShape([None, out_h, out_w, self.filters]))]
            )
            current_input = all_deconv_currents[:, t, :, :, :]
            
            if self.noise_std > 0:
                noise = tf.random.normal(shape=tf.shape(v_mem), mean=0.0, stddev=self.noise_std)
                current_input = current_input + noise

            v_mem = self.beta * v_mem + current_input
            
            # --- SPATIAL LATERAL INHIBITION (Contrast Sharpening) ---
            if t > 0:
                spikes_t_minus_1 = spike_trains.read(t-1)
                spikes_t_minus_1 = tf.reshape(spikes_t_minus_1, [batch_size, out_h, out_w, self.filters])
                blur = tf.nn.avg_pool2d(spikes_t_minus_1, ksize=3, strides=1, padding="SAME")
                v_mem = v_mem - (blur * 0.25)
            
            spikes = surrogate_spike(v_mem, self.threshold)
            v_mem = v_mem - (self.threshold * spikes)
            spike_trains = spike_trains.write(t, spikes)
            
        stacked_spikes = spike_trains.stack()
        spikes_time_batch = tf.transpose(stacked_spikes, perm=[1, 0, 2, 3, 4])
        
        # Save state
        self.v_mem.assign(v_mem[:1, :, :, :])
        
        return tf.reshape(spikes_time_batch, [batch_size, time_steps, out_h * out_w * self.filters])

    def apply_stdp(self, learning_rate=1e-4, decay=1e-5):
        pass # Deconv geometric mapping updates skipped natively for stability

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

    def apply_stdp(self, learning_rate=1e-4, decay=1e-5):
        for layer in self.layers:
            if hasattr(layer, 'apply_stdp'):
                layer.apply_stdp(learning_rate, decay)

    def prune(self, threshold=0.005):
        return sum([layer.prune(threshold) for layer in self.layers if hasattr(layer, 'prune')])

    def grow(self, threshold=0.1):
        return sum([layer.grow(threshold) for layer in self.layers if hasattr(layer, 'grow')])

    def reset_state(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_state'):
                layer.reset_state()

    def get_variables(self):
        vars = []
        for layer in self.layers:
            vars.extend(layer.get_variables())
        return vars
