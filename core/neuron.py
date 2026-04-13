import tensorflow as tf
from core.functions import surrogate_spike

class LIFCortexLayer:
    """
    Leaky Integrate-and-Fire (LIF) Cortex Layer.
    Processes inputs over discrete biological time steps instead of instantly.
    """
    def __init__(self, input_size, num_neurons, beta=0.9, threshold=1.0, noise_std=0.01, init_stddev=0.1, persistence=1.0, facilitation=False):
        super().__init__()
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.beta = beta  
        self.threshold = threshold
        self.threshold_variable = tf.Variable(float(threshold), trainable=False, name="homeostatic_threshold")
        self.noise_std = tf.Variable(float(noise_std), trainable=False, name="homeostatic_noise_std")
        self.baseline_noise = float(noise_std)
        self.persistence = tf.Variable(float(persistence), trainable=False, name="synaptic_persistence")
        
        # --- CHEMICAL ECHOES (Short-Term Synaptic Facilitation) ---
        self.facilitation = facilitation
        self.synaptic_u = tf.Variable(initial_value=tf.zeros((1, num_neurons)), trainable=False, name="utilization")
        self.synaptic_x = tf.Variable(initial_value=tf.ones((1, num_neurons)), trainable=False, name="availability")
        # Pre-calculated decay constants for 16-32 step sequences
        self.u_decay = 0.9  # Facilitation decay
        self.x_recovery = 0.8 # Depression recovery
        self.U_inc = 0.2   # Utilization increment per spike
        
        init_w = tf.random.normal(shape=(input_size, num_neurons), mean=0.0, stddev=init_stddev)
        self.weights = tf.Variable(initial_value=init_w, trainable=True, name="synaptic_weights")
        self.biases = tf.Variable(initial_value=tf.zeros(shape=(num_neurons,)), trainable=True, name="resting_potentials")
        
        # STRUCTURAL NEUROPLASTICITY VARIABLES
        self.synaptic_mask = tf.Variable(initial_value=tf.ones_like(self.weights), trainable=False, name="synaptic_mask")
        
        # Eligibility trace accumulation for Sleep Spawning (By-passes TF graph scope tracing limitations)
        self.hebbian_trace = tf.Variable(initial_value=tf.zeros_like(self.weights), trainable=False, name="hebbian_trace")
        
        # --- LONG TERM MEMORY (Myelination v4.1) ---
        # Tracks synaptic durability. High permanence protects from pruning.
        self.permanence = tf.Variable(initial_value=tf.zeros_like(self.weights), trainable=False, name="synaptic_permanence")
        
        # BIOLOGICAL PERSISTENCE: State variables for temporal continuity and fatigue
        self.v_mem = tf.Variable(initial_value=tf.zeros((1, num_neurons)), trainable=False, name="membrane_state")
        self.t_state = tf.Variable(initial_value=tf.ones((1, num_neurons)) * threshold, trainable=False, name="dynamic_threshold")
        
        # SELECTIVE HABITUATION (Sensory Adaptation / Ignoring the Nose)
        self.habituation_state = tf.Variable(initial_value=tf.zeros((1, input_size)), trainable=False, name="habituation_buffer")

    def reset_state(self):
        self.v_mem.assign(tf.zeros_like(self.v_mem))
        self.t_state.assign(tf.ones_like(self.t_state) * self.threshold)
        self.habituation_state.assign(tf.zeros_like(self.habituation_state))
        self.synaptic_u.assign(tf.zeros_like(self.synaptic_u))
        self.synaptic_x.assign(tf.ones_like(self.synaptic_x))

    def forward(self, inputs):
        curr_b = tf.shape(inputs)[0]
        curr_t = tf.shape(inputs)[1]
        
        # Load persistent state and expand to batch size
        v_mem = tf.tile(self.v_mem, [curr_b, 1])
        t_state = tf.tile(self.t_state, [curr_b, 1])
        spike_trains = tf.TensorArray(tf.float32, size=curr_t)
        
        active_weights = self.weights * self.synaptic_mask

        # --- SELECTIVE HABITUATION (Sensory Gating) ---
        # Subtract persistent background from the entire sensory stream at once
        h_state = tf.tile(self.habituation_state, [curr_b * curr_t, 1])
        flat_inputs = tf.reshape(inputs, [-1, self.input_size])
        gated_inputs = flat_inputs - (h_state * 0.5)
        
        # Projecting [Batch * Time, Input] -> [Batch * Time, Neurons]
        # v2.5 Override: If facilitation is ON, we matmul INSIDE the loop for per-step gain.
        if not self.facilitation:
            all_projections = tf.matmul(gated_inputs, active_weights) + self.biases
            all_projections = tf.reshape(all_projections, [curr_b, curr_t, self.num_neurons])
        else:
            # Placeholder for loop-based projection
            all_projections = None

        u = tf.tile(self.synaptic_u, [curr_b, 1])
        x = tf.tile(self.synaptic_x, [curr_b, 1])
        prev_spikes = tf.zeros((curr_b, self.num_neurons))

        for t in tf.range(curr_t):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (v_mem, tf.TensorShape([None, self.num_neurons])),
                    (t_state, tf.TensorShape([None, self.num_neurons])),
                    (u, tf.TensorShape([None, self.num_neurons])),
                    (x, tf.TensorShape([None, self.num_neurons])),
                    (prev_spikes, tf.TensorShape([None, self.num_neurons]))
                ]
            )
            
            if self.facilitation:
                # Update facilitation variables (Chemical Echoes)
                u = u * self.u_decay + self.U_inc * (1.0 - u) * prev_spikes
                x = x * self.x_recovery + (1.0 - x) - (u * x * prev_spikes)
                x = tf.clip_by_value(x, 0.0, 1.0)
                
                # Apply dynamic synaptic gain (u*x can reach ~1.0-2.0 depending on spike history)
                # This makes 'hot' synapses temporarily 2x stronger.
                dynamic_current = tf.matmul(gated_inputs[curr_b*t : curr_b*(t+1)], active_weights) * (u * 2.0)
                current_input = tf.reshape(dynamic_current, [curr_b, self.num_neurons]) + self.biases
            else:
                current_input = all_projections[:, t, :]
            
            # --- BIOLOGICAL SYNAPTIC JITTER ---
            # Prevents neural deadlock by vibrating potentials closer to the threshold.
            if self.noise_std > 0:
                noise = tf.random.normal(shape=tf.shape(v_mem), mean=0.0, stddev=self.noise_std)
                current_input = current_input + noise
            
            v_mem = self.beta * v_mem + current_input
            
            # v0.1.8: Biological Potential Clipping (Prevent NaN explosions)
            v_mem = tf.clip_by_value(v_mem, -10.0, 10.0)
            
            # --- LATERAL INHIBITION ---
            spikes_t_minus_1 = spike_trains.read(t-1) if t > 0 else tf.zeros((curr_b, self.num_neurons))
            spikes_t_minus_1 = tf.reshape(spikes_t_minus_1, [curr_b, self.num_neurons])
            # Boosted inhibition from 0.25 to 0.45 for stability (Equilibrium Patch)
            inhibition = tf.reduce_mean(spikes_t_minus_1, axis=1, keepdims=True) * 0.45 
            v_mem = v_mem - inhibition
            
            # --- SPIKE GENERATION WITH DYNAMIC THRESHOLD ---
            spikes = surrogate_spike(v_mem, t_state)
            
            # --- HYPERPOLARIZATION & RECALIBRATED FATIGUE ---
            # Reset v_mem to a moderate negative value (Soft Reset)
            v_mem = v_mem - (t_state * spikes * 2.0)
            # Threshold jumps more gently (prevents flickering lockout)
            t_state = t_state + (spikes * 0.5)
            # Threshold decays back toward base biologically (Regional Variable)
            t_state = self.threshold_variable + (t_state - self.threshold_variable) * 0.9


            spike_trains = spike_trains.write(t, spikes)

        stacked_spikes = spike_trains.stack()
        final_spikes = tf.transpose(stacked_spikes, perm=[1, 0, 2])
        
        # Save persistent state (Biological Memory) for the next cycle
        self.v_mem.assign(v_mem[:1, :])
        self.t_state.assign(t_state[:1, :])
        self.synaptic_u.assign(u[:1, :])
        self.synaptic_x.assign(x[:1, :])
        
        # Update habituation state (Global sensory average)
        # Done outside the loop to ensure graph consistency
        avg_input = tf.reduce_mean(gated_inputs, axis=0, keepdims=True)
        self.habituation_state.assign(self.habituation_state * 0.9 + avg_input * 0.1)
        
        # Store physical synaptic memory rates DURING inference for Deep Sleep tracing
        self.last_input_rate = tf.reduce_mean(inputs, axis=[0, 1]) 
        self.last_output_rate = tf.reduce_mean(final_spikes, axis=[0, 1]) 
        
        return final_spikes

    def update_hebbian_trace(self):
        if hasattr(self, 'last_input_rate') and hasattr(self, 'last_output_rate'):
            demand = tf.expand_dims(self.last_input_rate, 1) * tf.expand_dims(self.last_output_rate, 0)
            self.hebbian_trace.assign_add(demand)

    def apply_stdp(self, learning_rate=1e-4, decay=1e-5, metabolic_tax=0.0, dopamine=1.0):
        """ 
        Strictly decoupled, biologically plausible local synaptic update. 
        Modulated by global dopamine levels (Reward-Directed Plasticity).
        """
        # 1. Sign-Aware Hebbian Update (Maintenance of E/I balance)
        # Dopamine multiplier enhances plasticity during 'Meaningful' moments.
        reward_modulated_lr = learning_rate * dopamine
        effective_lr = reward_modulated_lr * tf.sign(self.weights)
        
        # 2. Austerity Decay: Metabolic tax physically starves noisy synapses
        # Synaptic Tagging: Active synapses (high trace) resist decay (min_decay_multiplier)
        # min_decay_multiplier logic prevents 'Consolidated' skills from being pruned.
        tag_protection = tf.cast(self.hebbian_trace > 0.5, tf.float32) * 0.1 # 90% protection
        actual_decay = (decay + (metabolic_tax * 0.005)) * self.persistence * (1.0 - tag_protection)
        
        delta = (effective_lr * self.hebbian_trace) - (actual_decay * self.weights)
        self.weights.assign_add(delta * self.synaptic_mask)

        # --- SYNAPTIC PERMANENCE (LTM Archiving) ---
        # Myelination: If reward is high, grow permanence (lock-in)
        # Gold Medal Threshold: 1.5 (Aha! moments)
        myelin_growth = tf.where(dopamine > 1.5, self.hebbian_trace * 0.05, tf.zeros_like(self.hebbian_trace))
        
        # Demyelination: If apathy is sustained, slowly forget (Risk Mitigation)
        # Apathy Threshold: 0.5 (Frustration/Mistakes)
        forgetting = tf.where(dopamine < 0.5, 0.001, 0.0)
        
        self.permanence.assign(tf.clip_by_value(self.permanence * (1.0 - forgetting) + myelin_growth, 0.0, 1.0))

        # 3. HOMEOSTATIC SYNAPTIC SCALING (Normalization)
        # Prevents any one neuron from becoming 'infinite' and seizing the network.
        # Every neuron has a fixed synaptic 'budget'.
        abs_weights = tf.abs(self.weights)
        total_inward_strength = tf.reduce_sum(abs_weights, axis=0, keepdims=True)
        # Target budget: ~20.0 to 40.0 units of total synaptic weight per neuron
        budget = 30.0 
        scaling_factor = tf.where(total_inward_strength > budget, budget / (total_inward_strength + 1e-6), tf.ones_like(total_inward_strength))
        self.weights.assign(self.weights * scaling_factor)

    def prune(self, threshold=0.005):
        # Sever weak synaptic connections entirely
        # v4.1: Protect "Gold Medal" synapses from pruning (Shield threshold: 0.2 permanence)
        weak_synapses = tf.cast(tf.abs(self.weights) < threshold, tf.float32)
        permanently_locked = tf.cast(self.permanence > 0.2, tf.float32)
        
        # Survival rule: (Strong enough) OR (Permanently Locked)
        survival_mask = 1.0 - (weak_synapses * (1.0 - permanently_locked))
        
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
        return [self.weights, self.biases, self.permanence]

class ConvLIFCortexLayer:
    """ Biological approximation of Retinotopy via restricted spatial receptive fields. """
    def __init__(self, input_shape, filters, kernel_size, stride=1, beta=0.9, threshold=1.0, noise_std=0.01, init_stddev=0.1, persistence=1.0):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.beta = beta
        self.threshold = threshold
        self.threshold_variable = tf.Variable(float(threshold), trainable=False, name="conv_homeostatic_threshold")
        self.noise_std = tf.Variable(float(noise_std), trainable=False, name="homeostatic_noise_std")
        self.baseline_noise = float(noise_std)
        self.persistence = tf.Variable(float(persistence), trainable=False, name="synaptic_persistence")
        
        init_w = tf.random.normal(shape=(kernel_size, kernel_size, input_shape[-1], filters), mean=0.0, stddev=init_stddev)
        self.weights = tf.Variable(initial_value=init_w, trainable=True, name="conv_weights")
        self.biases = tf.Variable(initial_value=tf.zeros(filters), trainable=True, name="conv_biases")
        
        self.synaptic_mask = tf.Variable(initial_value=tf.ones_like(self.weights), trainable=False, name="conv_mask")
        self.hebbian_trace = tf.Variable(initial_value=tf.zeros_like(self.weights), trainable=False, name="hebbian_trace")
        self.permanence = tf.Variable(initial_value=tf.zeros_like(self.weights), trainable=False, name="conv_permanence")
        
        out_h = input_shape[0] // stride
        out_w = input_shape[1] // stride
        self.v_mem = tf.Variable(initial_value=tf.zeros((1, out_h, out_w, filters)), trainable=False, name="conv_membrane")
        self.t_state = tf.Variable(initial_value=tf.ones((1, out_h, out_w, filters)) * threshold, trainable=False, name="conv_dynamic_threshold")
        
        # Habituation for spatial features
        self.habituation_state = tf.Variable(initial_value=tf.zeros((1, input_shape[0], input_shape[1], input_shape[2])), trainable=False, name="conv_habit_buffer")

    def reset_state(self):
        self.v_mem.assign(tf.zeros_like(self.v_mem))
        self.t_state.assign(tf.ones_like(self.t_state) * self.threshold)
        self.habituation_state.assign(tf.zeros_like(self.habituation_state))

    def forward(self, inputs):
        curr_b = tf.shape(inputs)[0]
        curr_t = tf.shape(inputs)[1]
        
        spatial_inputs = tf.reshape(inputs, [curr_b, curr_t, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        
        out_h = self.input_shape[0] // self.stride
        out_w = self.input_shape[1] // self.stride
        
        v_mem = tf.tile(self.v_mem, [curr_b, 1, 1, 1])
        t_state = tf.tile(self.t_state, [curr_b, 1, 1, 1])
        spike_trains = tf.TensorArray(tf.float32, size=curr_t)
        active_weights = self.weights * self.synaptic_mask
        
        # --- SELECTIVE HABITUATION (Spatial Gating) ---
        flat_spatial = tf.reshape(spatial_inputs, [-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        h_state = tf.tile(self.habituation_state, [curr_b * curr_t, 1, 1, 1])
        gated_spatial = flat_spatial - (h_state * 0.5)
        
        all_conv_currents = tf.nn.conv2d(gated_spatial, active_weights, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.biases
        all_conv_currents = tf.reshape(all_conv_currents, [curr_b, curr_t, out_h, out_w, self.filters])

        for t in tf.range(curr_t):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (v_mem, tf.TensorShape([None, out_h, out_w, self.filters])),
                    (t_state, tf.TensorShape([None, out_h, out_w, self.filters]))
                ]
            )
            current_input = all_conv_currents[:, t, :, :, :]
            
            # --- BIOLOGICAL SYNAPTIC JITTER ---
            if self.noise_std > 0:
                noise = tf.random.normal(shape=tf.shape(v_mem), mean=0.0, stddev=self.noise_std)
                current_input = current_input + noise

            v_mem = self.beta * v_mem + current_input
            
            # --- SPATIAL LATERAL INHIBITION (Contrast Sharpening) ---
            if t > 0:
                spikes_t_minus_1 = spike_trains.read(t-1)
                spikes_t_minus_1 = tf.reshape(spikes_t_minus_1, [curr_b, out_h, out_w, self.filters])
                blur = tf.nn.avg_pool2d(spikes_t_minus_1, ksize=3, strides=1, padding="SAME")
                v_mem = v_mem - (blur * 0.25)
            
            # --- SPIKE GENERATION WITH DYNAMIC THRESHOLD ---
            spikes = surrogate_spike(v_mem, t_state)
            
            # --- HYPERPOLARIZATION & RECALIBRATED FATIGUE ---
            v_mem = v_mem - (t_state * spikes * 2.0)
            t_state = t_state + (spikes * 0.5)
            t_state = self.threshold_variable + (t_state - self.threshold_variable) * 0.9

            # Update spatial habituation (Retinal persistence)
            # Use gated_spatial[batch*t] as sensory source to update the habituation buffer
            step_spatial = tf.reshape(gated_spatial, [curr_b, curr_t, self.input_shape[0], self.input_shape[1], self.input_shape[2]])[:, t, :, :, :]
            avg_frame = tf.reduce_mean(step_spatial, axis=0, keepdims=True)
            self.habituation_state.assign(self.habituation_state * 0.9 + avg_frame * 0.1)

            spike_trains = spike_trains.write(t, spikes)
            
        stacked_spikes = spike_trains.stack()
        spikes_time_batch = tf.transpose(stacked_spikes, perm=[1, 0, 2, 3, 4])
        
        # Save state
        self.v_mem.assign(v_mem[:1, :, :, :])
        self.t_state.assign(t_state[:1, :, :, :])
        
        self.last_in_rate = tf.reduce_mean(spatial_inputs, axis=[0, 1, 2, 3]) 
        self.last_out_rate = tf.reduce_mean(spikes_time_batch, axis=[0, 1, 2, 3]) 
        
        return tf.reshape(spikes_time_batch, [curr_b, curr_t, out_h * out_w * self.filters])

    def update_hebbian_trace(self):
        if hasattr(self, 'last_in_rate') and hasattr(self, 'last_out_rate'):
            demand_matrix = tf.expand_dims(self.last_in_rate, 1) * tf.expand_dims(self.last_out_rate, 0)
            demand_block = tf.reshape(demand_matrix, [1, 1, self.input_shape[-1], self.filters])
            demand_block = tf.tile(demand_block, [self.kernel_size, self.kernel_size, 1, 1])
            self.hebbian_trace.assign_add(demand_block)

    def apply_stdp(self, learning_rate=1e-4, decay=1e-5, metabolic_tax=0.0, dopamine=1.0):
        # Convolutional STDP update modulated by dopamine
        reward_modulated_lr = learning_rate * dopamine
        actual_decay = decay + (metabolic_tax * 0.005)
        delta = (reward_modulated_lr * self.hebbian_trace) - (actual_decay * self.weights)
        self.weights.assign_add(delta * self.synaptic_mask)
        
        # --- SYNAPTIC PERMANENCE (Conv LTM) ---
        myelin_growth = tf.where(dopamine > 1.5, self.hebbian_trace * 0.05, tf.zeros_like(self.hebbian_trace))
        forgetting = tf.where(dopamine < 0.5, 0.001, 0.0)
        self.permanence.assign(tf.clip_by_value(self.permanence * (1.0 - forgetting) + myelin_growth, 0.0, 1.0))
        
        # Homeostatic Scaling for Conv filters (Target budget: 15.0 units per filter)
        abs_weights = tf.abs(self.weights)
        total_inward = tf.reduce_sum(abs_weights, axis=[0, 1, 2], keepdims=True)
        budget = 15.0
        scaling = tf.where(total_inward > budget, budget / (total_inward + 1e-6), tf.ones_like(total_inward))
        self.weights.assign(self.weights * scaling)

    def prune(self, threshold=0.005):
        weak_synapses = tf.cast(tf.abs(self.weights) < threshold, tf.float32)
        permanently_locked = tf.cast(self.permanence > 0.2, tf.float32)
        survival_mask = 1.0 - (weak_synapses * (1.0 - permanently_locked))
        
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
        return [self.weights, self.biases, self.permanence]

class RecurrentLIFCortexLayer(LIFCortexLayer):
    """ Biological Top-Down Attention mechanism """
    def __init__(self, input_size, num_neurons, beta=0.9, threshold=1.0, noise_std=0.01, init_stddev=0.1, facilitation=False):
        super().__init__(input_size, num_neurons, beta, threshold, noise_std, init_stddev, facilitation=facilitation)
        
        init_rw = tf.random.normal(shape=(num_neurons, num_neurons), mean=0.0, stddev=init_stddev)
        self.recurrent_weights = tf.Variable(initial_value=init_rw, trainable=True, name="recurrent_weights")
        self.recurrent_synaptic_mask = tf.Variable(initial_value=tf.ones_like(self.recurrent_weights), trainable=False, name="recurrent_mask")
        
        self.recurrent_hebbian_trace = tf.Variable(initial_value=tf.zeros_like(self.recurrent_weights), trainable=False, name="recurrent_trace")
        self.recurrent_permanence = tf.Variable(initial_value=tf.zeros_like(self.recurrent_weights), trainable=False, name="recurrent_permanence")
        
        # Recurrent state persistence
        self.prev_spikes = tf.Variable(initial_value=tf.zeros((1, num_neurons)), trainable=False, name="prev_spikes_state")
        self.t_state = tf.Variable(initial_value=tf.ones((1, num_neurons)) * threshold, trainable=False, name="rec_dynamic_threshold")

    def reset_state(self):
        super().reset_state()
        self.prev_spikes.assign(tf.zeros_like(self.prev_spikes))
        self.t_state.assign(tf.ones_like(self.t_state) * self.threshold)

    def forward(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        v_mem = tf.tile(self.v_mem, [batch_size, 1])
        t_state = tf.tile(self.t_state, [batch_size, 1])
        prev_spikes = tf.tile(self.prev_spikes, [batch_size, 1])
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        
        active_weights = self.weights * self.synaptic_mask
        active_recurrent_weights = self.recurrent_weights * self.recurrent_synaptic_mask

        # --- SELECTIVE HABITUATION (Sensory Gating) ---
        flat_inputs = tf.reshape(inputs, [-1, self.input_size])
        h_state = tf.tile(self.habituation_state, [batch_size * time_steps, 1])
        gated_inputs = flat_inputs - (h_state * 0.5)
        
        # TEMPORAL SENSORY VECTORIZATION: Pre-project the feedforward stream.
        all_ff_projections = tf.matmul(gated_inputs, active_weights)
        all_ff_projections = tf.reshape(all_ff_projections, [batch_size, time_steps, self.num_neurons])

        u = tf.tile(self.synaptic_u, [batch_size, 1])
        x = tf.tile(self.synaptic_x, [batch_size, 1])

        for t in tf.range(time_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (v_mem, tf.TensorShape([None, self.num_neurons])),
                    (t_state, tf.TensorShape([None, self.num_neurons])),
                    (prev_spikes, tf.TensorShape([None, self.num_neurons])),
                    (u, tf.TensorShape([None, self.num_neurons])),
                    (x, tf.TensorShape([None, self.num_neurons]))
                ]
            )
            
            if self.facilitation:
                # Update facilitation variables (Chemical Echoes)
                # Frontal focus: Recurrent connections also facilitate
                u = u * self.u_decay + self.U_inc * (1.0 - u) * prev_spikes
                x = x * self.x_recovery + (1.0 - x) - (u * x * prev_spikes)
                x = tf.clip_by_value(x, 0.0, 1.0)
                
                # Dynamic Synaptic Gain
                ff_proj = tf.matmul(gated_inputs[batch_size*t : batch_size*(t+1)], active_weights)
                rec_proj = tf.matmul(prev_spikes, active_recurrent_weights)
                current_input = (ff_proj + rec_proj) * (u * 2.0)
                current_input = tf.reshape(current_input, [batch_size, self.num_neurons]) + self.biases
            else:
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
            
            # --- SPIKE GENERATION WITH DYNAMIC THRESHOLD ---
            spikes = surrogate_spike(v_mem, t_state)
            
            # --- HYPERPOLARIZATION & FATIGUE ---
            v_mem = v_mem - (t_state * spikes * 3.5)
            t_state = t_state + (spikes * 1.2)
            t_state = self.threshold_variable + (t_state - self.threshold_variable) * 0.9

            prev_spikes = spikes
            spike_trains = spike_trains.write(t, spikes)

            # Update habituation state (Running average of sensory input)
            step_input = tf.reshape(gated_inputs, [batch_size, time_steps, self.input_size])[:, t, :]
            avg_input = tf.reduce_mean(step_input, axis=0, keepdims=True)
            self.habituation_state.assign(self.habituation_state * 0.9 + avg_input * 0.1)

        stacked_spikes = spike_trains.stack()
        final_spikes = tf.transpose(stacked_spikes, perm=[1, 0, 2])
        
        # Save persistent states
        self.v_mem.assign(v_mem[:1, :])
        self.prev_spikes.assign(prev_spikes[:1, :])
        self.t_state.assign(t_state[:1, :])
        self.synaptic_u.assign(u[:1, :])
        self.synaptic_x.assign(x[:1, :])
        
        # Store Demand Trace averages for decoupled sleep phase plasticity
        self.last_input_rate = tf.reduce_mean(inputs, axis=[0, 1]) 
        self.last_output_rate = tf.reduce_mean(final_spikes, axis=[0, 1]) 
        
        return final_spikes

    def update_hebbian_trace(self):
        super().update_hebbian_trace()
        if hasattr(self, 'last_output_rate'):
            demand_rw = tf.expand_dims(self.last_output_rate, 1) * tf.expand_dims(self.last_output_rate, 0)
            self.recurrent_hebbian_trace.assign_add(demand_rw)

    def apply_stdp(self, learning_rate=1e-4, decay=1e-5, metabolic_tax=0.0, dopamine=1.0):
        # 1. Update Feedforward weights via super (supports dopamine)
        super().apply_stdp(learning_rate, decay, metabolic_tax, dopamine=dopamine)
        
        # 2. Update Recurrent weights (modulated by dopamine)
        reward_modulated_lr = learning_rate * dopamine
        delta_rw = (reward_modulated_lr * self.recurrent_hebbian_trace) - (decay * self.recurrent_weights)
        self.recurrent_weights.assign_add(delta_rw * self.recurrent_synaptic_mask)

        # --- RECURRENT PERMANENCE (Sequence LTM) ---
        myelin_growth_rw = tf.where(dopamine > 1.5, self.recurrent_hebbian_trace * 0.05, tf.zeros_like(self.recurrent_hebbian_trace))
        forgetting_rw = tf.where(dopamine < 0.5, 0.001, 0.0)
        self.recurrent_permanence.assign(tf.clip_by_value(self.recurrent_permanence * (1.0 - forgetting_rw) + myelin_growth_rw, 0.0, 1.0))

    def prune(self, threshold=0.005):
        pruned_fw = super().prune(threshold)
        weak_rw = tf.cast(tf.abs(self.recurrent_weights) < threshold, tf.float32)
        permanently_locked_rw = tf.cast(self.recurrent_permanence > 0.2, tf.float32)
        
        surv_mask = 1.0 - (weak_rw * (1.0 - permanently_locked_rw))
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
        return [self.weights, self.recurrent_weights, self.biases, self.permanence, self.recurrent_permanence]

class DeconvLIFCortexLayer:
    """ Biological Reverse-Retinotopy via Spatial Extrapolation. """
    def __init__(self, input_shape, filters, kernel_size, stride=2, beta=0.9, threshold=1.0, noise_std=0.01, init_stddev=0.1):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.beta = beta
        self.threshold = threshold
        self.threshold_variable = tf.Variable(float(threshold), trainable=False, name="deconv_homeostatic_threshold")
        self.noise_std = tf.Variable(float(noise_std), trainable=False, name="homeostatic_noise_std")
        self.baseline_noise = float(noise_std)
        
        # Deconvolution geometrically dilates matrices outward mathematically.
        # Kernel Shape: [height, width, output_channels, input_channels]
        init_w = tf.random.normal(shape=(kernel_size, kernel_size, filters, input_shape[-1]), mean=0.0, stddev=init_stddev)
        self.weights = tf.Variable(initial_value=init_w, trainable=True, name="deconv_weights")
        self.biases = tf.Variable(initial_value=tf.zeros(filters), trainable=True, name="deconv_biases")
        
        self.synaptic_mask = tf.Variable(initial_value=tf.ones_like(self.weights), trainable=False, name="conv_mask")
        self.permanence = tf.Variable(initial_value=tf.zeros_like(self.weights), trainable=False, name="deconv_permanence")
        
        out_h = input_shape[0] * stride
        out_w = input_shape[1] * stride
        self.v_mem = tf.Variable(initial_value=tf.zeros((1, out_h, out_w, filters)), trainable=False, name="deconv_membrane")
        self.t_state = tf.Variable(initial_value=tf.ones((1, out_h, out_w, filters)) * threshold, trainable=False, name="deconv_dynamic_threshold")

    def reset_state(self):
        self.v_mem.assign(tf.zeros_like(self.v_mem))
        self.t_state.assign(tf.ones_like(self.t_state) * self.threshold)

    def forward(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        spatial_inputs = tf.reshape(inputs, [batch_size, time_steps, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        
        out_h = self.input_shape[0] * self.stride
        out_w = self.input_shape[1] * self.stride
        
        v_mem = tf.tile(self.v_mem, [batch_size, 1, 1, 1])
        t_state = tf.tile(self.t_state, [batch_size, 1, 1, 1])
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        active_weights = self.weights * self.synaptic_mask
        
        # --- SELECTIVE HABITUATION (Spatial Gating) ---
        flat_p_inputs = tf.reshape(spatial_inputs, [-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        # Note: Deconv layers initialized without habituation buffer in previous step.
        # Adding it now.
        if not hasattr(self, 'habituation_state'):
            self.habituation_state = tf.Variable(initial_value=tf.zeros((1, self.input_shape[0], self.input_shape[1], self.input_shape[2])), trainable=False, name="deconv_habit_buffer")
            
        h_state = tf.tile(self.habituation_state, [batch_size * time_steps, 1, 1, 1])
        gated_inputs = flat_p_inputs - (h_state * 0.5)
        
        output_full_shape = [batch_size * time_steps, out_h, out_w, self.filters]
        all_deconv_currents = tf.nn.conv2d_transpose(gated_inputs, active_weights, output_shape=output_full_shape, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.biases
        all_deconv_currents = tf.reshape(all_deconv_currents, [batch_size, time_steps, out_h, out_w, self.filters])

        for t in tf.range(time_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (v_mem, tf.TensorShape([None, out_h, out_w, self.filters])),
                    (t_state, tf.TensorShape([None, out_h, out_w, self.filters]))
                ]
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
            
            # --- SPIKE GENERATION WITH DYNAMIC THRESHOLD ---
            spikes = surrogate_spike(v_mem, t_state)
            
            # --- HYPERPOLARIZATION & FATIGUE ---
            v_mem = v_mem - (t_state * spikes * 3.5)
            t_state = t_state + (spikes * 1.2)
            t_state = self.threshold_variable + (t_state - self.threshold_variable) * 0.9

            spike_trains = spike_trains.write(t, spikes)
            
        stacked_spikes = spike_trains.stack()
        spikes_time_batch = tf.transpose(stacked_spikes, perm=[1, 0, 2, 3, 4])
        
        # Save state
        self.v_mem.assign(v_mem[:1, :, :, :])
        self.t_state.assign(t_state[:1, :, :, :])
        
        return tf.reshape(spikes_time_batch, [batch_size, time_steps, out_h * out_w * self.filters])

    def update_hebbian_trace(self):
        """ Deconv layers use geometric mapping; traces are currently bypassed for motor stability. """
        pass

    def apply_stdp(self, learning_rate=1e-4, decay=1e-5, metabolic_tax=0.0, dopamine=1.0):
        """ Deconv STDP is bypassed to maintain geometric visual consistency. """
        pass

    def prune(self, threshold=0.005):
        """ Deconv pruning bypassed for motor-geometric stability. """
        return 0

    def grow(self, threshold=0.1):
        """ Deconv spawning bypassed for motor-geometric stability. """
        return 0

    def get_variables(self):
        return [self.weights, self.biases, self.permanence]

class SubCortexNetwork:
    """ Regional processing center routing over time. """
    def __init__(self, name="SubCortex"):
        self.name = name
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x, inject_x=None, inject_index=None):
        for i, layer in enumerate(self.layers):
            if inject_x is not None and i == inject_index:
                # v2.5 Biological Injection (Top-Down Focus)
                # Ensure time-steps match by tiling if necessary
                if len(tf.shape(inject_x)) == 3:
                     x = x + inject_x
                else:
                     x = x + tf.expand_dims(inject_x, 1)
            x = layer.forward(x)
        return x

    def update_hebbian_trace(self):
        for layer in self.layers:
            if hasattr(layer, 'update_hebbian_trace'):
                layer.update_hebbian_trace()

    def apply_stdp(self, learning_rate=1e-4, decay=1e-5, metabolic_tax=0.0, dopamine=1.0):
        for layer in self.layers:
            if hasattr(layer, 'apply_stdp'):
                layer.apply_stdp(learning_rate, decay, metabolic_tax, dopamine=dopamine)

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
