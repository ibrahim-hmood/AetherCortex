import tensorflow as tf
from core.neuron import LIFCortexLayer, RecurrentLIFCortexLayer
from core.functions import surrogate_spike

class HippocampalIndexLayer(RecurrentLIFCortexLayer):
    """
    Episodic memory hub. High-plasticity recurrent network.
    Uses 'Fast Weights' logic to quickly latch sensory patterns.
    """
    def __init__(self, input_size, num_neurons, beta=0.8, threshold=0.25, noise_std=0.1):
        # Lower beta (0.8) and lower threshold (0.5) to capture fleeting patterns
        super().__init__(input_size, num_neurons, beta=beta, threshold=threshold, noise_std=noise_std)
        # Fast Weight Trace: Biologically inspired rapid synaptic shift
        self.fast_weights = tf.Variable(initial_value=tf.zeros_like(self.weights), trainable=False, name="fast_synaptic_trace")

    def forward(self, inputs):
        # Standard recurrent forward logic but we add the 'Fast Weights' contribution
        # This allows the hippocampus to recall patterns that just occurred
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        v_mem = tf.tile(self.v_mem, [batch_size, 1])
        t_state = tf.tile(self.t_state, [batch_size, 1])
        prev_spikes = tf.tile(self.prev_spikes, [batch_size, 1])
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        
        # Combine permanent weights with rapid episodic weights
        effective_ff_weights = (self.weights + self.fast_weights) * self.synaptic_mask
        active_recurrent_weights = self.recurrent_weights * self.recurrent_synaptic_mask

        flat_inputs = tf.reshape(inputs, [-1, self.input_size])
        all_ff_projections = tf.matmul(flat_inputs, effective_ff_weights) + self.biases
        all_ff_projections = tf.reshape(all_ff_projections, [batch_size, time_steps, self.num_neurons])

        for t in tf.range(time_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (v_mem, tf.TensorShape([None, self.num_neurons])),
                    (t_state, tf.TensorShape([None, self.num_neurons])),
                    (prev_spikes, tf.TensorShape([None, self.num_neurons]))
                ]
            )
            ff_in = all_ff_projections[:, t, :]
            rec_in = tf.matmul(prev_spikes, active_recurrent_weights)
            current_input = ff_in + rec_in + self.biases
            
            v_mem = self.beta * v_mem + current_input
            spikes = surrogate_spike(v_mem, t_state)
            
            # Reset and Fatigue (Refractory)
            v_mem = v_mem - (t_state * spikes * 3.5)
            # Increased fatigue from 1.5 to 3.0 to prevent seizure latching (Equilibrium Patch)
            t_state = t_state + (spikes * 3.0) 
            t_state = self.threshold + (t_state - self.threshold) * 0.95

            prev_spikes = spikes
            spike_trains = spike_trains.write(t, spikes)

            # --- ONE-SHOT FAST PLASTICITY (Episodic Latching) ---
            # Biologically approximated STDP that happens IN-inference
            # This 'latches' the pattern into the fast_weights buffer
            if t > 0:
                outer = tf.matmul(tf.transpose(tf.reduce_mean(inputs[:, t-1:t, :], axis=0)), tf.reduce_mean(spikes, axis=0, keepdims=True))
                self.fast_weights.assign(self.fast_weights * 0.99 + outer * 0.01)

        stacked_spikes = spike_trains.stack()
        final_spikes = tf.transpose(stacked_spikes, perm=[1, 0, 2])
        
        self.v_mem.assign(v_mem[:1, :])
        self.prev_spikes.assign(prev_spikes[:1, :])
        self.t_state.assign(t_state[:1, :])
        
        return final_spikes

class GatedStriatalLayer(LIFCortexLayer):
    """
    Basal Ganglia core. Implements action gating / winner-take-all shushing.
    Only allows 'confident' motor spikes to reach the periphery.
    """
    def __init__(self, input_size, num_neurons, threshold=1.0):
        super().__init__(input_size, num_neurons, threshold=threshold)
        # Inhibitory Gating State (Global Shush)
        self.gate_threshold = tf.Variable(0.1, trainable=False, name="action_gate_barrier")

    def forward(self, inputs):
        # This layer acts as a 'Cleaner'. It suppresses all spikes below a 
        # relative competition threshold.
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        v_mem = tf.tile(self.v_mem, [batch_size, 1])
        t_state = tf.tile(self.t_state, [batch_size, 1])
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        
        active_weights = self.weights * self.synaptic_mask
        flat_inputs = tf.reshape(inputs, [-1, self.input_size])
        all_projections = tf.matmul(flat_inputs, active_weights) + self.biases
        all_projections = tf.reshape(all_projections, [batch_size, time_steps, self.num_neurons])

        # Global Competitive Inhibition: Calculate noise floor across all neurons
        # This is the 'Substantia Nigra' signal
        noise_floor = tf.reduce_mean(all_projections, axis=[1, 2], keepdims=True) * self.gate_threshold

        for t in tf.range(time_steps):
            current_input = all_projections[:, t, :]
            
            # --- COMPETITIVE GATING ---
            # Shush all signals that are weaker than the noise_floor
            gated_input = tf.where(current_input > noise_floor[:, 0, :], current_input, tf.zeros_like(current_input))
            
            v_mem = self.beta * v_mem + gated_input
            spikes = surrogate_spike(v_mem, t_state)
            
            v_mem = v_mem - (t_state * spikes * 5.0) # Even deeper reset for clear action selection
            t_state = t_state + (spikes * 2.0)
            t_state = self.threshold + (t_state - self.threshold) * 0.9

            spike_trains = spike_trains.write(t, spikes)

        stacked_spikes = spike_trains.stack()
        final_spikes = tf.transpose(stacked_spikes, perm=[1, 0, 2])
        
        self.v_mem.assign(v_mem[:1, :])
        self.t_state.assign(t_state[:1, :])
        
        return final_spikes

class SaliencyAmygdalaLayer(LIFCortexLayer):
    """
    Monitors sensory tension. Outputs a 'Fear' signal to the trainer.
    Does not allow signals back into the connective path, just monitors.
    """
    def __init__(self, input_size, num_neurons=64):
        super().__init__(input_size, num_neurons, threshold=1.0, beta=0.5)
        # v4.4 Gradient Stabilization: Since Amygdala output is discarded in current 
        # connectome and used only for telemetry, we mark it as non-trainable 
        # to prevent 'Gradient Void' warnings.
        self.weights.assign(self.weights.read_value()) # Force Variable
        self.weights = tf.Variable(self.weights, trainable=False, name="amygdala_synapses")
        self.biases = tf.Variable(self.biases, trainable=False, name="amygdala_bias")
        self.saliency_state = tf.Variable(0.05, trainable=False, name="saliency_fear_level")
        self.baseline_fear = 0.05

    def get_variables(self):
        # Amygdala is an observer; it does not participate in global backprop
        return []

    def forward(self, inputs):
        # processes sensory input to calculate global 'Tension'
        # v4.5: FORMAL GRADIENT SHIELD - disconnect from training tape
        inputs = tf.stop_gradient(inputs)
        
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        # Standard forward pass to generate internal amygdala spikes
        v_mem = tf.tile(self.v_mem, [batch_size, 1])
        t_state = tf.tile(self.t_state, [batch_size, 1])
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        
        # We use a copy of inputs to calculate saliency
        flat_inputs = tf.reshape(inputs, [-1, self.input_size])
        all_projections = tf.matmul(flat_inputs, self.weights) + self.biases
        
        # Leaky Integrator for Saliency: Fear now has temporal 'weight'
        # Decays by 10% per step, increases based on raw activation
        current_saliency = tf.reduce_mean(tf.abs(all_projections))
        new_saliency = (self.saliency_state * 0.9) + (current_saliency * 0.1)
        self.saliency_state.assign(new_saliency)
        all_projections = tf.reshape(all_projections, [batch_size, time_steps, self.num_neurons])

        for t in tf.range(time_steps):
            v_mem = self.beta * v_mem + all_projections[:, t, :]
            spikes = surrogate_spike(v_mem, t_state)
            v_mem = v_mem - (t_state * spikes * 2.0)
            spike_trains = spike_trains.write(t, spikes)

        stacked_spikes = spike_trains.stack()
        
        # Saliency = Average firing rate of the Amygdala
        fear = tf.reduce_mean(stacked_spikes)
        # Update persistent state (baseline_fear + dynamic excitement)
        new_fear = tf.maximum(self.baseline_fear, self.saliency_state * 0.95 + fear * 0.05)
        self.saliency_state.assign(new_fear)
        
        return tf.transpose(stacked_spikes, perm=[1, 0, 2])

class CerebellarSmoothCore(LIFCortexLayer):
    """
    High-density temporal filter. Smoothes jittery broca spikes.
    Biologically equivalent to the Granule Cell layer.
    """
    def __init__(self, input_size, num_neurons=512):
        # High num_neurons (512) for high resolution predictive smoothing
        # Pediatric Surgery: Lowered from 0.2 to 0.1 to facilitate babbling.
        super().__init__(input_size, num_neurons, threshold=0.1, beta=0.99) # High beta for 'smearing' over time

    def forward(self, inputs):
        # This layer acts as a 'moving average' on spikes
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        v_mem = tf.tile(self.v_mem, [batch_size, 1])
        t_state = tf.tile(self.t_state, [batch_size, 1])
        spike_trains = tf.TensorArray(tf.float32, size=time_steps)
        
        active_weights = self.weights * self.synaptic_mask
        flat_inputs = tf.reshape(inputs, [-1, self.input_size])
        all_projections = tf.matmul(flat_inputs, active_weights) + self.biases
        all_projections = tf.reshape(all_projections, [batch_size, time_steps, self.num_neurons])

        # Temporal Integration pass
        for t in tf.range(time_steps):
            # The Cerebellum preserves the signal longer (beta 0.99)
            v_mem = self.beta * v_mem + all_projections[:, t, :]
            spikes = surrogate_spike(v_mem, t_state)
            
            # Refractory period is SHORT to allow high frequency smoothing
            v_mem = v_mem - (t_state * spikes * 0.8)
            t_state = t_state + (spikes * 0.1)
            t_state = self.threshold + (t_state - self.threshold) * 0.9

            spike_trains = spike_trains.write(t, spikes)

        stacked_spikes = spike_trains.stack()
        return tf.transpose(stacked_spikes, perm=[1, 0, 2])
