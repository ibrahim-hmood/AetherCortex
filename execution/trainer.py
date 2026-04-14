import tensorflow as tf
import numpy as np

class BrainTrainer:
    """
    Uses TensorFlow strictly as a math execution engine over SNN sequences.
    Implemented Biological Predictive Coding with Homeostatic Regulation.
    """
    def __init__(self, brain_connectome):
        self.brain = brain_connectome
        self.base_lr = 0.0003
        self.optimizer = tf.optimizers.Adam(learning_rate=self.base_lr)
        
        # HOMEOSTATIC PARAMETERS: Evolving based on neural behavior
        self.metabolic_cost = tf.Variable(0.2, trainable=False, name="metabolic_cost")
        self.pos_weight = tf.Variable(30.0, trainable=False, name="linguistic_reward")
        self.focus_word = "" # v4.2 Focused Curriculum
        self.current_activity = tf.Variable(0.0, trainable=False, name="spike_density")
        
        # History for stagnation detection
        self.prev_epoch_loss = tf.Variable(0.0, trainable=False)
        self.prev_loss = tf.Variable(1.0, trainable=False)
        self.stagnation_counter = tf.Variable(0, trainable=False)

        # v4.4: Validation Metrics (Real-Time Fractions)
        self.total_validations = tf.Variable(0, dtype=tf.int64, trainable=False, name="total_validations")
        self.good_validations = tf.Variable(0, dtype=tf.int64, trainable=False, name="good_validations")
        self.bad_validations = tf.Variable(0, dtype=tf.int64, trainable=False, name="bad_validations")
        
        # v4.5: Atomic Vocabulary Mastery (Word-level permanence tracking)
        # managed in eagar mode to avoid tf.function string limitations
        self.vocab_mastery = {} 

    def update_homeostasis(self, epoch_loss, regional_activity=None):
        """
        REGIONAL HOMEOSTASIS (v2.0: Synaptic Surgery Patch)
        Performs independent metabolic regulation for each cortical hub.
        """
        # --- 1. GLOBAL METABOLIC AUTO-REGULATION (Legacy fallback) ---
        global_activity = float(self.current_activity.numpy())
        
        # Sense if the brain is "ignoring" the tax
        activity_delta = abs(global_activity - getattr(self, 'prev_activity', 0.0))
        self.prev_activity = global_activity
        
        # Define base metabolic trend
        if global_activity > 0.08: self.metabolic_cost.assign_add(0.1)
        elif global_activity < 0.03: self.metabolic_cost.assign_sub(0.05)
        self.metabolic_cost.assign(tf.clip_by_value(self.metabolic_cost, 0.1, 10.0))

        # --- 2. THALAMIC REGULATION (v3.0 Homeostatic Autonomy) ---
        # We no longer perform 'surgery' from the trainer.
        # The brain now managing its own internal regional pressure.
        pass

        # --- 2. SYNAPTIC COOLING & DEFIBRILLATION (Jumpstart) ---
        # Resolve the Metabolic Paradox: Only apply noise if needed.
        current_noise = self.brain.get_noise_state()
        
        if global_activity < 0.0001:
            # TOTAL SILENCE detected. Apply high noise jumpstart to force first spikes.
            # 0.40 noise is strong enough to reliably cross the default 1.0 threshold.
            print(f">>> [Homeostasis] NEURAL COMA: Defibrillating with 0.40 noise jitter (Current: {global_activity:.6%}%).")
            self.brain.modulate_noise(target_level=0.40)
        elif global_activity < 0.01:
            # Near silence. Maintain moderate jumpstart.
            print(f">>> [Homeostasis] Neural Faint: Maintaining 0.15 noise jitter.")
            self.brain.modulate_noise(target_level=0.15)
        elif global_activity > 0.20:
            # DYNAMIC COOLING: Proportional attenuation based on seizure intensity.
            # Adjusted Infancy Floor: Only cool if hitting > 20% activity.
            attenuation = float(np.exp(-20.0 * (global_activity - 0.20)))
            print(f">>> [Homeostasis] DYNAMIC COOLING: Seizure detected ({global_activity:.2%}). Attenuating noise by factor {attenuation:.4f}.")
            self.brain.modulate_noise(attenuation_factor=attenuation)
        elif global_activity > 0.10:
            # Brain is active. Gradually decay back toward 'Cooling' baseline.
            self.brain.modulate_noise(decay_rate=0.2) # Cool 20% toward baseline per epoch
            
        # --- 3. global_activity STAGNATION DETECTION (The "Adrenaline" Kick) ---
        # Fixed the 'Adrenaline Paradox': Only trigger for silent ruts, not seizures.
        if activity_delta < 0.002 and global_activity < 0.05:
            self.stagnation_counter.assign_add(1)
            print(f"> [Homeostasis] Neural SILENCE Rut Detected ({global_activity:.2%} density is fixed).")
        elif global_activity > 0.50:
            # SEIZURE RESET: If we are in a seizure, kill the adrenaline counter immediately.
            self.stagnation_counter.assign(0)
            self.optimizer.learning_rate.assign(self.base_lr)
            print(">>> [Homeostasis] SEIZURE RESET: Killing adrenaline triggers to stabilize.")
        elif global_activity < 0.07:
            # VICTORY RESET: Only clear the stagnation counter if we actually became sparse.
            self.stagnation_counter.assign(0)
            self.optimizer.learning_rate.assign(self.base_lr)
            if self.pos_weight.numpy() > 30.0:
                self.pos_weight.assign_sub(5.0)

        # TRIGGER ADRENALINE: Only permit if NOT in a seizure state.
        if self.stagnation_counter.numpy() >= 3 and global_activity < 0.20:
            # ADRENALINE KICK: Force weights to shift by spiking LR and motor rewards.
            # This triggers every epoch until the global_activity drops below 7%.
            self.optimizer.learning_rate.assign(0.001)
            self.pos_weight.assign_add(10.0)
            print(f"\n[Homeostasis] ADRENALINE SPIKE: Overriding persistence rut (LR: 0.001, Reward: {self.pos_weight.numpy():.1f})")
        
        # --- 4. REWARD SUPPRESSION (De-motivating the Seizure) ---
        if global_activity > 0.80:
            # If the brain is screaming, we remove all reward motivation.
            print(">>> [Homeostasis] REWARD SUPPRESSION: Starving the seizure of dopamine.")
            self.pos_weight.assign(5.0)
        
        # --- 3. AMYGDALA MODULATION (Saliency-Based Tension) ---
        # Emotional weighting: If the Amygdala detected high saliency (sensory tension),
        # we temporarily increase the metabolic tax to force high-focus sparsity.
        saliency_fear = float(getattr(self.brain.amygdala, 'saliency_state', 0.0))
        tension_multiplier = 1.0 + (saliency_fear * 2.0) # Fear can triple the tax
        
        effective_tax = self.metabolic_cost.numpy() * tension_multiplier
        
        # --- 4. REWARD CURRICULUM ---
        self.pos_weight.assign(tf.clip_by_value(self.pos_weight, 5.0, 100.0))
        self.prev_epoch_loss.assign(epoch_loss)

        print(f"> Internal State | Tax: {effective_tax:.2f} (Fear: {saliency_fear:.2f}) | P-Weight: {self.pos_weight.numpy():.1f} | global_activity: {global_activity:.2%}")

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 30, 49152], dtype=tf.float32), 
        tf.TensorSpec(shape=[None, 30, 300], dtype=tf.float32), 
        tf.TensorSpec(shape=[None, 30, 300], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.bool)
    ])
    def train_predictive_step(self, visual_train_t, auditory_train_t, target_speech_spikes_t_plus_1, bio_train_mode=False):
        # --- GLOBAL MEMBRANE DISCHARGE ---
        # Clear all residual electrical potential before every sentence.
        # This breaks Neural Rigor Mortis / Seizure loops.
        self.brain.reset_state()
        
        if bio_train_mode:
            # v4.5: PURE BIOLOGICAL LEARNING (BackProp Removed)
            # The brain now learns strictly through local R-STDP and homeostatic scaling.
            # There is no global gradient tape tracking the weights in this mode.
            
            # --- DYNAMIC FORWARD PASS ---
            brocas_spikes, visual_spikes, internal_activity, regional_activity = self.brain.forward(visual_train_t, auditory_train_t)
            self.current_activity.assign(internal_activity)
            
            # --- TELEMETRY LOSS (Diagnostic Only) ---
            weighted_mask = 1.0 + target_speech_spikes_t_plus_1 * (self.pos_weight - 1.0)
            text_loss = tf.reduce_mean(weighted_mask * tf.square(brocas_spikes - target_speech_spikes_t_plus_1))
            visual_loss = tf.reduce_mean(tf.square(
                tf.reduce_mean(visual_spikes, axis=1) - tf.reduce_mean(visual_train_t, axis=1)
            ))
            total_loss = text_loss + visual_loss + (internal_activity * self.metabolic_cost)

            # --- LIBMIC DOPAMINE SIGNAL ---
            # Reward is based on loss improvement (Differential Dopamine)
            accuracy = 1.0 / (text_loss + 1e-6)
            baseline = 1.0 / (self.prev_loss + 1e-6)
            dopamine = accuracy / (baseline + 1e-6)
            self.prev_loss.assign(text_loss)
            
            dopamine = tf.maximum(dopamine, 0.25)
            self.brain.dopamine_level.assign(dopamine)
            
            # v4.3 DYNAMIC NEURAL TAXATION
            new_tax = 0.01 + (tf.stop_gradient(total_loss) * 0.2)
            self.metabolic_cost.assign(tf.clip_by_value(new_tax, 0.01, 2.0))
            
            scaled_reward = 10.0 * dopamine 
            self.pos_weight.assign(tf.clip_by_value(scaled_reward, 1.0, 50.0))

            # --- TRACK VALIDATION METRICS ---
            self.total_validations.assign_add(1)
            if dopamine > 1.05: 
                self.good_validations.assign_add(1)
            elif dopamine < 0.80: 
                self.bad_validations.assign_add(1)

            # --- BIOLOGICAL PLASTICITY (R-STDP) ---
            self.brain.update_hebbian_traces()
            self.brain.apply_homeostatic_regulation(regional_activity)
            
            # No Optimizer update here. Learning driven strictly by synaptic traces + dopamine
            self.brain.apply_stdp(learning_rate=1e-4, metabolic_tax=self.metabolic_cost)

            return total_loss, brocas_spikes, visual_spikes, regional_activity
        else:
            # Standard AI Hybrid Training (Maintained for reference/heavy initialization)
            with tf.GradientTape() as tape:
                drop_vision = tf.random.uniform([]) < 0.5
                forward_visual = tf.cond(drop_vision, lambda: tf.zeros_like(visual_train_t), lambda: visual_train_t)

                brocas_spikes, visual_spikes, internal_activity, regional_activity = self.brain.forward(forward_visual, auditory_train_t)
                self.current_activity.assign(internal_activity)
                
                weighted_mask = 1.0 + target_speech_spikes_t_plus_1 * (self.pos_weight - 1.0)
                text_loss = tf.reduce_mean(weighted_mask * tf.square(brocas_spikes - target_speech_spikes_t_plus_1))
                visual_loss = tf.reduce_mean(tf.square(
                    tf.reduce_mean(visual_spikes, axis=1) - tf.reduce_mean(visual_train_t, axis=1)
                ))
                total_loss = text_loss + visual_loss + (internal_activity * self.metabolic_cost)

            trainable_vars = self.brain.get_variables()
            gradients = tape.gradient(total_loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            self.brain.update_hebbian_traces()
            self.brain.apply_homeostatic_regulation(regional_activity)

            return total_loss, brocas_spikes, visual_spikes, regional_activity
            
    def sleep_consolidation(self):
        """
        Deep Sleep Consolidation (v4.0): 
        Offline synaptic pruning and growth phase at the end of every epoch.
        """
        print("\n--- INITIATING DEEP SLEEP (SYNAPTIC CONSOLIDATION) ---")
        # Metabolic baseline restoration during sleep
        self.brain.dopamine_level.assign(1.0)
        
        # 1. Pruning Phase: Sever connections that didn't receive enough dopamine/demand
        pruned_count = self.brain.prune(threshold=0.005)
        
        # 2. Growth Phase: Spawn new synapses in high-demand 'Insight' pathways
        grown_count = self.brain.grow(threshold=0.1)
        
        print(f">> Sleep Complete | Pruned: {int(pruned_count)} | Grown: {int(grown_count)}")
        return pruned_count, grown_count

    def record_word_mastery(self, word, brocas_output):
        """
        v4.5: Neural Fingerprinting.
        Calculates how 'permanent' (myelinated) the connection for a specific 
        word has become by checking the active ensemble in the language area.
        """
        if not word or word == "" or brocas_output is None:
            return
            
        # 1. Identify which neurons fired for this word (mean over time dimension)
        spikes = tf.reduce_mean(brocas_output, axis=1) # [batch, neurons]
        active_mask = tf.cast(spikes > 0.05, tf.float32)
        
        # 2. Grab the permanence of the language area
        # We look at the primary Broca's area layers
        try:
            # permanence is [input_dim, output_dim]. We care about the output neurons'
            # total structural health. We use the synaptic permanence mask.
            perm_matrix = self.brain.frontal_language.brocas_area.layers[0].permanence
            
            # Average permanence of synapses leading TO the active output neurons
            active_permanence = tf.reduce_mean(perm_matrix, axis=0) # [output_dim]
            word_stability = tf.reduce_sum(active_permanence * active_mask) / (tf.reduce_sum(active_mask) + 1e-6)
            
            # Update the global mastery registry (managed eagerly)
            current_score = float(word_stability.numpy())
            
            # Clean the word (strip newlines/spaces)
            clean_word = word.strip().upper()
            if clean_word:
                # Keep the HIGHEST mastery seen (don't decay mastery if it was once locked)
                self.vocab_mastery[clean_word] = max(self.vocab_mastery.get(clean_word, 0.0), current_score)
        except Exception:
            pass
