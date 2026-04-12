import tensorflow as tf

class BrainTrainer:
    """
    Uses TensorFlow strictly as a math execution engine over SNN sequences.
    Implemented Biological Predictive Coding.
    """
    def __init__(self, brain_connectome):
        self.brain = brain_connectome
        self.optimizer = tf.optimizers.Adam(learning_rate=0.0003)

    @tf.function
    def train_predictive_step(self, visual_train_t, auditory_train_t, target_speech_spikes_t_plus_1, bio_train_mode=False):
        """
        Sensory input at time T is used to predict the speech spike train at time T+1.
        target_speech_spikes_t_plus_1: [batch, time_steps, auditory_dim]
            A sequential one-hot spike train where each time step holds the spike
            for exactly one character — the biologically correct motor target.
        """
        if bio_train_mode:
            # === TRUE BIOTRAIN (NO BACKPROP, STDP ONLY) ===
            brocas_spikes, visual_spikes = self.brain.forward(visual_train_t, auditory_train_t)

            # POSITIVE-WEIGHTED TEMPORAL LOSS:
            # Reduced to 10.0 to lower the "pressure to scream" once awake.
            pos_weight = 10.0
            weighted_mask = 1.0 + target_speech_spikes_t_plus_1 * (pos_weight - 1.0)
            text_loss = tf.reduce_mean(weighted_mask * tf.square(brocas_spikes - target_speech_spikes_t_plus_1))
            # Visual auto-encoding loss
            visual_loss = tf.reduce_mean(tf.square(
                tf.reduce_mean(visual_spikes, axis=1) - tf.reduce_mean(visual_train_t, axis=1)
            ))
            # BIOMIMETIC METABOLIC COST (ATP BUDGET)
            # Cost increased to 1.0 to aggressively quench the "Percent-Scream."
            global_activity = (tf.reduce_mean(brocas_spikes) + tf.reduce_mean(visual_spikes)) / 2.0
            total_loss = text_loss + visual_loss + (global_activity * 1.0)
            
            # Autonomic biological updates without derivatives
            self.brain.update_hebbian_traces()
            self.brain.apply_stdp(learning_rate=1e-4)

            return total_loss, brocas_spikes, visual_spikes
        else:
            with tf.GradientTape() as tape:
                # Force cross-modal learning by randomly blanking vision 50% of the time.
                # This teaches the PFC to reconstruct language using auditory text spikes alone.
                drop_vision = tf.random.uniform([]) < 0.5
                forward_visual = tf.cond(drop_vision, lambda: tf.zeros_like(visual_train_t), lambda: visual_train_t)

                brocas_spikes, visual_spikes = self.brain.forward(forward_visual, auditory_train_t)

                # POSITIVE-WEIGHTED TEMPORAL SEQUENCE LOSS:
                # Reduced for maturity.
                pos_weight = 10.0
                weighted_mask = 1.0 + target_speech_spikes_t_plus_1 * (pos_weight - 1.0)
                text_loss = tf.reduce_mean(weighted_mask * tf.square(brocas_spikes - target_speech_spikes_t_plus_1))

                # Auto-encoding visual loss
                visual_loss = tf.reduce_mean(tf.square(
                    tf.reduce_mean(visual_spikes, axis=1) - tf.reduce_mean(visual_train_t, axis=1)
                ))

                # BIOMIMETIC METABOLIC COST (ATP BUDGET)
                # Adds counter-pressure to noise, rewarding sparsity and precision.
                global_activity = (tf.reduce_mean(brocas_spikes) + tf.reduce_mean(visual_spikes)) / 2.0
                total_loss = text_loss + visual_loss + (global_activity * 1.0)

            trainable_vars = self.brain.get_variables()
            gradients = tape.gradient(total_loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Decoupled plasticity: update Hebbian traces outside the inference loop
            self.brain.update_hebbian_traces()

            return total_loss, brocas_spikes, visual_spikes
