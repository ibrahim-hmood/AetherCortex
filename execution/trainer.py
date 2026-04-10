import tensorflow as tf

class BrainTrainer:
    """
    Uses TensorFlow strictly as a math execution engine over SNN sequences.
    Implemented Biological Predictive Coding.
    """
    def __init__(self, brain_connectome):
        self.brain = brain_connectome
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def train_predictive_step(self, visual_train_t, auditory_train_t, target_text_rates_t_plus_1):
        """
        Sensory input at time T is used to predict the text/response at time T+1.
        """
        with tf.GradientTape() as tape:
            # Force cross-modal learning by randomly dropping out the biological visual input 50% of the time.
            # This teaches the prefrontal cortex to reconstruct images strictly using auditory text spikes.
            drop_vision = tf.random.uniform([]) < 0.5
            forward_visual = tf.cond(drop_vision, lambda: tf.zeros_like(visual_train_t), lambda: visual_train_t)
            
            brocas_spikes, visual_spikes = self.brain.forward(forward_visual, auditory_train_t)
            
            # Predict the rates for linguistic output
            predicted_text_rates = tf.reduce_mean(brocas_spikes, axis=1) # [batch, features]
            predicted_visual_rates = tf.reduce_mean(visual_spikes, axis=1) # [batch, features]
            
            # Loss against reality at time T+1 (Text)
            text_loss = tf.reduce_mean(tf.square(predicted_text_rates - target_text_rates_t_plus_1))
            
            # Auto-encoding visual loss (Calculate against the TRUE visual input, so the brain learns to hallucinate it)
            target_visual_rates = tf.reduce_mean(visual_train_t, axis=1)
            visual_loss = tf.reduce_mean(tf.square(predicted_visual_rates - target_visual_rates))
            
            total_loss = text_loss + visual_loss
            
        trainable_vars = self.brain.get_variables()
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return total_loss, brocas_spikes, visual_spikes
