from core.neuron import LIFCortexLayer, RecurrentLIFCortexLayer, SubCortexNetwork

class TemporalLobe:
    """
    Temporal Lobe: The seat of auditory processing and semantic comprehension.
    Equivalent to the Superior Temporal Gyrus.
    """
    def __init__(self, auditory_dim=300, internal_dim=512, threshold=1.0, noise_std=0.01, init_stddev=0.1, persistence=1.0, facilitation=False):
        # A1: Primary Auditory Cortex (Pure frequency detection)
        self.primary_auditory = SubCortexNetwork(name="Primary Auditory Cortex")
        self.primary_auditory.add_layer(LIFCortexLayer(auditory_dim, internal_dim, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev, persistence=persistence, facilitation=facilitation))
        
        # Wernicke's Area: High-level comprehension & semantic feature extraction
        # UPGRADE v4.0: Recurrent Splicing for semantic persistence
        self.wernickes_area = SubCortexNetwork(name="Wernicke's Area")
        self.wernickes_area.add_layer(RecurrentLIFCortexLayer(internal_dim, internal_dim, threshold=threshold * 0.9, noise_std=noise_std, init_stddev=init_stddev, facilitation=True))

    def process_comprehension(self, auditory_input_spike_train, feedback_input=None, habituation_gain=0.85):
        """Processes sound into semantic thoughts for the Parietal Hub"""
        # Inject self-talk feedback if present
        if feedback_input is not None:
             auditory_input_spike_train = auditory_input_spike_train + feedback_input
             
        a1_out = self.primary_auditory.forward(auditory_input_spike_train, habituation_gain=habituation_gain)
        comprehension_out = self.wernickes_area.forward(a1_out, habituation_gain=habituation_gain)
        return comprehension_out

    def reset_state(self):
        self.primary_auditory.reset_state()
        self.wernickes_area.reset_state()

    def get_variables(self):
        return self.primary_auditory.get_variables() + self.wernickes_area.get_variables()

    @property
    def layers(self):
        """Exposes all internal sub-layers for global connectome traversal."""
        return self.primary_auditory.layers + self.wernickes_area.layers
