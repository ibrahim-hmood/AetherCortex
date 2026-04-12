from core.neuron import LIFCortexLayer, SubCortexNetwork

class AuditoryLanguageCortex:
    """
    Temporal Lobe & Language Centers equivalent using Spiking Neural Networks.
    """
    def __init__(self, auditory_dim=256, internal_dim=128, executive_dim=256, threshold=1.0, noise_std=0.01, init_stddev=0.1):
        # A1: Primary Auditory Cortex
        self.primary_auditory = SubCortexNetwork(name="Primary Auditory Cortex")
        self.primary_auditory.add_layer(LIFCortexLayer(auditory_dim, internal_dim, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev))
        
        # Wernicke's Area: Comprehension
        self.wernickes_area = SubCortexNetwork(name="Wernicke's Area")
        self.wernickes_area.add_layer(LIFCortexLayer(internal_dim, internal_dim, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev))

        # Broca's Area: Generation
        self.brocas_area = SubCortexNetwork(name="Broca's Area")
        self.brocas_area.add_layer(LIFCortexLayer(executive_dim, internal_dim, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev))
        # Final motor strip should be slightly more sensitive for "babbling"
        self.brocas_area.add_layer(LIFCortexLayer(internal_dim, auditory_dim, threshold=threshold * 0.8, noise_std=noise_std * 1.5, init_stddev=init_stddev))

    def process_comprehension(self, auditory_input_spike_train):
        """Forward pass over time from ear -> A1 -> Wernicke's"""
        a1_out = self.primary_auditory.forward(auditory_input_spike_train)
        comprehension_out = self.wernickes_area.forward(a1_out)
        return comprehension_out

    def process_generation_prep(self, thought_spike_train):
        """Converts an abstract thought train into speech/text preparation"""
        return self.brocas_area.forward(thought_spike_train)

    def reset_state(self):
        self.primary_auditory.reset_state()
        self.wernickes_area.reset_state()
        self.brocas_area.reset_state()

    def get_variables(self):
        return (self.primary_auditory.get_variables() + 
                self.wernickes_area.get_variables() + 
                self.brocas_area.get_variables())
