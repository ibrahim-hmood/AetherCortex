from core.neuron import LIFCortexLayer, SubCortexNetwork

class AuditoryLanguageCortex:
    """
    Temporal Lobe & Language Centers equivalent using Spiking Neural Networks.
    """
    def __init__(self, auditory_dim=256, internal_dim=128, executive_dim=256):
        # A1: Primary Auditory Cortex
        self.primary_auditory = SubCortexNetwork(name="Primary Auditory Cortex")
        self.primary_auditory.add_layer(LIFCortexLayer(auditory_dim, internal_dim))
        
        # Wernicke's Area: Comprehension
        self.wernickes_area = SubCortexNetwork(name="Wernicke's Area")
        self.wernickes_area.add_layer(LIFCortexLayer(internal_dim, internal_dim))

        # Broca's Area: Generation
        self.brocas_area = SubCortexNetwork(name="Broca's Area")
        self.brocas_area.add_layer(LIFCortexLayer(executive_dim, internal_dim))
        # Ensure final output matches the sensory auditory target dimension (256) for predictive coding loss
        self.brocas_area.add_layer(LIFCortexLayer(internal_dim, auditory_dim))

    def process_comprehension(self, auditory_input_spike_train):
        """Forward pass over time from ear -> A1 -> Wernicke's"""
        a1_out = self.primary_auditory.forward(auditory_input_spike_train)
        comprehension_out = self.wernickes_area.forward(a1_out)
        return comprehension_out

    def process_generation_prep(self, thought_spike_train):
        """Converts an abstract thought train into speech/text preparation"""
        return self.brocas_area.forward(thought_spike_train)

    def get_variables(self):
        return (self.primary_auditory.get_variables() + 
                self.wernickes_area.get_variables() + 
                self.brocas_area.get_variables())
