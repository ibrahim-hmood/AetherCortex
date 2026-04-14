from core.neuron import LIFCortexLayer, SubCortexNetwork

class FrontalLanguageCortex:
    """
    Broca's Area: Motor Speech Assembly Center.
    Located in the Frontal Lobe (Inferior Frontal Gyrus).
    """
    def __init__(self, semantic_input_dim=512, motor_output_dim=300, threshold=1.0, noise_std=0.01, init_stddev=0.1, persistence=1.0, facilitation=False):
        # Broca's Area: Motor speech assembly & grammatical sequencing
        self.brocas_area = SubCortexNetwork(name="Broca's Area")
        self.brocas_area.add_layer(LIFCortexLayer(semantic_input_dim, motor_output_dim, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev, persistence=persistence, facilitation=facilitation))

    def process_generation_prep(self, gated_thought_train, habituation_gain=0.85):
        """Converts intent into motor speech patterns"""
        return self.brocas_area.forward(gated_thought_train, habituation_gain=habituation_gain)

    def reset_state(self):
        self.brocas_area.reset_state()

    def get_variables(self):
        return self.brocas_area.get_variables()

    @property
    def layers(self):
        """Exposes all internal sub-layers for global connectome traversal."""
        return self.brocas_area.layers
