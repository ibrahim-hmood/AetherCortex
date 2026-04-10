from core.neuron import LIFCortexLayer, SubCortexNetwork, RecurrentLIFCortexLayer

class ExecutiveFrontalCortex(SubCortexNetwork):
    """
    Prefrontal Cortex analogue using Spiking Neural Networks. 
    Integrates spike trains from all other cortices over time to form a 'thought'.
    """
    def __init__(self, combined_sensory_dim=256, cognitive_dim=512):
        super().__init__(name="Prefrontal Cortex")
        
        # Assimilating multi-modal perception
        self.add_layer(LIFCortexLayer(combined_sensory_dim, cognitive_dim))
        
        # High-level decision making (With Attention/Recurrent loops)
        self.add_layer(RecurrentLIFCortexLayer(cognitive_dim, cognitive_dim))
        
        # Preparing to route back down
        self.add_layer(LIFCortexLayer(cognitive_dim, cognitive_dim // 2))
