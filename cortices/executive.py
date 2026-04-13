from core.neuron import LIFCortexLayer, SubCortexNetwork, RecurrentLIFCortexLayer

class ExecutiveFrontalCortex(SubCortexNetwork):
    """
    Prefrontal Cortex analogue using Spiking Neural Networks. 
    Integrates spike trains from all other cortices over time to form a 'thought'.
    """
    def __init__(self, combined_sensory_dim=256, cognitive_dim=512, threshold=1.0, noise_std=0.01, init_stddev=0.1, facilitation=False):
        super().__init__(name="Prefrontal Cortex")
        
        # Assimilating multi-modal perception
        self.add_layer(LIFCortexLayer(combined_sensory_dim, cognitive_dim, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev, facilitation=facilitation))
        
        # High-level decision making (With Attention/Recurrent loops)
        self.add_layer(RecurrentLIFCortexLayer(cognitive_dim, cognitive_dim, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev, facilitation=facilitation))
        
        self.add_layer(LIFCortexLayer(cognitive_dim, cognitive_dim, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev, facilitation=facilitation))
