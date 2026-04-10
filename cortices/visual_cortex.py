from core.neuron import LIFCortexLayer, SubCortexNetwork, ConvLIFCortexLayer

class VisualCortex(SubCortexNetwork):
    """
    Occipital Lobe equivalent based on Spiking Neural Networks.
    Processes visual spike trains from the thalamus.
    """
    def __init__(self, input_shape=(32, 32, 3), v3_dim=128):
        super().__init__(name="Visual/Occipital Lobe")
        
        # V1: Primary visual cortex (Spatial Edge detection)
        # Input 32x32x3 (3072) -> 16x16x16 (4096)
        self.add_layer(ConvLIFCortexLayer(input_shape=input_shape, filters=16, kernel_size=5, stride=2))
        
        # V2: Secondary visual cortex (Geometric abstraction)
        # Input 16x16x16 (4096) -> 8x8x32 (2048)
        self.add_layer(ConvLIFCortexLayer(input_shape=(16, 16, 16), filters=32, kernel_size=3, stride=2))
        
        # V3: Higher level visual processing / Latent Thalamic Bottleneck
        # Dense projection from 2048 to semantic 128
        self.add_layer(LIFCortexLayer(2048, v3_dim))
