from core.neuron import LIFCortexLayer, SubCortexNetwork, ConvLIFCortexLayer

class VisualCortex(SubCortexNetwork):
    """
    Occipital Lobe equivalent based on Spiking Neural Networks.
    Processes visual spike trains from the thalamus.
    """
    def __init__(self, input_shape=(64, 64, 3), v3_dim=128):
        super().__init__(name="Visual/Occipital Lobe")
        
        # Foveal Saccade (Retinal Bottleneck compression)
        # Input HD 64x64x3 -> Compress instantly to 16x16x16 via Spatial Pooling Stride 4
        self.add_layer(ConvLIFCortexLayer(input_shape=input_shape, filters=16, kernel_size=7, stride=4))
        
        # V1: Primary visual cortex (Spatial Edge detection)
        # Input 16x16x16 -> 8x8x32
        self.add_layer(ConvLIFCortexLayer(input_shape=(16, 16, 16), filters=32, kernel_size=5, stride=2))
        
        # V2: Secondary visual cortex (Geometric abstraction)
        # Input 8x8x32 -> 4x4x64
        self.add_layer(ConvLIFCortexLayer(input_shape=(8, 8, 32), filters=64, kernel_size=3, stride=2))
        
        # V3: Higher level visual processing / Latent Thalamic Bottleneck
        # Dense projection from 1024 to semantic 128
        self.add_layer(LIFCortexLayer(1024, v3_dim))
