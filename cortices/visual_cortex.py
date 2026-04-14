from core.neuron import LIFCortexLayer, SubCortexNetwork, ConvLIFCortexLayer

class VisualCortex(SubCortexNetwork):
    """
    Occipital Lobe equivalent based on Spiking Neural Networks.
    Processes visual spike trains from the thalamus.
    """
    def __init__(self, input_shape=(128, 128, 3), v3_dim=128, threshold=1.0, noise_std=0.01, init_stddev=0.1, facilitation=False):
        super().__init__(name="Visual/Occipital Lobe")
        
        # Foveal Saccade (Retinal Bottleneck compression)
        # 128x128 -> 32x32
        self.add_layer(ConvLIFCortexLayer(input_shape=input_shape, filters=16, kernel_size=7, stride=4, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev, facilitation=facilitation))
        
        # V1: Primary visual cortex (Spatial Edge detection)
        # 32x32 -> 16x16
        self.add_layer(ConvLIFCortexLayer(input_shape=(32, 32, 16), filters=32, kernel_size=5, stride=2, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev, facilitation=facilitation))
        
        # V2: Secondary visual cortex (Geometric abstraction)
        # 16x16 -> 8x8
        self.add_layer(ConvLIFCortexLayer(input_shape=(16, 16, 32), filters=64, kernel_size=3, stride=2, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev, facilitation=facilitation))
        
        # V3: Higher level visual processing / Latent Thalamic Bottleneck
        # 8*8*64 = 4096
        self.add_layer(LIFCortexLayer(4096, v3_dim, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev, facilitation=facilitation))
