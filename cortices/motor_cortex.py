from core.neuron import LIFCortexLayer, SubCortexNetwork, DeconvLIFCortexLayer
import tensorflow as tf

class VisualMotorCortex(SubCortexNetwork):
    """
    Biological Motor/Decoder sequence.
    Receives highly abstract dense concepts from the Prefrontal Cortex and recursively unpacks them spatially 
    across widening geometric layers until physically flashing a High-Definition Retina grid.
    """
    def __init__(self, executive_dim=512, decode_shape=(128, 128, 3), threshold=1.0, noise_std=0.01, init_stddev=0.1):
        super().__init__(name="Visual Motor Decoder Lobe")
        
        # D1: Abstract Thought Expansion (Bottleneck Reconstruction)
        # We expand the 512 thought-seeds back to 4096 spatial anchors (8x8x64)
        self.add_layer(LIFCortexLayer(executive_dim, 4096, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev))
        
        # D2: Cortical Layer Extrapolation (8x8 -> 16x16)
        self.add_layer(DeconvLIFCortexLayer(input_shape=(8, 8, 64), filters=32, kernel_size=3, stride=2, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev))
        
        # D3: Semantic Expansion (16x16 -> 32x32)
        self.add_layer(DeconvLIFCortexLayer(input_shape=(16, 16, 32), filters=16, kernel_size=3, stride=2, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev))
        
        # D4: HD Physics Saccade (32x32 -> 128x128)
        self.add_layer(DeconvLIFCortexLayer(input_shape=(32, 32, 16), filters=3, kernel_size=5, stride=4, threshold=threshold, noise_std=noise_std, init_stddev=init_stddev))

    def forward(self, x):
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        
        # 1. Project dense thought out to 4096 parameters
        dense_expansion = self.layers[0].forward(x)
        
        # 2. Reshape mathematically to a biological 8x8 grid so transpose convolutions can take over spatially
        spatial_thought = tf.reshape(dense_expansion, [batch_size, time_steps, 8, 8, 64])
        
        # 3. Push through D2, D3, D4 sequential Deconv expansions
        out = self.layers[1].forward(spatial_thought)
        out = self.layers[2].forward(out)
        final_physical_grid = self.layers[3].forward(out)
        
        # Final output shape is [batch, time, height, width, channels(3)] -> Flatten to [batch, time, 49152]
        flat_hd_grid = tf.reshape(final_physical_grid, [batch_size, time_steps, -1])
        return flat_hd_grid
