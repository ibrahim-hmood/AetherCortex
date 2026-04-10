from core.neuron import LIFCortexLayer, SubCortexNetwork, DeconvLIFCortexLayer
import tensorflow as tf

class VisualMotorCortex(SubCortexNetwork):
    """
    Biological Motor/Decoder sequence.
    Receives highly abstract dense concepts from the Prefrontal Cortex and recursively unpacks them spatially 
    across widening geometric layers until physically flashing a High-Definition Retina grid.
    """
    def __init__(self, semantic_input_dim=256, target_hd_shape=(64, 64, 3)):
        super().__init__(name="Visual Motor Decoder Lobe")
        
        # D1: Abstract Thought Expansion
        # Input: 256 dense -> LIF -> 1024 (which is interpreted as a 4x4x64 grid).
        self.add_layer(LIFCortexLayer(semantic_input_dim, 1024))
        
        # D2: Cortical Layer Extrapolation (4x4 -> 8x8)
        # We manually structure the flat output into a 4x4x64 block internally inside the connectome natively.
        self.add_layer(DeconvLIFCortexLayer(input_shape=(4, 4, 64), filters=32, kernel_size=3, stride=2))
        
        # D3: Semantic Expansion (8x8 -> 16x16)
        self.add_layer(DeconvLIFCortexLayer(input_shape=(8, 8, 32), filters=16, kernel_size=3, stride=2))
        
        # D4: HD Physics Saccade (16x16 -> 64x64)
        # Brutally explodes the 16x16 grid natively into the 64x64 physical boundary.
        self.add_layer(DeconvLIFCortexLayer(input_shape=(16, 16, 16), filters=3, kernel_size=5, stride=4))

    def forward(self, x):
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        
        # 1. Project dense thought out to 4096 parameters
        dense_expansion = self.layers[0].forward(x)
        
        # 2. Reshape mathematically to a biological 4x4 grid so transpose convolutions can take over spatially
        spatial_thought = tf.reshape(dense_expansion, [batch_size, time_steps, 1024])
        
        # 3. Push through D2, D3, D4 sequential Deconv expansions
        out = self.layers[1].forward(spatial_thought)
        out = self.layers[2].forward(out)
        final_physical_grid = self.layers[3].forward(out)
        
        # Final output shape is [batch, time, height, width, channels(3)] -> Flatten to [batch, time, 49152]
        flat_hd_grid = tf.reshape(final_physical_grid, [batch_size, time_steps, -1])
        return flat_hd_grid
