# Biomimetic Brain Progress - Core Upgrades

### Architecture V0.0.1 -> V0.0.2

1. **Connectome Rewiring**: Re-routed the Visual Motor Strip to draw natively from the Prefrontal Cortex (256-dim) rather than the Visual thought stream, forcing true cross-modal generational inference rather than copying the sensory input directly.

2. **Persistence Upgrades**: Implemented memory trace tracking. `save_weights()` and `load_weights()` inject standard `.npy` numpy logic so the SNN trains conditionally upon previous sessions without rebooting biological matrices.

3. **Cross-Modal Sensory Dropout**: Configured `trainer.py` to randomly enforce a 50% Visual Dropout during training epochs. This forcibly prevents the Model from memorizing pixels as an autoencoder, requiring it to "hallucinate" the missing blanks based purely on textual correlations. 

4. **Retinotopic Connectivity (Spatial Convolutions)**: Upgraded `VisualCortex` and `neuron.py` by engineering the `ConvLIFCortexLayer`. Replaces purely arbitrary flat line dense mathematics with biological overlapping 2D geometric matrices. Permits explicit physical feature tracking (like recognizing "where" the top of a face exists).

5. **Thalamic Bottlenecking (Biological Latent Compression)**: Scaled the structural geometry of the Visual pathway to enforce an aggressive bottleneck at the Parietal integration layer, funneling 3072 spatial geometry dimensions into 128 semantic dimensions allowing for immense spatial scaling without memory burnouts.

6. **Top-Down Recurrent Attention (Binding Problem resolved)**: Developed the `RecurrentLIFCortexLayer` and injected it into both Parietal Hub and Prefrontal Cortex. Over 15 time steps, these layers dynamically feedback onto themselves contextually, physically binding semantic adjectives ("Blonde") tightly with mapped spatial entities ("Hair") so physical features stop bleeding into the wrong geographic objects. 

7. **Temporal Integration & Video Streaming**: Rewrote Tokenization routing sequences. Images are no longer frozen across all 15 time-steps randomly just to satisfy SNN dimensions. The Thalamic inputs correctly map chronological frames natively into biological sequential steps, allowing actual physics and temporal motion calculations. Fully untethered training loop allows alternating video/audio/image batches continuously!

8. **Neuromuscular Video Upscaling**: Added native Nearest-Neighbor upscaling directly into `motor_decoder.py` just before physical file generation to completely prevent MP4 codecs from corrupting video headers when presented with the extremely rigid 32x32 dimensional boundaries of biological arrays.

9. **Structural Architect Replication**: Replaced basic Numpy weight extraction with `@classmethod load_model` and `save_model`. The brain now natively dumps a configuration JSON describing its topologies directly alongside its genetic weights into `biological_model/`, completely preventing tensor shape mismatches and crashes when executing heavily evolved topologies.

### Architecture V0.0.2 -> V0.0.3

10. **Synaptic Pruning**: Injected `trainable=False` `synaptic_masks` into all `core/neuron.py` matrices. Added a Deep Sleep cycle to the training epoch loop that permanently severs weak/silent branches mathematically, forcing intense sparsity and noise reduction.

11. **Hebbian Structural Neuroplasticity (Spawning)**: Added native `grow()` methods to all neuron geometries. Employs "cells that fire together, wire together" logic to mathematically resurrect and spawn new physical synaptic ties globally if the textual arrays and visual arrays both detect high simultaneous firing demand despite pruned connections.

12. **Lateral Spatial Inhibition**: Eradicated generated "color bleed" and "ghosting" by injecting negative `inhibition_pools` into the membrane potential loop natively. When a pixel spikes, it geographically subtracts electrical voltage from its physical grid-neighbors. Forces extremely sharp, definitive boundaries (acts as a biological contrast filter).

13. **AutoGraph Eligibility Traces**: Resolved massive C++ dimension-tracking crashes during `tf.function` compilations by removing out-of-scope python arrays. Rewrote Hebbian plasticity tracking into native `tf.Variable` Eligibility Traces `self.hebbian_trace` that passively accumulate during inference smoothly without breaking boundary logic!

14. **Motor Retinal Persistence**: Scaled generation latency buffers. Added native Phosphor Decay `cv2.addWeighted()` into the `MotorDecoder`. Prevented biological computational delays (which took 14 frames for the thought to reach the visual motor strip) from generating completely pitch black glitch videos by dynamically extending generation bounds to 75 Frames and bleeding light mathematically over time!

15. **Maximum Exposure Flat Decoding**: Replaced standard averaging (`reduce_mean`) with `reduce_max` accumulation when translating temporal spikes down into a flat 2D `.png` image, completely eradicating pitch-black images caused mathematically by temporal sparsity dilution.
