import tensorflow as tf
import numpy as np
import os
from cortices.visual_cortex import VisualCortex
from cortices.motor_cortex import VisualMotorCortex
from cortices.auditory_language import AuditoryLanguageCortex
from cortices.executive import ExecutiveFrontalCortex
from core.neuron import LIFCortexLayer, RecurrentLIFCortexLayer

class BrainConnectome:
    """
    Connects the individual SNNs into a complete biological system.
    Signals flow as spike trains [batch, time_steps, features].
    """
    def __init__(self, visual_input_dim=12288, auditory_input_dim=256):
        # Global Sensitization Strategy:
        # We use different 'Biological Volumes' for different regions to ensure the
        # signal survives the journey from the Retina to the Tongue (Speech).
        
        # 1. Visual Sensitivity (The Entrance)
        self.visual_cortex = VisualCortex(
            input_shape=(64, 64, 3), 
            v3_dim=128,
            threshold=0.5,
            noise_std=0.05,
            init_stddev=0.15
        )
        
        # 2. Language/Phonological Sensitivity (The Internal Voice)
        # Higher noise (0.15) and lower threshold (0.3) to force 'Babbling'
        language_threshold = 0.3
        language_noise = 0.15
        
        self.auditory_cortex = AuditoryLanguageCortex(
            auditory_dim=auditory_input_dim, 
            internal_dim=128, 
            executive_dim=256,
            threshold=language_threshold,
            noise_std=language_noise,
            init_stddev=0.2
        )
        
        # Integration point (Parietal Hub) - Uses Recurrent Attention to bind traits
        self.integration_layer = RecurrentLIFCortexLayer(128 + 128, 256, threshold=language_threshold, noise_std=language_noise)
        
        # VWFA: Phonological Loop bridge (Visual -> Auditory)
        # Ultra-sensitive (threshold 0.2) to ensure the first few shapes result in sound.
        self.vwfa_bridge = RecurrentLIFCortexLayer(128, 256, threshold=0.2, noise_std=language_noise, init_stddev=0.3)
        
        # 3. Executive Sensitivity (The Gatekeeper)
        # Lowered threshold (0.4) to prevent the PFC from blocking infant signals
        self.prefrontal_cortex = ExecutiveFrontalCortex(
            combined_sensory_dim=256, 
            cognitive_dim=512,
            threshold=0.4,
            noise_std=0.1,
            init_stddev=0.15
        )
        
        # 4. Motor Sensitivity (The Output)
        # Slightly higher threshold (0.6) for stability in physical projection
        self.visual_motor_strip = VisualMotorCortex(
            semantic_input_dim=256, 
            target_hd_shape=(64, 64, 3),
            threshold=0.6,
            noise_std=0.05,
            init_stddev=0.1
        )

    def forward(self, processed_vision_train, processed_audio_train):
        """
        Forward pass propagates spike trains across all timesteps implicitly.
        Returns both the motor speech train (Broca's) and the visual comprehension train (V3 output).
        """
        visual_thought = self.visual_cortex.forward(processed_vision_train)
        
        # BIOMIMETIC READING (VWFA): Bridging Visual shapes into the Phonological Loop
        # This allows the brain to "hear" the text it is seeing.
        internal_voice = self.vwfa_bridge.forward(visual_thought)
        
        # Combine physical external sound with the internal hallucinated voice of the text
        audio_stream = processed_audio_train + internal_voice
        
        audio_thought = self.auditory_cortex.process_comprehension(audio_stream)
        
        # Parietal integration
        combined = tf.concat([visual_thought, audio_thought], axis=-1)
        integrated = self.integration_layer.forward(combined)
        
        executive_decision = self.prefrontal_cortex.forward(integrated)
        
        # Generate language response
        response_spikes_broca = self.auditory_cortex.process_generation_prep(executive_decision)
        
        # Generate visual imagination (Reverse projection from executive decision)
        imagined_visual_spikes = self.visual_motor_strip.forward(executive_decision)
        
        return response_spikes_broca, imagined_visual_spikes

    def apply_plasticity_and_pruning(self, prune_threshold=0.005, grow_threshold=0.1):
        pruned = 0.0
        grown = 0.0
        
        for region in [self.visual_cortex, self.auditory_cortex, self.integration_layer, self.vwfa_bridge, self.prefrontal_cortex, self.visual_motor_strip]:
            if hasattr(region, 'prune'): pruned += region.prune(prune_threshold)
            if hasattr(region, 'grow'): grown += region.grow(grow_threshold)
            
        return pruned, grown

    def update_hebbian_traces(self):
        for region in [self.visual_cortex, self.auditory_cortex, self.integration_layer, self.vwfa_bridge, self.prefrontal_cortex, self.visual_motor_strip]:
            if hasattr(region, 'update_hebbian_trace'):
                region.update_hebbian_trace()

    def apply_stdp(self, learning_rate=1e-4, decay=1e-5):
        for region in [self.visual_cortex, self.auditory_cortex, self.integration_layer, self.vwfa_bridge, self.prefrontal_cortex, self.visual_motor_strip]:
            if hasattr(region, 'apply_stdp'):
                region.apply_stdp(learning_rate, decay)

    def reset_state(self):
        """Clears all neural memory across the entire connectome."""
        self.visual_cortex.reset_state()
        self.auditory_cortex.reset_state()
        self.integration_layer.reset_state()
        self.vwfa_bridge.reset_state()
        self.prefrontal_cortex.reset_state()
        self.visual_motor_strip.reset_state()

    def get_variables(self):
        return (self.visual_cortex.get_variables() + 
                self.auditory_cortex.get_variables() + 
                self.integration_layer.get_variables() +
                self.vwfa_bridge.get_variables() +
                self.prefrontal_cortex.get_variables() +
                self.visual_motor_strip.get_variables())

    def save_weights(self, filepath="brain_weights.npy"):
        vars_list = self.get_variables()
        weights = [v.numpy() for v in vars_list]
        np.save(filepath, np.array(weights, dtype=object), allow_pickle=True)
        print(f"> Biological traces saved to {filepath}")

    def load_weights(self, filepath="brain_weights.npy"):
        if os.path.exists(filepath):
            weights = np.load(filepath, allow_pickle=True)
            vars_list = self.get_variables()
            
            if len(weights) != len(vars_list):
                print(f"> [WARNING] Synaptic length mismatch. Expected {len(vars_list)} blocks, found {len(weights)}. Overwriting with new neural structure.")
                return

            for v, w in zip(vars_list, weights):
                if v.shape != w.shape:
                    print(f"> [WARNING] Parameter shape mismatch detected on {v.name}. Evolved connectome geometry detected. Restarting neural traces from infancy.")
                    return
                v.assign(w)
            print(f"> Biological traces restored from {filepath}")
        else:
            print(f"> No traces found at {filepath}, starting from infancy.")

    def save_model(self, model_dir="biological_model"):
        import json
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        config = {
            "visual_input_dim": 12288,
            "auditory_input_dim": 256
        }
        
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f)
            
        self.save_weights(os.path.join(model_dir, "brain_weights.npy"))
        print(f"> Entire Connectome structure and biological architecture archived to '{model_dir}/'")

    @classmethod
    def load_model(cls, model_dir="biological_model"):
        import json
        config_path = os.path.join(model_dir, "config.json")
        weights_path = os.path.join(model_dir, "brain_weights.npy")
        
        if not os.path.exists(config_path):
            print(f"> [WARNING] Architectural config not found in {model_dir}. Booting from scratch.")
            return cls()
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Natively reconstruct the explicit biological scaffolding
        brain = cls(visual_input_dim=config.get("visual_input_dim", 12288), 
                    auditory_input_dim=config.get("auditory_input_dim", 256))
                    
        # Inject the physical synaptic bloodlines into the skeleton
        brain.load_weights(weights_path)
        return brain
