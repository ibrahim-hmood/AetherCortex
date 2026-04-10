import tensorflow as tf
import numpy as np
import os
from cortices.visual_cortex import VisualCortex
from cortices.auditory_language import AuditoryLanguageCortex
from cortices.executive import ExecutiveFrontalCortex
from core.neuron import LIFCortexLayer, RecurrentLIFCortexLayer

class BrainConnectome:
    """
    Connects the individual SNNs into a complete biological system.
    Signals flow as spike trains [batch, time_steps, features].
    """
    def __init__(self, visual_input_dim=3072, auditory_input_dim=256):
        # We assume 32x32x3 spatial geometry for visual inputs natively bridging the thalamus
        self.visual_cortex = VisualCortex(input_shape=(32, 32, 3), v3_dim=128)
        self.auditory_cortex = AuditoryLanguageCortex(auditory_dim=auditory_input_dim, internal_dim=128, executive_dim=256)
        
        # Integration point (Parietal Hub) - Uses Recurrent Attention to bind traits
        self.integration_layer = RecurrentLIFCortexLayer(128 + 128, 256)
        self.prefrontal_cortex = ExecutiveFrontalCortex(combined_sensory_dim=256, cognitive_dim=512)
        
        # Visual Motor Strip: Maps Prefrontal thought (256) back identically to visual sensory width (1024 or 3072 depending on main init)
        self.visual_motor_strip = LIFCortexLayer(256, visual_input_dim)

    def forward(self, processed_vision_train, processed_audio_train):
        """
        Forward pass propagates spike trains across all timesteps implicitly.
        Returns both the motor speech train (Broca's) and the visual comprehension train (V3 output).
        """
        visual_thought = self.visual_cortex.forward(processed_vision_train)
        audio_thought = self.auditory_cortex.process_comprehension(processed_audio_train)
        
        # Parietal integration
        combined = tf.concat([visual_thought, audio_thought], axis=-1)
        integrated = self.integration_layer.forward(combined)
        
        executive_decision = self.prefrontal_cortex.forward(integrated)
        
        # Generate language response
        response_spikes_broca = self.auditory_cortex.process_generation_prep(executive_decision)
        
        # Generate visual imagination (Reverse projection from executive decision)
        imagined_visual_spikes = self.visual_motor_strip.forward(executive_decision)
        
        return response_spikes_broca, imagined_visual_spikes

    def get_variables(self):
        return (self.visual_cortex.get_variables() + 
                self.auditory_cortex.get_variables() + 
                self.integration_layer.get_variables() +
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
            "visual_input_dim": 3072,
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
        brain = cls(visual_input_dim=config.get("visual_input_dim", 3072), 
                    auditory_input_dim=config.get("auditory_input_dim", 256))
                    
        # Inject the physical synaptic bloodlines into the skeleton
        brain.load_weights(weights_path)
        return brain
