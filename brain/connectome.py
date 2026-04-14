import tensorflow as tf
import numpy as np
import os
from cortices.visual_cortex import VisualCortex
from cortices.motor_cortex import VisualMotorCortex
from cortices.temporal_lobe import TemporalLobe
from cortices.auditory_language import FrontalLanguageCortex
from core.neuron import LIFCortexLayer, RecurrentLIFCortexLayer

class BrainConnectome:
    """
    Connects the individual SNNs into a complete biological system.
    Signals flow as spike trains [batch, time_steps, features].
    """
    def __init__(self, visual_input_dim=49152, auditory_input_dim=300, threshold=0.05):
        # Global Sensitization Strategy:
        # We use different 'Biological Volumes' for different regions to ensure the
        # signal survives the journey from the Retina to the Tongue (Speech).
        
        # 1. Sensory Centers
        self.visual_cortex = VisualCortex(input_shape=(128, 128, 3), v3_dim=512, threshold=threshold, facilitation=True)
        # Persistence = 0.1 protects the linguistic auditory centers from metabolic famine
        self.temporal_lobe = TemporalLobe(auditory_dim=auditory_input_dim, internal_dim=512, threshold=threshold, persistence=0.1, facilitation=True)
        
        # 2. VWFA: Bridging Visual shapes into the Phonological Loop
        # High persistence ensures the reading bridge survives high-noise periods
        # Pediatric Surgery: Lowered from 0.5 to 0.2 to enable vision-to-audio bridge
        self.vwfa_bridge = LIFCortexLayer(512, auditory_input_dim, threshold=0.2, persistence=0.1, facilitation=True)
        
        # 3. Integration & Executive
        from cortices.subcortical import HippocampalIndexLayer, GatedStriatalLayer, SaliencyAmygdalaLayer, CerebellarSmoothCore
        from cortices.executive import ExecutiveFrontalCortex
        
        self.integration_layer = RecurrentLIFCortexLayer(1024, 1024, threshold=threshold, facilitation=True) # Visual + Audio combined
        self.hippocampus = HippocampalIndexLayer(1024, 1024, threshold=0.1)
        
        # v2.5 Biological Upgrade: Early Infancy Hypersensitivity (0.2 -> 0.05)
        self.prefrontal_cortex = ExecutiveFrontalCortex(combined_sensory_dim=1024, cognitive_dim=512, threshold=0.05, facilitation=True)
        
        # --- REENTRY STATE (Working Memory Loops) ---
        self.prev_pfc_spikes = tf.Variable(tf.zeros((1, 512)), trainable=False, name="feedback_pfc")
        self.prev_broca_spikes = tf.Variable(tf.zeros((1, auditory_input_dim)), trainable=False, name="feedback_broca")

        
        # 4. Sub-cortical Hubs
        self.amygdala = SaliencyAmygdalaLayer(1024, num_neurons=64)
        self.basal_ganglia = GatedStriatalLayer(512, 512, threshold=0.2)
        self.cerebellum = CerebellarSmoothCore(300, 300)
        
        # 5. Motor hubs
        # Broca's Area: Protected from pruning via persistence=0.1
        # Aligned input to PFC output (512) to prevent information loss
        # v2.5 Biological Upgrade: Early Infancy Hypersensitivity (0.2 -> 0.05)
        self.frontal_language = FrontalLanguageCortex(semantic_input_dim=512, motor_output_dim=auditory_input_dim, threshold=0.05, persistence=0.1, facilitation=True)
        self.visual_motor_strip = VisualMotorCortex(executive_dim=512, decode_shape=(128, 128, 3), threshold=threshold)

        # --- LIBMIC REWARD SYSTEM (v4.0 Dopamine Upgrade) ---
        self.dopamine_level = tf.Variable(1.0, trainable=False, name="global_dopamine")
        
        # --- THE NEURAL PUPIL (v0.2.8 Dynamic Thalamic Gating) ---
        # Starts at neutral (0.5). Dilates when silent (down to 0.1), contracts when noisy (up to 0.95).
        self.habituation_pupil = tf.Variable(0.5, trainable=False, name="thalamic_pupil")
        
        # --- REGIONAL METABOLIC MAP (v3.0 Homeostatic Autonomy) ---
        self.regional_threshold_map = {
            "visual": self.visual_cortex.layers[0].threshold_variable,
            "temporal": self.temporal_lobe.primary_auditory.layers[0].threshold_variable,
            "parietal": self.integration_layer.threshold_variable,
            "executive": self.prefrontal_cortex.layers[0].threshold_variable,
            "broca": self.frontal_language.brocas_area.layers[0].threshold_variable,
            "hippocampus": self.hippocampus.threshold_variable,
            "motor_strip": self.visual_motor_strip.layers[0].threshold_variable
        }
        
        # Equilibrium Targets (8-10% is the biological 'sweet spot' for sparse coding)
        self.regional_target_map = {
            "visual": 0.10,
            "temporal": 0.08,
            "parietal": 0.08,
            "executive": 0.05, # Higher sensitivity for executive hubs
            "broca": 0.05,
            "hippocampus": 0.15, # Memory hubs can handle higher excitement
            "motor_strip": 0.10
        }
        
        self.homeostatic_drift_rate = 0.02 # Infancy Boost: Faster sensitivity adaptation

    def forward(self, processed_vision_train, processed_audio_train):
        """
        Forward pass propagates spike trains across all timesteps implicitly.
        Incorporates Thalamic Sensory Gating (TRN): Dampens inputs if activity is high.
        """
        # --- THALAMIC SENSORY GATING (TRN Shell) ---
        last_activity = getattr(self, 'last_spike_density', 0.0)
        
        # 1. Scalar Gating (Gain Control)
        gating_factor = float(np.exp(-10.0 * max(0.0, last_activity - 0.15)))
        
        # 2. Habituation Gating (The "Neural Pupil" v0.2.8)
        # Dilate for silence, contract for noise.
        # v0.2.8: Aggressive recovery for silence; fast contraction for seizures.
        if last_activity > 0.10: 
            self.habituation_pupil.assign_add(0.08) # Faster sunglasses (Protection)
        elif last_activity < 0.02: 
            self.habituation_pupil.assign_sub(0.05) # Faster night-vision (Discovery)
        
        # v0.2.8: Lowered max cap to 0.65 to prevent 'Total Blackout'
        self.habituation_pupil.assign(tf.clip_by_value(self.habituation_pupil, 0.05, 0.65))
        current_gate = self.habituation_pupil.read_value()
        
        # --- REENTRANT FEEDBACK (Top-Down Focus) ---
        # PFC output from previous step influences High-Level Visual Comprehension (Attention/V3)
        visual_thought = self.visual_cortex.forward(processed_vision_train, habituation_gain=current_gate)
        # Apply Top-Down gain to the semantic visual representation
        # v0.2.8: Dampen feedback when pupil is contracted to help break seizures
        feedback_gate = (1.0 - current_gate)
        pfc_feedback = tf.expand_dims(self.prev_pfc_spikes, 1) * (0.05 * feedback_gate)
        visual_thought = visual_thought + pfc_feedback
        
        # BIOMIMETIC READING (VWFA): Bridging Visual shapes into the Phonological Loop
        internal_voice = self.vwfa_bridge.forward(visual_thought, habituation_gain=current_gate)
        
        # Broca's output from previous step is heard by the ears (Phonological Loop)
        # v0.2.8: Dampen feedback when pupil is contracted
        feedback_gate = (1.0 - current_gate)
        broca_feedback = tf.expand_dims(self.prev_broca_spikes, 1) * (0.1 * feedback_gate)
        # Combine physical external sound and internal voice of text
        audio_stream = processed_audio_train + (internal_voice * 0.1) + broca_feedback
        
        # Temporal processing (Comprehension)
        audio_thought = self.temporal_lobe.process_comprehension(audio_stream, habituation_gain=current_gate)
        
        # --- SUB-CORTICAL INTERVENTION (Saliency) ---
        sensory_hub = tf.concat([visual_thought, audio_thought], axis=-1)
        self.amygdala.forward(sensory_hub) # Amygdala is internal, doesn't need the gate
        
        # Parietal integration
        integrated = self.integration_layer.forward(sensory_hub, habituation_gain=current_gate)
        
        # --- SUB-CORTICAL INTERVENTION (Memory) ---
        # Hippocampus uses indexing, bypasses sensory gate
        memory_boosted_integrated = self.hippocampus.forward(integrated)
        
        executive_decision = self.prefrontal_cortex.forward(memory_boosted_integrated, habituation_gain=current_gate)
        
        # --- SUB-CORTICAL INTERVENTION (Action Selection) ---
        # Basal Ganglia gated intent
        gated_executive_intent = self.basal_ganglia.forward(executive_decision)
        
        # Generate language response using the GATED intent
        # --- SYNAPTIC GAIN INJECTION v2.2 ---
        # 1.5x Multiplier to overdrive the Basal Ganglia gate
        # v0.2.8: Broca is an OUTPUT area; it is no longer suppressed by the sensory habituation pupil.
        raw_response_spikes = self.frontal_language.process_generation_prep(gated_executive_intent * 1.5)
        
        # --- SUB-CORTICAL INTERVENTION (Motor Precision) ---
        response_spikes_broca = self.cerebellum.forward(raw_response_spikes)
        
        # --- GLOBAL INTERNAL PROBE ---
        internal_density = (
            tf.reduce_mean(executive_decision) + 
            tf.reduce_mean(visual_thought) + 
            tf.reduce_mean(integrated)
        ) / 3.0
        
        # Generate visual imagination
        imagined_visual_spikes = self.visual_motor_strip.forward(executive_decision)
        
        # --- REGIONAL ACTIVITY MONITORING ---
        regional_activity = {
            "visual": tf.reduce_mean(visual_thought),
            "temporal": tf.reduce_mean(audio_thought),
            "parietal": tf.reduce_mean(integrated),
            "executive": tf.reduce_mean(executive_decision),
            "broca": tf.reduce_mean(response_spikes_broca), 
            "vwfa": tf.reduce_mean(internal_voice),
            "hippocampus": tf.reduce_mean(memory_boosted_integrated),
            "motor_strip": tf.reduce_mean(imagined_visual_spikes),
            "cerebellum": tf.reduce_mean(response_spikes_broca),
            "global": internal_density
        }
        
        # --- UPDATE REENTRY PERSISTENCE ---
        self.prev_pfc_spikes.assign(executive_decision[:1, -1, :])
        self.prev_broca_spikes.assign(response_spikes_broca[:1, -1, :])
        
        return response_spikes_broca, imagined_visual_spikes, internal_density, regional_activity

    def apply_homeostatic_regulation(self, regional_activity):
        """
        Autonomous Thalamic Regulation: Adjusts regional thresholds in real-time.
        If a region is too loud, it becomes less sensitive. 
        If a region is too quiet, it sensitized itself at drift_rate.
        """
        for region, actual_density in regional_activity.items():
            if region == "global": continue
            
            threshold_var = self.regional_threshold_map.get(region)
            target = self.regional_target_map.get(region, 0.08)
            
            if threshold_var is not None:
                # Proportional Drift logic (Using native TF ops for graph stability)
                delta = tf.cast(actual_density, tf.float32) - target
                
                # If too loud (> target), drift threshold UP
                # If too quiet (< target), drift threshold DOWN
                drift = delta * self.homeostatic_drift_rate
                
                # Defensive check: don't let drift happen too fast
                drift = tf.clip_by_value(drift, -0.05, 0.05)
                
                # Apply the drift directly to the variable
                threshold_var.assign_add(drift)
                
                # Biological Safety Clipping (v3.0 Floor: 0.01)
                threshold_var.assign(tf.clip_by_value(threshold_var, 0.01, 3.0))

    def update_hebbian_traces(self):
        """Accumulates synaptic demand based on temporal correlations."""
        for layer in self.all_layers():
            layer.update_hebbian_trace()

    def apply_stdp(self, learning_rate=1e-4, decay=1e-5, metabolic_tax=0.0):
        """ 
        Global synaptic update pass modulated by the current Dopamine state. 
        """
        d_val = self.dopamine_level.read_value()
        for layer in self.all_layers():
            layer.apply_stdp(
                learning_rate=learning_rate, 
                decay=decay, 
                metabolic_tax=metabolic_tax,
                dopamine=d_val
            )

    def prune(self, threshold=0.005):
        """Severs unused synaptic connections across the entire brain."""
        total_pruned = 0
        for layer in self.all_layers():
            total_pruned += layer.prune(threshold=threshold)
        return total_pruned

    def grow(self, threshold=0.1):
        """Spawns new synaptic connections in high-demand pathways."""
        total_grown = 0
        for layer in self.all_layers():
            total_grown += layer.grow(threshold=threshold)
        return total_grown

    def get_retinal_view(self):
        """Retrieves both RAW and GATED V1 inputs for diagnostic comparison."""
        try:
            # The first layer of VisualCortex is our V1/Retina entry point
            entry_layer = self.visual_cortex.layers[0]
            views = {}
            if hasattr(entry_layer, 'last_gated_input'):
                views['gated'] = entry_layer.last_gated_input.read_value().numpy()
            if hasattr(entry_layer, 'last_raw_input'):
                views['raw'] = entry_layer.last_raw_input.read_value().numpy()
            
            return views if views else None
        except Exception:
            pass
        return None

    def get_permanence_map(self):
        """Calculates the average Synaptic Permanence (LTM) for every brain region."""
        # Mapping regions to their internal layer aggregations
        regions = {
            "visual": self.visual_cortex.layers,
            "temporal": self.temporal_lobe.layers,
            "parietal": [self.integration_layer],
            "executive": self.prefrontal_cortex.layers,
            "broca": self.frontal_language.layers,
            "hippocampus": [self.hippocampus],
            "motor_strip": self.visual_motor_strip.layers,
            "cerebellum": [self.cerebellum]
        }
        
        perm_map = {}
        for region, layer_list in regions.items():
            perms = []
            for layer in layer_list:
                if hasattr(layer, 'permanence'):
                    perms.append(tf.reduce_mean(layer.permanence))
                if hasattr(layer, 'recurrent_permanence'):
                    perms.append(tf.reduce_mean(layer.recurrent_permanence))
            
            if perms:
                perm_map[region] = tf.reduce_mean(perms)
            else:
                perm_map[region] = 0.0
                
        # Global permanence average
        all_perms = list(perm_map.values())
        perm_map["global"] = sum(all_perms) / len(all_perms) if all_perms else 0.0
        return perm_map

    def all_layers(self):
        """Iterator for every synaptic layer in the brain."""
        layers = []
        layers.extend(self.visual_cortex.layers)
        layers.extend(self.temporal_lobe.layers)
        layers.append(self.vwfa_bridge)
        layers.append(self.integration_layer)
        layers.append(self.hippocampus)
        layers.extend(self.prefrontal_cortex.layers)
        layers.append(self.amygdala)
        layers.append(self.basal_ganglia)
        layers.append(self.cerebellum)
        layers.extend(self.frontal_language.layers)
        layers.extend(self.visual_motor_strip.layers)
        return layers

    def reset_state(self):
        """Clears all neural memory across the entire connectome."""
        regions = [
            self.visual_cortex, self.temporal_lobe, self.integration_layer, 
            self.vwfa_bridge, self.prefrontal_cortex, self.visual_motor_strip,
            self.hippocampus, self.basal_ganglia, self.amygdala, self.cerebellum, self.frontal_language
        ]
        for region in regions:
            if hasattr(region, 'reset_state'):
                region.reset_state()

    def get_variables(self):
        vars = self.visual_cortex.get_variables()
        vars += self.temporal_lobe.get_variables()
        vars += self.vwfa_bridge.get_variables()
        vars += self.integration_layer.get_variables()
        vars += self.prefrontal_cortex.get_variables()
        vars += self.frontal_language.get_variables()
        vars += self.visual_motor_strip.get_variables()
        vars += self.hippocampus.get_variables()
        vars += self.basal_ganglia.get_variables()
        vars += self.amygdala.get_variables()
        vars += self.cerebellum.get_variables()
        return vars

    def get_noise_state(self):
        """Returns the current average noise level across the connectome."""
        regions = [
            self.visual_cortex, self.temporal_lobe, self.integration_layer,
            self.vwfa_bridge, self.prefrontal_cortex, self.visual_motor_strip,
            self.hippocampus, self.basal_ganglia, self.amygdala, self.cerebellum, self.frontal_language
        ]
        noises = []
        for region in regions:
            if hasattr(region, 'noise_std'):
                noises.append(float(region.noise_std.numpy()))
            elif hasattr(region, 'layers'):
                for layer in region.layers:
                    if hasattr(layer, 'noise_std'):
                        noises.append(float(layer.noise_std.numpy()))
        return sum(noises) / len(noises) if noises else 0.0

    def modulate_noise(self, target_level=None, decay_rate=None, attenuation_factor=None):
        """
        Adjusts the noise floor. 
        """
        regions = [
            self.visual_cortex, self.temporal_lobe, self.integration_layer, 
            self.vwfa_bridge, self.prefrontal_cortex, self.visual_motor_strip,
            self.hippocampus, self.basal_ganglia, self.amygdala, self.cerebellum, self.frontal_language
        ]
        for region in regions:
            # Handle standard layers
            if hasattr(region, 'noise_std'):
                current = region.noise_std.numpy()
                if target_level is not None:
                    region.noise_std.assign(target_level)
                elif attenuation_factor is not None and hasattr(region, 'baseline_noise'):
                    region.noise_std.assign(region.baseline_noise * attenuation_factor)
                elif decay_rate is not None and hasattr(region, 'baseline_noise'):
                    # Decay toward baseline: new = current + (baseline - current) * rate
                    target = region.baseline_noise
                    new_noise = current + (target - current) * decay_rate
                    region.noise_std.assign(new_noise)
            
            # Handle composite regions (like AuditoryLanguageCortex which has nested layers)
            if hasattr(region, 'layers'):
                for layer in region.layers:
                    if hasattr(layer, 'noise_std'):
                        current = layer.noise_std.numpy()
                        if target_level is not None:
                            layer.noise_std.assign(target_level)
                        elif attenuation_factor is not None and hasattr(layer, 'baseline_noise'):
                            layer.noise_std.assign(layer.baseline_noise * attenuation_factor)
                        elif decay_rate is not None and hasattr(layer, 'baseline_noise'):
                            target = layer.baseline_noise
                            new_noise = current + (target - current) * decay_rate
                            layer.noise_std.assign(new_noise)

    def save_weights(self, filepath="brain_weights.npy"):
        weights = [var.numpy() for var in self.get_variables()]
        try:
            np.save(filepath, np.array(weights, dtype=object), allow_pickle=True)
        except PermissionError:
            print(f">>> [System] WARNING: Permission denied writing to {filepath}. File likely locked by monitor. Skipping save.")
        except Exception as e:
            print(f">>> [System] ERROR saving weights: {e}")
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
            "visual_input_dim": 49152,
            "auditory_input_dim": 300
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
        brain = cls(visual_input_dim=config.get("visual_input_dim", 49152), 
                    auditory_input_dim=config.get("auditory_input_dim", 300))
                    
        # Inject the physical synaptic bloodlines into the skeleton
        brain.load_weights(weights_path)
        return brain
