import tensorflow as tf
import os
import glob
import numpy as np
import random
import threading
import argparse
from collections import deque

# Enable Edge-AI Hardware Mixed Precision (Float16) only if a GPU is available
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("> Enabled mixed_float16 precision for Edge hardware efficiency.")
    else:
        print("> GPU not detected. Using float32 for maximum CPU numerical stability.")
except Exception as e:
    print(f"> Failed to set mixed precision: {e}")

from tokenizer.sensory_tokenizer import SensoryTokenizer
from data_ingestion.multimedia_loader import MultimediaLoader
from execution.trainer import BrainTrainer
from brain.connectome import BrainConnectome
from curriculum import SensoryCurriculum
from diagnostics.neural_stream import streamer

# Configuration
TIME_STEPS = 30
EPOCHS = 100
DATASET_DIR = "dataset"
MODEL_DIR = "biological_model"

# Curriculum Focus (v4.2)
focus_word = ""

def ensure_dummy_dataset():
    """Generates a dummy multimodal dataset if the folder is empty so we can test right away."""
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        png_files = glob.glob(os.path.join(DATASET_DIR, "*.png"))
        if len(png_files) == 0:
            print("> No dataset found. Generating dummy physiological dataset...")
            
            # We'll create basic concepts
            concepts = {
                "apple": {"text": "A red apple", "color": [255, 0, 0]},
                "ocean": {"text": "The blue ocean", "color": [0, 0, 255]},
                "grass": {"text": "Green grass", "color": [0, 255, 0]}
            }
            
            for name, data in concepts.items():
                # 1. Removed textual file dependencies (Title acts as text)
                
                # 2. Write image (A solid colored block representing the concept)
                # Using basic tf image encoding
                img_array = np.full((32, 32, 3), data["color"], dtype=np.uint8)
                img_string = tf.io.encode_png(img_array)
                tf.io.write_file(os.path.join(DATASET_DIR, f"{name}.png"), img_string)
                
                # Note: We skip audio/video, the MultimediaLoader natively handles missing files gracefully.
            
def get_dataset_basenames():
    img_files = glob.glob(os.path.join(DATASET_DIR, "*.png")) + glob.glob(os.path.join(DATASET_DIR, "*.jpg")) + glob.glob(os.path.join(DATASET_DIR, "*.jpeg"))
    vid_files = glob.glob(os.path.join(DATASET_DIR, "*.mp4"))
    txt_files = glob.glob(os.path.join(DATASET_DIR, "*.txt"))
    all_files = img_files + vid_files + txt_files
    # Return unique basenames without extension
    return list(set([os.path.splitext(os.path.basename(f))[0] for f in all_files]))

def async_plasticity(brain, prune_threshold, grow_threshold):
    pruned, grown = brain.apply_plasticity_and_pruning(prune_threshold, grow_threshold)
    print(f"\n>> [Autonomic System] Eradicated {int(pruned)} dead synapses (Forgetting).")
    print(f">> [Autonomic System] Spawned {int(grown)} new synapses (Learning).\n")

def build_sensory_dataset(loader, tokenizer, basenames, time_steps):
    def gen():
        for base in basenames:
            img_path = os.path.join(DATASET_DIR, f"{base}.png")
            vid_path = os.path.join(DATASET_DIR, f"{base}.mp4")
            txt_path = os.path.join(DATASET_DIR, f"{base}.txt")
            
            # --- CONVERSATIONAL TEXT SLIDING WINDOW ---
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                
                chunk_size = 6  # SLOW-READING FIX: 1 character every 5 steps
                if len(full_text) > 10:
                    for i in range(0, max(1, len(full_text) - chunk_size), chunk_size):
                        p_str = full_text[i : i + chunk_size]
                        t_str = full_text[i + 1 : i + 1 + chunk_size]
                        if len(t_str) < len(p_str):
                            t_str += " " * (len(p_str) - len(t_str))
                            
                        # BIOMIMETIC READING: Route the text visually as letter-shapes
                        vis_t = tokenizer.thalamic_routing("text", p_str, time_steps=time_steps)
                        
                        # The Target is the sequential speech spike train for T+1 text
                        targ = tokenizer.process_text_as_audio(t_str, time_steps=time_steps)
                        
                        # No external sound during reading
                        aud_t = tf.zeros((1, time_steps, 256), dtype=tf.float32)
                        
                        yield vis_t, aud_t, targ

            # --- TRADITIONAL CROSS-MODAL LABELING ---
            text_str = base
            # Sequential speech target: each char fires at its own time step
            targ = tokenizer.process_text_as_audio(text_str, time_steps=time_steps)
            
            # External sound is zero
            aud_t = tf.zeros((1, time_steps, 256), dtype=tf.float32)
            
            if os.path.exists(vid_path):
                frames = loader.load_video_frames(vid_path, max_frames=time_steps)
                vis_t = tokenizer.thalamic_routing("video", frames, time_steps=time_steps)
                yield vis_t, aud_t, targ
                
            if os.path.exists(img_path):
                img_t = loader.load_image(img_path)
                vis_t = tokenizer.thalamic_routing("vision", img_t, time_steps=time_steps)
                yield vis_t, aud_t, targ

    # Native tensor dimensions out of the thalamus 
    sig = (
        tf.TensorSpec(shape=(1, time_steps, 12288), dtype=tf.float32),
        tf.TensorSpec(shape=(1, time_steps, 256), dtype=tf.float32),
        tf.TensorSpec(shape=(1, time_steps, 256), dtype=tf.float32)  # Sequential speech target
    )
    # Autotune creates an asynchronous pipeline on the CPU, feeding the Edge GPU relentlessly
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(tf.data.AUTOTUNE)

def train():
    parser = argparse.ArgumentParser(description="Biomimetic SNN Trainer")
    parser.add_argument("--biotrain", action="store_true", help="Enable strictly biological training (Curriculum, STDP, Sleep Replay, Active Inference). Disables Backprop.")
    parser.add_argument("--epochs", type=int, default=0, help="Number of epochs to train for. Bypasses interactive input if > 0.")
    parser.add_argument("--focus", type=str, default="", help="Focus training on a specific word/concept")
    args, _ = parser.parse_known_args()
    
    bio_train_mode = args.biotrain
    focus_word = args.focus
    print(f"--- Biological Connectome Training Sequence | BIOTRAIN: {bio_train_mode} | FOCUS: {focus_word or 'None'} ---")
    
    ensure_dummy_dataset()
    
    loader = MultimediaLoader(visual_target_size=(64, 64))
    
    # 64x64x3 = 12288 native dimensions
    tokenizer = SensoryTokenizer(visual_dim=12288, auditory_dim=256)
    
    print("> Subconsciously loading Brain Connectome topography...")
    brain = BrainConnectome.load_model(MODEL_DIR)
    
    trainer = BrainTrainer(brain)
    trainer.focus_word = focus_word
    curriculum = SensoryCurriculum(DATASET_DIR)
    
    basenames = get_dataset_basenames()
    if not basenames:
        print("No training data found even after initialization. Exiting.")
        return

    print(f"--- Beginning Continuous Learning Loop | {curriculum.get_status()} ---")
    
    # Use command line epochs if provided, otherwise ask
    if args.epochs > 0:
        epochs = args.epochs
    else:
        epochs = input("How many epochs to train for? ")
        if(epochs is None or epochs == ""):
            epochs = 1000
        epochs = int(epochs)
    
    hippocampus = deque(maxlen=50) # Offline structural sleep buffer
    current_fovea = (0.0, 0.0) # Optical target center for active inference
    cranky_counter = 0 # Triggers reactive sleep for persistent seizures
    
    if not bio_train_mode:
        sensory_stream = build_sensory_dataset(loader, tokenizer, basenames, TIME_STEPS)
    
    # --- CONNECT TO NEURO-MONITOR ---
    streamer.connect()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        if bio_train_mode:
            # Stage-aware data selection
            stage_basenames = curriculum.get_stage_data(basenames)
            random.shuffle(stage_basenames)

            for base in stage_basenames:
                img_path = None
                for ext in [".png", ".jpg", ".jpeg"]:
                    if os.path.exists(os.path.join(DATASET_DIR, f"{base}{ext}")):
                        img_path = os.path.join(DATASET_DIR, f"{base}{ext}")
                        break
                if img_path is None:
                    img_path = os.path.join(DATASET_DIR, f"{base}.png") # fallback
                
                vid_path = os.path.join(DATASET_DIR, f"{base}.mp4")
                txt_path = os.path.join(DATASET_DIR, f"{base}.txt")
                
                # --- CURRICULUM LEVEL 2: CONVERSATIONAL TEXT ---
                if curriculum.current_level >= SensoryCurriculum.LEVEL_READING and os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        full_text = f.read()
                    
                    chunk_size = 6  # SLOW-READING FIX: 1 character every 5 steps
                    if len(full_text) > 10:
                        for i in range(0, max(1, len(full_text) - chunk_size), chunk_size):
                            p_str = full_text[i : i + chunk_size]
                            t_str = full_text[i + 1 : i + 1 + chunk_size]
                            if len(t_str) < len(p_str):
                                t_str += " " * (len(p_str) - len(t_str))
                            
                            vis_t = tokenizer.thalamic_routing("text", p_str, time_steps=TIME_STEPS)
                            targ = tokenizer.process_text_as_audio(t_str, time_steps=TIME_STEPS)
                            
                            aud_t = tf.zeros((1, TIME_STEPS, 256), dtype=tf.float32)
                            
                            brain.reset_state()
                            loss, brocas, imagined_vis, activity_map = trainer.train_predictive_step(vis_t, aud_t, targ, bio_train_mode=True, context_str=p_str)
                            
                            # --- STREAM TO DASHBOARD ---
                            p_map = brain.get_permanence_map()
                            streamer.stream_state(activity_map, permanence_map=p_map, context_text=p_str, mode="training")
                            
                            epoch_loss += float(loss)
                            hippocampus.append((vis_t, aud_t, targ))
                            curriculum.report_step(float(loss))

                # --- CURRICULUM LEVEL 1: CROSS-MODAL VISUAL LABELING ---
                text_str = base
                targ = tokenizer.process_text_as_audio(text_str, time_steps=TIME_STEPS)
                aud_t = tf.zeros((1, TIME_STEPS, 256), dtype=tf.float32)
                
                vis_t = None
                if os.path.exists(vid_path):
                    frames = loader.load_video_frames(vid_path, max_frames=TIME_STEPS, bio_train_mode=True, epoch=epoch, fovea_offset=current_fovea)
                    vis_t = tokenizer.thalamic_routing("video", frames, time_steps=TIME_STEPS)
                elif os.path.exists(img_path):
                    img_t = loader.load_image(img_path, bio_train_mode=True, epoch=epoch, fovea_offset=current_fovea)
                    vis_t = tokenizer.thalamic_routing("vision", img_t, time_steps=TIME_STEPS)
                
                if vis_t is not None:
                    brain.reset_state()
                    loss, brocas, imagined_vis, activity_map = trainer.train_predictive_step(vis_t, aud_t, targ, bio_train_mode=True, context_str=base)
                    
                    # --- STREAM TO DASHBOARD ---
                    p_map = brain.get_permanence_map()
                    streamer.stream_state(activity_map, permanence_map=p_map, context_text=f"Labeling: {base}", mode="training")
                    
                    epoch_loss += float(loss)
                    hippocampus.append((vis_t, aud_t, targ))
                    curriculum.report_step(float(loss))
                    
                    # Update Thalamic Shell state for gating
                    brain.last_spike_density = float(trainer.current_activity.numpy())
                    
                    # --- HOMEOSTATIC PANIC (v4.0 Reactive Sleep) ---
                    if brain.last_spike_density > 0.25:
                        cranky_counter += 1
                        if cranky_counter > 3:
                            print("\n>>> [Homeostasis] Brain is OVER-EXCITED. Triggering Emergency Consolidation...")
                            trainer.sleep_consolidation()
                            cranky_counter = 0
                            brain.reset_state()
                    else:
                        cranky_counter = 0

                    # Optical gaze persistence
                    new_x = tf.reduce_mean(imagined_vis) * 2.0 - 1.0 
                    new_y = tf.math.reduce_std(imagined_vis) * 2.0 - 1.0
                    current_fovea = (current_fovea[0] * 0.9 + float(new_x) * 0.1, current_fovea[1] * 0.9 + float(new_y) * 0.1)
        else:
            for visual_train_t, auditory_train_t, target_text_rates in sensory_stream:
                brain.reset_state()
                loss, _, _, activity_map = trainer.train_predictive_step(visual_train_t, auditory_train_t, target_text_rates, bio_train_mode=False)
                
                # --- STREAM TO DASHBOARD ---
                p_map = brain.get_permanence_map()
                streamer.stream_state(activity_map, permanence_map=p_map, context_text="Standard BackProp Step", mode="training")
                
                epoch_loss += loss.numpy()
            
        print(f"Epoch {epoch}/{epochs} | Bio-Loss: {epoch_loss:.4f} | {curriculum.get_status()}")
        
        # HOMEOSTATIC REGULATION: Update internal metabolism and rewards based on epoch performance
        trainer.update_homeostasis(epoch_loss, regional_activity=activity_map if 'activity_map' in locals() else None)
        
        # v0.1.8: Infancy Auto-Archiving (Save the FIRST stable epoch immediately)
        if epoch == 1:
            print("> [Infancy] archiving first stable neural traces...")
            brain.save_model(MODEL_DIR)
        
        # --- DEEP SLEEP (v4.0 Synaptic Consolidation) ---
        # At the end of every epoch, she pulls the hippocampal traces into physical synapses.
        if bio_train_mode:
            if len(hippocampus) > 0:
                print("\n> Initiating Rapid Hippocampal Replay...")
                # Replay recent high-saliency events through the brain one last time
                for h_vis, h_aud, h_targ in list(hippocampus):
                    trainer.train_predictive_step(h_vis, h_aud, h_targ, bio_train_mode=True, context_str="Hippocampal Replay")
                hippocampus.clear()
            
            # Physical Structural Consolidation (Pruning weak, Growing strong)
            trainer.sleep_consolidation()

            
        if epoch % 10 == 0:
            brain.save_model(MODEL_DIR)
            
    print("--- Training Concluded ---")
    brain.save_model(MODEL_DIR)

if __name__ == "__main__":
    train()
