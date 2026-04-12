import tensorflow as tf
import os
import glob
import numpy as np
import threading
import argparse
from collections import deque

# Enable Edge-AI Hardware Mixed Precision (Float16)
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("> Enabled mixed_float16 precision for Edge hardware efficiency.")
except Exception as e:
    print(f"> Failed to set mixed precision: {e}")

from tokenizer.sensory_tokenizer import SensoryTokenizer
from data_ingestion.multimedia_loader import MultimediaLoader
from execution.trainer import BrainTrainer
from brain.connectome import BrainConnectome

# Configuration
TIME_STEPS = 30
EPOCHS = 100
DATASET_DIR = "dataset"
MODEL_DIR = "biological_model"

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
    img_files = glob.glob(os.path.join(DATASET_DIR, "*.png"))
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
                
                chunk_size = 128
                if len(full_text) > 10:
                    for i in range(0, max(1, len(full_text) - chunk_size), chunk_size):
                        p_str = full_text[i : i + chunk_size]
                        t_str = full_text[i + 1 : i + 1 + chunk_size]
                        if len(t_str) < len(p_str):
                            t_str += " " * (len(p_str) - len(t_str))
                            
                        # BIOMIMETIC READING: Route the text visually as letter-shapes
                        vis_t = tokenizer.thalamic_routing("text", p_str, time_steps=time_steps)
                        
                        # The Target remains the Speech/ASCII representation for the Motor cortex
                        aud_targ = tokenizer.process_text_as_audio(t_str, time_steps=time_steps)
                        targ = tf.reduce_mean(aud_targ, axis=1)
                        
                        # No external sound during reading
                        aud_t = tf.zeros((1, time_steps, 256), dtype=tf.float32)
                        
                        yield vis_t, aud_t, targ

            # --- TRADITIONAL CROSS-MODAL LABELING ---
            text_str = base
            # Use ASCII for target prediction (Motor mapping)
            aud_targ = tokenizer.process_text_as_audio(text_str, time_steps=time_steps)
            targ = tf.reduce_mean(aud_targ, axis=1)
            
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
        tf.TensorSpec(shape=(1, 256), dtype=tf.float32)
    )
    # Autotune creates an asynchronous pipeline on the CPU, feeding the Edge GPU relentlessly
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(tf.data.AUTOTUNE)

def train():
    parser = argparse.ArgumentParser(description="Biomimetic SNN Trainer")
    parser.add_argument("--biotrain", action="store_true", help="Enable strictly biological training (Curriculum, STDP, Sleep Replay, Active Inference). Disables Backprop.")
    args, _ = parser.parse_known_args()
    
    bio_train_mode = args.biotrain
    print(f"--- Biological Connectome Training Sequence | BIOTRAIN: {bio_train_mode} ---")
    
    ensure_dummy_dataset()
    
    loader = MultimediaLoader(visual_target_size=(64, 64))
    
    # 64x64x3 = 12288 native dimensions
    tokenizer = SensoryTokenizer(visual_dim=12288, auditory_dim=256)
    
    print("> Subconsciously loading Brain Connectome topography...")
    brain = BrainConnectome.load_model(MODEL_DIR)
    
    trainer = BrainTrainer(brain)
    
    basenames = get_dataset_basenames()
    if not basenames:
        print("No training data found even after initialization. Exiting.")
        return

    print("--- Beginning Continuous Learning Loop ---")
    
    #Ask the user for the number of epochs
    epochs = input("How many epochs to train for? ")
    if(epochs is None or epochs == ""):
        epochs = 1000
    epochs = int(epochs)
    
    hippocampus = deque(maxlen=50) # Offline structural sleep buffer
    current_fovea = (0.0, 0.0) # Optical target center for active inference
    
    if not bio_train_mode:
        sensory_stream = build_sensory_dataset(loader, tokenizer, basenames, TIME_STEPS)
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        if bio_train_mode:
            # Active Inference cannot be purely asynchronous in a tf.data pipeline because T+1 depends on Motor Spikes of T!
            for base in basenames:
                img_path = os.path.join(DATASET_DIR, f"{base}.png")
                vid_path = os.path.join(DATASET_DIR, f"{base}.mp4")
                txt_path = os.path.join(DATASET_DIR, f"{base}.txt")
                
                # --- CONVERSATIONAL TEXT SLIDING WINDOW ---
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        full_text = f.read()
                    
                    chunk_size = 128
                    if len(full_text) > 10:
                        for i in range(0, max(1, len(full_text) - chunk_size), chunk_size):
                            p_str = full_text[i : i + chunk_size]
                            t_str = full_text[i + 1 : i + 1 + chunk_size]
                            if len(t_str) < len(p_str):
                                t_str += " " * (len(p_str) - len(t_str))
                            
                            vis_t = tokenizer.thalamic_routing("text", p_str, time_steps=TIME_STEPS)
                            aud_targ = tokenizer.process_text_as_audio(t_str, time_steps=TIME_STEPS)
                            targ = tf.reduce_mean(aud_targ, axis=1)
                            
                            aud_t = tf.zeros((1, TIME_STEPS, 256), dtype=tf.float32)
                            
                            brain.reset_state()
                            loss, brocas, imagined_vis = trainer.train_predictive_step(vis_t, aud_t, targ, bio_train_mode=True)
                            epoch_loss += float(loss)
                            hippocampus.append((vis_t, aud_t, targ))

                # --- CROSS-MODAL VISUAL LABELING ---
                text_str = base
                # Speech Target
                aud_targ = tokenizer.process_text_as_audio(text_str, time_steps=TIME_STEPS)
                targ = tf.reduce_mean(aud_targ, axis=1)
                
                aud_t = tf.zeros((1, TIME_STEPS, 256), dtype=tf.float32)
                
                vis_t = None
                if os.path.exists(vid_path):
                    frames = loader.load_video_frames(vid_path, max_frames=TIME_STEPS, bio_train_mode=True, epoch=epoch, fovea_offset=current_fovea)
                    vis_t = tokenizer.thalamic_routing("video", frames, time_steps=TIME_STEPS)
                elif os.path.exists(img_path):
                    img_t = loader.load_image(img_path, bio_train_mode=True, epoch=epoch, fovea_offset=current_fovea)
                    vis_t = tokenizer.thalamic_routing("vision", img_t, time_steps=TIME_STEPS)
                
                if vis_t is not None:
                    # Physical Experience computes Surprise natively while auto-updating STDP
                    brain.reset_state()
                    loss, brocas, imagined_vis = trainer.train_predictive_step(vis_t, aud_t, targ, bio_train_mode=True)
                    epoch_loss += float(loss)
                    
                    # Store memory in hippocampus for deep sleep (without explicit derivatives)
                    hippocampus.append((vis_t, aud_t, targ))
                    
                    # Motor strip directs optical gaze for NEXT physical event based on its abstraction
                    motor_x = tf.reduce_mean(imagined_vis) * 2.0 - 1.0 
                    motor_y = tf.math.reduce_std(imagined_vis) * 2.0 - 1.0
                    current_fovea = (float(motor_x), float(motor_y))
        else:
            for visual_train_t, auditory_train_t, target_text_rates in sensory_stream:
                brain.reset_state()
                loss, _, _ = trainer.train_predictive_step(visual_train_t, auditory_train_t, target_text_rates, bio_train_mode=False)
                epoch_loss += loss.numpy()
            
        print(f"Epoch {epoch}/{epochs} | Bio-Loss: {epoch_loss:.4f}")
        
        if (epoch % 5 == 0):
            if bio_train_mode and len(hippocampus) > 0:
                print("\n> Initiating Offline Sleep (Rapid Hippocampal STDP Replay) - Synchronous embedding...")
                for h_vis, h_aud, h_targ in list(hippocampus):
                    trainer.train_predictive_step(h_vis, h_aud, h_targ, bio_train_mode=True)
                hippocampus.clear()
            
            print("\n> Initiating Autonomic Biological Deep Sleep (Pruning on Background Thread)...")
            sleep_thread = threading.Thread(target=async_plasticity, args=(brain, 0.005, 0.1))
            sleep_thread.daemon = True
            sleep_thread.start()

            
        if epoch % 10 == 0:
            brain.save_model(MODEL_DIR)
            
    print("--- Training Concluded ---")
    brain.save_model(MODEL_DIR)

if __name__ == "__main__":
    train()
