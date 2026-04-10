import tensorflow as tf
import os
import glob
import numpy as np

from tokenizer.sensory_tokenizer import SensoryTokenizer
from data_ingestion.multimedia_loader import MultimediaLoader
from execution.trainer import BrainTrainer
from brain.connectome import BrainConnectome

# Configuration
TIME_STEPS = 15
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
    all_files = img_files + vid_files
    # Return unique basenames without extension
    return list(set([os.path.splitext(os.path.basename(f))[0] for f in all_files]))

def train():
    print("--- Biological Connectome Training Sequence ---")
    ensure_dummy_dataset()
    
    loader = MultimediaLoader(visual_target_size=(32, 32))
    tokenizer = SensoryTokenizer(visual_dim=3072, auditory_dim=256)
    
    print("> Subconsciously loading Brain Connectome topography...")
    brain = BrainConnectome.load_model(MODEL_DIR)
    
    trainer = BrainTrainer(brain)
    
    basenames = get_dataset_basenames()
    if not basenames:
        print("No training data found even after initialization. Exiting.")
        return

    print("--- Beginning Continuous Learning Loop ---")
    
    #Ask the user for the number of epochs
    epochs = int(input("How many epochs to train for? "))
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        for base in basenames:
            img_path = os.path.join(DATASET_DIR, f"{base}.png")
            vid_path = os.path.join(DATASET_DIR, f"{base}.mp4")
            
            # The title serves as the text prompt natively
            text_str = base
            
            # 1. Process Video if it exists
            if os.path.exists(vid_path):
                frames = loader.load_video_frames(vid_path, max_frames=TIME_STEPS)
                visual_train_t = tokenizer.thalamic_routing("video", frames, time_steps=TIME_STEPS)
                auditory_train_t = tokenizer.thalamic_routing("text", text_str, time_steps=TIME_STEPS)
                target_text_rates = tf.reduce_mean(auditory_train_t, axis=1)
                
                loss, _, _ = trainer.train_predictive_step(visual_train_t, auditory_train_t, target_text_rates)
                epoch_loss += loss.numpy()

            # 2. Process Image if it exists
            if os.path.exists(img_path):
                img_tensor = loader.load_image(img_path)
                visual_train_t = tokenizer.thalamic_routing("vision", img_tensor, time_steps=TIME_STEPS)
                auditory_train_t = tokenizer.thalamic_routing("text", text_str, time_steps=TIME_STEPS)
                target_text_rates = tf.reduce_mean(auditory_train_t, axis=1)
                
                loss, _, _ = trainer.train_predictive_step(visual_train_t, auditory_train_t, target_text_rates)
                epoch_loss += loss.numpy()
            
        print(f"Epoch {epoch}/{epochs} | Bio-Loss: {epoch_loss:.4f}")
        
        # Periodically save structural traces
        if epoch % 10 == 0:
            brain.save_model(MODEL_DIR)
            
    print("--- Training Concluded ---")
    brain.save_model(MODEL_DIR)

if __name__ == "__main__":
    train()
