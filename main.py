import tensorflow as tf
from tokenizer.sensory_tokenizer import SensoryTokenizer
from data_ingestion.multimedia_loader import MultimediaLoader
from tokenizer.motor_decoder import MotorDecoder
from brain.connectome import BrainConnectome
import time

def main():
    # Massive temporal bounds for Generation (Allows long hallucination sequence passing the initial 14-frame layer delay)
    TIME_STEPS = 75

    print("--- Biological Prompt Generation Boot Sequence ---")
    loader = MultimediaLoader(visual_target_size=(64, 64))
    tokenizer = SensoryTokenizer(visual_dim=12288, auditory_dim=256)
    decoder = MotorDecoder(visual_decode_shape=(64, 64, 3))
    
    print("> Waking up Brain Connectome (Loading HD topology)...")
    brain = BrainConnectome.load_model("biological_model")
    
    print("\n--- Prompting Phase ---")
    # Taking generative text prompt from end user
    prompt = input("Enter a text prompt to generate (Images, Audio, Video, Text): ")
    if not prompt: prompt = "DEFAULT"
    
    print(f"> Translating [{prompt}] into auditory thalamus spike paths...")
    text_spikes_t = tokenizer.thalamic_routing("text", prompt, time_steps=TIME_STEPS)
    
    # To force generation entirely from the prompt, we physically zero out the senses (blindfolding)
    blind_visual = tf.zeros((1, TIME_STEPS, 12288), dtype=tf.float32)
    blind_audio = tf.zeros((1, TIME_STEPS, 256), dtype=tf.float32)
    
    sensory_audio = text_spikes_t + blind_audio

    print("\n--- Biological Inference Step ---")
    start_time = time.time()
    # Generative pass uses straight inference
    brocas_out, visual_out = brain.forward(blind_visual, sensory_audio)
    print(f">> Complex thought formed in {time.time() - start_time:.4f} seconds.")

    print("\n--- Multi-Modal Generation Phase ---")
    print("> Executing inverse muscular decodings...")
    
    # 1. Output Visual Domain
    decoder.decode_to_image(visual_out, filepath="prompted_imagination.png")
    decoder.decode_to_video(visual_out, filepath="prompted_animation.mp4")
    
    # 2. Output Auditory/Linguistics Domain
    text_res = decoder.decode_to_text(brocas_out)
    print(f">> Generated Brain Text String: {text_res}")
    decoder.decode_to_audio(brocas_out, filepath="prompted_speech.wav")
    
    print("\n> Physical execution complete. Artifacts successfully materialized on disk.")

if __name__ == "__main__":
    main()
