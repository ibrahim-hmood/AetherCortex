import tensorflow as tf
import numpy as np
import argparse
import time
try:
    import cv2
except ImportError:
    cv2 = None

from tokenizer.sensory_tokenizer import SensoryTokenizer
from data_ingestion.multimedia_loader import MultimediaLoader
from tokenizer.motor_decoder import MotorDecoder
from brain.connectome import BrainConnectome

def main():
    parser = argparse.ArgumentParser(description="Biological Inference Engine")
    parser.add_argument("--biogen", action="store_true", help="Enable extreme biologically realistic generative loops.")
    args, _ = parser.parse_known_args()

    TIME_STEPS = 75

    print("--- Biological Prompt Generation Boot Sequence ---")
    loader = MultimediaLoader(visual_target_size=(64, 64))
    tokenizer = SensoryTokenizer(visual_dim=12288, auditory_dim=256)
    decoder = MotorDecoder(visual_decode_shape=(64, 64, 3))
    
    print("> Waking up Brain Connectome (Loading HD topology)...")
    brain = BrainConnectome.load_model("biological_model")
    
    print("\n--- Prompting Phase ---")
    prompt = input("Enter a text prompt to spark generation: ")
    if not prompt: prompt = "DEFAULT"
    
    # Ensure the brain starts with a fresh electrical potential for this prompt
    brain.reset_state()
    
    print(f"> Translating [{prompt}] into visual cortex reading paths (Eye Scanning)...")
    # BIOMIMETIC READING: The prompt enters through the Visual channel as letter-shapes
    visual_text_spikes = tokenizer.thalamic_routing("text", prompt, time_steps=TIME_STEPS)
    
    blind_visual = tf.zeros((1, TIME_STEPS, 12288), dtype=tf.float32)
    blind_audio = tf.zeros((1, TIME_STEPS, 256), dtype=tf.float32)

    if not args.biogen:
        # ORIGINAL INFERENCE
        print("\n--- Standard Biological Inference Step (Reading and Thinking) ---")
        start_time = time.time()
        # Feed the visual text, and zero external sound
        brocas_out, visual_out = brain.forward(visual_text_spikes, blind_audio)
        print(f">> Complex thought formed in {time.time() - start_time:.4f} seconds.")

        print("\n--- Multi-Modal Generation Phase ---")
        decoder.decode_to_image(visual_out, filepath="prompted_imagination.png")
        decoder.decode_to_video(visual_out, filepath="prompted_animation.mp4")
        
        text_res = decoder.decode_to_text(brocas_out)
        print(f">> Generated Brain Text String: {text_res}")
        decoder.decode_to_audio(brocas_out, filepath="prompted_speech.wav")
        print("\n> Physical execution complete. Artifacts materialized.")
        return

    # === BIOGEN MODE ===
    print("\n--- Biological Generative Modes ---")
    print("1: Biological Auto-Regression (Text Loop)")
    print("2: Hallucination Feedback Loop (Video Dreaming)")
    print("3: Latent Directed Dreaming (Deep Image Exposure)")
    print("4: Active Inference Canvas (Saccadic Drawing)")
    mode = input("Select a generation strategy (1/2/3/4): ").strip()
    
    if mode == "1":
        print("\n> Initiating Biological Auto-Regression (Conversing with self)...")
        cycles = 10
        # Start by reading the prompt (Visual)
        current_vis_spikes = visual_text_spikes
        # Inner-voice starts empty
        current_audio_spikes = blind_audio
        
        full_sentence = []
        for i in range(cycles):
            brocas_out, _ = brain.forward(current_vis_spikes, current_audio_spikes)
            
            # After cycle 1, the prompt "fades" from view, relying on auditory memory/feedback
            current_vis_spikes = blind_visual
            
            word = decoder.decode_to_text(brocas_out)
            word = word.strip() if word.strip() else "_"
            full_sentence.append(word)
            print(f"Cycle {i+1} Thought: {word}")
            # PHONOLOGICAL LOOP: Hearing your own voice (Auditory Feedback)
            current_audio_spikes = tokenizer.thalamic_routing("audio_text", word, time_steps=TIME_STEPS)
        print(f"\n>> Final Monologue: {' '.join(full_sentence)}")

    elif mode == "2":
        print("\n> Initiating Hallucination Feedback Loop (Recursive Video)...")
        frames = 20
        # Start by reading the concept visually (Reading the word "ocean")
        current_vis_spikes = visual_text_spikes
        for t in range(frames):
            print(f"> Dreaming frame {t+1}/{frames}...")
            _, visual_out = brain.forward(current_vis_spikes, blind_audio)
            
            img_arr = decoder.decode_to_image(visual_out, filepath=f"temp_dream_{t}.png")
            img_tensor = tf.convert_to_tensor(img_arr, dtype=tf.float32)
            # Reconstruct visual input from the dream
            current_vis_spikes = tokenizer.thalamic_routing("vision", img_tensor, time_steps=TIME_STEPS)
        print(f"\n>> Video frames compiled (temp_dream_X.png). Ready for FFMPEG!")

    elif mode == "3":
        print("\n> Initiating Latent Directed Dreaming (300-Frame Neural Exposure)...")
        LONG_STEPS = 300
        # Only read for the first 30 frames
        prompt_segment = tokenizer.thalamic_routing("text", prompt, time_steps=30)
        silence_segment = tf.zeros((1, LONG_STEPS - 30, 12288), dtype=tf.float32)
        long_visual = tf.concat([prompt_segment, silence_segment], axis=1)
        long_audio = tf.zeros((1, LONG_STEPS, 256), dtype=tf.float32)
        
        _, visual_out = brain.forward(long_visual, long_audio)
        decoder.decode_to_image(visual_out, filepath="deep_dream.png")
        print("\n>> Deep Latent Dream image mapping materialized into deep_dream.png")

    elif mode == "4":
        print("\n> Initiating Active Inference Canvas (Saccadic Drawing)...")
        canvas = np.zeros((128, 128, 3), dtype=np.uint8)
        current_fovea_x = 0.0 # -1.0 to 1.0 offsets
        current_fovea_y = 0.0
        
        cycles = 15
        for i in range(cycles):
            print(f"> Saccade Cycle {i+1}/{cycles} | Fovea: ({current_fovea_x:.2f}, {current_fovea_y:.2f})")
            
            # Combine the physical eye view (patch) with the abstract reading (prompt)
            if i < 3:
                # First cycles: Read the prompt visually to prime the imagination
                vis_input = tokenizer.thalamic_routing("text", prompt, time_steps=TIME_STEPS)
            else:
                # Later cycles: Look at the actual canvas
                vis_input = tokenizer.thalamic_routing("vision", vis_tensor, time_steps=TIME_STEPS)
            
            brocas_out, visual_out = brain.forward(vis_input, blind_audio)
            
            img_arr = decoder.decode_to_image(visual_out, filepath="temp.png")
            
            if cv2 is not None:
                # Organically blend the visual motor thought onto the actual physical canvas
                canvas[y_off:y_off+64, x_off:x_off+64] = cv2.addWeighted(canvas[y_off:y_off+64, x_off:x_off+64], 0.3, img_arr, 0.7, 0)
            
            # Motor cortex drives purely abstract coordinate translation
            motor_x = tf.reduce_mean(brocas_out) * 5.0 - 2.5
            motor_y = tf.math.reduce_std(brocas_out) * 5.0 - 2.5
            current_fovea_x = float(motor_x)
            current_fovea_y = float(motor_y)
            
        if cv2 is not None:
            cv2.imwrite("saccadic_drawing.png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            print("\n>> Active Inference Drawing preserved natively in saccadic_drawing.png")
        else:
            print("\n>> Skip saving saccadic_drawing.png (cv2 missing)")

if __name__ == "__main__":
    main()
