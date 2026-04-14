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
from diagnostics.neural_stream import streamer

def main():
    parser = argparse.ArgumentParser(description="Biological Inference Engine")
    parser.add_argument("--biogen", action="store_true", help="Enable extreme biologically realistic generative loops.")
    args, _ = parser.parse_known_args()

    TIME_STEPS = 30

    print("--- Biological Prompt Generation Boot Sequence ---")
    loader = MultimediaLoader(visual_target_size=(128, 128))
    tokenizer = SensoryTokenizer(visual_dim=49152, auditory_dim=300)
    decoder = MotorDecoder(visual_decode_shape=(128, 128, 3))
    
    print("> Waking up Brain Connectome (Loading HD topology)...")
    brain = BrainConnectome.load_model("biological_model")

    # --- CONNECT TO NEURO-MONITOR ---
    streamer.connect()
    
    print("\n--- Prompting Phase ---")
    prompt = input("Enter a text prompt to spark generation: ")
    if not prompt: prompt = "DEFAULT"
    
    # Ensure the brain starts with a fresh electrical potential for this prompt
    brain.reset_state()
    
    print(f"> Translating [{prompt}] into visual cortex reading paths (Eye Scanning)...")
    # BIOMIMETIC READING: The prompt enters through the Visual channel as letter-shapes
    visual_text_spikes = tokenizer.thalamic_routing("text", prompt, time_steps=TIME_STEPS)
    
    blind_visual = tf.zeros((1, TIME_STEPS, 49152), dtype=tf.float32)
    blind_audio = tf.zeros((1, TIME_STEPS, 300), dtype=tf.float32)

    if not args.biogen:
        # ORIGINAL INFERENCE
        # --- TEMPORAL SPLICING (v3.3: Step-by-Step 'Thinking' Visualization) ---
        print("\n--- Standard Biological Inference Step (Cognitive Ripple) ---")
        start_time = time.time()
        
        # We process the 30-step prompt millisecond-by-millisecond to see the signal propagation
        all_broca = []
        all_visual = []
        
        for t in range(TIME_STEPS):
            # Slice exactly 1 millisecond of the sensory spike stream
            vis_step = visual_text_spikes[:, t:t+1, :]
            aud_step = blind_audio[:, t:t+1, :]
            
            # Sub-millisecond forward pass
            brocas_step, visual_step, _, activity_map = brain.forward(vis_step, aud_step)
            all_broca.append(brocas_step)
            all_visual.append(visual_step)
            
            # --- STREAM TO DASHBOARD (One update per millisecond) ---
            step_context = f"Reading: {prompt} [{t+1}/{TIME_STEPS}ms]"
            p_map = brain.get_permanence_map()
            retinal_view = brain.get_retinal_view()
            streamer.stream_state(activity_map, permanence_map=p_map, context_text=step_context, mode="inference", retinal_proto=retinal_view)
            
            # Cognitive Pacing: Ensures the ripple is visible on the web dashboard
            time.sleep(0.01)
        
        # Stack the results back into a 30-step tensor for the motor decoders
        brocas_out = tf.concat(all_broca, axis=1)
        visual_out = tf.concat(all_visual, axis=1)
        
        print(f">> Full cognitive cycle completed in {time.time() - start_time:.4f} seconds.")

        print("\n--- Multi-Modal Generation Phase ---")
        decoder.decode_to_image(visual_out, filepath="prompted_imagination.png")
        
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
            all_broca = []
            for sub_t in range(TIME_STEPS):
                vis_step = current_vis_spikes[:, sub_t:sub_t+1, :]
                aud_step = current_audio_spikes[:, sub_t:sub_t+1, :]
                
                brocas_step, _, _, activity_map = brain.forward(vis_step, aud_step)
                all_broca.append(brocas_step)
                
                # --- STREAM TO DASHBOARD (Inner Voice Activity) ---
                current_context = " ".join(full_sentence) if full_sentence else "Starting monologue..."
                p_map = brain.get_permanence_map()
                streamer.stream_state(activity_map, permanence_map=p_map, context_text=f"{current_context} [{sub_t+1}/30ms]", mode="generation")
                time.sleep(0.005)

            brocas_out = tf.concat(all_broca, axis=1)
            
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
            
            all_vis = []
            for sub_t in range(TIME_STEPS):
                # Process each dream-millisecond individually
                vis_step = current_vis_spikes[:, sub_t:sub_t+1, :]
                aud_step = blind_audio[:, sub_t:sub_t+1, :]
                
                _, visual_step, _, activity_map = brain.forward(vis_step, aud_step)
                all_vis.append(visual_step)
                
                # --- STREAM TO DASHBOARD (Flicker Effect) ---
                p_map = brain.get_permanence_map()
                streamer.stream_state(activity_map, permanence_map=p_map, context_text=f"Dreaming frame {t+1} [{sub_t+1}/30ms]", mode="generation")
                time.sleep(0.005) # Faster delay for dreaming
            
            # Reconstruct the full 30-step hallucination
            visual_out = tf.concat(all_vis, axis=1)
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
        silence_segment = tf.zeros((1, LONG_STEPS - 30, 49152), dtype=tf.float32)
        long_visual = tf.concat([prompt_segment, silence_segment], axis=1)
        long_audio = tf.zeros((1, LONG_STEPS, 300), dtype=tf.float32)
        
        all_vis = []
        for t in range(LONG_STEPS):
            vis_step = long_visual[:, t:t+1, :]
            aud_step = long_audio[:, t:t+1, :]
            
            _, visual_step, _, activity_map = brain.forward(vis_step, aud_step)
            all_vis.append(visual_step)
            
            # --- STREAM TO DASHBOARD (Deep Wave) ---
            if t % 5 == 0: # Stream every 5ms for the long 300ms dream to avoid network lag
                p_map = brain.get_permanence_map()
                streamer.stream_state(activity_map, permanence_map=p_map, context_text=f"Deep Exposure Dream [{t+1}/300ms]", mode="generation")
            time.sleep(0.002)
        
        visual_out = tf.concat(all_vis, axis=1)
        
        decoder.decode_to_image(visual_out, filepath="deep_dream.png")
        # Final Refresh AFTER file is written
        p_map = brain.get_permanence_map()
        streamer.stream_state(activity_map, permanence_map=p_map, context_text="Latent Directed Dreaming Finalized", mode="generation")
        print("\n>> Deep Latent Dream image mapping materialized into deep_dream.png")

    elif mode == "4":
        print("\n> Initiating Active Inference Canvas (Saccadic Drawing)...")
        # v4.5: Expanded Canvas to allow 128x128 eye to 'roam' across larger paper
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        current_fovea_x = 0.0 # -1.0 to 1.0 offsets
        current_fovea_y = 0.0
        
        cycles = 15
        for i in range(cycles):
            # Calculate pixel offsets from foveal coordinates (-1.0 to 1.0 range)
            # Center (0,0) -> Offsets (64, 64) for a 128x128 patch on a 256x256 canvas
            x_off = int(np.clip(64 + current_fovea_x * 64, 0, 128))
            y_off = int(np.clip(64 + current_fovea_y * 64, 0, 128))
            
            print(f"> Saccade Cycle {i+1}/{cycles} | Fovea: ({current_fovea_x:.2f}, {current_fovea_y:.2f}) -> Offset: ({x_off}, {y_off})")
            
            # Extract the current foveal patch (HD 128x128)
            patch = canvas[y_off:y_off+128, x_off:x_off+128]
            vis_tensor = tf.convert_to_tensor(patch, dtype=tf.float32)
            
            # Combine the physical eye view (patch) with the abstract reading (prompt)
            if i < 3:
                # First cycles: Read the prompt visually to prime the imagination
                vis_input = tokenizer.thalamic_routing("text", prompt, time_steps=TIME_STEPS)
            else:
                # Later cycles: Look at the actual canvas through the eyes (Active Inference)
                vis_input = tokenizer.thalamic_routing("vision", vis_tensor, time_steps=TIME_STEPS)
            
            all_broca = []
            all_vis = []
            
            # v4.5: Neural Refresh - clear out 'sticky' electrical patterns before every saccade
            brain.reset_state()
            
            for sub_t in range(TIME_STEPS):
                # Process each saccade-millisecond individually
                vis_step = vis_input[:, sub_t:sub_t+1, :]
                aud_step = blind_audio[:, sub_t:sub_t+1, :]
                
                brocas_step, visual_step, _, activity_map = brain.forward(vis_step, aud_step)
                all_broca.append(brocas_step)
                all_vis.append(visual_step)
                
                # --- STREAM TO DASHBOARD (Active Saccade ripple) ---
                p_map = brain.get_permanence_map()
                streamer.stream_state(activity_map, permanence_map=p_map, context_text=f"Saccade {i+1} [{sub_t+1}/30ms]", mode="generation")
                time.sleep(0.005)
            
            brocas_out = tf.concat(all_broca, axis=1)
            visual_out = tf.concat(all_vis, axis=1)
            
            img_arr = decoder.decode_to_image(visual_out, filepath="temp.png")
            
            if cv2 is not None:
                # v4.5: Synchronized HD blending (128x128)
                canvas[y_off:y_off+128, x_off:x_off+128] = cv2.addWeighted(canvas[y_off:y_off+128, x_off:x_off+128], 0.3, img_arr, 0.7, 0)
            
            # v4.5: KINETIC MOTOR DYNAMICS
            # Instead of absolute position, motor spikes now provide directional MOMENTUM (Relative movement)
            # Scaling reduced (10.0 -> 0.5) because movement is now cumulative over 15 cycles.
            motor_jitter_x = (tf.reduce_mean(brocas_out) * 1.5 - 0.75) 
            motor_jitter_y = (tf.math.reduce_std(brocas_out) * 1.5 - 0.75)
            
            current_fovea_x += float(motor_jitter_x)
            current_fovea_y += float(motor_jitter_y)
            
            # Boundary Hardening: Prevent the eye from drifting off the paper (-1.0 to 1.0 logic range)
            current_fovea_x = np.clip(current_fovea_x, -1.0, 1.0)
            current_fovea_y = np.clip(current_fovea_y, -1.0, 1.0)
            
        if cv2 is not None:
            cv2.imwrite("saccadic_drawing.png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            # Final Refresh AFTER file is written
            p_map = brain.get_permanence_map()
            streamer.stream_state(activity_map, permanence_map=p_map, context_text="Active Inference Drawing Finalized", mode="generation")
            print("\n>> Active Inference Drawing preserved natively in saccadic_drawing.png")
        else:
            print("\n>> Skip saving saccadic_drawing.png (cv2 missing)")

if __name__ == "__main__":
    main()
