import tensorflow as tf
import numpy as np
import argparse
import time
import os

from tokenizer.sensory_tokenizer import SensoryTokenizer
from data_ingestion.multimedia_loader import MultimediaLoader
from brain.connectome import BrainConnectome
from diagnostics.neural_stream import streamer

def run_temporal_probe(prompt="red apple"):
    TIME_STEPS = 30
    MODEL_DIR = "biological_model"

    print(f"\n--- HYPOTHESIS TEST: Temporal Probe for '{prompt}' ---")
    
    # 1. Initialize Scaffolding
    tokenizer = SensoryTokenizer(visual_dim=49152, auditory_dim=300)
    
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory {MODEL_DIR} not found.")
        return

    print("> Loading Brain Connectome...")
    brain = BrainConnectome.load_model(MODEL_DIR)
    
    # 2. Connect to Monitor (v1.5 Telemetry)
    streamer.connect()
    
    # 2. Process Sensory Input
    # We use "text" routing which renders it as visual shapes (biomimetic reading)
    print(f"> Routing prompt '{prompt}' through visual pathways...")
    visual_input = tokenizer.thalamic_routing("text", prompt, time_steps=TIME_STEPS)
    auditory_input = tf.zeros((1, TIME_STEPS, 300), dtype=tf.float32)
    
    # 3. Forward Pass (Temporal Splicing)
    print("> Propagating spikes through cortices (1ms resolution)...")
    brain.reset_state()
    
    all_broca = []
    
    for t in range(TIME_STEPS):
        vis_step = visual_input[:, t:t+1, :]
        aud_step = auditory_input[:, t:t+1, :]
        
        # Forward millisecond
        # v0.2.8: Optimized capture for Regional Telemetry
        broca_step, _, _, regional_activity = brain.forward(vis_step, aud_step)
        all_broca.append(broca_step)
        
        # Stream state to Dashboard
        retinal_view = brain.get_retinal_view()
        streamer.stream_state(regional_activity, context_text=f"Probing: {prompt}", mode="inference", retinal_proto=retinal_view)
        
    # Stack the full spike train [1, 30, 256]
    brocas_out = tf.concat(all_broca, axis=1)
    rates_np = brocas_out[0].numpy() # [30, 256]

    # 4. DIAGNOSTIC PROBE: Ignore Burst Mode, perform Millisecond Winner-Take-All
    print("\n[TEMPORAL PROBE RESULTS]")
    print("-" * 60)
    print(f"{'Time':<6} | {'Winner':<10} | {'Activity':<10} | {'Runners Up'}")
    print("-" * 60)

    recovered_chars = []
    
    for t in range(TIME_STEPS):
        # Only look within printable ASCII (32-126)
        printable = rates_np[t, 32:127]
        max_rate = float(np.max(printable))
        
        if max_rate > 0.05: # Detection floor
            # Get Top 3 candidates at this millisecond
            top_indices = np.argsort(printable)[-3:][::-1]
            winners = [chr(idx + 32) for idx in top_indices if printable[idx] > 0.05]
            
            main_winner = winners[0]
            runners_up = ", ".join(winners[1:]) if len(winners) > 1 else "None"
            
            print(f"{t+1:>3} ms  | {main_winner:<10} | {max_rate:<10.4f} | {runners_up}")
            
            if not recovered_chars or main_winner != recovered_chars[-1]:
                recovered_chars.append(main_winner)
        else:
            print(f"{t+1:>3} ms  | {'(silent)':<10} | {max_rate:<10.4f} | -")

    print("-" * 60)
    final_sequence = "".join(recovered_chars)
    print(f"\n>> Recovered Temporal Sequence: {final_sequence}")
    print(f">> Hypothesis: If the sequence contains fragments of '{prompt}', the flashcard data survived the noise.")
    
    # 5. Cleanup
    streamer.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="red apple")
    args = parser.parse_args()
    
    run_temporal_probe(args.prompt)
