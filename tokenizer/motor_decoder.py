import tensorflow as tf
import numpy as np
import os
try:
    import cv2
except ImportError:
    cv2 = None

class MotorDecoder:
    """
    Inverse Tokenizer representing the Motor Strips and musculature.
    Translates Brain Spike trains back into Images, Video, Audio, and Text.
    """
    def __init__(self, visual_decode_shape=(64, 64, 3)):
        self.visual_decode_shape = visual_decode_shape
        self.flat_visual_dim = visual_decode_shape[0] * visual_decode_shape[1] * visual_decode_shape[2]

    def decode_to_image(self, visual_cortex_spikes, filepath="generated_output.png", gain=2.0):
        """
        Collapses structural visual spikes across time into a single static image frame.
        """
        # Using reduce_max performs biological 'light accumulation'
        firing_rates = tf.reduce_max(visual_cortex_spikes, axis=1)[0]
        
        # Apply Motor Gain to ensure sparse spikes are visible
        firing_rates = firing_rates * gain
        
        if tf.shape(firing_rates)[0] < self.flat_visual_dim:
            padded = tf.pad(firing_rates, [[0, self.flat_visual_dim - tf.shape(firing_rates)[0]]])
        else:
            padded = firing_rates[:self.flat_visual_dim]
            
        img_tensor = tf.reshape(padded, self.visual_decode_shape)
        # Clip to ensure intensities don't overflow
        img_tensor = tf.clip_by_value(img_tensor, 0.0, 1.0)
        img_array = tf.cast(img_tensor * 255.0, tf.uint8).numpy()
        
        if cv2 is not None:
            bgr_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, bgr_array)
            print(f">> Generated Static Image (Gain {gain}) saved to {filepath}")
        return img_array

    def decode_to_video(self, visual_cortex_spikes, filepath="generated_animation.mp4", fps=5):
        """
        Preserves the sequential nature of Spiking computation by rendering each 
        biological time step as an individual physical frame on screen.
        """
        if cv2 is None:
            return

        batch_idx = 0
        time_steps = tf.shape(visual_cortex_spikes)[1]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Upscaled to 256x256. Codecs (like mp4v) often corrupt buffers smaller than 64x64 macroblocks.
        upscale_dim = 256
        out = cv2.VideoWriter(filepath, fourcc, float(fps), (upscale_dim, upscale_dim))
        
        # Biological Retinal Persistence (Hold light natively to fix rapid black flicker)
        persistent_frame = np.zeros((upscale_dim, upscale_dim, 3), dtype=np.uint8)
        
        for t in range(time_steps):
            frame_rates = visual_cortex_spikes[batch_idx, t, :]
            
            # Skip physiological empty void (Latency during early brain computation propagation)
            if tf.reduce_max(frame_rates) == 0.0:
                continue
                
            if tf.shape(frame_rates)[0] < self.flat_visual_dim:
                padded = tf.pad(frame_rates, [[0, self.flat_visual_dim - tf.shape(frame_rates)[0]]])
            else:
                padded = frame_rates[:self.flat_visual_dim]
                
            img_tensor = tf.reshape(padded, self.visual_decode_shape)
            img_array = tf.cast(img_tensor * 255.0, tf.uint8).numpy()
            bgr_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Neuromuscular upscaling to preserve blocky shapes on the mp4
            upscaled_frame = cv2.resize(bgr_array, (upscale_dim, upscale_dim), interpolation=cv2.INTER_NEAREST)
            
            # Blend current spike heavily into the decaying phosphor trace
            persistent_frame = cv2.addWeighted(persistent_frame, 0.75, upscaled_frame, 0.8, 0)
            
            out.write(persistent_frame)
            
        out.release()
        print(f">> Generated Animation ({time_steps} frames) saved to {filepath}")

    def decode_to_text(self, brocas_spikes):
        """
        ADAPTIVE BIOMIMETIC DECODER:
        Detects the current firing regime and reads accordingly.

        BURST / RATE-CODING MODE  (early training, bag-coded weights):
          Neurons fire in a dense cloud across all time steps.  The decoder
          collapses across time and reads the printable ASCII neurons in
          dominance order (most-active first).  This recovers the babble that
          was working before, but now sorted by strength instead of by index.

        SEQUENTIAL / TEMPORAL MODE  (after sequential training converges):
          Broca's area fires one phoneme per time-step.  The decoder reads
          left-to-right, takes the printable-range Winner-Take-All at each
          step, and merges consecutive duplicates (held phonemes).

        The transition is automatic: fewer than 4 neurons firing per step on
        average means the sequential regime has emerged.
        """
        # brocas_spikes: [batch, time_steps, num_neurons]
        spikes_t  = brocas_spikes[0]          # [time_steps, num_neurons]
        rates_np  = spikes_t.numpy()          # eager, values are 0.0 or 1.0

        # --- Regime detection -------------------------------------------
        avg_active = float((rates_np > 0.5).sum(axis=1).mean())

        if avg_active <= 4.0:
            # ── SEQUENTIAL (TEMPORAL WTA) ──────────────────────────────
            # Only look within printable ASCII (32-126) for the winner.
            generated = ""
            last_char = None
            for t in range(rates_np.shape[0]):
                printable = rates_np[t, 32:127]          # slice to printable range
                if len(printable) == 0:
                    continue
                max_rate = float(printable.max())
                if max_rate > 0.5:
                    # argmax within the printable slice → offset back to full index
                    winner_idx = int(np.argmax(printable)) + 32
                    char = chr(winner_idx)
                    if char != last_char:                # merge held phonemes
                        generated += char
                    last_char = char
                else:
                    last_char = None                     # silence resets hold
            return generated

        else:
            # ── BURST / RATE-CODING ────────────────────────────────────
            # Aggregate how often each neuron fired across all time steps.
            # This gives a firing-rate vector for each neuron.
            mean_rates = rates_np.mean(axis=0)           # [num_neurons]
            printable_rates = mean_rates[32:127]         # only printable ASCII

            # Sort by dominance (highest firing neuron first).
            ranked_local = np.argsort(printable_rates)[::-1]
            generated = ""
            for local_idx in ranked_local:
                if printable_rates[local_idx] > 0.01:   # any activity at all
                    generated += chr(int(local_idx) + 32)
                else:
                    break
                if len(generated) >= 32:                 # reasonable cap
                    break
            return generated


    def decode_to_audio(self, brocas_spikes, filepath="generated_speech.wav", sample_rate=44100):
        """
        Unpacks discrete neural firing rates in language regions back into physical 
        sine-wave amplitudes mapping a synthesized audio track.
        """
        firing_rates = tf.reduce_mean(brocas_spikes, axis=1)[0]
        
        # Audio operates from -1.0 to 1.0 conceptually
        audio_array = (firing_rates.numpy() * 2.0) - 1.0 
        
        # Fill array to reach 1 second of audio time
        repeats = sample_rate // len(audio_array) if len(audio_array) > 0 else sample_rate
        if repeats == 0: repeats = 1
        
        wave = np.repeat(audio_array, repeats)[:sample_rate]
        wave_tensor = tf.convert_to_tensor(wave, dtype=tf.float32)
        wave_tensor = tf.expand_dims(wave_tensor, -1) # Requirements for audio codec: [samples, channels]
        
        # Utilize TensorFlow's native biological encoder trick
        wav_raw = tf.audio.encode_wav(wave_tensor, sample_rate=sample_rate)
        tf.io.write_file(filepath, wav_raw)
        
        print(f">> Generated Biological Audio track saved to {filepath}")
