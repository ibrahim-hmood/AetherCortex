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
    def __init__(self, visual_decode_shape=(32, 32, 3)):
        self.visual_decode_shape = visual_decode_shape
        self.flat_visual_dim = visual_decode_shape[0] * visual_decode_shape[1] * visual_decode_shape[2]

    def decode_to_image(self, visual_cortex_spikes, filepath="generated_output.png"):
        """
        Collapses structural visual spikes across time into a single static image frame.
        """
        firing_rates = tf.reduce_mean(visual_cortex_spikes, axis=1)[0]
        
        if tf.shape(firing_rates)[0] < self.flat_visual_dim:
            padded = tf.pad(firing_rates, [[0, self.flat_visual_dim - tf.shape(firing_rates)[0]]])
        else:
            padded = firing_rates[:self.flat_visual_dim]
            
        img_tensor = tf.reshape(padded, self.visual_decode_shape)
        img_array = tf.cast(img_tensor * 255.0, tf.uint8).numpy()
        
        if cv2 is not None:
            bgr_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, bgr_array)
            print(f">> Generated Static Image saved to {filepath}")
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
        
        for t in range(time_steps):
            frame_rates = visual_cortex_spikes[batch_idx, t, :]
            
            if tf.shape(frame_rates)[0] < self.flat_visual_dim:
                padded = tf.pad(frame_rates, [[0, self.flat_visual_dim - tf.shape(frame_rates)[0]]])
            else:
                padded = frame_rates[:self.flat_visual_dim]
                
            img_tensor = tf.reshape(padded, self.visual_decode_shape)
            img_array = tf.cast(img_tensor * 255.0, tf.uint8).numpy()
            bgr_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Neuromuscular upscaling to preserve blocky shapes on the mp4
            upscaled_frame = cv2.resize(bgr_array, (upscale_dim, upscale_dim), interpolation=cv2.INTER_NEAREST)
            out.write(upscaled_frame)
            
        out.release()
        print(f">> Generated Animation ({time_steps} frames) saved to {filepath}")

    def decode_to_text(self, brocas_spikes):
        """Translates language spikes back to ASCII text."""
        firing_rates = tf.reduce_mean(brocas_spikes, axis=1)[0]
        ascii_values = tf.cast(firing_rates * 255.0, tf.int32).numpy()
        
        generated_string = ""
        for val in ascii_values:
            if 32 <= val <= 126:
                generated_string += chr(val)
        return generated_string

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
