import tensorflow as tf
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

class SensoryTokenizer:
    """
    Acts as the sensory organs (eyes, ears) and thalamus for SNNs.
    Converts raw world data into mathematical arrays, and injects them
    as constant currents over T discrete time steps.
    """
    def __init__(self, visual_dim=1024, auditory_dim=256):
        self.visual_dim = visual_dim
        self.auditory_dim = auditory_dim

    def process_image(self, image_tensor, time_steps):
        flat = tf.reshape(tf.cast(image_tensor, tf.float32), [-1])
        flat = flat / 255.0
        
        current_len = tf.shape(flat)[0]
        if current_len < self.visual_dim:
            flat = tf.pad(flat, [[0, self.visual_dim - current_len]])
        else:
            flat = flat[:self.visual_dim]
        
        batch_flat = tf.expand_dims(flat, 0) # [1, features]
        # REPLICATE ACROSS TIME: Constant current injection for SNN
        return tf.repeat(tf.expand_dims(batch_flat, 1), time_steps, axis=1)

    def process_video(self, video_frames, time_steps):
        # Maps each physical frame into distinct biological time steps naturally
        processed_frames = []
        for frame in video_frames:
            flat = tf.reshape(tf.cast(frame, tf.float32), [-1]) / 255.0
            current_len = tf.shape(flat)[0]
            if current_len < self.visual_dim:
                flat = tf.pad(flat, [[0, self.visual_dim - current_len]])
            else:
                flat = flat[:self.visual_dim]
            processed_frames.append(flat)
            
        # Ensure we have exactly `time_steps` sequence bounds
        while len(processed_frames) < time_steps:
             processed_frames.append(processed_frames[-1] if len(processed_frames) > 0 else tf.zeros([self.visual_dim]))
             
        processed_frames = processed_frames[:time_steps]
        stacked = tf.stack(processed_frames)
        return tf.expand_dims(stacked, 0)

    def process_audio(self, audio_waveform, time_steps):
        audio = tf.cast(audio_waveform, tf.float32)
        current_len = tf.shape(audio)[0]
        if current_len < self.auditory_dim:
            audio = tf.pad(audio, [[0, self.auditory_dim - current_len]])
        else:
            audio = audio[:self.auditory_dim]
            
        batch_audio = tf.expand_dims(audio, 0)
        return tf.repeat(tf.expand_dims(batch_audio, 1), time_steps, axis=1)

    def process_text_visually(self, text_string, time_steps):
        """
        BIOMIMETIC READING: Renders text as a rolling visual sequence.
        This allows the Visual Cortex to scan characters over time.
        """
        if cv2 is None:
            # Fallback to zeros if cv2 missing
            return tf.zeros((1, time_steps, self.visual_dim), dtype=tf.float32)

        # Create a single long strip for the entire text
        char_w = 25
        canvas_width = max(64, len(text_string) * char_w + 128)
        canvas_height = 64
        strip = np.zeros((canvas_height, canvas_width, 1), dtype=np.uint8)
        
        # Draw the text centrally on the strip with a more legible biological scale
        # 0.8 scale fits perfectly in the 64px vertical space
        cv2.putText(strip, text_string, (64, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
        
        processed_frames = []
        # The text scrolls from right to left across the 64x64 window
        total_scroll = canvas_width - 64
        shift_per_step = total_scroll / time_steps if time_steps > 1 else 0
        
        for t in range(time_steps):
            x_start = int(t * shift_per_step)
            frame_gray = strip[:, x_start:x_start+64]
            # Convert to RGB to match the visual cortex input format
            frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
            
            flat = tf.reshape(tf.cast(frame_rgb, tf.float32), [-1]) / 255.0
            current_len = tf.shape(flat)[0]
            if current_len < self.visual_dim:
                flat = tf.pad(flat, [[0, self.visual_dim - current_len]])
            else:
                flat = flat[:self.visual_dim]
            processed_frames.append(flat)
            
        stacked = tf.stack(processed_frames)
        return tf.expand_dims(stacked, 0)

    def process_text_as_audio(self, text_string, time_steps):
        """
        BIOMIMETIC SEQUENTIAL SUBVOCALIZATION:
        Maps each character to its own dedicated time step, mirroring how Broca's
        area fires motor commands for phonemes one at a time, not all at once.
        At time step T, only the neuron whose index equals the ASCII value of the
        T-th character fires. This is the correct biological target for the decoder.
        
        Shape: [1, time_steps, auditory_dim]
        """
        # One spike per time step: spikes[t, ascii(char_t)] = 1.0
        spikes = np.zeros((time_steps, self.auditory_dim), dtype=np.float32)
        for t, char in enumerate(text_string):
            if t >= time_steps:
                break
            ascii_val = ord(char)
            if ascii_val < self.auditory_dim:
                spikes[t, ascii_val] = 1.0

        return tf.expand_dims(tf.convert_to_tensor(spikes), 0)  # [1, time_steps, auditory_dim]

    def thalamic_routing(self, sensory_type, raw_data, time_steps=30):
        if sensory_type == "vision":
            return self.process_image(raw_data, time_steps)
        elif sensory_type == "video":
            return self.process_video(raw_data, time_steps)
        elif sensory_type == "audio":
            return self.process_audio(raw_data, time_steps)
        elif sensory_type == "text":
            return self.process_text_visually(raw_data, time_steps)
        elif sensory_type == "audio_text":
            return self.process_text_as_audio(raw_data, time_steps)
        else:
            raise ValueError("Unknown sensory type.")
