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
    def __init__(self, visual_dim=1024, auditory_dim=300):
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
        BIOMIMETIC READING: Renders text centered on a 128x128 canvas.
        This allows the Visual Cortex to scan characters immediately.
        """
        if cv2 is None:
            # Fallback to zeros if cv2 missing
            return tf.zeros((1, time_steps, self.visual_dim), dtype=tf.float32)

        # Create a static 128x128 canvas
        canvas_h, canvas_w = 128, 128
        frame = np.zeros((canvas_h, canvas_w, 1), dtype=np.uint8)
        
        # Calculate centering coordinates (v4.5: High Contrast Boost)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.3
        thickness = 3
        (text_w, text_h), baseline = cv2.getTextSize(text_string, font, font_scale, thickness)
        
        # Center horizontally and vertically
        text_x = (canvas_w - text_w) // 2
        text_y = (canvas_h + text_h) // 2
        
        cv2.putText(frame, text_string, (text_x, text_y), font, font_scale, 255, thickness)
        
        # Convert to RGB and normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        flat = tf.reshape(tf.cast(frame_rgb, tf.float32), [-1]) / 255.0
        
        # Static reading: Use the same frame for all time steps
        current_len = tf.shape(flat)[0]
        # v4.4: visual_dim is now 128*128*3 = 49152
        if current_len < self.visual_dim:
            flat = tf.pad(flat, [[0, self.visual_dim - current_len]])
        else:
            flat = flat[:self.visual_dim]
            
        processed_frame = tf.expand_dims(flat, 0)
        return tf.repeat(tf.expand_dims(processed_frame, 0), time_steps, axis=1)

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
