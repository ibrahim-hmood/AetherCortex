import tensorflow as tf

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

    def process_text_as_audio(self, text_string, time_steps):
        ascii_vals = [float(ord(c)) for c in text_string]
        simulated_audio = tf.constant(ascii_vals, dtype=tf.float32) / 255.0 
        return self.process_audio(simulated_audio, time_steps)

    def thalamic_routing(self, sensory_type, raw_data, time_steps=15):
        if sensory_type == "vision":
            return self.process_image(raw_data, time_steps)
        elif sensory_type == "video":
            return self.process_video(raw_data, time_steps)
        elif sensory_type == "audio":
            return self.process_audio(raw_data, time_steps)
        elif sensory_type == "text":
            return self.process_text_as_audio(raw_data, time_steps)
        else:
            raise ValueError("Unknown sensory type.")
