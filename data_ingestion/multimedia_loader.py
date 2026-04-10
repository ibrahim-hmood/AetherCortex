import tensorflow as tf
import os
try:
    import cv2
except ImportError:
    cv2 = None

class MultimediaLoader:
    """
    Ingests physical files from disk directly into raw TensorFlow numeric arrays.
    These arrays are then passed to the SensoryTokenizer.
    """
    def __init__(self, visual_target_size=(32, 32)):
        self.target_size = visual_target_size

    def load_image(self, filepath):
        """Loads a generic PNG/JPEG directly into an RGB tensor."""
        if not os.path.exists(filepath):
            return tf.zeros((self.target_size[0], self.target_size[1], 3), dtype=tf.float32)
            
        raw = tf.io.read_file(filepath)
        # Decode and ensure 3 channels
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        # Scaled to physical dimensional limits
        img = tf.image.resize(img, self.target_size)
        return img

    def load_video_frames(self, filepath, max_frames=2):
        """Loads physical MP4 frames. Needs cv2."""
        if cv2 is None or not os.path.exists(filepath):
            # Fallback to generating simulated blank frames if cv2 uninstalled or file missing
            return [tf.zeros((self.target_size[0], self.target_size[1], 3), dtype=tf.float32) for _ in range(max_frames)]

        frames = []
        cap = cv2.VideoCapture(filepath)
        count = 0
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV loads as BGR, convert to biological RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = tf.convert_to_tensor(rgb_frame, dtype=tf.float32)
            resized = tf.image.resize(tensor_frame, self.target_size)
            frames.append(resized)
            count += 1
        cap.release()
        
        # Pad if video had fewer frames than expected
        while len(frames) < max_frames:
            frames.append(tf.zeros((self.target_size[0], self.target_size[1], 3), dtype=tf.float32))
            
        return frames

    def load_audio(self, filepath):
        """Loads raw audio. Simplified by returning a dummy tensor for architecture sake."""
        # Biologically, we would decode the WAV via librosa. 
        # Here we return a shaped raw audio array simulation.
        return tf.random.normal((500,))
        
    def load_text(self, filepath):
        if not os.path.exists(filepath):
            return "Simulated physical text read from empty file."
        with open(filepath, 'r') as f:
            return f.read()
