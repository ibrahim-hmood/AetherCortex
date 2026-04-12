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

    def __biological_processor(self, img, bio_train_mode, epoch, fovea_offset):
        if not bio_train_mode:
            return tf.image.resize(img, self.target_size)
            
        # === ACTIVE INFERENCE ===
        # Fovea offset selects a smaller (64x64) patch of the optical field (128x128).
        field_size = (self.target_size[0] * 2, self.target_size[1] * 2)
        field_img = tf.image.resize(img, field_size)
        
        # Convert fovea_offset [-1.0, 1.0] to pixel offsets centrally
        max_y = field_size[0] - self.target_size[0]
        max_x = field_size[1] - self.target_size[1]
        
        # Clip just in case the motor strip sends erratic spikes
        f_x = tf.clip_by_value(fovea_offset[0], -1.0, 1.0)
        f_y = tf.clip_by_value(fovea_offset[1], -1.0, 1.0)
        
        y_off = int((f_y + 1.0) / 2.0 * max_y)
        x_off = int((f_x + 1.0) / 2.0 * max_x)
        
        crop = tf.image.crop_to_bounding_box(field_img, y_off, x_off, self.target_size[0], self.target_size[1])
        
        # === EXTREME CURRICULUM LEARNING ===
        # Throttle visual capacity early in development
        curriculum_res = max(4, min(self.target_size[0], int(epoch * 4.0)))
        
        # Blur the information down, then scale it back to identical geometric neural dimensions
        blurry = tf.image.resize(crop, (curriculum_res, curriculum_res))
        return tf.image.resize(blurry, self.target_size)

    def load_image(self, filepath, bio_train_mode=False, epoch=1, fovea_offset=(0.0, 0.0)):
        """Loads a generic PNG/JPEG directly into an RGB tensor."""
        if not os.path.exists(filepath):
            return tf.zeros((self.target_size[0], self.target_size[1], 3), dtype=tf.float32)
            
        raw = tf.io.read_file(filepath)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        return self.__biological_processor(img, bio_train_mode, epoch, fovea_offset)

    def load_video_frames(self, filepath, max_frames=2, bio_train_mode=False, epoch=1, fovea_offset=(0.0, 0.0)):
        """Loads physical MP4 frames. Needs cv2."""
        if cv2 is None or not os.path.exists(filepath):
            return [tf.zeros((self.target_size[0], self.target_size[1], 3), dtype=tf.float32) for _ in range(max_frames)]

        frames = []
        cap = cv2.VideoCapture(filepath)
        count = 0
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = tf.convert_to_tensor(rgb_frame, dtype=tf.float32)
            processed = self.__biological_processor(tensor_frame, bio_train_mode, epoch, fovea_offset)
            frames.append(processed)
            count += 1
        cap.release()
        
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
