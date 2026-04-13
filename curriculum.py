import os
import glob
import random

class SensoryCurriculum:
    """
    Manages the developmental progression of the SNN.
    Stabilizes infancy through simpler visual-token mappings before
    progressing to complex multimodal sequencing.
    """
    LEVEL_FLASHCARDS = 1  # Single image -> Single token (Filename)
    LEVEL_READING = 2     # Continuous text sliding window
    
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.current_level = self.LEVEL_FLASHCARDS
        self.stability_counter = 0
        self.required_stable_steps = 100
        self.loss_threshold = 50.0

    def get_stage_data(self, basenames):
        """
        Filters the dataset basenames based on the current developmental maturity.
        """
        if self.current_level == self.LEVEL_FLASHCARDS:
            # Flashcards focus on files that have images
            flashcards = []
            for base in basenames:
                if os.path.exists(os.path.join(self.dataset_dir, f"{base}.png")) or os.path.exists(os.path.join(self.dataset_dir, f"{base}.jpg")) or os.path.exists(os.path.join(self.dataset_dir, f"{base}.jpeg")):
                    flashcards.append(base)
            return flashcards
        else:
            # Levels above flashcards use the full corpus
            return basenames

    def report_step(self, loss):
        """
        Updates maturity based on performance.
        Refined v0.1.9: Reset and cap logic.
        """
        if loss < self.loss_threshold:
            self.stability_counter = min(self.required_stable_steps, self.stability_counter + 1)
        else:
            # Regression: If loss spikes, we stay at the current level longer
            self.stability_counter = max(0, self.stability_counter - 1)
            
        if self.stability_counter >= self.required_stable_steps:
            if self.current_level < self.LEVEL_READING:
                self.current_level += 1
                self.stability_counter = 0 # CRITICAL RESET
                print(f"\n>>> [Curriculum] LEVEL UP! Brain has matured to Level {self.current_level}.")
                return True
        return False

    def get_status(self):
        stage_name = "Flashcards" if self.current_level == self.LEVEL_FLASHCARDS else "Reading"
        display_counter = min(self.stability_counter, self.required_stable_steps)
        progress = (display_counter / self.required_stable_steps) * 100
        return f"Stage: {stage_name} ({progress:.1f}% toward next Level)"
