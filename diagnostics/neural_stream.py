import socketio
import threading
import time
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

class NeuralStreamer:
    """
    Biological Telemetry Client (v1.1).
    Streams regional synaptic activity from the SNN training/inference loops 
    to the Neuro-Monitoring Dashboard.
    """
    def __init__(self, server_url="http://127.0.0.1:5005"):
        self.server_url = server_url
        self.connected = False
        self.sio = None
        
        # Robust SocketIO instantiation
        try:
            if hasattr(socketio, 'Client'):
                self.sio = socketio.Client()
            else:
                print(">>> [Diagnostics] Warning: socketio.Client not found. Checking for alternative exports...")
                # Some versions/shadowing might export it differently
        except Exception as e:
            print(f">>> [Diagnostics] SocketIO init failure: {e}")

    def _convert_tensors(self, data):
        """Recursively convert Tensors/Numpy to native Python floats for JSON serialization."""
        if isinstance(data, dict):
            return {k: self._convert_tensors(v) for k, v in data.items()}
        
        # Handle TensorFlow Tensors (Symbolic or Eager)
        if tf is not None and hasattr(data, "numpy"):
            try:
                return float(data.numpy())
            except Exception:
                return 0.0 # Fallback for strictly symbolic tensors
        
        # Handle Numpy types
        if isinstance(data, (np.float32, np.float64, np.float16)):
            return float(data)
            
        return data

    def connect(self):
        if not self.sio:
            return
            
        try:
            self.sio.connect(self.server_url)
            self.connected = True
            print(f">>> [Diagnostics] Connected to Neuro-Monitor at {self.server_url}")
        except Exception:
            self.connected = False
            print(f">>> [Diagnostics] Monitor server not detected. GUI disabled.")

    def stream_state(self, regional_activity, context_text="Thinking...", mode="training"):
        if not self.connected or not self.sio:
            return
            
        # Convert all Tensors to floats before sending
        clean_activity = self._convert_tensors(regional_activity)
            
        data = {
            "activity": clean_activity,
            "context": str(context_text),
            "mode": mode,
            "timestamp": time.time()
        }
        
        try:
            self.sio.emit('neural_update', data)
        except Exception:
            pass

    def disconnect(self):
        if self.connected:
            self.sio.disconnect()

# Singleton instance
streamer = NeuralStreamer()
