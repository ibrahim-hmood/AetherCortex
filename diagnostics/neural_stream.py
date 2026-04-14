import socketio
import threading
import time
import numpy as np
import base64
import cv2

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
        
        # Handle Numpy types (floats and ints)
        if isinstance(data, (np.float32, np.float64, np.float16, np.int64, np.int32, np.uint8)):
            return float(data) if "float" in str(type(data)) else int(data)
            
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

    def stream_state(self, regional_activity, permanence_map=None, context_text="", mode="inference", retinal_proto=None, validation_metrics=None, vocab_health=None):
        if not self.connected: 
            return
        
        # v4.4: Pack Validation Metrics
        self.last_validation = validation_metrics or getattr(self, 'last_validation', {"good": 0, "bad": 0, "total": 0})
        self.last_vocab = vocab_health or getattr(self, 'last_vocab', {})
            
        # Convert all Tensors to floats before sending
        # v4.5: Robustness fix - ensure these are at least empty dicts to prevent UI bailout
        clean_activity = self._convert_tensors(regional_activity) if regional_activity else {}
        clean_permanence = self._convert_tensors(permanence_map) if permanence_map else {}
        
        # v0.2.8: Thalamic Retinal Encoding (Dual-Feed Support)
        retinal_base64 = None
        raw_base64 = None
        
        if retinal_proto is not None:
            try:
                # Handle dictionary input (Dual Feed) or single tensor
                if isinstance(retinal_proto, dict):
                    gated_img = retinal_proto.get('gated')
                    raw_img = retinal_proto.get('raw')
                else:
                    gated_img = retinal_proto
                    raw_img = None

                def encode_retina(proto):
                    if proto is None: return None
                    img_array = np.squeeze(proto)
                    
                    # v0.2.8: Dynamic Range Stretching (Min-Max Scaling)
                    # Ensures faint signals are visible on the dashboard
                    f_min, f_max = np.min(img_array), np.max(img_array)
                    if f_max > f_min:
                        img_array = (img_array - f_min) / (f_max - f_min)
                    
                    img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
                    _, buffer = cv2.imencode('.png', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                    return base64.b64encode(buffer).decode('utf-8')

                retinal_base64 = encode_retina(gated_img)
                raw_base64 = encode_retina(raw_img)
            except Exception:
                pass
            
        data = {
            "activity": clean_activity,
            "permanence": clean_permanence,
            "context": str(context_text),
            "mode": mode,
            "retinal_feed": retinal_base64,
            "raw_feed": raw_base64,
            "validation": self._convert_tensors(self.last_validation),
            "vocabulary": self._convert_tensors(self.last_vocab),
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
