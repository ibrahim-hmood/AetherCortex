from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
import os

app = Flask(__name__, template_folder='web', static_folder='web')
app.config['SECRET_KEY'] = 'biological_secret_key'
# Use 'threading' async mode to avoid OpenSSL/Eventlet compatibility issues in WSL
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# The path to the generated brain image
BRAIN_IMAGE_PATH = r"C:\Users\bremo\.gemini\antigravity\brain\1696faa3-7f61-4aa1-b87d-25df1dbd2bbe\human_brain_lateral_view_1776043729451.png"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/saccadic_drawing.png')
def get_saccadic_drawing():
    return send_from_directory(os.getcwd(), "saccadic_drawing.png")

@app.route('/deep_dream.png')
def get_deep_dream():
    return send_from_directory(os.getcwd(), "deep_dream.png")

@socketio.on('neural_update')
def handle_neural_update(data):
    # Broadcast the neural state to all connected dashboard clients
    socketio.emit('ui_update', data)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("      NEURO-MONITORING DASHBOARD (v1.0)")
    print("      Biological Visualization Active")
    print("      Access at: http://127.0.0.1:5005")
    print("="*50 + "\n")
    socketio.run(app, host="0.0.0.0", port=5005, debug=False)
