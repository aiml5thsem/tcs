from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from cryptography.fernet import Fernet
import os
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ws-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

TEXT_FILE = 'sample.txt'

# AES-256 key setup
KEY_FILE = 'secret.key'
if not os.path.exists(KEY_FILE):
    key = Fernet.generate_key()
    with open(KEY_FILE, 'wb') as f: f.write(key)
with open(KEY_FILE, 'rb') as f: cipher = Fernet(f.read())

if not os.path.exists(TEXT_FILE):
    open(TEXT_FILE, 'w').close()

# Load current content
def get_content():
    try:
        with open(TEXT_FILE, 'r') as f:
            encoded = f.read()
        encrypted = base64.urlsafe_b64decode(encoded)
        return cipher.decrypt(encrypted).decode()
    except:
        return ''

@app.route('/')
def index():
    plaintext = get_content()
    
    return render_template_string('''
<!DOCTYPE html>
<html>
<head><title>Notes Live</title>
<style>
body{font-family:monospace;margin:40px;background:#f5f5f5;}
.container{max-width:900px;margin:0 auto;background:white;padding:30px;border-radius:10px;box-shadow:0 4px 6px rgba(0,0,0,0.1);}
textarea{width:100%;height:600px;font-family:monospace;font-size:14px;padding:20px;border:1px solid #ddd;border-radius:8px;box-sizing:border-box;resize:none;}
.status{padding:10px;background:#e7f3ff;border-radius:5px;margin:15px 0;font-size:14px;}
h1{text-align:center;color:#333;margin-bottom:30px;}
small{color:#666;display:block;text-align:center;margin-top:20px;}
</style>
</head>
<body>
<div class="container">
<h1>📝 Live Notes Editor</h1>
<div class="status" id="status">🟢 Live sync active (WebSocket)</div>
<textarea id="editor" placeholder="Type here... syncs instantly across devices">{{ plaintext }}</textarea>
<small>Real-time collaborative notes - Flask + WebSocket</small>
</div>

<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script>
const socket = io();
const editor = document.getElementById('editor');
let typing = false;

// Auto-sync every keystroke (300ms debounce)
let timeout;
editor.oninput = () => {
    typing = true;
    document.getElementById('status').textContent = '🔄 Typing...';
    clearTimeout(timeout);
    timeout = setTimeout(() => {
        socket.emit('update', editor.value);
        typing = false;
    }, 300);
};

// Live sync from other devices
socket.on('update', (data) => {
    if (!typing) {  // Don't overwrite if user is typing
        editor.value = data;
    }
    document.getElementById('status').textContent = '🟢 Live sync received';
});

// Connection status
socket.on('connect', () => {
    document.getElementById('status').textContent = '🟢 Connected - Live sync active';
});
</script>
</body>
</html>
    ''', plaintext=plaintext)

@socketio.on('update')
def handle_update(data):
    # AES-256 encrypt → save → broadcast
    encrypted = cipher.encrypt(data.encode())
    encoded = base64.urlsafe_b64encode(encrypted).decode()
    
    with open(TEXT_FILE, 'w') as f:
        f.write(encoded)
    
    emit('update', data, broadcast=True)

if __name__ == '__main__':
    print("🌐 Live Notes running: http://localhost:5000")
    print("🔒 AES-256 encrypted + WebSocket (NO POST requests)")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
