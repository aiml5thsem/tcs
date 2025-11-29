from flask import Flask, render_template_string, request
from flask_socketio import SocketIO, emit
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'simple-key'
socketio = SocketIO(app, cors_allowed_origins="*")

TEXT_FILE = 'sample.txt'

# Create file if not exists
if not os.path.exists(TEXT_FILE):
    with open(TEXT_FILE, 'w') as f:
        f.write('')

@app.route('/')
def index():
    # Load content server-side, embed in HTML (no /content route)
    with open(TEXT_FILE, 'r') as f:
        content = f.read()
    
    return render_template_string('''
<!DOCTYPE html>
<html>
<head><title>Live Sync</title></head>
<body style="font-family:monospace;margin:20px;">
    <h3>Live Text Sync → sample.txt</h3>
    <textarea id="editor" style="width:100%;height:500px;font-size:14px;font-family:monospace;border:1px solid #ccc;"
              placeholder="Type here... instant sync!">{{ content }}</textarea>
    
    <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
    <script>
        const socket = io();
        const editor = document.getElementById('editor');
        
        // Debounced sync (300ms)
        let timeout;
        editor.addEventListener('input', ()=>{
            clearTimeout(timeout);
            timeout = setTimeout(()=>{
                socket.emit('update', editor.value);
            }, 300);
        });
        
        // Live sync from others
        socket.on('update', (content)=>{
            editor.value = content;
        });
    </script>
</body>
</html>
    ''', content=content)

@socketio.on('update')
def handle_update(data):
    content = data
    with open(TEXT_FILE, 'w') as f:
        f.write(content)
    emit('update', content, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
