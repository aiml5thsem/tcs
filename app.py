from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Flask Test App</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; text-align: center; background: #f8f9fa; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        img { max-width: 200px; margin: 20px 0; border-radius: 8px; }
        h1 { color: #333; }
        p { color: #666; font-size: 18px; }
    </style>
</head>
<body>
    <div class="container">
        <img src="{{ url_for('static', filename='icon.png') }}" alt="Icon">
        <h1>👋 Hello! Welcome to Flask Application</h1>
        <p>This is a simple Flask test app.<br>Just learning Flask basics 🚀</p>
        <hr>
        <p><small>Flask version test - 2025</small></p>
    </div>
</body>
</html>
    ''')

if __name__ == '__main__':
    print("🌐 Flask Test App running on http://localhost:5000")
    print("📁 Put your icon.png in 'static/' folder")
    app.run(host='0.0.0.0', port=5000, debug=False)
