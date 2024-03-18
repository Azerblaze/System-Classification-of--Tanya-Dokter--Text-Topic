import webbrowser
import subprocess

# Run the Python program
subprocess.Popen(["python", "app.py"])

# Open browser to localhost:5000
webbrowser.open("http://localhost:5000")