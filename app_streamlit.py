import sys
import os
import subprocess

# Ensure the root project directory is in the import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Launching Streamlit Annotator...")
    # Form absolute path to ui handler 
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "streamlit_app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
