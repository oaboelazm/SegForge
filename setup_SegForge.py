import subprocess
import sys
import os

def install_requirements():
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print(f"Error: {req_file} not found.")
        return

    print("🚀 Starting SegForge Environment Setup...")
    
    with open(req_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

    total = len(lines)
    print(f"Found {total} dependencies to install/check.\n")

    for i, line in enumerate(lines):
        print(f"[{i+1}/{total}] Installing: {line}...")
        try:
            # We use -q to keep it clean, but let the user see the current package
            subprocess.check_call([sys.executable, "-m", "pip", "install", line, "--quiet"])
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing {line}: {e}")
            continue

    print("\n✅ Environment setup complete! You can now run 'python app.py' or 'streamlit run app_streamlit.py'.")

if __name__ == "__main__":
    install_requirements()
