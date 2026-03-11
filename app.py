import sys
import os

# Ensure the root project directory is in the import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.gradio_app import create_app

if __name__ == "__main__":
    app = create_app()
    print("Launching SAM Annotator...")
    # Using share=True to guarantee a public huggingface URL useful for Kaggle/Colab!
    app.launch(debug=True, share=True)
