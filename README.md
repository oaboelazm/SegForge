# SegForge

A powerful, interactive, modular tool that helps you generate segmentation datasets from images using the Segment Anything Model (SAM). Runs seamlessly on your local machine, Google Colab, or Kaggle.

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oaboelazm/SegForge/blob/main/notebooks/SegForge_Interactive_Notebook.ipynb) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oaboelazm/SegForge/blob/main/notebooks/SegForge_Interactive_Notebook.ipynb)

## ✨ Features

1. **Interactive Segmentation Engine**: Point-and-click segmentation applying Meta's SAM. Add positive object clicks and negative background exclusion clicks.
2. **Dataset Export Zip**: Generates standard deep learning structures containing the original image, pure binary mask representations, normalized YOLO txt outputs, and unified COCO annotations.
3. **Modular and Maintainable**: Written with a clean split distinguishing the neural network inference model from the user interface and the bounding geometry processing.

## 🚀 Quickstart (Local)

1. Clone the repository:

```bash
git clone https://github.com/oaboelazm/SegForge.git
cd SegForge
```

2. Install Requirements:

```bash
pip install -r requirements.txt
```

3. Launch App:

**Option A (Gradio - Default):**

```bash
python app.py
```

A local interface URL and a public shareable URL will be generated!

**Option B (Streamlit - Alternative):**

```bash
python app_streamlit.py
```

A local Streamlit interface URL will be generated!

## ☁️ Cloud 1-Click Installation

Simply click on the **Open In Colab** or **Kaggle** badges at the top of the repository to initialize the notebook script directly!

## 📂 Modular Structure

```
SegForge/
│
├── core/
│   ├── dataset_exporter.py    # YOLO/COCO translation and bundle export
│   ├── mask_utils.py          # Bounding box & Polygon algebra
│   └── sam_manager.py         # SAM weight loader & Torch execution
│
├── notebooks/
│   └── SegForge_Interactive_Notebook.ipynb # Cloud entrypoint
│
├── ui/
│   ├── gradio_app.py          # Gradio interface component layout
│   └── streamlit_app.py       # Streamlit interface component layout
│
├── app.py                     # Primary Application Wrapper (Gradio)
├── app_streamlit.py           # Alternative Application Wrapper (Streamlit)
├── requirements.txt           # Pip dependencies
└── README.md
```
