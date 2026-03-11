# SegForge

A powerful, interactive, modular tool that helps you generate segmentation datasets from images using the Segment Anything Model (SAM). Runs seamlessly on your local machine, Google Colab, or Kaggle.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oaboelazm/SegForge/blob/main/notebooks/SegForge_Interactive_Notebook.ipynb) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oaboelazm/SegForge/blob/main/notebooks/SegForge_Interactive_Notebook.ipynb)

[**Full English Documentation 📖**](file:///c:/Users/dell/Desktop/Projects/26/SegForge/DOCUMENTATION.md)

## ✨ Capabilities

- **Automated Mask Generation:** Upload images and dynamically generate distinct isolation masks by clicking on the object you want using **Segment Anything (SAM)**.
- **Advanced Selection Modes:** Pinpoint exact features using Positive and Negative point prompts, or draw bounding boxes to enforce strict containment.
- **Magic Mask Refinement:** Enhance fuzzy edges natively by generating algorithmic boundary constraints post-segmentation.
- **Batch Dataset Conversion:** Seamlessly convert entire existing YOLO object detection datasets straight into pixel-perfect segmentation masks using the fully automated bulk processor tab.
- **Dataset Management:** Label each discrete instance individually and review your image's objects interactively overlaying your results.
- **One-Click Export:** Compile everything instantaneously into COCO JSON annotations, YOLO txt polygon formats, and raw `.png` bitmasks bundled cleanly inside a `.zip`.
- **Dual Interfaces:** Features identically synced interfaces across Gradio (built for sharing and cloud work) or Streamlit (ideal for local analysis).

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
├── core/
│   ├── sam_manager.py       # Core SAM inference engine (Points, Multi-mask, Boxes)
│   ├── dataset_exporter.py  # COCO & YOLO semantic format converters
│   ├── mask_utils.py        # Mask Refinement & Post-processing Algorithms
│   └── batch_processor.py   # Bulk Detection -> Segmentation Logic
├── ui/
│   ├── gradio_app.py        # Gradio interface definitions & event controllers
│   └── streamlit_app.py     # Streamlit interface definitions
├── notebooks/
│   └── SegForge_Interactive_Notebook.ipynb # Ready-to-use Colab/Kaggle notebook
├── .gitignore
├── app.py                   # Gradio execution entry point
├── app_streamlit.py         # Streamlit execution entry point
└── requirements.txt         # Dependencies
```
