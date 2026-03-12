# SegForge Documentation 🚀

Welcome to the official documentation for **SegForge**, a modular and interactive dataset generation tool powered by the **Segment Anything Model (SAM2.1)**.

---

## 📖 Introduction

SegForge is designed to bridge the gap between simple object detection and high-quality semantic segmentation. Whether you are starting from scratch or already have a labeled detection dataset, SegForge provides the tools to generate pixel-perfect masks with minimal effort.

---

## 🏗️ Architecture Overview

The project is built with a modular architecture to ensure maintainability and scalability:

- **`core/`**: The brain of the application.
  - `sam_manager.py`: Handles SAM 2.1 model loading, point/box inference, and confidence-based mask selection.
  - `batch_processor.py`: Automates the conversion of YOLO detection datasets into segmentation masks.
  - `dataset_exporter.py`: Logic for exporting results into COCO JSON, YOLO Polygon TXT, and PNG formats.
  - `mask_utils.py`: Contains morphological post-processing scripts and refinement point generation algorithms.
- **`ui/`**: User interface definitions.
  - `gradio_app.py`: Layout for the Gradio-based interface (ideal for cloud/sharing).
  - `streamlit_app.py`: Layout for the Streamlit-based interface (ideal for local/heavy usage).
- **`app.py` / `app_streamlit.py`**: Entry points for launching the respective UIs.
- **`setup_project.py`**: Dedicated script for installing dependencies with real-time feedback.

---

## 🔥 Key Features

### 1. Interactive Annotation

Segment specific objects in your images using multiple prompt modes:

- **Positive Points (Green)**: Mark areas that belong to the object.
- **Negative Points (Red)**: Mark areas to exclude from the mask.
- **Bounding Box (Blue)**: Draw a box around the object to guide SAM.

### 2. Magic Mask Refinement

If a generated mask is slightly "fuzzy" or spills over the edges, use the **Refine Mask** button. This algorithm automatically detects the mask's boundary and places negative points just outside it to sharpen the edges.

### 3. CPU Optimization

SegForge automatically detects your hardware and scales the model. It uses the **Tiny** SAM 2.1 variant on CPU for lag-free performance and the **Large** variant on GPU for maximum precision.

### 4. Morphological Post-processing

Every mask is automatically cleaned using:

- **Closing**: Fills small holes within the mask.
- **Opening**: Removes tiny noise artifacts around the edges.
- **Hole Filling**: Ensures the object is a solid, continuous shape.

### 4. Batch Detection to Segmentation

Transform an entire **YOLO Object Detection** dataset into a **Segmentation** dataset:

1. Upload a ZIP containing `/images` and `/labels`.
2. SegForge reads the YOLO bounding boxes.
3. SAM processes each box to find the exact mask.
4. Download the new dataset in COCO/YOLO/PNG formats.

---

## 🛠️ Usage Guide

### Getting Started Locally

1. **Clone the repo.**
2. **Setup the environment**:
   Use the provided setup script to see installation progress:
   ```bash
   python setup_SegForge.py
   ```
3. **Run the app**:
   - Gradio: `python app.py`
   - Streamlit: `streamlit run app_streamlit.py`

> [!IMPORTANT]
> **Weight Download Progress**: On the first run, SegForge will download the SAM 2.1 weights (~800MB). A `tqdm` progress bar will appear in your console showing the download speed and estimated time remaining.

### Interactive Mode Workflow

1. **Upload** one or more images.
2. **Select** an image from the dropdown/list.
3. **Choose a mode**: Positive Point, Negative Point, or Bounding Box.
4. **Click/Drag** on the image to add prompts.
5. Click **Generate Mask**.
6. (Optional) Click **Refine Mask** to sharpen edges.
7. Enter a **Class Label** and click **Save Object & Mask**.
8. Once finished with all images, click **Export Dataset**.

### Batch Mode Workflow

1. Navigate to the **Batch Conversion** tab.
2. Upload a **.zip** file with this structure:
   ```text
   dataset.zip
   ├── images/
   │   ├── img1.jpg
   │   └── img2.jpg
   └── labels/
   │   ├── img1.txt (YOLO format: class x_center y_center width height)
   │   └── img2.txt
   └── classes.txt (Optional: List of class names, one per line)
   ```
3. Click **Start Bulk Conversion**.
4. Review the **Visual Validation Gallery** to check accuracy.
5. Download the final segmented dataset ZIP.

---

## 📂 Export Formats

SegForge exports a complete package ready for training:

- **COCO JSON**: Standard format for most segmentation frameworks.
- **YOLO Seg**: Normalized polygon coordinates for YOLOv8/v9/v10/v11 segmentation.
- **PNG Masks**: Raw binary masks (grayscale) for each image.
- **Dataset Info**: A `classes.txt` file mapping IDs to names.

---

## ⚙️ Requirements

- Python 3.9+
- PyTorch (CUDA recommended, PyTorch 2.3+ for SAM 2.1)
- SAM 2.1 (`sam2` package)
- OpenCV
- Gradio or Streamlit

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the refinement algorithms or add new export formats.
