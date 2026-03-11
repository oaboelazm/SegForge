import os
import sys
import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

# Ensure path includes project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sam_manager import SAMManager
from core.dataset_exporter import DatasetExporter
from core.mask_utils import COLORS

st.set_page_config(layout="wide", page_title="SegForge SAM Annotator")

TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@st.cache_resource
def get_sam_manager():
    return SAMManager()

@st.cache_resource
def get_exporter():
    return DatasetExporter()

sam_manager = get_sam_manager()
exporter = get_exporter()

def render_image(image_np, saved_objects, current_mask, points, labels):
    if image_np is None:
        return None
    overlay = image_np.copy()
    for i, obj in enumerate(saved_objects):
        mask = obj["mask"]
        color = np.array(COLORS[i % len(COLORS)], dtype=np.uint8)
        mask_indices = mask > 0
        overlay[mask_indices] = overlay[mask_indices] * 0.7 + color * 0.3
        
    if current_mask is not None:
        color = np.array([0, 255, 0], dtype=np.uint8)
        mask_indices = current_mask > 0
        overlay[mask_indices] = overlay[mask_indices] * 0.5 + color * 0.5
        
    for pt, lbl in zip(points, labels):
        c = (0, 255, 0) if lbl == 1 else (255, 0, 0)
        cv2.circle(overlay, (pt[0], pt[1]), 5, c, -1)
        cv2.circle(overlay, (pt[0], pt[1]), 2, (255, 255, 255), -1)
        
    return overlay

st.title("🧠 Segment Anything Model (SAM) - Interactive Dataset Generator")
st.markdown("Upload images, dynamically segment objects via points, assign class labels, and export datasets directly to your device.")

# Initialize session state variables
if "dataset" not in st.session_state:
    st.session_state.dataset = {}
if "points" not in st.session_state:
    st.session_state.points = []
if "labels" not in st.session_state:
    st.session_state.labels = []
if "current_mask" not in st.session_state:
    st.session_state.current_mask = None
if "current_filepath" not in st.session_state:
    st.session_state.current_filepath = None
if "image_np" not in st.session_state:
    st.session_state.image_np = None

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 1. Upload & Select Images")
    uploaded_files = st.file_uploader("Upload Local Images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
    
    filepaths = []
    if uploaded_files:
        for file in uploaded_files:
            filepath = os.path.join(TEMP_UPLOAD_DIR, file.name)
            # Only dump file if it doesn't already exist or it was updated
            if not os.path.exists(filepath):
                with open(filepath, "wb") as f:
                    f.write(file.getbuffer())
                    
            if filepath not in filepaths:
                filepaths.append(filepath)
                
            if filepath not in st.session_state.dataset:
                st.session_state.dataset[filepath] = {"objects": []}
                
    st.markdown(f"Total images tracked in instance: {len(st.session_state.dataset.keys())}")
    
    selected_file = st.selectbox(
        "Select Opened Image to Annotate", 
        list(st.session_state.dataset.keys()), 
        format_func=lambda x: os.path.basename(x) if x else ""
    )
    
    if selected_file != st.session_state.current_filepath:
        st.session_state.current_filepath = selected_file
        st.session_state.points = []
        st.session_state.labels = []
        st.session_state.current_mask = None
        
        if selected_file and os.path.exists(selected_file):
            img = cv2.imread(selected_file)
            st.session_state.image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with st.spinner(f"Setting image embeddings for {os.path.basename(selected_file)}..."):
                sam_manager.set_image(st.session_state.image_np)
        else:
            st.session_state.image_np = None

    st.markdown("### 2. Segmentation Settings")
    point_type = st.radio("Click Type", ["Positive (Object)", "Negative (Background)"])
    if st.button("Clear Current Clicks"):
        st.session_state.points = []
        st.session_state.labels = []
        st.session_state.current_mask = None
        st.rerun()
        
    st.markdown("### 3. Mask Management")
    class_name = st.text_input("Class Label", placeholder="e.g., car, person, dog")
    if st.button("Save Object & Mask", type="primary"):
        if st.session_state.current_mask is None or len(st.session_state.points) == 0:
            st.warning("No mask detected! Click on the image first to segment.")
        elif not class_name:
            st.warning("Enter a class label first!")
        else:
            st.session_state.dataset[st.session_state.current_filepath]["objects"].append({
                "mask": st.session_state.current_mask,
                "class_name": class_name
            })
            st.session_state.points = []
            st.session_state.labels = []
            st.session_state.current_mask = None
            st.success(f"Saved mask for class '{class_name}'.")
            st.rerun()
            
    st.markdown("### 4. Export")
    if st.button("Prepare Dataset Export (ZIP)", type="primary"):
        if not st.session_state.dataset:
            st.warning("Dataset is empty. Please upload images and annotate first.")
        else:
            with st.spinner("Preparing export bundle..."):
                zip_path = exporter.export(st.session_state.dataset)
            with open(zip_path, "rb") as fp:
                st.download_button(
                    label="⬇️ Download Output Dataset ZIP",
                    data=fp,
                    file_name="dataset_export.zip",
                    mime="application/zip",
                )

with col2:
    if st.session_state.image_np is not None:
        saved_objects = st.session_state.dataset.get(st.session_state.current_filepath, {}).get("objects", [])
        
        # Render composite layout
        rendered_image = render_image(
            st.session_state.image_np,
            saved_objects,
            st.session_state.current_mask,
            st.session_state.points,
            st.session_state.labels
        )
        
        st.markdown("**Interactive View (Click to segment)**")
        # Ensure we capture clicks dynamically using length of points as a cache invalidate-key
        coords = streamlit_image_coordinates(rendered_image, key=f"img_coords_{len(st.session_state.points)}")
        
        if coords is not None:
            st.session_state.points.append([coords['x'], coords['y']])
            st.session_state.labels.append(1 if "Positive" in point_type else 0)
            st.session_state.current_mask = sam_manager.predict(st.session_state.points, st.session_state.labels)
            st.rerun()
            
        st.markdown("**Saved Objects for Current Image**")
        if saved_objects:
            obj_data = [{"ID": i, "Class Name": obj["class_name"]} for i, obj in enumerate(saved_objects)]
            st.table(obj_data)
            
            del_id = st.number_input("Object ID to Delete", min_value=0, max_value=len(saved_objects)-1, step=1)
            if st.button("Delete Mask by ID"):
                st.session_state.dataset[st.session_state.current_filepath]["objects"].pop(int(del_id))
                st.rerun()
        else:
            st.info("No objects saved yet for this image.")
    else:
        st.info("Upload and select an image to start annotating.")
