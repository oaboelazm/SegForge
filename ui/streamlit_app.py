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
from core.mask_utils import COLORS, postprocess_mask, get_refinement_points
from core.batch_processor import process_batch_yolo

# Global Page Config (MUST BE FIRST)
st.set_page_config(page_title="SegForge - SAM Dataset Engine", layout="wide")

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

def render_image(image_np, saved_objects, current_mask, points, labels, box=None):
    if image_np is None:
        return None
    overlay = image_np.copy()
    
    # Draw saved objects
    for i, obj in enumerate(saved_objects):
        mask = obj["mask"]
        color = np.array(COLORS[i % len(COLORS)], dtype=np.uint8)
        mask_indices = mask > 0
        overlay[mask_indices] = overlay[mask_indices] * 0.7 + color * 0.3
        
    # Draw current active mask prediction
    if current_mask is not None:
        color = np.array([0, 255, 0], dtype=np.uint8)
        mask_indices = current_mask > 0
        overlay[mask_indices] = overlay[mask_indices] * 0.5 + color * 0.5
        
    # Draw active bounding box
    if box and len(box) == 4:
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        
    # Draw prompt points
    for pt, lbl in zip(points, labels):
        c = (0, 255, 0) if lbl == 1 else (255, 0, 0)
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 5, c, -1)
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
        
    return overlay

# --- Session State Initialization ---
if "dataset" not in st.session_state:
    st.session_state.dataset = {}
if "points" not in st.session_state:
    st.session_state.points = []
if "labels" not in st.session_state:
    st.session_state.labels = []
if "box" not in st.session_state:
    st.session_state.box = []
if "current_mask" not in st.session_state:
    st.session_state.current_mask = None
if "current_filepath" not in st.session_state:
    st.session_state.current_filepath = None
if "image_np" not in st.session_state:
    st.session_state.image_np = None
if "tracked_filepaths" not in st.session_state:
    st.session_state.tracked_filepaths = []


# --- Sidebar UI ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/oaboelazm/SegForge/main/assets/logo.png", width=100) # Placeholder for logo if exists
    st.title("SegForge Controls")
    
    # Engine status in sidebar
    status_info = sam_manager.get_status_info()
    st.info(f"🚀 {status_info['full_status']}")
    
    st.divider()
    
    st.header("1. Data Input")
    uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            filepath = os.path.join(TEMP_UPLOAD_DIR, file.name)
            if not os.path.exists(filepath):
                with open(filepath, "wb") as f:
                    f.write(file.getbuffer())
            if filepath not in st.session_state.tracked_filepaths:
                st.session_state.tracked_filepaths.append(filepath)
            if filepath not in st.session_state.dataset:
                st.session_state.dataset[filepath] = {"objects": []}

    selected_file = st.selectbox(
        "Active Image", 
        st.session_state.tracked_filepaths, 
        format_func=lambda x: os.path.basename(x) if x else "None"
    )

    # Change detected -> handle load
    if selected_file != st.session_state.current_filepath:
        st.session_state.current_filepath = selected_file
        st.session_state.points = []
        st.session_state.labels = []
        st.session_state.box = []
        st.session_state.current_mask = None
        
        if selected_file and os.path.exists(selected_file):
            img = cv2.imread(selected_file)
            st.session_state.image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with st.spinner("Setting embeddings..."):
                sam_manager.set_image(st.session_state.image_np)
        else:
            st.session_state.image_np = None

    st.divider()
    st.header("2. Prompt Settings")
    point_type = st.radio("Mode", ["Positive Point", "Negative Point", "Bounding Box"], horizontal=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Generate", use_container_width=True, type="primary"):
            valid_box = st.session_state.box if len(st.session_state.box) == 4 else None
            if not st.session_state.points and not valid_box:
                st.warning("Needs prompt!")
            else:
                raw_mask = sam_manager.predict(points=st.session_state.points, labels=st.session_state.labels, box=valid_box)
                st.session_state.current_mask = postprocess_mask(raw_mask)
                st.rerun()
    with col_b:
        if st.button("Refine", use_container_width=True):
            if st.session_state.current_mask is None:
                st.warning("Gen mask first!")
            else:
                neg_pts = get_refinement_points(st.session_state.current_mask, num_points=15, dilation_iters=2)
                for pt in neg_pts:
                    st.session_state.points.append(list(pt))
                    st.session_state.labels.append(0)
                valid_box = st.session_state.box if len(st.session_state.box) == 4 else None
                raw_mask = sam_manager.predict(points=st.session_state.points, labels=st.session_state.labels, box=valid_box)
                st.session_state.current_mask = postprocess_mask(raw_mask)
                st.rerun()

    if st.button("Clear Prompts", use_container_width=True):
        st.session_state.points = []
        st.session_state.labels = []
        st.session_state.box = []
        st.rerun()

    st.divider()
    st.header("3. Label & Save")
    class_name = st.text_input("Class Name", placeholder="e.g. car")
    if st.button("💾 Save Object", use_container_width=True, type="secondary"):
        if st.session_state.current_mask is None:
            st.warning("No mask!")
        elif not class_name:
            st.warning("No label!")
        else:
            st.session_state.dataset[st.session_state.current_filepath]["objects"].append({
                "mask": st.session_state.current_mask,
                "class_name": class_name
            })
            st.session_state.points = []
            st.session_state.labels = []
            st.session_state.box = []
            st.session_state.current_mask = None
            st.toast(f"Saved {class_name}!")
            st.rerun()

    st.divider()
    if st.button("📥 Export ZIP", use_container_width=True):
        if not st.session_state.dataset:
            st.error("Empty dataset!")
        else:
            with st.spinner("Zipping..."):
                zip_path = exporter.export(st.session_state.dataset)
            with open(zip_path, "rb") as fp:
                st.download_button("Download ZIP", data=fp, file_name="exported_masks.zip", use_container_width=True)


# --- Main Application Area ---
st.title("🧠 SegForge - Segment Anything Engine")

tab_interactive, tab_batch = st.tabs(["🖌️ Interactive Annotation", "📦 Batch Processing"])

with tab_interactive:
    if st.session_state.image_np is not None:
        saved_objects = st.session_state.dataset.get(st.session_state.current_filepath, {}).get("objects", [])
        
        # Render the preview
        rendered = render_image(
            st.session_state.image_np,
            saved_objects,
            st.session_state.current_mask,
            st.session_state.points,
            st.session_state.labels,
            st.session_state.box
        )
        
        st.caption("Click to add points, or two clicks for Bounding Box corners.")
        coords = streamlit_image_coordinates(rendered, key=f"img_{len(st.session_state.points)}_{len(st.session_state.box)}")
        
        if coords:
            x, y = coords['x'], coords['y']
            if "Bounding Box" in point_type:
                if len(st.session_state.box) == 0 or len(st.session_state.box) == 4:
                    st.session_state.box = [x, y]
                elif len(st.session_state.box) == 2:
                    x1, y1 = st.session_state.box
                    st.session_state.box = [min(x1, x), min(y1, y), max(x1, x), max(y1, y)]
            else:
                st.session_state.points.append([x, y])
                st.session_state.labels.append(1 if "Positive" in point_type else 0)
            st.rerun()
            
        with st.expander("Object List Management"):
            if saved_objects:
                for i, obj in enumerate(saved_objects):
                    cols = st.columns([4, 1])
                    cols[0].write(f"ID {i}: {obj['class_name']}")
                    if cols[1].button("🗑️", key=f"del_{i}"):
                        st.session_state.dataset[st.session_state.current_filepath]["objects"].pop(i)
                        st.rerun()
            else:
                st.write("No objects saved for this image.")
    else:
        st.info("👈 Upload and select an image from the sidebar to begin.")


with tab_batch:
    st.header("Bulk Conversion (YOLO -> SAM)")
    st.info("Upload a ZIP containing 'images' and 'labels' folders.")
    
    batch_zip = st.file_uploader("Upload YOLO ZIP", type=["zip"])
    
    if batch_zip and st.button("🚀 Process All", type="primary"):
        import zipfile, tempfile, shutil
        temp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(batch_zip, 'r') as z:
                z.extractall(temp_dir)
            
            # Simple dir check
            img_dir = os.path.join(temp_dir, 'images')
            lbl_dir = os.path.join(temp_dir, 'labels')
            
            if not os.path.exists(img_dir):
                # check subfolders
                for d in os.listdir(temp_dir):
                    if os.path.isdir(os.path.join(temp_dir, d, 'images')):
                        img_dir = os.path.join(temp_dir, d, 'images')
                        lbl_dir = os.path.join(temp_dir, d, 'labels')
                        break
            
            if os.path.exists(img_dir):
                prog = st.progress(0)
                def cb(p, t): prog.progress(p, text=t)
                
                out_zip, previews = process_batch_yolo(img_dir, lbl_dir, sam_manager, exporter, progress_callback=cb)
                st.session_state.batch_previews = previews
                
                with open(out_zip, "rb") as f:
                    st.download_button("📥 Download Result", f, file_name="batch_sam_export.zip")
            else:
                st.error("Missing 'images' folder in ZIP.")
        finally:
            shutil.rmtree(temp_dir)

    if "batch_previews" in st.session_state:
        st.divider()
        st.subheader("Results Preview")
        cols = st.columns(2)
        import random
        idxs = random.sample(range(len(st.session_state.batch_previews)), min(4, len(st.session_state.batch_previews)))
        for i, idx in enumerate(idxs):
            cols[i%2].image(st.session_state.batch_previews[idx])
