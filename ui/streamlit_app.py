import os
import sys
import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

# Ensure path includes project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset_exporter import DatasetExporter
from core.mask_utils import COLORS, postprocess_mask, get_refinement_points
from core.batch_processor import process_batch_yolo

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
        
    # Draw bounding box
    if box and len(box) == 4:
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        
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
if "box" not in st.session_state:
    st.session_state.box = []
if "current_mask" not in st.session_state:
    st.session_state.current_mask = None
if "current_filepath" not in st.session_state:
    st.session_state.current_filepath = None
if "image_np" not in st.session_state:
    st.session_state.image_np = None

tab_interactive, tab_batch = st.tabs(["Interactive Annotation", "Batch Conversion (Det -> Seg)"])

with tab_interactive:
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
        st.session_state.box = []
        st.session_state.current_mask = None
        
        if selected_file and os.path.exists(selected_file):
            img = cv2.imread(selected_file)
            st.session_state.image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with st.spinner(f"Setting image embeddings for {os.path.basename(selected_file)}..."):
                sam_manager.set_image(st.session_state.image_np)
        else:
            st.session_state.image_np = None

    st.markdown("### 2. Segmentation Settings")
    point_type = st.radio("Selection Mode", ["Positive Point", "Negative Point", "Bounding Box"])
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Generate Mask", type="primary"):
            valid_box = st.session_state.box if len(st.session_state.box) == 4 else None
            if not st.session_state.points and not valid_box:
                st.warning("Add at least one point or bounding box first!")
            else:
                raw_mask = sam_manager.predict(points=st.session_state.points, labels=st.session_state.labels, box=valid_box)
                st.session_state.current_mask = postprocess_mask(raw_mask)
                st.rerun()
    with col_b:
        if st.button("Refine Mask", type="secondary"):
            if st.session_state.current_mask is None:
                st.warning("Generate a mask first before refining!")
            else:
                neg_pts = get_refinement_points(st.session_state.current_mask, num_points=15, dilation_iters=2)
                for pt in neg_pts:
                    st.session_state.points.append(list(pt))
                    st.session_state.labels.append(0)
                valid_box = st.session_state.box if len(st.session_state.box) == 4 else None
                raw_mask = sam_manager.predict(points=st.session_state.points, labels=st.session_state.labels, box=valid_box)
                st.session_state.current_mask = postprocess_mask(raw_mask)
                st.rerun()
                
    st.write("---")
    
    col_c, col_d, col_e = st.columns(3)
    with col_c:
        if st.button("Clear Prompts"):
            st.session_state.points = []
            st.session_state.labels = []
            st.session_state.box = []
            st.rerun()
    with col_d:
        if st.button("Clear Mask"):
            st.session_state.current_mask = None
            st.rerun()
    with col_e:
        if st.button("Reset Image"):
            st.session_state.points = []
            st.session_state.labels = []
            st.session_state.box = []
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
            st.session_state.box = []
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
            st.session_state.labels,
            st.session_state.box
        )
        
        st.markdown("**Interactive View (Click/Drag depending on mode)**")
        coords = streamlit_image_coordinates(rendered_image, key=f"img_coords_{len(st.session_state.points)}_{len(st.session_state.box)}")
        
        if coords is not None:
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
        

with tab_batch:
    st.header("Batch Conversion (Detection -> Segmentation)")
    st.markdown("Upload a zip containing your `images` and `labels` directories to automatically scale YOLO bounding boxes into exact structural SAM masks.")
    
    batch_zip = st.file_uploader("Upload YOLO Dataset (.zip)", type=["zip"], key="batch_zip_uploader")
    
    if batch_zip is not None:
        if st.button("Start Bulk Conversion", type="primary"):
            import zipfile
            import tempfile
            import shutil
            
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(batch_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
            extracted_images_dir = None
            extracted_labels_dir = None
            
            for root, dirs, files in os.walk(temp_dir):
                if 'images' in dirs and not extracted_images_dir:
                    extracted_images_dir = os.path.join(root, 'images')
                if 'labels' in dirs and not extracted_labels_dir:
                    extracted_labels_dir = os.path.join(root, 'labels')
                    
            if not extracted_images_dir or not extracted_labels_dir:
                st.error("Invalid ZIP structure. Could not find 'images' and 'labels' folders.")
            else:
                with st.spinner("Processing dataset through SAM. This can take several minutes depending on dataset scale..."):
                    try:
                        out_zip, previews = process_batch_yolo(
                            extracted_images_dir,
                            extracted_labels_dir,
                            sam_manager,
                            exporter
                        )
                        
                        st.success("Conversion Complete!")
                        with open(out_zip, "rb") as fp:
                            st.download_button(
                                label="⬇️ Download Exported Dataset ZIP",
                                data=fp,
                                file_name="batch_segmented_dataset.zip",
                                mime="application/zip",
                            )
                            
                        # Show visual validation gallery
                        if previews:
                            st.markdown("### Visual Validation Sample")
                            cols = st.columns(2)
                            for i, view in enumerate(previews):
                                with cols[i % 2]:
                                    st.image(view, use_container_width=True)
                                    
                    except Exception as e:
                        st.error(f"Failed processing: {str(e)}")
                        
            shutil.rmtree(temp_dir)
