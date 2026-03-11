import os
import cv2
import numpy as np
import gradio as gr
from core.sam_manager import SAMManager
from core.dataset_exporter import DatasetExporter
from core.mask_utils import COLORS, postprocess_mask, get_refinement_points
from core.batch_processor import process_batch_yolo

def create_app():
    # Initialize Core Engines
    sam_manager = SAMManager()
    exporter = DatasetExporter()
    
    with gr.Blocks(title="SAM Image Annotator") as app:
        gr.Markdown("# 🧠 Segment Anything Model (SAM) - Interactive Dataset Generator")
        gr.Markdown("Upload images, dynamically segment objects via points, assign class labels, and export datasets directly to your device or Kaggle/Colab.")
        
        with gr.Tabs():
            with gr.TabItem("Interactive Annotation"):
                # --- Tab 1: Interactive ---
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Upload & Select Images")
                file_uploader = gr.File(file_count="multiple", label="Upload Local Images", file_types=["image"])
                file_list_dropdown = gr.Dropdown(choices=[], label="Select Opened Image to Annotate", interactive=True)
                
                gr.Markdown("### 2. Segmentation Settings")
                with gr.Group():
                    point_type_radio = gr.Radio(
                        choices=["Positive Point", "Negative Point", "Bounding Box"],
                        value="Positive Point",
                        label="Selection Mode"
                    )
                    
                    with gr.Row():
                        generate_mask_btn = gr.Button("Generate Mask", variant="primary")
                        refine_mask_btn = gr.Button("Refine Mask", variant="secondary")
                        
                    with gr.Row():
                        clear_points_btn = gr.Button("Clear Prompts")
                        clear_mask_btn = gr.Button("Clear Mask")
                        reset_img_btn = gr.Button("Reset Image", variant="stop")
                
                gr.Markdown("### 3. Mask Management")
                with gr.Group():
                    class_name_input = gr.Textbox(label="Class Label", placeholder="e.g., car, person, dog")
                    save_mask_btn = gr.Button("Save Object & Mask", variant="primary")
                
                gr.Markdown("### 4. Export")
                with gr.Group():
                    export_btn = gr.Button("Export Dataset (COCO, YOLO, PNG)", variant="primary")
                    export_file = gr.File(label="Download Formatted Dataset ZIP")
            
            with gr.Column(scale=2):
                image_viewer = gr.Image(type="numpy", label="Interactive View (Click/Drag depending on mode)")
                
                objects_display = gr.Dataframe(
                    headers=["ID", "Class Name", "Status"],
                    col_count=(3, "fixed"),
                    label="Saved Objects list for current image",
                    interactive=False
                )
                
                with gr.Row():
                    delete_idx_input = gr.Number(label="Object ID to Delete", precision=0)
                    delete_btn = gr.Button("Delete Mask by ID", variant="stop")
                
                # Session State Variables
                dataset_state = gr.State({}) # Data schema: { filepath: {"objects": [{"mask": ndarray, "class_name": str}]} }
                current_image_path = gr.State(None)
                current_image_np = gr.State(None)
                
                # Point and segmentation states
                points_state = gr.State([])
                labels_state = gr.State([])
                box_state = gr.State([]) # Stores [x1, y1, x2, y2]
                current_mask_state = gr.State(None)
                    
            with gr.TabItem("Batch Conversion (Detection -> Segmentation)"):
                # --- Tab 2: Batch Processing ---
                gr.Markdown("Convert a YOLO-formatted object detection dataset into exact semantic masks using SAM in one click.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Step 1: Upload Archive**")
                        gr.Markdown("Upload a single `.zip` containing your `images` and `labels` directories directly.")
                        batch_zip_uploader = gr.File(label="Upload Dataset ZIP", file_types=[".zip"])
                        
                        gr.Markdown("**Step 2: Processing**")
                        batch_start_btn = gr.Button("Start Conversion", variant="primary")
                        batch_export_file = gr.File(label="Download Generated Segments ZIP", interactive=False)
                        
                    with gr.Column(scale=1):
                        gr.Markdown("**Step 3: Preview Output Selection**")
                        batch_gallery = gr.Gallery(label="Samples from Conversion", columns=2, rows=2, object_fit="contain", height="auto")
        
        # -------------------------------------------------------------------------------------------------------------------------
        # --- TAB 1 : Helper Methods ---
        def render_image(image_np, saved_objects, current_mask, points, labels, box):
            if image_np is None:
                return None
            
            # Use original image copy for overlay
            overlay = image_np.copy()
            
            # Overlay all saved masks
            for i, obj in enumerate(saved_objects):
                mask = obj["mask"]
                color = np.array(COLORS[i % len(COLORS)], dtype=np.uint8)
                mask_indices = mask > 0
                overlay[mask_indices] = overlay[mask_indices] * 0.7 + color * 0.3
                
            # Overlay active (unsaved) mask
            if current_mask is not None:
                color = np.array([0, 255, 0], dtype=np.uint8)
                mask_indices = current_mask > 0
                overlay[mask_indices] = overlay[mask_indices] * 0.5 + color * 0.5
                
            # Draw bounding box
            if len(box) == 4:
                cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2) # Blue Box in RGB natively? Gradio expects RGB, cv2 deals with BGR usually.
                # In RGB arrays, Blue is (0, 0, 255). 
                
            # Draw point click annotations
            for pt, lbl in zip(points, labels):
                c = (0, 255, 0) if lbl == 1 else (255, 0, 0) # Green for positive, Red for negative
                cv2.circle(overlay, (pt[0], pt[1]), 5, c, -1)
                cv2.circle(overlay, (pt[0], pt[1]), 2, (255, 255, 255), -1)
                
            return overlay
            
        # --- Event Callbacks ---
        def on_files_uploaded(files, current_dataset):
            if not files:
                return gr.update(choices=[], value=None), current_dataset
                
            choices = [f.name for f in files]
            dataset = current_dataset.copy()
            
            # Populate state map
            for c in choices:
                if c not in dataset:
                    dataset[c] = {"objects": []}
                    
            return gr.update(choices=choices, value=choices[0]), dataset
            
        def on_image_selected(filepath, dataset):
            if not filepath:
                return None, None, [], [], [], None, None, []
                
            image_np = cv2.imread(filepath)
            # Gradio reads correctly if converted from BGR to RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            # Initialize SAM with the image
            print(f"Setting image embeddings for {os.path.basename(filepath)}...")
            sam_manager.set_image(image_np)
            
            saved_objects = dataset.get(filepath, {}).get("objects", [])
            obj_data = [[i, obj["class_name"], "Saved"] for i, obj in enumerate(saved_objects)]
            
            render = render_image(image_np, saved_objects, None, [], [], [])
            return filepath, image_np, [], [], [], None, render, obj_data
            
        def on_image_click(evt: gr.SelectData, image_np, filepath, dataset, points, labels, box, point_type, current_mask):
            if image_np is None:
                return points, labels, box, render_image(image_np, dataset.get(filepath, {}).get("objects", []), current_mask, points, labels, box)
                
            x, y = evt.index
            
            if "Bounding Box" in point_type:
                # We need exactly 2 points for a box
                if len(box) == 0 or len(box) == 4:
                    # New box start
                    box = [x, y]
                elif len(box) == 2:
                    # Form box [x_min, y_min, x_max, y_max]
                    x1, y1 = box
                    x_min, x_max = min(x1, x), max(x1, x)
                    y_min, y_max = min(y1, y), max(y1, y)
                    box = [x_min, y_min, x_max, y_max]
            else:
                # Point handling
                points.append([x, y])
                labels.append(1 if "Positive" in point_type else 0)
            
            saved_objects = dataset.get(filepath, {}).get("objects", [])
            render = render_image(image_np, saved_objects, current_mask, points, labels, box)
            return points, labels, box, render
            
        def on_generate_mask(filepath, dataset, image_np, points, labels, box):
            if image_np is None:
                return None, None
                
            # Only predict if there are adequate prompts
            valid_box = box if len(box) == 4 else None
            if not points and not valid_box:
                gr.Warning("Add at least one point or bounding box first!")
                return None, render_image(image_np, dataset.get(filepath, {}).get("objects", []), None, points, labels, box)

            raw_mask = sam_manager.predict(points=points, labels=labels, box=valid_box)
            clean_mask = postprocess_mask(raw_mask)
            
            saved_objects = dataset.get(filepath, {}).get("objects", [])
            render = render_image(image_np, saved_objects, clean_mask, points, labels, box)
            
            return clean_mask, render
            
        def on_refine_mask(filepath, dataset, image_np, points, labels, box, current_mask):
            if current_mask is None:
                gr.Warning("Generate a mask first before refining!")
                return points, labels, current_mask, render_image(image_np, dataset.get(filepath, {}).get("objects", []), current_mask, points, labels, box)
                
            # 1. Derive boundary reflection points
            neg_pts = get_refinement_points(current_mask, num_points=15, dilation_iters=2)
            for pt in neg_pts:
                points.append(list(pt))
                labels.append(0) # Negative
                
            # 2. Re-trigger full prediction algorithm mapping all points iteratively
            valid_box = box if len(box) == 4 else None
            refined_mask = sam_manager.predict(points=points, labels=labels, box=valid_box)
            refined_clean_mask = postprocess_mask(refined_mask)
            
            saved_objects = dataset.get(filepath, {}).get("objects", [])
            render = render_image(image_np, saved_objects, refined_clean_mask, points, labels, box)
            
            return points, labels, refined_clean_mask, render
            
        def on_save_mask(filepath, dataset, current_mask, class_name, image_np, points):
            if current_mask is None or len(points) == 0:
                gr.Warning("No mask detected! Click on the image first to segment.")
                return dataset, current_mask, points, [], None, None
                
            if not class_name:
                gr.Warning("Enter a class label first!")
                return dataset, current_mask, points, [], None, None
                
            d = dataset.copy()
            d[filepath]["objects"].append({
                "mask": current_mask,
                "class_name": class_name
            })
            
            saved_objects = d[filepath]["objects"]
            obj_data = [[i, obj["class_name"], "Saved"] for i, obj in enumerate(saved_objects)]
            render = render_image(image_np, saved_objects, None, [], [], [])
            
            gr.Info(f"Saved mask for class '{class_name}'.")
            # Clear interaction states
            return d, None, [], [], [], render, obj_data
            
        def on_clear_points(filepath, dataset, image_np, current_mask):
            saved_objects = dataset.get(filepath, {}).get("objects", [])
            render = render_image(image_np, saved_objects, current_mask, [], [], [])
            return [], [], [], render
            
        def on_clear_mask(filepath, dataset, image_np, points, labels, box):
            saved_objects = dataset.get(filepath, {}).get("objects", [])
            render = render_image(image_np, saved_objects, None, points, labels, box)
            return None, render
            
        def on_reset_image(filepath, dataset, image_np):
            saved_objects = dataset.get(filepath, {}).get("objects", [])
            render = render_image(image_np, saved_objects, None, [], [], [])
            return [], [], [], None, render
            
        def on_delete_mask(filepath, dataset, delete_idx, image_np, points, labels, box, current_mask):
            if delete_idx is None:
                return dataset, render_image(image_np, dataset.get(filepath, {}).get("objects", []), current_mask, points, labels, box), [[i, obj["class_name"], "Saved"] for i, obj in enumerate(dataset.get(filepath, {}).get("objects", []))]
                
            d = dataset.copy()
            if filepath in d and 0 <= int(delete_idx) < len(d[filepath]["objects"]):
                d[filepath]["objects"].pop(int(delete_idx))
                
            saved_objects = d.get(filepath, {}).get("objects", [])
            obj_data = [[i, obj["class_name"], "Saved"] for i, obj in enumerate(saved_objects)]
            
            render = render_image(image_np, saved_objects, current_mask, points, labels, box)
            return d, render, obj_data
            
        def build_dataset_export(dataset):
            zip_path = exporter.export(dataset)
            return zip_path
                
        # --- Interface Connectors ---
        file_uploader.upload(
            on_files_uploaded, 
            inputs=[file_uploader, dataset_state], 
            outputs=[file_list_dropdown, dataset_state]
        )
        
        file_list_dropdown.change(
            on_image_selected,
            inputs=[file_list_dropdown, dataset_state],
            outputs=[current_image_path, current_image_np, points_state, labels_state, box_state, current_mask_state, image_viewer, objects_display]
        )
        
        image_viewer.select(
            on_image_click,
            inputs=[current_image_np, current_image_path, dataset_state, points_state, labels_state, box_state, point_type_radio, current_mask_state],
            outputs=[points_state, labels_state, box_state, image_viewer]
        )
        
        generate_mask_btn.click(
            on_generate_mask,
            inputs=[current_image_path, dataset_state, current_image_np, points_state, labels_state, box_state],
            outputs=[current_mask_state, image_viewer]
        )
        
        refine_mask_btn.click(
            on_refine_mask,
            inputs=[current_image_path, dataset_state, current_image_np, points_state, labels_state, box_state, current_mask_state],
            outputs=[points_state, labels_state, current_mask_state, image_viewer]
        )
        
        clear_points_btn.click(
            on_clear_points,
            inputs=[current_image_path, dataset_state, current_image_np, current_mask_state],
            outputs=[points_state, labels_state, box_state, image_viewer]
        )
        
        clear_mask_btn.click(
            on_clear_mask,
            inputs=[current_image_path, dataset_state, current_image_np, points_state, labels_state, box_state],
            outputs=[current_mask_state, image_viewer]
        )
        
        reset_img_btn.click(
            on_reset_image,
            inputs=[current_image_path, dataset_state, current_image_np],
            outputs=[points_state, labels_state, box_state, current_mask_state, image_viewer]
        )
        
        save_mask_btn.click(
            on_save_mask,
            inputs=[current_image_path, dataset_state, current_mask_state, class_name_input, current_image_np, points_state],
            outputs=[dataset_state, current_mask_state, points_state, labels_state, box_state, image_viewer, objects_display]
        )
        
        delete_btn.click(
            on_delete_mask,
            inputs=[current_image_path, dataset_state, delete_idx_input, current_image_np, points_state, labels_state, box_state, current_mask_state],
            outputs=[dataset_state, image_viewer, objects_display]
        )
        
        export_btn.click(
            build_dataset_export,
            inputs=[dataset_state],
            outputs=[export_file]
        )
        
        
        # -------------------------------------------------------------------------------------------------------------------------
        # --- TAB 2 : Helper Methods ---
        
        def run_batch_conversion(zip_file_obj):
            if zip_file_obj is None:
                gr.Warning("Upload a dataset .zip archive first.")
                return None, None
                
            import zipfile
            import tempfile
            import shutil
            
            temp_dir = tempfile.mkdtemp()
            # 1. Unzip uploaded package
            with zipfile.ZipFile(zip_file_obj.name, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
            # Locate directories dynamically inside the archive structure
            extracted_images_dir = None
            extracted_labels_dir = None
            
            for root, dirs, files in os.walk(temp_dir):
                if 'images' in dirs and not extracted_images_dir:
                    extracted_images_dir = os.path.join(root, 'images')
                if 'labels' in dirs and not extracted_labels_dir:
                    extracted_labels_dir = os.path.join(root, 'labels')
                    
            if not extracted_images_dir or not extracted_labels_dir:
                gr.Warning("Invalid ZIP structure. Could not find 'images' and 'labels' folders.")
                shutil.rmtree(temp_dir)
                return None, None
                
            # 2. Fire YOLO -> Mask conversion
            try:
                out_zip, previews = process_batch_yolo(
                    extracted_images_dir,
                    extracted_labels_dir,
                    sam_manager,
                    exporter
                )
            except Exception as e:
                gr.Warning(f"Conversion failed: {str(e)}")
                shutil.rmtree(temp_dir)
                return None, None
                
            # Clean up temp
            shutil.rmtree(temp_dir)
            
            return out_zip, previews
            
        # --- Interface Connectors Tab 2 ---
        batch_start_btn.click(
            run_batch_conversion,
            inputs=[batch_zip_uploader],
            outputs=[batch_export_file, batch_gallery]
        )
        
    return app
