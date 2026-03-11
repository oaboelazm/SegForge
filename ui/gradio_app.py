import os
import cv2
import numpy as np
import gradio as gr
from core.sam_manager import SAMManager
from core.dataset_exporter import DatasetExporter
from core.mask_utils import COLORS

def create_app():
    # Initialize Core Engines
    sam_manager = SAMManager()
    exporter = DatasetExporter()
    
    with gr.Blocks(title="SAM Image Annotator") as app:
        gr.Markdown("# 🧠 Segment Anything Model (SAM) - Interactive Dataset Generator")
        gr.Markdown("Upload images, dynamically segment objects via points, assign class labels, and export datasets directly to your device or Kaggle/Colab.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload & Select Images")
                file_uploader = gr.File(file_count="multiple", label="Upload Local Images", file_types=["image"])
                file_list_dropdown = gr.Dropdown(choices=[], label="Select Opened Image to Annotate", interactive=True)
                
                gr.Markdown("### 2. Segmentation Settings")
                with gr.Group():
                    point_type_radio = gr.Radio(
                        choices=["Positive (Object)", "Negative (Background)"],
                        value="Positive (Object)",
                        label="Click Type"
                    )
                    clear_points_btn = gr.Button("Clear Current Clicks", variant="secondary")
                
                gr.Markdown("### 3. Mask Management")
                with gr.Group():
                    class_name_input = gr.Textbox(label="Class Label", placeholder="e.g., car, person, dog")
                    save_mask_btn = gr.Button("Save Object & Mask", variant="primary")
                
                gr.Markdown("### 4. Export")
                with gr.Group():
                    export_btn = gr.Button("Export Dataset (COCO, YOLO, PNG)", variant="primary")
                    export_file = gr.File(label="Download Formatted Dataset ZIP")
            
            with gr.Column(scale=2):
                image_viewer = gr.Image(type="numpy", label="Interactive View (Click to segment)")
                
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
        current_mask_state = gr.State(None)
        
        # --- Helper for rendering ---
        def render_image(image_np, saved_objects, current_mask, points, labels):
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
                
            # Draw point click annotations
            for pt, lbl in zip(points, labels):
                c = (0, 255, 0) if lbl == 1 else (255, 0, 0)
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
                return None, None, [], [], None, None, []
                
            image_np = cv2.imread(filepath)
            # Gradio reads correctly if converted from BGR to RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            # Initialize SAM with the image
            print(f"Setting image embeddings for {os.path.basename(filepath)}...")
            sam_manager.set_image(image_np)
            
            saved_objects = dataset.get(filepath, {}).get("objects", [])
            obj_data = [[i, obj["class_name"], "Saved"] for i, obj in enumerate(saved_objects)]
            
            render = render_image(image_np, saved_objects, None, [], [])
            return filepath, image_np, [], [], None, render, obj_data
            
        def on_image_click(evt: gr.SelectData, image_np, filepath, dataset, points, labels, point_type, current_mask):
            if image_np is None:
                return points, labels, current_mask, None
                
            x, y = evt.index
            points.append([x, y])
            labels.append(1 if "Positive" in point_type else 0)
            
            # Run SAM inference
            current_mask = sam_manager.predict(points, labels)
            
            saved_objects = dataset.get(filepath, {}).get("objects", [])
            render = render_image(image_np, saved_objects, current_mask, points, labels)
            
            return points, labels, current_mask, render
            
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
            render = render_image(image_np, saved_objects, None, [], [])
            
            gr.Info(f"Saved mask for class '{class_name}'.")
            # Clear interaction states
            return d, None, [], [], render, obj_data
            
        def on_clear_points(filepath, dataset, image_np):
            saved_objects = dataset.get(filepath, {}).get("objects", [])
            render = render_image(image_np, saved_objects, None, [], [])
            return [], [], None, render
            
        def on_delete_mask(filepath, dataset, delete_idx, image_np):
            if delete_idx is None:
                return dataset, None, None
                
            d = dataset.copy()
            if filepath in d and 0 <= int(delete_idx) < len(d[filepath]["objects"]):
                d[filepath]["objects"].pop(int(delete_idx))
                
            saved_objects = d.get(filepath, {}).get("objects", [])
            obj_data = [[i, obj["class_name"], "Saved"] for i, obj in enumerate(saved_objects)]
            
            render = render_image(image_np, saved_objects, None, [], [])
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
            outputs=[current_image_path, current_image_np, points_state, labels_state, current_mask_state, image_viewer, objects_display]
        )
        
        image_viewer.select(
            on_image_click,
            inputs=[current_image_np, current_image_path, dataset_state, points_state, labels_state, point_type_radio, current_mask_state],
            outputs=[points_state, labels_state, current_mask_state, image_viewer]
        )
        
        clear_points_btn.click(
            on_clear_points,
            inputs=[current_image_path, dataset_state, current_image_np],
            outputs=[points_state, labels_state, current_mask_state, image_viewer]
        )
        
        save_mask_btn.click(
            on_save_mask,
            inputs=[current_image_path, dataset_state, current_mask_state, class_name_input, current_image_np, points_state],
            outputs=[dataset_state, current_mask_state, points_state, labels_state, image_viewer, objects_display]
        )
        
        delete_btn.click(
            on_delete_mask,
            inputs=[current_image_path, dataset_state, delete_idx_input, current_image_np],
            outputs=[dataset_state, image_viewer, objects_display]
        )
        
        export_btn.click(
            build_dataset_export,
            inputs=[dataset_state],
            outputs=[export_file]
        )
        
    return app
