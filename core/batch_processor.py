import os
import cv2
import glob
import numpy as np
from PIL import Image

def process_batch_yolo(images_dir, labels_dir, sam_manager, exporter, class_names=None):
    """
    Reads all images in images_dir and their corresponding YOLO txt files in labels_dir.
    Applies SAM on each bounding box and collates them, returning the path of the exported ZIP
    and a tuple of up to 4 preview images visualizing the result.
    """
    
    # Supported image formats
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for e in exts:
        image_paths.extend(glob.glob(os.path.join(images_dir, e)))
        image_paths.extend(glob.glob(os.path.join(images_dir, e.upper())))

    if not image_paths:
        raise ValueError("No images found in the uploaded directory.")

    # We build the dataset dict using the exact same schema expected by `DatasetExporter`
    dataset_state = {}
    preview_images = []

    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        name_no_ext, _ = os.path.splitext(base_name)
        label_path = os.path.join(labels_dir, f"{name_no_ext}.txt")
        
        if not os.path.exists(label_path):
            continue # Skip images with no YOLO bounding boxes
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load image embeddings once per image!
        sam_manager.set_image(img_rgb)
        
        dataset_state[img_path] = {"objects": []}
        
        # Used for assembling a visual preview overlay
        preview_overlay = img_rgb.copy()
        mask_overlay = np.zeros_like(img_rgb, dtype=np.uint8)
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                c_x, c_y, b_w, b_h = map(float, parts[1:5])
                
                # De-normalize YOLO bounds back into pixels
                pixel_x = c_x * w
                pixel_y = c_y * h
                pixel_w = b_w * w
                pixel_h = b_h * h
                
                # Form explicit SAM Box format: [x_min, y_min, x_max, y_max]
                x_min = max(0, int(pixel_x - (pixel_w / 2)))
                y_min = max(0, int(pixel_y - (pixel_h / 2)))
                x_max = min(w, int(pixel_x + (pixel_w / 2)))
                y_max = min(h, int(pixel_y + (pixel_h / 2)))
                
                box = [x_min, y_min, x_max, y_max]
                
                # Fetch mask using pure bounding box logic!
                mask = sam_manager.predict(box=box)
                
                # Name resolution (fallback to 'class_X' if no names provided)
                c_name = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"
                
                dataset_state[img_path]["objects"].append({
                    "mask": mask,
                    "class_name": c_name
                })
                
                # Tint the mask green on the visual preview overlay
                color = np.array([0, 255, 0], dtype=np.uint8)
                mask_indices = mask > 0
                mask_overlay[mask_indices] = color
                
        # Alpha blend the masks over the preview
        preview_overlay = cv2.addWeighted(preview_overlay, 1.0, mask_overlay, 0.5, 0)
        
        # Keep up to 4 visual previews to pass back to the UI
        if len(preview_images) < 4:
            preview_images.append(preview_overlay)

    # Trigger DatasetExporter on the fully synthesized bulk dataset
    if not dataset_state:
        raise ValueError("No valid bounding boxes found across the uploaded dataset.")
        
    zip_export_path = exporter.export(dataset_state)
    return zip_export_path, preview_images
