import os
import cv2
import glob
import numpy as np
from PIL import Image
from core.mask_utils import COLORS

def process_batch_yolo(images_dir, labels_dir, sam_manager, exporter, class_names=None, progress_callback=None):
    """
    Reads all images in images_dir and their corresponding YOLO txt files in labels_dir.
    Applies SAM on each bounding box and collates them, returning the path of the exported ZIP
    and a tuple of up to 4 preview images visualizing the result.
    """
    
    # Supported image formats
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    image_paths = []
    for e in exts:
        image_paths.extend(glob.glob(os.path.join(images_dir, e)))
        image_paths.extend(glob.glob(os.path.join(images_dir, e.upper())))

    if not image_paths:
        raise ValueError("No images found in the uploaded directory.")

    # Try to find class names if not provided
    if not class_names:
        # Look for classes.txt in images_dir, labels_dir, or their parent
        possible_class_files = [
            os.path.join(labels_dir, "classes.txt"),
            os.path.join(os.path.dirname(labels_dir), "classes.txt"),
            os.path.join(images_dir, "classes.txt"),
            os.path.join(os.path.dirname(images_dir), "classes.txt"),
            os.path.join(labels_dir, "_classes.txt")
        ]
        for cf in possible_class_files:
            if os.path.exists(cf):
                with open(cf, "r") as f:
                    class_names = [line.strip() for line in f.readlines() if line.strip()]
                break

    # We build the dataset dict using the exact same schema expected by `DatasetExporter`
    dataset_state = {}
    preview_images = []
    
    total_images = len(image_paths)
    for i, img_path in enumerate(image_paths):
        if progress_callback:
            progress_callback((i + 1) / total_images, f"Processing {os.path.basename(img_path)} ({i+1}/{total_images})")
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
                
                # Get color based on class_id for consistency
                color_tuple = COLORS[class_id % len(COLORS)]
                color = np.array(color_tuple, dtype=np.uint8)
                
                # Tint the mask with the assigned color
                mask_indices = mask > 0
                mask_overlay[mask_indices] = color

                # Draw class text on the overlay using the same color
                text_pos = (x_min, max(y_min - 10, 20))
                cv2.putText(preview_overlay, c_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_tuple, 2)
                
        # Alpha blend the masks over the preview
        preview_overlay = cv2.addWeighted(preview_overlay, 1.0, mask_overlay, 0.5, 0)
        
        # Keep up to 50 visual previews to support UI randomization
        if len(preview_images) < 50:
            preview_images.append(preview_overlay)

    # Trigger DatasetExporter on the fully synthesized bulk dataset
    if not dataset_state:
        raise ValueError("No valid bounding boxes found across the uploaded dataset.")
        
    zip_export_path = exporter.export(dataset_state)
    return zip_export_path, preview_images
