import os
import json
import shutil
import cv2
import zipfile
import numpy as np
from core.mask_utils import mask_to_polygon, get_bbox, get_yolo_polygon, get_coco_polygon

class DatasetExporter:
    def __init__(self, base_dir="dataset"):
        self.base_dir = base_dir
        
    def export(self, dataset):
        """
        Export the current dataset dictionary to standard representations.
        Returns the path to the zipped dataset bundle.
        """
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
            
        os.makedirs(os.path.join(self.base_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "labels"), exist_ok=True) # YOLO format
        os.makedirs(os.path.join(self.base_dir, "annotations"), exist_ok=True) # COCO format
        
        coco_annotations = {
            "info": {"description": "SAM Generated Dataset", "date_created": "2023"},
            "images": [],
            "annotations": [],
            "categories": []
        }
        category_mapping = {}
        cat_id_counter = 1
        ann_id_counter = 1
        img_id_counter = 1
        
        for filepath, data in dataset.items():
            objects = data.get("objects", [])
            if not objects:
                continue
                
            orig_name = os.path.basename(filepath)
            img_name, _ = os.path.splitext(orig_name)
            
            # Copy original image
            dest_img_path = os.path.join(self.base_dir, "images", orig_name)
            shutil.copy2(filepath, dest_img_path)
            
            # Setup COCO Image entry
            img = cv2.imread(filepath)
            if img is None:
                continue
            h, w = img.shape[:2]
            
            coco_annotations["images"].append({
                "id": img_id_counter,
                "file_name": f"images/{orig_name}",
                "width": w,
                "height": h
            })
            
            yolo_lines = []
            
            for obj_idx, obj in enumerate(objects):
                cat_name = obj["class_name"]
                if cat_name not in category_mapping:
                    category_mapping[cat_name] = cat_id_counter
                    coco_annotations["categories"].append({
                        "id": cat_id_counter,
                        "name": cat_name,
                        "supercategory": "none"
                    })
                    cat_id_counter += 1
                    
                cat_id = category_mapping[cat_name]
                mask = obj["mask"]
                
                # Target 1: Pure Binary Mask PNG (for Mask R-CNN, Unet, etc.)
                mask_png_name = f"{img_name}_mask_{obj_idx}_{cat_name}.png"
                cv2.imwrite(os.path.join(self.base_dir, "masks", mask_png_name), (mask * 255).astype(np.uint8))
                
                # Convert to math geometry (Polygons / BBox)
                polygons = mask_to_polygon(mask)
                bbox = get_bbox(mask)
                area = int(np.sum(mask))
                
                # Target 2: COCO Format Segmentation & BBox Schema
                coco_poly = []
                for p in polygons:
                    coco_poly.append(get_coco_polygon(p))
                    
                coco_annotations["annotations"].append({
                    "id": ann_id_counter,
                    "image_id": img_id_counter,
                    "category_id": cat_id,
                    "segmentation": coco_poly,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })
                ann_id_counter += 1
                
                # Target 3: Normalized YOLO format using top bounding polygon representation
                if len(polygons) > 0:
                    norm_poly = get_yolo_polygon(polygons[0], w, h)
                    poly_str = " ".join([f"{p:.6f}" for p in norm_poly])
                    # Assuming index 0-based for darknet standard
                    yolo_lines.append(f"{cat_id - 1} {poly_str}")
                    
            if yolo_lines:
                with open(os.path.join(self.base_dir, "labels", f"{img_name}.txt"), "w") as f:
                    f.write("\n".join(yolo_lines))
                    
            img_id_counter += 1
            
        # Finish COCO
        with open(os.path.join(self.base_dir, "annotations", "coco_annotations.json"), "w") as f:
            json.dump(coco_annotations, f, indent=4)
            
        # Bundle ZIP mapping
        zip_path = "dataset_export.zip"
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.base_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Relative archive path dataset/...
                    arcname = os.path.relpath(file_path, os.path.dirname(self.base_dir))
                    zipf.write(file_path, arcname=arcname)
                    
        return zip_path
