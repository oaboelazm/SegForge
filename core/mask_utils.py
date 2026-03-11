import numpy as np
import cv2

def mask_to_polygon(mask):
    """Convert a boolean mask to a list of polygons."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if contour.shape[0] >= 3:
            # Returns [N, 2] shaped list of points
            poly = contour.reshape(-1, 2).tolist()
            polygons.append(poly)
    return polygons

def get_bbox(mask):
    """Get the bounding box [x_min, y_min, width, height] for COCO."""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x_min, x_max = float(x_indices.min()), float(x_indices.max())
    y_min, y_max = float(y_indices.min()), float(y_indices.max())
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def get_yolo_polygon(polygon, img_width, img_height):
    """Normalize polygon coordinates for YOLO format."""
    norm_poly = []
    for pt in polygon:
        norm_poly.append(pt[0] / img_width)
        norm_poly.append(pt[1] / img_height)
    return norm_poly

def get_coco_polygon(polygon):
    """Flatten polygon for COCO format."""
    flat_poly = []
    for pt in polygon:
        flat_poly.extend([float(pt[0]), float(pt[1])])
    return flat_poly

def generate_colors(num_colors=100):
    np.random.seed(42)
    return np.random.randint(0, 255, size=(num_colors, 3), dtype=np.uint8).tolist()

COLORS = generate_colors(100)
