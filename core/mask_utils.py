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

def postprocess_mask(mask, morph_kernel_size=5):
    """Clean up mask edges with morphological operations and fill holes."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    
    # Closing to remove small holes, then Opening to remove slight noise
    closed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    
    # Fill remaining internal holes
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(opened_mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    
    return filled_mask > 0

def get_refinement_points(mask, num_points=20, dilation_iters=3):
    """
    Generate negative bounding points just outside the current mask.
    This helps clean up fuzzy borders in subsequent segmentation loops.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find outer boundary by taking the difference between dilated and original mask
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_uint8, kernel, iterations=dilation_iters)
    boundary = cv2.bitwise_xor(dilated, mask_uint8)
    
    # Get all boundary pixels
    y_indices, x_indices = np.where(boundary > 0)
    if len(y_indices) == 0:
        return []
        
    points = list(zip(x_indices, y_indices))
    
    # Sub-sample uniformly to prevent passing thousands of points to SAM
    step = max(1, len(points) // num_points)
    sampled_points = points[::step][:num_points]
    
    return sampled_points
