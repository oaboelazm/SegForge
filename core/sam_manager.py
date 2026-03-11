import os
import urllib.request
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_predictor import SAM2ImagePredictor

# SAM 2.1 Large Weights
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM_CHECKPOINT_PATH = "sam2.1_hiera_large.pt"
# The config name should match the hierarchical model structure name in the sam2 package
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

class SAMManager:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.predictor = None
        
    def download_weights(self):
        if not os.path.exists(SAM_CHECKPOINT_PATH):
            print(f"Downloading SAM 2.1 weights to {SAM_CHECKPOINT_PATH}. This may take a few minutes...")
            urllib.request.urlretrieve(SAM_CHECKPOINT_URL, SAM_CHECKPOINT_PATH)
            print("Download complete.")
            
    def load_model(self):
        self.download_weights()
        if self.model is None:
            print(f"Loading SAM 2.1 Model ({SAM_CONFIG})...")
            # SAM 2 build_sam2 returns the model
            self.model = build_sam2(SAM_CONFIG, SAM_CHECKPOINT_PATH, device=self.device)
            self.predictor = SAM2ImagePredictor(self.model)
            print("SAM 2.1 loaded successfully.")
            
    def set_image(self, image_np):
        if self.predictor is None:
            self.load_model()
        self.predictor.set_image(image_np)
        
    def predict(self, points=None, labels=None, box=None):
        if self.predictor is None:
            raise RuntimeError("Model is not loaded. Please set image first.")
            
        input_point = np.array(points) if points and len(points) > 0 else None
        input_label = np.array(labels) if labels and len(labels) > 0 else None
        input_box = np.array(box) if box and len(box) == 4 else None
        
        # SAM 2.1 predict API
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=True,
        )
        
        # Sort by score to get the best mask
        best_mask_idx = np.argmax(scores)
        return masks[best_mask_idx]
