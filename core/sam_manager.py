import os
import urllib.request
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"

class SAMManager:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.sam = None
        self.predictor = None
        
    def download_weights(self):
        if not os.path.exists(SAM_CHECKPOINT_PATH):
            print(f"Downloading SAM weights to {SAM_CHECKPOINT_PATH}. This may take a few minutes...")
            urllib.request.urlretrieve(SAM_CHECKPOINT_URL, SAM_CHECKPOINT_PATH)
            print("Download complete.")
            
    def load_model(self):
        self.download_weights()
        if self.sam is None:
            print("Loading SAM Model...")
            self.sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
            self.sam.to(device=self.device)
            self.predictor = SamPredictor(self.sam)
            print("SAM loaded successfully.")
            
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
        
        # SAM returns 3 overlapping masks when multimask_output=True
        # We process them and extract the one with the highest confidence score.
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=True,
        )
        
        best_mask_idx = np.argmax(scores)
        return masks[best_mask_idx]
