import os
import urllib.request
import torch
import numpy as np
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Model Configuration Mapping
MODELS = {
    "large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "path": "sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml"
    },
    "tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "path": "sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml"
    }
}

class SAMManager:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_type = "large" if self.device == "cuda" else "tiny"
        self.model_info = MODELS[self.model_type]
        
        # CPU Optimizations
        if self.device == "cpu":
            # Set threads to match physical cores for better performance
            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
            
        self.model = None
        self.predictor = None
        
    def download_weights(self):
        path = self.model_info["path"]
        url = self.model_info["url"]
        
        if not os.path.exists(path):
            print(f"Downloading SAM 2.1 {self.model_type.upper()} weights to {path}. This may take a few minutes...")
            
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=path) as t:
                def reporthook(blocknum, blocksize, totalsize):
                    t.total = totalsize
                    t.update(blocknum * blocksize - t.n)

                urllib.request.urlretrieve(url, path, reporthook=reporthook)
            print("Download complete.")
            
    def load_model(self):
        self.download_weights()
        if self.model is None:
            config = self.model_info["config"]
            path = self.model_info["path"]
            
            print(f"Loading SAM 2.1 {self.model_type.upper()} Model (Device: {self.device})...")
            self.model = build_sam2(config, path, device=self.device)
            self.predictor = SAM2ImagePredictor(self.model)
            print(f"SAM 2.1 {self.model_type.upper()} loaded successfully.")
            
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
        
        # Use autocast for performance if on GPU, otherwise standard on CPU
        with torch.no_grad():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    masks, scores, _ = self.predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        box=input_box,
                        multimask_output=True,
                    )
            else:
                masks, scores, _ = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box,
                    multimask_output=True,
                )
        
        best_mask_idx = np.argmax(scores)
        return masks[best_mask_idx]

    def get_status_info(self):
        """Returns info about current loading state for the UI."""
        device_name = "CUDA (GPU)" if self.device == "cuda" else "CPU"
        model_name = f"SAM 2.1 {self.model_type.upper()}"
        
        return {
            "device": self.device,
            "device_name": device_name,
            "model_type": self.model_type,
            "model_name": model_name,
            "status": "Ready" if self.model is not None else "Awaiting Load",
            "full_status": f"🚀 Engine: {model_name} | Hardware: {device_name}"
        }
