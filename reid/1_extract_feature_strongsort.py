import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import gc
import gdown
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from argparse import ArgumentParser
from pathlib import Path

import torchreid
from torchreid.utils import FeatureExtractor
from torchreid.metrics import compute_distance_matrix

# Add the upper directory containing strong_sort to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import StrongSORT
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# Ensure the weights directory exists
weights_dir = Path("weights")
weights_dir.mkdir(parents=True, exist_ok=True)

raw_data_root = '../data'

W, H = 1920, 1080
data_list = {
    'test': ['camera_0008', 'camera_0019', 'camera_0028']
}
sample_rate = 1  # because we want to test on all frames
confidence_threshold = 0.3  # Set confidence threshold

det_path = '../runs/detect/inference/txt'
exp_path = '../runs/reid/inference'
reid_model_ckpt = weights_dir / 'osnet_x0_25_msmt17.pt'

val_transforms = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

reid_extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=reid_model_ckpt,
    image_size=[256, 128],
    device='cuda'
)

# Initialize StrongSORT
cfg = get_config()
cfg.merge_from_file('strong_sort/configs/strong_sort.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
strongsort_tracker = StrongSORT(
    weights_dir / 'osnet_x0_25_msmt17.pt',
    device,
    max_dist=cfg.STRONGSORT.MAX_DIST,
    max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
    max_age=cfg.STRONGSORT.MAX_AGE,
    n_init=cfg.STRONGSORT.N_INIT,
    nn_budget=cfg.STRONGSORT.NN_BUDGET,
    mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
    ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
)

for split in ['test']:
    for folder in data_list[split]:

        det_txt_path = os.path.join(det_path, f'{folder}.txt')
        print(f"Extracting feature from {det_txt_path}")

        dets = np.genfromtxt(det_txt_path, dtype=str, delimiter=',')

        # start extracting frame features
        cur_frame = 0
        emb = np.array([None] * len(dets))  # initialize the feature array

        for idx, (camera_id, _, frame_id, x, y, w, h, score, class_id) in enumerate(dets):
            x, y, w, h = map(float, [x, y, w, h])
            frame_id = str(int(frame_id))  # remove leading space

            if idx % 1000 == 0:
                print(f'Processing frame {frame_id} | {idx}/{len(dets)}')

            img_path = os.path.join(raw_data_root, split, folder, frame_id.zfill(5) + '.jpg')
            img = Image.open(img_path)
            img_np = np.array(img)  # Convert PIL image to NumPy array

            img_crop = img.crop((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
            img_crop = val_transforms(img_crop.convert('RGB')).unsqueeze(0)
            feature = reid_extractor(img_crop).cpu().detach().numpy()[0]

            feature = feature / np.linalg.norm(feature)
            emb[idx] = feature

            # Prepare detection for StrongSORT
            bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
            detections = np.array([bbox])
            confidences = np.array([score])
            classes = np.array([int(class_id)])  # Ensure the class_id is a Python integer

            # Update StrongSORT with the detection
            strongsort_tracker.update(detections, confidences, classes, img_np)

        # Save embeddings
        emb_save_path = os.path.join(exp_path, f'{folder}.npy')
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        np.save(emb_save_path, emb)

        # Save tracked results
        tracks = strongsort_tracker.tracker
        tracks_save_path = os.path.join(exp_path, f'{folder}_tracks.npy')
        np.save(tracks_save_path, tracks)
