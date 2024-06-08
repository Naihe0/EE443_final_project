# Import necessary libraries
from PIL import Image
import os
import numpy as np
import torch
import torchvision.transforms as T
from torchreid.reid.utils import FeatureExtractor

# Define paths and settings
raw_data_root = './data'  # Root directory for raw data
data_list = {
    'test': ['camera_0008', 'camera_0019', 'camera_0028']  # List of test data folders
}
det_path = './runs/detect/inference/txt'  # Path to detection result files
exp_path = './runs/reid/inference'  # Path to save extracted features
reid_model_ckpt_1 = './reid/osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth'  # Checkpoint for first ReID model
reid_model_ckpt_2 = './reid/resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth'  # Checkpoint for second ReID model

# Define transformations for input images
val_transforms = T.Compose([
    T.Resize([256, 128]),  # Resize images to 256x128
    T.ToTensor(),  # Convert images to tensors
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
])

# Initialize feature extractors with pre-trained models
reid_extractor_1 = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=reid_model_ckpt_1,
    image_size=[256, 128],
    device='cuda'
)

reid_extractor_2 = FeatureExtractor(
    model_name='resnet50',
    model_path=reid_model_ckpt_2,
    image_size=[256, 128],
    device='cuda'
)

def extract_features():
    """
    Function to extract features from images using two ReID models.
    Features from both models are concatenated and normalized before saving.
    """
    for split in ['test']:
        for folder in data_list[split]:
            det_txt_path = os.path.join(det_path, f'{folder}.txt')
            print(f"Extracting features from {det_txt_path}")

            dets = np.genfromtxt(det_txt_path, dtype=str, delimiter=',')
            emb = [None] * len(dets)  # Initialize the feature array

            for idx, (camera_id, _, frame_id, x, y, w, h, score, _) in enumerate(dets):
                x, y, w, h = map(float, [x, y, w, h])
                frame_id = str(int(frame_id))

                if idx % 1000 == 0:
                    print(f'Processing frame {frame_id} | {idx}/{len(dets)}')

                img_path = os.path.join(raw_data_root, split, folder, frame_id.zfill(5) + '.jpg')
                img = Image.open(img_path)
                img_crop = img.crop((x - w/2, y - h/2, x + w/2, y + h/2))
                img_crop = val_transforms(img_crop.convert('RGB')).unsqueeze(0).cuda()

                # Extract features using both ReID models
                feature_1 = reid_extractor_1(img_crop).detach()[0]
                feature_2 = reid_extractor_2(img_crop).detach()[0]

                # Concatenate and normalize features
                concatenated_feature = torch.cat((feature_1, feature_2)).cuda()
                concatenated_feature = concatenated_feature / torch.norm(concatenated_feature)

                emb[idx] = concatenated_feature.cpu().numpy()

            # Save the extracted features to a .npy file
            emb_save_path = os.path.join(exp_path, f'{folder}.npy')
            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
            np.save(emb_save_path, np.array(emb))

if __name__ == "__main__":
    extract_features()
