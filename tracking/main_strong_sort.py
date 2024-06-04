import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from strong_sort import StrongSORT
from Processing import postprocess
from PIL import Image

raw_data_root = './data'
data_list = {
    'test': ['camera_0008', 'camera_0019', 'camera_0028']
}
sample_rate = 1
vis_flag = True

exp_path = './runs/tracking/inference'
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
det_path = './runs/detect/inference/txt'
emb_path = './runs/reid/inference'

confidence_threshold = 0.3

def main():
    model_path = './reid/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'
    strong_sort = StrongSORT(model_path=model_path, device='cuda', conf_thres=confidence_threshold)
    postprocessing = postprocess(number_of_people=20, cluster_method='kmeans')

    for split in ['test']:
        for folder in data_list[split]:
            det_txt_path = os.path.join(det_path, f'{folder}.txt')
            emb_npy_path = os.path.join(emb_path, f'{folder}.npy')
            tracking_txt_path = os.path.join(exp_path, f'{folder}.txt')

            detection = np.loadtxt(det_txt_path, delimiter=',', dtype=None)
            embedding = np.load(emb_npy_path, allow_pickle=True)

            high_conf_indices = detection[:, 7] >= confidence_threshold
            detection = detection[high_conf_indices]
            embedding = embedding[high_conf_indices]

            img_dir = os.path.join(raw_data_root, split, folder)
            img_files = sorted(os.listdir(img_dir))
            
            all_tracklets = []

            for img_file in img_files:
                img_path = os.path.join(img_dir, img_file)
                img = Image.open(img_path)
                frame_id = int(img_file.split('.')[0])
                frame_detections = detection[detection[:, 2] == frame_id]
                frame_embeddings = embedding[detection[:, 2] == frame_id]

                print(f'\rProcessing frame {frame_id}', end='')

                tracklets = strong_sort.update(frame_detections, frame_embeddings, img)
                all_tracklets.extend(tracklets)

            print(f'Interpolating missing detections for {folder}')
            # Interpolate missing detections
            all_tracklets = postprocessing.interpolate_missing_detections(all_tracklets)

            features = np.array([trk.final_features for trk in all_tracklets])

            print(f'Clustering features for {folder}')
            labels = postprocessing.run(features)

            tracking_result = []

            print('Writing results ...')

            camera_id = folder.split('_')[-1]  # Extract camera ID from folder name

            for i, trk in enumerate(all_tracklets):
                final_tracking_id = labels[i] + 1
                for idx in range(len(trk.boxes)):
                    frame = trk.times[idx]
                    x, y, w, h = trk.boxes[idx]

                    result = '{},{},{},{},{},{},{},-1,-1\n'.format(camera_id, final_tracking_id, frame, x-w/2, y-h/2, w, h)
                    tracking_result.append(result)
        
            print(f'Saving tracking results at {tracking_txt_path}')

            with open(tracking_txt_path, 'w') as f:
                f.writelines(tracking_result)

if __name__ == "__main__":
    print("Starting tracking process...")
    main()
    print("Tracking process completed.")
