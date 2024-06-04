import os
import numpy as np
from byte_tracker import BYTETracker
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
    byte_tracker = BYTETracker()
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

                frame_detections_with_scores = np.hstack((frame_detections[:, 3:7], frame_detections[:, 7:8]))
                tracklets = byte_tracker.update(frame_detections_with_scores, frame_embeddings)
                all_tracklets.extend(tracklets)

            print(f'\nInterpolating missing detections for {folder}')
            # Interpolate missing detections
            all_tracklets = postprocessing.interpolate_missing_detections(all_tracklets)

            # Filter out tracks without valid final features
            valid_tracklets = [trk for trk in all_tracklets if trk.final_features is not None and not np.isnan(trk.final_features).any()]

            if len(valid_tracklets) == 0:
                print(f"No valid tracklets found for {folder}. Skipping.")
                continue

            features = np.array([trk.final_features for trk in valid_tracklets])

            print(f'Clustering features for {folder}')
            labels = postprocessing.run(features)

            tracking_result = []

            print('Writing results ...')

            camera_id = folder.split('_')[-1]  # Extract camera ID from folder name

            for i, trk in enumerate(valid_tracklets):
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
