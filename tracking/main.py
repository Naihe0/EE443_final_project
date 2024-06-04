import os
import os.path as osp
import sys
import time

import numpy as np
from IoU_Tracker import tracker
from Processing import postprocess


raw_data_root = './data'

W, H = 1920, 1080
data_list = {
    'test': ['camera_0008', 'camera_0019', 'camera_0028']
}
sample_rate = 1 # because we want to test on all frames
vis_flag = True # set to True to save the visualizations

exp_path = './runs/tracking/inference'
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
det_path = './runs/detect/inference/txt'
emb_path = './runs/reid/inference'

confidence_threshold = 0.3  # Set your confidence threshold

for split in ['test']:
    for folder in data_list[split]:
        det_txt_path = os.path.join(det_path, f'{folder}.txt')
        emb_npy_path = os.path.join(emb_path, f'{folder}.npy')
        tracking_txt_path = os.path.join(exp_path, f'{folder}.txt')

        detection = np.loadtxt(det_txt_path, delimiter=',', dtype=None)
        embedding = np.load(emb_npy_path, allow_pickle=True)

        # Filter detections based on confidence threshold
        high_conf_indices = detection[:, 7] >= confidence_threshold
        detection = detection[high_conf_indices]
        embedding = embedding[high_conf_indices]

        print(f"Getting bounding boxes from {det_txt_path} (number of detections: {len(detection)}")
        print(f"Getting features from {emb_npy_path} (number of embeddings: {len(embedding)}")


        camera_id = int(folder.split('_')[-1])
        print(f"Tracking on camera {camera_id}")

        mot = tracker()
        postprocessing = postprocess(number_of_people=20, cluster_method='kmeans')

        # Run the IoU tracking
        tracklets = mot.run(detection, embedding)

        features = np.array([trk.final_features for trk in tracklets])

        # Run the Post Processing to merge the tracklets
        labels = postprocessing.run(features) # The label represents the final tracking ID, it starts from 0. We will make it start from 1 later.

        tracking_result = []

        print('Writing Result ... ')

        for i, trk in enumerate(tracklets):
            final_tracking_id = labels[i]+1 # make it starts with 1
            for idx in range(len(trk.boxes)):

                frame = trk.times[idx]
                x, y, w, h = trk.boxes[idx]
                
                result = '{},{},{},{},{},{},{},-1,-1 \n'.format(camera_id, final_tracking_id, frame, x-w/2, y-h/2, w, h )
        
                tracking_result.append(result)
        
        print('Save tracking results at {}'.format(tracking_txt_path))

        with open(tracking_txt_path, 'w') as f:
            f.writelines(tracking_result)
