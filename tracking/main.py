# Import necessary libraries
import os
import numpy as np
from IoU_Tracker import Tracker
from Processing import postprocess

# Define paths and settings
raw_data_root = './data'  # Root directory for raw data
data_list = {
    'test': ['camera_0008', 'camera_0019', 'camera_0028']  # List of test data folders
}
sample_rate = 1  # Sample rate for processing
vis_flag = True  # Visualization flag

# Define paths for tracking results and ensure directory exists
exp_path = './runs/tracking/inference'
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

# Define paths for detection and embedding results
det_path = './runs/detect/inference/txt'
emb_path = './runs/reid/inference'

confidence_threshold = 0.3  # Confidence threshold for filtering detections

def main():
    """
    Main function to perform multi-object tracking using IoU Tracker and post-processing with clustering.
    """
    for split in ['test']:
        for folder in data_list[split]:
            # Define paths for detection, embedding, and tracking result files
            det_txt_path = os.path.join(det_path, f'{folder}.txt')
            emb_npy_path = os.path.join(emb_path, f'{folder}.npy')
            tracking_txt_path = os.path.join(exp_path, f'{folder}.txt')

            # Load detection and embedding data
            detection = np.loadtxt(det_txt_path, delimiter=',', dtype=None)
            embedding = np.load(emb_npy_path, allow_pickle=True)

            # Filter detections and embeddings based on confidence threshold
            high_conf_indices = detection[:, 7] >= confidence_threshold
            detection = detection[high_conf_indices]
            embedding = embedding[high_conf_indices]

            # Print information about loaded data
            print(f"Getting bounding boxes from {det_txt_path} (number of detections: {len(detection)})")
            print(f"Getting features from {emb_npy_path} (number of embeddings: {len(embedding)})")

            camera_id = int(folder.split('_')[-1])
            print(f"Tracking on camera {camera_id}")

            # Initialize Tracker and post-processing
            mot = Tracker()
            postprocessing = postprocess(number_of_people=max(len(detection), 20), cluster_method='kmeans')

            # Run tracker with detections and embeddings
            tracklets = mot.run(detection, embedding)

            # Interpolate missing detections (if needed)
            # tracklets = postprocessing.interpolate_missing_detections(tracklets)

            # Extract final features from tracklets
            features = np.array([trk.final_features for trk in tracklets])

            # Perform clustering on extracted features to assign labels
            labels = postprocessing.run(features)

            tracking_result = []

            print('Writing Result ... ')

            # Write tracking results to file
            for i, trk in enumerate(tracklets):
                final_tracking_id = labels[i] + 1
                for idx in range(len(trk.boxes)):
                    frame = trk.times[idx]
                    x, y, w, h = trk.boxes[idx]

                    result = '{},{},{},{},{},{},{},-1,-1\n'.format(camera_id, final_tracking_id, frame, x-w/2, y-h/2, w, h)
                    tracking_result.append(result)
        
            print('Save tracking results at {}'.format(tracking_txt_path))

            # Save tracking results to file
            with open(tracking_txt_path, 'w') as f:
                f.writelines(tracking_result)

if __name__ == "__main__":
    main()
