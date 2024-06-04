import os
import os.path as osp
import cv2
import numpy as np

results = "./runs/tracking/inference_strongsort/camera_0008_reformat.txt"
image_path = "./data/test/camera_0008"
vis_path = "./runs/tracking/inference_strongsort/vis_0008"

if not os.path.exists(vis_path):
    os.makedirs(vis_path)

# load the tracking results
tracking_results = np.loadtxt(results, delimiter=',', dtype=None)

# get the unique frame IDs
frame_ids = np.unique(tracking_results[:, 2])

# group the tracking results by frame ID
tracking_results = [tracking_results[tracking_results[:, 2] == frame_id] for frame_id in frame_ids]

for frame_id, tracking_result in zip(frame_ids, tracking_results):
    # pad the frame_id with zeros to 5
    frame_id = str(int(float(frame_id))).zfill(5)
    img_path = osp.join(image_path, frame_id + '.jpg')
    print(f"\rVisualizing frame {frame_id} from {img_path}", end="")

    new_img_path = osp.join(vis_path, frame_id + '.jpg')

    img = cv2.imread(img_path)
    for track in tracking_result:
        x, y, w, h = map(int, track[3:7])
        track_id = int(track[1])
        # print(f"Draw bounding box at ({x}, {y}, {w}, {h}) with track id {track_id}")
        # draw bounding box with track id
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(track_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(new_img_path, img)
