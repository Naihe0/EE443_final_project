import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from torchreid.reid.utils import FeatureExtractor
import torchvision.transforms as T
import torch

def calculate_iou(bbox1, bbox2):
    x1_1, y1_1, w1, h1 = bbox1
    x1_2, y1_2, w2, h2 = bbox2
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area_bbox1 = w1 * h1
    area_bbox2 = w2 * h2

    iou = intersection_area / float(area_bbox1 + area_bbox2 - intersection_area)
    return iou

def calculate_cosine_similarity(feature1, feature2):
    return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))

class Tracklet:
    def __init__(self, tracking_ID, box, feature, time):
        self.ID = tracking_ID
        self.boxes = [box]
        self.features = [feature]
        self.times = [time]
        self.cur_box = box
        self.cur_feature = feature
        self.alive = True
        self.is_activated = True  # Add this line
        self.kf = self.initialize_kalman_filter(box)
        self.final_features = None
    
    def initialize_kalman_filter(self, box):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([[1,0,0,0,1,0,0],
                         [0,1,0,0,0,1,0],
                         [0,0,1,0,0,0,1],
                         [0,0,0,1,0,0,0],
                         [0,0,0,0,1,0,0],
                         [0,0,0,0,0,1,0],
                         [0,0,0,0,0,0,1]])
        kf.H = np.array([[1,0,0,0,0,0,0],
                         [0,1,0,0,0,0,0],
                         [0,0,1,0,0,0,0],
                         [0,0,0,1,0,0,0]])
        kf.R *= 10.
        kf.P *= 10.
        kf.Q *= 0.01
        kf.x[:4] = np.array([box[0], box[1], box[2], box[3]]).reshape((4, 1))
        return kf
    
    def update(self, box, feature, time):
        self.cur_box = box
        self.kf.update(np.array([box[0], box[1], box[2], box[3]]))
        self.boxes.append(box)
        self.cur_feature = feature
        self.features.append(feature)
        self.times.append(time)
        self.is_activated = True  # Ensure it's set during updates
    
    def predict(self):
        self.kf.predict()
        self.cur_box = self.kf.x[:4].reshape((4,))
    
    def close(self):
        self.alive = False
        self.is_activated = False  # Set to False when tracklet is closed
    
    def get_avg_features(self):
        self.final_features = np.mean(self.features, axis=0)

class StrongSORT:
    def __init__(self, model_path, device='cuda', conf_thres=0.5, max_time_lost=30):
        self.conf_thres = conf_thres
        self.max_time_lost = max_time_lost
        self.tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.track_id = 0
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=model_path,
            device=device
        )

    def update(self, detections, embeddings, img):
        self.frame_id += 1
        # print(f'\rUpdating frame {self.frame_id}', end='')
        activated_tracks = []
        lost_tracks = []
        removed_tracks = []

        for track in self.tracks:
            if not track.is_activated:
                continue
            track.predict()
            if self.frame_id - track.times[-1] > self.max_time_lost:
                track.close()
                lost_tracks.append(track)

        high_conf_detections = [d for d in detections if d[7] >= self.conf_thres]
        low_conf_detections = [d for d in detections if d[7] < self.conf_thres]

        high_conf_features = self.extract_features(high_conf_detections, img) if len(high_conf_detections) > 0 else None
        low_conf_features = self.extract_features(low_conf_detections, img) if len(low_conf_detections) > 0 else None

        if high_conf_features is not None:
            matches, u_tracks, u_detections = self.match(high_conf_detections, self.tracks, high_conf_features)

            for i, j in matches:
                track = self.tracks[i]
                det = high_conf_detections[j]
                track.update(det[3:7], high_conf_features[j], self.frame_id)
                activated_tracks.append(track)

            for i in u_tracks:
                track = self.tracks[i]
                if self.frame_id - track.times[-1] > self.max_time_lost:
                    track.close()
                    removed_tracks.append(track)

            for j in u_detections:
                det = high_conf_detections[j]
                new_track = Tracklet(self.track_id, det[3:7], high_conf_features[j], self.frame_id)
                self.track_id += 1
                activated_tracks.append(new_track)

        if low_conf_features is not None:
            matches, u_tracks, u_detections = self.match(low_conf_detections, self.tracks, low_conf_features)

            for i, j in matches:
                track = self.tracks[i]
                det = low_conf_detections[j]
                track.update(det[3:7], low_conf_features[j], self.frame_id)
                activated_tracks.append(track)

        self.tracks = [t for t in self.tracks if t.alive] + activated_tracks
        self.lost_tracks = lost_tracks
        self.removed_tracks = removed_tracks

        return self.tracks

    def match(self, detections, tracks, features):
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_cost = 1 - calculate_iou(track.cur_box, det[3:7])
                feature_cost = 1 - calculate_cosine_similarity(track.cur_feature, features[j])
                cost_matrix[i, j] = 0.5 * iou_cost + 0.5 * feature_cost

        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        matches = []
        for row, col in zip(row_inds, col_inds):
            if cost_matrix[row, col] <= 1:
                matches.append((row, col))

        u_tracks = [i for i in range(len(tracks)) if i not in row_inds]
        u_detections = [j for j in range(len(detections)) if j not in col_inds]

        return matches, u_tracks, u_detections

    def extract_features(self, detections, img):
        transform = T.Compose([
            T.Resize([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        crops = []
        for det in detections:
            x, y, w, h = map(int, det[3:7])
            crop = img.crop((x, y, x + w, y + h)).convert('RGB')
            crop = transform(crop)
            crops.append(crop)
        crops = torch.stack(crops)
        features = self.extractor(crops)
        return features.cpu().detach().numpy()
