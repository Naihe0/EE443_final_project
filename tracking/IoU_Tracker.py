import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def calculate_feature_cost(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)

# calculate the overlap ratio of two bounding boxes
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
    
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
    area_bbox1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    area_bbox2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

    iou = intersection_area / float(area_bbox1 + area_bbox2 - intersection_area)
    return iou

# base class for tracklet
class tracklet:
    def __init__(self, tracking_ID, box, feature, time):
        self.ID = tracking_ID
        self.boxes = [box]
        self.features = [feature]
        self.times = [time]
        self.cur_box = box
        self.cur_feature = feature
        self.alive = True
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
    
    def predict(self):
        self.kf.predict()
        self.cur_box = self.kf.x[:4].reshape((4,))
    
    def close(self):
        self.alive = False
    
    def get_avg_features(self):
        self.final_features = sum(self.features) / len(self.features)

# class for multi-object tracker
class tracker:
    def __init__(self):
        self.all_tracklets = []
        self.cur_tracklets = []

    def run(self, detections, features=None):
        for frame_id in range(0, 3600):
            if frame_id % 100 == 0:
                print(f'Tracking | cur_frame {frame_id} | total frame 3600')

            inds = detections[:, 2] == frame_id
            cur_frame_detection = detections[inds]
            if features is not None:
                cur_frame_features = features[inds]

            for track in self.cur_tracklets:
                track.predict()
            
            if len(self.cur_tracklets) == 0:
                for idx in range(len(cur_frame_detection)):
                    new_tracklet = tracklet(len(self.all_tracklets) + 1, cur_frame_detection[idx][3:7], cur_frame_features[idx], frame_id)
                    self.cur_tracklets.append(new_tracklet)
                    self.all_tracklets.append(new_tracklet)
            else:
                cost_matrix = np.zeros((len(self.cur_tracklets), len(cur_frame_detection)))
                for i in range(len(self.cur_tracklets)):
                    for j in range(len(cur_frame_detection)):
                        iou_cost = 1 - calculate_iou(self.cur_tracklets[i].cur_box, cur_frame_detection[j][3:7])
                        feature_cost = calculate_feature_cost(self.cur_tracklets[i].cur_feature, cur_frame_features[j])
                        cost_matrix[i][j] = 0.5 * iou_cost + 0.5 * feature_cost

                row_inds, col_inds = linear_sum_assignment(cost_matrix)
                matches = min(len(row_inds), len(col_inds))

                for idx in range(matches):
                    row, col = row_inds[idx], col_inds[idx]
                    if cost_matrix[row, col] == 1:
                        self.cur_tracklets[row].close()
                        new_tracklet = tracklet(len(self.all_tracklets) + 1, cur_frame_detection[col][3:7], cur_frame_features[col], frame_id)
                        self.cur_tracklets.append(new_tracklet)
                        self.all_tracklets.append(new_tracklet)
                    else:
                        self.cur_tracklets[row].update(cur_frame_detection[col][3:7], cur_frame_features[col], frame_id)

                for idx, det in enumerate(cur_frame_detection):
                    if idx not in col_inds:
                        new_tracklet = tracklet(len(self.all_tracklets) + 1, det[3:7], cur_frame_features[idx], frame_id)
                        self.cur_tracklets.append(new_tracklet)
                        self.all_tracklets.append(new_tracklet)
            
            self.cur_tracklets = [trk for trk in self.cur_tracklets if trk.alive]            

        final_tracklets = self.all_tracklets

        for trk_id in range(len(final_tracklets)):
            final_tracklets[trk_id].get_avg_features()

        return final_tracklets
