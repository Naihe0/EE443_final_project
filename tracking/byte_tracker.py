import numpy as np
from yolox.tracker.byte_tracker import BYTETracker as ByteTrack
from yolox.tracking_utils.timer import Timer
# from yolox.tracking_utils.log import logger

class TrackState:
    Tentative = 0
    Confirmed = 1
    Deleted = 2

class Track:
    def __init__(self, tlwh, score, track_id, time):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.track_id = track_id
        self.score = score
        self.state = TrackState.Tentative
        self.hits = 0
        self.age = 0
        self.time_since_update = 0
        self.features = []
        self.times = [time]
        self.boxes = [self.tlwh]
        self.final_features = None

    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def predict(self):
        self.age += 1
        self.time_since_update += 1

    def update(self, tlwh, score, time, feature=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.Confirmed

        self.times.append(time)
        self.boxes.append(tlwh)

        if len(self.times) != len(self.boxes):
            raise ValueError("Mismatch between times and boxes lengths")

        if feature is not None:
            self.features.append(feature)
            self.final_features = np.mean(self.features, axis=0)

    def mark_missed(self):
        self.state = TrackState.Deleted

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

class BYTETracker:
    def __init__(self, conf_thres=0.3, max_age=30, min_hits=3, track_buffer=30):
        self.conf_thres = conf_thres
        self.max_age = max_age
        self.min_hits = min_hits
        self.track_buffer = track_buffer
        self.tracker = ByteTrack(track_thresh=self.conf_thres)
        self.timer = Timer()
        self.tracks = []

    def update(self, detections, features=None):
        outputs = []
        if detections.shape[0] > 0:
            online_targets = self.tracker.update(detections, [img.shape[0], img.shape[1]], self.timer)
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                    outputs.append(np.concatenate((tlwh, [t.track_id], [t.score])).reshape(1, -1))
        if len(outputs) > 0:
            outputs = np.concatenate(outputs, axis=0)
        return outputs
