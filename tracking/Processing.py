import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

class postprocess:
    def __init__(self, number_of_people, cluster_method):
        self.n = number_of_people
        self.cluster_method_name = cluster_method
    
    def run(self, features):
        print('Start Clustering')
        
        n_clusters = min(self.n, len(features))
        
        if self.cluster_method_name == 'kmeans':
            cluster_method = KMeans(n_clusters=n_clusters, random_state=0)
        else:
            raise NotImplementedError
        
        cluster_method.fit(features)
        
        print('Finish Clustering')
        
        return cluster_method.labels_

    def interpolate_missing_detections(self, tracklets):
        print('Start Interpolating Missing Detections')
        for trk in tracklets:
            if len(trk.times) < 2:
                continue  # Cannot interpolate with fewer than 2 points

            times = np.array(trk.times)
            complete_times = np.arange(times.min(), times.max() + 1)
            boxes = np.array(trk.boxes)
            if len(times) != boxes.shape[0]:
                print(f"Mismatch lengths: times={len(times)}, boxes={boxes.shape[0]}")
                continue  # Skip interpolation if lengths do not match
            
            interp_boxes = np.zeros((len(complete_times), boxes.shape[1]))
            for i in range(boxes.shape[1]):
                interp_boxes[:, i] = np.interp(complete_times, times, boxes[:, i])

            trk.times = complete_times.tolist()
            trk.boxes = interp_boxes.tolist()

        print('Finish Interpolating Missing Detections')
        return tracklets
