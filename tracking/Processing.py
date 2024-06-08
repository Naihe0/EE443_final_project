# Import necessary libraries
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

# Class for post-processing tracking results
class postprocess:
    def __init__(self, number_of_people, cluster_method):
        self.n = number_of_people  # Number of clusters/people
        self.cluster_method_name = cluster_method  # Clustering method to use
    
    # Run the clustering algorithm on the features
    def run(self, features):
        print('Start Clustering')
        
        # Determine the number of clusters
        n_clusters = min(self.n, len(features))
        
        # Choose the clustering method
        if self.cluster_method_name == 'kmeans':
            cluster_method = KMeans(n_clusters=n_clusters, random_state=0)
        else:
            raise NotImplementedError  # Raise an error if the clustering method is not implemented
        
        # Fit the clustering method to the features
        cluster_method.fit(features)
        
        print('Finish Clustering')
        
        # Return the cluster labels
        return cluster_method.labels_

    # Interpolate missing detections for the tracklets
    def interpolate_missing_detections(self, tracklets):
        print('Start Interpolating Missing Detections')
        for trk in tracklets:
            if len(trk.times) < 2:
                continue  # Cannot interpolate with fewer than 2 points

            times = np.array(trk.times)  # Convert times to a numpy array
            complete_times = np.arange(times.min(), times.max() + 1)  # Generate a complete time sequence
            boxes = np.array(trk.boxes)  # Convert boxes to a numpy array
            
            if len(times) != boxes.shape[0]:
                print(f"Mismatch lengths: times={len(times)}, boxes={boxes.shape[0]}")
                continue  # Skip interpolation if lengths do not match
            
            interp_boxes = np.zeros((len(complete_times), boxes.shape[1]))  # Initialize an array for interpolated boxes
            for i in range(boxes.shape[1]):
                interp_boxes[:, i] = np.interp(complete_times, times, boxes[:, i])  # Interpolate each coordinate

            trk.times = complete_times.tolist()  # Update tracklet times with the complete time sequence
            trk.boxes = interp_boxes.tolist()  # Update tracklet boxes with the interpolated boxes

        print('Finish Interpolating Missing Detections')
        return tracklets  # Return the updated tracklets
