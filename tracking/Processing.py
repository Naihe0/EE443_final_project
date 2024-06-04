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
