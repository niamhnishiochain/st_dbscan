"""
ADAPTED VERSION OF THE ST-DBSCAN ALGORITHM TO ACCEPT EPS3

ST-DBSCAN - fast scalable implementation of ST DBSCAN
            scales also to memory by splitting into frames
            and merging the clusters together
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.utils import check_array
from sklearn.neighbors import NearestNeighbors
import warnings


class ST_DBSCAN():
    """
    A class to perform the ST_DBSCAN clustering
    Parameters
    ----------
    eps1 : float, default=0.5
        The spatial density threshold (maximum spatial distance) between 
        two points to be considered related.
    eps2 : float, default=10
        The temporal threshold (maximum temporal distance) between two 
        points to be considered related.
    min_samples : int, default=5
        The number of samples required for a core point.
    metric : string default='euclidean'
        The used distance metric - more options are
        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
        ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘rogerstanimoto’, ‘sqeuclidean’,
        ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘yule’.
    n_jobs : int or None, default=-1
        The number of processes to start -1 means use all processors 
    Attributes
    ----------
    labels : array, shape = [n_samples]
        Cluster labels for the data - noise is defined as -1
    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

    Birant, Derya, and Alp Kut. "ST-DBSCAN: An algorithm for clustering spatial–temporal data." Data & Knowledge Engineering 60.1 (2007): 208-221.
    
    Peca, I., Fuchs, G., Vrotsou, K., Andrienko, N. V., & Andrienko, G. L. (2012). Scalable Cluster Analysis of Spatial Events. In EuroVA@ EuroVis.
    """

    def __init__(self,
                 eps1=0.5,
                 eps2=10,
                 eps3=0.1, 
                 min_samples=5,
                 metric='euclidean',
                 n_jobs=-1):
        self.eps1 = eps1
        self.eps2 = eps2
        self.eps3 = eps3
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X):
        """
        Apply the ST DBSCAN algorithm 
        ----------
        X : 2D numpy array with
            The first element of the array should be the time 
            attribute as float. The following positions in the array are 
            treated as spatial coordinates. The structure should look like this [[time_step1, x, y], [time_step2, x, y]..]
            For example 2D dataset:
            array([[0,0.45,0.43],
            [0,0.54,0.34],...])
        Returns
        -------
        self
        """
        # check if input is correct
        X = check_array(X)
        if X.shape[1] < 3:
            raise ValueError("Input must have at least 3 columns: time, x, y.")
       
        time = X[:, 0].reshape(-1, 1)
        space = X[:, 1:3] # Assumes x, y
        covars = X[:, 3:] if X.shape[1] > 3 else None # Remaining columns as covariates or not!

        if not self.eps1 > 0.0 or not self.eps2 > 0.0 or not self.min_samples > 0.0:
            raise ValueError('eps1, eps2, minPts must be positive')

        if len(X) < 20000:
            # Compute sqaured form Euclidean Distance Matrix for 'time' attribute and the spatial attributes
            time_dist = pdist(time, metric=self.metric)
            space_dist = pdist(space, metric=self.metric)

            if covars is not None and covars.shape[1] > 0:
                covar_dist = pdist(covars, metric=self.metric)
            else:
                covar_dist = np.zeros_like(time_dist)

            # Only keep distances where all criteria are met
            valid = (time_dist <= self.eps2) & \
                    (space_dist <= self.eps1) & \
                    (covar_dist <= self.eps3)
            
            # All other distance set to a large value to stop them clustering
            combined_dist = np.where(valid, space_dist, 2 * self.eps1)

            db = DBSCAN(eps=self.eps1,
                        min_samples=self.min_samples,
                        metric='precomputed')
            db.fit(squareform(combined_dist))

            self.labels = db.labels_

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # compute with sparse matrices
                # Compute sparse matrix for spatial distance
                nn_spatial = NearestNeighbors(metric=self.metric,
                                              radius=self.eps1)
                nn_spatial.fit(space)
                space_sp = nn_spatial.radius_neighbors_graph(space,
                                                           mode='distance')

                # Compute sparse matrix for temporal distance
                nn_time = NearestNeighbors(metric=self.metric,
                                           radius=self.eps2)
                nn_time.fit(time)
                time_sp = nn_time.radius_neighbors_graph(time,
                                                         mode='distance')
                
                # Compute sparse matrix for covariate distance
                if covars is not None and covars.shape[1] > 0:
                    nn_covars = NearestNeighbors(metric=self.metric,
                                             radius=self.eps3)
                    nn_covars.fit(covars)
                    covar_sp = nn_covars.radius_neighbors_graph(covars,                
                                                            mode='distance')
                else:
                    covar_sp = time_sp.copy()
                    # idea being no constraint here but not sure
                    covar_sp.data.fill(0.0)
                
                # Combine the sparse matrices where all three conditions are met
                combined_graph = time_sp.multiply(space_sp).multiply(covar_sp)

                db = DBSCAN(eps=self.eps1,
                            min_samples=self.min_samples,
                            metric='precomputed') # THIS IS WRONG I THINK! 
                # If 'precomputed' it expects actual distances, but now we have the product of the
                # distances
                # XNTS: FIGURE THIS OUT PLEASE  
                db.fit(combined_graph)

                self.labels = db.labels_

        return self

    def _merge_labels(self, prev_labels, new_labels, overlap_len):
        prev_overlap = prev_labels[-overlap_len:]
        new_overlap = new_labels[:overlap_len]
        mapper = {}

        for old, new in zip(prev_overlap, new_overlap):
            if new != -1 and new not in mapper:
                mapper[new] = old
        mapper[-1] = -1

        new_cluster_ids = set(new_labels)
        overlap_ids = set(new_overlap)
        unused_ids = new_cluster_ids - overlap_ids

        label_counter = max(prev_labels) + 1 if len(prev_labels) > 0 else 0
        for cid in unused_ids:
            if cid != -1:
                mapper[cid] = label_counter
                label_counter += 1

        return np.array([mapper[cid] for cid in new_labels])

    def fit_frame_split(self, X, frame_size, frame_overlap=None):
        """
        Apply the ST DBSCAN algorithm with splitting it into frames.
        ----------
        X : 2D numpy array with
            The first element of the array should be the time (sorted by time)
            attribute as float. The following positions in the array are 
            treated as spatial coordinates. The structure should look like this [[time_step1, x, y], [time_step2, x, y]..]
            For example 2D dataset:
            array([[0,0.45,0.43],
            [0,0.54,0.34],...])
        frame_size : float, default= None
            If not none the dataset is split into frames and merged aferwards
        frame_overlap : float, default=eps2
            If frame_size is set - there will be an overlap between the frames
            to merge the clusters afterwards 
        Returns
        -------
        self
        """
        # check if input is correct
        X = check_array(X)

        if frame_overlap is None:
            frame_overlap = self.eps2

        if frame_size <= 0 or frame_overlap <= 0 or frame_size < frame_overlap:
            raise ValueError('frame_size and/or frame_overlap not correct.')

        if not self.eps1 > 0.0 or not self.eps2 > 0.0 or not self.min_samples > 0.0:
            raise ValueError('eps1, eps2, minPts must be positive')

        # unique time points
        time = X[:, 0]
        unique_times = np.unique(X[:, 0])
        labels = None
        i = 0

        while i< len(unique_times):
            start_time = unique_times[i]
            end_time = start_time + frame_size

            frame_mask = (time >= start_time) & (time <= end_time)
            frame = X[frame_mask]
            self.fit(frame)

            current_labels = self.labels

            if labels is None:
                labels = current_labels
            else:
                overlap_start = end_time - frame_overlap
                overlap_mask = (time >= overlap_start) & (time <= end_time)
                overlap_len = np.sum(overlap_mask & frame_mask)

                remapped_labels = self._merge_labels(labels, current_labels, overlap_len)
                labels = np.concatenate([labels[:-overlap_len], remapped_labels])

            i += frame_size - frame_overlap

        self.labels = labels[:len(X)]
        return self
