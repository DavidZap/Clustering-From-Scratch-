import numpy as np
import random 
from sklearn.base import BaseEstimator, ClusterMixin
import utils


class myPCA:
    def __init__(self, n_components, method = "eigen"):
        
        self.n_components = n_components
        self.method = method
        self.components = None
        self.mean = None
        self.var_explained = None
        
    def fit(self, X):
        # center the data
        self.mean = np.mean(X, axis=0)
        self.transform(X)
        
    def transform(self, X):
        
        X = X - self.mean

        if self.method == "svd":
            U, s, Vt = np.linalg.svd(X)
            if self.n_components is not None:
                U = U[:, :self.n_components]
                s = s[:self.n_components]
                Vt = Vt[:self.n_components, :]
            self.components = Vt[:self.n_components].T
            self.var_explained = s / np.sum(s)

        elif self.method == "eigen":
            # Calculate the covariance matrix of X
            cov_X = np.cov(X, rowvar=False)
            # Calculate the eigenvalues and eigenvectors of the covariance matrix
            eigvals, eigvecs = np.linalg.eigh(cov_X)
            # Sort the eigenvectors by descending eigenvalues
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            # the rate of variance of the components 
            self.var_explained = eigvals / np.sum(eigvals)

            # store the first n_components eigenvectors as the principal components
            self.components = eigvecs[:, : self.n_components]
    
    def fit_transform(self, X):
        # center the data
        
        #Condition form Matrix length 
        if len(X) >1:
            X = X - self.mean
        else:
            X=(X-np.array(self.mean).reshape(1,-1))

        # project the data into the principal components
        X_transformed = np.dot(X, self.components)

        return X_transformed
    
    def inverse_transform(self,X,X_transformed): 
        
        X_reconstructed = X_transformed.dot(self.components.T) + np.mean(X, axis=0)

        
        return X_reconstructed
    
    def explained_variance_ratio(self):
        return self.var_explained
    

class myKmeans:
    
    def __init__(self, n_clusters, max_iters=1000, tol=1e-5,random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.random_state = random_state
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Set random seed for centroid initialization
        if self.random_state is not None:
            np.random.seed(self.random_state)
         
        # Initialize centroids randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for i in range(self.max_iters):
            # Assign each data point to the nearest centroid
            distances = self._calc_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for j in range(self.n_clusters):
                new_centroids[j] = np.mean(X[self.labels == j], axis=0)
                
            # Check for convergence
            if np.sum(np.abs(new_centroids - self.centroids)) < self.tol:
                break
                
            self.centroids = new_centroids
            
    def predict(self, X):
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)
    
    def _calc_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances



# Trae el calculo de los "get_params" y "set_params"
class myKMedoids(BaseEstimator, ClusterMixin):
    
    def __init__(self, n_clusters, max_iters=1000, tol=1e-5, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.cluster_centers_ = None
        self.tol = tol
        self.medoids = None
        self.labels = None
        self.random_state = random_state
    
        
    def fit(self, X):
        n_samples = X.shape[0]
        
        assert self.n_clusters <= n_samples, "El número de clusters debe ser menor o igual que el número de muestras."
        
        # Set random seed for centroid initialization
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize medoids randomly
        self.medoids = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        for i in range(self.max_iters):
            # Assign each data point to the nearest medoid
            distances = self._pairwise_distances(X, X[self.medoids])
            self.labels = np.argmin(distances, axis=1)
            
            # Update medoids
            new_medoids = np.zeros(self.n_clusters, dtype=np.int64)
            for j in range(self.n_clusters):
                mask = (self.labels == j)
                if np.sum(mask) > 0:
                    cluster_distances = self._pairwise_distances(X[mask], X[mask])
                    total_distance = np.sum(cluster_distances, axis=1)
                    new_medoids[j] = np.argmin(total_distance)
                else:
                    new_medoids[j] = self.medoids[j]

            # Check for convergence
            if np.all(new_medoids == self.medoids):
                self.cluster_centers_ = self.medoids
                break
                
            self.medoids = new_medoids
            
    def predict(self, X):
        distances = self._pairwise_distances(X, X[self.medoids])
        return np.argmin(distances, axis=1)

    
    def _pairwise_distances(self,X, Y=None):
        """
        Computes the pairwise distances between each pair of data points in the X matrix
        """
        if Y is None:
            Y = X

        # Compute squared distances
        distance = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)

        # Replace negative values with 0 (due to floating point errors)
        distance[distance < 0] = 0

        # Take the square root to obtain Euclidean distances
        distance = np.sqrt(distance)
        return distance


    
 