# 1) Research about the Spectral Clustering method, and answer the following questions:
* a. In which cases might it be more useful to apply?

Spectral Clustering can be more useful in cases where the data is non-linearly separable, and when traditional clustering methods such as K-means or Hierarchical clustering fail to produce meaningful clusters. Spectral clustering is particularly useful for data with complex structures such as graphs, images, and natural language data.

* b. What are the mathematical fundamentals of it?

The mathematical fundamentals of Spectral Clustering are based on the spectral graph theory, which explores the relationship between the eigenvectors and eigenvalues of a matrix representing a graph. Spectral clustering leverages the Laplacian matrix of a graph to transform the data points into a low-dimensional space where the clusters can be easily identified. The idea is to get the principal components of the matrix as representatives of the "most connected" regions. 

* c. What is the algorithm to compute it?

1. Given a data set, construct a similarity matrix that captures the similarity between each pair of data points.
2. Compute the Laplacian matrix from the similarity matrix.
3. Compute the first k eigenvectors of the Laplacian matrix.
4. Form a matrix with these eigenvectors as columns and normalize each row.
5. Cluster the resulting matrix using K-means clustering or another clustering algorithm.

* d. Does it hold any relation to some of the concepts previously mentioned in class? Which, and how?

Spectral Clustering is related to several concepts previously mentioned in class, such linear algebra, and dimensionality reduction. In particular, Spectral Clustering uses the Laplacian matrix, which is a key concept in graph theory, and eigenvectors and eigenvalues, which are fundamental concepts in linear algebra. Additionally, Spectral Clustering can be seen as a form of dimensionality reduction, as it transforms high-dimensional data into a low-dimensional space where the clusters can be easily identified.

# 2) Research about the Spectral Clustering method, and answer the following questions:

* a. In which cases might it be more useful to apply?

DBSCAN __(Density-Based Spatial Clustering of Applications with Noise)__ is a popular clustering algorithm that is useful when dealing with data that has *non-uniform density*. It is especially useful when the data is *spatially clustered*, and when the *number of clusters is not known a priori*. It can also handle noise and outliers well, making it a good choice for datasets with irregular shapes.

* b. What are the mathematical fundamentals of it?

The fundamental idea behind DBSCAN is to *group together points that are close to each other in a dense region of the data*, while separating points that are *further apart or in regions of lower density*. The algorithm works by defining a neighborhood around each point and then grouping points that are close enough to each other and have enough nearby neighbors into clusters. DBSCAN uses two key parameters: *epsilon (ε), which defines the radius of the neighborhood around each point*, and *minPts, which is the minimum number of points required to form a dense region*.

* c. Is there any relation between DBSCAN and Spectral Clustering? If so, what is it?

There is a relation between DBSCAN and Spectral Clustering in that both algorithms are used for clustering, but they use different approaches to achieve this. DBSCAN is a __density-based clustering algorithm that works by finding areas of high density in the data__, while Spectral Clustering is a __graph-based clustering algorithm that works by finding low-dimensional embeddings of the data and then clustering these embeddings__. Both algorithms can be effective in different scenarios, depending on the nature of the data and the desired output. However, *Spectral Clustering may be more suitable for datasets with complex geometric structures*, while *DBSCAN may be more suitable for datasets with varying densities*.

# 3) What is the elbow method in clustering? And which flaws does it pose to assess quality?

The elbow method is a popular technique used to determine the optimal number of clusters in a dataset. It involves plotting the sum of squared distances (SSE) of each data point from its closest centroid against the number of clusters, and looking for the "elbow" point in the graph where the SSE starts to decrease more slowly.

The idea behind the elbow method is that as the number of clusters increases, the SSE tends to decrease since the centroids are closer to their respective data points. However, at a certain point, adding more clusters will only marginally decrease the SSE, and may even increase it. This point is known as the elbow point and is considered the optimal number of clusters.

While the elbow method is simple and intuitive, it can be subjective and may not always provide the most accurate results. One of the main flaws of the elbow method is that it is sensitive to the shape of the data and the scale of the features. In some cases, the elbow point may not be well-defined or may be difficult to identify, making it challenging to determine the optimal number of clusters.

Moreover, the elbow method does not consider other metrics such as the cohesion and separation of the clusters, or the interpretability and usefulness of the resulting clusters. As such, it is important to use the elbow method as a starting point and supplement it with other methods to assess the quality of the clustering results. 

# 4) Remember the unsupervised Python package you created in the previous unit? 😀It’s time for an upgrade.
a. Implement the k-means module using Python and Numpy
b. Implement the k-medoids module using Python and Numpy
c. Remember to keep consistency with Scikit-Learn API as high as possible

The result upgraded Unsupervised package is in this [folder](dist)
