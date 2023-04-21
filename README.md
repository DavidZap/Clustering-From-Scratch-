# Research about the Spectral Clustering method, and answer the following questions:
* a. In which cases might it be more useful to apply?

Spectral Clustering can be more useful in cases where the data is non-linearly separable, and when traditional clustering methods such as K-means or Hierarchical clustering fail to produce meaningful clusters. Spectral clustering is particularly useful for data with complex structures such as graphs, images, and natural language data.

* b. What are the mathematical fundamentals of it?

The mathematical fundamentals of Spectral Clustering are based on the spectral graph theory, which explores the relationship between the eigenvectors and eigenvalues of a matrix representing a graph. Spectral clustering leverages the Laplacian matrix of a graph to transform the data points into a low-dimensional space where the clusters can be easily identified. The idea is to get the principal components of the matrix as representatives of the "most connected" regions. 

* c. The algorithm for Spectral Clustering can be summarized as follows:
