# Classification and Clustering Algorithms 

---

## 🔵 Classification Methods

### 1. **Logistic Regression**  
A linear model that predicts the probability of a class using a sigmoid function.

### 2. **K-Nearest Neighbors (KNN)**  
Classifies a data point based on the majority class among its nearest neighbors.

### 3. **Support Vector Machine (SVM)**  
Finds the optimal hyperplane that separates classes with the maximum margin.

### 4. **Decision Tree Classifier**  
Splits data recursively based on feature values to form a tree structure for classification.

### 5. **Random Forest Classifier**  
An ensemble of decision trees that vote to determine the final class label.

### 6. **Naive Bayes Classifier**  
Uses Bayes’ Theorem assuming feature independence to classify data.

### 7. **Gradient Boosting Classifier**  
Sequentially builds models to correct the errors of previous ones using boosting.

### 8. **XGBoost / LightGBM / CatBoost**  
Efficient gradient boosting methods known for speed and accuracy in structured data.

### 9. **Neural Networks (MLPClassifier)**  
A network of layers that learns complex patterns through backpropagation.

### 10. **Quadratic Discriminant Analysis (QDA)**  
A generative classifier that models each class with its own covariance matrix.

### 11. **Linear Discriminant Analysis (LDA)**  
Projects data onto a lower-dimensional space that best separates classes.

### 12. **Stochastic Gradient Descent Classifier (SGDClassifier)**  
Uses SGD optimization for large-scale and sparse classification problems.

### 13. **Voting Classifier**  
Combines multiple models' predictions using majority voting or averaging.

---

## 🟣 Clustering Methods

### 1. **K-Means Clustering**  
Partitions data into k clusters by minimizing the variance within each cluster.

### 2. **Hierarchical Clustering (Agglomerative/Divisive)**  
Builds a tree of clusters by either merging or splitting them recursively.

### 3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**  
Clusters points that are closely packed and labels others as noise.

### 4. **Mean Shift Clustering**  
Finds clusters by shifting centroids towards the highest density of data points.

### 5. **Gaussian Mixture Models (GMM)**  
Assumes data is generated from a mixture of several Gaussian distributions.

### 6. **OPTICS (Ordering Points To Identify the Clustering Structure)**  
Similar to DBSCAN but identifies clusters of varying density.

### 7. **Spectral Clustering**  
Uses the eigenvalues of a similarity matrix to perform dimensionality reduction before clustering.

### 8. **Affinity Propagation**  
Exchanges messages between points to decide exemplars and form clusters.

---

## 🟢 Dimensionality Reduction Methods (Often used before Clustering)

These aren't clustering algorithms themselves but are commonly used **before clustering** to reduce dimensions and reveal structure.

### 1. **UMAP (Uniform Manifold Approximation and Projection)**  
Reduces high-dimensional data to lower dimensions while preserving local and global structure—great for visualizing or prepping data for clustering.

### 2. **t-SNE (t-distributed Stochastic Neighbor Embedding)**  
Visualizes high-dimensional data by preserving local similarities in a 2D or 3D map.

### 3. **PCA (Principal Component Analysis)**  
Transforms features into uncorrelated components ranked by variance—often used before clustering.

---