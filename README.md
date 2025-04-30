# Dimensionality Reduction with PCA + Clustering using K-Means & DBSCAN

This project demonstrates how to use **Principal Component Analysis (PCA)** to reduce dimensionality of a dataset and then apply clustering techniques like **K-Means** and **DBSCAN** to find meaningful patterns in the data.

---

## 🧠 Dataset

We use the built-in **Digits** dataset from `sklearn.datasets`, which contains 8x8 images of handwritten digits (0–9).

- Number of samples: 1797
- Number of features: 64 (flattened 8×8 pixel values)

---

## 🔧 Workflow

### 1. **Data Preprocessing**
- Standardization with `StandardScaler` to normalize feature values.
  
### 2. **K-Means Clustering**
- **Elbow Method** is used to determine the optimal number of clusters (`k`) by plotting **inertia**.
- Final value of `k = 10` is chosen based on the elbow plot.
- K-Means is applied on both:
  - **Original 64D data**
  - **PCA-reduced 2D and 3D data** for visualization.

### 3. **PCA for Dimensionality Reduction**
- PCA is used to reduce:
  - From 64D → 2D (for 2D visualization)
  - From 64D → 3D (for 3D visualization)
- Allows easier plotting and understanding of cluster structures.

### 4. **DBSCAN Clustering**
- **DBSCAN** is run on the standardized 64D data.
- Grid search over various `eps` and `min_samples` values to test cluster formation.
- Final result is visualized in both 2D and 3D PCA space.

---

## 📊 Visualizations

- 📈 **Elbow Plot**: Helps determine best `k` for K-Means.
- 🟢 **2D & 3D scatter plots** for both:
  - K-Means clustering
  - DBSCAN clustering
- Color-coded by cluster labels.

---

## 📁 Dependencies

```bash
pip install numpy matplotlib scikit-learn
```


## 📌 Key Learnings
PCA is an effective tool to reduce data dimensions while preserving structure.

K-Means is sensitive to the choice of k but performs well with PCA-reduced data.

DBSCAN can find arbitrarily shaped clusters and identify noise points, especially useful after dimensionality reduction.

Visualization in 2D/3D helps understand and interpret clustering results better.

## ✅ Conclusion
This project demonstrates the complete pipeline of:

Data scaling

Dimensionality reduction with PCA

Clustering with both K-Means and DBSCAN

Visualizing high-dimensional data in 2D and 3D spaces
