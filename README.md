# 🎵 Amazon Music Clustering

## 📌 Project Overview

This project focuses on grouping Amazon Music songs into meaningful clusters based on their audio features using **unsupervised machine learning techniques**.

The goal is to identify patterns in music characteristics such as energy, danceability, and tempo, and group similar songs together.

---

## 🎯 Problem Statement

With millions of songs available, manually categorizing music is inefficient.
This project aims to automatically cluster songs based on their audio features to support:

* Music recommendation systems
* Playlist generation
* Music analysis

---

## 🧠 Approach

### 1. Data Preprocessing

* Removed unnecessary columns
* Checked for null values and duplicates
* Feature scaling using **StandardScaler**

---

### 2. Feature Selection

Selected relevant audio features:

* danceability
* energy
* loudness
* speechiness
* acousticness
* instrumentalness
* liveness
* valence
* tempo
* duration_ms

---

### 3. Clustering Techniques Used

#### ✅ K-Means Clustering

* Elbow method → optimal clusters ≈ 4–5
* Silhouette Score comparison:

  * k=4 → **0.23 (best)**
  * k=5 → 0.18
* Final selection: **k = 4**

---

#### ❌ DBSCAN Clustering

* eps determined using K-distance graph (~1.5)
* Silhouette Score: **negative (~ -0.03)**
* Result: Poor clustering → Rejected

---

#### ❌ Hierarchical Clustering (Agglomerative)

* Dendrogram suggested k=4
* Silhouette Score: **0.14**
* Lower than K-Means → Not selected

---

## 📊 Final Model

✔ **K-Means with k = 4 clusters**
✔ Best performance based on silhouette score

---

## 🏷️ Cluster Interpretation

| Cluster | Label                       | Description                   |
| ------- | --------------------------- | ----------------------------- |
| 0       | Calm / Acoustic Songs       | High acousticness, low energy |
| 1       | Melodic / Positive Songs    | Balanced energy and valence   |
| 2       | Speech-heavy (Rap / Spoken) | High speechiness              |
| 3       | High Energy / Dance Songs   | High energy & danceability    |

---

## 📈 Visualization

* PCA (2D) used for visualization
* Clusters show good separation
* Slight overlap in acoustic songs

---

## 📂 Project Structure

```
Amazon_Music_Clustering/
│
├── data/
│   └── single_genre_artists.csv
│
├── notebook/
│   └── Amazon_Music_Clustering.ipynb
│
└── README.md
```

---

## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

---

## 📊 Evaluation Metrics

* Silhouette Score
* Elbow Method (WCSS)
* Dendrogram

---

## 🚀 Results

* Successfully grouped songs into meaningful clusters
* Identified patterns in music features
* Can be used for recommendation systems

---

## 🔮 Future Improvements

* Use t-SNE for better visualization
* Try advanced clustering (Gaussian Mixture, HDBSCAN)
* Build a Streamlit app

---

## 👩‍💻 Author

**Sangeetha S**

---
