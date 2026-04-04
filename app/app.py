import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Amazon Music Clustering", layout="wide")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\sange\OneDrive\Documents\Amazon_Music_Clustering\data\single_genre_artists.csv")
    return df

df = load_data()

# -------------------------------
# IMPORTANT: Ensure these columns exist
# -------------------------------
# You MUST already have these in your dataset
# df['Predicted_Cluster']
# df['Cluster_Label']

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Cluster Visualization", "Songs Explorer", "Cluster Insights"]
)

# -------------------------------
# HOME PAGE
# -------------------------------
if page == "Home":
    st.title("🎵 Amazon Music Clustering Dashboard")

    st.markdown("""
    This dashboard helps explore song clusters based on audio features using K-Means clustering.
    
    **Features:**
    - Cluster visualization (PCA)
    - Filter songs by cluster
    - Explore song details
    - Understand cluster characteristics
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Songs", len(df))
    col2.metric("Number of Clusters", df['Predicted_Cluster'].nunique())
    col3.metric("Features Used", 10)

# -------------------------------
# PCA VISUALIZATION PAGE
# -------------------------------
elif page == "Cluster Visualization":
    st.title("📊 Cluster Visualization (PCA)")

    features = [
        'danceability','energy','loudness','speechiness',
        'acousticness','instrumentalness','liveness',
        'valence','tempo','duration_ms'
    ]

    X = df[features]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Cluster_Label"] = df["Cluster_Label"]

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster_Label",
        title="PCA Visualization of Clusters",
        width=1000,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# SONGS EXPLORER (YOUR IDEA 🔥)
# -------------------------------
elif page == "Songs Explorer":
    st.title("🎧 Explore Songs by Cluster")

    # FILTERS
    cluster_selected = st.selectbox(
        "Select Cluster",
        df["Cluster_Label"].unique()
    )

    filtered_df = df[df["Cluster_Label"] == cluster_selected]

    # Additional Filters
    energy_range = st.slider("Energy Range", 0.0, 1.0, (0.0, 1.0))
    dance_range = st.slider("Danceability Range", 0.0, 1.0, (0.0, 1.0))

    filtered_df = filtered_df[
        (filtered_df["energy"].between(energy_range[0], energy_range[1])) &
        (filtered_df["danceability"].between(dance_range[0], dance_range[1]))
    ]

    st.subheader(f"Songs in {cluster_selected}")

    st.dataframe(filtered_df[[
        'track_name','artist_name','genres',
        'energy','danceability','tempo',
        'Cluster_Label'
    ]])

# -------------------------------
# CLUSTER INSIGHTS PAGE
# -------------------------------
elif page == "Cluster Insights":
    st.title("📈 Cluster Insights")

    cluster_selected = st.selectbox(
        "Select Cluster for Analysis",
        df["Cluster_Label"].unique()
    )

    cluster_df = df[df["Cluster_Label"] == cluster_selected]

    st.subheader("Average Feature Values")

    avg_features = cluster_df.mean(numeric_only=True)

    avg_df = avg_features.reset_index()
    avg_df.columns = ["Feature", "Value"]

    fig = px.bar(
        avg_df,
        x="Feature",
        y="Value",
        title=f"Feature Distribution for {cluster_selected}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Size")
    st.write(f"Total Songs: {len(cluster_df)}")