import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Amazon Music Clustering Dashboard", layout="wide")

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F7FAFC;
    color: #1A1A1A;
}

/* Text */
html, body, [class*="css"] {
    color: #1A1A1A;
}

/* Titles */
h1, h2, h3 {
    color: #00A8E1;
    font-weight: 700;
}

/* LEFT WATERMARK */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 200px;
    height: 100%;
    background-image: url("https://cdn-icons-png.flaticon.com/512/727/727245.png");
    background-repeat: repeat-y;
    background-size: 100px;
    opacity: 0.04;
    z-index: 0;
}

/* RIGHT WATERMARK */
.stApp::after {
    content: "";
    position: fixed;
    top: 0;
    right: 0;
    width: 200px;
    height: 100%;
    background-image: url("https://cdn-icons-png.flaticon.com/512/727/727269.png");
    background-repeat: repeat-y;
    background-size: 100px;
    opacity: 0.04;
    z-index: 0;
}

/* Ensure content stays above */
.block-container {
    position: relative;
    z-index: 1;
}

</style>
""", unsafe_allow_html=True)


# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("../data/Final_Clustered_Data.csv")

df = load_data()

df_original = df.copy()

st.markdown("<br>", unsafe_allow_html=True)

title_placeholder = st.empty()

st.markdown("""
<style>
/* Disable typing inside selectbox */
div[data-baseweb="select"] input {
    pointer-events: none;
}

/* Optional: hide cursor */
div[data-baseweb="select"] input {
    caret-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# CENTERED NAVIGATION (IMPORTANT)
# -------------------------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    page = st.selectbox(
        "Navigate",
        ["Home", "Cluster Visualization", "Songs Explorer"],
        key="main_nav"
    )

# -------------------------
# HOME PAGE
# -------------------------
if page == "Home":
    
    title_placeholder.markdown(
        "<h1 style='text-align:center;'>🎵 Amazon Music Clustering Dashboard</h1>",
        unsafe_allow_html=True
    )

    #st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image(
            "../assets/amazon_music.png",
            width=600
        )

# -------------------------
# CLUSTER VISUALIZATION PAGE
# -------------------------
elif page == "Cluster Visualization":

    # Page Title
    title_placeholder.markdown(
        "<h1 style='text-align:center;'>📊 Cluster Visualization</h1>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # RADIO BUTTONS
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        plot_type = st.radio(
            "Choose Visualization",
            ["PCA Scatter", "Bar Plot", "Heatmap", "Distribution"],
            horizontal=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    # -------------------------
# FEATURE SELECTION (from your Colab)
# -------------------------

    features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms'
]

    X = df[features]

# -------------------------
# SCALING
# -------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# -------------------------
# PCA
# -------------------------

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

# -------------------------
# PCA DATAFRAME
# -------------------------

    PCA_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# Add cluster labels
    PCA_df['Cluster_Label'] = df['Cluster_Label']
    # PCA
    if plot_type == "PCA Scatter":

        fig, ax = plt.subplots(figsize=(5,3))

        sns.scatterplot(
    data=PCA_df,
    x='PC1',
    y='PC2',
    hue='Cluster_Label',
    palette='tab10',
    alpha=0.7,
    ax=ax
)

# Flip to match Colab
        PCA_df["PC2"] = -PCA_df["PC2"]

        ax.set_title("PCA Visualization of Song Clusters (K=4)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")

# Centered display
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.pyplot(fig, use_container_width=False)
            st.info("Clusters are clearly separated based on musical features.")


    # BAR PLOT
    elif plot_type == "Bar Plot":
         
        scaled_df = pd.DataFrame(X_scaled, columns=features)
        scaled_df["Cluster_Label"] = df["Cluster_Label"]


    # Use scaled data (IMPORTANT)
        cluster_avg = scaled_df.groupby("Cluster_Label").mean()

        fig, ax = plt.subplots(figsize=(5,3))   # small like PCA

        cluster_avg.T.plot(kind='bar', ax=ax)

        ax.set_title("Feature Comparison (Standardized)")
        ax.set_ylabel("Standardized Value")
        ax.set_xlabel("Features")

        ax.legend(title="Cluster")
        plt.xticks(rotation=45)

    # Centered layout
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.pyplot(fig, use_container_width=False)

        st.info("Each cluster has distinct feature characteristics.")

    # HEATMAP
    elif plot_type == "Heatmap":
        scaled_df = pd.DataFrame(X_scaled, columns=features)
        scaled_df["Cluster_Label"] = df["Cluster_Label"]

        cluster_avg = scaled_df.groupby("Cluster_Label").mean()

        fig, ax = plt.subplots(figsize=(7,5))   # slightly bigger

        sns.heatmap(
        cluster_avg,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        annot_kws={"size": 9},
        ax=ax
    )

        ax.set_title("Feature Comparison Across Clusters")
        ax.set_xlabel("Features")
        ax.set_ylabel("Cluster")

        plt.xticks(rotation=30)
        plt.tight_layout()

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.pyplot(fig, use_container_width=False)

            st.info("Heatmap shows feature intensity across clusters.")

    # DISTRIBUTION
    elif plot_type == "Distribution":

    # -------------------------
    # CENTERED DROPDOWN
    # -------------------------
        col1, col2, col3 = st.columns([1,2,1])

        with col2:
            feature = st.selectbox(
            "Select Feature",
            features
        )

        st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------
    # SMALLER PLOT
    # -------------------------
        fig, ax = plt.subplots(figsize=(5,3))   # same as PCA style

        sns.histplot(
        data=df,
        x=feature,
        hue="Cluster_Label",
        kde=True,
        bins=30,             
        ax=ax
    )

        ax.set_title(f"{feature} Distribution Across Clusters")

    # -------------------------
    # CENTER THE PLOT
    # -------------------------
        col1, col2, col3 = st.columns([1,2,1])

        with col2:
            st.pyplot(fig, use_container_width=False)

            st.info(f"{feature} distribution varies across clusters.")
        
# =========================================================
# SONGS EXPLORER
# =========================================================
elif page == "Songs Explorer":

    # -------------------------
    # TITLE (CENTERED)
    # -------------------------
    st.markdown(
        "<h1 style='text-align:center;'>🎧 Songs Explorer</h1>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------
    # CENTERED DROPDOWN
    # -------------------------
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        cluster_selected = st.radio(
            "Select Cluster",
            df["Cluster_Label"].unique()
        )

        if cluster_selected == "High Energy / Dance Songs":
            st.markdown(
    "<div style='background-color:#e6f2ff; padding:10px; border-radius:8px; text-align:center;'>High energy songs with strong beats and high danceability.</div>",
    unsafe_allow_html=True
)

        elif cluster_selected == "Calm / Acoustic Songs":
            st.info("Calm songs with high acousticness and lower energy.")

        elif cluster_selected == "Speech-heavy (Rap / Spoken)":
            st.info("Songs dominated by speech elements like rap or spoken content.")

        elif cluster_selected == "Melodic / Positive Songs":
            st.info("Melodic songs with balanced energy and positive valence.")


    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------
    # FILTER DATA
    # -------------------------
    cluster_df = df_original[df_original["Cluster_Label"] == cluster_selected]

    
    if cluster_selected == "High Energy / Dance Songs":
        cluster_df = cluster_df.sort_values(by="energy", ascending=False)

    elif cluster_selected == "Calm / Acoustic Songs":
        cluster_df = cluster_df.sort_values(by="acousticness", ascending=False)

    elif cluster_selected == "Speech-heavy (Rap / Spoken)":
        cluster_df = cluster_df.sort_values(by="speechiness", ascending=False)

    elif cluster_selected == "Melodic / Positive Songs":
        cluster_df = cluster_df.sort_values(by="valence", ascending=False)

   

    # -------------------------
    # CENTERED SUBTITLE
    # -------------------------
    st.markdown(
        f"<h3 style='text-align:center;'>Top Songs in {cluster_selected}</h3>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------
    # CENTERED TABLE
    # -------------------------
    col1, col2, col3 = st.columns([0.5,4,0.5])   # wider center for table

    cluster_df = df_original[df_original["Cluster_Label"] == cluster_selected]

    display_features = [
    'energy', 'danceability', 'acousticness',
    'speechiness', 'valence', 'tempo'
]

    with col2:
        st.dataframe(
        cluster_df[['name_song', 'name_artists', 'genres'] + display_features]
        .head(10)
        .rename(columns={
            'name_song': 'Song',
            'name_artists': 'Artist',
            'genres': 'Genre'
        }),
        use_container_width=True
    )