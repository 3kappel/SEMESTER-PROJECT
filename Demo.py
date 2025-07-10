import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to right, #f5f7fa, #c3cfe2);
        }
        h1, h2, h3 {
            color: #3a3a3a;
            font-family: 'Trebuchet MS', sans-serif;
        }
        .css-1d391kg {  # Header padding
            padding-top: 2rem;
        }
        .stButton > button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1 style='text-align: center;'>🎯 Mall Customer Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Uncover Hidden Insights & Group Customers by Behavior</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2190/2190561.png", width=100)
st.sidebar.title("🔧 Options")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])

# Main area
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    # Feature selection
    features = st.sidebar.multiselect("🔢 Select features for clustering", data.columns.tolist())

    if len(features) >= 2:
        # Standardize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[features])

        # Cluster number
        num_clusters = st.sidebar.slider("🎯 Number of clusters", 2, 10, 3)

        # KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        data['Cluster'] = kmeans.fit_predict(scaled_data)

        # Show cluster centers
        st.subheader("📌 Cluster Centers (approx.)")
        centers = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=features
        )
        st.dataframe(centers)

        # Scatter plot
        st.subheader("🎨 Cluster Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x=data[features[0]],
            y=data[features[1]],
            hue=data['Cluster'],
            palette='Set2',
            s=100,
            edgecolor='black',
            ax=ax
        )
        ax.set_title("Customer Segmentation Clusters", fontsize=16)
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        st.pyplot(fig)

    else:
        st.warning("⚠ Please select at least two features to create clusters.")
else:
    st.info("📁 Please upload a CSV file from the sidebar to get started.")