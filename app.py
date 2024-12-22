import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load Dataset
st.title("GDSC Dataset Analysis")
st.sidebar.title("Navigation")
data = pd.read_csv("./data/final_cleaned_data.csv")
bool_cols = ["CNA", "Gene Expression", "Methylation"]
numeric_cols = ["LN_IC50", "AUC", "Z_SCORE"]
data[bool_cols] = data[bool_cols].map(lambda x: 1 if x == "Y" else 0)

# Sidebar for Navigation
section = st.sidebar.radio(
    "Choose Section", ["Home", "Data Exploration", "Predictive Modeling", "Clustering"]
)

if section == "Home":
    st.header("Welcome to GDSC Dataset Analysis")
    st.write(
        """
        This app explores the GDSC dataset, providing insights into drug responses, genomic features, 
        and cancer cell lines. You can navigate through various sections to explore, predict, and cluster the data.
    """
    )
    st.write(
        f"Dataset contains **{data.shape[0]} rows** and **{data.shape[1]} columns**."
    )
    st.dataframe(data.head(10))

if section == "Data Exploration":
    st.header("Data Exploration")

    sns.set_theme(style="whitegrid")

    # Distribution Plot for LN_IC50
    st.subheader("Distribution of LN_IC50")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data["LN_IC50"], kde=True)
    st.pyplot(fig)

    # Correlation Heatmap for numeric features
    st.subheader("Correlation Heatmap")
    correlation_matrix = data[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    st.pyplot(fig)

    # Count plot for TCGA_DESC
    st.subheader("Cancer Type Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        y=data["TCGA_DESC"],
        order=data["TCGA_DESC"].value_counts().index,
        palette="viridis",
        hue=data["TCGA_DESC"],
    )
    st.pyplot(fig)

    # Boxplot
    st.subheader("AUC by Cancer Type")
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.boxplot(
        data=data,
        x="Cancer Type (matching TCGA label)",
        y="AUC",
        palette="Set2",
        ax=ax,
        hue="TARGET_PATHWAY",
    )
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    # Count Plot for Target Pathway
    st.subheader("Target Pathway Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        y=data["TARGET_PATHWAY"],
        order=data["TARGET_PATHWAY"].value_counts().index,
        palette="viridis",
        hue=data["TARGET_PATHWAY"],
    )

    # Scatter Plot
    st.subheader("Scatter Plot: LN_IC50 vs AUC")
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.scatterplot(
        data=data, x="LN_IC50", y="AUC", hue="TARGET_PATHWAY", alpha=0.7, ax=ax
    )
    st.pyplot(fig)

    # Pair Plot of numeric features
    st.subheader("Pair Plot of Numeric Features")
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.pairplot(
        data,
        vars=numeric_cols,
        hue="GDSC Tissue descriptor 1",
        palette="husl",
        diag_kind="kde",
        corner=True,
    )
    st.pyplot(fig)

if section == "Predictive Modeling":
    st.header("Predict Drug Response (AUC)")

    # Feature Selection
    features = st.multiselect(
        "Select Features", options=["LN_IC50", "CNA", "Gene Expression", "Methylation"]
    )
    if len(features) > 0:
        # Prepare Data
        X = data[features]
        y = data["AUC"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        st.write("Model Trained!")

        # Prediction
        st.subheader("Test the Model")
        inputs = {
            feature: st.number_input(feature, value=float(data[feature].mean()))
            for feature in features
        }
        inputs_df = pd.DataFrame([inputs])
        prediction = model.predict(inputs_df)
        st.write(f"Predicted AUC: **{prediction[0]:.2f}**")

if section == "Clustering":
    st.header("Clustering Cell Lines")

    # Select Features for Clustering
    cluster_features = st.multiselect(
        "Select Features for Clustering",
        options=["LN_IC50", "AUC", "CNA", "Gene Expression"],
    )
    if len(cluster_features) > 0:
        X = data[cluster_features]

        # K-Means Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X)
        data["Cluster"] = clusters

        # PCA for Visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        data["PCA1"], data["PCA2"] = X_pca[:, 0], X_pca[:, 1]

        st.subheader("Cluster Visualization (PCA)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=data, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", ax=ax
        )
        st.pyplot(fig)
