# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:47:42 2024

@author: Harshda Yewale
"""


import streamlit as st
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Load the dataset
data = pd.read_csv(r"C:\Users\Harshda Yewale\OneDrive\Documents\Desktop\p2 Deployment\updated_file.csv") 
                     # Replace with your dataset filename

data = data.drop('Encoded_Countries', axis=1)  # Drop the 'Encoded_Countries' column

# Preprocess the data if needed
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply PCA for visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# User Input
st.title("Cluster Prediction Model")
st.subheader("Enter Numerical Values for Variables")
user_input = {}
for col in data.columns:
    user_input[col] = st.number_input(col, value=0.0)

# Combine User Input with Dataset
user_df = pd.DataFrame([user_input])
combined_df = pd.concat([data, user_df], ignore_index=True)

# Apply scaler to the combined dataset
combined_scaled = scaler.transform(combined_df)

# Model Prediction
model = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='complete')  # Initialize clustering model
cluster_labels = model.fit_predict(combined_scaled)

# Predict Button
if st.button("Predict"):
    # Visualize
    plt.figure(figsize=(12, 8))

    # Plot original data
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c='blue', alpha=0.5, label="Original Data")

    # Plot user-entered data
    user_pca = pca.transform(user_df)
    plt.scatter(user_pca[:, 0], user_pca[:, 1], color='red', marker='x', label="User-entered Data")

    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title("Scatter Plot of Original and User-entered Data")
    plt.legend()

    st.pyplot(plt)
    cluster_label_user = cluster_labels[-1]  # Get the cluster label for the user's data point
    if cluster_label_user == 0:
        st.write("Your data point falls into Cluster 0.")
        st.write("Countries in Cluster 0 exhibit higher birth rates, slightly higher business tax rates, lower CO2 emissions, more bureaucratic processes for starting businesses, lower energy usage, lower GDP, lower health expenditure per capita and as a percentage of GDP, more hours required for tax compliance, higher infant mortality rates, lower internet usage, higher lending interest rates, lower female life expectancies, lower mobile phone adoption, and generally lower population and tourism figures")
    else:
        st.write("Your data point falls into Cluster 1.")
        st.write("Countries in Cluster 1 are characterized by lower birth rates, slightly lower business tax rates, higher CO2 emissions, simpler processes for starting businesses, higher energy usage, higher GDP, higher health expenditure per capita and as a percentage of GDP, fewer hours required for tax compliance, lower infant mortality rates, higher internet usage, lower lending interest rates, higher female life expectancies, higher mobile phone adoption, and generally higher population and tourism figures")
# Compute summary statistics for each cluster
cluster_summary = pd.DataFrame()
for cluster_label in range(model.n_clusters_):
        cluster_data = combined_df[model.labels_ == cluster_label]
        cluster_mean = cluster_data.mean()
        cluster_std = cluster_data.std()
        cluster_summary[f"Cluster {cluster_label} Mean"] = cluster_mean
        cluster_summary[f"Cluster {cluster_label} Std"] = cluster_std

     # Display the summary statist
st.write("Summary Statistics for Each Cluster:")
st.write(cluster_summary)
    