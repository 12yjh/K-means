# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:14:23 2024

@author: yjh12
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 5 10:14:23 2024

@author: yjh12
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist  # Used to calculate the distance between data points and cluster centers.

df1 = pd.read_csv('res_K-means/Merged-for-K-means_downtown_Hangzhou0808.csv') 

#%% Standardization and Normalization

# # Read column names
columns = df1.columns

# # Print column names
print(df1.columns)

df1_original = df1.copy()

# # Rename columns
df1.rename(columns={'near_dist': 'near_center'}, inplace=True)

# Select columns to be standardized
features = ['hb_peo_density', 'wb_peo_density', 'jh_ratio', 'poi_sum',
            'evenness', 'diversity', 'roaddensity', 'near_center', 'time_to_sub']

# Standardize selected columns in df1
scaler_standard = StandardScaler()
df1[features] = scaler_standard.fit_transform(df1[features])

# Normalize selected columns in df1
scaler_minmax = MinMaxScaler()
df1[features] = scaler_minmax.fit_transform(df1[features])

# df1.to_csv('8.8.1_final_standardized_normalized.csv', index=False, encoding='utf-8-sig')

#%% K-means Preprocessing
# Delete unnecessary columns for clustering preparation
# df1.columns
# df3 = df1.drop(['OBJECTID', 'ROAD_ID', 'FIRST_FIRST_fclass',
#         'FIRST_FIRST_name',  'Shape_Length', 'Shape_Area'], axis=1)
# %% Elbow Method to Determine the Best k Value

# Select the 'hb_peo_density', 'wb_peo_density', 'jh_ratio', 'poi_sum',
# 'evenness', 'diversity', 'roaddensity', 'near_center', 'time_to_sub' columns
X = df1[['hb_peo_density', 'wb_peo_density', 'jh_ratio', 'poi_sum',
          'evenness', 'diversity', 'roaddensity', 'near_center', 'time_to_sub']]
# X = df3.values[0:6750,]
# Initialize the range of K values and the list of average dispersions
K = range(2, 11)
meanDispersions = []

# Iterate over the range of K values, calculating the average dispersion for each K value
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Set random_state for reproducible results
    kmeans.fit(X)
    # Calculate the average dispersion
    meanDispersions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
print(X)  # Ensure X is not None and contains data
print(kmeans)  # Ensure kmeans is a KMeans instance
# # Set font display
# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # Set 'Times New Roman' font display
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Plot the elbow graph
plt.plot(K, meanDispersions, 'bx-')
plt.xlabel('k')
plt.ylabel('mean_deviation')
plt.title('Using the Elbow Method to Select the Number of Clusters')
# plt.show()

# Output the average dispersion for each k value
print("\nAverage dispersions for each k value:")
for k, dispersion in zip(K, meanDispersions):
    print(f"K = {k}, Average Dispersion = {dispersion:.4f}")

plt.savefig('elbow_method_plot_300dpi0810_3.png', dpi=300)

plt.show()



#%% Silhouette Coefficient to Select the Best K Value

from sklearn.metrics import silhouette_score

X = df1[['hb_peo_density', 'wb_peo_density', 'jh_ratio', 'poi_sum',
          'evenness', 'diversity', 'roaddensity', 'near_center', 'time_to_sub']].values

# Initialize the range of K values and the list of silhouette coefficients
K = range(2, 11)
silhouette_coefficients = []

# Iterate over the range of K values, calculating the silhouette coefficient for each K value
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Set random_state for reproducible results
    kmeans.fit(X)
    # Calculate the silhouette coefficient
    silhouette_coefficients.append(silhouette_score(X, kmeans.labels_))

# Plot the silhouette coefficient graph
plt.figure(figsize=(8, 6))  # Can set the size of the figure
plt.plot(K, silhouette_coefficients, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Analysis for Choosing the Optimal Number of Clusters')
plt.grid(True)  # Add grid lines

# Output the silhouette coefficient for each k value
print("\nSilhouette coefficients for each k value:")
for k, coeff in zip(K, silhouette_coefficients):
    print(f"K = {k}, Silhouette Coefficient = {coeff:.4f}")

# Choose the K value with the highest silhouette coefficient as the optimal number of clusters
best_k = K[silhouette_coefficients.index(max(silhouette_coefficients))]
print(f"\nThe optimal number of clusters is {best_k} with the highest silhouette coefficient of {max(silhouette_coefficients):.4f}")

# Save the image with a resolution of 300 DPI
plt.savefig('silhouette_coefficient_plot_300dpi08103.png', dpi=300)  # This code should not appear before any plt.show(), place show at the end
plt.show()

#%% K-means Clustering

# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# Select the 'MEAN_ndvi', 'FIRST_UHI', 'Traf_volume' columns
# X = df1[['MEAN_ndvi', 'FIRST_UHI', 'Traf_volume']]
 
kmeans = KMeans(n_clusters=3)
result = kmeans.fit_predict(X)
print(result)

# Get the clustering results and cluster centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Add the clustering results to the original DataFrame
df1['Cluster'] = result  # Add a new column 'Cluster' to store the clustering results

# Print the clustering results for inspection
print(df1[['hb_peo_density', 'wb_peo_density', 'jh_ratio', 'poi_sum',
            'evenness', 'diversity', 'roaddensity', 'near_center', 'time_to_sub', 'Cluster']].head())  # Print the first few rows to check the results

# This part of the code is for a personal project in an Australian research project, not used here, only for demonstration

# # Create a 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the 3D scatter plot of the clustering results
# scatter = ax.scatter(df1['MEAN_ndvi'], df1['FIRST_UHI'], df1['Traf_volume'], 
#                      c=df1['Cluster'], cmap='viridis', marker='o')

# # Add a color bar
# cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
# cbar.set_label('Cluster')

# # Set axis labels
# ax.set_xlabel('NDVI')
# ax.set_ylabel('UHI')
# ax.set_zlabel('Traf_Volume')

# # Set the chart title
# ax.set_title('3D K-Means Clustering Results')

# # Save the image (must run the entire block of code and place "plt.show()" at the end to save the result correctly)
# plt.savefig('3d_scatter_plot_08092.png', dpi=300) 
# Show the plot
plt.show()

# Plot the clustering results as a scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(df1['MEAN_ndvi'], df1['FIRST_UHI'], c=df1['Cluster'], cmap='viridis')
# plt.xlabel('MEAN_ndvi')
# plt.ylabel('FIRST_UHI')
# plt.title('K-Means Clustering Results')
# plt.colorbar(label='Cluster')
# plt.show()

# # Get the clustering results and cluster centers
# labels = kmeans.labels_
# centers = kmeans.cluster_centers_

# Print the cluster centers
print("Cluster Centers:")
print(centers)

# Save the center distribution table and the result table
column_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

centers_df = pd.DataFrame(centers, columns=column_names)

centers_df.rename(columns={'0': 'hb_peo_density', '1': 'wb_peo_density', '2': 'jh_ratio',
                            '3': 'poi_sum', '4': 'evenness', '5': 'diversity', '6': 'roaddensity',
                            '7': 'near_center', '8': 'time_to_sub'}, inplace=True)

# centers_df.to_csv('8.8.2_centers_0812.csv', index=False, encoding='utf-8-sig')
# df1.to_csv('8.8.3_k-means_outcome_0812.csv', index=False, encoding='utf-8-sig')













