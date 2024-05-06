# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:59:11 2024

@author: Kim Bjerge proposed by ChatGPT
"""

import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#image_size = 224
image_size = 512
num_features = 100
num_pca_components = 55

def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=num_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    feat = descriptors[0:num_features][:] # Only use first 100 features
    #print(feat.shape)
    featFlattend = np.ndarray.flatten(feat)
    #return descriptors
    return featFlattend

def cluster_images(image_paths, n_clusters):
    
    features = []
    for path in image_paths:
        feat = extract_features(path)
        #print(feat.shape)
        #features.extend(featFlattend)
        features.append(feat)
    
    # Standardize features
    scaler = StandardScaler() # removing the mean and scaling to unit variance
    features_scaled = scaler.fit_transform(features)
    
    # Reduce dimensionality with PCA
    pca = PCA(n_components=num_pca_components)  # Adjust the number of components as needed, feature reduction by 100
    features_pca = pca.fit_transform(features_scaled)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters)
    #kmeans.fit(features_pca)
    cluster_kmeans = kmeans.fit(features_pca).predict(features_pca)
    
    return cluster_kmeans, image_paths

# def sort_images_by_cluster(image_paths, kmeans):
#     sorted_images = [[] for _ in range(kmeans.n_clusters)]
    
#     # Standardize features
#     scaler = StandardScaler()
#     pca = PCA(n_components=50)  # Adjust the number of components as needed

#     for path in image_paths:
#         features = extract_features(path)
#         features = features.reshape(-1,1)
#         features_scaled = scaler.fit_transform(features)
#         features_pca = pca.fit_transform(features_scaled)
#         #kmeans.fit(features_all).predict(features_all)
#         #cluster_kmeans = kmeans.fit(features_pca).predict(features_pca) # predict
#         cluster_kmeans = kmeans.predict(features_pca) # predict
#         cluster_label = cluster_kmeans[0]
#         sorted_images[cluster_label].append(path)
    
#     return sorted_images

#%% MAIN
if __name__=='__main__':
    
    # Example usage
    image_folder = "data/euMoths/mixed/"
    #image_paths = [image_folder + name for name in os.listdir(image_folder) if name.endswith(".jpg")]
    image_names = os.listdir(image_folder) 
    image_paths = []
    for name in image_names:
        image_paths.append(image_folder + name)
    
    n_clusters = 5  # Number of clusters
    clusters, image_paths = cluster_images(image_paths, n_clusters)
    
    for cluster, image in zip(clusters, image_paths):
        print("Cluster", cluster, image)
    # sorted_images = sort_images_by_cluster(image_paths, kmeans)
    
    # # Display sorted images for each cluster
    # for cluster_idx, images_in_cluster in enumerate(sorted_images):
    #     print(f"Cluster {cluster_idx}:")
    #     for image_path in images_in_cluster:
    #         print(image_path)
            # You can display or process the images as needed
            # image = cv2.imread(image_path)
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
