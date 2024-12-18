# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:43:28 2023

@author: Kim BJerge
"""
import os
import random
import numpy as np
import torch
import argparse
#from torch import nn
from torch.utils.data import DataLoader

#from easyfsl.datasets import MiniImageNet
#from easyfsl.samplers import TaskSampler
#from easyfsl.utils import evaluate
from easyfsl.utils import predict_embeddings
from easyfsl.datasets import FeaturesDataset

from FewShotModelData import EmbeddingsModel, FewShotDataset

from torchvision.models import resnet50
from torchvision.models import resnet34
from torchvision.models import resnet18
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet34_Weights
from torchvision.models import ResNet18_Weights
from torchvision.models.efficientnet import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models.efficientnet import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score

from sklearn.manifold import TSNE
#from numpy import reshape
import seaborn as sns
import pandas as pd
from statistics import mode

# Gausion Mixtures Models plotting
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)

    for i in range(10):
    #for i in [5, 6, 7, 8, 9]:
        f = X[np.where(labels == i)]
        plt.plot(f[:, 0], f[:, 1], '.', c=c[i])
    #plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper left')
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7'], loc='lower right')
    
    w_factor = 0.5 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

def computeAccuracy(labels, predictions):
    
    predDic = {}
    for idx in range(len(predictions)):
        prediction = predictions[idx]
        if prediction in predDic:
            predDic[prediction].append(labels[idx])
        else:
            predDic[prediction] = []
            predDic[prediction].append(labels[idx])
    
    #print(predDic)

    TruePositives = 0
    for i, key in enumerate(predDic):    
        TruePositives += sum(predDic[key]==mode(predDic[key]))
        
    accuracy = TruePositives/len(predictions)
    return accuracy, TruePositives

def computeClusterPerformance(test_classes, features_all, labels_all):

    print("Kmeans clustering")
    kmeans = KMeans(n_clusters=test_classes, random_state=0, n_init="auto")
    predictions_kmeans = kmeans.fit(features_all).predict(features_all)
    accuracy, TP = computeAccuracy(np.array(labels_all), predictions_kmeans)
    print("Similar class (SC) score", str(test_classes) + " classes", accuracy, TP, len(predictions_kmeans))
    score = adjusted_rand_score(np.array(labels_all), predictions_kmeans)
    print("Rand index (RI) score", str(test_classes) + " classes", score)

    print("Gausian Mixture Models")
    gmm = GaussianMixture(n_components=test_classes, covariance_type='full', random_state=42)
    predictions_gmm = gmm.fit(features_all).predict(features_all)
    accuracy, TP = computeAccuracy(np.array(labels_all), predictions_gmm)
    print("Similar class (SC) score", str(test_classes) + " classes", accuracy, TP, len(predictions_gmm))
    score = adjusted_rand_score(np.array(labels_all), predictions_gmm)
    print("Rand index (RI) score", str(test_classes) + " classes", score)

    print("HSBSCAN clustering")
    #for max_clusters in [50, None]:
        #print("Max clusters", max_clusters)
    dbscan = HDBSCAN(min_cluster_size=3, store_centers='centroid')
    embs = features_all / np.linalg.norm(features_all, axis=1, keepdims=True)
    dbscan.fit(embs)
    predictions_dbscan = dbscan.labels_  
    centroids = dbscan.centroids_   
    accuracy, TP = computeAccuracy(np.array(labels_all), predictions_dbscan)
    print("Similar class (SC) score", str(test_classes) + " classes", accuracy, TP, len(predictions_dbscan))
    score = adjusted_rand_score(np.array(labels_all), predictions_dbscan)
    print("Rand index (RI) score", str(test_classes) + " classes", score)
    print("Number of centroids", len(centroids))
    #print(centroids)
    
    
#%% MAIN
if __name__=='__main__':

    plt.rcParams.update({'font.size': 12})
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ConvNeXt') #resnet18, resnet34, resnet50, EfficientNetB3, EfficientNetB4, ConvNeXt, ViTB16 
    parser.add_argument('--weights', default='ImageNet') #ImageNet, euMoths, CUB
    parser.add_argument('--dataset', default='euMoths') #miniImagenet, euMoths, CUB
    parser.add_argument('--method', default='SpecClust') #IsoForest, GMM, Kmeans, SpecClust, DBSCANClust, HDBSCANClust, Agglomerative ALL
    args = parser.parse_args()
  
    resDir = "./result/"
    dataDirMiniImageNet = "./data/mini_imagenet"
    dataDirEuMoths = "./data/euMoths"
    dataDirCUB = "./data/CUB"

    image_size = 224 # ResNet euMoths

    random_seed = 0
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    batch_size = 128
    n_workers = 12
    
    if args.dataset == 'euMoths':
        num_classes = 100  
        test_classes = 50
    if args.dataset == 'CUB':
        num_classes = 140
        test_classes = 30
    if args.dataset == 'miniImagenet':
        num_classes = 60
        test_classes = 20
   
    #%% Create model and prepare for training
    #DEVICE = "cuda"
    DEVICE = "cpu"
            
    modelName = "./modelsCluster/"
    if args.model == 'ViTB16':
        print('ViT-B-16')
        NetModel = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model = EmbeddingsModel(NetModel, num_classes, use_fc=False, modelName="ViTB16")
        feat_dim = 768
    if args.model == 'ConvNeXt':
        print('ConvNeXt Base')
        NetModel = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        model = EmbeddingsModel(NetModel, num_classes, use_fc=False, modelName="ConvNeXt")
        modelName += "ConvNeXt_euMoths_episodic_1_0721_221353_AdvLoss.pth"
        #modelName += "ConvNeXt_euMoths_episodic_2_0722_000232_AdvLoss.pth"
        feat_dim = 1024
    if args.model == 'EfficientNetB3':
        print('EfficientNetB3')
        NetModel = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1) # 82.00, 12.2M
        model = EmbeddingsModel(NetModel, num_classes, use_fc=False, modelName="effB3")
        feat_dim = 1536
    if args.model == 'EfficientNetB4':
        print('EfficientNetB4')
        NetModel = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1) # 83.38, 19.3M
        model = EmbeddingsModel(NetModel, num_classes, use_fc=False, modelName="effB4")
        feat_dim = 1792
           
    if args.model == 'resnet50':
        print('resnet50')
        NetModel = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # 80.86, 25.6M
        model = EmbeddingsModel(NetModel, num_classes, use_fc=False)
        #ResNetModel = resnet50(pretrained=True) # 80.86, 25.6M
        
        #modelName = "./modelsCluster/Resnet50_"+args.weights+"_classic_0_0613_172323_AdvLoss.pth" # SC 0.62 RI 0.40
        modelName = "./modelsAdv/Resnet50_"+args.weights+"_classic_0_0616_165100_AdvLoss.pth" # SC 0.62 RI 0.40
        
        
        #modelName = "./models/Resnet50_"+args.weights+"_model.pth"
        
        #modelName = "./models/Resnet50_"+args.weights+"_episodic_9_0612_163546_AdvLoss.pth"
        
        #modelName = "./models/Resnet50_"+args.weights+"_episodic_9_0608_230616_AdvLoss.pth" # univariant scatter 20 class 5-shot 6-query, acc, 0.71
        #modelName = "./models/Resnet50_"+args.weights+"_episodic_9_0609_074822_AdvLoss.pth" # univariant scatter 5 class 5-shot 6-query, acc, 0.64
        
        #modelName = "./models/Resnet50_"+args.weights+"_classic_0_0610_202623_AdvLoss.pth" # classic training 
        
        #modelName = "./models/Resnet50_"+args.weights+"_episodic_5_0609_203058_AdvLoss1.pth" # multivariant scatter 30 classes 5-shot 6-query, acc, 0.76 0.62 all best
        #modelName = "./models/Resnet50_"+args.weights+"_episodic_0_0610_120617_AdvLoss.pth" # CrossEntropy loss classes 5-shot 6-query, acc,
        
        #modelName = "./models/Resnet50_"+args.weights+"_episodic_5_1118_130758_AdvLoss.pth" # univariant scatter 5 class 5-shot 6-query, 0.75 best loss
        #modelName = "./models/Resnet50_"+args.weights+"_episodic_5_0506_074745_AdvLoss1.pth" # multivariant scatter 30 classes 5-shot 6-query, 0.75 best loss
        feat_dim = 2048
    if args.model == 'resnet34':
        print('resnet34')
        NetModel = resnet34(weights=ResNet34_Weights.IMAGENET1K_V2)
        model = EmbeddingsModel(NetModel, num_classes, use_fc=False)
        #ResNetModel = resnet34(pretrained=True) # 80.86, 25.6M
        modelName = "./models/Resnet34_"+args.weights+"_model.pth"
        feat_dim = 512
    if args.model == 'resnet18':
        print('resnet18')
        NetModel = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) 
        model = EmbeddingsModel(NetModel, num_classes, use_fc=False)
        #NetModel = resnet18(pretrained=True) # 80.86, 25.6M
        #modelName = "./models/Resnet18_"+args.weights+"_model.pth"
        
        # Best univariant model
        #modelName = "./models/Resnet18_"+args.weights+"_episodic_10_0218_092723_AdvLoss.pth" # univariant scatter 20 classes, 0.73
        
        #modelName = "./models/Resnet18_"+args.weights+"_episodic_10_0503_101857_AdvLoss.pth" # univariant scatter 40 classes, 0.71
        #modelName = "./models/Resnet18_"+args.weights+"_episodic_5_0502_214439_AdvLoss2.pth" # multivariant scatter 30 classes 5-shot 6-query, 0.71
        
        # Best multivariant model
        #modelName = "./models/Resnet18_"+args.weights+"_episodic_5_0502_214439_AdvLoss3.pth" # multivariant scatter 30 classes 5-shot 6-query, 0.71, train 0.98
       
        #modelName = "./models/Resnet18_"+args.weights+"_episodic_5_0502_214439_AdvLoss.pth" # multivariant scatter 30 classes 5-shot 6-query, 0.69
        #modelName = "./models/Resnet18_"+args.weights+"_episodic_5_0504_232800_AdvLoss.pth" # multivarian scatter 40 classes 7-shot 4-query 150 epochs, 0.65
        #modelName = "./models/Resnet18_"+args.weights+"_episodic_10_0506_073056_AdvLoss.pth" # multivariant scatter 30 classes 5-shot 6-query, alpha 1.0, train 0.2, very bad
        feat_dim = 512

    
    if args.weights == 'ImageNet':
        print('Using pretrained weights with ImageNet dataset')
    else:
        print('Using saved model weights', modelName)
        modelSaved = torch.load(modelName, map_location=torch.device(DEVICE))
        model.load_state_dict(modelSaved.state_dict())
        subDir = args.weights + '/'
        if os.path.exists(resDir+subDir) == False:
            os.mkdir(resDir+subDir)
            print("Create result directory", resDir+subDir)

    model.eval()
    model = model.to(DEVICE)
    
    #%% Create dataset
    if args.dataset == 'euMoths':
        test_set = FewShotDataset(split="train", image_size=image_size,  root=dataDirEuMoths, training=False)
        test_classes = num_classes
        #test_set = FewShotDataset(split="val", image_size=image_size,  root=dataDirEuMoths, training=False)
        #test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirEuMoths, training=False)
        print("euMoths Test dataset")
    if args.dataset == 'CUB':
        #test_set = FewShotDataset(split="train", image_size=image_size, training=False)
        #test_set = FewShotDataset(split="val", image_size=image_size, training=False)
        test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirCUB, training=False)
        print("CUB Test dataset")
    if args.dataset == 'miniImagenet':
        #test_set = MiniImageNet(root=dataDirMiniImageNet+'/images', specs_file=dataDirMiniImageNet+'/test.csv', image_size=image_size, training=False)
        #test_set = MiniImageNet(root=dataDirMiniImageNet+'/images', split="test", image_size=image_size, training=False)
        test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirMiniImageNet, training=False)
        print("miniImageNet Test dataset")
    
    dataloader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
        shuffle=True,
    )
    
    #%% Predict embeddings 
    embeddings_df = predict_embeddings(dataloader, model, device=DEVICE)
    
    #print(embeddings_df)
    #print(embeddings_df.dtypes)
        
    features_dataset = FeaturesDataset.from_dataframe(embeddings_df)
    
    #print(features_dataset[0])
    
    #Embedding
    #print(features_dataset[100][0].numpy())
    features_all = features_dataset[:][0].numpy()
    #Class_name
    #print(features_dataset[100][1])
    labels_all = features_dataset[:][1]

    #%% Plot feature space    
    
    if args.method == "ALL":
        computeClusterPerformance(test_classes, features_all, labels_all)   
    else:
        if args.method == 'GMM':
            print("Gausian Mixture Models")
            gmm = GaussianMixture(n_components=test_classes, covariance_type='full', random_state=42)
            predictions_all = gmm.fit(features_all).predict(features_all)
    
        if args.method == 'IsoForest': #NA
            print("Isolated Forest")
            forest = IsolationForest(n_estimators=test_classes, warm_start=True)
            predictions_all = forest.fit(features_all).predict(features_all)
           
        if args.method == 'Kmeans':
            print("Kmeans clustering")
            kmeans = KMeans(n_clusters=test_classes, random_state=0, n_init="auto")
            predictions_all = kmeans.fit(features_all).predict(features_all)
                   
        if args.method == 'SpecClust':
            print("Spectral clustering")
            sc = SpectralClustering(n_clusters=test_classes,
                                    affinity='nearest_neighbors', 
                                    assign_labels='kmeans',
                                    eigen_solver='arpack',
                                    random_state=0)
            predictions_all = sc.fit_predict(features_all)  
        
        if args.method == 'Agglomerative':
            ac = AgglomerativeClustering(
                linkage="average",
                #metric="cityblock",
                metric="euclidean",
                connectivity=None, # Structured none
                n_clusters=test_classes
            )
            predictions_all = ac.fit_predict(features_all)  
            
        if args.method == 'DBSCANClust': #NA
            dbscan = DBSCAN(eps=0.2, metric="cosine", n_jobs=6)
            embs = features_all / np.linalg.norm(features_all, axis=1, keepdims=True)
            dbscan.fit(embs)
            predictions_all = dbscan.labels_
            
        if args.method == 'HDBSCANClust': 
            dbscan = HDBSCAN(min_cluster_size=3, store_centers='centroid')
            embs = features_all / np.linalg.norm(features_all, axis=1, keepdims=True)
            dbscan.fit(embs)
            predictions_all = dbscan.labels_ 
            centroids = dbscan.centroids_ 
            print("Number of centroids", len(centroids))
            
        accuracy, TP = computeAccuracy(np.array(labels_all), predictions_all)
        print("Similarity", str(test_classes) + " classes", accuracy, TP, len(predictions_all))
        score = adjusted_rand_score(np.array(labels_all), predictions_all)
        print("Rand index score", str(test_classes) + " classes", score)
    
        #%% Select only 8 classes for visual illustration
        # index = np.where((predictions_all == 1) | 
        #                  (predictions_all == 4) |
        #                  (predictions_all == 7) |
        #                  (predictions_all == 10) |
        #                  (predictions_all == 12) |
        #                  (predictions_all == 15) |
        #                  (predictions_all == 17) |   
        #                  (predictions_all == 19))
        index = np.where((predictions_all == 1) | 
                         (predictions_all == 2) |
                         (predictions_all == 3) |
                         (predictions_all == 4) |
                         (predictions_all == 5) |
                         (predictions_all == 6) |
                         (predictions_all == 7) |   
                         (predictions_all == 8) |
                         (predictions_all == 9) |
                         (predictions_all == 10)| 
                         (predictions_all == 11) 
                         )
    
        features = features_all[index]
        predictions = predictions_all[index]
        labels = np.array(labels_all)[index]  
    
        accuracy, TP = computeAccuracy(labels, predictions)
        print("Cluster accuracy 11 classes", accuracy, TP, len(predictions))
        # Compute accuracy
        score = adjusted_rand_score(labels, predictions)
        print("Adjusted rand index score 11 classes ", score)
    
        # Best performance with ResNet18 - episodic training and scatter loss
        # Accuracy 50 classes 0.74, 8 classes 0.69
        # without episodic scatter loss training
        # Accuracy 50 classes 0.53, 8 classes 0.41
    
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(features)
        df = pd.DataFrame()
        df["y"] = predictions
        df["Comp-1"] = z[:,0]
        df["Comp-2"] = z[:,1]
        
        sns.scatterplot(x="Comp-1", y="Comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", 11),
        #                data=df).set(title="ImageNet train dataset T-SNE projection")
        #                data=df).set(title="Clustering of EU Moths (Pretrained)")
                       data=df).set(title="")
        plt.legend(bbox_to_anchor=(1.2, 1), borderaxespad=0)
        plt.show()
        #plot_gmm(gmm, features_all) 
    
