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

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet18, ResNet18_Weights

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

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
            
#%% MAIN
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50') #resnet18, resnet34, resnet50
    parser.add_argument('--weights', default='euMoths') #ImageNet, euMoths, CUB
    parser.add_argument('--dataset', default='euMoths') #miniImagenet, euMoths, CUB
    parser.add_argument('--method', default='Kmeans') #IsoForest, GMM, Kmeans, SpecClust
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
    
    if args.model == 'resnet50':
        print('resnet50')
        ResNetModel = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # 80.86, 25.6M
        #ResNetModel = resnet50(pretrained=True) # 80.86, 25.6M
        #modelName = "./models/Resnet50_"+args.weights+"_model.pth"
        modelName = "./models/Resnet50_"+args.weights+"_episodic_5_0506_074745_AdvLoss.pth" # multivariant scatter 30 classes 5-shot 6-query
        feat_dim = 2048
    if args.model == 'resnet34':
        print('resnet34')
        ResNetModel = resnet34(weights=ResNet34_Weights.IMAGENET1K_V2)
        #ResNetModel = resnet34(pretrained=True) # 80.86, 25.6M
        modelName = "./models/Resnet34_"+args.weights+"_model.pth"
        feat_dim = 512
    if args.model == 'resnet18':
        print('resnet18')
        ResNetModel = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) 
        ResNetModel = resnet18(pretrained=True) # 80.86, 25.6M
        #modelName = "./models/Resnet18_"+args.weights+"_model.pth"
        #modelName = "./models/Resnet18_"+args.weights+"_episodic_10_0218_092723_AdvLoss.pth" # univariant scatter 20 classes, 0.73
        #modelName = "./models/Resnet18_"+args.weights+"_episodic_10_0503_101857_AdvLoss.pth" # univariant scatter 40 classes, 0.71
        #modelName = "./models/Resnet18_"+args.weights+"_episodic_5_0502_214439_AdvLoss2.pth" # multivariant scatter 30 classes 5-shot 6-query, 0.71
        modelName = "./models/Resnet18_"+args.weights+"_episodic_5_0502_214439_AdvLoss3.pth" # multivariant scatter 30 classes 5-shot 6-query, 0.71, train 0.98
        #modelName = "./models/Resnet18_"+args.weights+"_episodic_5_0502_214439_AdvLoss.pth" # multivariant scatter 30 classes 5-shot 6-query, 0.69
        #modelName = "./models/Resnet18_"+args.weights+"_episodic_10_0503_101857_AdvLoss.pth" # multivarian scatter 40 classes 7-shot 4-query 50 epochs
        #modelName = "./models/Resnet18_"+args.weights+"_episodic_5_0504_232800_AdvLoss.pth" # multivarian scatter 40 classes 7-shot 4-query 150 epochs, 0.65
        feat_dim = 512

    model = EmbeddingsModel(ResNetModel, num_classes, use_fc=False)
    
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
        #test_set = FewShotDataset(split="train", image_size=image_size,  root=dataDirEuMoths, training=False)
        #test_classes = num_classes
        #test_set = FewShotDataset(split="val", image_size=image_size,  root=dataDirEuMoths, training=False)
        test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirEuMoths, training=False)
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
               
    if args.method == 'SpecClust': #NA
        print("Spectral clustering")
        sc = SpectralClustering(n_clusters=test_classes, affinity='precomputed', n_init=100,
                                assign_labels='discretize')
        predictions_all = sc.fit_predict(features_all)  
        
    accuracy, TP = computeAccuracy(np.array(labels_all), predictions_all)
    print("Accuracy", str(test_classes) + " classes", accuracy, TP, len(predictions_all))

    #%% Select only 8 classes for visual illustration
    index = np.where((predictions_all == 1) | 
                     (predictions_all == 4) |
                     (predictions_all == 7) |
                     (predictions_all == 10) |
                     (predictions_all == 12) |
                     (predictions_all == 15) |
                     (predictions_all == 17) |   
                     (predictions_all == 19))

    features = features_all[index]
    predictions = predictions_all[index]
    labels = np.array(labels_all)[index]  

    accuracy, TP = computeAccuracy(labels, predictions)
    print("Accuracy 8 classes", accuracy, TP, len(predictions))

    # Best performance with ResNet18 - episodic training and scatter loss
    # Accuracy 50 classes 0.74, 8 classes 0.69
    # without episodic scatter loss training
    # Accuracy 50 classes 0.53, 8 classes 0.41

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(features)
    df = pd.DataFrame()
    df["y"] = predictions
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 8),
    #                data=df).set(title="ImageNet train dataset T-SNE projection")
                    data=df).set(title="EU moths dataset T-SNE projection")
   

    #plot_gmm(gmm, features_all) 
    
