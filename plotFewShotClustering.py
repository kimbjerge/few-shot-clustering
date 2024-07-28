# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:26:30 2024

@author: Kim Bjerge
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plotClusterScoresSeeds(filePath, seeds, clusterName, clusterAlgo, text):

    ax = plt.gca()

    AMIscores = []
    ARIscores = []
    AMIscoresClassic = []
    ARIscoresClassic = []
    for seed in seeds:
        fileName = filePath + clusterName + str(seed) + ".txt"
        data_df = pd.read_csv(fileName)
        data_df = data_df.loc[data_df['ClusterAlgo'] == clusterAlgo]
        data_df_classic = data_df.loc[data_df['TrainMethod'] == "classic"]
        data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]  
    
        data_df = data_df.sort_values(["Alpha"])
        AMIscore = data_df['MIscore'].to_list()
        if len(AMIscore) == 11:
            AMIscores.append(AMIscore)
        ARIscore = data_df['RIscore'].to_list()
        if len(ARIscore) == 11:
            ARIscores.append(ARIscore)

        AMIscoresClassic.append(data_df_classic['MIscore'].to_list())
        ARIscoresClassic.append(data_df_classic['RIscore'].to_list())
        
        data_df.plot(kind='line',
                    x='Alpha',
                    y='MIscore',
                    style='.',
                    color='green', ax=ax)
        
        data_df.plot(kind='line',
                    x='Alpha',
                    y='RIscore',
                    style='.',
                    color='blue', ax=ax)
        
    alpha = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.plot(alpha, np.mean(AMIscores, 0), color="red")
    plt.scatter(0.0, np.mean(AMIscoresClassic), s=50, edgecolors='black', c="red")
    plt.plot(alpha, np.mean(ARIscores, 0), color="orange")
    plt.scatter(0.0, np.mean(ARIscoresClassic), s=50, edgecolors='black', c="orange")
    #plt.title("Clustering score vs. alpha values (Validate dataset)")
    plt.title("Clustering vs. alpha " + text)
    plt.ylabel('Score')
    plt.xlabel('Alpha')
    plt.ylim(0.25, 1.00) 
    #plt.xlim(0, 20)
    plt.legend(["AMI score", "ARI score"])
    plt.show()
    

def plotClusterRIMIScoreMulti(filePaths, clusterName, text):

    ax = plt.gca()

    AMIscores = []
    ARIscores = []
    AMIscoresClassic = []
    ARIscoresClassic = []
    for filePath in filePaths:
        fileName = filePath + clusterName
        data_df = pd.read_csv(fileName)
        data_df_classic = data_df.loc[data_df['TrainMethod'] == "classic"] 
        data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]  
    
        data_df = data_df.sort_values(["Alpha"])
        AMIscore = data_df['MIscore'].to_list()
        if len(AMIscore) == 11:
            AMIscores.append(AMIscore)
        ARIscore = data_df['RIscore'].to_list()
        if len(ARIscore) == 11:
            ARIscores.append(ARIscore)

        AMIscoresClassic.append(data_df_classic['MIscore'].to_list())
        ARIscoresClassic.append(data_df_classic['RIscore'].to_list())
        
        data_df.plot(kind='line',
                    x='Alpha',
                    y='MIscore',
                    style='.',
                    color='green', ax=ax)
        
        data_df.plot(kind='line',
                    x='Alpha',
                    y='RIscore',
                    style='.',
                    color='blue', ax=ax)
        
    alpha = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.plot(alpha, np.mean(AMIscores, 0), color="red")
    plt.scatter(0.1, np.mean(AMIscoresClassic), s=50, edgecolors='black', c="red")
    plt.plot(alpha, np.mean(ARIscores, 0), color="orange")
    plt.scatter(0.1, np.mean(ARIscoresClassic), s=50, edgecolors='black', c="orange")
    #plt.title("Clustering score vs. alpha values (Validate dataset)")
    plt.title("Clustering score vs. alpha " + text)
    plt.ylabel('Score')
    plt.xlabel('Alpha')
    plt.ylim(0.25, 0.95) 
    #plt.xlim(0, 20)
    plt.legend(["AMI score", "ARI score"])
    plt.show()


def plotClusterRIMIScore(data_df, text):

    ax = plt.gca()
    
    data_df = data_df.sort_values(["Alpha"])
    
    data_df.plot(kind='line',
                x='Alpha',
                y='MIscore',
                color='green', ax=ax)
    
    data_df.plot(kind='line',
                x='Alpha',
                y='RIscore',
                color='blue', ax=ax)
    
    #plt.title("Clustering score vs. alpha values (Validate dataset)")
    plt.title("Clustering score vs. alpha " + text)
    plt.ylabel('Score')
    plt.xlabel('Alpha')
    #plt.ylim(0.4, 0.8) 
    #plt.xlim(0, 20)
    plt.legend(["AMI score", "ARI score"])
    plt.show()
        
    
def plotClusterScore(data_df, text):

    ax = plt.gca()
    data_df.plot(kind='line',
                x='Alpha',
                y='RIscore',
                color='blue', ax=ax)
    
    data_df.plot(kind='line',
                x='Alpha',
                y='SCscore',
                color='green', ax=ax)

    #plt.title("Clustering score vs. alpha values (Validate dataset)")
    plt.title("Clustering score vs. alpha " + text)
    plt.ylabel('Score')
    plt.xlabel('Alpha')
    #plt.ylim(0.4, 0.8) 
    #plt.xlim(0, 20)
    plt.legend(["RI score", "SC score"])
    plt.show()
        

def plotFirstResults():

    #RIscore = []
    #alphaValues = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Fields "ModelDir,Model,TrainMethod,Dataset,ValTest,BatchSize,Classes,RIscore,SCscore,Alpha,ModelName\n"
    
    index = 1
    #data_df = pd.read_csv("./result/clustering/first/resnet50_euMoths_modelsRes50_" + str(index) + "_cluster_test.txt")
    #data_df = pd.read_csv("./result/clustering/first/resnet50_euMoths_cluster_validate.txt")
    data_df = pd.read_csv("./result/clustering/first/resnet50_euMoths_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterScore(data_df, "(ResNet50, EU Moths)")


    data_df = pd.read_csv("./result/clustering/first/resnet50_miniImagenet_cluster_test.txt")
    plotClusterScore(data_df, "(ResNet50, Mini)")
    data_df = pd.read_csv("./result/clustering/first/resnet50_CUB_cluster_test.txt")
    plotClusterScore(data_df, "(ResNet50, CUB)")
    data_df = pd.read_csv("./result/clustering/first/resnet50_tieredImagenet_cluster_test.txt")
    plotClusterScore(data_df, "(ResNet50, Tiered)")

    data_df = pd.read_csv("./result/clustering/first/efficientnetb3_euMoths_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterScore(data_df, "(EfficientNetB3, EU Moths)")
    
    data_df = pd.read_csv("./result/clustering/first/convnext_euMoths_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterScore(data_df, "(ConvNeXt, EU Moths)")

    data_df = pd.read_csv("./result/clustering/first/vitb16_euMoths_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterScore(data_df, "(ViT-B-16, EU Moths)")

def plotRandResults():

    #path = "./result/clustering/rand0/"
    #path = "./result/clustering/rand37/"
    #path = "./result/clustering/rand74/"
    #path = "./result/clustering/rand158/"
    path = "./result/clustering/rand261/"
    #plotFirstResults()
    data_df = pd.read_csv(path + "resnet50_euMoths_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterRIMIScore(data_df, "(ResNet50, EU Moths)")

    data_df = pd.read_csv(path + "efficientnetB3_euMoths_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterRIMIScore(data_df, "(EfficientNetB3, EU Moths)")
    
    data_df = pd.read_csv(path + "ConvNeXt_euMoths_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterRIMIScore(data_df, "(ConvNeXt, EU Moths)")
    
    data_df = pd.read_csv(path + "ViTB16_euMoths_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterRIMIScore(data_df, "(ViT-B-16, EU Moths)")
    
    data_df = pd.read_csv(path + "resnet50_CUB_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterRIMIScore(data_df, "(ResNet50, CUB)")
    
    data_df = pd.read_csv(path + "efficientnetB3_CUB_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterRIMIScore(data_df, "(EfficientNetB3, CUB)")
    
    data_df = pd.read_csv(path + "ConvNeXt_CUB_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterRIMIScore(data_df, "(ConvNeXt, CUB)")
    
    data_df = pd.read_csv(path + "ViTB16_CUB_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterRIMIScore(data_df, "(ViT-B-16, CUB)")

    data_df = pd.read_csv(path + "resnet50_miniImagenet_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterRIMIScore(data_df, "(ResNet50, Mini Imagenet)")
    
    data_df = pd.read_csv(path + "resnet50_tieredImagenet_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterRIMIScore(data_df, "(ResNet50, Tiered Imagenet)")

def plotRanResult2():

    paths = ["./result/clustering/rand0/",
             "./result/clustering/rand37/",
             "./result/clustering/rand74/",
             "./result/clustering/rand158/",
             "./result/clustering/rand261/"]
    
    plotClusterRIMIScoreMulti(paths, "resnet50_euMoths_cluster_test.txt", 
                              "(ResNet50, EU Moths)")
    plotClusterRIMIScoreMulti(paths, "efficientnetB3_euMoths_cluster_test.txt", 
                              "(EfficientNetB3, EU Moths)")
    plotClusterRIMIScoreMulti(paths, "ConvNeXt_euMoths_cluster_test.txt", 
                              "(ConvNeXt, EU Moths)")
    plotClusterRIMIScoreMulti(paths, "ViTB16_euMoths_cluster_test.txt", 
                              "(ViT-B-16, EU Moths)")
    plotClusterRIMIScoreMulti(paths, "resnet50_CUB_cluster_test.txt", 
                              "(ResNet50, CUB)")
    plotClusterRIMIScoreMulti(paths, "efficientnetB3_CUB_cluster_test.txt", 
                              "(EfficientNetB3, CUB)")
    plotClusterRIMIScoreMulti(paths, "ConvNeXt_CUB_cluster_test.txt", 
                              "(ConvNeXt, CUB)")
    plotClusterRIMIScoreMulti(paths, "ViTB16_CUB_cluster_test.txt", 
                              "(ViT-B-16, CUB)")
    plotClusterRIMIScoreMulti(paths, "resnet50_miniImagenet_cluster_test.txt", 
                              "(ResNet50, Mini Imagenet)")
    plotClusterRIMIScoreMulti(paths, "resnet50_tieredImagenet_cluster_test.txt", 
                              "(ResNet50, Tiered Imagenet)")

    
       
#%% MAIN
if __name__=='__main__':
    
    path = "./result/clustering/"
    
    seeds = [0, 37, 74, 158, 261]
    
    plotClusterScoresSeeds(path, seeds, "resnet50_euMoths_cluster_test_", "Kmeans",
                           "(ResNet50, EU Moths, K-means)")
    plotClusterScoresSeeds(path, seeds, "efficientnetB3_euMoths_cluster_test_", "Kmeans",
                           "(EfficientNetB3, EU Moths, K-means)")
    plotClusterScoresSeeds(path, seeds, "ConvNeXt_euMoths_cluster_test_", "Kmeans",
                           "(ConvNeXt, EU Moths, K-means)")
    plotClusterScoresSeeds(path, seeds, "ViTB16_CUB_cluster_test_", "Kmeans",
                           "(ViT-B-16, EU Moths, K-means)")

    plotClusterScoresSeeds(path, seeds, "resnet50_euMoths_cluster_test_", "SpecClust",
                           "(ResNet50, EU Moths, Spectral)")
    plotClusterScoresSeeds(path, seeds, "efficientnetB3_euMoths_cluster_test_", "SpecClust",
                           "(EfficientNetB3, EU Moths, Spectral)")
    plotClusterScoresSeeds(path, seeds, "ConvNeXt_euMoths_cluster_test_", "SpecClust",
                           "(ConvNeXt, EU Moths, Spectral)")
    plotClusterScoresSeeds(path, seeds, "ViTB16_euMoths_cluster_test_", "SpecClust",
                           "(ViT-B-16, EU Moths, Spectral)")
    
    plotClusterScoresSeeds(path, seeds, "resnet50_CUB_cluster_test_", "SpecClust",
                           "(ResNet50, CUB, Spectral)")
    plotClusterScoresSeeds(path, seeds, "efficientnetB3_CUB_cluster_test_", "SpecClust",
                           "(EfficientNetB3, CUB, Spectral)")
    plotClusterScoresSeeds(path, seeds, "ConvNeXt_CUB_cluster_test_", "SpecClust",
                           "(ConvNeXt, CUB, Spectral)")
    plotClusterScoresSeeds(path, seeds, "ViTB16_CUB_cluster_test_", "SpecClust",
                           "(ViT-B-16, CUB, Spectral)")
    
    plotClusterScoresSeeds(path, seeds, "resnet50_miniImagenet_cluster_test_", "SpecClust",
                           "(ResNet50, Mini, Spectral)")
    #plotClusterScoresSeeds(path, seeds, "efficientnetB3_miniImagenet_cluster_test_", "SpecClust",
    #                       "(EfficientNetB3, Mini, Spectral)")
    plotClusterScoresSeeds(path, seeds, "ConvNeXt_miniImagenet_cluster_test_", "SpecClust",
                           "(ConvNeXt, Mini, Spectral)")
    plotClusterScoresSeeds(path, seeds, "ViTB16_miniImagenet_cluster_test_", "SpecClust",
                           "(ViT-B-16, Mini, Spectral)")