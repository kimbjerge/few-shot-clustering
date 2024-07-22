# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:26:30 2024

@author: Kim Bjerge
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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
        
    
#%% MAIN
if __name__=='__main__':

    #RIscore = []
    #alphaValues = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Fields "ModelDir,Model,TrainMethod,Dataset,ValTest,BatchSize,Classes,RIscore,SCscore,Alpha,ModelName\n"
    
    index = 1
    data_df = pd.read_csv("./result/clustering/resnet50_euMoths_modelsRes50_" + str(index) + "_cluster_test.txt")
    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]
    plotClusterScore(data_df, "(ResNet50, EU Moths)")
    #data_df = pd.read_csv("./result/clustering/resnet50_euMoths_cluster_test.txt")
    #data_df = pd.read_csv("./result/clustering/resnet50_euMoths_modelsRes50_1_cluster_validate.txt")


    data_df = pd.read_csv("./result/clustering/resnet50_miniImagenet_cluster_test.txt")
    plotClusterScore(data_df, "(ResNet50, Mini)")
    data_df = pd.read_csv("./result/clustering/resnet50_CUB_cluster_test.txt")
    plotClusterScore(data_df, "(ResNet50, CUB)")
    data_df = pd.read_csv("./result/clustering/resnet50_tiered_imagenet_cluster_test.txt")
    plotClusterScore(data_df, "(ResNet50, Tiered)")

    data_df = pd.read_csv("./result/clustering/efficientnetb3_euMoths_cluster_test.txt")
    plotClusterScore(data_df, "(EfficientNetB3, EU Moths)")
    
    data_df = pd.read_csv("./result/clustering/convnext_euMoths_cluster_test.txt")
    plotClusterScore(data_df, "(ConvNeXt, EU Moths)")
    