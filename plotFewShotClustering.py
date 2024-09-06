# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:26:30 2024

@author: Kim Bjerge
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

alpha = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
seeds = [0, 37, 74, 158, 261]

def createTableDataPaper(pretrainedPath, filePath, datasetName, modelNames, clusterAlgos):

    print("===============================================================")
    print(datasetName, "ClusterAlgo Fine-tuned Method Alpha CA MI NMI ARI")
    tableText = ""
    for modelName in modelNames:
        for clusterAlgo in clusterAlgos:
            CA_pretrained = []
            CA_classic = []
            CA_episodic = []
            AMI_pretrained = []
            AMI_classic = []
            AMI_episodic = []
            NMI_pretrained = []
            NMI_classic = []
            NMI_episodic = []
            ARI_pretrained = []
            ARI_classic = []
            ARI_episodic = []
            filesExist = False
            for seed in seeds:
                fileName = modelName + '_' + datasetName  
                pretrainedFile = pretrainedPath + fileName + f'_cluster_test_0.txt' 
                trainedFile = filePath + fileName + f'_cluster_test_{seed}.txt' 
                if datasetName == 'tieredImagenet':
                    #trainedClassicFile = filePath + fileName + f'_cluster_test_{seed}C.txt' 
                    trainedClassicFile = filePath + fileName + f'_cluster_test_{seed}.txt' 
                else:
                    trainedClassicFile = filePath + fileName + f'_cluster_test_{seed}.txt' # Pre-trained on ImageNet
                #print(pretrainedFile, trainedFile)
                
                if os.path.exists(pretrainedFile):
                    data_df_pretrained = pd.read_csv(pretrainedFile)
                    data_df_pretrained = data_df_pretrained.loc[data_df_pretrained['ValTest'] == "Test"]
                else:
                    print("File doesn't exist", pretrainedFile)

                #data_df_pretrained = data_df_pretrained.loc[data_df_pretrained['clusterAlgo'] == clusterAlgo]
                if os.path.exists(trainedFile):

                    data_df_classic = pd.read_csv(trainedClassicFile)
                    data_df_classic = data_df_classic.loc[data_df_classic['ClusterAlgo'] == clusterAlgo]
                    data_df_classic = data_df_classic.loc[data_df_classic['TrainMethod'] == "classic"]

                    data_df = pd.read_csv(trainedFile)
                    data_df = data_df.loc[data_df['ClusterAlgo'] == clusterAlgo]
                    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]  
                    data_df = data_df.sort_values(["Alpha"])
                    
                    metricScore = "SCscore"
                    CA_pretrained.append(data_df_pretrained[metricScore].to_list())
                    classicScore = data_df_classic[metricScore].to_list()
                    if len(classicScore) > 0:
                        CA_classic.append(classicScore)
                    CA_episodic.append(data_df[metricScore].to_list())

                    metricScore = "MIscore"
                    AMI_pretrained.append(data_df_pretrained[metricScore].to_list())
                    classicScore = data_df_classic[metricScore].to_list()
                    if len(classicScore) > 0:
                        AMI_classic.append(classicScore)
                    AMI_episodic.append(data_df[metricScore].to_list())

                    metricScore = "NMIscore"
                    NMI_pretrained.append(data_df_pretrained[metricScore].to_list())
                    classicScore = data_df_classic[metricScore].to_list()
                    if len(classicScore) > 0:
                        NMI_classic.append(classicScore)
                    NMI_episodic.append(data_df[metricScore].to_list())
                        
                    metricScore = "RIscore"
                    ARI_episodic.append(data_df[metricScore].to_list())                    
                    ARI_pretrained.append(data_df_pretrained[metricScore].to_list())
                    classicScore = data_df_classic[metricScore].to_list()
                    if len(classicScore) > 0:
                        ARI_classic.append(classicScore)
                    ARI_episodic.append(data_df[metricScore].to_list())
                    filesExist = True
                else:
                    print("File doesn't exist", trainedFile)
            
            if filesExist:
                #ARI_pretrained_mean = np.mean(ARI_pretrained)

                CA_classic_mean = np.mean(CA_classic)
                CA_classic_std = np.std(CA_classic)       
                CA_episodic_mean = np.mean(CA_episodic, 0)
                CA_episodic_stda = np.std(CA_episodic, 0)
                CA_episodic_max = np.max(CA_episodic_mean)
                CA_episodic_alpha = alpha[np.argmax(CA_episodic_mean)]
                CA_episodic_std = CA_episodic_stda[np.argmax(CA_episodic_mean)]

                NMI_classic_mean = np.mean(NMI_classic)
                NMI_classic_std = np.std(NMI_classic)
                NMI_episodic_mean = np.mean(NMI_episodic, 0)
                NMI_episodic_stda = np.std(NMI_episodic, 0)
                NMI_episodic_max = np.max(NMI_episodic_mean)
                NMI_episodic_alpha = alpha[np.argmax(NMI_episodic_mean)]
                NMI_episodic_std = NMI_episodic_stda[np.argmax(NMI_episodic_mean)]

                AMI_classic_mean = np.mean(AMI_classic)
                AMI_classic_std = np.std(AMI_classic)
                AMI_episodic_mean = np.mean(AMI_episodic, 0)
                AMI_episodic_stda = np.std(AMI_episodic, 0)
                AMI_episodic_max = np.max(AMI_episodic_mean)
                AMI_episodic_alpha = alpha[np.argmax(AMI_episodic_mean)]
                AMI_episodic_std = AMI_episodic_stda[np.argmax(AMI_episodic_mean)]

                ARI_classic_mean = np.mean(ARI_classic)
                ARI_classic_std = np.std(ARI_classic)
                ARI_episodic_mean = np.mean(ARI_episodic, 0)
                ARI_episodic_stda = np.std(ARI_episodic, 0)
                ARI_episodic_max = np.max(ARI_episodic_mean)
                ARI_episodic_alpha = alpha[np.argmax(ARI_episodic_mean)]
                ARI_episodic_std = ARI_episodic_stda[np.argmax(ARI_episodic_mean)]

                if modelName == "resnet50":
                    modelName = "ResNet50v2"
                if modelName == "efficientnetb3":
                    modelName = "EfficientNetB3"
                if modelName == "convnext":
                    modelName = "ConvNeXt-B"
                if modelName == "vitb16":
                    modelName = "ViT-B/16"
                    
                if clusterAlgo == "Kmeans":
                    clusterName = "K-means"
                if clusterAlgo == "SpecClust":
                    clusterName = "Spectral"
                
                text = modelName + " & "
                text += clusterName + " & "
                text += "Classic & "
                text += "- & "
                text += "%0.3f(%0.3f) & " % (CA_classic_mean, CA_classic_std)
                text += "%0.3f(%0.3f) & " % (NMI_classic_mean, NMI_classic_std)
                text += "%0.3f(%0.3f) & " % (AMI_classic_mean, AMI_classic_std)
                text += "%0.3f(%0.3f) " % (ARI_classic_mean, ARI_classic_std)
                text += "\\\\\n"
                
                text += modelName + " & "
                text += clusterName + " & "
                text += "Episodic & "
                text += "%0.1f & " % (ARI_episodic_alpha)
                text += "%0.3f(%0.3f) & " % (CA_episodic_max, CA_episodic_std)
                text += "%0.3f(%0.3f) & " % (NMI_episodic_max, NMI_episodic_std)
                text += "%0.3f(%0.3f) & " % (AMI_episodic_max, AMI_episodic_std)
                text += "%0.3f(%0.3f) " % (ARI_episodic_max, ARI_episodic_std)
                text += "\\\\\n"
                
                print(text)
                print(CA_episodic_alpha, NMI_episodic_alpha, AMI_episodic_alpha, ARI_episodic_alpha)
                tableText += text
                
    return tableText

def plotBestScores(pretrainedPath, filePath, datasetName, metricScore, modelNames, clusterAlgos):
    
    plt.rcParams.update({'font.size': 14})
    figure = plt.figure(figsize=(11,11))
    figure.tight_layout(pad=1.0)

    subplots=[221, 222, 223, 224]    

    print("===============================================================")
    print(datasetName, "ClusterAlgo Pretrained Classic Episodic Alpha")
    for modelName in modelNames:
        for clusterAlgo in clusterAlgos:
            ARI_pretrained = []
            ARI_classic = []
            ARI_episodic = []
            filesExist = False
            for seed in seeds:
                fileName = modelName + '_' + datasetName  
                pretrainedFile = pretrainedPath + fileName + f'_cluster_test_0.txt' 
                trainedFile = filePath + fileName + f'_cluster_test_{seed}.txt' 
                if datasetName == 'tieredImagenet':
                    #trainedClassicFile = filePath + fileName + f'_cluster_test_{seed}C.txt' 
                    trainedClassicFile = filePath + fileName + f'_cluster_test_{seed}.txt' 
                else:
                    trainedClassicFile = filePath + fileName + f'_cluster_test_{seed}.txt' 
                #print(pretrainedFile, trainedFile)
                
                if os.path.exists(pretrainedFile):
                    data_df_pretrained = pd.read_csv(pretrainedFile)
                    data_df_pretrained = data_df_pretrained.loc[data_df_pretrained['ValTest'] == "Test"]
                else:
                    print("File doesn't exist", pretrainedFile)

                #data_df_pretrained = data_df_pretrained.loc[data_df_pretrained['clusterAlgo'] == clusterAlgo]
                if os.path.exists(trainedFile):

                    data_df_classic = pd.read_csv(trainedClassicFile)
                    data_df_classic = data_df_classic.loc[data_df_classic['ClusterAlgo'] == clusterAlgo]
                    data_df_classic = data_df_classic.loc[data_df_classic['TrainMethod'] == "classic"]

                    data_df = pd.read_csv(trainedFile)
                    data_df = data_df.loc[data_df['ClusterAlgo'] == clusterAlgo]
                    data_df = data_df.loc[data_df['TrainMethod'] == "episodic"]  
                    data_df = data_df.sort_values(["Alpha"])

                    ARI_pretrained.append(data_df_pretrained[metricScore].to_list())
                    classicScore = data_df_classic[metricScore].to_list()
                    if len(classicScore) > 0:
                        ARI_classic.append(classicScore)
                    ARI_episodic.append(data_df[metricScore].to_list())
                                        
                    filesExist = True
                else:
                    print("File doesn't exist", trainedFile)
            
            if filesExist:
                ARI_pretrained_mean = np.mean(ARI_pretrained)
                ARI_classic_mean = np.mean(ARI_classic)
                ARI_episodic_mean = np.mean(ARI_episodic, 0)
                ARI_episodic_max = np.max(ARI_episodic_mean)
                ARI_episodic_alpha = alpha[np.argmax(ARI_episodic_mean)]
                print(modelName, clusterAlgo, ARI_pretrained_mean, ARI_classic_mean, ARI_episodic_max, ARI_episodic_alpha)
            
            methods = ['Pretrained', 'Classic', r'Episodic ($\alpha$=' + f'{ARI_episodic_alpha})']
            plt.subplot(subplots[modelNames.index(modelName)])
            plt.bar(methods, [ARI_pretrained_mean, ARI_classic_mean, ARI_episodic_max], width=0.3, color=["blue", "green", "red"])
            if modelName == "resnet50":
                modelName = "ResNet50v2"
            if modelName == "efficientnetb3":
                modelName = "EfficientNetB3"
            if modelName == "convnext":
                modelName = "ConvNeXt-B"
            if modelName == "vitb16":
                modelName = "ViT-B/16"
            plt.title(modelName)
            plt.ylabel("ARI score")
            if datasetName == "miniImagenet": 
                plt.ylim(0.8, 1)            
            else:
                if datasetName == 'tieredImagenet':
                    plt.ylim(0.4, 1)            
                else:
                    plt.ylim(0.1, 1)
    
        plotName = datasetName
        if datasetName == "euMoths":
            plotName = "EU moths"
        if datasetName == "miniImagenet":
            plotName = "miniImageNet"
        if datasetName == "tieredImagenet":
            plotName = "tieredImageNet"
            
        #if clusterAlgo == "SpecClust":
        #    plt.suptitle(plotName+" dataset with Spectral clustering")
        #else:
        #    plt.suptitle(plotName+" dataset with K-means clustering")
            
                
    plt.show()  

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
        
    plt.plot(alpha, np.mean(AMIscores, 0), color="red")
    plt.scatter(0.0, np.mean(AMIscoresClassic), s=50, edgecolors='black', c="red")
    plt.plot(alpha, np.mean(ARIscores, 0), color="orange")
    plt.scatter(0.0, np.mean(ARIscoresClassic), s=50, edgecolors='black', c="orange")

    print(clusterName, clusterAlgo, "max AMIscore", np.max(np.mean(AMIscores, 0)), "Classic", np.mean(AMIscoresClassic))
    print(clusterName, clusterAlgo, "max ARIscore", np.max(np.mean(ARIscores, 0)), "Classic", np.mean(ARIscoresClassic))

    #plt.title("Clustering score vs. alpha values (Validate dataset)")
    plt.title("Clustering vs. alpha " + text)
    plt.ylabel('Score')
    plt.xlabel('Alpha')
    if "mini" in clusterName:
        plt.ylim(0.8, 1.00) 
    else:
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

def plotRandResults1():

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

def plotRanResult3():   
 
     path = "./result/clustering/"
     
     plotClusterScoresSeeds(path, seeds, "resnet50_euMoths_cluster_test_", "Kmeans",
                            "(ResNet50, EU Moths, K-means)")
     plotClusterScoresSeeds(path, seeds, "efficientnetB3_euMoths_cluster_test_", "Kmeans",
                            "(EfficientNetB3, EU Moths, K-means)")
     plotClusterScoresSeeds(path, seeds, "ConvNeXt_euMoths_cluster_test_", "Kmeans",
                            "(ConvNeXt, EU Moths, K-means)")
     plotClusterScoresSeeds(path, seeds, "ViTB16_CUB_cluster_test_", "Kmeans",
                            "(ViT-B/16, EU Moths, K-means)")

     plotClusterScoresSeeds(path, seeds, "resnet50_euMoths_cluster_test_", "SpecClust",
                            "(ResNet50, EU Moths, Spectral)")
     plotClusterScoresSeeds(path, seeds, "efficientnetB3_euMoths_cluster_test_", "SpecClust",
                            "(EfficientNetB3, EU Moths, Spectral)")
     plotClusterScoresSeeds(path, seeds, "ConvNeXt_euMoths_cluster_test_", "SpecClust",
                            "(ConvNeXt, EU Moths, Spectral)")
     plotClusterScoresSeeds(path, seeds, "ViTB16_euMoths_cluster_test_", "SpecClust",
                            "(ViT-B/16, EU Moths, Spectral)")
     
     plotClusterScoresSeeds(path, seeds, "resnet50_CUB_cluster_test_", "SpecClust",
                            "(ResNet50, CUB, Spectral)")
     plotClusterScoresSeeds(path, seeds, "efficientnetB3_CUB_cluster_test_", "SpecClust",
                            "(EfficientNetB3, CUB, Spectral)")
     plotClusterScoresSeeds(path, seeds, "ConvNeXt_CUB_cluster_test_", "SpecClust",
                            "(ConvNeXt, CUB, Spectral)")
     plotClusterScoresSeeds(path, seeds, "ViTB16_CUB_cluster_test_", "SpecClust",
                            "(ViT-B/16, CUB, Spectral)")
     
     plotClusterScoresSeeds(path, seeds, "resnet50_miniImagenet_cluster_test_", "SpecClust",
                            "(ResNet50, Mini, Spectral)")
     plotClusterScoresSeeds(path, seeds, "efficientnetB3_miniImagenet_cluster_test_", "SpecClust",
                            "(EfficientNetB3, Mini, Spectral)")
     plotClusterScoresSeeds(path, seeds, "ConvNeXt_miniImagenet_cluster_test_", "SpecClust",
                            "(ConvNeXt, Mini, Spectral)")
     plotClusterScoresSeeds(path, seeds, "ViTB16_miniImagenet_cluster_test_", "SpecClust",
                            "(ViT-B/16, Mini, Spectral)")      

#%% MAIN
if __name__=='__main__':
    
    #plotRanResult3()
        
        
    pretrainedPath = "./result/clusteringImgNet/"
    clusteringPath = "./result/clustering/"
    models = ["resnet50", "efficientnetb3", "convnext", "vitb16"]
    clusterAlgos = ["SpecClust"] # Kmeans, SpecClust
    metricScore = "RIscore" # RIscore, MIscore, NMIscore
    
    #dataSet = "tieredImagenet" # euMoths, CUB, miniImagenet, tieredImagenet
    #tableText = createTableDataPaper(pretrainedPath, clusteringPath, dataSet, models, ["Kmeans"])
    #tableText += createTableDataPaper(pretrainedPath, clusteringPath, dataSet, models, ["SpecClust"])
    #print(dataSet)
    #print(tableText)
    
    plotBestScores(pretrainedPath, clusteringPath, "euMoths", metricScore, models, clusterAlgos)
    plotBestScores(pretrainedPath, clusteringPath, "CUB", metricScore, models, clusterAlgos)
    plotBestScores(pretrainedPath, clusteringPath, "miniImagenet", metricScore, models, clusterAlgos)
    plotBestScores(pretrainedPath, clusteringPath, "tieredImagenet", metricScore, models, ["Kmeans"])
