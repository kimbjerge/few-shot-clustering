# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:58:31 2024

@author: Kim Bjerge
"""

import os
import random
import numpy as np
import torch
import argparse
from statistics import mode
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score

from torch.utils.data import DataLoader

from easyfsl.modules import resnet12
from easyfsl.datasets import FeaturesDataset
from easyfsl.utils import predict_embeddings

from FewShotModelData import EmbeddingsModel, FewShotDataset

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.efficientnet import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models.efficientnet import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

#%% K-means cluster evaluation - similar class (SC) score and rand index (RI) score     
def computeSimilarClassScore(labels, predictions):
    
    predDic = {}
    for idx in range(len(predictions)):
        prediction = predictions[idx]
        if prediction in predDic:
            predDic[prediction].append(labels[idx])
        else:
            predDic[prediction] = []
            predDic[prediction].append(labels[idx])
    
    TruePositives = 0
    for i, key in enumerate(predDic):    
        TruePositives += sum(predDic[key]==mode(predDic[key]))
        
    SCscore = TruePositives/len(predictions)
    return SCscore

def evaluateClustering(embeddings_model, val_loader, device, test_classes):
       
    print("Evaluate clustering with K-means on feature embeddings")
    
    embeddings_df = predict_embeddings(val_loader, embeddings_model, device=device)
            
    features_dataset = FeaturesDataset.from_dataframe(embeddings_df)
    features_all = features_dataset[:][0].numpy()
    labels_all = features_dataset[:][1]   
    
    kmeans = KMeans(n_clusters=test_classes, random_state=0, n_init="auto")
    predictions_all = kmeans.fit(features_all).predict(features_all)
    RIscore = adjusted_rand_score(np.array(labels_all), predictions_all)
    MIscore = adjusted_mutual_info_score(np.array(labels_all), predictions_all)
    NMIscore = normalized_mutual_info_score(np.array(labels_all), predictions_all)
    SCscore = computeSimilarClassScore(np.array(labels_all), predictions_all)
    print("Adjusted Rand Index (RI) score",  RIscore, 
          "Adjusted Mutual Info (MI) score", MIscore,
          "Normalized Mutual Info (MI) score", NMIscore,
          "Similar Class (SC) score", SCscore, "for classes", str(test_classes))
    
    return RIscore, MIscore, NMIscore, SCscore

def load_model(modelName, num_classes, argsModel, argsWeights):
    
    if argsModel == 'vitb16':
        print('ViT-B-16')
        NetModel = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model = EmbeddingsModel(NetModel, num_classes, use_fc=False, modelName="ViTB16")
        feat_dim = 768
    if argsModel == 'convnext':
        print('ConvNeXt Base')
        NetModel = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        model = EmbeddingsModel(NetModel, num_classes, use_fc=False, modelName="ConvNeXt")
        feat_dim = 1024
    if argsModel == 'efficientnetb3':
        print('EfficientNetB3')
        NetModel = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1) # 82.00, 12.2M
        model = EmbeddingsModel(NetModel, num_classes, use_fc=False, modelName="effB3")
        feat_dim = 1536
    if argsModel == 'efficientnetb4':
        print('EfficientNetB4')
        NetModel = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1) # 83.38, 19.3M
        model = EmbeddingsModel(NetModel, num_classes, use_fc=False, modelName="effB4")
        feat_dim = 1792
    if argsModel == 'resnet50':
        print('resnet50')
        ResNetModel = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # 80.86, 25.6M
        #ResNetModel = resnet50(pretrained=True) # 80.86, 25.6M
        model = EmbeddingsModel(ResNetModel, num_classes, use_fc=False)
        feat_dim = 2048
    if argsModel == 'resnet34':
        print('resnet34')
        ResNetModel = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1) 
        #ResNetModel = resnet34(pretrained=True) 
        model = EmbeddingsModel(ResNetModel, num_classes, use_fc=False)
        feat_dim = 512
    if argsModel == 'resnet18':
        print('resnet18')
        ResNetModel = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) 
        #ResNetModel = resnet18(pretrained=True) # 80.86, 25.6M
        model = EmbeddingsModel(ResNetModel, num_classes, use_fc=False)
        feat_dim = 512
    if argsModel == 'resnet12':
        print('resnet12')
        model = resnet12(use_fc=False, num_classes=num_classes) #.to(DEVICE)
        feat_dim = 64
    
    if argsWeights == 'ImageNet':
        print('Using pretrained weights with ImageNet dataset')
    else:
        print('Using saved model weights', modelName)
        modelSaved = torch.load(modelName, map_location=torch.device(DEVICE))
        model.load_state_dict(modelSaved.state_dict())

    model.eval()
    model = model.to(DEVICE)
    
    return model, feat_dim


def load_test_dataset(argsDataset, argsLearning):
    
    if argsDataset == 'Omniglot':
        if argsLearning:       
            test_set = FewShotDataset(split="val", image_size=image_size, root=dataDirOmniglot, training=False)
            print("Omniglot Val dataset")
        else:
            test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirOmniglot, training=False)
            print("Omniglot Test dataset")
    if argsDataset == 'euMoths':
        #test_set = FewShotDataset(split="train", image_size=image_size, root=dataDirEuMoths,training=False)
        if argsLearning:
            test_set = FewShotDataset(split="val", image_size=image_size, root=dataDirEuMoths, training=False)
            print("euMoths Val dataset")
        else:
            test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirEuMoths, training=False)
            print("euMoths Test dataset")
    if argsDataset == 'CUB':
        #test_set = FewShotDataset(split="train", image_size=image_size, root=dataDirCUB, training=False)
        if argsLearning:       
            test_set = FewShotDataset(split="val", image_size=image_size, root=dataDirCUB, training=False)
            print("CUB Val dataset")
        else:
            test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirCUB, training=False)
            print("CUB Test dataset")
    if argsDataset == 'tieredImagenet':
        if argsLearning:       
            test_set = FewShotDataset(split="val", image_size=image_size, root=dataDirTieredImageNet, training=False)
            print("tieredImagenet Val dataset")
        else:
            test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirTieredImageNet, training=False)
            print("tieredImagenet Test dataset")
    if argsDataset == 'miniImagenet':
        #test_set = MiniImageNet(root=dataDirMiniImageNet+'/images', specs_file=dataDirMiniImageNet+'/test.csv', image_size=image_size, training=False)
        #test_set = MiniImageNet(root=dataDirMiniImageNet+'/images', split="test", image_size=image_size, training=False)
        if argsLearning:       
            test_set = FewShotDataset(split="val", image_size=image_size, root=dataDirMiniImageNet, training=False)
            print("miniImageNet Val dataset")
        else:
            test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirMiniImageNet, training=False)
            print("miniImageNet Test dataset")
    
    return test_set

#%% MAIN
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    # Arguments to be changed 
    #parser.add_argument('--modelDir', default='modelsOmniglotAdvStd4') #Directory that contains Ominiglot models
    #parser.add_argument('--modelDir', default='modelsOmniglotAdvMulti4') #Directory that contains Ominiglot models
    parser.add_argument('--modelDir', default='modelsAlphaEUMoths20Ways') #Directory that contains Ominiglot models
    #parser.add_argument('--modelDir', default='modelsOmniglot') #Directory that contains Ominiglot models
    parser.add_argument('--batch', default='250', type=int) # training batch size
    parser.add_argument('--device', default='cpu') #cpu or cuda:0-3
    parser.add_argument('--validate', default='', type=bool) #default false when no parameter (Validate or test dataset)

    # Theses arguments must not be changed and will be updated based on the model name
    parser.add_argument('--model', default='') #resnet12 (Omniglot), resnet18, resnet34, resnet50, EfficientNetB3, EfficientNetB4, ConvNeXt, ViTB16 Must be empty
    parser.add_argument('--weights', default='') #ImageNet, mini_imagenet, tiered_imagenet, euMoths, CUB, Omniglot, Must be empty
    parser.add_argument('--dataset', default='') #miniImagenet, tieredImagenet, euMoths, CUB, Omniglot, Must be empty
    parser.add_argument('--alpha', default=0.1, type=float) # No effect
        
    args = parser.parse_args()
 
    random_seed = 0
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    n_workers = 16
    
    resDir = "./result/"
    dataDirMiniImageNet = "./data/mini_imagenet"
    dataDirEuMoths = "./data/euMoths"
    dataDirCUB = "./data/CUB"
    dataDirOmniglot = "./data/Omniglot"
    dataDirTieredImageNet = "./data/tiered_imagenet"
    subDir = "test/"

    if os.path.exists(resDir+subDir) == False:
        os.mkdir(resDir+subDir)
        print("Create result directory", resDir+subDir)
 
    DEVICE = args.device

    modelsInDir = []
    if args.modelDir == "":
        modelsInDir.append(args.model + '_' + args.dataset + '_classic_0_1234_123456.pth') # Create model name 
    else:
        modelsInDir = sorted(os.listdir(args.modelDir))
    #%% Create model and prepare for cluster testing
    for modelName in modelsInDir:
        if '.pth' in modelName:
        #if 'Resnet34_mini_imagenet_episodic_5_1116_141355_AdvLoss.pth' in modelName:
            modelNameSplit = modelName.split('_')
            #if args.model == '':
            args.model = modelNameSplit[0].lower()
                
            if modelNameSplit[2] == 'imagenet':
                if modelNameSplit[1] == 'mini':
                    nameData = "miniImagenet"
                    nameWeights = "mini_imagenet"
                else:
                    nameData = "tieredImagenet"
                    nameWeights = "tiered_imagenet"
                alpha_idx = 4
            else:
                nameData = modelNameSplit[1]
                nameWeights = nameData
                alpha_idx = 3
                
            #if args.weights == '':
            if args.modelDir != "":
                args.weights = nameWeights
            #if args.dataset == '':
            args.dataset = nameData

            args.alpha = int(modelNameSplit[alpha_idx])/10
            
            print(args)
        
            if args.model == 'resnet12':
                image_size = 28 # Omniglot dataset
            else:
                image_size = 224 # ResNet euMoths, EfficientNetB3 (300), ConvNeXt
                          
            num_classes = 100  
            if args.weights == 'CUB':
                num_classes = 140  
            if args.weights == 'Omniglot':
                num_classes = 3856  
            if args.weights == 'mini_imagenet':
                num_classes = 60
            if args.weights == 'tiered_imagenet':
                num_classes = 351 # Val 97, Test 160
            
            dataSetName = "Test"
            if args.validate: 
                dataSetName = "Validate"
                
            #%% Load model
            model, feat_dim = load_model(args.modelDir + '/' +modelName, num_classes, args.model, args.weights)

            #resFileName =  args.model + '_' +  args.dataset + '_' + args.modelDir + "_cluster_test.txt"
            resFileName =  args.model + '_' +  args.dataset + "_cluster_test.txt"
            line = "ModelDir,Model,TrainMethod,Dataset,ValTest,BatchSize,Classes,RIscore,MIscore,NMIscore,SCscore,Alpha,ModelName\n"
            if os.path.exists(resDir+subDir+resFileName):
                resFile = open(resDir+subDir+resFileName, "a")
            else:
                resFile = open(resDir+subDir+resFileName, "w")
                print(line)
                resFile.write(line)
                resFile.flush()                 
                
            #%% Prepare dataset (Validation or test dataset)
            test_set = load_test_dataset(args.dataset, args.validate)
            
            test_loader = DataLoader(
                test_set,
                batch_size=args.batch,
                num_workers=n_workers,
                pin_memory=True,
                shuffle=True,
            )
            test_classes = len(set(test_set.get_labels()))
            print("Test classes", test_classes)
    
            #%% Test clustering
            RIscore, MIscore, NMIscore, SCscore = evaluateClustering(
                model, test_loader, device=DEVICE, test_classes=test_classes
            ) 

            #%% Save results
            trainMethod = modelName.split('_')[2] # Classic or episodic
            if trainMethod == "imagenet": # Mini and Tiered imagenet 
                trainMethod = modelName.split('_')[3] # Classic or episodic                
            line = args.modelDir + ',' + args.model + ',' + trainMethod + ',' + args.dataset + ',' + dataSetName + ',' + str(args.batch) + ',' + str(test_classes) + ',' 
            line += str(RIscore) + ',' + str(MIscore) + ',' + str(NMIscore) + ',' + str(SCscore)  + ','
            line += str(args.alpha) + ',' + args.modelDir + '/' + modelName +  '\n'
            print(line)
            resFile.write(line)    
            resFile.flush()
            resFile.close()
            