# -*- coding: utf-8 -*-

"""
Created on Fri Jun 13 09:32:35 2024

@author: Kim Bjerge
"""

import random
import argparse
import numpy as np
import torch
from statistics import mean
from torch import nn
from tqdm import tqdm
from datetime import datetime

from PrototypicalNetworksNovelty import PrototypicalNetworksNovelty

from easyfsl.datasets import FeaturesDataset
from easyfsl.modules import resnet12
from easyfsl.methods import FewShotClassifier
from easyfsl.samplers import TaskSampler
from easyfsl.utils import evaluate
from easyfsl.utils import predict_embeddings

from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from statistics import mode

# See performance on ImageNet: https://pytorch.org/vision/0.18/models.html

from torchvision.models.efficientnet import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models.efficientnet import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
#from torchvision.models.efficientnet import efficientnet_b7 #, EfficientNet_B7_Weights

from FewShotModelData import EmbeddingsModel, FewShotDataset

IMAGENET_NORMALIZATION = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

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
    SCscore = computeSimilarClassScore(np.array(labels_all), predictions_all)
    print("Rand index (RI) score",  RIscore, "Similar class (SC) score", SCscore, "for classes", str(test_classes))
    
    return RIscore, SCscore
                       
#%% Classical training      
def train_epoch(entropyLossFunction: nn.CrossEntropyLoss, 
                   model_: nn.Module, 
                   data_loader: DataLoader, 
                   optimizer: Optimizer):
    
    all_loss = []
    model_.train()
    with tqdm(data_loader, total=len(data_loader), desc="Training") as tqdm_train:
        for images, labels in tqdm_train:
            optimizer.zero_grad()

            predictions = model_(images.to(DEVICE))
            loss = entropyLossFunction(predictions, labels.to(DEVICE))
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss="{}".format(mean(all_loss)))

    return mean(all_loss)


def classicTrain(model, modelName, train_loader, val_loader, few_shot_classifier,  
                 pretrained=False, m1=50, m2=80, n_epochs=100, learnRate=5e-4,  
                 test_classes=0, evaluateCluster=True):

    scheduler_milestones = [m1, m2] # From scratch with 1500 epochs
        
    learning_rate = learnRate

    scheduler_gamma = 0.1
   
    entropyLossFunction = nn.CrossEntropyLoss()
   
    train_optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    #train_optimizer = Adam(model.parameters(), lr=learning_rate) # Not working
    
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )
    
    #tb_logs_dir = Path("./logs")   
    #tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))
    log_dir = '-' + modelName.split('/')[2].replace(".pth", "")
    tb_writer = SummaryWriter(comment=log_dir)
    
    best_state = model.state_dict()
    best_validation_accuracy = 0.0
    validation_frequency = 5
    best_epoch = 0
    best_loss = 1000.0
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        average_loss = train_epoch(entropyLossFunction, model, train_loader, train_optimizer)
    
        if epoch % validation_frequency == validation_frequency - 1:
    
            # We use this very convenient method from EasyFSL's ResNet to specify
            # that the model shouldn't use its last fully connected layer during validation.
            model.set_use_fc(False)
            model.eval()
                    
            if evaluateCluster:
                RIscore, SCscore = evaluateClustering(
                    model, val_loader, device=DEVICE, test_classes=test_classes
                )      
                #validation_accuracy = (RIscore + SCscore)/2
                validation_accuracy = SCscore
            else:
                validation_accuracy = evaluate(
                    few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation"
                )
        
            model.set_use_fc(True)
    
            if validation_accuracy > best_validation_accuracy:
                best_epoch = epoch+1
                best_loss = average_loss
                best_validation_accuracy = validation_accuracy
                best_state = model.state_dict()
                print("Ding ding ding! We found a new best model!")
                torch.save(model, modelName)
                print("Best model saved", modelName)
    
            tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)
    
        tb_writer.add_scalar("Train/loss", average_loss, epoch)
    
        # Warn the scheduler that we did an epoch
        # so it knows when to decrease the learning rate
        train_scheduler.step()
    
    print("Best validation accuracy after epoch", best_validation_accuracy, best_epoch)
    
    return best_state, model, best_epoch, best_validation_accuracy, best_loss
    

#%% Episodic training      
def train_episodic_epoch(lossFunction, 
                         model: FewShotClassifier, 
                         data_loader: DataLoader, 
                         optimizer: Optimizer,
                         slossFunc,
                         alpha,
                         cosine):
    all_loss = []
    all_closs = []
    all_sloss = []
    all_scatter_between = []
    model.train()
    with tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Training"
    ) as tqdm_train:
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) in tqdm_train:

            optimizer.zero_grad()
            model.process_support_set(
                support_images.to(DEVICE), support_labels.to(DEVICE)
            )

            classification_scores = model(query_images.to(DEVICE))
            
            if slossFunc == "Triple": # These training methods do not work
                #closs = model.tripleMarginDistanceLoss(classification_scores, query_labels.to(DEVICE), margin=1.0) #10
                closs = model.contrastiveLoss(classification_scores, query_labels.to(DEVICE))
            else:
                closs = lossFunction(classification_scores, query_labels.to(DEVICE))

            if slossFunc == "Multi" or slossFunc == "MultiAlt" or slossFunc == "Triple":
                ScatterBetween, ScatterWithin, sloss = model.multivariantScatterLoss()
                if slossFunc == "MultiAlt":
                    sloss = 100000/ScatterBetween   
            else:
                correct_episodes = classification_scores[torch.max(classification_scores, 1)[1] == query_labels.to(DEVICE)]
                correct_scores = correct_episodes.max(1)[0]
                correct_pred_idx = correct_episodes.max(1)[1]            
                
                #Select scores part of correct predicitons that don't belong to the query label
                num_rows = correct_episodes.shape[0]
                num_cols = correct_episodes.shape[1]
                wrong_scores = torch.empty(num_rows*(num_cols-1)).to(DEVICE)
                idx = 0
                for i in range(num_rows):
                    for j in range(num_cols):
                        if j != correct_pred_idx[i]:
                            wrong_scores[idx]=correct_episodes[i][j]
                            idx += 1
                
                ScatterWithin = 1 # Mean only
                if slossFunc == "Var": # Mean and variance   
                    ScatterWithin = correct_scores.var() + wrong_scores.var()
                if slossFunc == "Std": # Mean and standard deviation
                    ScatterWithin = correct_scores.std() + wrong_scores.std()
                
                ScatterBetween = abs(correct_scores.mean() - wrong_scores.mean())
                sloss = ScatterWithin/ScatterBetween # Minimize scatter within related to scatter between      
 
            if torch.isnan(sloss): # Handling division with zero
                print("sloss nan")
                loss = closs 
            else:
                loss = alpha*sloss + (1-alpha)*closs
                
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())
            #all_closs.append(closs)
            all_closs.append(closs.item())
            all_sloss.append(sloss.item())
            all_scatter_between.append(ScatterBetween.item())

            tqdm_train.set_postfix( loss="{:.4f}".format(mean(all_loss)), 
                                    closs="{:.4f}".format(mean(all_closs)), 
                                    sloss="{:.4f}".format(mean(all_sloss)) )

    return mean(all_loss), mean(all_closs), mean(all_sloss), mean(all_scatter_between)


def CosineEmbeddingLoss(scores, labels, margin=0.6):
# https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html#torch.nn.CosineEmbeddingLoss    
  
    # 1 - score if y = 1
    # max(0, score) if y = -1
    
    #print(scores)
    #print(labels)
    predicted_idx = scores.max(1)[1]
    loss_sum = 0
    for idx in range(len(labels)):
        score = scores[idx][labels[idx]] 
        if labels[idx] == predicted_idx[idx]:
            loss_sum += 1 - score # y = 1, correct predicted
        else:
            loss_sum += max([0, score - margin]) # y = -1, wrongly predicted
        
    return loss_sum

def episodicTrain(model, modelName, train_loader, val_loader, few_shot_classifier, 
                  m1=5, m2=8, n_epochs=10, alpha=0.5, slossFunc="Multi", 
                  cosine=False, learnRate=0.001, pretrained=False, 
                  test_classes=50, evaluateCluster=True):  
    if cosine:
        entropyLossFunction = CosineEmbeddingLoss
        #entropyLossFunction = nn.CrossEntropyLoss()
        print("CosineEmbeddingLoss, margin = 0.6")
    else:
        entropyLossFunction = nn.CrossEntropyLoss()
        print("CrossEntropyLoss")
    
    #scheduler_milestones = [10, 30]
    #if n_epochs < 1000:
    #    scheduler_milestones = [60, 120] # From scratch with 200 epochs
    #else:
    scheduler_milestones = [m1, m2] # From scratch with 1500 epochs
        
    scheduler_gamma = 0.1
    learning_rate = learnRate # 1e-2
    
    train_optimizer = SGD(
        few_shot_classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )

    #tb_logs_dir = Path("./logs")   
    #tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))
    log_dir = '-' + modelName.split('/')[2].replace(".pth", "")
    tb_writer = SummaryWriter(comment=log_dir)

    # Train model
    best_state = few_shot_classifier.state_dict()
    best_loss = 1000.0
    best_validation_accuracy = 0.0
    best_scatter_between = 0.0
    best_epoch = 0
    for epoch in range(n_epochs):
        if epoch < 0:
            alphaUsed = 0.0 # Prioritize cross entropy loss in start
        else:
            alphaUsed = alpha
        print(f"Epoch {epoch} Alpha {alphaUsed}")
        average_loss, average_closs, average_sloss, average_scatter_between = train_episodic_epoch(entropyLossFunction, 
                                                                                                   few_shot_classifier, train_loader, 
                                                                                                   train_optimizer, slossFunc, 
                                                                                                   alphaUsed, cosine)
        
        if evaluateCluster:
            RIscore, SCscore = evaluateClustering(
                few_shot_classifier.backbone, val_loader, device=DEVICE, test_classes=test_classes
            ) 
            validation_accuracy = SCscore
        else:
            validation_accuracy = evaluate(
                few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation"
            )
        
        #if best_loss < average_loss: # Lowest training loss
        if validation_accuracy > best_validation_accuracy:
            best_epoch = epoch+1
            best_loss = average_loss
            best_validation_accuracy = validation_accuracy
            best_scatter_between = average_scatter_between
            best_state = few_shot_classifier.state_dict()
            torch.save(few_shot_classifier.backbone, modelName)
            #print(f"Lowest loss model saved with accuracy {(best_validation_accuracy):.4f} and loss {(best_loss):.4f}", modelName)
            print(f"Best model saved with accuracy {(best_validation_accuracy):.4f} and loss {(best_loss):.4f}", modelName)
            
        tb_writer.add_scalar("Train/loss", average_loss, epoch)
        tb_writer.add_scalar("Train/closs", average_closs, epoch)
        tb_writer.add_scalar("Train/sloss", average_sloss, epoch)
        tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)
    
        # Warn the scheduler that we did an epoch
        # so it knows when to decrease the learning rate
        train_scheduler.step()

    print(f"Best validation accuracy {(best_validation_accuracy):.4f} with loss {(best_loss):.4f} after epochs", best_epoch)
       
    return best_state, few_shot_classifier.backbone, best_epoch, best_validation_accuracy, best_scatter_between, best_loss


#%% Few shot testing of model        
def test(model, test_loader, few_shot_classifier, n_workers, DEVICE):
    
    model.eval()
    accuracy = evaluate(few_shot_classifier, test_loader, device=DEVICE, tqdm_prefix="Test")
    return accuracy

#%% Create models for training
def createModel(argsModel, argsPretrained):
    
    if argsModel == 'ViTB16':
        print('ViT-B-16')
        NetModel = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        modelName = "./modelsAdv/ViTB16_"
        model = EmbeddingsModel(NetModel, num_classes, use_softmax=False, use_fc=n_use_fc, modelName=argsModel)

    if argsModel == 'ConvNeXt':
        print('ConvNeXt Base')
        NetModel = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        modelName = "./modelsAdv/ConvNeXt_"
        model = EmbeddingsModel(NetModel, num_classes, use_softmax=False, use_fc=n_use_fc, modelName=argsModel)

    if argsModel == 'effB3':
        print('EfficientNetB3')
        NetModel = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1) # 82.00, 12.2M
        modelName = "./modelsAdv/EfficientNetB3_"
        model = EmbeddingsModel(NetModel, num_classes, use_softmax=False, use_fc=n_use_fc, modelName=argsModel)

    if argsModel == 'effB4':
        print('EfficientNetB4')
        NetModel = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1) # 83.38, 19.3M
        modelName = "./modelsAdv/EfficientNetB4_"
        model = EmbeddingsModel(NetModel, num_classes, use_softmax=False, use_fc=n_use_fc, modelName=argsModel)
        
    if argsModel == 'resnet50':
        print('resnet50')
        if argsPretrained:
            NetModel = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE) # 80.858, 25.6M
        else:
            NetModel = resnet50(weights=None).to(DEVICE) 
        modelName = "./modelsAdv/Resnet50_"
        model = EmbeddingsModel(NetModel, num_classes, use_softmax=False, use_fc=n_use_fc)
        
    if argsModel == 'resnet34':
        print('resnet34')
        if argsPretrained:
            NetModel = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(DEVICE) # 73.314, 21.8M
        else:
            NetModel = resnet34(weights=None).to(DEVICE)          
        modelName = "./modelsAdv/Resnet34_"  
        model = EmbeddingsModel(NetModel, num_classes, use_softmax=False, use_fc=n_use_fc)
        
    if argsModel == 'resnet18':
        print('resnet18')
        if argsPretrained:
            NetModel = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(DEVICE) # 69.758, 11.7M
        else:
            NetModel = resnet18(weights=None).to(DEVICE) 
        modelName = "./modelsAdv/Resnet18_"
        model = EmbeddingsModel(NetModel, num_classes, use_softmax=False, use_fc=n_use_fc)
        
    if argsModel == 'resnet12':
        print('resnet12')
        modelName = "./modelsAdv/Resnet12_"
        model = resnet12(use_fc=n_use_fc, num_classes=num_classes).to(DEVICE)
        
    return model, modelName

#%% Saving result to file  
def saveFSLArgs(modelName, args, best_epoch, valAccuracy, testAccuracy, scatterBetween, bestLoss):
    
    with open(modelName.replace('.pth', '.txt'), 'w') as f:
        line = "model,dataset,mode,cosine,epochs,m1,m2,slossFunc,alpha,cluster,pretrained,learnRate,device,trainTasks,"
        line += "valTasks,batch,way,shot,query,bestEpoch,valAccuracy,testAccuracy,meanBetween,trainLoss,modelName\n"
        print(line)
        f.write(line)
        line = args.model + ','
        line += args.dataset + ','
        line += args.mode + ',' 
        line += str(args.cosine) + ',' 
        line += str(args.epochs) + ',' 
        line += str(args.m1) + ','
        line += str(args.m2)  + ','
        line += args.slossFunc + ',' 
        line += str(args.alpha) + ','
        line += str(args.cluster) + ','
        line += str(args.pretrained) + ','
        line += str(args.learnRate) + ','
        line += args.device + ','
        line += str(args.tasks) + ',' 
        line += str(args.valTasks) + ','
        line += str(args.batch) + ',' 
        line += str(args.way) + ','
        line += str(args.shot) + ','
        line += str(args.query) + ','
        line += str(best_epoch) + ','
        line += str(valAccuracy) + ','
        line += str(testAccuracy) + ','
        line += str(scatterBetween) + ','
        line += str(bestLoss) + ','
        line += modelName + '\n'
        print(line)
        f.write(line)
        
# Saving result to file with cluster RI and SC scores
def saveClusterArgs(modelName, args, best_epoch, valRIscore, testRIscore, testSCscore, scatterBetween, bestLoss):
    
    with open(modelName.replace('.pth', '.txt'), 'w') as f:
        line = "model,dataset,mode,cosine,epochs,m1,m2,slossFunc,alpha,cluster,pretrained,learnRate,device,trainTasks,"
        line += "valTasks,batch,way,shot,query,bestEpoch,valRIscore,testRIscore,testSCscore,meanBetween,trainLoss,modelName\n"
        print(line)
        f.write(line)
        line = args.model + ','
        line += args.dataset + ','
        line += args.mode + ',' 
        line += str(args.cosine) + ',' 
        line += str(args.epochs) + ',' 
        line += str(args.m1) + ','
        line += str(args.m2)  + ','
        line += args.slossFunc + ',' 
        line += str(args.alpha) + ','
        line += str(args.cluster) + ','
        line += str(args.pretrained) + ','
        line += str(args.learnRate) + ','
        line += args.device + ','
        line += str(args.tasks) + ',' 
        line += str(args.valTasks) + ','
        line += str(args.batch) + ',' 
        line += str(args.way) + ','
        line += str(args.shot) + ','
        line += str(args.query) + ','
        line += str(best_epoch) + ','
        line += str(valRIscore) + ','
        line += str(testRIscore) + ','
        line += str(testSCscore) + ','
        line += str(scatterBetween) + ','
        line += str(bestLoss) + ','
        line += modelName + '\n'
        print(line)
        f.write(line)  
        
#%% MAIN
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='effB3') # resnet12, resnet18, resnet34, resnet50, effB3, effB4 (EfficientNet), ConvNeXt, ViTB16
    parser.add_argument('--dataset', default='euMoths') # euMoths, CUB, Omniglot (resnet12), mini_imagenet, tiered_imagenet
    parser.add_argument('--mode', default='classic') # classic, episodic
    parser.add_argument('--cosine', default='', type=bool) # default use Euclidian distance when no parameter ''
    parser.add_argument('--epochs', default=10, type=int) # epochs
    parser.add_argument('--m1', default=3, type=int) # learning rate scheduler for milstone 1 (epochs)
    parser.add_argument('--m2', default=6, type=int) # learning rate scheduler for rate milstone 2 (epochs)
    parser.add_argument('--slossFunc', default='Multi') # scatter loss function with variance (Var), standard deviation (Std) or only mean (Mean), multivariate (Multi), triple + multivariant (Triple)
    parser.add_argument('--alpha', default=0.0, type=float) # alpha parameter for sloss function (0-1)
    parser.add_argument('--pretrained', default='', type=bool) # default pretrained weigts is false ''
    parser.add_argument('--device', default='cpu') # training on cpu or cuda:0-3
    parser.add_argument('--tasks', default='250', type=int) # training tasks per epoch (*6 queries)
    parser.add_argument('--valTasks', default='100', type=int) # tasks used for validation
    parser.add_argument('--batch', default='250', type=int) # training batch size
    parser.add_argument('--way', default='5', type=int) # k-Ways for episodic training and few-shot validation
    parser.add_argument('--query', default='6', type=int) # n-Query for episodic training and few-shot validation
    parser.add_argument('--learnRate', default='0.001', type=float) # learn rate for episodic and classic training
    parser.add_argument('--shot', default='5', type=int) # n-shot for episodic training and few-shot validation
    parser.add_argument('--cluster', default='', type=bool) # default use FSL evaluation during training, 'True' use K-means clustering of embeddings
    args = parser.parse_args()
 
    dataDir = './data/' + args.dataset
    image_size = 224 # ResNet with euMoths, CUB and imagenet
    n_epochs = args.epochs # ImageNet pretrained weights - finetuning
    eval_cluster = args.cluster # Evaluate using K-means clustering on embeddings or FSL evaluation
    
    if  args.model == 'resnet12':
        
        if args.model == 'CUB':
            image_size = 84 # CUB dataset
        
        if args.dataset == 'Omniglot':
            image_size = 28 # Omniglot dataset
            
    if args.model == "effB3":
        image_size = 224 # 300
    if args.model == "effB4":
        image_size = 380 
            
    #image_size = 600 # EfficientNet B7
        
    random_seed = 0
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    batch_size = args.batch
    n_workers = 12
 
    n_way = args.way # 5 or 20 paper did
    n_shot = args.shot # For episodic training use 5 shot
    n_query = args.query
    n_tasks_per_epoch = args.tasks
    n_validation_tasks = args.valTasks
    n_test_tasks = 200
    
    #%% Create training and validation dataset loaders

    # Default for FSL transform
    transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**IMAGENET_NORMALIZATION),
            ]
        )
    
    # Settings for training moths order classifier
    # transform = transforms.Compose(
    #         [
    #             transforms.RandomResizedCrop(image_size),
    #             transforms.RandomAffine(40, scale=(.85, 1.15), shear=0),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.RandomVerticalFlip(),
    #             transforms.RandomPerspective(distortion_scale=0.2),
    #             transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    #             transforms.ToTensor(),
    #             transforms.Normalize(**IMAGENET_NORMALIZATION),
    #         ]
    #     )
   
    # Training dataset
    train_set = FewShotDataset(split="train",  image_size=image_size, root=dataDir, training=True, transform=transform)    
    if args.mode == 'classic':
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=n_workers,
            pin_memory=True,
            shuffle=True,
        )
        n_use_fc = True
        
    if args.mode == 'episodic': # Use task sample for episodic training
        train_sampler = TaskSampler(
            train_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks_per_epoch
        )
        train_loader = DataLoader(
            train_set,
            batch_sampler=train_sampler,
            num_workers=n_workers,
            pin_memory=True,
            collate_fn=train_sampler.episodic_collate_fn,
        )    
        n_use_fc = False
   
    # Validation dataset
    val_set = FewShotDataset(split="val",  image_size=image_size, root=dataDir, training=False)
 
    if eval_cluster:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            num_workers=n_workers,
            pin_memory=True,
            shuffle=True,
        )
    else:
        val_sampler = TaskSampler(
            val_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks
        )
        val_loader = DataLoader(
            val_set,
            batch_sampler=val_sampler,
            num_workers=n_workers,
            pin_memory=True,
            collate_fn=val_sampler.episodic_collate_fn,
        )
    
    
    #%% Create model and prepare for training
    DEVICE = args.device
    
    num_classes = len(set(train_set.get_labels()))
    print("Training classes", num_classes)
    test_classes = len(set(val_set.get_labels()))
    print("Validation classes", test_classes)
          
    now = datetime.now()
    dateTime = now.strftime("%m%d_%H%M%S")
        
    # Stable models https://pytorch.org/vision/stable/models.html   # Top 1, Accuracy
    #NetModel = efficientnet_b7(pretrained=True)                 # 84.122, 66.3M   
    
    model, modelName = createModel(args.model, args.pretrained)        
    modelName += args.dataset + '_' + args.mode + '_' + str(int(args.alpha*10)) + '_' + dateTime +"_AdvLoss.pth" 
        
    model = model.to(DEVICE)
    print("Saving model as", modelName)
    if eval_cluster:
        saveClusterArgs(modelName, args, 0, 0, 0, 0, 0, 0)
    else:
        saveFSLArgs(modelName, args, 0, 0, 0, 0, 0)

    if args.cosine:
        few_shot_classifier = PrototypicalNetworksNovelty(model).to(DEVICE)
        print("Use prototypical network with cosine distance to train and validate")
    else:
        #few_shot_classifier = PrototypicalNetworks(model).to(DEVICE)
        few_shot_classifier = PrototypicalNetworksNovelty(model, use_normcorr=3).to(DEVICE)
        print("Use prototypical network with euclidian distance to train and validate")
    
    #%% Classic or episodic training of model
    best_scatter_between = 0
    best_loss = 0
    if args.mode == 'classic':
        print("Classic training epochs", n_epochs)
        best_state, model, best_epoch, best_accuracy, best_loss = classicTrain(model, modelName, train_loader, val_loader, 
                                                                               few_shot_classifier, pretrained=args.pretrained,  
                                                                               m1=args.m1, m2=args.m2, n_epochs=n_epochs, 
                                                                               learnRate=args.learnRate,
                                                                               test_classes=test_classes,
                                                                               evaluateCluster=eval_cluster)
        model.set_use_fc(False)       
        #model.load_state_dict(best_state)

    if args.mode == 'episodic':
        print("Episodic training epochs", n_epochs)
        best_state, model, best_epoch, best_accuracy, best_scatter_between, best_loss = episodicTrain(model, modelName, 
                                                                                                      train_loader, val_loader, 
                                                                                                      few_shot_classifier, 
                                                                                                      m1=args.m1, m2=args.m2, 
                                                                                                      n_epochs=n_epochs, alpha=args.alpha, 
                                                                                                      slossFunc=args.slossFunc,
                                                                                                      cosine=args.cosine,
                                                                                                      learnRate=args.learnRate,
                                                                                                      pretrained=args.pretrained,
                                                                                                      test_classes=test_classes,
                                                                                                      evaluateCluster=eval_cluster)
        #few_shot_classifier.load_state_dict(best_state)
    

    #%% Evaluation on test dataset
    print('Using best saved model weights', modelName)
    modelSaved = torch.load(modelName, map_location=torch.device(DEVICE))
    model.load_state_dict(modelSaved.state_dict())
    few_shot_classifier.backbone.load_state_dict(modelSaved.state_dict())
        
    test_set = FewShotDataset(split="test", image_size=image_size, root=dataDir, training=False)
    if eval_cluster:
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            num_workers=n_workers,
            pin_memory=True,
            shuffle=True,
        )
        test_classes = len(set(test_set.get_labels()))
        model.set_use_fc(False)
        model.eval()
        testRIscore, testSCscore = evaluateClustering(model, test_loader, DEVICE, test_classes) # RI score
        saveClusterArgs(modelName, args, best_epoch, best_accuracy, testRIscore, testSCscore, best_scatter_between, best_loss)
        textLine = f"Cluster score valRI/testRI,testSC : {(100 * best_accuracy):.2f}%/{(100 * testRIscore):.2f}%,{(100 * testSCscore):.2f}%," + args.model + "," + args.dataset 
    else:
        test_sampler = TaskSampler(
            test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
        )
        test_loader = DataLoader(
            test_set,
            batch_sampler=test_sampler,
            num_workers=n_workers,
            pin_memory=True,
            collate_fn=test_sampler.episodic_collate_fn,
        )
        accuracy = test(model, test_loader, few_shot_classifier, n_workers, DEVICE)
        textLine = f"Accuracy val/test : {(100 * best_accuracy):.2f}%/{(100 * accuracy):.2f}%," + args.model + "," + args.dataset      
        saveFSLArgs(modelName, args, best_epoch, best_accuracy, accuracy, best_scatter_between, best_loss)

    textLine += "," + args.slossFunc + ',' + str(args.alpha) + "," + str(best_epoch) + "," + f"{(best_loss):.4f}," +  modelName + '\n'
    print(textLine)
    with open('ResultTrainAdvLoss.txt', 'a') as f:
        f.write(textLine)
