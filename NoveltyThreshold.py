# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:52:46 2023

@author: Kim Bjerge
"""
import math 
import numpy as np

"""
ImageNetNoveltyTh = {# CPU
        'resnet18': 0.6972,
        'resnet34': 0.6974,
        'resnet50': 0.6196  # 0.7287 5 shot
        }
ImageNetNoveltyThGPU = {# GPU other default weights than CPU???
        'resnet18': 0.8088,
        'resnet34': 0.8115,
        'resnet50': 0.8468
        }

euMothsNoveltyTh = {# GPU     CPU
        'resnet18': 0.7580,   #0.7582
        'resnet34': 0.7537,   #0.7593
        'resnet50': 0.7948,   #0.7932
        } 
CUBNoveltyTh = {    # CPU     GPU ????
        'resnet18': 0.7189,   #0.8128
        'resnet34': 0.7146,   #0.8194
        'resnet50': 0.7581    #0.8471
        } 
"""

# Learned avg and std on validation dataset with 1-shot and 6-way
ImageNetNoveltyAvgStd = {# avg, std
         'resnet18': [0.72226, 0.07111], 
         'resnet34': [0.73021, 0.07541], 
         'resnet50': [0.76526, 0.07281]  # Th 0.6196 1 shot, TH 0.7287 5 shot
         } 
euMothsNoveltyAvgStd = {# avg, std
         'resnet18': [0.77879, 0.05930], 
         'resnet34': [0.77685, 0.06190], 
         'resnet50': [0.80555, 0.05177]  
         } 
CUBNoveltyAvgStd = {# avg, std
         'resnet18': [0.71100, 0.05190], 
         'resnet34': [0.71902, 0.05467], 
         'resnet50': [0.76377, 0.04995]  
         }
OmniglotNoveltyAvgStd = {# avg, std
         'resnet12': [0.87853, 0.09809]
         #'resnet18': [0.71100, 0.05190], # NA
         #'resnet34': [0.71902, 0.05467], # NA 
         #'resnet50': [0.76377, 0.04995]  # NA 
         }  

# function for finding roots in second order equation
def equationroots(a, b, c): 
 
    # calculating discriminant using formula
    dis = b * b - 4 * a * c 
    sqrt_val = math.sqrt(abs(dis)) 
     
    # checking condition for discriminant
    if dis > 0: 
        print("real and different roots") 
        x1 = (-b + sqrt_val)/(2 * a)
        x2 = (-b - sqrt_val)/(2 * a)
        print(x1, x2) 
     
    elif dis == 0: 
        print("real and same roots")
        x1 = -b / (2 * a)
        x2 = 0
        print(x1) 
     
    # when discriminant is less than 0
    else:
        print("Complex Roots") 
        #print(- b / (2 * a), + i, sqrt_val) 
        #print(- b / (2 * a), - i, sqrt_val) 

    x = x1
    if x2 > 0 and x2 < 1: # Use the solution between 0-1
        x = x2
    return x, x2    

# Bayes classification for two classes 
# using the decision function d(x) = p(x|w)*P(w)
# k is the distribution of the known classes
# o is the distribution of the outliers 
def BayesTwoClassThreshold(var_k, mean_k, var_o, mean_o, k_way, m_outlier):
    
    std_k = math.sqrt(var_k)
    std_o = math.sqrt(var_o)
    a = (var_k - var_o)
    b = -2*(mean_o*var_k - mean_k*var_o)
    P = (math.sqrt(std_k)/math.sqrt(std_o))*(m_outlier/k_way)
    c = var_k*math.pow(mean_o,2) - var_o*math.pow(mean_k,2) - 2*var_o*var_k*math.log(P)  
    
    return equationroots(a, b, c)

def StdTimesTwoThredshold(mu_k, sigma_k):
    
    return mu_k - 2*sigma_k

def getLearnedThreshold(weightsName, modelName, n_shot):
    
    novelty_th = 0
    if weightsName == 'ImageNet':
        avg = ImageNetNoveltyAvgStd[modelName][0]
        std = ImageNetNoveltyAvgStd[modelName][1]
        #novelty_th = ImageNetNoveltyTh[modelName]
    if weightsName == 'Omniglot':
        avg = OmniglotNoveltyAvgStd[modelName][0]
        std = OmniglotNoveltyAvgStd[modelName][1]
    if weightsName == 'euMoths':
        avg = euMothsNoveltyAvgStd[modelName][0]
        std = euMothsNoveltyAvgStd[modelName][1]
        #novelty_th = euMothsNoveltyTh[modelName]
    if weightsName == 'CUB':
        avg = CUBNoveltyAvgStd[modelName][0]
        std = CUBNoveltyAvgStd[modelName][1]
        #novelty_th = CUBNoveltyTh[modelName]
    if weightsName == 'mini_imagenet':
        avg = 0.72226
        std = 0.07111
        
    novelty_th = avg - 2*(std/np.sqrt(n_shot)) # Mean filter sigma/sqrt(M)
    print("Novelty threshold", weightsName, modelName, avg, std, novelty_th)
    return novelty_th

