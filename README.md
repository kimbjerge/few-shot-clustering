# Few-shot-clustering

This code base contains the python code for clustering of images using few-shot learning and episodic training.

The method is described in the paper:

K. Bjerge, P. Bodesheim, H. Karstoft, “Deep Image Clustering with Model-Agnostic Meta-Learning,” Accepted by the International Conference on Computer Vision and Theory and Applications (VISAPP 2025).

The paper with link will be public in spring 2025.

Supplementary material with detailed results are found in:  ![Supplementary Material](https://github.com/kimbjerge/few-shot-clustering/blob/main/DeepImageClusteringMAML_SupplemtaryMaterial.pdf)
This document presents tables detailing the results of training deep learning models, including ResNet50v2, EfficientNetB3, ConvNeXt-B, and ViT-B/16, on various datasets such as EU Moths, Caltech Birds (CUB), tiered-ImageNet, and mini-ImageNet, as outlined in the accompanying paper.

# Installation conda environment on Windows

- conda create --name fslclust --file req_env_win.txt
- conda activate fslclust

The easy-few-shot-learning (easyfsl) framework has been used to boost our experiments with few-shot image classification. 
The framework contains libraries for 11 few-shot learning methods, handling of support and query data and Python code for resnet12 backend with episodic training.

Install the Python library easyfsl "pip install easyfsl" or use the GitHub:

https://github.com/sicara/easy-few-shot-learning

# Clustering of feature embeddings

python UnsupervisedGMM.py 

The code selectes one of the below datasets and computes the embedding features using a few-shot trained model with episodic training.
The resulting embeddings are clustered by selecting different clustering models such as GMM and K-means.

# Datasets used for training, validation, and testing
A copy of the prepared CU-Birds and EU-moths datasets can be downloaded from here:

https://drive.google.com/drive/folders/1xaAJG_-wGpqR0JRUAEjzbcZyS5GxrhNk

The miniImagenet and tieredImagenet dataset can be found here:

https://lyy.mpi-inf.mpg.de/mtl/download/Lmzjm9tX.html

The zipped files must be copied and unzipped to the folders:

 - data/CUB
 - data/euMoths
 - data/mini_imagenet
 - data/tiered_imagenet

## miniImageNet
This dataset presents a preprocessed version of the miniImageNet benchmark dataset used in few-shot learning. This version of miniImageNet is not resized to any particular size and is left to be the same size as they are in the ImageNet dataset.

Download and unzip the preprocessed version of the miniImageNet benchmark dataset from: https://www.kaggle.com/datasets/arjunashok33/miniimagenet

Copy the image files to data/mini_imagenet

The train.json, val.json, and test.json split the dataset into 60 train images, 20 validation images, and 20 test images.

With prepare/prepare_mini_imagenet.py it is possible to create another split of the miniImageNet dataset.

## Omniglot
The Omniglot data set is designed for developing more human-like learning algorithms. It contains 1623 different handwritten characters from 50 different alphabets and is used for few-shot learning research.

Alternatively to use the files on drive.google download and unzip the images_background.zip and images_evaluation.zip from the below GitHub. https://github.com/brendenlake/omniglot

Copy the images files to data/Omniglot

The train.json, val.json, and test.json split the dataset into 3856 train images, 40 validation images, and 40 test images.

With the Python script: prepare/prepare_Omniglot.py it is possible to create a customized split of the Omniglot dataset.

## CU-Birds (CUB)
The Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset is a widely-used dataset for fine-grained visual categorization task. It contains 11,788 images of 200 subcategories belonging to birds.

Alternatively to use the files on drive.google download and extract the dataset from a Github which provides a make download-cub recipe to download and extract the dataset. See https://github.com/sicara/easy-few-shot-learning

The train.json, val.json, and test.json split the dataset into 140 train images, 30 validation images, and 30 test images.

## EU-moths
This dataset presents a dataset of only 11 samples for each class of 200 classes of moth species.

Alternatively to use the files on drive.google download and unzip the Cropped images of the EU Moths dataset from: https://inf-cv.uni-jena.de/home/research/datasets/eu_moths_dataset/

Copy the image files to data/euMoths/images

The train.json, val.json, and test.json split the dataset into 100 train images, 50 validation images, and 50 test images.

With prepare/prepare_eu_moths.py it is possible to create another split of the EU moths dataset.

# Episodic training
Episodic training for domain generalization is the problem of learning models that generalize to novel testing domains with different statistics and classes than in the set of the known training domain. The method learns a model that generalizes well to a novel domain without any knowledge of the novel validation domain with new classes during episodic model training. It is also called the meta-learning paradigm, here we have a set of tasks to learn in each epoch. Each task also called an episode contains a support set of K-classes (K-way) with a N-shot of images for each class. A query set of images is matched with the support set using a few-shot Protypical network that compares the embeddings from the backbone of the convolutional neural network. The Prototypical network uses the Euclidian distance as a similarity function during training to find the best matching class in the support set. Episodic training can be performed with and without pre-trained weights where the backbone in our experiments is ResNet18, ResNet34, or ResNet50.

When training without pre-trained weights the model with the best accuracy is selected and stored.
When training with pre-trained weights the model with the lowest loss is selected and stored.
The models and results will be stored in the folder modelsAdv and tensorboard log files are stored in the folder runs.

To view the tensorboard log files write: tensorflow --logdir runs/

# CU-Birds and EU moths training with transfer learning
To train models on the CUB and EU-Moths dataset with pretrained weights from ImageNet the backbones ResNet18, ResNet34 and ResNet50 is provided. It is also possible to train miniImageNet with pre-trained weights, however, since miniImageNet is a subset of ImagneNet it would give unrealistic good results for domain adaptation since the same classes are included during pre-training and validation.

The Linux bash script/trainCUBPreAdv.sh contains command arguments to train with the CU-Birds dataset:

    python FewShotTrainingAdvLoss.py --model resnet18 --dataset CUB --mode episodic --slossFunc Std --alpha 0.5 --m1 120 --m2 190 --epochs 250 --learnRate 0.001 --pretrained True --tasks 500 --valTasks 100 --query 10 --device cuda:0

The linux bash script/traineuMothsPreAdv.sh contains command arguments to train with the EU-Moths dataset:

    python FewShotTrainingAdvLoss.py --model resnet18 --dataset euMoths --mode episodic --alpha 0.5 --m1 120 --m2 190 --epochs 250 --learnRate 0.001 --pretrained True --slossFunc Std --tasks 500 --valTasks 100 --query 6 --device cuda:0

The folder modelsFinalPreAdv contains the trained models with files that are generated for every model and contains arguments and results for training.
