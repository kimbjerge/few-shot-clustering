# few-shot-clustering
Python code for clustering of images using few-shot learning and episodic training

# Datasets used for training, validation, and testing
A copy of the prepared Omniglot, CU-Birds and EU-moths datasets can be downloaded from here:

https://drive.google.com/drive/folders/1xaAJG_-wGpqR0JRUAEjzbcZyS5GxrhNk

The zipped files must be copied and unzipped to the folders:

data/Omniglot
data/CUB
data/euMoths
miniImageNet
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
