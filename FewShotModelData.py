# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 18:57:52 2023

@author: Kim Bjerge

Defines CNN model pretrained and skipping fully connected layers
Embeddings is output from conv layers after flatten

Defines dataset used for few-shot testing

"""

from pathlib import Path
from torch import nn, Tensor
from easyfsl.datasets import EasySet

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
# Define EmbeddingsModel by parameter model and skip fc layers
class EmbeddingsModel(nn.Module):
    def __init__(
        self,
        model,
        num_classes: int=1000,
        use_softmax: bool=False,
        use_fc: bool=False,
        modelName: str="resnet"
        ):
        super().__init__()
        self.use_fc = use_fc
        self.use_sm = use_softmax
        self.model_ft = model 
        self.num_classes = num_classes
        if "resnet" in modelName:
            self.in_features = self.model_ft.fc.in_features # ResNet
        if "B3" in modelName: 
            self.in_features = self.model_ft.classifier[1].in_features #1536 EfficientNetB3
        if "B4" in modelName: 
            self.in_features = self.model_ft.classifier[1].in_features #1792 EfficientNetB4
        if "NeXt" in modelName:
            self.in_features = self.model_ft.classifier[2].in_features
        if "ViT" in modelName:
            self.in_features = self.model_ft.heads[0].in_features
                     
        # Only used when self.use_fc is True
        self.fc = nn.Linear(self.in_features, self.num_classes) # ResNet
        #self.classifier = nn.Linear(self.in_features, self.num_classes) # EfficientNet
        self.drop = nn.Dropout(p=0.25)
        self.softmax = nn.Softmax(dim=1)
        if "resnet" in modelName:
            self.model_ft.fc = nn.Identity() # Do nothing just pass input to output 
        if "eff" in modelName:
            self.model_ft.classifier = Identity()            
        if "NeXt" in modelName:
            self.model_ft.classifier[2] = Identity()
        if "ViT" in modelName:
            self.model_ft.heads[0] = Identity()
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.model_ft(x)   
        if self.use_fc:
            x = self.drop(x) # Dropout to add regularization
            x = self.fc(x)
            if self.use_sm:
                x = self.softmax(x)      
        return x # Ouput embeddings or softmax

    def set_use_fc(self, use_fc: bool):
        """
        Change the use_fc property. Allow to decide when and where the model should use its last
        fully connected layer.
        Args:
            use_fc: whether to set self.use_fc to True or False
        """
        self.use_fc = use_fc

# Define EmbeddingsModel by parameter model and skip fc layers
class EmbeddingsModelFC2(nn.Module):
    def __init__(
        self,
        model,
        num_classes: int=1000,
        use_softmax: bool=False,
        use_fc: bool=False
        ):
        super().__init__()
        self.use_fc = use_fc
        self.use_sm = use_softmax
        self.model_ft = model 
        self.num_classes = num_classes
        self.in_features = self.model_ft.fc.in_features # ResNet
        self.out_channels = int(self.in_features/4)
        print("EmbeddingsModelFC2 out channels", self.out_channels, use_softmax)
        #self.in_features = 1792 # EfficientNetB4
        # Only used when self.use_fc is True
        self.fc1 = nn.Linear(self.in_features, self.out_channels) # ResNet
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(self.out_channels, self.num_classes)
        self.drop = nn.Dropout(p=0.25)
        self.softmax = nn.Softmax(dim=1)
        self.model_ft.fc = nn.Identity() # Do nothing just pass input to output 
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.model_ft(x)   
        if self.use_fc:
            x = self.drop(x) # Dropout to add regularization
            x = self.fc2(self.relu(self.fc1(x)))
            if self.use_sm:
                x = self.softmax(x)      
        return x # Ouput embeddings or softmax

    def set_use_fc(self, use_fc: bool):
        """
        Change the use_fc property. Allow to decide when and where the model should use its last
        fully connected layer.
        Args:
            use_fc: whether to set self.use_fc to True or False
        """
        self.use_fc = use_fc        
        
# Define dataset by setting root path      
class FewShotDataset(EasySet):
    def __init__(self, split: str, image_size=84, root="./data/CUB", **kwargs):
        specs_file = Path(root) / f"{split}.json"
        if not specs_file.is_file():
            raise ValueError(
                f"Could not find specs file {specs_file.name} in {root}"
            )
        super().__init__(specs_file=specs_file, image_size=image_size, **kwargs)
     
