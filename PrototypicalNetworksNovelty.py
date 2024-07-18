"""
See original implementation (quite far from this one)
at https://github.com/jakesnell/prototypical-networks
"""

import torch

from torch import Tensor
from torch import nn
import matplotlib.pyplot as plt

from easyfsl.methods import FewShotClassifier

class PrototypicalNetworksNovelty(FewShotClassifier):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their euclidean distance to the prototypes.
    """
    def __init__(
        self,
        *args,
        use_normcorr: int = 0, # Default cosine distance, 3 euclidian distance
        **kwargs,
    ):
        """
        Build Prototypical Networks Novelty by calling the constructor of FewShotClassifier.
        Args:
            use_normcorr: use euclidian distance or normalized correlation to compute scores (0, 1, or 2)
        """
        super().__init__(*args, **kwargs)
        
        self.use_normcorr = use_normcorr
        
    def normxcorr_mean(self, proto_features, features):
             
        pf_mean = proto_features.mean(1)
        pf_std = proto_features.std(1)
        lpf = len(pf_mean)

        f_mean = features.mean(1)
        f_std = features.std(1)
        lf = len(f_mean)
        
        nxcorr = torch.zeros([lf, lpf], dtype=torch.float32)
        
        for i in range(lf):     
            features_sub_mean = torch.sub(features[i,:], f_mean[i])
            #features_sub_mean = features[i,:]
            for j in range(lpf):
               proto_features_sub_mean = torch.sub(proto_features[j,:], pf_mean[j])
               #proto_features_sub_mean = proto_features[j,:]
               nominator = torch.dot(features_sub_mean, proto_features_sub_mean)
               denominator = f_std[i]*pf_std[j]
               nxcorr[i,j] = nominator/denominator
        
        # plt.plot(features[0,:].tolist(), '.g') # label #3
        # plt.plot(proto_features[3,:].tolist(), '.r')
        # plt.show()
        # plt.plot(features[7,:].tolist(), '.g') # label #4
        # plt.plot(proto_features[2,:].tolist(), '.r')
        # plt.show()
        return nxcorr

    def normxcorr(self, proto_features, features):
             
        lpf = len(proto_features)
        lf = len(features)        
        nxcorr = torch.zeros([lf, lpf], dtype=torch.float32)     
        for i in range(lf): 
            feature_energy = torch.pow(features[i,:], 2).sum()
            for j in range(lpf):
               nominator = torch.dot(features[i,:], proto_features[j,:])
               denominator = feature_energy*torch.pow(proto_features[j,:], 2).sum()
               nxcorr[i,j] = nominator/torch.sqrt(denominator)
        
        # plt.plot(features[0,:].tolist(), '.g') # label #3
        # plt.plot(proto_features[3,:].tolist(), '.r')
        # plt.show()
        # plt.plot(features[7,:].tolist(), '.g') # label #4
        # plt.plot(proto_features[2,:].tolist(), '.r')
        # plt.show()
        return nxcorr
    
    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Predict query labels based on their distance to class prototypes in the feature space.
        Classification scores are the negative of euclidean distances.
        """
        scores_std = None
        
        # Extract the features of query images
        self.query_features = self.compute_features(query_images)
        self._raise_error_if_features_are_multi_dimensional(self.query_features)

        # Compute the euclidean distance from queries to prototypes
        if self.use_normcorr == 1: # Normalized correlation to mean of prototype features
            scores = self.normxcorr(self.prototypes, self.query_features)
        else:
            if self.use_normcorr == 2: # Mean of normalized correlation to prototype features
                self.k_way = len(torch.unique(self.support_labels))
                scores = torch.zeros([len(self.query_features) , self.k_way], dtype=torch.float32)    
                scores_std = torch.zeros([len(self.query_features) , self.k_way], dtype=torch.float32)    
                # Prototype i is the mean of all instances of features corresponding to labels == i
                for label in range(self.k_way):           
                    support_features = self.support_features[self.support_labels == label]
                    scores_label = self.normxcorr(support_features, self.query_features)
                    scores[:,label] = scores_label.mean(1)   
                    scores_std[:,label] = scores_label.std(1)
            else: # Euclidian of cosine distance to mean of prototype features
                if self.use_normcorr == 3: # Euclidian distance to prototypes
                    scores = self.l2_distance_to_prototypes(self.query_features)
                else: # Default cosine distance to prototypes (0) same as 1
                    scores = self.cosine_distance_to_prototypes(self.query_features)
                
        #return self.softmax_if_specified(scores), scores_std # Std not used
        return self.softmax_if_specified(scores)
    
    
    def multivariantScatterLoss(self):
        
        num_centers = len(self.prototypes)
        center_points = self.prototypes
        
        scatterBetweenSum = 0
        for i in range(num_centers-1):
            for j in range(num_centers - (i+1)):
                scatterDiff = center_points[i] - center_points[i+j+1]
                scatterBetween = scatterDiff @ torch.t(scatterDiff)
                scatterBetweenSum += scatterBetween
        
        support_features = self.support_features
        support_labels = self.support_labels
        
        scatterWithinSum = 0
        for i in range(num_centers):
            support_features_center = support_features[support_labels == i]
            for j in range(len(support_features_center)):
                scatterDiff = support_features_center[j] - center_points[i]
                scatterWithin = scatterDiff @ torch.t(scatterDiff)
                scatterWithinSum += scatterWithin
            
        #scatterWithinLoss = torch.sqrt(scatterWithinSum)
        #scatterBetweenLoss = torch.sqrt(scatterBetweenSum)
        scatterLoss = scatterWithinSum/scatterBetweenSum
        
        return scatterWithinSum, scatterBetweenSum, scatterLoss
    
    
    def tripleMarginDistanceLoss(self, dist_scores, labels, margin=1.0):
        
        k_way = len(self.prototypes)
        #n_shot = int(len(self.support_labels)/k_way)
        #q_query = int(len(self.query_features)/n_shot)
        q_len = len(dist_scores)

        tripleLoss = 0
        for i in range(q_len):
            scores = dist_scores[i]
            positive = scores[labels[i]]
            negative = (sum(scores) - positive)/(k_way-1)
            # tLoss = max(positive - negative + margin, 0) # Euclidian distance
            tLoss = (margin - positive) + negative # Cosine similarity
            tripleLoss += tLoss
                      
        return tripleLoss/q_len
    
    # Not good
    # def tripleMarginDistanceLoss(self, dist_scores, labels, margin=1.0):
        
    #     k_way = len(self.prototypes)
    #     #n_shot = int(len(self.support_labels)/k_way)
    #     #q_query = int(len(self.query_features)/n_shot)
    #     q_len = len(dist_scores)

    #     tripleLoss = 0
    #     for i in range(q_len):
    #         scores = dist_scores[i]
    #         positive = scores[labels[i]]
    #         negative = (sum(scores) - positive)/(k_way-1)
    #         tLoss = max(positive - negative + margin, 0)
    #         tripleLoss += tLoss
                      
    #     return tripleLoss/q_len

    # def contrastiveLoss(self, query, labels):
    
    # def tripleMarginDistanceLoss(self, labels):
        
    #     num_centers = len(self.prototypes)
    #     center_points = self.prototypes
    #     support_labels = self.support_labels
    #     n_shot = int(len(support_labels)/num_centers)
    #     query = self.query_features

    #     tripleLoss = 0
    #     for i in range(num_centers-1):
    #         anchor = center_points[i]
    #         label = support_labels[i*n_shot]
    #         positive = query[labels == label]
    #         negative = query[labels != label]
    #         triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), margin=10)
    #         tloss = triplet_loss(anchor, positive, negative)
    #         tripleLoss += tloss
            
    #     return tripleLoss/num_centers
      
    @staticmethod
    def is_transductive() -> bool:
        return False
