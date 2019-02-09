import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from tree import Tree
import numpy as np

class Forest(nn.Module):
    def __init__(self, in_features, num_trees, tree_depth, num_classes):
        super(Forest, self).__init__()

        self.in_features = in_features
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.num_classes = num_classes

        self.num_split_per_tree = 2 ** (self.tree_depth - 1) - 1
        assert self.num_split_per_tree <= self.in_features

        self.register_buffer('feature_mask', torch.zeros(self.num_split_per_tree, self.num_trees).long())

        self.linear = nn.Linear(in_features, in_features, bias=False)

        self.trees = nn.ModuleList()

        for i in range(self.num_trees):

            self.feature_mask[:, i] = torch.from_numpy(np.random.choice(self.in_features, self.num_split_per_tree))
            self.trees.append(Tree(tree_depth, num_classes))
    
    def forward(self, x):

        x = torch.sigmoid(self.linear(x))

        prob_trees = []

        for i, tree in enumerate(self.trees):
            
            feature_mask1 = self.feature_mask[:, i].squeeze()
            x1 = x[:, feature_mask1]
            
            prob_tree = tree(x1).unsqueeze(2)
            prob_trees.append(prob_tree)

        prob_forest = torch.cat(prob_trees, dim=2)
        prob_forest = torch.sum(prob_forest, dim=2) / self.num_trees

        return prob_forest
