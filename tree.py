import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Tree(nn.Module):

    def __init__(self, depth, num_classes=10):
        super(Tree, self).__init__()
        
        self.depth = depth
        self.num_classes = num_classes

        self.num_leaf = 2 ** (depth - 1)
        self.num_split = self.num_leaf - 1

        # leaf node distributions
        self.pi = Parameter(torch.rand((self.num_leaf, self.num_classes)))

    def forward(self, x):

        assert x.ndimension() == 2
        assert x.min() >= 0 and x.max() <= 1
        bs = x.size(0)
        assert x.size(1) == self.num_split, "%d vs %d" % (x.size(1), self.num_split)

        x = x.unsqueeze(dim=2)
        probabilities = torch.cat((x, 1-x), dim=2) # [bs, num_split, 2]

        start, end = 0, 1
        if x.is_cuda:
            tmp = torch.ones((bs, 1, 1)).cuda()
        else:
            tmp = torch.ones((bs, 1, 1))

        for d in range(self.depth-1):
            tmp = tmp.view(bs, -1, 1).repeat(1, 1, 2)
            prob = probabilities[:, start:end, :]
            tmp = tmp * prob
            start = end
            end = start + 2 ** (d + 1)

        tmp = tmp.view(bs, self.num_leaf)

        return torch.mm(tmp, F.softmax(self.pi, dim=1))

