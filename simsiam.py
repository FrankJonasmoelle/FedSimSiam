import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math


class ProjectionMLP(nn.Module):
    """Projection MLP f"""
    def __init__(self, in_features, h1_features, h2_features, out_features):
        super(ProjectionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features, h1_features),
            nn.BatchNorm1d(h1_features),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(h1_features, h2_features),
            nn.BatchNorm1d(h2_features),
            nn.ReLU(inplace=True)

        )
        self.l3 = nn.Sequential(
            nn.Linear(h1_features, out_features),
            nn.BatchNorm1d(out_features)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x
    

class PredictionMLP(nn.Module):
    """Prediction MLP h"""
    def __init__(self, in_features, hidden_features, out_features):
        super(PredictionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x
    

class SimSiam(nn.Module):
    def __init__(self):
        super(SimSiam, self).__init__()
        backbone = resnet18(weights=True) # TODO: Should weights be pretrained?
        num_ftrs = backbone.fc.in_features
        
        self.model = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = ProjectionMLP(num_ftrs, 2048, 2048, 2048)
        self.prediction = PredictionMLP(2048, 512, 2048)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1) # TODO
        z = self.projection(x)
        p = self.prediction(z)
        return z, p
    

def D(p, z):
    z = z.detach() # we don't backpropagate here
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p*z).sum(dim=1).mean()