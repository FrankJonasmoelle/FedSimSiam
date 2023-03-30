import torch.nn as nn

class DownstreamEvaluation(nn.Module):
    def __init__(self, simsiam_model):
        super(DownstreamEvaluation, self).__init__()
        self.simsiam = simsiam_model

        # freeze parameters         
        for param in self.simsiam.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(2048, 10)


    def forward(self, x):
        z, _ = self.simsiam(x)
        x = self.classifier(z)
        return x