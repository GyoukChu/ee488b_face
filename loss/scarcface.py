import torch
import torch.nn as nn
#conda install pytorch-metric-learning -c metric-learning -c pytorch
from pytorch_metric_learning import losses

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize=True
        self.num_classes=nClasses
        self.embedding_size=nOut

        self.criterion=losses.SubCenterArcFaceLoss(
            num_classes=self.num_classes, 
            embedding_size=self.embedding_size,
            **kwargs
        )

        print('Initialised SubCenter ArcFace Loss(K=3)')
        total_params = sum(p.numel() for p in self.criterion.parameters())
        print('Total loss func parameters: {:,}'.format(total_params))

    def forward(self, x, label=None):
        nloss = self.criterion(x, label)

        return nloss

# mat1 : batch_size * 
# mat2 : nOut * 