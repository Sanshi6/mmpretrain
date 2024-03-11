from mmengine.runner import load_checkpoint
import torch.nn as nn
from mmpretrain.models.backbones.ModifyRepVGG import create_RepVGG_B2, create_RepVGG_B1
import torch


class temp(nn.Module):
    def __init__(self):
        super(temp, self).__init__()
        self.conv1 = nn.Identity()
        self.backbone = None

    def forward(self, x):
        x = self.conv1(x)
        return x


# model = create_RepVGG_B1()
model = temp()
model.backbone = create_RepVGG_B1()
load_checkpoint(model, 'E:\CapstoneProject\mmclassification\pth\epoch_b1_13.pth')

model.features = model.backbone
torch.save(model.features.state_dict(), 'E:\CapstoneProject\mmclassification\pth\epoch_b1_13_backbone.pth')
