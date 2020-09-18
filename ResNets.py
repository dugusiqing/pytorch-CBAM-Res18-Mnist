import torch
import torchvision
from tensorboardX import SummaryWriter

model = torchvision.models.resnet18(pretrained=False)
dummy_input = torch.rand(1, 3, 224, 224)
with SummaryWriter(comment='ResNet18') as w:
    w.add_graph(model, (dummy_input, ))
