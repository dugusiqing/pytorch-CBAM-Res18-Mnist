import math
from torch.hub import load_state_dict_from_url
import gc
from mixPooling import *
gc.collect()

__all__ = ['ResNet', 'resnet18']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}


########################

####通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

#####空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
########################

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


######Res18和Res34的基础模块
class BasicBlock(nn.Module):
    expansion = 1
    # inplanes其实就是channel,叫法不同
    def __init__(self, inplanes, planes, stride=1, downsample=None,use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        #######通道注意力
        self.ca = ChannelAttention(planes)
        #######空间注意力
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.use_cbam = use_cbam
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #########加入通道注意力和空间注意力机制
        if self.use_cbam:
            out = self.ca(out) * out
            out = self.sa(out) * out

        # 把shortcut那的channel的维度统一
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10,use_cbam=False,use_mixpool=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,  # 因为mnist为（1，28，28）灰度图，因此输入通道数为1
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.use_mixpool = use_mixpool
        self.mixpool1 = MixPooling2D(kernel_size=3, stride=2,padding=1)

        # 网络的第一层加入注意力机制
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()
        self.layer1 = self._make_layer(block, 64, layers[0],use_cbam=use_cbam)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,use_cbam=use_cbam)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,use_cbam=use_cbam)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,use_cbam=use_cbam)
        # 网络的卷积层的最后一层加入注意力机制
        self.ca1 = ChannelAttention(self.inplanes)
        self.sa1 = SpatialAttention()

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.mixpool2 = MixPooling2D(kernel_size=7, stride=1, padding=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.use_cbam = use_cbam
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,use_cbam=False):
        # downsample 主要用来处理H(x)=F(x)+x中F(x)和xchannel维度不匹配问题
        downsample = None
        # self.inplanes为上个box_block的输出channel,planes为当前box_block块的输入channel
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,use_cbam))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            #layers.append(block(self.inplanes, planes, use_cbam))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.use_mixpool:
            x=self.mixpool1(x)
        else:
            x = self.maxpool(x)
        #######头部卷积引入CBAM
        if self.use_cbam:
            x = self.ca(x) * x
            x = self.sa(x) * x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #######尾部卷积引入CBAM
        if self.use_cbam:
            x = self.ca1(x) * x
            x = self.sa1(x) * x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def _resnet(arch, block, layers, pretrained, progress, use_cbam=False, use_mixpool=False,**kwargs):
    model = ResNet(block, layers, use_cbam=use_cbam,use_mixpool=use_mixpool, **kwargs)
    if pretrained and use_cbam :
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        new_state_dict = model.state_dict()
        new_state_dict.update(state_dict)
        model.load_state_dict(new_state_dict)
    return model

def resnet18(pretrained=False, progress=True, use_cbam=False,use_mixpool=False,**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,use_cbam=use_cbam,use_mixpool=use_mixpool,
                   **kwargs)


