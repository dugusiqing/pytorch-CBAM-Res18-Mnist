import numpy as np
import torch
import torch.nn as nn
import random
random.seed()

class MixPooling2D(nn.Module):
    def __init__(self, kernel_size=3, stride=2,padding=1):
        super(MixPooling2D, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.maxpool=nn.MaxPool2d(self.kernel_size,self.stride,padding=self.padding)
        self.avgpool=nn.AvgPool2d(self.kernel_size,self.stride,padding=self.padding)
        self.alpha_frequencies = np.zeros(2)
        self.alpha = random.uniform(0, 1)
    def forward(self, x):
        #######训练阶段，混合加权池化
        if self.training:
            self.alpha_frequencies[0] += self.alpha
            self.alpha_frequencies[1] += 1 - self.alpha
            out = self.alpha*self.maxpool(x)+(1-self.alpha)*self.avgpool(x)
        else:
        #######测试阶段，根据频次，选择池化方法
            if(self.alpha_frequencies[0] < self.alpha_frequencies[1]):
                out = self.avgpool(x)
            else:
                out = self.maxpool(x)
        return out



if __name__ == "__main__":
    print("="*10 + "MixPool2D" + "="*10)
    avgpool="avgpool"
    maxpool="maxpool"
    a= torch.Tensor([[1],[2]],[[3],[4]])
    b=torch.Tensor([[5],[6]],[[1],[3]])
    mixpool = MixPooling2D()
    print(mixpool.training)
    print(torch.lt(a,b))