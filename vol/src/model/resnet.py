import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1, bn_layer=False, bias=True):
    if bn_layer:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else: 
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True)
        )

def linear(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes), 
        nn.ReLU(inplace=True)
        )

############################
# ResNet PoseNet from TartanVO.
############################
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = conv(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return F.relu(out, inplace=True)


class ResNet(nn.Module):
    def __init__(self, in_dim = 2, out_dim = 1024):
        super(ResNet, self).__init__()
        inputnum = in_dim # 2 for the flow.
        blocknums = [2,2,3,4,6,7,3]
        outputnums = [32,64,64,128,128,256,256]

        self.firstconv = nn.Sequential(conv(inputnum, 32, 3, 2, 1, 1, False),
                                       conv(32, 32, 3, 1, 1, 1),
                                       conv(32, 32, 3, 1, 1, 1))

        self.inplanes = 32

        self.layer1 = self._make_layer(BasicBlock, outputnums[2], blocknums[2], 2, 1, 1) # 60, 60
        self.layer2 = self._make_layer(BasicBlock, outputnums[3], blocknums[3], 2, 1, 1) # 30, 30
        self.layer3 = self._make_layer(BasicBlock, outputnums[4], blocknums[4], 2, 1, 1) # 15, 15
        self.layer4 = self._make_layer(BasicBlock, outputnums[5], blocknums[5], 2, 1, 1) # 8, 8
        self.layer5 = self._make_layer(BasicBlock, outputnums[6], blocknums[6], 2, 1, 1) # 4, 4
        fcnum = outputnums[6] * 16

        # Hardcoded for 640x640 image for now.
        fc1 = linear(25600, 4096)
        fc2 = linear(4096, 2048)
        fc3 = linear(2048, out_dim)

        self.project_down = nn.Sequential(fc1, fc2, fc3)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = x.reshape(x.shape[0], -1)
        out = self.project_down(x)

        return out