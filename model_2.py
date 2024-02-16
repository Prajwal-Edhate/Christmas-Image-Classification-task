from email.mime import image
import torch
import torch.nn as nn
from torchvision.ops.stochastic_depth import StochasticDepth

class ConvNormActivation(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size = (3,3), stride = (2,2),padding=(1,1), silu = True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size = (1,1),stride = (1,1), bias = False),
                                        nn.BatchNorm2d(out_channels, affine = True, track_running_stats = True)]
        if silu:
            layers.append(nn.SiLU(inplace=True))
            layers[0]= nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size,stride = stride, padding = padding, bias = False)
            if in_channels == out_channels:
                layers[0] = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size,stride = stride, padding = padding, bias = False, groups = out_channels)
        
        super().__init__(*layers)
        self.out_channels = out_channels

class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = (1,1), stride = (1,1)):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.fc1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride)
        self.fc2 = nn.Conv2d(out_channels, in_channels, kernel_size = kernel_size, stride = stride)
        self.activation = nn.SiLU(inplace=True)
        self.scale_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.scale_activation(x)
        return x
        

class MBConv(nn.Module):

    def __init__(self, in_channels, mid_channels, squeeze_channels, out_channels, padding = (1,1), stride = (2,2), kernel_size=(3,3), num_convnorm = 1, num_squeeze = 1, p = 0.0, mode = 'row'):
        super().__init__()
        layers = [ConvNormActivation(in_channels=in_channels ,out_channels=mid_channels,silu=True)]
        if num_convnorm>1:
            layers[0] = ConvNormActivation(in_channels=in_channels ,out_channels=mid_channels,silu=True, padding= 0, kernel_size=(1,1), stride = (1,1))

        for _ in range(1,num_convnorm):
            layers.append(ConvNormActivation(in_channels=mid_channels, out_channels=mid_channels, silu=True, kernel_size=kernel_size, stride=stride, padding=padding))
        
        for _ in range(num_squeeze):
            layers.append(SqueezeExcitation(in_channels=mid_channels, out_channels=squeeze_channels))
        
        layers.append(ConvNormActivation(in_channels=mid_channels, out_channels=out_channels, silu=False))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(p = p, mode = mode)
    
    def forward(self,x):
        x = self.block(x)
        x = self.stochastic_depth(x)
        return x

class EfficientNet(nn.Module):

    def __init__(self,num_classes = 8):
        super(EfficientNet, self).__init__()
        layers = [ConvNormActivation(3,32,silu=True),
                  nn.Sequential(MBConv(32, 32,8,16),
                                MBConv(16, 16, 4, 16, p = 0.008695652173913044)),
                  nn.Sequential(MBConv(16, 96, 4, 24, num_convnorm=2, p=0.017391304347826087),
                                MBConv(24,144,6,24,num_convnorm=2,stride = (1,1), p = 0.026086956521739136),
                                MBConv(24,144,6,24,num_convnorm=2,stride=(1,1), p =0.034782608695652174)),
                                nn.Sequential(MBConv(24,144,6,40,num_convnorm=2, kernel_size=(5,5),padding=(2,2), p =0.043478260869565216),
                                              MBConv(40,240,10,40, num_convnorm=2, kernel_size=(5,5), padding=(2,2), stride=(1,1), p = 0.05217391304347827),
                                              MBConv(40,240, 10,40, num_convnorm=2, kernel_size=(5,5), padding=(2,2), stride=(1,1), p =0.06086956521739131))]
        layers.append(nn.Sequential(MBConv(40,240,10,80,num_convnorm=2,p = 0.06956521739130435),
                                    MBConv(80, 480, 20, 80, stride=(1,1), num_convnorm=2, p = 0.0782608695652174),
                                    MBConv(80, 480, 20, 80, stride=(1,1), num_convnorm=2, p = 0.08695652173913043),
                                    MBConv(80, 480, 20, 80, stride=(1,1), num_convnorm=2, p = 0.09565217391304348)))
        
        layers.append(nn.Sequential(MBConv(80,480,20,112, kernel_size=(5,5), stride=(1,1), padding=(2,2), num_convnorm=2, p = 0.10434782608695654),
                                    MBConv(112,672,28,112, kernel_size=(5,5), stride=(1,1), padding=(2,2), num_convnorm=2, p = 0.11304347826086956),
                                    MBConv(112,672,28,112, kernel_size=(5,5), stride=(1,1), padding=(2,2), num_convnorm=2, p = 0.12173913043478261),
                                    MBConv(112,672,28,112, kernel_size=(5,5), stride=(1,1), padding=(2,2), num_convnorm=2, p = 0.13043478260869565)))
        
        layers.append(nn.Sequential(MBConv(112, 672, 28, 192, kernel_size=(5,5), stride=(2,2), padding=(2,2), num_convnorm=2, p = 0.1391304347826087),
                                    MBConv(192,1152,48,192, kernel_size=(5,5), stride=(1,1), padding=(2,2), num_convnorm=2, p = 0.14782608695652175),
                                    MBConv(192,1152,48,192, kernel_size=(5,5), stride=(1,1), padding=(2,2), num_convnorm=2, p = 0.1565217391304348),
                                    MBConv(192,1152,48,192, kernel_size=(5,5), stride=(1,1), padding=(2,2), num_convnorm=2, p = 0.16521739130434784),
                                    MBConv(192,1152,48,192, kernel_size=(5,5), stride=(1,1), padding=(2,2), num_convnorm=2, p = 0.17391304347826086)))
        
        layers.append(nn.Sequential(MBConv(192,1152,48,320,kernel_size=(3,3), stride=(1,1), padding=(1,1),num_convnorm=2, p = 0.1826086956521739),
                                    MBConv(320, 1920,80,320, kernel_size=(3,3), stride=(1,1), padding=(1,1), num_convnorm=2, p = 0.19130434782608696)))
        
        layers.append(ConvNormActivation(320,1280,kernel_size=(1,1),stride=(1,1),padding=0, silu = True))
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Sequential(nn.Dropout(p = 0.2, inplace=True),
                                        nn.Linear(in_features = 1280, out_features = 1000, bias = True))

    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    