from email.mime import image
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = stride, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels,track_running_stats = True)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats = True)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.downsample(identity)
        x = self.relu(x)
        return x

class Network(nn.Module):
    
    def __init__(self, num_classes = 8):
        
        #############################
        # Initialize your network
        #############################
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, padding = 3, stride = 2, bias = False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride =2, padding = 1)
        self.layer1 = self._make_layer(64,64,3)
        self.layer2 = self._make_layer(64,128,4,2)
        self.layer3 = self._make_layer(128,256,6,2)
        self.layer4 =self._make_layer(256,512,3,2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes, bias = True)

    def _make_layer(self,in_channels,out_channels,num_blocks,stride = 1):
        layers = [BasicBlock(in_channels=in_channels,out_channels=out_channels, stride=stride)]
        for _ in range(1,num_blocks):
            layers.append(BasicBlock(in_channels=out_channels, out_channels=out_channels, stride = 1))
        return nn.Sequential(*layers)

        
    def forward(self, x):
        
        #############################
        # Implement the forward pass
        #############################
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), 'model.pkl')