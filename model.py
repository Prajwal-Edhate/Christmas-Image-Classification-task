from email.mime import image
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

#Implementation of ConvNext-Tiny with reference to https://github.com/facebookresearch/ConvNeXt.git

class LayerNorm(nn.Module):
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dimension, layer_scale_init_value = 1e-6):
        super().__init__()
        self.layer_scale_init_value = layer_scale_init_value
        self.dwconv = nn.Conv2d(dimension, dimension, kernel_size = 7, padding = 3, groups = dimension)
        self.norm = LayerNorm(dimension, eps=1e-6)
        self.pwconv1 = nn.Linear(dimension, 4 * dimension)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dimension, dimension)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dimension)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()

    def forward(self,x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0,3,1,2)
        x = input + self.drop_path(x)
        return x

class Network(nn.Module):

    def __init__(self, in_channels = 3, depths = [3,3,9,3], dims =[96, 192, 384, 768],
                 drop_path_rate = 0., layer_scale_init_value=1e-6, head_init_scale=1., num_classes = 8):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        dlayers = nn.Sequential(nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4), LayerNorm(dims[0],data_format="channels_first"))
        self.downsample_layers.append(dlayers)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(LayerNorm(dims[i],eps = 1e-6,data_format="channels_first"),
                                         nn.Conv2d(dims[i],dims[i+1],kernel_size=2,stride=2)))
        
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[Block(dims[i],1e-6) for j in range(depths[i])]))
        
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), 'model.pkl')
