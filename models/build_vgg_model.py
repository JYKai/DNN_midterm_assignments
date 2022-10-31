import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np


class VGG(nn.Module):
    
    def __init__(self, features, output_dim):
        super().__init__()

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

    
def get_vgg_layers(config, batch_norm=True):
    
    if config == 'vgg11':
        config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    elif config == 'vgg16':
        config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    else:
        print('I can not find the model config!')
    
    layers = []
    in_channels = 3

    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c

    return nn.Sequential(*layers)


def load_pretrained_model(model_name, output_dim):
    
    if model_name == 'vgg11':
        pretrained_model = models.vgg11_bn(pretrained=True)
    elif model_name == 'vgg16':
        pretrained_model = models.vgg16_bn(pretrained=True)
    
    IN_FEATURES = pretrained_model.classifier[-1].in_features
    OUTPUT_DIM = output_dim
    
    final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    pretrained_model.classifier[-1] = final_fc
    
    return pretrained_model
        
        
        