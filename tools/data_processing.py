# Import Library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import copy
import random


def iterator(data_root):
    
    ROOT = data_root
    
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
                               transforms.Resize(pretrained_size),
                               transforms.RandomRotation(5),
                               transforms.RandomHorizontalFlip(0.5),
                               transforms.RandomCrop(pretrained_size, padding=10),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=pretrained_means,
                                                    std=pretrained_stds)
                           ])

    test_transforms = transforms.Compose([
                               transforms.Resize(pretrained_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=pretrained_means,
                                                    std=pretrained_stds)
                           ])

    train_data = datasets.CIFAR10(ROOT,
                                  train=True,
                                  download=True,
                                  transform=train_transforms)

    test_data = datasets.CIFAR10(ROOT,
                                 train=False,
                                 download=True,
                             transform=test_transforms)
    
    return train_data, test_data, test_transforms


def split_data(train_data, test_data, test_transform, ratio=0.9):
    
    VALID_RATIO = ratio

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])
    
    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transform
    
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')
    
    return train_data, valid_data


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image
    
    
    
    
    
    
    
    
    
    