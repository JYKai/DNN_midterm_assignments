# Import Library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from tqdm.notebook import tqdm, trange
import numpy as np

import copy
import random
import time

import os
import argparse

from models.build_vgg_model import VGG
from models.build_vgg_model import get_vgg_layers
from models.build_vgg_model import load_pretrained_model

from tools.train_settings import train, evaluate, epoch_time
from tools.data_processing import iterator, split_data, normalize_image


def run_training(model_name, data_root, optimizer='adam', criterion='CE', learning_rate=1e-7, batch_size=128, output_dim=10, epochs=5, test=True):
    
    # Hyper-Parameters
    EPOCHS = epochs 
    BATCH_SIZE = batch_size 
    START_LR = learning_rate

    model_save_name = f'{model_name}_b{BATCH_SIZE}_e{EPOCHS}_lr{START_LR}.pt'
    model_save_path = f'./model_save/{model_save_name}'
    print("Complete Hyper-Parameter Settings!")
    
    
    # Build Model
    model_layer = get_vgg_layers(model_name) # args : model_name
    model = VGG(model_layer, output_dim) # args : output_dim
    pretrained_model = load_pretrained_model(model_name, output_dim)
    model.load_state_dict(pretrained_model.state_dict())
    print("Complete Build Model!")
    
    
    # Data Preprocessing
    train_data, test_data, test_transforms = iterator(data_root) # args : data_root
    train_data, valid_data = split_data(train_data, test_data, test_transforms)
    print("Complete Data Preprocessing")
    
    
    # DataLoader
    train_iterator = data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

    valid_iterator = data.DataLoader(valid_data,
                                     batch_size=BATCH_SIZE)

    test_iterator = data.DataLoader(test_data,
                                    batch_size=BATCH_SIZE)
    print("Complete DataLoader")
    
    
    # Training Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if criterion == 'CE':
        criterion = nn.CrossEntropyLoss()

    criterion = criterion.to(device)

    params = [
          {'params': model.features.parameters()},
          {'params': model.classifier.parameters()}
         ]

    if optimizer == 'adam':
        optimizer = optim.Adam(params, lr=START_LR)
    print("Complete Training Settings")

    tensordir_path = f'{model_name}_b{BATCH_SIZE}_e{EPOCHS}_lr{START_LR}'
    tensorboard_loss = f'Loss/{model_name}_b{BATCH_SIZE}_e{EPOCHS}_lr{START_LR}'
    tensorboard_acc = f'Acc/{model_name}_b{BATCH_SIZE}_e{EPOCHS}_lr{START_LR}'
    tensorboard_dir = f'logs/{tensordir_path}'
    writer = SummaryWriter(tensorboard_dir)
    print("Tensorboard Settings")
    
    
    # Training
    best_valid_loss = float('inf')

    for epoch in trange(EPOCHS, desc="Epochs"):
        
        start_time = time.monotonic()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        # writer.add_scalar(f'Train_{model_name}_b{BATCH_SIZE}_e{EPOCHS}', {'Loss': train_loss, 'Accuracy': train_acc}, epoch)

        writer.add_scalar('Train' + tensorboard_loss, train_loss, epoch+1)
        writer.add_scalar('Train' + tensorboard_acc, train_acc*100, epoch+1)

        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        # writer.add_scalar(f'Valid_{model_name}_b{BATCH_SIZE}_e{EPOCHS}', {'Loss': valid_loss, 'Accuracy': valid_acc}, epoch)

        # writer.add_scalar(tensorboard_acc, {'train': train_acc, 'valid': valid_acc}, epoch)       
        writer.add_scalar('Valid' + tensorboard_loss, valid_loss, epoch+1)
        writer.add_scalar('Valid' + tensorboard_acc, valid_acc*100, epoch+1)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    writer.close()
    
    if test:
        model.load_state_dict(torch.load(model_save_path))
        test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

def main():
    
    # Seed Settings
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # parser
    parser = argparse.ArgumentParser(description='Run a train & validate VGG')
    parser.add_argument('--model_name', type=str, default='vgg11', help='Name of module in the "models/" folder.')
    parser.add_argument('--data_root', type=str, default='./data', help='cifar10 dataset root')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer name')
    parser.add_argument('--criterion', type=str, default='CE', help='criterion name')
    parser.add_argument('--learning_rate', type=float, default=1e-7, help='LR size')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--output_dim', type=int, default=10, help='output_dim')
    parser.add_argument('--epochs', type=int, default=5, help='epochs')

    args = parser.parse_args()

    run_training(args.model_name, 
                 args.data_root, 
                 args.optimizer,
                 args.criterion,
                 args.learning_rate,
                 args.batch_size,
                 args.output_dim,
                 args.epochs)


if __name__ == '__main__':
    main()
