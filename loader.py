import torch
from torchvision import transforms,datasets,models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''
Hyperparameters and Network architecture are taken according  to the 
Author's Implementation.The model's highly depending upon Specific Architecture. So the Resnet Architecture is 
Inspired from Author's Implementation. Remaining is coded from Scratch
'''

# Hyperparameters
batch_size = 400
train_iterations = 1500

#loading the dataset
train_set = datasets.CIFAR10(root="data", train=True, download=True,
                                      transform=transforms.Compose([ transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

test_set = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

training_loader = DataLoader(train_set,batch_size=batch_size, shuffle=True,pin_memory=True)
validation_loader = DataLoader(test_set,batch_size=16,shuffle=True,pin_memory=True)

