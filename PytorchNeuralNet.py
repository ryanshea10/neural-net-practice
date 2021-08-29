#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 20:03:15 2021

@author: ryanshea
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt


"Define Hyperparameters"

input_size = 784 # images are 28x28
hidden_size = 100
num_classes = 10
epochs = 2
batch_size = 100
lr = .001


"Import MNIST Dataset and create Dataloader" # nn will be used to clasify imgs

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                           transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                           shuffle = False)

# examples = iter(train_loader)
# samples, labels = examples.next()
# print(samples.shape, labels.shape)

# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(samples[i][0], cmap='gray')

# plt.show()


"Create NN Class"

#Define nn model
class NeuralNet(nn.Module):
    
    # init neural net parameters (num_classes corresponds to the output size)
    def __init__(self, input_size, hidden_size, num_classes):
        # call the superclass constructor
        super(NeuralNet, self).__init__()
        # first linear layer (input to hidden connections)
        self.l1 = nn.Linear(input_size, hidden_size)
        # initialize hidden layer activation function
        self.relu = nn.ReLU()
        # second linear layer (hidden to output connections)
        self.l2 = nn.Linear(hidden_size, num_classes)
        
    # define the forward pass
    def forward(self, x):
        # apply first linear layer
        out = self.l1(x)
        # apply activations
        out = self.relu(out)
        # apply second linear layer
        out = self.l2(out)
        # the crossentropy loss will apply the softmax activation
        # so there is no need to perform the final activation
        return out

model = NeuralNet(input_size, hidden_size, num_classes)
    

"Initialize Loss Function and Optimizer"

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


"Create Training Loop"

total_steps = len(train_loader)

# main loop
for epoch in range(epochs):
    # loop over the batches
    for i, (imgs, labels) in enumerate(train_loader):
        # reshape img tensor (flatten image array)
        # the first dimension will become 100 which corresponds to the batch size
        imgs = imgs.reshape(-1, 28*28)
        
        # forward pass
        outputs = model(imgs)
        loss = loss_function(outputs, labels)
        
        # backward pass
        loss.backward()
        optimizer.step()
        
        # zero the gradients
        optimizer.zero_grad()
        
        if (i + 1) % 100 == 0:
            print(f'epoch: {epoch+1}/{epochs}, step {i+1}/{total_steps}, loss = {loss.item():.4f}')


"Evaluate Model"

# don't want to torch to track these operations so we detach it from the 
# computational graph using the following wrapper
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    
    for imgs, labels in test_loader:
        imgs = imgs.reshape(-1, 28*28)
        outputs = model(imgs)
        
        # returns (value, index) of the max value in the given dimension
        _, predictions = torch.max(outputs, 1)
        #add the number of samples in the current batch to the total n_samples
        n_samples += labels.shape[0]
        # add one to n_correct for each correct prediction
        n_correct += (predictions == labels).sum().item()
    
    acc = 100 * n_correct / n_samples
    print(f'accuracy  = {acc}')
