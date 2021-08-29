#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 17:54:48 2021

@author: ryanshea
"""

#import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

np.random.seed(10)

x = np.random.uniform(low=0, high=10, size=300).reshape(-1,1)
y = x**2 + np.random.normal(5, .5, 300).reshape(-1, 1)


#plt.scatter(x, y, c='red')

#scale data
x=StandardScaler().fit_transform(x)
y=StandardScaler().fit_transform(y)


def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivatives_sigmoid(x):
    return x*(1-x)



'Nonlinear Segment'

epoch=1000
lr=.001 
hiddenlayer_neurons= 20
inputlayer_neurons = 1
output_neurons = 1 

#initialize weights, the sizes are tailored so that the matrix multiplication works out
wh=np.random.normal(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.normal(size=(1,hiddenlayer_neurons))
wout=np.random.normal(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.normal(size=(1,output_neurons))

iterations=[]
mse_list=[]
for i in range(epoch):
    'Forward Pass'
    #multiply inputs by input to hidden weights
    hidden_layer_input1=np.dot(x,wh)
    #add bias 
    hidden_layer_input=hidden_layer_input1 + bh
    #activate the wieghted inputs
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    #multiply activations by hidden to output weights
    output1=np.dot(hiddenlayer_activations,wout)
    #add bias
    output=output1+bout
    
    mse=mean_squared_error(y, output)
    iterations.append(i)
    mse_list.append(mse)
    
    'Backward Pass'
    #calcuate error
    error=y-output
    #calculate gradient with respect to the hidden to output weights
    output_grad=np.dot(hiddenlayer_activations.T,error)*-1
    #calculate gradient with respect to the input to hidden weights
    #this is equal to the -1*error*(hidden to output weights)*x*f(x)*(1-f(x)) where f(x)=sigmoid
    #error*hidden-output is called the error at the hidden layer
    error_hidden=np.dot(error, wout.T)
    #f(x)*(1-f(x)) where f(x)=sigmoid is called the slope of the hidden layer
    slope_hidden=derivatives_sigmoid(hiddenlayer_activations)
    #calculate the final gradient
    hidden_grad=np.dot(x.T, error_hidden*slope_hidden)*-1
    #update parameters
    wout-=output_grad*lr
    wh-=hidden_grad*lr
    bout-=np.sum(error, axis=0, keepdims=True)*lr
    bh-=np.sum(slope_hidden*error_hidden, axis=0, keepdims=True)*lr
        
    

#mse_df=pd.DataFrame({'iterations':iterations, 'mse':mse_list})

plt.scatter(x, y, c='blue')
plt.scatter(x, output, c='red')


#plt.scatter(iterations, mse_list)