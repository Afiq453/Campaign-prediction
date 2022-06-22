# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 17:14:18 2022

@author: AMD
"""

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,BatchNormalization
from tensorflow.keras import Input
import numpy as np

class ModelCreation():
    def __init__(self):
        pass
    def simple_lstm_layer(self,X_train,num_node=128,
                          drop_rate=0.3,output_node=1):
        model = Sequential ()
        model.add(Input((np.shape(X_train)[1],))) 
        model.add(Dense(32, activation = 'relu', name ='Hidden_Layer_1'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(32, activation = 'relu', name ='Hidden_Layer_2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(4,activation='softmax', name='Output_layer'))
        model.summary() # to visualize
        
        return model