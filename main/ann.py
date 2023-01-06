# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 2:21:54 2022

@author: Abhishek
"""
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout, MaxPooling1D, Activation
import tensorflow as tf
import os
from tensorflow.keras.utils import get_custom_objects
import tempfile


def pick():
    def __getstate__(self):
        model_str = ""
        with tempfile.NameTemporaryFile(suffix='.hdf5', delete=False, dir=os.getcwd()) as fd:
            save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        model_dict = {'model_str': model_str}
        os.unlink(fd.name)
        return model_dict
    
    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False, dir=os.getcwd()) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = load_model(fd.name)
        os.unlink(fd.name)
        self.__dict__ = model.__dict__
        
    cls = Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__ 
    
    
class ann():
    def __init__(self):
        self.init = False
        pick()
        
    def modelArch(self, input_dim):
        
        self.model = Sequential()
        
        self.model.add(Dense(256, activation='relu', use_bias = True,input_shape=(input_dim,)))
        self.model.add(Dense(128, activation= 'relu'))
        self.model.add(Dropout(0.2))
        
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.2))
        
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(2, activation= 'sigmoid'))
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.init=True
        
        
    def fit(self, x_train, y_train, epochs_, batch_size_):
        self.model.log = self.model.fit(x_train, y_train, validation_split=0.2,epochs=epochs_,batch_size=batch_size_, verbose=1, shuffle=True)
        