# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:16:46 2022

@author: Abhishek
"""

from ann import ann
import pickle as pickle
import os
from tensorflow.keras.models import load_model



class lcd():
    def _init__(self):
        self.init=False
        
    def roboconfig(self, robot, humanoid, load_model_=False, path="."):
        self.robot = robot
        self.humanoid= humanoid
        
        if(self.humanoid):
            self.input_dim= 12
            
        else:
            self.input_dim = 7
            
            
        if(load_model_):
            self.ann = ann()
            self.ann.modelArch(self.input_dim)
            self.ann = load_model(path + '/'+self.robot + '_ANN', compile=False)
            self.model.log = pickle.load(open(path + '/'+self.robot + '_ANN_LOG', 'rb'))
            self.init = True  
            
        else:
            self.ann = ann()
            self.ann.modelArch(self.input_dim)
            
    def fit(self, train_data, labels, epochs, batch_size, save_model_=False):
        self.ann.fit(train_data, labels, epochs, batch_size)
        self.leg_probblities = self.ann.model.predict(train_data)
        self.model_log = self.ann.model.log.history
        
        
        if(save_model_):
            self.ann.model.save(self.robot + '_ANN')
            with open(self.robot + '_ANN_log', 'wb') as file_pi:
                pickle.dump(self.ann.model.log, file_pi)
                
        self.init = True
        
    def predict(self, data):
        leg_probablities = self.ann.model.predict(data.reshape(1,-1))
        return leg_probablities
    
    def test_prediction(self, data):
        leg_probablities = self.ann.model.predict(data)
        
        return leg_probablities
    
        
 
    
                              
    