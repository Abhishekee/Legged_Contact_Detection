# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:19:49 2022

@author: Abhishek
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import seaborn as sns
import scikitplot as skplt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from numpy import mean, std
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from lcd import lcd
from sklearn.metrics import f1_score
from sklearn.metrics import auc



humanoid  = True
robot = 'ATLAS'
noise = True
prefix = 'Dataset/'


def read_dataset(filename):
    df = pd.read_csv(filename)
    dataset = df.values
    dataset = dataset.astype('float32')
    
    return dataset

def put_noise(data, std):
    mu = 0
    s = np.random.normal(mu, std, data.shape[0])
    for k in range(0, data.shape[1]):
        for i in range(0, data.shape[0]):
            data[i,k] = data[i,k] + s[i]
            
    return data

def normalize_data(din, dmax):
    if(dmax != 0):
        dout = np.abs(din/dmax)
    else:
        dout = np.zeros(np.size(din))
        
    return dout

    
def remove(features, dataset):
    dataset = np.delete(dataset, features, axis=1)
    return dataset

def remove_outliers(dataset, labels):
    feature_mean = []
    feature_std = []
    
    for i in range(dataset.shape[1]):
        feature_mean.append(np.mean(dataset[:, i]))
        feature_std.append(np.std(dataset[:,i]))
        
    cut_off = []
    lower_bound = []
    upper_bound = []
    num_std = 3
    
    for i in range(dataset.shape[1]):
        cut_off.append(feature_std[i]*num_std)
        lower_bound.append(feature_mean[i] - cut_off[i])
        upper_bound.append(feature_mean[i] + cut_off[i])
        
        
    list_outliers = []
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            if (dataset[i,j] <= lower_bound[j]) or (dataset[i,j] >= upper_bound[j]):
                list_outliers.append(i)
                continue
    before_deletion = dataset.shape[0]
        
    dataset = np.delete(dataset, list_outliers, axis=0)
        
    labels = np.delete(labels, list_outliers, axis=0)
        
    return dataset, labels

def slip_fly_merger(labels):
    for i in range(labels.shape[0]):
        if labels[i] == 2:
            labels[i] = 1
    return labels

if __name__ == '__main__':
    dataset = read_dataset(prefix + 'ATLAS_21k_02ground.csv')
    labels = dataset[:, -1]
    dataset = np.delete(dataset, -1, axis=1)
    
    dataset, labels = remove_outliers(dataset, labels)
    
    labels = slip_fly_merger(labels)
    
    dataset, labels = remove_outliers(dataset, labels)
    
    
    if humanoid:
        
        if noise == True:
            dataset[:, :3] = put_noise(dataset[:,:3], 0.6325)
            dataset[:,3:6] = put_noise(dataset[:,3:6],0.03)        
            dataset[:,9:12] = put_noise(dataset[:,9:12],0.00523)
            
    else:
        
        dataset = remove([0,1,3,4,5], dataset)
        if noise == True:
            dataset[:, 0:1] = put_noise(dataset[:,0:1], 0.6325)
            dataset[:,1:4] = put_noise(dataset[:,1:4],0.03)        
            dataset[:,4:7] = put_noise(dataset[:,4:7],0.00523)
            
            
    for i in range(dataset.shape[1]):
        dataset[:,i] = normalize_data(dataset[:,i], np.max(abs(dataset[:,i])))
        
        
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size = 0.25, random_state = 15)
    y_train = to_categorical(y_train, num_classes = 2)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1])
    
    
    legged_contact = lcd()
    
    legged_contact.roboconfig(robot, humanoid)
    legged_contact.fit(X_train, y_train, 50, 32, True)
    
    
    
    predict_x = legged_contact.test_prediction(X_test)
    labels_x = np.argmax(predict_x, axis=1)
    confm = confusion_matrix(y_test, labels_x)
   
    
    auc = roc_auc_score(y_test, labels_x)
    fpr, tpr, thresholds =roc_curve(y_test, labels_x)
    print('ANN: AUC=%.3f' % (auc))
    plt.plot(fpr, tpr, linestyle='dotted', label = "ATLAS_21K")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    
    
    precision, recall, _ = precision_recall_curve(y_test, labels_x)
 
    
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label='ANN')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    plt.legend()
    plt.show()
        
    print(confm)
    print("Stable accuracy = ", confm[0,0]*100/(confm[0,0]+confm[0,1]))
    print("Slip accuracy = ", confm[1,1]*100/(confm[1,0]+confm[1,1]))
    
    group_names = ['True Pos','False Pos','False Neg','True Neg']

    group_counts = ["{0:0.0f}".format(value) for value in
                confm.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                     confm.flatten()/np.sum(confm)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(confm, annot=labels, fmt='', cmap='Blues')

    ax.set_title(' Confusion Matrix ');
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values ');
    ax.xaxis.set_ticklabels(['True','False'])
    ax.yaxis.set_ticklabels(['True','False'])
    plt.savefig("Confusion_Matrix_ATLAS_21K")
    plt.show()
    
    test_datasets = ['ATLAS_7k_04ground.csv','ATLAS_10k_05ground.csv','ATLAS_50k_mixedFriction.csv','TALOS_50k_mixedFriction.csv']
    
    
    for filename in test_datasets:
        
        unseen = read_dataset(prefix + filename)
        
        unseenlabels = unseen[:, -1]
        unseen = np.delete(unseen, -1, axis = 1)
        
        print("SLIP",np.count_nonzero(unseenlabels == 2))
        print("STABLE",np.count_nonzero(unseenlabels == 0))
        print("FLY",np.count_nonzero(unseenlabels == 1))

        unseen, unseenlabels = remove_outliers(unseen,unseenlabels)

        unseenlabels = slip_fly_merger(unseenlabels)
        
        
        if humanoid:
            if noise:
                unseen[:,:3]   = put_noise(unseen[:,:3],0.6325)      
                unseen[:,3:6]  = put_noise(unseen[:,3:6],0.0316)     
                unseen[:,6:9]  = put_noise(unseen[:,6:9],0.0078)
                unseen[:,9:12] = put_noise(unseen[:,9:12],0.00523)
        else:
            unseen = remove([0,1,3,4,5], unseen)
                
            if noise:
                unseen[:,0:1] =put_noise(unseen[:,0:1], 0.6325)
                unseen[:,1:4] =put_noise(unseen[:,1:4], 0.0078)
                unseen[:,4:7] =put_noise(unseen[:,4:7], 0.00523)
                    
                    
        for i in range(unseen.shape[1]):
            unseen[:,i] = normalize_data(unseen[:,i], np.max(abs(unseen[:,i])))
                
        unseen = unseen.reshape(unseen.shape[0], unseen.shape[1])
        predict_x1 = legged_contact.test_prediction(unseen) 
        labels_x1 = np.argmax(predict_x1,axis=1)
       
        auc = roc_auc_score(unseenlabels, labels_x1)
        fpr, tpr, thresholds =roc_curve(unseenlabels, labels_x1)
        print('ANN: AUC=%.3f' % (auc))
        plt.plot(fpr, tpr, linestyle='dotted', label = "Unseen")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
        
        precision, recall, _ = precision_recall_curve(unseenlabels, labels_x1)
     
        
        no_skill = len(y_test[y_test==1]) / len(y_test)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', label='ANN')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        
        plt.legend()
        plt.show()
        
        
        
        
        
        confm1 = confusion_matrix(unseenlabels, labels_x1)
            
        classes_x1 = np.argmax(predict_x1,axis=1)
        confm1 = confusion_matrix(unseenlabels,classes_x1)

      
        print(filename)
        print(confm1)
        print("Stable accuracy = ", confm1[0,0]*100/(confm1[0,0]+confm1[0,1]))
        print("Slip  accuracy = ", confm1[1,1]*100/(confm1[1,0]+confm1[1,1]))
        
        group_names = ['True Pos','False Pos','False Neg','True Neg']

        group_counts = ["{0:0.0f}".format(value) for value in
                    confm1.flatten()]

        group_percentages = ["{0:.2%}".format(value) for value in
                         confm1.flatten()/np.sum(confm1)]

        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]

        labels = np.asarray(labels).reshape(2,2)

        ax = sns.heatmap(confm1, annot=labels, fmt='', cmap='Greens')

        ax.set_title(' Confusion Matrix ');
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values ');
        ax.xaxis.set_ticklabels(['True','False'])
        ax.yaxis.set_ticklabels(['True','False'])
        plt.savefig("Confusion_Matrix_Unseen")
        plt.show()

        np.savetxt("Predictions", predict_x1)
        
    
    
        
               
    
    
    
    
