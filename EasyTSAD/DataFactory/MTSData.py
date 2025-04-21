'''
MTSDataSet read data and provides various method for preprocessing data in datasets, including MinMax normalization and Z-score normalization.

There may be some missing values and inconsecutive timestamps in some datasets. We complement the missing timestamps to make it continuous at the possible minimum intervals, meanwhile filling the n/a values  using the linear interpolation method.
 
 MTS:
    SMD
    SWaT
    WADIA
    SMAP
    MSL
    PSM
'''

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import json

import sys 
import os

from ..Controller import PathManager

class MTSData:
    '''
    TSData contains all information used for training, validation and test, including the dataset values and dataset information. Some typical preprocessing method are provided in class methods.
    
    Attributes:
        train (np.ndarray):
            The training set in numpy format;
        valid (np.ndarray):
            The validation set in numpy format;
        test (np.ndarray):
            The test set in numpy format;
        train_label (np.ndarray):
            The labels of training set in numpy format;
        test_label (np.ndarray):
            The labels of test set in numpy format;
        valid_label (np.ndarray):
            The labels of validation set in numpy format;
        info (dict):
            Some informations about the dataset, which might be useful.
    '''
    def __init__(self, train, valid, test, train_label, test_label, valid_label, info) -> None:
        self.train = train
        self.valid = valid
        self.test = test
        self.train_label = train_label
        self.test_label = test_label
        self.valid_label = valid_label
        self.info = info
    
    @classmethod
    def buildfrom(cls, types, dataset, data_name, train_proportion:float=1, valid_proportion:float=0):
        '''
        Build customized TSDataSet instance from numpy file.
        
        Args:
            types (str):
                The dataset type. One of "UTS" or "MTS";
            dataset (str):
                The dataset name where the curve comes from, e.g. "WSD";
            dataname (str):
                The curve's name (Including the suffix '.npy'), e.g. "1.npy";
                
        
        
        Returns:\n
         A TSDataSet instance.
        '''
        pm = PathManager.get_instance()
        train_path = pm.get_dataset_train_set(types, dataset, data_name)
        train = np.load(train_path)
        train_label_path = pm.get_dataset_train_label(types, dataset, data_name)
        train_label = np.load(train_label_path)
        
        test_path = pm.get_dataset_test_set(types, dataset, data_name)
        test = np.load(test_path)
        test_label_path = pm.get_dataset_test_label(types, dataset, data_name)
        test_label = np.load(test_label_path)
        
        if train_proportion > 0 and train_proportion < 1:
            split_idx = int(len(train) * train_proportion)
            train = train[-split_idx:]
            train_label = train_label[-split_idx:]
        
        valid = train
        valid_label = train_label
        
        if valid_proportion > 0 and valid_proportion < 1:
            split_idx = int(len(train) * valid_proportion)
            train, valid = train[:-split_idx], train[-split_idx:]
            train_label, valid_label = train_label[:-split_idx], train_label[-split_idx:]
        
        info_path = pm.get_dataset_info(types, dataset, data_name)
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        return cls(train, valid, test, train_label, test_label, valid_label, info)

    def min_max_norm(self, feature_range=(0,1)):
        '''
        Function to preprocess multivariate time series using Min Max Normalization and generate training set, validation set and test set according to your settings. Then the data are clipped to [feature_range.min - 1, feature_range.max + 1]
        
        Params:
         feature_range - tuple (min, max), default=(0, 1)
          Desired range of transformed data.
        '''
        n_features = self.train.shape[1]
        
        for i in range(n_features):
            scaler = MinMaxScaler(feature_range=feature_range).fit(self.train[:, i].reshape(-1, 1))
            
            self.train[:, i] = scaler.transform(self.train[:, i].reshape(-1, 1)).flatten()
            
            self.valid[:, i] = scaler.transform(self.valid[:, i].reshape(-1, 1)).flatten()
            self.valid[:, i] = np.clip(self.valid[:, i], a_min=feature_range[0]-2, a_max=feature_range[1]+2)
                
            self.test[:, i] = scaler.transform(self.test[:, i].reshape(-1, 1)).flatten()
            self.test[:, i] = np.clip(self.test[:, i], a_min=feature_range[0]-2, a_max=feature_range[1]+2)

    def z_score_norm(self):
        '''
        Function to preprocess multivariate time series using Standard (Z-score) Normalization and generate training set, validation set and test set.
        '''
        n_features = self.train.shape[1]
        
        for i in range(n_features):
            scaler = StandardScaler().fit(self.train[:, i].reshape(-1, 1))
            
            self.train[:, i] = scaler.transform(self.train[:, i].reshape(-1, 1)).flatten()
            
            self.valid[:, i] = scaler.transform(self.valid[:, i].reshape(-1, 1)).flatten()
                
            self.test[:, i] = scaler.transform(self.test[:, i].reshape(-1, 1)).flatten()
        
    def differential(self, p):
        for i in range(p):
            n_features = self.train.shape[1]
            
            for j in range(n_features):
                self.train[:, j] = np.pad(self.train[:, j], ((0,1)), 'edge') - np.pad(self.train[:, j], ((1,0)), 'edge')
                self.train[:, j] = self.train[:, j][:-1]
            
                self.valid[:, j] = np.pad(self.valid[:, j], ((0,1)), 'edge') - np.pad(self.valid[:, j], ((1,0)), 'edge')
                self.valid[:, j] = self.valid[:, j][:-1]
            
                self.test[:, j] = np.pad(self.test[:, j], ((0,1)), 'edge') - np.pad(self.test[:, j], ((1,0)), 'edge')
                self.test[:, j] = self.test[:, j][:-1]
