import torch
import torch.utils.data
import numpy as np
from .. import TSData
from typing import Tuple, Sequence, Dict

class  UTSOneByOneDataset(torch.utils.data.Dataset):
    '''
    The Dateset for one by one training and testing. 
    '''
    def __init__(self, tsData: TSData, phase:str, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        
        if phase == "train":
            self.len, = tsData.train.shape
            self.sample_num = max(self.len - self.window_size + 1, 0)
            X = torch.zeros((self.sample_num, self.window_size))
            
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(tsData.train[i : i + self.window_size])
                
        elif phase == "valid":
            self.len, = tsData.valid.shape
            self.sample_num = max(self.len - self.window_size + 1, 0)
            X = torch.zeros((self.sample_num, self.window_size))
            
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(tsData.valid[i : i + self.window_size])
                
        elif phase == "test":
            self.len, = tsData.test.shape
            self.sample_num = max(self.len - self.window_size + 1, 0)
            X = torch.zeros((self.sample_num, self.window_size))
            
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(tsData.test[i : i + self.window_size])
                
        else:
            raise ValueError('Arg "phase" in OneByOneDataset() must be one of "train", "valid", "test"')
            
        self.samples, self.targets = X, X
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, index):
        return self.samples[index, :], self.targets[index, :]
    
    
class  UTSAllInOneDataset(torch.utils.data.Dataset):
    '''
    The Dateset for one by one training and testing. 
    '''
    def __init__(self, datas: Dict[str, TSData], phase:str, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.sample_num = 0
        
        if phase == "train":
            for _, i in datas.items():
                ll, = i.train.shape
                self.sample_num += max(ll - self.window_size + 1, 0)
            
            X = torch.zeros((self.sample_num, self.window_size))
            cnt = 0
            for _, i in datas.items():
                ll, = i.train.shape
                ll = max(ll - self.window_size + 1, 0)
                for j in range(ll):
                    X[cnt, :] = torch.from_numpy(i.train[j: j + self.window_size])
                    cnt += 1
        
        elif phase == "valid":
            for _, i in datas.items():
                ll, = i.valid.shape
                self.sample_num += max(ll - self.window_size + 1, 0)
            
            X = torch.zeros((self.sample_num, self.window_size))
            cnt = 0
            for _, i in datas.items():
                ll, = i.valid.shape
                ll = max(ll - self.window_size + 1, 0)
                for j in range(ll):
                    X[cnt, :] = torch.from_numpy(i.valid[j: j + self.window_size])
                    cnt += 1
        
        else:
            raise ValueError('Arg "phase" in AllInOneDataset() must be one of "train", "valid"')
        
        assert cnt == self.sample_num
        
        self.samples, self.targets = X, X
    
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, index):
        return self.samples[index, :], self.targets[index, :]
        