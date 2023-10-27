import torch
import torch.utils.data
import numpy as np
from ...DataFactory import TSData
from typing import Tuple, Sequence, Dict

class  OneByOneDataset(torch.utils.data.Dataset):
    '''
    The Dateset for one by one training and testing. 
    '''
    def __init__(self, tsData: TSData, phase: str, window_size: int, step: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.step = step
        
        if phase == "train":
            self.len, = tsData.train.shape
            self.sample_num = max(0, (self.len - self.window_size) // self.step + 1)
            
            X = torch.zeros((self.sample_num, self.window_size))
            
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(tsData.train[i * step : i * step + self.window_size])
                
        elif phase == "valid":
            self.len, = tsData.valid.shape
            self.sample_num = max(0, (self.len - self.window_size) // self.step + 1)
            
            X = torch.zeros((self.sample_num, self.window_size))
            
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(tsData.valid[i * step : i * step + self.window_size])
                
        elif phase == "test":
            self.len, = tsData.test.shape
            self.sample_num = max(0, (self.len - self.window_size) // self.step + 1)
            
            X = torch.zeros((self.sample_num, self.window_size))
            
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(tsData.test[i * step : i * step + self.window_size])
                
        else:
            raise ValueError('Arg "phase" in OneByOneDataset() must be one of "train", "valid", "test"')
            
        self.samples, self.targets = torch.unsqueeze(X, -1), torch.unsqueeze(X, -1)
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, index):
        return self.samples[index, :, :], self.targets[index, :, :]
    
    
class  AllInOneDataset(torch.utils.data.Dataset):
    '''
    The Dateset for one by one training and testing. 
    '''
    def __init__(self, datas: Dict[str, TSData], phase:str, window_size: int, step: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.sample_num = 0
        self.step = step
        
        if phase == "train":
            for _, i in datas.items():
                ll, = i.train.shape
                self.sample_num += max(0, (ll - self.window_size) // self.step + 1)
            
            X = torch.zeros((self.sample_num, self.window_size))
            cnt = 0
            for _, i in datas.items():
                ll, = i.train.shape
                ll = max(0, (ll - self.window_size) // self.step + 1)
                for j in range(ll):
                    X[cnt, :] = torch.from_numpy(i.train[j * step : j * step + self.window_size])
                    cnt += 1
        
        elif phase == "valid":
            for _, i in datas.items():
                ll, = i.valid.shape
                self.sample_num += max(0, (ll - self.window_size) // self.step + 1)
            
            X = torch.zeros((self.sample_num, self.window_size))
            cnt = 0
            for _, i in datas.items():
                ll, = i.valid.shape
                ll = max(0, (ll - self.window_size) // self.step + 1)
                for j in range(ll):
                    X[cnt, :] = torch.from_numpy(i.valid[j * step : j * step + self.window_size])
                    cnt += 1
        
        else:
            raise ValueError('Arg "phase" in AllInOneDataset() must be one of "train", "valid"')
        
        assert cnt == self.sample_num
        
        self.samples, self.targets = torch.unsqueeze(X, -1), torch.unsqueeze(X, -1)
    
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, index):
        return self.samples[index, :, :], self.targets[index, :, :]
        