import torch
import torch.utils.data
import numpy as np
from ...DataFactory import TSData
from typing import Tuple, Sequence, Dict

class  OneByOneDataset(torch.utils.data.Dataset):
    '''
    The Dateset for one by one training and testing. 
    '''
    def __init__(self, data: np.ndarray, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        
        self.len, = data.shape
        
        self.sample_num = max(self.len - self.window_size + 1, 0)
        self.samples, self.targets = self.__getsamples(data)
        
    def __getsamples(self, data):
        X = torch.zeros((self.sample_num, self.window_size))
        
        for i in range(self.sample_num):
            X[i, :] = torch.from_numpy(data[i : i + self.window_size])
            
        return X, X
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, index):
        return self.samples[index, :], self.targets[index, :]
    
    
class  AllInOneDataset(torch.utils.data.Dataset):
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
            raise NotImplementedError
        
        assert cnt == self.sample_num
        
        self.samples, self.targets = X, X
    
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, index):
        return self.samples[index, :], self.targets[index, :]
        