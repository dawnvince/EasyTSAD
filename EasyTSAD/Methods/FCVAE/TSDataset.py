import numpy as np
import torch
from torch.utils.data import Dataset
from ...DataFactory import TSData
from typing import Tuple, Sequence, Dict

def missing_data_injection(x, y, z, rate):
    miss_size = int(rate * x.shape[0] * x.shape[1] * x.shape[2])
    row = torch.randint(low=0, high=x.shape[0], size=(miss_size,))
    col = torch.randint(low=0, high=x.shape[2], size=(miss_size,))
    x[row, :, col] = 0
    z[row, col] = 1
    return x, y, z

def point_ano(x, y, z, rate):
    aug_size = int(rate * x.shape[0])
    id_x = torch.randint(low=0, high=x.shape[0], size=(aug_size,))
    x_aug = x[id_x].clone()
    y_aug = y[id_x].clone()
    z_aug = z[id_x].clone()
    if x_aug.shape[1] == 1:
        ano_noise1 = torch.randint(low=1, high=20, size=(int(aug_size / 2),))
        ano_noise2 = torch.randint(
            low=-20, high=-1, size=(aug_size - int(aug_size / 2),)
        )
        ano_noise = (torch.cat((ano_noise1, ano_noise2), dim=0) / 2).to("cuda")
        x_aug[:, 0, -1] += ano_noise
        y_aug[:, -1] = torch.logical_or(y_aug[:, -1], torch.ones_like(y_aug[:, -1]))
    return x_aug, y_aug, z_aug


def seg_ano(x, y, z, rate, method):
    aug_size = int(rate * x.shape[0])
    idx_1 = torch.arange(aug_size)
    idx_2 = torch.arange(aug_size)
    while torch.any(idx_1 == idx_2):
        idx_1 = torch.randint(low=0, high=x.shape[0], size=(aug_size,))
        idx_2 = torch.randint(low=0, high=x.shape[0], size=(aug_size,))
    x_aug = x[idx_1].clone()
    y_aug = y[idx_1].clone()
    z_aug = z[idx_1].clone()
    time_start = torch.randint(low=7, high=x.shape[2], size=(aug_size,))  # seg start
    for i in range(len(idx_2)):
        if method == "swap":
            x_aug[i, :, time_start[i] :] = x[idx_2[i], :, time_start[i] :]
            y_aug[:, time_start[i] :] = torch.logical_or(
                y_aug[:, time_start[i] :], torch.ones_like(y_aug[:, time_start[i] :])
            )
    return x_aug, y_aug, z_aug

class  OneByOneDataset(torch.utils.data.Dataset):
    '''
    The Dateset for one by one training and testing. 
    '''
    def __init__(self, tsData: TSData, phase:str, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        
        if phase == "train":
            self.len, = tsData.train.shape
            self.sample_num = max(self.len - self.window_size + 1, 0)
            X = torch.zeros((self.sample_num, 1, self.window_size))
            Y = torch.zeros((self.sample_num, self.window_size))
            Z = torch.zeros((self.sample_num, self.window_size))
            
            for i in range(self.sample_num):
                X[i, 0, :] = torch.from_numpy(tsData.train[i : i + self.window_size])
                Y[i, :] = torch.from_numpy(tsData.train_label[i : i + self.window_size])
                
                
        elif phase == "valid":
            self.len, = tsData.valid.shape
            self.sample_num = max(self.len - self.window_size + 1, 0)
            X = torch.zeros((self.sample_num, 1, self.window_size))
            Y = torch.zeros((self.sample_num, self.window_size))
            Z = torch.zeros((self.sample_num, self.window_size))
            
            for i in range(self.sample_num):
                X[i, 0, :] = torch.from_numpy(tsData.valid[i : i + self.window_size])
                Y[i, :] = torch.from_numpy(tsData.valid_label[i : i + self.window_size])
                
        elif phase == "test":
            self.len, = tsData.test.shape
            self.sample_num = max(self.len - self.window_size + 1, 0)
            X = torch.zeros((self.sample_num, 1, self.window_size))
            Y = torch.zeros((self.sample_num, self.window_size))
            Z = torch.zeros((self.sample_num, self.window_size))
            
            for i in range(self.sample_num):
                X[i, 0, :] = torch.from_numpy(tsData.test[i : i + self.window_size])
                Y[i, :] = torch.from_numpy(tsData.test_label[i : i + self.window_size])
                
        else:
            raise ValueError('Arg "phase" in OneByOneDataset() must be one of "train", "valid", "test"')
            
        self.samples, self.targets, self.missing = X, Y, Z
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, index):
        return [self.samples[index, :, :], self.targets[index, :], self.missing[index, :]]
    
    
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
            
            X = torch.zeros((self.sample_num, 1, self.window_size))
            Y = torch.zeros((self.sample_num, self.window_size))
            Z = torch.zeros((self.sample_num, self.window_size))
            
            cnt = 0
            for _, i in datas.items():
                ll, = i.train.shape
                ll = max(ll - self.window_size + 1, 0)
                for j in range(ll):
                    X[cnt, 0, :] = torch.from_numpy(i.train[j: j + self.window_size])
                    Y[cnt, :] = torch.from_numpy(i.train_label[j : j + self.window_size])
                    cnt += 1
        
        elif phase == "valid":
            for _, i in datas.items():
                ll, = i.valid.shape
                self.sample_num += max(ll - self.window_size + 1, 0)
            
            X = torch.zeros((self.sample_num, 1, self.window_size))
            Y = torch.zeros((self.sample_num, self.window_size))
            Z = torch.zeros((self.sample_num, self.window_size))
            
            cnt = 0
            for _, i in datas.items():
                ll, = i.valid.shape
                ll = max(ll - self.window_size + 1, 0)
                for j in range(ll):
                    X[cnt, 0, :] = torch.from_numpy(i.valid[j: j + self.window_size])
                    Y[cnt, :] = torch.from_numpy(i.valid_label[j : j + self.window_size])
                    cnt += 1
        
        else:
            raise ValueError('Arg "phase" in AllInOneDataset() must be one of "train", "valid"')
        
        assert cnt == self.sample_num
        
        self.samples, self.targets, self.missing = X, Y, Z
    
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, index):
        return [self.samples[index, :, :], self.targets[index, :], self.missing[index, :]]