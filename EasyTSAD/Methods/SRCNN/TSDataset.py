from .Model import average_filter, spectral_residual
import torch
import torch.utils.data
import numpy as np
from ...DataFactory import TSData
from typing import Tuple, Sequence, Dict

class gen():
    def __init__(self, win_siz, step, nums):
        self.control = 0
        self.win_siz = win_siz
        self.step = step
        self.number = nums

    def generate_train_data(self, value, back_k=0):
        if back_k <= 5:
            back = back_k
        else:
            back = 5
        length = len(value)
        tmp = []
        for pt in range(self.win_siz, length - back, self.step):
            head = max(0, pt - self.win_siz)
            tail = min(length - back, pt)
            data = np.array(value[head:tail])
            data = data.astype(np.float64)
            num = np.random.randint(1, self.number)
            ids = np.random.choice(self.win_siz, num, replace=False)
            lbs = np.zeros(self.win_siz, dtype=np.int64)
            if (self.win_siz - 6) not in ids:
                self.control += np.random.random()
            else:
                self.control = 0
            if self.control > 100:
                ids[0] = self.win_siz - 6
                self.control = 0
            mean = np.mean(data)
            dataavg = average_filter(data)
            var = np.var(data)
            for id in ids:
                data[id] += (dataavg[id] + mean) * np.random.randn() * min((1 + var), 10)
                lbs[id] = 1
            tmp.append([data.tolist(), lbs.tolist()])
        return tmp
    
    
class  OneByOneTrainDataset(torch.utils.data.Dataset):
    '''
    The Dateset for one by one training and testing. 
    '''
    def __init__(self, tsData: TSData, window: int, step, num) -> None:
        super().__init__()
        self.genlen = 0
        self.len = self.genlen
        self.width = window
        
        generator = gen(window, step, num)
        self.kpinegraw = generator.generate_train_data(tsData.train)
        self.negrawlen = len(self.kpinegraw)
        print('length :', len(self.kpinegraw))
        
        self.len += self.negrawlen
        self.kpineglen = 0
        self.control = 0.
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        idx = index % self.negrawlen
        datas = self.kpinegraw[idx]
        datas = np.array(datas)
        data = datas[0, :].astype(np.float64)
        lbs = datas[1, :].astype(np.float64)
        wave = spectral_residual(data)
        waveavg = average_filter(wave)
        for i in range(self.width):
            if wave[i] < 0.001 and waveavg[i] < 0.001:
                lbs[i] = 0
                continue
            ratio = wave[i] / waveavg[i]
            if ratio < 1.0 and lbs[i] == 1:
                lbs[i] = 0
            if ratio > 5.0:
                lbs[i] = 1
        srscore = abs(wave - waveavg) / (waveavg + 0.01)
        sortid = np.argsort(srscore)
        for idx in sortid[-2:]:
            if srscore[idx] > 5:
                lbs[idx] = 1
        resdata = torch.from_numpy(100 * wave)
        reslb = torch.from_numpy(lbs)
        return resdata, reslb
    
    
class  AllInOneTrainDataset(torch.utils.data.Dataset):
    '''
    The Dateset for one by one training and testing. 
    '''
    def __init__(self, datas: Dict[str, TSData], window: int, step, num) -> None:
        super().__init__()
        self.genlen = 0
        self.len = self.genlen
        self.width = window
        
        self.kpinegraw = []
        generator = gen(window, step, num)
        for _, curve in datas.items(): 
            self.kpinegraw += generator.generate_train_data(curve.train)
        
        self.len += len(self.kpinegraw)
        self.kpineglen = 0
        self.control = 0.
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        idx = index % self.len
        datas = self.kpinegraw[idx]
        datas = np.array(datas)
        data = datas[0, :].astype(np.float64)
        lbs = datas[1, :].astype(np.float64)
        wave = spectral_residual(data)
        waveavg = average_filter(wave)
        for i in range(self.width):
            if wave[i] < 0.001 and waveavg[i] < 0.001:
                lbs[i] = 0
                continue
            ratio = wave[i] / waveavg[i]
            if ratio < 1.0 and lbs[i] == 1:
                lbs[i] = 0
            if ratio > 5.0:
                lbs[i] = 1
        srscore = abs(wave - waveavg) / (waveavg + 0.01)
        sortid = np.argsort(srscore)
        for idx in sortid[-2:]:
            if srscore[idx] > 5:
                lbs[idx] = 1
        resdata = torch.from_numpy(100 * wave)
        reslb = torch.from_numpy(lbs)
        return resdata, reslb