import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
import torchinfo
import tqdm
from ...DataFactory import TSData
from ...Exptools import EarlyStoppingTorch
from .. import BaseMethod
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from ...DataFactory.TorchDataSet.ReconstructWindow import UTSAllInOneDataset, UTSOneByOneDataset

class Model(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, input_len, out_len, cut_freq):
        super(Model, self).__init__()
        
        self.input_len = input_len
        self.out_len = out_len

        # Decompsition Kernel Size
        self.dominance_freq=cut_freq # 720/24
        self.length_ratio = self.out_len/self.input_len

        self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat) # complex layer for frequency upcampling]


    def forward(self, x):
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:,self.dominance_freq:]=0
        # low_x=torch.fft.irfft(low_specx, dim=1)
        low_specx = low_specx[:,0:self.dominance_freq]

        low_specxy_ = self.freq_upsampler(low_specx)
        # print(low_specxy_)
        low_specxy = torch.zeros([low_specxy_.size(0),int((self.out_len)/2+1)],dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:,0:low_specxy_.size(1)]=low_specxy_
        low_xy=torch.fft.irfft(low_specxy, dim=1)
        low_xy=low_xy * self.length_ratio # compemsate the length change

        # xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN
        xy=(low_xy) * torch.sqrt(x_var) +x_mean
        # return xy, low_xy* torch.sqrt(x_var)
        return xy
    
class FITS(BaseMethod):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.__anomaly_score = None
        
        self.cuda = True
        if self.cuda == True and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("=== Using CUDA ===")
        else:
            if self.cuda == True and not torch.cuda.is_available():
                print("=== CUDA is unavailable ===")
            self.device = torch.device("cpu")
            print("=== Using CPU ===")
            
        self.input_len = params["input_len"]
        self.out_len = params["out_len"]
        
        self.step_size = self.out_len // self.input_len
        assert self.step_size * self.input_len == self.out_len
        
        self.batch_size = params["batch_size"]
        self.model = Model(self.input_len, self.out_len, params["cut_freq"]).to(self.device)
        self.epochs = params["epochs"]
        learning_rate = params["lr"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)
    
    def train_valid_phase(self, tsTrain: TSData):
        
        train_loader = DataLoader(
            dataset=UTSOneByOneDataset(tsTrain, "train", window_size=self.out_len),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=UTSOneByOneDataset(tsTrain, "valid", window_size=self.out_len),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (x, target) in loop:
                x = x[:, ::self.step_size]
                x, target = x.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                
                output = self.model(x)
                loss = self.loss(output, target)
                loss.backward()

                self.optimizer.step()
                
                avg_loss += loss.cpu().item()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            
            self.model.eval()
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x = x[:, ::self.step_size]
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            valid_loss = avg_loss/max(len(valid_loader), 1)
            self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
            
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        train_loader = DataLoader(
            dataset=UTSAllInOneDataset(tsTrains, "train", window_size=self.out_len),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=UTSAllInOneDataset(tsTrains, "valid", window_size=self.out_len),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (x, target) in loop:
                x = x[:, ::self.step_size]
                x, target = x.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                
                output = self.model(x)
                loss = self.loss(output, target)
                loss.backward()

                self.optimizer.step()
                
                avg_loss += loss.cpu().item()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            
            self.model.eval()
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x = x[:, ::self.step_size]
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            valid_loss = avg_loss/max(len(valid_loader), 1)
            self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
        
    def test_phase(self, tsData: TSData):
        test_loader = DataLoader(
            dataset=UTSOneByOneDataset(tsData, "test", window_size=self.out_len),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        scores = []
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:
                x = x[:, ::self.step_size]
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                # loss = self.loss(output, target)
                mse = torch.sub(output, target).pow(2)
                scores.append(mse.cpu()[:,-1])
                loop.set_description(f'Testing: ')

        scores = torch.cat(scores, dim=0)
        scores = scores.numpy().flatten()

        assert scores.ndim == 1
        self.__anomaly_score = scores
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.input_len), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))