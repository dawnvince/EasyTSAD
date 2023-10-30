from typing import Dict
import tqdm
from ...DataFactory import TSData
from ...Exptools import EarlyStoppingTorch
from .. import BaseMethod
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchinfo import summary

from ...DataFactory.TorchDataSet import ReconstructWindow

class AEModel(nn.Module):
    def __init__(self, p, lat_dim_1, lat_dim_2) -> None:
        super().__init__()
        self.p = p
        self.enc = nn.Sequential(
            nn.Linear(p, lat_dim_1),
            nn.LeakyReLU(),
            nn.Linear(lat_dim_1, lat_dim_2),
            nn.LeakyReLU()
        )
        
        self.dec = nn.Sequential(
            nn.Linear(lat_dim_2, lat_dim_1),
            nn.LeakyReLU(),
            nn.Linear(lat_dim_1, p)
        )
        
    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x

class AE(BaseMethod):
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
            
        self.p = params["p"]
        self.lat_dim_1 = params["lat_dim_1"]
        self.lat_dim_2 = params["lat_dim_2"]
        
        self.batch_size = params["batch_size"]
        self.model = AEModel(self.p, self.lat_dim_1, self.lat_dim_2).to(self.device)
        self.epochs = params["epochs"]
        learning_rate = params["lr"]
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-3)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)
    
    def train_valid_phase(self, tsTrain: TSData):
        
        train_loader = DataLoader(
            dataset=ReconstructWindow.UTSOneByOneDataset(tsTrain, "train", window_size=self.p),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=ReconstructWindow.UTSOneByOneDataset(tsTrain, "valid", window_size=self.p),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (x, target) in loop:
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
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            valid_loss = avg_loss/max(len(valid_loader), 1)
            # self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
            
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        train_loader = DataLoader(
            dataset=ReconstructWindow.UTSAllInOneDataset(tsTrains, "train", window_size=self.p),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=ReconstructWindow.UTSAllInOneDataset(tsTrains, "valid", window_size=self.p),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (x, target) in loop:
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
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            valid_loss = avg_loss/max(len(valid_loader), 1)
            # self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
        
    def test_phase(self, tsData: TSData):
        test_loader = DataLoader(
            dataset=ReconstructWindow.UTSOneByOneDataset(tsData, "test", window_size=self.p),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        scores = []
        y_hat = []
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                # loss = self.loss(output, target)
                y_hat.append(output[:, -1])
                mse = torch.sub(output[:, -1], target[:, -1]).pow(2)
                scores.append(mse)
                loop.set_description(f'Testing: ')

        scores = torch.cat(scores, dim=0)
        scores = scores.cpu().numpy().flatten()
        
        y_hat = torch.cat(y_hat, dim=0)
        y_hat = y_hat.cpu().numpy().flatten()

        assert scores.ndim == 1
        self.__anomaly_score = scores
        self.y_hat = y_hat
    
    def get_y_hat(self) -> np.ndarray:
        return self.y_hat
    
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def param_statistic(self, save_file):
        model_stats = summary(self.model, (self.batch_size, self.p), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))