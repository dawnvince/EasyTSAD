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

from .TSDataset import UTSAllInOneDataset, UTSOneByOneDataset

SOS_token = 0

class LSTMModel(nn.Module):
    def __init__(self, window_size, input_size, 
                 hidden_dim, pred_len, num_layers, batch_size, device) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.input_size = input_size
        self.device = device
        
        self.lstm_encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        self.relu = nn.GELU()
        self.fc = nn.Linear(hidden_dim, pred_len)
        
    def forward(self, src):
        src = torch.unsqueeze(src, -1)
        outputs, _ = self.lstm_encoder(src)
        outputs = self.relu(outputs[:, -1, :])
        outputs = self.fc(outputs)
            
        return outputs
    
class LSTMADbeta(BaseMethod):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.__anomaly_score = None
        
        cuda = True
        self.y_hats = None
        
        self.cuda = cuda
        if self.cuda == True and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("=== Using CUDA ===")
        else:
            if self.cuda == True and not torch.cuda.is_available():
                print("=== CUDA is unavailable ===")
            self.device = torch.device("cpu")
            print("=== Using CPU ===")
        
        self.window_size = params["window_size"]
        self.pred_len = params["pred_len"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        
        input_size = params["input_size"]
        hidden_dim = params["hidden_dim"]
        num_layer = params["num_layer"]
        lr = params["lr"]
        
        self.model = LSTMModel(self.window_size, input_size, hidden_dim, self.pred_len, num_layer, batch_size=self.batch_size, device=self.device).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)
        
        self.mu = None
        self.sigma = None
        self.eps = 1e-10
        
    def train_valid_phase(self, tsTrain: TSData):
        train_loader = DataLoader(
            UTSOneByOneDataset(tsTrain, phase="train", 
                               window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            UTSOneByOneDataset(tsTrain, phase="valid", 
                               window_size=self.window_size, pred_len=self.pred_len),
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
            scores = []
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    
                    output = self.model(x)
                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                    
                    mse = torch.sub(output, target).pow(2)
                    scores.append(mse.cpu())
                    
            
            valid_loss = avg_loss/max(len(valid_loader), 1)
            self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop or epoch == self.epochs - 1:
                # fitting Gaussian Distribution
                if len(scores) > 0:
                    scores = torch.cat(scores, dim=0)
                    self.mu = torch.mean(scores)
                    self.sigma = torch.var(scores)
                    print(self.mu.size(), self.sigma.size())
                if self.early_stopping.early_stop:
                    print("   Early stopping<<<")
                break

    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        train_loader = DataLoader(
            UTSAllInOneDataset(tsTrains, phase="train", 
                               window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            UTSAllInOneDataset(tsTrains, phase="valid", 
                               window_size=self.window_size, pred_len=self.pred_len),
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
            scores = []
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    
                    output = self.model(x)
                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                    
                    mse = torch.sub(output, target).pow(2)
                    scores.append(mse.cpu())
                    
            
            valid_loss = avg_loss/max(len(valid_loader), 1)
            self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop or epoch == self.epochs - 1:
                # fitting Gaussian Distribution
                scores = torch.cat(scores, dim=0)
                self.mu = torch.mean(scores, dim=0)
                self.sigma = torch.var(scores, dim=0)
                print(self.mu.size(), self.sigma.size())
                if self.early_stopping.early_stop:
                    print("   Early stopping<<<")
                break

    def test_phase(self, tsData: TSData):
        test_loader = DataLoader(
            UTSOneByOneDataset(tsData, phase="test", 
                               window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        scores = []
        y_hats = []
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                mse = torch.sub(output, target).pow(2)
                y_hats.append(output.cpu())
                scores.append(mse.cpu())
                loop.set_description(f'Testing: ')

        scores = torch.cat(scores, dim=0)
        # scores = 0.5 * (torch.log(self.sigma + self.eps) + (scores - self.mu)**2 / (self.sigma+self.eps))
        
        scores = scores.numpy()
        y_hats = torch.cat(y_hats, dim=0)
        y_hats = y_hats.numpy()
        
        l, w = scores.shape
        
        new_scores = np.zeros((l - self.pred_len, w))
        for i in range(w):
            new_scores[:, i] = scores[self.pred_len - i:l-i, i]
        scores = np.mean(new_scores, axis=1)
        scores = np.pad(scores, (0, self.pred_len - 1), 'constant',constant_values=(0,0))
        
        new_y_hats = np.zeros((l - self.pred_len, w))
        for i in range(w):
            new_y_hats[:, i] = y_hats[self.pred_len - i:l-i, i]
        y_hats = np.mean(new_y_hats, axis=1)
        y_hats = np.pad(y_hats, (0, self.pred_len - 1), 'constant',constant_values=(0,0))

        assert scores.ndim == 1
        self.__anomaly_score = scores
        self.y_hats = y_hats
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        return self.y_hats
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.window_size), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))