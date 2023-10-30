from typing import Dict
import numpy as np
import torchinfo
from ...DataFactory import TSData
import torch
from torch import nn, optim
import tqdm
import os

from ...Exptools import EarlyStoppingTorch

from .. import BaseMethod
from .TSDataset import OneByOneDataset, AllInOneDataset
from .Model import DonutModel, MaskedVAELoss


class Donut(BaseMethod):
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
        
        self.window_size = params["window_size"]
        self.batch_size = params["batch_size"]        
        self.grad_clip = params["grad_clip"]
        self.num_epochs = params["num_epochs"]
        self.mc_samples = params["mc_samples"]
        
        input_dim = self.window_size
        hidden_dim = params["hidden_dim"]
        z_dim = params["z_dim"]
        inject_ratio = params["inject_ratio"]
        learning_rate = params["learning_rate"]
        l2_coff = params["l2_coff"]
        patience = params["earlystop_patience"]
        
        self.model = DonutModel(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim, mask_prob=inject_ratio).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=l2_coff)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.75)
        self.vaeloss = MaskedVAELoss()
        
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=patience)
        
    def train(self, train_loader, epoch):
        self.model.train(mode=True)
        avg_loss = 0
        loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
        for idx, (x, target) in loop:
            x, target = x.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            output = self.model(x)
            loss = self.vaeloss(output, (target,))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            
            avg_loss += loss.cpu().item()
            loop.set_description(f'Training Epoch [{epoch}/{self.num_epochs}]')
            loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                
    def valid(self, valid_loader, epoch):
        self.model.eval()
        avg_loss = 0
        loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                loss = self.vaeloss(output, (target,))
                avg_loss += loss.cpu().item()
                loop.set_description(f'Validation Epoch [{epoch}/{self.num_epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                
        return avg_loss/max(len(valid_loader), 1)
        
    def train_valid_phase(self, tsTrain: TSData):    
        train_loader = torch.utils.data.DataLoader(
            dataset=OneByOneDataset(tsTrain.train, window_size=self.window_size),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = torch.utils.data.DataLoader(
            dataset=OneByOneDataset(tsTrain.valid, window_size=self.window_size),
            batch_size=self.batch_size,
            shuffle=False
        )
                    
        for epoch in range(1, self.num_epochs + 1):
            self.train(train_loader, epoch)
            valid_loss = self.valid(valid_loader, epoch)
            self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
            
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):    
        train_loader = torch.utils.data.DataLoader(
            dataset=AllInOneDataset(tsTrains, phase="train", window_size=self.window_size),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = torch.utils.data.DataLoader(
            dataset=AllInOneDataset(tsTrains, phase="valid", window_size=self.window_size),
            batch_size=self.batch_size,
            shuffle=False
        )
                    
        for epoch in range(1, self.num_epochs + 1):
            self.train(train_loader, epoch)
            valid_loss = self.valid(valid_loader, epoch)
            self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
        
        
    def test_phase(self, tsData: TSData):
        
        test_loader = torch.utils.data.DataLoader(
            dataset=OneByOneDataset(tsData.test, window_size=self.window_size),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        scores = []
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for idx, (x, _) in loop:
                x = x.to(self.device)
                output = self.model(x, num_samples=self.mc_samples)
                scores.append(output.cpu())
                loop.set_description(f'Testing: ')

        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()
        
        assert scores.ndim == 1
        
        import shutil
        if self.save_path and os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
            
        self.__anomaly_score = scores
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        return super().get_y_hat
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.window_size), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))
    