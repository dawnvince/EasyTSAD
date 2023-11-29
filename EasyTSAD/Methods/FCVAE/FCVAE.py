from typing import Dict
from torch.utils.data import DataLoader
import argparse
from torch import optim
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import tqdm

from ...DataFactory import TSData

from .Model import CVAE, FCVAEModel
from .TSDataset import *
from .. import BaseMethod
from ...Exptools import EarlyStoppingTorch

class FCVAE(BaseMethod):
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
            
        self.hp = argparse.Namespace(**params)
        self.model = FCVAEModel(self.hp).to(self.device)
        
        self.optim = optim.Adam(self.model.parameters(), lr=self.hp.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=10)
        
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(self.save_path, patience=self.hp.patience)
        
    def batch_data_augmentation(self, x, y, z):
        # missing data injection
        if self.hp.point_ano_rate > 0:
            x_a, y_a, z_a = point_ano(x, y, z, self.hp.point_ano_rate)
            x = torch.cat((x, x_a), dim=0)
            y = torch.cat((y, y_a), dim=0)
            z = torch.cat((z, z_a), dim=0)
        if self.hp.seg_ano_rate > 0:
            x_a, y_a, z_a = seg_ano(
                x, y, z, self.hp.seg_ano_rate, method="swap"
            )
            x = torch.cat((x, x_a), dim=0)
            y = torch.cat((y, y_a), dim=0)
            z = torch.cat((z, z_a), dim=0)
        x, y, z = missing_data_injection(
            x, y, z, self.hp.missing_data_rate
        )
        return x, y, z
    
    def loss(self, x, y_all, z_all, mode="train"):
        y = (y_all[:, -1]).unsqueeze(1)
        if self.hp.use_label==1:
            mask = torch.logical_not(torch.logical_or(y_all, z_all))
        else:
            mask = torch.logical_not(z_all)
        mu_x, var_x, rec_x, mu, var, loss = self.model.forward(
            x,
            "train",
            mask,
        )
        loss_val = loss
        if mode == "test":
            mu_x_test, recon_prob = self.model.forward(x, "test", z_all)
            return mu_x, var_x, recon_prob, mu_x_test
        return loss_val
    
    def train(self, x, y_all, z_all):
        x, y_all, z_all = x.to(self.device), y_all.to(self.device), z_all.to(self.device)
        y_all2 = torch.zeros_like(y_all)
        x, y_all2, z_all = self.batch_data_augmentation(x, y_all2, z_all)
        
        if self.hp.use_label==1:
            mask = torch.logical_not(torch.logical_or(y_all2, z_all))
        else:
            mask = torch.logical_not(z_all)
        _, _, _, _, _, loss = self.model.forward(
            x,
            "train",
            mask,
        )
        
        return loss
    
    def valid(self, x, y_all, z_all):
        x, y_all, z_all = x.to(self.device), y_all.to(self.device), z_all.to(self.device)
        y_all_wo_label = torch.zeros_like(y_all)
        
        if self.hp.use_label==1:
            mask = torch.logical_not(torch.logical_or(y_all_wo_label, z_all))
        else:
            mask = torch.logical_not(z_all)
        _, _, _, _, _, loss = self.model.forward(
            x,
            "train",
            mask,
        )
        
        return loss
    
    def train_valid_phase(self, tsTrain: TSData):
        train_loader = DataLoader(
            dataset=OneByOneDataset(tsTrain, "train", self.hp.window),
            batch_size=self.hp.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=OneByOneDataset(tsTrain, "valid", self.hp.window),
            batch_size=self.hp.batch_size,
            shuffle=False
        )
        
        for epoch in range(1, self.hp.epochs + 1):
            ## Training
            train_loss = 0
            self.model.train()
            
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for i, (x, y_all, z_all) in loop:
                self.optim.zero_grad()
                
                loss = self.train(x, y_all, z_all)
                loss.backward()
                self.optim.step()
                
                train_loss += loss.item()
                
                loop.set_description(f'Training Epoch [{epoch}/{self.hp.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=train_loss/(i+1))
                
            ## Validation
            self.model.eval()
            valid_loss = 0
            
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for i, (x, y_all, z_all) in loop:
                    loss = self.valid(x, y_all, z_all)
                    valid_loss += loss.item()
                    
                    loop.set_description(f'Validation Epoch [{epoch}/{self.hp.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=valid_loss/(i+1))
                    
            valid_loss = valid_loss / len(valid_loader) + 1
            
            self.scheduler.step()
        
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
            
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        train_loader = DataLoader(
            dataset=AllInOneDataset(tsTrains, "train", self.hp.window),
            batch_size=self.hp.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=AllInOneDataset(tsTrains, "valid", self.hp.window),
            batch_size=self.hp.batch_size,
            shuffle=False
        )
        
        for epoch in range(1, self.hp.epochs + 1):
            ## Training
            train_loss = 0
            self.model.train()
            
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for i, (x, y_all, z_all) in loop:
                self.optim.zero_grad()
                
                loss = self.train(x, y_all, z_all)
                loss.backward()
                self.optim.step()
                
                train_loss += loss.item()
                
                loop.set_description(f'Training Epoch [{epoch}/{self.hp.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=train_loss/(i+1))
                
            ## Validation
            self.model.eval()
            valid_loss = 0
            
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for i, (x, y_all, z_all) in loop:
                    loss = self.train(x, y_all, z_all)
                    valid_loss += loss.item()
                    
                    loop.set_description(f'Validation Epoch [{epoch}/{self.hp.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=valid_loss/(i+1))
                    
            valid_loss = valid_loss / len(valid_loader) + 1
            
            self.scheduler.step()
        
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
            
    def test_phase(self, tsData: TSData):
        test_loader = torch.utils.data.DataLoader(
            dataset=OneByOneDataset(tsData, phase="test", window_size=self.hp.window),
            batch_size=self.hp.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        scores = []
        
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for i, (x, y_all, z_all) in loop:
                x, y_all, z_all = x.to(self.device), y_all.to(self.device), z_all.to(self.device)
                
                if self.hp.use_label==1:
                    mask = torch.logical_not(torch.logical_or(y_all, z_all))
                else:
                    mask = torch.logical_not(z_all)
                mu_x, var_x, rec_x, mu, var, loss = self.model.forward(
                    x,
                    "train",
                    mask,
                )
                loss_val = loss

                mu_x_test, recon_prob = self.model.forward(x, "test", z_all)
                # mu_x, var_x, recon_prob, mu_x_test 
                recon_prob = recon_prob[:, :, -1]
                scores.append(-1 * recon_prob.squeeze(1).cpu())
                
        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()
        
        assert scores.ndim == 1
        
        import shutil
        if self.save_path and os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
            
        self.__anomaly_score = scores
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def param_statistic(self, save_file):
        return 
        model_stats = torchinfo.summary(self.model, (self.hp.batch_size, self.hp.window), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))
            