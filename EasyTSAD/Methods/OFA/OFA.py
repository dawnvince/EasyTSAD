import argparse
from typing import Dict
import numpy as np
import torchinfo
from ...DataFactory import TSData
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import tqdm
import os

from ...Exptools import EarlyStoppingTorch
from .. import BaseMethod
from .TSDataset import AllInOneDataset, OneByOneDataset
from .GPT4TS import Model

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class OFA(BaseMethod):
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
            
        self.args = argparse.Namespace(**params)
        self.model = Model(self.args).float().to(self.device)
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.early_stopping = EarlyStoppingTorch(None, patience=self.args.patience)
        self.input_shape = (self.args.batch_size, self.args.seq_len, self.args.enc_in)
        
    def train_valid_phase(self, tsTrain: TSData):
        train_loader = DataLoader(
            dataset=OneByOneDataset(tsData=tsTrain, phase="train", window_size=self.args.seq_len),
            batch_size=self.args.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=OneByOneDataset(tsData=tsTrain, phase="valid", window_size=self.args.seq_len),
            batch_size=self.args.batch_size,
            shuffle=False
        )
        
        train_steps = len(train_loader)
        for epoch in range(1, self.args.epochs + 1):
            ## Training
            train_loss = 0
            self.model.train()
            
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for i, batch_x in loop:
                self.model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_x)
                
                loss.backward()
                self.model_optim.step()
                
                train_loss += loss.cpu().item()
                
                loop.set_description(f'Training Epoch [{epoch}/{self.args.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=train_loss/(i+1))
            
            ## Validation
            self.model.eval()
            total_loss = []
            
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for i, batch_x in loop:
                    batch_x = batch_x.float().to(self.device)

                    outputs = self.model(batch_x)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    pred = outputs.detach().cpu()
                    true = batch_x.detach().cpu()

                    loss = self.criterion(pred, true)
                    total_loss.append(loss)
                    loop.set_description(f'Valid Epoch [{epoch}/{self.args.epochs}]')
                    
            valid_loss = np.average(total_loss)
            loop.set_postfix(loss=loss.item(), valid_loss=valid_loss)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
            
            adjust_learning_rate(self.model_optim, epoch + 1, self.args)
            
            
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        train_loader = DataLoader(
            dataset=AllInOneDataset(datas=tsTrains, phase="train", window_size=self.args.seq_len),
            batch_size=self.args.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=AllInOneDataset(datas=tsTrains, phase="valid", window_size=self.args.seq_len),
            batch_size=self.args.batch_size,
            shuffle=False
        )
        
        train_steps = len(train_loader)
        for epoch in range(1, self.args.epochs + 1):
            ## Training
            train_loss = 0
            self.model.train()
            
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for i, batch_x in loop:
                self.model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_x)
                
                loss.backward()
                self.model_optim.step()
                
                train_loss += loss.cpu().item()
                
                loop.set_description(f'Training Epoch [{epoch}/{self.args.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=train_loss/(i+1))
            
            ## Validation
            self.model.eval()
            total_loss = []
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for i, batch_x in loop:
                    batch_x = batch_x.float().to(self.device)

                    outputs = self.model(batch_x)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    pred = outputs.detach().cpu()
                    true = batch_x.detach().cpu()

                    loss = self.criterion(pred, true)
                    total_loss.append(loss)
                    loop.set_description(f'Valid Epoch [{epoch}/{self.args.epochs}]')
                    
            valid_loss = np.average(total_loss)
            loop.set_postfix(loss=loss.item(), valid_loss=valid_loss)
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
            
            adjust_learning_rate(self.model_optim, epoch + 1, self.args)
            
            
    def test_phase(self, tsData: TSData):
        test_loader = DataLoader(
            dataset=OneByOneDataset(tsData, phase="test", window_size=self.args.seq_len),
            batch_size=self.args.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        attens_energy = []
        y_hats = []
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for i, batch_x in loop:
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                y_hat = torch.squeeze(outputs, -1)
                
                score = score.detach().cpu().numpy()[:, -1]
                y_hat = y_hat.detach().cpu().numpy()[:, -1]
                
                attens_energy.append(score)
                y_hats.append(y_hat)
                loop.set_description(f'Testing Phase: ')

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        scores = np.array(attens_energy)
        
        y_hats = np.concatenate(y_hats, axis=0).reshape(-1)
        y_hats = np.array(y_hats)

        assert scores.ndim == 1
        assert y_hats.shape == scores.shape
        
        import shutil
        self.save_path = None
        if self.save_path and os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
            
        self.__anomaly_score = scores
        self.y_hats = y_hats
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        return self.y_hats
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, self.input_shape, verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))