import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from einops import rearrange
from EasyTSAD.DataFactory import TSData
from .Model import DCdetectorModel
import warnings
import torchinfo
from ...DataFactory import TSData
from .. import BaseMethod
from .TSDataset import OneByOneDataset, AllInOneDataset
warnings.filterwarnings('ignore')

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), os.path.join(path, "DCdetector_checkpoint.pth"))
        self.val_loss_min = val_loss
        
class DCdetector(BaseMethod):
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
            
        self.win_size = params["win_size"]
        self.input_c = params["input_c"]
        self.output_c = params["output_c"]
        self.n_heads = params["n_heads"]
        self.d_model = params["d_model"]
        self.e_layers = params["e_layers"]
        self.patch_size = params["patch_size"]
        self.patience = params["earlystop_patience"]
        self.lr = params["learning_rate"]
        self.step = params["step"]
        self.num_epochs = params["num_epochs"]
        
        self.batch_size = params["batch_size"]
        
        self.model = DCdetectorModel(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads, d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size, channel=self.input_c)
        
        self.input_shape = (self.batch_size, self.win_size, self.input_c)
        
        if torch.cuda.is_available():
            self.model.cuda()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.save_path = "Model_ckpt"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            loss_1.append((prior_loss - series_loss).item())

        return np.average(loss_1)
    
    def train_valid_phase(self, tsTrain: TSData):
        print("======================TRAIN MODE======================")
        time_now = time.time()
        
        train_loader = torch.utils.data.DataLoader(
            dataset=OneByOneDataset(tsTrain, window_size=self.win_size, phase="train", step=self.step),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = torch.utils.data.DataLoader(
            dataset=OneByOneDataset(tsTrain, window_size=self.win_size, phase="valid", step=self.step),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        train_steps = len(train_loader)
        
        path = self.save_path
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, _) in enumerate(train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                series, prior = self.model(input)
                
                series_loss = 0.0
                prior_loss = 0.0

                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss = prior_loss - series_loss 

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
 
                loss.backward()
                self.optimizer.step()

            vali_loss1 = self.vali(valid_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s , vali_loss1: {2}".format(
                    epoch + 1, time.time() - epoch_time, vali_loss1))
            early_stopping(vali_loss1, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)   
            
    def train_valid_phase_all_in_one(self, tsTrains):
        train_loader = torch.utils.data.DataLoader(
            dataset=AllInOneDataset(tsTrains, phase="train", window_size=self.win_size, step=self.step),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = torch.utils.data.DataLoader(
            dataset=AllInOneDataset(tsTrains, phase="valid", window_size=self.win_size, step=self.step),
            batch_size=self.batch_size,
            shuffle=False
        )
        time_now = time.time()
        train_steps = len(train_loader)
        
        path = self.save_path
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, _) in enumerate(train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                series, prior = self.model(input)
                
                series_loss = 0.0
                prior_loss = 0.0

                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss = prior_loss - series_loss 

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
 
                loss.backward()
                self.optimizer.step()

            vali_loss1 = self.vali(valid_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s , vali_loss1: {2}".format(
                    epoch + 1, time.time() - epoch_time, vali_loss1))
            early_stopping(vali_loss1, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
            
    def test_phase(self, tsData: TSData):
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.save_path, "DCdetector_checkpoint.pth")))
        self.model.eval()
        temperature = 50
        attens_energy = []
        
        test_loader = torch.utils.data.DataLoader(
            dataset=OneByOneDataset(tsData, window_size=self.win_size, phase="thre", step=self.step),
            batch_size=self.batch_size,
            shuffle=False
        )

        print("======================TEST MODE======================")
        for i, (input_data, _) in enumerate(test_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        
        print("++++++++  ", test_energy.shape)
        assert test_energy.ndim == 1
        self.__anomaly_score = test_energy
        

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, self.input_shape, verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))
        
        