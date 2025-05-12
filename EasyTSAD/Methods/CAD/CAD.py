import os
from random import shuffle
from turtle import forward
from multiprocessing import cpu_count
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary, Callback
from pytorch_lightning.loggers import TensorBoardLogger

from collections import OrderedDict
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory import MTSData

import warnings
warnings.filterwarnings("ignore")


class MTSDataset(torch.utils.data.Dataset):
    
    def __init__(self, tsData: MTSData, set_type:str, window: int, horize: int):

        assert type(set_type) == type('str')
        self.set_type = set_type
        self.window = window
        self.horize = horize        
        
        if set_type == "train":
            rawdata = tsData.train
        elif set_type == "test":
            rawdata = tsData.test
        else:
            raise ValueError('Arg "set_type" in MTSDataset() must be one of "train", "test"')

        self.len, self.var_num = rawdata.shape
        self.sample_num = max(self.len - self.window - self.horize + 1, 0)
        self.samples, self.labels = self.__getsamples(rawdata)

    def __getsamples(self, data):
        X = torch.zeros((self.sample_num, self.window, self.var_num))
        Y = torch.zeros((self.sample_num, 1, self.var_num))

        for i in range(self.sample_num):
            start = i
            end = i + self.window
            X[i, :, :] = torch.from_numpy(data[start:end, :])
            Y[i, :, :] = torch.from_numpy(data[end+self.horize-1, :])

        return (X, Y)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :, :]]
        return sample

class Expert(nn.Module):
    def __init__(self, n_kernel, window, n_multiv, hidden_size, output_size, drop_out):
        super(Expert, self).__init__()
        self.conv = nn.Conv2d(1, n_kernel, (window, 1))
        self.dropout = nn.Dropout(drop_out)
        self.fc1 = nn.Linear(n_kernel * n_multiv, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(dim=1).contiguous()
        x = F.relu(self.conv(x))
        x = self.dropout(x)
        
        out = torch.flatten(x, start_dim=1).contiguous()
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, drop_out):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(drop_out)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class MMoE(pl.LightningModule):
    def __init__(self, config, seed=None):
        super(MMoE, self).__init__()
        self.hp = config 
        self.seed = config['seed']
        self.n_multiv = config['n_multiv']
        self.n_kernel = config['n_kernel']
        self.window = config['window']
        self.num_experts = config['num_experts']
        self.experts_out = config['experts_out']
        self.experts_hidden = config['experts_hidden']
        self.towers_hidden = config['towers_hidden']

        # task num = n_multiv
        self.tasks = config['n_multiv']
        self.criterion = config['criterion']
        self.exp_dropout = config['exp_dropout']
        self.tow_dropout = config['tow_dropout']
        self.conv_dropout = config['conv_dropout']
        self.lr = config['lr']

        self.softmax = nn.Softmax(dim=1)
        
        self.experts = nn.ModuleList([Expert(self.n_kernel, self.window, self.n_multiv, self.experts_hidden, self.experts_out, self.exp_dropout) \
            for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(self.window, self.num_experts), requires_grad=True) \
            for i in range(self.tasks)])
        self.share_gate = nn.Parameter(torch.randn(self.window, self.num_experts), requires_grad=True)
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden, self.tow_dropout) \
            for i in range(self.tasks)])

            
    def forward(self, x):
        experts_out = [e(x) for e in self.experts]
        experts_out_tensor = torch.stack(experts_out)
        
        gates_out = [self.softmax((x[:,:,i] @ self.w_gates[i]) * (1 - self.hp['sg_ratio']) + (x[:,:,i] @ self.share_gate) * self.hp['sg_ratio']) for i in range(self.tasks)]
        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_out_tensor for g in gates_out]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]
        
        tower_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        tower_output = torch.stack(tower_output, dim=0).permute(1,2,0)
        
        final_output = tower_output
        return final_output

    def loss(self, labels, predictions):
        if self.criterion == "l1":
            loss = F.l1_loss(predictions, labels)
        elif self.criterion == "l2":
            loss = F.mse_loss(predictions, labels)
        return loss
    
    def training_step(self, data_batch, batch_i):
        x, y = data_batch
        
        y_hat_ = self.forward(x)   
        
        loss_val = self.loss(y, y_hat_)
        self.log("val_loss", loss_val)
        output = OrderedDict({
            'loss': loss_val,
            'y' :y,
            'y_hat':y_hat_
        })
        return output
        
    def validation_step(self, data_batch, batch_i):
        x, y = data_batch
        
        y_hat_ = self.forward(x)

        loss_val = self.loss(y, y_hat_)
        
        self.log("val_loss", loss_val, on_step=False, on_epoch=True)
        output = OrderedDict({
            'val_loss': loss_val,
            'y' :y,
            'y_hat':y_hat_
        })
        return output

    def test_step(self, data_batch, batch_i):
        x, y = data_batch
        
        y_hat_ = self.forward(x)
        
        
        loss_val = self.loss(y, y_hat_)
        output = OrderedDict({
            'val_loss': loss_val,
            'y' :y,
            'y_hat':y_hat_
        })
        return output
    
    def cal_loss(self, y, y_hat):
        output = torch.sub(y, y_hat)
        output = torch.abs(output)
        if self.criterion == "l2":
            output = output.pow(2)
        
        mean_output = torch.mean(output, dim=1)
        max_output, _ = torch.max(output, dim=1)
        return mean_output, max_output
    
    def validation_step_end(self, outputs):
        y = outputs['y'].squeeze(1)
        y_hat = outputs['y_hat'].squeeze(1)
        loss_val, loss_max = self.cal_loss(y, y_hat)
        return [y, y_hat, loss_val]
    
    def validation_epoch_end(self, outputs):
        print("==============validation epoch end===============")
        y = torch.cat(([output[0] for output in outputs]),0)  
        y_hat = torch.cat(([output[1] for output in outputs]),0)  
        val_loss = torch.cat(([output[2] for output in outputs]), 0)
        np.set_printoptions(suppress=True)
            
    def test_step_end(self, outputs):
        y = outputs['y'].squeeze(1)
        y_hat = outputs['y_hat'].squeeze(1)
        loss_val, loss_max = self.cal_loss(y, y_hat)
        return [y, y_hat, loss_val, loss_max]
    
    def test_epoch_end(self, outputs):
        print("==============test epoch end===============")
        y = torch.cat(([output[0] for output in outputs]),0)  
        y_hat = torch.cat(([output[1] for output in outputs]),0)  
        val_loss = torch.cat(([output[2] for output in outputs]), 0)
        val_max = torch.cat(([output[3] for output in outputs]), 0)
        np.set_printoptions(suppress=True)
        
        if self.on_gpu:
            y = y.cpu()
            y_hat = y_hat.cpu()
            val_loss = val_loss.cpu()
            val_max = val_max.cpu()
        
        self.__anomaly_score = np.array(val_loss)
    
    def get_anomaly_score(self):
        return self.__anomaly_score
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
    
    def mydataloader(self, train, tsData=None, batch_s=0):
        set_type = train
        print(set_type + "data loader called...")
        train_sampler = None
        batch_size = self.hp['batch_size']

        if batch_s == 0:
            batch_size = self.hp['batch_size']
        else:
            batch_size = batch_s
        
        dataset = MTSDataset(tsData=tsData, set_type=set_type, \
            window=self.window, horize=self.hp['horize'])
        
        try:
            if self.on_gpu:
                train_sampler = DistributedSampler(dataset, rank=self.trainer.local_rank)
                batch_size = batch_size // self.trainer.world_size
        except Exception as e:
            print(e)
            print("=============GPU Setting ERROR================")
            
        if set_type == "train":
            shuffle_ = True
        else:
            shuffle_ = False
        
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle_,
            sampler=train_sampler,
            persistent_workers=False
        )
        
        return loader

class CAD(BaseMethod):
    def __init__(self, params:dict=None) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.cuda = True
        if self.cuda == True and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("=== Using CUDA ===")
        else:
            if self.cuda == True and not torch.cuda.is_available():
                print("=== CUDA is unavailable ===")
            self.device = torch.device("cpu")
            print("=== Using CPU ===")

        self.config = {
            'seed': 2023,
            'n_multiv': 51,
            'horize': 1,
            'window': 32,
            'batch_size': 128,

            'num_experts': 9,
            'n_kernel': 16,
            'experts_out': 128,
            'experts_hidden': 256,
            'towers_hidden': 32,
            'criterion': 'l2',
            'exp_dropout': 0.2,
            'tow_dropout': 0.1,
            'conv_dropout': 0.1,
            'sg_ratio': 0.7,
            'lr': 0.001
        }

        self.seed = self.config['seed']
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.logger = TensorBoardLogger(name="logs", save_dir="./")
        print("gpu is available: ", torch.cuda.is_available())
        if torch.cuda.is_available(): self.dev = "gpu"
        else: self.dev = "cpu"
        
        self.model = None
    

    def train_valid_phase(self, tsData: MTSData):
        print("Loading Model...")
        self.model = MMoE(self.config, self.seed)
        print("Model Built...")

        early_stop = EarlyStopping(
            monitor='val_loss', patience=5, verbose=True, mode='min'
        )
        
        cpkt_callback = ModelCheckpoint(
            monitor='val_loss', save_top_k=1, mode='min'
        )
        
        callback = [cpkt_callback, early_stop]
        self.trainer = Trainer(max_epochs=int(10) , callbacks=callback, logger=self.logger,\
            devices=1, accelerator=self.dev
        )
        
        self.trainer.fit(self.model, train_dataloaders=self.model.mydataloader(train='train', tsData=tsData), val_dataloaders=self.model.mydataloader(train="train", tsData=tsData))
        print("=========Train over============")


    def test_phase(self, tsData: MTSData):
        self.test_result = self.trainer.test(self.model, dataloaders=self.model.mydataloader(train="test", tsData=tsData))
        scores = self.model.get_anomaly_score()
        print(scores.shape)
        assert scores.ndim == 1
        self.__anomaly_score = scores

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def param_statistic(self, save_file):
        with open(save_file, 'w') as f:
            f.write('Over')