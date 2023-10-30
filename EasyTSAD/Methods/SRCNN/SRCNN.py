"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation ("Microsoft") grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

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
from .TSDataset import OneByOneTrainDataset, AllInOneTrainDataset
from .Model import *

from torch.autograd import Variable

class SRCNN(BaseMethod):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.__anomaly_score = None
        
        cuda = True
        self.y_hats = None
        
        self.cuda = cuda
        
        self.win_size = params["win_size"]
        self.lr = params["lr"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        self.step = params["step"]
        self.num = params["num"]
        
        if self.cuda == True and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("=== Using CUDA ===")
        else:
            if self.cuda == True and not torch.cuda.is_available():
                print("=== CUDA is unavailable ===")
            self.device = torch.device("cpu")
            print("=== Using CPU ===")
            
        self.model = Anomaly(self.win_size).to(self.device)
        bp_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.SGD(bp_parameters, lr=self.lr, momentum=0.9, weight_decay=0.0)
        
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=5)
        
        
    def train_valid_phase(self, tsTrain: TSData):
        train_loader = torch.utils.data.DataLoader(
            dataset=OneByOneTrainDataset(tsTrain, window=self.win_size, step=self.step, num=self.num),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (x, target) in loop:
                x, target = x.float().to(self.device), target.float().to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss1 = self.loss_function(output, target)
                loss1.backward()
                self.optimizer.step()
                nn.utils.clip_grad.clip_grad_norm(self.model.parameters(), 5.0)
                
                train_loss += loss1.cpu().item()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=train_loss/(idx + 1))
            
            self.adjust_lr(self.optimizer, epoch)
            
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        train_loader = torch.utils.data.DataLoader(
            dataset=AllInOneTrainDataset(tsTrains, window=self.win_size, step=self.step, num=self.num),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (x, target) in loop:
                x, target = x.float().to(self.device), target.float().to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss1 = self.loss_function(output, target)
                loss1.backward()
                self.optimizer.step()
                nn.utils.clip_grad.clip_grad_norm(self.model.parameters(), 5.0)
                
                train_loss += loss1.cpu().item()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=train_loss/(idx + 1))
            
            self.early_stopping(train_loss/len(train_loader), self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
            
            self.adjust_lr(self.optimizer, epoch)
            
    def test_phase(self, tsData: TSData):
        value = tsData.test
        length = len(value)
        back_k, backaddnum, step = 0, 5, 1
        win_size = self.win_size
        if back_k <= 5:
            back = back_k
        else:
            back = 5
        
        def modelwork(x):
            with torch.no_grad():
                x = torch.from_numpy(100 * x)
                x = x.float().to(self.device)
                x = torch.unsqueeze(x, 0)
                output = self.model(x)
            return output.detach().cpu().numpy().reshape(-1)
        
        detres = [0] * (win_size - backaddnum)
        scores = [0] * (win_size - backaddnum)
        
        for pt in range(win_size - backaddnum + back + step, length - back, step):
            head = max(0, pt - (win_size - backaddnum))
            tail = min(length, pt)
            wave = np.array(extend_series(value[head:tail + back]))
            mag = spectral_residual(wave)
            rawout = modelwork(mag)
            for ipt in range(pt - step - back, pt - back):
                scores.append(rawout[ipt - head].item())
        scores += [0] * (length - len(scores))
        self.__anomaly_score = np.array(scores)
        

    def adjust_lr(self, optimizer, epoch):
        base_lr = self.lr
        cur_lr = base_lr * (0.5 ** ((epoch + 10) // 10))
        for param in optimizer.param_groups:
            param['lr'] = cur_lr
    
    def loss_function(self, x, lb):
        l2_reg = 0.
        l2_weight = 0.
        for W in self.model.parameters():
            l2_reg = l2_reg + W.norm(2)
        kpiweight = torch.ones(lb.shape)
        kpiweight[lb == 1] = self.win_size // 100
        kpiweight = kpiweight.cuda()
        BCE = F.binary_cross_entropy(x, lb, weight=kpiweight, reduction='sum')
        return l2_reg * l2_weight + BCE
    
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.win_size), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))