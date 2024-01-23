from typing import Dict
import numpy as np
from EasyTSAD.DataFactory import TSData
import torch
from torch import nn
import torch.nn.functional as F
import torchinfo
import tqdm
from ...Exptools import EarlyStoppingTorch

from .. import BaseMethod
from ...DataFactory import TSData
from .TSDataset import OneByOneDataset, AllInOneDataset
from .model import TranADModel


class TranAD(BaseMethod):
    def __init__(self, params: dict) -> None:
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
        self.num_epochs = params["epochs"]
        patience = params["earlystop_patience"]

        self.model = TranADModel(feats=1).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=params["lr"], weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
        self.criterion = nn.MSELoss()

        self.save_path = (
            None if not params.get("save_path") else params.get("save_path")
        )
        self.early_stopping = EarlyStoppingTorch(
            save_path=self.save_path, patience=patience
        )

    def train_valid_phase(self, tsTrain: TSData):
        train_loader = torch.utils.data.DataLoader(
            dataset=OneByOneDataset(tsTrain.train, window_size=self.window_size),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # valid_loader = torch.utils.data.DataLoader(
        #     dataset=OneByOneDataset(
        #         tsTrain.valid, window_size=self.window_size, phase="valid"
        #     ),
        #     batch_size=self.batch_size,
        #     shuffle=False,
        # )

        for epoch in range(1, self.num_epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (x, _) in loop:
                x = x.to(self.device)
                x = x.unsqueeze(-1)
                bs = x.shape[0]
                x = x.permute(1, 0, 2)
                elem = x[-1, :, :].view(1, bs, 1)

                self.optimizer.zero_grad()
                z = self.model(x, elem)
                loss = (1 / epoch) * self.criterion(z[0], elem) + (
                    1 - 1 / epoch
                ) * self.criterion(z[1], elem)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                avg_loss += loss.cpu().item()
                loop.set_description(f"Training Epoch [{epoch}/{self.num_epochs}]")
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            # self.model.eval()
            # avg_loss = 0
            # loop = tqdm.tqdm(
            #     enumerate(valid_loader), total=len(valid_loader), leave=True
            # )
            # with torch.no_grad():
            #     for idx, x in loop:
            #         x = x.to(self.device)
            #         # x = x.unsqueeze(-1)
            #         bs = x.shape[0]
            #         x = x.permute(1, 0, 2)
            #         elem = x[-1, :, :].view(1, bs, 1)

            #         self.optimizer.zero_grad()
            #         z = self.model(x)
            #         loss = (1 / epoch) * self.criterion(z[0], elem) + (
            #             1 - 1 / epoch
            #         ) * self.criterion(z[1], elem)

            #         avg_loss += loss.cpu().item()
            #         loop.set_description(
            #             f"Validation Epoch [{epoch}/{self.num_epochs}]"
            #         )
            #         loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))
            self.scheduler.step()
            avg_loss = avg_loss / len(train_loader)
            self.early_stopping(avg_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break

    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        train_loader = torch.utils.data.DataLoader(
            dataset=AllInOneDataset(
                tsTrains, phase="train", window_size=self.window_size
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # valid_loader = torch.utils.data.DataLoader(
        #     dataset=AllInOneDataset(
        #         tsTrains, phase="valid", window_size=self.window_size
        #     ),
        #     batch_size=self.batch_size,
        #     shuffle=False,
        # )

        for epoch in range(1, self.num_epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (x, _) in loop:
                # breakpoint()
                x = x.to(self.device)
                x = x.unsqueeze(-1)
                bs = x.shape[0]
                x = x.permute(1, 0, 2)
                elem = x[-1, :, :].view(1, bs, 1)

                self.optimizer.zero_grad()
                z = self.model(x, elem)
                loss = (1 / epoch) * self.criterion(z[0], elem) + (
                    1 - 1 / epoch
                ) * self.criterion(z[1], elem)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                avg_loss += loss.cpu().item()
                loop.set_description(f"Training Epoch [{epoch}/{self.num_epochs}]")
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            # self.model.eval()
            # avg_loss = 0
            # loop = tqdm.tqdm(
            #     enumerate(valid_loader), total=len(valid_loader), leave=True
            # )
            # with torch.no_grad():
            #     for idx, x in loop:
            #         x = x.to(self.device)
            #         # x = x.unsqueeze(-1)
            #         bs = x.shape[0]
            #         x = x.permute(1, 0, 2)
            #         elem = x[-1, :, :].view(1, bs, 1)

            #         self.optimizer.zero_grad()
            #         z = self.model(x)
            #         loss = (1 / epoch) * self.criterion(z[0], elem) + (
            #             1 - 1 / epoch
            #         ) * self.criterion(z[1], elem)

            #         avg_loss += loss.cpu().item()
            #         loop.set_description(
            #             f"Validation Epoch [{epoch}/{self.num_epochs}]"
            #         )
            #         loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))
            self.scheduler.step()
            avg_loss = avg_loss / len(train_loader)

            self.early_stopping(avg_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break

    def test_phase(self, tsData: TSData):
        test_loader = torch.utils.data.DataLoader(
            dataset=OneByOneDataset(tsData.test, window_size=self.window_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model.eval()
        scores = []
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with torch.no_grad():
            for idx, (x, _) in loop:
                x = x.to(self.device)
                bs = x.shape[0]
                x = x.unsqueeze(-1)
                x = x.permute(1, 0, 2)
                elem = x[-1, :, :].view(1, bs, 1)
                # breakpoint()
                _, z = self.model(x, elem)
                loss = F.mse_loss(z, elem, reduction="none")[0].squeeze(1)
                scores.append(loss.cpu())

        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()

        self.__anomaly_score = scores

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
