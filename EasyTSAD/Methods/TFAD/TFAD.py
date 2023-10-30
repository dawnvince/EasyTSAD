import os
from typing import Dict, Optional, Tuple
from collections.abc import Callable

import numpy as np
import time
import torch
from torch import nn
import torch.fft
from torch.utils.data import DataLoader

# import torch.optim as optim
import torch_optimizer as optim

import pytorch_lightning as pl
import torchinfo

## unavailble in pl 2.0
# from pytorch_lightning.metric import Metric
from torchmetrics import Metric

from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
# from torchmetrics import Accuracy, Precision, Recall, ConfusionMatrix

# from .ts import TimeSeriesDataset
from . import model, utils
from .model.distances import CosineDistance, LpDistance, BinaryOnX1
from .model.outlier_exposure import coe_batch
from .model.mixup import mixup_batch, slow_slope
from .model.fft_aug import seasonal_shift, with_noise, other_fftshift, fft_aug
from .TSDataset import CroppedTimeSeriesDatasetTorch, TimeSeriesDataset, TimeSeries, TimeSeriesDatasetTorch, kpi_inject_anomalies
from .utils.donut_metrics import best_f1_search_grid, k_adjust_predicts

from ...DataFactory import TSData
from .. import BaseMethod

class TFAD(BaseMethod):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.__anomaly_score = None
        
        cuda = True
        self.y_hats = None
        
        self.cuda = cuda
        if self.cuda == True and torch.cuda.is_available():
            self.device = "gpu"
            print("=== Using CUDA ===")
        else:
            if self.cuda == True and not torch.cuda.is_available():
                print("=== CUDA is unavailable ===")
            self.device = "cpu"
            print("=== Using CPU ===")
        
        epochs = params["epochs"]
        ts_channels = params["ts_channels"]
        limit_val_batches = params["limit_val_batches"]
        num_sanity_val_steps = params["num_sanity_val_steps"]
        self.injection_method = params["injection_method"]
        self.ratio_injected_spikes = params["ratio_injected_spikes"]
        self.window_length = params["window_length"]
        self.suspect_window_length = params["suspect_window_length"]
        train_split_method = params["train_split_method"]
        self.num_series_in_train_batch = params["num_series_in_train_batch"]
        self.num_crops_per_series = params["num_crops_per_series"]
        rate_true_anomalies = params["rate_true_anomalies"]
        tcn_layers = params["tcn_layers"]
        tcn_out_channels = params["tcn_out_channels"]
        tcn_maxpool_out_channels = params["tcn_maxpool_out_channels"]
        embedding_rep_dim = params["embedding_rep_dim"]
        normalize_embedding = params["normalize_embedding"]
        distance = params["distance"]
        coe_rate = params["coe_rate"]
        mixup_rate = params["mixup_rate"]
        fft_sea_rate = params["fft_sea_rate"]
        fft_noise_rate = params["fft_noise_rate"]
        learning_rate = params["learning_rate"]
        check_val_every_n_epoch = params["check_val_every_n_epoch"]
        stride_roll_pred_val_test = params["stride_roll_pred_val_test"]
        max_windows_unfold_batch = params["max_windows_unfold_batch"]
        rnd_seed = params["rnd_seed"]
        self.label_reduction_method=params["label_reduction_method"]
        
        self.input_shape = (self.num_crops_per_series, ts_channels, self.window_length)
        
        tcn_kernel_size: int = 7
        
        self.rate_true_anomalies_used = 0
        
        self.lookahead_window = -1e10
        if train_split_method == "past_future_with_warmup":
            self.lookahead_window = self.window_length - self.suspect_window_length
        
        pl.trainer.seed_everything(rnd_seed)
        
        if distance == "cosine":
            # For the contrastive approach, the cosine distance is used
            distance = CosineDistance()
        elif distance == "L2":
            # For the contrastive approach, the L2 distance is used
            distance = LpDistance(p=2)
        elif distance == "non-contrastive":
            # For the non-contrastive approach, the classifier is
            # a neural-net based on the embedding of the whole window
            distance = BinaryOnX1(rep_dim=embedding_rep_dim, layers=1)
        
        self.model_dir = "pl_TFAD"
        self.test_save_path = os.path.join(self.model_dir, "test.npy")
        self.model = TFADModel(
            ts_channels=ts_channels,
            window_length=self.window_length,
            suspect_window_length=self.suspect_window_length,
            # hpars for encoder
            tcn_kernel_size=tcn_kernel_size,
            tcn_layers=tcn_layers,
            tcn_out_channels=tcn_out_channels,
            tcn_maxpool_out_channels=tcn_maxpool_out_channels,
            embedding_rep_dim=embedding_rep_dim,
            normalize_embedding=normalize_embedding,
            # hpars for classifier
            distance=distance,
            classification_loss=nn.BCELoss(),
            # hpars for anomalizers
            coe_rate=coe_rate,
            mixup_rate=mixup_rate,
            # hpars for validation and test
            stride_rolling_val_test=stride_roll_pred_val_test,
            max_windows_unfold_batch=max_windows_unfold_batch,
            # hpars for optimizer
            learning_rate=learning_rate,
            save_path=self.test_save_path
        )
        self.exp_name = None
        if self.exp_name is None:
            time_now = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
            self.exp_name = f"kpi-{time_now}"
        
        self.log_dir = "pl_log"
        logger = TensorBoardLogger(save_dir=self.log_dir, name=self.exp_name)
        
        self.checkpoint_cb = ModelCheckpoint(
            monitor="val_f1",
            dirpath=self.model_dir,
            filename="tfad-model-" + self.exp_name + "-{epoch:02d}-{val_f1:.4f}",
            save_top_k=1,
            mode="max",
        )
        
        self.trainer = Trainer(
            accelerator=self.device,
            default_root_dir=self.model_dir,
            logger=logger,
            min_epochs=epochs,
            max_epochs=epochs,
            limit_val_batches=limit_val_batches,
            num_sanity_val_steps=num_sanity_val_steps,
            check_val_every_n_epoch=check_val_every_n_epoch,
            callbacks=[self.checkpoint_cb],
            # callbacks=[checkpoint_cb, earlystop_cb, lr_logger],
    )
        
        
    def train_valid_phase(self, tsTrain: TSData):
        train_set = TimeSeriesDataset()
        train_set.append(
            TimeSeries(
                values=tsTrain.train,
                labels=tsTrain.train_label,
                item_id="one by one"
            )
        )
        
        train_set = TimeSeriesDataset(utils.take_n_cycle(train_set, len(train_set)))
        train_set_transformed = kpi_inject_anomalies(
            dataset=train_set,
            rate_true_anomalies_used=self.rate_true_anomalies_used,
            injection_method=self.injection_method,
            ratio_injected_spikes=self.ratio_injected_spikes,
        )
        # print("!!!!!!", len(train_set_transformed), train_set_transformed[1].shape)
        
        valid_set = TimeSeriesDataset()
        valid_set.append(
            TimeSeries(
                values=np.concatenate((tsTrain.train[-self.lookahead_window:], tsTrain.valid)),
                labels=np.concatenate((tsTrain.train_label[-self.lookahead_window:], tsTrain.valid_label)),
                item_id="one by one"
            )
        )
        valid_set = TimeSeriesDataset(utils.take_n_cycle(valid_set, len(valid_set)))
        
        train_loader = DataLoader(
            dataset=CroppedTimeSeriesDatasetTorch(
                ts_dataset=train_set_transformed,
                window_length=self.window_length,
                suspect_window_length=self.suspect_window_length,
                label_reduction_method=self.label_reduction_method,
                num_crops_per_series=self.num_crops_per_series,
            ),
            batch_size=self.num_series_in_train_batch,
            shuffle=True,
        )
        
        valid_loader = DataLoader(
            dataset=TimeSeriesDatasetTorch(valid_set),
            batch_size=1,
            shuffle=False,
        )
        
        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader
        )
        
        
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        train_set = TimeSeriesDataset()
        valid_set = TimeSeriesDataset()
        for curve_name, curve in tsTrains.items():
            train_set.append(
                TimeSeries(
                    values=curve.train,
                    labels=curve.train_label,
                    item_id=curve_name
                )
            )
            
            valid_set.append(
                TimeSeries(
                    values=np.concatenate((curve.train[-self.lookahead_window:], curve.valid)),
                    labels=np.concatenate((curve.train_label[-self.lookahead_window:], curve.valid_label)),
                    item_id=curve_name
                )
            )
            
        train_set = TimeSeriesDataset(utils.take_n_cycle(train_set, len(train_set)))
        train_set_transformed = kpi_inject_anomalies(
            dataset=train_set,
            rate_true_anomalies_used=self.rate_true_anomalies_used,
            injection_method=self.injection_method,
            ratio_injected_spikes=self.ratio_injected_spikes,
        )
        valid_set = TimeSeriesDataset(utils.take_n_cycle(valid_set, len(valid_set)))
        
        train_loader = DataLoader(
            dataset=CroppedTimeSeriesDatasetTorch(
                ts_dataset=train_set_transformed,
                window_length=self.window_length,
                suspect_window_length=self.suspect_window_length,
                label_reduction_method=self.label_reduction_method,
                num_crops_per_series=self.num_crops_per_series,
            ),
            batch_size=self.num_series_in_train_batch,
            shuffle=True,
        )
        
        valid_loader = DataLoader(
            dataset=TimeSeriesDatasetTorch(valid_set),
            batch_size=1,
            shuffle=False,
        )
        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader
        )
 
     
    def test_phase(self, tsData: TSData):
        test_set = TimeSeriesDataset()
        test_set.append(
            TimeSeries(
                values=np.concatenate((tsData.valid[-self.lookahead_window:], tsData.test)),
                labels=np.concatenate((tsData.valid_label[-self.lookahead_window:], tsData.test_label)),
                item_id="one by one"
            )
        )
        
        test_set = TimeSeriesDataset(utils.take_n_cycle(test_set, len(test_set)))
        test_loader = DataLoader(
            dataset=TimeSeriesDatasetTorch(test_set),
            batch_size=1,
            shuffle=False,
        )
        
        ckpt_file = [
            file for file in os.listdir(self.model_dir)
            if (file.endswith(".ckpt") and file.startswith("tfad-model-" + self.exp_name))
        ][-1]
        ckpt_path = self.model_dir +  "/" +  ckpt_file
        model = TFADModel.load_from_checkpoint(ckpt_path)
        evaluation_result = self.trainer.test(dataloaders=test_loader)
        
        scores = np.load(self.test_save_path).reshape(-1)
        self.__anomaly_score = scores
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, self.input_shape, verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))


def D_matrix(N):
    D = torch.zeros(N - 1, N)
    D[:, 1:] = torch.eye(N - 1)
    D[:, :-1] -= torch.eye(N - 1)
    return D


class CachePredictions(Metric):
    """Compute a number of metrics for  over all batches"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

        rank_zero_warn(
            "Metric `CachePredictions` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = _input_format_classification(preds, target)
        assert preds.shape == target.shape

        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        return self.preds, self.target


class hp_filter(nn.Module):
    """
        Hodrick Prescott Filter to decompose the series
    """

    def __init__(self, lamb):
        super(hp_filter, self).__init__()
        self.lamb = lamb

    def forward(self, x):
        x = x.permute(0, 2, 1)
        N = x.shape[1]
        D1 = D_matrix(N)
        D2 = D_matrix(N-1)
        D = torch.mm(D2, D1).to(device='cuda')

        g = torch.matmul(torch.inverse(torch.eye(N).to(device='cuda') + self.lamb * torch.mm(D.T, D)), x)
        res = x - g
        g = g.permute(0, 2, 1)
        res = res.permute(0, 2, 1)
        return res, g


class TFADModel(pl.LightningModule):
    """Neural Contrastive Detection in Time Series"""

    def __init__(
        self,
        # hparams for the input data
        ts_channels: int,
        window_length: int,
        suspect_window_length: int,
        # hparams for encoder
        tcn_kernel_size: int,
        tcn_layers: int,
        tcn_out_channels: int,
        tcn_maxpool_out_channels: int = 1,
        embedding_rep_dim: int = 64,
        normalize_embedding: bool = True,
        # hparams for classifier
        distance: nn.Module = CosineDistance(),
#         distance: nn.Module = NeuralDistance(),
        classification_loss: nn.Module = nn.BCELoss(),
        classifier_threshold: float = 0.5,
        threshold_grid_length_val: float = 0.10,
        threshold_grid_length_test: float = 0.05,
        # hparams for decomp
        hp_lamb: float = 6400,
        # hparams for weight of fft branch
        weight_fft_branch: float = 0.001,
        # hparams for batch
        coe_rate: float = 0.0,
        mixup_rate: float = 0.0,
        slow_slop: float = 0.25,
        fft_sea_rate: float = 0.0,
        fft_noise_rate: float = 0.0,
        rate_rn: float = 0.0,
        # hparams for validation and test
        stride_rolling_val_test: Optional[int] = None,
        val_labels_adj: bool = True,
        val_labels_adj_fun: Callable = k_adjust_predicts,
        test_labels_adj: bool = True,
        test_labels_adj_fun: Callable = k_adjust_predicts,
        max_windows_unfold_batch: Optional[int] = None,
        # hparams for optimizer
        learning_rate: float = 3e-4,
        save_path: str = None,
        k:int=7,
        *args,
        **kwargs,
    ) -> None:

        super().__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.k = k
        self.time = 0

        # Encoder Network
        self.encoder1 = model.TCNEncoder(
            in_channels=self.hparams.ts_channels,
            out_channels=self.hparams.embedding_rep_dim,
            kernel_size=self.hparams.tcn_kernel_size,
            tcn_channels=self.hparams.tcn_out_channels,
            tcn_layers=self.hparams.tcn_layers,
            tcn_out_channels=self.hparams.tcn_out_channels,
            maxpool_out_channels=self.hparams.tcn_maxpool_out_channels,
            normalize_embedding=self.hparams.normalize_embedding,
        )
        
        self.encoder2 = model.TCNEncoder(
            in_channels=self.hparams.ts_channels,
            out_channels=self.hparams.embedding_rep_dim,
            kernel_size=self.hparams.tcn_kernel_size,
            tcn_channels=self.hparams.tcn_out_channels,
            tcn_layers=self.hparams.tcn_layers,
            tcn_out_channels=self.hparams.tcn_out_channels,
            maxpool_out_channels=self.hparams.tcn_maxpool_out_channels,
            normalize_embedding=self.hparams.normalize_embedding,
        )
        
        self.encoder1f = model.TCNEncoder(
            in_channels=self.hparams.ts_channels,
            out_channels=self.hparams.embedding_rep_dim,
            kernel_size=self.hparams.tcn_kernel_size,
            tcn_channels=self.hparams.tcn_out_channels,
            tcn_layers=self.hparams.tcn_layers,
            tcn_out_channels=self.hparams.tcn_out_channels,
            maxpool_out_channels=self.hparams.tcn_maxpool_out_channels,
            normalize_embedding=self.hparams.normalize_embedding,
        )
        
        self.encoder2f = model.TCNEncoder(
            in_channels=self.hparams.ts_channels,
            out_channels=self.hparams.embedding_rep_dim,
            kernel_size=self.hparams.tcn_kernel_size,
            tcn_channels=self.hparams.tcn_out_channels,
            tcn_layers=self.hparams.tcn_layers,
            tcn_out_channels=self.hparams.tcn_out_channels,
            maxpool_out_channels=self.hparams.tcn_maxpool_out_channels,
            normalize_embedding=self.hparams.normalize_embedding,
        )

        # Contrast Classifier
        self.classifier = model.ContrastiveClasifier(
            distance=distance,
        )

        # Set classification loss
        self.classification_loss = classification_loss
        
        val_metrics = dict(
            cache_preds=CachePredictions(),
        )
        self.val_metrics = nn.ModuleDict(val_metrics)

        # Define test metrics
        # NOTE: We don't use torchmetrics directly because
        # we adjust the consider the best threshold for testing
        test_metrics = dict(
            cache_preds=CachePredictions(),
        )
        self.test_metrics = nn.ModuleDict(test_metrics)
        
        self.Decomp1 = hp_filter(lamb=self.hparams.hp_lamb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # The encoder could manage other window lengths,
        # but all training and validation is currently performed with a single length
        assert x.shape[-1] == self.hparams.window_length

        res, cyc = self.Decomp1(x)

        ts_whole_res_emb = self.encoder1(res)
        ts_context_res_emb = self.encoder1(res[..., : -self.hparams.suspect_window_length])
        
        ts_whole_cyc_emb = self.encoder2(cyc)
        ts_context_cyc_emb = self.encoder2(cyc[..., : -self.hparams.suspect_window_length])
        
        res_fft_whole = torch.fft.fft(res, dim=-1, norm='forward')
        cyc_fft_whole = torch.fft.fft(cyc, dim=-1, norm='forward')
        res_temp_whole = torch.cat((res_fft_whole.real, res_fft_whole.imag), -3)
        res_fft_ric_whole = torch.reshape(res_temp_whole.permute(1, 2, 0), [res_fft_whole.shape[-3], res_fft_whole.shape[-2], -1])
        cyc_temp_whole = torch.cat((cyc_fft_whole.real, cyc_fft_whole.imag), -3)
        cyc_fft_ric_whole = torch.reshape(cyc_temp_whole.permute(1, 2, 0), [cyc_fft_whole.shape[-3], cyc_fft_whole.shape[-2], -1]) 
        
        res_con = res[..., : -self.hparams.suspect_window_length]
        cyc_con = cyc[..., : -self.hparams.suspect_window_length]
        
        res_fft_con = torch.fft.fft(res_con, dim=-1, norm='forward')
        cyc_fft_con = torch.fft.fft(cyc_con, dim=-1, norm='forward')
        res_temp_con = torch.cat((res_fft_con.real, res_fft_con.imag), -3)
        res_fft_ric_con = torch.reshape(res_temp_con.permute(1, 2, 0), [res_fft_con.shape[-3], res_fft_con.shape[-2], -1])
        cyc_temp_con = torch.cat((cyc_fft_con.real, cyc_fft_con.imag), -3)
        cyc_fft_ric_con = torch.reshape(cyc_temp_con.permute(1, 2, 0), [cyc_fft_con.shape[-3], cyc_fft_con.shape[-2], -1]) 
        
        ts_whole_res_emb_f = self.encoder1f(res_fft_ric_whole)
        ts_context_res_emb_f = self.encoder1f(res_fft_ric_con)
        
        ts_whole_cyc_emb_f = self.encoder2f(cyc_fft_ric_whole)
        ts_context_cyc_emb_f = self.encoder2f(cyc_fft_ric_con)

        logits_anomaly = self.classifier(ts_whole_res_emb, ts_context_res_emb, ts_whole_res_emb_f, ts_context_res_emb_f, ts_whole_cyc_emb, ts_context_cyc_emb, ts_whole_cyc_emb_f, ts_context_cyc_emb_f, self.hparams.weight_fft_branch)
        
        return logits_anomaly

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> Dict[str, torch.Tensor]:

        x, y = self.xy_from_batch(batch)

        if self.hparams.coe_rate > 0:
            # print(" coe_rate x shape is", x.shape)
            x_oe, y_oe = coe_batch(
                x=x,
                y=y,
                coe_rate=self.hparams.coe_rate,
                suspect_window_length=self.hparams.suspect_window_length,
                random_start_end=True,
            )
            # Add COE to training batch
            x = torch.cat((x, x_oe), dim=0)
            y = torch.cat((y, y_oe), dim=0)

        # print(self.hparams.mixup_rate)
        if self.hparams.mixup_rate > 0.0:
            # print("mixup_rate x shape is", x.shape)
            x_mixup, y_mixup = mixup_batch(
                x=x,
                y=y,
                mixup_rate=self.hparams.mixup_rate,
            )
            # Add Mixup to training batch
            x = torch.cat((x, x_mixup), dim=0)
            y = torch.cat((y, y_mixup), dim=0)
            
        # print(self.hparams.slow_slop)
        if self.hparams.slow_slop > 0.0:
            # print("slow_slop x shape is", x.shape)
            x_mixup, y_mixup = slow_slope(
                x=x,
                y=y,
                mixup_rate=self.hparams.slow_slop,
            )
            # Add Mixup to training batch
            x = torch.cat((x, x_mixup), dim=0)
            y = torch.cat((y, y_mixup), dim=0)
            
        # 新的数据增强方式
        if self.hparams.fft_sea_rate > 0:
            x_fs, y_fs = coe_batch(
                x=x,
                y=y,
                coe_rate=self.hparams.fft_sea_rate,
                suspect_window_length=self.hparams.suspect_window_length,
                random_start_end=True,
                method = "multi_sea"
            )
            # Add COE to training batch
            x = torch.cat((x, x_fs), dim=0)
            y = torch.cat((y, y_fs), dim=0)
            
        if self.hparams.fft_noise_rate > 0:
            x_fs, y_fs = coe_batch(
                x=x,
                y=y,
                coe_rate=self.hparams.fft_sea_rate,
                suspect_window_length=self.hparams.suspect_window_length,
                random_start_end=True,
                method = "from_iad"
            )
            # Add COE to training batch
            x = torch.cat((x, x_fs), dim=0)
            y = torch.cat((y, y_fs), dim=0)
        
        ## WHERE IS remove_noise_norm ?
        
        # if self.hparams.rate_rn > 0.0:
        #     x_rn, y_rn = remove_noise_norm(
        #         x=x,
        #         y=y,
        #         rate_rn=self.hparams.rate_rn,
        #     )
        #     # Add rate_rn to training batch
        #     x = torch.cat((x, x_rn), dim=0)
        #     y = torch.cat((y, y_rn), dim=0)
        

        # Compute predictions
        logits_anomaly = self(x).squeeze()
        probs_anomaly = torch.sigmoid(logits_anomaly)

        # Calculate Loss
        loss = self.classification_loss(probs_anomaly, y)

        assert torch.isfinite(loss).item()

        # Logging loss
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {"loss": loss}

    def on_validation_epoch_start(self):
        # Reset all states in validation metrics
        for key in self.val_metrics.keys():
            self.val_metrics[key].reset()

    def validation_step(self, batch, batch_idx):
        x, y = self.xy_from_batch(batch)

        # Compute predictions
        probs_anomaly, _ = self.detect(
            ts=x,
            threshold_prob_vote=self.hparams.classifier_threshold,
            stride=int(self.hparams.stride_rolling_val_test)
            if self.hparams.stride_rolling_val_test
            else self.hparams.suspect_window_length,
        )

        # Eliminate timesteps with nan's in prediction
        nan_time_idx = torch.isnan(probs_anomaly).int().sum(dim=0).bool()
        y = y[:, ~nan_time_idx]
        probs_anomaly = probs_anomaly[:, ~nan_time_idx]
        target = y

        self.val_metrics["cache_preds"](preds=probs_anomaly, target=target)

    def on_validation_epoch_end(self):
        stage = "val"
        ### Compute metrics and find "best" threshold
        score, target = self.val_metrics["cache_preds"].compute()

        # score, target into lists of 1-d np.ndarray
        score_np, target_np = [], []
        for i in range(len(score)):
            score_i = score[i].cpu().numpy()
            target_i = target[i].cpu().numpy()
            assert (
                score_i.shape[0] == 1
            ), "Expected 1-d array with the predictad labels of the TimeSeries"
            assert (
                target_i.shape[0] == 1
            ), "Expected 1-d array with the observed labels of the TimeSeries"
            score_np.append(score_i[0, :])
            target_np.append(target_i[0, :])

        # # Criteria: best F1
        metrics_best, threshold_best = best_f1_search_grid(
            score=score_np,
            target=target_np,
            adjust_predicts_fun=self.hparams.val_labels_adj_fun if self.hparams.val_labels_adj else None,
            threshold_values=np.round(
                np.arange(0.0, 1.0, self.hparams.threshold_grid_length_val), decimals=5
            ),
            k=self.k,
            # threshold_bounds = [0.01, 0.99],
        )

        self.hparams.classifier_threshold = threshold_best
        self.log(
            f"classifier_threshold", self.hparams.classifier_threshold, prog_bar=True, logger=True
        )

        # Log metrics
        for key, value in metrics_best.items():
            self.log(f"{stage}_{key}", value, prog_bar=True if key == "f1" else False, logger=True)

    def test_step(self, batch, batch_idx):

        x, y = self.xy_from_batch(batch)

        # Compute predictions
        time1 = time.time()
        probs_anomaly, _ = self.detect(
            ts=x,
            threshold_prob_vote=self.hparams.classifier_threshold,
            stride=int(self.hparams.stride_rolling_val_test)
            if self.hparams.stride_rolling_val_test
            else self.hparams.suspect_window_length,
        )

        y_all = y
        probs_anomaly_all = probs_anomaly
        
        # Eliminate timesteps with nan's in prediction
        nan_time_idx = torch.isnan(probs_anomaly).int().sum(dim=0).bool()
        y = y[:, ~nan_time_idx]
        probs_anomaly = probs_anomaly[:, ~nan_time_idx]
        target = y
        time2 = time.time()
        # import numpy as np
        # np.savetxt("probs_anomaly"+str(batch_idx)+".csv", probs_anomaly_all.cpu().numpy().reshape(-1, 1), delimiter=',')
        # np.savetxt("target"+str(batch_idx)+".csv", y_all.cpu().numpy().reshape(-1, 1), delimiter=',')
        self.time += time2-time1
        self.test_metrics["cache_preds"](preds=probs_anomaly, target=target)

    def on_test_epoch_end(self):
        stage = "test"
        print('time',self.time)
        ### Compute metrics and find "best" threshold
        score, target = self.test_metrics["cache_preds"].compute()

        # score, target into lists of 1-d np.ndarray
        score_np, target_np = [], []
        for i in range(len(score)):
            score_i = score[i].cpu().numpy()
            target_i = target[i].cpu().numpy()
            assert (
                score_i.shape[0] == 1
            ), "Expected 1-d array with the predictad labels of the TimeSeries"
            assert (
                target_i.shape[0] == 1
            ), "Expected 1-d array with the observed labels of the TimeSeries"
            score_np.append(score_i[0, :])
            target_np.append(target_i[0, :])
            
        print(score_np)
        np.save(self.hparams.save_path, score_np)
        for key in self.test_metrics.keys():
            self.test_metrics[key].reset()

        # # Criteria: best F1
        # metrics_best, threshold_best = best_f1_search_grid(
        #     score=score_np,
        #     target=target_np,
        #     adjust_predicts_fun=k_adjust_predicts if self.hparams.test_labels_adj else None,
        #     threshold_values=np.round(
        #         np.arange(0.0, 1.0, self.hparams.threshold_grid_length_test), decimals=5
        #     ),
        #     k=self.k,
        #     # threshold_bounds = [0.01, 0.99],
        # )

        # metrics_best2, threshold_best2 = best_f1_search_grid2(
        #     score=score_np,
        #     target=target_np,
        #     adjust_predicts_fun=adjust_predicts_donut if self.hparams.test_labels_adj else None,
        #     threshold_values=np.round(
        #         np.arange(0.0, 1.0, self.hparams.threshold_grid_length_test), decimals=5
        #     ),
        #     # threshold_bounds = [0.01, 0.99],
        # )

        # self.hparams.classifier_threshold = threshold_best
        # self.log(
        #     f"classifier_threshold", self.hparams.classifier_threshold, prog_bar=True, logger=True
        # )
        
        # print(type(score_np))
        # print(type(score_np[0]))
        # score_save = np.array(score_np)
        # print(type(score_save))
        # print(score_save.shape)
        # print(type(score_save[0]))
        # print(score_save[0].shape)
        # print(score_save)
        # for i in range(score_save.shape[0]):
        #    print(score_save[i].shape)
        #    np.savetxt("score_"+str(i)+".csv", np.array(score_np[i]), delimiter=',')
        # print("here is ok")

        # Log metrics
        # for key, value in metrics_best.items():
        #     self.log(f"{stage}_{key}", value, prog_bar=True if key == "f1" else False, logger=True)
        # with open('./all_result.txt','a') as f:
        #     for key, value in metrics_best.items():
        #         f.write('{} {}\n'.format(str(key),str(value)))
        #     for key, value in metrics_best2.items():
        #         f.write('{} {}\n'.format(str(key),str(value)))
        #     f.write('\n')

    def configure_optimizers(self):
        # optim_class = optim.Adam
        optim_class = optim.Yogi
        # optim_class = optim.AdaBound

        optimizer = optim_class(self.parameters(), lr=self.learning_rate)

        return optimizer

    def detect(
        self,
        ts: torch.Tensor,
        threshold_prob_vote: float = 0.5,
        stride: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        """Deploys the model over a tensor representing the time series

        Args:
            ts: Tensor with the time series. Shape (batch_size, ts_channels, time)

        Output
            pred: Tensor with the estimated probability of each timestep being anomalous. Shape (batch_size, time)
        """

        assert 0 <= threshold_prob_vote <= 1

        if stride is None:
            stride = self.hparams.suspect_window_length

        batch_size, ts_channels, T = ts.shape
#         print("ts_channels is", ts_channels)
#         print("ts.shape is", ts.shape)
        num_windows = int(1 + (T - self.hparams.window_length) / stride)

        # Define functions for folding and unfolding the time series
        unfold_layer = nn.Unfold(
            kernel_size=(ts_channels, self.hparams.window_length), stride=stride
        )
        fold_layer = nn.Fold(
            output_size=(1, T), kernel_size=(1, self.hparams.window_length), stride=stride
        )

        # Currently, only 4-D input tensors (batched image-like tensors) are supported
        # images = (batch, channels, height, width)
        # we adapt our time series creating a height channel of dimension 1, and then
        ts_windows = unfold_layer(ts.unsqueeze(1))
#         print("ts_windows shape is", ts_windows.shape)

        assert ts_windows.shape == (
            batch_size,
            ts_channels * self.hparams.window_length,
            num_windows,
        )

        ts_windows = ts_windows.transpose(1, 2)
        ts_windows = ts_windows.reshape(
            batch_size, num_windows, ts_channels, self.hparams.window_length
        )
#         print("ts_windows shape after reshape is", ts_windows.shape)
        
        # Also posible via tensor method
        # ts_windows = ts.unfold(dimension=1,size=self.hparams.window_length,step=stride)

        with torch.no_grad():
            if self.hparams.max_windows_unfold_batch is None:
                logits_anomaly = self(ts_windows.flatten(start_dim=0, end_dim=1))
            else:
                # For very long time series, it is neccesary to process the windows in smaller chunks
                logits_anomaly = [
                    self(ts_windows_chunk)
                    for ts_windows_chunk in torch.split(
                        ts_windows.flatten(start_dim=0, end_dim=1),
                        self.hparams.max_windows_unfold_batch,
                        dim=0,
                    )
                ]
                logits_anomaly = torch.cat(logits_anomaly, dim=0)

        # Check model output shape: one label per (multivariate) window
        assert logits_anomaly.shape == (batch_size * num_windows, 1)

        # Repeat prediction for all timesteps in the suspect window, and reshape back before folding
        logits_anomaly = logits_anomaly.reshape(batch_size, num_windows, 1)
        logits_anomaly = logits_anomaly.repeat(1, 1, self.hparams.window_length)
        logits_anomaly[..., : -self.hparams.suspect_window_length] = np.nan
        logits_anomaly = logits_anomaly.transpose(1, 2)

        assert logits_anomaly.shape == (batch_size, self.hparams.window_length, num_windows)

        # Function to squeeze dimensions 1 and 2 after folding
        squeeze_fold = lambda x: x.squeeze(2).squeeze(1)

        ### Count the number of predictions per timestep ###
        # Indicates entries in logits_anomaly with a valid prediction
        id_suspect = torch.zeros_like(logits_anomaly)
        id_suspect[:, -self.hparams.suspect_window_length :] = 1.0
        num_pred = squeeze_fold(fold_layer(id_suspect))

        # Average of predicted probability of being anomalous for each timestep
        anomaly_probs = torch.sigmoid(logits_anomaly)
        # anomaly_probs_avg = squeeze_fold( fold_layer( anomaly_probs ) ) / num_pred
        anomaly_probs_nanto0 = torch.where(
            id_suspect == 1, anomaly_probs, torch.zeros_like(anomaly_probs)
        )
        anomaly_probs_avg = fold_layer(anomaly_probs_nanto0).squeeze(2).squeeze(1) / num_pred

        assert anomaly_probs_avg.shape == (batch_size, T)

        # Majority vote
        anomaly_votes = squeeze_fold(fold_layer(1.0 * (anomaly_probs > threshold_prob_vote)))
        anomaly_vote = 1.0 * (anomaly_votes > (num_pred / 2))

        assert anomaly_vote.shape == (batch_size, T)

        return anomaly_probs_avg, anomaly_vote

    def tsdetect(
        self,
        ts_dataset: TimeSeriesDataset,
        stride: Optional[int] = None,
        threshold_prob_vote: float = 0.5,
        *args,
        **kwargs,
    ) -> TimeSeriesDataset:
        """Deploys the model over a TimeSeriesDataset

        Args:
            ts_dataset: TimeSeriesDataset with the univariate time series.

        Output
            pred: Tensor with the estimated probability of each timestep being anomalous. Shape (batch, time)
        """

        assert not ts_dataset.nan_ts_values

        # Number of TimeSeries in the dataset
        N = len(ts_dataset)

        # Lengths of each TimeSeries
        ts_lengths = np.asarray([ts.shape[0] for ts in ts_dataset])
        same_length = np.all(ts_lengths == ts_lengths[0])

        ts_dataset_out = ts_dataset.copy()
        if same_length:
            # Stack all series and predict
            ts_torch = torch.stack(
                [
                    torch.tensor(ts.values.reshape(ts.shape), device=self.device)
                    for ts in ts_dataset
                ],
                dim=0,
            )
            ts_torch = ts_torch.transpose(1, 2)
            anomaly_probs_avg, anomaly_vote = self.detect(
                ts=ts_torch, threshold_prob_vote=threshold_prob_vote, stride=stride
            )
            anomaly_probs_avg = anomaly_probs_avg.cpu().numpy()
            anomaly_vote = anomaly_vote.cpu().numpy()
            # # Save prediction in dataset
            # for i, ts in enumerate(ts_dataset_out):
            #     ts.anomaly_probs_avg = anomaly_probs_avg
            #     ts.anomaly_vote = anomaly_vote
            # ts.labels = pred[i].squeeze().numpy()
        else:
            # Predict and save prediction in dataset
            anomaly_probs_avg, anomaly_vote = [], []
            for i, ts in enumerate(ts_dataset_out):
                ts_torch = (
                    torch.tensor(ts.values, device=self.device).reshape(ts.shape).T.unsqueeze(0)
                )
                if ts_torch.dim() == 2:
                    ts_torch.unsqueeze(1)
                anomaly_probs_avg_i, anomaly_vote_i = self.detect(
                    ts=ts_torch, threshold_prob_vote=threshold_prob_vote, stride=stride
                )
                anomaly_probs_avg.append(anomaly_probs_avg_i.cpu().numpy())
                anomaly_vote.append(anomaly_vote_i.cpu().numpy())
                # ts.labels = pred.squeeze().numpy()

        return anomaly_probs_avg, anomaly_vote

    @staticmethod
    def xy_from_batch(batch: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Fit batch dimensions for training and validation

        Args:
            batch : Tuple (x,y) generated by a dataloader (CroppedTimeSeriesDatasetTorch or TimeSeriesDatasetTorch)
                which provides x of shape (batch, number of crops, ts channels, time), and y of shape (batch, number of crops)

        This function flatten the first two dimensions: batch, ts sample.
        """

        x, y = batch

        # flatten first two dimensions
        if x.dim() == 4 and y.dim() == 2:
            x = torch.flatten(x, start_dim=0, end_dim=1)
            y = torch.flatten(y, start_dim=0, end_dim=1)

        return x, y
    
    
    