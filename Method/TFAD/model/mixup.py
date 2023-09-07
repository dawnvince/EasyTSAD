# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


import numpy as np
import torch


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    mixup_rate: float,
) -> torch.Tensor:
    """
    Args:
        x : Tensor of shape (batch, ts channels, time)
        y : Tensor of shape (batch, )
        mixup_rate : Number of generated anomalies as proportion of the batch size.
    """

    if mixup_rate == 0:
        raise ValueError(f"mixup_rate must be > 0.")
    batch_size = x.shape[0]
    mixup_size = int(batch_size * mixup_rate)  #

    # Select indices
    idx_1 = torch.arange(mixup_size)
    idx_2 = torch.arange(mixup_size)
    while torch.any(idx_1 == idx_2):
        idx_1 = torch.randint(low=0, high=batch_size, size=(mixup_size,)).type_as(x).long()
        idx_2 = torch.randint(low=0, high=batch_size, size=(mixup_size,)).type_as(x).long()

    # sample mixing weights:
    beta_param = float(0.05)
    beta_distr = torch.distributions.beta.Beta(
        torch.tensor([beta_param]), torch.tensor([beta_param])
    )
    weights = torch.from_numpy(np.random.beta(beta_param, beta_param, (mixup_size,))).type_as(x)
    oppose_weights = 1.0 - weights

    # Create contamination
    # corrected x_mix_2: idx_2
    x_mix_1 = x[idx_1].clone()
    x_mix_2 = x[idx_2].clone()
    x_mixup = (
        x_mix_1 * weights[:, None, None] + x_mix_2 * oppose_weights[:, None, None]
    )  # .detach()

    # Label as positive anomalies
    y_mixup = y[idx_1].clone() * weights + y[idx_2].clone() * oppose_weights

    return x_mixup, y_mixup


def slow_slope(
    x: torch.Tensor,
    y: torch.Tensor,
    mixup_rate: float,
) -> torch.Tensor:
    # print("x shape is", x.shape)
    # print("y shape is", y.shape)
    # print("y is", y)
    # print("x in slow_slop is", x)
    """
    Args:
        x : Tensor of shape (batch, ts channels, time)
        y : Tensor of shape (batch, )
        mixup_rate : Number of generated anomalies as proportion of the batch size.
    """

    if mixup_rate == 0:
        raise ValueError(f"mixup_rate must be > 0.")
    batch_size = x.shape[0]
    mixup_size = int(batch_size * mixup_rate)  #
    

    # Select indices
    idx_1 = torch.arange(mixup_size)
    x_mix_1 = x[idx_1].clone()

    # 构造一个单调增函数
    slop = torch.arange(x_mix_1[:, 0, :].shape[0])
    # print(slop.shape)
    

    # Create contamination
    # corrected x_mix_2: idx_2
    x_mixup = x[idx_1].clone()
    
#     print("x_mixup[:, 0, :] shape is", x_mixup[:][0][:].shape)
    
#     print("x_mixup[:, 0, :] is", x_mixup[:, 0, :])
    oe_size = int(x_mixup[:, 0, :].shape[1])
    # idx_2 = torch.arange(oe_size)
    
    s_r = torch.ones(slop.shape)
    s_c = (0.00001*torch.arange(oe_size))
    s_slop = torch.unsqueeze(s_r, dim=1)*torch.unsqueeze(s_c, dim=0)
    # print(s_slop.shape)
    # print(s_slop)
    
    
    x_mixup[:, 0, :] = x_mix_1[:, 0, :] + s_slop.cuda()
    
    # print("x_mixup[:, 0, :] in slow_slop is", x_mixup[:, 0, :])
    
    # Label as positive anomalies
    y_oe= torch.ones(mixup_size).type_as(y)

    return x_mixup, y_oe


# from iad.robust_filters.decomposition.robustSTL import RobustSTL


# def remove_noise_norm(
#     x: torch.Tensor,
#     y: torch.Tensor,
#     rate_rn: float,
# ) -> torch.Tensor:
#     """
#     Args:
#         x : Tensor of shape (batch, ts channels, time)
#         y : Tensor of shape (batch, )
#         mixup_rate : Number of generated anomalies as proportion of the batch size.
#     """
    
#     if mixup_rate == 0:
#         raise ValueError(f"mixup_rate must be > 0.")
#     batch_size = x.shape[0]
#     mixup_size = int(batch_size * rate_rn) 

#     # Select indices
#     idx_1 = torch.arange(mixup_size)
    
#     batch_data = x[idx_1][y==0].clone().numpy()

#     ## set no trend to extract
#     current_filter = RobustSTL(data_T=data_T,
#                      noise_toggle=True,
#                      noise_sigma_i=2.0,
#                      noise_sigma_d=2.5,
#                      noise_truncate=8.0,
#                      trend_toggle=True,
#                      trend_vlambda=trend_vlambda,  #成倍放大  趋势突变
#                      trend_vlambda_diff=trend_vlambda_diff,  #越小越平
#                      trend_solver_method='GADMM', # g_lasso_admm
#                      trend_solver_maxiters=20,
#                      trend_solver_show_progress=False,
#                      season_toggle=True,
#                      season_bilateral_period_num=2,  #周期参数
#                      season_neighbour_wdw_size=20,
#                      season_sigma_i=1, #4,
#                      season_sigma_d=2 #10,
#                     )

#     # first fit_transform
#     decomposed_out_part = current_filter.fit_transform(batch_data)

#     adjusted_trend = decomposed_out_part[:,0].reshape(-1,1)
#     adjusted_season = decomposed_out_part[:,1].reshape(-1,1)
#     irregular_data = decomposed_out_part[:,2].reshape(-1,1)
    
#     # x_rn = torch.from_numpy(adjusted_trend+adjusted_season)
#     x_rn = torch.from_numpy(adjusted_trend+adjusted_season+0.1*irregular_data)
#     y_rn = torch.zeros(y[idx_1][y==0].shape)
    
#     return x_rn, y_rn
