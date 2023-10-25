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

import itertools
from typing import Any, Iterator, List, Optional, Tuple, Union
from collections.abc import Sequence

import multiprocessing as mp
from multiprocessing.pool import ThreadPool

from pathlib import Path, PosixPath

from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt

import sklearn
from . import utils


class TimeSeries:
    """Base data structure for time series

    Attributes:
        values: Time series values. This can be a vector in the case of univariate time series, or
            a matrix for multivariate time series, with shape (Time, dimension).
        labels: Timestep label. These indicate the presence of an anomaly at each time.
        item_id: Identifies the time series, usually with a string.
        predicted_scores: (Optional) Indicate the current estimate of an anomaly at each timestep.
        indicator: Inform on injections on the time series.
            For example, The value -1 indicates may be interesting to sample windows containing this timestep.
            Could be used as a probability to sample from.
    """

    values: np.ndarray
    labels: np.ndarray
    item_id: Any
    predicted_scores: Optional[np.ndarray] = None
    indicator: np.ndarray

    def __init__(
        self,
        values: np.ndarray,
        labels: Optional[np.ndarray] = None,
        predicted_scores: Optional[np.ndarray] = None,
        item_id: Any = None,
        indicator: Optional[np.ndarray] = None,
    ):

        self.values = np.asarray(values, np.float32)
        self.labels = (
            np.asarray(labels, np.float32)
            if labels is not None
            else np.array([None] * len(values), dtype=np.float32)
        )
        assert len(self.values) == len(self.labels)
        self.item_id = item_id
        self.predicted_scores = predicted_scores
        self.indicator = np.zeros_like(self.labels).astype(int) if indicator is None else indicator

    def __len__(self):
        return len(self.values)

    def copy(self) -> "TimeSeries":
        return TimeSeries(
            values=self.values.copy(),
            labels=custom_copy(self.labels),
            predicted_scores=custom_copy(self.predicted_scores),
            indicator=custom_copy(self.indicator),
            item_id=self.item_id,
        )

    def __repr__(self):
        if self.values.ndim == 1:
            T, ts_channels = self.values.shape[0], 1
        elif self.values.ndim == 2:
            T, ts_channels = self.values.shape
        else:
            raise ValueError("values must be a vector or matrix.")
        return f"TimeSeries( item_id:{self.item_id!r}, channels:{ts_channels}, T:{T} \n values:{self.values!r} \n labels:{self.labels!r} \n predicted_scores:{self.predicted_scores!r})"

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key + 1, 1)

        ts = TimeSeries(
            values=self.values[key],
            labels=self.labels[key],
            predicted_scores=custom_slice(self.predicted_scores, key),
            indicator=custom_slice(self.indicator, key),
            item_id=None if self.item_id is None else f"slice of {self.item_id}",
        )
        return ts

    def append(self, ts: "TimeSeries"):
        ts_new = TimeSeries(
            values=np.concatenate([self.values, ts.values], axis=0),
            labels=np.concatenate([self.labels, ts.labels], axis=0),
            predicted_scores=custom_concat(self.predicted_scores, ts.predicted_scores, axis=0),
            indicator=custom_concat(self.indicator, ts.indicator, axis=0),
            item_id=f"({self.item_id},{ts.item_id})",
        )
        return ts_new

    @property
    def shape(self):
        if self.values.ndim == 1:
            T, ts_channels = self.values.shape[0], 1
        elif self.values.ndim == 2:
            T, ts_channels = self.values.shape
        else:
            raise ValueError("values must be a vector or matrix.")
        return T, ts_channels

    @property
    def nan_ts_values(self):
        return np.isnan(self.values).any()

    def plot(self, title: str = ""):
        T, ts_channels = self.shape
        fig, axs = plt.subplots(ts_channels, 1, figsize=(10, 2 * ts_channels))
        if ts_channels == 1:
            axs = np.array([axs])

        for i in range(ts_channels):
            values = self.values if ts_channels == 1 else self.values[:, i]
            axs[i].plot(values)
            ts_extrema = np.nanquantile(values, [0, 1])
            if ts_extrema[0] == ts_extrema[1]:
                ts_extrema[0] -= 1
                ts_extrema[1] += 1
            axs[i].fill_between(
                x=range(T),
                y1=np.ones(T) * ts_extrema[0],
                y2=np.where(self.labels == 1, ts_extrema[1], ts_extrema[0]),
                color="red",
                alpha=0.5,
                label="suspect window",
            )
            axs[i].fill_between(
                x=range(T),
                y1=np.ones(T) * ts_extrema[0],
                y2=np.where(self.indicator == 1, ts_extrema[1], ts_extrema[0]),
                color="blue",
                alpha=0.2,
                label="suspect window",
            )
            axs[i].set_ylim(ts_extrema[0], ts_extrema[1])
        fig.suptitle(title)
        return fig, axs


class TimeSeriesDataset(List[TimeSeries]):
    """Collection of Time Series."""

    def copy(self):
        return TimeSeriesDataset([ts.copy() for ts in self])

    def __repr__(self):
        return f"TimeSeriesDataset: {len(self)} TimeSeries"

    @property
    def nan_ts_values(self):
        return any([ts.nan_ts_values for ts in self])

    @property
    def shape(self):
        return [ts.shape for ts in self]

    def __getitem__(self, key):
        out = super().__getitem__(key)
        if isinstance(out, TimeSeries):
            return out
        elif isinstance(out, List):
            return TimeSeriesDataset(out)
        else:
            raise NotImplementedError()


from . import transforms as tr

def kpi_inject_anomalies(
    dataset: TimeSeriesDataset,
    rate_true_anomalies_used: float = 1.0,
    injection_method: str = ["None", "local_outliers"][-1],
    ratio_injected_spikes: float = None,
) -> TimeSeriesDataset:

    # dataset is transformed using a TimeSeriesTransform depending on the type of injection
    ts_transform = tr.LabelNoise(
        p_flip_1_to_0=1.0 - rate_true_anomalies_used
    )  # Ignore some true labels

    if injection_method == "None":
        ts_transform_iterator = ts_transform(dataset)
        dataset_transformed = utils.take_n_cycle(ts_transform_iterator, len(dataset))
        dataset_transformed = TimeSeriesDataset(dataset_transformed)
    elif injection_method == "local_outliers":
        # Inject synthetic anomalies: LocalOutlier
        # There are two types of series in this dataset: short and long
        # the "neighbourhood" defining the variance of the outliers are adjusted to this fact
        if ratio_injected_spikes is None:
            raise Exception
        else:
            anom_transform = tr.LocalOutlier(
                area_radius=2000,
                num_spikes=ratio_injected_spikes,
                spike_multiplier_range=(1.0, 4.0),
                direction_options=["increase"],
            )
            ts_transform = ts_transform + anom_transform

        # Generate many examples of injected time series
        multiplier = 5
        ts_transform_iterator = ts_transform(itertools.cycle(dataset))
        dataset_transformed = utils.take_n_cycle(
            ts_transform_iterator, multiplier * len(dataset)
        )
        dataset_transformed = TimeSeriesDataset(dataset_transformed)
    else:
        raise ValueError(f"injection_method = {injection_method} not supported!")

    return dataset_transformed

def custom_copy(array: Optional[np.ndarray]):
    if array is None:
        return None
    return array.copy()


def custom_slice(array: Optional[np.ndarray], key):
    if array is None:
        return None
    return array[key]


def custom_concat(array1: Optional[np.ndarray], array2: Optional[np.ndarray], *args, **kwargs):
    if (array1 is None) and (array2 is None):
        return None
    return np.concatenate([array1, array2], *args, **kwargs)


##### Functions for TimeSeries and TimeSeriesDataset  #####

## Why the data proprecessing module in its original code has so many F**king nesting ???
## Why spliting function contains additional operations ???

def ts_random_crop(
    ts: TimeSeries,
    length: int,
    num_crops: int = 1,
) -> TimeSeriesDataset:

    T = len(ts.values)

    if T < length:
        return []
    idx_end = np.random.randint(low=length, high=T, size=num_crops)

    out = [
        TimeSeries(
            values=_slice_pad_left(v=ts.values, end=idx_end[i], size=length, pad_value=np.nan),
            labels=_slice_pad_left(v=ts.labels, end=idx_end[i], size=length, pad_value=np.nan),
        )
        for i in range(num_crops)
    ]

    return out


def _slice_pad_left(v, end, size, pad_value):
    start = max(end - size, 0)
    v_slice = v[start:end]
    if len(v_slice) == size:
        return v_slice
    diff = size - len(v_slice)
    result = np.concatenate([[pad_value] * diff, v_slice])
    assert len(result) == size
    return result


def ts_rolling_window(
    ts: TimeSeries,
    window_length: int,
    stride: int = 1,
) -> TimeSeriesDataset:

    values_windows = _rolling_window(ts.values, window_length=window_length, stride=stride)
    labels_windows = _rolling_window(ts.labels, window_length=window_length, stride=stride)
    out = []
    n_windows = len(values_windows)
    for i in range(n_windows):
        out.append(TimeSeries(values=values_windows[i], labels=labels_windows[i]))

    return out


def _rolling_window(
    ts_array: np.ndarray,
    window_length: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Return a view to rolling windows of a time-series.
    """
    assert len(ts_array) >= window_length

    shape = (((len(ts_array) - window_length) // stride) + 1, window_length)
    strides = (ts_array.strides[0] * stride, ts_array.strides[0])
    return np.lib.stride_tricks.as_strided(ts_array, shape=shape, strides=strides)


def ts_split(
    ts: TimeSeries,
    indices_or_sections,
) -> TimeSeriesDataset:
    """Split a TimeSeries into multiple sub-TimeSeries.

    Args:
        ts : Time series to split
        indices_or_sections : Specify how to split,
            same logic as in np.split()
    """

    if not isinstance(indices_or_sections[0], Sequence):
        values_split = np.split(ts.values, indices_or_sections)
        labels_split = np.split(ts.labels, indices_or_sections)
        predicted_scores_split = (
            np.split(ts.predicted_scores, indices_or_sections) if ts.predicted_scores else None
        )
    else:
        values_split = _overlapping_split(ts.values, indices_or_sections)
        labels_split = _overlapping_split(ts.labels, indices_or_sections)
        predicted_scores_split = (
            _overlapping_split(ts.predicted_scores, indices_or_sections)
            if ts.predicted_scores
            else None
        )

    out = TimeSeriesDataset(
        [
            TimeSeries(
                values=values_split[i],
                labels=labels_split[i],
                item_id=f"{ts.item_id}_{i}" if ts.item_id else None,
                predicted_scores=predicted_scores_split[i] if ts.predicted_scores else None,
            )
            for i in range(len(values_split))
        ]
    )
    return out


def _overlapping_split(array, list_indices):
    l_splits = []
    for i in range(len(list_indices)):
        l_splits.append(array[list_indices[i][0] : list_indices[i][1]])
    return l_splits


def ts_to_array(
    ts: TimeSeriesDataset,
) -> np.array:
    """TimeSeries to numpy array

    Obtains a numpy array by stacking values and labels from a TimeSeries.
    The output dimension is (T,ts_channels+1)
    """
    out = np.concatenate((ts.values.reshape(ts.shape), ts.labels.reshape(ts.shape)), axis=1)
    return out


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

from typing import Optional

from functools import partial

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda

import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict

class TFADDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ts_dataset: TimeSeriesDataset,
        validation_ts_dataset: Optional[TimeSeriesDataset],
        test_ts_dataset: Optional[TimeSeriesDataset],
        window_length: int,
        suspect_window_length: int,
        num_series_in_train_batch: int,
        num_crops_per_series: int = 1,
        label_reduction_method: Optional[str] = [None, "any"][-1],
        stride_val_test: int = 1,
        num_workers: int = 0,
        *args,
        **kwargs,
    ) -> None:

        super().__init__()

        self.train_ts_dataset = train_ts_dataset
        self.validation_ts_dataset = validation_ts_dataset
        self.test_ts_dataset = test_ts_dataset

        hparams = AttributeDict(
            window_length=window_length,
            suspect_window_length=suspect_window_length,
            num_series_in_train_batch=num_series_in_train_batch,
            num_crops_per_series=num_crops_per_series,
            label_reduction_method=label_reduction_method,
            stride_val_test=stride_val_test,
            num_workers=num_workers,
        )
        self.hparams = hparams
        self.hparams.update(hparams)

        self.datasets = {}
        assert (
            not train_ts_dataset.nan_ts_values
        ), "TimeSeries in train_ts_dataset must not have nan values."
        self.datasets["train"] = CroppedTimeSeriesDatasetTorch(
            ts_dataset=train_ts_dataset,
            window_length=self.hparams.window_length,
            suspect_window_length=self.hparams.suspect_window_length,
            label_reduction_method=self.hparams.label_reduction_method,
            num_crops_per_series=self.hparams.num_crops_per_series,
        )

        if validation_ts_dataset is not None:
            assert (
                not validation_ts_dataset.nan_ts_values
            ), "TimeSeries in validation_ts_dataset must not have nan values."
            self.datasets["validation"] = TimeSeriesDatasetTorch(validation_ts_dataset)

        if test_ts_dataset is not None:
            assert (
                not test_ts_dataset.nan_ts_values
            ), "TimeSeries in test_ts_dataset must not have nan values."
            self.datasets["test"] = TimeSeriesDatasetTorch(test_ts_dataset)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.hparams[f"num_series_in_train_batch"],
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.datasets["validation"],
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.datasets["test"],
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )


class TimeSeriesDatasetTorch(Dataset):
    """Time series dataset

    Creates a pytorch dataset based on a TimeSeriesDataset.

    It is possible to apply transformation to the input TimeSeries or the windows.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
    ) -> None:
        """
        Args:
            dataset : TimeSeriesDataset with which serve as the basis for the Torch dataset.
        """
        self.dataset = dataset

        self.transform = Compose(
            [
                Lambda(lambda ts: [ts.values, ts.labels]),
                Lambda(
                    lambda vl: [np.expand_dims(vl[0], axis=1) if vl[0].ndim == 1 else vl[0], vl[1]]
                ),  # Add ts channel dimension, if needed
                Lambda(
                    lambda vl: [np.transpose(vl[0]), vl[1]]
                ),  # Transpose ts values, so the dimensions are (channel, time)
                Lambda(lambda x: [torch.from_numpy(x_i) for x_i in x]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x, y = self.transform(self.dataset[idx])

        return x, y


class CroppedTimeSeriesDatasetTorch(Dataset):
    """Cropped time series dataset

    Creates a pytorch dataset based on windows from a TimeSeriesDataset.

    Each window (a.k.a. crop) has length of window_length.

    The label y is based on the last 'suspect_window_length' time steps.
    The labels are aggregated according to label_reduction_method.

    It is possible to apply transformation to the input TimeSeries or each window.
    """

    def __init__(
        self,
        ts_dataset: TimeSeriesDataset,
        window_length: int,
        suspect_window_length: int,
        num_crops_per_series: int = 1,
        label_reduction_method: Optional[str] = [None, "any"][-1],
    ) -> None:
        """
        Args:
            ts_dataset : TimeSeriesDataset with which serve as the basis for the cropped windows
            window_length : Length of the (random) windows to be considered. If not specified, the whole series is returned.
            suspect_window_length : Number of timesteps considered at the end of each window
                to define whether a window is anomalous of not.
            num_crops_per_series : Number of random windows taken from each TimeSeries from dataset.
            label_reduction_method : Method used to reduce the labels in the suspect window.
                None : All labels in the suspect window are returned
                'any' : The anomalies of a window is anomalous is any timestep in the suspect_window_length is marked as anomalous.
        """
        self.ts_dataset = ts_dataset

        self.window_length = int(window_length) if window_length else None

        self.suspect_window_length = int(suspect_window_length)
        self.label_reduction_method = label_reduction_method

        self.num_crops_per_series = int(num_crops_per_series)

        # Validate that all TimeSeries in ts_dataset are longer than window_length
        ts_dataset_lengths = np.array([len(ts.values) for ts in self.ts_dataset])
        if any(ts_dataset_lengths < self.window_length):
            raise ValueError(
                "All TimeSeries in 'ts_dataset' must be of length greater or equal to 'window_length'"
            )

        self.cropping_fun = partial(
            ts_random_crop, length=self.window_length, num_crops=self.num_crops_per_series
        )

        # Apply ts_window_transform, to anomalize the window randomly
        self.transform = Compose(
            [
                # Pick a random crop from the selected TimeSeries
                Lambda(lambda x: self.cropping_fun(ts=x)),  # Output: List with cropped TimeSeries
                Lambda(
                    lambda x: (
                        np.stack([ts.values.reshape(ts.shape).T for ts in x], axis=0),
                        np.stack([ts.labels for ts in x], axis=0),
                    )
                ),  # output: tuple of two np.arrays (values, labels), with shapes (num_crops, N, T) and (num_crops, T)
                Lambda(
                    lambda x: [torch.from_numpy(x_i) for x_i in x]
                ),  # output: two torch Tensor (values, labels) with shapes (num_crops, N, T) and (num_crops, T)
            ]
        )

    def __len__(self):
        return len(self.ts_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x, y = self.transform(self.ts_dataset[idx])

        y_suspect = reduce_labels(
            y=y,
            suspect_window_length=self.suspect_window_length,
            reduction_method=self.label_reduction_method,
        )
        return x, y_suspect


def reduce_labels(
    y: torch.Tensor,
    suspect_window_length: int,
    reduction_method: Optional[str] = [None, "any"][-1],
) -> torch.Tensor:
    """Auxiliary function to reduce labels, one per batch element

    Args:
        y : Tensor with the labels to be reduced. Shape (batch, time).
        suspect_window_length : Number of timesteps considered at the end of each window
            to define whether a window is anomalous of not.
        reduction_method : Method used to reduce the labels in the suspect window.
            None : All labels in the suspect window are returned. The output is a 2D tensor.
            'any' : The anomalies of a window is anomalous if any timestep in the
                    suspect_window_length is marked as anomalous. The output is a 1D tensor.
    Output:
        y_suspect : Tensor with the reduced labels. Shape depends on the reduction_method used.
    """

    suspect_window_length = int(suspect_window_length)

    y_suspect = y[..., -suspect_window_length:]

    if reduction_method is None:
        pass
    elif reduction_method == "any":
        # Currently we will do:
        #   nan are valued as zero, unless
        #   if there are only nan's, y will be nan
        #     [0,0,0,0,0] -> 0
        #     [0,0,0,1,0] -> 1
        #     [nan,nan,nan,nan,nan] -> nan
        #     [0,0,0,nan,0] -> nan
        #     [0,nan,0,1,0] -> 1
        #     [nan,nan,nan,1,nan] -> 1
        y_nan = torch.isnan(y_suspect)
        if torch.any(y_nan).item():
            y_suspect = torch.where(
                y_nan, torch.zeros_like(y_suspect), y_suspect
            )  # Substitutes nan by 0
            y_suspect = (
                torch.sum(y_suspect, dim=1).bool().float()
            )  # Add to check if theres any 1 in each row
            y_suspect = torch.where(
                torch.sum(y_nan, dim=1).bool(), torch.full_like(y_suspect, float("nan")), y_suspect
            )  # put nan if all elements are nan
        else:
            y_suspect = torch.sum(y_suspect, dim=1).bool().float()
    else:
        raise ValueError(f"reduction_method = {reduction_method} not supported.")

    return y_suspect
