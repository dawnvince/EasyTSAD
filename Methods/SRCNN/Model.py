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
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

def make_layers(Bn=True, input=256):
    global configs
    layers = []
    layer = nn.Conv2d(input, input, kernel_size=1, stride=1, padding=0)
    layers.append(layer)
    if Bn:
        layers.append(nn.BatchNorm2d(input))

    for k, s, c in configs:
        if c == -1:
            layer = nn.Conv2d(kernel_size=k, stride=s, padding=0)
        else:
            now = []
            now.append(nn.Conv1d(input, c, kernel_size=k, stride=s, padding=0))
            input = c
            if Bn:
                now.append(nn.BatchNorm2d(input))
            now.append(nn.Relu(inplace=True))
            layer = nn.Sequential(*now)
        layers.append(layer)
    return nn.Sequential(*layers), input


class trynet(nn.Module):
    def __init__(self):
        super(trynet, self).__init__()
        self.layer1 = nn.Conv1d(1, 128, kernel_size=128, stride=0, padding=0)
        self.layer2 = nn.BatchNorm1d(128)

        self.feature = make_layers()
        
class Anomaly(nn.Module):
    def __init__(self, window=1024):
        self.window = window
        super(Anomaly, self).__init__()
        self.layer1 = nn.Conv1d(window, window, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv1d(window, 2 * window, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(2 * window, 4 * window)
        self.fc2 = nn.Linear(4 * window, window)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), self.window, 1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
    

def predict_next(values):
    """
    Predicts the next value by sum up the slope of the last value with previous values.
    Mathematically, g = 1/m * sum_{i=1}^{m} g(x_n, x_{n-i}), x_{n+1} = x_{n-m+1} + g * m,
    where g(x_i,x_j) = (x_i - x_j) / (i - j)
    :param values: list.
        a list of float numbers.
    :return : float.
        the predicted next value.
    """

    if len(values) <= 1:
        raise ValueError(f'data should contain at least 2 numbers')

    v_last = values[-1]
    n = len(values)

    slopes = [(v_last - v) / (n - 1 - i) for i, v in enumerate(values[:-1])]

    return values[1] + sum(slopes)
    
def extend_series(values, extend_num=5, look_ahead=5):
    """
    extend the array data by the predicted next value
    :param values: list.
        a list of float numbers.
    :param extend_num: int, default 5.
        number of values added to the back of data.
    :param look_ahead: int, default 5.
        number of previous values used in prediction.
    :return: list.
        The result array.
    """

    if look_ahead < 1:
        raise ValueError('look_ahead must be at least 1')

    extension = [predict_next(values[-look_ahead - 2:-1])] * extend_num
    return np.concatenate((values, extension), axis=0)

def average_filter(values, n=3):
    """
    Calculate the sliding window average for the give time series.
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    :param values: list.
        a list of float numbers
    :param n: int, default 3.
        window size.
    :return res: list.
        a list of value after the average_filter process.
    """

    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res

def fft(values):
    wave = np.array(values)
    trans = np.fft.fft(wave)
    realnum = np.real(trans)
    comnum = np.imag(trans)
    mag = np.sqrt(realnum ** 2 + comnum ** 2)
    mag += 1e-5
    spectral = np.exp(np.log(mag) - average_filter(np.log(mag)))
    trans.real = trans.real * spectral / mag
    trans.imag = trans.imag * spectral / mag
    wave = np.fft.ifft(trans)
    mag = np.sqrt(wave.real ** 2 + wave.imag ** 2)
    return mag

def spectral_residual(values):
    """
    This method transform a time series into spectral residual series
    :param values: list.
        a list of float values.
    :return: mag: list.
        a list of float values as the spectral residual values
    """
    EPS = 1e-8
    trans = np.fft.fft(values)
    mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)

    maglog = [np.log(item) if abs(item) > EPS else 0 for item in mag]

    spectral = np.exp(maglog - average_filter(maglog, n=3))

    trans.real = [ireal * ispectral / imag if abs(imag) > EPS else 0
                  for ireal, ispectral, imag in zip(trans.real, spectral, mag)]
    trans.imag = [iimag * ispectral / imag if abs(imag) > EPS else 0
                  for iimag, ispectral, imag in zip(trans.imag, spectral, mag)]

    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)

    return mag