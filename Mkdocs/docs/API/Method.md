# The built-in methods

## Statistical Methods
| Class Name | Requirements | Description | Ref. |
| ------------ | ------------- | ------------- | ------------- |
| MatrixProfile | matrixprofile(pypi) | | Matrix Profile XI: SCRIMP++: Time Series Motif Discovery at Interactive Speeds |
| SubLOF | sklearn | LOF in sequence manner | LOF: identifying density-based local outliers | 
| SAND | tslearn==0.4.1 | | SAND: streaming subsequence anomaly detection |
| SubOCSVM | sklearn | OCSVM in sequence manner | Support Vector Method for Novelty Detection |


## Prediction-based

| Class Name | Requirements | Description | Ref. |
| ------------ | ------------- | ------------- | ------------- |
| AR | pytorch | AutoRegression implemented by a torch linear (using first order difference)| Robust regression and outlier detection |
| LSTMADalpha | pytorch | LSTMAD in a seq2seq manner | Long Short Term Memory Networks for Anomaly Detection in Time Series |
| LSTMADbeta | pytorch | LSTMAD in a multi-step prediction manner | Long Short Term Memory Networks for Anomaly Detection in Time Series

## Reconstruction-based
| Class Name | Requirements | Description | Ref. |
| ------------ | ------------- | ------------- | ------------- |
| AE | pytorch | AutoEncoder | Sparse autoencoder |
| EncDecAD | pytorch | Combine LSTM and AE | LSTM-based encoder-decoder for multi- sensor anomaly detection
| SRCNN | pytorch | | Time-series anomaly detection service at microsoft |
| Amomaly Transformer | pytorch | | Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy |
| TFAD | pytorch-lightning | | TFAD: A decomposition time series anomaly detection architecture with time-frequency analysis | 

## VAE-based
| Class Name | Requirements | Description | Ref. |
| ------------ | ------------- | ------------- | ------------- |
| Donut | pytorch | Unsupervised anomaly 1032 detection via variational auto-encoder for seasonal kpis in web applications | 
| FCVAE | pytorch-lightning | Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective |

## Representation Learning
| Class Name | Requirements | Description | Ref. |
| ------------ | ------------- | ------------- | ------------- |
| DCDetector | pytorch | | DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection |

## General TimeSeries Models
| Class Name | Requirements | Description | Ref. |
| ------------ | ------------- | ------------- | ------------- |
| TimesNet | pytorch | | TIMESNET: TEMPORAL 2D-VARIATION MODELING FOR GENERAL TIME SERIES ANALYSIS |
| OFA | pytorch | One Fits all, freezing some GPT2 params |  One Fits All: Power General Time Series Analysis by Pretrained LM |
| FITS | pytorch | | FITS: Modeling Time Series with $10K$ Parameters |