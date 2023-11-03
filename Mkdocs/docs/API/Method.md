# The built-in methods

## Prediction-based

| Class Name | Description | Ref. |
| ------------ | ------------- | ------------- |
| AR | AutoRegression implemented by a torch linear (using first order difference)| Robust regression and outlier detection |
| LSTMADalpha | LSTMAD in a seq2seq manner | Long Short Term Memory Networks for Anomaly Detection in Time Series |
| LSTMADbeta | LSTMAD in a multi-step prediction manner | Long Short Term Memory Networks for Anomaly Detection in Time Series

## Reconstruction-based
| Class Name | Description | Ref. |
| ------------ | ------------- | ------------- |
| AE | AutoEncoder | Sparse autoencoder |
| EncDecAD | Combine LSTM and AE | LSTM-based encoder-decoder for multi- sensor anomaly detection
| SRCNN |  | Time-series anomaly detection service at microsoft |
| Amomaly Transformer | | Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy
| TimesNet | | TIMESNET: TEMPORAL 2D-VARIATION MODELING FOR GENERAL TIME SERIES ANALYSIS

## VAE-based
| Class Name | Description | Ref. |
| ------------ | ------------- | ------------- |
| Donut | | Unsupervised anomaly 1032 detection via variational auto-encoder for seasonal kpis in web applications | 
| FCVAE | | UnderReview

## Others
| Class Name | Description | Ref. |
| ------------ | ------------- | ------------- |
| TFAD | | TFAD: A decomposition time series anomaly detection architecture with time-frequency analysis | 