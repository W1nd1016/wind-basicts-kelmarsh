<img width="2222" height="590" alt="image" src="https://github.com/user-attachments/assets/7b11dde0-d50f-4d43-911a-5719f7989a40" /># WIND_BASICTS Kelmarsh wind forecasting

This repository builds an hourly spatiotemporal dataset for the Kelmarsh wind farm with 6 turbines, then trains and evaluates multiple forecasting baselines such as AGCRN, MTGNN, DGCRN, D2STGNN, LSTM, and STID.

The pipeline has two main parts

1. Data preparation  
   Merge SCADA and CERRA features, align everything to a common hourly timeline, normalize using train split statistics, and save numpy files into data/wind

2. Training and evaluation  
   Use a sliding window dataset with history length 24 hours and horizon 6 hours, then train models under tools

---

## Repository layout
```
WIND_BASICTS
├── BasicTS
├── configs
├── data
│   └── wind
│       ├── X.npy
│       ├── Y.npy
│       ├── mask.npy
│       ├── adj.npy
│       └── meta.json
├── datasets
│   ├── init.py
│   └── wind_dataset.py
├── models
│   ├── agcrn_bt.py
│   ├── agcrn_min.py
│   ├── d2stgnn_v058.py
│   ├── dgcrn_v058.py
│   ├── lstm_stf_v001.py
│   ├── mtgnn_v058.py
│   ├── stid_min.py
│   └── stid_official.py
├── output
├── raw_cerra
├── raw_scada
└── tools
├── prep_scada_cerra_kelmarsh.py
├── prep_scada_only_kelmarsh.py
├── train_agcrn_min.py
├── train_agcrn_v058.py
├── train_agcrn_mtgnn_head_v001.py
├── train_d2stgnn_v058.py
├── train_dgcrn_v058.py
├── train_lstm_stf_v001.py
├── train_mtgnn_v058.py
├── ensemble_agcrn_mtgnn_v001.py
├── ensemble_agcrn_mtgnn_v002.py
├── cal_mae_rmse.py
└── predict_plot_*.py
```
---

## Requirements

Python 3.9 or newer is recommended.

Minimal packages for preprocessing and training

- numpy
- pandas
- torch

Install

pip install numpy pandas torch

---

## Raw data

Put SCADA files into raw_scada

- Kelmarsh_WT_static.csv
- Turbine_Data_Kelmarsh_1_*.csv
- Turbine_Data_Kelmarsh_2_*.csv
- Turbine_Data_Kelmarsh_3_*.csv
- Turbine_Data_Kelmarsh_4_*.csv
- Turbine_Data_Kelmarsh_5_*.csv
- Turbine_Data_Kelmarsh_6_*.csv

Put CERRA files into raw_cerra

- CERRA_50m_KWF1.csv to CERRA_50m_KWF6.csv
- CERRA_75m_KWF1.csv to CERRA_75m_KWF6.csv
- CERRA_100m_KWF1.csv to CERRA_100m_KWF6.csv

---

## Data preparation

You can build the dataset in two ways.

### Option A SCADA plus CERRA

Run

python tools/prep_scada_cerra_kelmarsh.py

This will write the following files into data/wind

- X.npy normalized input features
- Y.npy normalized target power
- mask.npy availability mask for target power
- adj.npy adjacency matrix for 6 turbines
- meta.json feature names, normalization statistics, split indices, and the L and H settings

### Option B SCADA only

Run

python tools/prep_scada_only_kelmarsh.py

---

## Dataset and loader

The dataset class is in datasets/wind_dataset.py and returns one sample as

- x is the past 24 hours with all turbines and all features
- y is the next 6 hours power for all turbines
- m is the next 6 hours mask for all turbines

A training batch has

- x as batch by history by nodes by features
- y as batch by horizon by nodes
- m as batch by horizon by nodes

The time splits are sequential

- train is first 80 percent of the timeline
- val is next 10 percent
- test is last 10 percent

---

## Train AGCRN minimal script

Run

python tools/train_agcrn_min.py

Training uses

- masked MAE as the loss
- masked MAE and masked RMSE for validation
- early stopping with patience 15

The best checkpoint is saved to

- output/agcrn_min_best.pt

Then the script loads the best checkpoint and prints test MAE and RMSE.

---

## Other models

You can train other baselines with the provided scripts

- tools/train_mtgnn_v058.py
- tools/train_dgcrn_v058.py
- tools/train_d2stgnn_v058.py
- tools/train_lstm_stf_v001.py
- tools/train_agcrn_v058.py
- tools/train_agcrn_mtgnn_head_v001.py

Ensembling scripts

- tools/ensemble_agcrn_mtgnn_v001.py
- tools/ensemble_agcrn_mtgnn_v002.py

Metrics and plots

- tools/cal_mae_rmse.py
- tools/predict_plot_*.py
