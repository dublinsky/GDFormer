# GDFormer
Graph Diffusing trans-Former for traffic flow prediction. This repo. contains the data set and code for our paper submitted to Pattern Recognition Letters. Overview of this model is shown in the following figure, it is a sequence to sequence architecture, both of the encoders and decoders are constitued by the nodel designed GDA module and the auxiliaries.
![architecture](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/2664024110872.png)
The internal instructure of the encoder and the decoder are highlighted in the following figures,
![enc](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/2152128129298.png =357x)
![dec](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/3572128117165.png =357x)
Now we show how to reproduce the results shown in our paper. 
## 1. environment description
Hardware: Intel Core i5-7400, NVIDIA GeForce RTX 3080TI with 12G GPU RAM.
Software: Ubuntu 20.04 LTS, anaconda 4.10.1, the required packages installed is listed in ggp.yml contained in this folder
## 2. How to run it
The data sets for this paper, well processed METR-LA and PeMS03 has been contained in the "Datasets" folder.
When the required OS and anaconda have been installed, run the following command in the console,
```
conda env create -f ggp.yml
```
then activate the environment,
```
conda activate ggp
```
then run the following command,
```
python Train.py
```
you will obtain the outputs like this,
```
Training set: x with shape torch.Size([20556, 12, 207, 1]), y with shape torch.Size([20556, 13, 207, 1])
Valuational set: x with shape torch.Size([3426, 12, 207, 1]), y with shape torch.Size([3426, 13, 207, 1])
Testing set: x with shape torch.Size([10278, 12, 207, 1]), y with shape torch.Size([10278, 13, 207, 1])
Time 2021-08-31 09:56:29 | Loaded dataset >> METR
Time 2021-08-31 09:56:29 | Builded Model
Time 2021-08-31 09:57:34 Training 0 | training loss:  2.78823
Predicting Flow at 0 | valational MAPE:  5.863%, MAE:  2.51990, RMSE:  4.45442
Time 2021-08-31 09:58:39 Training 1 | training loss:  2.55756
Predicting Flow at 0 | valational MAPE:  5.698%, MAE:  2.40975, RMSE:  4.35384
... ...
```
Please note that the current setting is for the METR-LA data set, if you want to obtain that for the PeMS03 data set, you can open the file Train.py, and change the second last line, i.e., change 
```
config = METR_config
```
as
```
config = PeMS_config
```
and run the following command,
```
python Train.py
```
The corresponding output is given by,
```
Training set: x with shape torch.Size([5350, 12, 555, 1]), y with shape torch.Size([5350, 13, 555, 1])
Valuational set: x with shape torch.Size([892, 12, 555, 1]), y with shape torch.Size([892, 13, 555, 1])
Testing set: x with shape torch.Size([2674, 12, 555, 1]), y with shape torch.Size([2674, 13, 555, 1])
Time 2021-08-31 10:34:48 | Loaded dataset >> PeMS
Time 2021-08-31 10:34:48 | Builded Model
Time 2021-08-31 10:35:40 Training 0 | training loss:  19.62505
Predicting Flow at 0 | valational MAPE:  14.753%, MAE:  15.57753, RMSE:  24.87848
... ...
```
## 3. Visualization of the results
We have frozen the trained model, "METR.pt" and "PeMS.pt" in this folder.
Utilizing the following commnad to obtain the evaluation result,
```
python getMetrics.py
```
Then we will have "PeMS_enc_adj.npy" and "PeMS_dec_adj.npy" in "Datasets/PeMS/" folder and "METR_enc_adj.npy" and "METR_enc_adj.npy" in "Datasets/METR/" folder.
Run the following command to obtain the dynamic refreshing adjacency matrix visualization,
```
python getRes.py
```
The results of PeMS03 data set is given by,
![METR_0step_enc_adj](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/5931821150189.png =430x)
![METR_5step_enc_adj](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/5880723147793.png =430x)
![METR_0step_dec_adj](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/5887954130000.png =430x)
![METR_5step_dec_adj](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/76356126555.png =430x)
The results of METR-LA data set is given by,
![PeMS_0step_enc_adj](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/3316340145295.png =430x)
![METR_5step_enc_adj](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/1005341126536.png =430x)
![METR_0step_dec_adj](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/5198541148976.png =430x)
![METR_5step_dec_adj](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/1420742144112.png =430x)
To visualize the traffic flow prediction of randomly chosen sensors at randomly truncated time ticks, run the following command,
```
python getRes.py
```
The results of PeMS03 data set is given by,
![PeMS_443th_node_start_at667_flow](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/410056137658.png =430x)
![PeMS_539th_node_start_at1594_flow](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/3489556130792.png =430x)
The results of METR-LA data set is given by,
![METR_74th_node_start_at5301_flow](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/4479958121322.png =430x)
![METR_195th_node_start_at6096_flow](https://raw.githubusercontent.com/dublinsky/GDFormer/main/Images/5752858123826.png =430x)
