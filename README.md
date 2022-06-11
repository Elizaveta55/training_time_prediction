## Overtraining predictor

Structure:
- `predictor.py` - main program what monitors updates and predict overtraining. It return number of epoch what is predicted as a start of overtraining.
- `utils.py` describes some general def's.
- `config.yaml` states all variables and its values.
- `requirements.txt` lists all requirements to install.
- folder `models` containes saved checkpoints from valid and train LSTM models for OCR and NER logs.

file `predictor.py` contains implementation of train predictor. It monitors updates in log file (stated in `config.file_name`) and predict overtraining by exploring train and valid difference.

#### Programs outputs

There are general program output:

```
Processed [20/40], prediction building based on processed data for [40/60] ## no overtraining predicted
Processed [40/60], prediction building based on processed data for [60/80]
Overtraining is predicted on epoch 62 with probability 50.0
Processed [60/80], prediction building based on processed data for [80/100]
Overtraining is predicted on epoch 80 with probability 66.66666666666666
Processed [80/100], prediction building based on processed data for [100/120]
Overtraining is predicted on epoch 100 with probability 75.0
Processed [100/120], prediction building based on processed data for [120/140] ## program predicts start of overtraining, therefore it hasn't changed
Overtraining is predicted on epoch 100 with probability 60.0
Processed [120/140], prediction building based on processed data for [140/160]
Overtraining is predicted on epoch 100 with probability 50.0

```

Conclusion: Epochs 10000 (according to input file format) is predicted as an epoch from where overtraining has started.

#### Plots 

###### look_back

There is a trade-off: var "look_back" stands for amount of points we consider for 1 prediction. For example, for 150 epochs and look_back=10, the trainset will contain 140 instances: [t0, t1, t2,..., t9] -> t10, [t1, t2, t3,..., t10] -> t11, ... . It is benefitial to have lower value of look_back because there will be more instances in trainset. At the same time higher value of look_back gives better performance. Therefore one experiment was conducted to choose best look_back.

1. Train loss: OCR, look_back=10
![train_full](/images/ocr_smooth_train_closer_look_50_look_back_10.png)
2. Valid loss: OCR,  look_back=10
![valid_full](/images/ocr_smooth_valid_closer_look_50_look_back_10.png)
3. Train loss: OCR,  look_back=20
![train_short](/images/ocr_smooth_train_closer_look_50_look_back_20.png)
4. Valid loss: OCR,  look_back=20
![valid_short](/images/ocr_smooth_valid_closer_look_50_look_back_20.png)
5. Train loss: OCR,  look_back=30
![train_short](/images/ocr_smooth_train_closer_look_50_look_back_30.png)
6. Valid loss: OCR,  look_back=30
![valid_short](/images/ocr_smooth_valid_closer_look_50_look_back_30.png)

Conclusion: look_back should be chosen carefully, for our case look_back=30 is appropriate.


###### Smoothing technique

In order to keep look_back as low as possible without information loss smoothing technique was applied on the data.

1. Train loss: no smoothing, look_back=50
![train_full](/images/final_train_closer_look.png)
2. Valid loss: no smoothing,  look_back=50
![valid_full](/images/final_valid_closer_look.png)
3. Train loss: smoothing,  look_back=30
![train_short](/images/ocr_smooth_train_closer_look_50_look_back_30.png)
4. Valid loss: smoothing,  look_back=30
![valid_short](/images/ocr_smooth_valid_closer_look_50_look_back_30.png)

Conclusion: it is quite justifiable to use the smoothing technique in this case because it is observable that data has too intensive fluctations. Look_back=50 is too high price for maintaining its waves.


## Double LSTM

This part of implementation is about single LSTM model what has [train_loss_array, valid_loss_array] as an input and returns [predicted_train_value, predicted_valid_value]. 

#### Programs outputs

There are general program output:

```
Processed [20/40], prediction building based on processed data for [40/60]
Processed [40/60], prediction building based on processed data for [60/80]
Processed [60/80], prediction building based on processed data for [80/100]
Overtraining is predicted on epoch 80 with probability 33.33333333333333
Processed [80/100], prediction building based on processed data for [100/120]
Overtraining is predicted on epoch 80 with probability 25.0
Processed [100/120], prediction building based on processed data for [120/140]
Overtraining is predicted on epoch 80 with probability 20.0
Processed [120/140], prediction building based on processed data for [140/160]
Overtraining is predicted on epoch 80 with probability 16.666666666666664

```

Conclusion: Epochs 8000 (according to input file format) is predicted as an epoch from where overtraining has started.

#### Prediction curves

Dependency of look_back and prediction curve were explored:
1. Train loss: OCR, look_back=10
![train_full](/images/double_train_look_back_10.png)
2. Valid loss: OCR,  look_back=10
![valid_full](/images/double_valid_look_back_10.png)
3. Train loss: OCR,  look_back=20
![train_short](/images/double_train_look_back_20.png)
4. Valid loss: OCR,  look_back=20
![valid_short](/images/double_valid_look_back_20.png)
5. Train loss: OCR,  look_back=30
![train_short](/images/double_train_look_back_30.png)
6. Valid loss: OCR,  look_back=30
![valid_short](/images/double_valid_look_back_30.png)
7. Train loss: OCR, look_back=40
![train_full](/images/double_train_look_back_40.png)
8. Valid loss: OCR,  look_back=40
![valid_full](/images/double_valid_look_back_40.png)
9. Train loss: OCR,  look_back=50
![train_short](/images/double_train_look_back_50.png)
10. Valid loss: OCR,  look_back=50
![valid_short](/images/double_valid_look_back_50.png)

Conclusion: the most optimal trade-off is within look_back=40.


## Updated data

#### Programs outputs (SIMPLE)

There are general program output:

```
Processed [20/40], prediction building based on processed data for [40/60]
Processed [40/60], prediction building based on processed data for [60/80]
Overtraining is predicted on epoch 60 with probability 100.0
Processed [60/80], prediction building based on processed data for [80/100]
Overtraining is predicted on epoch 80 with probability 100.0
Processed [80/100], prediction building based on processed data for [100/120]
Overtraining is predicted on epoch 80 with probability 75.0
Processed [100/120], prediction building based on processed data for [120/140]
Overtraining is predicted on epoch 80 with probability 60.0
Processed [120/140], prediction building based on processed data for [140/160]
Overtraining is predicted on epoch 80 with probability 50.0
Processed [140/160], prediction building based on processed data for [160/180]
Overtraining is predicted on epoch 80 with probability 42.857142857142854
```
Conclusion: overtraining is predicted on epoch 8000 (according to input file format)

#### Prediction curve (DOUBLE)

1. Train loss: OCR, look_back=30
![train_full](/images/new_data_train.png)
2. Valid loss: OCR,  look_back=30
![valid_full](/images/new_data_valid.png)

#### Programs outputs (DOUBLE)

There are general program output:

```
Processed [20/40], prediction building based on processed data for [40/60]
Processed [40/60], prediction building based on processed data for [60/80]
Processed [60/80], prediction building based on processed data for [80/100]
Processed [80/100], prediction building based on processed data for [100/120]
Processed [100/120], prediction building based on processed data for [120/140]
Processed [120/140], prediction building based on processed data for [140/160]
Processed [140/160], prediction building based on processed data for [160/180]
```
Conclusion: no overtraining predicted

#### Prediction curve (DOUBLE)

1. Train loss: OCR, look_back=40
![train_full](/images/new_data_doubled_train.png)
2. Valid loss: OCR,  look_back=40
![valid_full](/images/new_data_doubled_valid.png)

##### Conclusion

Double model has better prediction. Simple model predicted overtraining on 80 (or 8000 according to input file format) epoch because of local val loss increase which did not last long but affected the overall prediction. It didn't changed afterwards because it is the strongest prediction. Therefore there is a necesity to update overtraining prediction logic for such outliers.

1. Train loss: Simple, look_back=40
![train_full](/images/90_train.png)
2. Valid loss: Simple,  look_back=40
![valid_full](/images/90_valid.png)

## Prediction logics

In order to identify underfitting or overfitting array a[0..n] of difference between VALID and TRAIN loss for every timestamp(=epoch) will be collected. The values should be normed related real loss values, therefore two arrays are considered:
**Input data**: a[i] = (y_valid[i] - y_train[i]) - difference of losses for every timestamp, mean[i] = mean(y_valid[i], y_train[i]) - mean between losses for every timestamp

Possible cases:
1. Normal behaviour:
    1. Train ~ Valid, valid is slightly bigger => 1) a[i]->0 for i->n-1, 2) a[i]>=0 with some error, 3) mean[i] doesn't increase/decrease suddenly for i->n-1
    2. Valid approximate to train within the training => 1) a[i]->0 for i->n-1
    3. Both losses are decreasing => 1) a[i]->0 and mean[i]->0 for i->n-1 SMOOTHLY
2. Overfitting:
    1. Difference between losses is getting higher => 1) a[i]-> +-infinity for i->n-1
3. Underfitting:
    1. Increasing training => 1) a[i]-> -infinity and mean[i]-> +infinity for i->n-1
    2. Sudden synchronous dip => 1) mean[i]-> 0 too sudden and a[i]->0 for i->n-1
    3. Valid significantly higher. It doesn't reduce over the training time => 1) a[i] !-> 0 for i->n-1
4. Unrepresentative Train Dataset:
    1. Both train and valid curves improves, but there are gaps between them. Valid is bigger => 1) a[i] changes sign, has intensively high values
5. Unrepresentative Validation Dataset:
    1. Train curve is good, but valid curve shows noisy movements around the training loss => 1) a[i] intensively changes sign, has high values 
    2. Valid curve is much lower -> 1) a[i] !-> 0 and a[i]<0 with some error


## Approaches to Predicting time and epoch

There are two applied approaches: LSTM and ARIMA. LSTM approach is implemented with deep learning model. It has to wait for considerable amount of epochs data in order to be trained to start predicting. Previous experiments showed the most optimal initial value of epochs for LSTM training is 40. It means we have to wait 40 epochs to get any prediction. Moreover, this model is not reliable much: it is teached on highly volatile data (becouse behaviour of first epochs is highly unstable) and will be reteached in 40 epoches, therefore first more reliable predictions will be given in 80 epochs. Apparantly it is not wise to use this heavy model because the trade-off is not beneficial for us, moreover, there is no way to generalize the prediction (unless we have diverse dataset of different training and valied losses)

Another approach is ARIMA - statisctcal/mathematical approach - it uses interpolation formula in order to predict the time-series. This approach can be implemented from the first epochs with relatively high reliability of prediction. It does not require any training therefore it is fast.

Disadvantage of both approaches is the usage of smoothing curves technique in order to stabilize the prediction. It will affect the prediction of earlystopping.

Next sections represent the obtained result and its plots to comapare one approach to another.

### Predicting time and epoch. LSTM model



#### File: "log_train_new1.txt" gives the following prediction:
```
Processed [20/40], prediction building based on processed data for [40/60]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0, in 88.08719
----------------------------------------------------------------
Processed [40/60], prediction building based on processed data for [60/80]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [60/80], prediction building based on processed data for [80/100]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 32, in 18062.26271
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [80/100], prediction building based on processed data for [100/120]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 32, in 18062.26271
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [100/120], prediction building based on processed data for [120/140]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 37, in 21077.10269 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
```

1. Losses, epoch 40: 
![train_full](/images/train_2_40.png)
2. Losses, epoch 80: 
![train_full](/images/train_2_80.png)
3. Losses, epoch 120: 
![train_full](/images/train_2_120.png)

#### File: "log_train_new2.txt" gives the following prediction:
```
Processed [20/40], prediction building based on processed data for [40/60]
Train Prediction on given data
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0, in 45.51252 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0, in 45.51252 ms
----------------------------------------------------------------
Processed [40/60], prediction building based on processed data for [60/80]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [60/80], prediction building based on processed data for [80/100]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 28, in 20297.18923 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [80/100], prediction building based on processed data for [100/120]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 31, in 22813.50261 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [100/120], prediction building based on processed data for [120/140]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 38, in 27500.27101 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------

```

1. Losses, epoch 40: 
![train_full](/images/train_1_40.png)
2. Losses, epoch 80: 
![train_full](/images/train_1_80.png)
3. Losses, epoch 120: 
![train_full](/images/train_1_120.png)

#### File: "log_train_new3.txt" gives the following prediction:
```
Processed [20/40], prediction building based on processed data for [40/60]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [40/60], prediction building based on processed data for [60/80]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [60/80], prediction building based on processed data for [80/100]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [80/100], prediction building based on processed data for [100/120]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [100/120], prediction building based on processed data for [120/140]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [120/140], prediction building based on processed data for [140/160]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 49, in 24468.38095 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [140/160], prediction building based on processed data for [160/180]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 49, in 24468.38095 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [160/180], prediction building based on processed data for [180/200]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 55, in 27894.95292 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [180/200], prediction building based on processed data for [200/220]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 64, in 33017.58103 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
```
1. Losses, epoch 50: 
![train_full](/images/train_3_50.png)
2. Losses, epoch 100: 
![train_full](/images/train_3_100.png)
3. Losses, epoch 150: 
![train_full](/images/train_3_150.png)
4. Losses, epoch 200: 
![train_full](/images/train_3_200.png)


#### File: "log_train_new4.txt" gives the following prediction:
```
Processed [20/40], prediction building based on processed data for [40/60]
Train Prediction on given data
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0, in 13.86965 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0, in 13.86965 ms
----------------------------------------------------------------
Processed [40/60], prediction building based on processed data for [60/80]
Train Prediction on given data
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0, in 13.86965 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0, in 13.86965 ms
----------------------------------------------------------------
Processed [60/80], prediction building based on processed data for [80/100]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [80/100], prediction building based on processed data for [100/120]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [100/120], prediction building based on processed data for [120/140]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [120/140], prediction building based on processed data for [140/160]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 47, in 31795.43487 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------

```

1. Losses, epoch 40: 
![train_full](/images/train_4_40.png)
2. Losses, epoch 80: 
![train_full](/images/train_4_80.png)
3. Losses, epoch 120: 
![train_full](/images/train_4_120.png)



#### File: "log_train_new5.txt" gives the following prediction:
```
Processed [20/40], prediction building based on processed data for [40/60]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [40/60], prediction building based on processed data for [60/80]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [60/80], prediction building based on processed data for [80/100]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [80/100], prediction building based on processed data for [100/120]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [100/120], prediction building based on processed data for [120/140]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [120/140], prediction building based on processed data for [140/160]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [140/160], prediction building based on processed data for [160/180]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [160/180], prediction building based on processed data for [180/200]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [180/200], prediction building based on processed data for [200/220]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [200/220], prediction building based on processed data for [220/240]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
```

1. Losses, epoch 50: 
![train_full](/images/train_5_50.png)
2. Losses, epoch 100: 
![train_full](/images/train_5_100.png)
3. Losses, epoch 150: 
![train_full](/images/train_5_150.png)
4. Losses, epoch 200: 
![train_full](/images/train_5_200.png)


#### File: "log_train_newu.txt" gives the following prediction:
```
Processed [20/40], prediction building based on processed data for [40/60]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [40/60], prediction building based on processed data for [60/80]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [60/80], prediction building based on processed data for [80/100]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [80/100], prediction building based on processed data for [100/120]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [100/120], prediction building based on processed data for [120/140]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [120/140], prediction building based on processed data for [140/160]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
```

1. Losses, epoch 50: 
![train_full](/images/arima_u_50.png)
2. Losses, epoch 100: 
![train_full](/images/arima_u_100.png)
3. Losses, epoch 150: 
![train_full](/images/arima_u_150.png)



#### File: "log_train_newx.txt" gives the following prediction:
```
Processed [20/40], prediction building based on processed data for [40/60]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [40/60], prediction building based on processed data for [60/80]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [60/80], prediction building based on processed data for [80/100]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [80/100], prediction building based on processed data for [100/120]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [100/120], prediction building based on processed data for [120/140]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [120/140], prediction building based on processed data for [140/160]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [140/160], prediction building based on processed data for [160/180]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [160/180], prediction building based on processed data for [180/200]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [180/200], prediction building based on processed data for [200/220]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [200/220], prediction building based on processed data for [220/240]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
```

1. Losses, epoch 50: 
![train_full](/images/arima_x_50.png)
2. Losses, epoch 100: 
![train_full](/images/arima_x_100.png)
3. Losses, epoch 150: 
![train_full](/images/arima_x_150.png)
4. Losses, epoch 200: 
![train_full](/images/arima_x_200.png)






### Predicting time and epoch, ARIMA model


#### File: "log_train_new1.txt" gives the following prediction:
```
Processed [0/20], prediction building based on processed data for [20/40]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 88.08719 ms
----------------------------------------------------------------
Processed [20/40], prediction building based on processed data for [40/60]
Train Prediction on given data
1Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 88.08719 ms
----------------------------------------------------------------
Processed [40/60], prediction building based on processed data for [60/80]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 3500 epoch, in 19837.29555 ms
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 88.08719 ms
----------------------------------------------------------------
Processed [60/80], prediction building based on processed data for [80/100]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 3200 epoch, in 18062.26271 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 3700 epoch, in 21077.10269 ms
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 88.08719 ms
Predicted early stopping 10200 epoch, in 70595.55797831995
----------------------------------------------------------------
Processed [80/100], prediction building based on processed data for [100/120]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 3200 epoch, in 18062.26271 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 88.08719 ms
----------------------------------------------------------------
Processed [100/120], prediction building based on processed data for [120/140]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 3700 epoch, in 21077.10269 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 88.08719 ms
----------------------------------------------------------------
Processed [120/140], prediction building based on processed data for [140/160]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 88.08719 ms
----------------------------------------------------------------
```

1. Losses, epoch 40: 
![train_full](/images/arima_1_40.png)
2. Losses, epoch 80: 
![train_full](/images/arima_1_80.png)
3. Losses, epoch 120: 
![train_full](/images/arima_1_120.png)

#### File: "log_train_new2.txt" gives the following prediction:
```
Processed [0/20], prediction building based on processed data for [20/40]
Train Prediction on given data
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 45.51252 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [20/40], prediction building based on processed data for [40/60]
Train Prediction on given data
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 45.51252 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [40/60], prediction building based on processed data for [60/80]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 3100 epoch, in 22813.50261 ms
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 45.51252 ms
----------------------------------------------------------------
Processed [60/80], prediction building based on processed data for [80/100]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 2800 epoch, in 20297.18923 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [80/100], prediction building based on processed data for [100/120]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 3100 epoch, in 22813.50261 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 45.51252 ms
Predicted early stopping 12000 epoch, in 98862.6056731285
----------------------------------------------------------------
Processed [100/120], prediction building based on processed data for [120/140]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 3800 epoch, in 27500.27101 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 45.51252 ms
----------------------------------------------------------------
```

1. Losses, epoch 40: 
![train_full](/images/arima_2_40.png)
2. Losses, epoch 80: 
![train_full](/images/arima_2_80.png)
3. Losses, epoch 120: 
![train_full](/images/arima_2_120.png)

#### File: "log_train_new3.txt" gives the following prediction:
```
Processed [0/20], prediction building based on processed data for [20/40]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 20.02098 ms
----------------------------------------------------------------
Processed [20/40], prediction building based on processed data for [40/60]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 20.02098 ms
----------------------------------------------------------------
Processed [40/60], prediction building based on processed data for [60/80]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [60/80], prediction building based on processed data for [80/100]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 20.02098 ms
----------------------------------------------------------------
Processed [80/100], prediction building based on processed data for [100/120]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 20.02098 ms
----------------------------------------------------------------
Processed [100/120], prediction building based on processed data for [120/140]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 5100 epoch, in 25705.59911 ms
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 20.02098 ms
----------------------------------------------------------------
Processed [120/140], prediction building based on processed data for [140/160]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 4900 epoch, in 24468.38095 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 6100 epoch, in 31297.20435 ms
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 20.02098 ms
----------------------------------------------------------------
Processed [140/160], prediction building based on processed data for [160/180]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 4900 epoch, in 24468.38095 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 6100 epoch, in 31297.20435 ms
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 20.02098 ms
Predicted early stopping 18100 epoch, in 108668.08896134941
----------------------------------------------------------------
Processed [160/180], prediction building based on processed data for [180/200]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 5500 epoch, in 27894.95292 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 20.02098 ms
----------------------------------------------------------------
Processed [180/200], prediction building based on processed data for [200/220]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 6400 epoch, in 33017.58103 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 20.02098 ms
----------------------------------------------------------------
```
1. Losses, epoch 50: 
![train_full](/images/arima_3_50.png)
2. Losses, epoch 100: 
![train_full](/images/arima_3_100.png)
3. Losses, epoch 150: 
![train_full](/images/arima_3_150.png)
4. Losses, epoch 200: 
![train_full](/images/arima_3_200.png)


#### File: "log_train_new4.txt" gives the following prediction:
```
Processed [0/20], prediction building based on processed data for [20/40]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 13.86965 ms
----------------------------------------------------------------
Processed [20/40], prediction building based on processed data for [40/60]
Train Prediction on given data
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 13.86965 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 13.86965 ms
----------------------------------------------------------------
Processed [40/60], prediction building based on processed data for [60/80]
Train Prediction on given data
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 13.86965 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 13.86965 ms
----------------------------------------------------------------
Processed [60/80], prediction building based on processed data for [80/100]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 13.86965 ms
----------------------------------------------------------------
Processed [80/100], prediction building based on processed data for [100/120]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 13.86965 ms
----------------------------------------------------------------
Processed [100/120], prediction building based on processed data for [120/140]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 6000 epoch, in 42752.01402 ms
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 13.86965 ms
----------------------------------------------------------------
Processed [120/140], prediction building based on processed data for [140/160]
Train Prediction on given data
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 4700 epoch, in 31795.43487 ms
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 2
Losses are having synchronously dip
Is predicted on epoch 5500 epoch, in 37176.80114 ms
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 13.86965 ms
----------------------------------------------------------------
```

1. Losses, epoch 40: 
![train_full](/images/arima_4_40.png)
2. Losses, epoch 80: 
![train_full](/images/arima_4_80.png)
3. Losses, epoch 120: 
![train_full](/images/arima_4_120.png)



#### File: "log_train_new5.txt" gives the following prediction:
```
Processed [0/20], prediction building based on processed data for [20/40]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 23.01907 ms
----------------------------------------------------------------
Processed [20/40], prediction building based on processed data for [40/60]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 23.01907 ms
----------------------------------------------------------------
Processed [40/60], prediction building based on processed data for [60/80]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 23.01907 ms
----------------------------------------------------------------
Processed [60/80], prediction building based on processed data for [80/100]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 23.01907 ms
----------------------------------------------------------------
Processed [80/100], prediction building based on processed data for [100/120]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 23.01907 ms
----------------------------------------------------------------
Processed [100/120], prediction building based on processed data for [120/140]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 23.01907 ms
Predicted early stopping 13900 epoch, in 88085.18965265594
----------------------------------------------------------------
Processed [120/140], prediction building based on processed data for [140/160]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 23.01907 ms
Predicted early stopping 15900 epoch, in 101843.57151366504
----------------------------------------------------------------
Processed [140/160], prediction building based on processed data for [160/180]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 23.01907 ms
Predicted early stopping 17800 epoch, in 117332.98754638033
----------------------------------------------------------------
Processed [160/180], prediction building based on processed data for [180/200]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [180/200], prediction building based on processed data for [200/220]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [200/220], prediction building based on processed data for [220/240]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
```

1. Losses, epoch 50: 
![train_full](/images/arima_5_50.png)
2. Losses, epoch 100: 
![train_full](/images/arima_5_100.png)
3. Losses, epoch 150: 
![train_full](/images/arima_5_150.png)
4. Losses, epoch 200: 
![train_full](/images/arima_5_200.png)


#### File: "log_train_newu.txt" gives the following prediction:
```
Processed [0/20], prediction building based on processed data for [20/40]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [20/40], prediction building based on processed data for [40/60]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [40/60], prediction building based on processed data for [60/80]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 134.25785 ms
----------------------------------------------------------------
Processed [60/80], prediction building based on processed data for [80/100]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 134.25785 ms
----------------------------------------------------------------
Processed [80/100], prediction building based on processed data for [100/120]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 134.25785 ms
----------------------------------------------------------------
Processed [100/120], prediction building based on processed data for [120/140]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 134.25785 ms
----------------------------------------------------------------
Processed [120/140], prediction building based on processed data for [140/160]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 134.25785 ms
----------------------------------------------------------------
```

1. Losses, epoch 40: 
![train_full](/images/arima_u_40.png)
2. Losses, epoch 80: 
![train_full](/images/arima_u_80.png)
3. Losses, epoch 120: 
![train_full](/images/arima_u_120.png)



#### File: "log_train_newx.txt" gives the following prediction:
```
Processed [0/20], prediction building based on processed data for [20/40]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [20/40], prediction building based on processed data for [40/60]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [40/60], prediction building based on processed data for [60/80]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 85.98034 ms
----------------------------------------------------------------
Processed [60/80], prediction building based on processed data for [80/100]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 85.98034 ms
----------------------------------------------------------------
Processed [80/100], prediction building based on processed data for [100/120]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 85.98034 ms
----------------------------------------------------------------
Processed [100/120], prediction building based on processed data for [120/140]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [120/140], prediction building based on processed data for [140/160]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 85.98034 ms
----------------------------------------------------------------
Processed [140/160], prediction building based on processed data for [160/180]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 85.98034 ms
Predicted early stopping 17700 epoch, in 305345.28604617575
----------------------------------------------------------------
Processed [160/180], prediction building based on processed data for [180/200]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [180/200], prediction building based on processed data for [200/220]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Training is active....
----------------------------------------------------------------
Processed [200/220], prediction building based on processed data for [220/240]
Train Prediction on given data
Training is active....
----------------------------------------------------------------
Train Prediction on predicted data:
Underfitting. 3
Losses are not reducing
Is predicted on epoch 0 epoch, in 85.98034 ms
----------------------------------------------------------------

```

1. Losses, epoch 40: 
![train_full](/images/arima_x_40.png)
2. Losses, epoch 80: 
![train_full](/images/arima_x_80.png)
3. Losses, epoch 120: 
![train_full](/images/arima_x_120.png)
4. Losses, epoch 160: 
![train_full](/images/arima_x_160.png)
5. Losses, epoch 200: 
![train_full](/images/arima_x_200.png)




### Solo early stopping
#### Arima 

##### log_1
```
[Predicted data]: predicted earlystopping(patience=20): 10200 epoch, in 70595.55797831995 from the start
```
Actual stop - 14900

##### log_2
```
[Predicted data]: predicted earlystopping(patience=20): 12000 epoch, in 98862.6056731285 from the start
```
Actual stop - 13300

##### log_3
```
[Predicted data]: predicted earlystopping(patience=20): 18100 epoch, in 108668.08896134941 from the start
```
Actual stop - 20300

##### log_4
```

```
Actual stop - 15100

##### log_5
```
[Predicted data]: predicted earlystopping(patience=20): 13900 epoch, in 88085.18965265594 from the start
[Predicted data]: predicted earlystopping(patience=20): 15900 epoch, in 101843.57151366504 from the start
[Predicted data]: predicted earlystopping(patience=20): 17800 epoch, in 117332.98754638033 from the start
```
Actual stop - 22800

##### log_6
```

```
Actual stop - NA

##### log_7
```
[Predicted data]: predicted earlystopping(patience=20): 17700 epoch, in 305345.28604617575
```
Actual stop - NA


##### log_8 - 1 million 
```
[Predicted data]: predicted earlystopping(patience=20): 31500 epoch, in 6 days, 11:27:09.914673
```
Actual stop - 30700


#### Analysis

ARIMA is not precise, it does not reflect wavy nature of data.

**Hypothesis 1** is based on changing the params of ARIMA

Auto Regressive Integrated Moving Average(ARIMA) — It is like a liner regression equation where the predictors depend on parameters (p,d,q) of the ARIMA model.
- p : This is the number of AR (Auto-Regressive) terms. Example — if p is 3 the predictor for y(t) will be y(t-1),y(t-2),y(t-3).
- q : This is the number of MA (Moving-Average) terms.
- d :This is the number of differences or the number of non-seasonal differences .

_Results_: predictions are highly unstable, apart from (1,1,1) params. The reason in the nature of data - it does not really have any seasons and bold trends.

**Hypothesis 2** should neglect negative performance of Hypothesis 1. 
We will explore only 80% of data to reduce the contribution of high variability of initial values of losses.

_Result_: instability, no improvements

Nevertheless, **ARIMA still performed better then LSTM**

**Alternative Approach**

[Comet.ml](https://www.comet.ml/site/predictive-early-stopping/) platform suggest a diverse range of tools for machine learning including prediction of early stopping. There are many tools available for academic purposes for free, but prediction of early stopping is only paid-based.

### Latest version

ARIMA is chosen to be a base model. Model has many parameters listed in ```config.yaml``` file, the base algorithm is: wait until 40 entities(or epochs) won't be processed, then start ARIMA prediction every 20 epochs as well as prediction of early stopping.

Prediction of early stopping for every new set of losses (set One stands for 40 epochs, set Two stands for 40+20 epochs, set Three stands for 40+20+20 epochs, etc) returns predicted epoch up to nearest 200 epochs.

There are two types of data used for predictions of early stopping: real or given data (the data we read from the file) and real data with predicted nearest 40 epochs (as the most possibly reliable data for future prediction). These two cases marked as _[Given data]_ and _[Predicted data]_

Criteria for early stopping:
1. Increasing Valid loss, marked as _INCREASING VALID_
2. Increasing difference between valid loss and train loss, particularly increasing valid loss while train loss is not that rapid, marked as _INCREASING VALID-train_

In the following sections outputs is provided for the data-files: ```log_train_1m.txt```, ```log_train_newx.txt```, ```log_train_newu.txt``` (latest files we have obtained fron NER training).

#### File log_train_1m.txt:

```
No early stopping is detected on 64000 near epochs, considering first 4000 epochs, on real data
No early stopping is detected on 68000 near epochs, considering first 8000 epochs, on real + predicted data
No early stopping is detected on 86000 near epochs, considering first 6000 epochs, on real data
No early stopping is detected on 90000 near epochs, considering first 10000 epochs, on real + predicted data
No early stopping is detected on 108000 near epochs, considering first 8000 epochs, on real data
No early stopping is detected on 112000 near epochs, considering first 12000 epochs, on real + predicted data
No early stopping is detected on 130000 near epochs, considering first 10000 epochs, on real data
No early stopping is detected on 134000 near epochs, considering first 14000 epochs, on real + predicted data
No early stopping is detected on 152000 near epochs, considering first 12000 epochs, on real data
No early stopping is detected on 156000 near epochs, considering first 16000 epochs, on real + predicted data
No early stopping is detected on 174000 near epochs, considering first 14000 epochs, on real data
     detected:  INCREASING TRAIN-VALID
[Predicted data]: predicted earlystopping(patience=20): 17900 epoch, in 3 days, 13:48:21.852680
No early stopping is detected on 196000 near epochs, considering first 16000 epochs, on real data
     detected:  INCREASING TRAIN-VALID
[Predicted data]: predicted earlystopping(patience=20): 19700 epoch, in 3 days, 22:56:13.457071
No early stopping is detected on 218000 near epochs, considering first 18000 epochs, on real data
     detected:  INCREASING TRAIN-VALID
[Predicted data]: predicted earlystopping(patience=20): 21800 epoch, in 4 days, 9:17:51.145587
No early stopping is detected on 240000 near epochs, considering first 20000 epochs, on real data
     detected:  INCREASING TRAIN-VALID
[Predicted data]: predicted earlystopping(patience=20): 23800 epoch, in 4 days, 19:51:09.610220
No early stopping is detected on 262000 near epochs, considering first 22000 epochs, on real data
No early stopping is detected on 266000 near epochs, considering first 26000 epochs, on real + predicted data
No early stopping is detected on 284000 near epochs, considering first 24000 epochs, on real data
No early stopping is detected on 288000 near epochs, considering first 28000 epochs, on real + predicted data
No early stopping is detected on 306000 near epochs, considering first 26000 epochs, on real data
No early stopping is detected on 310000 near epochs, considering first 30000 epochs, on real + predicted data
No early stopping is detected on 328000 near epochs, considering first 28000 epochs, on real data
No early stopping is detected on 332000 near epochs, considering first 32000 epochs, on real + predicted data
No early stopping is detected on 350000 near epochs, considering first 30000 epochs, on real data
     detected:  INCREASING TRAIN-VALID
[Predicted data]: predicted earlystopping(patience=20): 33700 epoch, in 6 days, 22:31:52.858304
```

1. Losses, epoch 40: 
![train_full](/images/1m_40.png)
2. Losses, epoch 80: 
![train_full](/images/1m_80.png)
3. Losses, epoch 120: 
![train_full](/images/1m_120.png)
4. Losses, epoch 160: 
![train_full](/images/1m_160.png)
5. Losses, epoch 200: 
![train_full](/images/1m_200.png)
6. Losses, epoch 240: 
![train_full](/images/1m_280.png)
7. Losses, epoch 280: 
![train_full](/images/1m_320.png)


#### File log_train_newx.txt:

```
No early stopping is detected on 64000 near epochs, considering first 4000 epochs, on real data
No early stopping is detected on 68000 near epochs, considering first 8000 epochs, on real + predicted data
No early stopping is detected on 86000 near epochs, considering first 6000 epochs, on real data
No early stopping is detected on 90000 near epochs, considering first 10000 epochs, on real + predicted data
No early stopping is detected on 108000 near epochs, considering first 8000 epochs, on real data
No early stopping is detected on 112000 near epochs, considering first 12000 epochs, on real + predicted data
No early stopping is detected on 130000 near epochs, considering first 10000 epochs, on real data
No early stopping is detected on 134000 near epochs, considering first 14000 epochs, on real + predicted data
No early stopping is detected on 152000 near epochs, considering first 12000 epochs, on real data
No early stopping is detected on 156000 near epochs, considering first 16000 epochs, on real + predicted data
No early stopping is detected on 174000 near epochs, considering first 14000 epochs, on real data
     detected:  INCREASING TRAIN-VALID
[Predicted data]: predicted earlystopping(patience=20): 17900 epoch, in 3 days, 13:48:21.852680
No early stopping is detected on 196000 near epochs, considering first 16000 epochs, on real data
     detected:  INCREASING TRAIN-VALID
[Predicted data]: predicted earlystopping(patience=20): 19700 epoch, in 3 days, 22:56:13.457071
No early stopping is detected on 218000 near epochs, considering first 18000 epochs, on real data
     detected:  INCREASING TRAIN-VALID
[Predicted data]: predicted earlystopping(patience=20): 21800 epoch, in 4 days, 9:17:51.145587
No early stopping is detected on 240000 near epochs, considering first 20000 epochs, on real data
     detected:  INCREASING TRAIN-VALID
[Predicted data]: predicted earlystopping(patience=20): 23800 epoch, in 4 days, 19:51:09.610220
No early stopping is detected on 262000 near epochs, considering first 22000 epochs, on real data
No early stopping is detected on 266000 near epochs, considering first 26000 epochs, on real + predicted data
```

1. Losses, epoch 40: 
![train_full](/images/newx_40.png)
2. Losses, epoch 80: 
![train_full](/images/newx_80.png)
3. Losses, epoch 120: 
![train_full](/images/newx_120.png)
4. Losses, epoch 160: 
![train_full](/images/newx_160.png)
5. Losses, epoch 200: 
![train_full](/images/newx_200.png)

#### File log_train_newu.txt:

```
No early stopping is detected on 64000 near epochs, considering first 4000 epochs, on real data
No early stopping is detected on 68000 near epochs, considering first 8000 epochs, on real + predicted data
No early stopping is detected on 86000 near epochs, considering first 6000 epochs, on real data
No early stopping is detected on 90000 near epochs, considering first 10000 epochs, on real + predicted data
No early stopping is detected on 108000 near epochs, considering first 8000 epochs, on real data
No early stopping is detected on 112000 near epochs, considering first 12000 epochs, on real + predicted data
No early stopping is detected on 130000 near epochs, considering first 10000 epochs, on real data
No early stopping is detected on 134000 near epochs, considering first 14000 epochs, on real + predicted data
No early stopping is detected on 152000 near epochs, considering first 12000 epochs, on real data
     detected:  INCREASING TRAIN-VALID
[Predicted data]: predicted earlystopping(patience=20): 15600 epoch, in 3 days, 16:41:24.576944
No early stopping is detected on 174000 near epochs, considering first 14000 epochs, on real data
No early stopping is detected on 178000 near epochs, considering first 18000 epochs, on real + predicted data
```

1. Losses, epoch 40: 
![train_full](/images/newu_40.png)
2. Losses, epoch 80: 
![train_full](/images/newu_80.png)
3. Losses, epoch 120: 
![train_full](/images/newu_120.png)

#### 
