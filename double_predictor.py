import argparse
import asyncio
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot
from keras.layers import Bidirectional
from statsmodels.tsa.arima.model import ARIMA


from utils import *
from statsmodels.tsa.ar_model import AutoReg

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, dest='config', default='config.yaml')
args = parser.parse_args()
config = Config(args.config)

data = pd.DataFrame()
is_first_operation=True


assert (config.how_much_train+config.val_boards)<=config.n_every_epoch_calc_lstm
assert config.look_back<config.n_every_epoch_calc_lstm

suspected_epoch = []
defference_losses = []
defference_mean = []
conunter_pred = 0

checkpoint_valid = ModelCheckpoint('model/model_trice_' + config.model + '.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
model_double = Sequential()
model_double.add(Bidirectional(LSTM(128, return_sequences=True)))
model_double.add(Dense(1))


def plot(type):
    y = (data['Train loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()
    x = (data['Elapsed time']).tolist()
    model_x = ARIMA(x, order=(1, 1, 1))
    model_x_fit = model_x.fit()
    x_to_add = model_x_fit.predict(len(x), len(x) + config.how_much_epochs_to_plot - 1)
    y=y[-config.how_much_epochs_to_plot:]
    x_plot=x[-config.how_much_epochs_to_plot:]
    x_plot_valid = x_plot
    y_plot = y
    model = model_double
    y_pred_plot = y + predict_new_data_from_pretrained(model, type, predict_count = config.how_much_epochs_to_plot)[0]
    x_pred_plot = x[-config.how_much_epochs_to_plot:]
    x_pred_plot.extend(x_to_add)
    pyplot.plot(x_pred_plot, y_pred_plot, label='Predicted_train')
    pyplot.plot(x_plot, y_plot, label='Actual_train')
    y_valid = (data['Valid loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()
    y_valid = y_valid[-config.how_much_epochs_to_plot:]
    y_plot_valid = y_valid
    y_pred_plot_valid = y_valid + predict_new_data_from_pretrained(model, type, predict_count=config.how_much_epochs_to_plot)[1]
    x_pred_plot_valid = x_pred_plot
    pyplot.plot(x_pred_plot_valid, y_pred_plot_valid, label='Predicted_valid')
    pyplot.plot(x_plot_valid, y_plot_valid, label='Actual_valid')
    pyplot.title('Losses')
    pyplot.xlabel('Input Variable (x)')
    pyplot.ylabel('Output Variable (y)')
    pyplot.legend()
    pyplot.show()


def predict_new_data(model, old_data, predict_count=config.predict_count, look_back=config.look_back):
    new_data = [list(), list()]
    old_data_copy = old_data[:]
    for i in range(predict_count):
        datapoint = np.array([[old_data_copy[0][-look_back:], old_data_copy[1][-look_back:]]])
        new_value = model.predict(datapoint.tolist())
        new_data[0].append(new_value[0][0][0])
        new_data[1].append(new_value[0][1][0])
        old_data_copy[0].append(new_value[0][0][0])
        old_data_copy[1].append(new_value[0][1][0])
    return new_data

def predict_new_data_from_pretrained(model, type, predict_count=config.predict_count):
    y_train = (data['Train loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()
    y_valid = (data['Valid loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()
    y_new = [y_train, y_valid]
    model.build(input_shape=np.array([[y_new[0][-config.look_back:], y_new[1][-config.look_back:]]]).shape)
    model.load_weights('model/model_' + type.lower() + '_' + config.model + '.h5')
    y_pred = predict_new_data(model, [y_new[0][-config.look_back:], y_new[1][-config.look_back:]], predict_count = predict_count)
    return y_pred

def train_model(model, checkpoint):
    y_train = (data['Train loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()
    y_valid = (data['Valid loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()
    y_new = [y_train, y_valid]
    data_x, data_y = create_dataset_double(y_new, config.look_back)

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(data_x[:config.how_much_train], data_y[:config.how_much_train], validation_data=(
        data_x[config.how_much_train - config.val_boards:config.how_much_train + config.val_boards],
        data_y[config.how_much_train - config.val_boards:config.how_much_train + config.val_boards]),
                    epochs=500, batch_size=128, verbose=1, shuffle=True, callbacks=[checkpoint])
    y_pred = predict_new_data(model, y_new)
    return y_pred


# a[i]->0 for i->n-1
# a[i]->0 and mean[i]->0 for i->n-1 SMOOTHLY
def is_approximate_to_zero_smooth(difference):
    ## FIX NOISY AND WAVIE DATA
    gradient = np.gradient(difference)
    overall_mean = np.mean(gradient)
    gradient_abs = abs(gradient)
    for i in range(int(len(gradient_abs)*0.3), len(gradient_abs) - 1):
        if gradient_abs[i]!=0 and gradient[i-1]<0:
            if gradient_abs[i+1]!=0:
                if gradient_abs[i]/(gradient_abs[i+1]) < 1:
                    if (gradient_abs[i+1]) > abs(overall_mean):
                        return False
        elif gradient_abs[i]!=0 and gradient[i-1]>0 and overall_mean>0:
            return False
    return True

# mean[i]-> 0 too sudden
def is_approximate_to_zero_sudden(difference):
    gradient = np.gradient(difference)
    overall_mean = np.mean(gradient)
    gradient_abs = abs(gradient)
    for i in range(int(len(gradient_abs) * 0.3), len(gradient_abs) - 1):
        if gradient_abs[i] != 0 and gradient[i - 1] < 0:
            if gradient_abs[i + 1] != 0:
                if gradient_abs[i] / (gradient_abs[i + 1]) < 1:
                    if (gradient_abs[i + 1])*0.5 > abs(overall_mean):
                        # print("gradient_abs[i + 1]", gradient_abs[i + 1]*0.7 , " abs(overall_mean)", abs(overall_mean))
                        # print(abs(overall_mean)/ gradient_abs[i + 1])
                        return True
        elif gradient_abs[i] != 0 and gradient[i - 1] > 0 and overall_mean > 0:
            return False
    return False

def epoch_approximate_to_zero_sudden(difference):
    gradient = np.gradient(difference)
    overall_mean = np.mean(gradient)
    gradient_abs = abs(gradient)
    for i in range(int(len(gradient_abs) * 0.3), len(gradient_abs) - 1):
        if gradient_abs[i] != 0 and gradient[i - 1] < 0:
            if gradient_abs[i + 1] != 0:
                if gradient_abs[i] / (gradient_abs[i + 1]) < 1:
                    if (gradient_abs[i + 1])*0.5 > abs(overall_mean):
                        return i+1

# a[i] !-> 0 for i->n-1
def is_not_approximate_to_zero(difference):
    gradient = np.gradient(difference)
    if np.mean(gradient)<0:
        count_decreasing = sum([1 for x in gradient if x <= 0])
        count_increasing = sum([1 for x in gradient if x > 0])
    else:
        count_decreasing = sum([1 for x in gradient if x < 0])
        count_increasing = sum([1 for x in gradient if x >= 0])
    if count_decreasing / (count_decreasing + count_increasing) < 0.4:
        return True
    return False

def epoch_not_approximate_to_zero(difference):
    gradient = np.gradient(difference)
    mean = np.mean(gradient)
    if mean<0:
        for i in range(len(gradient)):
            if gradient[i]>0 and gradient[i]>abs(mean):
                return i
    else:
        for i in range(len(gradient)):
            if gradient[i]>=0 and gradient[i]>abs(mean):
                return i

# a[i]-> -infinity for i->n-1
def is_approximate_to_positive_inf(difference):
    gradient = np.gradient(difference)
    coef = np.var(gradient) / np.mean(gradient)
    if coef > 0:
        if np.mean(gradient) > 0:
            return False
        return True
    else:
        return False


# a[i]-> +infinity for i->n-1
# mean[i]-> +infinity for i->n-1
def is_approximate_to_negative_inf(difference):
    gradient = np.gradient(difference)
    coef = np.var(gradient)/np.mean(gradient)
    if coef < 0:
        if np.mean(gradient) < 0:
            return False
        ## проверка на то, что ноль пройден
        return True
    else:
        return False


#a[i]>=0 with some error
def is_positive(difference):
    count_negative = sum([1 for x in difference if x < 0])
    count_positive = sum([1 for x in difference if x >= 0])
    if count_positive / (count_negative+count_positive)>0.8:
        return True
    else:
        return False

#a[i]<0 with some error
def is_negative(difference):
    count_negative = sum([1 for x in difference if x < 0])
    count_positive = sum([1 for x in difference if x >= 0])
    if count_negative / (count_negative + count_positive) > 0.8:
        return True
    else:
        return False

# mean[i] doesn't increase/decrease suddenly for i->n-1
def is_sudden_change(mean):
    window_size = int(len(mean) * 0.8)
    overall_coef = np.var(mean) / np.mean(mean)
    prev_coef = 0
    for i in range(int(len(mean) * 0.2), len(mean) - window_size):
        try:
            gradient = np.gradient(mean[i:i+window_size])
        except:
            print(mean)
            print(len(mean[i:i+window_size]))
            print(mean[i:i+window_size])
        coef = np.var(gradient) / np.mean(gradient)
        if abs(prev_coef) < abs(coef) and abs(coef) > 0.5 * abs(overall_coef):
            return True
        prev_coef = coef
    return False


def show_time_and_epoch(type):
    print(type)
    train_loss = (data['Train loss']).tolist()
    valid_loss = (data['Valid loss']).tolist()
    time = (data['Elapsed time']).tolist()
    model_time = ARIMA(time, order=(1, 1, 1))
    model_time_fit = model_time.fit()
    predicted_time = model_time_fit.predict(len(time), len(time) + config.how_much_epochs_to_plot - 1)
    time.extend(predicted_time)
    difference_losses = [valid_loss[i] - train_loss[i] for i in range(len(train_loss))]
    mean = [np.mean([valid_loss[i], train_loss[i]]) for i in range(len(train_loss))]

    if type == "Overfitting. 1":
        if is_approximate_to_negative_inf(difference_losses):
            print("Train losses are growing")
            for i in range(len(difference_losses)):
                if difference_losses[i] > abs(np.mean(mean)):
                    return i, time[i]
        elif is_approximate_to_positive_inf(difference_losses):
            print("Valid losses are growing")
            for i in range(len(difference_losses)):
                if difference_losses[i] < abs(np.mean(mean)):
                    return i, time[i]
    elif type == "Underfitting. 1":
        print("Train losses are growing")
        for i in range(len(difference_losses)):
            if difference_losses[i] > abs(np.mean(mean)):
                return i, time[i]
    elif type == "Underfitting. 2":
        print("Losses are having synchronously dip")
        i = epoch_approximate_to_zero_sudden(mean)
        return i, time[i]
    elif type == "Underfitting. 3":
        print("Losses are not reducing")
        i = epoch_not_approximate_to_zero(difference_losses)
        return i, time[i]
    else:
        print("Something went wrong")


def predict_training(difference, mean):
    is_predicted_non_normal = False
    if is_approximate_to_zero_smooth(difference) and is_positive(difference) and is_sudden_change(mean)==False:
        # print("Normal behavoiur. 1 Smooth")
        is_predicted_non_normal = False
    elif is_approximate_to_zero_sudden(difference) and is_positive(difference) and is_sudden_change(mean)==False:
        # print("Normal behavoiur. 1 Sudden")
        is_predicted_non_normal = False
    if is_approximate_to_zero_smooth(difference):
        # print("Normal behavoiur. 2 Smooth")
        is_predicted_non_normal = False
    elif is_approximate_to_zero_sudden(difference):
        # print("Normal behavoiur. 2 Sudden")
        is_predicted_non_normal = False
    if is_approximate_to_zero_smooth(difference) and is_approximate_to_zero_smooth(mean):
        # print("Normal behavoiur. 3")
        is_predicted_non_normal = False

    if is_approximate_to_negative_inf(difference) or is_approximate_to_positive_inf(difference):
        epoch, time = show_time_and_epoch("Overfitting. 1")
        print(f'Is predicted on epoch {epoch}, in {time} ms')
        is_predicted_non_normal = True
    if is_approximate_to_negative_inf(difference) and is_approximate_to_positive_inf(mean):
        epoch, time = show_time_and_epoch("Underfitting. 1")
        print(f'Is predicted on epoch {epoch}, in {time} ms')
        is_predicted_non_normal = True
    if is_approximate_to_zero_sudden(mean) and is_positive(difference):
        epoch, time = show_time_and_epoch("Underfitting. 2")
        is_predicted_non_normal = True
        print(f'Is predicted on epoch {epoch}, in {time} ms')
    if is_not_approximate_to_zero(difference):
        epoch, time = show_time_and_epoch("Underfitting. 3")
        print(f'Is predicted on epoch {epoch}, in {time} ms')
        is_predicted_non_normal = True

    if not is_predicted_non_normal:
        print("Training is active....")
    print("----------------------------------------------------------------")


def analyse_results(pred_double, position, eps = config.eps, counter_pred=conunter_pred):
    pred_valid = pred_double[1]
    pred_train = pred_double[0]
    difference_losses = [pred_valid[i] - pred_train[i] for i in range(len(pred_train))]
    mean = [np.mean([pred_valid[i], pred_train[i]]) for i in range(len(pred_train))]

    print("Train Prediction on predicted data:")
    predict_training(difference_losses, mean)


async def follow(thefile):
    # thefile.seek(0,2)
    while True:
        line = thefile.readline()
        if not line:
            await asyncio.sleep(0.1)
            continue
        if get_substring(line, start='Train loss: ' , end=','):
            yield float(get_substring(line, start='Train loss: ' , end=',')), \
                  float(get_substring(line, start='Valid loss: ' , end=',')), \
                  float(get_substring(line, start='Elapsed_time: ' , end='\n'))


async def main(path):
    global conunter_pred
    global is_first_operation
    global data
    global model_double
    is_trained_once = False
    if config.flag_train == False:
        is_trained_once = True
    async for train_loss, valid_loss, elapsed_time in follow(open(path, "r" , encoding='utf-8')):
        if train_loss:
            if is_first_operation==True:
                data = pd.DataFrame({"Train loss":train_loss, "Valid loss":valid_loss, "Elapsed time":elapsed_time}, columns=['Train loss', 'Valid loss', 'Elapsed time'], index=['Epoch'])
                suspected_epoch.append(0)
                defference_losses.append(valid_loss - train_loss)
                defference_mean.append(np.mean([valid_loss, train_loss], dtype=float))
            else:
                data = data.append({"Train loss":train_loss, "Valid loss":valid_loss, "Elapsed time":elapsed_time}, ignore_index=True)
                suspected_epoch.append(0)
                defference_mean.append(np.mean([valid_loss, train_loss]))
                defference_losses.append((valid_loss - train_loss))
                if config.flag_analyze and is_trained_once and data.shape[0] % config.predict_count == 0 and data.shape[0]>=config.look_back:
                    print("Processed [{0}/{1}], prediction building based on processed data for [{1}/{2}]".format(data.shape[0] - config.predict_count, data.shape[0], data.shape[0] + config.predict_count))
                    conunter_pred +=1
                    print("Train Prediction on given data")
                    predict_training(defference_losses, defference_mean)
                    analyse_results(predict_new_data_from_pretrained(model_double, 'trice', predict_count=3 * config.predict_count),
                                    data.shape[0])
            is_first_operation=False
        if config.flag_plot and is_trained_once and data.shape[0] % config.epoch_to_plot == 0:
            plot('double')
        if config.flag_train and data.shape[0] % config.n_every_epoch_calc_lstm == 0:
            print(train_model(model_double, checkpoint_valid))
            is_trained_once = True

asyncio.ensure_future(main(config.file_name))
loop = asyncio.get_event_loop()
loop.run_forever()
