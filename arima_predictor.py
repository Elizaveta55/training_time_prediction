import argparse
import asyncio
import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
import datetime
from utils import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, dest='config', default='config.yaml')
args = parser.parse_args()
config = Config(args.config)

data = pd.DataFrame()
is_first_operation = True

assert config.patience <= config.predict_count

suspected_epoch = []
defference_losses = []
defference_mean = []
conunter_pred = 0


def plot():
    y = (data['Train loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()
    x = (data['Elapsed time']).tolist()
    model_x = ARIMA(x, order=(1, 1, 1))
    model_x_fit = model_x.fit()
    x_to_add = model_x_fit.predict(len(x), len(x) + config.how_much_epochs_to_plot - 1)
    y = y[-config.how_much_epochs_to_plot:]
    x_plot = x[-config.how_much_epochs_to_plot:]
    x_plot_valid = x_plot
    y_plot = y
    y_pred_plot = predict_new_data_from_pretrained(predict_count=config.how_much_epochs_to_plot)[0][
                  -2 * config.how_much_epochs_to_plot:]
    x_pred_plot = x[-config.how_much_epochs_to_plot:]
    x_pred_plot.extend(x_to_add)
    pyplot.plot(x_pred_plot, y_pred_plot, label='Predicted_train')
    pyplot.plot(x_plot, y_plot, label='Actual_train')
    y_valid = (data['Valid loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()
    y_valid = y_valid[-config.how_much_epochs_to_plot:]
    y_plot_valid = y_valid
    y_pred_plot_valid = predict_new_data_from_pretrained(predict_count=config.how_much_epochs_to_plot)[1][
                        -2 * config.how_much_epochs_to_plot:]
    x_pred_plot_valid = x_pred_plot
    pyplot.plot(x_pred_plot_valid, y_pred_plot_valid, label='Predicted_valid')
    pyplot.plot(x_plot_valid, y_plot_valid, label='Actual_valid')
    pyplot.title('Losses')
    pyplot.xlabel('Time (x)')
    pyplot.ylabel('Loss (y)')
    pyplot.legend()
    pyplot.show()


def predict_new_data_from_pretrained(predict_count=config.predict_count):
    y_train = (data['Train loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()
    y_valid = (data['Valid loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()

    # model_train = ARIMA(y_train[int(len(y_train)*0.05):], order=(10, 1, 1))
    model_train = ARIMA(y_train, order=(1, 1, 1))
    model_train_fit = model_train.fit()
    predicted_train = model_train_fit.predict(len(y_train), len(y_train) + predict_count - 1)
    y_train.extend(predicted_train)

    model_valid = ARIMA(y_valid, order=(1, 1, 1))
    # model_valid = ARIMA(y_valid[int(len(y_train)*0.05):], order=(10, 1, 1))
    model_valid_fit = model_valid.fit()
    predicted_valid = model_valid_fit.predict(len(y_valid), len(y_valid) + predict_count - 1)
    y_valid.extend(predicted_valid)

    return [y_train, y_valid]



def predict_early_stopping(valid_loss, train_loss, type_data, patience=config.patience):
    count_iter=0
    initial_valid_size = len(valid_loss)
    while True:

        ## stage1: check valid losses is not incresing througtout given patience
        count_epochs_of_increasing_valid = 0
        gradient_to_check_increasing_valid = np.gradient(valid_loss)
        for i in range(len(valid_loss)):
            if gradient_to_check_increasing_valid[i] >= 0:
                count_epochs_of_increasing_valid += 1
                if count_epochs_of_increasing_valid == patience:
                    print("     detected: INCREASING VALID")
                    return i
            else:
                count_epochs_of_increasing_valid = 0

        ## stage2: check difference between train loss and valis loss is not increasing througtout given patience
        difference = [valid_loss[i] - train_loss[i] for i in range(len(valid_loss))]
        count_epochs_of_increasing_difference = 0
        gradient_to_check_increasing_difference = np.gradient(difference)
        for i in range(len(difference)):
            if gradient_to_check_increasing_difference[i] >= 0:
                count_epochs_of_increasing_difference += 1
                if count_epochs_of_increasing_difference == patience*2:
                    print("     detected:  INCREASING TRAIN-VALID")
                    return i
            else:
                count_epochs_of_increasing_difference = 0


        ## stage3: if none of listed above indicators of early stopping is detected -> increase prediction space (consider more predicted epochs)
        predicted_train, predicted_valid = predict_new_data_from_pretrained()
        train_loss.extend(predicted_train)
        valid_loss.extend(predicted_valid)

        count_iter+=1

        if count_iter >=10:
            print(f"No early stopping is detected on {len(valid_loss)*100} near epochs, considering first {initial_valid_size*100} epochs, on {type_data} data")
            return



# a[i]->0 for i->n-1
# a[i]->0 and mean[i]->0 for i->n-1 SMOOTHLY
def is_approximate_to_zero_smooth(difference):
    ## FIX NOISY AND WAVIE DATA
    gradient = np.gradient(difference)
    overall_mean = np.mean(gradient)
    gradient_abs = abs(gradient)
    for i in range(int(len(gradient_abs) * 0.3), len(gradient_abs) - 1):
        if gradient_abs[i] != 0 and gradient[i - 1] < 0:
            if gradient_abs[i + 1] != 0:
                if gradient_abs[i] / (gradient_abs[i + 1]) < 1:
                    if (gradient_abs[i + 1]) > abs(overall_mean):
                        return False
        elif gradient_abs[i] != 0 and gradient[i - 1] > 0 and overall_mean > 0:
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
                    if (gradient_abs[i + 1]) * 0.5 > abs(overall_mean):
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
                    if (gradient_abs[i + 1]) * 0.5 > abs(overall_mean):
                        return i + 1


# a[i] !-> 0 for i->n-1
def is_not_approximate_to_zero(difference):
    gradient = np.gradient(difference)
    if np.mean(gradient) < 0:
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
    if mean < 0:
        for i in range(len(gradient)):
            if gradient[i] > 0 and gradient[i] > abs(mean):
                return i
    else:
        for i in range(len(gradient)):
            if gradient[i] >= 0 and gradient[i] > abs(mean):
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
    coef = np.var(gradient) / np.mean(gradient)
    if coef < 0:
        if np.mean(gradient) < 0:
            return False
        ## проверка на то, что ноль пройден
        return True
    else:
        return False


# a[i]>=0 with some error
def is_positive(difference):
    count_negative = sum([1 for x in difference if x < 0])
    count_positive = sum([1 for x in difference if x >= 0])
    if count_positive / (count_negative + count_positive) > 0.8:
        return True
    else:
        return False


# a[i]<0 with some error
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
            gradient = np.gradient(mean[i:i + window_size])
        except:
            print(mean)
            print(len(mean[i:i + window_size]))
            print(mean[i:i + window_size])
        coef = np.var(gradient) / np.mean(gradient)
        if abs(prev_coef) < abs(coef) and abs(coef) > 0.5 * abs(overall_coef):
            return True
        prev_coef = coef
    return False


def show_time_and_epoch(difference, mean, type):
    print(type)
    time = (data['Elapsed time']).tolist()
    model_time = ARIMA(time, order=(1, 1, 1))
    model_time_fit = model_time.fit()
    predicted_time = model_time_fit.predict(len(time), len(time) + config.how_much_epochs_to_plot - 1)
    time.extend(predicted_time)

    if type == "Overfitting. 1":
        if is_approximate_to_negative_inf(difference):
            # print("Train losses are growing")
            for i in range(len(difference)):
                if difference[i] > abs(np.mean(mean)):
                    return i, time[i]
        elif is_approximate_to_positive_inf(difference):
            # print("Valid losses are growing")
            for i in range(len(difference)):
                if difference[i] < abs(np.mean(mean)):
                    return i, time[i]
    elif type == "Underfitting. 1":
        # print("Train losses are growing")
        for i in range(len(difference)):
            if difference[i] > abs(np.mean(mean)):
                return i, time[i]
    elif type == "Underfitting. 2":
        # print("Losses are having synchronously dip")
        i = epoch_approximate_to_zero_sudden(mean)
        return i, time[i]
    elif type == "Underfitting. 3":
        # print("Losses are not reducing")
        i = epoch_not_approximate_to_zero(difference)
        return i, time[i]
    else:
        print("Something went wrong")


def show_time(epoch):
    time = (data['Elapsed time']).tolist()
    model_time = ARIMA(time, order=(1, 1, 1))
    model_time_fit = model_time.fit()
    predicted_time = model_time_fit.predict(len(time), len(time) + config.how_much_epochs_to_plot - 1)
    time.extend(predicted_time)
    try:
        return time[epoch - 1]
    except:
        return time[-1]


def predict_training(difference, mean, valid_loss, train_loss, type_data):
    is_predicted_non_normal = False
    if is_approximate_to_zero_smooth(difference) and is_positive(difference) and is_sudden_change(mean) == False:
        # print("Normal behavoiur. 1 Smooth")
        is_predicted_non_normal = False
    elif is_approximate_to_zero_sudden(difference) and is_positive(difference) and is_sudden_change(mean) == False:
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
        epoch, time = show_time_and_epoch(difference, mean, "Overfitting. 1")
        print(f'Is predicted on epoch {epoch*100} epoch, in {time} ms')
        is_predicted_non_normal = True
    if is_approximate_to_negative_inf(difference) and is_approximate_to_positive_inf(mean):
        epoch, time = show_time_and_epoch(difference, mean, "Underfitting. 1")
        print(f'Is predicted on epoch {epoch*100} epoch, in {time} ms')
        is_predicted_non_normal = True
    if is_approximate_to_zero_sudden(mean) and is_positive(difference):
        epoch, time = show_time_and_epoch(difference, mean, "Underfitting. 2")
        is_predicted_non_normal = True
        print(f'Is predicted on epoch {epoch*100} epoch, in {time} ms')
    if is_not_approximate_to_zero(difference):
        epoch, time = show_time_and_epoch(difference, mean, "Underfitting. 3")
        print(f'Is predicted on epoch {epoch*100} epoch, in {time} ms')
        is_predicted_non_normal = True

    predicted_i = predict_early_stopping(valid_loss, train_loss, type_data)
    if predicted_i:

        print(f"Predicted early stopping {predicted_i*100} epoch, in {str(datetime.timedelta(seconds=show_time(predicted_i)))}")

    if not is_predicted_non_normal:
        print("Training is active....")
    print("----------------------------------------------------------------")


def analyse_results(pred_double, position, type_data, eps=config.eps, counter_pred=conunter_pred):
    pred_valid = (pd.Series(pred_double[1]).rolling(window=config.smooth_window, min_periods=1).mean()).tolist()[int(len(pred_double)*0.8):]
    pred_train = (pd.Series(pred_double[0]).rolling(window=config.smooth_window, min_periods=1).mean()).tolist()[int(len(pred_double)*0.8):]
    difference_losses = [pred_valid[i] - pred_train[i] for i in range(len(pred_train))]
    mean = [np.mean([pred_valid[i], pred_train[i]]) for i in range(len(pred_train))]

    print("Train Prediction on predicted data:")
    predict_training(difference_losses, mean, pred_valid, pred_train, type_data)


async def follow(thefile):
    # thefile.seek(0,2)
    while True:
        line = thefile.readline()
        if not line:
            await asyncio.sleep(0.1)
            continue
        if get_substring(line, start='Train loss: ', end=','):
            yield float(get_substring(line, start='Train loss: ', end=',')), \
                  float(get_substring(line, start='Valid loss: ', end=',')), \
                  float(get_substring(line, start='Elapsed_time: ', end='\n'))


async def main(path):
    global conunter_pred
    global is_first_operation
    global data
    async for train_loss, valid_loss, elapsed_time in follow(open(path, "r", encoding='utf-8')):
        if train_loss:
            if is_first_operation == True:
                data = pd.DataFrame({"Train loss": train_loss, "Valid loss": valid_loss, "Elapsed time": elapsed_time},
                                    columns=['Train loss', 'Valid loss', 'Elapsed time'], index=['Epoch'])
                suspected_epoch.append(0)
                defference_losses.append(valid_loss - train_loss)
                defference_mean.append(np.mean([valid_loss, train_loss], dtype=float))
            else:
                data = data.append({"Train loss": train_loss, "Valid loss": valid_loss, "Elapsed time": elapsed_time},
                                   ignore_index=True)
                suspected_epoch.append(0)
                defference_mean.append(np.mean([valid_loss, train_loss]))
                defference_losses.append((valid_loss - train_loss))
                valid = (data['Valid loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()
                train = (data['Train loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()
                if not config.flag_predict_only_earlystopping and data.shape[0] % config.predict_count == 0 and data.shape[0] >= config.look_back:
                    print("Processed [{0}/{1}], prediction building based on processed data for [{1}/{2}]".format(
                        data.shape[0] - config.predict_count, data.shape[0], data.shape[0] + config.predict_count))
                    conunter_pred += 1
                    print("Train Prediction on given data")
                    predict_training(defference_losses, defference_mean, valid, train, "real")
                    analyse_results(predict_new_data_from_pretrained(predict_count=2 * config.predict_count), data.shape[0], "real + pretrained")
                elif config.flag_predict_only_earlystopping and data.shape[0] % config.predict_count == 0 and data.shape[0] >= config.look_back:
                    conunter_pred += 1
                    predicted_i = predict_early_stopping(valid, train, "real")
                    if predicted_i:
                        print(f"[Given data]: predicted earlystopping(patience={config.patience}): {predicted_i*100} epoch, in {str(datetime.timedelta(seconds=show_time(predicted_i)))}")
                    train_predicted = (pd.Series(predict_new_data_from_pretrained(predict_count= 2 * config.predict_count)[0]).rolling(
                        window=config.smooth_window, min_periods=1).mean()).tolist()
                    valid_predicted = (pd.Series(predict_new_data_from_pretrained(predict_count= 2 * config.predict_count)[1]).rolling(
                        window=config.smooth_window, min_periods=1).mean()).tolist()
                    valid_predicted_i = predict_early_stopping(valid_predicted, train_predicted, "real + predicted")
                    if valid_predicted_i:
                        print(
                            f"[Predicted data]: predicted earlystopping(patience={config.patience}): {valid_predicted_i*100} epoch, in {str(datetime.timedelta(seconds=show_time(valid_predicted_i)))}")
            is_first_operation = False
        if config.flag_plot and data.shape[0] % config.epoch_to_plot == 0:
            plot()


asyncio.ensure_future(main(config.file_name))
loop = asyncio.get_event_loop()
loop.run_forever()