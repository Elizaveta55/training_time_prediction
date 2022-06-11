import argparse
import asyncio
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot
from keras.layers import Bidirectional
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, dest='config', default='config.yaml')
args = parser.parse_args()
config = Config(args.config)

data = pd.DataFrame()
is_first_operation=True


assert (config.how_much_train+config.val_boards)<=config.n_every_epoch_calc_lstm
assert config.look_back<config.n_every_epoch_calc_lstm

suspected_epoch = []
conunter_pred = 0

checkpoint_valid = ModelCheckpoint('model/model_valid_' + config.model + '.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
model_valid = Sequential()
model_valid.add(Bidirectional(LSTM(128, return_sequences=False)))
model_valid.add(Dense(1))

checkpoint_train = ModelCheckpoint('model/model_train_' + config.model + '.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
model_train = Sequential()
model_train.add(Bidirectional(LSTM(128, return_sequences=False)))
model_train.add(Dense(1))

def plot(type):
    y = (data[type + ' loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()
    y=y[-config.how_much_epochs_to_plot:]
    x_plot = [i for i in range(len(y))]
    y_plot = y
    if type == 'Valid':
        model = model_valid
    elif type == 'Train':
        model = model_train
    else:
        return
    y_pred_plot = y + predict_new_data_from_pretrained(model, type, predict_count = config.how_much_epochs_to_plot)
    x_pred_plot = [i for i in range(len(y_pred_plot))]
    pyplot.plot(x_pred_plot, y_pred_plot, label='Predicted')
    pyplot.plot(x_plot, y_plot, label='Actual')
    pyplot.title('' + type+ ' loss')
    pyplot.xlabel('Input Variable (x)')
    pyplot.ylabel('Output Variable (y)')
    pyplot.legend()
    pyplot.show()

def predict_new_data(model, old_data, predict_count=config.predict_count, look_back=config.look_back):
    new_data = []
    old_data_copy = old_data[:]
    for i in range(predict_count):
        new_value = model.predict(np.array([[old_data_copy[-look_back:]]]))
        new_data.append(new_value[0][0])
        old_data_copy.append(new_value[0][0])
    return new_data

def predict_new_data_from_pretrained(model, type, predict_count=config.predict_count):
    y = (data[type + ' loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()
    y_new = y[:]
    model.build(input_shape=np.array([[y_new[-config.look_back:]]]).shape)
    model.load_weights('model/model_' + type.lower() + '_' + config.model + '.h5')
    y_pred = predict_new_data(model, y_new[-config.look_back:], predict_count = predict_count)
    return y_pred

def train_model(model, type, checkpoint):
    y = (data[type + ' loss'].rolling(window=config.smooth_window, min_periods=1).mean()).tolist()
    y_new = y[:]
    data_x, data_y = create_dataset(y_new, config.look_back)
    data_x = np.reshape(data_x, (data_x.shape[0], 1, data_x.shape[1]))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(data_x[:config.how_much_train], data_y[:config.how_much_train], validation_data=(
        data_x[config.how_much_train - config.val_boards:config.how_much_train + config.val_boards],
        data_y[config.how_much_train - config.val_boards:config.how_much_train + config.val_boards]),
                    epochs=500, batch_size=128, verbose=1, shuffle=True, callbacks=[checkpoint])

    y_pred = predict_new_data(model, y_new)
    return y_pred

def analyse_results_detailed(pred_valid, pred_train, position, eps = config.eps):
    if len(suspected_epoch)<(position+len(pred_train)):
        suspected_epoch.extend([0]*((position+len(pred_train)) - len(suspected_epoch)))
    overtraining_prob = sum([pred_valid[i]-pred_train[i] for i in range(len(pred_train))])/len(pred_train)
    puredata_prob = sum([pred_train[i]-pred_valid[i] for i in range(len(pred_train))])/len(pred_train)
    if overtraining_prob >= eps:
        print('     Warning: There is a high probability of overtraining on predicted epoches')
        print('     Suspected epoch(-s) is(are): ')
        for i in range(len(pred_train)):
            if ((pred_valid[i] - pred_train[i]) > eps):
                print(f"           Epoch {position + i}, {((((pred_valid[i] - pred_train[i]) / eps) - 1) * 100):.0f}% bigger then epsilon")
                suspected_epoch[position+i]+=1
    if puredata_prob >=eps:
        print('     Warning: Train data distr and valid data distr are significantly different')

def analyse_results(pred_valid, pred_train, position, eps = config.eps, counter_pred=conunter_pred):
    if len(suspected_epoch)<(position+len(pred_train)):
        suspected_epoch.extend([0]*((position+len(pred_train)) - len(suspected_epoch)))
    overtraining_prob = sum([pred_valid[i]-pred_train[i] for i in range(len(pred_train))])/len(pred_train)
    if overtraining_prob >= eps:
        for i in range(len(pred_train)):
            if ((pred_valid[i] - pred_train[i]) > eps):
                suspected_epoch[position+i]+=1
    i_most_susp_epoch = np.argmax(suspected_epoch)
    if i_most_susp_epoch!=0:
        print(f"Overtraining is predicted on epoch {i_most_susp_epoch} with probability {suspected_epoch[i_most_susp_epoch]/conunter_pred * 100}")

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
    global model_valid, model_train
    async for train_loss, valid_loss, elapsed_time in follow(open(path, "r" , encoding='utf-8')):
        if train_loss:
            if is_first_operation==True:
                data = pd.DataFrame({"Train loss":train_loss, "Valid loss":valid_loss, "Elapsed time":elapsed_time}, columns=['Train loss', 'Valid loss', 'Elapsed time'], index=['Epoch'])
                suspected_epoch.append(0)
            else:
                data = data.append({"Train loss":train_loss, "Valid loss":valid_loss, "Elapsed time":elapsed_time}, ignore_index=True)
                suspected_epoch.append(0)
                if config.flag_analyze and data.shape[0] % config.predict_count == 0 and data.shape[0]>=config.look_back:
                    print("Processed [{0}/{1}], prediction building based on processed data for [{1}/{2}]".format(data.shape[0] - config.predict_count, data.shape[0], data.shape[0] + config.predict_count))
                    conunter_pred +=1
                    analyse_results(predict_new_data_from_pretrained(model_valid, 'Valid', predict_count=3*config.predict_count),
                                    predict_new_data_from_pretrained(model_train, 'Train', predict_count=3*config.predict_count),
                                    data.shape[0])
            is_first_operation=False
        if config.flag_plot and data.shape[0] % config.epoch_to_plot == 0:
            plot('Valid')
            plot('Train')
        if config.flag_train and data.shape[0] % config.n_every_epoch_calc_lstm == 0:
            print(train_model(model_valid, 'Valid', checkpoint_valid))
            print(train_model(model_train, 'Train', checkpoint_train))

asyncio.ensure_future(main(config.file_name))
loop = asyncio.get_event_loop()
loop.run_forever()