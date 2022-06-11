import yaml
from easydict import EasyDict as edict
import numpy as np

def Config(filename):
    with open(filename, 'r') as f:
        parser = edict(yaml.load(f, Loader=yaml.FullLoader))
    for x in parser:
        print('{}: {}'.format(x, parser[x]))
    return parser

def get_substring(s, start, end):
    try:
        return s.split(start)[1].split(end)[0]
    except BaseException:
        return None

def create_dataset(y, look_back):
    dataX, dataY = [], []
    for i in range(len(y) - look_back -1):
        dataX.append(y[i:(i+look_back)])
        dataY.append(y[i + look_back])
    return np.array(dataX), np.array(dataY)
