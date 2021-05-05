import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from app.core.settings import settings

time_points = settings.time_points
arg_num = settings.arg_num
class_num = settings.class_num


def sweden_dataset_preprocess(data_frame):
    copy = data_frame.copy()
    copy = copy[np.all((copy.label.apply(float.is_integer), copy.label != 0), axis=0)]
    
    return copy


def convert_time(str_time):
    """00:15:41.0000001 -> float"""

    split = str_time.split(':')
    new_value = float(split[0]) * 60 * 60
    new_value += float(split[1]) * 60
    new_value += float(split[2])

    return new_value


def preprocess(data_frame: pd.DataFrame, rate: int = 1, time_conv_f=None):
    copy = data_frame.iloc[::rate].copy()

    def encode(_label):
        _label = int(_label)
        index = -1

        if _label == 1:
            index = 0
        if 2 <= _label <= 4:
            index = 1
        if 5 <= _label <= 7:
            index = _label - 5
        new_label = [0. for _ in range(class_num)]
        new_label[index] = 1.

        return new_label

    if time_conv_f is not None:
        copy['timestamp'] = copy['timestamp'].apply(time_conv_f)

    grouped = copy.groupby('label')

    # Clustering by input/output
    x = []
    y = []
    for label, data in grouped:
        del data['label']
        x.append(data.to_numpy(copy=True))
        encoded_label = encode(label)
        y.append(np.array([encoded_label] * data.shape[0]).reshape(-1, class_num, 1))
    x = np.concatenate(x)
    y = np.concatenate(y)

    # Clustering by intervals
    x = np.reshape(x, (-1, arg_num))

    # Clustering by train/validate
    test_fraction = 0.2
    p = np.random.permutation(x.shape[0])
    split_idx = int(x.shape[0] * (1 - test_fraction))

    x_train = x[p[:split_idx]]
    x_valid = x[p[split_idx:]]

    y_train = y[p[:split_idx]]
    y_valid = y[p[split_idx:]]
    return x_train, x_valid, y_train, y_valid
