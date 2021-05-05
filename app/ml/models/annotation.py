import io

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

from app.core.settings import settings, new_columns, tf_swap
from app.models.schema.label import Label


def read_data_csv(file, test):
    try:
        if test:
            copy = pd.read_csv(file, skiprows=2, low_memory=False)
        else:
            copy = pd.read_csv(io.StringIO(file), skiprows=2, low_memory=False)
    except Exception as e:
        raise Exception(f'Чтение файла: {e}')

    try:
        copy.columns = new_columns
        copy = copy[tf_swap]
        copy = copy.dropna()
    except Exception as e:
        raise Exception(f'Формат данных [Количество столбцов != {len(new_columns)}]{e}')

    return copy


d_type = settings.d_type
time_points = settings.time_points
arg_num = settings.arg_num
class_num = settings.class_num


def annotate(file, test=False):
    model = load_model(f'{settings.dnn_model_path}model.h5')
    timestamps, dataset = preprocess(file, test, rate=10, conv=True)

    try:
        labels = model.predict(dataset)
    except Exception as e:
        raise Exception(f'Во время работы алгоритма: {e}')

    try:
        states = get_states(timestamps, labels)
    except Exception as e:
        raise Exception(f'Список пуст: {e}')

    return states


def get_states(timestamps, labels):
    switches = []
    idx = 0
    while idx < len(labels):
        def get_probability(y):
            _probability = [0 for _ in range(class_num)]
            for j in y:
                _probability[int(np.argmax(j))] += 1
            for j in range(len(_probability)):
                _probability[j] /= len(y)

            return _probability.index(max(_probability)), max(_probability)

        def check_timezone(_choice, _probability, new_step):
            if idx < len(labels) - new_step:
                new_choice, new_probability = get_probability(labels[idx:idx + new_step])
                if new_probability >= _probability:
                    _choice, _probability = new_choice, new_probability
                    return _choice, _probability, True
            else:
                new_choice, new_probability = get_probability(labels[idx:])
                if new_probability >= _probability:
                    _choice, _probability = new_choice, new_probability
                    return _choice, _probability, True
            return _choice, _probability, False

        choice = -1
        probability = 0
        step = 0
        steps = [3, 7, 10, 13]
        for cur_step in steps:
            choice, probability, new_step = check_timezone(choice, probability, cur_step)
            if new_step:
                step = cur_step

        print(len(labels), timestamps.shape[0])
        if not switches or switches[-1].mode != choice:
            switches.append(Label(start=timestamps[idx*time_points], end='', mode=choice))

        idx += step

    for i, sw in enumerate(switches[:-1]):
        sw.mode += 5
        sw.end = switches[i+1].start
    switches[-1].mode += 5
    switches[-1].end = timestamps[-1]

    return switches


def convert_time(time):
    """00:15:41.0000001 -> float"""

    split = time.split(':')
    new_value = float(split[0]) * 60 * 60
    new_value += float(split[1]) * 60
    new_value += float(split[2])
    return new_value


def preprocess(file, test, rate=1, conv=True):
    copy = read_data_csv(file, test)
    copy = copy.iloc[::rate].copy()

    try:
        timestamps = copy['timestamp'].to_numpy()
        if conv:
            copy['timestamp'] = copy['timestamp'].apply(convert_time)
        del copy['label']
        del copy['vehicleType']
        del copy['speed']
        del copy['latitude']
        del copy['longitude']
        dataset = copy.to_numpy(copy=True, dtype='float32')
        x = []
        for i in range(0, len(dataset)-60, 60):
            time_series = []
            for j in range(time_points):
                raw = dataset[i+j]
                time_series += [np.array([
                    raw[0],
                    (raw[1] ** 2 + raw[2] ** 2 + raw[3] ** 2) ** .5,
                    (raw[4] ** 2 + raw[5] ** 2 + raw[6] ** 2) ** .5
                ])]
            x += [np.array(time_series)]
        new_dataset = np.array(x)
    except Exception as e:
        raise Exception(f'Во время обработки данных: {e}')

    return timestamps, new_dataset
