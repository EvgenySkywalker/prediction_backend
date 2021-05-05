import io
import os
import time

import requests
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer, BatchNormalization, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.models import load_model

from app.core.settings import settings, new_columns, tf_swap
from app.ml.teaching import preprocess, convert_time, sweden_dataset_preprocess

time_points = settings.time_points
arg_num = settings.arg_num
class_num = settings.class_num
url = settings.url

if os.path.isfile('model.h5'):
    model = load_model(f'{settings.dnn_model_path}model.h5')
else:
    model = Sequential([
        Dense(arg_num, input_shape=(arg_num,)),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(.2),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(.2),
        Dense(class_num, activation='softmax')
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=1e-3),
        metrics=['categorical_accuracy'],
    )


def scheduler(epoch, lr):
    if epoch < 4:
        lr = 0.01
        return lr
    else:
        return lr * 3e-1


lr_sheduler = LearningRateScheduler(scheduler)


def train(x_train, x_val, y_train, y_val):
    print(time.strftime('%H:%M:%S'))
    _history = model.fit(
        x_train, y_train,
        epochs=10,
        verbose=2,
        validation_data=(x_val, y_val),
        initial_epoch=0,
        callbacks=[
            lr_sheduler
        ],
    )
    print(time.strftime('%H:%M:%S'))
    return _history


def post_model(_filename, _history):
    files = {'file': open(f'{settings.dnn_model_path}neuralnetwork_model_{filename}.h5', 'rb')}
    print(_filename)
    response = requests.post(url, files=files, data={'percent': _history.history['val_categorical_accuracy'][-1] * 100})
    print(response.status_code)


while True:
    print(url)
    r = requests.get(url)
    print(r.status_code)

    if r.status_code == requests.codes.ok:
        try:
            filename = r.headers['Content-Disposition'].split(';')[1].split('=')[1]
            filename = filename.replace('"', '').split('.')[0]
            print(filename)
            df = pd.read_csv(io.StringIO(r.content.decode('utf-8')), skiprows=1, low_memory=False)
            df.columns = new_columns
            df_copy = df[tf_swap]
            df_copy = df_copy.dropna()
            x_t, x_v, y_t, y_v = preprocess(df_copy, rate=10, time_conv_f=convert_time)
            history = train(x_t, x_v, y_t, y_v)
            model.save(f'{settings.dnn_model_path}neuralnetwork_model_{filename}.h5')
            if os.path.isfile('model.h5'):
                os.remove('model.h5')
            model.save('model.h5')
            post_model(filename, history)
        except Exception as e:
            print(f'Ошибка обучения: {e}')
            time.sleep(300)
    else:
        print('Нет данных для обучения')
        time.sleep(300)
