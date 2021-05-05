from uuid import getnode as get_mac

from pydantic import BaseSettings

# def
feature_count = 3
target_count = 3
# sw
# feature_count = 8
# target_count = 5

mac = int(str(get_mac()).replace('3', '2'))


new_columns = [
    'accX', 'accY', 'accZ',
    'gyrX', 'gyrY', 'gyrZ',
    'latitude', 'longitude',
    'speed', 'vehicleType', 'label',
    'timestamp'
]
tf_swap = [
    'timestamp',
    'accX', 'accY', 'accZ',
    'gyrX', 'gyrY', 'gyrZ',
    'latitude', 'longitude',
    'speed', 'vehicleType', 'label'
]


class Settings(BaseSettings):
    dnn_weights_path: str = 'app/assets/v1.2-acc1.0000-val_loss0.0000-ep002.hdf5'
    dnn_sw_weights_path: str = 'app/assets/sw_v1.2-acc1.0000-val_loss0.0000-ep001.hdf5'
    dnn_model_path: str = 'app/assets/'
    time_points: int = 60
    arg_num: int = feature_count
    class_num: int = target_count
    url: str = 'http://34.89.190.3/'+'file/'+str(mac)
    d_type: str = 'float64'


settings = Settings()
