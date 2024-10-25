import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def get_ori_data(sequence_length=60, stride=1, shuffle=False, seed=13, ori_data_filename=None, input_output_ratio=0.5):
    np.random.seed(seed)

    original_df = pd.read_csv(ori_data_filename, header=0)
    original_df = original_df.fillna(original_df.mean())

    start_sequence_range = list(range(0, original_df.shape[0] - sequence_length, stride))

    if shuffle:
        np.random.shuffle(start_sequence_range)
    split_size = int(sequence_length * input_output_ratio)
    splitted_original_data = np.array([original_df[start_index:start_index+sequence_length] for start_index in start_sequence_range])
    splitted_original_x = splitted_original_data[:, :split_size]
    splitted_original_y = splitted_original_data[:, split_size:]
    return splitted_original_x, splitted_original_y


def scale_data(ori_data, scaling_method='standard', scaler=None):
    reshaped_ori_data = ori_data.reshape(-1, 1)
    if scaler is None:
        assert scaling_method in ['standard', 'minmax'], 'Only standard and minmax scalers are currently supported'
        if scaling_method == 'minmax':
            scaler = MinMaxScaler()
            scaler.fit(reshaped_ori_data)
            scaler_params = [scaler.data_min_, scaler.data_max_]

        elif scaling_method == 'standard':
            scaler = StandardScaler()
            scaler.fit(reshaped_ori_data)
            scaler_params = [scaler.mean_, scaler.var_]
        scaled_ori_data = scaler.transform(reshaped_ori_data).reshape(ori_data.shape)
        return scaled_ori_data, scaler, scaler_params
    else:
        scaled_ori_data = scaler.transform(reshaped_ori_data).reshape(ori_data.shape)
        return scaled_ori_data, scaler, None
