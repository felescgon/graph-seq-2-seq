from torch.utils.data import DataLoader, TensorDataset
from models.EncoderDecoder import create_encoder_decoder_model
from models.TCN import TemporalConvNet
import torch
import torch.optim as optim
import os
import pickle
from helpers import index_splitter
from data_load import get_ori_data, scale_data
from tqdm import tqdm, trange
from trainer import StepByStep
import numpy as np
import argparse
from distutils import util
import torch.nn as nn
from models.Transformer import Transformer
def get_data_loaders(scaled_x_train_tensor, scaled_x_val_tensor, seq_len, batch_size, input_output_ratio):
    train_data = TensorDataset(scaled_x_train_tensor.float(), scaled_x_train_tensor[:, int(seq_len / 2):].float())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    source_test = scaled_x_val_tensor[:, :int(seq_len * input_output_ratio)]
    target_test = scaled_x_val_tensor[:, int(seq_len * input_output_ratio):]
    test_data = TensorDataset(source_test.float(), target_test.float())
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, train_data, test_data

def prepare_model(n_features, checkpoint):
    #print(args)
    checkpoint_context = checkpoint['model_params']
    seq_len = checkpoint_context["seq_len"] * 2
    hidden_dim = checkpoint_context["hidden_dim"]
    lr = checkpoint_context["lr"]
    num_layers = checkpoint_context["num_layers"]

    model_type = checkpoint_context['encoder_decoder_model']
    if model_type == "EncoderDecoder":
        rnn_module = checkpoint_context["rnn_module"]
        teacher_forcing = checkpoint_context["teacher_forcing"]
        normalization = checkpoint_context["normalization"]
        narrow_attn_heads = 0
        if "narrow_attn_heads" in checkpoint_context:
            narrow_attn_heads = checkpoint_context["narrow_attn_heads"]
        if rnn_module == "RNN":
            rnn_layer_module = nn.RNN
        elif rnn_module == "GRU":
            rnn_layer_module = nn.GRU
        else:
            rnn_layer_module = nn.LSTM
        model = create_encoder_decoder_model(n_features=n_features, hidden_dim=hidden_dim, rnn_layer_module=rnn_layer_module, rnn_layers=num_layers, seq_len=seq_len, teacher_forcing=teacher_forcing, normalization=normalization, narrow_attn_heads=narrow_attn_heads)
    elif model_type == "TCN":
        kernel_size = checkpoint_context["kernel_size"]
        num_channels = checkpoint_context["num_channels"]
        channels = [num_channels for _ in range(num_layers-1)]
        channels.append (n_features)
        model = TemporalConvNet(num_inputs=n_features, num_channels=channels, kernel_size=kernel_size, seq_len=seq_len)
    elif model_type == "Transformer":
        narrow_attn_heads = checkpoint_context["narrow_attn_heads"]
        model = Transformer(n_features=n_features, hidden_dim=hidden_dim, seq_len=checkpoint_context["seq_len"], narrow_attn_heads=narrow_attn_heads, num_layers=num_layers, dropout=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint

def export_checkpoint(experiment_dir, checkpoint_pth_file, args):
    input_output_ratio = 1/2
    checkpoint = torch.load(checkpoint_pth_file)
    checkpoint_context = checkpoint['model_params']
    model_type = checkpoint_context['encoder_decoder_model']
    seq_len = checkpoint_context["seq_len"] * 2
    batch_size = checkpoint_context["batch_size"]
    experiment_root_directory_name = args.experiment_directory_path
    scaler = None
    scaling_method = 'standard'
    data_available = os.path.exists(f'{experiment_root_directory_name}/scaler.pickle') and os.path.exists(
        f'{experiment_root_directory_name}/x_train_tensor.pickle') and os.path.exists(
        f'{experiment_root_directory_name}/y_train_tensor.pickle') and os.path.exists(
        f'{experiment_root_directory_name}/x_val_tensor.pickle') and os.path.exists(
        f'{experiment_root_directory_name}/y_val_tensor.pickle')
    if data_available:
        with open(f'{experiment_root_directory_name}/scaler.pickle', "rb") as fb:
            scaler = pickle.load(fb)
        with open(f'{experiment_root_directory_name}/x_train_tensor.pickle', "rb") as fb:
            x_train_tensor = pickle.load(fb)
        with open(f'{experiment_root_directory_name}/y_train_tensor.pickle', "rb") as fb:
            y_train_tensor = pickle.load(fb)
        with open(f'{experiment_root_directory_name}/x_val_tensor.pickle', "rb") as fb:
            x_val_tensor = pickle.load(fb)
        with open(f'{experiment_root_directory_name}/y_val_tensor.pickle', "rb") as fb:
            y_val_tensor = pickle.load(fb)
    else:
        scaling_method = checkpoint_context["scaling_method"] if "scaling_method" in checkpoint_context else 'standard'
        ori_data_filename = checkpoint_context["ori_data_filename"] if "ori_data_filename" in checkpoint_context else 'azure_v2'
        trace = checkpoint_context[
            "trace"] if "trace" in checkpoint_context else None
        x, y = get_ori_data(sequence_length=seq_len, stride=1, shuffle=True, seed=13,
                            ori_data_filename=ori_data_filename, input_output_ratio=input_output_ratio, trace=trace)
        train_idx, val_idx = index_splitter(len(x), [75, 25])
        x_tensor = torch.cat((torch.as_tensor(x), torch.as_tensor(y)), 1)  # full size training
        y_tensor = torch.as_tensor(y)
        x_train_tensor = x_tensor[train_idx]
        x_val_tensor = x_tensor[val_idx]
        y_train_tensor = y_tensor[train_idx]
        y_val_tensor = y_tensor[val_idx]
    if scaler is not None:
        scaled_x_train, _, _ = scale_data(x_train_tensor, scaler=scaler)
    else:
        scaled_x_train, scaler, _ = scale_data(x_train_tensor, scaling_method=scaling_method)
    scaled_y_train, _, _ = scale_data(y_train_tensor, scaler=scaler)
    scaled_x_val, _, _ = scale_data(x_val_tensor, scaler=scaler)
    scaled_y_val, _, _ = scale_data(y_val_tensor, scaler=scaler)
    scaled_x_train_tensor = torch.as_tensor(scaled_x_train)
    scaled_x_val_tensor = torch.as_tensor(scaled_x_val)
    model, optimizer, checkpoint = prepare_model(scaled_x_train_tensor.shape[2], checkpoint)
    train_loader, test_loader, train_data, test_data = get_data_loaders(scaled_x_train_tensor, scaled_x_val_tensor,
                                                                        seq_len, batch_size, input_output_ratio)
    epoch = checkpoint['epoch']
    experiment_root_directory_name = f'{experiment_dir}/epoch_{epoch}/'
    torch.manual_seed(43)
    sbs_transf = StepByStep(model, None, optimizer, device=args.device)
    sbs_transf.set_loaders(train_loader, test_loader)
    export_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    generated_data_directory_name = experiment_root_directory_name + "generated_data/"
    os.makedirs(generated_data_directory_name, exist_ok=True)
    export_iterator = iter(export_loader)
    n_samples_export = args.n_samples_export
    for i in trange(n_samples_export, leave=False, colour='green'):
        try:
            input_batch, output_batch = next(export_iterator)
        except StopIteration:
            export_iterator = iter(export_loader)
            input_batch, output_batch = next(export_iterator)
        if model_type == "Transformer":
            input_batch = torch.concat((input_batch, output_batch), dim=1)
        predicted_sequence = sbs_transf.predict(input_batch)[0]
        rescaled_sequence = np.reshape(scaler.inverse_transform(predicted_sequence.reshape(-1, 1)),
                                       predicted_sequence.shape)
        np.savetxt(f'{generated_data_directory_name}/sample_{i}.csv', rescaled_sequence, delimiter=",")

def main(args):
    #print(args)
    experiment_directories = []
    if args.recursive == True:
        root_dir = args.experiment_directory_path
        experiment_directories = []
        for subdir, dirs, files in os.walk(root_dir):
            if 'checkpoints' in dirs:
                experiment_directories.append(subdir)
    else:
        experiment_directories.append(args.experiment_directory_path)

    progress_bar = tqdm(experiment_directories, colour='red')
    for experiment_dir in progress_bar:
        progress_bar.set_description(f'Creating samples for {experiment_dir}')
        epoch = args.epoch
        if epoch == -1:
            checkpoints_dir = f'{experiment_dir}/checkpoints/'
            assert os.path.exists(checkpoints_dir) and len(os.listdir(checkpoints_dir)) > 0, f'{experiment_dir}checkpoints/ does not exist or is empty'
            checkpoint_paths = [f'{checkpoints_dir}{checkpoint_name}' for checkpoint_name in sorted(os.listdir(checkpoints_dir), key=lambda fileName: int(fileName.split('.')[0].split('_')[1]), reverse=True)]
        else:
            checkpoint_pth_file = f'{experiment_dir}/checkpoints/epoch_{epoch}.pth'
            checkpoint_paths = [checkpoint_pth_file]
        for checkpoint_path in tqdm(checkpoint_paths, leave=False, colour='yellow'):
            assert os.path.exists(checkpoint_path), f'{checkpoint_path} does not exist'
            export_checkpoint(experiment_dir, checkpoint_path, args)

if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_directory_path',
        type=str)
    parser.add_argument(
        '--epoch',
        default=-1,
        type=int)
    parser.add_argument(
        '--n_samples_export',
        default=10,
        type=int)
    parser.add_argument(
        '--recursive',
        default=False,
        type=lambda x: bool(util.strtobool(str(x))))
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        type=str)
    args = parser.parse_args()
    main(args)