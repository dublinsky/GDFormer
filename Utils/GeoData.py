import numpy as np
import torch

from torch.utils.data import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Flow(Dataset):
    def __init__(self, x, y, scaler):
        self.size = x.shape[0]
        self.x = x
        self.y = y
        self.scaler = scaler
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.size


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def split_data(data, x_offsets, y_offsets):
    length = data.shape[0]
    data_list = [np.expand_dims(data, axis=-1)]

    data = np.concatenate(data_list, axis=-1)
    x, y = [], []
    
    min_t = abs(min(x_offsets))
    max_t = abs(length - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device)


def generate_data(args):
    data = np.load(args['flow'])
    x_offsets = np.sort(
        np.concatenate((np.arange(1 - args['his'], 1, 1),))
    )

    y_offsets = np.sort(
        np.arange(1 - args['his'], args['next'] + 1, 1)
    )

    x, y = split_data(
        data,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
    )

    samples = x.shape[0]
    num_val = round(samples * args['val_rate'])
    num_train = round(samples * args['train_rate'])

    num_test = samples - num_val - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    scaler = StandardScaler(x_train.mean(), x_train.std())
    print(f'Training set: x with shape {x_train.shape}, y with shape {y_train.shape}')
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    print(f'Valuational set: x with shape {x_val.shape}, y with shape {y_val.shape}')
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]
    print(f'Testing set: x with shape {x_test.shape}, y with shape {y_test.shape}')

    return Flow(x_train, y_train, scaler), Flow(x_val, y_val, scaler), Flow(x_test, y_test, scaler)


def process_adj(adj_path):
    # print(f"Calculating Chebyshev polynomials up to order {degree}...")
    adj = np.load(adj_path)

    A = adj + np.identity(adj.shape[0])
    d = np.sum(A, axis=1)
    # sinvD = np.sqrt(np.mat(np.diag(d)).I)

    # L = np.mat(np.identity(adj.shape[0]) + sinvD * A * sinvD)

    # return torch.from_numpy(L).float().to(device)
    D = np.mat(np.diag(1. / d))
    L = np.matmul(A, D)
    return torch.from_numpy(L).float().to(device)
