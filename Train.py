import time
import os
import random
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Utils.GeoData import device, generate_data, process_adj
from Utils.Evaluation import evaluation
from Utils.MaskLoss import masked_mae_loss as masked_loss
from Model.GDFormer import GDF


def training(data, adj, model, args):
    optimizer = optim.Adam([
        {'params': model.parameters()},
    ], lr=args['learning_rate'])

    dataloader = DataLoader(data, batch_size=args['batch_size'], shuffle=True)
    samples = 0
    total_loss = 0.

    for x, y in dataloader:
        x, y = torch.squeeze(x, dim=-1), torch.squeeze(y, dim=-1)
        x, y = torch.permute(x, [0, 2, 1]), torch.permute(y, [0, 2, 1])
        ex_adj = adj.repeat([y.shape[0], 1 , 1])

        x = data.scaler.transform(x)

        truth_y = y[..., -args['next']: ].clone()
        y = data.scaler.transform(y)
        y[..., -args['next']: ] = 0.

        optimizer.zero_grad()

        pred = model(x, ex_adj, y, ex_adj)
        pred = data.scaler.inverse_transform(pred)
        
        # print(f'pred {pred.shape}, truth {truth_y.shape}')

        loss = masked_loss(pred[..., -args['next']: ], truth_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        samples += 1

    return total_loss / samples


def testing(data, adj, model, args, saving=False):
    dataloader = DataLoader(data, batch_size=args['batch_size'])
    preds = []
    truth = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = torch.squeeze(x, dim=-1), torch.squeeze(y, dim=-1)
            x, y = torch.permute(x, [0, 2, 1]), torch.permute(y, [0, 2, 1])
            ex_adj = adj.repeat([y.shape[0], 1 , 1])

            x = data.scaler.transform(x)
            
            truth_y = y[..., -args['next']: ].clone()
            y = data.scaler.transform(y)
            y[..., -args['next']: ] = 0.
            
            pred = model(x, ex_adj, y, ex_adj)
            pred = data.scaler.inverse_transform(pred)

            preds.append(pred[..., -args['next']: ])
            truth.append(truth_y)
    
    metrics = evaluation(torch.cat(preds, 0), torch.cat(truth, 0), args['next'],
                         args['dataset'], torch.tensor(0.), saving=saving)
    return metrics


def main(args):
    val_res_saved_path = f"./ValRes/{args['dataset']}_{args['diffusion_type']}_{args['adj_type']}.csv"
    if os.path.exists(val_res_saved_path):
        with open(val_res_saved_path, 'a') as e_file:
            e_file.write('pred_len,MAPE,MAE,RMSE')
    
    train_data, val_data, test_data = generate_data(args)
    adj = process_adj(args['adj'])

    print(f'Time {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) } | Loaded dataset >> {args["dataset"]}')

    model = GDF(args['his'], args['his'] + args['next'], args['embfeats'],
                args['g_nhead'], args['d_nhead'], args['kdim'], args['vdim'],
                args['hidfeats'], args['next'], args['nodes'], args['enc_layers'],
                args['dec_layers'], args['max_diff_step'], args['droprate'],
                args['diffusion_type'], args['adj_type'], args['dynamic_adj_saved'])
    
    model.to(device)
    print(f'Time {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) } | Builded Model')

    for ep in range(args['epochs']):
        loss = training(train_data, adj, model, args)
        metrics = testing(val_data, adj, model, args)
        print(f'Time {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) } Training {ep} | training loss: {loss: .5f}')
        for idx, m in enumerate(metrics):
            print(f'Predicting Flow at {idx} | valational MAPE: {m[0]: .3%}, MAE: {m[1]: .5f}, RMSE: {m[2]: .5f}')

        if args['val_res_saved']:
            with open(val_res_saved_path, 'a') as e_file:
                for idx, m in enumerate(metrics):
                    e_file.write(f'{idx},{metrics[0]: .5%},{metrics[1]: .5f},{metrics[2]: .5f}\n')

    if args['testing']:
        metrics = testing(test_data, adj, model, args, saving=True)

        print(f'Time {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) } | Testing results.')
        for idx, m in enumerate(metrics):
            print(f'Predicting Flow at {idx} | MAPE: {m[0]: .3%}, MAE: {m[1]: .5f}, RMSE: {m[2]: .5f}')

    if args['saving']:
        torch.save(model, args['saved_path'])


if __name__ == '__main__':
    seed = 567

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    HHY_config = {
        'dataset': 'HHY',
        'flow': './Datasets/HHY/Flow.npy',
        'adj': './Datasets/HHY/dists.npy',
        'nodes': 8,
        'batch_size': 16,
        'epochs': 250,
        'train_rate': 0.6,
        'val_rate': 0.1,
        'learning_rate': 0.001,
        'his': 12,
        'next': 1,
        'droprate': 0,
        'embfeats': 32,
        'hidfeats':64,
        'g_nhead': 8,
        'd_nhead': 8,
        'kdim': 16,
        'vdim': 16,
        'enc_layers': 2,
        'dec_layers': 2,
        'max_diff_step': 2,
        'testing': True,
        'saving': False,
        'saved_path': './HHY.pt',
        'diffusion_type': 'Attention',  # choose from PPR, HK Or Attention
        'adj_type': 'D',  # 'D' means Dynamic, 'S' means Static
        'val_res_saved': False,
        'dynamic_adj_saved': False,
    }
    
    METR_config = {
        'dataset': 'METR',
        'flow': './Datasets/METR/Flow.npy',
        'adj': './Datasets/METR/dists.npy',
        'nodes': 207,
        'batch_size': 8,
        'epochs': 60,  # about 2.5 hours
        'train_rate': 0.6,
        'val_rate': 0.1,
        'learning_rate': 0.001,
        'his': 12,
        'next': 1,
        'droprate': 0.1,
        'embfeats': 32,
        'hidfeats':64,
        'g_nhead': 8,
        'd_nhead': 8,
        'kdim': 16,
        'vdim': 16,
        'enc_layers': 2,
        'dec_layers': 2,
        'max_diff_step': 2,
        'testing': True,
        'saving': False,
        'saved_path': './METR.pt',
        # For Results, default setting: False
        'diffusion_type': 'Attention',  # choose from PPR, HK Or Attention
        'adj_type': 'D',  # 'D' means Dynamic, 'S' means Static
        'val_res_saved': False,
        'dynamic_adj_saved': False,
    }
    
    PeMS_config = {
        'dataset': 'PeMS',
        'flow': './Datasets/PeMS/Flow.npy',
        'adj': './Datasets/PeMS/dists.npy',
        'nodes': 555,
        'batch_size': 8,
        'epochs': 80,  # about 35 mins
        'train_rate': 0.6,
        'val_rate': 0.1,
        'learning_rate': 0.001,
        'his': 12,
        'next': 1,
        'droprate': 0,
        'embfeats': 32,
        'hidfeats':64,
        'g_nhead': 8,
        'd_nhead': 8,
        'kdim': 16,
        'vdim': 16,
        'enc_layers': 2,
        'dec_layers': 2,
        'max_diff_step': 2,
        'testing': True,
        'saving': False,
        'saved_path': './PeMS.pt',
        # For Results, default setting: False
        'diffusion_type': 'Attention',  # choose from PPR, HK Or Attention
        'adj_type': 'D',  # 'D' means Dynamic, 'S' means Static
        'val_res_saved': False,
        'dynamic_adj_saved': False,
    }
    
    config = HHY_config
    main(config)
