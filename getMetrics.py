import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from Utils.GeoData import generate_data, process_adj
from Utils.Evaluation import evaluation


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
    _, _, test_data = generate_data(args)
    adj = process_adj(args['adj'])

    print(f'Time {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) } | Loaded dataset >> {args["dataset"]}')

    model = torch.load(args['saved_path'])
    metrics = testing(test_data, adj, model, args)

    print(f'Time {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) } | Testing results.')
    for idx, m in enumerate(metrics):
        print(f'Predicting Flow at {idx} | MAPE: {m[0]: .3%}, MAE: {m[1]: .5f}, RMSE: {m[2]: .5f}')

    enc_adj = model.enc_adj.cpu().numpy()
    dec_adj = model.dec_adj.cpu().numpy()
    
    np.save(f"./Datasets/{args['dataset']}/{args['dataset']}_enc_adj.npy", enc_adj)
    np.save(f"./Datasets/{args['dataset']}/{args['dataset']}_dec_adj.npy", dec_adj)


if __name__ == '__main__':
    HHY_config = {
        'dataset': 'HHY',
        'flow': './Datasets/HHY/Flow.npy',
        'adj': './Datasets/HHY/dists.npy',
        'nodes': 8,
        'batch_size': 16,
        'epochs': 350,
        'train_rate': 0.6,
        'val_rate': 0.1,
        'learning_rate': 0.001,
        'his': 12,
        'next': 1,
        'droprate': 0,
        'embfeats': 16,
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
        # For Results, default setting: False
        'diffusion_type': 'PPR',  # choose from PPR, HK Or Attention
        'adj_type': 'D',  # 'D' means Dynamic, 'S'means static
        'val_res_saved': True,
        'dynamic_adj_saved': True,
    }
    
    METR_config = {
        'dataset': 'METR',
        'flow': './Datasets/METR/Flow.npy',
        'adj': './Datasets/METR/dists.npy',
        'nodes': 207,
        'batch_size': 8,
        'epochs': 120,  # about 2.5 hours
        'train_rate': 0.6,
        'val_rate': 0.1,
        'learning_rate': 0.001,
        'his': 12,
        'next': 1,
        'droprate': 0,
        'embfeats': 16,
        'hidfeats':64,
        'g_nhead': 8,
        'd_nhead': 8,
        'kdim': 16,
        'vdim': 16,
        'enc_layers': 2,
        'dec_layers': 2,
        'max_diff_step': 2,
        'testing': True,
        'saving': True,
        'saved_path': './METR.pt',
        # For Results, default setting: False
        'diffusion_type': 'Attention',  # choose from PPR, HK Or Attention
        'adj_type': 'D',  # 'D' means Dynamic, 'S'means static
        'val_res_saved': False,
        'dynamic_adj_saved': False,
    }
    
    PeMS_config = {
        'dataset': 'PeMS',
        'flow': './Datasets/PeMS/Flow.npy',
        'adj': './Datasets/PeMS/dists.npy',
        'nodes': 555,
        'batch_size': 4,
        'epochs': 20,  # about 35 mins
        'train_rate': 0.6,
        'val_rate': 0.1,
        'learning_rate': 0.001,
        'his': 12,
        'next': 1,
        'droprate': 0,
        'embfeats': 16,
        'hidfeats':64,
        'g_nhead': 8,
        'd_nhead': 8,
        'kdim': 16,
        'vdim': 16,
        'enc_layers': 2,
        'dec_layers': 2,
        'max_diff_step': 2,
        'testing': True,
        'saving': True,
        'saved_path': './PeMS.pt',
        # For Results, default setting: False
        'diffusion_type': 'Attention',  # choose from PPR, HK Or Attention
        'adj_type': 'D',  # 'D' means Dynamic, 'S'means static
        'val_res_saved': False,
        'dynamic_adj_saved': False,
    }
    
    config = HHY_config
    main(config)
