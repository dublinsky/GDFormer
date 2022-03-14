import numpy as np
import torch


def evaluation(preds, labels, predicted_length, name, null_val=torch.tensor(float('nan')), saving=False):
    metrics_over_timesteps = []
    for time_steps in range(predicted_length):
        if saving:
            np.save(f'./Datasets/{name}/{name}_pred.npy', preds[..., time_steps].cpu().numpy())
            np.save(f'./Datasets/{name}/{name}_true.npy', labels[..., time_steps].cpu().numpy())
        mape = masked_mape(preds[..., time_steps], labels[..., time_steps], null_val)
        mae = masked_mae(preds[..., time_steps], labels[..., time_steps], null_val)
        rmse = masked_rmse(preds[..., time_steps], labels[..., time_steps], null_val)

        metrics = [mape, mae, rmse]
        metrics_over_timesteps.append(metrics)
    return metrics_over_timesteps


def masked_mape(preds, labels, null_val=torch.tensor(float('nan'))):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        null_val_mtx = torch.ones_like(labels) * null_val
        mask = torch.ne(labels, null_val_mtx)
    
    zeros = torch.zeros_like(labels)
    mape = torch.where(mask, (labels - preds) / labels, zeros)
    mape = torch.mean(torch.abs(mape))
    return mape


def masked_mae(preds, labels, null_val=torch.tensor(float('nan'))):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        null_val_mtx = torch.ones_like(labels) * null_val
        mask = torch.ne(labels, null_val_mtx)
    
    zeros = torch.zeros_like(labels)
    mae = torch.where(mask, labels - preds, zeros)
    mae = torch.mean(torch.abs(mae))
    return mae


def masked_rmse(preds, labels, null_val=torch.tensor(float('nan'))):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        null_val_mtx = torch.ones_like(labels) * null_val
        mask = torch.ne(labels, null_val_mtx)
    
    zeros = torch.zeros_like(labels)
    rmse = torch.where(mask, labels - preds, zeros)
    rmse = torch.sqrt(torch.mean(torch.pow(rmse, 2)))
    return rmse
