import numpy as np
import pickle
import os
from scipy.stats import hmean

def open_folder(folder):
    data = []

    for file in os.listdir(folder):
        filepath = f'{folder}/{file}'
        print(f'reading {filepath}')
        with open(filepath, 'rb') as f:
            current_data = pickle.load(f)
            data.append(current_data)

    data = np.concatenate(data, axis=1)
    return data

def prepare_data_GL(data):
    exp_combined = data.reshape((9, -1, 4, 1))
    fdr_c, fomr_c, tnr_c, tpr_c = [], [], [], []
    bacc_c, f1_c, mcc_c = [], [], []

    for i in range(9):
        tns = exp_combined[i, :, 0, :]
        fns = exp_combined[i, :, 1, :]
        fps = exp_combined[i, :, 2, :]
        tps = exp_combined[i, :, 3, :]

        tnr = tns / (fps + tns)
        tpr = tps / (fns + tps)
        fdr = fps / (fps + tps)
        fomr = fns / (fns + tns)

        fdr_f = np.nan_to_num(fdr)
        fomr_f = np.nan_to_num(fomr)
        tnr_f = np.nan_to_num(tnr, nan=1)
        tpr_f = np.nan_to_num(tpr, nan=1)

        bacc = (tpr_f + tnr_f) / 2
        f1 = hmean([1 - fdr_f, tpr_f], axis=0)

        mcc_first = tpr_f * tnr_f * (1 - fdr_f) * (1 - fomr_f)
        mcc_second = (1 - tpr_f) * (1 - tnr_f) * fomr_f * fdr_f
        mcc = np.sqrt(mcc_first) - np.sqrt(mcc_second)

        fdr_c.append(fdr_f.mean(axis=0))
        fomr_c.append(fomr_f.mean(axis=0))
        tnr_c.append(tnr_f.mean(axis=0))
        tpr_c.append(tpr_f.mean(axis=0))

        bacc_c.append(bacc.mean(axis=0))
        f1_c.append(f1.mean(axis=0))
        mcc_c.append(mcc.mean(axis=0))

    fdr_c, fomr_c, tnr_c, tpr_c = np.stack(fdr_c), np.stack(fomr_c), np.stack(tnr_c), np.stack(tpr_c)
    bacc_c, f1_c, mcc_c = np.stack(bacc_c), np.stack(f1_c), np.stack(mcc_c)
    
    return fdr_c, fomr_c, tnr_c, tpr_c, bacc_c, f1_c, mcc_c

def prepare_data(data):
    exp_combined = data.reshape((9, -1, 4, 5))
    fdr_c, fomr_c, tnr_c, tpr_c = [], [], [], []
    bacc_c, f1_c, mcc_c = [], [], []

    for i in range(9):
        tns = exp_combined[i, :, 0, :]
        fns = exp_combined[i, :, 1, :]
        fps = exp_combined[i, :, 2, :]
        tps = exp_combined[i, :, 3, :]

        tnr = tns / (fps + tns)
        tpr = tps / (fns + tps)
        fdr = fps / (fps + tps)
        fomr = fns / (fns + tns)

        fdr_f = np.nan_to_num(fdr)
        fomr_f = np.nan_to_num(fomr)
        tnr_f = np.nan_to_num(tnr, nan=1)
        tpr_f = np.nan_to_num(tpr, nan=1)

        bacc = (tpr + tnr) / 2
        f1 = hmean([1 - fdr_f, tpr_f], axis=0)

        mcc_first = tpr_f * tnr_f * (1 - fdr_f) * (1 - fomr_f)
        mcc_second = (1 - tpr_f) * (1 - tnr_f) * fomr_f * fdr_f
        mcc = np.sqrt(mcc_first) - np.sqrt(mcc_second)

        fdr_c.append(fdr_f.mean(axis=0))
        fomr_c.append(fomr_f.mean(axis=0))
        tnr_c.append(tnr_f.mean(axis=0))
        tpr_c.append(tpr_f.mean(axis=0))

        bacc_c.append(bacc.mean(axis=0))
        f1_c.append(f1.mean(axis=0))
        mcc_c.append(mcc.mean(axis=0))

    fdr_c, fomr_c, tnr_c, tpr_c = np.stack(fdr_c), np.stack(fomr_c), np.stack(tnr_c), np.stack(tpr_c)
    bacc_c, f1_c, mcc_c = np.stack(bacc_c), np.stack(f1_c), np.stack(mcc_c)
    
    return fdr_c, fomr_c, tnr_c, tpr_c, bacc_c, f1_c, mcc_c