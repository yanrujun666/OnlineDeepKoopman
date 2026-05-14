import os
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random

from torch.utils.data import ConcatDataset

from src.loader import VehicleDynamicDataset5C
from src.encoder import CustomEncoderUnscaledv2WithoutNorm
from src.Koopman import Koopmanv1

#######################
# default
condition_dir_dict = {
    'AA': 'data/data_scenarios_for_differ_params_v8/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA-m_sprungA-LfmA',
    'AB': 'data/data_scenarios_for_differ_params_v8/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA-m_sprungA-LfmB',
    'AC': 'data/data_scenarios_for_differ_params_v8/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA-m_sprungA-LfmC',
    'BA': 'data/data_scenarios_for_differ_params_v8/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA-m_sprungB-LfmA',
    'BB': 'data/data_scenarios_for_differ_params_v8/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA-m_sprungB-LfmB',
    'BC': 'data/data_scenarios_for_differ_params_v8/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA-m_sprungB-LfmC',
    'CA': 'data/data_scenarios_for_differ_params_v8/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA-m_sprungC-LfmA',
    'CB': 'data/data_scenarios_for_differ_params_v8/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA-m_sprungC-LfmB',
    'CC': 'data/data_scenarios_for_differ_params_v8/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA-m_sprungC-LfmC',
}

DEVICE = torch.device('cuda:0')


if __name__ == '__main__':
    #######################
    # choose eval condition
    eval_condition = 'AB'
    eval_condition = eval_condition.upper()
    assert eval_condition in list(condition_dir_dict.keys()), 'no such condition'

    #######################
    valid_dir = condition_dir_dict[eval_condition]
    valid_files = [os.path.join(valid_dir, x) for x in sorted(os.listdir(valid_dir))]
    valid_dst = VehicleDynamicDataset5C(mat_fpath=valid_files[0], horizon=1, device='cuda:0', show_log=False, skip=100)
    valid_loader = torch.utils.data.DataLoader(valid_dst, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    #######################
    x = []
    y = []
    for sample in valid_loader:
        x.append(sample['x_est_gt'][0,0].item())
        y.append(sample['y_est_gt'][0,0].item())

    x = np.array(x)
    y = np.array(y)

    plt.plot(x, y)
    plt.savefig('trajectory.png')