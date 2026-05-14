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
CONTROL_MIN = torch.tensor([-1500, -1500, -1500, -1500, -1500, -1500, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]).to(DEVICE)
CONTROL_MAX = torch.tensor([1500, 1500, 1500, 1500, 1500, 1500, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2]).to(DEVICE)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时设置所有GPU

set_seed(42)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def eval(encoder, dkm, dst):
    qbar = tqdm.tqdm(dst)
    encoder.eval()
    dkm.eval()

    dkm_avg_eval_error = 0.0
    dkm_final_eval_error = 0.0
    yaw_avg_eval_error = 0.0
    yaw_final_eval_error = 0.0
    vx_eval_error = 0.0
    vy_eval_error = 0.0
    wz_eval_error = 0.0
    sample_cnt = 0

    for i, sample in enumerate(qbar):
        with torch.no_grad():
            states = torch.stack([
                sample['x_est_gt'],
                sample['y_est_gt'],
                sample['thetaz_est_gt'],
                sample['vx_est_gt'],
                sample['vy_est_gt'],
                sample['wz_est_gt'],
            ], dim=-1)
            controls = torch.cat([
                sample['T_motor']['total'],
                sample['delta_act']['total']
            ], dim=-1)
            states[:, :, :2] -= states[:, 0:1, :2].repeat(1, states.shape[1], 1)
            controls_norm = (controls - CONTROL_MIN) / (CONTROL_MAX - CONTROL_MIN + 1e-6)
            state_embeds_res = encoder(states)
            state_embeds_full = torch.cat([states, state_embeds_res], dim=-1)
            ITERS = states.shape[1] - 1
            result_embeds_full = dkm(state_embeds_full[:, :-1, :], controls_norm, iters=ITERS)
            est_states = result_embeds_full[..., :6]

        dkm_loss = torch.norm(est_states[:, :, :2] - states[:, 1:, :2], p=2, dim=-1) # multi-step prediction loss in state space
        yaw_loss = torch.abs(est_states[:, :, 2] - states[:, 1:, 2])
        
        dkm_avg_eval_error += torch.sum(dkm_loss)
        yaw_avg_eval_error += torch.sum(yaw_loss)

        fde_dkm_loss = torch.norm(est_states[:, -1, :2] - states[:, -1, :2],  p=2, dim=-1)
        yaw_dkm_loss = torch.abs(est_states[:, -1, 2] - states[:, -1, 2])

        dkm_final_eval_error += torch.sum(fde_dkm_loss)
        yaw_final_eval_error += torch.sum(yaw_dkm_loss)

        vx_eval_error += torch.sum(torch.abs(est_states[:, :, 3] - states[:, 1:, 3]))
        vy_eval_error += torch.sum(torch.abs(est_states[:, :, 4] - states[:, 1:, 4]))
        wz_eval_error += torch.sum(torch.abs(est_states[:, :, 5] - states[:, 1:, 5]))
        
        sample_cnt += states.shape[0]
    
    dkm_avg_eval_error /= (sample_cnt * ITERS)
    dkm_final_eval_error /= sample_cnt
    yaw_avg_eval_error /= (sample_cnt * ITERS)
    yaw_avg_eval_error *= (180.0 / np.pi)
    yaw_final_eval_error /= sample_cnt
    yaw_final_eval_error *= (180.0 / np.pi)
    vx_eval_error /= (sample_cnt * ITERS)
    vy_eval_error /= (sample_cnt * ITERS)
    wz_eval_error /= (sample_cnt * ITERS)
    
    print('Valid:')
    print('DKM model ADE: %.4f' % dkm_avg_eval_error.item(), 'FDE: %.4f' % dkm_final_eval_error.item())
    print('DKM model average yaw error: %.4f' % yaw_avg_eval_error.item(), 'final yaw error: %.4f' % yaw_final_eval_error.item())
    print('vx_eval_error: %.4f' % vx_eval_error.item(), 'vy_eval_error: %.4f' % vy_eval_error.item(), 'wz_eval_error: %.4f' % (wz_eval_error.item() * (180.0/np.pi)))

    return dkm_avg_eval_error, dkm_final_eval_error, vx_eval_error, vy_eval_error, wz_eval_error, yaw_avg_eval_error, yaw_final_eval_error


if __name__ == '__main__':
    #######################
    # choose eval condition
    eval_condition = 'CC'
    eval_condition = eval_condition.upper()
    assert eval_condition in list(condition_dir_dict.keys()), 'no such condition'

    #######################
    # choose eval length
    eval_length = 100
    if eval_length >= 300:
        skip = 25
    else:
        skip = 100
    valid_dir = condition_dir_dict[eval_condition]
    valid_files = [os.path.join(valid_dir, x) for x in sorted(os.listdir(valid_dir))]
    valid_dsts = []
    for file in valid_files:
        valid_dsts.append(VehicleDynamicDataset5C(mat_fpath=file, horizon=eval_length, device='cuda:0', show_log=False, skip=skip))
    valid_dst = ConcatDataset(valid_dsts)
    valid_loader = torch.utils.data.DataLoader(valid_dst, batch_size=512, shuffle=False, num_workers=0, drop_last=False)

    #######################
    # config model
    def build_model(ckpt_path):
        encoder = CustomEncoderUnscaledv2WithoutNorm(state_dim=6, hidden_dim=16, layer_depth=6)
        dkm = Koopmanv1(state_dim=6, hidden_dim=16, control_dim=12)

        print('Params of encoder:', count_parameters(encoder) / 1e6, 'M')
        print('Params of dkm:', count_parameters(dkm) / 1e6, 'M')

        ckpt = torch.load(ckpt_path)
        # print(ckpt['koopman'])
        print(ckpt['eval_results'])
        print(ckpt['epoch'])
        encoder.load_state_dict(ckpt['encoder'])
        dkm.load_state_dict(ckpt['koopman'])

        encoder = encoder.to(DEVICE)
        dkm = dkm.to(DEVICE)
        return encoder, dkm

    encoder, dkm = build_model(
        ckpt_path='base.pth'
    )

    eval(encoder, dkm, valid_loader)