import torch
import os
import numpy as np
import scipy.io as scio
import pandas as pd

from torch.utils.data import Dataset

from src.parse_data import load_vehicle_data, preprocess_trucksim_data_3dof, preprocess_trucksim_data_7dof


class VehicleDynamicDataset5C(Dataset):
    
    def __init__(self, mat_fpath, horizon=100, device=torch.device('cpu'), show_log=True, mode='7dof', skip=100):
        super(VehicleDynamicDataset5C, self).__init__()
        # load trucksim data
        self.trucksim_data_raw = load_vehicle_data(mat_fpath)
        self.mode = mode.lower()
        assert self.mode in ['3dof', '7dof']
        # preprocess
        if self.mode == '3dof':
            self.trucksim_data_ppd = preprocess_trucksim_data_3dof(self.trucksim_data_raw)
        else:
            self.trucksim_data_ppd = preprocess_trucksim_data_7dof(self.trucksim_data_raw)
        
        # show log
        if show_log:
            for k in self.trucksim_data_ppd.keys():
                tmp = self.trucksim_data_ppd[k]
                if isinstance(tmp, dict):
                    for kk in tmp.keys():
                        if isinstance(tmp[kk], dict):
                            for kkk in tmp[kk].keys():
                                print('%s [%s] (%s): ' % (k, kk, kkk), np.array(tmp[kk][kkk]).shape)
                        else:
                            print('%s [%s]: ' % (k, kk), np.array(tmp[kk]).shape)
                else:
                    print('%s: ' % k, np.array(tmp).shape)

            # print(self.trucksim_data_ppd['vehicle_dynamics']['Vx'][0], self.trucksim_data_ppd['vehicle_dynamics']['Vy'][0], self.trucksim_data_ppd['vehicle_dynamics']['YawRate'][0], self.trucksim_data_ppd['vehicle_dynamics']['WhlSpd_FL'][0], self.trucksim_data_ppd['vehicle_dynamics']['WhlSpd_FR'][0], self.trucksim_data_ppd['vehicle_dynamics']['WhlSpd_ML'][0], self.trucksim_data_ppd['vehicle_dynamics']['WhlSpd_MR'][0], self.trucksim_data_ppd['vehicle_dynamics']['WhlSpd_RL'][0], self.trucksim_data_ppd['vehicle_dynamics']['WhlSpd_RR'][0])
            # print(self.trucksim_data_ppd['initial_conditions']['vx0'], self.trucksim_data_ppd['initial_conditions']['vy0'], self.trucksim_data_ppd['initial_conditions']['wz0'], self.trucksim_data_ppd['initial_conditions']['w0'])


        condition = mat_fpath.split('/')[-2].split('_')[0]
        condition_embed = torch.zeros(5)
        condition_embed[0] = 1
        self.condition = condition_embed

        self.device = device

        self.skip = skip

        if horizon == -1:
            self.horizon = len(self.trucksim_data_ppd['control_input']['delta']) - self.skip - 1
        else:
            self.horizon = horizon
        
        self.total_samples = len(self.trucksim_data_ppd['control_input']['delta']) - self.horizon - self.skip
        # print(self.total_samples)

        # # trucksim: geometric center -> dynamic model: gravity center
        # sprung_mass = torch.tensor(4455.0)
        # load_mass = torch.tensor(5000.0)
        # unsprung_mass = torch.tensor(350.0 * 3)
        # g = torch.tensor(9.81)
        # sprung_c = torch.tensor(1.250)
        # load_c = torch.tensor(4.0)
        # axis1 = torch.tensor(0.0)
        # axis2 = torch.tensor(3.250)
        # axis3 = torch.tensor(6.500)

        # m = sprung_mass + unsprung_mass + load_mass
        
        # total_c = (sprung_mass * sprung_c + unsprung_mass / 3 * (axis1 + axis2 + axis3) + load_mass * load_c) / m

        # vy_trucksim = torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Vy']).to(self.device)
        # omega = torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['YawRate']).to(device)
        # vy_dynamics = vy_trucksim + (axis3 / 2 - total_c) * omega
        # self.trucksim_data_ppd['vehicle_dynamics']['Vy_processed'] = vy_dynamics

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):

        current_idx = idx + self.skip
        future_idx = idx + self.horizon +self.skip

        # get a clip of data
        clip = {

            # 控制输入
            'delta_act': {
                'total': torch.from_numpy(self.trucksim_data_ppd['control_input']['delta'][current_idx:future_idx, :]).to(self.device), # t ~ t+N
                'fl': torch.from_numpy(self.trucksim_data_ppd['actuators']['steer_angles']['FL_steer'][current_idx:future_idx]).to(self.device),
                'fr': torch.from_numpy(self.trucksim_data_ppd['actuators']['steer_angles']['FR_steer'][current_idx:future_idx]).to(self.device),
                'ml': torch.from_numpy(self.trucksim_data_ppd['actuators']['steer_angles']['ML_steer'][current_idx:future_idx]).to(self.device),
                'mr': torch.from_numpy(self.trucksim_data_ppd['actuators']['steer_angles']['MR_steer'][current_idx:future_idx]).to(self.device),
                'rl': torch.from_numpy(self.trucksim_data_ppd['actuators']['steer_angles']['RL_steer'][current_idx:future_idx]).to(self.device),
                'rr': torch.from_numpy(self.trucksim_data_ppd['actuators']['steer_angles']['RR_steer'][current_idx:future_idx]).to(self.device),
            },
            'T_motor': {
                'total': torch.from_numpy(self.trucksim_data_ppd['control_input']['torque'][current_idx:future_idx, :]).to(self.device), # t ~ t+N
                'fl': torch.from_numpy(self.trucksim_data_ppd['actuators']['wheel_torques']['FL_trq'][current_idx:future_idx]).to(self.device),
                'fr': torch.from_numpy(self.trucksim_data_ppd['actuators']['wheel_torques']['FR_trq'][current_idx:future_idx]).to(self.device),
                'ml': torch.from_numpy(self.trucksim_data_ppd['actuators']['wheel_torques']['ML_trq'][current_idx:future_idx]).to(self.device),
                'mr': torch.from_numpy(self.trucksim_data_ppd['actuators']['wheel_torques']['MR_trq'][current_idx:future_idx]).to(self.device),
                'rl': torch.from_numpy(self.trucksim_data_ppd['actuators']['wheel_torques']['RL_trq'][current_idx:future_idx]).to(self.device),
                'rr': torch.from_numpy(self.trucksim_data_ppd['actuators']['wheel_torques']['RR_trq'][current_idx:future_idx]).to(self.device),
            },

            # 状态序列
            'vx_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Vx'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, vx
            'vy_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Vy'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, vy
            'wz_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['YawRate'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, wz
            'x_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Xpos'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, x
            'y_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Ypos'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, y
            'thetaz_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Yaw'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, z
            'ax_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Ax'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, ax
            'ay_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Ay'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, ay
        
            # 时间戳
            'his_timestamp': torch.from_numpy(np.arange(current_idx, future_idx)), # t ~ t+N
            'est_timestamp': torch.tensor(future_idx), # t+N+1

            # 轮胎受力
            'Fx_FL': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fx_FL'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fx_FL
            'Fy_FL': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fy_FL'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fy_FL
            'Fz_FL': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fz_FL'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fz_FL
            'Fx_FR': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fx_FR'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fx_FR
            'Fy_FR': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fy_FR'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fy_FR
            'Fz_FR': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fz_FR'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fz_FR

            'Fx_ML': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fx_ML'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fx_FL
            'Fy_ML': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fy_ML'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fy_FL
            'Fz_ML': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fz_ML'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fz_FL
            'Fx_MR': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fx_MR'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fx_FR
            'Fy_MR': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fy_MR'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fy_FR
            'Fz_MR': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fz_MR'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fz_FR

            'Fx_RL': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fx_RL'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fx_FL
            'Fy_RL': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fy_RL'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fy_FL
            'Fz_RL': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fz_RL'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fz_FL
            'Fx_RR': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fx_RR'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fx_FR
            'Fy_RR': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fy_RR'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fy_FR
            'Fz_RR': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Fz_RR'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, Fz_FR

            # # 侧偏角（真值）
            # 'Alpha_FL': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Alpha_FL'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1
            # 'Alpha_FR': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Alpha_FR'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1,
            # 'Alpha_ML': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Alpha_ML'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, 
            # 'Alpha_MR': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Alpha_MR'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, 
            # 'Alpha_RL': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Alpha_RL'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, 
            # 'Alpha_RR': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Alpha_RR'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, 

            'condition': self.condition.to(self.device)
        }

        assert clip['delta_act']['total'].shape[0] == self.horizon
        assert clip['vx_est_gt'].shape[0] == self.horizon + 1

        return clip