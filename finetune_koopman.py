import torch
import tqdm
import torch.nn.functional as F
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import shutil
import random
from datetime import datetime

from torch.utils.data import ConcatDataset

from src.encoder import CustomEncoderUnscaledv2WithoutNorm
from src.Koopman import Koopmanv1
from src.loader import VehicleDynamicDataset5C
from src.monitor import EarlyStopping

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时设置所有GPU
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

set_seed(42)

def create_run_id():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def init_run_paths(script_name, condition, run_id):
    log_dir = os.path.join('log', script_name, condition)
    ckpt_dir = os.path.join('ckpt', script_name, condition)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(log_dir, f'{run_id}.log'), os.path.join(ckpt_dir, f'{run_id}.pth')


class RunLogger:
    def __init__(self, log_path, run_id):
        self.log_path = log_path
        self.run_id = run_id
        self.run_start = datetime.now()
        self._epoch_start = None

    def _emit(self, line):
        print(line, flush=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def blank(self):
        print(flush=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write('\n')

    def write(self, msg, phase=None, timestamp=True):
        parts = []
        if timestamp:
            parts.append(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]')
        if phase:
            parts.append(f'[{phase}]')
        parts.append(str(msg))
        self._emit(' '.join(parts))

    def write_header(self, config_lines):
        self.blank()
        self._emit('=' * 60)
        self._emit(f'run_id={self.run_id}')
        self._emit(f'Run started at {self.run_start.strftime("%Y-%m-%d %H:%M:%S")}')
        self.blank()
        for line in config_lines:
            if line.startswith('run_id='):
                continue
            self._emit(line)
        self.blank()
        self._emit('=' * 60)
        self.blank()

    def start_epoch(self, epoch):
        if epoch > 0:
            self.blank()
        self._epoch_start = time.time()
        self._emit(f'--- Epoch {epoch} ---')

    def end_epoch(self, epoch):
        elapsed = time.time() - self._epoch_start if self._epoch_start is not None else 0.0
        self.write(f'finished, elapsed={elapsed:.1f}s', phase='epoch')

    def finish_run(self):
        self.blank()
        total = (datetime.now() - self.run_start).total_seconds()
        self.write(f'Run finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', phase='summary')
        self.write(f'Total elapsed: {total:.1f}s', phase='summary')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

ITERS = 100
BATCH_SIZE = 512
MAX_EPOCH = 16
DEVICE = torch.device('cuda:0')
CONTROL_MIN = torch.tensor([-1500, -1500, -1500, -1500, -1500, -1500, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]).to(DEVICE)
CONTROL_MAX = torch.tensor([1500, 1500, 1500, 1500, 1500, 1500, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2]).to(DEVICE)
CONDITION = 'AB'


def build_model():
    encoder = CustomEncoderUnscaledv2WithoutNorm(state_dim=6, hidden_dim=16, layer_depth=6)
    dkm = Koopmanv1(state_dim=6, hidden_dim=16, control_dim=12)

    ckpt = torch.load("base.pth")
    encoder.load_state_dict(ckpt['encoder'])
    dkm.load_state_dict(ckpt['koopman'])

    print('Params of encoder:', count_parameters(encoder) / 1e6, 'M')
    print('Params of dkm:', count_parameters(dkm) / 1e6, 'M')

    return encoder, dkm


def build_loader():

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

    data_dir = condition_dir_dict[CONDITION]
    all_files = [os.path.join(data_dir, x) for x in sorted(os.listdir(data_dir))]

    train_dst = VehicleDynamicDataset5C(mat_fpath=all_files[0], horizon=ITERS, device=DEVICE, show_log=False, skip=100)
    print('Get %d samples in the training dataset.' % len(train_dst))
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    valid_dsts = []
    for file in all_files[1:]:
        valid_dsts.append(VehicleDynamicDataset5C(mat_fpath=file, horizon=ITERS, device=DEVICE, show_log=False, skip=100))
    valid_dst = ConcatDataset(valid_dsts)
    print('Get %d samples in the validation dataset.' % len(valid_dst))
    valid_loader = torch.utils.data.DataLoader(valid_dst, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    return train_loader, valid_loader


def build_optimizer(encoder, dkm):
    # parameters = list(encoder.parameters()) + list(dkm.parameters())
    parameters = list(dkm.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.00001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    return optimizer, scheduler


def train(encoder, dkm, loader, optimizer, scheduler, encoder_old=None, dkm_old=None, logger=None):
    qbar = tqdm.tqdm(loader)
    encoder.eval()
    dkm.train()
    epoch_avg_loss = 0.0
    dkm_avg_eval_error = 0.0
    dkm_final_eval_error = 0.0
    vx_eval_error = 0.0
    vy_eval_error = 0.0
    wz_eval_error = 0.0
    sample_cnt = 0
    for i, sample in enumerate(qbar):
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
        # transform to local frame
        states[:, :, :2] -= states[:, 0:1, :2].repeat(1, states.shape[1], 1)
        controls_norm = (controls - CONTROL_MIN) / (CONTROL_MAX - CONTROL_MIN + 1e-6)

        with torch.no_grad():
            state_embeds = encoder(states)
        state_embeds_full = torch.cat([states, state_embeds], dim=-1)
        result_embeds_full = dkm(state_embeds_full[:, :-1, :], controls_norm, iters=ITERS)
        est_states = result_embeds_full[:, :, :6]

        loss_pd_multi_fs = torch.mean(torch.norm((result_embeds_full - state_embeds_full[:, 1:, :]), p=2, dim=-1)) # multi-step prediction loss in feature space
        loss_pd_multi_ss_scaled = torch.mean(torch.norm((est_states - states[:, 1:, :]), p=2, dim=-1)) # # multi-step prediction loss in state space

        # loss_pd_single_ss = 0
        loss_pd_single_fs = 0
        loss_pd_single_ss_scaled = 0
        for i in range(ITERS):
            result_embeds_full_s = dkm(state_embeds_full[:, i:i+1, :], controls_norm[:, i:i+1, :], iters=1)
            est_states_s = result_embeds_full_s[:, :, :6]
            loss_pd_single_fs += torch.mean(torch.norm((result_embeds_full_s - state_embeds_full[:, i+1:i+2, :]), p=2, dim=-1))
            loss_pd_single_ss_scaled += torch.mean(torch.norm((est_states_s - states[:, i+1:i+2, :]), p=2, dim=-1))
        loss_pd_single_fs /= ITERS
        loss_pd_single_ss_scaled /= ITERS

        def phycise_informed_loss(est, gt, dt=0.01):

            dX = est[:, :-1, 3] * torch.cos(est[:, :-1, 2]) - est[:, :-1, 4] * torch.sin(est[:, :-1, 2])
            dY = est[:, :-1, 3] * torch.sin(est[:, :-1, 2]) + est[:, :-1, 4] * torch.cos(est[:, :-1, 2])

            dX_gt = gt[:, :-1, 3] * torch.cos(gt[:, :-1, 2]) - gt[:, :-1, 4] * torch.sin(gt[:, :-1, 2])
            dY_gt = gt[:, :-1, 3] * torch.sin(gt[:, :-1, 2]) + gt[:, :-1, 4] * torch.cos(gt[:, :-1, 2])

            dX_gt_w = torch.max(torch.abs(dX_gt), dim=-1).values + 1e-6
            dY_gt_w = torch.max(torch.abs(dY_gt), dim=-1).values + 1e-6
            dW_gt_w = torch.max(torch.abs(gt[:, :-1, 5]), dim=-1).values + 1e-6

            T = dX.shape[1]
            time_w = (torch.arange(T, dtype=torch.float32, device=est.device) / max(T - 1, 1)) ** 2 + 1.0
            time_w = time_w.unsqueeze(0).repeat(est.shape[0], 1) / T

            vx_pd_consistency_error = torch.abs((est[:, 1:, 0] - est[:, :-1, 0]) / dt - dX)
            vy_pd_consistency_error = torch.abs((est[:, 1:, 1] - est[:, :-1, 1]) / dt - dY)
            wz_pd_consistency_error = torch.abs((est[:, 1:, 2] - est[:, :-1, 2]) / dt - est[:, :-1, 5])

            vx_gt_consistency_error = torch.abs((gt[:, 1:, 0] - gt[:, :-1, 0]) / dt - dX_gt)
            vy_gt_consistency_error = torch.abs((gt[:, 1:, 1] - gt[:, :-1, 1]) / dt - dY_gt)
            wz_gt_consistency_error = torch.abs((gt[:, 1:, 2] - gt[:, :-1, 2]) / dt - gt[:, :-1, 5])

            vx_pd_consistency_error = torch.sum(time_w * F.relu(vx_pd_consistency_error - vx_gt_consistency_error), dim=-1)
            vy_pd_consistency_error = torch.sum(time_w * F.relu(vy_pd_consistency_error - vy_gt_consistency_error), dim=-1)
            wz_pd_consistency_error = torch.sum(time_w * F.relu(wz_pd_consistency_error - wz_gt_consistency_error), dim=-1)

            vx_rel_error = torch.mean(vx_pd_consistency_error / dX_gt_w) 
            vy_rel_error = torch.mean(vy_pd_consistency_error / dY_gt_w)
            wz_rel_error = torch.mean(wz_pd_consistency_error / dW_gt_w)

            return vx_rel_error + vy_rel_error + 0.01 * wz_rel_error

        pi_loss = phycise_informed_loss(est_states, states[:, 1:, :])

        loss_reg = 0.0
        # for param in encoder.parameters():
        #     loss_reg += torch.norm(param, p=2)
        for param in dkm.named_parameters():
            # print(param[0], param[1].shape, dkm_old[param[0]].shape)
            if dkm_old is not None:
                loss_reg += torch.norm(param[1] - dkm_old[param[0]], p=2)
            else:
                loss_reg += torch.norm(param[1], p=2)
        # print(loss_reg)

        loss =  (loss_pd_multi_ss_scaled + loss_pd_single_ss_scaled) + \
                    0.1 * (loss_pd_multi_fs + loss_pd_single_fs) + \
                        0.05 * pi_loss 
        
        
        scale = 1e-6
        loss += scale * loss_reg
        
        epoch_avg_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(dkm.parameters(), max_norm=5.0)
        optimizer.step()            

        qbar.set_postfix({'loss': f"{loss.item():.3f}", \
                            'L_ss_m': f"{loss_pd_multi_ss_scaled.item():.2f}", \
                                'L_ss_s': f"{loss_pd_single_ss_scaled.item():.2f}", \
                                    "L_fs_m": f"{loss_pd_multi_fs.item():.2f}", \
                                        "L_fs_s": f"{loss_pd_single_fs.item():.2f}", \
                                            "L_pi": f"{pi_loss.item():.2f}", \
                                                "L_reg": f"{loss_reg.item():.6f}"})
        
        dkm_loss = torch.norm(est_states[:, :, :2] - states[:, 1:, :2], p=2, dim=-1) # multi-step prediction loss in state space
        dkm_avg_eval_error += torch.sum(dkm_loss)
        fde_dkm_loss = torch.norm(est_states[:, -1, :2] - states[:, -1, :2],  p=2, dim=-1)
        dkm_final_eval_error += torch.sum(fde_dkm_loss)

        vx_eval_error += torch.sum(torch.abs(est_states[:, :, 3] - states[:, 1:, 3]))
        vy_eval_error += torch.sum(torch.abs(est_states[:, :, 4] - states[:, 1:, 4]))
        wz_eval_error += torch.sum(torch.abs(est_states[:, :, 5] - states[:, 1:, 5]))

        sample_cnt += dkm_loss.shape[0]

    dkm_avg_eval_error /= (sample_cnt * ITERS)
    dkm_final_eval_error /= sample_cnt
    vx_eval_error /= (sample_cnt * ITERS)
    vy_eval_error /= (sample_cnt * ITERS)
    wz_eval_error /= (sample_cnt * ITERS)
    
    lines = [
        'Train:',
        f'DKM model mean error: {dkm_avg_eval_error.item():.6f}, final error: {dkm_final_eval_error.item():.6f}',
        f'vx_eval_error: {vx_eval_error.item():.6f}, vy_eval_error: {vy_eval_error.item():.6f}, wz_eval_error: {wz_eval_error.item():.6f}',
    ]
    if logger:
        for line in lines:
            logger.write(line, phase='train')
    else:
        for line in lines:
            print(line)

    return encoder, dkm, epoch_avg_loss / len(qbar), dkm_avg_eval_error, dkm_final_eval_error, vx_eval_error, vy_eval_error, wz_eval_error


def eval(encoder, dkm, loader, ei, condition, logger=None, split='train'):
    qbar = tqdm.tqdm(loader)
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
            # transform to local frame
            states[:, :, :2] -= states[:, 0:1, :2].repeat(1, states.shape[1], 1)
            controls_norm = (controls - CONTROL_MIN) / (CONTROL_MAX - CONTROL_MIN + 1e-6)
            state_embeds_res = encoder(states)
            state_embeds_full = torch.cat([states, state_embeds_res], dim=-1)
            result_embeds_full = dkm(state_embeds_full[:, :-1, :], controls_norm, iters=ITERS)
            est_states = result_embeds_full[:, :, :6]
        
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
    
    phase = 'train_eval' if split == 'train' else 'valid'
    lines = [
        f'Eval on {condition} {split} set (epoch {ei}):',
        f'DKM model ADE: {dkm_avg_eval_error.item():.6f}, FDE: {dkm_final_eval_error.item():.6f}',
        f'DKM model average yaw error: {yaw_avg_eval_error.item():.6f}, final yaw error: {yaw_final_eval_error.item():.6f}',
        f'vx_eval_error: {vx_eval_error.item():.6f}, vy_eval_error: {vy_eval_error.item():.6f}, wz_eval_error: {wz_eval_error.item():.6f}',
    ]
    if logger:
        for line in lines:
            logger.write(line, phase=phase)
    else:
        for line in lines:
            print(line)

    return dkm_avg_eval_error, dkm_final_eval_error, vx_eval_error, vy_eval_error, wz_eval_error, yaw_avg_eval_error, yaw_final_eval_error

if __name__ == '__main__':

    run_id = create_run_id()
    log_filname, ckpt_filname = init_run_paths('finetune-koopman', CONDITION, run_id)
    logger = RunLogger(log_filname, run_id)

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
    data_dir = condition_dir_dict[CONDITION]

    logger.write_header([
        'script=finetune_koopman.py',
        f'condition={CONDITION}',
        f'data_dir={data_dir}',
        'base_checkpoint=base.pth',
        f'checkpoint_path={ckpt_filname}',
        f'log_path={log_filname}',
        f'device={DEVICE}',
        'seed=42',
        f'ITERS={ITERS}',
        f'BATCH_SIZE={BATCH_SIZE}',
        f'MAX_EPOCH={MAX_EPOCH}',
        'optimizer=Adam(lr=1e-5, weight_decay=1e-5) on koopman only',
        'scheduler=StepLR(step_size=50, gamma=0.9)',
        'trainable=koopman (encoder frozen)',
        'eval_split=train',
        'early_stopping_patience=1000',
        'early_stopping_restore_best_weights=False',
    ])

    encoder, dkm = build_model()
    encoder = encoder.to(DEVICE)
    dkm = dkm.to(DEVICE)

    train_loader, valid_loader = build_loader()

    optimizer, scheduler = build_optimizer(encoder, dkm)

    early_stopping = EarlyStopping(patience=1000, min_delta=0.0, restore_best_weights=False, weight_save_fpath=ckpt_filname)

    logger.write('Initial evaluation on train set before training', phase='train_eval')
    eval(encoder, dkm, train_loader, 0, CONDITION, logger=logger)

    for ei in range(MAX_EPOCH):
        logger.start_epoch(ei)
        encoder, dkm, train_loss, \
             dkm_avg_eval_error, dkm_final_eval_error, \
                vx_eval_error, vy_eval_error, wz_eval_error = train(
                    encoder, dkm, train_loader, optimizer, scheduler, logger=logger)
        logger.write(f'train epoch mean loss: {train_loss:.6f}', phase='train')

        results = {}

        dkm_avg_eval_error, dkm_final_eval_error, \
            vx_eval_error, vy_eval_error, wz_eval_error, \
                yaw_avg_eval_error, yaw_final_eval_error = eval(encoder, dkm, train_loader, ei, CONDITION, logger=logger)

        results['real'] = {
            'dkm_mde': dkm_avg_eval_error.item(),
            'dkm_fde': dkm_final_eval_error.item(),
            'dkm_mae': yaw_avg_eval_error.item(),
            'dkm_fae': yaw_final_eval_error.item(),
            'vx_error': vx_eval_error.item(),
            'vy_error': vy_eval_error.item(),
            'wz_error': wz_eval_error.item(),
        }
        
        model_weight = {
            'encoder': encoder.state_dict().copy(),
            'koopman': dkm.state_dict().copy(),
            'best_loss': train_loss,
            'eval_results': results,
            'epoch': ei,
            'run_id': run_id,
            'condition': CONDITION,
        }
        early_stopping(train_loss, model_weight)
        logger.end_epoch(ei)
        
        if early_stopping.early_stop or (ei == MAX_EPOCH - 1):
            best_weight = early_stopping.best_weights
            logger.write(f"Early stopping at epoch {best_weight['epoch']}", phase='summary')
            results = best_weight['eval_results']
            for ck in ['real']:
                logger.write(ck, phase='summary')
                result_c = results[ck]
                for rk in result_c.keys():
                    logger.write(f'{rk}: {result_c[rk]:.6f}', phase='summary')
            logger.write(f'Saved checkpoint: {ckpt_filname}', phase='summary')
            break

        scheduler.step()

    logger.finish_run()
