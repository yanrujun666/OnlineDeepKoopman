import argparse
from collections import deque
import os

import torch
import torch.nn.functional as F
import tqdm
import numpy as np
from torch.utils.data import ConcatDataset

import finetune_koopman as ft
from src.loader import VehicleDynamicDataset5C


def _prepare_sequence_batch(sample, encoder):
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

    # 与原训练流程保持一致：位置转局部坐标、控制归一化
    states[:, :, :2] -= states[:, 0:1, :2].repeat(1, states.shape[1], 1)
    controls_norm = (controls - ft.CONTROL_MIN) / (ft.CONTROL_MAX - ft.CONTROL_MIN + 1e-6)

    with torch.no_grad():
        embeds = encoder(states)
    states_full = torch.cat([states, embeds], dim=-1)  # [B, T+1, H]

    x_t = states_full[:, :-1, :]   # [B, T, H]
    y_t = states_full[:, 1:, :]    # [B, T, H]
    u_t = controls_norm            # [B, T, C]
    return x_t, y_t, u_t


def build_online_loaders(window_size):
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
    data_dir = condition_dir_dict[ft.CONDITION]
    all_files = [os.path.join(data_dir, x) for x in sorted(os.listdir(data_dir))]

    # 在线流式训练：严格按时间顺序，逐步读取
    train_dst = VehicleDynamicDataset5C(
        mat_fpath=all_files[0],
        horizon=1,
        device=ft.DEVICE,
        show_log=False,
        skip=100
    )
    train_loader = torch.utils.data.DataLoader(
        train_dst,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # 验证集复用原有长时域评估习惯（horizon=window_size）
    valid_dsts = []
    for file in all_files[1:]:
        valid_dsts.append(VehicleDynamicDataset5C(
            mat_fpath=file,
            horizon=window_size,
            device=ft.DEVICE,
            show_log=False,
            skip=100
        ))
    valid_dst = ConcatDataset(valid_dsts)
    valid_loader = torch.utils.data.DataLoader(
        valid_dst,
        batch_size=ft.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    print('Online-train samples:', len(train_dst))
    print('Validation samples:', len(valid_dst))
    return train_loader, valid_loader


def _rolling_horizon_metrics(dkm, x_hist, y_hist, u_hist):
    # x_hist, y_hist, u_hist 均为长度 W 的 list，元素 shape: [H]/[H]/[C]
    x0 = x_hist[0].unsqueeze(0).unsqueeze(0)                     # [1,1,H]
    u_seq = torch.stack(list(u_hist), dim=0).unsqueeze(0)        # [1,W,C]
    gt_seq = torch.stack(list(y_hist), dim=0)                    # [W,H]

    with torch.no_grad():
        pred_seq = dkm(x0, u_seq, iters=u_seq.shape[1])[0]       # [W,H]

    err_xy = torch.norm(pred_seq[:, :2] - gt_seq[:, :2], p=2, dim=-1)
    ade_xy = torch.mean(err_xy).item()
    fde_xy = err_xy[-1].item()
    mse_state = F.mse_loss(pred_seq[:, :6], gt_seq[:, :6]).item()
    return ade_xy, fde_xy, mse_state


def _optimize_on_window(
    dkm,
    optimizer,
    A_ref,
    B_ref,
    x_hist,
    y_hist,
    u_hist,
    update_epochs,
    reg_lambda
):
    x_win = torch.stack(list(x_hist), dim=0)  # [W,H]
    y_win = torch.stack(list(y_hist), dim=0)  # [W,H]
    u_win = torch.stack(list(u_hist), dim=0)  # [W,C]

    for _ in range(update_epochs):
        pred_next = x_win @ dkm.A + u_win @ dkm.B
        loss_state = F.mse_loss(pred_next[:, :6], y_win[:, :6])
        loss_reg = F.mse_loss(dkm.A, A_ref) + F.mse_loss(dkm.B, B_ref)
        loss = loss_state + reg_lambda * loss_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dkm.parameters(), max_norm=5.0)
        optimizer.step()

    return loss_state.item(), loss_reg.item()


def online_optimize_koopman(
    encoder,
    dkm,
    train_loader,
    window_size=100,
    trigger_threshold=0.2,
    update_epochs=50,
    lr=1e-4,
    reg_lambda=1e-3,
    max_steps=-1
):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    dkm.train()
    optimizer = torch.optim.Adam(dkm.parameters(), lr=lr)

    # 以初始参数作为正则中心，避免在线更新漂移过大
    A_ref = dkm.A.detach().clone()
    B_ref = dkm.B.detach().clone()

    x_window = deque(maxlen=window_size)  # x_t
    y_window = deque(maxlen=window_size)  # x_{t+1}
    u_window = deque(maxlen=window_size)  # u_t

    metrics = []
    trigger_count = 0
    qbar = tqdm.tqdm(train_loader, desc='Online Koopman event-trigger')
    for step_idx, sample in enumerate(qbar):
        if max_steps > 0 and step_idx >= max_steps:
            break

        x_seq, y_seq, u_seq = _prepare_sequence_batch(sample, encoder)  # [1,1,H], [1,1,H], [1,1,C]
        x_window.append(x_seq[0, 0, :].detach())
        y_window.append(y_seq[0, 0, :].detach())
        u_window.append(u_seq[0, 0, :].detach())

        if len(x_window) < window_size:
            continue

        ade_xy, fde_xy, mse_state = _rolling_horizon_metrics(dkm, x_window, y_window, u_window)

        triggered = False
        update_state_loss = 0.0
        update_reg_loss = 0.0
        if ade_xy > trigger_threshold:
            triggered = True
            trigger_count += 1
            update_state_loss, update_reg_loss = _optimize_on_window(
                dkm=dkm,
                optimizer=optimizer,
                A_ref=A_ref,
                B_ref=B_ref,
                x_hist=x_window,
                y_hist=y_window,
                u_hist=u_window,
                update_epochs=update_epochs,
                reg_lambda=reg_lambda
            )
            # 更新后再测一次当前窗口的100步误差
            ade_xy, fde_xy, mse_state = _rolling_horizon_metrics(dkm, x_window, y_window, u_window)

        metrics.append({
            'step': step_idx,
            'ade_xy_100': ade_xy,
            'fde_xy_100': fde_xy,
            'mse_state_100': mse_state,
            'triggered': int(triggered),
            'update_state_loss': update_state_loss,
            'update_reg_loss': update_reg_loss,
        })
        qbar.set_postfix({
            'step': step_idx,
            'ade100': f'{ade_xy:.4f}',
            'trig': int(triggered),
            'trig_cnt': trigger_count
        })

    return dkm, metrics


def main():
    parser = argparse.ArgumentParser(description='Online optimization of Koopman parameters with one-step loss and sliding window.')
    parser.add_argument('--condition', type=str, default='AB', help='Condition key, e.g. AB/AA/CC.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Torch device, e.g. cuda:0 or cpu.')
    parser.add_argument('--window-size', type=int, default=100, help='Sliding window length in time ticks.')
    parser.add_argument('--trigger-threshold', type=float, default=0.2, help='Trigger update when 100-step ADE is above this threshold.')
    parser.add_argument('--update-epochs', type=int, default=50, help='Gradient epochs on current window after trigger.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for online updates.')
    parser.add_argument('--reg-lambda', type=float, default=1e-3, help='Regularization weight around initial Koopman params.')
    parser.add_argument('--max-steps', type=int, default=-1, help='Optional: limit number of online steps.')
    parser.add_argument('--metric-log-path', type=str, default='log/online_koopman_metrics.csv', help='CSV path to save per-step 100-step prediction metrics.')
    parser.add_argument('--save-path', type=str, default='', help='Optional checkpoint save path.')
    args = parser.parse_args()

    ft.DEVICE = torch.device(args.device)
    ft.CONDITION = args.condition
    ft.CONTROL_MIN = ft.CONTROL_MIN.to(ft.DEVICE)
    ft.CONTROL_MAX = ft.CONTROL_MAX.to(ft.DEVICE)
    ft.WEIGHTS = ft.WEIGHTS.to(ft.DEVICE)

    encoder, dkm = ft.build_model()
    encoder = encoder.to(ft.DEVICE)
    dkm = dkm.to(ft.DEVICE)

    train_loader, val_loader = build_online_loaders(args.window_size)

    print('\nBefore online optimization:')
    ft.eval(encoder, dkm, val_loader, 0, ft.CONDITION)

    dkm, metric_logs = online_optimize_koopman(
        encoder=encoder,
        dkm=dkm,
        train_loader=train_loader,
        window_size=args.window_size,
        trigger_threshold=args.trigger_threshold,
        update_epochs=args.update_epochs,
        lr=args.lr,
        reg_lambda=args.reg_lambda,
        max_steps=args.max_steps
    )

    if args.metric_log_path:
        log_dir = os.path.dirname(args.metric_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(args.metric_log_path, 'w') as f:
            f.write('step,ade_xy_100,fde_xy_100,mse_state_100,triggered,update_state_loss,update_reg_loss\n')
            for m in metric_logs:
                f.write(
                    f"{m['step']},{m['ade_xy_100']:.8f},{m['fde_xy_100']:.8f},{m['mse_state_100']:.8f},"
                    f"{m['triggered']},{m['update_state_loss']:.8f},{m['update_reg_loss']:.8f}\n"
                )
        print(f'Saved online metrics to: {args.metric_log_path}')

    print('\nAfter online optimization:')
    dkm_avg_eval_error, dkm_final_eval_error, \
        vx_eval_error, vy_eval_error, wz_eval_error, \
        yaw_avg_eval_error, yaw_final_eval_error = ft.eval(encoder, dkm, val_loader, 0, ft.CONDITION)

    if args.save_path:
        weight = {
            'encoder': encoder.state_dict().copy(),
            'koopman': dkm.state_dict().copy(),
            'eval_results': {
                'dkm_mde': dkm_avg_eval_error.item(),
                'dkm_fde': dkm_final_eval_error.item(),
                'dkm_mae': yaw_avg_eval_error.item(),
                'dkm_fae': yaw_final_eval_error.item(),
                'vx_error': vx_eval_error.item(),
                'vy_error': vy_eval_error.item(),
                'wz_error': wz_eval_error.item(),
            }
        }
        torch.save(weight, args.save_path)
        print(f'Saved checkpoint to: {args.save_path}')


if __name__ == '__main__':
    main()
