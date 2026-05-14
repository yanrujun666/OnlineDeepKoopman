import numpy as np
import torch
import torch.nn as nn


class SevenDOFThreeAxleBicycleModelLinear(nn.Module):
    def __init__(self, batch_size=1, device='cpu', dt=0.01):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.dt = dt
        
        self.sprung_mass = torch.tensor(4455.0)
        self.load_mass = torch.tensor(5000.0)
        self.unsprung_mass = torch.tensor(350.0 * 3)
        
        # 固定参数
        self.g = torch.tensor(9.81, device=self.device)
        self.sprung_c = torch.tensor(1.250, device=self.device)
        self.load_c = torch.tensor(4.0, device=self.device)
        self.axis1 = torch.tensor(0.0, device=self.device)
        self.axis2 = torch.tensor(3.250, device=self.device)
        self.axis3 = torch.tensor(6.500, device=self.device)

        # 转动惯量
        self.all_Iz = torch.tensor(34823.2+5416.67, device=self.device)

        # 新增：高度参数（单位：m）
        self.h_sprung = torch.tensor(1.175, device=self.device)  # 簧上质量质心高度
        self.h_unsprung = torch.tensor(0.51/2, device=self.device)  # 簧下质量质心高度
        self.h_load = torch.tensor(2.050, device=self.device)  # 货物质心高度
        self.track_width = torch.tensor(2.030, device=self.device)  # 横向轴距
        
        # 轮胎参数
        self.mu = torch.tensor(1.0, device=self.device)
        self.R = torch.tensor(0.51, device=self.device)
        self.BCD_alpha = torch.tensor(-138541.1972258911, device=self.device)
        self.roll_resistant = torch.tensor(0.0041, device=self.device)

        # 空气阻力
        self.area = torch.tensor(10, device=self.device)
        self.air_density = torch.tensor(1.2258, device=self.device)
        self.air_resistant = torch.tensor(0.56, device=self.device)

        # 初始化状态变量 (batch_size, 6) - 7DOF模型
        self.register_buffer('state', torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device).repeat(batch_size, 1))
        
        # 初始化缓存变量，避免运行时检查
        self.ax_prev = torch.zeros(batch_size, device=self.device)
        self.ay_prev = torch.zeros(batch_size, device=self.device)
        # self.Fz_dynamic = None
        
    def forward(self, state, delta, torques):
        """计算状态导数 dy/dt = f(y, t) - 支持批量计算"""
        # state: (batch_size, 6) - 7DOF模型
        # delta: (batch_size, 6) - 6个车轮的转向角
        # torques: (batch_size, 6) - 6个车轮的驱动力矩

        vx, vy, omega, X, Y, psi = state.unbind(dim=1)

        # 从params中提取参数，每个参数形状为 (batch_size,)
        BCD_alpha = self.BCD_alpha.repeat(self.batch_size)
        all_Iz = self.all_Iz.repeat(self.batch_size)
        roll_resistant = self.roll_resistant.repeat(self.batch_size)
        air_resistant = self.air_resistant.repeat(self.batch_size)

        # 中间变量缓存 - 扩展到6个车轮
        alpha = torch.zeros(self.batch_size, 6, device=self.device)  # 6个车轮的侧偏角
        F_longi = torch.zeros(self.batch_size, 6, device=self.device)  # 6个车轮的纵向力
        F_lateral = torch.zeros(self.batch_size, 6, device=self.device)  # 6个车轮的侧偏力
        
        #####################################################################
        # 1. 预计算
        """预计算不依赖于状态的参数"""
        m = self.sprung_mass + self.unsprung_mass + self.load_mass
        
        # 计算总质心位置
        total_c = (self.sprung_mass * self.sprung_c + 
                self.unsprung_mass / 3 * (self.axis1 + self.axis2 + self.axis3) +
                self.load_mass * self.load_c) / m
        
        # 计算转动惯量 - 使用批次参数
        Iz = (all_Iz + self.sprung_mass * (self.sprung_c - total_c)**2 +
                self.load_mass * (self.load_c - total_c)**2)
        
        # 计算各轴到质心的距离
        lf = total_c - self.axis1
        lm = self.axis2 - total_c
        lr = self.axis3 - total_c
        
        # 计算总质心高度
        h_total = (self.sprung_mass * self.h_sprung + 
                    self.unsprung_mass * self.h_unsprung +
                    self.load_mass * self.h_load) / m
        
        # 更新车轮位置（基于实际质心位置）
        wheel_positions = torch.tensor([
            # [x, y] 相对于质心的位置
            [lf, self.track_width/2],   # 前左轮
            [lf, -self.track_width/2],  # 前右轮
            [-lm, self.track_width/2],    # 中左轮
            [-lm, -self.track_width/2],   # 中右轮
            [-lr, self.track_width/2],    # 后左轮
            [-lr, -self.track_width/2],   # 后右轮
        ], device=self.device)
        
        # 静态垂向载荷分布（6个车轮，每轴左右均分）
        k = 0.5
        r23 = k / (1-k)
        Fz_axis_static = torch.tensor([
            m * self.g * (lm + r23 * lr) / (lf * (1+r23) + lm + r23*lr),  # 前轴
            m * self.g * lf / (lf * (1+r23) + lm + r23*lr),  # 中轴
            m * self.g * lf * r23 / (lf * (1+r23) + lm + r23*lr)   # 后轴
        ])
        
        # 每个车轮的静态载荷（左右均分）
        Fz_static = torch.stack([
            Fz_axis_static[0] / 2,  # 前左轮
            Fz_axis_static[0] / 2,  # 前右轮
            Fz_axis_static[1] / 2,  # 中左轮
            Fz_axis_static[1] / 2,  # 中右轮
            Fz_axis_static[2] / 2,  # 后左轮
            Fz_axis_static[2] / 2,  # 后右轮
        ]).unsqueeze(0).repeat(self.batch_size, 1).to(self.device)
        #####################################################################

        #####################################################################
        # 3. 计算每个车轮处的速度
        # vx, vy: (batch_size,)
        # omega: (batch_size,)
        # 返回: (batch_size, 6, 2) - 6个车轮的[x, y]速度
        
        # 使用向量化操作替代循环
        wheel_positions_batch = wheel_positions.unsqueeze(0).expand(self.batch_size, -1, -1)  # (batch_size, 6, 2)

        # 计算车轮速度
        wheel_velocities_x = vx.unsqueeze(1) - omega.unsqueeze(1) * wheel_positions_batch[:, :, 1]
        wheel_velocities_y = vy.unsqueeze(1) + omega.unsqueeze(1) * wheel_positions_batch[:, :, 0]
        wheel_velocities = torch.stack([wheel_velocities_x, wheel_velocities_y], dim=2)
        #####################################################################
        
        #####################################################################
        # 4. 计算每个车轮的侧偏角
        # wheel_velocities: (batch_size, 6, 2)
        # delta: (batch_size, 6) - 6个车轮的转向角
        # 返回: (batch_size, 6) - 6个车轮的侧偏角
        vx_wheel = wheel_velocities[:, :, 0]
        vy_wheel = wheel_velocities[:, :, 1]

        alpha = - delta + torch.atan2(vy_wheel, vx_wheel)
        # alpha = - delta + vy_wheel / vx_wheel
        
        # 限制侧偏角范围
        alpha = torch.clamp(alpha, -torch.pi/2, torch.pi/2)
        #####################################################################

        #####################################################################
        # 5. 计算加速度（用于载荷转移）
        # 使用上一时刻的加速度估计值，或者初始化为0
        if not hasattr(self, 'ax_prev'):
            ax_prev = torch.zeros(self.batch_size, device=self.device)
        else:
            ax_prev = self.ax_prev.clone()
        if not hasattr(self, 'ay_prev'):
            ay_prev = torch.zeros(self.batch_size, device=self.device)
        else:
            ay_prev = self.ay_prev.clone()
        #####################################################################

        #####################################################################
        # 6. 更新动态垂向载荷
        # 纵向载荷转移（三轴）
        # 基于动态载荷转移系数 k=0.5 for tandem axles 2 & 3
        k = 0.5
        r = k / (1 - k)  

        # 计算轴距 L1 和 L2 from lf, lm, lr
        L1 = lf + lm  # 前轴到中轴的距离
        L2 = lr - lm  # 中轴到后轴的距离
        
        # 总纵向载荷转移力矩
        Fz_longi_transfer_total = m * ax_prev.unsqueeze(1) * h_total
        
        # 计算分母 for δF2
        denominator = L1 + r * (L1 + L2)
        F2 = Fz_longi_transfer_total / denominator

        # 计算 δF3 和 δF1
        F3 = r * F2
        F1 = - (F2 + F3)
        
        # 横向载荷转移（左右侧）
        # 横向载荷转移量 = 质量 × 横向加速度 × 质心高度 / (横向轴距/2)
        Fz_lateral_transfer = m * ay_prev.unsqueeze(1) * h_total / (self.track_width / 2)

        # 从静态载荷开始，重新计算动态载荷
        Fz_dynamic = Fz_static.clone()
        
        # 前轴
        Fz_dynamic[:, 0] += F1.squeeze(1) / 2  # 前左轮
        Fz_dynamic[:, 1] += F1.squeeze(1) / 2  # 前右轮
        # 中轴
        Fz_dynamic[:, 2] += F2.squeeze(1) / 2  # 中左轮
        Fz_dynamic[:, 3] += F2.squeeze(1) / 2  # 中右轮
        # 后轴
        Fz_dynamic[:, 4] += F3.squeeze(1) / 2  # 后左轮
        Fz_dynamic[:, 5] += F3.squeeze(1) / 2  # 后右轮
        
        # 横向载荷转移
        lateral_transfer_per_wheel = Fz_lateral_transfer / 6
        
        # 应用横向载荷转移
        Fz_dynamic[:, [0, 2, 4]] -= lateral_transfer_per_wheel
        Fz_dynamic[:, [1, 3, 5]] += lateral_transfer_per_wheel
        
        # 确保垂向载荷为正值
        Fz_dynamic = torch.clamp(Fz_dynamic, min=0.1)
        
        # 验证物理合理性
        total_load = torch.sum(Fz_dynamic, dim=1)
        expected_load = m * self.g
        load_error = torch.abs(total_load - expected_load)
        
        # 如果载荷分布异常，重置为静态载荷
        abnormal_mask = load_error > expected_load * 0.1
        if torch.any(abnormal_mask):
            print("[警告]检测到垂向载荷过小")
            exit(0)
        #####################################################################

        #####################################################################
        # 8. 轮胎力计算(简化为纯线性模型，只使用静态侧偏刚度)
        F_alpha = BCD_alpha.unsqueeze(1) * alpha
        #####################################################################

        #####################################################################
        # 9. 力分解和累计
        # 使用批次参数 roll_resistant，形状为 (batch_size,)
        Fx_drive = torques / self.R - roll_resistant.unsqueeze(1) * Fz_dynamic
        
        # 使用向量化操作替代循环
        cos_delta = torch.cos(delta)
        sin_delta = torch.sin(delta)
        
        # 驱动分量分解
        fx_d = Fx_drive * cos_delta
        fy_d = Fx_drive * sin_delta
        
        # 侧偏力分解
        fx_a = -F_alpha * sin_delta
        fy_a = F_alpha * cos_delta
        
        # 合力计算
        fx_total = fx_d + fx_a
        fy_total = fy_d + fy_a
        
        # 累计总力和力矩
        Fx_total = torch.sum(fx_total, dim=1)
        Fy_total = torch.sum(fy_total, dim=1)
        
        # 计算对质心的力矩（考虑车轮位置）
        # 使用向量化操作
        Mz_total = torch.sum(
            wheel_positions_batch[:, :, 0] * fy_total - 
            wheel_positions_batch[:, :, 1] * fx_total, 
            dim=1
        )
        
        # 存储各车轮力
        F_longi = Fx_drive
        F_lateral = F_alpha
        
        # 考虑空气阻力作用于整车纵向运动
        # 使用批次参数 air_resistant，形状为 (batch_size,)
        Fx_total -= air_resistant * self.area * self.air_density * (vx ** 2) / 2
        #####################################################################

        #####################################################################
        # 10. 动力学方程
        dvx = Fx_total / m + vy * omega
        dvy = Fy_total / m - vx * omega
        domega = Mz_total / Iz
        #####################################################################

        #####################################################################
        # 11. 加速度估计
        ax = dvx - vy * omega  # 纵向加速度
        ay = dvy + vx * omega  # 横向加速度
        
        self.ax_prev = ax
        self.ay_prev = ay
        #####################################################################

        #####################################################################
        # 12. 运动学方程
        sin_psi = torch.sin(psi)
        cos_psi = torch.cos(psi)
        dX = vx * cos_psi - vy * sin_psi
        dY = vx * sin_psi + vy * cos_psi
        dpsi = omega
        #####################################################################

        # 计算 lateral_center
        lateral_center = vy - (self.axis3 / 2 - total_c) * omega
        
        # 返回所有需要的信息，避免缓存
        derivatives = torch.stack([dvx, dvy, domega, dX, dY, dpsi], dim=1)

        return derivatives, {
            'alpha': alpha,
            'F_longi': F_longi,
            'F_lateral': F_lateral,
            'Fz': Fz_dynamic,
            'Fz_static': Fz_static,
            'Fz_dynamic': Fz_dynamic,
            'lateral_center': lateral_center,
            'total_c': total_c,
            'Iz': Iz,
            'ax': dvx,
            'ay': dvy,
        }

    def update(self, dt, delta, torques, method='euler'):
        """更新车辆状态 - 支持批量计算"""
        delta = torch.as_tensor(delta, device=self.device)
        torques = torch.as_tensor(torques, device=self.device)
        
        # 确保输入有正确的形状
        if delta.dim() == 1:
            delta = delta.unsqueeze(0).repeat(self.batch_size, 1)
        elif delta.dim() == 2 and delta.shape[0] != self.batch_size:
            delta = delta.repeat(self.batch_size, 1)
        
        if torques.dim() == 1:
            torques = torques.unsqueeze(0).repeat(self.batch_size, 1)
        elif torques.dim() == 2 and torques.shape[0] != self.batch_size:
            torques = torques.repeat(self.batch_size, 1)
        
        if method == 'euler':
            # 欧拉方法
            derivatives, aux_info = self.forward(self.state, delta, torques)
            self.state = self.state + dt * derivatives
            # 存储辅助信息用于 get_state
            self._last_aux_info = aux_info
        elif method == 'rk4':
            # 四阶龙格-库塔方法
            k1, _ = self.forward(self.state, delta, torques)
            k2, _ = self.forward(self.state + dt/2 * k1, delta, torques)
            k3, _ = self.forward(self.state + dt/2 * k2, delta, torques)
            k4, aux_info = self.forward(self.state + dt * k3, delta, torques)
            
            self.state = self.state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            # 使用最后一次的辅助信息
            self._last_aux_info = aux_info
        
        return self.get_state()

    def get_state(self):
        """获取当前状态和中间变量 - 支持批量计算"""
        vx, vy, omega, X, Y, psi = self.state.unbind(dim=1)
        
        # 使用最后一次计算的辅助信息
        aux_info = getattr(self, '_last_aux_info', {})
        
        return {
            'position': {'x': X, 'y': Y},
            'attitude': {'yaw': psi},
            'ax': aux_info.get('ax', torch.zeros_like(vx)),
            'ay': aux_info.get('ay', torch.zeros_like(vy)),
            'velocity': {
                'longitudinal': vx,
                'lateral': vy,
                'yaw_rate': omega,
                'lateral_center': aux_info.get('lateral_center', torch.zeros_like(vy))
            },
            'tire_forces': {
                'front_left': {
                    'Fx': aux_info.get('F_longi', torch.zeros_like(vx))[:, 0],
                    'Fy': aux_info.get('F_lateral', torch.zeros_like(vx))[:, 0],
                    'Fz': aux_info.get('Fz', torch.zeros_like(vx))[:, 0]
                },
                'front_right': {
                    'Fx': aux_info.get('F_longi', torch.zeros_like(vx))[:, 1],
                    'Fy': aux_info.get('F_lateral', torch.zeros_like(vx))[:, 1],
                    'Fz': aux_info.get('Fz', torch.zeros_like(vx))[:, 1]
                },
                'mid_left': {
                    'Fx': aux_info.get('F_longi', torch.zeros_like(vx))[:, 2],
                    'Fy': aux_info.get('F_lateral', torch.zeros_like(vx))[:, 2],
                    'Fz': aux_info.get('Fz', torch.zeros_like(vx))[:, 2]
                },
                'mid_right': {
                    'Fx': aux_info.get('F_longi', torch.zeros_like(vx))[:, 3],
                    'Fy': aux_info.get('F_lateral', torch.zeros_like(vx))[:, 3],
                    'Fz': aux_info.get('Fz', torch.zeros_like(vx))[:, 3]
                },
                'rear_left': {
                    'Fx': aux_info.get('F_longi', torch.zeros_like(vx))[:, 4],
                    'Fy': aux_info.get('F_lateral', torch.zeros_like(vx))[:, 4],
                    'Fz': aux_info.get('Fz', torch.zeros_like(vx))[:, 4]
                },
                'rear_right': {
                    'Fx': aux_info.get('F_longi', torch.zeros_like(vx))[:, 5],
                    'Fy': aux_info.get('F_lateral', torch.zeros_like(vx))[:, 5],
                    'Fz': aux_info.get('Fz', torch.zeros_like(vx))[:, 5]
                }
            },
            'tire_alpha': {
                'alpha_fl': aux_info.get('alpha', torch.zeros_like(vx))[:, 0],
                'alpha_fr': aux_info.get('alpha', torch.zeros_like(vx))[:, 1],
                'alpha_ml': aux_info.get('alpha', torch.zeros_like(vx))[:, 2],
                'alpha_mr': aux_info.get('alpha', torch.zeros_like(vx))[:, 3],
                'alpha_rl': aux_info.get('alpha', torch.zeros_like(vx))[:, 4],
                'alpha_rr': aux_info.get('alpha', torch.zeros_like(vx))[:, 5]
            },
            'load_transfer': {
                'Fz_static': aux_info.get('Fz_static', torch.zeros_like(vx)),
                'Fz_dynamic': aux_info.get('Fz_dynamic', torch.zeros_like(vx)),
            },
        }
        

    def reset_state(self, state=None, ax_prev=None, ay_prev=None):
        """重置车辆状态 - 支持批量计算"""
        if state is None:
            default_state = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=self.device)
            self.state = default_state.repeat(self.batch_size, 1)
        else:
            self.state = torch.as_tensor(state, device=self.device)

            if self.state.dim() == 1:
                self.state = self.state.unsqueeze(0).repeat(self.batch_size, 1)
            elif self.state.dim() == 2 and self.state.shape[0] != self.batch_size:
                self.state = self.state.repeat(self.batch_size, 1)

            # 侧向速度处理，trucksim是几何中心
            m = self.sprung_mass + self.unsprung_mass + self.load_mass
        
            total_c = (self.sprung_mass * self.sprung_c + 
                    self.unsprung_mass / 3 * (self.axis1 + self.axis2 + self.axis3) +
                    self.load_mass * self.load_c) / m

            lateral_center = self.state[:, 1]
            omega = self.state[:, 2]
            vy = lateral_center + (self.axis3 / 2 - total_c) * omega
            self.state[:, 1] = vy

        if ax_prev is None:
            self.ax_prev = torch.zeros(self.batch_size, device=self.device)
        else:
            self.ax_prev = torch.as_tensor(ax_prev, device=self.device)
            if self.ax_prev.shape[0] != self.batch_size:
                self.ax_prev = self.ax_prev.repeat(self.batch_size)
            assert self.ax_prev.shape[0] == self.batch_size

        if ay_prev is None:
            self.ay_prev = torch.zeros(self.batch_size, device=self.device)
        else:
            self.ay_prev = torch.as_tensor(ay_prev, device=self.device)
            if self.ay_prev.shape[0] != self.batch_size:
                self.ay_prev = self.ay_prev.repeat(self.batch_size)
            assert self.ay_prev.shape[0] == self.batch_size
