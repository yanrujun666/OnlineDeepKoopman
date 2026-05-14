import numpy as np
import scipy.io as scio


def load_vehicle_data(mat_path):
    """
    :param mat_path: MAT文件路径
    :return: 结构化字典数据，包含actuators、vehicle_dynamics和initial_conditions
    """
    try:
        # 加载原始数据（自动转换struct为字典）
        raw_data = scio.loadmat(mat_path, simplify_cells=True)
        
        # 数据架构容器
        parsed_data = {
            'actuators': {},
            'vehicle_dynamics': {},
            'initial_conditions': {}  # 新增初始条件容器
        }

        #-------------------------- 初始条件提取 --------------------------#
        # 检查是否存在initial_conditions字段
        if 'initial_conditions' in raw_data:
            # MATLAB结构体可能被转换为字典或numpy.void对象
            # 这里假设已经是简化后的字典结构
            initial_cond = raw_data['initial_conditions']
            
            # 提取目标变量（注意MATLAB变量名大小写敏感性）
            parsed_data['initial_conditions'] = {
                'wz0': initial_cond.get('wz0', None),  # 初始横摆角速度
                'vx0': initial_cond.get('vx0', None),  # 初始纵向速度
                'vy0': initial_cond.get('vy0', None),  # 初始横向速度
                'w0': initial_cond.get('w0', None)      # 初始车轮转速（可能需要确认命名）
            }
        else:
            print("警告: 未找到initial_conditions字段")

        if len(raw_data['Delta_act']) != len(raw_data['T_motor']):
            length = len(raw_data['T_motor'])
            raw_data['Delta_act'] = np.expand_dims(raw_data['Delta_act'], 0).repeat(length, 0)
        
        assert len(raw_data['Delta_act']) == len(raw_data['T_motor'])

        #-------------------------- Actuator Data --------------------------#
        parsed_data['actuators']['steer_angles'] = {
            k: raw_data['Delta_act'][:, i].astype(np.float32)
            for i, k in enumerate([
                'FL_steer', 'FR_steer',
                'ML_steer', 'MR_steer',
                'RL_steer', 'RR_steer'
            ])
        }

        parsed_data['actuators']['wheel_torques'] = {
            k: raw_data['T_motor'][:, i].astype(np.float32)
            for i, k in enumerate([
                'FL_trq', 'FR_trq',
                'ML_trq', 'MR_trq',
                'RL_trq', 'RR_trq'
            ])
        }

        #-------------------------- Vehicle Dynamics --------------------------#
        state_labels = [
            'Xpos', 'Ypos', 'Yaw',          # 0-2
            'Vx', 'Vy', 'YawRate',         # 3-5
            'Ax', 'Ay', 'YawAccel',         # 6-8
            *[f'WhlSpd_{pos}' for pos in ['FL', 'FR', 'ML', 'MR', 'RL', 'RR']],  # 9-14
            *[f'{force_direction}_{pos}' 
              for force_direction in ['Fx', 'Fy', 'Fz']
              for pos in ['FL', 'FR', 'ML', 'MR', 'RL', 'RR']
            ],  # 15-32
            'Alpha_FL', 'Alpha_FR', 'Alpha_ML', 'Alpha_MR', 'Alpha_RL', 'Alpha_RR',
            'Vy_processed'
        ]

        parsed_data['vehicle_dynamics'] = {
            k: raw_data['Vehicle_state_trucksim_39d'][:, i].astype(np.float32)
            for i, k in enumerate(state_labels[:39])
        }

        # parsed_data['vehicle_dynamics'] = {
        #     k: raw_data['Vehicle_state_trucksim_33d'][:, i].astype(np.float32)
        #     for i, k in enumerate(state_labels)
        # }


        return parsed_data

    except Exception as e:
        print(f"数据解析失败: {str(e)}, {str(mat_path)}")
        print(raw_data)
        exit(0)


def preprocess_trucksim_data_3dof(parsed_data):
    """将Trucksim的6轮数据转换为3轴平均数据"""
    # 获取原始数据引用
    actuators = parsed_data['actuators']
    
    # 转角和扭矩按轴分组索引
    axis_indices = [
        ['FL', 'FR'],   # 前轴 FL,FR
        ['ML', 'MR'],    # 中轴 ML,MR
        ['RL', 'RR']     # 后轴 RL,RR
    ]
    
    # 计算各轴平均转角和总扭矩
    delta_axis = []
    torque_axis = []
    for idx in axis_indices:
        # 转角取平均（注意单位是弧度）
        delta_avg = (actuators['steer_angles'][idx[0]+'_steer'] +actuators['steer_angles'][idx[1]+'_steer']) / 2 
        delta_axis.append(delta_avg)
        
        # 扭矩求和
        torque_sum = (actuators['wheel_torques'][idx[0]+'_trq'] + actuators['wheel_torques'][idx[1]+'_trq'])
        torque_axis.append(torque_sum)
    
    # 存储处理后的控制输入
    parsed_data['control_input'] = {
        'delta': np.column_stack(delta_axis),  # 各轴转角 [前,中,后]
        'torque': np.column_stack(torque_axis) # 各轴扭矩 [前,中,后]
    }
    
    return parsed_data


def preprocess_trucksim_data_7dof(parsed_data):
    # 获取原始数据引用
    actuators = parsed_data['actuators']
    
    # 转角和扭矩按轴分组索引
    axis_indices = [
        ['FL'], 
        ['FR'],
        ['ML'],
        ['MR'], 
        ['RL'], 
        ['RR'] 
    ]
    
    # 计算各轴平均转角和总扭矩
    delta_axis = []
    torque_axis = []
    for idx in axis_indices:
        # 转角
        delta_avg = actuators['steer_angles'][idx[0]+'_steer']
        delta_axis.append(delta_avg)
        
        # 扭矩
        torque_sum = actuators['wheel_torques'][idx[0]+'_trq']
        torque_axis.append(torque_sum)
    
    # 存储处理后的控制输入
    parsed_data['control_input'] = {
        'delta': np.column_stack(delta_axis),  # 各轴转角 [前,中,后]
        'torque': np.column_stack(torque_axis) # 各轴扭矩 [前,中,后]
    }
    
    return parsed_data