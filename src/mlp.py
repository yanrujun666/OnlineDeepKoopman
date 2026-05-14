import torch
import numpy as np
import torch.nn as nn


class MLPEncoderv1(nn.Module):
    
    def __init__(self, state_dim, hidden_dim, layer_depth):
        super(MLPEncoderv1, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.layer_depth = layer_depth

        layer_channels = [32, 64, self.hidden_dim-self.state_dim, self.hidden_dim-self.state_dim]

        self.in_fc = nn.Linear(self.state_dim, layer_channels[0], bias=True)
        self.act = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList()

        for i in range(len(layer_channels)-1):
            self.layers.append(nn.Linear(layer_channels[i], layer_channels[i+1], bias=True))
            if i != len(layer_channels)-2:
                self.layers.append(nn.ReLU(inplace=True))
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-1/np.sqrt(self.layer_depth), b=1/np.sqrt(self.layer_depth))
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.act(self.in_fc(x))
        for layer in self.layers:
            x = layer(x)
        return x
    

class MLPEncoderv0(nn.Module):
    
    def __init__(self, state_dim, hidden_dim, layer_depth):
        super(MLPEncoderv0, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64, bias=True),
            nn.GELU(),
            nn.Linear(64, 128, bias=True),
            nn.GELU(),
            nn.Linear(128, 256, bias=True),
            nn.GELU(),
            nn.Linear(256, 512, bias=True),
            nn.GELU(),
            nn.Linear(512, 512, bias=True),
            nn.GELU(),
            nn.Linear(512, 256, bias=True),
            nn.GELU(),
            nn.Linear(256, 64, bias=True),
            nn.GELU(),
            nn.Linear(64, hidden_dim-state_dim, bias=True),
        )

        self.layer_depth = len(self.layers)

        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-1/np.sqrt(self.layer_depth), b=1/np.sqrt(self.layer_depth))
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLPPredictorv0(nn.Module):
    
    def __init__(self, state_dim, control_dim):
        super(MLPPredictorv0, self).__init__()

        self.state_dim = state_dim

        self.layers = nn.Sequential(
            nn.Linear(state_dim+control_dim, 64, bias=True),
            nn.GELU(),
            nn.Linear(64, 128, bias=True),
            nn.GELU(),
            nn.Linear(128, 256, bias=True),
            nn.GELU(),
            nn.Linear(256, 512, bias=True),
            nn.GELU(),
            nn.Linear(512, 512, bias=True),
            nn.GELU(),
            nn.Linear(512, 256, bias=True),
            nn.GELU(),
            nn.Linear(256, 64, bias=True),
            nn.GELU(),
            nn.Linear(64, state_dim, bias=True),
        )

        self.layer_depth = len(self.layers)

        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.1/np.sqrt(self.layer_depth), b=0.1/np.sqrt(self.layer_depth))
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLPEncoderv0Norm(nn.Module):
    
    def __init__(self, state_dim, hidden_dim, layer_depth):
        super(MLPEncoderv0Norm, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.in_bn = nn.BatchNorm1d(state_dim)

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64, bias=True),
            nn.GELU(),
            nn.Linear(64, 128, bias=True),
            nn.GELU(),
            nn.Linear(128, 256, bias=True),
            nn.GELU(),
            nn.Linear(256, 512, bias=True),
            nn.GELU(),
            nn.Linear(512, 512, bias=True),
            nn.GELU(),
            nn.Linear(512, 256, bias=True),
            nn.GELU(),
            nn.Linear(256, 64, bias=True),
            nn.GELU(),
            nn.Linear(64, hidden_dim-state_dim, bias=True),
        )

        self.layer_depth = len(self.layers)

        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-1/np.sqrt(self.layer_depth), b=1/np.sqrt(self.layer_depth))
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        B, S, C = x.shape
        x = x.view(B*S, -1)
        x_norm = self.in_bn(x)
        x = x_norm.view(B, S, C)
        
        for layer in self.layers:
            x = layer(x)
        return x
    

# class MLPPredictorv0(nn.Module):
    
#     def __init__(self, state_dim, control_dim):
#         super(MLPPredictorv0, self).__init__()

#         self.state_dim = state_dim
#         self.control_dim = control_dim

#         self.layers = nn.Sequential(
#             nn.Linear(state_dim+control_dim, 64, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(64, 256, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 512, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 128, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, state_dim, bias=True),
#         )
        
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
    

class MLPEncoderv0Large(nn.Module):
    
    def __init__(self, state_dim, hidden_dim, layer_depth):
        super(MLPEncoderv0Large, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.layer_depth = layer_depth

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, hidden_dim-state_dim, bias=True),
        )
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class MLPDecoderv0(nn.Module):
    
    def __init__(self, state_dim, hidden_dim, layer_depth):
        super(MLPDecoderv0, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.layer_depth = layer_depth

        layer_channels = [self.hidden_dim, 64, 32, 16]

        self.layers = nn.ModuleList()

        for i in range(len(layer_channels)-1):
            self.layers.append(nn.Linear(layer_channels[i], layer_channels[i+1], bias=True))
            self.layers.append(nn.ReLU(inplace=True))

        self.out_fc = nn.Linear(16, self.state_dim, bias=True)
        self.act = nn.Sigmoid()

    #     # 初始化权重
    #     self.init_weights()
    
    # def init_weights(self):
    #     # Kaiming初始化（适用于ReLU激活函数）
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.uniform_(m.weight, a=-1/np.sqrt(self.layer_depth), b=1/np.sqrt(self.layer_depth))
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # x = self.act(self.out_fc(x))
        x = self.out_fc(x)
        return x
    

class MLPDecoderv1(nn.Module):
    
    def __init__(self, state_dim, hidden_dim, layer_depth):
        super(MLPDecoderv1, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.layer_depth = layer_depth

        layer_channels = [self.hidden_dim, 128, 64, 32]

        self.layers = nn.ModuleList()

        for i in range(len(layer_channels)-1):
            self.layers.append(nn.Linear(layer_channels[i], layer_channels[i+1], bias=True))
            self.layers.append(nn.ReLU(inplace=True))

        self.out_fc = nn.Linear(32, self.state_dim, bias=True)
        self.act = nn.Sigmoid()

    #     # 初始化权重
    #     self.init_weights()
    
    # def init_weights(self):
    #     # Kaiming初始化（适用于ReLU激活函数）
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.uniform_(m.weight, a=-1/np.sqrt(self.layer_depth), b=1/np.sqrt(self.layer_depth))
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.act(self.out_fc(x))
        # x = self.out_fc(x)
        return x


class MLPEncoder(nn.Module):
    
    def __init__(self, state_dim, hidden_dim, layer_depth):
        super(MLPEncoder, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.layer_depth = layer_depth

        self.in_fc = nn.Linear(self.state_dim, self.hidden_dim-self.state_dim, bias=True)
        self.act = nn.Tanh()

        self.layers = nn.ModuleList()

        for i in range(self.layer_depth - 1):
            self.layers.append(nn.Linear(self.hidden_dim-self.state_dim, self.hidden_dim-self.state_dim, bias=True))
            if i != self.layer_depth - 2:
                self.layers.append(nn.Tanh())
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        # Kaiming初始化（适用于ReLU激活函数）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-1/np.sqrt(self.layer_depth), b=1/np.sqrt(self.layer_depth))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.act(self.in_fc(x))
        for layer in self.layers:
            x = layer(x)
        return x


class MLPDecoder(nn.Module):
    
    def __init__(self, state_dim, hidden_dim, layer_depth):
        super(MLPDecoder, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.layer_depth = layer_depth

        self.layers = nn.ModuleList()

        for i in range(self.layer_depth - 1):
            self.layers.append(nn.Linear(self.hidden_dim-self.state_dim, self.hidden_dim-self.state_dim, bias=True))
            self.layers.append(nn.Tanh())

        self.out_fc = nn.Linear(self.hidden_dim-self.state_dim, self.state_dim, bias=False)
        # self.act = nn.Sigmoid()

        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        # Kaiming初始化（适用于ReLU激活函数）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out_fc(x)
        return x