import torch
import torch.nn as nn

from src.transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder


class BatchFourierPositionalEncoding(nn.Module):
    def __init__(self, L=10, input_dim=2):
        """
        :param L: 频率级数，控制输出维度 C = 2 * input_dim * L
        :param input_dim: 输入坐标维度（默认为2，即x和y）
        """
        super().__init__()
        self.L = L
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim * L
        
        # 预计算频率 2^k (k=0, 1, ..., L-1)
        self.register_buffer('freqs', 2.0 ** torch.arange(0, L, dtype=torch.float32))

    def forward(self, coords):
        """
        :param coords: 输入坐标，形状为 (Batch_size, ins_num, pts_num, input_dim)
        :return: 编码后的特征，形状为 (Batch_size, ins_num, pts_num, output_dim)
        """       
        # 计算所有频率的正弦和余弦
        # angle: (Batch_size, ins_num, pts_num, input_dim, L)
        coords_normalized = coords
        angle = coords_normalized.unsqueeze(-1) * self.freqs.view(1, 1, 1, 1, -1)
        sin_enc = torch.sin(2 * torch.pi * angle)  # (..., input_dim, L)
        cos_enc = torch.cos(2 * torch.pi * angle)  # (..., input_dim, L)
        
        # 将正弦和余弦交错排列
        pe = torch.stack([sin_enc, cos_enc], dim=-1)  # (..., input_dim, L, 2)
        pe = pe.flatten(start_dim=-3, end_dim=-1)     # (..., input_dim * L * 2)
        pe = torch.cat([coords, pe], dim=-1)
        
        return pe
    

class CustomEncoderUnscaledv2WithoutNorm(nn.Module):

    def __init__(self, state_dim, hidden_dim, layer_depth):
        super().__init__()

        self.in_embed = BatchFourierPositionalEncoding(L=8, input_dim=1)

        self.channel_fc = nn.Sequential(
            nn.Linear(16+1, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU()
        )

        encoder_depth = 2
        decoder_depth = 2

        self.state_pe = nn.Embedding(state_dim, 128)
        encoder_layer = TransformerEncoderLayer(128, 8, 128, 0.0, 'gelu', False)
        self.state_encoder = TransformerEncoder(encoder_layer, encoder_depth)

        decoder_layer = TransformerDecoderLayer(128, 8, 128, 0.0, "gelu", False)
        decoder_norm = nn.LayerNorm(128)
        self.state_decoder = TransformerDecoder(decoder_layer, decoder_depth, decoder_norm, return_intermediate=False)

        self.out_fc =nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim-state_dim),
        )

        self.skip_fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim-state_dim),
        )
        
    def forward(self, x):

        B, S, C = x.shape        
        x_embed = self.in_embed(x.unsqueeze(-1)) # B S state_dim C

        x_embed = self.channel_fc(x_embed)

        B, S, N, C = x_embed.shape
        pos = self.state_pe.weight.repeat(B*S, 1, 1)
        x_embed_flat = x_embed.view(B*S, N, C)
        x_embed_flat = x_embed_flat.permute(1, 0, 2)
        pos_flat = pos.permute(1, 0, 2)
        x_embed_fused = self.state_encoder(x_embed_flat, pos=pos_flat)

        fused_embed = torch.zeros((1, 1, C), device=x_embed.device).repeat(B*S, 1, 1).permute(1, 0, 2)
        fused_embed = self.state_decoder(fused_embed, x_embed_fused, pos=pos_flat)[0, 0]
        fused_embed = fused_embed.view(B, S, C)

        skip_embed = self.skip_fc(x)

        return self.out_fc(fused_embed) + skip_embed
