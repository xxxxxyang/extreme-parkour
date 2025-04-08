import torch
import torch.nn as nn
from .PAE import Model

class EnhancedTemporalBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg, time_range, window=1):
        super().__init__()
        activation = nn.ELU()
        # 基础特征提取
        self.base_backbone = base_backbone

        # Period latent (结合本体感受)
        if env_cfg == None:
            self.PAE = Model(32 + 53, 32, time_range, window)
        else:
            self.PAE = Model(32 + env_cfg.env.n_proprio, 32, time_range, window)


        # 时序聚合模块
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)

        # 运动连续性约束层
        self.motion_constraint = nn.Sequential(
            nn.Linear(512, 128),
            activation,
            nn.Linear(128, 32+2),
            nn.Tanh() # 限制在[-1,1]
        )

        self.hidden_states = None

    def forward(self, depth_seq, proprio_seq):
        """
        depth_seq: [B, T, H, W] 时序深度图
        proprio_seq: [B, T, D] 时序本体感受
        """
        B, T, H, W = depth_seq.shape

        # # 逐帧特征提取
        # spatial_latent = []
        # for t in range(T):
        #     latent = self.base_backbone(depth_seq[:, t])  # [B, 32]
        #     spatial_latent.append(latent)
        # spatial_latent = torch.stack(spatial_latent, dim=1)  # [B, T, 32]
        # 使用 3D CNN 提取特征
        spatial_latent = self.base_backbone(depth_seq)  # [B, T, 32]

        # # Period latent 提取
        # y, latent, period_signal, _= self.PAE(
        #     torch.cat([spatial_latent.permute(0, 2, 1), proprio_seq.permute(0, 2, 1)], dim=1)
        # )   # latent: [B, 32, T] y: [B, 32+n_proprio, T]

        # # rnn
        # depth_latent = latent[:, :, -1] # get the last time step [B, 32]
        # depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)

        # rnn
        depth_latent = spatial_latent[:, -1] # get the last time step [B, 32]
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)


        # output
        depth_latent = self.motion_constraint(depth_latent.squeeze(1))

        # return depth_latent, y, period_signal   # [B, 32+2], [B, (32+n_proprio) * T], [B, 32+n_proprio, T]
        return depth_latent
    
    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()


class DepthConv3DBackbone(nn.Module):
    def __init__(self, scandots_output_dim, output_activation=None, num_frames=1):
        super().__init__()

        # self.num_frames = num_frames    # number of frames to be stacked (equal to depth_buffer_len)
        activation = nn.ELU()

        # 3D CNN 模块
        self.image_compression = nn.Sequential(
            # 输入: [B, T, H, W] -> [B, 1, T, H, W]
            nn.Conv3d(in_channels=1, out_channels=scandots_output_dim//2, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            # 输出: [B, 16, T, H/2, W/2]
            nn.ELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # 输出: [B, 16, T, H/4, W/4]
            nn.Conv3d(in_channels=16, out_channels=scandots_output_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            # 输出: [B, 32, T, H/4, W/4]
            nn.ELU(),
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # 压缩 H 和 W 维度
            # 输出: [B, 32, T, 1, 1]
            nn.Flatten(start_dim=3),  # 展平 H 和 W 维度
            # 输出: [B, 32, T, 1]
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, depth_seq: torch.Tensor):
        """
        depth_seq: [B, T, H, W] 时序深度图
        输出: [B, T, 32]
        """
        # 添加通道维度 [B, T, H, W] -> [B, 1, T, H, W]
        depth_seq = depth_seq.unsqueeze(1)
        
        # 通过 3D CNN 提取特征
        compressed = self.image_compression(depth_seq)  # [B, 32, T]
        compressed = compressed.squeeze(-1)  # [B, 32, T]
        # 激活函数
        latent = self.output_activation(compressed.permute(0, 2, 1))  # [B, T, 32]
        
        return latent
    

# import torch
# import torch.nn as nn
# from .PAE import Model

# class EnhancedTemporalBackbone(nn.Module):
#     def __init__(self, base_backbone, env_cfg):
#         super().__init__()
#         activation = nn.ELU()
#         # 基础特征提取
#         self.base_backbone = base_backbone  
        
#         # 运动相位估计分支
#         self.phase_estimator = nn.Sequential(
#             nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
#             nn.ELU(),
#             nn.AdaptiveAvgPool1d(1),
#             nn.Linear(16, 1)
#         )
        
#         # 动态噪声门控
#         self.noise_gate = nn.Sequential(
#             nn.Linear(32 + 1, 16),  # 输入特征+相位
#             nn.Sigmoid()
#         )

#         # 视觉和本体感受融合模块
#         if env_cfg == None:
#             self.combination_mlp = nn.Sequential(
#                                     nn.Linear(32 + 53, 128),
#                                     activation,
#                                     nn.Linear(128, 32)
#                                 )
#         else:
#             self.combination_mlp = nn.Sequential(
#                                         nn.Linear(32 + env_cfg.env.n_proprio, 128),
#                                         activation,
#                                         nn.Linear(128, 32)
#                                     )

        
#         # 时序聚合模块
#         self.rnn = nn.GRU(
#             input_size=32, 
#             hidden_size=256,
#             bidirectional=True,
#             batch_first=True
#         )
        
#         # 运动连续性约束层
#         self.motion_constraint = nn.Sequential(
#             nn.Linear(256*2, 128),
#             nn.ELU(),
#             nn.Linear(128, 32+2)
#         )

#         self.hidden_states = None

#     def forward(self, depth_seq, proprio_seq):
#         """
#         depth_seq: [B, T, H, W] 时序深度图
#         proprio_seq: [B, T, D] 时序本体感受
#         """
#         T = depth_seq.shape[:1]
        
#         # 逐帧特征提取
#         spatial_latent = []
#         for t in range(T):
#             latent = self.base_backbone(depth_seq[t])  # [B, 32]
#             spatial_latent.append(latent)
#         spatial_latent = torch.stack(spatial_latent, dim=0)  # [B,T,32]
        
#         # 相位估计 (利用时序连续性)
#         phase_feats = self.phase_estimator(spatial_latent.permute(1,0))  # [B,1,T]
#         phase = torch.sin(phase_feats.squeeze())  # 约束在[-1,1] [T]
        
#         # 噪声门控 (相位敏感)
#         gated_feats = []
#         for t in range(T):
#             gate = self.noise_gate(
#                 torch.cat([spatial_latent[t], phase[t].unsqueeze(0)], dim=0)
#             )  # [16]
#             gated = spatial_latent[t] * gate.mean(dim=0, keepdim=True)  # [32]
#             gated_feats.append(gated)
#         gated_feats = torch.stack(gated_feats, dim=0)  # [T,32]

#         # 与本体感觉融合
#         combined_latent = self.combination_mlp(
#             torch.cat([gated_feats, proprio_seq], dim=1)
#         )  # [T,32]
        
#         # 双向时序融合
#         temporal_out, hidden_states = self.rnn(combined_latent.unsqueeze(0))  # [T,512]
        
#         # 运动连续性约束
#         output = self.motion_constraint(temporal_out[:,-1])  # 取最后时刻输出
        
#         return output, phase

