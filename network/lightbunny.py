import torch
import torch.nn as nn
from torch import Tensor


class LightBunnyConfig:
    raw_channel_in: int
    seq_len: int
    pred_len: int
    chunk_size: int
    enc_in: int
    d_model: int
    seg_len: int


# class Config:
#     raw_channel_in = n_electrodes
#     seq_len = seq_len
#     pred_len = seq_len
#     chunk_size = 25
#     enc_in = 994
#     d_model = 256
#     seg_len = 25


class ResidualLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.block1_linear1 = nn.Linear(in_features, in_features)
        self.block1_gelu1 = nn.GELU()
        self.block1_linear2 = nn.Linear(in_features, in_features)
        self.block1_gelu2 = nn.GELU()

        self.block2_linear1 = nn.Linear(in_features, in_features)
        self.block2_gelu1 = nn.GELU()
        self.block2_linear2 = nn.Linear(in_features, out_features)
        self.block2_downsample = nn.Linear(in_features, out_features)
        self.block2_gelu2 = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        iden = x
        out = self.block1_linear1(x)
        out = self.block1_gelu1(out)
        out = self.block1_linear2(out)
        out += iden
        out = self.block1_gelu2(out)

        iden = out
        out = self.block2_linear1(out)
        out = self.block2_gelu1(out)
        out = self.block2_linear2(out)
        out += self.block2_downsample(iden)
        out = self.block1_gelu2(out)

        return out


class IEBlock(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_node):
        super(IEBlock, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_node = num_node

        self.spatial_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim // 4),
        )

        self.channel_proj = nn.Linear(self.num_node, self.num_node)
        torch.nn.init.eye_(self.channel_proj.weight)

        self.output_proj = nn.Linear(self.hid_dim // 4, self.output_dim)

    def forward(self, x):
        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))
        x = self.output_proj(x.permute(0, 2, 1))

        x = x.permute(0, 2, 1)

        return x


class LightTS(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.chunk_size = configs.chunk_size
        assert self.seq_len % self.chunk_size == 0
        self.num_chunks = self.seq_len // self.chunk_size

        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.dropout = configs.dropout

        self.layer_1 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4,
            num_node=self.num_chunks,
        )

        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)

        self.layer_2 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4,
            num_node=self.num_chunks,
        )

        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)

        self.layer_3 = IEBlock(
            input_dim=self.d_model // 2,
            hid_dim=self.d_model // 2,
            output_dim=self.pred_len,
            num_node=self.enc_in,
        )

        self.ar = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        B, T, N = x.size()

        highway = self.ar(x.permute(0, 2, 1))
        highway = highway.permute(0, 2, 1)

        # continuous sampling
        x1 = x.reshape(B, self.num_chunks, self.chunk_size, N)
        x1 = x1.permute(0, 3, 2, 1)
        x1 = x1.reshape(-1, self.chunk_size, self.num_chunks)
        x1 = self.layer_1(x1)
        x1 = self.chunk_proj_1(x1).squeeze(dim=-1)

        # interval sampling
        x2 = x.reshape(B, self.chunk_size, self.num_chunks, N)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2.reshape(-1, self.chunk_size, self.num_chunks)
        x2 = self.layer_2(x2)
        x2 = self.chunk_proj_2(x2).squeeze(dim=-1)

        x3 = torch.cat([x1, x2], dim=-1)

        x3 = x3.reshape(B, N, -1)
        x3 = x3.permute(0, 2, 1)

        out = self.layer_3(x3)

        out = out + highway
        return out


class LightBunny(nn.Module):
    def __init__(self, configs: LightBunnyConfig):
        super().__init__()

        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.raw_channel_in = configs.raw_channel_in

        self.channelTransforming = ResidualLinear(self.raw_channel_in, self.d_model)
        self.rnn_pre = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.enc_in,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )

        self.lightTS = LightTS(configs)

    def forward(self, x: Tensor) -> Tensor:
        # b,t,c0 -> b,t,d -> b,t,c
        x = self.channelTransforming(x)
        x, _ = self.rnn_pre(x)

        x = self.lightTS(x)

        return x
