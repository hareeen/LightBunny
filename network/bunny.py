import torch
import torch.nn as nn
from torch import Tensor


# class Config:
#     raw_channel_in = n_electrodes
#     seq_len = seq_len
#     pred_len = seq_len
#     enc_in = 994
#     dropout = 0.2
#     d_model = 256
#     seg_len = 25


class BunnyConfig:
    seq_len: int
    pred_len: int
    enc_in: int
    raw_channel_in: int
    dropout: float
    d_model: int
    seg_len: int


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


class Bunny(nn.Module):
    def __init__(self, configs: BunnyConfig):
        super().__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.raw_channel_in = configs.raw_channel_in

        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        self.channelTransforming = ResidualLinear(self.raw_channel_in, self.d_model)
        self.rnn_pre = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.enc_in,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )

        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)

        # permute     b,s,c0 -> b,s,c -> b,c,s
        x = self.channelTransforming(x)
        x, _ = self.rnn_pre(x)
        x = x.permute(0, 2, 1)  # b,c,s

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        _, hn = self.rnn(x)  # bc,n,d  1,bc,d

        # m,d//2 -> 1,m,d//2 -> c,m,d//2
        # c,d//2 -> c,1,d//2 -> c,m,d//2
        # c,m,d -> cm,1,d -> bcm, 1, d
        pos_emb = (
            torch.cat(
                [
                    self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                    self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1),
                ],
                dim=-1,
            )
            .view(-1, 1, self.d_model)
            .repeat(batch_size, 1, 1)
        )

        _, hy = self.rnn(
            pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)
        )  # bcm,1,d  1,bcm,d

        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1)

        return y
