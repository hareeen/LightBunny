from torch import nn, Tensor


class ResidualLinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.block1_linear1 = nn.Linear(in_features, in_features)
        self.block1_elu1 = nn.ELU()
        self.block1_linear2 = nn.Linear(in_features, in_features)
        self.block1_elu2 = nn.ELU()

        self.block2_linear1 = nn.Linear(in_features, in_features)
        self.block2_elu1 = nn.ELU()
        self.block2_linear2 = nn.Linear(in_features, out_features)
        self.block2_downsample = nn.Linear(in_features, out_features)
        self.block2_elu2 = nn.ELU()

        self.block3_linear = nn.Linear(out_features, out_features)
        self.block3_elu = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        iden = x
        out = self.block1_linear1(x)
        out = self.block1_elu1(out)
        out = self.block1_linear2(out)
        out += iden
        out = self.block1_elu2(out)

        iden = out
        out = self.block2_linear1(out)
        out = self.block2_elu1(out)
        out = self.block2_linear2(out)
        out += self.block2_downsample(iden)
        out = self.block1_elu2(out)

        out = self.block3_linear(out)
        out = self.block3_elu(out)

        return out


class DeepSIF(nn.Module):
    def __init__(
        self,
        num_sensor: int,
        num_source: int,
        num_lstm_layers: int = 3,
        hidden_features: int = 500,
    ):
        super().__init__()

        self.spatial = ResidualLinearBlock(num_sensor, hidden_features)
        self.temporal = nn.LSTM(
            hidden_features, num_source, num_lstm_layers, batch_first=True
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.spatial(x)

        self.temporal.flatten_parameters()
        out = self.temporal(out)[0]

        return out

