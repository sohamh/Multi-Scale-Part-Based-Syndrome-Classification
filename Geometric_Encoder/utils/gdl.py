import torch
import torch.nn as nn
import torch.nn.functional as F


def Pool(x, trans, spiral=None, mode=None):
    if mode == 'max':
        out = torch.empty(x.shape[0], trans.shape[0], x.shape[-1])
        _, col = trans._indices()
        for i, c in enumerate(col):
            tmp, _ = torch.max(x[:, spiral[0], :], 1)
            out[:, i, :] = tmp

    elif mode == 'min':
        out = torch.empty(x.shape[0], trans.shape[0], x.shape[-1])
        _, col = trans._indices()
        for i, c in enumerate(col):
            tmp, _ = torch.min(x[:, spiral[0], :], 1)
            out[:, i, :] = tmp

    elif mode == 'mean':
        out = torch.empty(x.shape[0], trans.shape[0], x.shape[-1])
        _, col = trans._indices()
        for i, c in enumerate(col):
            tmp = torch.mean(x[:, spiral[0], :], 1)
            out[:, i, :] = tmp

    else:
        out = torch.matmul(trans.to_dense(), x)

    return out


class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices, pool=None):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.pool = pool
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform, self.conv.indices, self.pool)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out