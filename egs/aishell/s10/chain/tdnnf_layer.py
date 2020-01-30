#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


def constraint_orthonormal_internal(M):
    '''
    Refer to
        void ConstrainOrthonormalInternal(BaseFloat scale, CuMatrixBase<BaseFloat> *M)
    from
        https://github.com/kaldi-asr/kaldi/blob/master/src/nnet3/nnet-utils.cc#L982

    Note that we always use the **floating** case.
    '''
    assert M.ndim == 2

    num_rows = M.size(0)
    num_cols = M.size(1)

    assert num_rows <= num_cols

    # P = M * M^T
    P = torch.mm(M, M.t())
    P_PT = torch.mm(P, P.t())

    trace_P = torch.trace(P)
    trace_P_P = torch.trace(P_PT)

    scale = torch.sqrt(trace_P_P / trace_P)

    ratio = trace_P_P * num_rows / (trace_P * trace_P)
    assert ratio > 0.99

    update_speed = 0.125

    if ratio > 1.02:
        update_speed *= 0.5
        if ratio > 1.1:
            update_speed *= 0.5

    identity = torch.eye(num_rows, dtype=P.dtype, device=P.device)
    P = P - scale * scale * identity

    alpha = update_speed / (scale * scale)
    M = M - 4 * alpha * torch.mm(P, M)
    return M


class FactorizedTDNN(nn.Module):
    '''
    This class implements the following topology in kaldi:
      tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1

    References:
        - http://danielpovey.com/files/2018_interspeech_tdnnf.pdf
        - ConstrainOrthonormalInternal() from
        https://github.com/kaldi-asr/kaldi/blob/master/src/nnet3/nnet-utils.cc#L982
    '''

    def __init__(self, dim, bottleneck_dim, time_stride, bypass_scale=0.66):
        super().__init__()

        assert time_stride in [0, 1]
        assert abs(bypass_scale) <= 1

        self.bypass_scale = bypass_scale

        if time_stride == 0:
            kernel_size = 1
        else:
            kernel_size = 3

        self.kernel_size = kernel_size

        # WARNING(fangjun): kaldi uses [-1, 0] for the first linear layer
        # and [0, 1] for the second affine layer;
        # We use [-1, 0, 1] for the first linear layer

        # conv requires [N, C, T]
        self.conv = nn.Conv1d(in_channels=dim,
                              out_channels=bottleneck_dim,
                              kernel_size=kernel_size,
                              bias=False)

        # affine requires [N, T, C]
        self.affine = nn.Linear(in_features=bottleneck_dim, out_features=dim)

        # batchnorm requires [N, C, T]
        self.batchnorm = nn.BatchNorm1d(num_features=dim)

    def forward(self, x):
        # input x is of shape: [batch_size, feat_dim, seq_len] = [N, C, T]
        assert x.ndim == 3

        # save it for skip connection
        input_x = x

        x = self.conv(x)
        # at this point, x is [N, C, T]

        x = x.permute(0, 2, 1)
        # at this point, x is [N, T, C]

        x = self.affine(x)
        # at this point, x is [N, T, C]

        x = F.relu(x)
        # at this point, x is [N, T, C]

        x = x.permute(0, 2, 1)
        # at this point, x is [N, C, T]
        x = self.batchnorm(x)

        # TODO(fangjun): implement GeneralDropoutComponent in PyTorch

        # at this point, x is [N, C, T]
        if self.kernel_size == 3:
            x = self.bypass_scale * input_x[:, :, 1:-1] + x
        else:
            x = self.bypass_scale * input_x + x
        return x

    def constraint_orthonormal(self):
        state_dict = self.conv.state_dict()
        w = state_dict['weight']
        # w is of shape [out_channels, in_channels, kernel_size]
        out_channels = w.size(0)
        in_channels = w.size(1)
        kernel_size = w.size(2)

        w = w.reshape(out_channels, -1)

        num_rows = w.size(0)
        num_cols = w.size(1)

        need_transpose = False
        if num_rows > num_cols:
            w = w.t()
            need_transpose = True

        w = constraint_orthonormal_internal(w)

        if need_transpose:
            w = w.t()

        w = w.reshape(out_channels, in_channels, kernel_size)

        state_dict['weight'] = w
        self.conv.load_state_dict(state_dict)


def _test_constraint_orthonormal():

    def compute_loss(M):
        P = torch.mm(M, M.t())
        P_PT = torch.mm(P, P.t())

        trace_P = torch.trace(P)
        trace_P_P = torch.trace(P_PT)

        scale = torch.sqrt(trace_P_P / trace_P)

        identity = torch.eye(P.size(0), dtype=P.dtype, device=P.device)
        Q = P / (scale * scale) - identity
        loss = torch.norm(Q, p='fro')  # Frobenius norm

        return loss

    w = torch.randn(6, 8) * 10
    loss = []
    loss.append(compute_loss(w))
    for i in range(15):
        w = constraint_orthonormal_internal(w)
        loss.append(compute_loss(w))
    # TODO(fangjun): draw the loss using matplotlib
    print(loss)


def _test_factorized_tdnn():
    N = 1
    T = 10
    C = 4
    model = FactorizedTDNN(dim=C, bottleneck_dim=2, time_stride=1)
    x = torch.arange(N * T * C).reshape(N, C, T).float()
    y = model(x)
    assert y.size(2) == T - 2

    model = FactorizedTDNN(dim=C, bottleneck_dim=2, time_stride=0)
    y = model(x)
    assert y.size(2) == T


if __name__ == '__main__':
    torch.manual_seed(20200130)
    _test_factorized_tdnn()
    _test_constraint_orthonormal()
