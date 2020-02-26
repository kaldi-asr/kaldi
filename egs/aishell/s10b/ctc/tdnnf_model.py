#!/usr/bin/env python3

# Copyright 2019-2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from add_deltas_layer import AddDeltasLayer
from tdnnf_layer import FactorizedTDNN
from tdnnf_layer import OrthonormalLinear
from tdnnf_layer import PrefinalLayer


def get_tdnnf_model(input_dim, output_dim, hidden_dim, bottleneck_dim,
                    prefinal_bottleneck_dim, kernel_size_list,
                    subsampling_factor_list):
    model = TdnnfModel(input_dim=input_dim,
                       output_dim=output_dim,
                       hidden_dim=hidden_dim,
                       bottleneck_dim=bottleneck_dim,
                       prefinal_bottleneck_dim=prefinal_bottleneck_dim,
                       kernel_size_list=kernel_size_list,
                       subsampling_factor_list=subsampling_factor_list)
    return model


'''
input dim=43 name=input

# please note that it is important to have input layer with the name=input
# as the layer immediately preceding the fixed-affine-layer to enable
# the use of short notation for the descriptor
fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=exp/chain_cleaned_1c/tdnn1c_sp/configs/lda.mat

# the first splicing is moved before the lda layer, so no splicing here
relu-batchnorm-dropout-layer name=tdnn1 l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim-continuous=true dim=1024
tdnnf-layer name=tdnnf2 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=1
tdnnf-layer name=tdnnf3 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=1
tdnnf-layer name=tdnnf4 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=1
tdnnf-layer name=tdnnf5 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=0
tdnnf-layer name=tdnnf6 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
tdnnf-layer name=tdnnf7 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
tdnnf-layer name=tdnnf8 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
tdnnf-layer name=tdnnf9 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
tdnnf-layer name=tdnnf10 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
tdnnf-layer name=tdnnf11 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
tdnnf-layer name=tdnnf12 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
tdnnf-layer name=tdnnf13 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
linear-component name=prefinal-l dim=256 l2-regularize=0.008 orthonormal-constraint=-1.0

prefinal-layer name=prefinal-chain input=prefinal-l l2-regularize=0.008 big-dim=1024 small-dim=256
output-layer name=output include-log-softmax=false dim=3456 l2-regularize=0.002

prefinal-layer name=prefinal-xent input=prefinal-l l2-regularize=0.008 big-dim=1024 small-dim=256
output-layer name=output-xent dim=3456 learning-rate-factor=5.0 l2-regularize=0.002
'''


# Create a network like the above one
class TdnnfModel(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=1024,
                 bottleneck_dim=128,
                 prefinal_bottleneck_dim=256,
                 kernel_size_list=[3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3],
                 subsampling_factor_list=[1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1]):
        super().__init__()

        assert len(kernel_size_list) == len(subsampling_factor_list)
        num_layers = len(kernel_size_list)

        # deltas_layer requires [N, C, T]
        self.delta_layer = AddDeltasLayer()

        # batch_norm0 requires [N, C, T]
        self.batch_norm0 = nn.BatchNorm1d(num_features=input_dim * 3,
                                          affine=False)

        # tdnn1_affine requires [N, T, C]
        self.tdnn1_affine = nn.Linear(in_features=input_dim * 3,
                                      out_features=hidden_dim)

        # tdnn1_batchnorm requires [N, C, T]
        self.tdnn1_batchnorm = nn.BatchNorm1d(num_features=hidden_dim,
                                              affine=False)

        tdnnfs = []
        for i in range(num_layers):
            kernel_size = kernel_size_list[i]
            subsampling_factor = subsampling_factor_list[i]
            layer = FactorizedTDNN(dim=hidden_dim,
                                   bottleneck_dim=bottleneck_dim,
                                   kernel_size=kernel_size,
                                   subsampling_factor=subsampling_factor)
            tdnnfs.append(layer)

        # tdnnfs requires [N, C, T]
        self.tdnnfs = nn.ModuleList(tdnnfs)

        # prefinal_l affine requires [N, C, T]
        self.prefinal_l = OrthonormalLinear(
            dim=hidden_dim,
            bottleneck_dim=prefinal_bottleneck_dim,
            kernel_size=1)

        # prefinal_chain requires [N, C, T]
        self.prefinal_chain = PrefinalLayer(big_dim=hidden_dim,
                                            small_dim=prefinal_bottleneck_dim)

        # output_affine requires [N, T, C]
        self.output_affine = nn.Linear(in_features=prefinal_bottleneck_dim,
                                       out_features=output_dim)

        # prefinal_xent requires [N, C, T]
        self.prefinal_xent = PrefinalLayer(big_dim=hidden_dim,
                                           small_dim=prefinal_bottleneck_dim)

        self.output_xent_affine = nn.Linear(in_features=prefinal_bottleneck_dim,
                                            out_features=output_dim)

    # TODO(fangjun): avoid `permute`.
    def forward(self, x, feat_len_list):
        # input x is of shape: [batch_size, seq_len, input_dim] = [N, T, C]
        assert x.ndim == 3

        # at this point, x is [N, T, C]
        x = x.permute(0, 2, 1)

        # at this point, x is [N, C, T]
        x = self.delta_layer(x)

        # at this point, x is [N, C, T]
        x = self.batch_norm0(x)

        # at this point, x is [N, C, T]

        x = x.permute(0, 2, 1)

        # at this point, x is [N, T, C]

        x = self.tdnn1_affine(x)

        # at this point, x is [N, T, C]

        x = F.relu(x)

        x = x.permute(0, 2, 1)

        # at this point, x is [N, C, T]

        x = self.tdnn1_batchnorm(x)

        # tdnnf requires input of shape [N, C, T]
        for i in range(len(self.tdnnfs)):
            x = self.tdnnfs[i](x)

        # at this point, x is [N, C, T]

        x = self.prefinal_l(x)

        # at this point, x is [N, C, T]

        x = self.prefinal_chain(x)

        # at this point, x is [N, C, T]
        x = x.permute(0, 2, 1)

        # at this point, x is [N, T, C]
        x = self.output_affine(x)

        feat_len_list = (torch.tensor(feat_len_list).int() + 2) / 3

        return x, feat_len_list

    def constrain_orthonormal(self):
        for i in range(len(self.tdnnfs)):
            self.tdnnfs[i].constrain_orthonormal()

        self.prefinal_l.constrain_orthonormal()
        self.prefinal_chain.constrain_orthonormal()
        self.prefinal_xent.constrain_orthonormal()


if __name__ == '__main__':
    input_dim = 40
    output_dim = 218
    model = TdnnfModel(input_dim=input_dim, output_dim=output_dim)
    N = 1
    T = 150 + 29 + 29
    C = input_dim
    x = torch.arange(N * T * C).reshape(N, T, C).float()
    nnet_output = model(x)
    print(x.shape, nnet_output.shape)
    model.constrain_orthonormal()
