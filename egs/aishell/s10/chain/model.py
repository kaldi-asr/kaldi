#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load_lda_mat
'''
  input dim=$feat_dim name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=625
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=625
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=625
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=625

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=tdnn6 dim=625 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn6 dim=625 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
'''


def get_chain_model(feat_dim, output_dim, lda_mat_filename, hidden_dim,
                    kernel_size_list, stride_list):
    model = ChainModel(feat_dim=feat_dim,
                       output_dim=output_dim,
                       lda_mat_filename=lda_mat_filename,
                       hidden_dim=hidden_dim,
                       kernel_size_list=kernel_size_list,
                       stride_list=stride_list)
    return model


# Create a network like the above one
class ChainModel(nn.Module):

    def __init__(self,
                 feat_dim,
                 output_dim,
                 lda_mat_filename,
                 hidden_dim=625,
                 kernel_size_list=[1, 3, 3, 3, 3, 3],
                 stride_list=[1, 1, 3, 1, 1, 1],
                 frame_subsampling_factor=3):
        super().__init__()

        # at present, we current support only frame_subsampling_factor to be 3
        assert frame_subsampling_factor == 3

        assert len(kernel_size_list) == len(stride_list)
        num_layers = len(kernel_size_list)

        tdnns = []
        for i in range(num_layers):
            in_channels = hidden_dim
            if i == 0:
                in_channels = feat_dim * 3

            kernel_size = kernel_size_list[i]
            stride = stride_list[i]

            # we do not need to perform padding in Conv1d because it
            # has been included in left/right context while generating egs
            layer = nn.Conv1d(in_channels=in_channels,
                              out_channels=hidden_dim,
                              kernel_size=kernel_size,
                              stride=stride)
            tdnns.append(layer)

        self.tdnns = nn.ModuleList(tdnns)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_features=hidden_dim) for i in range(num_layers)
        ])

        self.prefinal_chain_tdnn = nn.Conv1d(in_channels=hidden_dim,
                                             out_channels=hidden_dim,
                                             kernel_size=1)
        self.prefinal_chain_batch_norm = nn.BatchNorm1d(num_features=hidden_dim)
        self.output_fc = nn.Linear(in_features=hidden_dim,
                                   out_features=output_dim)

        self.prefinal_xent_tdnn = nn.Conv1d(in_channels=hidden_dim,
                                            out_channels=hidden_dim,
                                            kernel_size=1)
        self.prefinal_xent_batch_norm = nn.BatchNorm1d(num_features=hidden_dim)
        self.output_xent_fc = nn.Linear(in_features=hidden_dim,
                                        out_features=output_dim)

        self.lda_A, self.lda_b = load_lda_mat(lda_mat_filename)

        assert feat_dim * 3 == self.lda_A.shape[0]

    def forward(self, x):
        # input x is of shape: [batch_size, seq_len, feat_dim] = [N, T, C]
        assert x.ndim == 3

        # to() does not copy data if lda_A is already in the expected device
        self.lda_A = self.lda_A.to(x.get_device())
        self.lda_b = self.lda_b.to(x.get_device())

        x = torch.matmul(x, self.lda_A) + self.lda_b

        # at this point, x is [N, T, C]

        x = x.permute(0, 2, 1)

        # at this point, x is [N, C, T]

        # Conv1d requires input of shape [N, C, T]
        for i in range(len(self.tdnns)):
            x = self.tdnns[i](x)
            x = F.relu(x)
            x = self.batch_norms[i](x)

        # at this point, x is [N, C, T]

        # we have two branches from this point on

        # first, for the chain branch
        x_chain = self.prefinal_chain_tdnn(x)
        x_chain = F.relu(x_chain)
        x_chain = self.prefinal_chain_batch_norm(x_chain)
        x_chain = x_chain.permute(0, 2, 1)
        # at this point, x_chain is [N, T, C]
        nnet_output = self.output_fc(x_chain)

        # now for the xent branch
        x_xent = self.prefinal_xent_tdnn(x)
        x_xent = F.relu(x_xent)
        x_xent = self.prefinal_xent_batch_norm(x_xent)
        x_xent = x_xent.permute(0, 2, 1)

        # at this point x_xent is [N, T, C]
        xent_output = self.output_xent_fc(x_xent)
        xent_output = F.log_softmax(xent_output, dim=-1)

        return nnet_output, xent_output
