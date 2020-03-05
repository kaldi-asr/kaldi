#!/usr/bin/env python3

# Copyright 2020 JD AI, Beijing, China (author: Lu Fan)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi
from kaldi import read_nnet3_model
from torch.utils.dlpack import from_dlpack
from torch.utils.dlpack import to_dlpack

class TestNnetNnet(unittest.TestCase):

    def test_nnet_nnet(self):
        kaldi.SelectGpuId('yes')
        final_mdl = "/mnt/cfs1_alias1/asr/users/fanlu/task/kaldi_recipe/pybind/s10.1/exp/chain_cleaned_1c/tdnn1c_sp/final.mdl"
        nnet = kaldi.read_nnet3_model(final_mdl)
        for i in range(nnet.NumComponents()):
            component = nnet.GetComponent(i)
            comp_type = component.Type()
            if "Affine" in comp_type or "TdnnComponent" in comp_type:
                linear_params = from_dlpack(component.LinearParams().to_dlpack())
                bias_params = from_dlpack(component.BiasParams().to_dlpack())
                print(linear_params.shape)
            elif "Batch" in comp_type:
                # stats_sum = from_dlpack(component.StatsSum().to_dlpack())
                # stats_sumsq = from_dlpack(component.StatsSumsq().to_dlpack())
                # print(stats_sum.shape)
                pass
            elif "LinearComponent" == comp_type:
                linear_params = from_dlpack(component.LinearParams().to_dlpack())
                print(linear_params.shape)

if __name__ == '__main__':
    unittest.main()