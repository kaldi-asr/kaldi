#!/usr/bin/env python3

# Copyright 2020 JD AI, Beijing, China (author: Lu Fan)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

try:
    import torch
    from torch.utils.dlpack import to_dlpack
    from torch.utils.dlpack import from_dlpack
except ImportError:
    print('This test needs PyTorch.')
    print('Please install PyTorch first.')
    print('PyTorch 1.3.0dev20191006 has been tested and is known to work.')
    sys.exit(0)

import kaldi

"""
input dim=40 name=input

# please note that it is important to have input layer with the name=input
# as the layer immediately preceding the fixed-affine-layer to enable
# the use of short notation for the descriptor
fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=$dir/configs/lda.mat

# the first splicing is moved before the lda layer, so no splicing here
relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=16
tdnnf-layer name=tdnnf2 $tdnnf_opts dim=16 bottleneck-dim=2 time-stride=1
linear-component name=prefinal-l dim=4 $linear_opts

prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=16 small-dim=4
output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=16 small-dim=4
output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
"""
class TestNnetNnet(unittest.TestCase):

    def test_nnet_nnet(self):
        if torch.cuda.is_available() == False:
            print('No GPU detected! Skip it')
            return

        if kaldi.CudaCompiled() == False:
            print('Kaldi is not compiled with CUDA! Skip it')
            return

        device_id = 0

        # Kaldi and PyTorch will use the same GPU
        kaldi.SelectGpuDevice(device_id=device_id)
        kaldi.CuDeviceAllowMultithreading()

        final_mdl = 'final.mdl'
        nnet = kaldi.read_nnet3_model(final_mdl)
        for i in range(nnet.NumComponents()):
            component = nnet.GetComponent(i)
            comp_type = component.Type()
            if comp_type in ['RectifiedLinearComponent', 'GeneralDropoutComponent',
                             'NoOpComponent']:
                continue
            comp_name = nnet.GetComponentName(i)
            if comp_name == 'lda':
                self.assertEqual(comp_type, 'FixedAffineComponent')
                linear_params = from_dlpack(
                    component.LinearParams().to_dlpack())
                bias_params = from_dlpack(component.BiasParams().to_dlpack())
                self.assertEqual(linear_params.shape, (120, 120))
                self.assertEqual(bias_params.shape, (120,))
            elif comp_name == 'tdnn1.affine':
                self.assertEqual(comp_type, 'NaturalGradientAffineComponent')
                linear_params = from_dlpack(
                    component.LinearParams().to_dlpack())
                bias_params = from_dlpack(component.BiasParams().to_dlpack())
                self.assertEqual(linear_params.shape, (16, 120))
                self.assertEqual(bias_params.shape, (16,))
            elif comp_name == 'tdnn1.batchnorm':
                self.assertEqual(comp_type, 'BatchNormComponent')
                component.SetTestMode(True)
                mean = from_dlpack(component.Mean().to_dlpack())
                var = from_dlpack(component.Var().to_dlpack())
                self.assertEqual(mean.shape, (16,))
                self.assertEqual(var.shape, (16,))
            elif comp_name == 'tdnnf2.linear':
                self.assertEqual(comp_type, 'TdnnComponent')
                linear_params = from_dlpack(
                    component.LinearParams().to_dlpack())
                self.assertEqual(linear_params.shape, (2, 32))
            elif comp_name == 'tdnnf2.affine':
                self.assertEqual(comp_type, 'TdnnComponent')
                linear_params = from_dlpack(
                    component.LinearParams().to_dlpack())
                bias_params = from_dlpack(component.BiasParams().to_dlpack())
                self.assertEqual(linear_params.shape, (16, 4))
                self.assertEqual(bias_params.shape, (16,))
            elif comp_name == 'tdnnf2.batchnorm':
                self.assertEqual(comp_type, 'BatchNormComponent')
                component.SetTestMode(True)
                mean = from_dlpack(component.Mean().to_dlpack())
                var = from_dlpack(component.Var().to_dlpack())
                self.assertEqual(mean.shape, (16,))
                self.assertEqual(var.shape, (16,))
            elif comp_name == 'prefinal-l':
                self.assertEqual(comp_type, 'LinearComponent')
                params = from_dlpack(component.Params().to_dlpack())
                self.assertEqual(params.shape, (4, 16))
            elif comp_name == 'prefinal-chain.affine':
                self.assertEqual(comp_type, 'NaturalGradientAffineComponent')
                linear_params = from_dlpack(
                    component.LinearParams().to_dlpack())
                bias_params = from_dlpack(component.BiasParams().to_dlpack())
                self.assertEqual(linear_params.shape, (16, 4))
                self.assertEqual(bias_params.shape, (16,))
            elif comp_name == 'prefinal-chain.batchnorm1':
                self.assertEqual(comp_type, 'BatchNormComponent')
                component.SetTestMode(True)
                mean = from_dlpack(component.Mean().to_dlpack())
                var = from_dlpack(component.Var().to_dlpack())
                self.assertEqual(mean.shape, (16,))
                self.assertEqual(var.shape, (16,))
            elif comp_name == 'prefinal-chain.linear':
                self.assertEqual(comp_type, 'LinearComponent')
                params = from_dlpack(component.Params().to_dlpack())
                self.assertEqual(linear_params.shape, (16, 4))
            elif comp_name == 'prefinal-chain.batchnorm2':
                self.assertEqual(comp_type, 'BatchNormComponent')
                component.SetTestMode(True)
                mean = from_dlpack(component.Mean().to_dlpack())
                var = from_dlpack(component.Var().to_dlpack())
                self.assertEqual(mean.shape, (4,))
                self.assertEqual(var.shape, (4,))
            elif comp_name == 'output.affine':
                self.assertEqual(comp_type, 'NaturalGradientAffineComponent')
                linear_params = from_dlpack(
                    component.LinearParams().to_dlpack())
                bias_params = from_dlpack(component.BiasParams().to_dlpack())
                self.assertEqual(linear_params.shape, (3448, 4))
                self.assertEqual(bias_params.shape, (3448,))
            elif comp_name == 'prefinal-xent.affine':
                self.assertEqual(comp_type, 'NaturalGradientAffineComponent')
                linear_params = from_dlpack(
                    component.LinearParams().to_dlpack())
                bias_params = from_dlpack(component.BiasParams().to_dlpack())
                self.assertEqual(linear_params.shape, (16, 4))
                self.assertEqual(bias_params.shape, (16,))
            elif comp_name == 'prefinal-xent.batchnorm1':
                self.assertEqual(comp_type, 'BatchNormComponent')
                component.SetTestMode(True)
                mean = from_dlpack(component.Mean().to_dlpack())
                var = from_dlpack(component.Var().to_dlpack())
                self.assertEqual(mean.shape, (16,))
                self.assertEqual(var.shape, (16,))
            elif comp_name == 'prefinal-xent.linear':
                self.assertEqual(comp_type, 'LinearComponent')
                params = from_dlpack(component.Params().to_dlpack())
                self.assertEqual(linear_params.shape, (16, 4))
            elif comp_name == 'prefinal-xent.batchnorm2':
                self.assertEqual(comp_type, 'BatchNormComponent')
                component.SetTestMode(True)
                mean = from_dlpack(component.Mean().to_dlpack())
                var = from_dlpack(component.Var().to_dlpack())
                self.assertEqual(mean.shape, (4,))
                self.assertEqual(var.shape, (4,))
            elif comp_name == 'output-xent.affine':
                self.assertEqual(comp_type, 'NaturalGradientAffineComponent')
                linear_params = from_dlpack(
                    component.LinearParams().to_dlpack())
                bias_params = from_dlpack(component.BiasParams().to_dlpack())
                self.assertEqual(linear_params.shape, (3448, 4))
                self.assertEqual(bias_params.shape, (3448,))
            else:
                self.assertEqual(comp_type, 'LogSoftmaxComponent')

if __name__ == '__main__':
    unittest.main()
