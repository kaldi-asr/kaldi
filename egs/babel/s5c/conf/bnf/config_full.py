#################################################
## PTDNN - Python Toolkit for Deep Neural Network
## Author: Yajie Miao
#################################################

import os
import sys

from utils.learn_rates import LearningRateExpDecay


class BnfExpConfig(object):

    def __init__(self):

        # working directory; by default, the pfiles should be here
        self.wdir = "WORK/"
        self.pretrain_data = self.wdir + 'train.pfile.gz'  # pretraining data
        self.pretrain_output = self.wdir + "rbm.ptr"       # pretraining output
    
        # finetuning data
        self.finetune_train_data = self.wdir + 'train.pfile.gz'   # finetune training data
        self.finetune_valid_data = self.wdir + 'valid.pfile.gz'   # finetune validation data
        self.finetune_output = self.wdir + "final.nnet.raw"           # finetune output
        self.nnet_kaldi_fmt = self.wdir + "final.nnet"

        # global config for nnet topo
        self.n_ins=250                                   # size of input data
        self.n_outs=N_OUTS                               # number of output targets.. we'll replace this with 
                                                         # the correct number when we move this to the right place.
        self.hidden_layers_sizes=[1024, 1024, 1024, 1024, 1024, 42, 1024] # hidden layer sizes
        self.bnf_layer_index = 6                         # the index of the Bottleneck layer
        self.pretrain_layer_num = 5                      # number of hidden layers to be pretrained
   
        # global config for data
        self.shuffle = True
        self.chunk_size = '200m'

        # pretraining batch size
        self.pretrain_batch_size = 128              # batch-size in pretraining                             

        # pretraining schedule
        self.pretrain_gbrbm_lr = 0.005              # learning rate for Gaussian-Bernoulli RBM
        self.pretrain_rbm_lr = 0.08                 # learning rate for Bernoulli-Bernoulli RBM
        self.initial_momentum = 0.5                 # initial momentum 
        self.final_momentum = 0.9                   # final momentum
        self.initial_momentum_epoch = 2             # for how many epochs do we use initial_momentum
        self.pretraining_epochs = 4                   # total epochs 

        # finetuning batch size
        self.finetune_batch_size = 256              # batch-size for finetuning

        # finetuning schedule
        self.finetune_momentum = 0.5                # momentum for finetuning
        self.lrate = LearningRateExpDecay(start_rate=0.04,             # starting learning rate
                                          scale_by = 0.5,               # decaying factor in ramping
                                          max_epochs = 1000,            # 'dump' epoch limit, never can be reached
                                          min_derror_ramp_start = 0.01, # min validation error difference to trigger ramping
                                          min_derror_stop = 0.01,       # min validation error difference to stop finetuning, after ramping
                                          init_error = 100)
