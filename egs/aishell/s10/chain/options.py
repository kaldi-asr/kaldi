#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import argparse
import os


def _str2bool(v):
    '''
    This function is modified from
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    '''
    if isinstance(v, bool):
        return v
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _set_inference_args(parser):
    parser.add_argument('--feats-scp',
                        dest='feats_scp',
                        help='feats.scp filename, required for inference',
                        type=str)

    parser.add_argument('--model-left-context',
                        dest='model_left_context',
                        help='model left context',
                        type=int,
                        default=-1)

    parser.add_argument('--model-right-context',
                        dest='model_right_context',
                        help='model right context',
                        type=int,
                        default=-1)

    parser.add_argument(
        '--save-as-compressed',
        dest='save_as_compressed',
        help='true to save the neural network output to CompressedMatrix,'
        ' false to Matrix<float>',
        type=_str2bool)


def _set_training_args(parser):
    parser.add_argument('--train.cegs-dir',
                        dest='cegs_dir',
                        help='cegs dir containing comibined cegs.*.scp',
                        type=str)

    parser.add_argument('--train.den-fst',
                        dest='den_fst_filename',
                        help='denominator fst filename',
                        type=str)

    parser.add_argument('--train.egs-left-context',
                        dest='egs_left_context',
                        help='egs left context',
                        type=int)

    parser.add_argument('--train.egs-right-context',
                        dest='egs_right_context',
                        help='egs right context',
                        type=int)

    parser.add_argument('--train.num-epochs',
                        dest='num_epochs',
                        help='number of epochs to train',
                        type=int)

    parser.add_argument('--train.lr',
                        dest='learning_rate',
                        help='learning rate',
                        type=float)

    parser.add_argument('--train.l2-regularize',
                        dest='l2_regularize',
                        help='l2 regularize',
                        type=float)


def _check_training_args(args):
    assert os.path.isdir(args.cegs_dir)

    assert os.path.isfile(args.den_fst_filename)

    assert args.egs_left_context > 0
    assert args.egs_right_context > 0

    assert args.num_epochs > 0
    assert args.learning_rate > 0
    assert args.l2_regularize > 0

    if args.checkpoint:
        assert os.path.exists(args.checkpoint)


def _check_inference_args(args):
    assert args.checkpoint is not None
    assert os.path.isfile(args.checkpoint)

    assert args.feats_scp is not None
    assert os.path.isfile(args.feats_scp)

    assert args.model_left_context > 0
    assert args.model_right_context > 0


def _check_args(args):

    if args.is_training:
        _check_training_args(args)
    else:
        _check_inference_args(args)

    if args.lda_mat_filename:
        assert os.path.isfile(args.lda_mat_filename)

    # although -1 means to use CPU in `kaldi.SelectGpuDevice()`
    # we do NOT want to use CPU here so we require it to be >= 0
    assert args.device_id >= 0

    assert args.feat_dim > 0
    assert args.output_dim > 0
    assert args.hidden_dim > 0

    assert args.kernel_size_list is not None
    assert len(args.kernel_size_list) > 0

    assert args.stride_list is not None
    assert len(args.stride_list) > 0

    args.kernel_size_list = [int(k) for k in args.kernel_size_list.split(', ')]

    args.stride_list = [int(k) for k in args.stride_list.split(', ')]

    assert len(args.kernel_size_list) == len(args.stride_list)

    assert args.log_level in ['debug', 'info', 'warning']


def get_args():
    parser = argparse.ArgumentParser(
        description='chain training in PyTorch with kaldi pybind')

    _set_training_args(parser)
    _set_inference_args(parser)

    parser.add_argument('--dir',
                        help='dir to save results. The user has to '
                        'create it before calling this script.',
                        required=True,
                        type=str)

    parser.add_argument('--device-id',
                        dest='device_id',
                        help='GPU device id',
                        required=True,
                        type=int)

    parser.add_argument('--is-training',
                        dest='is_training',
                        help='true for training, false for inference',
                        required=True,
                        type=_str2bool)

    parser.add_argument(
        '--lda-mat-filename',
        dest='lda_mat_filename',
        help='affine-transform-file in fixed-affine-layer of kaldi',
        default=None,
        type=str)

    parser.add_argument('--feat-dim',
                        dest='feat_dim',
                        help='nn input dimension',
                        required=True,
                        type=int)

    parser.add_argument('--output-dim',
                        dest='output_dim',
                        help='nn output dimension',
                        required=True,
                        type=int)

    parser.add_argument('--hidden-dim',
                        dest='hidden_dim',
                        help='nn hidden dimension',
                        required=True,
                        type=int)

    parser.add_argument('--kernel-size-list',
                        dest='kernel_size_list',
                        help='kernel size list',
                        required=True,
                        type=str)

    parser.add_argument('--stride-list',
                        dest='stride_list',
                        help='stride list',
                        required=True,
                        type=str)

    parser.add_argument('--log-level',
                        dest='log_level',
                        help='log level. valid values: debug, info, warning',
                        type=str,
                        default='info')

    parser.add_argument(
        '--checkpoint',
        dest='checkpoint',
        help='filename of the checkpoint, required for inference',
        type=str)

    args = parser.parse_args()

    _check_args(args)
    return args
