

# Copyright 2016 Vijayaditya Peddinti.
#           2016 Vimal Manohar
# Apache 2.0.

""" This is a module with method which will be used by scripts for
training of deep neural network acoustic model with frame-level objective.
"""

import logging
import math
import imp
import os
import sys

sys.path.append("steps/libs")
import common as common_lib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def GenerateEgs(data, alidir, egs_dir,
                left_context, right_context,
                valid_left_context, valid_right_context,
                run_opts, stage=0,
                feat_type='raw', online_ivector_dir=None,
                samples_per_iter=20000, frames_per_eg=20, srand=0,
                egs_opts=None, cmvn_opts=None, transform_dir=None):

    """ Wrapper for calling steps/nnet3/get_egs.sh

    Generates targets from alignment directory 'alidir', which contains
    the model final.mdl and alignments.
    """

    common_lib.RunKaldiCommand("""
steps/nnet3/get_egs.sh {egs_opts} \
  --cmd "{command}" \
  --cmvn-opts "{cmvn_opts}" \
  --feat-type {feat_type} \
  --transform-dir "{transform_dir}" \
  --online-ivector-dir "{ivector_dir}" \
  --left-context {left_context} --right-context {right_context} \
  --valid-left-context {valid_left_context} \
  --valid-right-context {valid_right_context} \
  --stage {stage} \
  --samples-per-iter {samples_per_iter} \
  --frames-per-eg {frames_per_eg} \
  --srand {srand} \
  {data} {alidir} {egs_dir}
  """.format(command=run_opts.command,
             cmvn_opts=cmvn_opts if cmvn_opts is not None else '',
             feat_type=feat_type,
             transform_dir=transform_dir if transform_dir is not None else '',
             ivector_dir=online_ivector_dir if online_ivector_dir is not None else '',
             left_context=left_context, right_context=right_context,
             valid_left_context=valid_left_context,
             valid_right_context=valid_right_context,
             stage=stage, samples_per_iter=samples_per_iter,
             frames_per_eg=frames_per_eg, srand=srand, data=data, alidir=alidir,
             egs_dir=egs_dir,
             egs_opts=egs_opts if egs_opts is not None else ''))
