

# Copyright 2016    Vijayaditya Peddinti.
#           2016    Vimal Manohar
# Apache 2.0.

""" This is a module with method which will be used by scripts for
training of deep neural network raw model (i.e. without acoustic model)
with frame-level objective.
"""

import logging

import libs.common as common_lib

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def generate_egs_using_targets(data, targets_scp, egs_dir,
                               left_context, right_context,
                               run_opts, stage=0,
                               left_context_initial=-1, right_context_final=-1,
                               feat_type='raw', online_ivector_dir=None,
                               target_type='dense', num_targets=-1,
                               samples_per_iter=20000, frames_per_eg_str="20",
                               srand=0, egs_opts=None, cmvn_opts=None,
                               transform_dir=None):
    """ Wrapper for calling steps/nnet3/get_egs_targets.sh

    This method generates egs directly from an scp file of targets, instead of
    getting them from the alignments (as with the method generate_egs() in
    module nnet3.train.frame_level_objf.acoustic_model).

    Args:
        target_type: "dense" if the targets are in matrix format
                     "sparse" if the targets are in posterior format
        num_targets: must be explicitly specified for "sparse" targets.
            For "dense" targets, this option is ignored and the target dim
            is computed from the target matrix dimension
        For other options, see the file steps/nnet3/get_egs_targets.sh
    """

    if target_type == 'dense':
        num_targets = common_lib.get_feat_dim_from_scp(targets_scp)
    else:
        if num_targets == -1:
            raise Exception("--num-targets is required if "
                            "target-type is sparse")

    common_lib.run_job(
        """steps/nnet3/get_egs_targets.sh {egs_opts} \
                --cmd "{command}" \
                --cmvn-opts "{cmvn_opts}" \
                --feat-type {feat_type} \
                --transform-dir "{transform_dir}" \
                --online-ivector-dir "{ivector_dir}" \
                --left-context {left_context} \
                --right-context {right_context} \
                --left-context-initial {left_context_initial} \
                --right-context-final {right_context_final} \
                --stage {stage} \
                --samples-per-iter {samples_per_iter} \
                --frames-per-eg {frames_per_eg_str} \
                --srand {srand} \
                --target-type {target_type} \
                --num-targets {num_targets} \
                {data} {targets_scp} {egs_dir}
        """.format(command=run_opts.egs_command,
                   cmvn_opts=cmvn_opts if cmvn_opts is not None else '',
                   feat_type=feat_type,
                   transform_dir=(transform_dir
                                  if transform_dir is not None
                                  else ''),
                   ivector_dir=(online_ivector_dir
                                if online_ivector_dir is not None
                                else ''),
                   left_context=left_context,
                   right_context=right_context,
                   left_context_initial=left_context_initial,
                   right_context_final=right_context_final,
                   stage=stage, samples_per_iter=samples_per_iter,
                   frames_per_eg_str=frames_per_eg_str, srand=srand,
                   num_targets=num_targets,
                   data=data,
                   targets_scp=targets_scp, target_type=target_type,
                   egs_dir=egs_dir,
                   egs_opts=egs_opts if egs_opts is not None else ''))
