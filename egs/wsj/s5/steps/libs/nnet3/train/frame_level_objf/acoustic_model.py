

# Copyright 2016    Vijayaditya Peddinti.
#           2016    Vimal Manohar
# Apache 2.0.

""" This is a module with method which will be used by scripts for
training of deep neural network acoustic model with frame-level objective.
"""

import logging

import libs.common as common_lib
import libs.nnet3.train.common as common_train_lib


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def generate_egs(data, alidir, egs_dir,
                 left_context, right_context,
                 run_opts, stage=0,
                 left_context_initial=-1, right_context_final=-1,
                 online_ivector_dir=None,
                 samples_per_iter=20000, frames_per_eg_str="20", srand=0,
                 egs_opts=None, cmvn_opts=None):

    """ Wrapper for calling steps/nnet3/get_egs.sh

    Generates targets from alignment directory 'alidir', which contains
    the model final.mdl and alignments.
    """

    common_lib.execute_command(
        """steps/nnet3/get_egs.sh {egs_opts} \
                --cmd "{command}" \
                --cmvn-opts "{cmvn_opts}" \
                --online-ivector-dir "{ivector_dir}" \
                --left-context {left_context} \
                --right-context {right_context} \
                --left-context-initial {left_context_initial} \
                --right-context-final {right_context_final} \
                --stage {stage} \
                --samples-per-iter {samples_per_iter} \
                --frames-per-eg {frames_per_eg_str} \
                --srand {srand} \
                {data} {alidir} {egs_dir}
        """.format(command=run_opts.egs_command,
                   cmvn_opts=cmvn_opts if cmvn_opts is not None else '',
                   ivector_dir=(online_ivector_dir
                                if online_ivector_dir is not None
                                else ''),
                   left_context=left_context,
                   right_context=right_context,
                   left_context_initial=left_context_initial,
                   right_context_final=right_context_final,
                   stage=stage, samples_per_iter=samples_per_iter,
                   frames_per_eg_str=frames_per_eg_str, srand=srand, data=data,
                   alidir=alidir, egs_dir=egs_dir,
                   egs_opts=egs_opts if egs_opts is not None else ''))


def prepare_initial_acoustic_model(dir, alidir, run_opts,
                                   srand=-3, input_model=None):
    """ Adds the first layer; this will also add in the lda.mat and
        presoftmax_prior_scale.vec. It will also prepare the acoustic model
        with the transition model.
        If 'input_model' is specified, no initial network preparation(adding
        the first layer) is done and this model is used as initial 'raw' model
        instead of '0.raw' model to prepare '0.mdl' as acoustic model by adding the
        transition model.
    """

    if input_model is None:
        common_train_lib.prepare_initial_network(dir, run_opts,
                                                 srand=srand)

    # Convert to .mdl, train the transitions, set the priors.
    common_lib.execute_command(
        """{command} {dir}/log/init_mdl.log \
                nnet3-am-init {alidir}/final.mdl {raw_mdl} - \| \
                nnet3-am-train-transitions - \
                "ark:gunzip -c {alidir}/ali.*.gz|" {dir}/0.mdl
        """.format(command=run_opts.command,
                   dir=dir, alidir=alidir,
                   raw_mdl=(input_model if input_model is not None
                            else '{0}/0.raw'.format(dir))))
