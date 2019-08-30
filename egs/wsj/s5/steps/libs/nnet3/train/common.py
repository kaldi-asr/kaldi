

# Copyright 2016    Vijayaditya Peddinti.
#           2016    Vimal Manohar
# Apache 2.0

"""This module contains classes and methods common to training of
nnet3 neural networks.
"""
from __future__ import division

import argparse
import glob
import logging
import os
import math
import re
import shutil

import libs.common as common_lib
from libs.nnet3.train.dropout_schedule import *

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class RunOpts(object):
    """A structure to store run options.

    Run options like queue.pl and run.pl, along with their memory
    and parallel training options for various types of commands such
    as the ones for training, parallel-training, running on GPU etc.
    """

    def __init__(self):
        self.command = None
        self.train_queue_opt = None
        self.combine_gpu_opt = None
        self.combine_queue_opt = None
        self.prior_gpu_opt = None
        self.prior_queue_opt = None
        self.parallel_train_opts = None

def get_outputs_list(model_file, get_raw_nnet_from_am=True):
    """ Generates list of output-node-names used in nnet3 model configuration.
        It will normally return 'output'.
    """
    if get_raw_nnet_from_am:
        outputs_list = common_lib.get_command_stdout(
            "nnet3-am-info --print-args=false {0} | "
            "grep -e 'output-node' | cut -f2 -d' ' | cut -f2 -d'=' ".format(model_file))
    else:
        outputs_list = common_lib.get_command_stdout(
            "nnet3-info --print-args=false {0} | "
            "grep -e 'output-node' | cut -f2 -d' ' | cut -f2 -d'=' ".format(model_file))

    return outputs_list.split()


def get_multitask_egs_opts(egs_dir, egs_prefix="",
                           archive_index=-1,
                           use_multitask_egs=False):
    """ Generates egs option for multitask(or multilingual) training setup,
        if {egs_prefix}output.*.ark or {egs_prefix}weight.*.ark files exists in egs_dir.
        Each line in {egs_prefix}*.scp has a corresponding line containing
        name of the output-node in the network and language-dependent weight in
        {egs_prefix}output.*.ark or {egs_prefix}weight.*.ark respectively.
        e.g. Returns the empty string ('') if use_multitask_egs == False,
        otherwise something like:
        '--output=ark:foo/egs/output.3.ark --weight=ark:foo/egs/weights.3.ark'
        i.e. egs_prefix is "" for train and
        "valid_diagnostic." for validation.

        Caution: archive_index is usually an integer, but may be a string ("JOB")
        in some cases.
    """
    multitask_egs_opts = ""
    egs_suffix = ".{0}".format(archive_index) if archive_index != -1 else ""

    if use_multitask_egs:
        output_file_name = ("{egs_dir}/{egs_prefix}output{egs_suffix}.ark"
                            "".format(egs_dir=egs_dir,
                                      egs_prefix=egs_prefix,
                                      egs_suffix=egs_suffix))
        output_rename_opt = ""
        if os.path.isfile(output_file_name):
            output_rename_opt = ("--outputs=ark:{output_file_name}".format(
                output_file_name=output_file_name))

        weight_file_name = ("{egs_dir}/{egs_prefix}weight{egs_suffix}.ark"
                            "".format(egs_dir=egs_dir,
                                      egs_prefix=egs_prefix,
                                      egs_suffix=egs_suffix))
        weight_opt = ""
        if os.path.isfile(weight_file_name):
            weight_opt = ("--weights=ark:{weight_file_name}"
                          "".format(weight_file_name=weight_file_name))

        multitask_egs_opts = (
            "{output_rename_opt} {weight_opt}".format(
                output_rename_opt=output_rename_opt,
                weight_opt=weight_opt))

    return multitask_egs_opts


def get_successful_models(num_models, log_file_pattern,
                          difference_threshold=1.0):
    assert num_models > 0

    parse_regex = re.compile(
        "LOG .* Overall average objective function for "
        "'output' is ([0-9e.\-+= ]+) over ([0-9e.\-+]+) frames")
    objf = []
    for i in range(num_models):
        model_num = i + 1
        logfile = re.sub('%', str(model_num), log_file_pattern)
        lines = open(logfile, 'r').readlines()
        this_objf = -100000.0
        for line_num in range(1, len(lines) + 1):
            # we search from the end as this would result in
            # lesser number of regex searches. Python regex is slow !
            mat_obj = parse_regex.search(lines[-1 * line_num])
            if mat_obj is not None:
                this_objf = float(mat_obj.groups()[0].split()[-1])
                break
        objf.append(this_objf)
    max_index = objf.index(max(objf))
    accepted_models = []
    for i in range(num_models):
        if (objf[max_index] - objf[i]) <= difference_threshold:
            accepted_models.append(i + 1)

    if len(accepted_models) != num_models:
        logger.warn("Only {0}/{1} of the models have been accepted "
                    "for averaging, based on log files {2}.".format(
                        len(accepted_models),
                        num_models, log_file_pattern))

    return [accepted_models, max_index + 1]


def get_average_nnet_model(dir, iter, nnets_list, run_opts,
                           get_raw_nnet_from_am=True):

    next_iter = iter + 1
    if get_raw_nnet_from_am:
        out_model = ("""- \| nnet3-am-copy --set-raw-nnet=-  \
                        {dir}/{iter}.mdl {dir}/{next_iter}.mdl""".format(
                            dir=dir, iter=iter,
                            next_iter=next_iter))
    else:
        out_model = "{dir}/{next_iter}.raw".format(
            dir=dir, next_iter=next_iter)

    common_lib.execute_command(
        """{command} {dir}/log/average.{iter}.log \
                nnet3-average {nnets_list} \
                {out_model}""".format(command=run_opts.command,
                                      dir=dir,
                                      iter=iter,
                                      nnets_list=nnets_list,
                                      out_model=out_model))


def get_best_nnet_model(dir, iter, best_model_index, run_opts,
                        get_raw_nnet_from_am=True):

    best_model = "{dir}/{next_iter}.{best_model_index}.raw".format(
        dir=dir,
        next_iter=iter + 1,
        best_model_index=best_model_index)

    if get_raw_nnet_from_am:
        out_model = ("""- \| nnet3-am-copy --set-raw-nnet=- \
                        {dir}/{iter}.mdl {dir}/{next_iter}.mdl""".format(
                            dir=dir, iter=iter, next_iter=iter + 1))
    else:
        out_model = "{dir}/{next_iter}.raw".format(dir=dir,
                                                   next_iter=iter + 1)

    common_lib.execute_command(
        """{command} {dir}/log/select.{iter}.log \
                nnet3-copy {best_model} \
                {out_model}""".format(command=run_opts.command,
                                      dir=dir, iter=iter,
                                      best_model=best_model,
                                      out_model=out_model))


def validate_chunk_width(chunk_width):
    """Validate a chunk-width string , returns boolean.
    Expected to be a string representing either an integer, like '20',
    or a comma-separated list of integers like '20,30,16'"""
    if not isinstance(chunk_width, str):
        return False
    a = chunk_width.split(",")
    assert len(a) != 0  # would be code error
    for elem in a:
        try:
            i = int(elem)
            if i < 1 and i != -1:
                return False
        except:
            return False
    return True


def principal_chunk_width(chunk_width):
    """Given a chunk-width string like "20" or "50,70,40", returns the principal
    chunk-width which is the first element, as an int.  E.g. 20, or 40."""
    if not validate_chunk_width(chunk_width):
        raise Exception("Invalid chunk-width {0}".format(chunk_width))
    return int(chunk_width.split(",")[0])


def validate_range_str(range_str):
    """Helper function used inside validate_minibatch_size_str().
    Returns true if range_str is a a comma-separated list of
    positive integers and ranges of integers, like '128',
    '128,256', or '64-128,256'."""
    if not isinstance(range_str, str):
        return False
    ranges = range_str.split(",")
    assert len(ranges) > 0
    for r in ranges:
        # a range may be either e.g. '64', or '128-256'
        try:
            c = [int(x) for x in r.split(":")]
        except:
            return False
        # c should be either e.g. [ 128 ], or  [64,128].
        if len(c) == 1:
            if c[0] <= 0:
                return False
        elif len(c) == 2:
            if c[0] <= 0 or c[1] < c[0]:
                return False
        else:
            return False
    return True


def validate_minibatch_size_str(minibatch_size_str):
    """Validate a minibatch-size string (returns bool).
    A minibatch-size string might either be an integer, like '256',
    a comma-separated set of integers or ranges like '128,256' or
    '64:128,256',  or a rule like '128=64:128/256=32,64', whose format
    is: eg-length1=size-range1/eg-length2=size-range2/....
    where a size-range is a comma-separated list of either integers like '16'
    or ranges like '16:32'.  An arbitrary eg will be mapped to the size-range
    for the closest of the listed eg-lengths (the eg-length is defined
    as the number of input frames, including context frames)."""
    if not isinstance(minibatch_size_str, str):
        return False
    a = minibatch_size_str.split("/")
    assert len(a) != 0  # would be code error

    for elem in a:
        b = elem.split('=')
        # We expect b to have length 2 in the normal case.
        if len(b) != 2:
            # one-element 'b' is OK if len(a) is 1 (so there is only
            # one choice)... this would mean somebody just gave "25"
            # or something like that for the minibatch size.
            if len(a) == 1 and len(b) == 1:
                return validate_range_str(elem)
            else:
                return False
        # check that the thing before the '=' sign is a positive integer
        try:
            if int(b[0]) <= 0:
                return False
        except:
            return False  # not an integer at all.

        if not validate_range_str(b[1]):
            return False
    return True


def halve_range_str(range_str):
    """Helper function used inside halve_minibatch_size_str().
    returns half of a range [but converting resulting zeros to
    ones], e.g. '16'->'8', '16,32'->'8,16', '64:128'->'32:64'.
    Returns true if range_str is a a comma-separated list of
    positive integers and ranges of integers, like '128',
    '128,256', or '64-128,256'."""

    ranges = range_str.split(",")
    halved_ranges = []
    for r in ranges:
        # a range may be either e.g. '64', or '128:256'
        c = [str(max(1, int(x)//2)) for x in r.split(":")]
        halved_ranges.append(":".join(c))
    return ','.join(halved_ranges)


def halve_minibatch_size_str(minibatch_size_str):
    """Halve a minibatch-size string, as would be validated by
    validate_minibatch_size_str (see docs for that).  This halves
    all the integer elements of minibatch_size_str that represent minibatch
    sizes (as opposed to chunk-lengths) and that are >1."""

    if not validate_minibatch_size_str(minibatch_size_str):
        raise Exception("Invalid minibatch-size string '{0}'".format(minibatch_size_str))

    a = minibatch_size_str.split("/")
    ans = []
    for elem in a:
        b = elem.split('=')
        # We expect b to have length 2 in the normal case.
        if len(b) == 1:
            return halve_range_str(elem)
        else:
            assert len(b) == 2
            ans.append('{0}={1}'.format(b[0], halve_range_str(b[1])))
    return '/'.join(ans)


def copy_egs_properties_to_exp_dir(egs_dir, dir):
    try:
        for file in ['cmvn_opts', 'splice_opts', 'info/final.ie.id', 'final.mat',
                     'global_cmvn.stats', 'online_cmvn']:
            file_name = '{dir}/{file}'.format(dir=egs_dir, file=file)
            if os.path.isfile(file_name):
                shutil.copy(file_name, dir)
    except IOError:
        logger.error("Error while trying to copy egs "
                     "property files to {dir}".format(dir=dir))
        raise


def parse_generic_config_vars_file(var_file):
    variables = {}
    try:
        var_file_handle = open(var_file, 'r')
        for line in var_file_handle:
            parts = line.split('=')
            field_name = parts[0].strip()
            field_value = parts[1].strip()
            if field_name in ['model_left_context', 'left_context']:
                variables['model_left_context'] = int(field_value)
            elif field_name in ['model_right_context', 'right_context']:
                variables['model_right_context'] = int(field_value)
            elif field_name == 'num_hidden_layers':
                if int(field_value) > 1:
                    raise Exception(
                        "You have num_hidden_layers={0} (real meaning: your config files "
                        "are intended to do discriminative pretraining).  Since Kaldi 5.2, "
                        "this is no longer supported --> use newer config-creation scripts, "
                        "i.e. xconfig_to_configs.py.".format(field_value))
            else:
                variables[field_name] = field_value

        return variables
    except ValueError:
        # we will throw an error at the end of the function so I will just pass
        pass

    raise Exception('Error while parsing the file {0}'.format(var_file))


def get_input_model_info(input_model):
    """ This function returns a dictionary with keys "model_left_context" and
        "model_right_context" and values equal to the left/right model contexts
        for input_model.
        This function is useful when using the --trainer.input-model option
        instead of initializing the model using configs.
    """
    variables = {}
    try:
        out = common_lib.get_command_stdout("""nnet3-info {0} | """
                                            """head -4 """.format(input_model))
        # out looks like this
        # left-context: 7
        # right-context: 0
        # num-parameters: 90543902
        # modulus: 1
        for line in out.split("\n"):
            parts = line.split(":")
            if len(parts) != 2:
                continue
            if parts[0].strip() ==  'left-context':
                variables['model_left_context'] = int(parts[1].strip())
            elif parts[0].strip() ==  'right-context':
                variables['model_right_context'] = int(parts[1].strip())

    except ValueError:
        pass
    return variables


def verify_egs_dir(egs_dir, feat_dim, ivector_dim, ivector_extractor_id,
                   left_context, right_context,
                   left_context_initial=-1, right_context_final=-1):
    try:
        egs_feat_dim = int(open('{0}/info/feat_dim'.format(
                                    egs_dir)).readline())

        egs_ivector_id = None
        try:
            egs_ivector_id = open('{0}/info/final.ie.id'.format(
                                        egs_dir)).readline().strip()
            if (egs_ivector_id == ""):
                egs_ivector_id = None;
        except:
            # it could actually happen that the file is not there
            # for example in cases where the egs were dumped by
            # an older version of the script
            pass

        try:
            egs_ivector_dim = int(open('{0}/info/ivector_dim'.format(
                egs_dir)).readline())
        except:
            egs_ivector_dim = 0
        egs_left_context = int(open('{0}/info/left_context'.format(
                                    egs_dir)).readline())
        egs_right_context = int(open('{0}/info/right_context'.format(
                                    egs_dir)).readline())
        try:
            egs_left_context_initial = int(open('{0}/info/left_context_initial'.format(
                        egs_dir)).readline())
        except:  # older scripts didn't write this, treat it as -1 in that case.
            egs_left_context_initial = -1
        try:
            egs_right_context_final = int(open('{0}/info/right_context_final'.format(
                        egs_dir)).readline())
        except:  # older scripts didn't write this, treat it as -1 in that case.
            egs_right_context_final = -1

        # if feat_dim was supplied as 0, it means the --feat-dir option was not
        # supplied to the script, so we simply don't know what the feature dim is.
        if (feat_dim != 0 and feat_dim != egs_feat_dim) or (ivector_dim != egs_ivector_dim):
            raise Exception("There is mismatch between featdim/ivector_dim of "
                            "the current experiment and the provided "
                            "egs directory")

        if (((egs_ivector_id is None) and (ivector_extractor_id is not None)) or
            ((egs_ivector_id is not None) and (ivector_extractor_id is None))):
            logger.warning("The ivector ids are used inconsistently. It's your "
                          "responsibility to make sure the ivector extractor "
                          "has been used consistently")
            logger.warning("ivector id for egs: {0} in dir {1}".format(egs_ivector_id, egs_dir))
            logger.warning("ivector id for extractor: {0}".format(ivector_extractor_id))
        elif ((egs_ivector_dim > 0) and (egs_ivector_id is None) and (ivector_extractor_id is None)):
            logger.warning("The ivector ids are not used. It's your "
                          "responsibility to make sure the ivector extractor "
                          "has been used consistently")
        elif ivector_extractor_id != egs_ivector_id:
            raise Exception("The egs were generated using a different ivector "
                            "extractor. id1 = {0}, id2={1}".format(
                                ivector_extractor_id, egs_ivector_id));

        if (egs_left_context < left_context or
            egs_right_context < right_context):
            raise Exception('The egs have insufficient (l,r) context ({0},{1}) '
                            'versus expected ({2},{3})'.format(
                                egs_left_context, egs_right_context,
                                left_context, right_context))

        # the condition on the initial/final context is an equality condition,
        # not an inequality condition, as there is no mechanism to 'correct' the
        # context (by subtracting context) while copying the egs, like there is
        # for the regular left-right context.  If the user is determined to use
        # previously dumped egs, they may be able to slightly adjust the
        # --egs.chunk-left-context-initial and --egs.chunk-right-context-final
        # options to make things matched up.  [note: the model l/r context gets
        # added in, so you have to correct for changes in that.]
        if (egs_left_context_initial != left_context_initial or
            egs_right_context_final != right_context_final):
            raise Exception('The egs have incorrect initial/final (l,r) context '
                            '({0},{1}) versus expected ({2},{3}).  See code from '
                            'where this exception was raised for more info'.format(
                                egs_left_context_initial, egs_right_context_final,
                                left_context_initial, right_context_final))

        frames_per_eg_str = open('{0}/info/frames_per_eg'.format(
                             egs_dir)).readline().rstrip()
        if not validate_chunk_width(frames_per_eg_str):
            raise Exception("Invalid frames_per_eg in directory {0}/info".format(
                    egs_dir))
        num_archives = int(open('{0}/info/num_archives'.format(
                                    egs_dir)).readline())

        return [egs_left_context, egs_right_context,
                frames_per_eg_str, num_archives]
    except (IOError, ValueError):
        logger.error("The egs dir {0} has missing or "
                     "malformed files.".format(egs_dir))
        raise


def compute_presoftmax_prior_scale(dir, alidir, num_jobs, run_opts,
                                   presoftmax_prior_scale_power=-0.25):

    # getting the raw pdf count
    common_lib.execute_command(
        """{command} JOB=1:{num_jobs} {dir}/log/acc_pdf.JOB.log \
                ali-to-post "ark:gunzip -c {alidir}/ali.JOB.gz|" ark:- \| \
                post-to-tacc --per-pdf=true  {alidir}/final.mdl ark:- \
                {dir}/pdf_counts.JOB""".format(command=run_opts.command,
                                               num_jobs=num_jobs,
                                               dir=dir,
                                               alidir=alidir))

    common_lib.execute_command(
        """{command} {dir}/log/sum_pdf_counts.log \
                vector-sum --binary=false {dir}/pdf_counts.* {dir}/pdf_counts \
        """.format(command=run_opts.command, dir=dir))

    for file in glob.glob('{0}/pdf_counts.*'.format(dir)):
        os.remove(file)
    pdf_counts = common_lib.read_kaldi_matrix('{0}/pdf_counts'.format(dir))[0]
    scaled_counts = smooth_presoftmax_prior_scale_vector(
        pdf_counts,
        presoftmax_prior_scale_power=presoftmax_prior_scale_power,
        smooth=0.01)

    output_file = "{0}/presoftmax_prior_scale.vec".format(dir)
    common_lib.write_kaldi_matrix(output_file, [scaled_counts])
    common_lib.force_symlink("../presoftmax_prior_scale.vec",
                             "{0}/configs/presoftmax_prior_scale.vec".format(
                                dir))


def smooth_presoftmax_prior_scale_vector(pdf_counts,
                                         presoftmax_prior_scale_power=-0.25,
                                         smooth=0.01):
    total = sum(pdf_counts)
    average_count = float(total) / len(pdf_counts)
    scales = []
    for i in range(len(pdf_counts)):
        scales.append(math.pow(pdf_counts[i] + smooth * average_count,
                               presoftmax_prior_scale_power))
    num_pdfs = len(pdf_counts)
    scaled_counts = [x * float(num_pdfs) / sum(scales) for x in scales]
    return scaled_counts


def prepare_initial_network(dir, run_opts, srand=-3, input_model=None):
    if input_model is not None:
        shutil.copy(input_model, "{0}/0.raw".format(dir))
        return
    if os.path.exists(dir+"/configs/init.config"):
        common_lib.execute_command(
            """{command} {dir}/log/add_first_layer.log \
                    nnet3-init --srand={srand} {dir}/init.raw \
                    {dir}/configs/final.config {dir}/0.raw""".format(
                        command=run_opts.command, srand=srand,
                        dir=dir))
    else:
        common_lib.execute_command(
            """{command} {dir}/log/init_model.log \
           nnet3-init --srand={srand} {dir}/configs/final.config {dir}/0.raw""".format(
                        command=run_opts.command, srand=srand,
                        dir=dir))


def get_model_combine_iters(num_iters, num_epochs,
                      num_archives, max_models_combine,
                      num_jobs_final):
    """ Figures out the list of iterations for which we'll use those models
        in the final model-averaging phase.  (note: it's a weighted average
        where the weights are worked out from a subset of training data.)"""

    approx_iters_per_epoch_final = float(num_archives) / num_jobs_final
    # Note: it used to be that we would combine over an entire epoch,
    # but in practice we very rarely would use any weights from towards
    # the end of that range, so we are changing it to use not
    # approx_iters_per_epoch_final, but instead:
    # approx_iters_per_epoch_final/2 + 1,
    # dividing by 2 to use half an epoch, and adding 1 just to make sure
    # it's not zero.

    # First work out how many iterations we want to combine over in the final
    # nnet3-combine-fast invocation.
    # The number we use is:
    # min(max(max_models_combine, approx_iters_per_epoch_final/2+1),
    #     iters/2)
    # But if this value is > max_models_combine, then the models
    # are subsampled to get these many models to combine.

    num_iters_combine_initial = min(int(approx_iters_per_epoch_final/2) + 1,
                                    int(num_iters/2))

    if num_iters_combine_initial > max_models_combine:
        subsample_model_factor = int(
            float(num_iters_combine_initial) / max_models_combine)
        num_iters_combine = num_iters_combine_initial
        models_to_combine = set(range(
            num_iters - num_iters_combine_initial + 1,
            num_iters + 1, subsample_model_factor))
        models_to_combine.add(num_iters)
    else:
        subsample_model_factor = 1
        num_iters_combine = min(max_models_combine, num_iters//2)
        models_to_combine = set(range(num_iters - num_iters_combine + 1,
                                      num_iters + 1))

    return models_to_combine


def get_current_num_jobs(it, num_it, start, step, end):
    "Get number of jobs for iteration number 'it' of range('num_it')"

    ideal = float(start) + (end - start) * float(it) / num_it
    if ideal < step:
        return int(0.5 + ideal)
    else:
        return int(0.5 + ideal / step) * step


def get_learning_rate(iter, num_jobs, num_iters, num_archives_processed,
                      num_archives_to_process,
                      initial_effective_lrate, final_effective_lrate):
    if iter + 1 >= num_iters:
        effective_learning_rate = final_effective_lrate
    else:
        effective_learning_rate = (
                initial_effective_lrate
                * math.exp(num_archives_processed
                           * math.log(float(final_effective_lrate) / initial_effective_lrate)
                           / num_archives_to_process))

    return num_jobs * effective_learning_rate


def should_do_shrinkage(iter, model_file, shrink_saturation_threshold,
                        get_raw_nnet_from_am=True):

    if iter == 0:
        return True

    if get_raw_nnet_from_am:
        output = common_lib.get_command_stdout(
            "nnet3-am-info {0} 2>/dev/null | "
            "steps/nnet3/get_saturation.pl".format(model_file))
    else:
        output = common_lib.get_command_stdout(
            "nnet3-info 2>/dev/null {0} | "
            "steps/nnet3/get_saturation.pl".format(model_file))
    output = output.strip().split("\n")
    try:
        assert len(output) == 1
        saturation = float(output[0])
        assert saturation >= 0 and saturation <= 1
    except:
        raise Exception("Something went wrong, could not get "
                        "saturation from the output '{0}' of "
                        "get_saturation.pl on the info of "
                        "model {1}".format(output, model_file))
    return saturation > shrink_saturation_threshold


def remove_nnet_egs(egs_dir):
    common_lib.execute_command("steps/nnet2/remove_egs.sh {egs_dir}".format(
            egs_dir=egs_dir))


def clean_nnet_dir(nnet_dir, num_iters, egs_dir,
                   preserve_model_interval=100,
                   remove_egs=True,
                   get_raw_nnet_from_am=True):
    try:
        if remove_egs:
            remove_nnet_egs(egs_dir)

        for iter in range(num_iters):
            remove_model(nnet_dir, iter, num_iters, None,
                         preserve_model_interval,
                         get_raw_nnet_from_am=get_raw_nnet_from_am)
    except (IOError, OSError):
        logger.error("Error while cleaning up the nnet directory")
        raise


def remove_model(nnet_dir, iter, num_iters, models_to_combine=None,
                 preserve_model_interval=100,
                 get_raw_nnet_from_am=True):
    if iter % preserve_model_interval == 0:
        return
    if models_to_combine is not None and iter in models_to_combine:
        return
    if get_raw_nnet_from_am:
        file_name = '{0}/{1}.mdl'.format(nnet_dir, iter)
    else:
        file_name = '{0}/{1}.raw'.format(nnet_dir, iter)

    if os.path.isfile(file_name):
        os.remove(file_name)


def positive_int(arg):
   val = int(arg)
   if (val <= 0):
      raise argparse.ArgumentTypeError("must be positive int: '%s'" % arg)
   return val


class CommonParser(object):
    """Parser for parsing common options related to nnet3 training.

    This argument parser adds common options related to nnet3 training
    such as egs creation, training optimization options.
    These are used in the nnet3 train scripts
    in steps/nnet3/train*.py and steps/nnet3/chain/train.py
    """

    parser = argparse.ArgumentParser(add_help=False)

    def __init__(self,
                 include_chunk_context=True,
                 default_chunk_left_context=0):
        # feat options
        self.parser.add_argument("--feat.online-ivector-dir", type=str,
                                 dest='online_ivector_dir', default=None,
                                 action=common_lib.NullstrToNoneAction,
                                 help="""directory with the ivectors extracted
                                 in an online fashion.""")
        self.parser.add_argument("--feat.cmvn-opts", type=str,
                                 dest='cmvn_opts', default=None,
                                 action=common_lib.NullstrToNoneAction,
                                 help="A string specifying '--norm-means' "
                                 "and '--norm-vars' values")

        # egs extraction options.  there is no point adding the chunk context
        # option for non-RNNs (by which we mean basic TDNN-type topologies), as
        # it wouldn't affect anything, so we disable them if we know in advance
        # that we're not supporting RNN-type topologies (as in train_dnn.py).
        if include_chunk_context:
            self.parser.add_argument("--egs.chunk-left-context", type=int,
                                     dest='chunk_left_context',
                                     default=default_chunk_left_context,
                                     help="""Number of additional frames of input
                                 to the left of the input chunk. This extra
                                 context will be used in the estimation of RNN
                                 state before prediction of the first label. In
                                 the case of FF-DNN this extra context will be
                                 used to allow for frame-shifts""")
            self.parser.add_argument("--egs.chunk-right-context", type=int,
                                     dest='chunk_right_context', default=0,
                                     help="""Number of additional frames of input
                                     to the right of the input chunk. This extra
                                     context will be used in the estimation of
                                     bidirectional RNN state before prediction of
                                 the first label.""")
            self.parser.add_argument("--egs.chunk-left-context-initial", type=int,
                                     dest='chunk_left_context_initial', default=-1,
                                     help="""Number of additional frames of input
                                 to the left of the *first* input chunk extracted
                                 from an utterance.  If negative, defaults to
                                 the same as --egs.chunk-left-context""")
            self.parser.add_argument("--egs.chunk-right-context-final", type=int,
                                     dest='chunk_right_context_final', default=-1,
                                     help="""Number of additional frames of input
                                 to the right of the *last* input chunk extracted
                                 from an utterance.  If negative, defaults to the
                                 same as --egs.chunk-right-context""")
        self.parser.add_argument("--egs.dir", type=str, dest='egs_dir',
                                 default=None,
                                 action=common_lib.NullstrToNoneAction,
                                 help="""Directory with egs. If specified this
                                 directory will be used rather than extracting
                                 egs""")
        self.parser.add_argument("--egs.stage", type=int, dest='egs_stage',
                                 default=0,
                                 help="Stage at which get_egs.sh should be "
                                 "restarted")
        self.parser.add_argument("--egs.opts", type=str, dest='egs_opts',
                                 default=None,
                                 action=common_lib.NullstrToNoneAction,
                                 help="""String to provide options directly
                                 to steps/nnet3/get_egs.sh script""")

        # trainer options
        self.parser.add_argument("--trainer.srand", type=int, dest='srand',
                                 default=0,
                                 help="""Sets the random seed for model
                                 initialization and egs shuffling.
                                 Warning: This random seed does not control all
                                 aspects of this experiment.  There might be
                                 other random seeds used in other stages of the
                                 experiment like data preparation (e.g. volume
                                 perturbation).""")
        self.parser.add_argument("--trainer.num-epochs", type=float,
                                 dest='num_epochs', default=8.0,
                                 help="Number of epochs to train the model")
        self.parser.add_argument("--trainer.shuffle-buffer-size", type=int,
                                 dest='shuffle_buffer_size', default=5000,
                                 help=""" Controls randomization of the samples
                                 on each iteration. If 0 or a large value the
                                 randomization is complete, but this will
                                 consume memory and cause spikes in disk I/O.
                                 Smaller is easier on disk and memory but less
                                 random.  It's not a huge deal though, as
                                 samples are anyway randomized right at the
                                 start.  (the point of this is to get data in
                                 different minibatches on different iterations,
                                 since in the preconditioning method, 2 samples
                                 in the same minibatch can affect each others'
                                 gradients.""")
        self.parser.add_argument("--trainer.max-param-change", type=float,
                                 dest='max_param_change', default=2.0,
                                 help="""The maximum change in parameters
                                 allowed per minibatch, measured in Frobenius
                                 norm over the entire model""")
        self.parser.add_argument("--trainer.samples-per-iter", type=int,
                                 dest='samples_per_iter', default=400000,
                                 help="This is really the number of egs in "
                                 "each archive.")
        self.parser.add_argument("--trainer.lda.rand-prune", type=float,
                                 dest='rand_prune', default=4.0,
                                 help="Value used in preconditioning "
                                 "matrix estimation")
        self.parser.add_argument("--trainer.lda.max-lda-jobs", type=int,
                                 dest='max_lda_jobs', default=10,
                                 help="Max number of jobs used for "
                                 "LDA stats accumulation")
        self.parser.add_argument("--trainer.presoftmax-prior-scale-power",
                                 type=float,
                                 dest='presoftmax_prior_scale_power',
                                 default=-0.25,
                                 help="Scale on presofmax prior")
        self.parser.add_argument("--trainer.optimization.proportional-shrink", type=float,
                                 dest='proportional_shrink', default=0.0,
                                 help="""If nonzero, this will set a shrinkage (scaling)
                        factor for the parameters, whose value is set as:
                        shrink-value=(1.0 - proportional-shrink * learning-rate), where
                        'learning-rate' is the learning rate being applied
                        on the current iteration, which will vary from
                        initial-effective-lrate*num-jobs-initial to
                        final-effective-lrate*num-jobs-final.
                        Unlike for train_rnn.py, this is applied unconditionally,
                        it does not depend on saturation of nonlinearities.
                        Can be used to roughly approximate l2 regularization.""")

        # Parameters for the optimization
        self.parser.add_argument(
            "--trainer.optimization.initial-effective-lrate", type=float,
            dest='initial_effective_lrate', default=0.0003,
            help="Learning rate used during the initial iteration")
        self.parser.add_argument(
            "--trainer.optimization.final-effective-lrate", type=float,
            dest='final_effective_lrate', default=0.00003,
            help="Learning rate used during the final iteration")
        self.parser.add_argument("--trainer.optimization.num-jobs-initial",
                                 type=int, dest='num_jobs_initial', default=1,
                                 help="Number of neural net jobs to run in "
                                 "parallel at the start of training")
        self.parser.add_argument("--trainer.optimization.num-jobs-final",
                                 type=int, dest='num_jobs_final', default=8,
                                 help="Number of neural net jobs to run in "
                                 "parallel at the end of training")
        self.parser.add_argument("--trainer.optimization.num-jobs-step",
            type=positive_int,  metavar='N', dest='num_jobs_step', default=1,
            help="""Number of jobs increment, when exceeds this number. For
            example, if N=3, the number of jobs may progress as 1, 2, 3, 6, 9...""")
        self.parser.add_argument("--trainer.optimization.max-models-combine",
                                 "--trainer.max-models-combine",
                                 type=int, dest='max_models_combine',
                                 default=20,
                                 help="""The maximum number of models used in
                                 the final model combination stage.  These
                                 models will themselves be averages of
                                 iteration-number ranges""")
        self.parser.add_argument("--trainer.optimization.max-objective-evaluations",
                                 "--trainer.max-objective-evaluations",
                                 type=int, dest='max_objective_evaluations',
                                 default=30,
                                 help="""The maximum number of objective
                                 evaluations in order to figure out the
                                 best number of models to combine. It helps to
                                 speedup if the number of models provided to the
                                 model combination binary is quite large (e.g.
                                 several hundred).""")
        self.parser.add_argument("--trainer.optimization.do-final-combination",
                                 dest='do_final_combination', type=str,
                                 action=common_lib.StrToBoolAction,
                                 choices=["true", "false"], default=True,
                                 help="""Set this to false to disable the final
                                 'combine' stage (in this case we just use the
                                 last-numbered model as the final.mdl).""")
        self.parser.add_argument("--trainer.optimization.combine-sum-to-one-penalty",
                                 type=float, dest='combine_sum_to_one_penalty', default=0.0,
                                 help="""This option is deprecated and does nothing.""")
        self.parser.add_argument("--trainer.optimization.momentum", type=float,
                                 dest='momentum', default=0.0,
                                 help="""Momentum used in update computation.
                                 Note: we implemented it in such a way that it
                                 doesn't increase the effective learning
                                 rate.""")
        self.parser.add_argument("--trainer.dropout-schedule", type=str,
                                 action=common_lib.NullstrToNoneAction,
                                 dest='dropout_schedule', default=None,
                                 help="""Use this to specify the dropout
                                 schedule.  You specify a piecewise linear
                                 function on the domain [0,1], where 0 is the
                                 start and 1 is the end of training; the
                                 function-argument (x) rises linearly with the
                                 amount of data you have seen, not iteration
                                 number (this improves invariance to
                                 num-jobs-{initial-final}).  E.g. '0,0.2,0'
                                 means 0 at the start; 0.2 after seeing half
                                 the data; and 0 at the end.  You may specify
                                 the x-value of selected points, e.g.
                                 '0,0.2@0.25,0' means that the 0.2
                                 dropout-proportion is reached a quarter of the
                                 way through the data.   The start/end x-values
                                 are at x=0/x=1, and other unspecified x-values
                                 are interpolated between known x-values.  You
                                 may specify different rules for different
                                 component-name patterns using 'pattern1=func1
                                 pattern2=func2', e.g. 'relu*=0,0.1,0
                                 lstm*=0,0.2,0'.  More general should precede
                                 less general patterns, as they are applied
                                 sequentially.""")
        self.parser.add_argument("--trainer.add-option", type=str,
                                 dest='train_opts', action='append', default=[],
                                 help="""You can use this to add arbitrary options that
                                 will be passed through to the core training code (nnet3-train
                                 or nnet3-chain-train)""")
        self.parser.add_argument("--trainer.optimization.backstitch-training-scale",
                                 type=float, dest='backstitch_training_scale',
                                 default=0.0, help="""scale of parameters changes
                                 used in backstitch training step.""")
        self.parser.add_argument("--trainer.optimization.backstitch-training-interval",
                                 type=int, dest='backstitch_training_interval',
                                 default=1, help="""the interval of minibatches
                                 that backstitch training is applied on.""")
        self.parser.add_argument("--trainer.compute-per-dim-accuracy",
                                 dest='compute_per_dim_accuracy',
                                 type=str, choices=['true', 'false'],
                                 default=False,
                                 action=common_lib.StrToBoolAction,
                                 help="Compute train and validation "
                                 "accuracy per-dim")

        # General options
        self.parser.add_argument("--stage", type=int, default=-4,
                                 help="Specifies the stage of the experiment "
                                 "to execution from")
        self.parser.add_argument("--exit-stage", type=int, default=None,
                                 help="If specified, training exits before "
                                 "running this stage")
        self.parser.add_argument("--cmd", type=str, dest="command",
                                 action=common_lib.NullstrToNoneAction,
                                 help="""Specifies the script to launch jobs.
                                 e.g. queue.pl for launching on SGE cluster
                                        run.pl for launching on local machine
                                 """, default="queue.pl")
        self.parser.add_argument("--egs.cmd", type=str, dest="egs_command",
                                 action=common_lib.NullstrToNoneAction,
                                 help="Script to launch egs jobs")
        self.parser.add_argument("--use-gpu", type=str,
                                 choices=["true", "false", "yes", "no", "wait"],
                                 help="Use GPU for training. "
                                 "Note 'true' and 'false' are deprecated.",
                                 default="yes")
        self.parser.add_argument("--cleanup", type=str,
                                 action=common_lib.StrToBoolAction,
                                 choices=["true", "false"], default=True,
                                 help="Clean up models after training")
        self.parser.add_argument("--cleanup.remove-egs", type=str,
                                 dest='remove_egs', default=True,
                                 action=common_lib.StrToBoolAction,
                                 choices=["true", "false"],
                                 help="If true, remove egs after experiment")
        self.parser.add_argument("--cleanup.preserve-model-interval",
                                 dest="preserve_model_interval",
                                 type=int, default=100,
                                 help="""Determines iterations for which models
                                 will be preserved during cleanup.
                                 If mod(iter,preserve_model_interval) == 0
                                 model will be preserved.""")

        self.parser.add_argument("--reporting.email", dest="email",
                                 type=str, default=None,
                                 action=common_lib.NullstrToNoneAction,
                                 help=""" Email-id to report about the progress
                                 of the experiment.  NOTE: It assumes the
                                 machine on which the script is being run can
                                 send emails from command line via. mail
                                 program. The Kaldi mailing list will not
                                 support this feature.  It might require local
                                 expertise to setup. """)
        self.parser.add_argument("--reporting.interval",
                                 dest="reporting_interval",
                                 type=float, default=0.1,
                                 help="""Frequency with which reports have to
                                 be sent, measured in terms of fraction of
                                 iterations.
                                 If 0 and reporting mail has been specified
                                 then only failure notifications are sent""")


import unittest

class SelfTest(unittest.TestCase):

    def test_halve_minibatch_size_str(self):
        self.assertEqual('32', halve_minibatch_size_str('64'))
        self.assertEqual('32,8:16', halve_minibatch_size_str('64,16:32'))
        self.assertEqual('1', halve_minibatch_size_str('1'))
        self.assertEqual('128=32/256=20,40:50', halve_minibatch_size_str('128=64/256=40,80:100'))


    def test_validate_chunk_width(self):
        for s in [ '64', '64,25,128' ]:
            self.assertTrue(validate_chunk_width(s), s)


    def test_validate_minibatch_size_str(self):
        # Good descriptors.
        for s in [ '32', '32,64', '1:32', '1:32,64', '64,1:32', '1:5,10:15',
                   '128=64:128/256=32,64', '1=2/3=4', '1=1/2=2/3=3/4=4' ]:
            self.assertTrue(validate_minibatch_size_str(s), s)
        # Bad descriptors.
        for s in [ None, 42, (43,), '', '1:', ':2', '3,', ',4', '5:6,', ',7:8',
                   '9=', '10=10/', '11=11/11', '12=1:2//13=1:3' '14=/15=15',
                   '16/17=17', '/18=18', '/18', '//19', '/' ]:
            self.assertFalse(validate_minibatch_size_str(s), s)


    def test_get_current_num_jobs(self):
        niters = 12
        self.assertEqual([2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8],
                         [get_current_num_jobs(i, niters, 2, 1, 9)
                              for i in range(niters)])
        self.assertEqual([2, 3, 3, 3, 3, 6, 6, 6, 6, 6, 9, 9],
                         [get_current_num_jobs(i, niters, 2, 3, 9)
                              for i in range(niters)])


if __name__ == '__main__':
    unittest.main()
