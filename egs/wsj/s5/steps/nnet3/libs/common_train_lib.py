

# Copyright 2016 Vijayaditya Peddinti.
#           2016 Vimal Manohar
# Apache 2.0.

import subprocess
import logging
import math
import re
import time
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# a class to store run options
class RunOpts:
    def __init__(self):
        self.command = None
        self.train_queue_opt = None
        self.combine_queue_opt = None
        self.prior_gpu_opt = None
        self.prior_queue_opt = None
        self.parallel_train_opts = None

def AddCommonTrainArgs(parser):
    # feat options
    parser.add_argument("--feat.online-ivector-dir", type=str, dest='online_ivector_dir',
                        default = None, action = common_train_lib.NullstrToNoneAction,
                        help="""directory with the ivectors extracted in
                        an online fashion.""")
    parser.add_argument("--feat.cmvn-opts", type=str, dest='cmvn_opts',
                        default = None, action = common_train_lib.NullstrToNoneAction,
                        help="A string specifying '--norm-means' and '--norm-vars' values")

    # egs extraction options
    parser.add_argument("--egs.chunk-left-context", type=int, dest='chunk_left_context',
                        default = 0,
                        help="Number of additional frames of input to the left"
                        " of the input chunk. This extra context will be used"
                        " in the estimation of RNN state before prediction of"
                        " the first label. In the case of FF-DNN this extra"
                        " context will be used to allow for frame-shifts")
    parser.add_argument("--egs.chunk-right-context", type=int, dest='chunk_right_context',
                        default = 0,
                        help="Number of additional frames of input to the right"
                        " of the input chunk. This extra context will be used"
                        " in the estimation of bidirectional RNN state before"
                        " prediction of the first label.")
    parser.add_argument("--egs.transform_dir", type=str, dest='transform_dir',
                        default = None, action = common_train_lib.NullstrToNoneAction,
                        help="""String to provide options directly to steps/nnet3/get_egs.sh script""")
    parser.add_argument("--egs.dir", type=str, dest='egs_dir',
                        default = None, action = common_train_lib.NullstrToNoneAction,
                        help="""Directory with egs. If specified this directory
                        will be used rather than extracting egs""")
    parser.add_argument("--egs.stage", type=int, dest='egs_stage',
                        default = 0, help="Stage at which get_egs.sh should be restarted")
    parser.add_argument("--egs.opts", type=str, dest='egs_opts',
                        default = None, action = common_train_lib.NullstrToNoneAction,
                        help="""String to provide options directly to steps/nnet3/get_egs.sh script""")

    # trainer options
    parser.add_argument("--trainer.srand", type=int, dest='srand',
                        default = 0,
                        help="Sets the random seed for model initialization and egs shuffling. "
                        "Warning: This random seed does not control all aspects of this experiment. "
                        "There might be other random seeds used in other stages of the experiment "
                        "like data preparation (e.g. volume perturbation).")
    parser.add_argument("--trainer.num-epochs", type=int, dest='num_epochs',
                        default = 8,
                        help="Number of epochs to train the model")
    parser.add_argument("--trainer.prior-subset-size", type=int, dest='prior_subset_size',
                        default = 20000,
                        help="Number of samples for computing priors")
    parser.add_argument("--trainer.num-jobs-compute-prior", type=int, dest='num_jobs_compute_prior',
                        default = 10,
                        help="The prior computation jobs are single threaded and run on the CPU")
    parser.add_argument("--trainer.max-models-combine", type=int, dest='max_models_combine',
                        default = 20,
                        help="The maximum number of models used in the final model combination stage. These models will themselves be averages of iteration-number ranges")
    parser.add_argument("--trainer.shuffle-buffer-size", type=int, dest='shuffle_buffer_size',
                        default = 5000,
                        help=""" Controls randomization of the samples on each
                        iteration. If 0 or a large value the randomization is
                        complete, but this will consume memory and cause spikes
                        in disk I/O.  Smaller is easier on disk and memory but
                        less random.  It's not a huge deal though, as samples
                        are anyway randomized right at the start.
                        (the point of this is to get data in different
                        minibatches on different iterations, since in the
                        preconditioning method, 2 samples in the same minibatch
                        can affect each others' gradients.""")
    parser.add_argument("--trainer.add-layers-period", type=int, dest='add_layers_period',
                        default=2,
                        help="The number of iterations between adding layers"
                        "during layer-wise discriminative training.")
    parser.add_argument("--trainer.max-param-change", type=float, dest='max_param_change',
                        default=2.0,
                        help="""The maximum change in parameters allowed
                        per minibatch, measured in Frobenius norm over
                        the entire model""")
    parser.add_argument("--trainer.samples-per-iter", type=int, dest='samples_per_iter',
                        default=400000,
                        help="This is really the number of egs in each archive.")
    parser.add_argument("--trainer.lda.rand-prune", type=float, dest='rand_prune',
                        default=4.0,
                        help="""Value used in preconditioning matrix estimation""")
    parser.add_argument("--trainer.lda.max-lda-jobs", type=float, dest='max_lda_jobs',
                        default=10,
                        help="""Max number of jobs used for LDA stats accumulation""")

    # Parameters for the optimization
    parser.add_argument("--trainer.optimization.initial-effective-lrate", type=float, dest='initial_effective_lrate',
                        default = 0.0003,
                        help="Learning rate used during the initial iteration")
    parser.add_argument("--trainer.optimization.final-effective-lrate", type=float, dest='final_effective_lrate',
                        default = 0.00003,
                        help="Learning rate used during the final iteration")
    parser.add_argument("--trainer.optimization.num-jobs-initial", type=int, dest='num_jobs_initial',
                        default = 1,
                        help="Number of neural net jobs to run in parallel at the start of training")
    parser.add_argument("--trainer.optimization.num-jobs-final", type=int, dest='num_jobs_final',
                        default = 8,
                        help="Number of neural net jobs to run in parallel at the end of training")
    parser.add_argument("--trainer.optimization.max-models-combine", type=int, dest='max_models_combine',
                        default = 20,
                        help = """ The is the maximum number of models we give to the
                                   final 'combine' stage, but these models will themselves
                                   be averages of iteration-number ranges. """)
    parser.add_argument("--trainer.optimization.momentum", type=float, dest='momentum',
                        default = 0.0,
                        help="""Momentum used in update computation.
                        Note: we implemented it in such a way that
                        it doesn't increase the effective learning rate.""")
    # General options
    parser.add_argument("--stage", type=int, default=-4,
                        help="Specifies the stage of the experiment to execution from")
    parser.add_argument("--exit-stage", type=int, default=None,
                        help="If specified, training exits before running this stage")
    parser.add_argument("--cmd", type=str, action = common_train_lib.NullstrToNoneAction,
                        dest = "command",
                        help="""Specifies the script to launch jobs.
                        e.g. queue.pl for launching on SGE cluster
                             run.pl for launching on local machine
                        """, default = "queue.pl")
    parser.add_argument("--egs.cmd", type=str, action = common_train_lib.NullstrToNoneAction,
                        dest = "egs_command",
                        help="""Script to launch egs jobs""", default = "queue.pl")
    parser.add_argument("--use-gpu", type=str, action = common_train_lib.StrToBoolAction,
                        choices = ["true", "false"],
                        help="Use GPU for training", default=True)
    parser.add_argument("--cleanup", type=str, action = common_train_lib.StrToBoolAction,
                        choices = ["true", "false"],
                        help="Clean up models after training", default=True)
    parser.add_argument("--cleanup.remove-egs", type=str, dest='remove_egs',
                        default = True, action = common_train_lib.StrToBoolAction,
                        choices = ["true", "false"],
                        help="""If true, remove egs after experiment""")
    parser.add_argument("--cleanup.preserve-model-interval", dest = "preserve_model_interval",
                        type=int, default=100,
                        help="Determines iterations for which models will be preserved during cleanup. If mod(iter,preserve_model_interval) == 0 model will be preserved.")

    parser.add_argument("--reporting.email", dest = "email",
                        type=str, default=None, action = common_train_lib.NullstrToNoneAction,
                        help=""" Email-id to report about the progress of the experiment.
                              NOTE: It assumes the machine on which the script is being run can send
                              emails from command line via. mail program. The
                              Kaldi mailing list will not support this feature.
                              It might require local expertise to setup. """)
    parser.add_argument("--reporting.interval", dest = "reporting_interval",
                        type=int, default=0.1,
                        help="Frequency with which reports have to be sent, measured in terms of fraction of iterations. If 0 and reporting mail has been specified then only failure notifications are sent")

def SendMail(message, subject, email_id):
    try:
        subprocess.Popen('echo "{message}" | mail -s "{subject}" {email} '.format(
            message = message,
            subject = subject,
            email = email_id), shell=True)
    except Exception as e:
        logger.info(" Unable to send mail due to error:\n {error}".format(error = str(e)))
        pass

def StrToBool(values):
    if values == "true":
        return True
    elif values == "false":
        return False
    else:
        raise ValueError

class StrToBoolAction(argparse.Action):
    """ A custom action to convert bools from shell format i.e., true/false
        to python format i.e., True/False """
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, StrToBool(values))
        except ValueError:
            raise Exception("Unknown value {0} for --{1}".format(values, self.dest))

class NullstrToNoneAction(argparse.Action):
    """ A custom action to convert empty strings passed by shell
        to None in python. This is necessary as shell scripts print null strings
        when a variable is not specified. We could use the more apt None
        in python. """
    def __call__(self, parser, namespace, values, option_string=None):
            if values.strip() == "":
                setattr(namespace, self.dest, None)
            else:
                setattr(namespace, self.dest, values)


def CheckIfCudaCompiled():
    p = subprocess.Popen("cuda-compiled")
    p.communicate()
    if p.returncode == 1:
        return False
    else:
        return True


class KaldiCommandException(Exception):
    def __init__(self, command, err):
        Exception.__init__(self,"There was an error while running the command {0}\n".format(command)+"-"*10+"\n"+err)

def RunKaldiCommand(command, wait = True):
    """ Runs commands frequently seen in Kaldi scripts. These are usually a
        sequence of commands connected by pipes, so we use shell=True """
    #logger.info("Running the command\n{0}".format(command))
    p = subprocess.Popen(command, shell = True,
                         stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE)

    if wait:
        [stdout, stderr] = p.communicate()
        if p.returncode is not 0:
            raise KaldiCommandException(command, stderr)
        return stdout, stderr
    else:
        return p

def GetSuccessfulModels(num_models, log_file_pattern, difference_threshold=1.0):
    assert(num_models > 0)

    parse_regex = re.compile("LOG .* Overall average objective function for 'output' is ([0-9e.\-+]+) over ([0-9e.\-+]+) frames")
    objf = []
    for i in range(num_models):
        model_num = i + 1
        logfile = re.sub('%', str(model_num), log_file_pattern)
        lines = open(logfile, 'r').readlines()
        this_objf = -100000
        for line_num in range(1, len(lines) + 1):
            # we search from the end as this would result in
            # lesser number of regex searches. Python regex is slow !
            mat_obj = parse_regex.search(lines[-1*line_num])
            if mat_obj is not None:
                this_objf = float(mat_obj.groups()[0])
                break;
        objf.append(this_objf);
    max_index = objf.index(max(objf))
    accepted_models = []
    for i in range(num_models):
        if (objf[max_index] - objf[i]) <= difference_threshold:
            accepted_models.append(i+1)

    if len(accepted_models) != num_models:
        logger.warn("""Only {0}/{1} of the models have been accepted
for averaging, based on log files {2}.""".format(len(accepted_models),
                                                 num_models, log_file_pattern))

    return [accepted_models, max_index+1]

def GetAverageNnetModel(dir, iter, nnets_list, run_opts,
                        get_raw_nnet_from_am = True, shrink = None):
    scale = 1.0
    if shrink is not None:
        scale = shrink

    new_iter = iter + 1
    if get_raw_nnet_from_am:
        out_model = """- \| nnet3-am-copy --set-raw-nnet=- --scale={scale} \
{dir}/{iter}.mdl {dir}/{new_iter}.mdl""".format(dir = dir, iter = iter,
                                                new_iter = new_iter,
                                                scale = scale)
    else:
        if shrink is not None:
            out_model = """- \| nnet3-copy --scale={scale} \
- {dir}/{new_iter}.raw""".format(dir = dir, new_iter = new_iter, scale = scale)
        else:
            out_model = "{dir}/{new_iter}.raw".format(dir = dir,
                                                      new_iter = new_iter)

    RunKaldiCommand("""
{command} {dir}/log/average.{iter}.log \
nnet3-average {nnets_list} \
{out_model}""".format(command = run_opts.command,
               dir = dir,
               iter = iter,
               nnets_list = nnets_list,
               out_model = out_model))

def GetBestNnetModel(dir, iter, best_model_index, run_opts,
                     get_raw_nnet_from_am = True, shrink = None):
    scale = 1.0
    if shrink is not None:
        scale = shrink

    best_model = '{dir}/{next_iter}.{best_model_index}.raw'.format(
            dir = dir,
            next_iter = iter + 1,
            best_model_index = best_model_index)

    if get_raw_nnet_from_am:
        out_model = """- \| nnet3-am-copy --set-raw-nnet=- \
{dir}/{iter}.mdl {dir}/{next_iter}.mdl""".format(dir = dir, iter = iter,
                                                 new_iter = iter + 1)
    else:
        out_model = '{dir}/{next_iter}.raw'.format(dir = dir,
                                                   next_iter = iter + 1)

    RunKaldiCommand("""
{command} {dir}/log/select.{iter}.log \
nnet3-copy --scale={scale} {best_model} \
{out_model}""".format(command = run_opts.command,
               dir = dir, iter = iter,
               best_model =  best_model,
               out_model = out_model, scale = scale))

def GetNumberOfLeaves(alidir):
    [stdout, stderr] = RunKaldiCommand("tree-info {0}/tree 2>/dev/null | grep num-pdfs".format(alidir))
    parts = stdout.split()
    assert(parts[0] == "num-pdfs")
    num_leaves = int(parts[1])
    if num_leaves == 0:
        raise Exception("Number of leaves is 0")
    return num_leaves

def GetNumberOfJobs(alidir):
    try:
        num_jobs = int(open('{0}/num_jobs'.format(alidir), 'r').readline().strip())
    except IOError, ValueError:
        raise Exception('Exception while reading the number of alignment jobs')
    return num_jobs

def GetIvectorDim(ivector_dir = None):
    if ivector_dir is None:
        return 0
    [stdout_val, stderr_val] = RunKaldiCommand("feat-to-dim --print-args=false scp:{dir}/ivector_online.scp -".format(dir = ivector_dir))
    ivector_dim = int(stdout_val)
    return ivector_dim

def GetFeatDim(feat_dir):
    [stdout_val, stderr_val] = RunKaldiCommand("feat-to-dim --print-args=false scp:{data}/feats.scp -".format(data = feat_dir))
    feat_dim = int(stdout_val)
    return feat_dim

def GetFeatDimFromScp(feat_scp):
    [stdout_val, stderr_val] =  RunKaldiCommand("feat-to-dim --print-args=false scp:{feat_scp} -".format(feat_scp = feat_scp))
    feat_dim = int(stdout_val)
    return feat_dim

def ReadKaldiMatrix(matrix_file):
    try:
        lines = map(lambda x: x.split(), open(matrix_file).readlines())
        first_field = lines[0][0]
        last_field = lines[-1][-1]
        lines[0] = lines[0][1:]
        lines[-1] = lines[-1][:-1]
        if not (first_field == "[" and last_field == "]"):
            raise Exception("Kaldi matrix file has incorrect format, only text format matrix files can be read by this script")
        for i in range(len(lines)):
            lines[i] = map(lambda x: int(float(x)), lines[i])
        return lines
    except IOError:
        raise Exception("Error while reading the kaldi matrix file {0}".format(matrix_file))

def WriteKaldiMatrix(output_file, matrix):
    # matrix is a list of lists
    file = open(output_file, 'w')
    file.write("[ ")
    num_rows = len(matrix)
    if num_rows == 0:
        raise Exception("Matrix is empty")
    num_cols = len(matrix[0])

    for row_index in range(len(matrix)):
        if num_cols != len(matrix[row_index]):
            raise Exception("All the rows of a matrix are expected to have the same length")
        file.write(" ".join(map(lambda x: str(x), matrix[row_index])))
        if row_index != num_rows - 1:
            file.write("\n")
    file.write(" ]")
    file.close()

import shutil
def CopyEgsPropertiesToExpDir(egs_dir, dir):
    try:
        for file in ['cmvn_opts', 'splice_opts', 'final.mat']:
            file_name = '{dir}/{file}'.format(dir = egs_dir, file = file)
            if os.path.isfile(file_name):
                shutil.copy2(file_name, dir)
    except IOError:
        raise Exception("Error while trying to copy egs property files to {dir}".format(dir = dir))

def SplitData(data, num_jobs):
   RunKaldiCommand("utils/split_data.sh {data} {num_jobs}".format(data = data,
                                                                  num_jobs = num_jobs))

def ParseModelConfigVarsFile(var_file):
    try:
        var_file_handle = open(var_file, 'r')
        model_left_context = None
        model_right_context = None
        num_hidden_layers = None
        for line in var_file_handle:
            parts = line.split('=')
            field_name = parts[0].strip()
            field_value = parts[1]
            if field_name in ['model_left_context', 'left_context']:
                model_left_context = int(field_value)
            elif field_name in ['model_right_context', 'right_context']:
                model_right_context = int(field_value)
            elif field_name == 'num_hidden_layers':
                num_hidden_layers = int(field_value)

        if model_left_context is not None and model_right_context is not None and num_hidden_layers is not None:
            return [model_left_context, model_right_context, num_hidden_layers]

    except ValueError:
        # we will throw an error at the end of the function so I will just pass
        pass

    raise Exception('Error while parsing the file {0}'.format(var_file))

def ParseGenericConfigVarsFile(var_file):
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
                variables['num_hidden_layers'] = int(field_value)
            else:
                variables[field_name] = field_value
        return variables
    except ValueError:
        # we will throw an error at the end of the function so I will just pass
        pass

    raise Exception('Error while parsing the file {0}'.format(var_file))

def VerifyEgsDir(egs_dir, feat_dim, ivector_dim, left_context, right_context):
    try:
        egs_feat_dim = int(open('{0}/info/feat_dim'.format(egs_dir)).readline())
        egs_ivector_dim = int(open('{0}/info/ivector_dim'.format(egs_dir)).readline())
        egs_left_context = int(open('{0}/info/left_context'.format(egs_dir)).readline())
        egs_right_context = int(open('{0}/info/right_context'.format(egs_dir)).readline())
        if (feat_dim != egs_feat_dim) or (ivector_dim != egs_ivector_dim):
            raise Exception('There is mismatch between featdim/ivector_dim of the current experiment and the provided egs directory')

        if (egs_left_context < left_context) or (egs_right_context < right_context):
            raise Exception('The egs have insufficient context')

        frames_per_eg = int(open('{0}/info/frames_per_eg'.format(egs_dir)).readline())
        num_archives = int(open('{0}/info/num_archives'.format(egs_dir)).readline())

        return [egs_left_context, egs_right_context, frames_per_eg, num_archives]
    except IOError, ValueError:
        raise Exception('The egs dir {0} has missing or malformed files'.format(egs_dir))

import os, errno

def ForceSymlink(file1, file2):
    try:
        os.symlink(file1, file2)
    except OSError, e:
        if e.errno == errno.EEXIST:
            os.remove(file2)
            os.symlink(file1, file2)

def ComputePresoftmaxPriorScale(dir, alidir, num_jobs, run_opts,
                                presoftmax_prior_scale_power = -0.25):

    # getting the raw pdf count
    RunKaldiCommand("""
{command} JOB=1:{num_jobs} {dir}/log/acc_pdf.JOB.log \
ali-to-post "ark:gunzip -c {alidir}/ali.JOB.gz|" ark:- \| \
post-to-tacc --per-pdf=true  {alidir}/final.mdl ark:- {dir}/pdf_counts.JOB
     """.format(command = run_opts.command,
                num_jobs = num_jobs,
                dir = dir,
                alidir = alidir))

    RunKaldiCommand("""
{command} {dir}/log/sum_pdf_counts.log \
vector-sum --binary=false {dir}/pdf_counts.* {dir}/pdf_counts
       """.format(command = run_opts.command,  dir = dir))

    import glob
    for file in glob.glob('{0}/pdf_counts.*'.format(dir)):
        os.remove(file)
    pdf_counts = ReadKaldiMatrix('{0}/pdf_counts'.format(dir))[0]
    scaled_counts = SmoothPresoftmaxPriorScaleVector(pdf_counts, presoftmax_prior_scale_power = presoftmax_prior_scale_power, smooth = 0.01)

    output_file = "{0}/presoftmax_prior_scale.vec".format(dir)
    WriteKaldiMatrix(output_file, [scaled_counts])
    ForceSymlink("../presoftmax_prior_scale.vec", "{0}/configs/presoftmax_prior_scale.vec".format(dir))

def SmoothPresoftmaxPriorScaleVector(pdf_counts, presoftmax_prior_scale_power = -0.25, smooth = 0.01):
    total = sum(pdf_counts)
    average_count = total/len(pdf_counts)
    scales = []
    for i in range(len(pdf_counts)):
        scales.append(math.pow(pdf_counts[i] + smooth * average_count, presoftmax_prior_scale_power))
    num_pdfs = len(pdf_counts)
    scaled_counts = map(lambda x: x * float(num_pdfs) / sum(scales), scales)
    return scaled_counts


def PrepareInitialNetwork(dir, run_opts):
    RunKaldiCommand("""
{command} {dir}/log/add_first_layer.log \
   nnet3-init --srand=-3 {dir}/init.raw {dir}/configs/layer1.config {dir}/0.raw     """.format(command = run_opts.command,
               dir = dir))

def VerifyIterations(num_iters, num_epochs, num_hidden_layers,
                     num_archives, max_models_combine, add_layers_period,
                     num_jobs_final):
    """ Verifies that number of iterations are sufficient for various
        phases of training."""

    finish_add_layers_iter = num_hidden_layers * add_layers_period

    if num_iters <= (finish_add_layers_iter + 2):
        raise Exception(' There are insufficient number of epochs. These are not even sufficient for layer-wise discriminatory training.')


    approx_iters_per_epoch_final = num_archives/num_jobs_final
    # First work out how many iterations we want to combine over in the final
    # nnet3-combine-fast invocation.  (We may end up subsampling from these if the
    # number exceeds max_model_combine).  The number we use is:
    # min(max(max_models_combine, approx_iters_per_epoch_final),
    #     1/2 * iters_after_last_layer_added)
    half_iters_after_add_layers = (num_iters - finish_add_layers_iter)/2
    num_iters_combine = min(max(max_models_combine, approx_iters_per_epoch_final), half_iters_after_add_layers)
    return num_iters_combine

def GetRealignIters(realign_times, num_iters,
                    num_jobs_initial, num_jobs_final):
    """ Takes the realign_times string and identifies the approximate
        iterations at which realignments have to be done."""
    # realign_times is a space seperated string of values between 0 and 1

    realign_iters = []
    for realign_time in realign_times.split():
        realign_time = float(realign_time)
        assert(realign_time > 0 and realign_time < 1)
        if num_jobs_initial == num_jobs_final:
            realign_iter = int(0.5 + num_iters * realign_time)
        else:
            realign_iter = math.sqrt((1 - realign_time) * math.pow(num_jobs_initial, 2)
                            + realign_time * math.pow(num_jobs_final, 2))
            realign_iter = realign_iter - num_jobs_initial
            realign_iter = realign_iter / (num_jobs_final - num_jobs_initial)
            realign_iter = realign_iter * num_iters
        realign_iters.append(int(realign_iter))

    return realign_iters

def Align(dir, data, lang, run_opts, iter = None, transform_dir = None,
          online_ivector_dir = None):

    alidir = '{dir}/ali{ali_suffix}'.format(dir = dir,
               ali_suffix = "_iter_{0}".format(iter) if iter is not None else "")

    logger.info("Aligning the data{gpu}with {num_jobs} jobs.".format(
        gpu = " using gpu " if run_opts.realign_use_gpu else " ",
        num_jobs = run_opts.realign_num_jobs ))
    RunKaldiCommand("""
steps/nnet3/align.sh --nj {num_jobs_align} --cmd "{align_cmd} {align_queue_opt}" \
        --use-gpu {align_use_gpu} \
        --transform-dir "{transform_dir}" \
        --online-ivector-dir "{online_ivector_dir}" \
        --iter "{iter}" {data} {lang} {dir} {alidir}
    """.format(dir = dir, align_use_gpu = "yes" if run_opts.realign_use_gpu else "no",
               align_cmd = run_opts.realign_command,
               align_queue_opt = run_opts.realign_queue_opt,
               num_jobs_align = run_opts.realign_num_jobs,
               transform_dir = transform_dir if transform_dir is not None else "",
               online_ivector_dir = online_ivector_dir if online_ivector_dir is not None else "",
               iter = iter if iter is not None else "",
               alidir = alidir,
               lang = lang, data = data))
    return alidir

def Realign(dir, iter, feat_dir, lang, prev_egs_dir, cur_egs_dir,
            prior_subset_size, num_archives, run_opts,
            transform_dir = None, online_ivector_dir = None):
    raise Exception("Realignment stage has not been implemented in nnet3")
    logger.info("Getting average posterior for purposes of adjusting the priors.")
    # Note: this just uses CPUs, using a smallish subset of data.
    # always use the first egs archive, which makes the script simpler;
    # we're using different random subsets of it.

    avg_post_vec_file = ComputeAveragePosterior(dir, iter, prev_egs_dir,
                            num_archives, prior_subset_size, run_opts)

    avg_post_vec_file = "{dir}/post.{iter}.vec".format(dir=dir, iter=iter)
    logger.info("Re-adjusting priors based on computed posteriors")
    model = '{0}/{1}.mdl'.format(dir, iter)
    AdjustAmPriors(dir, model, avg_post_vec_file, model, run_opts)

    alidir = Align(dir, feat_dir, lang, run_opts, iter,
                   transform_dir, online_ivector_dir)
    RunKaldiCommand("""
steps/nnet3/relabel_egs.sh --cmd "{command}" --iter {iter} {alidir} \
    {prev_egs_dir} {cur_egs_dir}""".format(
            command = run_opts.command,
            iter = iter,
            dir = dir,
            alidir = alidir,
            prev_egs_dir = prev_egs_dir,
            cur_egs_dir = cur_egs_dir))

def GetLearningRate(iter, num_jobs, num_iters, num_archives_processed,
                    num_archives_to_process,
                    initial_effective_lrate, final_effective_lrate):
    if iter + 1 >= num_iters:
        effective_learning_rate = final_effective_lrate
    else:
        effective_learning_rate =  initial_effective_lrate * math.exp(num_archives_processed * math.log(final_effective_lrate/ initial_effective_lrate)/num_archives_to_process)

    return num_jobs * effective_learning_rate

def DoShrinkage(iter, model_file, non_linearity, shrink_threshold,
                get_raw_nnet_from_am = True):

    if iter == 0:
        return True

    try:
        if get_raw_nnet_from_am:
            output, error = RunKaldiCommand("nnet3-am-info --print-args=false {model_file} | grep {non_linearity}".format(non_linearity = non_linearity, model_file = model_file))
        else:
            output, error = RunKaldiCommand("nnet3-info --print-args=false {model_file} | grep {non_linearity}".format(non_linearity = non_linearity, model_file = model_file))
        output = output.strip().split("\n")
        # eg.
        # component name=Lstm1_f type=SigmoidComponent, dim=1280, count=5.02e+05, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.06,0.17,0.19,0.24 0.28,0.33,0.44,0.62,0.79 0.96,0.99,1.0,1.0), mean=0.482, stddev=0.198], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.0001,0.003,0.004,0.03 0.12,0.18,0.22,0.24,0.25 0.25,0.25,0.25,0.25), mean=0.198, stddev=0.0591]

        mean_pattern = re.compile(".*deriv-avg=.*mean=([0-9\.]+).*")
        total_mean_deriv = 0
        num_derivs = 0
        for line in output:
            mat_obj = mean_pattern.search(line)
            if mat_obj is None:
                raise Exception("Something went wrong, unable to find deriv-avg in the line \n{0}".format(line))
            mean_deriv = float(mat_obj.groups()[0])
            total_mean_deriv += mean_deriv
            num_derivs += 1
        if total_mean_deriv / num_derivs < shrink_threshold:
            return True
    except ValueError:
        raise Exception("Error while parsing the model info output")

    return False

def ComputeAveragePosterior(dir, iter, egs_dir, num_archives,
                            prior_subset_size, run_opts,
                            get_raw_nnet_from_am = True):
    # Note: this just uses CPUs, using a smallish subset of data.
    """ Computes the average posterior of the network"""
    import glob
    for file in glob.glob('{0}/post.{1}.*.vec'.format(dir, iter)):
        os.remove(file)

    if run_opts.num_jobs_compute_prior > num_archives:
        egs_part = 1
    else:
        egs_part = 'JOB'

    if get_raw_nnet_from_am:
        model = "nnet3-am-copy --raw=true {dir}/combined.mdl -|".format(dir = dir)
    else:
        model = "{dir}/final.raw".format(dir = dir)

    RunKaldiCommand("""
{command} JOB=1:{num_jobs_compute_prior} {prior_queue_opt} {dir}/log/get_post.{iter}.JOB.log \
    nnet3-subset-egs --srand=JOB --n={prior_subset_size} ark:{egs_dir}/egs.{egs_part}.ark ark:- \| \
    nnet3-merge-egs --measure-output-frames=true --minibatch-size=128 ark:- ark:- \| \
    nnet3-compute-from-egs {prior_gpu_opt} --apply-exp=true \
    {model} ark:- ark:- \| \
matrix-sum-rows ark:- ark:- \| vector-sum ark:- {dir}/post.{iter}.JOB.vec
    """.format(command = run_opts.command,
               dir = dir, model = model,
               num_jobs_compute_prior = run_opts.num_jobs_compute_prior,
               prior_queue_opt = run_opts.prior_queue_opt,
               iter = iter, prior_subset_size = prior_subset_size,
               egs_dir = egs_dir, egs_part = egs_part,
               prior_gpu_opt = run_opts.prior_gpu_opt))

    # make sure there is time for $dir/post.{iter}.*.vec to appear.
    time.sleep(5)
    avg_post_vec_file = "{dir}/post.{iter}.vec".format(dir=dir, iter=iter)
    RunKaldiCommand("""
{command} {dir}/log/vector_sum.{iter}.log \
    vector-sum {dir}/post.{iter}.*.vec {output_file}
        """.format(command = run_opts.command,
                   dir = dir, iter = iter, output_file = avg_post_vec_file))

    for file in glob.glob('{0}/post.{1}.*.vec'.format(dir, iter)):
        os.remove(file)
    return avg_post_vec_file

def AdjustAmPriors(dir, input_model, avg_posterior_vector, output_model, run_opts):
    RunKaldiCommand("""
{command} {dir}/log/adjust_priors.final.log \
nnet3-am-adjust-priors {input_model} {avg_posterior_vector} {output_model}
    """.format(command = run_opts.command,
               dir = dir, input_model = input_model,
               avg_posterior_vector = avg_posterior_vector,
               output_model = output_model))

def RemoveEgs(egs_dir):
    RunKaldiCommand("steps/nnet2/remove_egs.sh {egs_dir}".format(egs_dir=egs_dir))

def CleanNnetDir(nnet_dir, num_iters, egs_dir, num_iters_combine = None,
                 preserve_model_interval = 100,
                 remove_egs = True,
                 get_raw_nnet_from_am = True):
    try:
        if remove_egs:
            RemoveEgs(egs_dir)

        for iter in range(num_iters):
            RemoveModel(nnet_dir, iter, num_iters, 1,
                        preserve_model_interval,
                        get_raw_nnet_from_am = get_raw_nnet_from_am)
    except (IOError, OSError) as err:
        logger.warning("Error while cleaning up the nnet directory")
        raise err

def RemoveModel(nnet_dir, iter, num_iters, num_iters_combine = None,
                preserve_model_interval = 100,
                get_raw_nnet_from_am = True):
    if iter % preserve_model_interval == 0:
        return
    if num_iters_combine is not None and iter >= num_iters - num_iters_combine + 1 :
        return
    if get_raw_nnet_from_am:
        file_name = '{0}/{1}.mdl'.format(nnet_dir, iter)
    else:
        file_name = '{0}/{1}.raw'.format(nnet_dir, iter)

    if os.path.isfile(file_name):
        os.remove(file_name)

def ComputeLifterCoeffs(lifter, dim):
    coeffs = [0] * dim
    for i in range(0, dim):
        coeffs[i] = 1.0 + 0.5 * lifter * math.sin(math.pi * i / float(lifter));

    return coeffs

def ComputeIdctMatrix(K, N, cepstral_lifter=0):
    matrix = [[0] * K for i in range(N)]
    # normalizer for X_0
    normalizer = math.sqrt(1.0 / float(N));
    for j in range(0, N):
        matrix[j][0] = normalizer;
    # normalizer for other elements
    normalizer = math.sqrt(2.0 / float(N));
    for k in range(1, K):
      for n in range(0, N):
        matrix[n][k] = normalizer * math.cos(math.pi / float(N) * (n + 0.5) * k);

    if cepstral_lifter != 0:
        lifter_coeffs = ComputeLifterCoeffs(cepstral_lifter, K)
        for k in range(0, K):
          for n in range(0, N):
            matrix[n][k] = matrix[n][k] / lifter_coeffs[k];

    return matrix

def WriteIdctMatrix(feat_dim, cepstral_lifter, file_path):
    # generate the IDCT matrix and write to the file
    idct_matrix = ComputeIdctMatrix(feat_dim, feat_dim, cepstral_lifter)
    # append a zero column to the matrix, this is the bias of the fixed affine component
    for k in range(0, feat_dim):
        idct_matrix[k].append(0)
    WriteKaldiMatrix(file_path, idct_matrix)

