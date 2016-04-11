#!/usr/bin/env python


# Copyright 2016 Vijayaditya Peddinti.
# Apache 2.0.


# this script is based on steps/nnet3/lstm/train.sh

import os
import subprocess
import argparse
import sys
import pprint
import logging
import imp
import traceback
import shutil
import random
import math
import glob

train_lib = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')
data_lib = imp.load_source('dtl', 'utils/data/data_lib.py')
chain_lib = imp.load_source('ncl', 'steps/nnet3/chain/nnet3_chain_lib.py')
nnet3_log_parse = imp.load_source('nlp', 'steps/nnet3/report/nnet3_log_parse_lib.py')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting chain model trainer (train.py)')


def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="""
Generates training examples used to train the 'chain' system (and also the"""
" validation examples used for diagnostics), and puts them in separate archives.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--cmd", type=str, action = train_lib.NullstrToNoneAction, dest="command",
                        help="Specifies the script to launch jobs."
                        " e.g. queue.pl for launching on SGE cluster run.pl"
                        " for launching on local machine", default = "queue.pl")
    # feat options
    parser.add_argument("--feat.dir", type=str, dest='feat_dir', required = True,
                        help="Directory with features used for training the neural network.")
    parser.add_argument("--feat.online-ivector-dir", type=str, dest='online_ivector_dir',
                        default = None, action = train_lib.NullstrToNoneAction,
                        help="directory with the ivectors extracted in an online fashion.")
    parser.add_argument("--feat.cmvn-opts", type=str, dest='cmvn_opts',
                        default = None, action = train_lib.NullstrToNoneAction,
                        help="A string specifying '--norm-means' and '--norm-vars' values")

    # egs extraction options

    parser.add_argument("--cut-zero-frames", type=int, default=-1,
                        help="Number of frames (measured before subsampling)"
                        " to zero the derivative on each side of a cut point"
                        " (if set, activates new-style derivative weights)")
    parser.add_argument("--frame-subsampling-factor", type=int, default=3,
                        help="Frames-per-second of features we train on."
                        " Divided by frames-per-second at output of the chain model.")
    parser.add_argument("--alignment-subsampling-factor", type=int, default=3,
                        help="Frames-per-second of input alignments."
                        " Divided by frames-per-second at output of the chain model.")
    parser.add_argument("--chunk-width", type=int, default = [150], action=append,
                        help="Number of output labels in each example.")
    parser.add_argument("--chunk-overlap-per-eg", type=int, default = 0,
                        help="Number of supervised frames of overlap that we"
                        " aim per eg. It can be used to avoid data wastage when"
                        " using --left-deriv-truncate and --right-deriv-truncat"
                        " options in the training script")
    parser.add_argument("--chunk-left-context", type=int, default = 4,
                        help="Number of additional frames of input to the left"
                        " of the input chunk. This extra context will be used"
                        " in the estimation of RNN state before prediction of"
                        " the first label. In the case of FF-DNN this extra"
                        " context will be used to allow for frame-shifts")
    parser.add_argument("--chunk-right-context", type=int, default = 4,
                        help="Number of additional frames of input to the right"
                        " of the input chunk. This extra context will be used"
                        " in the estimation of bidirectional RNN state before"
                        " prediction of the first label.")
    parser.add_argument("--valid-left-context", type=int, default = None,
                        help=" Amount of left-context for validation egs, typically"
                        " used in recurrent architectures to ensure matched"
                        " condition with training egs")
    parser.add_argument("--valid-right-context", type=int, default = None,
                        help=" Amount of right-context for validation egs, typically"
                        " used in recurrent architectures to ensure matched"
                        " condition with training egs")
    parser.add_argument("--compress", type=str, default = True,
                        action = train_lib.StrToBoolAction,
                        choices = ["true", "false"],
                        help="If false, disables compression. Might be necessary"
                        " to check if results will be affected.")
    parser.add_argument("--num-utts-subset", type=int, default = 300,
                        help="Number of utterances in valudation and training"
                        " subsets used for shrinkage and diagnostics")
    parser.add_argument("--num-train-egs-combine", type=int, default=1000,
                        help="Training examples for combination weights at the"
                        " very end.")
    parser.add_argument("--num-valid-egs-combine", type=int, default=0,
                        help="Validation examples for combination weights at the"
                        " very end.")
    parser.add_argument("--num-egs-diagnostic", type=int, default=400,
                        help="Numer of frames for 'compute-probs' jobs")
    parser.add_argument("--frames-per-iter", type=int, default=400000,
                        help="Number of of supervised-frames seen per job of"
                        " training iteration. Measured at the sampling rate of"
                        " the features used. This is just a guideline; the script"
                        " will pick a number that divides the number of samples"
                        " in the entire data")
    parser.add_argument("--right-tolerance", type=int, default=None, help="")
    parser.add_argument("--left-tolerance", type=int, default=None, help="")


    parser.add_argument("--max-shuffle-jobs-run", type=int, default=50,
                        help="Limits the number of shuffle jobs which are"
                        " simultaneously run. Data shuffling jobs are fairly CPU"
                        " intensive as they include the nnet3-chain-normalize-egs"
                        " command; so we can run significant number of jobs"
                        " without overloading the disks.")
    parser.add_argument("--num-jobs", type=int, default=15,
                        help="Number of jobs to be run")

    parser.add_argument("--lat-dir", type=str, required = True,
                        help="Directory with alignments used for training the neural network.")
    parser.add_argument("--chain-dir", type=str, required = True,
                        help="Directory with trans_mdl, tree, normalization.fst")
    parser.add_argument("--dir", type=str, required = True,
                        help="Directory to store the examples")

    print(' '.join(sys.argv))
    print(sys.argv)

    args = parser.parse_args()

    [args, run_opts] = ProcessArgs(args)

    return [args, run_opts]

def ProcessArgs(args):
    # process the options
    if args.chunk_width < 1:
        raise Exception("--egs.chunk-width should have a minimum value of 1")

    if args.chunk_left_context < 0:
        raise Exception("--egs.chunk-left-context should be non-negative")

    if args.chunk_right_context < 0:
        raise Exception("--egs.chunk-right-context should be non-negative")

    if args.valid_left_context is None:
        args.valid_left_context = args.chunk_left_context
    if args.valid_right_context is None:
        args.valid_right_context = args.chunk_right_context

    return args

def CheckForRequiredFiles(feat_dir, chain_dir, lat_dir, online_ivector_dir = None):
    required_files = ['{0}/feats.scp'.format(feat_dir), '{0}/lat.1.gz'.format(lat_dir),
                      '{0}/final.mdl'.format(lat_dir), '{0}/0.trans_mdl'.format(chain_dir),
                       '{0}/tree'.format(chain_dir), '{0}/normalization.fst'.format(chain_dir)]
    if online_ivector_dir is not None:
        required_files.append('{0}/ivector_online.scp'.format(online_ivector_dir))
        required_files.append('{0}/ivector_period'.format(online_ivector_dir))

    for file in required_files:
        if not os.path.isfile(file):
            raise Exception('Expected {0} to exist.'.format(file))


def SampleUtts(feat_dir, num_utts_subset, min_duration, exclude_list=None):
    utt2durs_dict = data_lib.GetUtt2Dur(feat_dir)
    utt2durs = utt2durs_dict.items()
    utt2uniq, uniq2utt = data_lib.GetUtt2Uniq(feat_dir)
    if num_utts_subset is None:
        num_utts_subset = len(utt2durs)
        if exclude_list is not None:
            num_utts_subset = num_utts_subset - len(exclude_list)

    random.shuffle(utt2durs)
    sampled_utts = []

    index = 0
    num_trials = 0
    while (len(sampled_utts) < num_utts_subset) and (num_trials <= len(utt2durs)):
        if utt2durs[index][-1] >= min_duration:
            if utt2uniq is not None:
                uniq_id = utt2uniq[utt2durs[index][0]]
                utts2add = uniq2utt[uniq_id]
            else:
                utts2add = [utt2durs[index][0]]
            for utt in utts2add:
                if exclude_list is not None and utt in exclude_list:
                    continue
            for utt in utts2add:
                sampled_utts.append(utt)
            index = index + 1
        num_trials = num_trials + 1

    if len(sampled_utts) < num_utts_subset:
        raise Exception("Number of utterances which have duration of at least "
                "{md} seconds is really low. Please check your data.".format(md = min_duration))
    sampled_utts_durs = []
    for utt in sampled_utts:
        sampled_utts_durs.append([utt, utt2durs_dict[utt]])
    return sampled_utts, sampled_utts_durs

def WriteList(listd, file_name):
    file_handle = open(file_name, 'w')
    for item in listd:
        file_handle.write(str(item)+"\n")
    file_handle.close()

def GetMaxOpenFiles():
    stdout, stderr = train_lib.RunKaldiCommand("ulimit -n")
    return int(stdout)

def CopyTrainingLattices(lat_dir, dir, cmd, num_lat_jobs):
    train_lib.RunKaldiCommand("""
  {cmd} --max-jobs-run 6 JOB=1:{nj} {dir}/log/lattice_copy.JOB.log \
    lattice-copy "ark:gunzip -c {latdir}/lat.JOB.gz|" ark,scp:{dir}/lat.JOB.ark,{dir}/lat.JOB.scp""".format(cmd = cmd, nj = num_lat_jobs, dir = dir,
                       latdir = lat_dir))

    total_lat_file = open('{0}/lat.scp'.format(dir), 'w')
    for id in range(1, num_lat_jobs+1):
        lat_file_name = '{0}/lat.{1}.scp'.format(dir, id)
        lat_lines = ''.join(open(lat_file_name, 'r').readlines())
        total_lat_file.write(lat_lines)
    total_lat_file.close()

def GetFeatIvectorStrings(dir, feat_dir, split_feat_dir, cmvn_opt_string, ivector_dir = None):

    train_feats_function=lambda list_file: "ark,s,cs:utils/filter_scp.pl {list_file} {sdir}/JOB/feats.scp | apply-cmvn {cmvn} --utt2spk=ark:{sdir}/JOB/utt2spk scp:{sdir}/JOB/cmvn.scp scp:- ark:- |".format(sdir = split_feat_dir, cmvn = cmvn_opt_string, list_file = list_file)
    valid_feats="ark,s,cs:utils/filter_scp.pl {dir}/valid_uttlist {fdir}/feats.scp | apply-cmvn {cmvn} --utt2spk=ark:{fdir}/utt2spk scp:{fdir}/cmvn.scp scp:- ark:- |".format(dir = dir, fdir = feat_dir, cmvn = cmvn_opt_string)
    train_subset_feats="ark,s,cs:utils/filter_scp.pl {dir}/train_subset_uttlist  {fdir}/feats.scp | apply-cmvn {cmvn} --utt2spk=ark:{fdir}/utt2spk scp:{fdir}/cmvn.scp scp:- ark:- |".format(dir = dir, fdir = feat_dir, cmvn = cmvn_opt_string)

    if ivector_dir is not None:
        ivector_period = train_lib.GetIvectorPeriod(ivector_dir)
        ivector_opt="--ivectors='ark,s,cs:utils/filter_scp.pl {sdir}/JOB/utt2spk {idir}/ivector_online.scp | subsample-feats --n=-{period} scp:- ark:- |'".format(sdir = split_feat_dir, idir = ivector_dir, period = ivector_period)
        valid_ivector_opt="--ivectors='ark,s,cs:utils/filter_scp.pl {dir}/valid_uttlist {idir}/ivector_online.scp | subsample-feats --n=-{period} scp:- ark:- |'".format(dir = dir, idir = ivector_dir, period = ivector_period)
        train_subset_ivector_opt="--ivectors='ark,s,cs:utils/filter_scp.pl {dir}/train_subset_uttlist {idir}/ivector_online.scp | subsample-feats --n=-{period} scp:- ark:- |'".format(dir = dir, idir = ivector_dir, period = ivector_period)
    else:
        ivector_opt = ''
        valid_ivector_opt = ''
        train_subset_ivector_opt = ''

    return {'train_feats_function':train_feats_function, 'valid_feats':valid_feats, 'train_subset_feats':train_subset_feats,
            'ivector_opts':ivector_opt, 'valid_ivector_opts':valid_ivector_opt, 'train_subset_ivector_opts':train_subset_ivector_opt}

def GetEgsOptions(left_context, right_context,
                  valid_left_context, valid_right_context,
                  chunk_width, frames_overlap_per_eg,
                  frame_subsampling_factor, alignment_subsampling_factor,
                  left_tolerance, right_tolerance, compress, cut_zero_frames):

    if valid_left_context is None:
        valid_left_context = left_context
    if valid_right_context is None:
        valid_right_context = right_context

    train_egs_opts_func = lambda chunk_width:  "--left-context={lc} --right-context={rc} --num-frames={cw} --num-frames-overlap={fope} --frame-subsampling-factor={fsf} --compress={comp} --cut-zero-frames={czf}".format(lc = left_context, rc = right_context,
              cw = chunk_width, fope = frames_overlap_per_eg,
              fsf = frame_subsampling_factor, comp = compress,
              czf = cut_zero_frames)

    # don't do the overlap thing for the validation data.
    valid_egs_opts="--left-context={vlc} --right-context={vrc} --num-frames={cw} --frame-subsampling-factor={fsf} --compress={comp}".format(vlc = valid_left_context,
            vrc = valid_right_context, cw = chunk_width,
            fsf = frame_subsampling_factor, comp = compress)

    supervision_opts="--lattice-input=true --frame-subsampling-factor={asf}".format(asf = alignment_subsampling_factor)
    if right_tolerance is not None:
        supervision_opts="{sup} --right-tolerance={rt}".format(sup = supervision_opts, rt = right_tolerance)

    if left_tolerance is not None:
        supervision_opts="{sup} --left-tolerance={lt}".format(sup = supervision_opts, lt = left_tolerance)

    return {'train_egs_opts_function' : train_egs_opts_func,
            'valid_egs_opts' : valid_egs_opts,
            'supervision_opts' : supervision_opts}

def GenerateValidTrainSubsetEgs(dir, lat_dir, chain_dir,
                                feat_ivector_strings, egs_opts,
                                num_train_egs_combine,
                                num_valid_egs_combine,
                                num_egs_diagnostic, cmd):
    valid_utts = map(lambda x: x.strip(), open('{0}/valid_uttlist'.format(dir), 'r').readlines())
    train_utts = map(lambda x: x.strip(), open('{0}/train_subset_uttlist'.format(dir), 'r').readlines())
    lat_scp = map(lambda x: x.strip(), open('{0}/lat.scp'.format(dir), 'r').readlines())
    utt_set = set(valid_utts + train_utts)

    lat_scp_special = []
    for line in lat_scp:
        if line.split()[0] in utt_set:
            lat_scp_special.append(line)
    file_handle = open('{0}/lat_special.scp'.format(dir), 'w')
    file_handle.write('\n'.join(lat_scp_special))
    file_handle.close()

    logger.info("Creating validation subset examples.")
    train_lib.RunKaldiCommand("""
  {cmd} {dir}/log/create_valid_subset.log \
    lattice-align-phones --replace-output-symbols=true {ldir}/final.mdl scp:{dir}/lat_special.scp ark:- \| \
    chain-get-supervision {sup_opt} {cdir}/tree {cdir}/0.trans_mdl \
      ark:- ark:- \| \
    nnet3-chain-get-egs {v_iv_opt} {v_egs_opt} {cdir}/normalization.fst \
      "{v_feats}" ark,s,cs:- "ark:{dir}/valid_all.cegs" """.format(
          cmd = cmd, dir = dir, ldir = lat_dir, cdir = chain_dir,
          sup_opt = egs_opts['supervision_opts'],
          v_egs_opt = egs_opts['valid_egs_opts'],
          v_iv_opt = feat_ivector_strings['valid_ivector_opts'],
          v_feats = feat_ivector_strings['valid_feats']))

    logger.info("Creating train subset examples.")
    train_lib.RunKaldiCommand("""
  {cmd} {dir}/log/create_train_subset.log \
    lattice-align-phones --replace-output-symbols=true {ldir}/final.mdl scp:{dir}/lat_special.scp ark:- \| \
    chain-get-supervision {sup_opt} \
    {cdir}/tree {cdir}/0.trans_mdl ark:- ark:- \| \
    nnet3-chain-get-egs {t_iv_opt} {v_egs_opt} {cdir}/normalization.fst \
       "{t_feats}" ark,s,cs:- "ark:{dir}/train_subset_all.cegs" """.format(
          cmd = cmd, dir = dir, ldir = lat_dir, cdir = chain_dir,
          sup_opt = egs_opts['supervision_opts'],
          v_egs_opt = egs_opts['valid_egs_opts'],
          t_iv_opt = feat_ivector_strings['train_subset_ivector_opts'],
          t_feats = feat_ivector_strings['train_subset_feats']))

    logger.info("... Getting subsets of validation examples for diagnostics and combination.")
    train_lib.RunKaldiCommand("""
  {cmd} {dir}/log/create_valid_subset_combine.log \
    nnet3-chain-subset-egs --n={nve_combine} ark:{dir}/valid_all.cegs \
    ark:{dir}/valid_combine.cegs""".format(
        cmd = cmd, dir = dir, nve_combine = num_valid_egs_combine))


    train_lib.RunKaldiCommand("""
  {cmd} {dir}/log/create_valid_subset_diagnostic.log \
    nnet3-chain-subset-egs --n={ne_diagnostic} ark:{dir}/valid_all.cegs \
    ark:{dir}/valid_diagnostic.cegs""".format(
        cmd = cmd, dir = dir, ne_diagnostic = num_egs_diagnostic))

    train_lib.RunKaldiCommand("""
  {cmd} {dir}/log/create_train_subset_combine.log \
    nnet3-chain-subset-egs --n={nte_combine} ark:{dir}/train_subset_all.cegs \
    ark:{dir}/train_combine.cegs""".format(
        cmd = cmd, dir = dir, nte_combine = num_train_egs_combine))

    train_lib.RunKaldiCommand("""
  {cmd} {dir}/log/create_train_subset_diagnostic.log \
    nnet3-chain-subset-egs --n={ne_diagnostic} ark:{dir}/train_subset_all.cegs \
    ark:{dir}/train_diagnostic.cegs""".format(
        cmd = cmd, dir = dir, ne_diagnostic = num_egs_diagnostic))

    train_lib.RunKaldiCommand(""" cat {dir}/valid_combine.cegs {dir}/train_combine.cegs > {dir}/combine.cegs""".format(dir = dir))

    # perform checks
    for file_name in '{0}/combine.cegs {0}/train_diagnostic.cegs {0}/valid_diagnostic.cegs'.format(dir).split():
        if os.path.getsize(file_name) == 0:
            raise Exception("No examples in {0}".format(file_name))

    # clean-up
    for file_name in '{0}/valid_all.cegs {0}/train_subset_all.cegs {0}/train_combine.cegs {0}/valid_combine.cegs'.format(dir).split():
        os.remove(file_name)

# method to assign the train_utts to a partition with appropriate chunk_width
# Algorithm:
def GetTrainUttPartitions(chunk_widths, frame_shift, train_utts_durs, feat_dir, frames_per_iter):

    utt_partition_preference = {}
    for item in train_utts_durs:
        utt_name = item[0]
        width = int(item[1] * frame_shift)

        # get the partition preference order
        # get distances different chunk boundaries
        distances = []
        additional_cost = 10
        for cw in chunk_widths:
            distance_from_left_boundary = width % cw
            distance_from_right_boundary = cw - distance_from_left_boundary
            if distance_from_left_boundary > distance_from_right_boundary:
                distance = distance_from_right_boundary
            else:
                distance = distance_from_left_boundary + additional_cost
            distances.append([cw, distance])
        distances = sorted(distances, key = lambda x:x[1])
        utt_partition_preference[utt_name] = map(lambda x: x[0], distances)

    utts_per_chunk_width = {}
    for cw in chunk_widths:
        utts_per_chunk_width[cw] = []

    # get the initial assignment
    for utt, preferences in utt_partition_preference.items():
        utts_per_chunk_width[preferences.pop(0)].append(utt)

    # merge partitions until each of them has atleast frames_per_iter frames
    for iteration in xrange(len(chunk_widths) - 1):
        has_min_num_frames = True
        for chunk_width in chunk_widths:
            if GetNumFrames(feat_dir, utts_per_chunk_width[chunk_width]) <= frames_per_iter:
                has_min_num_frames = False
                # we don't want to use this chunk_width if there are less
                # than frames_per_iter frames as there will not be sufficient
                # randomization
                for utt in utts_per_chunk_width.pop(chunk_width):
                    preferences = utt_partition_preference[utt]
                    while len(preferences) > 0:
                        key = preferences.pop(0)
                        if utts_per_chunk_width.has_key(key):
                            utts_per_chunk_width[key].append(utt)
                            break

        if has_min_num_frames:
            break
    return utts_per_chunk_width

def GenerateTrainingExamplesFromUtts(dir, lat_dir, chain_dir, feat_dir,
                                     utts, chunk_width,
                                     train_feats_string, train_egs_opts_string,
                                     supervision_opts, ivector_opts,
                                     num_jobs, max_shuffle_jobs_run,
                                     frames_per_iter, cmd, only_shuffle):

    # The examples will go round-robin to egs_list.  Note: we omit the
    # 'normalization.fst' argument while creating temporary egs: the phase of egs
    # preparation that involves the normalization FST is quite CPU-intensive and
    # it's more convenient to do it later, in the 'shuffle' stage.  Otherwise to
    # make it efficient we need to use a large 'nj', like 40, and in that case
    # there can be too many small files to deal with, because the total number of
    # files is the product of 'nj' by 'num_archives_intermediate', which might be
    # quite large.
    num_frames = data_lib.GetNumFrames(feat_dir, utts)
    num_archives = num_frames / frames_per_iter + 1
    max_open_files = GetMaxOpenFiles()
    num_archives_intermediate = num_archives
    archives_multiple = 1
    while (num_archives_intermediate+4) > max_open_files:
      archives_multiple = archives_multiple + 1
      num_archives_intermediate = num_archives / archives_multiple
    num_archives = num_archives_intermediate * archives_multiple
    egs_per_archive = num_frames/(chunk_width * num_archives)

    if egs_per_archive > frames_per_iter:
        raise Exception("egs_per_archive({epa}) > frames_per_iter({fpi}). This is an error in the logic for determining egs_per_archive".format(epa = egs_per_achive, fpi = frames_per_iter))

    logger.info("Splitting a total of {nf} frames into {na} archives, each with {epa} egs.".format(nf = num_frames, na = num_archives, epa = egs_per_archive))

    if os.path.isdir('{0}/storage'.format(dir)):
        # this is a striped directory, so create the softlinks
        data_lib.CreateDataLinks(["{dir}/cegs.{x}.ark".format(dir = dir, x = x) for x in range(1, num_archives + 1)])
        for x in range(1, num_archives_intermediate + 1):
            data_lib.CreateDataLinks(["{dir}/cegs_orig.{y}.{x}.ark".format(dir = dir, x = x, y = y) for y in range(1, num_jobs + 1)])

    split_feat_dir = "{0}/split{1}".format(feat_dir, num_jobs)
    egs_list = ' '.join(['ark:{dir}/cegs_orig.JOB.{ark_num}.ark'.format(dir=dir, ark_num = x) for x in range(1, num_archives_intermediate + 1)])
    if not only_shuffle:
        train_lib.RunKaldiCommand("""
        {cmd} JOB=1:{nj} {dir}/log/get_egs.JOB.log \
        utils/filter_scp.pl {sfdir}/JOB/utt2spk {dir}/../lat.scp \| \
        lattice-align-phones --replace-output-symbols=true {ldir}/final.mdl scp:- ark:- \| \
        chain-get-supervision {sup_opts} \
         {cdir}/tree {cdir}/0.trans_mdl ark:- ark:- \| \
        nnet3-chain-get-egs {iv_opts} {egs_opts} \
         "{feats}" ark,s,cs:- ark:- \| \
        nnet3-chain-copy-egs --random=true --srand=JOB ark:- {egs_list}""".format(
            cmd = cmd, nj = num_jobs, dir = dir, ldir = lat_dir,
            sfdir = split_feat_dir, cdir = chain_dir,
            sup_opts = supervision_opts, iv_opts = ivector_opts, egs_opts = train_egs_opts_string,
            feats = train_feats_string, egs_list = egs_list))


    logger.info("Recombining and shuffling order of archives on disk")
    egs_list = ' '.join(['{dir}/cegs_orig.{n}.JOB.ark'.format(dir=dir, n = x) for x in range(1, num_jobs + 1)])
    if archives_multiple == 1:
        # there are no intermediate archives so just shuffle egs across
        # jobs and dump them into a single output
        train_lib.RunKaldiCommand("""
    {cmd} --max-jobs-run {msjr} --mem 8G JOB=1:{nai} {dir}/log/shuffle.JOB.log \
      nnet3-chain-normalize-egs {cdir}/normalization.fst "ark:cat {egs_list}|" ark:- \| \
      nnet3-chain-shuffle-egs --srand=JOB ark:- ark:{dir}/cegs.JOB.ark""".format(
              cmd = cmd, msjr = max_shuffle_jobs_run,
              nai = num_archives_intermediate, cdir = chain_dir,
              dir = dir, egs_list = egs_list))
    else:
        # there are intermediate archives so we shuffle egs across jobs
        # and split them into archives_multiple output archives
        output_archives = ' '.join(["ark:{dir}/cegs/JOB.{ark_num}.ark".format(dir = dir, ark_num = x) for x in range(1, archives_multple)])

        train_lib.RunKaldiCommand("""
    {cmd} --max-jobs-run {msjr} --mem 8G JOB=1:{nai} {dir}/log/shuffle.JOB.log \
      nnet3-chain-normalize-egs {cdir}/normalization.fst "ark:cat {egs_list}|" ark:- \| \
      nnet3-chain-shuffle-egs --srand=JOB ark:- ark:- \| \
      nnet3-chain-copy-egs ark:- {oarks}""".format(
          cmd = cmd, msjr = max_shuffle_jobs_run,
          nai = num_archives_intermediate, cdir = chain_dir,
          dir = dir, egs_list = egs_list, oarks = output_archives))

        # archives were created as cegs.x.y.ark
        # linking them to cegs.i.ark format which is expected by the training
        # scripts
        for i in range(1, num_archives_intermediate):
            for j in range(1, archives_multiple):
                archive_index = (i-1) * archive_multiple + j
                ForceSymLink("cegs.{0}.ark".format(archive_index),
                             "{dir}/cegs.{i}.{j}.ark".format(dir = dir, i = i, j = j))

        Cleanup(dir, archive_multiple)

def ForceSymLink(source, target):
    import os, errno
    try:
        os.symlink(source, target)
    except OSError, e:
        if e.errno == errno.EEXIST:
            os.remove(target)
            os.symlink(source, target)
        else:
            raise e

def Cleanup(dir, archive_multiple):
    logger.info("Removing temporary lattices")
    for lat_file in glob.glob('{0}/lat.*'.format(dir)):
        os.remove(lat_file)

    logger.info("Removing temporary alignments")
    for file_name in '{0}/ali.ark {0}/ali.scp'.format(dir).split():
        os.remove(file_name)
    for file_name in '{0}/trans.ark {0}/trans.scp'.format(dir).split():
        try:
            os.remove(file_name)
        except OSError:
            pass

    if archive_multiple > 1:
        # there will be some extra soft links we want to delete
        for file_name in glob.glob('{0}/cegs.*.*.ark'.format(dir)):
            os.remove(file_name)

def CreateDirectory(dir):
    import os, errno
    try:
        os.makedirs(dir)
    except OSError, e:
        if e.errno == errno.EEXIST:
            pass

def GenerateTrainingExamples(dir, lat_dir, chain_dir, feat_dir,
                             feat_ivector_strings, egs_opts,
                             train_utts_durs, chunk_widths,
                             frame_shift, frames_per_iter,
                             cmd, num_jobs, max_shuffle_jobs_run, only_shuffle):

    utt_partitions = GetTrainUttPartitions(chunk_widths, frame_shift,
                                           train_utts_durs, feat_dir,
                                           frames_per_iter)
    cur_dir_info = []
    total_ark_list = []
    for chunk_width in utt_partitions.keys():
        logger.info("Generating training examples for chunk width {0}".format(chunk_width))
        utts = utt_partitions[chunk_width]
        cur_dir = '{0}/cw{1}'.format(dir.strip("/"), chunk_width)
        CreateDirectory(cur_dir)
        cur_dir_info.append([chunk_width, cur_dir])
        utt_list_file = '{0}/train_uttlist'.format(cur_dir)
        handle = open(utt_list_file, 'w')
        handle.write("\n".join(utts))
        handle.close()

        # generate the training options string with the given chunk_width
        train_egs_opt_func = egs_opts['train_egs_opts_function']
        train_egs_opts_string = train_egs_opt_func(chunk_width)
        # generate the feature vector string with the utt list for the
        # current chunk width
        train_feats_func = feat_ivector_strings['train_feats_function']
        train_feats_string = train_feats_func(utt_list_file)

        if os.path.isdir('{0}/storage'.format(dir)):
            # dir was striped, so stripe cur_dir
            real_paths = [os.path.realpath(x).strip("/") for x in glob.glob('{0}/storage/*'.format(dir))]
            cur_real_paths = ['{0}_cw_{1}'.format(x, chunk_width) for x in real_paths]
            train_lib.RunKaldiCommand("""
                utils/create_split_dir.pl {target_dirs} {dir}/storage""".format(target_dirs = " ".join(cur_real_paths), dir = cur_dir))

        GenerateTrainingExamplesFromUtts(cur_dir, lat_dir, chain_dir, feat_dir,
                                         utts, chunk_width,
                                         train_feats_string, train_egs_opts_string,
                                         egs_opts['supervision_opts'],
                                         feat_ivector_strings['ivector_opts'],
                                         num_jobs, max_shuffle_jobs_run,
                                         frames_per_iter, cmd, only_shuffle)

        total_ark_list = total_ark_list + glob.glob(cur_dir+'/cegs.*.ark')

    # create soft links for the ark files in the parent directory
    random.shuffle(total_ark_list)
    dir_path = os.path.abspath(dir)+"/"
    for ark_index in range(1, len(total_ark_list)+1):
        relative_path = os.path.abspath(total_ark_list[ark_index - 1]).split(dir_path)[1]
        ForceSymLink(relative_path, '{dir}/cegs.{ai}.ark'.format(dir = dir, ai = ark_index))

    # write the info about the individual partition egs_dirs
    partition_file = open(dir.strip()+'/utt_partition_info', 'w')
    for chunk_width,cur_dir in cur_dir_info:
        partition_file.write("--chunk_width {0} --egs-dir {1}".format(chunk_width, cur_dir))
    partition_file.close()

def GenerateChainEgs(chain_dir, lat_dir, egs_dir, feat_dir,
                    online_ivector_dir = None,
                    chunk_width = [150],
                    chunk_left_context = 4,
                    chunk_right_context = 4,
                    valid_left_context = None,
                    valid_right_context = None,
                    chunk_overlap_per_eg = 0,
                    cmd = "run.pl", stage = 0,
                    cmvn_opts = None,
                    compress = True,
                    num_utts_subset = 300,
                    num_train_egs_combine = 1000,
                    num_valid_egs_combine = 0,
                    num_egs_diagnostic = 400,
                    frames_per_iter = 400000,
                    frames_overlap_per_eg = 0,
                    left_tolerance = None,
                    right_tolerance = None,
                    cut_zero_frames = -1,
                    max_shuffle_jobs_run = 50,
                    num_jobs = 15,
                    frame_subsampling_factor = 3,
                    alignment_subsampling_factor = 3):

    # Check files
    CheckForRequiredFiles(feat_dir, chain_dir, lat_dir, online_ivector_dir)

    for directory in '{0}/log {0}/info'.format(egs_dir).split():
        if not os.path.exists(directory):
            os.makedirs(directory)

    frame_shift = data_lib.GetFrameShift(feat_dir)
    min_duration = float(max(chunk_width)) * frame_shift
    valid_utts = SampleUtts(feat_dir, num_utts_subset, min_duration)[0]
    train_subset_utts = SampleUtts(feat_dir, num_utts_subset, min_duration, exclude_list = valid_utts)[0]
    train_utts, train_utts_durs = SampleUtts(feat_dir, None, -1, exclude_list = valid_utts)

    WriteList(valid_utts, '{0}/valid_uttlist'.format(egs_dir))
    WriteList(train_subset_utts, '{0}/train_subset_uttlist'.format(egs_dir))

    num_lat_jobs = train_lib.GetNumberOfJobs(lat_dir)
    if stage <= 1:
        logger.info("Copying training lattices.")
        CopyTrainingLattices(lat_dir, egs_dir, cmd, num_lat_jobs)


    # split the training data into parts for individual jobs
    # we will use the same number of jobs as that used for alignment
    split_feat_dir = train_lib.SplitData(feat_dir, num_jobs)
    feat_ivector_strings = GetFeatIvectorStrings(egs_dir, feat_dir,
            split_feat_dir, cmvn_opts, ivector_dir = online_ivector_dir)

    egs_opts = GetEgsOptions(chunk_left_context, chunk_right_context,
                             valid_left_context, valid_right_context,
                             chunk_width[0], frames_overlap_per_eg,
                             frame_subsampling_factor,
                             alignment_subsampling_factor,
                             left_tolerance, right_tolerance,
                             compress, cut_zero_frames)

    if stage <= 2:
        logger.info("Generating validation and training subset examples")

        GenerateValidTrainSubsetEgs(egs_dir, lat_dir, chain_dir,
                                    feat_ivector_strings, egs_opts,
                                    num_train_egs_combine,
                                    num_valid_egs_combine,
                                    num_egs_diagnostic, cmd)

    if (stage <= 4):
        only_shuffle = False
        if (stage > 3):
            # only shuffle the existing archives
            only_shuffle = True

        logger.info("Generating training examples on disk.")
        GenerateTrainingExamples(egs_dir, lat_dir, chain_dir, feat_dir,
                                 feat_ivector_strings, egs_opts,
                                 train_utts_durs, chunk_width,
                                 frame_shift, frames_per_iter,
                                 cmd, num_jobs, max_shuffle_jobs_run,
                                 only_shuffle)

def Main():
    args = GetArgs()
    GenerateChainEgs(args.chain_dir, args.lat_dir, args.dir, args.feat_dir,
                     online_ivector_dir = args.online_ivector_dir,
                     chunk_width = args.chunk_width,
                     chunk_left_context = args.chunk_left_context,
                     chunk_right_context = args.chunk_right_context,
                     valid_left_context = args.valid_left_context,
                     valid_right_context = args.valid_right_context,
                     chunk_overlap_per_eg = args.chunk_overlap_per_eg,
                     cmd = args.cmd, stage = args.stage,
                     cmvn_opts = args.cmvn_opts,
                     compress = args.compress,
                     num_utts_subset = args.num_utts_subset,
                     num_train_egs_combine = args.num_train_egs_combine,
                     num_valid_egs_combine = args.num_valid_egs_combine,
                     num_egs_diagnostic = args.num_egs_diagnostic,
                     frames_per_iter = args.frames_per_iter,
                     left_tolerance = args.left_tolerance,
                     right_tolerance = args.right_tolerance,
                     cut_zero_frames = args.cut_zero_frames,
                     max_shuffle_jobs_run = args.max_shuffle_jobs_run,
                     num_jobs = args.num_jobs,
                     frame_subsampling_factor = args.frame_subsampling_factor,
                     alignment_subsampling_factor = args.alignment_subsampling_factor)

if __name__ == "__main__":
    Main()
