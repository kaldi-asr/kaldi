#!/usr/bin/env python

# Copyright 2016    Vijayaditya Peddinti
#           2016    Vimal Manohar
# Apache 2.0.

from __future__ import print_function
import os
import argparse
import sys
import logging
import shlex
import random
import math
import glob

sys.path.insert(0, 'steps')
import libs.data as data_lib
import libs.common as common_lib

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Getting egs for training')


def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="""Generates training examples used to train the 'nnet3'
        network (and also the validation examples used for diagnostics),
        and puts them in separate archives.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--cmd", type=str, default="run.pl",
                        help="Specifies the script to launch jobs."
                        " e.g. queue.pl for launching on SGE cluster run.pl"
                        " for launching on local machine")
    # feat options
    parser.add_argument("--feat.dir", type=str, dest='feat_dir', required=True,
                        help="Directory with features used for training "
                        "the neural network.")
    parser.add_argument("--feat.online-ivector-dir", type=str,
                        dest='online_ivector_dir',
                        default=None, action=common_lib.NullstrToNoneAction,
                        help="directory with the ivectors extracted in an "
                        "online fashion.")
    parser.add_argument("--feat.cmvn-opts", type=str, dest='cmvn_opts',
                        default=None, action=common_lib.NullstrToNoneAction,
                        help="A string specifying '--norm-means' and "
                        "'--norm-vars' values")
    parser.add_argument("--feat.apply-cmvn-sliding", type=str,
                        dest='apply_cmvn_sliding',
                        default=False, action=common_lib.StrToBoolAction,
                        help="Apply CMVN sliding, instead of per-utteance "
                        "or speakers")

    # egs extraction options
    parser.add_argument("--frames-per-eg", type=int, default=8,
                        help="""Number of frames of labels per example.
                        more->less disk space and less time preparing egs, but
                        more I/O during training.
                        note: the script may reduce this if
                        reduce-frames-per-eg is true.""")
    parser.add_argument("--left-context", type=int, default=4,
                        help="""Amount of left-context per eg (i.e. extra
                        frames of input features not present in the output
                        supervision).""")
    parser.add_argument("--right-context", type=int, default=4,
                        help="Amount of right-context per eg")
    parser.add_argument("--valid-left-context", type=int, default=None,
                        help="""Amount of left-context for validation egs,
                        typically used in recurrent architectures to ensure
                        matched condition with training egs""")
    parser.add_argument("--valid-right-context", type=int, default=None,
                        help="""Amount of right-context for validation egs,
                        typically used in recurrent architectures to ensure
                        matched condition with training egs""")
    parser.add_argument("--compress-input", type=str, default=True,
                        action=common_lib.StrToBoolAction,
                        choices=["true", "false"],
                        help="If false, disables compression. Might be "
                        "necessary to check if results will be affected.")
    parser.add_argument("--input-compress-format", type=int, default=0,
                        help="Format used for compressing the input features")

    parser.add_argument("--reduce-frames-per-eg", type=str, default=True,
                        action=common_lib.StrToBoolAction,
                        choices=["true", "false"],
                        help="""If true, this script may reduce the
                        frames-per-eg if there is only one archive and even
                        with the reduced frames-per-eg, the number of
                        samples-per-iter that would result is less than or
                        equal to the user-specified value.""")

    parser.add_argument("--num-utts-subset", type=int, default=300,
                        help="Number of utterances in validation and training"
                        " subsets used for shrinkage and diagnostics")
    parser.add_argument("--num-utts-subset-valid", type=int,
                        help="Number of utterances in validation"
                        " subset used for diagnostics")
    parser.add_argument("--num-utts-subset-train", type=int,
                        help="Number of utterances in training"
                        " subset used for shrinkage and diagnostics")
    parser.add_argument("--num-train-egs-combine", type=int, default=10000,
                        help="Training examples for combination weights at the"
                        " very end.")
    parser.add_argument("--num-valid-egs-combine", type=int, default=0,
                        help="Validation examples for combination weights at "
                        "the very end.")
    parser.add_argument("--num-egs-diagnostic", type=int, default=4000,
                        help="Numer of frames for 'compute-probs' jobs")

    parser.add_argument("--samples-per-iter", type=int, default=400000,
                        help="""This is the target number of egs in each
                        archive of egs (prior to merging egs).  We probably
                        should have called it egs_per_iter. This is just a
                        guideline; it will pick a number that divides the
                        number of samples in the entire data.""")

    parser.add_argument("--stage", type=int, default=0,
                        help="Stage to start running script from")
    parser.add_argument("--num-jobs", type=int, default=6,
                        help="""This should be set to the maximum number of
                        jobs you are comfortable to run in parallel; you can
                        increase it if your disk speed is greater and you have
                        more machines.""")
    parser.add_argument("--srand", type=int, default=0,
                        help="Rand seed for nnet3-copy-egs and "
                        "nnet3-shuffle-egs")
    parser.add_argument("--generate-egs-scp", type=str,
                        default=False, action=common_lib.StrToBoolAction,
                        help="Generate scp files in addition to archives")

    parser.add_argument("--targets-parameters", type=str, action='append',
                        required=True, dest='targets_para_array',
                        help="""Parameters for targets. Each set of parameters
                        corresponds to a separate output node of the neural
                        network. The targets can be sparse or dense.
                        The parameters used are:
                        --targets-rspecifier=<targets_rspecifier>
                            # rspecifier for the targets, can be alignment or
                            # matrix.
                        --num-targets=<n>
                            # targets dimension. required for sparse feats.
                        --target-type=<dense|sparse>""")

    parser.add_argument("--dir", type=str, required=True,
                        help="Directory to store the examples")

    print(' '.join(sys.argv))
    print(sys.argv)

    args = parser.parse_args()

    args = process_args(args)

    return args


def process_args(args):
    # process the options
    if args.num_utts_subset_valid is None:
        args.num_utts_subset_valid = args.num_utts_subset

    if args.num_utts_subset_train is None:
        args.num_utts_subset_train = args.num_utts_subset

    if args.valid_left_context is None:
        args.valid_left_context = args.left_context
    if args.valid_right_context is None:
        args.valid_right_context = args.right_context

    if (args.left_context < 0 or args.right_context < 0
            or args.valid_left_context < 0 or args.valid_right_context < 0):
        raise Exception(
            "--{,valid-}{left,right}-context should be non-negative")

    return args


def check_for_required_files(feat_dir, targets_scps, online_ivector_dir=None):
    required_files = ['{0}/feats.scp'.format(feat_dir),
                      '{0}/cmvn.scp'.format(feat_dir)]
    if online_ivector_dir is not None:
        required_files.append('{0}/ivector_online.scp'.format(
            online_ivector_dir))
        required_files.append('{0}/ivector_period'.format(
            online_ivector_dir))
    required_files.extend(targets_scps)

    for file in required_files:
        if not os.path.isfile(file):
            raise Exception('Expected {0} to exist.'.format(file))


def parse_targets_parameters_array(para_array):
    targets_parser = argparse.ArgumentParser()
    targets_parser.add_argument("--output-name", type=str, required=True,
                                help="Name of the output. e.g. output-xent")
    targets_parser.add_argument("--dim", type=int, default=-1,
                                help="Target dimension (required for sparse "
                                "targets")
    targets_parser.add_argument("--target-type", type=str, default="dense",
                                choices=["dense", "sparse"],
                                help="Dense for matrix format")
    targets_parser.add_argument("--targets-scp", type=str, required=True,
                                help="Scp file of targets; can be posteriors "
                                "or matrices")
    targets_parser.add_argument("--compress", type=str, default=True,
                                action=common_lib.StrToBoolAction,
                                help="Specifies whether the output must be "
                                "compressed")
    targets_parser.add_argument("--compress-format", type=int, default=0,
                                help="Format for compressing target")
    targets_parser.add_argument("--deriv-weights-scp", type=str, default="",
                                help="Per-frame deriv weights for this output")
    targets_parser.add_argument("--scp2ark-cmd", type=str, default="",
                                help="""The command that is used to convert
                                targets scp to archive. e.g. An scp of
                                alignments can be converted to posteriors using
                                ali-to-post""")

    targets_parameters = [targets_parser.parse_args(shlex.split(x))
                          for x in para_array]

    for t in targets_parameters:
        if not os.path.isfile(t.targets_scp):
            raise Exception("Expected {0} to exist.".format(t.targets_scp))

        if t.target_type == "dense":
            dim = common_lib.get_feat_dim_from_scp(t.targets_scp)
            if t.dim != -1 and t.dim != dim:
                raise Exception('Mismatch in --dim provided and feat dim for '
                                'file {0}; {1} vs {2}'.format(t.targets_scp,
                                                              t.dim, dim))
            t.dim = -dim

    return targets_parameters


def sample_utts(feat_dir, num_utts_subset, min_duration, exclude_list=None):
    utt2durs_dict = data_lib.get_utt2dur(feat_dir)
    utt2durs = utt2durs_dict.items()
    utt2uniq, uniq2utt = data_lib.get_utt2uniq(feat_dir)
    if num_utts_subset is None:
        num_utts_subset = len(utt2durs)
        if exclude_list is not None:
            num_utts_subset = num_utts_subset - len(exclude_list)

    random.shuffle(utt2durs)
    sampled_utts = []

    index = 0
    num_trials = 0
    while (len(sampled_utts) < num_utts_subset
           and num_trials <= len(utt2durs)):
        if utt2durs[index][-1] >= min_duration:
            if utt2uniq is not None:
                uniq_id = utt2uniq[utt2durs[index][0]]
                utts2add = uniq2utt[uniq_id]
            else:
                utts2add = [utt2durs[index][0]]
            exclude_utt = False
            if exclude_list is not None:
                for utt in utts2add:
                    if utt in exclude_list:
                        exclude_utt = True
                        break
            if not exclude_utt:
                for utt in utts2add:
                    sampled_utts.append(utt)

        else:
            logger.info("Skipping utterance %s of length %f",
                        utt2uniq[utt2durs[index][0]], utt2durs[index][1])
        index = index + 1
        num_trials = num_trials + 1
    if exclude_list is not None:
        assert(len(set(exclude_list).intersection(sampled_utts)) == 0)
    if len(sampled_utts) < num_utts_subset:
        raise Exception(
            """Number of utterances which have duration of at least {md}
            seconds is really low (required={rl}, available={al}).  Please
            check your data.""".format(
                md=min_duration, al=len(sampled_utts), rl=num_utts_subset))

    sampled_utts_durs = []
    for utt in sampled_utts:
        sampled_utts_durs.append([utt, utt2durs_dict[utt]])
    return sampled_utts, sampled_utts_durs


def write_list(listd, file_name):
    file_handle = open(file_name, 'w')
    assert(type(listd) == list)
    for item in listd:
        file_handle.write(str(item)+"\n")
    file_handle.close()


def get_max_open_files():
    stdout, stderr = common_lib.run_kaldi_command("ulimit -n")
    return int(stdout)


def get_feat_ivector_strings(dir, feat_dir, split_feat_dir,
                             cmvn_opt_string, ivector_dir=None,
                             apply_cmvn_sliding=False):

    if not apply_cmvn_sliding:
        train_feats = ("ark,s,cs:utils/filter_scp.pl --exclude "
                       "{dir}/valid_uttlist {sdir}/JOB/feats.scp | "
                       "apply-cmvn {cmvn} --utt2spk=ark:{sdir}/JOB/utt2spk "
                       "scp:{sdir}/JOB/cmvn.scp scp:- ark:- |".format(
                           dir=dir, sdir=split_feat_dir,
                           cmvn=cmvn_opt_string))
        valid_feats = ("ark,s,cs:utils/filter_scp.pl {dir}/valid_uttlist "
                       "{fdir}/feats.scp | "
                       "apply-cmvn {cmvn} --utt2spk=ark:{fdir}/utt2spk "
                       "scp:{fdir}/cmvn.scp scp:- ark:- |".format(
                           dir=dir, fdir=feat_dir, cmvn=cmvn_opt_string))
        train_subset_feats = ("ark,s,cs:utils/filter_scp.pl "
                              "{dir}/train_subset_uttlist  {fdir}/feats.scp | "
                              "apply-cmvn {cmvn} --utt2spk=ark:{fdir}/utt2spk "
                              "scp:{fdir}/cmvn.scp scp:- ark:- |".format(
                                  dir=dir, fdir=feat_dir,
                                  cmvn=cmvn_opt_string))

        def feats_subset_func(subset_list):
            return ("ark,s,cs:utils/filter_scp.pl {subset_list} "
                    "{fdir}/feats.scp | "
                    "apply-cmvn {cmvn} --utt2spk=ark:{fdir}/utt2spk "
                    "scp:{fdir}/cmvn.scp scp:- ark:- |".format(
                        dir=dir, subset_list=subset_list,
                        fdir=feat_dir, cmvn=cmvn_opt_string))

    else:
        train_feats = ("ark,s,cs:utils/filter_scp.pl --exclude "
                       "{dir}/valid_uttlist {sdir}/JOB/feats.scp | "
                       "apply-cmvn-sliding scp:{sdir}/JOB/cmvn.scp scp:- "
                       "ark:- |".format(dir=dir, sdir=split_feat_dir,
                                        cmvn=cmvn_opt_string))

        def feats_subset_func(subset_list):
            return ("ark,s,cs:utils/filter_scp.pl {subset_list} "
                    "{fdir}/feats.scp | "
                    "apply-cmvn-sliding {cmvn} scp:{fdir}/cmvn.scp scp:- "
                    "ark:- |".format(dir=dir, subset_list=subset_list,
                                     fdir=feat_dir, cmvn=cmvn_opt_string))

        train_subset_feats = feats_subset_func(
            "{0}/train_subset_uttlist".format(dir))
        valid_feats = feats_subset_func("{0}/valid_uttlist".format(dir))

    if ivector_dir is not None:
        ivector_period = common_lib.GetIvectorPeriod(ivector_dir)
        ivector_opt = ("--ivectors='ark,s,cs:utils/filter_scp.pl "
                       "{sdir}/JOB/utt2spk {idir}/ivector_online.scp | "
                       "subsample-feats --n=-{period} scp:- ark:- |'".format(
                           sdir=split_feat_dir, idir=ivector_dir,
                           period=ivector_period))
        valid_ivector_opt = ("--ivectors='ark,s,cs:utils/filter_scp.pl "
                             "{dir}/valid_uttlist {idir}/ivector_online.scp | "
                             "subsample-feats --n=-{period} "
                             "scp:- ark:- |'".format(
                                 dir=dir, idir=ivector_dir,
                                 period=ivector_period))
        train_subset_ivector_opt = (
            "--ivectors='ark,s,cs:utils/filter_scp.pl "
            "{dir}/train_subset_uttlist {idir}/ivector_online.scp | "
            "subsample-feats --n=-{period} scp:- ark:- |'".format(
                dir=dir, idir=ivector_dir, period=ivector_period))
    else:
        ivector_opt = ''
        valid_ivector_opt = ''
        train_subset_ivector_opt = ''

    return {'train_feats': train_feats,
            'valid_feats': valid_feats,
            'train_subset_feats': train_subset_feats,
            'feats_subset_func': feats_subset_func,
            'ivector_opts': ivector_opt,
            'valid_ivector_opts': valid_ivector_opt,
            'train_subset_ivector_opts': train_subset_ivector_opt,
            'feat_dim': common_lib.get_feat_dim(feat_dir),
            'ivector_dim': common_lib.get_ivector_dim(ivector_dir)}


def get_egs_options(targets_parameters, frames_per_eg,
                    left_context, right_context,
                    valid_left_context, valid_right_context,
                    compress_input,
                    input_compress_format=0, length_tolerance=0):

    train_egs_opts = []
    train_egs_opts.append("--left-context={0}".format(left_context))
    train_egs_opts.append("--right-context={0}".format(right_context))
    train_egs_opts.append("--num-frames={0}".format(frames_per_eg))
    train_egs_opts.append("--compress-input={0}".format(compress_input))
    train_egs_opts.append("--input-compress-format={0}".format(
                            input_compress_format))
    train_egs_opts.append("--compress-targets={0}".format(
                            ':'.join(["true" if t.compress else "false"
                                      for t in targets_parameters])))
    train_egs_opts.append("--targets-compress-formats={0}".format(
                            ':'.join([str(t.compress_format)
                                      for t in targets_parameters])))
    train_egs_opts.append("--length-tolerance={0}".format(length_tolerance))
    train_egs_opts.append("--output-names={0}".format(
                            ':'.join([t.output_name
                                      for t in targets_parameters])))
    train_egs_opts.append("--output-dims={0}".format(
                            ':'.join([str(t.dim)
                                      for t in targets_parameters])))

    valid_egs_opts = (
        "--left-context={vlc} --right-context={vrc} "
        "--num-frames={n} --compress-input={comp} "
        "--input-compress-format={icf} --compress-targets={ct} "
        "--targets-compress-formats={tcf} --length-tolerance={tol} "
        "--output-names={names} --output-dims={dims}".format(
            vlc=valid_left_context, vrc=valid_right_context, n=frames_per_eg,
            comp=compress_input, icf=input_compress_format,
            ct=':'.join(["true" if t.compress else "false"
                         for t in targets_parameters]),
            tcf=':'.join([str(t.compress_format)
                          for t in targets_parameters]),
            tol=length_tolerance,
            names=':'.join([t.output_name
                            for t in targets_parameters]),
            dims=':'.join([str(t.dim) for t in targets_parameters])))

    return {'train_egs_opts': " ".join(train_egs_opts),
            'valid_egs_opts': valid_egs_opts}


def get_targets_list(targets_parameters, subset_list):
    targets_list = []
    for t in targets_parameters:
        rspecifier = "ark,s,cs:" if t.scp2ark_cmd != "" else "scp,s,cs:"
        rspecifier += get_subset_rspecifier(t.targets_scp, subset_list)
        rspecifier += t.scp2ark_cmd
        deriv_weights_rspecifier = ""
        if t.deriv_weights_scp != "":
            deriv_weights_rspecifier = "scp,s,cs:{0}".format(
                get_subset_rspecifier(t.deriv_weights_scp, subset_list))
        this_targets = '''"{rspecifier}" "{dw}"'''.format(
            rspecifier=rspecifier, dw=deriv_weights_rspecifier)

        targets_list.append(this_targets)
    return " ".join(targets_list)


def get_subset_rspecifier(scp_file, subset_list):
    if scp_file == "":
        return ""
    return "utils/filter_scp.pl {subset} {scp} |".format(subset=subset_list,
                                                         scp=scp_file)


def split_scp(scp_file, num_jobs):
    out_scps = ["{0}.{1}".format(scp_file, n) for n in range(1, num_jobs + 1)]
    common_lib.run_kaldi_command("utils/split_scp.pl {scp} {oscps}".format(
                                    scp=scp_file,
                                    oscps=' '.join(out_scps)))
    return out_scps


def generate_valid_train_subset_egs(dir, targets_parameters,
                                    feat_ivector_strings, egs_opts,
                                    num_train_egs_combine,
                                    num_valid_egs_combine,
                                    num_egs_diagnostic, cmd,
                                    num_jobs=1,
                                    generate_egs_scp=False):

    if generate_egs_scp:
        valid_combine_output = ("ark,scp:{0}/valid_combine.egs,"
                                "{0}/valid_combine.egs.scp".format(dir))
        valid_diagnostic_output = ("ark,scp:{0}/valid_diagnostic.egs,"
                                   "{0}/valid_diagnostic.egs.scp".format(dir))
        train_combine_output = ("ark,scp:{0}/train_combine.egs,"
                                "{0}/train_combine.egs.scp".format(dir))
        train_diagnostic_output = ("ark,scp:{0}/train_diagnostic.egs,"
                                   "{0}/train_diagnostic.egs.scp".format(dir))
    else:
        valid_combine_output = "ark:{0}/valid_combine.egs".format(dir)
        valid_diagnostic_output = "ark:{0}/valid_diagnostic.egs".format(dir)
        train_combine_output = "ark:{0}/train_combine.egs".format(dir)
        train_diagnostic_output = "ark:{0}/train_diagnostic.egs".format(dir)

    wait_pids = []

    logger.info("Creating validation and train subset examples.")

    split_scp('{0}/valid_uttlist'.format(dir), num_jobs)
    split_scp('{0}/train_subset_uttlist'.format(dir), num_jobs)

    valid_pid = common_lib.run_kaldi_command(
        """{cmd} JOB=1:{nj} {dir}/log/create_valid_subset.JOB.log \
          nnet3-get-egs-multiple-targets {v_iv_opt} {v_egs_opt} "{v_feats}" \
          {targets} ark,scp:{dir}/valid_all.JOB.egs,"""
        """{dir}/valid_all.JOB.egs.scp""".format(
            cmd=cmd, nj=num_jobs, dir=dir,
            v_egs_opt=egs_opts['valid_egs_opts'],
            v_iv_opt=feat_ivector_strings['valid_ivector_opts'],
            v_feats=feat_ivector_strings['feats_subset_func'](
                '{dir}/valid_uttlist.JOB'.format(dir=dir)),
            targets=get_targets_list(
                targets_parameters,
                '{dir}/valid_uttlist.JOB'.format(dir=dir))),
        wait=False)

    train_pid = common_lib.run_kaldi_command(
        """{cmd} JOB=1:{nj} {dir}/log/create_train_subset.JOB.log \
          nnet3-get-egs-multiple-targets {t_iv_opt} {v_egs_opt} "{t_feats}" \
          {targets} ark,scp:{dir}/train_subset_all.JOB.egs,"""
        """{dir}/train_subset_all.JOB.egs.scp""".format(
            cmd=cmd, nj=num_jobs, dir=dir,
            v_egs_opt=egs_opts['valid_egs_opts'],
            t_iv_opt=feat_ivector_strings['train_subset_ivector_opts'],
            t_feats=feat_ivector_strings['feats_subset_func'](
                '{dir}/train_subset_uttlist.JOB'.format(dir=dir)),
            targets=get_targets_list(
                targets_parameters,
                '{dir}/train_subset_uttlist.JOB'.format(dir=dir))),
        wait=False)

    wait_pids.append(valid_pid)
    wait_pids.append(train_pid)

    for pid in wait_pids:
        stdout, stderr = pid.communicate()
        if pid.returncode != 0:
            raise Exception(stderr)

    valid_egs_all = ' '.join(
        ['{dir}/valid_all.{n}.egs.scp'.format(dir=dir, n=n)
         for n in range(1, num_jobs + 1)])
    train_subset_egs_all = ' '.join(
        ['{dir}/train_subset_all.{n}.egs.scp'.format(dir=dir, n=n)
         for n in range(1, num_jobs + 1)])

    wait_pids = []
    logger.info("... Getting subsets of validation examples for diagnostics "
                " and combination.")
    pid = common_lib.run_kaldi_command(
        """{cmd} {dir}/log/create_valid_subset_combine.log \
            cat {valid_egs_all} \| nnet3-subset-egs --n={nve_combine} \
            scp:- {valid_combine_output}""".format(
                cmd=cmd, dir=dir, valid_egs_all=valid_egs_all,
                nve_combine=num_valid_egs_combine,
                valid_combine_output=valid_combine_output),
        wait=False)
    wait_pids.append(pid)

    pid = common_lib.run_kaldi_command(
        """{cmd} {dir}/log/create_valid_subset_diagnostic.log \
            cat {valid_egs_all} \| nnet3-subset-egs --n={ne_diagnostic} \
            scp:- {valid_diagnostic_output}""".format(
                cmd=cmd, dir=dir, valid_egs_all=valid_egs_all,
                ne_diagnostic=num_egs_diagnostic,
                valid_diagnostic_output=valid_diagnostic_output),
        wait=False)
    wait_pids.append(pid)

    pid = common_lib.run_kaldi_command(
        """{cmd} {dir}/log/create_train_subset_combine.log \
            cat {train_subset_egs_all} \| \
            nnet3-subset-egs --n={nte_combine} \
            scp:- {train_combine_output}""".format(
                cmd=cmd, dir=dir, train_subset_egs_all=train_subset_egs_all,
                nte_combine=num_train_egs_combine,
                train_combine_output=train_combine_output),
        wait=False)
    wait_pids.append(pid)

    pid = common_lib.run_kaldi_command(
        """{cmd} {dir}/log/create_train_subset_diagnostic.log \
            cat {train_subset_egs_all} \| \
            nnet3-subset-egs --n={ne_diagnostic} \
            scp:- {train_diagnostic_output}""".format(
                cmd=cmd, dir=dir, train_subset_egs_all=train_subset_egs_all,
                ne_diagnostic=num_egs_diagnostic,
                train_diagnostic_output=train_diagnostic_output),
        wait=False)
    wait_pids.append(pid)

    for pid in wait_pids:
        stdout, stderr = pid.communicate()
        if pid.returncode != 0:
            raise Exception(stderr)

    common_lib.run_kaldi_command(
        """cat {dir}/valid_combine.egs {dir}/train_combine.egs > \
                {dir}/combine.egs""".format(dir=dir))

    if generate_egs_scp:
        common_lib.run_kaldi_command(
            """cat {dir}/valid_combine.egs.scp {dir}/train_combine.egs.scp > \
                    {dir}/combine.egs.scp""".format(dir=dir))
        common_lib.run_kaldi_command(
            "rm {dir}/valid_combine.egs.scp {dir}/train_combine.egs.scp"
            "".format(dir=dir))

    # perform checks
    for file_name in ('{0}/combine.egs {0}/train_diagnostic.egs '
                      '{0}/valid_diagnostic.egs'.format(dir).split()):
        if os.path.getsize(file_name) == 0:
            raise Exception("No examples in {0}".format(file_name))

    # clean-up
    for x in ('{0}/valid_all.*.egs {0}/train_subset_all.*.egs '
              '{0}/valid_all.*.egs.scp {0}/train_subset_all.*.egs.scp '
              '{0}/train_combine.egs '
              '{0}/valid_combine.egs'.format(dir).split()):
        for file_name in glob.glob(x):
            os.remove(file_name)


def generate_training_examples_internal(dir, targets_parameters, feat_dir,
                                        train_feats_string,
                                        train_egs_opts_string,
                                        ivector_opts,
                                        num_jobs, frames_per_eg,
                                        samples_per_iter, cmd, srand=0,
                                        reduce_frames_per_eg=True,
                                        only_shuffle=False,
                                        dry_run=False,
                                        generate_egs_scp=False):

    # The examples will go round-robin to egs_list.  Note: we omit the
    # 'normalization.fst' argument while creating temporary egs: the phase of
    # egs preparation that involves the normalization FST is quite
    # CPU-intensive and it's more convenient to do it later, in the 'shuffle'
    # stage.  Otherwise to make it efficient we need to use a large 'nj', like
    # 40, and in that case there can be too many small files to deal with,
    # because the total number of files is the product of 'nj' by
    # 'num_archives_intermediate', which might be quite large.
    num_frames = data_lib.get_num_frames(feat_dir)
    num_archives = (num_frames) / (frames_per_eg * samples_per_iter) + 1

    reduced = False
    while (reduce_frames_per_eg and frames_per_eg > 1
           and num_frames / ((frames_per_eg-1)*samples_per_iter) == 0):
        frames_per_eg -= 1
        num_archives = 1
        reduced = True

    if reduced:
        logger.info("Reduced frames-per-eg to {0} "
                    "because amount of data is small".format(frames_per_eg))

    max_open_files = get_max_open_files()
    num_archives_intermediate = num_archives
    archives_multiple = 1
    while (num_archives_intermediate+4) > max_open_files:
        archives_multiple = archives_multiple + 1
        num_archives_intermediate = int(math.ceil(float(num_archives)
                                        / archives_multiple))
    num_archives = num_archives_intermediate * archives_multiple
    egs_per_archive = num_frames/(frames_per_eg * num_archives)

    if egs_per_archive > samples_per_iter:
        raise Exception(
            """egs_per_archive({epa}) > samples_per_iter({fpi}).
            This is an error in the logic for determining
            egs_per_archive""".format(epa=egs_per_archive,
                                      fpi=samples_per_iter))

    if dry_run:
        if generate_egs_scp:
            for i in range(1, num_archives_intermediate + 1):
                for j in range(1, archives_multiple + 1):
                    archive_index = (i-1) * archives_multiple + j
                    common_lib.force_symlink(
                        "egs.{0}.ark".format(archive_index),
                        "{dir}/egs.{i}.{j}.ark".format(dir=dir, i=i, j=j))
        cleanup(dir, archives_multiple, generate_egs_scp)
        return {'num_frames': num_frames,
                'num_archives': num_archives,
                'egs_per_archive': egs_per_archive}

    logger.info("Splitting a total of {nf} frames into {na} archives, "
                "each with {epa} egs.".format(nf=num_frames, na=num_archives,
                                              epa=egs_per_archive))

    if os.path.isdir('{0}/storage'.format(dir)):
        # this is a striped directory, so create the softlinks
        data_lib.create_data_links(["{dir}/egs.{x}.ark".format(dir=dir, x=x)
                                  for x in range(1, num_archives + 1)])
        for x in range(1, num_archives_intermediate + 1):
            data_lib.create_data_links(
                ["{dir}/egs_orig.{y}.{x}.ark".format(dir=dir, x=x, y=y)
                 for y in range(1, num_jobs + 1)])

    split_feat_dir = "{0}/split{1}".format(feat_dir, num_jobs)
    egs_list = ' '.join(
        ['ark:{dir}/egs_orig.JOB.{ark_num}.ark'.format(dir=dir, ark_num=x)
         for x in range(1, num_archives_intermediate + 1)])

    if not only_shuffle:
        common_lib.run_kaldi_command(
            """{cmd} JOB=1:{nj} {dir}/log/get_egs.JOB.log \
                    nnet3-get-egs-multiple-targets {iv_opts} {egs_opts} \
                    "{feats}" {targets} ark:- \| \
                    nnet3-copy-egs --random=true --srand=$[JOB+{srand}] \
                    ark:- {egs_list}""".format(
                        cmd=cmd, nj=num_jobs, dir=dir, srand=srand,
                        iv_opts=ivector_opts, egs_opts=train_egs_opts_string,
                        feats=train_feats_string,
                        targets=get_targets_list(targets_parameters,
                                                 '{sdir}/JOB/utt2spk'.format(
                                                    sdir=split_feat_dir)),
                        egs_list=egs_list))

    logger.info("Recombining and shuffling order of archives on disk")
    egs_list = ' '.join(['{dir}/egs_orig.{n}.JOB.ark'.format(dir=dir, n=x)
                         for x in range(1, num_jobs + 1)])

    if archives_multiple == 1:
        # there are no intermediate archives so just shuffle egs across
        # jobs and dump them into a single output

        if generate_egs_scp:
            output_archive = ("ark,scp:{dir}/egs.JOB.ark,"
                              "{dir}/egs.JOB.scp".format(dir=dir))
        else:
            output_archive = "ark:{dir}/egs.JOB.ark".format(dir=dir)

        common_lib.run_kaldi_command(
            """{cmd} --max-jobs-run {msjr} JOB=1:{nai} \
            {dir}/log/shuffle.JOB.log \
                nnet3-shuffle-egs --srand=$[JOB+{srand}] \
                "ark:cat {egs_list}|" {output_archive}""".format(
                    cmd=cmd, msjr=num_jobs,
                    nai=num_archives_intermediate, srand=srand,
                    dir=dir, egs_list=egs_list,
                    output_archive=output_archive))

        if generate_egs_scp:
            out_egs_handle = open("{0}/egs.scp".format(dir), 'w')
            for i in range(1, num_archives_intermediate + 1):
                for line in open("{0}/egs.{1}.scp".format(dir, i)):
                    print (line.strip(), file=out_egs_handle)
            out_egs_handle.close()
    else:
        # there are intermediate archives so we shuffle egs across jobs
        # and split them into archives_multiple output archives
        if generate_egs_scp:
            output_archives = ' '.join(
                ["ark,scp:{dir}/egs.JOB.{ark_num}.ark,"
                 "{dir}/egs.JOB.{ark_num}.scp".format(
                    dir=dir, ark_num=x)
                 for x in range(1, archives_multiple + 1)])
        else:
            output_archives = ' '.join(
                ["ark:{dir}/egs.JOB.{ark_num}.ark".format(
                    dir=dir, ark_num=x)
                 for x in range(1, archives_multiple + 1)])
        # archives were created as egs.x.y.ark
        # linking them to egs.i.ark format which is expected by the training
        # scripts
        for i in range(1, num_archives_intermediate + 1):
            for j in range(1, archives_multiple + 1):
                archive_index = (i-1) * archives_multiple + j
                common_lib.force_symlink(
                    "egs.{0}.ark".format(archive_index),
                    "{dir}/egs.{i}.{j}.ark".format(dir=dir, i=i, j=j))

        common_lib.run_kaldi_command(
            """{cmd} --max-jobs-run {msjr} JOB=1:{nai} \
                {dir}/log/shuffle.JOB.log \
                    nnet3-shuffle-egs --srand=$[JOB+{srand}] \
                    "ark:cat {egs_list}|" ark:- \| \
                    nnet3-copy-egs ark:- {oarks}""".format(
                        cmd=cmd, msjr=num_jobs,
                        nai=num_archives_intermediate, srand=srand,
                        dir=dir, egs_list=egs_list, oarks=output_archives))

        if generate_egs_scp:
            out_egs_handle = open("{0}/egs.scp".format(dir), 'w')
            for i in range(1, num_archives_intermediate + 1):
                for j in range(1, archives_multiple + 1):
                    for line in open("{0}/egs.{1}.{2}.scp".format(dir, i, j)):
                        print (line.strip(), file=out_egs_handle)
            out_egs_handle.close()

    cleanup(dir, archives_multiple, generate_egs_scp)
    return {'num_frames': num_frames,
            'num_archives': num_archives,
            'egs_per_archive': egs_per_archive}


def cleanup(dir, archives_multiple, generate_egs_scp=False):
    logger.info("Removing temporary archives in {0}.".format(dir))
    for file_name in glob.glob("{0}/egs_orig*".format(dir)):
        real_path = os.path.realpath(file_name)
        data_lib.try_to_delete(real_path)
        data_lib.try_to_delete(file_name)

    if archives_multiple > 1 and not generate_egs_scp:
        # there will be some extra soft links we want to delete
        for file_name in glob.glob('{0}/egs.*.*.ark'.format(dir)):
            os.remove(file_name)


def create_directory(dir):
    import errno
    try:
        os.makedirs(dir)
    except OSError, e:
        if e.errno == errno.EEXIST:
            pass


def generate_training_examples(dir, targets_parameters, feat_dir,
                               feat_ivector_strings, egs_opts,
                               frame_shift, frames_per_eg, samples_per_iter,
                               cmd, num_jobs, srand=0,
                               only_shuffle=False, dry_run=False,
                               generate_egs_scp=False):

    # generate the training options string with the given chunk_width
    train_egs_opts = egs_opts['train_egs_opts']
    # generate the feature vector string with the utt list for the
    # current chunk width
    train_feats = feat_ivector_strings['train_feats']

    if os.path.isdir('{0}/storage'.format(dir)):
        real_paths = [os.path.realpath(x).strip("/")
                      for x in glob.glob('{0}/storage/*'.format(dir))]
        common_lib.run_kaldi_command(
            """utils/create_split_dir.pl {target_dirs} \
                    {dir}/storage""".format(
                        target_dirs=" ".join(real_paths), dir=dir))

    info = generate_training_examples_internal(
        dir=dir, targets_parameters=targets_parameters,
        feat_dir=feat_dir, train_feats_string=train_feats,
        train_egs_opts_string=train_egs_opts,
        ivector_opts=feat_ivector_strings['ivector_opts'],
        num_jobs=num_jobs, frames_per_eg=frames_per_eg,
        samples_per_iter=samples_per_iter, cmd=cmd,
        srand=srand,
        only_shuffle=only_shuffle,
        dry_run=dry_run,
        generate_egs_scp=generate_egs_scp)

    return info


def write_egs_info(info, info_dir):
    for x in ['num_frames', 'num_archives', 'egs_per_archive',
              'feat_dim', 'ivector_dim',
              'left_context', 'right_context', 'frames_per_eg']:
        write_list([info['{0}'.format(x)]], '{0}/{1}'.format(info_dir, x))


def generate_egs(egs_dir, feat_dir, targets_para_array,
                 online_ivector_dir=None,
                 frames_per_eg=8,
                 left_context=4,
                 right_context=4,
                 valid_left_context=None,
                 valid_right_context=None,
                 cmd="run.pl", stage=0,
                 cmvn_opts=None, apply_cmvn_sliding=False,
                 compress_input=True,
                 input_compress_format=0,
                 num_utts_subset_train=300,
                 num_utts_subset_valid=300,
                 num_train_egs_combine=1000,
                 num_valid_egs_combine=0,
                 num_egs_diagnostic=4000,
                 samples_per_iter=400000,
                 num_jobs=6,
                 srand=0,
                 generate_egs_scp=False):

    for directory in '{0}/log {0}/info'.format(egs_dir).split():
        create_directory(directory)

    print (cmvn_opts if cmvn_opts is not None else '',
           file=open('{0}/cmvn_opts'.format(egs_dir), 'w'))
    print ("true" if apply_cmvn_sliding else "false",
           file=open('{0}/apply_cmvn_sliding'.format(egs_dir), 'w'))

    targets_parameters = parse_targets_parameters_array(targets_para_array)

    # Check files
    check_for_required_files(feat_dir,
                             [t.targets_scp for t in targets_parameters],
                             online_ivector_dir)

    frame_shift = data_lib.get_frame_shift(feat_dir)
    min_duration = frames_per_eg * frame_shift
    valid_utts = sample_utts(feat_dir, num_utts_subset_valid, min_duration)[0]
    train_subset_utts = sample_utts(feat_dir, num_utts_subset_train,
                                    min_duration, exclude_list=valid_utts)[0]
    train_utts, train_utts_durs = sample_utts(feat_dir, None, -1,
                                              exclude_list=valid_utts)

    write_list(valid_utts, '{0}/valid_uttlist'.format(egs_dir))
    write_list(train_subset_utts, '{0}/train_subset_uttlist'.format(egs_dir))
    write_list(train_utts, '{0}/train_uttlist'.format(egs_dir))

    # split the training data into parts for individual jobs
    # we will use the same number of jobs as that used for alignment
    split_feat_dir = common_lib.split_data(feat_dir, num_jobs)
    feat_ivector_strings = get_feat_ivector_strings(
        dir=egs_dir, feat_dir=feat_dir, split_feat_dir=split_feat_dir,
        cmvn_opt_string=cmvn_opts,
        ivector_dir=online_ivector_dir,
        apply_cmvn_sliding=apply_cmvn_sliding)

    egs_opts = get_egs_options(targets_parameters=targets_parameters,
                               frames_per_eg=frames_per_eg,
                               left_context=left_context,
                               right_context=right_context,
                               valid_left_context=valid_left_context,
                               valid_right_context=valid_right_context,
                               compress_input=compress_input,
                               input_compress_format=input_compress_format)

    if stage <= 2:
        logger.info("Generating validation and training subset examples")

        generate_valid_train_subset_egs(
            dir=egs_dir,
            targets_parameters=targets_parameters,
            feat_ivector_strings=feat_ivector_strings,
            egs_opts=egs_opts,
            num_train_egs_combine=num_train_egs_combine,
            num_valid_egs_combine=num_valid_egs_combine,
            num_egs_diagnostic=num_egs_diagnostic,
            cmd=cmd,
            num_jobs=num_jobs,
            generate_egs_scp=generate_egs_scp)

    logger.info("Generating training examples on disk.")
    info = generate_training_examples(
        dir=egs_dir,
        targets_parameters=targets_parameters,
        feat_dir=feat_dir,
        feat_ivector_strings=feat_ivector_strings,
        egs_opts=egs_opts,
        frame_shift=frame_shift,
        frames_per_eg=frames_per_eg,
        samples_per_iter=samples_per_iter,
        cmd=cmd,
        num_jobs=num_jobs,
        srand=srand,
        only_shuffle=True if stage > 3 else False,
        dry_run=True if stage > 4 else False,
        generate_egs_scp=generate_egs_scp)

    info['feat_dim'] = feat_ivector_strings['feat_dim']
    info['ivector_dim'] = feat_ivector_strings['ivector_dim']
    info['left_context'] = left_context
    info['right_context'] = right_context
    info['frames_per_eg'] = frames_per_eg

    write_egs_info(info, '{dir}/info'.format(dir=egs_dir))


def main():
    args = get_args()
    generate_egs(args.dir, args.feat_dir, args.targets_para_array,
                 online_ivector_dir=args.online_ivector_dir,
                 frames_per_eg=args.frames_per_eg,
                 left_context=args.left_context,
                 right_context=args.right_context,
                 valid_left_context=args.valid_left_context,
                 valid_right_context=args.valid_right_context,
                 cmd=args.cmd, stage=args.stage,
                 cmvn_opts=args.cmvn_opts,
                 apply_cmvn_sliding=args.apply_cmvn_sliding,
                 compress_input=args.compress_input,
                 input_compress_format=args.input_compress_format,
                 num_utts_subset_train=args.num_utts_subset_train,
                 num_utts_subset_valid=args.num_utts_subset_valid,
                 num_train_egs_combine=args.num_train_egs_combine,
                 num_valid_egs_combine=args.num_valid_egs_combine,
                 num_egs_diagnostic=args.num_egs_diagnostic,
                 samples_per_iter=args.samples_per_iter,
                 num_jobs=args.num_jobs,
                 srand=args.srand,
                 generate_egs_scp=args.generate_egs_scp)


if __name__ == "__main__":
    main()
