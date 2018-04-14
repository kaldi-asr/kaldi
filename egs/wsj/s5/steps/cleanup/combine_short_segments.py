#!/usr/bin/env python

# Copyright 2016 Vijayaditya Peddinti
# Apache 2.0

from __future__ import print_function
import argparse
import sys
import os
import subprocess
import errno
import copy
import shutil
import warnings

def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="""
    **Warning, this script is deprecated.  Please use utils/data/combine_short_segments.sh**
    This script concatenates segments in the input_data_dir to ensure that"""
    " the segments in the output_data_dir have a specified minimum length.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument("--minimum-duration", type=float, required = True,
                        help="Minimum duration of the segments in the output directory")
    parser.add_argument("--input-data-dir", type=str, required = True)
    parser.add_argument("--output-data-dir", type=str, required = True)

    print(' '.join(sys.argv))
    args = parser.parse_args()
    return args

def RunKaldiCommand(command, wait = True):
    """ Runs commands frequently seen in Kaldi scripts. These are usually a
        sequence of commands connected by pipes, so we use shell=True """
    p = subprocess.Popen(command, shell = True,
                         stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE)

    if wait:
        [stdout, stderr] = p.communicate()
        if p.returncode is not 0:
            raise Exception("There was an error while running the command {0}\n".format(command)+"-"*10+"\n"+stderr)
        return stdout, stderr
    else:
        return p

def MakeDir(dir):
    try:
        os.mkdir(dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise exc
        raise Exception("Directory {0} already exists".format(dir))
        pass

def CheckFiles(input_data_dir):
    for file_name in ['spk2utt', 'text', 'utt2spk', 'feats.scp']:
        file_name = '{0}/{1}'.format(input_data_dir, file_name)
        if not os.path.exists(file_name):
            raise Exception("There is no such file {0}".format(file_name))

def ParseFileToDict(file, assert2fields = False, value_processor = None):
    if value_processor is None:
        value_processor = lambda x: x[0]

    dict = {}
    for line in open(file, 'r'):
        parts = line.split()
        if assert2fields:
            assert(len(parts) == 2)

        dict[parts[0]] = value_processor(parts[1:])
    return dict

def WriteDictToFile(dict, file_name):
    file = open(file_name, 'w')
    keys = dict.keys()
    keys.sort()
    for key in keys:
        value = dict[key]
        if type(value) in [list, tuple] :
            if type(value) is tuple:
                value = list(value)
            value.sort()
            value = ' '.join(value)
        file.write('{0}\t{1}\n'.format(key, value))
    file.close()


def ParseDataDirInfo(data_dir):
    data_dir_file = lambda file_name: '{0}/{1}'.format(data_dir, file_name)

    utt2spk = ParseFileToDict(data_dir_file('utt2spk'))
    spk2utt = ParseFileToDict(data_dir_file('spk2utt'), value_processor = lambda x: x)
    text = ParseFileToDict(data_dir_file('text'), value_processor = lambda x: " ".join(x))
    # we want to assert feats.scp has just 2 fields, as we don't know how
    # to process it otherwise
    feat = ParseFileToDict(data_dir_file('feats.scp'), assert2fields = True)
    utt2dur = ParseFileToDict(data_dir_file('utt2dur'), value_processor = lambda x: float(x[0]))
    utt2uniq = None
    if os.path.exists(data_dir_file('utt2uniq')):
        utt2uniq = ParseFileToDict(data_dir_file('utt2uniq'))
    return utt2spk, spk2utt, text, feat, utt2dur, utt2uniq


def GetCombinedUttIndexRange(utt_index, utts, utt_durs, minimum_duration):
    # We want the minimum number of concatenations
    # to reach the minimum_duration. If two concatenations satisfy
    # the minimum duration constraint we choose the shorter one.
    left_index = utt_index - 1
    right_index = utt_index + 1
    num_remaining_segments = len(utts) - 1
    cur_utt_dur = utt_durs[utts[utt_index]]

    while num_remaining_segments > 0:

        left_utt_dur = 0
        if left_index >= 0:
            left_utt_dur = utt_durs[utts[left_index]]
        right_utt_dur = 0
        if right_index <= len(utts) - 1:
            right_utt_dur = utt_durs[utts[right_index]]

        right_combined_utt_dur = cur_utt_dur + right_utt_dur
        left_combined_utt_dur = cur_utt_dur + left_utt_dur
        left_right_combined_utt_dur = cur_utt_dur + left_utt_dur + right_utt_dur

        combine_left_exit = False
        combine_right_exit = False
        if right_combined_utt_dur >= minimum_duration:
            if left_combined_utt_dur >= minimum_duration:
                if left_combined_utt_dur <= right_combined_utt_dur:
                    combine_left_exit = True
                else:
                    combine_right_exit = True
            else:
                combine_right_exit = True
        elif left_combined_utt_dur >= minimum_duration:
            combine_left_exit = True
        elif left_right_combined_utt_dur >= minimum_duration :
            combine_left_exit = True
            combine_right_exit = True

        if combine_left_exit and combine_right_exit:
            cur_utt_dur = left_right_combined_utt_dur
            break
        elif combine_left_exit:
            cur_utt_dur = left_combined_utt_dur
            # move back the right_index as we don't need to combine it
            right_index = right_index - 1
            break
        elif combine_right_exit:
            cur_utt_dur = right_combined_utt_dur
            # move back the left_index as we don't need to combine it
            left_index = left_index + 1
            break

        # couldn't satisfy minimum duration requirement so continue search
        if left_index >= 0:
            num_remaining_segments = num_remaining_segments - 1
        if right_index <= len(utts) - 1:
            num_remaining_segments = num_remaining_segments - 1

        left_index = left_index - 1
        right_index = right_index + 1

        cur_utt_dur = left_right_combined_utt_dur
    left_index = max(0, left_index)
    right_index = min(len(utts)-1, right_index)
    return left_index, right_index, cur_utt_dur


def WriteCombinedDirFiles(output_dir, utt2spk, spk2utt, text, feat, utt2dur, utt2uniq):
    out_dir_file = lambda file_name: '{0}/{1}'.format(output_dir, file_name)
    total_combined_utt_list = []
    for speaker in spk2utt.keys():
        utts = spk2utt[speaker]
        for utt in utts:
            if type(utt) is tuple:
                #this is a combined utt
                total_combined_utt_list.append((speaker, utt))

    for speaker, combined_utt_tuple in total_combined_utt_list:
        combined_utt_list = list(combined_utt_tuple)
        combined_utt_list.sort()
        new_utt_name = "-".join(combined_utt_list)+'-appended'

        # updating the utt2spk dict
        for utt in combined_utt_list:
            spk_name = utt2spk.pop(utt)
        utt2spk[new_utt_name] = spk_name

        # updating the spk2utt dict
        spk2utt[speaker].remove(combined_utt_tuple)
        spk2utt[speaker].append(new_utt_name)

        # updating the text dict
        combined_text = []
        for utt in combined_utt_list:
            combined_text.append(text.pop(utt))
        text[new_utt_name] = ' '.join(combined_text)

        # updating the feat dict
        combined_feat = []
        for utt in combined_utt_list:
            combined_feat.append(feat.pop(utt))
        feat_command = "concat-feats --print-args=false {feats} - |".format(feats = " ".join(combined_feat))
        feat[new_utt_name] = feat_command

        # updating utt2dur
        combined_dur = 0
        for utt in combined_utt_list:
            combined_dur += utt2dur.pop(utt)
        utt2dur[new_utt_name] = combined_dur

        # updating utt2uniq
        if utt2uniq is not None:
            combined_uniqs = []
            for utt in combined_utt_list:
                combined_uniqs.append(utt2uniq.pop(utt))
            # utt2uniq file is used to map perturbed data to original unperturbed
            # versions so that the training cross validation sets can avoid overlap
            # of data however if perturbation changes the length of the utterance
            # (e.g. speed perturbation) the utterance combinations in each
            # perturbation of the original recording can be very different. So there
            # is no good way to find the utt2uniq mapping so that we can avoid
            # overlap.
            utt2uniq[new_utt_name] = combined_uniqs[0]


    WriteDictToFile(utt2spk, out_dir_file('utt2spk'))
    WriteDictToFile(spk2utt, out_dir_file('spk2utt'))
    WriteDictToFile(feat, out_dir_file('feats.scp'))
    WriteDictToFile(text, out_dir_file('text'))
    if utt2uniq is not None:
        WriteDictToFile(utt2uniq, out_dir_file('utt2uniq'))
    WriteDictToFile(utt2dur, out_dir_file('utt2dur'))


def CombineSegments(input_dir, output_dir, minimum_duration):
    utt2spk, spk2utt, text, feat, utt2dur, utt2uniq = ParseDataDirInfo(input_dir)
    total_combined_utt_list = []

    # copy the duration dictionary so that we can modify it
    utt_durs = copy.deepcopy(utt2dur)
    speakers = spk2utt.keys()
    speakers.sort()
    for speaker in speakers:

        utts = spk2utt[speaker] # this is an assignment of the reference
        # In WriteCombinedDirFiles the values of spk2utt will have the list
        # of combined utts which will be used as reference

        # we make an assumption that the sorted uttlist corresponds
        # to contiguous segments. This is true only if utt naming
        # is done according to accepted conventions
        # this is an easily violatable assumption. Have to think of a better
        # way to do this.
        utts.sort()
        utt_index = 0
        while utt_index < len(utts):
            if utt_durs[utts[utt_index]] < minimum_duration:
                left_index, right_index, cur_utt_dur = GetCombinedUttIndexRange(utt_index, utts, utt_durs, minimum_duration)
                if not cur_utt_dur >= minimum_duration:
                    # this is a rare occurrence, better make the user aware of this
                    # situation and let them deal with it
                    warnings.warn('Speaker {0} does not have enough utterances to satisfy the minimum duration '
                                  'constraint. Not modifying these utterances'.format(speaker))
                    utt_index = utt_index + 1
                    continue
                combined_duration = 0
                combined_utts = []
                # update the utts_dur dictionary
                for utt in utts[left_index:right_index + 1]:
                    combined_duration += utt_durs.pop(utt)
                    if type(utt) is tuple:
                        for item in utt:
                            combined_utts.append(item)
                    else:
                        combined_utts.append(utt)
                combined_utts = tuple(combined_utts) # converting to immutable type to use as dictionary key
                assert(cur_utt_dur == combined_duration)

                # now modify the utts list
                combined_indices = range(left_index, right_index + 1)
                # start popping from the largest index so that the lower
                # indexes are valid
                for i in combined_indices[::-1]:
                    utts.pop(i)
                utts.insert(left_index, combined_utts)
                utt_durs[combined_utts] = combined_duration
                utt_index = left_index
            utt_index = utt_index + 1
    WriteCombinedDirFiles(output_dir, utt2spk, spk2utt, text, feat, utt2dur, utt2uniq)

def Main():
    print("""steps/cleanup/combine_short_segments.py: warning: this script is deprecated and will be removed.
          Please use utils/data/combine_short_segments.sh""", file = sys.stderr)
    args = GetArgs()

    CheckFiles(args.input_data_dir)
    MakeDir(args.output_data_dir)
    feat_lengths = {}
    segments_file = '{0}/segments'.format(args.input_data_dir)

    RunKaldiCommand("utils/data/get_utt2dur.sh {0}".format(args.input_data_dir))

    CombineSegments(args.input_data_dir, args.output_data_dir, args.minimum_duration)

    RunKaldiCommand("utils/utt2spk_to_spk2utt.pl {od}/utt2spk > {od}/spk2utt".format(od = args.output_data_dir))
    if os.path.exists('{0}/cmvn.scp'.format(args.input_data_dir)):
        shutil.copy('{0}/cmvn.scp'.format(args.input_data_dir), args.output_data_dir)

    RunKaldiCommand("utils/fix_data_dir.sh {0}".format(args.output_data_dir))
if __name__ == "__main__":
    Main()

