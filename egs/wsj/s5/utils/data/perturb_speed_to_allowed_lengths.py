#!/usr/bin/env python3

# Copyright     2017  Hossein Hadian
# Apache 2.0


""" This script perturbs speeds of utterances to force their lengths to some
    allowed lengths spaced by a factor (like 10%)
"""

import argparse
import os
import sys
import copy
import math
import logging

sys.path.insert(0, 'steps')
import libs.common as common_lib

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_args():
    parser = argparse.ArgumentParser(description="""This script copies the 'srcdir'
                                   data directory to output data directory 'dir'
                                   while modifying the utterances so that there are
                                   3 copies of each utterance: one with the same
                                   speed, one with a higher speed (not more than
                                   factor% faster) and one with a lower speed
                                   (not more than factor% slower)""")
    parser.add_argument('factor', type=float, default=12,
                        help='Spacing (in percentage) between allowed lengths.')
    parser.add_argument('srcdir', type=str,
                        help='path to source data dir')
    parser.add_argument('dir', type=str, help='output dir')
    parser.add_argument('--coverage-factor', type=float, default=0.05,
                        help="""Percentage of durations not covered from each
                             side of duration histogram.""")
    parser.add_argument('--frame-shift', type=int, default=10,
                        help="""Frame shift in milliseconds.""")
    parser.add_argument('--frame-length', type=int, default=25,
                        help="""Frame length in milliseconds.""")
    parser.add_argument('--frame-subsampling-factor', type=int, default=3,
                        help="""Chain frame subsampling factor.
                             See steps/nnet3/chain/train.py""")
    parser.add_argument('--speed-perturb', type=str, choices=['true','false'],
                        default='true',
                        help="""If false, no speed perturbation will occur, i.e.
                             only 1 copy of each utterance will be
                             saved, which is modified to have an allowed length
                             by using extend-wav-with-silence.""")
    args = parser.parse_args()
    args.speed_perturb = True if args.speed_perturb == 'true' else False
    return args

class Utterance(object):
    """ This class represents a Kaldi utterance
        in a data directory like data/train
    """

    def __init__(self, uid, wavefile, speaker, transcription, dur):
        self.wavefile = (wavefile if wavefile.rstrip(" \t\r\n").endswith('|') else
                         'cat {} |'.format(wavefile))
        self.speaker = speaker
        self.transcription = transcription
        self.id = uid
        self.dur = float(dur)

    def to_kaldi_utt_str(self):
        return self.id + " " + self.transcription

    def to_kaldi_wave_str(self):
        return self.id + " " + self.wavefile

    def to_kaldi_dur_str(self):
        return "{} {:0.3f}".format(self.id, self.dur)


def read_kaldi_datadir(dir):
    """ Read a data directory like
        data/train as a list of utterances
    """

    # check to make sure that no segments file exists as this script won't work
    # with data directories which use a segments file.
    if os.path.isfile(os.path.join(dir, 'segments')):
        logger.info("The data directory '{}' seems to use a 'segments' file. "
                    "This script does not yet support a 'segments' file. You'll need "
                    "to use utils/data/extract_wav_segments_data_dir.sh "
                    "to convert the data dir so it does not use a 'segments' file. "
                    "Exiting...".format(dir))
        sys.exit(1)

    logger.info("Loading the data from {}...".format(dir))
    utterances = []
    wav_scp = read_kaldi_mapfile(os.path.join(dir, 'wav.scp'))
    text = read_kaldi_mapfile(os.path.join(dir, 'text'))
    utt2dur = read_kaldi_mapfile(os.path.join(dir, 'utt2dur'))
    utt2spk = read_kaldi_mapfile(os.path.join(dir, 'utt2spk'))

    num_fail = 0
    for utt in wav_scp:
        if utt in text and utt in utt2dur and utt in utt2spk:
            utterances.append(Utterance(utt, wav_scp[utt], utt2spk[utt],
                                  text[utt], utt2dur[utt]))
        else:
            num_fail += 1

    if float(len(utterances)) / len(wav_scp) < 0.5:
        logger.info("More than half your data is problematic. Try "
                    "fixing using fix_data_dir.sh.")
        sys.exit(1)

    logger.info("Successfully read {} utterances. Failed for {} "
                "utterances.".format(len(utterances), num_fail))
    return utterances


def read_kaldi_mapfile(path):
    """ Read any Kaldi mapping file - like text, .scp files, etc.
    """

    m = {}
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip(" \t\r\n")
            sp_pos = line.find(' ')
            key = line[:sp_pos]
            val = line[sp_pos+1:]
            m[key] = val
    return m

def generate_kaldi_data_files(utterances, outdir):
    """ Write out a list of utterances as Kaldi data files into an
        output data directory.
    """

    logger.info("Exporting to {}...".format(outdir))
    speakers = {}

    with open(os.path.join(outdir, 'text'), 'w', encoding='latin-1') as f:
        for utt in utterances:
            f.write(utt.to_kaldi_utt_str() + "\n")

    with open(os.path.join(outdir, 'wav.scp'), 'w', encoding='latin-1') as f:
        for utt in utterances:
            f.write(utt.to_kaldi_wave_str() + "\n")

    with open(os.path.join(outdir, 'utt2dur'), 'w', encoding='latin-1') as f:
        for utt in utterances:
            f.write(utt.to_kaldi_dur_str() + "\n")

    with open(os.path.join(outdir, 'utt2spk'), 'w', encoding='latin-1') as f:
        for utt in utterances:
            f.write(utt.id + " " + utt.speaker + "\n")
            if utt.speaker not in speakers:
                speakers[utt.speaker] = [utt.id]
            else:
                speakers[utt.speaker].append(utt.id)

    with open(os.path.join(outdir, 'spk2utt'), 'w', encoding='latin-1') as f:
        for s in speakers:
            f.write(s + " ")
            for utt in speakers[s]:
                f.write(utt + " ")
            f.write('\n')

    logger.info("Successfully wrote {} utterances to data "
                "directory '{}'".format(len(utterances), outdir))

def find_duration_range(utterances, coverage_factor):
    """Given a list of utterances, find the start and end duration to cover

     If we try to cover
     all durations which occur in the training set, the number of
     allowed lengths could become very large.

     Returns
     -------
     start_dur: int
     end_dur: int
    """
    durs = []
    for u in utterances:
        durs.append(u.dur)
    durs.sort()
    to_ignore_dur = 0
    tot_dur = sum(durs)
    for d in durs:
        to_ignore_dur += d
        if to_ignore_dur * 100.0 / tot_dur > coverage_factor:
            start_dur = d
            break
    to_ignore_dur = 0
    for d in reversed(durs):
        to_ignore_dur += d
        if to_ignore_dur * 100.0 / tot_dur > coverage_factor:
            end_dur = d
            break
    if start_dur < 0.3:
        start_dur = 0.3  # a hard limit to avoid too many allowed lengths --not critical
    return start_dur, end_dur


def find_allowed_durations(start_dur, end_dur, args):
    """Given the start and end duration, find a set of
       allowed durations spaced by args.factor%. Also write
       out the list of allowed durations and the corresponding
       allowed lengths (in frames) on disk.

     Returns
     -------
     allowed_durations: list of allowed durations (in seconds)
    """

    allowed_durations = []
    d = start_dur
    with open(os.path.join(args.dir, 'allowed_durs.txt'), 'w', encoding='latin-1') as durs_fp, \
           open(os.path.join(args.dir, 'allowed_lengths.txt'), 'w', encoding='latin-1') as lengths_fp:
        while d < end_dur:
            length = int(d * 1000 - args.frame_length) / args.frame_shift + 1
            if length % args.frame_subsampling_factor != 0:
                length = (args.frame_subsampling_factor *
                              (length // args.frame_subsampling_factor))
                d = (args.frame_shift * (length - 1.0)
                     + args.frame_length + args.frame_shift / 2) / 1000.0
            allowed_durations.append(d)
            durs_fp.write("{}\n".format(d))
            lengths_fp.write("{}\n".format(int(length)))
            d *= args.factor
    return allowed_durations



def perturb_utterances(utterances, allowed_durations, args):
    """Given a set of utterances and a set of allowed durations, generate
       an extended set of perturbed utterances (all having an allowed duration)

     Returns
     -------
     perturbed_utterances: list of pertubed utterances
    """

    perturbed_utterances = []
    for u in utterances:
        # find i such that: allowed_durations[i-1] <= u.dur <= allowed_durations[i]
        # i = len(allowed_durations) --> no upper bound
        # i = 0         --> no lower bound
        if u.dur < allowed_durations[0]:
            i = 0
        elif u.dur > allowed_durations[-1]:
            i = len(allowed_durations)
        else:
            i = 1
            while i < len(allowed_durations):
                if u.dur <= allowed_durations[i] and u.dur >= allowed_durations[i - 1]:
                    break
                i += 1

        if i > 0 and args.speed_perturb:  # we have a smaller allowed duration
            allowed_dur = allowed_durations[i - 1]
            speed = u.dur / allowed_dur
            if max(speed, 1.0/speed) > args.factor:  # this could happen for very short/long utterances
                continue
            u1 = copy.deepcopy(u)
            u1.id = 'pv1-' + u.id
            u1.speaker = 'pv1-' + u.speaker
            u1.wavefile = '{} sox -t wav - -t wav - speed {} | '.format(u.wavefile, speed)
            u1.dur = allowed_dur
            perturbed_utterances.append(u1)


        if i < len(allowed_durations):  # we have a larger allowed duration
            allowed_dur2 = allowed_durations[i]
            speed = u.dur / allowed_dur2
            if max(speed, 1.0/speed) > args.factor:
                continue

            ## Add two versions for the second allowed_duration
            ## one version is by using speed modification using sox
            ## the other is by extending by silence
            if args.speed_perturb:
                u2 = copy.deepcopy(u)
                u2.id = 'pv2-' + u.id
                u2.speaker = 'pv2-' + u.speaker
                u2.wavefile = '{} sox -t wav - -t wav - speed {} | '.format(u.wavefile, speed)
                u2.dur = allowed_dur2
                perturbed_utterances.append(u2)

            delta = allowed_dur2 - u.dur
            if delta <= 1e-4:
                continue
            u3 = copy.deepcopy(u)
            u3.id = 'pv3-' + u.id
            u3.speaker = 'pv3-' + u.speaker
            u3.wavefile = '{} extend-wav-with-silence --extra-silence-length={} - - | '.format(u.wavefile, delta)
            u3.dur = allowed_dur2
            perturbed_utterances.append(u3)
    return perturbed_utterances



def main():
    args = get_args()
    args.factor = 1.0 + args.factor / 100.0

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    utterances = read_kaldi_datadir(args.srcdir)

    start_dur, end_dur = find_duration_range(utterances, args.coverage_factor)
    logger.info("Durations in the range [{},{}] will be covered. "
                "Coverage rate: {}%".format(start_dur, end_dur,
                                      100.0 - args.coverage_factor * 2))
    logger.info("There will be {} unique allowed lengths "
                "for the utterances.".format(int(math.log(end_dur / start_dur)/
                                                 math.log(args.factor))))

    allowed_durations = find_allowed_durations(start_dur, end_dur, args)

    perturbed_utterances = perturb_utterances(utterances, allowed_durations,
                                              args)

    generate_kaldi_data_files(perturbed_utterances, args.dir)


if __name__ == '__main__':
      main()
