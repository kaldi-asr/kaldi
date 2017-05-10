#! /usr/bin/env python

# Copyright 2016    Vimal Manohar
# Apache 2.0

from __future__ import print_function
import argparse
import random

def get_args():
    parser = argparse.ArgumentParser(description="""This script converts a
    wav.scp into split wav.scp that can be converted into noise-set-paramters
    that can be passed to steps/data/reverberate_data_dir.py.  The wav files in
    wav.scp is trimmed randomly into pieces based on options such options such
    as --max-duration, --skip-initial-duration and --num-parts-per-minute.""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--max-duration", type=float, default=30,
                        help="Maximum duration in seconds of the created "
                        "signal pieces")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Minimum duration in seconds of the created "
                        "signal pieces")
    parser.add_argument("--skip-initial-duration", type=float, default=5,
                        help="The duration in seconds of the original signal "
                        "that will be ignored while creating the pieces")
    parser.add_argument("--num-parts-per-minute", type=int, default=3,
                        help="Used to control the number of parts to create "
                        "from a recording")
    parser.add_argument("--sampling-rate", type=float, default=8000,
                        help="Required sampling rate of the output signals.")
    parser.add_argument('--random-seed', type=int, default=0,
                        help='seed to be used in the random split of signals')
    parser.add_argument("wav_scp", type=str,
                        help="The input wav.scp")
    parser.add_argument("reco2dur", type=str,
                        help="""Durations of the recordings corresponding to the
                        input wav.scp""")
    parser.add_argument("out_utt2dur", type=str,
                        help="Output utt2dur corresponding to split wavs")
    parser.add_argument("out_wav_scp", type=str,
                        help="Output wav.scp corresponding to split wavs")

    args = parser.parse_args()

    return args


def get_noise_set(reco, reco_dur, wav_rspecifier_split, sampling_rate,
                num_parts, max_duration, min_duration, skip_initial_duration):
    noise_set = []
    for i in range(num_parts):
        utt = "{0}-{1}".format(reco, i+1)

        start_time = round(random.random() * (reco_dur - skip_initial_duration)
                           + skip_initial_duration, 2)
        duration = min(round(random.random() * (max_duration-min_duration)
                             + min_duration, 2),
                       reco_dur - start_time)
        if duration < min_duration:
            continue

        if len(wav_rspecifier_split) == 1:
            rspecifier = ("sox -D {wav} -r {sr} -t wav - "
                          "trim {st} {dur} |".format(
                              wav=wav_rspecifier_split[0],
                              sr=sampling_rate, st=start_time, dur=duration))
        else:
            rspecifier = ("{wav} sox -D -t wav - -r {sr} -t wav - "
                          "trim {st} {dur} |".format(
                              wav=" ".join(wav_rspecifier_split),
                              sr=sampling_rate, st=start_time, dur=duration))

        noise_set.append( (utt, rspecifier, duration) )
    return noise_set


def main():
    args = get_args()
    random.seed(args.random_seed)

    reco2dur = {}
    for line in open(args.reco2dur):
        parts = line.strip().split()
        if len(parts) != 2:
            raise Exception(
                "Expecting reco2dur to contain lines of the format "
                "<reco-id> <duration>; Got {0}".format(line))
        reco2dur[parts[0]] = float(parts[1])

    out_wav_scp = open(args.out_wav_scp, 'w')
    out_utt2dur = open(args.out_utt2dur, 'w')

    for line in open(args.wav_scp):
        parts = line.strip().split()
        reco = parts[0]
        dur = reco2dur[reco]

        num_parts = int(float(args.num_parts_per_minute) / 60 * reco2dur[reco])

        noise_set = get_noise_set(
            reco, reco2dur[reco], wav_rspecifier_split=parts[1:],
            sampling_rate=args.sampling_rate, num_parts=num_parts,
            max_duration=args.max_duration, min_duration=args.min_duration,
            skip_initial_duration=args.skip_initial_duration)

        for utt, rspecifier, dur in noise_set:
            print ("{0} {1}".format(utt, rspecifier), file=out_wav_scp)
            print ("{0} {1}".format(utt, dur), file=out_utt2dur)

    out_wav_scp.close()
    out_utt2dur.close()


if __name__ == '__main__':
    main()
