#! /usr/bin/env python

# Copyright 2016  Vimal Manohar
# Apache 2.0.

"""This script converts kaldi-style utt2spk and segments to a NIST RTTM
file.

The RTTM format is
<type> <file-id> <channel-id> <begin-time> \
        <duration> <ortho> <stype> <name> <conf>

<type> = SPEAKER for each segment.
<file-id> - the File-ID of the recording
<channel-id> - the Channel-ID, usually 1
<begin-time> - start time of segment
<duration> - duration of segment
<ortho> - <NA> (this is ignored)
<stype> - <NA> (this is ignored)
<name> - speaker name or id
<conf> - <NA> (this is ignored)

"""

import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script converts kaldi-style utt2spk and
        segments to a NIST RTTM file""")

    parser.add_argument("--reco2file-and-channel", type=argparse.FileType('r'),
                        help="""Input reco2file_and_channel.
                        The format is <recording-id> <file-id> <channel-id>.
                        If not provided, then <recording-id> is taken as the
                        <file-id> with <channel-id> = 1.""")
    parser.add_argument("utt2spk", type=argparse.FileType('r'),
                        help="Input utt2spk file")
    parser.add_argument("segments", type=argparse.FileType('r'),
                        help="Input segments file")
    parser.add_argument("rttm_file", type=str, type=argparse.FileType('w'),
                        help="Output RTTM file")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    reco2file_and_channel = {}
    if args.reco2file_and_channel is not None:
        for line in args.reco2file_and_channel:
            parts = line.strip().split()
            reco2file_and_channel[parts[0]] = (parts[1], parts[2])

    utt2spk = {}
    for line in args.utt2spk:
        parts = line.strip().split()
        utt2spk[parts[0]] = parts[1]
    args.utt2spk.close()

    for line in args.segments:
        parts = line.strip().split()

        utt = parts[0]
        spkr = utt2spk[utt]

        reco = parts[1]

        if args.reco2file_and_channel is not None:
            try:
                file_id, channel = reco2file_and_channel[reco]
            except KeyError as e:
                raise RuntimeError(
                    "Could not find recording {0} in {1}".format(
                        reco, args.reco2file_and_channel.name))

        start_time = float(parts[2])
        duration = float(parts[3]) - start_time

        print("SPEAKER {0} {1} {2:7.2f} {3:7.2f} "
              "<NA> <NA> {4} <NA>\n".format(
                  file_id, channel, start_time,
                  duration, spkr), file=args.rttm_file))


if __name__ == '__main__':
    main()
