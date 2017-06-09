#! /usr/bin/env python

# Copyright 2016  Vimal Manohar
# Apache 2.0.

"""This script converts a NIST RTTM file with
speaker info into kaldi utt2spk and segments.

The RTTM format is
<type> <file-id> <channel-id> <begin-time> \
        <duration> <ortho> <stype> <name> <conf>

We only process lines with <type> = SPEAKER
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
        description="""This script converts an RTTM with
        speaker info into kaldi utt2spk and segments""")
    parser.add_argument("--use-reco-id-as-spkr", type=str,
                        choices=["true", "false"],
                        help="Use the recording ID based on RTTM and "
                        "reco2file_and_channel as the speaker")
    parser.add_argument("--reco2file-and-channel", type=argparse.FileType('r'),
                        help="""Input reco2file_and_channel.
                        The format is <recording-id> <file-id> <channel-id>.
                        If not provided, then the <file-id> will be
                        used as the <recording-id> when creating segments file.
                        """)
    parser.add_argument("rttm_file", type=argparse.FileType('r'),
                        help="""Input RTTM file.
                        The format of the RTTM file is
                        <type> <file-id> <channel-id> <begin-time>
                        <duration> <NA> <NA> <speaker> <NA>""")
    parser.add_argument("utt2spk", type=argparse.FileType('w'),
                        help="Output utt2spk file")
    parser.add_argument("segments", type=argparse.FileType('w'),
                        help="Output segments file")

    args = parser.parse_args()

    args.use_reco_id_as_spkr = bool(args.use_reco_id_as_spkr == "true")

    return args


def main():
    args = get_args()

    file_and_channel2reco = {}
    if args.reco2file_and_channel is not None:
        for line in args.reco2file_and_channel:
            parts = line.strip().split()
            file_and_channel2reco[(parts[1], parts[2])] = parts[0]

    for line in args.rttm_file:
        parts = line.strip().split()
        if parts[0] != "SPEAKER":
            continue

        file_id = parts[1]
        channel = parts[2]

        if args.reco2file_and_channel is not None:
            try:
                reco = file_and_channel2reco[(file_id, channel)]
            except KeyError:
                raise RuntimeError(
                    "Could not find recording with (file_id, channel) "
                    "= ({0},{1}) in {2}".format(
                        file_id, channel,
                        args.reco2file_and_channel.name))
        else:
            reco = file_id

        start_time = float(parts[3])
        end_time = start_time + float(parts[4])

        if args.use_reco_id_as_spkr:
            spkr = reco
        else:
            spkr = parts[7]

        st = int(start_time * 100)
        end = int(end_time * 100)
        utt = "{0}-{1:06d}-{2:06d}".format(spkr, st, end)

        args.utt2spk.write("{0} {1}\n".format(utt, spkr))
        args.segments.write("{0} {1} {2:7.2f} {3:7.2f}\n".format(
            utt, reco, start_time, end_time))


if __name__ == '__main__':
    main()
