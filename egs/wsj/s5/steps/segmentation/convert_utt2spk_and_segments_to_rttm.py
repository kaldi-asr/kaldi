#! /usr/bin/env python

"""This script converts kaldi-style utt2spk and segments to an RTTM"""

import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script converts kaldi-style utt2spk and
        segments to an RTTM""")

    parser.add_argument("utt2spk", type=str,
                        help="Input utt2spk file")
    parser.add_argument("segments", type=str,
                        help="Input segments file")
    parser.add_argument("reco2file_and_channel", type=str,
                        help="""Input reco2file_and_channel.
                        The format is <recording-id> <file-id> <channel-id>.""")
    parser.add_argument("rttm_file", type=str,
                        help="Output RTTM file")

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    reco2file_and_channel = {}
    for line in open(args.reco2file_and_channel):
        parts = line.strip().split()
        reco2file_and_channel[parts[0]] = (parts[1], parts[2])

    utt2spk = {}
    with open(args.utt2spk, 'r') as utt2spk_reader:
        for line in utt2spk_reader:
            parts = line.strip().split()
            utt2spk[parts[0]] = parts[1]

    with open(args.rttm_file, 'w') as rttm_writer:
        for line in open(args.segments, 'r'):
            parts = line.strip().split()

            utt = parts[0]
            spkr = utt2spk[utt]

            reco = parts[1]

            try:
                file_id, channel = reco2file_and_channel[reco]
            except KeyError as e:
                raise Exception("Could not find recording {0} in {1}: "
                                "{2}\n".format(reco,
                                               args.reco2file_and_channel,
                                               str(e)))

            start_time = float(parts[2])
            duration = float(parts[3]) - start_time

            rttm_writer.write("SPEAKER {0} {1} {2:7.2f} {3:7.2f} "
                              "<NA> <NA> {4} <NA>\n".format(
                                  file_id, channel, start_time,
                                  duration, spkr))

if __name__ == '__main__':
    main()
