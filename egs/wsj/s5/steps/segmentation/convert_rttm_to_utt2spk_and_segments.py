#! /usr/bin/env python

"""This script converts an RTTM with
speaker info into kaldi utt2spk and segments"""

import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script converts an RTTM with
        speaker info into kaldi utt2spk and segments""")
    parser.add_argument("--use-reco-id-as-spkr", type=str,
                        choices=["true", "false"],
                        help="Use the recording ID based on RTTM and "
                        "reco2file_and_channel as the speaker")
    parser.add_argument("rttm_file", type=str,
                        help="""Input RTTM file.
                        The format of the RTTM file is
                        <type> <file-id> <channel-id> <begin-time> """
                        """<end-time> <NA> <NA> <speaker> <conf>""")
    parser.add_argument("reco2file_and_channel", type=str,
                        help="""Input reco2file_and_channel.
                        The format is <recording-id> <file-id> <channel-id>.""")
    parser.add_argument("utt2spk", type=str,
                        help="Output utt2spk file")
    parser.add_argument("segments", type=str,
                        help="Output segments file")

    args = parser.parse_args()

    args.use_reco_id_as_spkr = bool(args.use_reco_id_as_spkr == "true")

    return args

def main():
    args = get_args()

    file_and_channel2reco = {}
    for line in open(args.reco2file_and_channel):
        parts = line.strip().split()
        file_and_channel2reco[(parts[1], parts[2])] = parts[0]

    utt2spk_writer = open(args.utt2spk, 'w')
    segments_writer = open(args.segments, 'w')
    for line in open(args.rttm_file):
        parts = line.strip().split()
        if parts[0] != "SPEAKER":
            continue

        file_id = parts[1]
        channel = parts[2]

        try:
            reco = file_and_channel2reco[(file_id, channel)]
        except KeyError as e:
            raise Exception("Could not find recording with "
                            "(file_id, channel) "
                            "= ({0},{1}) in {2}: {3}\n".format(
                                file_id, channel,
                                args.reco2file_and_channel, str(e)))

        start_time = float(parts[3])
        end_time = start_time + float(parts[4])

        if args.use_reco_id_as_spkr:
            spkr = reco
        else:
            spkr = parts[7]

        st = int(start_time * 100)
        end = int(end_time * 100)
        utt = "{0}-{1:06d}-{2:06d}".format(spkr, st, end)

        utt2spk_writer.write("{0} {1}\n".format(utt, spkr))
        segments_writer.write("{0} {1} {2:7.2f} {3:7.2f}\n".format(
            utt, reco, start_time, end_time))

if __name__ == '__main__':
    main()
