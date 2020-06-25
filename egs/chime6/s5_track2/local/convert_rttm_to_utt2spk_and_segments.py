#! /usr/bin/env python
# Copyright   2019   Vimal Manohar
# Apache 2.0.

"""This script converts an RTTM with
speaker info into kaldi utt2spk and segments"""

import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script converts an RTTM with
        speaker info into kaldi utt2spk and segments""")
    parser.add_argument("--use-reco-id-as-spkr", type=str,
                        choices=["true", "false"], default="false",
                        help="Use the recording ID based on RTTM and "
                        "reco2file_and_channel as the speaker")
    parser.add_argument("--append-reco-id-to-spkr", type=str,
                        choices=["true", "false"], default="false",
                        help="Append recording ID to the speaker ID")

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
    args.append_reco_id_to_spkr = bool(args.append_reco_id_to_spkr == "true")

    if args.use_reco_id_as_spkr:
        if args.append_reco_id_to_spkr:
            raise Exception("Appending recording ID to speaker does not make sense when using --use-reco-id-as-spkr=true")

    return args

def main():
    args = get_args()

    file_and_channel2reco = {}
    utt2spk={}
    segments={}
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
            if args.append_reco_id_to_spkr:
                spkr = reco + "-" + parts[7]
            else:
                spkr = parts[7]

        st = int(start_time * 100)
        end = int(end_time * 100)
        utt = "{0}-{1:06d}-{2:06d}".format(spkr, st, end)
        utt2spk[utt]=spkr
        segments[utt]=(reco, start_time, end_time)

    for uttid_id in sorted(utt2spk):
        utt2spk_writer.write("{0} {1}\n".format(uttid_id, utt2spk[uttid_id]))
        segments_writer.write("{0} {1} {2:7.2f} {3:7.2f}\n".format(
            uttid_id, segments[uttid_id][0], segments[uttid_id][1], segments[uttid_id][2]))

if __name__ == '__main__':
    main()
