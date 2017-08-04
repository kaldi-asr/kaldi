#! /usr/bin/env python

import argparse
import logging
import sys

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
    parser = argparse.ArgumentParser(
        description="""This script converts CTM to per-frame weights by the word
        posteriors in the CTM as the weights.""")

    parser.add_argument("--frame-shift", type=float, default=0.01,
                        help="Frame shift value in seconds")
    parser.add_argument("--default-weight", type=float, default=1.0,
                        help="Default weight on silence frames")
    parser.add_argument("segments_in", type=str, help="Input segments file")
    parser.add_argument("ctm_in", type=str, help="Input utterance-level CTM "
                        "file i.e. the first column has utterance-ids")
    parser.add_argument("weights_out", type=str, help="Output per-frame "
                        "weights vector written in Kaldi text archive format")

    args = parser.parse_args()

    return args


def run(args):
    utt2num_frames = {}
    with common_lib.smart_open(args.segments_in) as segments_reader:
        for line in segments_reader.readlines():
            parts = line.strip().split()
            if len(parts) not in [4, 5]:
                raise RuntimeError("Invalid line {0} in segments file {1}"
                                   "".format(line.strip(), args.segments_in))
            utt2num_frames[parts[0]] = int((float(parts[3]) - float(parts[2]))
                                           / args.frame_shift + 0.5)

    num_utt = 0
    with common_lib.smart_open(args.ctm_in) as ctm_reader, \
            common_lib.smart_open(args.weights_out, 'w') as weights_writer:
        prev_utt = None
        weights = []
        for line in ctm_reader.readlines():
            parts = line.strip().split()
            if len(parts) not in [5, 6]:
                raise RuntimeError("Invalid line {0} in CTM file {1}"
                                   "".format(line.strip(), args.ctm_in))

            utt = parts[0]
            if utt != prev_utt:
                if prev_utt is not None:
                    assert len(weights) >= utt2num_frames[prev_utt]
                    common_lib.write_vector_ascii(weights_writer, weights,
                                                  key=prev_utt)
                weights = [args.default_weight for x in
                           range(utt2num_frames[utt])]

            start_time = float(parts[2])
            dur = float(parts[3])
            prob = 1.0 if len(parts) == 5 else float(parts[5])

            start_frame = int(start_time / args.frame_shift + 0.5)
            length = int(dur / args.frame_shift)

            if len(weights) < start_frame + length:
                weights.extend([args.default_weight for x in
                                   range(len(weights), start_frame + length)])
                for x in range(start_frame, start_frame + length):
                    weights[x] = prob

            assert len(weights) >= start_frame + length
            prev_utt = utt
            num_utt += 1
        assert len(weights) >= utt2num_frames[prev_utt]
        common_lib.write_vector_ascii(weights_writer, weights,
                                      key=prev_utt)

    if num_utt == 0:
        raise RuntimeError("Failed to process any utterances")


def main():
    args = get_args()
    run(args)


if __name__ == "__main__":
    main()
