#!/usr/bin/env python

# Copyright 2017  Vimal Manohar
# Apache 2.0

"""
This script converts arc-info into targets for training
speech activity detection network. The output is a matrix archive
with each matrix having 3 columns -- silence, speech and garbage.
The posterior probabilities of the phones of each of the classes are
summed up to get the target matrix values.
"""

import argparse
import logging
import sys

sys.path.insert(0, 'steps')
import libs.common as common_lib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    parser = argparse.ArgumentParser(
        description="""This script converts arc-info into targets for training
        speech activity detection network. The output is a matrix archive
        with each matrix having 3 columns -- silence, speech and garbage.
        The posterior probabilities of the phones of each of the classes are
        summed up to get the target matrix values.
        """)

    parser.add_argument("--silence-phones", type=str,
                        required=True,
                        help="File containing a list of phones that will be "
                        "treated as silence")
    parser.add_argument("--garbage-phones", type=str,
                        required=True,
                        help="File containing a list of phones that will be "
                        "treated as garbage class")
    parser.add_argument("--max-phone-length", type=int, default=50,
                        help="""Maximum number of frames allowed for a speech
                        phone above which the arc is treated as garbage.""")

    parser.add_argument("arc_info", type=str,
                        help="Arc info file (output of lattice-arc-post). "
                        "See the help for lattice-arc-post for information "
                        "about the format of this input.")
    parser.add_argument("targets_file", type=str,
                        help="File to write targets matrix archive in text "
                        "format")
    args = parser.parse_args()
    return args


def run(args):
    silence_phones = {}
    with common_lib.smart_open(args.silence_phones) as silence_phones_fh:
        for line in silence_phones_fh:
            silence_phones[line.strip().split()[0]] = 1

    if len(silence_phones) == 0:
        raise RuntimeError("Could not find any phones in {silence}"
                           "".format(silence=args.silence_phones))

    garbage_phones = {}
    with common_lib.smart_open(args.garbage_phones) as garbage_phones_fh:
        for line in garbage_phones_fh:
            word = line.strip().split()[0]
            if word in silence_phones:
                raise RuntimeError("Word '{word}' is in both {silence} "
                                   "and {garbage}".format(
                                       word=word,
                                       silence=args.silence_phones,
                                       garbage=args.garbage_phones))
            garbage_phones[word] = 1

    if len(garbage_phones) == 0:
        raise RuntimeError("Could not find any phones in {garbage}"
                           "".format(garbage=args.garbage_phones))

    num_utts = 0
    num_err = 0
    targets = []
    prev_utt = ""

    with common_lib.smart_open(args.arc_info) as arc_info_reader, \
            common_lib.smart_open(args.targets_file, 'w') as targets_writer:
        for line in arc_info_reader:
            try:
                parts = line.strip().split()
                utt = parts[0]

                if utt != prev_utt:
                    if prev_utt != "":
                        if len(targets) > 0:
                            num_utts += 1
                            common_lib.write_matrix_ascii(
                                targets_writer, targets, key=prev_utt)
                        else:
                            num_err += 1
                    prev_utt = utt
                    targets = []

                start_frame = int(parts[1])
                num_frames = int(parts[2])
                post = float(parts[3])
                phone = parts[4]

                if start_frame + num_frames > len(targets):
                    for t in range(len(targets), start_frame + num_frames):
                        targets.append([0, 0, 0])
                    assert start_frame + num_frames == len(targets)

                for t in range(start_frame, start_frame + num_frames):
                    if phone in silence_phones:
                        targets[t][0] += post
                    elif num_frames > args.max_phone_length:
                        targets[t][2] += post
                    elif phone in garbage_phones:
                        targets[t][2] += post
                    else:
                        targets[t][1] += post
            except Exception:
                logger.error("Failed to process line {line} in {f}"
                             "".format(line=line.strip(), f=args.arc_info))
                logger.error("len(targets) = {l}".format(l=len(targets)))
                raise

    if prev_utt != "":
        if len(targets) > 0:
            num_utts += 1
            common_lib.write_matrix_ascii(args.targets_file, targets,
                                          key=prev_utt)
        else:
            num_err += 1

    logger.info("Wrote {num_utts} targets; failed with {num_err}"
                "".format(num_utts=num_utts, num_err=num_err))
    if num_utts == 0 or num_err >= num_utts / 2:
        raise RuntimeError


def main():
    args = get_args()

    try:
        run(args)
    except Exception:
        raise


if __name__ == "__main__":
    main()
