#! /usr/bin/env python

# Copyright 2017  Vimal Manohar
# Apache 2.0

import argparse
import logging
import numpy as np
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
        speech activity detection network.""")

    parser.add_argument("--silence-words", type=argparse.FileType('r'),
                        required=True,
                        help="File containing a list of words that will be "
                        "treated as silence")
    parser.add_argument("--garbage-words", type=argparse.FileType('r'),
                        required=True,
                        help="File containing a list of words that will be "
                        "treated as garbage class")
    parser.add_argument("--max-phone-length", type=int, default=50,
                        help="""Maximum number of frames allowed for a
                        word containing a single phone
                        above which the arc is treated as garbage.""")

    parser.add_argument("arc_info", type=argparse.FileType('r'),
                        help="Arc info file (output of lattice-arc-post")
    parser.add_argument("targets_file", type=argparse.FileType('w'),
                        help="File to write targets matrix archive in text "
                        "format")
    args = parser.parse_args()
    return args


def run(args):
    silence_words = {}
    for line in args.silence_words:
        silence_words[line.strip().split()[0]] = 1
    args.silence_words.close()

    if len(silence_words) == 0:
        raise RuntimeError("Could not find any words in {silence}"
                           "".format(silence=args.silence_words.name))

    garbage_words = {}
    for line in args.garbage_words:
        word = line.strip().split()[0]
        if word in silence_words:
            raise RuntimeError("Word '{word}' is in both {silence} "
                               "and {garbage}".format(
                                   word=word,
                                   silence=args.silence_words.name,
                                   garbage=args.garbage_words.name))
        garbage_words[word] = 1
    args.garbage_words.close()

    if len(garbage_words) == 0:
        raise RuntimeError("Could not find any words in {garbage}"
                           "".format(garbage=args.garbage_words.name))

    num_utts = 0
    num_err = 0
    targets = np.array([])
    prev_utt = ""
    for line in args.arc_info:
        try:
            parts = line.strip().split()
            utt = parts[0]

            if utt != prev_utt:
                if prev_utt != "":
                    if targets.shape[0] > 0:
                        num_utts += 1
                        common_lib.write_matrix_ascii(args.targets_file, targets,
                                                      key=prev_utt)
                    else:
                        num_err += 1
                prev_utt = utt
                targets = np.array([])

            start_frame = int(parts[1])
            num_frames = int(parts[2])
            post = float(parts[3])
            word = parts[4]

            num_phones = 0
            if len(parts) > 5:
                if "," in parts[5]:
                    num_phones = len(parts) - 6
                else:
                    num_phones = len(parts) - 5

            if start_frame + num_frames > targets.shape[0]:
                targets.resize(start_frame + num_frames, 3)

            if word in silence_words:
                targets[start_frame:(start_frame + num_frames), 0] += post
            elif word in garbage_words:
                targets[start_frame:(start_frame + num_frames), 2] += post
            else:
                if num_phones == 1:
                    if num_frames > args.max_phone_length:
                        targets[start_frame:(start_frame + num_frames), 2] += post
                targets[start_frame:(start_frame + num_frames), 1] += post
        except Exception:
            logger.error("Failed to process line {line} in {f}"
                         "".format(line=line.strip(), f=args.arc_info.name))
            logger.error("len(targets) = {l}".format(l=len(targets)))
            raise

    if prev_utt != "":
        if len(targets) > 0:
            num_utts += 1
            common_lib.write_matrix_ascii(args.targets_file, targets,
                                          key=prev_utt)
        else:
            num_err += 1

    args.targets_file.close()

    logger.info("Wrote {num_utts} targets; failed with {num_err}"
                "".format(num_utts=num_utts, num_err=num_err))
    if num_utts == 0 or num_err >= num_utts / 2:
        raise RuntimeError


def main():
    args = get_args()

    try:
        run(args)
    except Exception:
        logger.error("Script failed; traceback = ", exc_info=True)
        raise SystemExit(1)
    finally:
        for f in [args.arc_info, args.targets_file,
                  args.silence_words, args.garbage_words]:
            if f is not None:
                f.close()


if __name__ == "__main__":
    main()
