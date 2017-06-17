#! /usr/bin/env python

# Copyright 2016  Vimal Manohar
# Apache 2.0

"""Prepares a graph directory with a simple HMM topology for segmentation.
"""

from __future__ import print_function
import argparse
import logging
import math
import os
import sys
import traceback

sys.path.insert(0, 'steps')
import libs.common as common_lib


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    parser = argparse.ArgumentParser(
        description="This script generates a graph directory for decoding with "
        "a simple HMM model.\n"
        "It needs as an input classes_info file with the format:\n"
        "<class-id (1-indexed)> <initial-probability> <self-loop-probability> "
        "<min-duration> <list-of-pairs>,\n"
        "where each pair is <destination-class>:<transition-probability>.\n"
        "destination-class -1 is used to represent final probability.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--transition-scale", type=float, default=1.0,
                        help="""Scale on transition probabilities relative to
                        LM weights""")
    parser.add_argument("--loopscale", type=float, default=0.1,
                        help="""Scale on self-loop log-probabilities relative
                        to LM weights""")

    parser.add_argument("--min-silence-duration", type=float, default=0.03,
                        help="""Minimum duration for silence""")
    parser.add_argument("--min-speech-duration", type=float, default=0.3,
                        help="""Minimum duration for speech""")
    parser.add_argument("--max-speech-duration", type=float, default=10.0,
                        help="""Maximum duration for speech""")
    parser.add_argument("--frame-shift", type=float, default=0.03,
                        help="""Frame shift in seconds""")

    parser.add_argument("--initial-silence-probability", type=float,
                        default=0.5,
                        help="Initial probability for transition into silence")
    parser.add_argument("--sil-to-speech-probability", type=float, default=0.1,
                        help="Transition probability for silence to speech")
    parser.add_argument("--speech-to-sil-probability", type=float, default=0.1,
                        help="Transition probability for speech to silence")
    parser.add_argument("--final-probability", type=float, default=1e-5,
                        help="Final probability")

    parser.add_argument("output_graph", type=argparse.FileType('w'),
                        help="Output graph")
    args = parser.parse_args()

    args.min_states_silence = int(args.min_silence_duration / args.frame_shift
                                  + 0.5)
    args.min_states_speech = int(args.min_speech_duration / args.frame_shift
                                 + 0.5)
    args.max_states_speech = int(args.max_speech_duration / args.frame_shift
                                 + 0.5)

    return args


def print_states(args, file_handle):
    # Initial transition to silence
    print ("0 1 1 1 {0}".format(
                -math.log(args.initial_silence_probability)),
           file=file_handle)
    silence_start_state = 1

    # Silence min duration transitions
    # 1->2, 2->3 and so on until
    # (1 + min_states_silence - 2) -> (1 + min_states_silence - 1)  ...
    for state in range(silence_start_state,
                       silence_start_state + args.min_states_silence - 1):
        print ("{state} {next_state} 1 1 {cost}".format(
                    state=state, next_state=state + 1,
                    cost=-math.log(1.0 - args.final_probability)),
               file=file_handle)
    silence_last_state = silence_start_state + args.min_states_silence - 1

    # Silence self-loop
    print ("{state} {state} 1 1 {cost}".format(
                state=silence_last_state,
                cost=-math.log(1.0 - args.sil_to_speech_probability
                               - args.final_probability)),
           file=file_handle)

    speech_start_state = silence_last_state + 1
    # Initial transition to speech
    print ("0 {state} 2 2 {cost}".format(
                state=speech_start_state,
                cost=-math.log(1.0 - args.initial_silence_probability)),
           file=file_handle)

    # Silence to speech transition
    print ("{sil_state} {speech_state} 2 2 {cost}".format(
                sil_state=silence_last_state,
                speech_state=speech_start_state,
                cost=-math.log(args.sil_to_speech_probability)),
           file=file_handle)

    # Speech min duration
    for state in range(speech_start_state,
                       speech_start_state + args.min_states_speech - 1):
        print ("{state} {next_state} 2 2 {cost}".format(
                    state=state, next_state=state + 1,
                    cost=-math.log(1.0 - args.final_probability)),
               file=file_handle)

    # Speech max duration
    for state in range(speech_start_state + args.min_states_speech - 1,
                       speech_start_state + args.max_states_speech - 1):
        print ("{state} {next_state} 2 2 {cost}".format(
                    state=state, next_state=state + 1,
                    cost=-math.log(1.0 - args.speech_to_sil_probability
                                   - args.final_probability)),
               file=file_handle)

        print ("{state} {sil_state} 1 1 {cost}".format(
                    state=state, sil_state=silence_start_state,
                    cost=-math.log(args.speech_to_sil_probability)),
               file=file_handle)
    speech_last_state = speech_start_state + args.max_states_speech - 1

    print ("{state} {sil_state} 1 1 {cost}".format(
                state=speech_last_state, sil_state=silence_start_state,
                cost=-math.log(1.0 - args.final_probability)))

    for state in range(1, speech_last_state + 1):
        print ("{state} {cost}".format(
                    state=state, cost=-math.log(args.final_probability)),
               file=file_handle)


def main():
    try:
        args = get_args()
        print_states(args, args.output_graph)
    except Exception:
        logger.error("Failed preparing graph")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
