#!/usr/bin/env python

# Copyright 2017  Vimal Manohar
# Apache 2.0

import argparse
import sys
sys.path.insert(0, 'steps')

import libs.common as common_lib

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script writes to stdout a transformation matrix
    to convert a 3x1 probability vector to a
    2x1 pseudo-likelihood vector by first dividing by 3x1 priors vector.""")

    parser.add_argument("--priors", type=str, default=None,
                        action=common_lib.NullstrToNoneAction,
                        help="Priors vector used to remove the priors from "
                        "the neural network output posteriors to "
                        "convert them to likelihoods")

    parser.add_argument("--sil-in-speech-weight", type=float,
                        default=0.0,
                        help="The fraction of silence probability "
                        "to add to speech")
    parser.add_argument("--speech-in-sil-weight", type=float,
                        default=0.0,
                        help="The fraction of speech probability "
                        "to add to silence")
    parser.add_argument("--garbage-in-speech-weight", type=float,
                        default=0.0,
                        help="The fraction of garbage probability "
                        "to add to speech")
    parser.add_argument("--garbage-in-sil-weight", type=float,
                        default=0.0,
                        help="The fraction of garbage probability "
                        "to add to silence")
    parser.add_argument("--sil-scale", type=float,
                        default=1.0, help="""Scale on the silence probability
                        (make this more than one to encourage
                        decoding silence).""")

    args = parser.parse_args()

    return args


def run(args):
    priors = [[1.0, 1.0, 1.0]]
    if args.priors is not None:
        priors = common_lib.read_matrix_ascii(args.priors)
        if len(priors) != 0 and len(priors[0]) != 3:
            raise RuntimeError("Invalid dimension for priors {0}"
                               "".format(priors))

    priors_sum = sum(priors[0])
    sil_prior = priors[0][0] / priors_sum
    speech_prior = priors[0][1] / priors_sum
    garbage_prior = priors[0][2] / priors_sum

    transform_mat = [[args.sil_scale / sil_prior,
                      args.speech_in_sil_weight / speech_prior,
                      args.garbage_in_sil_weight / garbage_prior],
                     [args.sil_in_speech_weight / sil_prior,
                      1.0 / speech_prior,
                      args.garbage_in_speech_weight / garbage_prior]]

    common_lib.write_matrix_ascii(sys.stdout, transform_mat)


def main():
    args = get_args()
    run(args)


if __name__ == "__main__":
    main()
