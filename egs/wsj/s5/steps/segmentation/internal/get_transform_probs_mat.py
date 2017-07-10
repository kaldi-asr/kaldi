#!/usr/bin/env python

# Copyright 2017  Vimal Manohar
# Apache 2.0

import argparse
import sys
sys.path.insert(0, 'steps')

import libs.common as common_lib

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script gets a transformation matrix
    to convert a 3x1 probability vector to a
    2x1 pseudo-likelihood vector after dividing by priors""")

    parser.add_argument("--sil-prior", type=str, default=1,
                        help="Prior on the silence output as "
                        "learned by the network")
    parser.add_argument("--speech-prior", type=str, default=1,
                        help="Prior on the speech output as "
                        "learned by the network")
    parser.add_argument("--garbage-prior", type=str, default=1,
                        help="Prior on the garbage output as "
                        "learned by the network")

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

    args = parser.parse_args()

    return args


def run(args):
    priors = [[1, 1]]
    if args.priors is not None:
        priors = common_lib.read_matrix_ascii(args.priors)
        if len(priors) != 0 and len(priors[0]) != 2:
            raise RuntimeError("Invalid dimension for priors {0}"
                               "".format(priors))

    priors_sum = (args.sil_prior + args.speech_prior
                 + args.garbage_prior)
    garbage_prior = args.garbage_prior / priors_sum
    sil_prior = args.sil_prior / priors_sum
    speech_prior = args.speech_prior / priors_sum

    transform_mat = [[1.0 / sil_prior,
                      args.speech_in_sil_weight / speech_prior,
                      args.garbage_in_sil_weight / garbage_prior],
                     [1.0 / sil_prior,
                      args.sil_in_speech_weight / speech_prior,
                      args.garbage_in_speech_weight / garbage_prior]]

    common_lib.write_matrix_ascii(sys.stdout, transform_mat)


def main():
    args = get_args()
    run(args)


if __name__ == "__main__":
    main()
