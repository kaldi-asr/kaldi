#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0
#
#
'''
This file is adapted from src/bin/latgen-faster-mapped.cc.

Note that there is no **mapped** in the filename since we
do not use a transition model for mapping pdf ids to transition ids.

Since this Python script is just a thin wrapper about the C++ code,
there should not be any performance problem.

You can write another `src/bin/latgen-faster.cc` if you are
still worrying about the performance.
'''
#
#

# TODO(fangjun): refer to src/bin/latgen-faster-mapped parallel.cc to
# implement latgen-faster-parallel.py

import sys

import kaldi
from kaldi import fst


def main():
    usage = kaldi.StringArg('''\
Generate lattices, reading log-likelihoods as matrices

Usage: latgen-faster [options]  fst-rxfilename loglikes-rspecifier \
lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]
''')

    allow_partial = kaldi.BoolArg(False)
    acoustic_scale = kaldi.FloatArg(0.1)
    word_syms_filename = kaldi.StringArg()

    config = kaldi.LatticeFasterDecoderConfig()

    po = kaldi.ParseOptions(usage)

    config.Register(po)

    po.Register('acoustic-scale', acoustic_scale,
                'Scaling factor for acoustic likelihoods')

    po.Register('word-symbol-table', word_syms_filename,
                'Symbol table for words [for debug output]')

    po.Register('allow-partial', allow_partial,
                'If true, produce output even if end state was not reached.')

    po.Read(sys.argv)

    if po.NumArgs() < 3 or po.NumArgs() > 5:
        po.PrintUsage()
        sys.exit(1)

    fst_in_str = po.GetArg(1)
    log_likes_rspecifier = po.GetArg(2)
    lattice_wspecifier = po.GetArg(3)
    words_wspecifier = po.GetOptArg(4)
    alignment_wspecifier = po.GetOptArg(5)

    determinize = config.determinize_lattice
    compact_lattice_writer = kaldi.CompactLatticeWriter()
    lattice_writer = kaldi.LatticeWriter()

    if determinize:
        assert compact_lattice_writer.Open(lattice_wspecifier) == True
    else:
        assert lattice_writer.Open(lattice_wspecifier) == True

    words_writer = kaldi.IntVectorWriter(words_wspecifier)
    alignments_writer = kaldi.IntVectorWriter(alignment_wspecifier)

    word_syms = fst.SymbolTable()

    if word_syms_filename:
        word_syms = fst.SymbolTable.ReadText(word_syms_filename.value)

    # TODO(fangjun): support a table of FSTs

    tot_like = 0.0
    frame_count = 0
    num_success = 0
    num_fail = 0

    loglike_reader = kaldi.SequentialMatrixReader(log_likes_rspecifier)

    # WARNING(fangjun): fst_in_str has to be a **const** fst.
    # If it is a vector fst, you will get an error
    # while creating the subsequent LatticeFasterDecoder.
    tlg_fst = fst.ReadFstKaldiGeneric(fst_in_str)

    decoder = kaldi.LatticeFasterDecoder(tlg_fst, config)

    trans_model = kaldi.TransitionModel()  # a dummy transition model

    for key, value in loglike_reader:
        if value.NumRows() == 0:
            print('zero length utterance: {}'.format(key))
            num_fail += 1
            continue

        decodable = kaldi.DecodableMatrixScaled(likes=value,
                                                scale=acoustic_scale.value)

        is_succeeded, likelihood = kaldi.DecodeUtteranceLatticeFaster(
            decoder=decoder,
            decodable=decodable,
            trans_model=trans_model,
            word_syms=word_syms,
            utt=key,
            acoustic_scale=acoustic_scale.value,
            determinize=determinize,
            allow_partial=allow_partial.value,
            alignments_writer=alignments_writer,
            words_writer=words_writer,
            compact_lattice_writer=compact_lattice_writer,
            lattice_writer=lattice_writer)

        if is_succeeded:
            tot_like += likelihood
            frame_count += value.NumRows()
            num_success += 1
        else:
            num_fail += 1

    print('Done {num_success} utterances, failed for {num_fail}'.format(
        num_success=num_success, num_fail=num_fail))

    print('Overall log-likelihood per frame is {} over {} frames'.format(
        tot_like / frame_count, frame_count))

    if num_success != 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
