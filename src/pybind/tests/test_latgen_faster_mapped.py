#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0
'''

This file is adapted from src/bin/latgen-faster-mapped.cc
and shows you how to decode in Python with kaldi pybind.

Before running this script, you should have finished `egs/aishell/s10/run.sh`.

If you have not run the above recipe, then you can replace relevant
files, e.g., HCLG.fst, nnet_outut.scp, etc, in this scripts
with your own.

'''

import unittest

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import kaldi
from kaldi import fst


class TestLatGenFasterMapped(unittest.TestCase):

    def test(self):
        usage = 'Generate lattices, reading log-likelihoods as matrices\n'
        ' (model is needed only for the integer mappings in its transition-model)\n'
        po = kaldi.ParseOptions(usage)

        allow_partial = kaldi.BoolArg(False)
        acoustic_scale = kaldi.FloatArg(0.1)
        config = kaldi.LatticeFasterDecoderConfig()

        if not os.path.exists(
                '../../../egs/aishell/s10/exp/chain/graph/HCLG.fst'):
            print('Please execute kaldi/egs/aishell/s10/run.sh first')
            print('and souce path.sh in it before running this script')
            print('Or replace relevant files in this test with your own')
            print('Skip this test')
            return
        os.chdir('../../../egs/aishell/s10')

        # We use ParseOptions here to show you how to parse options in Python
        # with kaldi pybind.
        # You do NOT need to do this way in reality.

        config.Register(po)

        po.Register('acoustic-scale', acoustic_scale,
                    'Scaling factor for acoustic likelihoods')

        po.Register(
            'allow-partial', allow_partial,
            'If true, produce output even if end state was not reached.')

        args = [
            'a.out',
            '--acoustic-scale=1.0',
            '--allow-partial=true',
            '--beam=12.0',
            '--lattice-beam=4.0',
            '--max-active=7000',
            '--max-mem=50000000',
            '--min-active=200',
        ]
        po.Read(args)

        word_syms_filename = 'exp/chain/graph/words.txt'
        trans_model_filename = 'exp/chain/0.trans_mdl'
        hclg_filename = 'exp/chain/graph/HCLG.fst'
        lat_wspecifier = 'ark:|lattice-scale --acoustic-scale=10 ark:- ark:- | gzip -c >lat.gz'

        trans_model = kaldi.read_transition_model(trans_model_filename)

        determinize = config.determinize_lattice
        compact_lattice_writer = kaldi.CompactLatticeWriter()
        lattice_writer = kaldi.LatticeWriter()
        if determinize:
            self.assertTrue(compact_lattice_writer.Open(lat_wspecifier))
        else:
            self.assertTrue(lattice_writer.Open(lat_wspecifier))

        words_writer = kaldi.IntVectorWriter()
        alignments_writer = kaldi.IntVectorWriter()

        word_syms = fst.SymbolTable.ReadText(word_syms_filename)

        nnet_output_scp = 'scp:head -n2 exp/chain/inference/test/nnet_output.scp|'
        loglike_reader = kaldi.SequentialMatrixReader(nnet_output_scp)
        hclg_fst = fst.ReadFstKaldiGeneric(hclg_filename)

        decoder = kaldi.LatticeFasterDecoder(hclg_fst, config)

        num_fail = 0
        for key, value in loglike_reader:
            if value.NumRows() == 0:
                print('zero length utterance: {}'.format(key))
                num_fail += 1
                continue
            decodable = kaldi.DecodableMatrixScaledMapped(
                trans_model, value, acoustic_scale.value)
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


if __name__ == '__main__':
    unittest.main()
'''

NOTE(fangjun): You should see the following output (the decoded results are in Chinese):

kaldi/src/pybind/tests$ ./test_latgen_faster_mapped.py
a.out --acoustic-scale=1.0 --allow-partial=true --beam=12.0 --lattice-beam=4.0 --max-active=7000 --max-mem=50000000 --min-active=200
lattice-scale --acoustic-scale=10 ark:- ark:-
BAC009S0764W0121 甚至 出现 交易 几乎 停滞 的 情况
LOG (a.out[5.5.733~4-4c87]:DecodeUtteranceLatticeFaster():decoder-wrappers.cc:375) Log-like per frame for utterance BAC009S0764W0121 is 1.23619 over 140 frames.
BAC009S0764W0122 一二 线 城市 虽然 也 处于 调整 中
LOG (a.out[5.5.733~4-4c87]:DecodeUtteranceLatticeFaster():decoder-wrappers.cc:375) Log-like per frame for utterance BAC009S0764W0122 is 1.29542 over 137 frames.
LOG (lattice-scale[5.5.733~4-4c87]:main():lattice-scale.cc:107) Done 2 lattices.
.
----------------------------------------------------------------------
Ran 1 test in 0.561s

OK
'''
