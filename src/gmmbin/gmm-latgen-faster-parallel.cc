// gmmbin/gmm-latgen-faster-parallel.cc

// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Daniel Povey)
//                2014  Guoguo Chen

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "base/timer.h"
#include "feat/feature-functions.h"  // feature reversal
#include "util/kaldi-thread.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::Fst;
    using fst::StdArc;

    const char *usage =
        "Decode features using GMM-based model.  Uses multiple decoding threads,\n"
        "but interface and behavior is otherwise the same as gmm-latgen-faster\n"
        "Usage: gmm-latgen-faster-parallel [options] model-in (fst-in|fsts-rspecifier) "
        "features-rspecifier lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    BaseFloat log_sum_exp_prune = 0.0;
    LatticeFasterDecoderConfig latgen_config;
    TaskSequencerConfig sequencer_config; // has --num-threads option

    std::string word_syms_filename;
    latgen_config.Register(&po);
    sequencer_config.Register(&po);
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("log-sum-exp-prune", &log_sum_exp_prune,
                "If >0, pruning parameter to minimize exp()'s.  Suggest 3 to 5; "
                "larger is more exact.");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "If true, produce output even if end state was not reached.");

    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        lattice_wspecifier = po.GetArg(4),
        words_wspecifier = po.GetOptArg(5),
        alignment_wspecifier = po.GetOptArg(6);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    bool determinize = latgen_config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_done = 0, num_err = 0;
    Fst<StdArc> *decode_fst = NULL; // only used if there is a single
                                          // decoding graph.

    TaskSequencer<DecodeUtteranceLatticeFasterClass> sequencer(sequencer_config);

    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      // Input FST is just one FST, not a table of FSTs.

      decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
      timer.Reset();

      {
        for (; !feature_reader.Done(); feature_reader.Next()) {
          std::string utt = feature_reader.Key();
          Matrix<BaseFloat> *features =
              new Matrix<BaseFloat>(feature_reader.Value());
          feature_reader.FreeCurrent();
          if (features->NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_err++;
            delete features;
            continue;
          }

          LatticeFasterDecoder *decoder = new LatticeFasterDecoder(*decode_fst,
                                                                   latgen_config);
          // takes ownership of "features"
          DecodableAmDiagGmmScaled *gmm_decodable =
              new DecodableAmDiagGmmScaled(am_gmm, trans_model,
                                           acoustic_scale,
                                           log_sum_exp_prune,
                                           features);

          DecodeUtteranceLatticeFasterClass *task =
              new DecodeUtteranceLatticeFasterClass(
                  decoder, gmm_decodable, // takes ownership of these two.
                  trans_model, word_syms, utt, acoustic_scale, determinize,
                  allow_partial, &alignment_writer, &words_writer,
                  &compact_lattice_writer, &lattice_writer,
                  &tot_like, &frame_count, &num_done, &num_err, NULL);

          sequencer.Run(task); // takes ownership of "task",
          // and will delete it when done.
        }
      }
    } else { // We have different FSTs for different utterances.
      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string utt = fst_reader.Key();
        if (!feature_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no features available.";
          num_err++;
          continue;
        }
        Matrix<BaseFloat> *features = new Matrix<BaseFloat>(
            feature_reader.Value(utt));
        if (features->NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_err++;
          delete features;
          continue;
        }

        // the "decoder" object takes ownership of the new FST object.
        LatticeFasterDecoder *decoder = new LatticeFasterDecoder(
            latgen_config,
            new VectorFst<StdArc>(fst_reader.Value()));

        // The "decodable" object takes ownership of the features.
        DecodableAmDiagGmmScaled *gmm_decodable =
            new DecodableAmDiagGmmScaled(am_gmm, trans_model, acoustic_scale,
                                         log_sum_exp_prune, features);

        DecodeUtteranceLatticeFasterClass *task =
            new DecodeUtteranceLatticeFasterClass(
                decoder, gmm_decodable, // takes ownership of these two.
                trans_model, word_syms, utt, acoustic_scale, determinize,
                allow_partial, &alignment_writer, &words_writer,
                &compact_lattice_writer, &lattice_writer,
                &tot_like, &frame_count, &num_done, &num_err, NULL);
        sequencer.Run(task); // takes ownership of "task",
        // and will delete it when done.
      }
    }
    sequencer.Wait();

    delete decode_fst;

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Decoded with " << sequencer_config.num_threads << " threads.";
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor per thread assuming 100 frames/sec is "
              << (sequencer_config.num_threads * elapsed * 100.0 / frame_count);
    KALDI_LOG << "Done " << num_done << " utterances, failed for "
              << num_err;
    KALDI_LOG << "Overall log-likelihood per frame is "
              << (tot_like/frame_count) << " over "
              << frame_count << " frames.";

    delete word_syms;
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
