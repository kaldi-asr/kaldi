// nnet2bin/nnet-latgen-faster-parallel.cc

// Copyright 2009-2013   Microsoft Corporation
//                       Johns Hopkins University (author: Daniel Povey)
//                2014   Guoguo Chen

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
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/lattice-faster-decoder.h"
#include "nnet2/decodable-am-nnet.h"
#include "util/timer.h"
#include "thread/kaldi-task-sequence.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Generate lattices using neural net model.\n"
        "Usage: nnet-latgen-faster-parallel [options] <nnet-in> <fst-in|fsts-rspecifier> <features-rspecifier>"
        " <lattice-wspecifier> [ <words-wspecifier> [<alignments-wspecifier>] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    LatticeFasterDecoderConfig config;
    TaskSequencerConfig sequencer_config; // has --num-threads option
    std::string spkvecs_rspecifier, utt2spk_rspecifier;
    
    std::string word_syms_filename;
    sequencer_config.Register(&po);
    config.Register(&po);
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial, "If true, produce output even if end state was not reached.");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Rspecifier for a vector that describes each speaker; "
                "only needed if the neural net was trained this way.");
    po.Register("utt2spk", &utt2spk_rspecifier, "Rspecifier for map from utterance to speaker; only relevant "
                "in conjunction with the --spk-vecs option.");
    
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
    AmNnet am_nnet;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;

    TaskSequencer<DecodeUtteranceLatticeFasterClass> sequencer(sequencer_config);
    
    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    RandomAccessBaseFloatVectorReaderMapped spkvecs_reader(spkvecs_rspecifier,
                                                           utt2spk_rspecifier);
    // We support reading in a vector to describe each speaker, if the neural
    // net requires this (i.e. it was trained with this).
    
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_done = 0, num_err = 0;
    VectorFst<StdArc> *decode_fst = NULL;
    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

      decode_fst = fst::ReadFstKaldi(fst_in_str);

      {
    
        for (; !feature_reader.Done(); feature_reader.Next()) {
          std::string utt = feature_reader.Key();
          const Matrix<BaseFloat> &features (feature_reader.Value());
          if (features.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_err++;
            continue;
          }
          Vector<BaseFloat> spk_info;
          if (spkvecs_reader.IsOpen()) {
            if (spkvecs_reader.HasKey(utt)) {
              spk_info = spkvecs_reader.Value(utt);
            } else {
              KALDI_WARN << "Cannot find speaker vector for " << utt
                         << " (skipping this utterance).";
              continue;
            }
          }
          bool pad_input = true;
          DecodableAmNnetParallel *nnet_decodable = new DecodableAmNnetParallel(
              trans_model, am_nnet,
              new CuMatrix<BaseFloat>(features),
              new CuVector<BaseFloat>(spk_info),
              pad_input, acoustic_scale);

          LatticeFasterDecoder *decoder = new LatticeFasterDecoder(*decode_fst,
                                                                   config);

          DecodeUtteranceLatticeFasterClass *task =
              new DecodeUtteranceLatticeFasterClass(
                  decoder, nnet_decodable, // takes ownership of these two.
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
        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_err++;
          continue;
        }

        // This constructor of LatticeFasterDecoder takes ownership of the FST.
        LatticeFasterDecoder *decoder =
            new LatticeFasterDecoder(config, fst_reader.Value().Copy());

        Vector<BaseFloat> spk_info;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(utt)) {
            spk_info = spkvecs_reader.Value(utt);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << utt
                       << " (skipping this utterance).";
            continue;
          }
        }
        bool pad_input = true;
        DecodableAmNnetParallel *nnet_decodable = new DecodableAmNnetParallel(
            trans_model, am_nnet,
            new CuMatrix<BaseFloat>(features),
            new CuVector<BaseFloat>(spk_info),
            pad_input, acoustic_scale);

        DecodeUtteranceLatticeFasterClass *task =
            new DecodeUtteranceLatticeFasterClass(
                decoder, nnet_decodable, // takes ownership of these two.
                trans_model, word_syms, utt, acoustic_scale, determinize,
                allow_partial, &alignment_writer, &words_writer,
                &compact_lattice_writer, &lattice_writer,
                &tot_like, &frame_count, &num_done, &num_err, NULL);

        sequencer.Run(task); // takes ownership of "task",
                             // and will delete it when done.
      }
    }
    sequencer.Wait(); // Waits for all tasks to be done.
    if (decode_fst != NULL) delete decode_fst;   
    
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_done << " utterances, failed for "
              << num_err;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
              << frame_count<<" frames.";

    if (word_syms) delete word_syms;
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
