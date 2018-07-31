// sgmm2bin/sgmm2-latgen-faster.cc

// Copyright 2009-2012  Saarland University;  Microsoft Corporation;
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

#include <string>
using std::string;

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "sgmm2/am-sgmm2.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "sgmm2/decodable-am-sgmm2.h"
#include "base/timer.h"

namespace kaldi {

// the reference arguments at the beginning are not const as the style guide
// requires, but are best viewed as inputs.
bool ProcessUtterance(LatticeFasterDecoder &decoder,
                      const AmSgmm2 &am_sgmm,
                      const TransitionModel &trans_model,
                      double log_prune,
                      double acoustic_scale,
                      const Matrix<BaseFloat> &features,
                      RandomAccessInt32VectorVectorReader &gselect_reader,
                      RandomAccessBaseFloatVectorReaderMapped &spkvecs_reader,
                      const fst::SymbolTable *word_syms,
                      const std::string &utt,
                      bool determinize,
                      bool allow_partial,
                      Int32VectorWriter *alignments_writer,
                      Int32VectorWriter *words_writer,
                      CompactLatticeWriter *compact_lattice_writer,
                      LatticeWriter *lattice_writer,
                      double *like_ptr) { // puts utterance's like in like_ptr on success.
  using fst::Fst;

  Sgmm2PerSpkDerivedVars spk_vars;
  if (spkvecs_reader.IsOpen()) {
    if (spkvecs_reader.HasKey(utt)) {
      spk_vars.SetSpeakerVector(spkvecs_reader.Value(utt));
      am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
    } else {
      KALDI_WARN << "Cannot find speaker vector for " << utt << ", not decoding this utterance";
      return false; // We could use zero, but probably the user would want to know about this
      // (this would normally be a script error or some kind of failure).
    }
  }
  if (!gselect_reader.HasKey(utt) ||
      gselect_reader.Value(utt).size() != features.NumRows()) {
    KALDI_WARN << "No Gaussian-selection info available for utterance "
               << utt << " (or wrong size)";
  }

  const std::vector<std::vector<int32> > &gselect =
      gselect_reader.Value(utt);
  
  DecodableAmSgmm2Scaled sgmm_decodable(am_sgmm, trans_model, features, gselect,
                                        log_prune, acoustic_scale, &spk_vars);

  return DecodeUtteranceLatticeFaster(
      decoder, sgmm_decodable, trans_model, word_syms, utt, acoustic_scale,
      determinize, allow_partial, alignments_writer, words_writer,
      compact_lattice_writer, lattice_writer, like_ptr);
}

} // end namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;

    const char *usage =
        "Decode features using SGMM-based model.\n"
        "Usage:  sgmm2-latgen-faster [options] <model-in> (<fst-in>|<fsts-rspecifier>) "
        "<features-rspecifier> <lattices-wspecifier> [<words-wspecifier> [<alignments-wspecifier>] ]\n";
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 0.1;
    bool allow_partial = false;
    BaseFloat log_prune = 5.0;
    string word_syms_filename, gselect_rspecifier, spkvecs_rspecifier,
        utt2spk_rspecifier;

    LatticeFasterDecoderConfig decoder_opts;
    decoder_opts.Register(&po);    

    po.Register("acoustic-scale", &acoustic_scale,
        "Scaling factor for acoustic likelihoods");
    po.Register("log-prune", &log_prune,
                "Pruning beam used to reduce number of exp() evaluations.");
    po.Register("word-symbol-table", &word_syms_filename,
        "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "Produce output even when final state was not reached");
    po.Register("gselect", &gselect_rspecifier,
                "rspecifier for precomputed per-frame Gaussian indices.");
    po.Register("spk-vecs", &spkvecs_rspecifier,
                "rspecifier for speaker vectors");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    if (gselect_rspecifier == "")
      KALDI_ERR << "--gselect option is required.";

    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        lattice_wspecifier = po.GetArg(4),
        words_wspecifier = po.GetOptArg(5),
        alignment_wspecifier = po.GetOptArg(6);

    TransitionModel trans_model;
    kaldi::AmSgmm2 am_sgmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }

    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    bool determinize = decoder_opts.determinize_lattice;    
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

    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    RandomAccessBaseFloatVectorReaderMapped spkvecs_reader(spkvecs_rspecifier,
                                                           utt2spk_rspecifier);

    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_err = 0;

    Timer timer;
        
    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) { // a single FST.
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      // It's important that we initialize decode_fst after feature_reader, as it
      // can prevent crashes on systems installed without enough virtual memory.
      // It has to do with what happens on UNIX systems if you call fork() on a
      // large process: the page-table entries are duplicated, which requires a
      // lot of virtual memory.
      Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
      timer.Reset(); // exclude graph loading time.
      
      {
        LatticeFasterDecoder decoder(*decode_fst, decoder_opts);
    
        const std::vector<std::vector<int32> > empty_gselect;

        for (; !feature_reader.Done(); feature_reader.Next()) {
          string utt = feature_reader.Key();
          const Matrix<BaseFloat> &features(feature_reader.Value());
          if (features.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_err++;
            continue;
          }
          double like;
          if (ProcessUtterance(decoder, am_sgmm, trans_model, log_prune, acoustic_scale,
                               features, gselect_reader, spkvecs_reader, word_syms,
                               utt, determinize, allow_partial,
                               &alignment_writer, &words_writer, &compact_lattice_writer,
                               &lattice_writer, &like)) {
            tot_like += like;
            frame_count += features.NumRows();
            KALDI_LOG << "Log-like per frame for utterance " << utt << " is "
                      << (like / features.NumRows()) << " over "
                      << features.NumRows() << " frames.";
            num_success++;
          } else { num_err++; }
        }
      }
      delete decode_fst; // only safe to do this after decoder goes out of scope.
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
        LatticeFasterDecoder decoder(fst_reader.Value(), decoder_opts);
        double like;

        if (ProcessUtterance(decoder, am_sgmm, trans_model, log_prune, acoustic_scale,
                             features, gselect_reader, spkvecs_reader, word_syms,
                             utt, determinize, allow_partial,
                             &alignment_writer, &words_writer, &compact_lattice_writer,
                             &lattice_writer, &like)) {
          tot_like += like;
          frame_count += features.NumRows();
          KALDI_LOG << "Log-like per frame for utterance " << utt << " is "
                    << (like / features.NumRows()) << " over "
                    << features.NumRows() << " frames.";
          num_success++;
        } else { num_err++; }
      }
    }
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_err;
    KALDI_LOG << "Overall log-likelihood per frame = " << (tot_like/frame_count)
              << " over " << frame_count << " frames.";

    delete word_syms;
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


