// bin/latgen-tracking-mapped.cc

// Copyright 2012  BUT (author: Mirko Hannemann)
//      2012-2013  Johns Hopkins University (author: Daniel Povey)
//           2014  Guoguo Chen

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
//#include "gmm/am-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/lattice-tracking-decoder.h"
//#include "decoder/decodable-am-diag-gmm.h"
#include "decoder/decodable-matrix.h"
#include "util/timer.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Generate lattices using likelihood matrices, using arc lattices from forward path.\n"
        "Usage: latgen-tracking-mapped [options] model-in (fst-in|fsts-rspecifier) loglike-rspecifier"
        " arcs-rspecifier lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    LatticeTrackingDecoderConfig config;
    
    std::string word_syms_filename;
    config.Register(&po);
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");

    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial, "If true, produce output even if end state was not reached.");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 5 || po.NumArgs() > 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        arcs_rspecifier = po.GetArg(4),
        lattice_wspecifier = po.GetArg(5),
        words_wspecifier = po.GetOptArg(6),
        alignment_wspecifier = po.GetOptArg(7);
    
    TransitionModel trans_model;
    //AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      //am_gmm.Read(ki.Stream(), binary);
    }

    // load arc graphs for each utterance to track first pass tokens
    RandomAccessTableReader<fst::VectorFstHolder> arcs_reader(arcs_rspecifier);

    bool determinize = config.determinize_lattice;
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
    int num_success = 0, num_fail = 0;

    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      // Input FST is just one FST, not a table of FSTs.
      VectorFst<StdArc> *decode_fst = fst::ReadFstKaldi(fst_in_str);

      {
        LatticeTrackingDecoder decoder(*decode_fst, config);
    
        for (; !feature_reader.Done(); feature_reader.Next()) {
          std::string utt = feature_reader.Key();
          Matrix<BaseFloat> loglikes (feature_reader.Value());
          feature_reader.FreeCurrent();
          if (loglikes.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_fail++;
            continue;
          }
          
          //DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
          //                                       acoustic_scale);
          DecodableMatrixScaledMapped decodable(trans_model, loglikes, acoustic_scale);

          if (!arcs_reader.HasKey(utt)) {
            KALDI_WARN << "Can't decode utterance " << utt
                       << " because arc graph isn't available.";
            num_fail++;
            continue;
          }
          const fst::VectorFst<fst::StdArc> &arc_graph = arcs_reader.Value(utt);

          double like;
          if (DecodeUtteranceLatticeTracking(
                  decoder, decodable, trans_model, arc_graph, word_syms, utt,
                  acoustic_scale, determinize, allow_partial, &alignment_writer,
                  &words_writer, &compact_lattice_writer, &lattice_writer,
                  &like)) {
            tot_like += like;
            frame_count += loglikes.NumRows();
            num_success++;
          } else num_fail++;
        }
      }
      delete decode_fst; // delete this only after decoder goes out of scope.
    } else { // We have different FSTs for different utterances.
      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);          
      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string utt = fst_reader.Key();
        if (!feature_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no loglikes available.";
          num_fail++;
          continue;
        }
        const Matrix<BaseFloat> &loglikes = feature_reader.Value(utt);
        if (loglikes.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_fail++;
          continue;
        }

        LatticeTrackingDecoder decoder(fst_reader.Value(), config);
        //DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
        //                                       acoustic_scale);
        DecodableMatrixScaledMapped decodable(trans_model, loglikes, acoustic_scale);
                                               
        if (!arcs_reader.HasKey(utt)) {
          KALDI_WARN << "Can't decode utterance " << utt
                     << " because arc graph isn't available.";
          num_fail++;
          continue;
        }
        const fst::VectorFst<fst::StdArc> &arc_graph = arcs_reader.Value(utt);


        double like;
        if (DecodeUtteranceLatticeTracking(
                decoder, decodable, trans_model, arc_graph, word_syms, utt,
                acoustic_scale, determinize, allow_partial, &alignment_writer,
                &words_writer, &compact_lattice_writer, &lattice_writer,
                &like)) {
          tot_like += like;
          frame_count += loglikes.NumRows();
          num_success++;
        } else num_fail++;
      }
    }
      
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
              << frame_count<<" frames.";

    if (word_syms) delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
