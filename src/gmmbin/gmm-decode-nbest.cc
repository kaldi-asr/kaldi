// gmmbin/gmm-decode-nbest.cc

// Copyright 2009-2011  Microsoft Corporation, Mirko Hannemann

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
#include "hmm/transition-model.h"
#include "fst/fstlib.h"
#include "fstext/fstext-lib.h"
#include "decoder/nbest-decoder.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "util/timer.h"
#include "lat/kaldi-lattice.h" // for CompactLatticeArc
#include "fstext/lattice-utils.h" // for ConvertLattice

using namespace kaldi;

fst::Fst<fst::StdArc> *ReadNetwork(std::string filename) {
  // read decoding network FST
  Input ki(filename); // use ki.Stream() instead of is.
  if (!ki.Stream().good()) KALDI_ERR << "Could not open decoding-graph FST "
                                      << filename;

  fst::FstHeader hdr;
  if (!hdr.Read(ki.Stream(), "<unknown>")) {
    KALDI_ERR << "Reading FST: error reading FST header.";
  }
  if (hdr.ArcType() != fst::StdArc::Type()) {
    KALDI_ERR << "FST with arc type " << hdr.ArcType() << " not supported.";
  }
  fst::FstReadOptions ropts("<unspecified>", &hdr);

  fst::Fst<fst::StdArc> *decode_fst = NULL;

  if (hdr.FstType() == "vector") {
    decode_fst = fst::VectorFst<fst::StdArc>::Read(ki.Stream(), ropts);
  } else if (hdr.FstType() == "const") {
    decode_fst = fst::ConstFst<fst::StdArc>::Read(ki.Stream(), ropts);
  } else {
    KALDI_ERR << "Reading FST: unsupported FST type: " << hdr.FstType();
  }
  if (decode_fst == NULL) { // fst code will warn.
    KALDI_ERR << "Error reading FST (after reading header).";
    return NULL;
  } else {
    return decode_fst;
  }
}


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;

    const char *usage =
      "Decode features using GMM-based model, producing N-best lattice output.\n"
      "Note: this program was mainly intended to validate the lattice generation\n"
      "algorithm and is not very useful; in general, processing the\n"
      "lattices into n-best lists will be more efficient.\n"
      "Usage:\n"
      " gmm-decode-nbest [options] <model-in> <fst-in> <features-rspecifier> "
        "<nbest-lattice-wspecifier> <words-wspecifier> [<alignments-wspecifier>]\n";
    ParseOptions po(usage);
    bool allow_partial = true;
    BaseFloat acoustic_scale = 0.1;
    
    std::string word_syms_filename;
    NBestDecoderOptions decoder_opts;
    decoder_opts.Register(&po, true);  // true == include obscure settings.
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "Produce output even when final state was not reached");
    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_in_filename = po.GetArg(2),
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

    CompactLatticeWriter compact_lattice_writer;
    if (!compact_lattice_writer.Open(lattice_wspecifier)) {
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;
    }

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    // It's important that we initialize decode_fst after feature_reader, as it
    // can prevent crashes on systems installed without enough virtual memory.
    // It has to do with what happens on UNIX systems if you call fork() on a
    // large process: the page-table entries are duplicated, which requires a
    // lot of virtual memory.
    fst::Fst<fst::StdArc> *decode_fst = ReadNetwork(fst_in_filename);

    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;
    NBestDecoder decoder(*decode_fst, decoder_opts);

    Timer timer;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      Matrix<BaseFloat> features (feature_reader.Value());
      feature_reader.FreeCurrent();
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << key;
        num_fail++;
        continue;
      }

      DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                             acoustic_scale);
      decoder.Decode(&gmm_decodable);

      fst::VectorFst<CompactLatticeArc> decoded;  // output FST.
      bool was_final;
      int32 nbest;
      BaseFloat nbest_beam;
      if (decoder.GetNBestLattice(&decoded, &was_final, &nbest, &nbest_beam)) {
        if (!was_final) {
          if (allow_partial) {
            KALDI_WARN << "Decoder did not reach end-state, "
               << "outputting partial traceback since --allow-partial=true";
          } else {
            KALDI_WARN << "Decoder did not reach end-state, "
               << "output partial traceback with --allow-partial=true";
            num_fail++;
            KALDI_WARN << "Did not successfully decode utterance " << key
                   << ", len = " << features.NumRows();
            continue; // next utterance
          }
        }
        num_success++;
        KALDI_LOG << "retrieved:" << nbest << " tokens, effective beam:" << nbest_beam;

//        std::cout << "n-best paths:\n";
//        fst::FstPrinter<CompactLatticeArc> fstprinter(decoded, NULL, NULL, NULL, false, true);
//        fstprinter.Print(&std::cout, "standard output");

        if (acoustic_scale != 0.0) // We'll write the lattice without acoustic scaling
          fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &decoded);
        compact_lattice_writer.Write(key, decoded);

        fst::VectorFst<CompactLatticeArc> decoded1;
        ShortestPath(decoded, &decoded1);
        fst::VectorFst<LatticeArc> utterance;
        ConvertLattice(decoded1, &utterance, true);

        std::vector<int32> alignment;
        std::vector<int32> words;
        LatticeWeight weight;
        frame_count += features.NumRows();

        GetLinearSymbolSequence(utterance, &alignment, &words, &weight);

        words_writer.Write(key, words);
        if (alignment_writer.IsOpen())
          alignment_writer.Write(key, alignment);
        if (word_syms != NULL) {
          std::cerr << key << ' ';
          for (size_t i = 0; i < words.size(); i++) {
            std::string s = word_syms->Find(words[i]);
            if (s == "")
              KALDI_ERR << "Word-id " << words[i] <<" not in symbol table.";
            std::cerr << s << ' ';
          }
          std::cerr << '\n';
        }
        BaseFloat like = -(weight.Value1() - weight.Value2());
        // KALDI_LOG << "final weight:" << weight.Value1() << "," << weight.Value2();
        tot_like += like;
        KALDI_LOG << "Log-like per frame for utterance " << key << " is "
                  << (like / features.NumRows()) << " over "
                  << features.NumRows() << " frames.";
      } else {
        num_fail++;
        KALDI_WARN << "Did not successfully decode utterance " << key
                   << ", len = " << features.NumRows();
      }
    }

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
              << frame_count<<" frames.";

    if (word_syms) delete word_syms;    
    delete decode_fst;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


