// sgmmbin/sgmm-decode-faster.cc

// Copyright 2009-2011  Saarland University;  Microsoft Corporation

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
#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/decodable-am-sgmm.h"
#include "util/timer.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Decode features using SGMM-based model.\n"
        "Usage:  sgmm-decode-faster [options] <model-in> <fst-in> "
        "<features-rspecifier> <words-wspecifier> [alignments-wspecifier]\n";
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 0.1;
    BaseFloat log_prune = 5.0;
    string word_syms_filename, gselect_rspecifier, spkvecs_rspecifier,
        utt2spk_rspecifier;

    FasterDecoderOptions decoder_opts;
    decoder_opts.Register(&po, true);  // true == include obscure settings.
    kaldi::SgmmGselectConfig sgmm_opts;
    sgmm_opts.Register(&po);

    po.Register("acoustic-scale", &acoustic_scale,
        "Scaling factor for acoustic likelihoods");
    po.Register("log-prune", &log_prune,
        "Pruning beam used to reduce number of exp() evaluations.");
    po.Register("word-symbol-table", &word_syms_filename,
        "Symbol table for words [for debug output]");
    po.Register("gselect", &gselect_rspecifier,
                "rspecifier for precomputed per-frame Gaussian indices.");
    po.Register("spk-vecs", &spkvecs_rspecifier,
                "rspecifier for speaker vectors");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_in_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        words_wspecifier = po.GetArg(4),
        alignment_wspecifier = po.GetOptArg(5);

    TransitionModel trans_model;
    kaldi::AmSgmm am_sgmm;
    {
      bool binary;
      Input is(model_in_filename, &binary);
      trans_model.Read(is.Stream(), binary);
      am_sgmm.Read(is.Stream(), binary);
    }

    Int32VectorWriter words_writer(words_wspecifier);
    Int32VectorWriter alignment_writer;
    if (alignment_wspecifier != "")
      if (!alignment_writer.Open(alignment_wspecifier))
        KALDI_ERR << "Failed to open alignments output.";

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") {
      word_syms = fst::SymbolTable::ReadText(word_syms_filename);
      if (!word_syms)
        KALDI_EXIT << "Could not read symbol table from file "
                   << word_syms_filename;
    }

    RandomAccessInt32VectorVectorReader gselect_reader;
    if (!gselect_rspecifier.empty())
      if (!gselect_reader.Open(gselect_rspecifier))
        KALDI_ERR << "Cannot open stream to read gaussian-selection indices";

    RandomAccessTokenReader utt2spk_reader;
    if (!utt2spk_rspecifier.empty())  // per-speaker adaptation
      if (!utt2spk_reader.Open(utt2spk_rspecifier))
        KALDI_ERR << "Could not open the utt2spk map: " << utt2spk_rspecifier;

    RandomAccessBaseFloatVectorReader spkvecs_reader;
    if (!spkvecs_rspecifier.empty())
      if (!spkvecs_reader.Open(spkvecs_rspecifier))
        KALDI_ERR << "Cannot read speaker vectors.";

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    // It's important that we initialize decode_fst after feature_reader, as it
    // can prevent crashes on systems installed without enough virtual memory.
    // It has to do with what happens on UNIX systems if you call fork() on a
    // large process: the page-table entries are duplicated, which requires a
    // lot of virtual memory.
    VectorFst<StdArc> *decode_fst = NULL;
    {
      std::ifstream is(fst_in_filename.c_str(), std::ifstream::binary);
      if (!is.good()) KALDI_EXIT << "Could not open decoding-graph FST "
                                << fst_in_filename;
      decode_fst =
          VectorFst<StdArc>::Read(is, fst::FstReadOptions(fst_in_filename));
      if (decode_fst == NULL)  // fst code will warn.
        exit(1);
    }

    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;
    FasterDecoder decoder(*decode_fst, decoder_opts);

    Timer timer;
    const std::vector<std::vector<int32> > empty_gselect;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      string utt = feature_reader.Key();
      Matrix<BaseFloat> features(feature_reader.Value());
      feature_reader.FreeCurrent();
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_fail++;
        continue;
      }

      string utt_or_spk;
      if (utt2spk_rspecifier.empty()) utt_or_spk = utt;
      else {
        if (!utt2spk_reader.HasKey(utt)) {
          KALDI_WARN << "Utterance " << utt << " not present in utt2spk map; "
                     << "skipping this utterance.";
          num_fail++;
          continue;
        } else {
          utt_or_spk = utt2spk_reader.Value(utt);
        }
      }

      SgmmPerSpkDerivedVars spk_vars;
      if (spkvecs_reader.IsOpen()) {
        if (spkvecs_reader.HasKey(utt_or_spk)) {
          spk_vars.v_s = spkvecs_reader.Value(utt_or_spk);
          am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
        } else {
          KALDI_WARN << "Cannot find speaker vector for " << utt_or_spk;
        }
      }  // else spk_vars is "empty"

      bool has_gselect = false;
      if (gselect_reader.IsOpen()) {
        has_gselect = gselect_reader.HasKey(utt)
                      && gselect_reader.Value(utt).size() == features.NumRows();
        if (!has_gselect)
          KALDI_WARN << "No Gaussian-selection info available for utterance "
                     << utt << " (or wrong size)";
      }
      const std::vector<std::vector<int32> > *gselect =
          (has_gselect ? &gselect_reader.Value(utt) : &empty_gselect);

      DecodableAmSgmmScaled sgmm_decodable(sgmm_opts, am_sgmm, spk_vars,
                                           trans_model, features, *gselect,
                                           log_prune, acoustic_scale);
      decoder.Decode(&sgmm_decodable);

      KALDI_LOG << "Length of file is " << features.NumRows();

      VectorFst<StdArc> decoded;  // linear FST.
      bool saw_endstate = decoder.GetOutput(true,  // consider only final states
                                            &decoded);

      if (saw_endstate || decoder.GetOutput(false, &decoded)) {
        num_success++;
        if (!saw_endstate) {
          KALDI_WARN << "Decoder did not reach end-state, "
                     << "outputting partial traceback.";
        }
        std::vector<int32> alignment;
        std::vector<int32> words;
        StdArc::Weight weight;
        frame_count += features.NumRows();

        GetLinearSymbolSequence(decoded, &alignment, &words, &weight);

        words_writer.Write(utt, words);
        if (alignment_writer.IsOpen())
          alignment_writer.Write(utt, alignment);
        if (word_syms != NULL) {
          std::cerr << utt << ' ';
          for (size_t i = 0; i < words.size(); i++) {
            std::string s = word_syms->Find(words[i]);
            if (s == "")
              KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
            std::cerr << s << ' ';
          }
          std::cerr << '\n';
        }
        BaseFloat like = -weight.Value();
        tot_like += like;
        KALDI_LOG << "Log-like per frame for utterance " << utt << " is "
                  << (like / features.NumRows());

      } else {
        num_fail++;
        KALDI_WARN << "Did not successfully decode utterance " << utt
                   << ", len = " << features.NumRows();
      }
    }
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame = " << (tot_like/frame_count)
              << " over " << frame_count << " frames.";
    delete decode_fst;
    if (num_success != 0)
      return 0;
    else
      return 1;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


