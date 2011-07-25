// gmmbin/gmm-decode-kaldi.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "decoder/kaldi-decoder-left.h"
// you can either use left or right: without or with reorder option
#include "decoder/decodable-am-diag-gmm.h"
#include "util/timer.h"

typedef fst::ConstFst<fst::StdArc> FstType;

int main(int argc, char *argv[])
{
  try {
#ifdef _MSC_VER
    if (0) { new FstType(* static_cast<fst::VectorFst<fst::StdArc>*> (NULL)); }
#endif
    using namespace kaldi;
	using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Decode features using GMM-based model.\n"
        "Usage:   gmm-decode-kaldi [options] model-in fst-in features-rspecifier words-wspecifier\n";

    ParseOptions po(usage);
    BaseFloat acoustic_scale = 0.1;
    std::string word_syms_filename;
    KaldiDecoderOptions decoder_opts;
    decoder_opts.Register(&po, true);  // true == include obscure settings.
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_in_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        words_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input is(model_in_filename, &binary);
      trans_model.Read(is.Stream(), binary);
      am_gmm.Read(is.Stream(), binary);
    }


    Int32VectorWriter words_writer(words_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") {
      word_syms = fst::SymbolTable::ReadText(word_syms_filename);
      if (!word_syms)
        KALDI_EXIT << "Could not read symbol table from file "<<word_syms_filename;
    }

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    // It's important that we initialize decode_fst after feature_reader, as it
    // can prevent crashes on systems installed without enough virtual memory.
    // It has to do with what happens on UNIX systems if you call fork() on a
    // large process: the page-table entries are duplicated, which requires a
    // lot of virtual memory.
    FstType *decode_fst = NULL;
    {
      VectorFst<StdArc> *read_fst = NULL;
      std::ifstream is(fst_in_filename.c_str(), std::ifstream::binary);
      if (!is.good()) KALDI_EXIT << "Could not open decoding-graph FST "
                                << fst_in_filename;
      read_fst = VectorFst<StdArc>::Read(is, fst::FstReadOptions((std::string)fst_in_filename));
      if (read_fst == NULL) // fst code will warn.
        exit(1);
      decode_fst = new FstType(*read_fst);
      // copy to ConstFst.  If memory
      // exhausted here, should copy as ConstFst to disk.
    }

    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    KaldiDecoder<DecodableAmDiagGmmScaled, FstType> decoder(decoder_opts);

    Timer timer;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &features = feature_reader.Value();

      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << key;
        continue;
      }

      DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                             acoustic_scale);
      fst::VectorFst<fst::StdArc> *word_links = decoder.Decode(*decode_fst, &gmm_decodable);

      KALDI_LOG << "Length of file is " << features.NumRows();
      if (word_links == NULL) {
        KALDI_WARN << "Could not decode file " << key;
      } else {
        std::vector<kaldi::int32> words;
        StdArc::Weight weight;
        GetLinearSymbolSequence(*word_links, static_cast<std::vector<kaldi::int32>*>(NULL),
                                &words, &weight);

        frame_count += features.NumRows();

        words_writer.Write(key, words);

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
        BaseFloat like = -weight.Value();
        tot_like += like;
        KALDI_LOG << "Log-like per frame for utterance " << key <<"[index "
                  << key << "] is " << (like / features.NumRows());
        delete word_links;
      }
    }
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);

    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count)
              << " over " << frame_count << " frames.";
    
    if (word_syms) delete word_syms;
    delete decode_fst;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


