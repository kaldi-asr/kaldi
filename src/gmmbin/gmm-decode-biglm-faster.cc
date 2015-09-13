// gmmbin/gmm-decode-biglm-faster.cc

// Copyright 2009-2011  Gilles Boulianne  Microsoft Corporation

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
#include "fstext/fstext-lib.h"
#include "decoder/biglm-faster-decoder.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "base/timer.h"

namespace kaldi {

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

}



int main(int argc, char *argv[])
{
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::Fst;
    using fst::StdArc;
    using fst::ReadFstKaldi;

    const char *usage =
        "Decode features using GMM-based model.\n"
        "User supplies LM used to generate decoding graph, and desired LM;\n"
        "this decoder applies the difference during decoding\n"
        "Usage:  gmm-decode-biglm-faster [options] model-in fst-in oldlm-fst-in newlm-fst-in features-rspecifier words-wspecifier [alignments-wspecifier [lattice-wspecifier]]\n";
    ParseOptions po(usage);
    bool allow_partial = true;    
    BaseFloat acoustic_scale = 0.1;
    
    std::string word_syms_filename;
    BiglmFasterDecoderOptions decoder_opts;
    decoder_opts.Register(&po, true);  // true == include obscure settings.
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "Produce output even when final state was not reached");

    po.Read(argc, argv);

    if (po.NumArgs() < 6 || po.NumArgs() > 8) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        old_lm_fst_rxfilename = po.GetArg(3),
        new_lm_fst_rxfilename = po.GetArg(4),
        feature_rspecifier = po.GetArg(5),
        words_wspecifier = po.GetArg(6),
        alignment_wspecifier = po.GetOptArg(7),
        lattice_wspecifier = po.GetOptArg(8);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    CompactLatticeWriter clat_writer(lattice_wspecifier);
    
    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") {
      word_syms = fst::SymbolTable::ReadText(word_syms_filename);
      if (!word_syms)
        KALDI_ERR << "Could not read symbol table from file "<<word_syms_filename;
    }

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);


    // It's important that we initialize decode_fst after feature_reader, as it
    // can prevent crashes on systems installed without enough virtual memory.
    // It has to do with what happens on UNIX systems if you call fork() on a
    // large process: the page-table entries are duplicated, which requires a
    // lot of virtual memory.
    Fst<StdArc> *decode_fst = ReadNetwork(fst_rxfilename);
    
    VectorFst<StdArc> *old_lm_fst = ReadFstKaldi(old_lm_fst_rxfilename);
    ApplyProbabilityScale(-1.0, old_lm_fst); // Negate old LM probs...
    
    VectorFst<StdArc> *new_lm_fst = ReadFstKaldi(new_lm_fst_rxfilename);


    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;

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
      fst::BackoffDeterministicOnDemandFst<StdArc> old_lm_dfst(*old_lm_fst);
      fst::BackoffDeterministicOnDemandFst<StdArc> new_lm_dfst(*new_lm_fst);
      fst::ComposeDeterministicOnDemandFst<StdArc> compose_dfst(&old_lm_dfst,
                                                                &new_lm_dfst);
      fst::CacheDeterministicOnDemandFst<StdArc> cache_dfst(&compose_dfst);
      
      BiglmFasterDecoder decoder(*decode_fst, decoder_opts, &cache_dfst);
      
      DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                             acoustic_scale);
      decoder.Decode(&gmm_decodable);

      std::cerr << "Length of file is "<<features.NumRows()<<'\n';

      fst::VectorFst<LatticeArc> decoded;  // linear FST.

      if ( (allow_partial || decoder.ReachedFinal())
           && decoder.GetBestPath(&decoded) ) {
        if (!decoder.ReachedFinal())
          KALDI_WARN << "Decoder did not reach end-state, "
                     << "outputting partial traceback since --allow-partial=true";
        num_success++;
        if (!decoder.ReachedFinal())
          KALDI_WARN << "Decoder did not reach end-state, outputting partial traceback.";
        std::vector<int32> alignment;
        std::vector<int32> words;
        LatticeWeight weight;
        frame_count += features.NumRows();

        GetLinearSymbolSequence(decoded, &alignment, &words, &weight);

        words_writer.Write(key, words);
        if (alignment_writer.IsOpen())
          alignment_writer.Write(key, alignment);
          
        if (lattice_wspecifier != "") {
          if (acoustic_scale != 0.0) // We'll write the lattice without acoustic scaling
            fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &decoded);
          fst::VectorFst<CompactLatticeArc> clat;
          ConvertLattice(decoded, &clat, true);
          clat_writer.Write(key, clat);
        }
          
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
        BaseFloat like = -weight.Value1() -weight.Value2();
        tot_like += like;
        KALDI_LOG << "Log-like per frame for utterance " << key << " is "
                  << (like / features.NumRows()) << " over "
                  << features.NumRows() << " frames.";
        KALDI_VLOG(2) << "Cost for utterance " << key << " is "
                      << weight.Value1() << " + " << weight.Value2();
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

    delete word_syms;    
    delete decode_fst;
    delete old_lm_fst;
    delete new_lm_fst;
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


