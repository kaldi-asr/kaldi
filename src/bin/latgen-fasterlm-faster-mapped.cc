// bin/latgen-fasterlm-faster-mapped .cc

// Copyright      2018  Zhehuai Chen

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
#include "decoder/decoder-wrappers.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"
#include "lm/faster-arpa-lm.h"
#include "decoder/lattice-biglm-faster-decoder.h"


namespace kaldi {
// Takes care of output.  Returns true on success.
bool DecodeUtterance(LatticeBiglmFasterDecoder &decoder, // not const but is really an input.
                     DecodableInterface &decodable, // not const but is really an input.
                     const TransitionModel &trans_model,
                     const fst::SymbolTable *word_syms,
                     std::string utt,
                     double acoustic_scale,
                     bool determinize,
                     bool allow_partial,
                     Int32VectorWriter *alignment_writer,
                     Int32VectorWriter *words_writer,
                     CompactLatticeWriter *compact_lattice_writer,
                     LatticeWriter *lattice_writer,
                     double *like_ptr) {  // puts utterance's like in like_ptr on success.
  using fst::VectorFst;

  if (!decoder.Decode(&decodable)) {
    KALDI_WARN << "Failed to decode file " << utt;
    return false;
  }
  if (!decoder.ReachedFinal()) {
    if (allow_partial) {
      KALDI_WARN << "Outputting partial output for utterance " << utt
                 << " since no final-state reached\n";
    } else {
      KALDI_WARN << "Not producing output for utterance " << utt
                 << " since no final-state reached and "
                 << "--allow-partial=false.\n";
      return false;
    }
  }

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  { // First do some stuff with word-level traceback...
    VectorFst<LatticeArc> decoded;
    decoder.GetBestPath(&decoded);
    if (decoded.NumStates() == 0)
      // Shouldn't really reach this point as already checked success.
      KALDI_ERR << "Failed to get traceback for utterance " << utt;

    std::vector<int32> alignment;
    std::vector<int32> words;
    GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
    num_frames = alignment.size();
    if (words_writer->IsOpen())
      words_writer->Write(utt, words);
    assert(!alignment_writer);
    //if (alignment_writer->IsOpen())
    //  alignment_writer->Write(utt, alignment);
    if (word_syms != NULL) {
      std::cerr << utt << ' ';
      for (size_t i = 0; i < words.size(); i++) {
        std::string s = word_syms->Find(words[i]);
        if (s == "")
          KALDI_ERR << "Word-id " << words[i] <<" not in symbol table.";
        std::cerr << s << ' ';
      }
      std::cerr << '\n';
    }
    likelihood = -(weight.Value1() + weight.Value2());
  }

  // Get lattice, and do determinization if requested.
  Lattice lat;
  decoder.GetRawLattice(&lat);
  if (lat.NumStates() == 0)
    KALDI_ERR << "Unexpected problem getting lattice for utterance " << utt;
  fst::Connect(&lat);
  if (determinize) {
    CompactLattice clat;
    if (!DeterminizeLatticePhonePrunedWrapper(
            trans_model,
            &lat,
            decoder.GetOptions().lattice_beam,
            &clat,
            decoder.GetOptions().det_opts))
      KALDI_WARN << "Determinization finished earlier than the beam for "
                 << "utterance " << utt;
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &clat);
    compact_lattice_writer->Write(utt, clat);
  } else {
    Lattice fst;
    decoder.GetRawLattice(&fst);
    if (fst.NumStates() == 0)
      KALDI_ERR << "Unexpected problem getting lattice for utterance "
                << utt;
    fst::Connect(&fst); // Will get rid of this later... shouldn't have any
    // disconnected states there, but we seem to.
    if (acoustic_scale != 0.0) // We'll write the lattice without acoustic scaling
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &fst); 
    lattice_writer->Write(utt, fst);
  }
  KALDI_LOG << "Log-like per frame for utterance " << utt << " is "
            << (likelihood / num_frames) << " over "
            << num_frames << " frames.";
  KALDI_VLOG(2) << "Cost for utterance " << utt << " is "
                << weight.Value1() << " + " << weight.Value2();
  *like_ptr = likelihood;
  return true;
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::Fst;
    using fst::StdArc;
    using fst::ReadFstKaldi;

    const char *usage =
        "Generate lattices using on-the-fly composition.\n"
        "e.g. HCLG_1 - G_1 + (G_2a \\dynamic_int G_2b) \n"
        "User supplies LM used to generate decoding graph, and desired LM;\n"
        "this decoder applies the difference during decoding\n"
        "Usage: latgen-fasterlm-faster-mapped [options] model-in(for ctc, the model is ignored) HCLG-1-fstin "
        "G-1-oldlm G-1-weight G-2a-newlm G-2a-weight G-2b-newlm G-2b-weight G-2c... features-rspecifier"
        " lattice-wspecifier  words-wspecifier \n"
        "Notably, we always make G-1-weight = -1\n"
        "ctc example: /fgfs/users/zhc00/works/dyn_dec/kaldi_ctc/README\n"
        "hmm example: /fgfs/users/zhc00/works/dyn_dec/kaldi_minilibri/README\n"
        ;
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    int32 symbol_size = 0;
    bool ctc = false;
    LatticeBiglmFasterDecoderConfig config;
    config.Register(&po);

    ArpaParseOptions arpa_options;
    arpa_options.Register(&po);
    po.Register("ctc", &ctc, "is ctc decoding");
    po.Register("symbol-size", &symbol_size, "symbol table size");
    po.Register("unk-symbol", &arpa_options.unk_symbol,
                "Integer corresponds to unknown-word in language model. -1 if "
                "no such word is provided.");
    po.Register("bos-symbol", &arpa_options.bos_symbol,
                "Integer corresponds to <s>. You must set this to your actual "
                "BOS integer.");
    po.Register("eos-symbol", &arpa_options.eos_symbol,
                "Integer corresponds to </s>. You must set this to your actual "
                "EOS integer.");


    std::string word_syms_filename;
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");

    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial, "If true, produce output even if end state was not reached.");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 6 ) {
      po.PrintUsage();
      exit(1);
    }
   
    int start_lm = 3;
    int end_lm = po.NumArgs() - 3;
    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2),
        feature_rspecifier = po.GetArg(po.NumArgs() - 2),
        lattice_wspecifier = po.GetArg(po.NumArgs() - 1),
        words_wspecifier = po.GetOptArg(po.NumArgs());
 
    assert((end_lm - start_lm+1) % 2 == 0); // one lm one weight
    //old_lm_fst_rxfilename = po.GetArg(3),
    //new_lm_fst_rxfilename = po.GetArg(4),   

    TransitionModel trans_model;
    if (!ctc)
        ReadKaldiObject(model_in_filename, &trans_model);

    /*
    FasterArpaLm old_lm;
    ReadKaldiObject(old_lm_fst_rxfilename, &old_lm);
    FasterArpaLmDeterministicFst old_lm_dfst(old_lm);
    ApplyProbabilityScale(-1.0, old_lm_dfst); // Negate old LM probs...
    */
    int lm_num=(end_lm-start_lm+1)/2;
    std::vector<FasterArpaLm> lm_vec;
    std::vector<FasterArpaLmDeterministicFst> dlm_vec;
    std::vector<fst::ComposeDeterministicOnDemandFst<StdArc>> clm_vec;
    lm_vec.reserve(lm_num);
    dlm_vec.reserve(lm_num);
    clm_vec.reserve(lm_num-1);
    for ( int i = start_lm; i < end_lm; i+=2 ) {
      std::string s_lm = po.GetArg(i);
      float w =  atof(po.GetArg(i+1).c_str());
      lm_vec.emplace_back(arpa_options, s_lm, symbol_size, w);
      dlm_vec.emplace_back(lm_vec.back());
      if (i == start_lm) continue;
      else if (i == start_lm+2) {
        clm_vec.emplace_back(&dlm_vec.front(),&dlm_vec.back());
      } else {
        clm_vec.emplace_back(&clm_vec.back(),&dlm_vec.back());
      }
    }
    // multiple compose
    fst::CacheDeterministicOnDemandFst<StdArc> cache_dfst(&clm_vec.back(), 1e7);

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);

    //Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;

    double elapsed=0;
    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      // Input FST is just one FST, not a table of FSTs.
      Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);

      {
        LatticeBiglmFasterDecoder decoder(*decode_fst, config, &cache_dfst);
        timer.Reset();
    
        for (; !feature_reader.Done(); feature_reader.Next()) {
          std::string utt = feature_reader.Key();
          Matrix<BaseFloat> features (feature_reader.Value());
          feature_reader.FreeCurrent();
          if (features.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_fail++;
            continue;
          }
         
          DecodableInterface* decodable = NULL;
          if (!ctc) 
            decodable = new DecodableMatrixScaledMapped(trans_model, features, acoustic_scale);
          else {
            decodable = new DecodableMatrixScaledMappedCtc(features, acoustic_scale);
            decoder.GetOptions().det_opts.phone_determinize = false; // disable DeterminizeLatticePhonePrunedFirstPass
          }

          double like;
          if (DecodeUtterance(decoder, *decodable, trans_model, word_syms,
                              utt, acoustic_scale, determinize, allow_partial,
                              NULL, &words_writer,
                              &compact_lattice_writer, &lattice_writer,
                              &like)) {
            tot_like += like;
            frame_count += features.NumRows();
            num_success++;
          } else num_fail++;
          delete decodable;
        }
        elapsed = timer.Elapsed();
      }
      delete decode_fst; // delete this only after decoder goes out of scope.
    } else { // We have different FSTs for different utterances.
      assert(0);
    }
      
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
              << frame_count<<" frames.";

    delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
