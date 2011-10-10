// sgmmbin/sgmm-latgen-simple.cc

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
#include "decoder/lattice-simple-decoder.h"
#include "decoder/decodable-am-sgmm.h"
#include "util/timer.h"

namespace kaldi {
// Takes care of output.  Returns total like.
double ProcessDecodedOutput(const LatticeSimpleDecoder &decoder,
                            const fst::SymbolTable *word_syms,
                            std::string utt,
                            double acoustic_scale,
                            bool determinize,
                            Int32VectorWriter *alignment_writer,
                            Int32VectorWriter *words_writer,
                            CompactLatticeWriter *compact_lattice_writer,
                            LatticeWriter *lattice_writer) {
  using fst::VectorFst;
  
  double likelihood;
  { // First do some stuff with word-level traceback...
    VectorFst<LatticeArc> decoded;
    if (!decoder.GetBestPath(&decoded)) 
      // Shouldn't really reach this point as already checked success.
      KALDI_ERR << "Failed to get traceback for utterance " << utt;

    std::vector<int32> alignment;
    std::vector<int32> words;
    LatticeWeight weight;
    GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
    if (words_writer->IsOpen())
      words_writer->Write(utt, words);
    if (alignment_writer->IsOpen())
      alignment_writer->Write(utt, alignment);
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

  if (determinize) {
    CompactLattice fst;
    if (!decoder.GetLattice(&fst))
      KALDI_ERR << "Unexpected problem getting lattice for utterance "
                << utt;
    if (acoustic_scale != 0.0) // We'll write the lattice without acoustic scaling
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &fst); 
    compact_lattice_writer->Write(utt, fst);
  } else {
    Lattice fst;
    if (!decoder.GetRawLattice(&fst)) 
      KALDI_ERR << "Unexpected problem getting lattice for utterance "
                << utt;
    fst::Connect(&fst); // Will get rid of this later... shouldn't have any
    // disconnected states there, but we seem to.
    if (acoustic_scale != 0.0) // We'll write the lattice without acoustic scaling
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &fst); 
    lattice_writer->Write(utt, fst);
  }
  return likelihood;
}

}


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Decode features using SGMM-based model.\n"
        "Usage:  sgmm-latgen-simple [options] <model-in> <fst-in> "
        "<features-rspecifier> <lattices-wspecifier> [<words-wspecifier> [<alignments-wspecifier>] ]\n";
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 0.1;
    bool allow_partial = false;
    BaseFloat log_prune = 5.0;
    string word_syms_filename, gselect_rspecifier, spkvecs_rspecifier,
        utt2spk_rspecifier;

    LatticeSimpleDecoderConfig decoder_opts;
    SgmmGselectConfig sgmm_opts;
    decoder_opts.Register(&po);    
    sgmm_opts.Register(&po);

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

    std::string model_in_filename = po.GetArg(1),
        fst_in_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        lattice_wspecifier = po.GetArg(4),
        words_wspecifier = po.GetOptArg(5),
        alignment_wspecifier = po.GetOptArg(6);

    TransitionModel trans_model;
    kaldi::AmSgmm am_sgmm;
    {
      bool binary;
      Input is(model_in_filename, &binary);
      trans_model.Read(is.Stream(), binary);
      am_sgmm.Read(is.Stream(), binary);
    }

    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    bool determinize = decoder_opts.determinize_lattice;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_EXIT << "Could not open table for writing lattices: "
                 << lattice_wspecifier;
    
    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_EXIT << "Could not read symbol table from file "
                   << word_syms_filename;

    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);

    RandomAccessTokenReader utt2spk_reader(utt2spk_rspecifier);

    RandomAccessBaseFloatVectorReader spkvecs_reader(spkvecs_rspecifier);
    
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
    LatticeSimpleDecoder decoder(*decode_fst, decoder_opts);
    
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

      if (!decoder.Decode(&sgmm_decodable)) {
        KALDI_WARN << "Failed to decode file " << utt;
        num_fail++;
        continue;
      }

      frame_count += features.NumRows();
      double like;
      if (!decoder.ReachedFinal()) {
        if (allow_partial) {
          KALDI_WARN << "Outputting partial output for utterance " << utt
                     << " since no final-state reached\n";
          num_fail++;
        } else {
          KALDI_WARN << "Not producing output for utterance " << utt
                     << " since no final-state reached and "
                     << "--allow-partial=false.\n";
          continue;
        }
      }
      like = ProcessDecodedOutput(decoder, word_syms, utt, acoustic_scale,
                                  determinize, &alignment_writer, &words_writer,
                                  &compact_lattice_writer, &lattice_writer);
      tot_like += like;
      KALDI_LOG << "Log-like per frame for utterance " << utt << " is "
                << (like / features.NumRows()) << " over "
                << features.NumRows() << " frames.";
      num_success++;
    }
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame = " << (tot_like/frame_count)
              << " over " << frame_count << " frames.";

    if (word_syms) delete word_syms;
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


