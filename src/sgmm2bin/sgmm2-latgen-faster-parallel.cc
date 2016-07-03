// sgmm2bin/sgmm2-latgen-faster-parallel.cc

// Copyright 2009-2013  Saarland University;  Microsoft Corporation;
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
#include "thread/kaldi-task-sequence.h"
#include "base/timer.h"

namespace kaldi {

// the reference arguments at the beginning are not const as the style guide
// requires, but are best viewed as inputs.
void ProcessUtterance(const AmSgmm2 &am_sgmm,
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
                      LatticeFasterDecoder *decoder, // Takes ownership of this.
                      double *like_sum,
                      int64 *frame_sum,
                      int32 *num_done,
                      int32 *num_err,
                      TaskSequencer<DecodeUtteranceLatticeFasterClass> *sequencer) {
  using fst::VectorFst;
  using std::vector;

  Sgmm2PerSpkDerivedVars *spk_vars = new Sgmm2PerSpkDerivedVars; // decodable
  // will take ownership.
  if (spkvecs_reader.IsOpen()) {
    if (spkvecs_reader.HasKey(utt)) {
      spk_vars->SetSpeakerVector(spkvecs_reader.Value(utt));
      am_sgmm.ComputePerSpkDerivedVars(spk_vars);
    } else {
      KALDI_WARN << "Cannot find speaker vector for " << utt << ", not decoding this utterance";
      delete spk_vars;
      (*num_err)++;
      return;
    }
  }
  if (!gselect_reader.HasKey(utt) ||
      gselect_reader.Value(utt).size() != features.NumRows()) {
    KALDI_WARN << "No Gaussian-selection info available for utterance "
               << utt << " (or wrong size)";
  }

  // decodable will take ownership.
  vector<vector<int32> > *gselect = new std::vector<vector<int32> >(
      gselect_reader.Value(utt));

  Matrix<BaseFloat> *new_feats = new Matrix<BaseFloat>(features); // decodable
  // will take ownership of this.

  // This takes ownership of new_feats, gselect, and spk_vars
  DecodableAmSgmm2Scaled *sgmm_decodable = new DecodableAmSgmm2Scaled(
      am_sgmm, trans_model, new_feats, gselect,
      spk_vars, log_prune, acoustic_scale);

  // takes ownership of decoder and sgmm_decodable.
  DecodeUtteranceLatticeFasterClass *task =
      new DecodeUtteranceLatticeFasterClass(
          decoder, sgmm_decodable, trans_model, word_syms, utt, acoustic_scale,
          determinize, allow_partial, alignments_writer, words_writer,
          compact_lattice_writer, lattice_writer, like_sum, frame_sum, num_done,
          num_err, NULL);

  sequencer->Run(task); // takes ownership.
}

} // end namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Decode features using SGMM-based model.  This version accepts the --num-threads\n"
        "option but otherwise behaves identically to sgmm2-latgen-faster\n"
        "Usage:  sgmm2-latgen-faster-parallel [options] <model-in> (<fst-in>|<fsts-rspecifier>) "
        "<features-rspecifier> <lattices-wspecifier> [<words-wspecifier> [<alignments-wspecifier>] ]\n";
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 0.1;
    bool allow_partial = false;
    BaseFloat log_prune = 5.0;
    string word_syms_filename, gselect_rspecifier, spkvecs_rspecifier,
        utt2spk_rspecifier;

    LatticeFasterDecoderConfig decoder_opts;
    TaskSequencerConfig sequencer_config; // has --num-threads option
    decoder_opts.Register(&po);
    sequencer_config.Register(&po);

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

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;    
    int num_done = 0, num_err = 0;
    Timer timer;
    VectorFst<StdArc> *decode_fst = NULL;
    fst::SymbolTable *word_syms = NULL;
    
    TaskSequencer<DecodeUtteranceLatticeFasterClass> sequencer(
        sequencer_config);
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

    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_filename;

    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    RandomAccessBaseFloatVectorReaderMapped spkvecs_reader(spkvecs_rspecifier,
                                                           utt2spk_rspecifier);
        
    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) { // a single FST.
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      // It's important that we initialize decode_fst after feature_reader, as it
      // can prevent crashes on systems installed without enough virtual memory.
      // It has to do with what happens on UNIX systems if you call fork() on a
      // large process: the page-table entries are duplicated, which requires a
      // lot of virtual memory.
      decode_fst = fst::ReadFstKaldi(fst_in_str);
      timer.Reset(); // exclude graph loading time.
      
      {
        for (; !feature_reader.Done(); feature_reader.Next()) {
          string utt = feature_reader.Key();
          const Matrix<BaseFloat> &features(feature_reader.Value());
          if (features.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_err++;
            continue;
          }

          // ProcessUtterance will take ownership of this.
          LatticeFasterDecoder *decoder = new LatticeFasterDecoder(
              *decode_fst, decoder_opts);

          ProcessUtterance(am_sgmm, trans_model, log_prune, acoustic_scale,
                           features, gselect_reader, spkvecs_reader, word_syms,
                           utt, determinize, allow_partial,
                           &alignment_writer, &words_writer, &compact_lattice_writer,
                           &lattice_writer, decoder, &tot_like, &frame_count,
                           &num_done, &num_err, &sequencer);
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
        VectorFst<StdArc> *fst = fst_reader.Value().Copy(); // Note: this does
        // a shallow copy because OpenFst is "smart" about these things and
        // does reference counting.  The constructor of LatticeFasterDecoder
        // takes ownership of this FST (note: LatticeFasterDecoder has 2
        // constructors, one of which takes ownership and one of which does not).
        LatticeFasterDecoder *decoder = new LatticeFasterDecoder(decoder_opts,
                                                                 fst);

        // ProcessUtterance takes ownership of "decoder".
        ProcessUtterance(am_sgmm, trans_model, log_prune, acoustic_scale,
                         features, gselect_reader, spkvecs_reader, word_syms,
                         utt, determinize, allow_partial,
                         &alignment_writer, &words_writer, &compact_lattice_writer,
                         &lattice_writer, decoder, &tot_like, &frame_count,
                         &num_done, &num_err, &sequencer);
      }
    }
    sequencer.Wait(); // Wait till all tasks are done.
    
    delete decode_fst; 
    delete word_syms;
    
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Decoded with " << sequencer_config.num_threads << " threads.";
    KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
              << "s: real-time factor per thread assuming 100 frames/sec is "
              << (sequencer_config.num_threads * elapsed * 100.0 / frame_count);
    KALDI_LOG << "Done " << num_done << " utterances, failed for "
              << num_err;
    KALDI_LOG << "Overall log-likelihood per frame = " << (tot_like/frame_count)
              << " over " << frame_count << " frames.";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


