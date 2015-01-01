// sgmm2bin/sgmm2-align-compiled.cc

// Copyright 2009-2012  Microsoft Corporation;  Saarland University
//           2012-2014 Johns Hopkins University (Daniel Povey)

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
#include "sgmm2/am-sgmm2.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/training-graph-compiler.h"
#include "sgmm2/decodable-am-sgmm2.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Align features given [SGMM-based] models.\n"
        "Usage: sgmm2-align-compiled [options] model-in graphs-rspecifier "
        "feature-rspecifier alignments-wspecifier\n"
        "e.g.: sgmm2-align-compiled 1.mdl ark:graphs.fsts scp:train.scp ark:1.ali\n";

    ParseOptions po(usage);
    bool binary = true;
    AlignConfig align_config;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;
    BaseFloat log_prune = 5.0;
    std::string gselect_rspecifier, spkvecs_rspecifier, utt2spk_rspecifier;

    align_config.Register(&po);    
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("log-prune", &log_prune, "Pruning beam used to reduce number "
                "of exp() evaluations.");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic "
                "likelihoods");
    po.Register("transition-scale", &transition_scale, "Scaling factor for "
                "some transition probabilities [see also self-loop-scale].");
    po.Register("self-loop-scale", &self_loop_scale, "Scaling factor for "
                "self-loop versus non-self-loop probability mass [controls "
                "most transition probabilities.]");
    po.Register("gselect", &gselect_rspecifier, "Precomputed Gaussian indices "
                "(rspecifier)");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    if (gselect_rspecifier == "")
      KALDI_ERR << "--gselect option is mandatory.";
    
    std::string model_in_filename = po.GetArg(1),
        fst_rspecifier = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        alignment_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmSgmm2 am_sgmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }
    
    SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);

    RandomAccessBaseFloatVectorReaderMapped spkvecs_reader(spkvecs_rspecifier,
                                                           utt2spk_rspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    int num_done = 0, num_err = 0, num_retry = 0;
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;

    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string utt = fst_reader.Key();
      if (!feature_reader.HasKey(utt)) {
        KALDI_WARN << "No feature found for utterance " << utt;
        num_err++;
        continue;
      }
      VectorFst<StdArc> decode_fst(fst_reader.Value());
      // stops copy-on-write of the fst by deleting the fst inside the reader,
      // since we're about to mutate the fst by adding transition probs.
      fst_reader.FreeCurrent();
      
      const Matrix<BaseFloat> &features = feature_reader.Value(utt);
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_err++;
        continue;
      }

      Sgmm2PerSpkDerivedVars spk_vars;
      if (spkvecs_reader.IsOpen()) {
        if (spkvecs_reader.HasKey(utt)) {
          spk_vars.SetSpeakerVector(spkvecs_reader.Value(utt));
          am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
        } else {
          KALDI_WARN << "Cannot find speaker vector for " << utt;
          num_err++;
          continue;
        }
      }  // else spk_vars is "empty"

      if (!gselect_reader.HasKey(utt)
          && gselect_reader.Value(utt).size() != features.NumRows()) {
        KALDI_WARN << "No Gaussian-selection info available for utterance "
                   << utt << " (or wrong size)";
        num_err++;
      }
      const std::vector<std::vector<int32> > &gselect =
          gselect_reader.Value(utt);

      {  // Add transition-probs to the FST.
        std::vector<int32> disambig_syms;  // empty.
        AddTransitionProbs(trans_model, disambig_syms,
                           transition_scale, self_loop_scale,
                           &decode_fst);
      }

      DecodableAmSgmm2Scaled sgmm_decodable(am_sgmm, trans_model, features, gselect,
                                            log_prune, acoustic_scale, &spk_vars);

      AlignUtteranceWrapper(align_config, utt,
                            acoustic_scale, &decode_fst, &sgmm_decodable,
                            &alignment_writer, NULL,
                            &num_done, &num_err, &num_retry,
                            &tot_like, &frame_count);
      
    }

    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count)
              << " over " << frame_count<< " frames.";
    KALDI_LOG << "Retried " << num_retry << " out of "
              << (num_done + num_err) << " utterances.";
    KALDI_LOG << "Done " << num_done << ", errors on " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


