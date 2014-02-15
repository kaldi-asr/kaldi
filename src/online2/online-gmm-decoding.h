// online2/online-gmm-decoding.h

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_ONLINE2_ONLINE_GMM_DECODING_H_
#define KALDI_ONLINE2_ONLINE_GMM_DECODING_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "transform/basis-fmllr-diag-gmm.h"
#include "transform/fmllr-diag-gmm.h"
#include "online2/online-feature-pipeline.h"
#include "online2/online-gmm-decodable.h"
#include "decoder/lattice-faster-decoder.h"
#include "hmm/transition-model.h"
#include "gmm/am-diag-gmm.h"


namespace kaldi {


struct OnlineGmmDecodingConfig {
  OnlineFeaturePipelineConfig feature_config;

  BaseFloat fmllr_lattice_beam;
  BasisFmllrOptions basis_opts; // options for basis-fMLLR adaptation.

  LatticeFasterDecoderConfig faster_decoder_opts;
  
  // rxfilename for model trained with online-CMN features:
  std::string online_alimdl_rxfilename;
  // rxfilename for model used for estimating fMLLR transforms (only needed if
  // different from online_alimdl_rxfilename):
  std::string mdl_rxfilename;
  // rxfilename for possible discriminatively trained
  // model (only needed if different from mdl_rxfilename).
  std::string final_mdl_rxfilename;

  BaseFloat acoustic_scale;

  std::string silence_phones;
  
  int32 adaptation_threshold; // number of frames after which we first adapt.
                              // TODO: set this, make sure it's used (from calling code?)
  
  OnlineGmmDecodingConfig():  fmllr_lattice_beam(3.0), acoustic_scale(0.1) { }
  
  void Register(OptionsItf *po) {
    feature_config.Register(po);
    basis_opts.Register(po);
    faster_decoder_opts.Register(po);
    po->Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po->Register("silence-phones", &silence_phones,
                 "Colon-separated list of integer ids of silence phones, e.g. "
                 "1:2:3 (affects adaptation).");
  }
};


/**
   This class is used to read, store and give access to the models used for 3
   phases of decoding (first-pass with online-CMN features; the ML models used
   for estimating transforms; and the discriminatively trained models).  It
   takes care of the logic whereby if, say, the last model isn't given we
   default to the second model, and so on, and it interpretes the filenames
   from the config object.  It is passed as a const reference to other
   objects in this header. */
class OnlineGmmDecodingModels {
 public:
  OnlineGmmDecodingModels(const OnlineGmmDecodingConfig &config);

  const TransitionModel &GetTransitionModel() const;

  const AmDiagGmm &GetOnlineAlignmentModel() const;

  const AmDiagGmm &GetModel() const;

  const AmDiagGmm &GetFinalModel() const;

 private:
  // The transition-model is only needed for its integer ids, and these need to
  // be identical for all 3 models, so we only store one (it doesn't matter
  // which one).
  TransitionModel tmodel_; 
  // The model trained with online-CMVN features:
  AmDiagGmm online_alignment_model_;
  // The ML-trained model used to get transforms, if supplied (otherwise use
  // online_alignment_model_):
  AmDiagGmm model_;
  // The discriminatively trained model (if supplied);
  // otherwise use model_ if supplied, otherwise use
  // online_alignment_model_:
  AmDiagGmm final_model_;   
};


struct SpeakerAdaptationState {
  OnlineCmvnState cmvn_state;
  FmllrDiagGmmAccs spk_stats;
  Matrix<BaseFloat> cur_transform;
};

/**
   You will instantiate this class when you want to decode a single
   utterance using the online-decoding setup.  This is an alternative
   than manually putting things together yourself.
*/
class SingleUtteranceGmmDecoder {
 public:
  SingleUtteranceGmmDecoder(const OnlineGmmDecodingConfig &config,
                            const OnlineGmmDecodingModels &models,                            
                            const OnlineFeaturePipeline &feature_prototype,
                            const fst::Fst<fst::StdArc> &fst,
                            const SpeakerAdaptationState &adaptation_state);

  OnlineFeaturePipeline &FeaturePipeline() { return *feature_pipeline_; }

  /// advance the first pass as far as we can.
  void AdvanceFirstPass();

  /// Returns true if we already have an fMLLR transform.  The user will
  /// already know this; the call is for convenience.  
  bool HasTransform() const;
  
  /// Estimate the [basis-]fMLLR transform and apply it to the features.
  /// This will get used if you call RescoreLattice() or if you just
  /// continue decoding; however to get it applied retroactively
  /// you'd have to call RescoreLattice().
  /// "end_of_utterance" just affects how we interpret the final-probs in the
  /// lattice.
  void EstimateFmllr(bool end_of_utterance);

  
  
  void GetAdaptationState(SpeakerAdaptationState *adaptation_state);

  ~SingleUtteranceGmmDecoder();
 private:
  void AccumulateFmllrStats(const CompactLattice &clat,
                            FmllrDiagGmmAccs *spk_stats);

  
  OnlineGmmDecodingConfig config_;
  const OnlineGmmDecodingModels &models_;
  OnlineFeaturePipeline *feature_pipeline_;
  const SpeakerAdaptationState &orig_adaptation_state_;
  SpeakerAdaptationState adaptation_state_;
  LatticeFasterDecoder decoder_;
  
};

  


}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_GMM_DECODING_H_
