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
#include "hmm/posterior.h"


namespace kaldi {


struct OnlineGmmDecodingConfig {
  BaseFloat fmllr_lattice_beam;

  BasisFmllrOptions basis_opts; // options for basis-fMLLR adaptation.

  LatticeFasterDecoderConfig faster_decoder_opts;
  
  // rxfilename for model trained with online-CMN features:
  std::string online_alimdl_rxfilename;
  // rxfilename for model used for estimating fMLLR transforms (only needed if
  // different from online_alimdl_rxfilename):
  std::string model_rxfilename;
  // rxfilename for possible discriminatively trained
  // model (only needed if different from model_rxfilename).
  std::string rescore_model_rxfilename;
  // rxfilename for the BasisFmllrEstimate object containing the basis
  // used for basis-fMLLR.
  std::string fmllr_basis_rxfilename;

  BaseFloat acoustic_scale;

  std::string silence_phones;
  BaseFloat silence_weight;
  
  int32 adaptation_threshold; // number of frames after which we first adapt.
                              // TODO: set this, make sure it's used (from calling code?)
  
  OnlineGmmDecodingConfig():  fmllr_lattice_beam(3.0), acoustic_scale(0.1),
                              silence_weight(0.1) { }
  
  void Register(OptionsItf *po) {
    { // register basis_opts with prefix, there are getting to be too many
      // options.
      ParseOptions basis_po("basis", po);
      basis_opts.Register(&basis_po);
    }
    faster_decoder_opts.Register(po);
    po->Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po->Register("silence-phones", &silence_phones,
                 "Colon-separated list of integer ids of silence phones, e.g. "
                 "1:2:3 (affects adaptation).");
    po->Register("silence-weight", &silence_weight,
                 "Weight applied to silence frames for fMLLR estimation (if "
                 "--silence-phones option is supplied)");
    po->Register("fmllr-lattice-beam", &fmllr_lattice_beam, "Beam used in "
                 "pruning lattices for fMLLR estimation");
    po->Register("online-alignment-model", &online_alimdl_rxfilename,
                 "(Extended) filename for model trained with online CMN "
                 "features, e.g. from apply-cmvn-online.");
    po->Register("model", &model_rxfilename, "(Extended) filename for model, "
                 "typically the one used for fMLLR computation.  Required option.");
    po->Register("rescore-model", &rescore_model_rxfilename, "(Extended) filename "
                 "for model to rescore lattices with, e.g. discriminatively trained"
                 "model, if it differs from that supplied to --model option.  Must"
                 "have the same tree.");
    po->Register("fmllr-basis", &fmllr_basis_rxfilename, "(Extended) filename "
                 "of fMLLR basis object, as output by gmm-basis-fmllr-training");
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

  const BasisFmllrEstimate &GetFmllrBasis() const;
  
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
  AmDiagGmm rescore_model_;
  // The following object contains the basis elements for
  // "Basis fMLLR".
  BasisFmllrEstimate fmllr_basis_;
};


struct SpeakerAdaptationState {
  OnlineCmvnState cmvn_state;
  FmllrDiagGmmAccs spk_stats;
  Matrix<BaseFloat> transform;
};

/**
   You will instantiate this class when you want to decode a single
   utterance using the online-decoding setup.  This is an alternative
   to manually putting things together yourself.
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
  bool HaveTransform() const;
  
  /// Estimate the [basis-]fMLLR transform and apply it to the features.
  /// This will get used if you call RescoreLattice() or if you just
  /// continue decoding; however to get it applied retroactively
  /// you'd have to call RescoreLattice().
  /// "end_of_utterance" just affects how we interpret the final-probs in the
  /// lattice.  This should generally be true if you think you've reached
  /// the end of the grammar, and false otherwise.
  void EstimateFmllr(bool end_of_utterance);
  
  void GetAdaptationState(SpeakerAdaptationState *adaptation_state) const;

  /// Gets the lattice.  If rescore_if_needed is true, and if there is any point
  /// in rescoring the state-level lattice (see RescoringIsNeeded()).  The
  /// output lattice has any acoustic scaling in it (which will typically be
  /// desirable in an online-decoding context); if you want an un-scaled
  /// lattice, scale it using ScaleLattice() with the inverse of the acoustic weight.
  /// "end_of_utterance" will be true if you want the final-probs to be included.
  void GetLattice(bool rescore_if_needed,
                  bool end_of_utterance,
                  CompactLattice *clat) const;

  /// This function outputs to "final_relative_cost", if non-NULL, a number >= 0
  /// that will be close to zero if the final-probs were close to the best probs
  /// active on the final frame.  (the output to final_relative_cost is based on
  /// the first-pass decoding).  If it's close to zero (e.g. < 5, as a guess),
  /// it means you reached the end of the grammar with good probability, which
  /// can be taken as a good sign that the input was OK.
  BaseFloat FinalRelativeCost() { return decoder_.FinalRelativeCost(); }

  ~SingleUtteranceGmmDecoder();
 private:
  // Note: the GauPost this outputs is indexed by pdf-id, not transition-id as
  // normal.
  bool GetGaussianPosteriors(bool end_of_utterance, GauPost *gpost);

  /// Returns true if doing a lattice rescoring pass would have any point, i.e.
  /// if we have estimated fMLLR during this utterance, or if we have a
  /// discriminative model that differs from the fMLLR model *and* we currently
  /// have fMLLR features.
  bool RescoringIsNeeded() const;

  OnlineGmmDecodingConfig config_;
  std::vector<int32> silence_phones_; // sorted, unique list of silence phones,
                                      // derived from config_
  const OnlineGmmDecodingModels &models_;
  OnlineFeaturePipeline *feature_pipeline_;
  const SpeakerAdaptationState &orig_adaptation_state_;
  // adaptation_state_ generally reflects the "current" state of the adptation.
  // Note: adaptation_state_.cmvn_state is just copied from
  // orig_adaptation_state, the function GetAdaptationState() gets the CMVN
  // state.
  SpeakerAdaptationState adaptation_state_;
  LatticeFasterDecoder decoder_;
  
};

  


}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_GMM_DECODING_H_
