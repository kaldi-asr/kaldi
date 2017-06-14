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
#include "online2/online-endpoint.h"
#include "decoder/lattice-faster-online-decoder.h"
#include "hmm/transition-model.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/posterior.h"


namespace kaldi {
/// @addtogroup  onlinedecoding OnlineDecoding
/// @{



/// This configuration class controls when to re-estimate the basis-fMLLR during
/// online decoding.  The basic model is to re-estimate it on a certain time t
/// (e.g. after 1 second) and then at a set of times forming a geometric series,
/// e.g. 1.5, 1.5^2, etc.  We specify different configurations for the first
/// utterance of a speaker (which requires more frequent adaptation), and for
/// subsequent utterances.  We also re-estimate fMLLR at the end of every
/// utterance, but this is done directly from the calling code, not by the class
/// SingleUtteranceGmmDecoder.
struct OnlineGmmDecodingAdaptationPolicyConfig {
  BaseFloat adaptation_first_utt_delay;
  BaseFloat adaptation_first_utt_ratio;
  BaseFloat adaptation_delay;
  BaseFloat adaptation_ratio;
  OnlineGmmDecodingAdaptationPolicyConfig():
      adaptation_first_utt_delay(2.0),
      adaptation_first_utt_ratio(1.5),
      adaptation_delay(5.0),
      adaptation_ratio(2.0) { }

  void Register(OptionsItf *opts) {
    opts->Register("adaptation-first-utt-delay", &adaptation_first_utt_delay,
                   "Delay before first basis-fMLLR adaptation for first utterance "
                   "of each speaker");
    opts->Register("adaptation-first-utt-ratio", &adaptation_first_utt_ratio,
                   "Ratio that controls frequency of fMLLR adaptation for first "
                   "utterance of each speaker");
    opts->Register("adaptation-delay", &adaptation_delay,
                   "Delay before first basis-fMLLR adaptation for not-first "
                   "utterances of each speaker");
    opts->Register("adaptation-ratio", &adaptation_ratio,
                   "Ratio that controls frequency of fMLLR adaptation for "
                   "not-first utterances of each speaker");
  }
  
  /// Check that configuration values make sense.
  void Check() const;
  
  /// This function returns true if we are scheduled
  /// to re-estimate fMLLR somewhere in the interval
  /// [ chunk_begin_secs, chunk_end_secs ).
  bool DoAdapt(BaseFloat chunk_begin_secs,
               BaseFloat chunk_end_secs,
               bool is_first_utterance) const;
      
};


struct OnlineGmmDecodingConfig {
  BaseFloat fmllr_lattice_beam;
  
  BasisFmllrOptions basis_opts; // options for basis-fMLLR adaptation.

  LatticeFasterDecoderConfig faster_decoder_opts;
  
  OnlineGmmDecodingAdaptationPolicyConfig adaptation_policy_opts;
  
  // rxfilename for model trained with online-CMN features
  // (only needed if different from model_rxfilename)
  std::string online_alimdl_rxfilename;
  // rxfilename for model used for estimating fMLLR transforms
  std::string model_rxfilename;
  // rxfilename for possible discriminatively trained model
  // (only needed if different from model_rxfilename)
  std::string rescore_model_rxfilename;
  // rxfilename for the BasisFmllrEstimate object containing the basis
  // used for basis-fMLLR.
  std::string fmllr_basis_rxfilename;

  BaseFloat acoustic_scale;

  std::string silence_phones;
  BaseFloat silence_weight;
  

  OnlineGmmDecodingConfig():  fmllr_lattice_beam(3.0), acoustic_scale(0.1),
                              silence_weight(0.1) { }
  
  void Register(OptionsItf *opts) {
    { // register basis_opts with prefix, there are getting to be too many
      // options.
      ParseOptions basis_po("basis", opts);
      basis_opts.Register(&basis_po);
    }
    adaptation_policy_opts.Register(opts);
    faster_decoder_opts.Register(opts);
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Scaling factor for acoustic likelihoods");
    opts->Register("silence-phones", &silence_phones,
                   "Colon-separated list of integer ids of silence phones, e.g. "
                   "1:2:3 (affects adaptation).");
    opts->Register("silence-weight", &silence_weight,
                   "Weight applied to silence frames for fMLLR estimation (if "
                   "--silence-phones option is supplied)");
    opts->Register("fmllr-lattice-beam", &fmllr_lattice_beam, "Beam used in "
                   "pruning lattices for fMLLR estimation");
    opts->Register("online-alignment-model", &online_alimdl_rxfilename,
                   "(Extended) filename for model trained with online CMN "
                   "features, e.g. from apply-cmvn-online.");
    opts->Register("model", &model_rxfilename, "(Extended) filename for model, "
                   "typically the one used for fMLLR computation.  Required option.");
    opts->Register("rescore-model", &rescore_model_rxfilename, "(Extended) filename "
                   "for model to rescore lattices with, e.g. discriminatively trained"
                   "model, if it differs from that supplied to --model option.  Must"
                   "have the same tree.");
    opts->Register("fmllr-basis", &fmllr_basis_rxfilename, "(Extended) filename "
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
  // The model trained with online-CMVN features
  // (if supplied, otherwise use model_)
  AmDiagGmm online_alignment_model_;
  // The ML-trained model used to get transforms (required)
  AmDiagGmm model_;
  // The discriminatively trained model
  // (if supplied, otherwise use model_)
  AmDiagGmm rescore_model_;
  // The following object contains the basis elements for
  // "Basis fMLLR".
  BasisFmllrEstimate fmllr_basis_;
};


struct OnlineGmmAdaptationState {
  OnlineCmvnState cmvn_state;
  FmllrDiagGmmAccs spk_stats;
  Matrix<BaseFloat> transform;

  // Writing and reading of the state of the object
  void Write(std::ostream &out_stream, bool binary) const;
  void Read(std::istream &in_stream, bool binary);


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
                            const OnlineGmmAdaptationState &adaptation_state);
  
  OnlineFeaturePipeline &FeaturePipeline() { return *feature_pipeline_; }

  /// advance the decoding as far as we can.  May also estimate fMLLR after
  /// advancing the decoding, depending on the configuration values in
  /// config_.adaptation_policy_opts.  [Note: we expect the user will also call
  /// EstimateFmllr() at utterance end, which should generally improve the
  /// quality of the estimated transforms, although we don't rely on this].
  void AdvanceDecoding();

  /// Finalize the decoding. Cleanups and prunes remaining tokens, so the final result
  /// is faster to obtain.
  void FinalizeDecoding();

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
  
  void GetAdaptationState(OnlineGmmAdaptationState *adaptation_state) const;

  /// Gets the lattice.  If rescore_if_needed is true, and if there is any point
  /// in rescoring the state-level lattice (see RescoringIsNeeded()), it will
  /// rescore the lattice.  The output lattice has any acoustic scaling in it
  /// (which will typically be desirable in an online-decoding context); if you
  /// want an un-scaled lattice, scale it using ScaleLattice() with the inverse
  /// of the acoustic weight.  "end_of_utterance" will be true if you want the
  /// final-probs to be included.
  void GetLattice(bool rescore_if_needed,
                  bool end_of_utterance,
                  CompactLattice *clat) const;

  /// Outputs an FST corresponding to the single best path through the current
  /// lattice. If "use_final_probs" is true AND we reached the final-state of
  /// the graph then it will include those as final-probs, else it will treat
  /// all final-probs as one.
  void GetBestPath(bool end_of_utterance,
                   Lattice *best_path) const;

  /// This function outputs to "final_relative_cost", if non-NULL, a number >= 0
  /// that will be close to zero if the final-probs were close to the best probs
  /// active on the final frame.  (the output to final_relative_cost is based on
  /// the first-pass decoding).  If it's close to zero (e.g. < 5, as a guess),
  /// it means you reached the end of the grammar with good probability, which
  /// can be taken as a good sign that the input was OK.
  BaseFloat FinalRelativeCost() { return decoder_.FinalRelativeCost(); }


  /// This function calls EndpointDetected from online-endpoint.h,
  /// with the required arguments.
  bool EndpointDetected(const OnlineEndpointConfig &config);

  ~SingleUtteranceGmmDecoder();
 private:
  bool GetGaussianPosteriors(bool end_of_utterance, GaussPost *gpost);

  /// Returns true if doing a lattice rescoring pass would have any point, i.e.
  /// if we have estimated fMLLR during this utterance, or if we have a
  /// discriminative model that differs from the fMLLR model *and* we currently
  /// have fMLLR features.
  bool RescoringIsNeeded() const;

  OnlineGmmDecodingConfig config_;
  std::vector<int32> silence_phones_; // sorted, unique list of silence phones,
                                      // derived from config_
  const OnlineGmmDecodingModels &models_;
  OnlineFeaturePipeline *feature_pipeline_;  // owned here.
  const OnlineGmmAdaptationState &orig_adaptation_state_;
  // adaptation_state_ generally reflects the "current" state of the
  // adaptation. Note: adaptation_state_.cmvn_state is just copied from
  // orig_adaptation_state, the function GetAdaptationState() gets the CMVN
  // state.
  OnlineGmmAdaptationState adaptation_state_;
  LatticeFasterOnlineDecoder decoder_;
};

  
/// @} End of "addtogroup onlinedecoding"

}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_GMM_DECODING_H_
