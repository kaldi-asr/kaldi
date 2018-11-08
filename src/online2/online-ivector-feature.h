// online2/online-ivector-feature.h

// Copyright 2013-2014   Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_ONLINE2_ONLINE_IVECTOR_FEATURE_H_
#define KALDI_ONLINE2_ONLINE_IVECTOR_FEATURE_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "itf/online-feature-itf.h"
#include "gmm/diag-gmm.h"
#include "feat/online-feature.h"
#include "ivector/ivector-extractor.h"
#include "decoder/lattice-faster-online-decoder.h"

namespace kaldi {
/// @addtogroup  onlinefeat OnlineFeatureExtraction
/// @{

/// @file
/// This file contains code for online iVector extraction in a form compatible
/// with OnlineFeatureInterface.  It's used in online-nnet2-feature-pipeline.h.

/// This class includes configuration variables relating to the online iVector
/// extraction, but not including configuration for the "base feature",
/// i.e. MFCC/PLP/filterbank, which is an input to this feature.  This
/// configuration class can be used from the command line, but before giving it
/// to the code we create a config class called
/// OnlineIvectorExtractionInfo which contains the actual configuration
/// classes as well as various objects that are needed.  The principle is that
/// any code should be callable from other code, so we didn't want to force
/// configuration classes to be read from disk.
struct OnlineIvectorExtractionConfig {
  std::string lda_mat_rxfilename;  // to read the LDA+MLLT matrix
  std::string global_cmvn_stats_rxfilename; // to read matrix of global CMVN
                                            // stats
  std::string splice_config_rxfilename;  // to read OnlineSpliceOptions
  std::string cmvn_config_rxfilename;  // to read in OnlineCmvnOptions
  std::string diag_ubm_rxfilename;  // reads type DiagGmm.
  std::string ivector_extractor_rxfilename;  // reads type IvectorExtractor

  // the following four configuration values should in principle match those
  // given to the script extract_ivectors_online.sh, although none of them are
  // super-critical.
  int32 ivector_period;  // How frequently we re-estimate iVectors.
  int32 num_gselect;  // maximum number of posteriors to use per frame for
                      // iVector extractor.
  BaseFloat min_post;  // pruning threshold for posteriors for the iVector
                       // extractor.
  BaseFloat posterior_scale;  // Scale on posteriors used for iVector
                              // extraction; can be interpreted as the inverse
                              // of a scale on the log-prior.
  BaseFloat max_count;  // Maximum stats count we allow before we start scaling
                        // down stats (if nonzero).. this prevents us getting
                        // atypical-looking iVectors for very long utterances.
                        // Interpret this as a number of frames times
                        // posterior_scale, typically 1/10 of a frame count.

  int32 num_cg_iters;  // set to 15.  I don't believe this is very important, so it's
                       // not configurable from the command line for now.


  // If use_most_recent_ivector is true, we always return the most recent
  // available iVector rather than the one for the current frame.  This means
  // that if audio is coming in faster than we can process it, we will return a
  // more accurate iVector.
  bool use_most_recent_ivector;

  // If true, always read ahead to NumFramesReady() when getting iVector stats.
  bool greedy_ivector_extractor;

  // max_remembered_frames is the largest number of frames it will remember
  // between utterances of the same speaker; this affects the output of
  // GetAdaptationState(), and has the effect of limiting the number of frames
  // of both the CMVN stats and the iVector stats.  Setting this to a smaller
  // value means the adaptation is less constrained by previous utterances
  // (assuming you provided info from a previous utterance of the same speaker
  // by calling SetAdaptationState()).
  BaseFloat max_remembered_frames;

  OnlineIvectorExtractionConfig(): ivector_period(10), num_gselect(5),
                                   min_post(0.025), posterior_scale(0.1),
                                   max_count(0.0), num_cg_iters(15),
                                   use_most_recent_ivector(true),
                                   greedy_ivector_extractor(false),
                                   max_remembered_frames(1000) { }

  void Register(OptionsItf *opts) {
    opts->Register("lda-matrix", &lda_mat_rxfilename, "Filename of LDA matrix, "
                   "e.g. final.mat; used for iVector extraction. ");
    opts->Register("global-cmvn-stats", &global_cmvn_stats_rxfilename,
                   "(Extended) filename for global CMVN stats, used in iVector "
                   "extraction, obtained for example from "
                   "'matrix-sum scp:data/train/cmvn.scp -', only used for "
                   "iVector extraction");
    opts->Register("cmvn-config", &cmvn_config_rxfilename, "Configuration "
                   "file for online CMVN features (e.g. conf/online_cmvn.conf),"
                   "only used for iVector extraction.  Contains options "
                   "as for the program 'apply-cmvn-online'");
    opts->Register("splice-config", &splice_config_rxfilename, "Configuration file "
                   "for frame splicing (--left-context and --right-context "
                   "options); used for iVector extraction.");
    opts->Register("diag-ubm", &diag_ubm_rxfilename, "Filename of diagonal UBM "
                   "used to obtain posteriors for iVector extraction, e.g. "
                   "final.dubm");
    opts->Register("ivector-extractor", &ivector_extractor_rxfilename,
                   "Filename of iVector extractor, e.g. final.ie");
    opts->Register("ivector-period", &ivector_period, "Frequency with which "
                   "we extract iVectors for neural network adaptation");
    opts->Register("num-gselect", &num_gselect, "Number of Gaussians to select "
                   "for iVector extraction");
    opts->Register("min-post", &min_post, "Threshold for posterior pruning in "
                   "iVector extraction");
    opts->Register("posterior-scale", &posterior_scale, "Scale for posteriors in "
                   "iVector extraction (may be viewed as inverse of prior scale)");
    opts->Register("max-count", &max_count, "Maximum data count we allow before "
                   "we start scaling the stats down (if nonzero)... helps to make "
                   "iVectors from long utterances look more typical.  Interpret "
                   "as a frame-count times --posterior-scale, typically 1/10 of "
                   "a number of frames.  Suggest 100.");
    opts->Register("use-most-recent-ivector", &use_most_recent_ivector, "If true, "
                   "always use most recent available iVector, rather than the "
                   "one for the designated frame.");
    opts->Register("greedy-ivector-extractor", &greedy_ivector_extractor, "If "
                   "true, 'read ahead' as many frames as we currently have available "
                   "when extracting the iVector.  May improve iVector quality.");
    opts->Register("max-remembered-frames", &max_remembered_frames, "The maximum "
                   "number of frames of adaptation history that we carry through "
                   "to later utterances of the same speaker (having a finite "
                   "number allows the speaker adaptation state to change over "
                   "time).  Interpret as a real frame count, i.e. not a count "
                   "scaled by --posterior-scale.");
  }
};

/// This struct contains various things that are needed (as const references)
/// by class OnlineIvectorExtractor.
struct OnlineIvectorExtractionInfo {

  Matrix<BaseFloat> lda_mat;  // LDA+MLLT matrix.
  Matrix<double> global_cmvn_stats;  // Global CMVN stats.

  OnlineCmvnOptions cmvn_opts;  // Options for online CMN/CMVN computation.
  OnlineSpliceOptions splice_opts;  // Options for frame splicing
                                    // (--left-context,--right-context)

  DiagGmm diag_ubm;
  IvectorExtractor extractor;

  // the following configuration variables are copied from
  // OnlineIvectorExtractionConfig, see comments there.
  int32 ivector_period;
  int32 num_gselect;
  BaseFloat min_post;
  BaseFloat posterior_scale;
  BaseFloat max_count;
  int32 num_cg_iters;
  bool use_most_recent_ivector;
  bool greedy_ivector_extractor;
  BaseFloat max_remembered_frames;

  OnlineIvectorExtractionInfo(const OnlineIvectorExtractionConfig &config);

  void Init(const OnlineIvectorExtractionConfig &config);

  // This constructor creates a version of this object where everything
  // is empty or zero.
  OnlineIvectorExtractionInfo();

  void Check() const;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineIvectorExtractionInfo);
};

/// This class stores the adaptation state from the online iVector extractor,
/// which can help you to initialize the adaptation state for the next utterance
/// of the same speaker in a more informed way.
struct OnlineIvectorExtractorAdaptationState {
  // CMVN state for the features used to get posteriors for iVector extraction;
  // online CMVN is not used for the features supplied to the neural net,
  // instead the iVector is used.

  // Adaptation state for online CMVN (used for getting posteriors for iVector)
  OnlineCmvnState cmvn_state;

  /// Stats for online iVector estimation.
  OnlineIvectorEstimationStats ivector_stats;

  /// This constructor initializes adaptation-state with no prior speaker history.
  OnlineIvectorExtractorAdaptationState(const OnlineIvectorExtractionInfo &info):
      cmvn_state(info.global_cmvn_stats),
      ivector_stats(info.extractor.IvectorDim(),
                    info.extractor.PriorOffset(),
                    info.max_count) { }

  /// Copy constructor
  OnlineIvectorExtractorAdaptationState(
      const OnlineIvectorExtractorAdaptationState &other);

  /// Scales down the stats if needed to ensure the number of frames in the
  /// speaker-specific CMVN stats does not exceed max_remembered_frames
  /// and the data-count in the iVector stats does not exceed
  /// max_remembered_frames * posterior_scale.  [the posterior_scale
  /// factor is necessary because those stats have already been scaled
  /// by that factor.]
  void LimitFrames(BaseFloat max_remembered_frames,
                   BaseFloat posterior_scale);

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};




/// OnlineIvectorFeature is an online feature-extraction class that's responsible
/// for extracting iVectors from raw features such as MFCC, PLP or filterbank.
/// Internally it processes the raw features using two different pipelines, one
/// online-CMVN+splice+LDA, and one just splice+LDA. It gets GMM posteriors from
/// the CMVN-normalized features, and with those and the unnormalized features
/// it obtains iVectors.

class OnlineIvectorFeature: public OnlineFeatureInterface {
 public:
  /// Constructor.  base_feature is for example raw MFCC or PLP or filterbank
  /// features, whatever was used to train the iVector extractor.
  /// "info" contains all the configuration information as well as
  /// things like the iVector extractor that we won't be modifying.
  /// Caution: the class keeps a const reference to "info", so don't
  /// delete it while this class or others copied from it still exist.
  explicit OnlineIvectorFeature(const OnlineIvectorExtractionInfo &info,
                                OnlineFeatureInterface *base_feature);

  // This version of the constructor accepts per-frame weights (relates to
  // downweighting silence).  This is intended for use in offline operation,
  // i.e. during training.  [will implement this when needed.]
  //explicit OnlineIvectorFeature(const OnlineIvectorExtractionInfo &info,
  //     std::vector<BaseFloat> frame_weights,
  //OnlineFeatureInterface *base_feature);


  // Member functions from OnlineFeatureInterface:

  /// Dim() will return the iVector dimension.
  virtual int32 Dim() const;
  virtual bool IsLastFrame(int32 frame) const;
  virtual int32 NumFramesReady() const;
  virtual BaseFloat FrameShiftInSeconds() const;
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  /// Set the adaptation state to a particular value, e.g. reflecting previous
  /// utterances of the same speaker; this will generally be called after
  /// constructing a new instance of this class.
  void SetAdaptationState(
      const OnlineIvectorExtractorAdaptationState &adaptation_state);


  /// Get the adaptation state; you may want to call this before destroying this
  /// object, to get adaptation state that can be used to improve decoding of
  /// later utterances of this speaker.
  void GetAdaptationState(
      OnlineIvectorExtractorAdaptationState *adaptation_state) const;

  virtual ~OnlineIvectorFeature();

  // Some diagnostics (not present in generic interface):
  // UBM log-like per frame:
  BaseFloat UbmLogLikePerFrame() const;
  // Objective improvement per frame from iVector estimation, versus default iVector
  // value, measured at utterance end.
  BaseFloat ObjfImprPerFrame() const;

  // returns number of frames seen (but not counting the posterior-scale).
  BaseFloat NumFrames() const {
    return ivector_stats_.NumFrames() / info_.posterior_scale;
  }


  // If you are downweighting silence, you can call
  // OnlineSilenceWeighting::GetDeltaWeights and supply the output to this class
  // using UpdateFrameWeights().  The reason why this call happens outside this
  // class, rather than this class pulling in the data weights, relates to
  // multi-threaded operation and also from not wanting this class to have
  // excessive dependencies.
  //
  // You must either always call this as soon as new data becomes available
  // (ideally just after calling AcceptWaveform), or never call it for the
  // lifetime of this object.
  void UpdateFrameWeights(
      const std::vector<std::pair<int32, BaseFloat> > &delta_weights);

 private:

  // This accumulates i-vector stats for a set of frames, specified as pairs
  // (t, weight).  The weights do not have to be positive.  (In the online
  // silence-weighting that we do, negative weights can occur if we change our
  // minds about the assignment of a frame as silence vs. non-silence).
  void UpdateStatsForFrames(
      const std::vector<std::pair<int32, BaseFloat> > &frame_weights);

  // Returns a modified version of info_.min_post, which is opts_.min_post if
  // weight is 1.0 or -1.0, but gets larger if fabs(weight) is small... but no
  // larger than 0.99.  (This is an efficiency thing, to not bother processing
  // very small counts).
  BaseFloat GetMinPost(BaseFloat weight) const;

  // This is the original UpdateStatsUntilFrame that is called when there is
  // no data-weighting involved.
  void UpdateStatsUntilFrame(int32 frame);

  // This is the new UpdateStatsUntilFrame that is called when there is
  // data-weighting (i.e. when the user has been calling UpdateFrameWeights()).
  void UpdateStatsUntilFrameWeighted(int32 frame);

  void PrintDiagnostics() const;

  const OnlineIvectorExtractionInfo &info_;

  OnlineFeatureInterface *base_;  // The feature this is built on top of
                                  // (e.g. MFCC); not owned here

  OnlineFeatureInterface *lda_;  // LDA on top of raw+splice features.
  OnlineCmvn *cmvn_;  // the CMVN that we give to the lda_normalized_.
  OnlineFeatureInterface *lda_normalized_;  // LDA on top of CMVN+splice

  // the following is the pointers to OnlineFeatureInterface objects that are
  // owned here and which we need to delete.
  std::vector<OnlineFeatureInterface*> to_delete_;

  /// the iVector estimation stats
  OnlineIvectorEstimationStats ivector_stats_;

  /// num_frames_stats_ is the number of frames of data we have already
  /// accumulated from this utterance and put in ivector_stats_.  Each frame t <
  /// num_frames_stats_ is in the stats.  In case you are doing the
  /// silence-weighted iVector estimation, with UpdateFrameWeights() being
  /// called, this variable is still used but you may later have to revisit
  /// earlier frames to adjust their weights... see the code.
  int32 num_frames_stats_;

  /// delta_weights_ is written to by UpdateFrameWeights,
  /// in the case where the iVector estimation is silence-weighted using the decoder
  /// traceback.  Its elements are consumed by UpdateStatsUntilFrameWeighted().
  /// We provide std::greater<std::pair<int32, BaseFloat> > > as the comparison type
  /// (default is std::less) so that the lowest-numbered frame, not the highest-numbered
  /// one, will be returned by top().
  std::priority_queue<std::pair<int32, BaseFloat>,
                      std::vector<std::pair<int32, BaseFloat> >,
                      std::greater<std::pair<int32, BaseFloat> > > delta_weights_;

  /// this is only used for validating that the frame-weighting code is not buggy.
  std::vector<BaseFloat> current_frame_weight_debug_;

  /// delta_weights_provided_ is set to true if UpdateFrameWeights was ever called; it's
  /// used to detect wrong usage of this class.
  bool delta_weights_provided_;
  /// The following is also used to detect wrong usage of this class; it's set
  /// to true if UpdateStatsUntilFrame() was ever called.
  bool updated_with_no_delta_weights_;

  /// if delta_weights_ was ever called, this keeps track of the most recent
  /// frame that ever had a weight.  It's mostly for detecting errors.
  int32 most_recent_frame_with_weight_;

  /// The following is only needed for diagnostics.
  double tot_ubm_loglike_;

  /// Most recently estimated iVector, will have been
  /// estimated at the greatest time t where t <= num_frames_stats_ and
  /// t % info_.ivector_period == 0.
  Vector<double> current_ivector_;

  /// if info_.use_most_recent_ivector == false, we need to store
  /// the iVector we estimated each info_.ivector_period frames so that
  /// GetFrame() can return the iVector that was active on that frame.
  /// ivectors_history_[i] contains the iVector we estimated on
  /// frame t = i * info_.ivector_period.
  std::vector<Vector<BaseFloat>* > ivectors_history_;

};


struct OnlineSilenceWeightingConfig {
  std::string silence_phones_str;
  // The weighting factor that we apply to silence phones in the iVector
  // extraction.  This option is only relevant if the --silence-phones option is
  // set.
  BaseFloat silence_weight;

  // Transition-ids that get repeated at least this many times (if
  // max_state_duration > 0) are treated as silence.
  BaseFloat max_state_duration;

  // This is the scale that we apply to data that we don't yet have a decoder
  // traceback for, in the online silence
  BaseFloat new_data_weight;

  bool Active() const {
    return !silence_phones_str.empty() && silence_weight != 1.0;
  }

  OnlineSilenceWeightingConfig():
      silence_weight(1.0), max_state_duration(-1) { }

  void Register(OptionsItf *opts) {
    opts->Register("silence-phones", &silence_phones_str, "(RE weighting in "
                   "iVector estimation for online decoding) List of integer ids of "
                   "silence phones, separated by colons (or commas).  Data that "
                   "(according to the traceback of the decoder) corresponds to "
                   "these phones will be downweighted by --silence-weight.");
    opts->Register("silence-weight", &silence_weight, "(RE weighting in "
                   "iVector estimation for online decoding) Weighting factor for frames "
                   "that the decoder trace-back identifies as silence; only "
                   "relevant if the --silence-phones option is set.");
    opts->Register("max-state-duration", &max_state_duration, "(RE weighting in "
                   "iVector estimation for online decoding) Maximum allowed "
                   "duration of a single transition-id; runs with durations longer "
                   "than this will be weighted down to the silence-weight.");
  }
  // e.g. prefix = "ivector-silence-weighting"
  void RegisterWithPrefix(std::string prefix, OptionsItf *opts) {
    ParseOptions po_prefix(prefix, opts);
    this->Register(&po_prefix);
  }
};

// This class is responsible for keeping track of the best-path traceback from
// the decoder (efficiently) and computing a weighting of the data based on the
// classification of frames as silence (or not silence)... also with a duration
// limitation, so data from a very long run of the same transition-id will get
// weighted down.  (this is often associated with misrecognition or silence).
class OnlineSilenceWeighting {
 public:
  // Note: you would initialize a new copy of this object for each new
  // utterance.
  // The frame-subsampling-factor is used for newer nnet3 models, especially
  // chain models, when the frame-rate of the decoder is different from the
  // frame-rate of the input features.  E.g. you might set it to 3 for such
  // models.

  OnlineSilenceWeighting(const TransitionModel &trans_model,
                         const OnlineSilenceWeightingConfig &config,
			 int32 frame_subsampling_factor = 1);

  bool Active() const { return config_.Active(); }

  // This should be called before GetDeltaWeights, so this class knows about the
  // traceback info from the decoder.  It records the traceback information from
  // the decoder using its BestPathEnd() and related functions.
  // It will be instantiated for FST == fst::Fst<fst::StdArc> and fst::GrammarFst.
  template <typename FST>
  void ComputeCurrentTraceback(const LatticeFasterOnlineDecoderTpl<FST> &decoder);

  // Calling this function gets the changes in weight that require us to modify
  // the stats... the output format is (frame-index, delta-weight).  The
  // num_frames_ready argument is the number of frames available at the input
  // (or equivalently, output) of the online iVector extractor class, which may
  // be more than the currently available decoder traceback.  How many frames
  // of weights it outputs depends on how much "num_frames_ready" increased
  // since last time we called this function, and whether the decoder traceback
  // changed.  Negative delta_weights might occur if frames previously
  // classified as non-silence become classified as silence if the decoder's
  // traceback changes.  You must call this function with "num_frames_ready"
  // arguments that only increase, not decrease, with time.  You would provide
  // this output to class OnlineIvectorFeature by calling its function
  // UpdateFrameWeights with the output.
  void GetDeltaWeights(
      int32 num_frames_ready_in,
      std::vector<std::pair<int32, BaseFloat> > *delta_weights);

 private:
  const TransitionModel &trans_model_;
  const OnlineSilenceWeightingConfig &config_;

  int32 frame_subsampling_factor_;

  unordered_set<int32> silence_phones_;

  struct FrameInfo {
    // The only reason we need the token pointer is to know far back we have to
    // trace before the traceback is the same as what we previously traced back.
    void *token;
    int32 transition_id;
    // current_weight is the weight we've previously told the iVector
    // extractor to use for this frame, if any.  It may not equal the
    // weight we "want" it to use (any difference between the two will
    // be output when the user calls GetDeltaWeights().
    BaseFloat current_weight;
    FrameInfo(): token(NULL), transition_id(-1), current_weight(0.0) {}
  };

  // gets the frame at which we need to begin our processing in
  // GetDeltaWeights...  normally this is equal to
  // num_frames_output_and_correct_, but it may be earlier in case
  // max_state_duration is relevant.
  int32 GetBeginFrame();

  // This contains information about any previously computed traceback;
  // when the traceback changes we use this variable to compare it with the
  // previous traceback.
  // It's indexed at the frame-rate of the decoder (may be different
  // by 'frame_subsampling_factor_' from the frame-rate of the features.
  std::vector<FrameInfo> frame_info_;

  // This records how many frames have been output and that currently reflect
  // the traceback accurately.  It is used to avoid GetDeltaWeights() having to
  // visit each frame as far back as t = 0, each time it is called.
  // GetDeltaWeights() sets this to the number of frames that it output, and
  // ComputeCurrentTraceback() then reduces it to however far it traced back.
  // However, we may have to go further back in time than this in order to
  // properly honor the "max-state-duration" config.  This, if needed, is done
  // in GetDeltaWeights() before outputting the delta weights.
  int32 num_frames_output_and_correct_;
};


/// @} End of "addtogroup onlinefeat"
}  // namespace kaldi

#endif  // KALDI_ONLINE2_ONLINE_IVECTOR_FEATURE_H_
