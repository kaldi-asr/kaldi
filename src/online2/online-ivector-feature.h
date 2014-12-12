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
                                   use_most_recent_ivector(true),
                                   greedy_ivector_extractor(false),
                                   max_remembered_frames(1000) { }
  
  void Register(OptionsItf *po) {
    po->Register("lda-matrix", &lda_mat_rxfilename, "Filename of LDA matrix, "
                 "e.g. final.mat; used for iVector extraction. ");
    po->Register("global-cmvn-stats", &global_cmvn_stats_rxfilename,
                 "(Extended) filename for global CMVN stats, used in iVector "
                 "extraction, obtained for example from "
                 "'matrix-sum scp:data/train/cmvn.scp -', only used for "
                 "iVector extraction");
    po->Register("cmvn-config", &cmvn_config_rxfilename, "Configuration "
                 "file for online CMVN features (e.g. conf/online_cmvn.conf),"
                 "only used for iVector extraction");
    po->Register("splice-config", &splice_config_rxfilename, "Configuration file "
                 "for frame splicing (--left-context and --right-context "
                 "options); used for iVector extraction.");
    po->Register("diag-ubm", &diag_ubm_rxfilename, "Filename of diagonal UBM "
                 "used to obtain posteriors for iVector extraction, e.g. "
                 "final.dubm");
    po->Register("ivector-extractor", &ivector_extractor_rxfilename,
                 "Filename of iVector extractor, e.g. final.ie");
    po->Register("ivector-period", &ivector_period, "Frequency with which "
                 "we extract iVectors for neural network adaptation");
    po->Register("num-gselect", &num_gselect, "Number of Gaussians to select "
                 "for iVector extraction");
    po->Register("min-post", &min_post, "Threshold for posterior pruning in "
                 "iVector extraction");
    po->Register("posterior-scale", &posterior_scale, "Scale for posteriors in "
                 "iVector extraction (may be viewed as inverse of prior scale)");
    po->Register("use-most-recent-ivector", &use_most_recent_ivector, "If true, "
                 "always use most recent available iVector, rather than the "
                 "one for the designated frame.");
    po->Register("greedy-ivector-extractor", &greedy_ivector_extractor, "If "
                 "true, 'read ahead' as many frames as we currently have available "
                 "when extracting the iVector.  May improve iVector quality.");
    po->Register("max-remembered-frames", &max_remembered_frames, "The maximum "
                 "number of frames of adaptation history that we carry through "
                 "to later utterances of the same speaker (having a finite "
                 "number allows the speaker adaptation state to change over "
                 "time");
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
                    info.extractor.PriorOffset()) { }
  
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
  
  // Member functions from OnlineFeatureInterface:

  /// Dim() will return the iVector dimension.
  virtual int32 Dim() const;
  virtual bool IsLastFrame(int32 frame) const;
  virtual int32 NumFramesReady() const;
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
  
 private:
  virtual void UpdateStatsUntilFrame(int32 frame);
  void PrintDiagnostics() const;
  
  const OnlineIvectorExtractionInfo &info_;

  // base_ is the base feature; it is not owned here.
  OnlineFeatureInterface *base_;
  // the following online-feature-extractor pointers are owned here:
  OnlineSpliceFrames *splice_; // splice on top of raw features.
  OnlineTransform *lda_;  // LDA on top of raw+splice features.
  OnlineCmvn *cmvn_;
  OnlineSpliceFrames *splice_normalized_; // splice on top of CMVN feats.
  OnlineTransform *lda_normalized_;  // LDA on top of CMVN+splice

  /// the iVector estimation stats
  OnlineIvectorEstimationStats ivector_stats_;

  /// num_frames_stats_ is the number of frames of data we have already
  /// accumulated from this utterance and put in ivector_stats_.
  int32 num_frames_stats_;

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

/// @} End of "addtogroup onlinefeat"
}  // namespace kaldi

#endif  // KALDI_ONLINE2_ONLINE_NNET2_FEATURE_PIPELINE_H_
