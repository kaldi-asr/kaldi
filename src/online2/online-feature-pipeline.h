// online2/online-feature-pipeline.h

// Copyright 2013   Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_ONLINE2_ONLINE_FEATURE_PIPELINE_H_
#define KALDI_ONLINE2_ONLINE_FEATURE_PIPELINE_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "online2/online-feature.h"

namespace kaldi {
/// @addtogroup  onlinefeat OnlineFeatureExtraction
/// @{


/// This configuration class is to set up OnlineFeaturePipelineConfig, which
/// in turn is the configuration class for OnlineFeaturePipeline.
/// Instead of taking the options for the parts of the feature pipeline
/// directly, it reads in the names of configuration classes.
/// I'm conflicted about whether this is a wise thing to do, but I think
/// for ease of scripting it's probably better to do it like this.
/// 
struct OnlineFeaturePipelineCommandLineConfig {
  std::string mfcc_config;
  std::string plp_config;
  // later will have:
  // std::string pitch_config;
  std::string cmvn_config;
  std::string global_cmvn_stats_rxfilename;
  std::string delta_config;
  std::string splice_config;
  std::string lda_rxfilename;
  
  OnlineFeaturePipelineCommandLineConfig() { }

  void Register(OptionsItf *po) {
    po->Register("mfcc-config", &mfcc_config, "Configuration class file for MFCC "
                 "features (e.g. conf/mfcc.conf)");
    po->Register("plp-config", &plp_config, "Configuration class file for PLP "
                 "features (e.g. conf/plp.conf)");
    po->Register("cmvn-config", &cmvn_config, "Configuration class "
                 "file for online CMVN features (e.g. conf/online_cmvn.conf)");
    po->Register("global-cmvn-stats", &global_cmvn_stats_rxfilename,
                 "(Extended) filename for global CMVN stats, e.g. obtained from "
                 "'matrix-sum scp:data/train/cmvn.scp -'");
    po->Register("delta-config", &delta_config, "Configuration class file for "
                 "delta feature computation (if not supplied, will not apply "
                 "delta features; supply empty config to use defaults.)");
    po->Register("splice-config", &splice_config, "Configuration class file for "
                 "frame splicing, if done (e.g. prior to LDA)");
    po->Register("lda-matrix", &lda_rxfilename, "Filename of LDA matrix (if using "
                 "LDA), e.g. exp/foo/final.mat");
  }
};



/// This configuration class is responsible for storing the configuration
/// options for OnlineFeaturePipeline, but it does not set them.  To do that you
/// should use OnlineFeaturePipelineCommandLineConfig, which can read in in the
/// configuration from config files on disk.  The reason for structuring it this
/// way with two config files, is to make it easier to configure from code as
/// well as from the command line.
struct OnlineFeaturePipelineConfig {
  OnlineFeaturePipelineConfig():
      feature_type("mfcc"), splice_frames(false), apply_deltas(true) { }

  OnlineFeaturePipelineConfig(
      const OnlineFeaturePipelineCommandLineConfig &cmdline_config);
  
  std::string feature_type; // "mfcc" or "plp", for now.
                            // When we add pitch we'll have a separate "add_pitch"
                            // boolean variable.
  MfccOptions mfcc_opts; // options for MFCC computation, if feature_type == "mfcc"
  PlpOptions plp_opts; // Options for PLP computation, if feature_type == "plp"

  OnlineCmvnOptions cmvn_opts; // Options for online CMN/CMVN computation.

  bool splice_frames;
  OnlineSpliceOptions splice_opts; // Options for frame splicing, if done.
  
  bool apply_deltas;
  DeltaFeaturesOptions delta_opts; // Options for delta computation, if done.

  std::string lda_rxfilename; // Filename for reading LDA or LDA+MLLT matrix, if
                              // used.
  std::string global_cmvn_stats_rxfilename; // Filename used for reading global
                                            // CMVN stats
};



/// OnlineFeaturePipeline is a class that's responsible for putting together the
/// various stages of the feature-processing pipeline, in an online setting.
/// This does not attempt to be fully generic, we just try to handle the common
/// case.  Since the online-decoding code needs to "know about" things like CMN
/// and fMLLR in order to do adaptation, it's hard to make this completely
/// generic.
class OnlineFeaturePipeline: public OnlineFeatureInterface {
 public:
  OnlineFeaturePipeline(const OnlineFeaturePipelineConfig &cfg);

  /// Member functions from OnlineFeatureInterface:
  virtual int32 Dim() const;
  virtual bool IsLastFrame(int32 frame) const;
  virtual int32 NumFramesReady() const;
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);
  
  void FreezeCmvn(); // stop it from moving further (do this when you start using
                     // fMLLR).  This will crash if NumFramesReady() == 0.

  /// Set the CMVN state to a particular value (will generally be
  /// called after Copy().
  void SetCmvnState(const OnlineCmvnState &cmvn_state);
  void GetCmvnState(OnlineCmvnState *cmvn_state);
  
  /// Accept more data to process (won't actually process it, will
  /// just copy it).   sampling_rate is necessary just to assert
  /// it equals what's in the config.
  void AcceptWaveform(BaseFloat sampling_rate,
                      const VectorBase<BaseFloat> &waveform);
  
  // This object is used to set the fMLLR transform.  Call it with
  // the empty matrix if you want to stop it using any transform.
  void SetTransform(const MatrixBase<BaseFloat> &transform);


  // Returns true if an fMLLR transform has been set using
  // SetTransform().
  bool HaveFmllrTransform() { return fmllr_ != NULL; }
  
  /// returns a newly initialized copy of *this-- this does not duplicate all
  /// the internal state or the speaker-adaptation state, but gives you a
  /// freshly initialized version of this object, as if you had initialized it
  /// using the constructor that takes the config file.  After calling this you
  /// may want to call SetCmvnState() and SetTransform().
  OnlineFeaturePipeline *New() const;

  virtual ~OnlineFeaturePipeline();
  
 private:
  /// The following constructor is used internally in the New() function;
  /// it has the same effect as initializing from just "cfg", but avoids
  /// re-reading the LDA transform from disk.
  OnlineFeaturePipeline(const OnlineFeaturePipelineConfig &cfg,
                        const Matrix<BaseFloat> &lda_mat,
                        const Matrix<BaseFloat> &global_cmvn_stats);

  /// Init() is to be called from the constructor; it assumes the pointer
  /// members are all uninitialized but config_ and lda_mat_ are
  /// initialized.
  void Init();
  
  OnlineFeaturePipelineConfig config_;
  Matrix<BaseFloat> lda_mat_; // LDA matrix, if supplied.
  Matrix<BaseFloat> global_cmvn_stats_; // Global CMVN stats.
  
  OnlineBaseFeature *base_feature_;
  // base_feature_ is the MFCC or PLP feature.
  // In future if we want to append pitch features, we'll add a pitch_ member
  // here and a member that appends the pitch and mfcc/plp features.

  OnlineCmvn *cmvn_;
  OnlineFeatureInterface *splice_or_delta_; // This may be NULL if we're not
                                            // doing splicing or deltas.
  OnlineFeatureInterface *lda_; // If non-NULL, the LDA or LDA+MLLT transform.

  /// returns lda_ if it exists, else splice_or_delta_, else cmvn_.  If this
  /// were not private we would have const and non-const versions returning
  /// const and non-const pointers.
  OnlineFeatureInterface* UnadaptedFeature() const;

  OnlineFeatureInterface *fmllr_; // non-NULL if we currently have an fMLLR
                                  // transform.

  /// returns adapted feature if fmllr_ exists, else UnadaptedFeature().  If
  /// this were not private we would have const and non-const versions returning
  /// const and non-const pointers.
  OnlineFeatureInterface* AdaptedFeature() const;
};




/// @} End of "addtogroup onlinefeat"
}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_FEATURE_PIPELINE_H_
