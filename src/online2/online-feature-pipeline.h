// online2/online-feature-pipeline.h

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


#ifndef KALDI_ONLINE2_ONLINE_FEATURE_PIPELINE_H_
#define KALDI_ONLINE2_ONLINE_FEATURE_PIPELINE_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "feat/online-feature.h"
#include "feat/pitch-functions.h"

namespace kaldi {
/// @addtogroup  onlinefeat OnlineFeatureExtraction
/// @{

/// @file
/// This file contains a class OnlineFeaturePipeline for online feature
/// extraction, which puts together various pieces into something that
/// has a convenient interface.

/// This configuration class is to set up OnlineFeaturePipelineConfig, which
/// in turn is the configuration class for OnlineFeaturePipeline.
/// Instead of taking the options for the parts of the feature pipeline
/// directly, it reads in the names of configuration classes.
/// I'm conflicted about whether this is a wise thing to do, but I think
/// for ease of scripting it's probably better to do it like this.
struct OnlineFeaturePipelineCommandLineConfig {
  std::string feature_type;
  std::string mfcc_config;
  std::string plp_config;
  std::string fbank_config;
  bool add_pitch;
  std::string pitch_config;
  std::string pitch_process_config;
  std::string cmvn_config;
  std::string global_cmvn_stats_rxfilename;
  bool add_deltas;
  std::string delta_config;
  bool splice_feats;
  std::string splice_config;
  std::string lda_rxfilename;

  OnlineFeaturePipelineCommandLineConfig() :
    feature_type("mfcc"), add_pitch(false), add_deltas(false),
    splice_feats(false) { }

  void Register(OptionsItf *opts) {
    opts->Register("feature-type", &feature_type,
                   "Base feature type [mfcc, plp, fbank]");
    opts->Register("mfcc-config", &mfcc_config, "Configuration file for "
                   "MFCC features (e.g. conf/mfcc.conf)");
    opts->Register("plp-config", &plp_config, "Configuration file for "
                   "PLP features (e.g. conf/plp.conf)");
    opts->Register("fbank-config", &fbank_config, "Configuration file for "
                   "filterbank features (e.g. conf/fbank.conf)");
    opts->Register("add-pitch", &add_pitch, "Append pitch features to raw "
                   "MFCC/PLP features.");
    opts->Register("pitch-config", &pitch_config, "Configuration file for "
                   "pitch features (e.g. conf/pitch.conf)");
    opts->Register("pitch-process-config", &pitch_process_config,
                   "Configuration file for post-processing pitch features "
                   "(e.g. conf/pitch_process.conf)");
    opts->Register("cmvn-config", &cmvn_config, "Configuration class "
                   "file for online CMVN features (e.g. conf/online_cmvn.conf)");
    opts->Register("global-cmvn-stats", &global_cmvn_stats_rxfilename,
                   "(Extended) filename for global CMVN stats, e.g. obtained "
                   "from 'matrix-sum scp:data/train/cmvn.scp -'");
    opts->Register("add-deltas", &add_deltas,
                   "Append delta features.");
    opts->Register("delta-config", &delta_config, "Configuration file for "
                   "delta feature computation (if not supplied, will not apply "
                   "delta features; supply empty config to use defaults.)");
    opts->Register("splice-feats", &splice_feats, "Splice features with left and "
                   "right context.");
    opts->Register("splice-config", &splice_config, "Configuration file "
                   "for frame splicing, if done (e.g. prior to LDA)");
    opts->Register("lda-matrix", &lda_rxfilename, "Filename of LDA matrix (if "
                   "using LDA), e.g. exp/foo/final.mat");
  }
};



/// This configuration class is responsible for storing the configuration
/// options for OnlineFeaturePipeline, but it does not set them.  To do that you
/// should use OnlineFeaturePipelineCommandLineConfig, which can read in the
/// configuration from config files on disk.  The reason for structuring it this
/// way with two config files, is to make it easier to configure from code as
/// well as from the command line.
struct OnlineFeaturePipelineConfig {
  OnlineFeaturePipelineConfig():
      feature_type("mfcc"), add_pitch(false), add_deltas(true),
      splice_feats(false) { }

  OnlineFeaturePipelineConfig(
      const OnlineFeaturePipelineCommandLineConfig &cmdline_config);

  BaseFloat FrameShiftInSeconds() const;

  std::string feature_type;  // "mfcc" or "plp" or "fbank"

  MfccOptions mfcc_opts;  // options for MFCC computation,
                          // if feature_type == "mfcc"
  PlpOptions plp_opts;  // Options for PLP computation, if feature_type == "plp"
  FbankOptions fbank_opts;  // Options for filterbank computation, if
                            // feature_type == "fbank"

  bool add_pitch;
  PitchExtractionOptions pitch_opts;  // Options for pitch extraction, if done.
  ProcessPitchOptions pitch_process_opts;  // Options for pitch
                                                   // processing

  OnlineCmvnOptions cmvn_opts;  // Options for online CMN/CMVN computation.

  bool add_deltas;
  DeltaFeaturesOptions delta_opts;  // Options for delta computation, if done.

  bool splice_feats;
  OnlineSpliceOptions splice_opts;  // Options for frame splicing, if done.

  std::string lda_rxfilename;  // Filename for reading LDA or LDA+MLLT matrix,
                               // if used.
  std::string global_cmvn_stats_rxfilename;  // Filename used for reading global
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
  explicit OnlineFeaturePipeline(const OnlineFeaturePipelineConfig &cfg);

  /// Member functions from OnlineFeatureInterface:
  virtual int32 Dim() const;
  virtual bool IsLastFrame(int32 frame) const;
  virtual int32 NumFramesReady() const;
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  // This is supplied for debug purposes.
  void GetAsMatrix(Matrix<BaseFloat> *feats);
  
  void FreezeCmvn();  // stop it from moving further (do this when you start
                      // using fMLLR). This will crash if NumFramesReady() == 0.

  /// Set the CMVN state to a particular value (will generally be
  /// called after Copy().
  void SetCmvnState(const OnlineCmvnState &cmvn_state);
  void GetCmvnState(OnlineCmvnState *cmvn_state);

  /// Accept more data to process (won't actually process it, will
  /// just copy it).   sampling_rate is necessary just to assert
  /// it equals what's in the config.
  void AcceptWaveform(BaseFloat sampling_rate,
                      const VectorBase<BaseFloat> &waveform);

  BaseFloat FrameShiftInSeconds() const {
    return config_.FrameShiftInSeconds();
  }

  // InputFinished() tells the class you won't be providing any
  // more waveform.  This will help flush out the last few frames
  // of delta or LDA features, and finalize the pitch features
  // (making them more accurate).
  void InputFinished();

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
  Matrix<BaseFloat> lda_mat_;  // LDA matrix, if supplied.
  Matrix<BaseFloat> global_cmvn_stats_;  // Global CMVN stats.

  OnlineBaseFeature *base_feature_;        // MFCC/PLP
  OnlinePitchFeature *pitch_;              // Raw pitch
  OnlineProcessPitch *pitch_feature_;  // Processed pitch
  OnlineFeatureInterface *feature_;        // CMVN (+ processed pitch)

  OnlineCmvn *cmvn_;
  OnlineFeatureInterface *splice_or_delta_;  // This may be NULL if we're not
                                             // doing splicing or deltas.

  OnlineFeatureInterface *lda_;  // If non-NULL, the LDA or LDA+MLLT transform.

  /// returns lda_ if it exists, else splice_or_delta_, else cmvn_.  If this
  /// were not private we would have const and non-const versions returning
  /// const and non-const pointers.
  OnlineFeatureInterface* UnadaptedFeature() const;

  OnlineFeatureInterface *fmllr_;  // non-NULL if we currently have an fMLLR
                                   // transform.

  /// returns adapted feature if fmllr_ exists, else UnadaptedFeature().  If
  /// this were not private we would have const and non-const versions returning
  /// const and non-const pointers.
  OnlineFeatureInterface* AdaptedFeature() const;
};




/// @} End of "addtogroup onlinefeat"
}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_FEATURE_PIPELINE_H_
