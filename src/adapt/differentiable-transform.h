// adapt/differentiable-transform.h

// Copyright      2018  Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_TRANSFORM_DIFFERENTIABLE_TRANSFORM_H_
#define KALDI_TRANSFORM_DIFFERENTIABLE_TRANSFORM_H_

#include <vector>

#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"
#include "cudamatrix/cu-matrix.h"
#include "adapt/differentiable-transform-itf.h"
#include "adapt/differentiable-fmllr.h"


// This header contains the 'base-cases' of DifferentiableTransform: namely,
// FmllrTransform and MeanOnlyTransform.  See also generic-transform.h where
// sequence, append and no-op types are defined.
namespace kaldi {
namespace differentiable_transform {


/**
   This is a version of the transform class that implements fMLLR (with
   spherical variances, to make the update equations non-iterative); see
   differentiable-fmllr.h where the core parts of this are implemented,
   this provides the interface compatible with DifferentiableTransform.

   Please see the comments in class DifferentiableTransform (in
   differentiable-transform-itf.h) for the meaning and usage of the various
   interface functions and their parameters.
*/
class FmllrTransform: public DifferentiableTransform {
 public:
  int32 InitFromConfig(int32 cur_pos,
                       std::vector<ConfigLine> *config_lines) override;


  int32 Dim() const override { return dim_; }

  MinibatchInfoItf* TrainingForward(
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      CuMatrixBase<BaseFloat> *output) const override;

  virtual void TrainingBackward(
      const CuMatrixBase<BaseFloat> &input,
      const CuMatrixBase<BaseFloat> &output_deriv,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      MinibatchInfoItf *minibatch_info,
      CuMatrixBase<BaseFloat> *input_deriv) const override;

  void Accumulate(
      int32 final_iter,
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors) override;

  void Estimate(int32 final_iter) override;

  int32 NumFinalIterations() override { return 1; }

  SpeakerStatsItf *GetEmptySpeakerStats() const override;

  void TestingAccumulate(
      const MatrixBase<BaseFloat> &input,
      const SubPosterior &posteriors,
      SpeakerStatsItf *speaker_stats) const override;

  void TestingForward(
      const MatrixBase<BaseFloat> &input,
      const SpeakerStatsItf &speaker_stats,
      MatrixBase<BaseFloat> *output) const override;

  FmllrTransform(const FmllrTransform &other);

  FmllrTransform(): target_model_(NULL) { }

  std::string Type() const override { return "FmllrTransform"; }

  DifferentiableTransform* Copy() const override;

  void Write(std::ostream &os, bool binary) const override;

  void Read(std::istream &is, bool binary) override;

  ~FmllrTransform();
 private:
  int32 dim_;

  FmllrEstimatorOptions fmllr_opts_;

  // Note: this target model is only for use in test time.  We allocate it the
  // first time Accumulate() is called.  In training time we estimate it
  // minibatch by minibatch (which is why we don't expect to have that many
  // classes).  At the end of training we'll accumulate stats here in
  // Accumulate(), and Estimate() will estimate it.
  GaussianEstimator *target_model_;
};

class FmllrMinibatchInfo: public MinibatchInfoItf {
 public:

  FmllrMinibatchInfo(int32 num_classes, int32 dim, int32 num_speakers);

  GaussianEstimator target_model;

  // One estimator of Fmllr per speaker.  Make them pointers so we don't have to
  // implement self-constructor for class FmllrEstimator.
  std::vector<FmllrEstimator*> estimators;

  ~FmllrMinibatchInfo();
};

class FmllrSpeakerStats: public SpeakerStatsItf {
 public:
  // Caution: this object maintains references to mu and s, so it's not a good
  // idea to let the target-model (which lives in the FmllrTransform object) be
  // deleted during the lifetime of this object.
  FmllrSpeakerStats(const FmllrEstimatorOptions &opts,
                    const MatrixBase<BaseFloat> &mu,
                    const VectorBase<BaseFloat> &s):
      estimator(opts, mu, s) { }

  void Estimate() override { estimator.Estimate(); }

  FmllrEstimator estimator;

  ~FmllrSpeakerStats() { }
};

/**
   This version of the transform class does a mean normalization: adding an
   offset to its input so that the difference (per speaker) of the transformed
   class means from the speaker-independent class means is minimized.
   This is like a mean-only fMLLR with fixed (say, unit) covariance model.
 */
class MeanOnlyTransform: public DifferentiableTransform {
 public:
  /*
    Example config line:

    MeanOnlyTransform dim=100
   */
  int32 InitFromConfig(int32 cur_pos,
                       std::vector<ConfigLine> *config_lines) override;


  int32 Dim() const override { return dim_; }

  MinibatchInfoItf* TrainingForward(
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      CuMatrixBase<BaseFloat> *output) const override;

  virtual void TrainingBackward(
      const CuMatrixBase<BaseFloat> &input,
      const CuMatrixBase<BaseFloat> &output_deriv,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      MinibatchInfoItf *minibatch_info,
      CuMatrixBase<BaseFloat> *input_deriv) const override;

  void Accumulate(
      int32 final_iter,
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors) override;

  void Estimate(int32 final_iter) override;

  int32 NumFinalIterations() override { return 1; }

  SpeakerStatsItf *GetEmptySpeakerStats() const override;

  void TestingAccumulate(
      const MatrixBase<BaseFloat> &input,
      const SubPosterior &posteriors,
      SpeakerStatsItf *speaker_stats) const override;

  void TestingForward(
      const MatrixBase<BaseFloat> &input,
      const SpeakerStatsItf &speaker_stats,
      MatrixBase<BaseFloat> *output) const override;

  MeanOnlyTransform(const MeanOnlyTransform &other);

  MeanOnlyTransform(): target_model_(NULL) { }

  std::string Type() const override { return "MeanOnlyTransform"; }

  DifferentiableTransform* Copy() const override;

  void Write(std::ostream &os, bool binary) const override;

  void Read(std::istream &is, bool binary) override;

  ~MeanOnlyTransform();
 private:
  int32 dim_;

  // Note: this target model is only for use in test time.  We allocate it the
  // first time Accumulate() is called.  In training time we estimate it
  // minibatch by minibatch (which is why we don't expect to have that many
  // classes).  At the end of training we'll accumulate stats here in
  // Accumulate(), and Estimate() will estimate it.
  GaussianEstimator *target_model_;
};

class MeanOnlyTransformMinibatchInfo: public MinibatchInfoItf {
 public:

  MeanOnlyTransformMinibatchInfo(int32 num_classes, int32 dim,
                                 int32 num_speakers);

  GaussianEstimator target_model;

  // One estimator of offset per speaker.  Make them pointers so we don't have to
  // implement self-constructor for class FmllrEstimator.
  std::vector<MeanOnlyTransformEstimator*> estimators;

  ~MeanOnlyTransformMinibatchInfo();
};

class MeanOnlyTransformSpeakerStats: public SpeakerStatsItf {
 public:
  // Caution: this object maintains a reference to mu, so it's not a good idea
  // to let the target-model (which lives in the FmllrTransform object) be
  // deleted during the lifetime of this object.
  MeanOnlyTransformSpeakerStats(const MatrixBase<BaseFloat> &mu):
      estimator(mu) { }

  void Estimate() override { estimator.Estimate(); }

  MeanOnlyTransformEstimator estimator;

  ~MeanOnlyTransformSpeakerStats() { }
};





} // namespace differentiable_transform
} // namespace kaldi

#endif  // KALDI_TRANSFORM_DIFFERENTIABLE_TRANSFORM_H_
