// adapt/generic-transform.h

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


#ifndef KALDI_TRANSFORM_GENERIC_TRANSFORM_H_
#define KALDI_TRANSFORM_GENERIC_TRANSFORM_H_

#include <vector>
#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"
#include "cudamatrix/cu-matrix.h"
#include "adapt/differentiable-transform-itf.h"

// This header contains 'generic' forms of differentiable transform, which allow
// you to append more basic transforms together or concatenate them dimension-wise.
// Also it includes a no-op transform.

namespace kaldi {
namespace differentiable_transform {


/**
   This is a version of the transform class that does nothing.  It's potentially
   useful for situations where you want to apply speaker normalization to some
   dimensions of the feature vector but not to others.
 */
class NoOpTransform: public DifferentiableTransform {
 public:

  int32 InitFromConfig(int32 cur_pos,
                       std::vector<ConfigLine> *config_lines) override;

  int32 Dim() const override { return dim_; }

  MinibatchInfoItf* TrainingForward(
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      CuMatrixBase<BaseFloat> *output) const override {
    output->CopyFromMat(input);
    return NULL;
  }
  virtual void TrainingBackward(
      const CuMatrixBase<BaseFloat> &input,
      const CuMatrixBase<BaseFloat> &output_deriv,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      MinibatchInfoItf *minibatch_info,
      CuMatrixBase<BaseFloat> *input_deriv) const override {
    KALDI_ASSERT(minibatch_info == NULL);
    input_deriv->AddMat(1.0, output_deriv);
  }

  virtual int32 NumFinalIterations() { return 0; }

  void Accumulate(
      int32 final_iter,
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors) override { }


  SpeakerStatsItf *GetEmptySpeakerStats() const override {
    return new SpeakerStatsItf();
  }

  void TestingAccumulate(
      const MatrixBase<BaseFloat> &input,
      const SubPosterior &posteriors,
      SpeakerStatsItf *speaker_stats) const override { }

  void TestingForward(
      const MatrixBase<BaseFloat> &input,
      const SpeakerStatsItf &speaker_stats,
      MatrixBase<BaseFloat> *output) const override {
    output->CopyFromMat(input);
  }

  void Estimate(int32 final_iter) override { }


  NoOpTransform(): dim_(-1) { }

  NoOpTransform(const NoOpTransform &other):
      DifferentiableTransform(other),
      dim_(other.dim_) { }

  DifferentiableTransform* Copy() const override {
    return new NoOpTransform(*this);
  }

  std::string Type() const override { return "NoOpTransform"; }

  void Write(std::ostream &os, bool binary) const override;

  void Read(std::istream &is, bool binary) override;

 private:
  int32 dim_;
};


/**
   This is a version of the transform class that does a sequence of other
   transforms, specified by other instances of the DifferentiableTransform
   interface.  For instance: fMLLR followed by another fMLLR, or mean normalization
   followed by fMLLR.  The reason this might make sense is that you'd get a better
   estimate of the speaker-adapted class means if you do some kind of speaker
   normalization before estimating those class means.

   Caution: the framework currently implicitly assumes that the
   final one of the supplied transforms subsumes the previous ones
   (as in fMLLR subsumes mean subtraction, or fMLLR subsumes a previous
   fMLLR of the same dimension).  This means that in test time the
   first of the two transforms may be ignored and only the second one
   performed.  This is in order to keep a single-pass adaptation framework
   in test time.  The sequence of transforms still makes a difference
   because it affects how we compute the adaptation model (i.e., it's
   more like a speaker-adapted model than a speaker independent model,
   to use traditional ASR terminology).
 */
class SequenceTransform: public DifferentiableTransform {
 public:
  int32 InitFromConfig(int32 cur_pos,
                       std::vector<ConfigLine> *config_lines) override;

  int32 Dim() const override;
  void SetNumClasses(int32 num_classes) override;
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

  int32 NumFinalIterations() override;

  void Accumulate(
      int32 final_iter,
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors) override;

  void Estimate(int32 final_iter) override;

  SpeakerStatsItf *GetEmptySpeakerStats() const override {
    // See comment at the top of this class for an explanation.
    return transforms_.back()->GetEmptySpeakerStats();
  }

  void TestingAccumulate(
      const MatrixBase<BaseFloat> &input,
      const SubPosterior &posteriors,
      SpeakerStatsItf *speaker_stats) const override;

  void TestingForward(
      const MatrixBase<BaseFloat> &input,
      const SpeakerStatsItf &speaker_stats,
      MatrixBase<BaseFloat> *output) const override;

  SequenceTransform(const SequenceTransform &other);

  SequenceTransform() { }

  DifferentiableTransform* Copy() const override {
    return new SequenceTransform(*this);
  }

  std::string Type() const override { return "SequenceTransform"; }

  void Write(std::ostream &os, bool binary) const override;

  void Read(std::istream &is, bool binary) override;

  ~SequenceTransform() override;
 private:
  std::vector<DifferentiableTransform*> transforms_;
};

// This is the type actually returned by TrainingForward() for SequenceTransform.
// It contains a list of other MinibatchInfo, together with the outputs for all
// but the last call.
class SequenceMinibatchInfo: public MinibatchInfoItf {
 public:
  std::vector<MinibatchInfoItf*> info_vec;
  // outputs.size() will be info.size() - 1.
  std::vector<CuMatrix<BaseFloat> > outputs;

  ~SequenceMinibatchInfo() override;
};


class AppendSpeakerStats: public SpeakerStatsItf {
 public:
  AppendSpeakerStats() { }

  std::vector<SpeakerStatsItf*> stats;

  void Estimate() override;

  ~AppendSpeakerStats();
};

/**
   This is a version of the transform class that consists of a number of other
   transforms, appended dimension-wise, so its feature dimension is the sum of
   the dimensions of the constituent transforms-- e.g. this could be used to
   implement block-diagonal fMLLR, or a structure where some dimensions are
   adapted and some are not.
 */
class AppendTransform: public DifferentiableTransform {
 public:
  int32 InitFromConfig(int32 cur_pos,
                       std::vector<ConfigLine> *config_lines) override;

  int32 Dim() const override;
  void SetNumClasses(int32 num_classes) override;
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

  int32 NumFinalIterations() override;

  void Accumulate(
      int32 final_iter,
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors) override;

  SpeakerStatsItf *GetEmptySpeakerStats() const override;

  void TestingAccumulate(
      const MatrixBase<BaseFloat> &input,
      const SubPosterior &posteriors,
      SpeakerStatsItf *speaker_stats) const override;

  virtual void TestingForward(
      const MatrixBase<BaseFloat> &input,
      const SpeakerStatsItf &speaker_stats,
      MatrixBase<BaseFloat> *output) const override;

  void Estimate(int32 final_iter) override;

  AppendTransform(const AppendTransform &other);

  AppendTransform() { }

  DifferentiableTransform* Copy() const override {
    return new AppendTransform(*this);
  }

  std::string Type() const override { return "AppendTransform"; }

  void Write(std::ostream &os, bool binary) const override;

  void Read(std::istream &is, bool binary) override;

  ~AppendTransform();
 private:
  std::vector<DifferentiableTransform*> transforms_;
};


// This is the type created by TrainingForward() for AppendTransform.
// It just contains a list of other MinibatchInfo.
class AppendMinibatchInfo: public MinibatchInfoItf {
 public:
  std::vector<MinibatchInfoItf*> info_vec;

  ~AppendMinibatchInfo() override;
};


} // namespace differentiable_transform
} // namespace kaldi

#endif  // KALDI_TRANSFORM_GENERIC_TRANSFORM_H_
