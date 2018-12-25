// adapt/differentiable-transform.cc

// Copyright     2018  Johns Hopkins University (author: Daniel Povey)

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

#include "adapt/differentiable-transform.h"


// This header contains the 'base-cases' of DifferentiableTransform: namely,
// FmllrTransform and MeanOnlyTransform.  See also generic-transform.h where
// sequence, append and no-op types are defined.
namespace kaldi {
namespace differentiable_transform {

FmllrMinibatchInfo::FmllrMinibatchInfo(
    int32 num_classes, int32 dim, int32 num_speakers):
    target_model(num_classes, dim),
    estimators(num_speakers, NULL) { }

FmllrMinibatchInfo::~FmllrMinibatchInfo() {
  for (size_t i = 0; i < estimators.size(); i++)
    delete estimators[i];
}


int32 FmllrTransform::InitFromConfig(
    int32 cur_pos,
    std::vector<ConfigLine> *config_lines) {
  KALDI_ASSERT(cur_pos < int32(config_lines->size()));
  ConfigLine *line = &((*config_lines)[cur_pos]);
  KALDI_ASSERT(line->FirstToken() == Type());

  if (!line->GetValue("dim", &dim_) || dim_ <= 0)
    KALDI_ERR << "Dimension 'dim' must be specified for FmllrTransform, config "
        "line is: " << line->WholeLine();
  fmllr_opts_.ReadFromConfig(line);
  if (line->HasUnusedValues())
    KALDI_ERR << "Some configuration values were not used: '"
              << line->UnusedValues() << "', in line: "
              << line->WholeLine();
  return cur_pos + 1;
}


FmllrTransform::FmllrTransform(const FmllrTransform &other):
    DifferentiableTransform(other),
    dim_(other.dim_), fmllr_opts_(other.fmllr_opts_),
    target_model_(other.target_model_ == NULL ? NULL :
                  new GaussianEstimator(*other.target_model_)) { }

DifferentiableTransform *FmllrTransform::Copy() const {
  return new FmllrTransform(*this);
}

void FmllrTransform::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<FmllrTransform>");
  WriteToken(os, binary, "<NumClasses>");
  WriteBasicType(os, binary, num_classes_);
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  fmllr_opts_.Write(os, binary);
  if (target_model_ != NULL) {
    WriteToken(os, binary, "<TargetModel>");
    target_model_->Write(os, binary);
  } else {
    WriteToken(os, binary, "<NoTargetModel>");
  }
  WriteToken(os, binary, "</FmllrTransform>");
}

void FmllrTransform::Read(std::istream &is, bool binary) {
  delete target_model_;
  target_model_ = NULL;
  ExpectOneOrTwoTokens(is, binary, "<FmllrTransform>", "<NumClasses>");
  ReadBasicType(is, binary, &num_classes_);
  ExpectToken(is, binary, "<Dim>");
  ReadBasicType(is, binary, &dim_);
  fmllr_opts_.Read(is, binary);
  std::string tok;
  ReadToken(is, binary, &tok);
  if (tok == "<TargetModel>") {
    target_model_ = new GaussianEstimator(num_classes_, dim_);
  } // else "<NoTargetModel>".
  ExpectToken(is, binary, "</FmllrTransform>");
}


MinibatchInfoItf* FmllrTransform::TrainingForward(
    const CuMatrixBase<BaseFloat> &input,
    int32 num_chunks,
    int32 num_spk,
    const Posterior &posteriors,
    CuMatrixBase<BaseFloat> *output) const  {
  int32 num_classes = num_classes_,
      dim = dim_, num_frames = input.NumRows();
  KALDI_ASSERT(SameDim(input, *output) && input.NumCols() == dim &&
               int32(posteriors.size()) == input.NumRows());
  KALDI_ASSERT(num_chunks % num_spk == 0 && num_spk > 1 &&
              num_frames % num_chunks == 0);
  int32 chunks_per_spk = num_chunks / num_spk,
      frames_per_chunk = num_frames / num_chunks;

  FmllrMinibatchInfo *ans = new FmllrMinibatchInfo(num_classes,
                                                   dim, num_spk);

  // The input is in CuMatrix, i.e. it's on the GPU if we're using a GPU.  For
  // now we just transfer everything to CPU, which of course is not optimal; we
  // may later implement some of the deeper parts of this on GPU if the methods
  // turn out to be effective.
  Matrix<BaseFloat> input_cpu(input),
      output_cpu(num_frames, dim, kUndefined);

  // First estimate the target model (Gaussian means and spherical variances).
  ans->target_model.AccStats(input_cpu, posteriors);
  ans->target_model.Estimate(fmllr_opts_);

  for (int32 s = 0; s < num_spk; s++)
    ans->estimators[s] = new FmllrEstimator(fmllr_opts_,
                                            ans->target_model.GetMeans(),
                                            ans->target_model.GetVars());


  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    int32 speaker = chunk / chunks_per_spk;
    SubMatrix<BaseFloat> this_input(input_cpu.RowData(chunk),
                                    frames_per_chunk,  // num-rows
                                    dim,  // num-cols
                                    input_cpu.Stride() * num_chunks); // stride
    SubPosterior this_posteriors(posteriors,
                                 chunk, // offset
                                 frames_per_chunk, // num_frames
                                 num_chunks);  // stride
    ans->estimators[speaker]->AccStats(this_input, this_posteriors);
  }
  BaseFloat objf_impr = 0.0;
  for (int32 s = 0; s < num_spk; s++) {
    BaseFloat this_impr = ans->estimators[s]->Estimate();
    objf_impr += this_impr / num_spk;
  }
  // objf_impr is now the average objective-function improvement per frame.
  // We will later find a better way to display this.
  KALDI_LOG << "Objective function improvement per frame is "
            << objf_impr;

  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    int32 speaker = chunk / chunks_per_spk;
    SubMatrix<BaseFloat>
        this_input(input_cpu.RowData(chunk), frames_per_chunk, dim,
                   input_cpu.Stride() * num_chunks),
        this_output(output_cpu.RowData(chunk),
                    frames_per_chunk, dim, output_cpu.Stride() * num_chunks);
    ans->estimators[speaker]->AdaptFeatures(this_input, &this_output);
  }
  output->CopyFromMat(output_cpu);
  return ans;
}

void FmllrTransform::TrainingBackward(
    const CuMatrixBase<BaseFloat> &input,
    const CuMatrixBase<BaseFloat> &output_deriv,
    int32 num_chunks,
    int32 num_spk,
    const Posterior &posteriors,
    MinibatchInfoItf *minibatch_info,
    CuMatrixBase<BaseFloat> *input_deriv) const {
  FmllrMinibatchInfo *info = dynamic_cast<FmllrMinibatchInfo*>(minibatch_info);
  KALDI_ASSERT(info != NULL && "Wrong type of minibatch info supplied.");

  int32 dim = dim_, num_frames = input.NumRows();
  KALDI_ASSERT(SameDim(input, output_deriv) && input.NumCols() == dim &&
               SameDim(input, *input_deriv) &&
               int32(posteriors.size()) == input.NumRows());
  KALDI_ASSERT(num_chunks % num_spk == 0 && num_spk > 1 &&
              num_frames % num_chunks == 0);
  int32 chunks_per_spk = num_chunks / num_spk,
      frames_per_chunk = num_frames / num_chunks;

  // For now we just transfer everything to the CPU.
  Matrix<BaseFloat> input_cpu(input),
      output_deriv_cpu(output_deriv),
      input_deriv_cpu(num_frames, dim);

  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    int32 speaker = chunk / chunks_per_spk;
    SubMatrix<BaseFloat> this_input(
        input_cpu.RowData(chunk), frames_per_chunk,
        dim, input_cpu.Stride() * num_chunks),
        this_output_deriv(output_deriv_cpu.RowData(chunk),
                          frames_per_chunk, dim,
                          output_deriv_cpu.Stride() * num_chunks),
        this_input_deriv(input_deriv_cpu.RowData(chunk),
                         frames_per_chunk, dim,
                         input_deriv_cpu.Stride() * num_chunks);
    info->estimators[speaker]->AdaptFeaturesBackward(
        this_input, this_output_deriv, &this_input_deriv);
  }

  for (int32 s = 0; s < num_spk; s++)
    info->estimators[s]->EstimateBackward();

  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    int32 speaker = chunk / chunks_per_spk;
    SubMatrix<BaseFloat> this_input(
        input_cpu.RowData(chunk), frames_per_chunk,
        dim, input_cpu.Stride() * num_chunks),
        this_output_deriv(output_deriv_cpu.RowData(chunk),
                          frames_per_chunk, dim,
                          output_deriv_cpu.Stride() * num_chunks),
        this_input_deriv(input_deriv_cpu.RowData(chunk),
                         frames_per_chunk, dim,
                         input_deriv_cpu.Stride() * num_chunks);
    SubPosterior this_posteriors(posteriors, chunk,
                                 frames_per_chunk, num_chunks);
    info->estimators[speaker]->AccStatsBackward(
        this_input, this_posteriors, &this_input_deriv);
  }

  for (int32 s = 0; s < num_spk; s++)
    info->target_model.AddToOutputDerivs(info->estimators[s]->GetMeanDeriv(),
                                        info->estimators[s]->GetVarDeriv());

  info->target_model.AccStatsBackward(input_cpu, posteriors, &input_deriv_cpu);
  // These TrainingBackward() functions are all supposed to add to the
  // 'input_deriv'.
  CuMatrix<BaseFloat> input_deriv_temp(input_deriv->NumRows(),
                                       input_deriv->NumCols(),
                                       kUndefined);
  input_deriv_temp.CopyFromMat(input_deriv_cpu);
  input_deriv->AddMat(1.0, input_deriv_temp);

  delete info;
}


void FmllrTransform::Accumulate(
    int32 final_iter,
    const CuMatrixBase<BaseFloat> &input,
    int32 num_chunks,
    int32 num_spk,
    const Posterior &posteriors) {
  KALDI_ASSERT(final_iter == 0);
  if (target_model_ == NULL)
    target_model_ = new GaussianEstimator(num_classes_, dim_);
  Matrix<BaseFloat> input_cpu(input);
  target_model_->AccStats(input_cpu, posteriors);
}


void FmllrTransform::Estimate(int32 final_iter) {
  KALDI_ASSERT(final_iter == 0 && target_model_ != NULL);
  target_model_->Estimate(fmllr_opts_);
}


SpeakerStatsItf *FmllrTransform::GetEmptySpeakerStats() const {
  KALDI_ASSERT(target_model_ != NULL &&
               target_model_->GetMeans().NumRows() != 0 &&
               "You're trying to do adaptation with speaker transforms on "
               "which you haven't done the final phase of training.");
  return new FmllrSpeakerStats(fmllr_opts_, target_model_->GetMeans(),
                               target_model_->GetVars());
}

void FmllrTransform::TestingAccumulate(
    const MatrixBase<BaseFloat> &input,
    const SubPosterior &posteriors,
    SpeakerStatsItf *speaker_stats) const {
  FmllrSpeakerStats *stats = dynamic_cast<FmllrSpeakerStats*>(
      speaker_stats);
  KALDI_ASSERT(stats != NULL && "Wrong type of speaker stats supplied.");
  stats->estimator.AccStats(input, posteriors);
}

void FmllrTransform::TestingForward(
      const MatrixBase<BaseFloat> &input,
      const SpeakerStatsItf &speaker_stats,
      MatrixBase<BaseFloat> *output) const {
  const FmllrSpeakerStats *stats = dynamic_cast<const FmllrSpeakerStats*>(
      &speaker_stats);
  KALDI_ASSERT(stats != NULL && "Wrong type of speaker stats supplied.");
  KALDI_ASSERT(stats->estimator.IsEstimated() &&
               "You can't call TestingForward() without calling Estimate() on "
               "the speaker stats.");
  stats->estimator.AdaptFeatures(input, output);
}

FmllrTransform::~FmllrTransform() {
  delete target_model_;
}


MeanOnlyTransformMinibatchInfo::MeanOnlyTransformMinibatchInfo(
    int32 num_classes, int32 dim, int32 num_speakers):
    target_model(num_classes, dim),
    estimators(num_speakers, NULL) { }

MeanOnlyTransformMinibatchInfo::~MeanOnlyTransformMinibatchInfo() {
  for (size_t i = 0; i < estimators.size(); i++)
    delete estimators[i];
}


int32 MeanOnlyTransform::InitFromConfig(
    int32 cur_pos,
    std::vector<ConfigLine> *config_lines) {
  KALDI_ASSERT(cur_pos < int32(config_lines->size()));
  ConfigLine *line = &((*config_lines)[cur_pos]);
  KALDI_ASSERT(line->FirstToken() == Type());

  if (!line->GetValue("dim", &dim_) || dim_ <= 0)
    KALDI_ERR << "Dimension 'dim' must be specified for MeanOnlyTransform, config "
        "line is: " << line->WholeLine();
  if (line->HasUnusedValues())
    KALDI_ERR << "Some configuration values were not used: '"
              << line->UnusedValues() << "', in line: "
              << line->WholeLine();
  return cur_pos + 1;
}

MeanOnlyTransform::MeanOnlyTransform(const MeanOnlyTransform &other):
    DifferentiableTransform(other),
    dim_(other.dim_), target_model_(other.target_model_ == NULL ? NULL :
                                    new GaussianEstimator(*other.target_model_)) { }

void MeanOnlyTransform::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<MeanOnlyTransform>");
  WriteToken(os, binary, "<NumClasses>");
  WriteBasicType(os, binary, num_classes_);
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  if (target_model_ != NULL) {
    WriteToken(os, binary, "<TargetModel>");
    target_model_->Write(os, binary);
  } else {
    WriteToken(os, binary, "<NoTargetModel>");
  }
  WriteToken(os, binary, "</MeanOnlyTransform>");
}

void MeanOnlyTransform::Read(std::istream &is, bool binary) {
  delete target_model_;
  target_model_ = NULL;
  ExpectOneOrTwoTokens(is, binary, "<MeanOnlyTransform>", "<NumClasses>");
  ReadBasicType(is, binary, &num_classes_);
  ExpectToken(is, binary, "<Dim>");
  ReadBasicType(is, binary, &dim_);
  std::string tok;
  ReadToken(is, binary, &tok);
  if (tok == "<TargetModel>") {
    target_model_ = new GaussianEstimator(num_classes_, dim_);
  } // else "<NoTargetModel>".
  ExpectToken(is, binary, "</MeanOnlyTransform>");
}


MinibatchInfoItf* MeanOnlyTransform::TrainingForward(
    const CuMatrixBase<BaseFloat> &input,
    int32 num_chunks,
    int32 num_spk,
    const Posterior &posteriors,
    CuMatrixBase<BaseFloat> *output) const  {
  int32 num_classes = num_classes_,
      dim = dim_, num_frames = input.NumRows();
  KALDI_ASSERT(SameDim(input, *output) && input.NumCols() == dim &&
               int32(posteriors.size()) == input.NumRows());
  KALDI_ASSERT(num_chunks % num_spk == 0 && num_spk > 1 &&
              num_frames % num_chunks == 0);
  int32 chunks_per_spk = num_chunks / num_spk,
      frames_per_chunk = num_frames / num_chunks;

  MeanOnlyTransformMinibatchInfo *ans = new MeanOnlyTransformMinibatchInfo(num_classes,
                                                   dim, num_spk);

  // The input is in CuMatrix, i.e. it's on the GPU if we're using a GPU.  For
  // now we just transfer everything to CPU, which of course is not optimal; we
  // may later implement some of the deeper parts of this on GPU if the methods
  // turn out to be effective.
  Matrix<BaseFloat> input_cpu(input),
      output_cpu(num_frames, dim, kUndefined);

  // First estimate the target model (Gaussian means and spherical variances).
  // We use the default options: they only affect the variances, which we won't
  // be using.
  ans->target_model.AccStats(input_cpu, posteriors);
  FmllrEstimatorOptions default_opts;
  ans->target_model.Estimate(default_opts);

  for (int32 s = 0; s < num_spk; s++)
    ans->estimators[s] = new MeanOnlyTransformEstimator(
        ans->target_model.GetMeans());


  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    int32 speaker = chunk / chunks_per_spk;
    SubMatrix<BaseFloat> this_input(input_cpu.RowData(chunk),
                                    frames_per_chunk,  // num-rows
                                    dim,  // num-cols
                                    input_cpu.Stride() * num_chunks); // stride
    SubPosterior this_posteriors(posteriors,
                                 chunk, // offset
                                 frames_per_chunk, // num_frames
                                 num_chunks);  // stride
    ans->estimators[speaker]->AccStats(this_input, this_posteriors);
  }
  for (int32 s = 0; s < num_spk; s++)
    ans->estimators[s]->Estimate();

  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    int32 speaker = chunk / chunks_per_spk;
    SubMatrix<BaseFloat>
        this_input(input_cpu.RowData(chunk), frames_per_chunk, dim,
                   input_cpu.Stride() * num_chunks),
        this_output(output_cpu.RowData(chunk),
                    frames_per_chunk, dim, output_cpu.Stride() * num_chunks);
    ans->estimators[speaker]->AdaptFeatures(this_input, &this_output);
  }
  output->CopyFromMat(output_cpu);
  return ans;
}


DifferentiableTransform *MeanOnlyTransform::Copy() const {
  return new MeanOnlyTransform(*this);
}

void MeanOnlyTransform::TrainingBackward(
    const CuMatrixBase<BaseFloat> &input,
    const CuMatrixBase<BaseFloat> &output_deriv,
    int32 num_chunks,
    int32 num_spk,
    const Posterior &posteriors,
    MinibatchInfoItf *minibatch_info,
    CuMatrixBase<BaseFloat> *input_deriv) const {
  MeanOnlyTransformMinibatchInfo *info =
      dynamic_cast<MeanOnlyTransformMinibatchInfo*>(minibatch_info);
  KALDI_ASSERT(info != NULL && "Wrong type of minibatch info supplied.");

  int32 dim = dim_, num_frames = input.NumRows();
  KALDI_ASSERT(SameDim(input, output_deriv) && input.NumCols() == dim &&
               SameDim(input, *input_deriv) &&
               int32(posteriors.size()) == input.NumRows());
  KALDI_ASSERT(num_chunks % num_spk == 0 && num_spk > 1 &&
              num_frames % num_chunks == 0);
  int32 chunks_per_spk = num_chunks / num_spk,
      frames_per_chunk = num_frames / num_chunks;

  // For now we just transfer everything to the CPU.
  Matrix<BaseFloat> input_cpu(input),
      output_deriv_cpu(output_deriv),
      input_deriv_cpu(num_frames, dim);

  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    int32 speaker = chunk / chunks_per_spk;
    SubMatrix<BaseFloat> this_input(
        input_cpu.RowData(chunk), frames_per_chunk,
        dim, input_cpu.Stride() * num_chunks),
        this_output_deriv(output_deriv_cpu.RowData(chunk),
                          frames_per_chunk, dim,
                          output_deriv_cpu.Stride() * num_chunks),
        this_input_deriv(input_deriv_cpu.RowData(chunk),
                         frames_per_chunk, dim,
                         input_deriv_cpu.Stride() * num_chunks);
    info->estimators[speaker]->AdaptFeaturesBackward(
        this_input, this_output_deriv, &this_input_deriv);
  }

  for (int32 s = 0; s < num_spk; s++)
    info->estimators[s]->EstimateBackward();

  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    int32 speaker = chunk / chunks_per_spk;
    SubMatrix<BaseFloat> this_input(
        input_cpu.RowData(chunk), frames_per_chunk,
        dim, input_cpu.Stride() * num_chunks),
        this_output_deriv(output_deriv_cpu.RowData(chunk),
                          frames_per_chunk, dim,
                          output_deriv_cpu.Stride() * num_chunks),
        this_input_deriv(input_deriv_cpu.RowData(chunk),
                         frames_per_chunk, dim,
                         input_deriv_cpu.Stride() * num_chunks);
    SubPosterior this_posteriors(posteriors, chunk,
                                 frames_per_chunk, num_chunks);
    info->estimators[speaker]->AccStatsBackward(
        this_input, this_posteriors, &this_input_deriv);
  }

  for (int32 s = 0; s < num_spk; s++) {
    Vector<BaseFloat> var_derivs(num_classes_);  // zero.
    info->target_model.AddToOutputDerivs(info->estimators[s]->GetMeanDeriv(),
                                         var_derivs);
  }

  info->target_model.AccStatsBackward(input_cpu, posteriors, &input_deriv_cpu);
  // These TrainingBackward() functions are all supposed to add to the
  // 'input_deriv'.
  CuMatrix<BaseFloat> input_deriv_temp(input_deriv->NumRows(),
                                       input_deriv->NumCols(),
                                       kUndefined);
  input_deriv_temp.CopyFromMat(input_deriv_cpu);
  input_deriv->AddMat(1.0, input_deriv_temp);
  delete info;
}


void MeanOnlyTransform::Accumulate(
    int32 final_iter,
    const CuMatrixBase<BaseFloat> &input,
    int32 num_chunks,
    int32 num_spk,
    const Posterior &posteriors) {
  KALDI_ASSERT(final_iter == 0);
  if (target_model_ == NULL)
    target_model_ = new GaussianEstimator(num_classes_, dim_);
  Matrix<BaseFloat> input_cpu(input);
  target_model_->AccStats(input_cpu, posteriors);
}

void MeanOnlyTransform::Estimate(int32 final_iter) {
  KALDI_ASSERT(final_iter == 0 && target_model_ != NULL);
  // The options only affect the estimates of the variance, which we don't use
  // here, so we use the default options.
  FmllrEstimatorOptions default_opts;
  target_model_->Estimate(default_opts);
}



SpeakerStatsItf *MeanOnlyTransform::GetEmptySpeakerStats() const {
  KALDI_ASSERT(target_model_ != NULL &&
               target_model_->GetMeans().NumRows() != 0 &&
               "You're trying to do adaptation with speaker transforms on "
               "which you haven't done the final phase of training.");
  return new MeanOnlyTransformSpeakerStats(target_model_->GetMeans());
}

void MeanOnlyTransform::TestingAccumulate(
    const MatrixBase<BaseFloat> &input,
    const SubPosterior &posteriors,
    SpeakerStatsItf *speaker_stats) const {
  MeanOnlyTransformSpeakerStats *stats = dynamic_cast<MeanOnlyTransformSpeakerStats*>(
      speaker_stats);
  KALDI_ASSERT(stats != NULL && "Wrong type of speaker stats supplied.");
  stats->estimator.AccStats(input, posteriors);
}

void MeanOnlyTransform::TestingForward(
      const MatrixBase<BaseFloat> &input,
      const SpeakerStatsItf &speaker_stats,
      MatrixBase<BaseFloat> *output) const {
  const MeanOnlyTransformSpeakerStats *stats = dynamic_cast<const MeanOnlyTransformSpeakerStats*>(
      &speaker_stats);
  KALDI_ASSERT(stats != NULL && "Wrong type of speaker stats supplied.");
  KALDI_ASSERT(stats->estimator.IsEstimated() &&
               "You can't call TestingForward() without calling Estimate() on "
               "the speaker stats.");
  stats->estimator.AdaptFeatures(input, output);
}

MeanOnlyTransform::~MeanOnlyTransform() {
  delete target_model_;
}



}  // namespace differentiable_transform
}  // namespace kaldi
