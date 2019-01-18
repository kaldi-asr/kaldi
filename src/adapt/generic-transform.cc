// adapt/generic-transform.cc

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

#include "adapt/differentiable-transform-itf.h"
#include "adapt/generic-transform.h"

namespace kaldi {
namespace differentiable_transform {


int32 NoOpTransform::InitFromConfig(
    int32 cur_pos,
    std::vector<ConfigLine> *config_lines) {
  KALDI_ASSERT(cur_pos < int32(config_lines->size()));
  ConfigLine *line = &((*config_lines)[cur_pos]);
  KALDI_ASSERT(line->FirstToken() == Type());
  if (!line->GetValue("dim", &dim_) || dim_ <= 0)
    KALDI_ERR << "Dimension 'dim' must be specified for NoOpTransform, config "
        "line is: " << line->WholeLine();
  if (line->HasUnusedValues())
    KALDI_ERR << "Some configuration values were not used: '"
              << line->UnusedValues() << "', in line: "
              << line->WholeLine();
  return cur_pos + 1;
}


void NoOpTransform::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NoOpTransform>");
  WriteToken(os, binary, "<NumClasses>");
  WriteBasicType(os, binary, num_classes_);
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "</NoOpTransform>");
}

void NoOpTransform::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<NoOpTransform>", "<NumClasses>");
  ReadBasicType(is, binary, &num_classes_);
  ExpectToken(is, binary, "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "</NoOpTransform>");
}


int32 SequenceTransform::InitFromConfig(
    int32 cur_pos,
    std::vector<ConfigLine> *config_lines) {
  KALDI_ASSERT(cur_pos < int32(config_lines->size()) &&
               transforms_.empty());
  ConfigLine *line = &((*config_lines)[cur_pos]);
  KALDI_ASSERT(line->FirstToken() == Type());
  int32 num_transforms = -1;
  if (!line->GetValue("num-transforms", &num_transforms) ||
      num_transforms <= 0)
    KALDI_ERR << "Config value num-transforms must be specified for "
        "SequenceTransform, line is: " << line->WholeLine();
  if (line->HasUnusedValues())
    KALDI_ERR << "Some configuration values were not used: '"
              << line->UnusedValues() << "', in line: "
              << line->WholeLine();
  cur_pos++;

  int32 dim = 0;
  for (int32 i = 0; i < num_transforms; i++) {
    if (cur_pos >= int32(config_lines->size()))
      KALDI_ERR << "Config file lacks enough lines for SequenceTransform.";
    ConfigLine *other_line = &((*config_lines)[cur_pos]);
    std::string transform_type = other_line->FirstToken();
    DifferentiableTransform *transform = NewTransformOfType(transform_type);
    if (transform == NULL)
      KALDI_ERR << "Could not find transform of type " << transform_type;
    cur_pos = transform->InitFromConfig(cur_pos, config_lines);
    if (i == 0) {
      dim = transform->Dim();
    } else if (dim != transform->Dim()) {
      KALDI_ERR << "Transforms used in SequenceTransform have inconsistent dim: "
                << dim << " vs " << transform->Dim();
    }
    transforms_.push_back(transform);
  }
  return cur_pos;
}


SequenceTransform::SequenceTransform(const SequenceTransform &other):
    DifferentiableTransform(other),
    transforms_(other.transforms_.size(), NULL) {
  for (size_t i = 0; i < other.transforms_.size(); i++)
    transforms_[i] = other.transforms_[i]->Copy();
}


void SequenceTransform::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SequenceTransform>");
  WriteToken(os, binary, "<NumClasses>");
  WriteBasicType(os, binary, num_classes_);
  WriteToken(os, binary, "<NumTransforms>");
  int32 num_transforms = transforms_.size();
  WriteBasicType(os, binary, num_transforms);
  for (int32 i = 0; i < num_transforms; i++)
    transforms_[i]->Write(os, binary);
  WriteToken(os, binary, "</SequenceTransform>");
}

void SequenceTransform::Read(std::istream &is, bool binary) {
  while (!transforms_.empty()) {
    delete transforms_.back();
    transforms_.pop_back();
  }
  ExpectOneOrTwoTokens(is, binary, "<SequenceTransform>", "<NumClasses>");
  ReadBasicType(is, binary, &num_classes_);
  ExpectToken(is, binary, "<NumTransforms>");
  int32 num_transforms;
  ReadBasicType(is, binary, &num_transforms);
  for (int32 i = 0; i < num_transforms; i++) {
    std::string tok;
    ReadToken(is, binary, &tok);
    DifferentiableTransform *transform;
    if (!(transform = NewTransformOfType(tok)))
      KALDI_ERR << "Expected the name of a transform, got "
                << tok << " (maybe you should recompile?)";
    transform->Read(is, binary);
    transforms_.push_back(transform);
  }
  ExpectToken(is, binary, "</SequenceTransform>");
}

void SequenceTransform::Add(const DifferentiableTransform &other_in) {
  const SequenceTransform *other = dynamic_cast<const SequenceTransform*>(
      &other_in);
  KALDI_ASSERT(transforms_.size() == other->transforms_.size());
  for (size_t i = 0; i < transforms_.size(); i++)
    transforms_[i]->Add(*(other->transforms_[i]));
}

int32 SequenceTransform::Dim() const {
  size_t num_transforms = transforms_.size();
  KALDI_ASSERT(num_transforms > 0);
  return transforms_[0]->Dim();
}

void SequenceTransform::SetNumClasses(int32 num_classes) {
  KALDI_ASSERT(num_classes > 0);
  num_classes_ = num_classes;
  for (size_t i = 0; i < transforms_.size(); i++) {
    transforms_[i]->SetNumClasses(num_classes);
  }
}

SequenceTransform::~SequenceTransform() {
  for (size_t i = 0; i < transforms_.size(); i++)
    delete transforms_[i];
}

MinibatchInfoItf* SequenceTransform::TrainingForward(
    const CuMatrixBase<BaseFloat> &input,
    int32 num_chunks,
    int32 num_spk,
    const Posterior &posteriors,
    CuMatrixBase<BaseFloat> *output) const {
  KALDI_ASSERT(SameDim(input, *output) &&
               !transforms_.empty());
  SequenceMinibatchInfo *ans = new SequenceMinibatchInfo();

  const CuMatrixBase<BaseFloat> *last_output = &input;
  CuMatrixBase<BaseFloat> *this_output;

  ans->outputs.resize(transforms_.size() - 1);

  for (size_t i = 0; i < transforms_.size(); i++) {
    if (i + 1 == transforms_.size()) {
      this_output = output;
    } else {
      // not the final transform.
      ans->outputs[i].Resize(output->NumRows(), output->NumCols(), kUndefined);
      this_output = &(ans->outputs[i]);
    }
    ans->info_vec.push_back(transforms_[i]->TrainingForward(
        *last_output, num_chunks, num_spk, posteriors, this_output));
    last_output = this_output;
  }
  return ans;
}

void SequenceTransform::TrainingBackward(
      const CuMatrixBase<BaseFloat> &input,
      const CuMatrixBase<BaseFloat> &output_deriv,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      MinibatchInfoItf *minibatch_info,
      CuMatrixBase<BaseFloat> *input_deriv) const {
  KALDI_ASSERT(SameDim(input, output_deriv) && SameDim(input, *input_deriv));

  SequenceMinibatchInfo *info = dynamic_cast<SequenceMinibatchInfo*>(minibatch_info);
  KALDI_ASSERT(info != NULL && "Mismatched MinibatchInfo type?");

  CuMatrix<BaseFloat> temp_deriv(input.NumRows(),
                                 input.NumCols());
  int32 num_transforms = transforms_.size();
  KALDI_ASSERT(num_transforms > 0);

  const CuMatrixBase<BaseFloat> *cur_output_deriv = &output_deriv;

  for (int32 i = num_transforms - 1; i >= 0; i--) {
    const CuMatrixBase<BaseFloat> *cur_input = (i == 0 ? &input :
                                                &(info->outputs[i-1]));
    CuMatrixBase<BaseFloat> *cur_input_deriv;
    if (i == 0) {
      cur_input_deriv = input_deriv;
    } else if (i == num_transforms - 1) {
      cur_input_deriv = &temp_deriv;
    } else {
      // this matrix is no longer needed, store the intermediate deriv here.
      cur_input_deriv = &(info->outputs[i]);
      cur_input_deriv->SetZero();
    }
    transforms_[i]->TrainingBackward(*cur_input, *cur_output_deriv,
                                     num_chunks, num_spk, posteriors,
                                     info->info_vec[i], cur_input_deriv);
    info->info_vec[i] = NULL;  // Prevent it from being deleted twice.
    cur_output_deriv = cur_input_deriv;
  }
  delete info;  // This function took ownership.
}

int32 SequenceTransform::NumFinalIterations() {
  int32 ans = 0;
  for (size_t i = 0; i < transforms_.size(); i++)
    ans += transforms_[i]->NumFinalIterations();
  return ans;
}

void SequenceTransform::Accumulate(
    int32 final_iter,
    const CuMatrixBase<BaseFloat> &input,
    int32 num_chunks,
    int32 num_spk,
    const Posterior &posteriors) {
  CuMatrix<BaseFloat> temp;
  const CuMatrixBase<BaseFloat> *cur_input = &input;

  int32 prev_final_iters = 0;
  for (size_t i = 0; i < transforms_.size(); i++) {
    int32 nf = transforms_[i]->NumFinalIterations();
    if (final_iter < prev_final_iters + nf) {
      transforms_[i]->Accumulate(final_iter - prev_final_iters,
                                 *cur_input, num_chunks, num_spk,
                                 posteriors);
      return;
    } else {
      KALDI_ASSERT(i + 1 < transforms_.size());
      // We have to propagate the features through this transform.
      CuMatrix<BaseFloat> this_output(input.NumRows(), input.NumCols(),
                                      kUndefined);
      transforms_[i]->TestingForwardBatch(*cur_input, num_chunks, num_spk,
                                          posteriors, &this_output);
      temp.Swap(&this_output);
      cur_input = &temp;
    }
    prev_final_iters += nf;
  }
  KALDI_ERR << "final_iter out of range.";
}

void SequenceTransform::Estimate(int32 final_iter) {
  CuMatrix<BaseFloat> temp;

  int32 prev_final_iters = 0;
  for (size_t i = 0; i < transforms_.size(); i++) {
    int32 nf = transforms_[i]->NumFinalIterations();
    if (final_iter < prev_final_iters + nf) {
      transforms_[i]->Estimate(final_iter - prev_final_iters);
      return;
    }
    prev_final_iters += nf;
  }
  KALDI_ERR << "final_iter out of range.";
}

void SequenceTransform::TestingAccumulate(
    const MatrixBase<BaseFloat> &input,
    const SubPosterior &posteriors,
    SpeakerStatsItf *speaker_stats) const {
  transforms_.back()->TestingAccumulate(input, posteriors,
                                        speaker_stats);
}

void SequenceTransform::TestingForward(
    const MatrixBase<BaseFloat> &input,
    const SpeakerStatsItf &speaker_stats,
    MatrixBase<BaseFloat> *output) const {
  transforms_.back()->TestingForward(input, speaker_stats, output);
}


SequenceMinibatchInfo::~SequenceMinibatchInfo() {
  for (size_t i = 0; i < info_vec.size(); i++)
    delete info_vec[i];
}



int32 AppendTransform::InitFromConfig(
    int32 cur_pos,
    std::vector<ConfigLine> *config_lines) {
  KALDI_ASSERT(cur_pos < int32(config_lines->size()) &&
               transforms_.empty());
  ConfigLine *line = &((*config_lines)[cur_pos]);
  KALDI_ASSERT(line->FirstToken() == Type());
  int32 num_transforms = -1;
  if (!line->GetValue("num-transforms", &num_transforms) ||
      num_transforms <= 0)
    KALDI_ERR << "Config value num-transforms must be specified for "
        "AppendTransform, line is: " << line->WholeLine();
  if (line->HasUnusedValues())
    KALDI_ERR << "Some configuration values were not used: '"
              << line->UnusedValues() << "', in line: "
              << line->WholeLine();
  cur_pos++;

  for (int32 i = 0; i < num_transforms; i++) {
    if (cur_pos >= int32(config_lines->size()))
      KALDI_ERR << "Config file lacks enough lines for AppendTransform.";
    ConfigLine *other_line = &((*config_lines)[cur_pos]);
    std::string transform_type = other_line->FirstToken();
    DifferentiableTransform *transform = NewTransformOfType(transform_type);
    if (transform == NULL)
      KALDI_ERR << "Could not find transform of type " << transform_type;
    cur_pos = transform->InitFromConfig(cur_pos, config_lines);
    transforms_.push_back(transform);
  }
  return cur_pos;
}



AppendTransform::AppendTransform(const AppendTransform &other):
    DifferentiableTransform(other),
    transforms_(other.transforms_.size(), NULL) {
  for (size_t i = 0; i < other.transforms_.size(); i++)
    transforms_[i] = other.transforms_[i]->Copy();
}



void AppendTransform::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<AppendTransform>");
  WriteToken(os, binary, "<NumClasses>");
  WriteBasicType(os, binary, num_classes_);
  WriteToken(os, binary, "<NumTransforms>");
  int32 num_transforms = transforms_.size();
  WriteBasicType(os, binary, num_transforms);
  for (int32 i = 0; i < num_transforms; i++)
    transforms_[i]->Write(os, binary);
  WriteToken(os, binary, "</AppendTransform>");
}

void AppendTransform::Read(std::istream &is, bool binary) {
  while (!transforms_.empty()) {
    delete transforms_.back();
    transforms_.pop_back();
  }
  ExpectOneOrTwoTokens(is, binary, "<AppendTransform>", "<NumClasses>");
  ReadBasicType(is, binary, &num_classes_);
  ExpectToken(is, binary, "<NumTransforms>");
  int32 num_transforms;
  ReadBasicType(is, binary, &num_transforms);
  for (int32 i = 0; i < num_transforms; i++) {
    std::string tok;
    ReadToken(is, binary, &tok);
    DifferentiableTransform *transform;
    if (!(transform = NewTransformOfType(tok)))
      KALDI_ERR << "Expected the name of a transform, got "
                << tok << " (maybe you should recompile?)";
    transform->Read(is, binary);
    transforms_.push_back(transform);
  }
  ExpectToken(is, binary, "</AppendTransform>");
}

void AppendTransform::Add(const DifferentiableTransform &other_in) {
  const AppendTransform *other = dynamic_cast<const AppendTransform*>(
      &other_in);
  KALDI_ASSERT(transforms_.size() == other->transforms_.size());
  for (size_t i = 0; i < transforms_.size(); i++)
    transforms_[i]->Add(*(other->transforms_[i]));
}

int32 AppendTransform::Dim() const {
  size_t num_transforms = transforms_.size();
  KALDI_ASSERT(num_transforms > 0);
  int32 ans = 0;
  for (size_t i = 0; i < num_transforms; i++)
    ans += transforms_[i]->Dim();
  return ans;
}

void AppendTransform::SetNumClasses(int32 num_classes) {
  num_classes_ = num_classes;
  for (size_t i = 0; i < transforms_.size(); i++) {
    transforms_[i]->SetNumClasses(num_classes);
  }
}

AppendTransform::~AppendTransform() {
  for (size_t i = 0; i < transforms_.size(); i++)
    delete transforms_[i];
}


MinibatchInfoItf* AppendTransform::TrainingForward(
    const CuMatrixBase<BaseFloat> &input,
    int32 num_chunks,
    int32 num_spk,
    const Posterior &posteriors,
    CuMatrixBase<BaseFloat> *output) const {
  KALDI_ASSERT(input.NumCols() == Dim() &&
               SameDim(input, *output));
  AppendMinibatchInfo *ans = new AppendMinibatchInfo();
  int32 dim_offset = 0;
  for (size_t i = 0; i < transforms_.size(); i++) {
    int32 this_dim = transforms_[i]->Dim();
    CuSubMatrix<BaseFloat> input_part = input.ColRange(dim_offset, this_dim),
        output_part = output->ColRange(dim_offset, this_dim);
    ans->info_vec.push_back(transforms_[i]->TrainingForward(
        input_part, num_chunks, num_spk, posteriors, &output_part));
    dim_offset += this_dim;
  }
  KALDI_ASSERT(dim_offset == input.NumCols());
  return ans;
}

void AppendTransform::TrainingBackward(
      const CuMatrixBase<BaseFloat> &input,
      const CuMatrixBase<BaseFloat> &output_deriv,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      MinibatchInfoItf *minibatch_info,
      CuMatrixBase<BaseFloat> *input_deriv) const {
  AppendMinibatchInfo *info = dynamic_cast<AppendMinibatchInfo*>(minibatch_info);
  KALDI_ASSERT(info != NULL && "Mismatched MinibatchInfo type?");

  int32 dim_offset = 0;
  for (size_t i = 0; i < transforms_.size(); i++) {
    int32 this_dim = transforms_[i]->Dim();
    CuSubMatrix<BaseFloat> input_part = input.ColRange(dim_offset, this_dim),
        output_deriv_part = output_deriv.ColRange(dim_offset, this_dim),
        input_deriv_part = input_deriv->ColRange(dim_offset, this_dim);
    transforms_[i]->TrainingBackward(
        input_part, output_deriv_part, num_chunks, num_spk,
        posteriors, info->info_vec[i], &input_deriv_part);
    info->info_vec[i] = NULL;  // Prevent it from being deleted twice.
    dim_offset += this_dim;
  }
  KALDI_ASSERT(dim_offset == input.NumCols());
  delete info;  // This function took ownership.
}

int32 AppendTransform::NumFinalIterations() {
  int32 ans = 0;
  for (size_t i = 0; i < transforms_.size(); i++)
    ans = std::max<int32>(ans, transforms_[i]->NumFinalIterations());
  return ans;
}


void AppendTransform::Accumulate(
    int32 final_iter,
    const CuMatrixBase<BaseFloat> &input,
    int32 num_chunks,
    int32 num_spk,
    const Posterior &posteriors) {
  int32 num_final_iters = 0,
      dim_offset = 0;
  for (size_t i = 0; i < transforms_.size(); i++) {
    int32 this_nf = transforms_[i]->NumFinalIterations(),
        this_dim = transforms_[i]->Dim();
    if (final_iter < this_nf)
      transforms_[i]->Accumulate(final_iter,
                                 input.ColRange(dim_offset, this_dim),
                                 num_chunks, num_spk, posteriors);
    if (this_nf > num_final_iters)
      num_final_iters = this_nf;
    dim_offset += this_dim;
  }
  KALDI_ASSERT(final_iter >= 0 && final_iter < num_final_iters);
}

void AppendTransform::Estimate(int32 final_iter) {
  for (size_t i = 0; i < transforms_.size(); i++) {
    int32 this_nf = transforms_[i]->NumFinalIterations();
    if (final_iter < this_nf) {
      transforms_[i]->Estimate(final_iter);
    }
  }
}

AppendMinibatchInfo::~AppendMinibatchInfo() {
  for (size_t i = 0; i < info_vec.size(); i++)
    delete info_vec[i];
}

SpeakerStatsItf* AppendTransform::GetEmptySpeakerStats() const {
  AppendSpeakerStats *ans = new AppendSpeakerStats();
  for (size_t i = 0; i < transforms_.size(); i++)
    ans->stats.push_back(transforms_[i]->GetEmptySpeakerStats());
  return ans;
}

void AppendTransform::TestingAccumulate(
    const MatrixBase<BaseFloat> &input,
    const SubPosterior &posteriors,
    SpeakerStatsItf *speaker_stats) const {
  AppendSpeakerStats *stats = dynamic_cast<AppendSpeakerStats*>(speaker_stats);
  KALDI_ASSERT(stats != NULL && stats->stats.size() == transforms_.size() &&
               "Wrong type of stats supplied to AppendTransform.");
  int32 dim_offset = 0;
  for (size_t i = 0; i < transforms_.size(); i++) {
    int32 this_dim = transforms_[i]->Dim();
    SubMatrix<BaseFloat> input_part = input.ColRange(dim_offset, this_dim);
    transforms_[i]->TestingAccumulate(input_part, posteriors,
                                      stats->stats[i]);
    dim_offset += this_dim;
  }
  KALDI_ASSERT(dim_offset == input.NumCols());
}


void AppendTransform::TestingForward(
    const MatrixBase<BaseFloat> &input,
    const SpeakerStatsItf &speaker_stats,
    MatrixBase<BaseFloat> *output) const {
  const AppendSpeakerStats *stats =
      dynamic_cast<const AppendSpeakerStats*>(&speaker_stats);
  KALDI_ASSERT(stats != NULL && stats->stats.size() == transforms_.size() &&
               "Wrong type of stats supplied to AppendTransform.");
  int32 dim_offset = 0;
  for (size_t i = 0; i < transforms_.size(); i++) {
    int32 this_dim = transforms_[i]->Dim();
    SubMatrix<BaseFloat> input_part = input.ColRange(dim_offset, this_dim),
        output_part = output->ColRange(dim_offset, this_dim);
    transforms_[i]->TestingForward(input_part, *(stats->stats[i]),
                                   &output_part);
    dim_offset += this_dim;
  }
  KALDI_ASSERT(dim_offset == input.NumCols());
}

void AppendSpeakerStats::Estimate() {
  for (size_t i = 0; i < stats.size(); i++)
    stats[i]->Estimate();
}

AppendSpeakerStats::~AppendSpeakerStats() {
  for (size_t i = 0; i < stats.size(); i++)
    delete stats[i];
}


}  // namespace differentiable_transform
}  // namespace kaldi
