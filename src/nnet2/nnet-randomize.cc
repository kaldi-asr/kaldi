// nnet2/nnet-randomize.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/nnet-randomize.h"

namespace kaldi {
namespace nnet2 {

void NnetDataRandomizer::RandomizeSamples() {
  KALDI_ASSERT(samples_.empty());
  int32 spk_data_size;
  
  for (size_t i = 0; i < data_.size(); i++) { // For each training file
    const TrainingFile &tfile = *(data_[i]);
    if (i == 0) spk_data_size = tfile.spk_info.Dim();
    else KALDI_ASSERT(tfile.spk_info.Dim() == spk_data_size);
    KALDI_ASSERT(tfile.feats.NumRows() ==
                 static_cast<int32>(tfile.pdf_post.size()));
    for (size_t j = 0; j < tfile.pdf_post.size(); j++)
      samples_.push_back(std::make_pair(i, j));
  }
  std::random_shuffle(samples_.begin(), samples_.end());

  if (num_samples_tgt_ == -1) { // set this variable.
    if (config_.num_samples > 0 && config_.num_epochs > 0) 
      KALDI_ERR << "You cannot set both of the --num-samples and --num-epochs "
                << "options to greater than zero.";
    if (config_.num_samples > 0)
      num_samples_tgt_ = config_.num_samples;
    else if (config_.num_epochs > 0)
      num_samples_tgt_ = static_cast<int>(config_.num_epochs *
                                          samples_.size());
    else 
      KALDI_ERR << "At least one of --num-samples and --num-epochs must be "
                << "greater than zero.";
    KALDI_ASSERT(num_samples_tgt_ > 0);
  }
}

NnetDataRandomizer::NnetDataRandomizer(int32 left_context_,
                                       int32 right_context_,
                                       const NnetDataRandomizerConfig &config):
    left_context_(left_context_), right_context_(right_context_), config_(config) {
  num_samples_returned_ = 0;
  num_samples_tgt_ = -1; // We'll set this the first time we call Done() or Value(),
  // inside RandomizeSamples().
}

void NnetDataRandomizer::AddTrainingFile(const Matrix<BaseFloat> &feats,
                                         const Vector<BaseFloat> &spk_info,
                                         const Posterior &pdf_post) {
  TrainingFile *tf = new TrainingFile(feats, spk_info, pdf_post);
  data_.push_back(tf);
}

NnetDataRandomizer::~NnetDataRandomizer() {
  for (size_t i = 0; i < data_.size(); i++)
    delete data_[i];
}

void NnetDataRandomizer::GetExample(const std::pair<int32, int32> &pair,
                                    NnetExample *example) const {
  int32 file_index = pair.first,
      frame_index = pair.second;
  KALDI_ASSERT(static_cast<size_t>(file_index) < data_.size());
  const TrainingFile &tf = *(data_[file_index]);
  KALDI_ASSERT(static_cast<size_t>(frame_index) < tf.pdf_post.size());
  example->labels = tf.pdf_post[frame_index];
  example->spk_info = tf.spk_info;
  Matrix<BaseFloat> input_frames(left_context_ + 1 + right_context_,
                                 tf.feats.NumCols());
  int32 start_frame = frame_index - left_context_,
      end_frame = frame_index + left_context_;
  for (int32 frame = start_frame; frame <= end_frame; frame++) {
    SubVector<BaseFloat> dest(input_frames, frame - start_frame);
    int32 frame_limited = frame; // we'll duplicate the start/end frame if we
    // cross the boundary of the utterance.
    if (frame_limited < 0)
      frame_limited = 0;
    if (frame_limited >= tf.feats.NumRows())
      frame_limited = tf.feats.NumRows() - 1;
    tf.feats.CopyRowToVec(frame_limited, &dest);
  }
  example->input_frames.CopyFromMat(input_frames); // this call resizes. 
}

bool NnetDataRandomizer::Done() {
  if (data_.empty()) return true;  // no data, so must be done.
  if (num_samples_tgt_ == -1) RandomizeSamples();  // first time called.
  if (num_samples_returned_ >= num_samples_tgt_) return true;
  if (samples_.empty()) RandomizeSamples();
  KALDI_ASSERT(!samples_.empty());
  return false;
}

const NnetExample &NnetDataRandomizer::Value() {
  KALDI_ASSERT(!Done());  // implies !samples_.empty().
  GetExample(samples_.back(), &cur_example_);
  return cur_example_;
}

void NnetDataRandomizer::Next() {
  KALDI_ASSERT(!Done());  // implies !samples_.empty().
  samples_.pop_back();
  num_samples_returned_++;
}

} // namespace nnet2
} // namespace kaldi
