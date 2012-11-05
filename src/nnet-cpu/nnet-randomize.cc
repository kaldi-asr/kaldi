// nnet/nnet-randomize.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet-cpu/nnet-randomize.h"

namespace kaldi {

void NnetDataRandomizer::RandomizeSamplesRecurse(
    std::vector<std::vector<std::pair<int32, int32> > > *samples_by_pdf,
    std::vector<std::pair<int32, int32> > *samples) {
  int32 tot_num_samples = 0, num_pdfs = samples_by_pdf->size();
  for (int32 i = 0; i < num_pdfs; i++)
    tot_num_samples += (*samples_by_pdf)[i].size();
  int32 cutoff = 250; // Just hardcode this.  When the #samples is
  // smaller than a plausible minibatch size, there's no point randomizing at
  // any smaller scale.  We will randomize the order of what we put out,
  // using standard methods; we just won't do this super-careful randomization
  // that tries to minimize variance of counts of different labels.
  
  if (tot_num_samples < cutoff) {   // Append all the pdfs to "samples".
    // Base case.
    size_t cur_size = samples->size();
    for (int32 i = 0; i < num_pdfs; i++)
      samples->insert(samples->end(),
                      (*samples_by_pdf)[i].begin(), (*samples_by_pdf)[i].end());
    // Randomize the samples we just added, so they're not in order by pdf.
    std::random_shuffle(samples->begin() + cur_size, samples->end());
  } else {
    std::vector<std::vector<std::pair<int32, int32> > > samples1(num_pdfs),
        samples2(num_pdfs);
    // Divide up samples_by_pdf into two pieces and recurse.  For each pdf
    // we try to ensure as even as possible a balance for its samples: we split
    // them in two, and if there's an odd number, allocate the odd one randomly
    // to the left or right.  (Has less variance than Bernoulli distribution.)
    for (int32 i = 0; i < num_pdfs; i++) {
      size_t size = (*samples_by_pdf)[i].size(); // #samples for this pdf.
      size_t half_size = size / 2; // Will round down.
      if (size % 2 != 0 && rand() % 2 == 0) // If odd #samples, allocate
        half_size++; // odd sample randomly to left or right.
      std::vector<std::pair<int32, int32> >::const_iterator
          begin_iter = (*samples_by_pdf)[i].begin(),
          middle_iter = begin_iter + half_size,
          end_iter = begin_iter + size;
      samples1[i].insert(samples1[i].end(), begin_iter, middle_iter);
      samples2[i].insert(samples2[i].end(), middle_iter, end_iter);
      std::vector<std::pair<int32, int32> > temp;
      (*samples_by_pdf)[i].swap(temp); // Trick to free up memory.
    }
    {
      std::vector<std::vector<std::pair<int32, int32> > > temp;
      samples_by_pdf->swap(temp); // Trick to free up memory.
    }
    RandomizeSamplesRecurse(&samples1, samples);
    {
      std::vector<std::vector<std::pair<int32, int32> > > temp;
      samples1.swap(temp); // Trick to free up memory.
    }
    RandomizeSamplesRecurse(&samples2, samples);
  }
}

void NnetDataRandomizer::GetRawSamples(
    std::vector<std::vector<std::pair<int32, int32> > > *pdf_counts) {
  pdf_counts->clear();
  int32 spk_data_size = 0;
  for (size_t i = 0; i < data_.size(); i++) { // For each training file
    const TrainingFile &tfile = *(data_[i]);
    if (i == 0) spk_data_size = tfile.spk_info.Dim();
    else KALDI_ASSERT(tfile.spk_info.Dim() == spk_data_size);
    KALDI_ASSERT(tfile.feats.NumRows() ==
                 static_cast<int32>(tfile.labels.size()));
    for (size_t j = 0; j < tfile.labels.size(); j++) {
      int32 pdf = tfile.labels[j];
      KALDI_ASSERT(pdf >= 0);
      if (static_cast<int32>(pdf_counts->size()) <= pdf)
        pdf_counts->resize(pdf+1);
      // The pairs are pairs of (file-index, frame-index).
      (*pdf_counts)[pdf].push_back(std::make_pair(i, j));
    }
  }
}

void NnetDataRandomizer::RandomizeSamples() {
  KALDI_ASSERT(samples_.empty());

  // The samples, indexed first by pdf.
  std::vector<std::vector<std::pair<int32, int32> > > samples_by_pdf;
  GetRawSamples(&samples_by_pdf);

  int32 num_pdfs = samples_by_pdf.size();
  // counts of each pdf.
  Vector<BaseFloat> reweighted_counts(num_pdfs);
  for (int32 i = 0; i < reweighted_counts.Dim(); i++)
    reweighted_counts(i) = static_cast<BaseFloat>(samples_by_pdf[i].size());
  KALDI_ASSERT(config_.frequency_power >= 0.0 &&
               config_.frequency_power <= 1.0);
  BaseFloat old_max_count = reweighted_counts.Max();
  KALDI_ASSERT(old_max_count > 0.0);  
  reweighted_counts.ApplyPow(config_.frequency_power);  
  // We now modify "reweighted_counts_" so that the maximum count
  // is the same as it was before.  This ensures that all counts
  // are at least as large as they were before we reweighted
  // (since the largest count will have been decreased by the
  // largest factor).
  reweighted_counts.Scale(old_max_count / reweighted_counts.Max());

  pdf_weights_.Resize(num_pdfs);
  /*
    For each pdf:
    Work out how many times each instance of that pdf appears.
    It's all about minimizing variance, so we don't just draw
    from a distribution with the right expectation.
   */
  for (int32 label = 0; label < reweighted_counts.Dim(); label++) {
    std::vector<std::pair<int32, int32> > &this_samples = samples_by_pdf[label];
    BaseFloat orig_count = this_samples.size(),
               new_count = reweighted_counts(label);
    KALDI_ASSERT(new_count >= orig_count);
    if (orig_count == 0) {
      pdf_weights_(label) = 0.0; // don't care.
      // nothing else to do.
    } else {
      int32 num_repeats = static_cast<int32>(new_count / orig_count); // Number
      // of times we repeat each sample; round down.  Some samples we'll repeat
      // one more time (num_extra samples)
      BaseFloat num_extra_float = new_count - orig_count*num_repeats;
      int32 num_extra = static_cast<int32>(num_extra_float); // round down.
      if (RandUniform() < num_extra_float - num_extra) num_extra++;
      // num_extra is an integer that is either slightly more or slightly
      // less than num_extra_float, but has the same expectation.

      { // set pdf_weights_[i] to the weight necessary to rescale the new
        // count to the old, original count.
        BaseFloat actual_count = num_repeats*orig_count + num_extra;
        pdf_weights_(label) = static_cast<BaseFloat>(orig_count) / actual_count;
      }
      // Randomize order of samples.  The first "num_extra" samples will
      // get count num_repeats+1, the rest will get count num_repeats.
      std::random_shuffle(this_samples.begin(), this_samples.end());

      std::vector<std::pair<int32, int32> > this_samples_new; // with repeats.
      for (int32 i = 0; i < static_cast<int32>(this_samples.size()); i++) {
        int32 n = num_repeats;
        if (i < num_extra) n++;
        for (int32 j = 0; j < n; j++) this_samples_new.push_back(this_samples[i]);
      }
      std::random_shuffle(this_samples_new.begin(), this_samples_new.end());
      this_samples = this_samples_new; // Write back to where we got it from.
    }
  }
  RandomizeSamplesRecurse(&samples_by_pdf, &samples_); // destroys samples_by_pdf.
  
  // Renormalize pdf_weights_ so that after weighting by the frequency
  // with which the trainer will see different pdfs, the average count
  // is 1.  This is an attempt to make the same learning rates applicable
  // (approximately), regardless what type of reweighting we used.
  // Note: we use the "reweighted_counts", which are floats not the actual
  // integerized counts, because otherwise I think expectations would  be
  // wrong in some exremely small way (due to the nonlinearity in the inverse
  // function).
  pdf_weights_.Scale(reweighted_counts.Sum()/
                     VecVec(pdf_weights_, reweighted_counts));

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
                                         const std::vector<int32> &labels) {
  TrainingFile *tf = new TrainingFile(feats, spk_info, labels);
  data_.push_back(tf);
}

NnetDataRandomizer::~NnetDataRandomizer() {
  for (size_t i = 0; i < data_.size(); i++)
    delete data_[i];
}

void NnetDataRandomizer::GetExample(const std::pair<int32, int32> &pair,
                                    NnetTrainingExample *example) const {
  int32 file_index = pair.first,
      frame_index = pair.second;
  KALDI_ASSERT(static_cast<size_t>(file_index) < data_.size());
  const TrainingFile &tf = *(data_[file_index]);
  KALDI_ASSERT(static_cast<size_t>(frame_index) < tf.labels.size());
  int32 label = tf.labels[frame_index];
  KALDI_ASSERT(label >= 0 && label < pdf_weights_.Dim());
  example->weight = pdf_weights_(label);
  example->label = label;
  example->spk_info = tf.spk_info;
  example->input_frames.Resize(left_context_ + 1 + right_context_,
                               tf.feats.NumCols());
  int32 start_frame = frame_index - left_context_,
      end_frame = frame_index + left_context_;
  for (int32 frame = start_frame; frame <= end_frame; frame++) {
    SubVector<BaseFloat> dest(example->input_frames, frame - start_frame);
    int32 frame_limited = frame; // we'll duplicate the start/end frame if we
    // cross the boundary of the utterance.
    if (frame_limited < 0)
      frame_limited = 0;
    if (frame_limited >= tf.feats.NumRows())
      frame_limited = tf.feats.NumRows() - 1;
    tf.feats.CopyRowToVec(frame_limited, &dest);
  }
}

bool NnetDataRandomizer::Done() {
  if (data_.empty()) return true;  // no data, so must be done.
  if (num_samples_tgt_ == -1) RandomizeSamples();  // first time called.
  if (num_samples_returned_ >= num_samples_tgt_) return true;
  if (samples_.empty()) RandomizeSamples();
  KALDI_ASSERT(!samples_.empty());
  return false;
}

const NnetTrainingExample &NnetDataRandomizer::Value() {
  KALDI_ASSERT(!Done());  // implies !samples_.empty().
  GetExample(samples_.back(), &cur_example_);
  return cur_example_;
}

void NnetDataRandomizer::Next() {
  KALDI_ASSERT(!Done());  // implies !samples_.empty().
  samples_.pop_back();
  num_samples_returned_++;
}

} // namespace
