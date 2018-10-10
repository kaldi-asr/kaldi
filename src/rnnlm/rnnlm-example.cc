// rnnlm/rnnlm-example.cc

// Copyright 2017  Daniel Povey

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

#include <numeric>
#include "rnnlm/rnnlm-example.h"

namespace kaldi {
namespace rnnlm {

void RnnlmExample::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<RnnlmExample>");
  WriteToken(os, binary, "<VocabSize>");
  WriteBasicType(os, binary, vocab_size);
  WriteToken(os, binary, "<NumChunks>");
  WriteBasicType(os, binary, num_chunks);
  WriteToken(os, binary, "<ChunkLength>");
  WriteBasicType(os, binary, chunk_length);
  WriteToken(os, binary, "<SampleGroupSize>");
  WriteBasicType(os, binary, sample_group_size);
  WriteToken(os, binary, "<NumSamples>");
  WriteBasicType(os, binary, num_samples);
  WriteToken(os, binary, "<InputWords>");
  WriteIntegerVector(os, binary, input_words);
  WriteToken(os, binary, "<OutputWords>");
  WriteIntegerVector(os, binary, output_words);
  WriteToken(os, binary, "<OutputWeights>");
  output_weights.Write(os, binary);
  WriteToken(os, binary, "<SampledWords>");
  WriteIntegerVector(os, binary, sampled_words);
  WriteToken(os, binary, "<SampleInvProbs>");
  sample_inv_probs.Write(os, binary);
  WriteToken(os, binary, "</RnnlmExample>");
}

void RnnlmExample::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<RnnlmExample>");
  ExpectToken(is, binary, "<VocabSize>");
  ReadBasicType(is, binary, &vocab_size);
  ExpectToken(is, binary, "<NumChunks>");
  ReadBasicType(is, binary, &num_chunks);
  ExpectToken(is, binary, "<ChunkLength>");
  ReadBasicType(is, binary, &chunk_length);
  ExpectToken(is, binary, "<SampleGroupSize>");
  ReadBasicType(is, binary, &sample_group_size);
  ExpectToken(is, binary, "<NumSamples>");
  ReadBasicType(is, binary, &num_samples);
  ExpectToken(is, binary, "<InputWords>");
  ReadIntegerVector(is, binary, &input_words);
  ExpectToken(is, binary, "<OutputWords>");
  ReadIntegerVector(is, binary, &output_words);
  ExpectToken(is, binary, "<OutputWeights>");
  output_weights.Read(is, binary);
  ExpectToken(is, binary, "<SampledWords>");
  ReadIntegerVector(is, binary, &sampled_words);
  ExpectToken(is, binary, "<SampleInvProbs>");
  sample_inv_probs.Read(is, binary);
  ExpectToken(is, binary, "</RnnlmExample>");
}

RnnlmExampleSampler::RnnlmExampleSampler(
    const RnnlmEgsConfig &config, const SamplingLm &arpa_sampling):
    config_(config), arpa_sampling_(arpa_sampling) {
  config_.Check();

  // The unigram distribution from the LM, modified according to
  // config_.special_symbol_prob and config_.uniform_prob_mass...
  std::vector<BaseFloat> unigram_distribution =
      arpa_sampling.GetUnigramDistribution();
  double sum = std::accumulate(unigram_distribution.begin(),
                                  unigram_distribution.end(),
                                  0.0);
  KALDI_ASSERT(std::fabs(sum - 1.0) < 0.01 &&
               "Unigram distribution from ARPA does not sum "
               "to (close to) 1");
  int32 num_words = unigram_distribution.size();
  if (config_.uniform_prob_mass > 0.0) {
    BaseFloat x = config_.uniform_prob_mass / (num_words - 1);
    for (int32 i = 1; i < num_words; i++)
      if (i != config_.bos_symbol && i != config_.brk_symbol)
        unigram_distribution[i] += x;
  }
  // If these are not zero, either something is wrong with your language model
  // or you supplied the wrong --bos-symbol or --brk-symbol options.  We allow
  // tiny values because the ARPA files sometimes give -99 as the unigram prob
  // for <s>.
  KALDI_ASSERT(unigram_distribution[config_.bos_symbol] < 1.0e-10);
  // we don't check that the <brk> symbol has very tiny prob because
  // it could have accumulated some probability mass via smoothing;
  // this is harmless.

  unigram_distribution[config_.bos_symbol] = config_.special_symbol_prob;
  unigram_distribution[config_.brk_symbol] = config_.special_symbol_prob;
  double new_sum = std::accumulate(unigram_distribution.begin(),
                                   unigram_distribution.end(),
                                   0.0),
      scale = 1.0 / new_sum;
  // rescale so it sums to almost 1; this is a requirement of the constructor
  // of class Sampler.

  int32 num_words_nonzero_prob = 0;
  for (std::vector<BaseFloat>::iterator iter = unigram_distribution.begin(),
           end = unigram_distribution.end(); iter != end; ++iter) {
    if (*iter != 0.0) num_words_nonzero_prob++;
    *iter *= scale;
  }

  if (config_.num_samples > num_words_nonzero_prob) {
    KALDI_WARN << "The number of samples (--num-samples=" << config_.num_samples
               << ") exceeds the number of words with nonzero probability "
               << num_words_nonzero_prob << " -> not doing sampling.  You could "
               << "skip creating the ARPA file, and not provide it, which "
               << "might save some bother.";
    config_.num_samples = 0;
  }
  if (config_.num_samples == 0) {
    sampler_ = NULL;
  } else {
    sampler_ = new Sampler(unigram_distribution);
  }
}


void RnnlmExampleSampler::SampleForMinibatch(RnnlmExample *minibatch) const {
  if (sampler_ == NULL) return;  // we're not actually sampling.
  KALDI_ASSERT(minibatch->chunk_length == config_.chunk_length &&
               minibatch->num_chunks == config_.num_chunks_per_minibatch &&
               config_.chunk_length % config_.sample_group_size == 0 &&
               static_cast<int32>(minibatch->input_words.size()) ==
               config_.chunk_length * config_.num_chunks_per_minibatch);
  int32 num_samples = config_.num_samples,
      sample_group_size = config_.sample_group_size,
      chunk_length = config_.chunk_length,
      num_groups = chunk_length / sample_group_size;
  minibatch->num_samples = num_samples;
  minibatch->sample_group_size = sample_group_size;
  minibatch->sampled_words.resize(num_groups * num_samples);
  minibatch->sample_inv_probs.Resize(num_groups * num_samples);

  for (int32 g = 0; g < num_groups; g++) {
    SampleForGroup(g, minibatch);
  }
}


void RnnlmExampleSampler::SampleForGroup(int32 g,
                                         RnnlmExample *minibatch) const {
  // All words that appear on the output are required to appear in the sample.  we
  // need to figure what this set of words is.
  int32 num_chunks_per_minibatch = config_.num_chunks_per_minibatch;
  std::vector<int32> words_we_must_sample;
  for (int32 t = g * config_.sample_group_size;
       t < (g + 1) * config_.sample_group_size; t++) {
    for (int32 n = 0; n < num_chunks_per_minibatch; n++) {
      int32 i = t * num_chunks_per_minibatch + n;
      int32 output_word = minibatch->output_words[i];
      words_we_must_sample.push_back(output_word);
    }
  }
  SortAndUniq(&words_we_must_sample);

  // 'hist_weights' is a representation of a weighted set of histories.
  std::vector<std::pair<std::vector<int32>, BaseFloat> > hist_weights;
  GetHistoriesForGroup(g, *minibatch, &hist_weights);
  KALDI_ASSERT(!hist_weights.empty());  // we made sure of this.

  // 'higher_order_probs' and 'unigram_weight' are a compact representation of
  // an (unnormalized) distribution that is the suitably weighted sum of the
  // distributions that the language model predicts given the history states
  // present in 'hist_weights'.
  // We represent the distribution in this way, instead of just as a vector,
  // so that it is efficient even when the vocabulary size is very large.
  std::vector<std::pair<int32, BaseFloat> > higher_order_probs;
  BaseFloat unigram_weight = arpa_sampling_.GetDistribution(hist_weights,
                                                            &higher_order_probs);

  // 'sample' will be a list of pairs (integer word-id, inclusion probability).
  std::vector<std::pair<int32, BaseFloat> > sample;

  // the 'sampler_' object knows how to sample from an unnormalized distribution
  // represented as unigram-weight and a list of higher-than-unigram (word-id,
  // additional-weight) pairs.
  int32 num_samples = config_.num_samples;
  sampler_->SampleWords(num_samples, unigram_weight,
                        higher_order_probs, words_we_must_sample,
                        &sample);
  KALDI_ASSERT(sample.size() == static_cast<size_t>(num_samples));
  std::sort(sample.begin(), sample.end());
  // write to the 'sampled_words' and 'sample_inv_probs' arrays.
  for (int32 s = 0; s < num_samples; s++) {
    int32 i = (g * num_samples) + s;
    minibatch->sampled_words[i] = sample[s].first;
    KALDI_ASSERT(sample[s].second > 0.0);
    minibatch->sample_inv_probs(i) = 1.0 / sample[s].second;
  }
  RenumberOutputWordsForGroup(g, minibatch);
}

void RnnlmExampleSampler::RenumberOutputWordsForGroup(
    int32 g, RnnlmExample *minibatch) const {
  int32 sample_group_size = config_.sample_group_size,
      num_samples = config_.num_samples,
      num_chunks_per_minibatch = config_.num_chunks_per_minibatch,
      num_outputs_per_group = sample_group_size * num_chunks_per_minibatch,
      vocab_size = minibatch->vocab_size;

  // get the range of 'sampled_words' that covers this group.
  const int32 *sampled_words_ptr = &(minibatch->sampled_words[0]),
      *sampled_words_begin = sampled_words_ptr + (g * num_samples),
      *sampled_words_end = sampled_words_begin + num_samples;

  int32 *output_words_ptr = &(minibatch->output_words[0]),
      *output_words_iter = output_words_ptr + (g * num_outputs_per_group),
      *output_words_end = output_words_iter + num_outputs_per_group;
  for (; output_words_iter != output_words_end; ++output_words_iter) {
    int32 output_word = *output_words_iter;
    // note: output_word is > 0 because epsilon won't ever occur there,
    // although in a sense 0 is a valid output-word id.
    KALDI_ASSERT(output_word > 0 && output_word < vocab_size);
    const int32 *sampled_words_ptr = std::lower_bound(sampled_words_begin,
                                                      sampled_words_end,
                                                      output_word);
    if (*sampled_words_ptr != output_word) {
      KALDI_ERR << "Output word not found in samples (indicates code error)";
    }
    int32 renumbered_output_word = sampled_words_ptr - sampled_words_begin;
    *output_words_iter = renumbered_output_word;
  }
}


void RnnlmExampleSampler::GetHistoriesForGroup(
    int32 g, const RnnlmExample &minibatch,
    std::vector<std::pair<std::vector<int32>, BaseFloat> > *hist_weights) const {
  // initially store as an unordered_map so we can remove duplicates.

  // hist_to_weight maps from the history to the (unnormalized) weight for that
  // history.  It represents a weighted combination of history-states that we
  // will get a distribution for (from the ARPA LM) and sample from.
  std::unordered_map<std::vector<int32>, BaseFloat, VectorHasher<int32> > hist_to_weight;

  hist_weights->clear();
  KALDI_ASSERT(arpa_sampling_.Order() > 0);
  int32 max_history_length = arpa_sampling_.Order() - 1,
      num_chunks_per_minibatch = config_.num_chunks_per_minibatch;

  // This block sets up the 'hist_to_weight' map.  Note: sample_group_size
  // will normally be small, like 1, 2 or 4.
  for (int32 t = g * config_.sample_group_size;
       t < (g + 1) * config_.sample_group_size; t++) {
    for (int32 n = 0; n < num_chunks_per_minibatch; n++) {
      int32 i = t * num_chunks_per_minibatch + n;
      BaseFloat this_weight = minibatch.output_weights(i);
      KALDI_ASSERT(this_weight >= 0);
      if (this_weight == 0.0)
        continue;
      std::vector<int32> history;
      GetHistory(t, n, minibatch, max_history_length, &history);
      // note: if the key did not exist in the map, it is as
      // if the value were zero, see here:
      // https://stackoverflow.com/questions/8943261/stdunordered-map-initialization
      // .. this is at least since C++11, maybe since C++03.
      hist_to_weight[history] += this_weight;
    }
  }
  if (hist_to_weight.empty()) {
    KALDI_WARN << "No histories seen (we don't expect to see this very often)";
    std::vector<int32> empty_history;
    hist_to_weight[empty_history] = 1.0;
  }
  std::unordered_map<std::vector<int32>, BaseFloat, VectorHasher<int32> >::const_iterator
      iter = hist_to_weight.begin(), end = hist_to_weight.end();
  hist_weights->reserve(hist_to_weight.size());
  for (; iter != end; ++iter)
    hist_weights->push_back(std::pair<std::vector<int32>, BaseFloat>(
        iter->first, iter->second));
}

void RnnlmExampleSampler::GetHistory(
    int32 t, int32 n,
    const RnnlmExample &minibatch,
    int32 max_history_length,
    std::vector<int32> *history) const {
  history->reserve(max_history_length);
  history->clear();
  int32 num_chunks_per_minibatch = config_.num_chunks_per_minibatch;

  // e.g. if 'max_history_length' is 2, we iterate over t_step = [0, -1].
  // you'll notice that the first history-position we look for when
  // predicting position 't' is 'hist_t = t + 0 = t'.  This may be
  // surprising-- you might be expecting that t-1 would be the first
  // position we'd look at-- but notice that we're looking at the
  // input word, not the output word.
  for (int32 t_step = 0; t_step > -max_history_length; t_step--) {
    int32 hist_t = t + t_step;
    KALDI_ASSERT(hist_t >= 0);  // .. or we should have done 'break' below
                                // before reaching this value of t_step.  If
                                // this assert fails it means that a minibatch
                                // doesn't start with input_word equal to
                                // bos_symbol or brk_symbol, which is a bug.
    int32 i = hist_t * num_chunks_per_minibatch + n,
        history_word = minibatch.input_words[i];
    history->push_back(history_word);
    if (history_word == config_.bos_symbol ||
        history_word == config_.brk_symbol)
      break;
  }
  // we want the most recent word to be the last word in 'history', so the order
  // needs to be reversed.
  std::reverse(history->begin(), history->end());
}



void RnnlmExampleCreator::AcceptSequence(
    BaseFloat weight, const std::vector<int32> &words) {
  CheckSequence(weight, words);
  SplitSequenceIntoChunks(weight, words);
  num_sequences_processed_++;
  while (chunks_.size() > static_cast<size_t>(config_.chunk_buffer_size)) {
    if (!ProcessOneMinibatch())
      break;
  }
}

RnnlmExampleCreator::~RnnlmExampleCreator() {
  Flush();
  BaseFloat words_per_chunk = num_words_processed_ * 1.0 /
      num_chunks_processed_,
      chunks_per_minibatch = num_chunks_processed_ * 1.0 /
      num_minibatches_written_;
  KALDI_LOG << "Combined " << num_sequences_processed_ << "/"
            << num_chunks_processed_
            << " sequences/chunks into " << num_minibatches_written_
            << " minibatches (" << chunks_.size()
            << " chunks left over)";
 KALDI_LOG << "Overall there were "
           << words_per_chunk << " words per chunk; "
           << chunks_per_minibatch << " chunks per minibatch.";
 for (size_t i = 0; i < chunks_.size(); i++)
   delete chunks_[i];
}

RnnlmExampleCreator::SingleMinibatchCreator::SingleMinibatchCreator(
    const RnnlmEgsConfig &config):
    config_(config),
    eg_chunks_(config_.num_chunks_per_minibatch) {
  for (int32 i = 0; i < config_.num_chunks_per_minibatch; i++)
    empty_eg_chunks_.push_back(i);
}

bool RnnlmExampleCreator::SingleMinibatchCreator::AcceptChunk(
    RnnlmExampleCreator::SequenceChunk *chunk) {
  int32 chunk_len = chunk->Length();
  if (chunk_len == config_.chunk_length) {  // maximum-sized chunk.
    if (empty_eg_chunks_.empty()) {
      return false;
    } else  {
      int32 i = empty_eg_chunks_.back();
      KALDI_ASSERT(size_t(i) < eg_chunks_.size() && eg_chunks_[i].empty());
      eg_chunks_[i].push_back(chunk);
      empty_eg_chunks_.pop_back();
      return true;
    }
  } else {  // smaller-sized chunk than maximum chunk size.
    KALDI_ASSERT(chunk_len < config_.chunk_length);
    // Find the index best_i into partial_eg_chunks_, such
    // that partial_eg_chunks_[best_i] is a pair (best_j,
    // best_space_left) such that space_left >= chunk_len, with
    // best_space_left as small as possible.
    int32 best_i = -1, best_j = -1,
        best_space_left = std::numeric_limits<int32>::max(),
        size = partial_eg_chunks_.size();
    for (int32 i = 0; i < size; i++) {
      int32 this_space_left = partial_eg_chunks_[i].second;
      if (this_space_left >= chunk_len && this_space_left < best_space_left) {
        best_i = i;
        best_j = partial_eg_chunks_[i].first;
        best_space_left = this_space_left;
      }
    }
    if (best_i != -1) {
      partial_eg_chunks_[best_i] = partial_eg_chunks_.back();
      partial_eg_chunks_.pop_back();
    } else {
      // consume a currently-unused chunk, if available.
      if (empty_eg_chunks_.empty()) {
        return false;
      } else {
        best_j = empty_eg_chunks_.back();
        empty_eg_chunks_.pop_back();
        best_space_left = config_.chunk_length;
      }
    }
    int32 new_space_left = best_space_left - chunk_len;
    KALDI_ASSERT(new_space_left >= 0);
    if (new_space_left > 0) {
      partial_eg_chunks_.push_back(std::pair<int32, int32>(best_j,
                                                           new_space_left));
    }
    eg_chunks_[best_j].push_back(chunk);
    return true;
  }
}


RnnlmExampleCreator::SingleMinibatchCreator::~SingleMinibatchCreator() {
  for (size_t i = 0; i < eg_chunks_.size(); i++)
    for (size_t j = 0; j < eg_chunks_[i].size(); j++)
      delete eg_chunks_[i][j];
}


void RnnlmExampleCreator::SingleMinibatchCreator::CreateMinibatchOneSequence(
    int32 n, RnnlmExample *minibatch) {
  // Much of the code here is about figuring out what to do if we haven't
  // completely used up the potential length of the sequence.  We first try
  // giving extra left-context to any split-up pieces of sequence that could potentially
  // use extra left-context; when that avenue is exhausted, we
  // pad at the end with </s> symbols with zero weight.


  KALDI_ASSERT(static_cast<size_t>(n) < eg_chunks_.size());
  const std::vector<SequenceChunk*> &this_chunks = eg_chunks_[n];
  int32 num_chunks = this_chunks.size();
  // note: often num_chunks will be 1, occasionally 0 (if we've run out of
  // data), and sometimes more than 1 (if we're appending multiple chunks
  // together because they were shorter than config_.chunk_length).


  // total_current_chunk_length is the total Length() of all the chunks.
  int32 total_current_chunk_length = 0;
  for (int32 c = 0; c < num_chunks; c++) {
    total_current_chunk_length += this_chunks[c]->Length();
  }
  KALDI_ASSERT(total_current_chunk_length <= config_.chunk_length);
  int32 extra_length_available = config_.chunk_length - total_current_chunk_length;

  while (true) {
    bool changed = false;
    for (int32 c = 0; c < num_chunks; c++) {
      if (this_chunks[c]->context_begin > 0 && extra_length_available > 0) {
        changed = true;
        this_chunks[c]->context_begin--;
        extra_length_available--;
      }
    }
    if (!changed)
      break;
  }

  int32 pos = 0;  // position in the sequence (we increase this every time a word
                  // gets added).
  for (int32 c = 0; c < num_chunks; c++) {
    SequenceChunk &chunk = *(this_chunks[c]);

    // note: begin and end are the indexes of the first and the last-plus-one
    // words in the sequence that we *predict*.
    // you can think of real_begin as the index of the first real word in the
    // sequence that we use as left context (however it will be preceded by
    // either a <s> or a <brk>, depending whether 'real_begin' is 0 or >0).
    // For these positions that are only used as left context, and not predicted
    // the weight of the output (predicted) word is zero.  'begin' is the index
    // of the first predicted word.
    int32 context_begin = chunk.context_begin,
        begin = chunk.begin,
        end = chunk.end;
    for (int32 i = context_begin; i < end; i++) {
      int32 output_word = (*chunk.sequence)[i],
          input_word;
      if (i == context_begin) {
        if (context_begin == 0) input_word = config_.bos_symbol;
        else input_word = config_.brk_symbol;
      } else {
        input_word = (*chunk.sequence)[i - 1];
      }
      BaseFloat weight = (i < begin ? 0.0 : chunk.weight);
      Set(n, pos, input_word, output_word, weight, minibatch);
      pos++;
    }
  }
  for (; pos < config_.chunk_length; pos++) {
    // fill the rest with <s> as input and </s> as output
    // and weight of 0.0.  The symbol-id doesn't really matter
    // so we pick ones that we know are valid inputs and outputs.
    int32 input_word = config_.bos_symbol,
        output_word = config_.eos_symbol;
    BaseFloat weight = 0.0;
    Set(n, pos, input_word, output_word, weight, minibatch);
  }
}


void RnnlmExampleCreator::SingleMinibatchCreator::Set(
    int32 n, int32 t, int32 input_word, int32 output_word,
    BaseFloat weight, RnnlmExample *minibatch) const {
  KALDI_ASSERT(n >= 0 && n < config_.num_chunks_per_minibatch &&
               t >= 0 && t < config_.chunk_length &&
               weight >= 0.0);

  int32 i = t * config_.num_chunks_per_minibatch + n;
  minibatch->input_words[i] = input_word;
  minibatch->output_words[i] = output_word;
  minibatch->output_weights(i) = weight;
}


void RnnlmExampleCreator::SingleMinibatchCreator::CreateMinibatch(
    RnnlmExample *minibatch) {
  minibatch->vocab_size = config_.vocab_size;
  minibatch->num_chunks = config_.num_chunks_per_minibatch;
  minibatch->chunk_length = config_.chunk_length;
  minibatch->num_samples = config_.num_samples;
  int32 num_words = config_.chunk_length * config_.num_chunks_per_minibatch;
  minibatch->input_words.resize(num_words);
  minibatch->output_words.resize(num_words);
  minibatch->output_weights.Resize(num_words);
  minibatch->sampled_words.clear();
  for (int32 n = 0; n < config_.num_chunks_per_minibatch; n++) {
    CreateMinibatchOneSequence(n, minibatch);
  }
}

RnnlmExampleCreator::SequenceChunk* RnnlmExampleCreator::GetRandomChunk() {
  KALDI_ASSERT(!chunks_.empty());
  int32 pos = RandInt(0, chunks_.size() - 1);
  SequenceChunk *ans = chunks_[pos];
  chunks_[pos] = chunks_.back();
  chunks_.pop_back();
  return ans;
}

bool RnnlmExampleCreator::ProcessOneMinibatch() {
  // A couple of configuration values that are not important enough
  // to go in the config...
  // 'chunks_proportion' controls when we discard a small number of
  // chunks rather than form a new minibatch, after we've finished
  // reading the data and have a small bit left over.
  const BaseFloat chunks_proportion = 0.0;  // TODO: revert to 0.5.
  // 'max_rejections' is the maximum number of successive chunks that
  // can be rejected for being 'too big', before we give up an accept
  // the minibatch as-is.
  const int32 max_rejections = 5;

  if (chunks_.size() <
      std::max<size_t>(1,
                       config_.num_chunks_per_minibatch * chunks_proportion)) {
    // there's not enough data to form one minibatch.
    return false;
  }
  SingleMinibatchCreator s(config_);
  int32 cur_rejections = 0;
  while (!chunks_.empty() && cur_rejections < max_rejections) {
    int32 i = RandInt(0, chunks_.size() - 1);
    if (s.AcceptChunk(chunks_[i])) {
      num_chunks_processed_++;
      num_words_processed_ += chunks_[i]->Length();
      chunks_[i] = chunks_.back();
      chunks_.pop_back();
      cur_rejections = 0;
    } else {
      cur_rejections++;
    }
  }
  RnnlmExample *minibatch = new RnnlmExample();
  s.CreateMinibatch(minibatch);
  std::ostringstream os;
  os << "minibatch-" << num_minibatches_written_;
  std::string key = os.str();
  num_minibatches_written_++;
  if (minibatch_sampler_ == NULL) {
    // write it directly from this function.
    writer_->Write(key, *minibatch);
    delete minibatch;
  } else {
    // the sampling, since it can be slow, will be done in parallel by as many
    // background threads as the user specified via the --num-threads option.
    // SamplerTask will also write it out.
    sampling_sequencer_.Run(new SamplerTask(*minibatch_sampler_,
                                            key, writer_, minibatch));
  }
  return true;
}


void RnnlmExampleCreator::SplitSequenceIntoChunks(
    BaseFloat weight, const std::vector<int32> &words) {
  std::shared_ptr<std::vector<int32> > ptr (new std::vector<int32>());
  ptr->reserve(words.size() + 1);
  ptr->insert(ptr->end(), words.begin(), words.end());
  ptr->push_back(config_.eos_symbol);  // add the terminating </s>.

  int32 sequence_length = ptr->size();  // == words.size() + 1
  if (sequence_length <= config_.chunk_length) {
    chunks_.push_back(new SequenceChunk(config_, ptr, weight,
                                        0, sequence_length));
  } else {
    std::vector<int32> chunk_lengths;
    ChooseChunkLengths(sequence_length, &chunk_lengths);
    int32 cur_start = 0;
    for (size_t i = 0; i < chunk_lengths.size(); i++) {
      int32 this_end = cur_start + chunk_lengths[i];
      chunks_.push_back(new SequenceChunk(config_, ptr, weight,
                                          cur_start, this_end));
      cur_start = this_end;
    }
  }
}

// see comment in rnnlm-example.h, by its declaration.
void RnnlmExampleCreator::ChooseChunkLengths(
    int32 sequence_length,
    std::vector<int32> *chunk_lengths) {
  KALDI_ASSERT(sequence_length > config_.chunk_length);
  chunk_lengths->clear();
  int32 tot = sequence_length - config_.min_split_context,
     chunk_length_no_context = config_.chunk_length - config_.min_split_context;
  KALDI_ASSERT(chunk_length_no_context > 0);
  // divide 'tot' into pieces of size <= config_.chunk_length - config_.min_split_context.

  // note:
  for (int32 i = 0; i < tot / chunk_length_no_context; i++)
    chunk_lengths->push_back(chunk_length_no_context);
  KALDI_ASSERT(!chunk_lengths->empty());
  int32 remaining_size = tot % chunk_length_no_context;
  if (remaining_size != 0) {
    // put the smaller piece in a random location.
    (*chunk_lengths)[RandInt(0, chunk_lengths->size() - 1)] = remaining_size;
    chunk_lengths->push_back(chunk_length_no_context);
  }
  (*chunk_lengths)[0] += config_.min_split_context;
  KALDI_ASSERT(std::accumulate(chunk_lengths->begin(), chunk_lengths->end(), 0)
               == sequence_length);
}

void RnnlmExampleCreator::CheckSequence(
    BaseFloat weight,
    const std::vector<int32> &words) {
  KALDI_ASSERT(weight > 0.0);
  int32 bos_symbol = config_.bos_symbol,
      brk_symbol = config_.brk_symbol,
      eos_symbol = config_.eos_symbol,
      vocab_size = config_.vocab_size;
  for (size_t i = 0; i < words.size(); i++) {
    // note: eos_symbol within a sequence isn't disallowed; this
    // is allowed as a way to encode multiple turns of a conversation,
    // and similar scenarios.
    KALDI_ASSERT(words[i] != bos_symbol && words[i] != brk_symbol &&
                 words[i] > 0 && words[i] < vocab_size);
  }
  if (!words.empty() && words.back() == eos_symbol) {
    // we may rate-limit this warning eventually if people legitimately need to
    // do this.
    KALDI_WARN << "Raw word sequence contains </s> at the end.  "
        "Is this a bug in your data preparation?  We'll add another one.";
  }
}

void RnnlmExampleCreator::Check() const {
  config_.Check();
  if (minibatch_sampler_ != NULL) {
    if (minibatch_sampler_->VocabSize() > config_.vocab_size) {
      KALDI_ERR << "Option --vocab-size=" << config_.vocab_size
                << " is inconsistent with the language model.";
    }
  }
}

void RnnlmExampleCreator::Process(std::istream &is) {
  int32 num_lines = 0;
  std::vector<int32> words;
  std::string line;
  while (getline(is, line)) {
    num_lines++;
    std::istringstream line_is(line);
    BaseFloat weight;
    line_is >> weight;
    words.clear();
    int32 word;
    while (line_is >> word) {
      words.push_back(word);
    }
    if (!line_is.eof()) {
      KALDI_ERR << "Could not interpret input: " << line;
    }
    this->AcceptSequence(weight, words);
  }
  KALDI_LOG << "Processed " << num_lines << " lines of input.";
}

void RnnlmExample::Swap(RnnlmExample *other) {
  std::swap(vocab_size, other->vocab_size);
  std::swap(num_chunks, other->num_chunks);
  std::swap(chunk_length, other->chunk_length);
  std::swap(sample_group_size, other->sample_group_size);
  std::swap(num_samples, other->num_samples);
  input_words.swap(other->input_words);
  output_words.swap(other->output_words);
  output_weights.Swap(&(other->output_weights));
  sampled_words.swap(other->sampled_words);
  sample_inv_probs.Swap(&(other->sample_inv_probs));
}

}  // namespace rnnlm
}  // namespace kaldi
