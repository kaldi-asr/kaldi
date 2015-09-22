// ctc/cctc-transition-model.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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


#include "ctc/cctc-transition-model.h"
#include "cudamatrix/cu-matrix.h"

namespace kaldi {
namespace ctc {


BaseFloat CctcTransitionModel::GraphLabelToLmProb(int32 graph_label) const {
  int32 history = (graph_label - 1) / (num_phones_ + 1),
      phone = (graph_label - 1) % (num_phones_ + 1);
  // if graph_label is out of range, we'll get a segmentation fault or
  // an assertion in the operator () of Vector.
  // note: phone might be zero for blank, in which case we'd return 1.0.
  return history_state_info_[history].phone_lm_prob(phone);
}

int32 CctcTransitionModel::GraphLabelToHistoryState(int32 graph_label) const {
  int32 history = (graph_label - 1) / (num_phones_ + 1);
  KALDI_ASSERT(static_cast<size_t>(history) < history_state_info_.size());
  return history;
}

int32 CctcTransitionModel::GetNextHistoryState(int32 history_state,
                                               int32 phone) const {
  KALDI_ASSERT(static_cast<size_t>(history_state) < history_state_info_.size() &&
               phone >= 0 && phone <= num_phones_);
  return history_state_info_[history_state].next_history_state[phone];
}

BaseFloat CctcTransitionModel::GetLmProb(int32 history_state,
                                         int32 phone) const {
  KALDI_ASSERT(static_cast<size_t>(history_state) < history_state_info_.size() &&
               phone >= 0 && phone <= num_phones_);
  return history_state_info_[history_state].phone_lm_prob(phone);
}

int32 CctcTransitionModel::GetOutputIndex(int32 history_state,
                                          int32 phone) const {
  KALDI_ASSERT(static_cast<size_t>(history_state) < history_state_info_.size() &&
               phone >= 0 && phone <= num_phones_);
  return history_state_info_[history_state].output_index[phone];
}

int32 CctcTransitionModel::GraphLabelToNextHistoryState(
    int32 graph_label) const {
  int32 history = (graph_label - 1) / (num_phones_ + 1),
      phone = (graph_label - 1) % (num_phones_ + 1);
  KALDI_ASSERT(static_cast<size_t>(history) < history_state_info_.size());
  return history_state_info_[history].next_history_state[phone];
}

int32 CctcTransitionModel::InitialHistoryState() const {
  return initial_history_state_;
}

int32 CctcTransitionModel::GetGraphLabel(int32 history_state,
                                         int32 phone) const {
  KALDI_ASSERT(static_cast<size_t>(phone) <= static_cast<size_t>(num_phones_) &&
               static_cast<size_t>(history_state) < history_state_info_.size());
  return history_state * (num_phones_ + 1)  + phone + 1;
}

int32 CctcTransitionModel::GraphLabelToOutputIndex(int32 graph_label) const {
  int32 history = (graph_label - 1) / (num_phones_ + 1),
      phone = (graph_label - 1) % (num_phones_ + 1);
  return history_state_info_[history].output_index[phone];
}

void CctcTransitionModel::Check() const {
  int32 num_phones = num_phones_;
  KALDI_ASSERT(num_phones > 0);
  KALDI_ASSERT(phone_left_context_ >= 0);
  KALDI_ASSERT(num_output_indexes_ > 0);
  KALDI_ASSERT(num_non_blank_indexes_ > 0 &&
               num_non_blank_indexes_ < num_output_indexes_);
  int32 num_histories = history_state_info_.size();
  KALDI_ASSERT(static_cast<size_t>(initial_history_state_) <
               history_state_info_.size());
  std::vector<bool> output_index_seen(num_output_indexes_, false);
  for (int32 h = 0; h < num_histories; h++) {
    const HistoryStateInfo &info = history_state_info_[h];
    // see blank should not change the history state.
    KALDI_ASSERT(info.next_history_state[0] == h);
    for (int32 p = 1; p <= num_phones; p++) {
      int32 next_h = info.next_history_state[p];
      KALDI_ASSERT(next_h >= 0 && next_h < num_histories);
    }
    // output-index if we predict blank should be after the
    // non-blank indexes.
    KALDI_ASSERT(info.output_index[0] >= num_non_blank_indexes_);
    output_index_seen[info.output_index[0]] = true;
    for (int32 p = 1; p <= num_phones; p++) {
      int32 output_index = info.output_index[p];
      KALDI_ASSERT(output_index < num_non_blank_indexes_);
      output_index_seen[output_index] = true;
    }
    KALDI_ASSERT(info.phone_lm_prob.Min() > 0.0);
    // check that LM prob for blank is 1.0.
    KALDI_ASSERT(info.phone_lm_prob(0) == 1.0);
    // .. and that the LM probs of the real phones (>0) sum to one. 
    KALDI_ASSERT(fabs(info.phone_lm_prob.Sum() - 2.0) < 0.001);
  }
  int32 num_not_seen = 0;
  for (int32 output_index = 0; output_index < num_output_indexes_;
       output_index++) {
    if (!output_index_seen[output_index]) {
      num_not_seen++;
      KALDI_WARN << "Output index " << output_index << " is never used.";
    }
  }
  // We don't believe it should ever happen that output indexes are not
  // seen, so assert that it doesn't happen.
  KALDI_ASSERT(num_not_seen == 0 &&
               "...this assert may later need to be revised/removed");
  

  // Do a spot check that after seeing phone_left_context_ real phones,
  // we always get to the same history state regardless of where we started.
  int32 num_test = 50;
  for (int32 i = 0; i < num_test; i++) {
    int32 h1 = RandInt(0, num_histories - 1),
        h2 = RandInt(0, num_histories - 1);
    for (int32 n = 0; n < phone_left_context_; n++) {
      int32 p = RandInt(1, num_phones);  // a real phone.
      h1 = history_state_info_[h1].next_history_state[p];
      h2 = history_state_info_[h2].next_history_state[p];
    }
    KALDI_ASSERT(h1 == h2 && "Test of phone_left_context_ failed.");
  }
}

void CctcTransitionModel::Write(std::ostream &os, bool binary) const {
  Check();
  WriteToken(os, binary, "<CctcTransitionModel>");
  if (!binary) os << "\n";
  WriteToken(os, binary, "<NumPhones>");
  WriteBasicType(os, binary, num_phones_);
  if (!binary) os << "\n";
  WriteToken(os, binary, "<PhoneLeftContext>");
  WriteBasicType(os, binary, phone_left_context_);
  if (!binary) os << "\n";
  WriteToken(os, binary, "<NumOutputIndexes>");
  WriteBasicType(os, binary, num_output_indexes_);
  if (!binary) os << "\n";
  WriteToken(os, binary, "<NumNonBlankIndexes>");
  WriteBasicType(os, binary, num_non_blank_indexes_);
  if (!binary) os << "\n";
  WriteToken(os, binary, "<InitialHistoryState>");
  WriteBasicType(os, binary, initial_history_state_);
  if (!binary) os << "\n";
  WriteToken(os, binary, "<NumHistoryStates>");
  int32 num_history_states = history_state_info_.size();
  WriteBasicType(os, binary, num_history_states);
  if (!binary) os << "\n";
  WriteToken(os, binary, "<HistoryStates>");
  for (int32 h = 0; h < num_history_states; h++) {
    const HistoryStateInfo &info = history_state_info_[h];
    WriteIntegerVector(os, binary, info.next_history_state);
    if (!binary) os << "\n";
    WriteIntegerVector(os, binary, info.output_index);
    if (!binary) os << "\n";
    info.phone_lm_prob.Write(os, binary);
  }
  WriteToken(os, binary, "</CctcTransitionModel>");  
}


void CctcTransitionModel::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<CctcTransitionModel>");
  ExpectToken(is, binary, "<NumPhones>");
  ReadBasicType(is, binary, &num_phones_);
  ExpectToken(is, binary, "<PhoneLeftContext>");
  ReadBasicType(is, binary, &phone_left_context_);
  ExpectToken(is, binary, "<NumOutputIndexes>");
  ReadBasicType(is, binary, &num_output_indexes_);
  ExpectToken(is, binary, "<NumNonBlankIndexes>");
  ReadBasicType(is, binary, &num_non_blank_indexes_);
  ExpectToken(is, binary, "<InitialHistoryState>");
  ReadBasicType(is, binary, &initial_history_state_);
  ExpectToken(is, binary, "<NumHistoryStates>");
  int32 num_history_states = history_state_info_.size();
  ReadBasicType(is, binary, &num_history_states);
  KALDI_ASSERT(num_history_states > 0 && num_history_states < 10000000);
  history_state_info_.resize(num_history_states);
  ExpectToken(is, binary, "<HistoryStates>");
  for (int32 h = 0; h < num_history_states; h++) {
    HistoryStateInfo &info = history_state_info_[h];
    ReadIntegerVector(is, binary, &info.next_history_state);
    ReadIntegerVector(is, binary, &info.output_index);
    info.phone_lm_prob.Read(is, binary);
  }
  ExpectToken(is, binary, "</CctcTransitionModel>");
  Check();
}


void CctcTransitionModel::ComputeWeights(Matrix<BaseFloat> *weights) const {
  int32 num_history_states = history_state_info_.size(),
      num_output_indexes = num_output_indexes_,
      num_phones = num_phones_;
  weights->Resize(num_history_states,
                  num_output_indexes);
  for (int32 h = 0; h < num_history_states; h++) {
    const HistoryStateInfo &info = history_state_info_[h];
    SubVector<BaseFloat> row(*weights, h);
    for (int32 p = 0; p <= num_phones; p++) {
      int32 output_index = info.output_index[p];
      BaseFloat lm_prob = info.phone_lm_prob(p);
      row(output_index) += lm_prob;
    }
  }
}

void CctcTransitionModel::ComputeWeights(CuMatrix<BaseFloat> *cu_weights) const {
  Matrix<BaseFloat> weights;
  ComputeWeights(&weights);
  cu_weights->Resize(0, 0);
  cu_weights->Swap(&weights);
}

CctcTransitionModelCreator::CctcTransitionModelCreator(
    const ContextDependency &ctx_dep,
    const LanguageModel &phone_lang_model):
    ctx_dep_(ctx_dep),
    phone_lang_model_(phone_lang_model) { }


void CctcTransitionModelCreator::InitCctcTransitionModel(
    CctcTransitionModel *model) {
  lm_hist_state_map_.Init(phone_lang_model_);
  KALDI_LOG << "Phone language model has "
            << lm_hist_state_map_.NumLmHistoryStates() << " history states.";
  KALDI_LOG << "Decision tree has " << (ctx_dep_.ContextWidth() - 1)
            << " phones of left context.";
  num_tree_leaves_ = ctx_dep_.NumPdfs();
  num_output_indexes_ = num_tree_leaves_ + lm_hist_state_map_.NumLmHistoryStates();
  KALDI_LOG << "There are " << num_output_indexes_ << " output indexes, = "
            << num_tree_leaves_ << " for non-blank, and "
            << lm_hist_state_map_.NumLmHistoryStates() << " for blank.";
  
  GetInitialHistoryStates();
  while (MergeHistoryStatesOnePass());
  OutputToTransitionModel(model);
  model->Check();
}


void CctcTransitionModelCreator::GetInitialHistories(SetType *hist_set) const {
  hist_set->clear();
  int32 num_phones = phone_lang_model_.VocabSize();
  KALDI_ASSERT(ctx_dep_.CentralPosition() == ctx_dep_.ContextWidth() - 1 &&
               "Tree for CCTC model must have left context only.");

  int32 tree_left_context = ctx_dep_.ContextWidth() - 1;  

  std::vector<std::vector<int32> > hist_state_queue;
  
  for (int32 i = 0; i < lm_hist_state_map_.NumLmHistoryStates(); i++) {
    const std::vector<int32> &hist = lm_hist_state_map_.GetHistoryForState(i);
    hist_set->insert(hist);
    if (hist.size() < static_cast<size_t>(tree_left_context))
      hist_state_queue.push_back(hist);
  }

  while (!hist_state_queue.empty()) {
    std::vector<int32> vec = hist_state_queue.back();
    hist_state_queue.pop_back();
    // vec will be shorter than tree_left_context.  make sure each more-specific
    // history formed by prepending a phone to vec, is in the set.
    for (int32 p = 0; p <= num_phones; p++) {
      if (p != 0 && !vec.empty() && vec[0] == 0) {
        // We can't have real phones followed by zero (which in this context
        // represents the beginning of sentence).
        continue;
      }
      std::vector<int32> more_specific_vec;
      more_specific_vec.reserve(vec.size() + 1);
      more_specific_vec.push_back(p);
      more_specific_vec.insert(more_specific_vec.end(), vec.begin(), vec.end());
      // if we inserted it (i.e. if it was a new history) and it's shorter
      // than the tree's left context, we have to add it to the queue for
      // further processing.
      if (hist_set->insert(more_specific_vec).second &&
          more_specific_vec.size() < static_cast<size_t>(tree_left_context))
        hist_state_queue.push_back(more_specific_vec);
    }
  }
  // Now erase all history-states that have a shorter history-vector than
  // tree_left_context, as we already added more-specific versions of these.
  SetType::iterator iter = hist_set->begin(), end = hist_set->end();
  while (iter != end) {
    const std::vector<int32> &vec = *iter;
    if (vec.size() < static_cast<size_t>(tree_left_context))
      iter = hist_set->erase(iter);  // erase() returns iterator to the next
                                     // element.
    else
      ++iter;
  }
  
  KALDI_LOG << "Initial set of histories (pre-merging) has "
            << hist_set->size() << " elements.";
}


int32 CctcTransitionModelCreator::GetOutputIndex(
    const std::vector<int32> &hist, int32 phone) const {
  int32 context_width = ctx_dep_.ContextWidth();
  KALDI_ASSERT(hist.size() >= context_width - 1);
  if (phone == 0) {  // The blank phone -> output state is LM history state,
                     // offset by num_tree_leaves_.
    return num_tree_leaves_ + lm_hist_state_map_.GetLmHistoryState(hist);
  } else {
    std::vector<int32> phone_in_context(context_width);
    // if "hist" has unnecessary left context that needs to be truncated,
    // hist_offset will be > 0.
    int32 hist_offset = static_cast<int32>(hist.size()) - (context_width - 1);
    for (int32 i = 0; i + 1 < context_width; i++)
      phone_in_context[i] = hist[hist_offset + i];
    phone_in_context[context_width - 1] = phone;
    int32 pdf_class = 0;  // This is hard-coded; the tree should be built
                          // with a single pdf-class that should be zero.
    int32 pdf_id;
    bool ok = ctx_dep_.Compute(phone_in_context, pdf_class, &pdf_id);
    if (!ok) {
      std::ostringstream os;
      for (size_t i = 0; i < context_width; i++)
        os << phone_in_context[i] << ' ';
      KALDI_ERR << "Could not get tree leaf for the following phone-in-context: "
                << os.str();
    }
    KALDI_ASSERT(pdf_id >= 0 && pdf_id < num_tree_leaves_);
    return pdf_id;
  }
}


void CctcTransitionModelCreator::GetInitialHistoryStates() {
  SetType hist_set;
  GetInitialHistories(&hist_set);
  KALDI_ASSERT(!hist_set.empty());
  // The rest of this function will convert each of the members of hist_set into
  // history-states, and set up the initial version of the associated info.
  
  MapType hist_to_state;
  
  // First get a numbering for the initial history vectors, and
  // a vector of the vectors.
  std::vector<std::vector<int32> > hist_vec;
  hist_vec.reserve(hist_set.size());
  {
    unordered_set<std::vector<int32>, VectorHasher<int32> >::iterator
        iter = hist_set.begin(), end = hist_set.end();
    for (; iter != end; ) {
      const std::vector<int32> &vec = *iter;
      hist_to_state[vec] = hist_vec.size();
      hist_vec.push_back(vec);
      iter = hist_set.erase(iter);  // this erase function moves iter to the
                                    // next element.  We save memory by erasing
                                    // the input as we go.
    }
    hist_set.clear();
    hist_set.rehash(1);  // save memory.
  }
  KALDI_ASSERT(hist_vec.size() == hist_to_state.size());

  CreateHistoryInfo(hist_vec, hist_to_state);

  {
    // Work out the index of the initial history-state that appears
    // at the beginning of the sentence.  This is the one whose
    // vector is [ 0 0 ], i.e. zeros repeated up to the
    // the left-context of the decision tree, but at least one zero
    // if the phone LM is not a 1-gram (because this is how it
    // represents the beginning-of-sentence history).
    int32 tree_left_context = ctx_dep_.ContextWidth() - 1,
        start_state_left_context = std::max<int32>(
            tree_left_context, phone_lang_model_.NgramOrder() > 1 ? 1 : 0);
    std::vector<int32> sentence_start_hist(start_state_left_context, 0);
    MapType::iterator iter;
    if ((iter = hist_to_state.find(sentence_start_hist)) == hist_to_state.end())
      KALDI_ERR << "Cannot find history state for beginning of sentence.";
    initial_history_state_ = iter->second;
  }
}

void CctcTransitionModelCreator::CreateHistoryInfo(
    const std::vector<std::vector<int32> > &hist_vec,
    const MapType &hist_to_state) {
  int32 num_histories = hist_vec.size(),  // before merging.
      num_phones = phone_lang_model_.VocabSize(),
      tree_left_context = ctx_dep_.ContextWidth() - 1;  
  
  history_states_.resize(num_histories);

  for (int32 h = 0; h < num_histories; h++) {
    const std::vector<int32> &hist = hist_vec[h];
    HistoryState &state = history_states_[h];
    state.lm_history_state = lm_hist_state_map_.GetLmHistoryState(hist);
    state.output_index.resize(num_phones + 1);
    state.next_history_state.resize(num_phones + 1);
    state.history = hist;  // this member only needed for ease of debugging.
    KALDI_ASSERT(hist.size() >= static_cast<size_t>(tree_left_context));
    for (int32 phone = 0; phone <= num_phones; phone++)
      state.output_index[phone] = GetOutputIndex(hist, phone);
    state.next_history_state[0] = 0;  // it would transition to itself, but
    // this would prevent merging when we
    // compared this value.
    for (int32 phone = 1; phone <= num_phones; phone++) {
      std::vector<int32> next_hist(hist);
      next_hist.push_back(phone);
      // Now back off next_hist until it corresponds to a known history state.
      MapType::const_iterator iter, end = hist_to_state.end();
      while ((iter = hist_to_state.find(next_hist)) == end) {
        KALDI_ASSERT(!next_hist.empty());
        next_hist.erase(next_hist.begin());  // erase 1st element (back off)
      }
      int32 next_state_index = iter->second;
      state.next_history_state[phone] = next_state_index;
    }
  }
}

bool CctcTransitionModelCreator::MergeHistoryStatesOnePass() {
  int32 num_history_states = history_states_.size();
  std::vector<int32> old2new_history_state(num_history_states);

  int32 new_num_history_states = 0;

  // This maps from const HistoryState* to its new numbering.
  HistoryMapType hist_to_new;
  
  for (int32 h = 0; h < num_history_states; h++) {
    const HistoryState *hist_state = &(history_states_[h]);
    std::pair<const HistoryState*, int32> pair_to_insert(hist_state,
                                                         new_num_history_states);
    std::pair<HistoryMapType::iterator, bool>
        returned_pair = hist_to_new.insert(pair_to_insert);
    bool was_inserted = returned_pair.second;
    if (was_inserted) {
      old2new_history_state[h] = new_num_history_states;
      new_num_history_states++;  // we have allocated that index.
    } else {
      int32 other_h = returned_pair.first->second;
      old2new_history_state[h] = other_h;
    }
  }
  // save memory.
  hist_to_new.clear();
  hist_to_new.rehash(1);
  
  if (new_num_history_states == num_history_states) {
    // No duplicates were found, nothing was changed
    return false;
  }
  KALDI_ASSERT(new_num_history_states > 0);
  std::vector<HistoryState> new_history_states(new_num_history_states);
  // Renumber the history states.
  for (int32 h = 0; h < num_history_states; h++) {
    new_history_states[old2new_history_state[h]] = history_states_[h];
  }
  // Make sure at the "next_history_state" member variable is updated
  // to reflect the new numbering.
  for (int32 h = 0; h < new_num_history_states; h++) {
    HistoryState &hist_state = new_history_states[h];
    std::vector<int32> &next_state_vec = hist_state.next_history_state;
    // Note: we don't map element zero of the vector because that's always zero,
    // and it doesn't contain a valid history-state index.
    for (std::vector<int32>::iterator iter = next_state_vec.begin() + 1;
         iter != next_state_vec.end(); ++iter) {
      int32 old = *iter;
      KALDI_ASSERT(static_cast<size_t>(old) < old2new_history_state.size());
      *iter = old2new_history_state[old];
    }
  }
  history_states_.swap(new_history_states);
  initial_history_state_ = old2new_history_state[initial_history_state_];
  
  KALDI_LOG << "Merged " << num_history_states << " history states to "
            << new_num_history_states << ".";
  return true;  // we merged at least one state.
}
    
void CctcTransitionModelCreator::OutputToTransitionModel(
    CctcTransitionModel *trans_model) const {
  KALDI_ASSERT(!history_states_.empty());
  // first clear some stuff, just in case.
  trans_model->weights_.Resize(0, 0);
  trans_model->history_state_info_.clear();
  
  int32 num_histories = history_states_.size(),
      num_phones = phone_lang_model_.VocabSize(),
      ngram_order = phone_lang_model_.NgramOrder(),
      tree_context_width = ctx_dep_.ContextWidth(); 
  
  trans_model->num_phones_ = num_phones;
  trans_model->phone_left_context_ = std::max(ngram_order,
                                             tree_context_width) - 1;
  trans_model->num_output_indexes_ = num_output_indexes_;
  trans_model->num_non_blank_indexes_ = num_tree_leaves_;
  trans_model->initial_history_state_ = initial_history_state_;
  KALDI_ASSERT(initial_history_state_ < num_histories);
  trans_model->history_state_info_.resize(num_histories);
  for (int32 h = 0; h < num_histories; h++)
    OutputHistoryState(h, trans_model);
};
  

void CctcTransitionModelCreator::OutputHistoryState(
    int32 h, CctcTransitionModel *trans_model) const {
  int32 num_phones = phone_lang_model_.VocabSize();
  const HistoryState &src = history_states_[h];
  CctcTransitionModel::HistoryStateInfo &info =
      trans_model->history_state_info_[h];
  info.next_history_state.resize(num_phones + 1);
  info.next_history_state = src.next_history_state;
  // The next history-state after we see blank (phone 0) is h itself,
  // because blank doesn't advance the history state.
  info.next_history_state[0] = h;
  info.output_index = src.output_index;
  // while "src" has the lm_history_state stored, we want to simplify the
  // representation in "trans_model", so we directly store the vector of
  // probabilities.

  info.phone_lm_prob.Resize(num_phones + 1, kUndefined);
  info.phone_lm_prob(0) = 0.0;  // Set the prob for phone 0 (blank) to 0
                                // temporarily to help us renormalize
  int32 lm_history_state = src.lm_history_state;

  for (int32 p = 0; p <= num_phones; p++)
    info.phone_lm_prob(p) = lm_hist_state_map_.GetProb(phone_lang_model_,
                                                       lm_history_state, p);
  // language model should sum to one over its output space.
  KALDI_ASSERT(fabs(1.0 - info.phone_lm_prob.Sum()) < 1.001);
  // eos_prob is the probability of the end-of-sequence/end-of-sentence symbol.
  BaseFloat eos_prob = info.phone_lm_prob(0); 
  KALDI_ASSERT(info.phone_lm_prob(0) < 1.0);  // If EOS had all the prob mass,
                                              // removing it and renormalizing
                                              // would be a problem.
  // The following two lines of code may seem a bit confusing.  Before doing the
  // following, position 0 in the phone_lm_prob vector was the probability of
  // end-of-sequence.  After the next lines of code, position 0 in the
  // phone_lm_prob vector corresponds to the blank symbol, which we by fiat set
  // to 1.0 [this isn't of course a valid probability model, but it doesn't
  // matter, it's just something that we train the network against and as long
  // as we're consistent everything will be OK].  However, the CCTC model
  // doesn't care about the end-of-sequence probability; it's not in its output
  // space.  The reason why it doesn't make sense to model the end-of-sequence
  // is that CCTC is a conditional model, where we predict the phone sequence
  // given the acoustic sequence.  If we had a special symbol for end of
  // acoustic sequence, then we would predict the end of the phone sequence with
  // probability one when ever we saw the end of the acoustic sequence (well,
  // assuming we had no real phone to flush out.. assume the model is smart
  // enough to have done this before).  So there is no point trying to predict
  // the end of the phone sequence while we're still seeing real acoustic data.
  // So we remove the probability mass from the eos symbol and renormalize the
  // language model to sum to one.
  info.phone_lm_prob.Scale(1.0 / (1.0 - eos_prob));
  info.phone_lm_prob(0) = 1.0;
}

}  // namespace ctc
}  // namespace kaldi
