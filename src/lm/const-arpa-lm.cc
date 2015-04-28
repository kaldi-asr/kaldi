// lm/const-arpa-lm.cc

// Copyright 2014  Guoguo Chen

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

#include <sstream>

#include "lm/const-arpa-lm.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"

namespace kaldi {

// Auxiliary struct for converting ConstArpaLm format langugae model to Arpa
// format.
struct ArpaLine {
  std::vector<int32> words; // Sequence of words to be printed.
  float logprob;            // Logprob corresponds to word sequence.
  float backoff_logprob;    // Backoff_logprob corresponds to word sequence.
  // Comparison function for sorting.
  bool operator < (const ArpaLine &other) const {
    if (words.size() < other.words.size()) {
      return true;
    } else if (words.size() > other.words.size()) {
      return false;
    } else {
      return words < other.words;
    }
  }
};

// Auxiliary class to build ConstArpaLm. We first use this class to figure out
// the relative address of different LmStates, and then put everything into one
// block in memory.
class LmState {
 public:
  union ChildType {
    // If child is not the final order, we keep the pointer to its LmState.
    LmState* state;

    // If child is the final order, we only keep the log probability for it.
    float prob;
  };

  struct ChildrenVectorLessThan {
    bool operator()(
        const std::pair<int32, union ChildType>& lhs,
        const std::pair<int32, union ChildType>& rhs ) const {
      return lhs.first < rhs.first;
    }
  };

  LmState(const bool is_unigram, const bool is_child_final_order,
          const float logprob, const float backoff_logprob) :
      is_unigram_(is_unigram), is_child_final_order_(is_child_final_order),
      logprob_(logprob), backoff_logprob_(backoff_logprob) {}

  void SetMyAddress(const int64 address) {
    my_address_ = address;
  }

  void AddChild(const int32 word, LmState* child_state) {
    KALDI_ASSERT(!is_child_final_order_);
    ChildType child;
    child.state = child_state;
    children_.push_back(std::make_pair(word, child));
  }

  void AddChild(const int32 word, const float child_prob) {
    KALDI_ASSERT(is_child_final_order_);
    ChildType child;
    child.prob = child_prob;
    children_.push_back(std::make_pair(word, child));
  }

  int64 MyAddress() const {
    return my_address_;
  }

  bool IsUnigram() const {
    return is_unigram_;
  }

  bool IsChildFinalOrder() const {
    return is_child_final_order_;
  }

  float Logprob() const {
    return logprob_;
  }

  float BackoffLogprob() const {
    return backoff_logprob_;
  }

  int32 NumChildren() const {
    return children_.size();
  }

  std::pair<int32, union ChildType> GetChild(const int32 index) {
    KALDI_ASSERT(index < children_.size());
    KALDI_ASSERT(index >= 0);
    return children_[index];
  }

  void SortChildren() {
    std::sort(children_.begin(), children_.end(), ChildrenVectorLessThan());
  }

  // Checks if the current LmState is a leaf.
  bool IsLeaf() const {
    return (backoff_logprob_ == 0.0 && children_.empty());
  }

  // Computes the size of the memory that the current LmState would take in
  // <lm_states> array. It's the number of 4-byte chunks.
  int32 MemSize() const {
    if (IsLeaf() && !is_unigram_) {
      // We don't create an entry in this case; the logprob will be stored in
      // the same int32 that we would normally store the pointer in.
      return 0;
    } else {
      // We store the following information:
      // logprob, backoff_logprob, children.size() and children data.
      return (3 + 2 * children_.size());
    }
  }

 private:
  // Unigram states will have LmStates even if they are leaves, therefore we
  // need to note when this is a unigram or not.
  bool is_unigram_;

  // If the current LmState has an order of (final_order - 1), then its child
  // must be the final order. We only keep the log probability for its child.
  bool is_child_final_order_;

  // When we compute the addresses of the LmStates as offsets into <lm_states_>
  // pointer, and put the offsets here. Note that this is just offset, not
  // actual pointer.
  int64 my_address_;

  // Language model log probability of the current sequence. For example, if
  // this state is "A B", then it would be the logprob of "A -> B".
  float logprob_;

  // Language model backoff log probability of the current sequence, e.g., state
  // "A B -> X" backing off to "B -> X".
  float backoff_logprob_;

  // List of children. 
  std::vector<std::pair<int32, union ChildType> > children_;
};

// Class to build ConstArpaLm from Arpa format language model. It relies on the
// auxiliary class LmState above.
class ConstArpaLmBuilder {
 public:
  ConstArpaLmBuilder(
      const bool natural_base, const int32 bos_symbol,
      const int32 eos_symbol, const int32 unk_symbol) :
      natural_base_(natural_base), bos_symbol_(bos_symbol),
      eos_symbol_(eos_symbol), unk_symbol_(unk_symbol) {
    ngram_order_ = 0;
    num_words_ = 0;
    overflow_buffer_size_ = 0;
    lm_states_size_ = 0;
    max_address_offset_ = pow(2, 30) - 1;
    is_built_ = false;
    lm_states_ = NULL;
    unigram_states_ = NULL;
    overflow_buffer_ = NULL; 
  }

  ~ConstArpaLmBuilder() {
    unordered_map<std::vector<int32>,
                  LmState*, VectorHasher<int32> >::iterator iter;
    for (iter = seq_to_state_.begin(); iter != seq_to_state_.end(); ++iter) {
      delete iter->second;
    }
    if (is_built_) {
      delete[] lm_states_;
      delete[] unigram_states_;
      delete[] overflow_buffer_;
    }
  }

  // Reads in the Arpa format language model, parses it and creates LmStates.
  void Read(std::istream &is, bool binary);

  // Writes ConstArpaLm.
  void Write(std::ostream &os, bool binary) const;

  // Builds ConstArpaLm.
  void Build();

  void SetMaxAddressOffset(const int32 max_address_offset) {
    KALDI_WARN << "You are changing <max_address_offset_>; the default should "
        << "not be changed unless you are in testing mode.";
    max_address_offset_ = max_address_offset;
  }

 private:
  struct WordsAndLmStatePairLessThan {
    bool operator()(
        const std::pair<std::vector<int32>*, LmState*>& lhs,
        const std::pair<std::vector<int32>*, LmState*>& rhs ) const {
      return *(lhs.first) < *(rhs.first);
    }
  };

 private:
  // If true, use natural base e for log-prob, otherwise use base 10. The
  // default base in Arpa format language model is base 10.
  bool natural_base_;

  // Indicating if ConstArpaLm has been built or not.
  bool is_built_;

  // Maximum relative address for the child. We put it here just for testing.
  // The default value is 30-bits and should not be changed except for testing.
  int32 max_address_offset_;

  // Integer corresponds to <s>.
  int32 bos_symbol_;

  // Integer corresponds to </s>.
  int32 eos_symbol_;

  // Integer corresponds to unknown-word. -1 if no unknown-word symbol is
  // provided.
  int32 unk_symbol_;

  // N-gram order of language model. This can be figured out from "/data/"
  // section in Arpa format language model.
  int32 ngram_order_;

  // Index of largest word-id plus one. It defines the end of <unigram_states_>
  // array.
  int32 num_words_;

  // Number of entries in the overflow buffer for pointers that couldn't be
  // represented as a 30-bit relative index.
  int32 overflow_buffer_size_;

  // Size of the <lm_states_> array, which will be needed by I/O.
  int32 lm_states_size_;

  // Memory blcok for storing LmStates.
  int32* lm_states_;

  // Memory block for storing pointers of unigram LmStates.
  int32** unigram_states_;

  // Memory block for storing pointers of the LmStates that have large relative
  // address to their parents.
  int32** overflow_buffer_;

  // Hash table from word sequences to LmStates.
  unordered_map<std::vector<int32>,
                LmState*, VectorHasher<int32> > seq_to_state_;
};

// Reads in the Arpa format language model, parses it and puts the word sequence
// into the corresponding LmState in <seq_to_state_>.
void ConstArpaLmBuilder::Read(std::istream &is, bool binary) {
  if (binary) {
    KALDI_ERR << "binary-mode reading is not implemented for "
        << "ConstArpaLmBuilder.";
  }

  std::string line;

  // Number of n-grams from "\data\" section. Those numbers should match the
  // actual number of n-grams from "\N-grams:" sections.
  // Note that when we convert the words in the Arpa format language model into
  // integers, we remove lines with OOV words. We also modify the n-gram counts
  // in "\data\" correspondingly.
  std::vector<int32> num_ngrams;

  // Processes "\data\" section.
  bool keyword_found = false;
  while (getline(is, line) && !is.eof()) {
    // The section keywords starts with backslash. We terminate the while loop
    // if a new section is found.
    if (!line.empty() && line[0] == '\\') {
      if (line.find("-grams:") != std::string::npos) break;
      if (line.find("\\end\\") != std::string::npos) break;
    }

    std::vector<std::string> col;
    SplitStringToVector(line, " \t", true, &col);

    // Looks for keyword "\data\".
    if (!keyword_found && col.size() == 1 && col[0] == "\\data\\") {
      KALDI_LOG << "Reading \"\\data\\\" section.";
      keyword_found = true;
    }

    // Enters "\data\" section, and looks for patterns like"ngram 1=1000", which
    // means there are 1000 unigrams.
    if (keyword_found && col.size() == 2 && col[0] == "ngram") {
      std::vector<std::string> sub_col;
      SplitStringToVector(col[1], "=", true, &sub_col);
      if (sub_col.size() == 2) {
        int32 order, ngram_count;
        if (!ConvertStringToInteger(sub_col[0], &order)) {
          KALDI_ERR << "bad line: " << line << "; fail to convert "
              << sub_col[0] << " to integer.";
        }
        if (!ConvertStringToInteger(sub_col[1], &ngram_count)) {
          KALDI_ERR << "bad line: " << line << "; fail to convert "
              << sub_col[1] << " to integer.";
        }
        if (num_ngrams.size() <= order) {
          num_ngrams.resize(order + 1);
        }
        num_ngrams[order] = ngram_count;
      }
    }
  }
  KALDI_ASSERT(num_ngrams.size() > 0);
  ngram_order_ = num_ngrams.size() - 1;

  // Processes "\N-grams:" section.
  int32 max_word_id = 0;
  for (int32 cur_order = 1; cur_order < num_ngrams.size(); ++cur_order) {
    // Skips n-grams with zero count.
    if (num_ngrams[cur_order] == 0) continue;

    keyword_found = false;
    int32 ngram_count = 0;
    std::ostringstream keyword;
    keyword << "\\" << cur_order << "-grams:";
    // We use "do ... while" loop since one line has already been read.
    do {
      // The section keywords starts with backslash. We terminate the while loop
      // if a new section is found.
      if (!line.empty() && line[0] == '\\') {
        if (line.find("-grams:") != std::string::npos && keyword_found) break;
        if (line.find("\\end\\") != std::string::npos) break;
      }

      std::vector<std::string> col;
      SplitStringToVector(line, " \t", true, &col);

      // Looks for keyword "\N-gram:" if the keyword has not been located.
      if (!keyword_found && col.size() == 1 && col[0] == keyword.str()) {
        KALDI_LOG << "Reading \"" << keyword.str() << "\" section.";
        ngram_count = 0;
        keyword_found = true;
        continue;
      }

      // Enters "\N-grams:" section if the keyword has been located.
      if (keyword_found && col.size() > 0) {
        KALDI_ASSERT(col.size() >= 1 + cur_order);
        KALDI_ASSERT(col.size() <= 2 + cur_order);// backoff_logprob could be 0.
        if (cur_order == ngram_order_ && col.size() == 2 + cur_order) {
          KALDI_ERR << "Backoff probability detected for final-order entry \""
              << line << "\".";
        }
        ngram_count++;

        // If backoff_logprob is 0, it will not appear in Arpa format language
        // model. We put it back so the processing afterwards will be easier.
        if (col.size() == 1 + cur_order) {
          col.push_back("0");
        }

        // Creates LmState for the current word sequence.
        bool is_unigram = (cur_order == 1) ? true : false;
        float logprob;
        float backoff_logprob;
        KALDI_ASSERT(ConvertStringToReal(col[0], &logprob));
        KALDI_ASSERT(ConvertStringToReal(col[1 + cur_order], &backoff_logprob));
        if (natural_base_) {
          logprob *= log(10);
          backoff_logprob *= log(10);
        }
       
        // If <ngram_order_> is larger than 1, then we do not create LmState for
        // the final order entry. We only keep the log probability for it.
        LmState *lm_state = NULL;
        if (cur_order != ngram_order_ || ngram_order_ == 1) {
          lm_state = new LmState(is_unigram,
                                 (cur_order == ngram_order_ - 1),
                                 logprob, backoff_logprob);
        }

        // Figures out the sequence of words.
        std::vector<int32> seq(cur_order, 0);
        for (int32 index = 0; index < cur_order; ++index) {
          int32 word;
          if (!ConvertStringToInteger(col[1 + index], &word)) {
            KALDI_ERR << "bad line: " << line << "; fail to convert "
                << col[1 + index] << " to integer.";
          }
          seq[index] = word;
        }

        // If <ngram_order_> is larger than 1, then we do not insert LmState to
        // <seq_to_state_>.
        if (cur_order != ngram_order_ || ngram_order_ == 1) {
          KALDI_ASSERT(lm_state != NULL);
          KALDI_ASSERT(seq_to_state_.find(seq) == seq_to_state_.end());
          seq_to_state_[seq] = lm_state;
        }

        // If n-gram order is larger than 1, we have to add possible child to
        // existing LmStates. We have the following two assumptions:
        // 1. N-grams are processed from small order to larger ones, i.e., from
        //    1, 2, ... to the highest order.
        // 2. If a n-gram exists in the Arpa format language model, then the
        //    "history" n-gram also exists. For example, if "A B C" is a valid
        //    n-gram, then "A B" is also a valid n-gram.
        if (cur_order > 1) {
          std::vector<int32> hist(seq.begin(), seq.begin() + cur_order - 1);
          int32 word = seq[seq.size() - 1];
          unordered_map<std::vector<int32>,
                        LmState*, VectorHasher<int32> >::iterator hist_iter;
          hist_iter = seq_to_state_.find(hist);
          KALDI_ASSERT(hist_iter != seq_to_state_.end());
          if (cur_order != ngram_order_ || ngram_order_ == 1) {
            KALDI_ASSERT(lm_state != NULL);
            KALDI_ASSERT(!hist_iter->second->IsChildFinalOrder());
            hist_iter->second->AddChild(word, lm_state);
          } else {
            KALDI_ASSERT(lm_state == NULL);
            KALDI_ASSERT(hist_iter->second->IsChildFinalOrder());
            hist_iter->second->AddChild(word, logprob);
          }
        } else {
          // Figures out <max_word_id>.
          KALDI_ASSERT(seq.size() == 1);
          if (seq[0] > max_word_id) {
            max_word_id = seq[0];
          }
        }
      }
    } while (getline(is, line) && !is.eof());
    if (ngram_count > num_ngrams[cur_order] ||
        (ngram_count == 0 && num_ngrams[cur_order] != 0)) {
      KALDI_ERR << "Header said there would be " << num_ngrams[cur_order]
                << " n-grams of order " << cur_order << ", but we saw "
                << ngram_count;
    }
  }

  // <num_words_> is <max_word_id> plus 1.
  num_words_ = max_word_id + 1;
}

// ConstArpaLm can be built in the following steps, assuming we have already
// created LmStates <seq_to_state_>:
// 1. Sort LmStates lexicographically.
//    This enables us to compute relative address. When we say lexicographic, we
//    treat the word-ids as letters. After sorting, the LmStates are in the
//    following order:
//    ...
//    A B
//    A B A
//    A B B
//    A B C
//    ...
//    where each line represents a LmState.
// 2. Update <my_address> in LmState, which is relative to the first element in
//    <sorted_vec>.
// 3. Put the following structure into the memory block
//    struct LmState {
//      float logprob;
//      float backoff_logprob;
//      int32 num_children;
//      std::pair<int32, int32> [] children;
//    }
//
//    At the same time, we will also create two special buffers:
//    <unigram_states_>
//    <overflow_buffer_>
void ConstArpaLmBuilder::Build() {
  // STEP 1: sorting LmStates lexicographically.
  // Vector for holding the sorted LmStates.
  std::vector<std::pair<std::vector<int32>*, LmState*> > sorted_vec;
  unordered_map<std::vector<int32>,
                LmState*, VectorHasher<int32> >::iterator iter;
  for (iter = seq_to_state_.begin(); iter != seq_to_state_.end(); ++iter) {
    if (iter->second->MemSize() > 0) {
      sorted_vec.push_back(
          std::make_pair(const_cast<std::vector<int32>*>(&(iter->first)),
                         iter->second));
    }
  }

  std::sort(sorted_vec.begin(), sorted_vec.end(),
            WordsAndLmStatePairLessThan());

  // STEP 2: updating <my_address> in LmState.
  for (int32 i = 0; i < sorted_vec.size(); ++i) {
    lm_states_size_ += sorted_vec[i].second->MemSize();
    if (i == 0) {
      sorted_vec[i].second->SetMyAddress(0);
    } else {
      sorted_vec[i].second->SetMyAddress(sorted_vec[i - 1].second->MyAddress()
          + sorted_vec[i - 1].second->MemSize());
    }
  }

  // STEP 3: creating memory block to store LmStates.
  // Reserves a memory block for LmStates.
  int32 lm_states_index = 0;
  try {
    lm_states_ = new int32[lm_states_size_];
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
  }

  // Puts data into memory block.
  unigram_states_ = new int32*[num_words_];
  std::vector<int32*> overflow_buffer_vec;
  for (int32 i = 0; i < num_words_; ++i) {
    unigram_states_[i] = NULL;
  }
  for (int32 i = 0; i < sorted_vec.size(); ++i) {
    // Current address.
    int32* parent_address = lm_states_ + lm_states_index;

    // Adds logprob.
    float logprob = sorted_vec[i].second->Logprob();
    lm_states_[lm_states_index++] = *reinterpret_cast<int32*>(&logprob);

    // Adds backoff_logprob.
    float backoff_logprob = sorted_vec[i].second->BackoffLogprob();
    lm_states_[lm_states_index++] = *reinterpret_cast<int32*>(&backoff_logprob);

    // Adds num_children.
    lm_states_[lm_states_index++] = sorted_vec[i].second->NumChildren();

    // Adds children, there are 3 cases:
    // 1. Child is a leaf and not unigram
    // 2. Child is not a leaf or is unigram
    //    2.1 Relative address can be represented by 30 bits
    //    2.2 Relative address cannot be represented by 30 bits
    sorted_vec[i].second->SortChildren();
    for (int32 j = 0; j < sorted_vec[i].second->NumChildren(); ++j) {
      int32 child_info;
      if (sorted_vec[i].second->IsChildFinalOrder() ||
          sorted_vec[i].second->GetChild(j).second.state->MemSize() == 0) {
        // Child is a leaf and not unigram. In this case we will not create an
        // entry in <lm_states_>; instead, we put the logprob in the place where
        // we normally store the poitner.
        float child_logprob;
        if (sorted_vec[i].second->IsChildFinalOrder()) {
          child_logprob = sorted_vec[i].second->GetChild(j).second.prob;
        } else {
          child_logprob =
              sorted_vec[i].second->GetChild(j).second.state->Logprob();
        }
        child_info = *reinterpret_cast<int32*>(&child_logprob);
        child_info &= ~1;   // Sets the last bit to 0 so <child_info> is even.
      } else {
        // Child is not a leaf or is unigram.
        int64 offset =
            sorted_vec[i].second->GetChild(j).second.state->MyAddress()
            - sorted_vec[i].second->MyAddress();
        KALDI_ASSERT(offset > 0);
        if (offset <= max_address_offset_) {
          // Relative address can be represented by 30 bits.
          child_info = offset * 2;
          child_info |= 1;
        } else {
          // Relative address cannot be represented by 30 bits, we have to put
          // the child address into <overflow_buffer_>.
          int32* abs_address = parent_address + offset;
          overflow_buffer_vec.push_back(abs_address);
          int32 overflow_buffer_index = overflow_buffer_vec.size() - 1;
          child_info = overflow_buffer_index * 2;
          child_info |= 1;
          child_info *= -1;
        }
      }
      // Child word.
      lm_states_[lm_states_index++] = sorted_vec[i].second->GetChild(j).first;
      // Child info.
      lm_states_[lm_states_index++] = child_info;
    }

    // If the current state corresponds to an unigram, then create a separate
    // loop up table to improve efficiency, since those will be looked up pretty
    // frequently.
    if (sorted_vec[i].second->IsUnigram()) {
      KALDI_ASSERT(sorted_vec[i].first->size() == 1);
      unigram_states_[(*sorted_vec[i].first)[0]] = parent_address;
    }
  }
  KALDI_ASSERT(lm_states_size_ == lm_states_index);

  // Move <overflow_buffer_> from vector holder to array.
  overflow_buffer_size_ = overflow_buffer_vec.size();
  overflow_buffer_ = new int32*[overflow_buffer_size_];
  for (int32 i = 0; i < overflow_buffer_size_; ++i) {
    overflow_buffer_[i] = overflow_buffer_vec[i];
  }

  is_built_ = true;
}

void ConstArpaLmBuilder::Write(std::ostream &os, bool binary) const {
  if (!binary) {
    KALDI_ERR << "text-mode writing is not implemented for ConstArpaLmBuilder.";
  }
  KALDI_ASSERT(is_built_);

  // Creates ConstArpaLm.
  ConstArpaLm const_arpa_lm(bos_symbol_, eos_symbol_, unk_symbol_, ngram_order_,
                            num_words_, overflow_buffer_size_, lm_states_size_,
                            unigram_states_, overflow_buffer_, lm_states_);
  const_arpa_lm.Write(os, binary);
}

void ConstArpaLm::Write(std::ostream &os, bool binary) const {
  KALDI_ASSERT(initialized_);
  if (!binary) {
    KALDI_ERR << "text-mode writing is not implemented for ConstArpaLm.";
  }

  // Misc info.
  WriteBasicType(os, binary, bos_symbol_);
  WriteBasicType(os, binary, eos_symbol_);
  WriteBasicType(os, binary, unk_symbol_);
  WriteBasicType(os, binary, ngram_order_);

  // LmStates section.
  WriteBasicType(os, binary, lm_states_size_);
  for (int32 i = 0; i < lm_states_size_; ++i) {
    WriteBasicType(os, binary, lm_states_[i]);
  }

  // Unigram section. We write memory offset to disk instead of the absolute
  // pointers.
  WriteBasicType(os, binary, num_words_);
  for (int32 i = 0; i < num_words_; ++i) {
    // The relative address here is a little bit tricky:
    // 1. If the original address is NULL, then we set the relative address to
    //    zero.
    // 2. If the original address is not NULL, we set it to the following:
    //      unigram_states_[i] - lm_states_ + 1
    //    we plus 1 to ensure that the above value is positive.
    int64 tmp_address = (unigram_states_[i] == NULL) ? 0 :
        unigram_states_[i] - lm_states_ + 1;
    WriteBasicType(os, binary, tmp_address);
  }

  // Overflow section. We write memory offset to disk instead of the absolute
  // pointers.
  WriteBasicType(os, binary, overflow_buffer_size_);
  for (int32 i = 0; i < overflow_buffer_size_; ++i) {
    // The relative address here is a little bit tricky:
    // 1. If the original address is NULL, then we set the relative address to
    //    zero.
    // 2. If the original address is not NULL, we set it to the following:
    //      overflow_buffer_[i] - lm_states_ + 1
    //    we plus 1 to ensure that the above value is positive.
    int64 tmp_address = (overflow_buffer_[i] == NULL) ? 0 :
        overflow_buffer_[i] - lm_states_ + 1;
    WriteBasicType(os, binary, tmp_address);
  }
}

void ConstArpaLm::Read(std::istream &is, bool binary) {
  KALDI_ASSERT(!initialized_);
  if (!binary) {
    KALDI_ERR << "text-mode reading is not implemented for ConstArpaLm.";
  }

  // Misc info.
  ReadBasicType(is, binary, &bos_symbol_);
  ReadBasicType(is, binary, &eos_symbol_);
  ReadBasicType(is, binary, &unk_symbol_);
  ReadBasicType(is, binary, &ngram_order_);

  // LmStates section.
  ReadBasicType(is, binary, &lm_states_size_);
  lm_states_ = new int32[lm_states_size_];
  for (int32 i = 0; i < lm_states_size_; ++i) {
    ReadBasicType(is, binary, &lm_states_[i]);
  }

  // Unigram section. We write memory offset to disk instead of the absolute
  // pointers.
  ReadBasicType(is, binary, &num_words_);
  unigram_states_ = new int32*[num_words_];
  for (int32 i = 0; i < num_words_; ++i) {
    int64 tmp_address;
    ReadBasicType(is, binary, &tmp_address);
    // Check out how we compute the relative address in ConstArpaLm::Write().
    unigram_states_[i] =
        (tmp_address == 0) ? NULL : lm_states_ + tmp_address - 1;
  }

  // Overflow section. We write memory offset to disk instead of the absolute
  // pointers.
  ReadBasicType(is, binary, &overflow_buffer_size_);
  overflow_buffer_ = new int32*[overflow_buffer_size_];
  for (int32 i = 0; i < overflow_buffer_size_; ++i) {
    int64 tmp_address;
    ReadBasicType(is, binary, &tmp_address);
    // Check out how we compute the relative address in ConstArpaLm::Write().
    overflow_buffer_[i] =
        (tmp_address == 0) ? NULL : lm_states_ + tmp_address - 1;
  }
  KALDI_ASSERT(ngram_order_ > 0);
  KALDI_ASSERT(bos_symbol_ < num_words_ && bos_symbol_ > 0);
  KALDI_ASSERT(eos_symbol_ < num_words_ && eos_symbol_ > 0);
  KALDI_ASSERT(unk_symbol_ < num_words_ &&
               (unk_symbol_ > 0 || unk_symbol_ == -1));
  lm_states_end_ = lm_states_ + lm_states_size_ - 1;
  memory_assigned_ = true;
  initialized_ = true;;
}

bool ConstArpaLm::HistoryStateExists(const std::vector<int32>& hist) const {
  // We do not create LmState for empty word sequence, but technically it is the
  // history state of all unigrams.
  if (hist.size() == 0) {
    return true;
  }

  // Tries to locate the LmState of the given word sequence.
  int32* lm_state = GetLmState(hist);
  if (lm_state == NULL) {
    // <lm_state> does not exist means <hist> has no child.
    return false;
  } else {
    // Note that we always create LmState for unigrams, so even if <lm_state> is
    // not NULL, we still have to check if it has child.
    KALDI_ASSERT(lm_state >= lm_states_);
    KALDI_ASSERT(lm_state + 2 <= lm_states_end_);
    // <lm_state + 2> points to <num_children>.
    if (*(lm_state + 2) > 0) {
      return true;
    } else {
      return false;
    }
  }
  return true;
}

float ConstArpaLm::GetNgramLogprob(const int32 word,
                                   const std::vector<int32>& hist) const {
  KALDI_ASSERT(initialized_);

  // If the history size plus one is larger than <ngram_order_>, remove the old
  // words.
  std::vector<int32> mapped_hist(hist);
  while (mapped_hist.size() >= ngram_order_) {
    mapped_hist.erase(mapped_hist.begin(), mapped_hist.begin() + 1);
  }
  KALDI_ASSERT(mapped_hist.size() + 1 <= ngram_order_);

  // TODO(guoguo): check with Dan if this is reasonable.
  // Maps possible out-of-vocabulary words to <unk>. If a word does not have a
  // corresponding LmState, we treat it as <unk>. We map it to <unk> if <unk> is
  // specified.
  int32 mapped_word = word;
  if (unk_symbol_ != -1) {
    KALDI_ASSERT(mapped_word >= 0);
    if (mapped_word >= num_words_ || unigram_states_[mapped_word] == NULL) {
      mapped_word = unk_symbol_;
    }
    for (int32 i = 0; i < mapped_hist.size(); ++i) {
      KALDI_ASSERT(mapped_hist[i] >= 0);
      if (mapped_hist[i] >= num_words_ ||
          unigram_states_[mapped_hist[i]] == NULL) {
        mapped_hist[i] = unk_symbol_;
      }
    }
  }

  // Loops up n-gram probability.
  return GetNgramLogprobRecurse(mapped_word, mapped_hist);
}

float ConstArpaLm::GetNgramLogprobRecurse(
    const int32 word, const std::vector<int32>& hist) const {
  KALDI_ASSERT(initialized_);
  KALDI_ASSERT(hist.size() + 1 <= ngram_order_);

  // Unigram case.
  if (hist.size() == 0) {
    if (word >= num_words_ || unigram_states_[word] == NULL) {
      // If <unk> is defined, then the word sequence should have already been
      // mapped to <unk> is necessary; this is for the case where <unk> is not
      // defined.
      return std::numeric_limits<float>::min();
    } else {
      return *reinterpret_cast<float*>(unigram_states_[word]);
    }
  }

  // High n-gram orders.
  float logprob = 0.0;
  float backoff_logprob = 0.0;
  int32* state;
  if ((state = GetLmState(hist)) != NULL) {
    int32 child_info;
    int32* child_lm_state = NULL;
    if (GetChildInfo(word, state, &child_info)) {
      DecodeChildInfo(child_info, state, &child_lm_state, &logprob);
      return logprob;
    } else {
      backoff_logprob = *reinterpret_cast<float*>(state + 1);
    }
  }
  std::vector<int32> new_hist(hist);
  new_hist.erase(new_hist.begin(), new_hist.begin() + 1);
  return backoff_logprob + GetNgramLogprobRecurse(word, new_hist);
}

int32* ConstArpaLm::GetLmState(const std::vector<int32>& seq) const {
  KALDI_ASSERT(initialized_);

  // No LmState exists for empty word sequence.
  if (seq.size() == 0) return NULL;

  // If <unk> is defined, then the word sequence should have already been mapped
  // to <unk> is necessary; this is for the case where <unk> is not defined.
  if (seq[0] >= num_words_ || unigram_states_[seq[0]] == NULL) return NULL;
  int32* parent = unigram_states_[seq[0]];

  int32 child_info;
  int32* child_lm_state = NULL;
  float logprob;
  for (int32 i = 1; i < seq.size(); ++i) {
    if (!GetChildInfo(seq[i], parent, &child_info)) {
      return NULL;
    }
    DecodeChildInfo(child_info, parent, &child_lm_state, &logprob);
    if (child_lm_state == NULL) {
      return NULL;
    } else {
      parent = child_lm_state;
    }
  }
  return parent;
}

bool ConstArpaLm::GetChildInfo(const int32 word,
                               int32* parent, int32* child_info) const {
  KALDI_ASSERT(initialized_);

  KALDI_ASSERT(parent != NULL);
  KALDI_ASSERT(parent >= lm_states_);
  KALDI_ASSERT(child_info != NULL);

  KALDI_ASSERT(parent + 2 <= lm_states_end_);
  int32 num_children = *(parent + 2);
  KALDI_ASSERT(parent + 2 + 2 * num_children <= lm_states_end_);

  if (num_children == 0) return false;

  // A binary search into the children memory block.
  int32 start_index = 1;
  int32 end_index = num_children;
  while (start_index <= end_index) {
    int32 mid_index = round((start_index + end_index) / 2);
    int32 mid_word = *(parent + 1 + 2 * mid_index);
    if (mid_word == word) {
      *child_info = *(parent + 2 + 2 * mid_index);
      return true;
    } else if (mid_word < word) {
      start_index = mid_index + 1;
    } else {
      end_index = mid_index - 1;
    }
  }

  return false;
}

void ConstArpaLm::DecodeChildInfo(const int32 child_info,
                                  int32* parent,
                                  int32** child_lm_state,
                                  float* logprob) const {
  KALDI_ASSERT(initialized_);

  KALDI_ASSERT(logprob != NULL);
  if (child_info % 2 == 0) {
    // Child is a leaf, only returns the log probability.
    *child_lm_state = NULL;
    int32 tmp_child_info = child_info;
    *logprob = *reinterpret_cast<float*>(&tmp_child_info);
  } else {
    int32 child_offset = child_info / 2;
    if (child_offset > 0) {
      *child_lm_state = parent + child_offset;
      *logprob = *reinterpret_cast<float*>(*child_lm_state);
    } else {
      KALDI_ASSERT(-child_offset < overflow_buffer_size_);
      *child_lm_state = overflow_buffer_[-child_offset];
      *logprob = *reinterpret_cast<float*>(*child_lm_state);
    }
    KALDI_ASSERT(*child_lm_state >= lm_states_);
    KALDI_ASSERT(*child_lm_state <= lm_states_end_);
  }
}

void ConstArpaLm::WriteArpaRecurse(int32* lm_state,
                                   const std::vector<int32>& seq,
                                   std::vector<ArpaLine> *output) const {
  if (lm_state == NULL) return;

  KALDI_ASSERT(lm_state >= lm_states_);
  KALDI_ASSERT(lm_state + 2 <= lm_states_end_);

  // Inserts the current LmState to <output>.
  ArpaLine arpa_line;
  arpa_line.words = seq;
  arpa_line.logprob = *reinterpret_cast<float*>(lm_state);
  arpa_line.backoff_logprob = *reinterpret_cast<float*>(lm_state + 1);
  output->push_back(arpa_line);

  // Scans for possible children, and recursively adds child to <output>.
  int32 num_children = *(lm_state + 2);
  KALDI_ASSERT(lm_state + 2 + 2 * num_children <= lm_states_end_);
  for (int32 i = 0; i < num_children; ++i) {
    std::vector<int32> new_seq(seq);
    new_seq.push_back(*(lm_state + 3 + 2 * i));
    int32 child_info = *(lm_state + 4 + 2 * i);
    float logprob;
    int32* child_lm_state = NULL;
    DecodeChildInfo(child_info, lm_state, &child_lm_state, &logprob);

    if (child_lm_state == NULL) {
      // Leaf case.
      ArpaLine child_arpa_line;
      child_arpa_line.words = new_seq;
      child_arpa_line.logprob = logprob;
      child_arpa_line.backoff_logprob = 0.0;
      output->push_back(child_arpa_line);
    } else {
      WriteArpaRecurse(child_lm_state, new_seq, output);
    }
  }
}

void ConstArpaLm::WriteArpa(std::ostream &os) const {
  KALDI_ASSERT(initialized_);

  std::vector<ArpaLine> tmp_output;
  for (int32 i = 0; i < num_words_; ++i) {
    if (unigram_states_[i] != NULL) {
      std::vector<int32> seq(1, i);
      WriteArpaRecurse(unigram_states_[i], seq, &tmp_output);
    }
  }

  // Sorts ArpaLines and collects head information.
  std::sort(tmp_output.begin(), tmp_output.end());
  std::vector<int32> ngram_count(1, 0);
  for (int32 i = 0; i < tmp_output.size(); ++i) {
    if (tmp_output[i].words.size() >= ngram_count.size()) {
      ngram_count.resize(tmp_output[i].words.size() + 1);
      ngram_count[tmp_output[i].words.size()] = 1;
    } else {
      ngram_count[tmp_output[i].words.size()] += 1;
    }
  }

  // Writes the header.
  os << std::endl;
  os << "\\data\\" << std::endl;
  for (int32 i = 1; i < ngram_count.size(); ++i) {
    os << "ngram " << i << "=" << ngram_count[i] << std::endl;
  }

  // Writes n-grams.
  int32 current_order = 0;
  for (int32 i = 0; i < tmp_output.size(); ++i) {
    // Beginning of a n-gram section. 
    if (tmp_output[i].words.size() != current_order) {
      current_order = tmp_output[i].words.size();
      os << std::endl;
      os << "\\" << current_order << "-grams:" << std::endl;
    }

    // Writes logprob.
    os << tmp_output[i].logprob << '\t';

    // Writes word sequence.
    for (int32 j = 0; j < tmp_output[i].words.size(); ++j) {
      os << tmp_output[i].words[j];
      if (j != tmp_output[i].words.size() - 1) {
        os << " ";
      }
    }

    // Writes backoff_logprob if it is not zero.
    if (tmp_output[i].backoff_logprob != 0.0) {
      os << '\t' << tmp_output[i].backoff_logprob;
    }
    os << std::endl;
  }

  os << std::endl << "\\end\\" << std::endl;
}

ConstArpaLmDeterministicFst::ConstArpaLmDeterministicFst(
    const ConstArpaLm& lm) : lm_(lm) {
  // Creates a history state for <s>.
  std::vector<Label> bos_state(1, lm_.BosSymbol());
  state_to_wseq_.push_back(bos_state);
  wseq_to_state_[bos_state] = 0;
  start_state_ = 0;
}

fst::StdArc::Weight ConstArpaLmDeterministicFst::Final(StateId s) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());
  const std::vector<Label>& wseq = state_to_wseq_[s];
  float logprob = lm_.GetNgramLogprob(lm_.EosSymbol(), wseq);
  return Weight(-logprob);
}

bool ConstArpaLmDeterministicFst::GetArc(StateId s,
                                         Label ilabel, fst::StdArc *oarc) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());
  std::vector<Label> wseq = state_to_wseq_[s];

  float logprob = lm_.GetNgramLogprob(ilabel, wseq);
  if (logprob == std::numeric_limits<float>::min()) {
    return false;
  }

  // Locates the next state in ConstArpaLm. Note that OOV and backoff have been
  // taken care of in ConstArpaLm.
  wseq.push_back(ilabel);
  while (wseq.size() >= lm_.NgramOrder()) {
    // History state has at most lm_.NgramOrder() -1 words in the state.
    wseq.erase(wseq.begin(), wseq.begin() + 1);
  }
  while (!lm_.HistoryStateExists(wseq)) {
    KALDI_ASSERT(wseq.size() > 0);
    wseq.erase(wseq.begin(), wseq.begin() + 1);
  }

  std::pair<const std::vector<Label>, StateId> wseq_state_pair(
      wseq, static_cast<Label>(state_to_wseq_.size()));

  // Attemps to insert the current <wseq_state_pair>. If the pair already exists
  // then it returns false.
  typedef MapType::iterator IterType;
  std::pair<IterType, bool> result = wseq_to_state_.insert(wseq_state_pair);

  // If the pair was just inserted, then also add it to <state_to_wseq_>.
  if (result.second == true)
    state_to_wseq_.push_back(wseq);

  // Creates the arc.
  oarc->ilabel = ilabel;
  oarc->olabel = ilabel;
  oarc->nextstate = result.first->second;
  oarc->weight = Weight(-logprob);

  return true;
}

bool BuildConstArpaLm(const bool natural_base, const int32 bos_symbol,
                      const int32 eos_symbol, const int32 unk_symbol,
                      const std::string& arpa_rxfilename,
                      const std::string& const_arpa_wxfilename) {
  ConstArpaLmBuilder lm_builder(natural_base, bos_symbol,
                                eos_symbol, unk_symbol);
  ReadKaldiObject(arpa_rxfilename, &lm_builder);
  lm_builder.Build();
  WriteKaldiObject(lm_builder, const_arpa_wxfilename, true);
  return true;
}

} // namespace kaldi
