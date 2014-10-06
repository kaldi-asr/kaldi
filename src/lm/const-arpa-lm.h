// lm/const-arpa-lm.h

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

#ifndef KALDI_LM_CONST_ARPA_LM_H_
#define KALDI_LM_CONST_ARPA_LM_H_

#include "base/kaldi-common.h"
#include "fstext/deterministic-fst.h"
#include "util/common-utils.h"

namespace kaldi {

// Forward declaration of Auxiliary struct ArpaLine.
struct ArpaLine;

class ConstArpaLm {
 public:

  // Default constructor, will be used if you are going to load the ConstArpaLm
  // format language model from disk.
  ConstArpaLm() {
    lm_states_ = NULL;
    unigram_states_ = NULL;
    overflow_buffer_ = NULL;
    memory_assigned_ = false;
    initialized_ = false;
  }

  // Special constructor, will be used when you initialize ConstArpaLm from
  // scratch through this constructor.
  ConstArpaLm(const int32 bos_symbol, const int32 eos_symbol,
              const int32 unk_symbol, const int32 ngram_order,
              const int32 num_words, const int32 overflow_buffer_size,
              const int32 lm_states_size, int32** unigram_states,
              int32** overflow_buffer, int32* lm_states) :
      bos_symbol_(bos_symbol), eos_symbol_(eos_symbol),
      unk_symbol_(unk_symbol), ngram_order_(ngram_order),
      num_words_(num_words), overflow_buffer_size_(overflow_buffer_size),
      lm_states_size_(lm_states_size), unigram_states_(unigram_states),
      overflow_buffer_(overflow_buffer), lm_states_(lm_states) {
    KALDI_ASSERT(unigram_states_ != NULL);
    KALDI_ASSERT(overflow_buffer_ != NULL);
    KALDI_ASSERT(lm_states_ != NULL);
    KALDI_ASSERT(ngram_order_ > 0);
    KALDI_ASSERT(bos_symbol_ < num_words_ && bos_symbol_ > 0);
    KALDI_ASSERT(eos_symbol_ < num_words_ && eos_symbol_ > 0);
    KALDI_ASSERT(unk_symbol_ < num_words_ &&
                 (unk_symbol_ > 0 || unk_symbol_ == -1));
    lm_states_end_ = lm_states_ + lm_states_size_ - 1;
    memory_assigned_ = false;
    initialized_ = true;
  }

  ~ConstArpaLm() {
    if (memory_assigned_) {
      delete[] lm_states_;
      delete[] unigram_states_;
      delete[] overflow_buffer_;
    }
  }

  // Reads the ConstArpaLm format language model.
  void Read(std::istream &is, bool binary);

  // Writes the language model in ConstArpaLm format.
  void Write(std::ostream &os, bool binary) const;

  // Creates Arpa format language model from ConstArpaLm format, and writes it
  // to output stream. This will be useful in testing.
  void WriteArpa(std::ostream &os) const;

  // Wrapper of GetNgramLogprobRecurse. It first maps possible out-of-vocabulary
  // words to <unk>, if <unk> is defined, and then calls GetNgramLogprobRecurse.
  float GetNgramLogprob(const int32 word, const std::vector<int32>& hist) const;

  // Returns true if the history word sequence <hist> has successor, which means
  // <hist> will be a state in the FST format language model.
  bool HistoryStateExists(const std::vector<int32>& hist) const;

  int32 BosSymbol() const { return bos_symbol_; }
  int32 EosSymbol() const { return eos_symbol_; }
  int32 UnkSymbol() const { return unk_symbol_; }
  int32 NgramOrder() const { return ngram_order_; }

 private:
  // Loops up n-gram probability for given word sequence. Backoff is handled by
  // recursively calling this function. 
  float GetNgramLogprobRecurse(const int32 word,
                               const std::vector<int32>& hist) const;

  // Given a word sequence, find the address of the corresponding LmState.
  // Returns NULL if no corresponding LmState is found.
  //
  // If the word sequence exists in n-gram language model, but it is a leaf and
  // is not an unigram, we still return NULL, since there is no LmState struct
  // reserved for this sequence. 
  int32* GetLmState(const std::vector<int32>& seq) const;

  // Given a pointer to the parent, find the child_info that corresponds to
  // given word. The parent has the following structure:
  // struct LmState {
  //   float logprob;
  //   float backoff_logprob;
  //   int32 num_children;
  //   std::pair<int32, int32> [] children;
  // }
  // It returns false if the child is not found.
  bool GetChildInfo(const int32 word, int32* parent, int32* child_info) const;

  // Decodes <child_info> to get log probability and child LmState. In the leaf
  // case, only <logprob> will be returned, and <child_address> will be NULL.
  void DecodeChildInfo(const int32 child_info, int32* parent,
                       int32** child_lm_state, float* logprob) const;

  void WriteArpaRecurse(int32* lm_state,
                        const std::vector<int32>& seq,
                        std::vector<ArpaLine> *output) const;

  // We assign memory in Read(). If it is called, we have to release memory in
  // the destructor.
  bool memory_assigned_;

  // Makes sure that the language model has been loaded before using it.
  bool initialized_;

  // Integer corresponds to <s>.
  int32 bos_symbol_;

  // Integer corresponds to </s>.
  int32 eos_symbol_;

  // Integer corresponds to unknown-word. -1 if no unknown-word symbol is
  // provided.
  int32 unk_symbol_;

  // N-gram order of the language model.
  int32 ngram_order_;

  // Index of largest word-id plus one. It defines the end of <unigram_states_>
  // array.
  int32 num_words_;

  // Number of entries in the overflow buffer for pointers that couldn't be
  // represented as a 30-bit relative index.
  int32 overflow_buffer_size_;

  // Size of the <lm_states_> array, which will be needed by I/O.
  int32 lm_states_size_;

  // Points to the end of <lm_states_>. We use this information to check if
  // there is any illegal visit to the un-reserved memory.
  int32* lm_states_end_;

  // Loopup table for pointers of unigrams. The pointer could be NULL, for
  // example for those words that are in words.txt, but not in the language
  // model.
  int32** unigram_states_;

  // Technically a 32-bit number cannot represent a possibly 64-bit pointer. We
  // therefore use "relative" address instead of "absolute" address, which will
  // be a small number most of the time. This buffer is for the case where the
  // relative address has more than 30-bits.
  int32** overflow_buffer_;

  // Memory chunk that contains the actual LmStates. One LmState has the
  // following structure:
  //
  // struct LmState {
  //   float logprob;
  //   float backoff_logprob;
  //   int32 num_children;
  //   std::pair<int32, int32> [] children;
  // }
  //
  // Note that the floating point representation has 4 bytes, int32 also has 4
  // bytes, therefore one LmState will occupy the following number of bytes:
  //
  // x = 1 + 1 + 1 + 2 * children.size() = 3 + 2 * children.size() 
  int32* lm_states_;
};

/**
 This class wraps a ConstArpaLm format language model with the interface defined
 in DeterministicOnDemandFst.
 */
class ConstArpaLmDeterministicFst :
    public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  ConstArpaLmDeterministicFst(const ConstArpaLm& lm);

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual StateId Start() { return start_state_; }

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual Weight Final(StateId s);

  virtual bool GetArc(StateId s, Label ilabel, fst::StdArc* oarc);

 private:
  typedef unordered_map<std::vector<Label>,
                        StateId, VectorHasher<Label> > MapType;
  StateId start_state_;
  MapType wseq_to_state_;
  std::vector<std::vector<Label> > state_to_wseq_;
  const ConstArpaLm& lm_;
};

// Reads in an Arpa format language model and converts it into ConstArpaLm
// format. We assume that the words in the input Arpa format language model have
// been converted into integers.
bool BuildConstArpaLm(const bool natural_base, const int32 bos_symbol,
                      const int32 eos_symbol, const int32 unk_symbol,
                      const std::string& arpa_rxfilename,
                      const std::string& const_arpa_wxfilename);

} // namespace kaldi

#endif  // KALDI_LM_CONST_ARPA_LM_H_
