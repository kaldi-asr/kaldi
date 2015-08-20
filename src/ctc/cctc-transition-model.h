// lat/cctc-transition-model.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


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


#ifndef KALDI_CTC_CCTC_TRANSITION_MODEL_H_
#define KALDI_CTC_CCTC_TRANSITION_MODEL_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "cudamatrix/cu-matrix.h"
#include "ctc/language-model.h"

namespace kaldi {
namespace ctc {

// CTC means Connectionist Temporal Classification, see the paper by Graves et
// al.  CCTC means context-dependent CTC, it's an extension of the original model.
//

// This is the 'experimental' version of the CTC model.. it's not exactly the same as
// the original CTC formulation, unless you run with zero left context (and even
// then it's not really the same due to the existence of the language model)..  This
// class contains everything you need except the actual neural net.  The idea is
// to make it independent of the type of the neural net.  Note: there is no
// transition model in this framework; this class is the closest thing we have
// to a transition model and we may later have to figure out how to do things
// like lattice time alignment with it.
//
// Now, the decision tree doesn't let us predict blanks... how do we decide how
// many different pdf-ids to give blank (i.e. as a function of the context)?
// One way would be just to have as many blank labels as there are distinct
// language model states... in a bigram model this will be as many as there
// are phones so this will be fairly manageable.
//
// Mappings for (H,p)
//
//         H -> history_id.   The history_id is an integer that we define, that is sufficient to identify the language-model
//                            state and also together with p and the acoustic
//                            probs gives us all the required info for the
//                            denominator of (1).  So we can tabulate the
//                            denominator of (1) by history_id.  The mechanism of
//                            computing the mapping consists of working out
//                            the lm_history_state and history_weight_vector separately
//                            for each H, and then assigning the same history_id to
//                            sets of histories that map to the same values.
//
//  (history_id,p) -> model_id.  We ensure that the history_id contains
//                            sufficient context that this map can be computed.  In fact it's
//                            *almost* implicit in the above definition, and together with
//                            the history_id->lm_history_state map is sufficient to define
//                            the history_id.
//
//  (history_id,p) -> next history_id.  We may need to split the history_ids slightly in order
//                            to ensure that this mapping can always be computed.  But it will
//                            make graph compilation easier.
//
//
// HERE.
//   How to compute history_ids?  First make all language model states history_ids.
//   Then for each history_id that is shorter than the decision tree left-context (e.g. 1 or 2),
//   make sure that for each p, the extended sequence (p, H) is a history_id.
//
//   At this point we compute the maps
//           history_id,p -> model_id.  [compute this is a vector<int32>].  hash this somehow?
//           history_id,p -> next_history_id.  [can make this a vector too.]
//   ... after each time we get a candidate pair to merge based on the model_id map, we can
//       compare the next_history_id thing, because this will almost always succeed.
//
//   Now we 
//
//
//   First assign each distinct H up to the context of the
//        decision tree a unique history_id.  For each of these, ask: do we need to
//        make it more specific as a language-model history state?  
//       [IsLmHistoryStateSuffix()].  If so, figure out the
//  
//
//
//
//   Then compute the maps:
//        history_id,p -> model_id  [some kind of hashable object here?  could do it with
//                                   approximate hashing function to save memory.]
//
//       
//
//
//
//
//
//      (H,p) -> model_id.    A model_id corresponds to a column of the output of
//                            the neural net.  There is a distinct model_id for each state of the
//                            decision tree, plus for each LM history state (i.e. each
//                            lm_history_state) there is a model_id for (lm_history_state,blank).
//                            [note: the decision to have a blank per LM-state was made for
//                            convenience, we could have had as many or as few as we wanted].
//
// 
// history_id -> lm_history_state.  This maps directly to the language model state
//                               of the phone-level language model.
// history_id -> example phonetic context.
// (lm_history_state,p)  -> lm_prob ==  p_{lm}(p | H)..  define this as 1 if p == blank.
//                            
// history_id -> history_weight_vector.   A history-weight vector is what we
//                              have to dot with the output acoustic probs to get
//                              the denominator \sum_q p_{lm}(q | H) * p_{ac}(H,q)
//                              We get it by starting with the zero vector vec = [0...], and then
//                              for each phone q, compute model_id as a function of (H,q),
//                              and do vec[model_id] += p_{lm}(q | H).  Define p_{lm}(blank | H)
//                              as 1 for these purposes.
//
//      (H,p) -> pair_id.     This is an index sufficient to compute the history_id and model_id, i.e.
//    pair_id -> history_id
//    pair_id -> model_id
//
// we can tabulate the mapping from H to history_id- this will be manageable up to
// two phones of left context.
//
// Prior to training or testing:  we compute the matrix of history-weight-vectors,so we 
//                   can compute them all at once.
//
// During training:  - our fst has (H,p) pairs as labels, where p is sometimes blank.
//                     We pre-compute the mapping from (H,p) to
//                     (history_id, model_id, log(lm_prob)).  The log-like is just
//                     a difference of logs.  The log(lm_prob) is only needed for
//                     diagnostic purposes, it doesn't affect the training.
//
// During testing -   initially our fst has (H,p) pairs as labels (like
//                    ilabel indexes).  To make it more determinizable and minimizable
//                    we can modify it by mapping the ilabel indexes to pair_ids;
//                    we can just discard the lm_probs at this point, or if for
//                    interpolation purposes we want to add them with a small weight,
//                    we can add them to the transition costs.
//
// let p be a phone or blank.
// How do we go from (H, p) to a graph-label?  - Map the H to a history_id
// using a table-based approach where we ensure that some truncation of the history
// always has an element in the table.
//  Map the (H, p) to a model-id using the decision tree or (for blanks) the LM
//  history-state map.
// Map the triple (history_id, model_id, phone) to a graph-label using a hash.
// We need to ensure that a (history_id, phone) always maps to another
// 
//
//
//
//
//
// Cctc corresponds to context-dependent connectionist temporal classification;
// it's an extension to the original model in which we make the output
// probabilities dependent on the previously emitted symbols.
// This class includes everything but the neural net; is is similar in
// function to the TransitionModel class in the HMM-based approach.
class CctcTransitionModel {
 public:
  // You generally won't need to know what the left-context is, as viewing it as
  // a graph or an acceptor may be easier (i.e. a map from
  // (state,phone)->state).
  int32 PhoneLeftContext() const { return phone_left_context_; }

  // The number of output-indexes-- i.e. the output dimension that the neural
  // net you use with this should have.  It will equal ctx_dep_.NumPdfs() +
  // language_model_.NumHistoryStates().  The output_indexes are numbered from 0
  // to NumOutputIndexes() - 1.
  int32 NumOutputIndexes() const { return num_output_indexes_; }

  // just in case you need to know it, this returns the number of output
  // indexes that correspond to non-blank symbols (i.e. real phones).  The
  // non-blank indexes come first.
  int32 NumNonBlankIndexes() const { return num_non_blank_indexes_; }

  // return the number of history-states the model contains.
  int32 NumHistoryStates() const { return history_state_info_.size(); }

  // return the number of phones.  Phones are one-based, so NumPhones() is the
  // index of the largest phone, but phone 0 is used to mean the blank symbol.
  int32 NumPhones() const { return num_phones_; }

  // returns the matrix of weights, used for calculating denominator
  // probabilities: row index is history-state index from 0 to
  // NumHistoryStates() - 1, column index is neural-net output index, from 0 to
  // NumOutputIndexes() - 1.
  const CuMatrix<BaseFloat> &GetWeights() const { return weights_; }
  
  // A graph-label is a similar concept to a transition-id in HMM-based models;
  // it's a one-based index that appears on the input side of a decoding graph
  // or training graph.  A graph label can be mapped to the phone (which may be
  // 0 for blank, and which in this context means the rightmost, or predicted,
  // phone); to the output-index which is the row of the output of the nnet that
  // forms the numerator of the expression; and to the history_state, which
  // identifies the row of the weights_ matrix (and which is usually more
  // specific than the lm_history_state); and to the LM probability
  // corresponding to this transition (this will be 1.0 for blank symbols).
  int32 NumGraphLabels() const { return NumHistoryStates() * (num_phones_+1); }
  
  // Maps graph-label to phone (i.e. the predicted phone, or 0 for blank).
  int32 GraphLabelToPhone(int32 g) const { return g % (num_phones_ + 1); }

  // Maps graph-label to the corresponding language model probability
  // (will be 1.0 if this is for a blank).
  BaseFloat GraphLabelToLmProb(int32 graph_label) const;

  // Maps graph-label to the history-state, i.e. to the row of the
  // denominator_weights_ matrix for that graph-label.
  int32 GraphLabelToHistoryState(int32 graph_label) const;

  // Maps graph-label to the next history-state, i.e. one you'd be in if you
  // added some other phone to the right of the existing phone-sequence.  (The
  // identity of that added phone wouldn't matter as it wouldn't be part of the
  // history).
  int32 GraphLabelToNextHistoryState(int32 graph_label) const;  

  // Returns the history-state at the beginning of an utterance, corresponding
  // to beginning of sentence.
  int32 InitialHistoryState() const;
  
  // Given a history-state and a phone (or 0 for blank), gives the
  // corresponding graph label.
  int32 PairToGraphLabel(int32 history_state, int32 phone) const;

  // Given a history-state and a phone (or 0 for blank), gives the
  // next history state.
  int32 GetNextHistoryState(int32 history_state, int32 phone) const;

  // Returns the language model probability of this phone given this history
  // state (or zero for blank).
  BaseFloat GetLmProb(int32 history_state, int32 phone) const;  

  // Returns the output-index for this phone [or 0 for blank] given this history
  // state.
  int32 GetOutputIndex(int32 history_state, int32 phone) const;  

  
  // Maps graph-label to the output index (between zero NumOutputIndexes() - 1),
  // which will be used to look up (in the nnet output) the numerator of the
  // expression for the likelihood of this phone (or blank).
  int32 GraphLabelToOutputIndex(int32 graph_label) const;

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

 protected:
  // Check that the contents of this object make sense (does not check
  // weights_, which is a derived variable).
  void Check() const;
  
  // This function computes weights_.
  void ComputeWeights();
  
  struct HistoryStateInfo {
    // The next history-state we'd get to after appending the phone p>0 is
    // tabluated as next_history_state[p].  next_history_state[0] equals the
    // index of this history-state, because 0 means blank and blank isn't
    // included in the phone history.
    std::vector<int32> next_history_state;

    // This tabulates the neural network output-index we get for (this-history-state, p)
    // indexed by the phone p (or zero for blank)
    std::vector<int32> output_index;

    // This tabulates the language model probability we get for each
    // phone p > 0 following this history state.  The LM prob
    // for blank (0) is defined as 1.0, as we let the acoustic model take
    // care of predicting it, so phone_lm_prob(0) is set to 1.0.
    Vector<BaseFloat> phone_lm_prob;
  };

  // The graph-label for a (history-state, phone-or-blank) pair (h,p) is equal to
  // h * (num_phones_ + 1) + p
  

  // the number of phones that this model is buit for (this is one-based, so the
  // highest numbered phone is indexed num_phones_; 0 is used for the blank
  // symbol).
  int32 num_phones_;

  // The maximum number of phones of left context that this model cares about;
  // equals the larger of (the tree context width, and the language-model n-gram
  // order) minus one.
  int32 phone_left_context_;
  
  // the dimension of the neural network output.
  int32 num_output_indexes_;
  // The number of the output indexes that don't correspond to various
  // left-contexts of the blank symbol.  Non-blank indexes come before blank
  // ones.
  int32 num_non_blank_indexes_;

  // The index of the history state that we have at the start of a sentence.
  int32 initial_history_state_;
  
  // The vector of information on history states (indexed by history-state).
  // All other information in this class is derived from this, 
  // num_phones_, num_output_indexes_ and num_non_blank_indexes_.
  std::vector<HistoryStateInfo> history_state_info_;

  // This matrix is a derived variable; its row is the history_state,
  // and its column dimension is the output dimension of the neural net.
  // A row of this will be dotted with the output of the neural net to
  // get the denominator of the probability value.
  CuMatrix<BaseFloat> weights_;
  
  friend class CctcTransitionModelCreator;
};

// This class creates (i.e. sets up all the indexes in) class CctcTransitionModel,
// which basically contains a bunch of index mappings and phone language model
// probabilities; it doesn't include the actual neural net.
class CctcTransitionModelCreator {
 public:
  // This class stores const references to these arguments.
  CctcTransitionModelCreator(const ContextDependency &ctx_dep,
                             const LanguageModel &phone_lang_model);

  void InitCctcTransitionModel(CctcTransitionModel *model);
 private:
  typedef unordered_set<std::vector<int32>, VectorHasher<int32> > SetType;
  typedef unordered_map<std::vector<int32>, int32, VectorHasher<int32> > MapType;
  
  
  // This function outputs all the initial histories (pre-merging), represented
  // as vectors of phonetic left-context.
  void GetInitialHistories(SetType *hist_set) const;

  // called from GetInitialHistoryStates().  Writes to history_state_info_.
  // Input is a vector of histories represented as phone left-context
  // histories, each of length at least equal to the decision-tree left context.
  void CreateHistoryInfo(const std::vector<std::vector<int32> > &hist_vec,
                         const MapType &hist_to_state);
  
  // writes to history_state_info_ the initial history states, pre-merging.
  void GetInitialHistoryStates();

  // Does one pass of merging the history states; returns true if anything
  // was merged.  If it returns true you need to call it again as more things
  // may now be mergeable.
  bool MergeHistoryStatesOnePass();

  // Once the history states have been computed and merged, this outputs
  // them to the CctcTransitionModel (where some extra information is computed).
  void OutputToTransitionModel(CctcTransitionModel *model) const;

  // Called from OutputToTransitionModel, this function outputs
  // the info for one history state
  void OutputHistoryState(int32 h, CctcTransitionModel *model) const;  

  // Returns the neural-net output index for this history and this
  // phone.   hist.size() must >= ctx_dep_.ContextWidth() - 1.
  // If phone > 0 (not blank) it uses the decision tree, else
  // it works out the language-model history state and computes
  // an index by adding that to num_tree_leaves_.
  int32 GetOutputIndex(const std::vector<int32> &hist,
                       int32 phone) const;
  
  
  // This struct contains all the info we need for constructing the history
  // states and merging ones that behave the same.
  struct HistoryState {
    int32 lm_history_state;  // the language model history state for this
                             // history state.

    // For each phone p (including zero for blank), output_index[p] contains the
    // index into the neural network's output that we use for the numerator of the
    // likelihoood.  For non-blank phone this corresponds to a decision tree state.
    // note, output_indexes are zero-based.
    std::vector<int32> output_index;
    
    // For each next-phone p > 0, contains the index of the history-state that
    // we get to after appending phone p to the sequence.  The blank phone
    // (phone 0) doesn't affect history states (so it would transition to
    // itself), but we always set the zeroth element of this vector to zero
    // because setting it to the index of this history-state would prevent
    // merging.
    std::vector<int32> next_history_state;

    // This member is provided only for possible debugging use in future, is not
    // needed for most of the code, and is not compared in the operator ==.  It
    // represents a lanugage-model history vector (a sequence of context
    // phones); after merging states, it simply represents an arbitrarily chosen
    // history vector, one out of many merged ones.1
    std::vector<int32> history;

    bool operator == (const HistoryState &other) const {
      return lm_history_state == other.lm_history_state &&
          output_index == other.output_index &&
          next_history_state == other.next_history_state;
    }
  };

  // hashing object that hashes struct HistoryState (from a pointer).
  struct HistoryStateHasher {
    size_t operator () (const HistoryState *const hist_info) const {
      VectorHasher<int32> vec_hasher;
      size_t p1 = 31; 
      return p1 * hist_info->lm_history_state +
          vec_hasher(hist_info->output_index) +
          vec_hasher(hist_info->next_history_state);
    }
  };
  struct HistoryStateEqual {
    size_t operator () (const HistoryState *const hist_info1,
                        const HistoryState *const hist_info2) const {
      return (*hist_info1 == *hist_info2);
    }
  };

  
  typedef unordered_map<const HistoryState*, int32,
                        HistoryStateHasher, HistoryStateEqual> HistoryMapType;
  

  const ContextDependency &ctx_dep_;
  const LanguageModel &phone_lang_model_;
  LmHistoryStateMap lm_hist_state_map_;

  // num_tree_leaves equals ctx_dep_.NumPdfs() (cached because the function is slow).
  int32 num_tree_leaves_;
  // equals num_tree_leaves_ + lm_hist_state_map_.NumLmHistoryStates().
  // the first num_tree_leaves_ indexes are for non-blank phones.
  // the reset are for blanks in the respective phone-language-model contexts.
  int32 num_output_indexes_;

  // the history state that corresponds to the start of a sentence.
  int32 initial_history_state_;
  
  std::vector<HistoryState> history_states_;
};



// ... regarding computing the CTC objective and its derivative from
//  .. a block of frames.
// ... Firstly, how do we interpret the output layer of the CTC model
//     during training?
//
//   - A 'conventional mode', where there is a decision tree used for non-blank
//     symbols that map them to a context-dependent phone index (in which we include blank).
//     Could get this from a decision tree with a shared pdf-index.  The
//     matrix-row will be like a kind of pdf-id.
//     in this mode the normalizer is just the sum of all the output log-probs,
//     and we can use the log-softmax and just look up the probs.  Here, we
//     ignore the context of the blank symbol.
//     
//   - An 'experimental mode', where only left-context is supported (i.e. the
//     misnamed 'central-phone' must be the right-most phone), and any decoding
//     graph has to specify this left-context for all symbols including blank.
//
//     We should probably at least max-normalize and then do exp on the GPU
//     card, to avoid exp's and logs happening on the CPU and slowing things
//     down.
//
//     Symbols in this setup are specified as sets of phones, e.g. suppose the
//     left-context is 2, the set could be written something like
//     [aeiou][nr][k].  The numerator of this prob will be the sum of the probs
//     of all symbols that match the precise phone sequence e.g. 'irk', and the
//     denominator will be the sum of the probs of all symbols that match the
//     left part.
//
//     Obviously the
//
//     For each left-context e.g. 'an' that has been seen, we can cache
//   
// 
//     The probability of the central phone is the prob. of the symbol that
//     matches
//
//     e.g. 'ab' (assuming a is the left context and b is the phone) divided by
//     the sum of probs 'a*' for all phones plus blank.
// 
//  We don't want the model to spend energy learning the LM-probs.  So train
//  with LM-probs as part of the baseline model, assuming a certain number of
//  phones to the left are known, and let it focus on the residual prediction
//  problem.
//
//  We want the residual probability of, given the LM-probs and given what LM
//  state we're in, what is the factor we need to multiply each phone's prob by?
//  View this as MMI on a model with a phone-ngram LM.
//
//  - Next- compute a left-context decision tree (probably biphone) that
//    determines a mapping from (history,phone) = (H,p) to a phone-id.
//
//  - Compute a suitable phone-ngram LM; express it as a table [?]
//
//  - Given a particular left-context phonetic history H, we'll be modeling the probability of the
//    next phone [or blank] p as:
//
//     p(p | H) = { p_{lm}(p | H) * p_{ac}(H,p)) } / { \sum_q p_{lm}(q | H) * p_{ac}(H,p) }    (1)
//
//    Each H,p maps to a pdf-id.
//
//    The denominator sum... how do we do this?  Assume we can enumerate all histories.
//    Compute it as a vector for each history, and then work out the unique vectors.
//    Each one will become a history-state
//
//    During decoding we'll already have the phone probabilities being modeled
//    by a more accurate language model based on words, so we'll divide out the
//    above by the phone-level LM-prob to give:
//
//       p_{decoding}(p | H) = p(p | H) / p_{lm}(p | H).
//
//    or for blank, p_{decoding}(blank | H) = p(blank | H) with no division by
//    the LM prob (since this hasn't been already accounted for in the decoding
//    graph).  An easier way to look at this might be that we take p(p|H) from
//    the simple model we used in training, and multiply it by a correction
//    factor which equals the ratio between our "accurate" language model and
//    the inaccurate phone-bigram language model that we trained with.
//
//    Here, p_{ac}(p,H) is a trainable quantity that we train to maximize
//    p(p | H).  As such we can write it however we want; I chose a comma p,H rather
//    than a pipe symbol p|H.
//
//    
//
//



}  // namespace ctc
}  // namespace kaldi

#endif  // KALDI_CTC_CCTC_TRANSITION_MODEL_H_

