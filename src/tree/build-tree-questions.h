// tree/build-tree-questions.h

// Copyright 2009-2011  Microsoft Corporation

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

#ifndef KALDI_TREE_BUILD_TREE_QUESTIONS_H_
#define KALDI_TREE_BUILD_TREE_QUESTIONS_H_

#include "util/stl-utils.h"
#include "tree/context-dep.h"

namespace kaldi {


/// \addtogroup tree_group
/// @{
/// Typedef for statistics to build trees.
typedef std::vector<std::pair<EventType, Clusterable*> > BuildTreeStatsType;

/// Typedef used when we get "all keys" from a set of stats-- used in specifying
/// which kinds of questions to ask.
typedef enum { kAllKeysInsistIdentical, kAllKeysIntersection, kAllKeysUnion } AllKeysType;

/// @}

/// \defgroup tree_group_questions Question sets for decision-tree clustering
///  See \ref tree_internals (and specifically \ref treei_func_questions) for context.
/// \ingroup tree_group
/// @{

/// QuestionsForKey is a class used to define the questions for a key,
/// and also options that allow us to refine the question during tree-building
/// (i.e. make a question specific to the location in the tree).
/// The Questions class handles aggregating these options for a set
/// of different keys.
struct QuestionsForKey {  // Configuration class associated with a particular key
  // (of type EventKeyType).  It also contains the questions themselves.
  std::vector<std::vector<EventValueType> > initial_questions;
  RefineClustersOptions refine_opts;  // if refine_opts.max_iter == 0,
  // we just pick from the initial questions.
  
  QuestionsForKey(int32 num_iters = 5): refine_opts(num_iters, 2) {
    // refine_cfg with 5 iters and top-n = 2 (this is no restriction because
    // RefineClusters called with 2 clusters; would get set to that anyway as
    // it's the only possible value for 2 clusters).  User has to add questions.
    // This config won't work as-is, as it has no questions.
  }

  void Check() const {
    for (size_t i = 0;i < initial_questions.size();i++) KALDI_ASSERT(IsSorted(initial_questions[i]));
  }

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  // copy and assign allowed.
};

/// This class defines, for each EventKeyType, a set of initial questions that
/// it tries and also a number of iterations for which to refine the questions to increase
/// likelihood. It is perhaps a bit more than an options class, as it contains the
/// actual questions.
class Questions {  // careful, this is a class.
 public:
  const QuestionsForKey &GetQuestionsOf(EventKeyType key) const {
    std::map<EventKeyType, size_t>::const_iterator iter;
    if ( (iter = key_idx_.find(key)) == key_idx_.end()) {
      KALDI_ERR << "Questions: no options for key "<< key;
    }
    size_t idx = iter->second;
    KALDI_ASSERT(idx < key_options_.size());
    key_options_[idx]->Check();
    return *(key_options_[idx]);
  }
  void SetQuestionsOf(EventKeyType key, const QuestionsForKey &options_of_key) {
    options_of_key.Check();
    if (key_idx_.count(key) == 0) {
      key_idx_[key] = key_options_.size();
      key_options_.push_back(new QuestionsForKey());
      *(key_options_.back()) = options_of_key;
    } else {
      size_t idx = key_idx_[key];
      KALDI_ASSERT(idx < key_options_.size());
      *(key_options_[idx]) = options_of_key;
    }
  }
  void GetKeysWithQuestions(std::vector<EventKeyType> *keys_out) const {
    KALDI_ASSERT(keys_out != NULL);
    CopyMapKeysToVector(key_idx_, keys_out);
  }
  const bool HasQuestionsForKey(EventKeyType key) const { return (key_idx_.count(key) != 0); }
  ~Questions() { kaldi::DeletePointers(&key_options_); }


  /// Initializer with arguments.  After using this you would have to set up the config for each key you
  /// are going to use, or use InitRand().
  Questions() { }


  /// InitRand attempts to generate "reasonable" random questions.  Only
  /// of use for debugging.  This initializer creates a config that is
  /// ready to use.
  /// e.g. num_iters_refine = 0 means just use stated questions (if >1, will use
  /// different questions at each split of the tree).
  void InitRand(const BuildTreeStatsType &stats, int32 num_quest, int32 num_iters_refine, AllKeysType all_keys_type);

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
 private:
  std::vector<QuestionsForKey*> key_options_;
  std::map<EventKeyType, size_t> key_idx_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(Questions);
};

/// @}

}// end namespace kaldi

#endif // KALDI_TREE_BUILD_TREE_QUESTIONS_H_
