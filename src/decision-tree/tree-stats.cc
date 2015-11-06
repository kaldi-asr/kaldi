// decision-tree/tree-stats.cc

// Copyright 2015   Vimal Manohar

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

#ifndef DECISION_TREE_TREE_STATS_H
#define DECISION_TREE_TREE_STATS_H

#include "decision-tree/boolean-io-funcs.h"
#include "decision-tree/tree-stats.h"

namespace kaldi {
namespace decision_tree_classifier {

void BooleanTreeStats::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<BooleanTreeStats>");
  WriteToken(os, binary, "<NumClasses>");
  WriteBasicType(os, binary, num_classes_);
  for (auto it = stats_.begin(); it != stats_.end(); ++it) {
    WriteToken(os, binary, "<Vector>");
    WriteBooleanVector(os, binary, it->first);
    WriteToken(os, binary, "<Counts>");
    (it->second).Write(os, binary);
  }
  WriteToken(os, binary, "</BooleanTreeStats>");
}

void BooleanTreeStats::Read(std::istream &is, bool binary, bool add) {
  ExpectToken(is, binary, "<BooleanTreeStats>");
  ExpectToken(is, binary, "<NumClasses>");
  ReadBasicType(is, binary, &num_classes_);
  std::string token;
  ReadToken(is, binary, &token);

  if (!add) {
    stats_.clear();
    stats_map_.clear();
  }

  while (token != "</BooleanTreeStats>") {
    if (token != "<Vector>") {
      KALDI_ERR << "Unexpected token " << token 
                << "; expecting <Vector> or </BooleanTreeStats>";
    }

    std::vector<bool> vec;
    ReadBooleanVector(is, binary, &vec);

    ExpectToken(is, binary, "<Counts>");

    if (!add) {
      KALDI_ASSERT(stats_map_.count(vec) == 0);
      stats_map_.emplace(vec, NumStats());
      stats_.emplace_back(vec, Vector<BaseFloat>(num_classes_));
      stats_.back().second.Read(is, binary);
    } else {
      auto it = stats_map_.find(vec);
      if (it == stats_map_.end()) {
        stats_map_.emplace(vec, NumStats());
        stats_.emplace_back(vec, Vector<BaseFloat>(num_classes_));
        stats_.back().second.Read(is, binary);
      } else {
        stats_[it->second].second.Read(is, binary, true);
      }
    }
    
    ReadToken(is, binary, &token);
  }
}

void BooleanTreeStats::Accumulate(const std::vector<bool> &bits,
                                  int32 label) {
  auto it = stats_map_.find(bits);
  if (it == stats_map_.end()) {
    Vector<BaseFloat> init_tree_stats(num_classes_);
    init_tree_stats(label) = 1.0;
    stats_map_.emplace(bits, NumStats());
    stats_.emplace_back(bits, init_tree_stats);
  } else {
    (stats_[it->second].second)(label) += 1.0;
  }
}

void BooleanTreeStats::Accumulate(const std::pair<std::vector<bool>, 
                                                  Vector<BaseFloat> > &stat) {
  auto it = stats_map_.find(stat.first);
  if (it == stats_map_.end()) {
    stats_map_.emplace(stat.first, NumStats());
    stats_.push_back(stat);
  } else {
    KALDI_ASSERT(NumClasses() == stat.second.Dim());
    stats_[it->second].second.AddVec(1.0, stat.second);
  }
}

void BooleanTreeStats::Accumulate(const std::vector<bool> &vec,
                                  const Vector<BaseFloat> &counts) {
  auto it = stats_map_.find(vec);
  if (it == stats_map_.end()) {
    stats_map_.emplace(vec, NumStats());
    stats_.emplace_back(vec, counts);
  } else {
    KALDI_ASSERT(NumClasses() == counts.Dim());
    stats_[it->second].second.AddVec(1.0, counts);
  }
}

void BooleanTreeStats::AddStats(const BooleanTreeStats &other) {
  if (NumClasses() == -1) 
    num_classes_ = other.NumClasses();
  else if (num_classes_ != other.NumClasses()) {
    KALDI_ERR << "Trying to add stats with different number of classes; "
              << NumClasses() << " vs " << other.NumClasses();
  }

  const std::vector<std::pair<std::vector<bool>, Vector<BaseFloat> > > &other_stats = other.Stats(); 

  for (auto it = other_stats.begin(); it != other_stats.end(); ++it) {
    Accumulate(*it);
  }
}

}
}

#endif // DECISION_TREE_TREE_STATS_H
