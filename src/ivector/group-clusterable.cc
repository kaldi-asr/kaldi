// ivector/group-clusterable.cc

// Copyright 2016  David Snyder

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


#include "ivector/group-clusterable.h"

namespace kaldi {

BaseFloat GroupClusterable::Objf() const {
  // TODO (current only using Distance())
  return total_distance_;
}

void GroupClusterable::SetZero() {
  points_.clear();
  total_distance_ = 0;
}

void GroupClusterable::Add(const Clusterable &other_in) {
  const GroupClusterable *other =
      static_cast<const GroupClusterable*>(&other_in);

  for (std::set<int32>::const_iterator itr_i = points_.begin();
       itr_i != points_.end(); ++itr_i) {
    for (std::set<int32>::const_iterator itr_j = other->points_.begin();
         itr_j != other->points_.end(); ++itr_j) {
      total_distance_ += (*scores_)(*itr_i, *itr_j);
    }
  }
  total_distance_ += other->total_distance_;
  points_.insert(other->points_.begin(), other->points_.end());
}

void GroupClusterable::Sub(const Clusterable &other_in) {
  const GroupClusterable *other =
      static_cast<const GroupClusterable*>(&other_in);
  for (std::set<int32>::const_iterator itr_i = points_.begin();
       itr_i != points_.end(); ++itr_i) {
    for (std::set<int32>::const_iterator itr_j = other->points_.begin();
         itr_j != other->points_.end(); ++itr_j) {
      total_distance_ -= (*scores_)(*itr_i, *itr_j);
    }
  }
  total_distance_ -= other->total_distance_;
  for (std::set<int32>::const_iterator itr_i = other->points_.begin();
       itr_i != other->points_.end(); ++itr_i) {
    points_.erase(*itr_i);
  }
}

BaseFloat GroupClusterable::Normalizer() const {
  return points_.size();
}

Clusterable *GroupClusterable::Copy() const {
  GroupClusterable *ans = new GroupClusterable(points_, scores_);
  return ans;
}

void GroupClusterable::Scale(BaseFloat f) {
  // TODO
  return;
}

void GroupClusterable::Write(std::ostream &os, bool binary) const {
  // TODO
  return;
}

Clusterable *GroupClusterable::ReadNew(std::istream &is, bool binary) const {
  // TODO
  return NULL;
}

BaseFloat GroupClusterable::Distance(const Clusterable &other_in) const {
  const GroupClusterable *other =
      static_cast<const GroupClusterable*>(&other_in);
  GroupClusterable *copy = static_cast<GroupClusterable*>(this->Copy());
  copy->Add(*other);
  BaseFloat ans = (copy->Objf() - other->Objf() - this->Objf())
    / (other->Normalizer() * this->Normalizer());
  return ans;
}
}
