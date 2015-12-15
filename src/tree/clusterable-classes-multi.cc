// tree/clusterable-classes-multi.cc

// Copyright 2015 Hainan Xu

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

#include <algorithm>
#include <string>
#include "base/kaldi-math.h"
#include "itf/clusterable-itf.h"
#include "tree/clusterable-classes.h"

namespace kaldi {


// ============================================================================
// Implementation of EntropyClusterable class.
// ============================================================================

void EntropyClusterable::ChangeLeafWithMapping(int32 tree_index,
                             const std::vector<int>& mapping) {
            // mapping maps old_index to mapping[old_index]
  KALDI_ASSERT(leaf_assignment_to_count_.size() == 1);
  // this should not be called on any clusterable object 
  // that is a sum of multiple clusterable objects
  int32 old_leaf = leaf_assignment_to_count_.begin()->first[tree_index];
  int32 new_leaf = mapping[old_leaf];
  KALDI_ASSERT(new_leaf != -1);

  this->SetLeafValue(tree_index, new_leaf);
}

BaseFloat EntropyClusterable::DistanceEntropy(const Clusterable &other,
                                              size_t tree_index) const {
  Clusterable *copy = this->Copy();
  copy->Add(other);
  EntropyClusterable *entro = dynamic_cast<EntropyClusterable*>(copy);
  KALDI_ASSERT(entro != NULL);
  entro->SetLeafValue(tree_index, -1);  // -1 is just a dummy here
                            // negative to make sure it's not identical 
                            // to any other leaf-value
  BaseFloat ans = this->Objf() + other.Objf() - copy->Objf();
  // now the ans has the entropy term

  if (ans < 0) {
    // This should not happen. Check if it is more than just rounding error.
    if (std::fabs(ans) > 0.01 * (1.0 + std::fabs(copy->Objf()))) {
      KALDI_WARN << "Negative number returned (badly defined Clusterable "
                 << "class?): ans= " << ans;
    }
    ans = 0;
  }
  delete copy;
  return ans;
}

void EntropyClusterable::CopyFromGaussian(const Clusterable* other) {
  const GaussClusterable *g = dynamic_cast<const GaussClusterable*>(other);
  this->count_ = g->count_;
  this->stats_ = g->stats_;
  this->var_floor_ = g->var_floor_;
  this->lambda_ = 0;
}

//for debug, probably will delete 
double EntropyClusterable::CheckCount() const{
  if (leaf_assignment_to_count_.size() == 0) {
    // this would probably only happen in random generation of stats
    // where maps aren't set up yet
    return count_;
  }
  double sum = 0;
  for (MapType::const_iterator it = leaf_assignment_to_count_.begin(); 
                               it != leaf_assignment_to_count_.end(); 
                               it++) {
    sum += it->second;
  }
  KALDI_ASSERT(std::abs(sum-count_) == 0);
  return sum;
}

void EntropyClusterable::AddStats(const VectorBase<BaseFloat> &vec,
                                  BaseFloat weight) {
  count_ += weight;
  stats_.Row(0).AddVec(weight, vec);
  stats_.Row(1).AddVec2(weight, vec);
}

void EntropyClusterable::SetZero() {
  count_ = 0;
  stats_.SetZero();
  leaf_assignment_to_count_.clear();
}

void EntropyClusterable::Add(const Clusterable &other_in) {
  KALDI_ASSERT(other_in.Type() == "entropy");
  const EntropyClusterable *other =
      dynamic_cast<const EntropyClusterable*>(&other_in);
  count_ += other->count_;
  stats_.AddMat(1.0, other->stats_);
  
  for (MapType::const_iterator it = other->leaf_assignment_to_count_.begin(); 
                               it != other->leaf_assignment_to_count_.end();
                               it++ ) {
    if (leaf_assignment_to_count_.count(it->first) == 0) {
      leaf_assignment_to_count_.insert(MapType::value_type(it->first,
                                                           it->second));
    } else {
      // already exist
      leaf_assignment_to_count_[it->first] += it->second;
    }
  }
}

void EntropyClusterable::AddAndMerge(Clusterable &other_in,
                                     size_t tree_index) {
  KALDI_ASSERT(other_in.Type() == "entropy");
  EntropyClusterable *other =
      dynamic_cast<EntropyClusterable*>(&other_in);
  count_ += other->count_;
  stats_.AddMat(1.0, other->stats_);
  
  MapType::iterator it;

  it = leaf_assignment_to_count_.begin();
  int32 leaf = it->first[tree_index];

  // do some assertions
  while (it != leaf_assignment_to_count_.end()) {
    KALDI_ASSERT(leaf == it->first[tree_index]);
    it++;
  }

  other->SetLeafValue(tree_index, leaf);

  for (MapType::const_iterator it = other->leaf_assignment_to_count_.begin(); 
                               it != other->leaf_assignment_to_count_.end();
                               it++ ) {
    if (leaf_assignment_to_count_.count(it->first) == 0) {
      leaf_assignment_to_count_.insert(MapType::value_type(it->first,
                                                           it->second));
    } else {
      // already exist
      leaf_assignment_to_count_[it->first] += it->second;
    }
  }
//  this->CheckCount(); // for debug 
}

void EntropyClusterable::Sub(const Clusterable &other_in) {
  KALDI_ASSERT(other_in.Type() == "entropy");
  const EntropyClusterable *other =
      static_cast<const EntropyClusterable*>(&other_in);
  count_ -= other->count_;
  stats_.AddMat(-1.0, other->stats_);
  
  for (MapType::const_iterator it = other->leaf_assignment_to_count_.begin(); 
                               it != other->leaf_assignment_to_count_.end();
                               it++ ) {
    if (leaf_assignment_to_count_.count(it->first) == 0) {
      // this is not quite possible
      KALDI_ASSERT(it->second == 0); 
      leaf_assignment_to_count_.erase(it->first);
    }
    else if (leaf_assignment_to_count_[it->first] == it->second) {
      leaf_assignment_to_count_.erase(it->first);
    }
    else {
      // already exist and should be bigger
      double this_count = leaf_assignment_to_count_[it->first];
      double other_count = it->second;

      // avoiding floating point error
      KALDI_ASSERT(this_count - other_count > -0.0001 );
      leaf_assignment_to_count_[it->first] -= it->second;
    }
  }
}

Clusterable* EntropyClusterable::Copy() const {
  KALDI_ASSERT(stats_.NumRows() == 2);
  EntropyClusterable *ans = new EntropyClusterable(stats_.NumCols(),
                                                   var_floor_);
  ans->lambda_ = this->lambda_;

  ans->num_trees_ = num_trees_;
  ans->leaf_assignment_to_count_ = leaf_assignment_to_count_;
  ans->count_ = count_;
  ans->stats_ = stats_;
  return ans;
}

void EntropyClusterable::Scale(BaseFloat f) {
  KALDI_ASSERT(f >= 0.0);
  count_ *= f;
  stats_.Scale(f);
  for (MapType::const_iterator it = leaf_assignment_to_count_.begin(); 
                               it != leaf_assignment_to_count_.end();
                               it++ ) {
    leaf_assignment_to_count_[it->first] *= f;
  }
}

void EntropyClusterable::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "GCL");  // magic string.
  WriteBasicType(os, binary, count_);
  WriteBasicType(os, binary, var_floor_);
  stats_.Write(os, binary);
}

Clusterable* EntropyClusterable::ReadNew(std::istream &is, bool binary) const {
  EntropyClusterable *gc = new EntropyClusterable();
  gc->Read(is, binary);
  return gc;
}

void EntropyClusterable::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "GCL");  // magic string.
  ReadBasicType(is, binary, &count_);
  ReadBasicType(is, binary, &var_floor_);
  stats_.Read(is, binary);
}

void EntropyClusterable::SetLeafValue(int32 tree_index, int32 leaf) {
  MapType::iterator it = leaf_assignment_to_count_.begin();
  if (it == leaf_assignment_to_count_.end()) {
    std::vector<int32> v(num_trees_, -1);
    v[tree_index] = leaf;
    leaf_assignment_to_count_[v] = count_;
    return;
  }
  MapType new_map;
  for (; it != leaf_assignment_to_count_.end(); it++) {
    std::vector<int32> v = it->first;
    double c = it->second;
    v[tree_index] = leaf;  // the new leaf
    new_map[v] += c;
  }
  
  leaf_assignment_to_count_ = new_map;
}

int32 EntropyClusterable::GetNumTrees() const {
  return num_trees_;
}

void EntropyClusterable::SetNumTrees(int32 num_trees) {
  num_trees_ = num_trees;
  if (leaf_assignment_to_count_.size() == 0) {
    std::vector<int32> v(num_trees, -1);
    leaf_assignment_to_count_.insert(MapType::value_type(v, count_));
  }
  else {
    //This assertion here is because this function should only be called
    //** before ** the decision tree splitting starts
    //so, each stats would be associated with at most 1 leaf for each tree
    KALDI_ASSERT(leaf_assignment_to_count_.size() == 1);
    KALDI_ASSERT(leaf_assignment_to_count_.begin()->first.size() == num_trees);
  }
}

double EntropyClusterable::GetEntropy(int32 tree_index) const {
  KALDI_ASSERT(tree_index >= -1 && tree_index < num_trees_);
  double ans = 0.0;
  if (tree_index == -1) { // joint entropy
    for (MapType::const_iterator it = leaf_assignment_to_count_.begin();
                                 it != leaf_assignment_to_count_.end();
                                 it++) {
      if (it->second == 0.0) {
        continue;
      }
      ans += it->second * log(count_ / it->second);
    }
  }
  else {
    typedef unordered_map<int32, double> IntMap;
    IntMap int_map;
    for (MapType::const_iterator it = leaf_assignment_to_count_.begin();
                                 it != leaf_assignment_to_count_.end();
                                 it++) {
      if (int_map.count(it->first[tree_index]) == 0) { // new element
        int_map.insert(IntMap::value_type(it->first[tree_index], it->second));
      }
      else {
        int_map[it->first[tree_index]] += it->second;
      }
    }
    //now we apply the same procedure in the if{} section 
    // to calculate the marginal entropy
    for (IntMap::const_iterator it = int_map.begin();
                                it != int_map.end();
                                it ++) {
      ans += it->second * log(count_ / it->second);
    }
  }
  return ans / count_;
}

BaseFloat EntropyClusterable::Objf() const {
  if (count_ <= 0.0) {
    if (count_ < -0.1) {
      KALDI_WARN << "GaussClusterable::Objf(), count is negative " << count_;
    }
    return 0.0;
  } else {
    size_t dim = stats_.NumCols();
    Vector<double> vars(dim);
    double objf_per_frame = 0.0;
    for (size_t d = 0; d < dim; d++) {
      double mean(stats_(0, d) / count_), var = stats_(1, d) / count_ - mean
          * mean, floored_var = std::max(var, var_floor_);
      vars(d) = floored_var;
      objf_per_frame += -0.5 * var / floored_var;
    }
    objf_per_frame += -0.5 * (vars.SumLog() + M_LOG_2PI * dim);
    if (KALDI_ISNAN(objf_per_frame)) {
      KALDI_WARN << "GaussClusterable::Objf(), objf is NaN\n";
      return 0.0;
    }

    double entropy_term = 0.0;
    for (MapType::const_iterator it = leaf_assignment_to_count_.begin();  
                                 it != leaf_assignment_to_count_.end(); 
                                 it++) {
      double count = it->second;
      if (count == 0) {
        continue; //otherwise we get a NaN for objf
      }
      entropy_term += count * log(count);
    }
    entropy_term += this->count_ * log(this->count_) / this->num_trees_;

    // it should be minus here before the entropy-term
    return objf_per_frame * count_ - lambda_ * entropy_term; 
  }
}

} // end of namespace kaldi
