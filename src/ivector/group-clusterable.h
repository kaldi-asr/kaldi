// ivector/group-clusterable.h

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

#ifndef KALDI_IVECTOR_GROUP_CLUSTERABLE_H_
#define KALDI_IVECTOR_GROUP_CLUSTERABLE_H_

#include <vector>
#include <set>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "itf/clusterable-itf.h"

namespace kaldi {

class GroupClusterable: public Clusterable {
 public:
  GroupClusterable(const std::set<int32> &points,
    const Matrix<BaseFloat> *scores):
  points_(points),
  scores_(scores),
  total_distance_(0) {
    for (std::set<int32>::iterator itr_i = points_.begin();
      itr_i != points_.end(); ++itr_i) {
      for (std::set<int32>::iterator itr_j = itr_i;
        itr_j != points_.end(); ++itr_j) {
        total_distance_ += (*scores_)(*itr_i, *itr_j);
      }
    }
  }
  virtual std::string Type() const { return "group"; }
  virtual BaseFloat Objf() const;
  virtual void SetZero();
  virtual void Add(const Clusterable &other_in);
  virtual void Sub(const Clusterable &other_in);
  virtual BaseFloat Normalizer() const;
  virtual Clusterable *Copy() const;
  virtual void Scale(BaseFloat f);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Clusterable *ReadNew(std::istream &is, bool binary) const;
  virtual ~GroupClusterable() {}
  virtual BaseFloat Distance(const Clusterable &other_in) const;

 private:
  std::set<int32> points_;
  const Matrix<BaseFloat> * scores_; // Scores between all elements
  BaseFloat total_distance_;
};

}

#endif  // KALDI_IVECTOR_GROUP_CLUSTERABLE_H_
