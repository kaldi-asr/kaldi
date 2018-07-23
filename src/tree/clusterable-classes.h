// tree/clusterable-classes.h

// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//                2014  Daniel Povey

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

#ifndef KALDI_TREE_CLUSTERABLE_CLASSES_H_
#define KALDI_TREE_CLUSTERABLE_CLASSES_H_ 1

#include <string>
#include "itf/clusterable-itf.h"
#include "matrix/matrix-lib.h"

namespace kaldi {

/// \addtogroup clustering_group
/// @{

/// ScalarClusterable clusters scalars with x^2 loss.
class ScalarClusterable: public Clusterable {
 public:
  ScalarClusterable(): x_(0), x2_(0), count_(0) {}
  explicit ScalarClusterable(BaseFloat x): x_(x), x2_(x*x), count_(1) {}
  virtual std::string Type() const { return "scalar"; }
  virtual BaseFloat Objf() const;
  virtual void SetZero() { count_ = x_ = x2_ = 0.0; }
  virtual void Add(const Clusterable &other_in);
  virtual void Sub(const Clusterable &other_in);
  virtual Clusterable* Copy() const;
  virtual BaseFloat Normalizer() const {
    return static_cast<BaseFloat>(count_);
  }

  // Function to write data to stream. Will organize input later [more complex]
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Clusterable* ReadNew(std::istream &is, bool binary) const;

  std::string Info();  // For debugging.
  BaseFloat Mean() { return (count_ != 0 ? x_/count_ : 0.0); }
  private:
  BaseFloat x_;
  BaseFloat x2_;
  BaseFloat count_;

  void Read(std::istream &is, bool binary);
};


/// GaussClusterable wraps Gaussian statistics in a form accessible
/// to generic clustering algorithms.
class GaussClusterable: public Clusterable {
 public:
  GaussClusterable(): count_(0.0), var_floor_(0.0) {}
  GaussClusterable(int32 dim, BaseFloat var_floor):
      count_(0.0), stats_(2, dim), var_floor_(var_floor) {}

  GaussClusterable(const Vector<BaseFloat> &x_stats,
                   const Vector<BaseFloat> &x2_stats,
                   BaseFloat var_floor, BaseFloat count);

  virtual std::string Type() const {  return "gauss"; }
  void AddStats(const VectorBase<BaseFloat> &vec, BaseFloat weight = 1.0);
  virtual BaseFloat Objf() const;
  virtual void SetZero();
  virtual void Add(const Clusterable &other_in);
  virtual void Sub(const Clusterable &other_in);
  virtual BaseFloat Normalizer() const { return count_; }
  virtual Clusterable *Copy() const;
  virtual void Scale(BaseFloat f);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Clusterable *ReadNew(std::istream &is, bool binary) const;
  virtual ~GaussClusterable() {}

  BaseFloat count() const { return count_; }
  // The next two functions are not const-correct, because of SubVector.
  SubVector<double> x_stats() const { return stats_.Row(0); }
  SubVector<double> x2_stats() const { return stats_.Row(1); }
 private:
  double count_;
  Matrix<double> stats_; // two rows: sum, then sum-squared.
  double var_floor_;  // should be common for all objects created.

  void Read(std::istream &is, bool binary);
};

/// @} end of "addtogroup clustering_group"

inline void GaussClusterable::SetZero() {
  count_ = 0;
  stats_.SetZero();
}

inline GaussClusterable::GaussClusterable(const Vector<BaseFloat> &x_stats,
                                          const Vector<BaseFloat> &x2_stats,
                                          BaseFloat var_floor, BaseFloat count):
    count_(count), stats_(2, x_stats.Dim()), var_floor_(var_floor) {
  stats_.Row(0).CopyFromVec(x_stats);
  stats_.Row(1).CopyFromVec(x2_stats);
}


/// VectorClusterable wraps vectors in a form accessible to generic clustering
/// algorithms.  Each vector is associated with a weight; these could be 1.0.
/// The objective function (to be maximized) is the negated sum of squared
/// distances from the cluster center to each vector, times that vector's
/// weight.
class VectorClusterable: public Clusterable {
 public:
  VectorClusterable(): weight_(0.0), sumsq_(0.0) {}

  VectorClusterable(const Vector<BaseFloat> &vector,
                    BaseFloat weight);

  virtual std::string Type() const {  return "vector"; }
  // Objf is negated weighted sum of squared distances.
  virtual BaseFloat Objf() const;
  virtual void SetZero() { weight_ = 0.0; sumsq_ = 0.0; stats_.Set(0.0); }
  virtual void Add(const Clusterable &other_in);
  virtual void Sub(const Clusterable &other_in);
  virtual BaseFloat Normalizer() const { return weight_; }
  virtual Clusterable *Copy() const;
  virtual void Scale(BaseFloat f);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Clusterable *ReadNew(std::istream &is, bool binary) const;
  virtual ~VectorClusterable() {}

 private:
  double weight_;  // sum of weights of the source vectors.  Never negative.
  Vector<double> stats_; // Equals the weighted sum of the source vectors.
  double sumsq_;  // Equals the sum over all sources, of weight_ * vec.vec,
                  // where vec = stats_ / weight_.  Used in computing
                  // the objective function.
  void Read(std::istream &is, bool binary);
};



}  // end namespace kaldi.

#endif  // KALDI_TREE_CLUSTERABLE_CLASSES_H_
