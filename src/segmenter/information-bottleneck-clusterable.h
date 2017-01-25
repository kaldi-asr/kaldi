// segmenter/information-bottleneck-clusterable.h

// Copyright 2017   Vimal Manohar (Johns Hopkins University)

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

#ifndef KALDI_SEGMENTER_INFORMATION_BOTTLENECK_CLUSTERABLE_H_
#define KALDI_SEGMENTER_INFORMATION_BOTTLENECK_CLUSTERABLE_H_

#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"
#include "itf/clusterable-itf.h"

namespace kaldi {

class InformationBottleneckClusterable: public Clusterable {
 public:
  /// Constructor used for creating empty object e.g. when reading from file.
  InformationBottleneckClusterable(): total_count_(0.0) { }

  /// Constructor initializing the relevant variable dimension.
  /// Used for making Copy() of object.
  InformationBottleneckClusterable(int32 relevance_dim) :
    total_count_(0.0), p_yp_c_(relevance_dim) { }

  /// Constructor initializing from input stats corresponding to a
  /// segment.
  InformationBottleneckClusterable(int32 id, BaseFloat count,
                                   const VectorBase<BaseFloat> &relevance_dist):
      total_count_(0.0), p_yp_c_(relevance_dist.Dim()) {
    AddStats(id, count, relevance_dist);
  }

  /// Return a copy of this object.
  virtual Clusterable* Copy() const;

  /// Return the objective function, which is
  /// N(c) * (-r * H(Y|c) + ibeta * H(X|c))
  /// where N(c) is the total count in the cluster
  ///       H(Y|c) is the conditional entropy of the relevance
  ///            variable distribution
  ///       H(X|c) is the conditional entropy of the input variable
  ///            distribution
  ///       r is the weight on the relevant variables
  ///       ibeta is the weight on the input variables
  virtual BaseFloat Objf(BaseFloat relevance_factor,
                         BaseFloat input_factor) const;

  /// Return the objective function with the default values
  /// for relevant_factor (1.0) and input_factor (0.1)
  virtual BaseFloat Objf() const { return Objf(1.0, 0.1); }

  /// Return the count in this cluster.
  virtual BaseFloat Normalizer() const { return total_count_; }

  /// Set stats to empty.
  virtual void SetZero() {
    counts_.clear();
    p_yp_c_.Resize(0);
    total_count_ = 0.0;
  }

  /// Add stats to this object
  virtual void AddStats(int32 id, BaseFloat count,
                        const VectorBase<BaseFloat> &relevance_dist);

  /// Add other stats.
  virtual void Add(const Clusterable &other);
  /// Subtract other stats.
  virtual void Sub(const Clusterable &other);
  /// Scale the stats by a positive number f.
  virtual void Scale(BaseFloat f);

  /// Return a string that describes the clusterable type.
  virtual std::string Type() const { return "information-bottleneck"; }

  /// Write data to stream.
  virtual void Write(std::ostream &os, bool binary) const;

  /// Read data from a stream and return the corresponding object (const
  /// function; it's a class member because we need access to the vtable
  /// so generic code can read derived types).
  virtual Clusterable* ReadNew(std::istream &is, bool binary) const;

  /// Read data from stream
  virtual void Read(std::istream &is, bool binary);

  /// Return the objective function of the combined object this + other.
  virtual BaseFloat ObjfPlus(const Clusterable &other,
                             BaseFloat relevance_factor,
                             BaseFloat input_factor) const;

  /// Same as the above function, but using default values for
  /// relevance_factor (1.0) and input_factor (0.1)
  virtual BaseFloat ObjfPlus(const Clusterable &other) const {
    return ObjfPlus(other, 1.0, 0.1);
  }

  /// Return the objective function of the combined object this + other.
  virtual BaseFloat ObjfMinus(const Clusterable &other,
                              BaseFloat relevance_factor,
                              BaseFloat input_factor) const;

  /// Same as the above function, but using default values for
  /// relevance_factor (1.0) and input_factor (0.1)
  virtual BaseFloat ObjfMinus(const Clusterable &other) const {
    return ObjfMinus(other, 1.0, 0.1);
  }

  /// Return the objective function decrease from merging the two
  /// clusters.
  /// Always a non-negative number.
  virtual BaseFloat Distance(const Clusterable &other,
                             BaseFloat relevance_factor,
                             BaseFloat input_factor) const;

  /// Same as the above function, but using default values for
  /// relevance_factor (1.0) and input_factor (0.1)
  virtual BaseFloat Distance(const Clusterable &other) const {
    return Distance(other, 1.0, 0.1);
  }

  virtual ~InformationBottleneckClusterable() {}

  /// Public accessors
  virtual const Vector<BaseFloat>& RelevanceDist() const { return p_yp_c_; }
  virtual int32 RelevanceDim() const { return p_yp_c_.Dim(); }

  virtual const std::map<int32, BaseFloat>& Counts() const { return counts_; }

 private:
  /// A list of the original segments this cluster contains along with
  /// their corresponding counts.
  std::map<int32, BaseFloat> counts_;

  /// Total count in this cluster.
  BaseFloat total_count_;

  /// Relevant variable distribution.
  /// TODO: Make sure that this is a valid probability distribution.
  Vector<BaseFloat> p_yp_c_;
};

/// Returns the KL Divergence between two probability distributions.
BaseFloat KLDivergence(const VectorBase<BaseFloat> &p1,
                       const VectorBase<BaseFloat> &p2);

}  // end namespace kaldi

#endif  // KALDI_SEGMENTER_INFORMATION_BOTTLENECK_CLUSTERABLE_H_
