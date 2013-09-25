// transform/compressed-transform-stats.h

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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


#ifndef KALDI_TRANSFORM_COMPRESSED_TRANSFORM_STATS_H_
#define KALDI_TRANSFORM_COMPRESSED_TRANSFORM_STATS_H_

#include <vector>

#include "transform/transform-common.h"

namespace kaldi {

// The purpose of this class is to compress the AffineXformStats into less
// memory for easier storage and transmission across the network.  It was a
// feature requested by particular user of Kaldi.  It's based on the
// CompressedMatrix class, which compresses a matrix into around one byte per
// element, but before applying that, we first use various techniques to
// normalize the range of elements of the stats and to make it so that the
// compressed G matrices will still be positive definite.  [Basically, we
// compress the Cholesky of each G_i, and we first normalize all the G_i to have
// the same trace.]  We also mess with the K stats a bit, to ensure that the
// derivative of the "compressed" transform taken where the transformation
// matrix is the "default" matrix, is the same as the derivative of the
// un-compressed matrix.  [I.e. we correct the stored K to account for the
// compression of G.]

class CompressedAffineXformStats {
 public:
  CompressedAffineXformStats(): beta_(0.0) { }
  CompressedAffineXformStats(const AffineXformStats &input) {
    CopyFromAffineXformStats(input);
  }
  void CopyFromAffineXformStats(const AffineXformStats &input);
  
  void CopyToAffineXformStats(AffineXformStats *output) const;

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  private:
  // Note: normally we don't use float, only BaseFloat.  In this case
  // it seems more appropriate to use float (since the stuff in G_ is
  // already a lot more approximate than float.)
  float beta_;
  Matrix<float> K_;
  CompressedMatrix G_; // This dim x [ 1 + (0.5*(dim+1)*(dim+2))] matrix
  // stores the contents of the G_ matrix of the AffineXform Stats, in a
  // compressed form.

  // Convert one G matrix into linearized, normalized form ready
  // for compression.
  static void PrepareOneG(const SpMatrix<double> &Gi, double beta,
                          SubVector<double> *linearized);
  // Reverse the process of PrepareOneG.
  static void ExtractOneG(const SubVector<double> &linearized, double beta,
                          SpMatrix<double> *Gi);
  
};


} // namespace kaldi

#endif  // KALDI_TRANSFORM_COMPRESSED_TRANSFORM_STATS_H_
