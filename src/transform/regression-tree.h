// transform/regression-tree.h

// Copyright 2009-2011  Saarland University
// Author:  Arnab Ghoshal

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


#ifndef KALDI_TRANSFORM_REGRESSION_TREE_H_
#define KALDI_TRANSFORM_REGRESSION_TREE_H_

#include <utility>
#include <vector>

#include "base/kaldi-common.h"
#include "tree/cluster-utils.h"
#include "gmm/am-diag-gmm.h"
#include "transform/transform-common.h"

namespace kaldi {

/** \class RegressionTree
 *  A regression tree is a clustering of Gaussian densities in an acoustic
 *  model, such that the group of Gaussians at each node of the tree are
 *  transformed by the same transform. Each node is thus called a regression
 *  class.
 */
class RegressionTree {
 public:
  RegressionTree() {}

  /// Top-down clustering of the Gaussians in a model based on their means.
  /// If sil_indices is nonempty, will put silence in a special class
  /// using a top-level split.
  void BuildTree(const Vector<BaseFloat> &state_occs,
                 const std::vector<int32> &sil_indices,
                 const AmDiagGmm &am,
                 int32 max_clusters);

  /// Parses the regression tree and finds the nodes whose occupancies (read
  /// from stats_in) are greater than min_count. The regclass_out vector has
  /// size equal to number of baseclasses, and contains the regression class
  /// index for each baseclass. The stats_out vector has size equal to number
  /// of regression classes. Return value is true if at least one regression
  /// class passed the count cutoff, false otherwise.
  bool GatherStats(const std::vector<AffineXformStats*> &stats_in,
                   double min_count,
                   std::vector<int32> *regclasses_out,
                   std::vector<AffineXformStats*> *stats_out) const;

  void Write(std::ostream &out, bool binary) const;
  void Read(std::istream &in, bool binary, const AmDiagGmm &am);

  /// Accessors (const)
  int32 NumBaseclasses() const { return num_baseclasses_; }
  const std::vector< std::pair<int32, int32> >& GetBaseclass(int32 bclass)
      const { return baseclasses_[bclass]; }
  int32 Gauss2BaseclassId(size_t pdf_id, size_t gauss_id) const {
    return gauss2bclass_[pdf_id][gauss_id];
  }

 private:
  int32 num_nodes_;  ///< Total (non-leaf+leaf) nodes

  /// For each node, index of its parent: size = num_nodes_
  /// If 0 <= i < num_baseclasses_, then i is a leaf of the tree (a base class);
  /// else a non-leaf node.  parents_[i] > i, except for the top node
  /// (last-numbered one), for which parents_[i] == i.
  std::vector<int32> parents_;
  int32 num_baseclasses_;  ///< Number of leaf nodes
  /// Each baseclass (leaf of regression tree) is a vector of Gaussian indices.
  /// Each Gaussian in the model is indexed by (pdf, gaussian) indices pair.
  std::vector< std::vector< std::pair<int32, int32> > > baseclasses_;
  /// Mapping from (pdf, gaussian) indices to baseclasses
  std::vector< std::vector<int32> > gauss2bclass_;

  void MakeGauss2Bclass(const AmDiagGmm &am);

  // Cannot have copy constructor and assigment operator
  KALDI_DISALLOW_COPY_AND_ASSIGN(RegressionTree);

};

}  // namespace kaldi

#endif  // KALDI_TRANSFORM_REGRESSION_TREE_H_
