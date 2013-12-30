// transform/regression-tree.cc

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

#include <string>
#include <utility>
using std::pair;
#include <vector>
using std::vector;

#include "transform/regression-tree.h"
#include "tree/clusterable-classes.h"
#include "util/common-utils.h"

namespace kaldi {

/// Top-down clustering of the Gaussians in a model based on their means.
void RegressionTree::BuildTree(const Vector<BaseFloat> &state_occs,
                               const std::vector<int32> &sil_indices,
                               const AmDiagGmm &am,
                               int32 max_clusters) {
  KALDI_ASSERT(IsSortedAndUniq(sil_indices));
  int32 dim = am.Dim(),
      num_pdfs = static_cast<int32>(am.NumPdfs());
  vector<Clusterable*> gauss_means;
  // For each Gaussianin the model, the pair of (pdf, gaussian) indices.
  vector< pair<int32, int32> > gauss_indices;
  Vector<BaseFloat> tmp_mean(dim);
  Vector<BaseFloat> tmp_var(dim);
  BaseFloat var_floor = 0.01;

  gauss2bclass_.resize(num_pdfs);
  gauss_means.reserve(am.NumGauss());  // NOT resize, uses push_back
  gauss_indices.reserve(am.NumGauss());  // NOT resize, uses push_back

  for (int32 pdf_index = 0; pdf_index < num_pdfs; pdf_index++) {
    gauss2bclass_[pdf_index].resize(am.GetPdf(pdf_index).NumGauss());
    for (int32 num_gauss = am.GetPdf(pdf_index).NumGauss(),
        gauss_index = 0; gauss_index < num_gauss; ++gauss_index) {
      // don't include silence while clustering...
      if (std::binary_search(sil_indices.begin(), sil_indices.end(), pdf_index))
        continue;

      am.GetGaussianMean(pdf_index, gauss_index, &tmp_mean);
      am.GetGaussianVariance(pdf_index, gauss_index, &tmp_var);
      tmp_var.AddVec2(1.0, tmp_mean);  // make it x^2 stats.
      BaseFloat this_weight =  state_occs(pdf_index) *
                               (am.GetPdf(pdf_index).weights())(gauss_index);
      tmp_mean.Scale(this_weight);
      tmp_var.Scale(this_weight);
      gauss_indices.push_back(std::make_pair(pdf_index, gauss_index));
      gauss_means.push_back(new GaussClusterable(tmp_mean, tmp_var, var_floor,
                                                 this_weight));
    }
  }

  vector<int32> leaves;
  vector<int32> clust_parents;
  int32 num_leaves;
  TreeClusterOptions opts;  // Use default options or get from somewhere else
  TreeCluster(gauss_means,
              (sil_indices.empty() ? max_clusters : max_clusters-1),
              NULL /* clusters not needed */,
              &leaves, &clust_parents, &num_leaves, opts);

  if (sil_indices.empty()) {  // no special treatment of silence...
    num_baseclasses_ = static_cast<int32>(num_leaves);
    baseclasses_.resize(num_leaves);
    parents_.resize(clust_parents.size());
    for (int32 i = 0, num_nodes = clust_parents.size(); i < num_nodes; i++) {
      parents_[i] = static_cast<int32>(clust_parents[i]);
    }
    num_nodes_ = static_cast<int32>(clust_parents.size());
    for (int32 i = 0; i < static_cast<int32>(gauss_indices.size()); i++) {
      baseclasses_[leaves[i]].push_back(gauss_indices[i]);
      gauss2bclass_[gauss_indices[i].first][gauss_indices[i].second] = leaves[i];
    }
  } else {
    // separate top-level split between silence and speech...
    // silence is node zero and new parent is last-numbered one.
    num_baseclasses_ = static_cast<int32>(num_leaves+1);  // +1 to include 0 == silence
    baseclasses_.resize(num_leaves+1);  // +1 to include 0 == silence
    parents_.resize(clust_parents.size()+2);  // +1 to include 0 == silence, +parent.

    int32 top_node = clust_parents.size() + 1;
    for (int32 i = 0; i < static_cast<int32>(clust_parents.size()); i++) {
      parents_[i+1] = clust_parents[i]+1;  // handle offsets
    }
    parents_[0] = top_node;
    parents_[clust_parents.size()] = top_node;  // old top node's parent is new top node.
    parents_[top_node] = top_node;  // being own parent is sign of being top node.

    num_nodes_ = static_cast<int32>(clust_parents.size() + 2);
    // Assign nonsilence Gaussians to their assigned classes (add one
    // to all leaf indices, make room for silence class).
    for (int32 i = 0; i < static_cast<int32>(gauss_indices.size()); i++) {
      baseclasses_[leaves[i]+1].push_back(gauss_indices[i]);
      gauss2bclass_[gauss_indices[i].first][gauss_indices[i].second] = leaves[i]+1;
    }
    // Assign silence Gaussians to zero'th baseclass.
    for (int32 i = 0; i < static_cast<int32>(sil_indices.size()); i++) {
      int32 pdf_index = sil_indices[i];
      for (int32 j = 0; j < am.GetPdf(pdf_index).NumGauss(); j++) {
        baseclasses_[0].push_back(std::make_pair(pdf_index, j));
        gauss2bclass_[pdf_index][j] = 0;
      }
    }
  }
  DeletePointers(&gauss_means);
}


static bool GetActiveParents(int32 node, const vector<int32> &parents,
                             const vector<bool> &is_active,
                             vector<int32> *active_parents_out) {
  KALDI_ASSERT(parents.size() == is_active.size());
  KALDI_ASSERT(static_cast<size_t>(node) < parents.size());
  active_parents_out->clear();
  bool ret_val = false;
  while (node < static_cast<int32> (parents.size() - 1)) {  // exclude the root
    node = parents[node];
    if (is_active[node]) {
      active_parents_out->push_back(node);
      ret_val = true;
    }
  }
  return ret_val;  // will return if not starting from root
  if (node == static_cast<int32> (parents.size() - 1)) {  // root node
    if (is_active[node]) {
      active_parents_out->push_back(node);
      return true;
    } else {
      return false;
    }
  }
  KALDI_ASSERT(false);  // Never reached
}

/// Parses the regression tree and finds the nodes whose occupancies (read
/// from stats_in) are greater than min_count. The regclass_out vector has
/// size equal to number of baseclasses, and contains the regression class
/// index for each baseclass. The stats_out vector has size equal to number
/// of regression classes. Return value is true if at least one regression
/// class passed the count cutoff, false otherwise.
bool RegressionTree::GatherStats(const vector<AffineXformStats*> &stats_in,
                                 double min_count,
                                 vector<int32> *regclasses_out,
                                 vector<AffineXformStats*> *stats_out) const {
  KALDI_ASSERT(static_cast<int32>(stats_in.size()) == num_baseclasses_);
  if (static_cast<int32>(regclasses_out->size()) != num_baseclasses_)
    regclasses_out->resize(static_cast<size_t>(num_baseclasses_), -1);
  if (num_baseclasses_ == 1)  // Only root node in tree
    KALDI_ASSERT(num_nodes_ == 1);

  double total_occ = 0.0;
  int32 num_regclasses = 0;
  vector<double> node_occupancies(num_nodes_, 0.0);
  vector<bool> generate_xform(num_nodes_, false);
  vector<int32> regclasses(num_nodes_, -1);

  // Go through the leaves (baseclasses) and find where to generate transforms
  for (int32 bclass = 0; bclass < num_baseclasses_; bclass++) {
    total_occ += stats_in[bclass]->beta_;
    node_occupancies[bclass] = stats_in[bclass]->beta_;
    if (num_baseclasses_ != 1) {  // Don't count twice if tree only has root.
      node_occupancies[parents_[bclass]] += node_occupancies[bclass];
    }
    if (node_occupancies[bclass] < min_count) {
      // Not enough count, so pass the responsibility to the parent.
      generate_xform[bclass] = false;
      generate_xform[parents_[bclass]] = true;
    } else {  // generate at the leaf level.
      generate_xform[bclass] = true;
      regclasses[bclass] = num_regclasses++;
    }
  }
  // Check whether there is enough data for the single global transform (at
  // the root of the regression tree). If not, no transforms will be computed.
  if (total_occ < min_count) {
    // Make all baseclasses use the unit transform at the root.
    for (int32 bclass = 0; bclass < num_baseclasses_; bclass++) {
      (*regclasses_out)[bclass] = 0;
    }
    DeletePointers(stats_out);
    stats_out->clear();
    KALDI_WARN << "Not enough data to compute global transform. Occupancy at "
               << "root = " << total_occ << "<"  << min_count;
    return false;
  }

  // Now go through the non-leaf nodes and find where to generate transforms.
  // Iterates only till num_nodes_ - 1 so that it doesn't count root twice.
  for (int32 node = num_baseclasses_; node < num_nodes_ - 1; node++) {
    node_occupancies[parents_[node]] += node_occupancies[node];
    // Only bother with generating transforms if a child asked for it.
    if (generate_xform[node]) {
      if (node_occupancies[node] < min_count) {
        // Not enough count, so pass the responsibility to the parent.
        generate_xform[node] = false;
        generate_xform[parents_[node]] = true;
      } else {  // transform will be generated at this level.
        regclasses[node] = num_regclasses++;
      }
    }
  }

  AssertEqual(node_occupancies[num_nodes_-1], total_occ, 1.0e-9);
  // If needed, generate a transform at the root.
  if (generate_xform[num_nodes_-1] && regclasses[num_nodes_-1] < 0) {
    KALDI_ASSERT(node_occupancies[num_nodes_-1] >= min_count);
    regclasses[num_nodes_-1] = num_regclasses++;
  }

  // Initialize the accumulators for output stats.
  // NOTE: memory is allocated here; be careful to delete the pointers
  stats_out->resize(num_regclasses);
  for (int32 r = 0; r < num_regclasses; r++) {
    (*stats_out)[r] = new AffineXformStats();
    (*stats_out)[r]->Init(stats_in[0]->dim_, stats_in[0]->G_.size());
  }

  // Finally go through the tree again and add stats
  vector<int32> active_parents;
  for (int32 bclass = 0; bclass < num_baseclasses_; bclass++) {
    if (generate_xform[bclass]) {
      KALDI_ASSERT(regclasses[bclass] > -1);
      (*stats_out)[regclasses[bclass]]->CopyStats(*(stats_in[bclass]));
      (*regclasses_out)[bclass] = regclasses[bclass];
      if (GetActiveParents(bclass, parents_, generate_xform, &active_parents)) {
        // Some other baseclass has less count
        for (vector<int32>::const_iterator p = active_parents.begin(),
            endp = active_parents.end(); p != endp; ++p) {
          KALDI_ASSERT(regclasses[*p] > -1);
          (*stats_out)[regclasses[*p]]->Add(*(stats_in[bclass]));
        }
      }
    } else {
      bool found = GetActiveParents(bclass, parents_, generate_xform,
                                    &active_parents);
      KALDI_ASSERT(found);  // must have active parents
      for (vector<int32>::const_iterator p = active_parents.begin(),
          endp = active_parents.end(); p != endp; ++p) {
        KALDI_ASSERT(regclasses[*p] > -1);
        (*stats_out)[regclasses[*p]]->Add(*(stats_in[bclass]));
      }
      (*regclasses_out)[bclass] = regclasses[active_parents[0]];
    }
  }

  KALDI_ASSERT(num_regclasses <= num_baseclasses_);
  return true;
}

void RegressionTree::Write(std::ostream &out, bool binary) const {
  WriteToken(out, binary, "<REGTREE>");
  WriteToken(out, binary, "<NUMNODES>");
  WriteBasicType(out, binary, num_nodes_);
  if (!binary) out << '\n';
  WriteToken(out, binary, "<PARENTS>");
  if (!binary) out << '\n';
  WriteIntegerVector(out, binary, parents_);
  WriteToken(out, binary, "</PARENTS>");
  if (!binary) out << '\n';

  WriteToken(out, binary, "<BASECLASSES>");
  if (!binary) out << '\n';
  WriteToken(out, binary, "<NUMBASECLASSES>");
  WriteBasicType(out, binary, num_baseclasses_);
  if (!binary) out << '\n';
  for (int32 bclass = 0; bclass < num_baseclasses_; bclass++) {
    WriteToken(out, binary, "<CLASS>");
    WriteBasicType(out, binary, bclass);
    WriteBasicType(out, binary, static_cast<int32>(
        baseclasses_[bclass].size()));
    if (!binary) out << '\n';
    for (vector< pair<int32, int32> >::const_iterator
        it = baseclasses_[bclass].begin(), end = baseclasses_[bclass].end();
        it != end; it++) {
      WriteBasicType(out, binary, it->first);
      WriteBasicType(out, binary, it->second);
      if (!binary) out << '\n';
    }

    WriteToken(out, binary, "</CLASS>");
    if (!binary) out << '\n';
  }
  WriteToken(out, binary, "</BASECLASSES>");
  if (!binary) out << '\n';
}

void RegressionTree::Read(std::istream &in, bool binary,
                          const AmDiagGmm &am) {
  int32 total_gauss = 0;
  ExpectToken(in, binary, "<REGTREE>");
  ExpectToken(in, binary, "<NUMNODES>");
  ReadBasicType(in, binary, &num_nodes_);
  KALDI_ASSERT(num_nodes_ > 0);
  parents_.resize(static_cast<size_t>(num_nodes_));
  ExpectToken(in, binary, "<PARENTS>");
  ReadIntegerVector(in, binary, &parents_);
  ExpectToken(in, binary, "</PARENTS>");

  ExpectToken(in, binary, "<BASECLASSES>");
  ExpectToken(in, binary, "<NUMBASECLASSES>");
  ReadBasicType(in, binary, &num_baseclasses_);
  KALDI_ASSERT(num_baseclasses_ >0);
  baseclasses_.resize(static_cast<size_t>(num_baseclasses_));
  for (int32 bclass = 0; bclass < num_baseclasses_; bclass++) {
    ExpectToken(in, binary, "<CLASS>");
    int32 class_id, num_comp, pdf_id, gauss_id;
    ReadBasicType(in, binary, &class_id);
    ReadBasicType(in, binary, &num_comp);
    KALDI_ASSERT(class_id == bclass && num_comp > 0);
    total_gauss += num_comp;
    baseclasses_[bclass].reserve(num_comp);

    for (int32 i = 0; i < num_comp; i++) {
      ReadBasicType(in, binary, &pdf_id);
      ReadBasicType(in, binary, &gauss_id);
      KALDI_ASSERT(pdf_id >= 0 && gauss_id >= 0);
      baseclasses_[bclass].push_back(std::make_pair(pdf_id, gauss_id));
    }

    ExpectToken(in, binary, "</CLASS>");
  }
  ExpectToken(in, binary, "</BASECLASSES>");

  if (total_gauss != am.NumGauss())
    KALDI_ERR << "Expecting " << am.NumGauss() << " Gaussians in "
        "regression tree, found " << total_gauss;
  MakeGauss2Bclass(am);
}

void RegressionTree::MakeGauss2Bclass(const AmDiagGmm &am) {
  gauss2bclass_.resize(am.NumPdfs());
  for (int32 pdf_index = 0, num_pdfs = am.NumPdfs(); pdf_index < num_pdfs;
      ++pdf_index) {
    gauss2bclass_[pdf_index].resize(am.NumGaussInPdf(pdf_index));
  }

  int32 total_gauss = 0;
  for (int32 bclass_index = 0; bclass_index < num_baseclasses_;
      ++bclass_index) {
    vector< pair<int32, int32> >::const_iterator itr =
        baseclasses_[bclass_index].begin(), end =
            baseclasses_[bclass_index].end();
    for (; itr != end; ++itr) {
      KALDI_ASSERT(itr->first < am.NumPdfs() &&
          itr->second < am.NumGaussInPdf(itr->first));
      gauss2bclass_[itr->first][itr->second] = bclass_index;
      total_gauss++;
    }
  }

  if (total_gauss != am.NumGauss())
    KALDI_ERR << "Expecting " << am.NumGauss() << " Gaussians in "
        "regression tree, found " << total_gauss;
}

}  // namespace kaldi

