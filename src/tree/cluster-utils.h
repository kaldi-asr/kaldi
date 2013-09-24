// tree/cluster-utils.h

// Copyright 2012   Arnab Ghoshal
// Copyright 2009-2011  Microsoft Corporation;  Saarland University

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

#ifndef KALDI_TREE_CLUSTER_UTILS_H_
#define KALDI_TREE_CLUSTER_UTILS_H_

#include <vector>
#include "matrix/matrix-lib.h"
#include "itf/clusterable-itf.h"

namespace kaldi {

/// \addtogroup clustering_group_simple
/// @{

/// Returns the total objective function after adding up all the
/// statistics in the vector (pointers may be NULL).
BaseFloat SumClusterableObjf(const std::vector<Clusterable*> &vec);

/// Returns the total normalizer (usually count) of the cluster (pointers may be NULL).
BaseFloat SumClusterableNormalizer(const std::vector<Clusterable*> &vec);

/// Sums stats (ptrs may be NULL). Returns NULL if no non-NULL stats present.
Clusterable *SumClusterable(const std::vector<Clusterable*> &vec);

/** Fills in any (NULL) holes in "stats" vector, with empty stats, because
 *  certain algorithms require non-NULL stats.  If "stats" nonempty, requires it
 *  to contain at least one non-NULL pointer that we can call Copy() on.
 */
void EnsureClusterableVectorNotNull(std::vector<Clusterable*> *stats);


/** Given stats and a vector "assignments" of the same size (that maps to
 * cluster indices), sums the stats up into "clusters."  It will add to any
 * stats already present in "clusters" (although typically "clusters" will be
 * empty when called), and it will extend with NULL pointers for any unseen
 * indices. Call EnsureClusterableStatsNotNull afterwards if you want to ensure
 * all non-NULL clusters. Pointer in "clusters" are owned by caller. Pointers in
 * "stats" do not have to be non-NULL.
 */
void AddToClusters(const std::vector<Clusterable*> &stats,
                   const std::vector<int32> &assignments,
                   std::vector<Clusterable*> *clusters);


/// AddToClustersOptimized does the same as AddToClusters (it sums up the stats
/// within each cluster, except it uses the sum of all the stats ("total") to
/// optimize the computation for speed, if possible.  This will generally only be
/// a significant speedup in the case where there are just two clusters, which
/// can happen in algorithms that are doing binary splits; the idea is that we
/// sum up all the stats in one cluster (the one with the fewest points in it),
/// and then subtract from the total.
void AddToClustersOptimized(const std::vector<Clusterable*> &stats,
                            const std::vector<int32> &assignments,
                            const Clusterable &total,
                            std::vector<Clusterable*> *clusters);

/// @} end "addtogroup clustering_group_simple"

/// \addtogroup clustering_group_algo
/// @{

// Note, in the algorithms below, it is assumed that the input "points" (which
// is std::vector<Clusterable*>) is all non-NULL.

/** A bottom-up clustering algorithm. There are two parameters that control how
 *  many clusters we get: a "max_merge_thresh" which is a threshold for merging
 *  clusters, and a min_clust which puts a floor on the number of clusters we want. Set
 *  max_merge_thresh = large to use the min_clust only, or min_clust to 0 to use
 *  the max_merge_thresh only.
 *
 *  The algorithm is:
 *  \code
 *      while (num-clusters > min_clust && smallest_merge_cost <= max_merge_thresh)
 *          merge the closest two clusters.
 *  \endcode
 *
 *  @param points [in] Points to be clustered (may not contain NULL pointers)
 *  @param thresh [in] Threshold on cost change from merging clusters; clusters
 *               won't be merged if the cost is more than this
 *  @param min_clust [in] Minimum number of clusters desired; we'll stop merging
 *                  after reaching this number.
 *  @param clusters_out [out] If non-NULL, will be set to a vector of size equal
 *                 to the number of output clusters, containing the clustered
 *                 statistics.  Must be empty when called.
 *  @param assignments_out [out] If non-NULL, will be resized to the number of
 *                 points, and each element is the index of the cluster that point
 *                 was assigned to.
 *  @return Returns the total objf change relative to all clusters being separate, which is
 *    a negative.  Note that this is not the same as what the other clustering algorithms return.
 */
BaseFloat ClusterBottomUp(const std::vector<Clusterable*> &points,
                          BaseFloat thresh,
                          int32 min_clust,
                          std::vector<Clusterable*> *clusters_out,
                          std::vector<int32> *assignments_out);

/** This is a bottom-up clustering where the points are pre-clustered in a set
 *  of compartments, such that only points in the same compartment are clustered
 *  together. The compartment and pair of points with the smallest merge cost
 *  is selected and the points are clustered. The result stays in the same
 *  compartment. The code does not merge compartments, and hence assumes that
 *  the number of compartments is smaller than the 'min_clust' option.
 *  The clusters in "clusters_out" are newly allocated and owned by the caller.
 */
BaseFloat ClusterBottomUpCompartmentalized(
    const std::vector< std::vector<Clusterable*> > &points, BaseFloat thresh,
    int32 min_clust, std::vector< std::vector<Clusterable*> > *clusters_out,
    std::vector< std::vector<int32> > *assignments_out);


struct RefineClustersOptions {
  int32 num_iters;  // must be >= 0.  If zero, does nothing.
  int32 top_n;  // must be >= 2.
  RefineClustersOptions() : num_iters(100), top_n(5) {}
  RefineClustersOptions(int32 num_iters_in, int32 top_n_in)
      : num_iters(num_iters_in), top_n(top_n_in) {}
  // include Write and Read functions because this object gets written/read as
  // part of the QuestionsForKeyOptions class.
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};

/** RefineClusters is mainly used internally by other clustering algorithms.
 *
 *  It starts with a given assignment of points to clusters and
 *  keeps trying to improve it by moving points from cluster to cluster, up to
 *  a maximum number of iterations.
 *
 *  "clusters" and "assignments" are both input and output variables, and so
 *  both MUST be non-NULL.
 *
 *  "top_n" (>=2) is a pruning value: more is more exact, fewer is faster. The
 *  algorithm initially finds the "top_n" closest clusters to any given point,
 *  and from that point only consider move to those "top_n" clusters. Since
 *  RefineClusters is called multiple times from ClusterKMeans (for instance),
 *  this is not really a limitation.
 */
BaseFloat RefineClusters(const std::vector<Clusterable*> &points,
                         std::vector<Clusterable*> *clusters /*non-NULL*/,
                         std::vector<int32> *assignments /*non-NULL*/,
                         RefineClustersOptions cfg = RefineClustersOptions());

struct ClusterKMeansOptions {
  RefineClustersOptions refine_cfg;
  int32 num_iters;
  int32 num_tries;  // if >1, try whole procedure >once and pick best.
  bool verbose;
  ClusterKMeansOptions()
      : refine_cfg(), num_iters(20), num_tries(2), verbose(true)  {}
};

/** ClusterKMeans is a K-means-like clustering algorithm. It starts with
 *  pseudo-random initialization of points to clusters and uses RefineClusters
 *  to iteratively improve the cluster assignments.  It does this for
 *  multiple iterations and picks the result with the best objective function.
 *
 *
 *  ClusterKMeans implicitly uses rand(). It will not necessarily return
 *  the same value on different calls.  Use srand() if you want consistent
 *  results.
 *  The algorithm used in ClusterKMeans is a "k-means-like" algorithm that tries
 *  to be as efficient as possible.  Firstly, since the algorithm it uses
 *  includes random initialization, it tries the whole thing cfg.num_tries times
 *  and picks the one with the best objective function.  Each try, it does as
 *  follows: it randomly initializes points to clusters, and then for
 *  cfg.num_iters iterations it calls RefineClusters().  The options to
 *  RefineClusters() are given by cfg.refine_cfg.  Calling RefineClusters once
 *  will always be at least as good as doing one iteration of reassigning points to
 *  clusters, but will generally be quite a bit better (without taking too
 *  much extra time).
 *
 *  @param points [in]  points to be clustered (must be all non-NULL).
 *  @param num_clust [in] number of clusters requested (it will always return exactly
 *                 this many, or will fail if num_clust > points.size()).
 *  @param clusters_out [out] may be NULL; if non-NULL, should be empty when called.
 *          Will be set to a vector of statistics corresponding to the output clusters.
 *  @param assignments_out [out] may be NULL; if non-NULL, will be set to a vector of
 *             same size as "points", which says for each point which cluster
 *              it is assigned to.
 *  @param cfg [in] configuration class specifying options to the algorithm.
 *  @return Returns the objective function improvement versus everything being
 *     in the same cluster.
 *
 */
BaseFloat ClusterKMeans(const std::vector<Clusterable*> &points,
                        int32 num_clust,  // exact number of clusters
                        std::vector<Clusterable*> *clusters_out,  // may be NULL
                        std::vector<int32> *assignments_out,  // may be NULL
                        ClusterKMeansOptions cfg = ClusterKMeansOptions());

struct TreeClusterOptions  {
  ClusterKMeansOptions kmeans_cfg;
  int32 branch_factor;
  BaseFloat thresh;  // Objf change: if >0, may be used to control number of leaves.
  TreeClusterOptions()
      : kmeans_cfg(), branch_factor(2), thresh(0) {
    kmeans_cfg.verbose = false;
  }
};

/** TreeCluster is a top-down clustering algorithm, using a binary tree (not
 *  necessarily balanced). Returns objf improvement versus having all points
 *  in one cluster.  The algorithm is:
 *     - Initialize to 1 cluster (tree with 1 node).
 *     - Maintain, for each cluster, a "best-binary-split" (using ClusterKMeans
 *       to do so). Always split the highest scoring cluster, until we can do no
 *       more splits.
 *
 *  @param points [in] Data points to be clustered
 *  @param max_clust  [in] Maximum number of clusters (you will get exactly this number,
 *                if there are at least this many points, except if you set the
 *                cfg.thresh value nonzero, in which case that threshold may limit
 *                the number of clusters.
 *  @param clusters_out [out] If non-NULL, will be set to the a vector whose first
 *                (*num_leaves_out) elements are the leaf clusters, and whose
 *                subsequent elements are the nonleaf nodes in the tree, in
 *                topological order with the root node last.  Must be empty vector
 *                when this function is called.
 *  @param assignments_out [out] If non-NULL, will be set to a vector to a vector the
 *               same size as "points", where assignments[i] is the leaf node index i
 *               to which the i'th point gets clustered.
 *  @param clust_assignments_out [out] If non-NULL, will be set to a vector the same size
 *                as clusters_out  which says for each node (leaf or nonleaf), the
 *                index of its parent.  For the root node (which is last),
 *                assignments_out[i] == i.  For each i, assignments_out[i]>=i, i.e.
 *                any node's parent is higher numbered than itself.  If you don't need
 *                this information, consider using instead the ClusterTopDown function.
 *  @param num_leaves_out [out] If non-NULL, will be set to the number of leaf nodes
 *                in the tree.
 *  @param cfg [in] Configuration object that controls clustering behavior.  Most
 *                 important value is "thresh", which provides an alternative mechanism
 *                 [other than max_clust] to limit the number of leaves.
 */
BaseFloat TreeCluster(const std::vector<Clusterable*> &points,
                      int32 max_clust,  // max number of leaf-level clusters.
                      std::vector<Clusterable*> *clusters_out,
                      std::vector<int32> *assignments_out,
                      std::vector<int32> *clust_assignments_out,
                      int32 *num_leaves_out,
                      TreeClusterOptions cfg = TreeClusterOptions());


/**
 *  A clustering algorithm that internally uses TreeCluster,
 *  but does not give you the information about the structure of the tree.
 *  The "clusters_out" and "assignments_out" may be NULL if the outputs are not
 *  needed.
 *
 *  @param points [in]  points to be clustered (must be all non-NULL).
 *  @param max_clust [in] Maximum number of clusters (you will get exactly this number,
 *                if there are at least this many points, except if you set the
 *                cfg.thresh value nonzero, in which case that threshold may limit
 *                the number of clusters.
 *  @param clusters_out [out] may be NULL; if non-NULL, should be empty when called.
 *           Will be set to a vector of statistics corresponding to the output clusters.
 *  @param assignments_out [out] may be NULL; if non-NULL, will be set to a vector of
 *           same size as "points", which says for each point which cluster
 *            it is assigned to.
 *  @param cfg [in] Configuration object that controls clustering behavior.  Most
 *                important value is "thresh", which provides an alternative mechanism
 *                [other than max_clust] to limit the number of leaves.
*/
BaseFloat ClusterTopDown(const std::vector<Clusterable*> &points,
                         int32 max_clust,  // max number of clusters.
                         std::vector<Clusterable*> *clusters_out,
                         std::vector<int32> *assignments_out,
                         TreeClusterOptions cfg = TreeClusterOptions());

/// @} end of "addtogroup clustering_group_algo"

}  // end namespace kaldi.

#endif  // KALDI_TREE_CLUSTER_UTILS_H_
