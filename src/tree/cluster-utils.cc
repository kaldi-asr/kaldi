// tree/cluster-utils.cc

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

#include <functional>
#include <queue>
#include <vector>
using std::vector;

#include "base/kaldi-math.h"
#include "util/stl-utils.h"
#include "tree/cluster-utils.h"

namespace kaldi {

typedef uint16 uint_smaller;
typedef int16 int_smaller;

// ============================================================================
// Some convenience functions used in the clustering routines
// ============================================================================

BaseFloat SumClusterableObjf(const std::vector<Clusterable*> &vec) {
  BaseFloat ans = 0.0;
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i] != NULL) {
      BaseFloat objf = vec[i]->Objf();
      if (KALDI_ISNAN(objf)) {
        KALDI_WARN << "SumClusterableObjf, NaN objf";
      } else {
        ans += objf;
      }
    }
  }
  return ans;
}

BaseFloat SumClusterableNormalizer(const std::vector<Clusterable*> &vec) {
  BaseFloat ans = 0.0;
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i] != NULL) {
      BaseFloat objf = vec[i]->Normalizer();
      if (KALDI_ISNAN(objf)) {
        KALDI_WARN << "SumClusterableObjf, NaN objf";
      } else {
        ans += objf;
      }
    }
  }
  return ans;
}

Clusterable* SumClusterable(const std::vector<Clusterable*> &vec) {
  Clusterable *ans = NULL;
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i] != NULL) {
      if (ans == NULL)
        ans = vec[i]->Copy();
      else
        ans->Add(*(vec[i]));
    }
  }
  return ans;
}

void EnsureClusterableVectorNotNull(std::vector<Clusterable*> *stats) {
  KALDI_ASSERT(stats != NULL);
  std::vector<Clusterable*>::iterator itr = stats->begin(), end = stats->end();
  if (itr == end) return;  // Nothing to do.
  Clusterable *nonNullExample = NULL;
  for (; itr != end; ++itr) {
    if (*itr != NULL) {
      nonNullExample = *itr;
      break;
    }
  }
  if (nonNullExample == NULL) {
    KALDI_ERR << "All stats are NULL.";  // crash. logic error.
  }
  itr = stats->begin();
  Clusterable *nonNullExampleCopy = nonNullExample->Copy();
  // sets stats to zero. do this just once (on copy) for efficiency.
  nonNullExampleCopy->SetZero();
  for (; itr != end; ++itr) {
    if (*itr == NULL) {
      *itr = nonNullExampleCopy->Copy();
    }
  }
  delete nonNullExampleCopy;
}

void AddToClusters(const std::vector<Clusterable*> &stats,
                   const std::vector<int32> &assignments,
                   std::vector<Clusterable*> *clusters) {
  KALDI_ASSERT(assignments.size() == stats.size());
  int32 size = stats.size();
  if (size == 0) return;  // Nothing to do.
  KALDI_ASSERT(clusters != NULL);
  int32 max_assignment = *std::max_element(assignments.begin(),
                                           assignments.end());
  if (static_cast<int32> (clusters->size()) <= max_assignment)
    clusters->resize(max_assignment + 1, NULL);  // extend with NULLs.
  for (int32 i = 0; i < size; i++) {
    if (stats[i] != NULL) {
      if ((*clusters)[assignments[i]] == NULL)
        (*clusters)[assignments[i]] = stats[i]->Copy();
      else
        (*clusters)[assignments[i]]->Add(*(stats[i]));
    }
  }
}


void AddToClustersOptimized(const std::vector<Clusterable*> &stats,
                            const std::vector<int32> &assignments,
                            const Clusterable &total,
                            std::vector<Clusterable*> *clusters) {
#ifdef KALDI_PARANOID
  // Make sure "total" is actually the sum of stats in "stats".
  {
    BaseFloat stats_norm = SumClusterableNormalizer(stats),
    tot_norm = total.Normalizer();
    AssertEqual(stats_norm, tot_norm, 0.01);
  }
#endif

  KALDI_ASSERT(assignments.size() == stats.size());
  int32 size = stats.size();
  if (size == 0) return;  // Nothing to do.
  KALDI_ASSERT(clusters != NULL);
  int32 num_clust = 1 + *std::max_element(assignments.begin(),
                                          assignments.end());
  if (static_cast<int32> (clusters->size()) < num_clust)
    clusters->resize(num_clust, NULL);  // extend with NULLs.
  std::vector<int32> num_stats_for_cluster(num_clust, 0);
  int32 num_total_stats = 0;
  for (int32 i = 0; i < size; i++) {
    if (stats[i] != NULL) {
      num_total_stats++;
      num_stats_for_cluster[assignments[i]]++;
    }
  }
  if (num_total_stats == 0) return;  // Nothing to do.

  //  it will only ever be efficient to subtract for at most one index.
  int32 subtract_index = -1;
  for (int32 c = 0; c < num_clust; c++) {
    if (num_stats_for_cluster[c] > num_total_stats - num_stats_for_cluster[c]) {
      subtract_index = c;
      if ((*clusters)[c] == NULL)
        (*clusters)[c] = total.Copy();
      else
        (*clusters)[c]->Add(total);
      break;
    }
  }

  for (int32 i = 0; i < size; i++) {
    if (stats[i] != NULL) {
      int32 assignment = assignments[i];
      if (assignment != (int32) subtract_index) {
        if ((*clusters)[assignment] == NULL)
          (*clusters)[assignment] = stats[i]->Copy();
        else
          (*clusters)[assignment]->Add(*(stats[i]));
      }
      if (subtract_index != -1 && assignment != subtract_index)
        (*clusters)[subtract_index]->Sub(*(stats[i]));
    }
  }
}

// ============================================================================
// Bottom-up clustering routines
// ============================================================================

class BottomUpClusterer {
 public:
  BottomUpClusterer(const std::vector<Clusterable*> &points,
                    BaseFloat max_merge_thresh,
                    int32 min_clust,
                    std::vector<Clusterable*> *clusters_out,
                    std::vector<int32> *assignments_out)
      : ans_(0.0), points_(points), max_merge_thresh_(max_merge_thresh),
        min_clust_(min_clust), clusters_(clusters_out != NULL? clusters_out
            : &tmp_clusters_), assignments_(assignments_out != NULL ?
                assignments_out : &tmp_assignments_) {
    nclusters_ = npoints_ = points.size();
    dist_vec_.resize((npoints_ * (npoints_ - 1)) / 2);
  }

  BaseFloat Cluster();
  ~BottomUpClusterer() { DeletePointers(&tmp_clusters_); }

 private:
  void Renumber();
  void InitializeAssignments();
  void SetInitialDistances();  ///< Sets up distances and queue.
  /// CanMerge returns true if i and j are existing clusters, and the distance
  /// (negated objf-change) "dist" is accurate (i.e. not outdated).
  bool CanMerge(int32 i, int32 j, BaseFloat dist);
  /// Merge j into i and delete j.
  void MergeClusters(int32 i, int32 j);
  /// Reconstructs the priority queue from the distances.
  void ReconstructQueue();

  void SetDistance(int32 i, int32 j);
  BaseFloat& Distance(int32 i, int32 j) {
    KALDI_ASSERT(i < npoints_ && j < i);
    return dist_vec_[(i * (i - 1)) / 2 + j];
  }

  BaseFloat ans_;
  const std::vector<Clusterable*> &points_;
  BaseFloat max_merge_thresh_;
  int32 min_clust_;
  std::vector<Clusterable*> *clusters_;
  std::vector<int32> *assignments_;

  std::vector<Clusterable*> tmp_clusters_;
  std::vector<int32> tmp_assignments_;

  std::vector<BaseFloat> dist_vec_;
  int32 nclusters_;
  int32 npoints_;
  typedef std::pair<BaseFloat, std::pair<uint_smaller, uint_smaller> > QueueElement;
  // Priority queue using greater (lowest distances are highest priority).
  typedef std::priority_queue<QueueElement, std::vector<QueueElement>,
      std::greater<QueueElement>  > QueueType;
  QueueType queue_;
};

BaseFloat BottomUpClusterer::Cluster() {
  KALDI_VLOG(2) << "Initializing cluster assignments.";
  InitializeAssignments();
  KALDI_VLOG(2) << "Setting initial distances.";
  SetInitialDistances();

  KALDI_VLOG(2) << "Clustering...";
  while (nclusters_ > min_clust_ && !queue_.empty()) {
    std::pair<BaseFloat, std::pair<uint_smaller, uint_smaller> > pr = queue_.top();
    BaseFloat dist = pr.first;
    int32 i = (int32) pr.second.first, j = (int32) pr.second.second;
    queue_.pop();
    if (CanMerge(i, j, dist)) MergeClusters(i, j);
  }
  KALDI_VLOG(2) << "Renumbering clusters to contiguous numbers.";
  Renumber();
  return ans_;
}

void BottomUpClusterer::Renumber() {
  KALDI_VLOG(2) << "Freeing up distance vector.";
  {
    vector<BaseFloat> tmp;
    tmp.swap(dist_vec_);
  }

// Commented the following out since it was causing the process to take up too
// much memory with larger models. While the swap() method of STL types swaps 
// the data pointers, std::swap() creates a temporary copy. -Arnab
//  KALDI_VLOG(2) << "Freeing up the queue";
//   // first free up memory by getting rid of queue.  this is a special trick
//   // to force STL to free memory.
//  {
//    QueueType tmp;
//    std::swap(tmp, queue_);
//  }

  // called after clustering, renumbers to make clusters contiguously
  // numbered. also processes assignments_ to remove chains of references.
  KALDI_VLOG(2) << "Creating new copy of non-NULL clusters.";
  std::vector<uint_smaller> mapping(npoints_, static_cast<uint_smaller> (-1));  // mapping from intermediate to final clusters.
  std::vector<Clusterable*> new_clusters(nclusters_);
  int32 clust = 0;
  for (int32 i = 0; i < npoints_; i++) {
    if ((*clusters_)[i] != NULL) {
      KALDI_ASSERT(clust < nclusters_);
      new_clusters[clust] = (*clusters_)[i];
      mapping[i] = clust;
      clust++;
    }
  }
  KALDI_ASSERT(clust == nclusters_);

  KALDI_VLOG(2) << "Creating new copy of assignments.";
  std::vector<int32> new_assignments(npoints_);
  for (int32 i = 0; i < npoints_; i++) {  // now reprocess assignments_.
    int32 ii = i;
    while ((*assignments_)[ii] != ii)
      ii = (*assignments_)[ii];  // follow the chain.
    KALDI_ASSERT((*clusters_)[ii] != NULL);  // cannot have assignment to nonexistent cluster.
    KALDI_ASSERT(mapping[ii] != static_cast<uint_smaller>(-1));
    new_assignments[i] = mapping[ii];
  }
  clusters_->swap(new_clusters);
  assignments_->swap(new_assignments);
}

void BottomUpClusterer::InitializeAssignments() {
  clusters_->resize(npoints_);
  assignments_->resize(npoints_);
  for (int32 i = 0; i < npoints_; i++) {  // initialize as 1-1 mapping.
    (*clusters_)[i] = points_[i]->Copy();
    (*assignments_)[i] = i;
  }
}

void BottomUpClusterer::SetInitialDistances() {
  for (int32 i = 0; i < npoints_; i++) {
    for (int32 j = 0; j < i; j++) {
      BaseFloat dist = (*clusters_)[i]->Distance(*((*clusters_)[j]));
      dist_vec_[(i * (i - 1)) / 2 + j] = dist;
      if (dist <= max_merge_thresh_)
        queue_.push(std::make_pair(dist, std::make_pair(static_cast<uint_smaller>(i),
            static_cast<uint_smaller>(j))));
    }
  }
}

bool BottomUpClusterer::CanMerge(int32 i, int32 j, BaseFloat dist) {
  KALDI_ASSERT(i != j && i < npoints_ && j < npoints_);
  if ((*clusters_)[i] == NULL || (*clusters_)[j] == NULL)
    return false;
  BaseFloat cached_dist = dist_vec_[(i * (i - 1)) / 2 + j];
  return (std::fabs(cached_dist - dist) <= 1.0e-05 * std::fabs(dist));
}

void BottomUpClusterer::MergeClusters(int32 i, int32 j) {
  KALDI_ASSERT(i != j && i < npoints_ && j < npoints_);
  (*clusters_)[i]->Add(*((*clusters_)[j]));
  delete (*clusters_)[j];
  (*clusters_)[j] = NULL;
  // note that we may have to follow the chain within "assignment_" to get
  // final assignments.
  (*assignments_)[j] = i;
  // subtract negated objective function change, i.e. add objective function
  // change.
  ans_ -= dist_vec_[(i * (i - 1)) / 2 + j];
  nclusters_--;
  // Now update "distances".
  for (int32 k = 0; k < npoints_; k++) {
    if (k != i && (*clusters_)[k] != NULL) {
      if (k < i)
        SetDistance(i, k);  // SetDistance requires k < i.
      else
        SetDistance(k, i);
    }
  }
}

void BottomUpClusterer::ReconstructQueue() {
  // empty queue [since there is no clear()]
  {
    QueueType tmp;
    std::swap(tmp, queue_);
  }
  for (int32 i = 0; i < npoints_; i++) {
    if ((*clusters_)[i] != NULL) {
      for (int32 j = 0; j < i; j++) {
        if ((*clusters_)[j] != NULL) {
          BaseFloat dist = dist_vec_[(i * (i - 1)) / 2 + j];
          if (dist <= max_merge_thresh_) {
            queue_.push(std::make_pair(dist, std::make_pair(
                static_cast<uint_smaller>(i), static_cast<uint_smaller>(j))));
          }
        }
      }
    }
  }
}

void BottomUpClusterer::SetDistance(int32 i, int32 j) {
  KALDI_ASSERT(i < npoints_ && j < i && (*clusters_)[i] != NULL
         && (*clusters_)[j] != NULL);
  BaseFloat dist = (*clusters_)[i]->Distance(*((*clusters_)[j]));
  dist_vec_[(i * (i - 1)) / 2 + j] = dist;  // set the distance in the array.
  if (dist < max_merge_thresh_) {
    queue_.push(std::make_pair(dist, std::make_pair(static_cast<uint_smaller>(i),
        static_cast<uint_smaller>(j))));
  }
  // every time it's at least twice the maximum possible size.
  if (queue_.size() >= static_cast<size_t> (npoints_ * npoints_)) {
    // Control memory use by getting rid of orphaned queue entries
    ReconstructQueue();
  }
}



BaseFloat ClusterBottomUp(const std::vector<Clusterable*> &points,
                          BaseFloat max_merge_thresh,
                          int32 min_clust,
                          std::vector<Clusterable*> *clusters_out,
                          std::vector<int32> *assignments_out) {
  KALDI_ASSERT(max_merge_thresh >= 0.0 && min_clust >= 0);
  KALDI_ASSERT(!ContainsNullPointers(points));
  int32 npoints = points.size();
  // make sure fits in uint_smaller and does not hit the -1 which is reserved.
  KALDI_ASSERT(sizeof(uint_smaller)==sizeof(uint32) ||
               npoints < static_cast<int32>(static_cast<uint_smaller>(-1)));

  KALDI_VLOG(2) << "Initializing clustering object.";
  BottomUpClusterer bc(points, max_merge_thresh, min_clust, clusters_out, assignments_out);
  BaseFloat ans = bc.Cluster();
  if (clusters_out) KALDI_ASSERT(!ContainsNullPointers(*clusters_out));
  return ans;
}


// ============================================================================
// Compartmentalized bottom-up clustering routines
// ============================================================================

struct CompBotClustElem {
  BaseFloat dist;
  int32 compartment, point1, point2;
  CompBotClustElem(BaseFloat d, int32 comp, int32 i, int32 j)
      : dist(d), compartment(comp), point1(i), point2(j) {}
};

bool operator > (const CompBotClustElem &a, const CompBotClustElem &b) {
  return a.dist > b.dist;
}

class CompartmentalizedBottomUpClusterer {
 public:
  CompartmentalizedBottomUpClusterer(
      const vector< vector<Clusterable*> > &points, BaseFloat max_merge_thresh,
      int32 min_clust)
      : points_(points), max_merge_thresh_(max_merge_thresh),
        min_clust_(min_clust) {
    ncompartments_ = points.size();
    nclusters_ = 0;
    npoints_.resize(ncompartments_);
    for (int32 comp = 0; comp < ncompartments_; comp++) {
      npoints_[comp] = points[comp].size();
      nclusters_ += npoints_[comp];
    }
  }
  BaseFloat Cluster(vector< vector<Clusterable*> > *clusters_out,
                    vector< vector<int32> > *assignments_out);
  ~CompartmentalizedBottomUpClusterer() {
    for (vector< vector<Clusterable*> >::iterator itr = clusters_.begin(),
         end = clusters_.end(); itr != end; ++itr)
      DeletePointers(&(*itr));
  }

 private:
  // Renumbers to make clusters contiguously numbered. Called after clustering.
  // Also processes assignments_ to remove chains of references.
  void Renumber(int32 compartment);
  void InitializeAssignments();
  void SetInitialDistances();  ///< Sets up distances and queue.
  /// CanMerge returns true if i and j are existing clusters, and the distance
  /// (negated objf-change) "dist" is accurate (i.e. not outdated).
  bool CanMerge(int32 compartment, int32 i, int32 j, BaseFloat dist);
  /// Merge j into i and delete j. Returns obj function change.
  BaseFloat MergeClusters(int32 compartment, int32 i, int32 j);
  /// Reconstructs the priority queue from the distances.
  void ReconstructQueue();

  void SetDistance(int32 compartment, int32 i, int32 j);

  const vector< vector<Clusterable*> > &points_;
  BaseFloat max_merge_thresh_;
  int32 min_clust_;
  vector< vector<Clusterable*> > clusters_;
  vector< vector<int32> > assignments_;
  vector< vector<BaseFloat> > dist_vec_;
  int32 ncompartments_, nclusters_;
  vector<int32> npoints_;
  // Priority queue using greater (lowest distances are highest priority).
  typedef std::priority_queue< CompBotClustElem, std::vector<CompBotClustElem>,
      std::greater<CompBotClustElem> > QueueType;
  QueueType queue_;
};

BaseFloat CompartmentalizedBottomUpClusterer::Cluster(
    vector< vector<Clusterable*> > *clusters_out,
    vector< vector<int32> > *assignments_out) {
  InitializeAssignments();
  SetInitialDistances();
  BaseFloat total_obj_change = 0;

  while (nclusters_ > min_clust_ && !queue_.empty()) {
    CompBotClustElem qelem = queue_.top();
    queue_.pop();
    if (CanMerge(qelem.compartment, qelem.point1, qelem.point2, qelem.dist))
      total_obj_change += MergeClusters(qelem.compartment, qelem.point1,
                                        qelem.point2);
  }
  for (int32 comp = 0; comp < ncompartments_; comp++)
    Renumber(comp);
  if (clusters_out != NULL) clusters_out->swap(clusters_); 
  if (assignments_out != NULL) assignments_out->swap(assignments_);
  return total_obj_change;
}

void CompartmentalizedBottomUpClusterer::Renumber(int32 comp) {
  // first free up memory by getting rid of queue.  this is a special trick
  // to force STL to free memory.
  {
    QueueType tmp;
    std::swap(tmp, queue_);
  }

  // First find the number of surviving clusters in the compartment.
  int32 clusts_in_compartment = 0;
  for (int32 i = 0; i < npoints_[comp]; i++) {
    if (clusters_[comp][i] != NULL)
      clusts_in_compartment++;
  }
  KALDI_ASSERT(clusts_in_compartment <= nclusters_);

  // mapping from intermediate to final clusters.
  vector<uint_smaller> mapping(npoints_[comp], static_cast<uint_smaller> (-1));
  vector<Clusterable*> new_clusters(clusts_in_compartment);

  // Now copy the surviving clusters in a fresh array.
  clusts_in_compartment = 0;
  for (int32 i = 0; i < npoints_[comp]; i++) {
    if (clusters_[comp][i] != NULL) {
      new_clusters[clusts_in_compartment] = clusters_[comp][i];
      mapping[i] = clusts_in_compartment;
      clusts_in_compartment++;
    }
  }

  // Next, process the assignments.
  std::vector<int32> new_assignments(npoints_[comp]);
  for (int32 i = 0; i < npoints_[comp]; i++) {
    int32 ii = i;
    while (assignments_[comp][ii] != ii)
      ii = assignments_[comp][ii];  // follow the chain.
    // cannot assign to nonexistent cluster.
    KALDI_ASSERT(clusters_[comp][ii] != NULL);
    KALDI_ASSERT(mapping[ii] != static_cast<uint_smaller>(-1));
    new_assignments[i] = mapping[ii];
  }
  clusters_[comp].swap(new_clusters);
  assignments_[comp].swap(new_assignments);
}

void CompartmentalizedBottomUpClusterer::InitializeAssignments() {
  clusters_.resize(ncompartments_);
  assignments_.resize(ncompartments_);
  for (int32 comp = 0; comp < ncompartments_; comp++) {
    clusters_[comp].resize(npoints_[comp]);
    assignments_[comp].resize(npoints_[comp]);
    for (int32 i = 0; i < npoints_[comp]; i++) {  // initialize as 1-1 mapping.
      clusters_[comp][i] = points_[comp][i]->Copy();
      assignments_[comp][i] = i;
    }
  }
}

void CompartmentalizedBottomUpClusterer::SetInitialDistances() {
  dist_vec_.resize(ncompartments_);
  for (int32 comp = 0; comp < ncompartments_; comp++) {
    dist_vec_[comp].resize((npoints_[comp] * (npoints_[comp] - 1)) / 2);
    for (int32 i = 0; i < npoints_[comp]; i++)
      for (int32 j = 0; j < i; j++)
        SetDistance(comp, i, j);
  }
}

bool CompartmentalizedBottomUpClusterer::CanMerge(int32 comp, int32 i, int32 j,
                                                  BaseFloat dist) {
  KALDI_ASSERT(comp < ncompartments_ && i < npoints_[comp] && j < i);
  if (clusters_[comp][i] == NULL || clusters_[comp][j] == NULL)
    return false;
  BaseFloat cached_dist = dist_vec_[comp][(i * (i - 1)) / 2 + j];
  return (std::fabs(cached_dist - dist) <= 1.0e-05 * std::fabs(dist));
}

BaseFloat CompartmentalizedBottomUpClusterer::MergeClusters(int32 comp, int32 i,
                                                            int32 j) {
  KALDI_ASSERT(comp < ncompartments_ && i < npoints_[comp] && j < i);
  clusters_[comp][i]->Add(*(clusters_[comp][j]));
  delete clusters_[comp][j];
  clusters_[comp][j] = NULL;
  // note that we may have to follow the chain within "assignment_" to get
  // final assignments.
  assignments_[comp][j] = i;
  // objective function change.
  BaseFloat ans = -dist_vec_[comp][(i * (i - 1)) / 2 + j];
  nclusters_--;
  // Now update "distances".
  for (int32 k = 0; k < npoints_[comp]; k++) {
    if (k != i && clusters_[comp][k] != NULL) {
      if (k < i)
        SetDistance(comp, i, k);  // SetDistance requires k < i.
      else
        SetDistance(comp, k, i);
    }
  }
  // Control memory use by getting rid of orphaned queue entries every time
  // it's at least twice the maximum possible size.
  if (queue_.size() >= static_cast<size_t> (nclusters_ * nclusters_)) {
    ReconstructQueue();
  }
  return ans;
}

void CompartmentalizedBottomUpClusterer::ReconstructQueue() {
  // empty queue [since there is no clear()]
  {
    QueueType tmp;
    std::swap(tmp, queue_);
  }
  for (int32 comp = 0; comp < ncompartments_; comp++) {
    for (int32 i = 0; i < npoints_[comp]; i++) {
      if (clusters_[comp][i] == NULL) continue;
      for (int32 j = 0; j < i; j++) {
        if (clusters_[comp][j] == NULL) continue;
        SetDistance(comp, i, j);
      }
    }
  }
}

void CompartmentalizedBottomUpClusterer::SetDistance(int32 comp,
                                                     int32 i, int32 j) {
  KALDI_ASSERT(comp < ncompartments_ && i < npoints_[comp] && j < i);
  KALDI_ASSERT(clusters_[comp][i] != NULL && clusters_[comp][j] != NULL);
  BaseFloat dist = clusters_[comp][i]->Distance(*(clusters_[comp][j]));
  dist_vec_[comp][(i * (i - 1)) / 2 + j] = dist;
  if (dist < max_merge_thresh_) {
    queue_.push(CompBotClustElem(dist, comp, static_cast<uint_smaller>(i),
        static_cast<uint_smaller>(j)));
  }
}



BaseFloat ClusterBottomUpCompartmentalized(
    const std::vector< std::vector<Clusterable*> > &points, BaseFloat thresh,
    int32 min_clust, std::vector< std::vector<Clusterable*> > *clusters_out,
    std::vector< std::vector<int32> > *assignments_out) {
  KALDI_ASSERT(thresh >= 0.0 && min_clust >= 0);
  KALDI_ASSERT(min_clust >= points.size());  // Code does not merge compartments.
  int32 npoints = 0;
  for (vector< vector<Clusterable*> >::const_iterator itr = points.begin(),
           end = points.end(); itr != end; ++itr) {
    KALDI_ASSERT(!ContainsNullPointers(*itr));
    npoints += itr->size();
  }
  // make sure fits in uint_smaller and does not hit the -1 which is reserved.
  KALDI_ASSERT(sizeof(uint_smaller)==sizeof(uint32) ||
               npoints < static_cast<int32>(static_cast<uint_smaller>(-1)));

  CompartmentalizedBottomUpClusterer bc(points, thresh, min_clust);
  BaseFloat ans = bc.Cluster(clusters_out, assignments_out);
  if (clusters_out) {
    for (vector< vector<Clusterable*> >::iterator itr = clusters_out->begin(),
             end = clusters_out->end(); itr != end; ++itr) {
      KALDI_ASSERT(!ContainsNullPointers(*itr));
    }
  }
  return ans;
}


// ============================================================================
// Clustering through refinement routines
// ============================================================================

class RefineClusterer {

 public:
  // size used in point_info structure (we store a lot of these so don't want
  // to just make it int32). Also used as a time-id (cannot have more moves of
  // points, than can fit in this time). Must be big enough to store num-clust.
  typedef int32 LocalInt;
  typedef uint_smaller ClustIndexInt;

  RefineClusterer(const std::vector<Clusterable*> &points,
                  std::vector<Clusterable*> *clusters,
                  std::vector<int32> *assignments,
                  RefineClustersOptions cfg)
      : points_(points), clusters_(clusters), assignments_(assignments),
        cfg_(cfg) {
    KALDI_ASSERT(cfg_.top_n >= 2);
    num_points_ = points_.size();
    num_clust_ = static_cast<int32> (clusters->size());

    // so can fit clust-id in LocalInt
    if (cfg_.top_n > (int32) num_clust_) cfg_.top_n
        = static_cast<int32> (num_clust_);
    KALDI_ASSERT(cfg_.top_n == static_cast<int32>(static_cast<ClustIndexInt>(cfg_.top_n)));
    t_ = 0;
    my_clust_index_.resize(num_points_);
    // will set all PointInfo's to 0 too (they will be up-to-date).
    clust_time_.resize(num_clust_, 0);
    clust_objf_.resize(num_clust_);
    for (int32 i = 0; i < num_clust_; i++)
      clust_objf_[i] = (*clusters_)[i]->Objf();
    info_.resize(num_points_ * cfg_.top_n);
    ans_ = 0;
    InitPoints();
  }

  BaseFloat Refine() {
    if (cfg_.top_n <= 1) return 0.0;  // nothing to do.
    Iterate();
    return ans_;
  }
  // at some point check cfg_.top_n > 1 after maxing to num_clust_.
 private:
  void InitPoint(int32 point) {
    // Find closest clusters to this point.
    // distances are really negated objf changes, ignoring terms that don't vary with the "other" cluster.

    std::vector<std::pair<BaseFloat, LocalInt> > distances;
    distances.reserve(num_clust_-1);
    int32 my_clust = (*assignments_)[point];
    Clusterable *point_cl = points_[point];

    for (int32 clust = 0;clust < num_clust_;clust++) {
      if (clust != my_clust) {
        Clusterable *tmp = (*clusters_)[clust]->Copy();
        tmp->Add(*point_cl);
        BaseFloat other_clust_objf = clust_objf_[clust];
        BaseFloat other_clust_plus_me_objf = (*clusters_)[clust]->ObjfPlus(* (points_[point]));

        BaseFloat distance = other_clust_objf-other_clust_plus_me_objf;  // negated delta-objf, with only "varying" terms.
        distances.push_back(std::make_pair(distance, (LocalInt)clust));
        delete tmp;
      }
    }
    if ((cfg_.top_n-1-1) >= 0) {
      std::nth_element(distances.begin(), distances.begin()+(cfg_.top_n-1-1), distances.end());
    }
    // top_n-1 is the # of elements we want to retain.  -1 because we need the iterator
    // that points to the end of that range (i.e. not potentially off the end of the array).

    for (int32 index = 0;index < cfg_.top_n-1;index++) {
      point_info &info = GetInfo(point, index);
      int32 clust = distances[index].second;
      info.clust = clust;
      BaseFloat distance = distances[index].first;
      BaseFloat other_clust_objf = clust_objf_[clust];
      BaseFloat other_clust_plus_me_objf = -(distance - other_clust_objf);
      info.objf = other_clust_plus_me_objf;
      info.time = 0;
    }
    // now put the last element in, which is my current cluster.
    point_info &info = GetInfo(point, cfg_.top_n-1);
    info.clust = my_clust;
    info.time = 0;
    info.objf = (*clusters_)[my_clust]->ObjfMinus(*(points_[point]));
    my_clust_index_[point] = cfg_.top_n-1;
  }
  void InitPoints() {
    // finds, for each point, the closest cfg_.top_n clusters (including its own cluster).
    // this may be the most time-consuming step of the algorithm.
    for (int32 p = 0;p < num_points_;p++) InitPoint(p);
  }
  void Iterate() {
    int32 iter, num_iters = cfg_.num_iters;
    for (iter = 0;iter < num_iters;iter++) {
      int32 cur_t = t_;
      for (int32 point = 0;point < num_points_;point++) {
        if (t_+1 == 0) {
          KALDI_WARN << "Stopping iterating at int32 moves";
          return;  // once we use up all time points, must return-- this
                  // should rarely happen as int32 is large.
        }
        ProcessPoint(point);
      }
      if (t_ == cur_t) break;  // nothing changed so we converged.
    }
  }
  void MovePoint(int32 point, int32 new_index) {
    // move point to a different cluster.
    t_++;
    int32 old_index = my_clust_index_[point];  // index into info
    // array corresponding to current cluster.
    KALDI_ASSERT(new_index < cfg_.top_n  && new_index != old_index);
    point_info &old_info = GetInfo(point, old_index),
        &new_info = GetInfo(point, new_index);
    my_clust_index_[point] = new_index;  // update to new index.

    int32 old_clust = old_info.clust, new_clust = new_info.clust;
    KALDI_ASSERT( (*assignments_)[point] == old_clust);
    (*assignments_)[point] = new_clust;
    (*clusters_)[old_clust]->Sub( *(points_[point]) );
    (*clusters_)[new_clust]->Add( *(points_[point]) );
    UpdateClust(old_clust);
    UpdateClust(new_clust);
  }
  void UpdateClust(int32 clust) {
    KALDI_ASSERT(clust < num_clust_);
    clust_objf_[clust] = (*clusters_)[clust]->Objf();
    clust_time_[clust] = t_;
  }
  void ProcessPoint(int32 point) {
    // note: calling code uses the fact
    // that it only ever increases t_ by one.
    KALDI_ASSERT(point < num_points_);
    // (1) Make sure own-cluster like is updated.
    int32 self_index = my_clust_index_[point];  // index <cfg_.top_n of own cluster.
    point_info &self_info = GetInfo(point, self_index);
    int32 self_clust = self_info.clust;  // cluster index of own cluster.
    KALDI_ASSERT(self_index < cfg_.top_n);
    UpdateInfo(point, self_index);

    float own_clust_objf = clust_objf_[self_clust];
    float own_clust_minus_me_objf = self_info.objf;  // objf of own cluster minus self.
    // Now check the other "close" clusters and see if we want to move there.
           for (int32 index = 0;index < cfg_.top_n;index++) {
      if (index != self_index) {
        UpdateInfo(point, index);
        point_info &other_info = GetInfo(point, index);
        BaseFloat other_clust_objf = clust_objf_[other_info.clust];
        BaseFloat other_clust_plus_me_objf = other_info.objf;
        BaseFloat impr = other_clust_plus_me_objf + own_clust_minus_me_objf
            - other_clust_objf - own_clust_objf;
        if (impr > 0) {  // better to switch...
          ans_ += impr;
          MovePoint(point, index);
          return;  // the stuff we precomputed at the top is invalidated now, and it's
          // easiest just to wait till next time we visit this point.
        }
      }
    }
  }

  void UpdateInfo(int32 point, int32 idx) {
    point_info &pinfo = GetInfo(point, idx);
    if (pinfo.time < clust_time_[pinfo.clust]) {  // it's not up-to-date...
      Clusterable *tmp_cl = (*clusters_)[pinfo.clust]->Copy();
      if (idx == my_clust_index_[point]) {
        tmp_cl->Sub( *(points_[point]) );
      } else{
        tmp_cl->Add( *(points_[point]) );
      }
      pinfo.time = t_;
      pinfo.objf = tmp_cl->Objf();
      delete tmp_cl;
    }
  }

  typedef struct {
    LocalInt clust;
    LocalInt time;
    BaseFloat objf;  // Objf of this cluster plus this point (or minus, if own cluster).
  } point_info;

  point_info &GetInfo(int32 point, int32 idx) {
    KALDI_ASSERT(point < num_points_ && idx < cfg_.top_n);
    int32 i = point*cfg_.top_n + idx;
    KALDI_PARANOID_ASSERT(i < static_cast<int32>(info_.size()));
    return info_[i];
  }

  const std::vector<Clusterable*> &points_;
  std::vector<Clusterable*> *clusters_;
  std::vector<int32> *assignments_;

  std::vector<point_info> info_;  // size is [num_points_ * cfg_.top_n].
  std::vector<ClustIndexInt> my_clust_index_;  // says for each point, which index 0...cfg_.top_n-1 currently
                                            // corresponds to its own cluster.

  std::vector<LocalInt> clust_time_;  // Modification time of cluster.
  std::vector<BaseFloat> clust_objf_;  // [clust], objf for cluster.

  BaseFloat ans_;  // objf improvement.

  int32 num_clust_;
  int32 num_points_;
  int32 t_;
  RefineClustersOptions cfg_;  // note, we change top_n in config; don't make this member a reference member.
};


BaseFloat RefineClusters(const std::vector<Clusterable*> &points,
                         std::vector<Clusterable*> *clusters,
                         std::vector<int32> *assignments,
                         RefineClustersOptions cfg) {
#ifndef KALDI_PARANOID // don't do this check in "paranoid" mode as we want to expose bugs.
  if (cfg.num_iters <= 0) { return 0.0; } // nothing to do.
#endif
  KALDI_ASSERT(clusters != NULL && assignments != NULL);
  KALDI_ASSERT(!ContainsNullPointers(points) && !ContainsNullPointers(*clusters));
  RefineClusterer rc(points, clusters, assignments, cfg);
  BaseFloat ans = rc.Refine();
  KALDI_ASSERT(!ContainsNullPointers(*clusters));
  return ans;
}

// ============================================================================
// K-means like clustering routines
// ============================================================================

/// ClusterKMeansOnce is called internally by ClusterKMeans; it is equivalent
/// to calling ClusterKMeans with cfg.num_tries == 1.  It returns the objective
/// function improvement versus everything being in one cluster.

BaseFloat ClusterKMeansOnce(const std::vector<Clusterable*> &points,
                            int32 num_clust,
                            std::vector<Clusterable*> *clusters_out,
                            std::vector<int32> *assignments_out,
                            ClusterKMeansOptions &cfg) {
  std::vector<int32> my_assignments;
  int32 num_points = points.size();
  KALDI_ASSERT(clusters_out != NULL);
  KALDI_ASSERT(num_points != 0);
  KALDI_ASSERT(num_clust <= num_points);

  KALDI_ASSERT(clusters_out->empty());  // or we wouldn't know what to do with pointers in there.
  clusters_out->resize(num_clust, (Clusterable*)NULL);
  assignments_out->resize(num_points);

  {  // This block assigns points to clusters.
    // This is done pseudo-randomly using Rand() so that
    // if we call ClusterKMeans multiple times we get different answers (so we can choose
    // the best if we want).
    int32 skip;  // randomly choose a "skip" that's coprime to num_points.
    if (num_points == 1) {
      skip = 1;
    } else {
      skip = 1 + (Rand() % (num_points-1));  // a number between 1 and num_points-1.
      while (Gcd(skip, num_points) != 1) {  // while skip is not coprime to num_points...
        if (skip == num_points-1) skip = 0;
        skip++;  // skip is now still betweeen 1 and num_points-1.  will cycle through
        // all of 1...num_points-1.
      }
    }
    int32 i, j, count = 0;
    for (i = 0, j = 0; count != num_points;i = (i+skip)%num_points, j = (j+1)%num_clust, count++) {
      // i cycles pseudo-randomly through all points; j skips ahead by 1 each time
      // modulo num_points.
      // assign point i to cluster j.
      if ((*clusters_out)[j] == NULL) (*clusters_out)[j] = points[i]->Copy();
      else (*clusters_out)[j]->Add(*(points[i]));
      (*assignments_out)[i] = j;
    }
  }


  BaseFloat normalizer = SumClusterableNormalizer(*clusters_out);
  BaseFloat ans;
  {  // work out initial value of "ans" (objective function improvement).
    Clusterable *all_stats = SumClusterable(*clusters_out);
    ans = SumClusterableObjf(*clusters_out) - all_stats->Objf();  // improvement just from the random
    // initialization.
    if (ans < -0.01 && ans < -0.01 * fabs(all_stats->Objf())) {  // something bad happend.
      KALDI_WARN << "ClusterKMeans: objective function after random assignment to clusters is worse than in single cluster: "<< (all_stats->Objf()) << " changed by " << ans << ".  Perhaps your stats class has the wrong properties?";
    }
    delete all_stats;
  }
  for (int32 iter = 0;iter < cfg.num_iters;iter++) {
    // Keep refining clusters by reassigning points.
    BaseFloat objf_before;
    if (cfg.verbose) objf_before =SumClusterableObjf(*clusters_out);
    BaseFloat impr = RefineClusters(points, clusters_out, assignments_out, cfg.refine_cfg);
    BaseFloat objf_after;
    if (cfg.verbose) objf_after = SumClusterableObjf(*clusters_out);
    ans += impr;
    if (cfg.verbose)
      KALDI_LOG << "ClusterKMeans: on iteration "<<(iter)<<", objf before = "<<(objf_before)<<", impr = "<<(impr)<<", objf after = "<<(objf_after)<<", normalized by "<<(normalizer)<<" = "<<(objf_after/normalizer);
    if (impr == 0) break;
  }
  return ans;
}

BaseFloat ClusterKMeans(const std::vector<Clusterable*> &points,
                        int32 num_clust,
                        std::vector<Clusterable*> *clusters_out,
                        std::vector<int32> *assignments_out,
                        ClusterKMeansOptions cfg) {
  if (points.size() == 0) {
    if (clusters_out) KALDI_ASSERT(clusters_out->empty());  // or we wouldn't know whether to free the pointers.
    if (assignments_out) assignments_out->clear();
    return 0.0;
  }
  KALDI_ASSERT(cfg.num_tries>=1 && cfg.num_iters>=1);
  if (clusters_out) KALDI_ASSERT(clusters_out->empty());  // or we wouldn't know whether to deallocate.
  if (cfg.num_tries == 1) {
    std::vector<int32> assignments;
    return ClusterKMeansOnce(points, num_clust, clusters_out, (assignments_out != NULL?assignments_out:&assignments), cfg);
  } else {  // multiple tries.
    if (clusters_out) {
      KALDI_ASSERT(clusters_out->empty());  // we don't know the ownership of any pointers in there, otherwise.
    }
    BaseFloat best_ans = 0.0;
    for (int32 i = 0;i < cfg.num_tries;i++) {
      std::vector<Clusterable*> clusters_tmp;
      std::vector<int32> assignments_tmp;
      BaseFloat ans = ClusterKMeansOnce(points, num_clust, &clusters_tmp, &assignments_tmp, cfg);
      KALDI_ASSERT(!ContainsNullPointers(clusters_tmp));
      if (i == 0 || ans > best_ans) {
        best_ans = ans;
        if (clusters_out) {
          if (clusters_out->size()) DeletePointers(clusters_out);
          *clusters_out = clusters_tmp;
          clusters_tmp.clear();  // suppress deletion of pointers.
        }
        if (assignments_out) *assignments_out = assignments_tmp;
      }
      // delete anything remaining in clusters_tmp (we cleared it if we used
      // the pointers.
      DeletePointers(&clusters_tmp);
    }
    return best_ans;
  }
}

// ============================================================================
// Routines for clustering using a top-down tree
// ============================================================================

class TreeClusterer {
 public:
  TreeClusterer(const std::vector<Clusterable*> &points,
                int32 max_clust,
                TreeClusterOptions cfg):
      points_(points), max_clust_(max_clust), ans_(0.0), cfg_(cfg)
  {
    KALDI_ASSERT(cfg_.branch_factor > 1);
    Init();
  }
  BaseFloat Cluster(std::vector<Clusterable*> *clusters_out,
                    std::vector<int32> *assignments_out,
                    std::vector<int32> *clust_assignments_out,
                    int32 *num_leaves_out) {
    while (static_cast<int32>(leaf_nodes_.size()) < max_clust_ && !queue_.empty()) {
      std::pair<BaseFloat, Node*> pr = queue_.top();
      queue_.pop();
      ans_ += pr.first;
      DoSplit(pr.second);
    }
    CreateOutput(clusters_out, assignments_out, clust_assignments_out,
                 num_leaves_out);
    return ans_;
  }

  ~TreeClusterer() {
    for (int32 leaf = 0; leaf < static_cast<int32>(leaf_nodes_.size());leaf++) {
      delete leaf_nodes_[leaf]->node_total;
      DeletePointers(&(leaf_nodes_[leaf]->leaf.clusters));
      delete leaf_nodes_[leaf];
    }
    for (int32 nonleaf = 0; nonleaf < static_cast<int32>(nonleaf_nodes_.size()); nonleaf++) {
      delete nonleaf_nodes_[nonleaf]->node_total;
      delete nonleaf_nodes_[nonleaf];
    }
  }


 private:
  struct Node {
    bool is_leaf;
    int32 index;  // index into leaf_nodes or nonleaf_nodes as applicable.
    Node *parent;
    Clusterable *node_total;  // sum of all data with this node.
    struct {
      std::vector<Clusterable*> points;
      std::vector<int32> point_indices;
      BaseFloat best_split;
      std::vector<Clusterable*> clusters;  // [branch_factor]... if we do split.
      std::vector<int32> assignments;  // assignments of points to clusters.
    } leaf;
    std::vector<Node*> children;  // vector of size branch_factor.   if non-leaf.
    // pointers not owned here but in vectors leaf_nodes_, nonleaf_nodes_.
  };


  void CreateOutput(std::vector<Clusterable*> *clusters_out,
                    std::vector<int32> *assignments_out,
                    std::vector<int32> *clust_assignments_out,
                    int32 *num_leaves_out) {
   if (num_leaves_out) *num_leaves_out = leaf_nodes_.size();
    if (assignments_out)
      CreateAssignmentsOutput(assignments_out);
    if (clust_assignments_out)
      CreateClustAssignmentsOutput(clust_assignments_out);
    if (clusters_out)
      CreateClustersOutput(clusters_out);
  }

  // This creates the output index corresponding to an index "index" into the array nonleaf_nodes_.
  // reverse numbering so root node is last.
  int32 NonleafOutputIndex(int32 index) {
    return leaf_nodes_.size() + nonleaf_nodes_.size() - 1 - index;
  }
  void CreateAssignmentsOutput(std::vector<int32> *assignments_out) {
    assignments_out->clear();
    assignments_out->resize(points_.size(), (int32)(-1));  // fill with -1.
    for (int32 leaf = 0; leaf < static_cast<int32>(leaf_nodes_.size()); leaf++) {
      std::vector<int32> &indices = leaf_nodes_[leaf]->leaf.point_indices;
      for (int32 i = 0; i < static_cast<int32>(indices.size()); i++) {
        KALDI_ASSERT(static_cast<size_t>(indices[i]) < points_.size());
        KALDI_ASSERT((*assignments_out)[indices[i]] == (int32)(-1));  // check we're not assigning twice.
        (*assignments_out)[indices[i]] = leaf;
      }
    }
#ifdef KALDI_PARANOID
    for (size_t i = 0;i<assignments_out->size();i++) KALDI_ASSERT((*assignments_out)[i] != (int32)(-1));
#endif
  }
  void CreateClustAssignmentsOutput(std::vector<int32> *clust_assignments_out) {
    clust_assignments_out->resize(leaf_nodes_.size() + nonleaf_nodes_.size());
    for (int32 leaf = 0; leaf < static_cast<int32>(leaf_nodes_.size()); leaf++) {
      int32 parent_index;
      if (leaf_nodes_[leaf]->parent == NULL) {  // tree with only one node.
        KALDI_ASSERT(leaf_nodes_.size() == 1&&nonleaf_nodes_.size() == 0 && leaf == 0);
        parent_index = 0;
      } else {
        if (leaf_nodes_[leaf]->parent->is_leaf) parent_index = leaf_nodes_[leaf]->parent->index;
        else parent_index = NonleafOutputIndex(leaf_nodes_[leaf]->parent->index);
      }
      (*clust_assignments_out)[leaf] = parent_index;
    }
    for (int32 nonleaf = 0; nonleaf < static_cast<int32>(nonleaf_nodes_.size()); nonleaf++) {
      int32 index = NonleafOutputIndex(nonleaf);
      int32 parent_index;
      if (nonleaf_nodes_[nonleaf]->parent == NULL) parent_index = index;  // top node.  make it own parent.
      else {
        KALDI_ASSERT(! nonleaf_nodes_[nonleaf]->parent->is_leaf);  // parent is nonleaf since child is nonleaf.
        parent_index = NonleafOutputIndex(nonleaf_nodes_[nonleaf]->parent->index);
      }
      (*clust_assignments_out)[index] = parent_index;
    }
  }
  void CreateClustersOutput(std::vector<Clusterable*> *clusters_out) {
    clusters_out->resize(leaf_nodes_.size() + nonleaf_nodes_.size());
    for (int32 leaf = 0; leaf < static_cast<int32>(leaf_nodes_.size()); leaf++) {
      (*clusters_out)[leaf] = leaf_nodes_[leaf]->node_total;
      leaf_nodes_[leaf]->node_total = NULL;  // suppress delete.
    }
    for (int32 nonleaf = 0; nonleaf < static_cast<int32>(nonleaf_nodes_.size()); nonleaf++) {
      int32 index = NonleafOutputIndex(nonleaf);
      (*clusters_out)[index] = nonleaf_nodes_[nonleaf]->node_total;
      nonleaf_nodes_[nonleaf]->node_total = NULL;  // suppress delete.
    }
  }
  void DoSplit(Node *node) {
    KALDI_ASSERT(node->is_leaf && node->leaf.best_split > cfg_.thresh*0.999);  // 0.999 is to avoid potential floating-point weirdness under compiler optimizations.
    KALDI_ASSERT(node->children.size() == 0);
    node->children.resize(cfg_.branch_factor);
    for (int32 i = 0;i < cfg_.branch_factor;i++) {
      Node *child = new Node;
      node->children[i] = child;
      child->is_leaf = true;
      child->parent = node;
      child->node_total = node->leaf.clusters[i];
      if (i == 0) {
        child->index = node->index;  // assign node's own index in leaf_nodes_ to 1st child.
        leaf_nodes_[child->index] = child;
      } else {
        child->index = leaf_nodes_.size();  // generate new indices for other children.
        leaf_nodes_.push_back(child);
      }
    }

    KALDI_ASSERT(node->leaf.assignments.size() == node->leaf.points.size());
    KALDI_ASSERT(node->leaf.point_indices.size() == node->leaf.points.size());
    for (int32 i = 0; i < static_cast<int32>(node->leaf.points.size()); i++) {
      int32 child_index = node->leaf.assignments[i];
      KALDI_ASSERT(child_index < static_cast<int32>(cfg_.branch_factor));
      node->children[child_index]->leaf.points.push_back(node->leaf.points[i]);
      node->children[child_index]->leaf.point_indices.push_back(node->leaf.point_indices[i]);
    }
    node->leaf.points.clear();
    node->leaf.point_indices.clear();
    node->leaf.clusters.clear();  // already assigned pointers to children.
    node->leaf.assignments.clear();
    node->is_leaf = false;
    node->index = nonleaf_nodes_.size();  // new index at end of nonleaf_nodes_.
    nonleaf_nodes_.push_back(node);
    for (int32 i = 0;i < static_cast<int32>(cfg_.branch_factor);i++)
      FindBestSplit(node->children[i]);
  }
  void FindBestSplit(Node *node) {
    // takes a leaf node that has just been set up, and does ClusterKMeans with k = cfg_branch_factor.
    KALDI_ASSERT(node->is_leaf);
    if (node->leaf.points.size() == 0) {
      KALDI_WARN << "Warning: tree clustering: leaf with no data";
      node->leaf.best_split = 0; return;
    }
    if (node->leaf.points.size()<=1) { node->leaf.best_split = 0; return; }
    else {
      // use kmeans.
      BaseFloat impr = ClusterKMeans(node->leaf.points,
                                     cfg_.branch_factor,
                                     &node->leaf.clusters,
                                     &node->leaf.assignments,
                                     cfg_.kmeans_cfg);
      node->leaf.best_split = impr;
      if (impr > cfg_.thresh)
        queue_.push(std::make_pair(impr, node));
    }
  }
  void Init() {  // Initializes top node.
    Node *top_node = new Node;
    top_node->index = leaf_nodes_.size();  // == 0 currently.
    top_node->parent = NULL;  // no parent since root of tree.
    top_node->is_leaf = true;
    leaf_nodes_.push_back(top_node);
    top_node->leaf.points = points_;
    top_node->node_total = SumClusterable(points_);
    top_node->leaf.point_indices.resize(points_.size());
    for (size_t i = 0;i<points_.size();i++) top_node->leaf.point_indices[i] = i;
    FindBestSplit(top_node);  // this should always be called when new node is created.
  }

  std::vector<Node*> leaf_nodes_;
  std::vector<Node*> nonleaf_nodes_;

  const std::vector<Clusterable*> &points_;
  int32 max_clust_;
  BaseFloat ans_;  // objf improvement.

  std::priority_queue<std::pair<BaseFloat, Node*> > queue_;  // contains leaves.

  TreeClusterOptions cfg_;
};


BaseFloat TreeCluster(const std::vector<Clusterable*> &points,
                      int32 max_clust,  // this is a max only.
                      std::vector<Clusterable*> *clusters_out,
                      std::vector<int32> *assignments_out,
                      std::vector<int32> *clust_assignments_out,
                      int32 *num_leaves_out,
                      TreeClusterOptions cfg) {
  if (points.size() == 0) {
    if (clusters_out) clusters_out->clear();
    if (assignments_out) assignments_out->clear();
    if (clust_assignments_out) clust_assignments_out->clear();
    if (num_leaves_out) *num_leaves_out = 0;
    return 0.0;
  }
  TreeClusterer tc(points, max_clust, cfg);
  BaseFloat ans = tc.Cluster(clusters_out, assignments_out, clust_assignments_out, num_leaves_out);
  if (clusters_out) KALDI_ASSERT(!ContainsNullPointers(*clusters_out));
  return ans;
}


BaseFloat ClusterTopDown(const std::vector<Clusterable*> &points,
                         int32 max_clust,  // max # of clusters.
                         std::vector<Clusterable*> *clusters_out,
                         std::vector<int32> *assignments_out,
                         TreeClusterOptions cfg) {
  int32 num_leaves = 0;
  BaseFloat ans = TreeCluster(points, max_clust, clusters_out, assignments_out, NULL, &num_leaves, cfg);
  if (clusters_out != NULL) {
    for (size_t j = num_leaves;j<clusters_out->size();j++) delete (*clusters_out)[j];
    clusters_out->resize(num_leaves);  // number of leaf-level clusters in tree.
  }
  return ans;
}


void RefineClustersOptions::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<RefineClustersOptions>");
  WriteBasicType(os, binary, num_iters);
  WriteBasicType(os, binary, top_n);
  WriteToken(os, binary, "</RefineClustersOptions>");
}

void RefineClustersOptions::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<RefineClustersOptions>");
  ReadBasicType(is, binary, &num_iters);
  ReadBasicType(is, binary, &top_n);
  ExpectToken(is, binary, "</RefineClustersOptions>");
}


} // end namespace kaldi.
