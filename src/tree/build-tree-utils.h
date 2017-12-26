// tree/build-tree-utils.h

// Copyright 2009-2011  Microsoft Corporation

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

#ifndef KALDI_TREE_BUILD_TREE_UTILS_H_
#define KALDI_TREE_BUILD_TREE_UTILS_H_

#include "tree/build-tree-questions.h"

// build-tree-questions.h needed for this typedef:
// typedef std::vector<std::pair<EventType, Clusterable*> > BuildTreeStatsType;
// and for other #includes.

namespace kaldi {


///   \defgroup tree_group_lower Low-level functions for manipulating statistics and event-maps
///    See \ref tree_internals and specifically \ref treei_func for context.
///   \ingroup tree_group
///
///   @{



/// This frees the Clusterable* pointers in "stats", where non-NULL, and sets them to NULL.
/// Does not delete the pointer "stats" itself.
void DeleteBuildTreeStats(BuildTreeStatsType *stats);

/// Writes BuildTreeStats object.  This works even if pointers are NULL.
void WriteBuildTreeStats(std::ostream &os, bool binary,
                         const BuildTreeStatsType &stats);

/// Reads BuildTreeStats object.  The "example" argument must be of the same
/// type as the stats on disk, and is needed for access to the correct "Read"
/// function.  It was organized this way for easier extensibility (so adding new
/// Clusterable derived classes isn't painful)
void ReadBuildTreeStats(std::istream &is, bool binary,
                        const Clusterable &example, BuildTreeStatsType *stats);

/// Convenience function e.g. to work out possible values of the phones from just the stats.
/// Returns true if key was always defined inside the stats.
/// May be used with and == NULL to find out of key was always defined.
bool PossibleValues(EventKeyType key, const BuildTreeStatsType &stats,
                    std::vector<EventValueType> *ans);


/// Splits stats according to the EventMap, indexing them at output by the
/// leaf type.   A utility function.  NOTE-- pointers in stats_out point to
/// the same memory location as those in stats.  No copying of Clusterable*
/// objects happens.  Will add to stats in stats_out if non-empty at input.
/// This function may increase the size of vector stats_out as necessary
/// to accommodate stats, but will never decrease the size.
void SplitStatsByMap(const BuildTreeStatsType &stats_in, const EventMap &e,
                     std::vector<BuildTreeStatsType> *stats_out);

/// SplitStatsByKey splits stats up according to the value of a particular key,
/// which must be always defined and nonnegative.  Like MapStats.  Pointers to
/// Clusterable* in stats_out are not newly allocated-- they are the same as the
/// ones in stats_in.  Generally they will still be owned at stats_in (user can
/// decide where to allocate ownership).
void SplitStatsByKey(const BuildTreeStatsType &stats_in, EventKeyType key,
                     std::vector<BuildTreeStatsType> *stats_out);


/// Converts stats from a given context-window (N) and central-position (P) to a
/// different N and P, by possibly reducing context.  This function does a job
/// that's quite specific to the "normal" stats format we use.  See \ref
/// tree_window for background.  This function may delete some keys and change
/// others, depending on the N and P values.  It expects that at input, all keys
/// will either be -1 or lie between 0 and oldN-1.  At output, keys will be
/// either -1 or between 0 and newN-1.
/// Returns false if we could not convert the stats (e.g. because newN is larger
/// than oldN).
bool ConvertStats(int32 oldN, int32 oldP, int32 newN, int32 newP,
                  BuildTreeStatsType *stats);


/// FilterStatsByKey filters the stats according the value of a specified key.
/// If include_if_present == true, it only outputs the stats whose key is in
/// "values"; otherwise it only outputs the stats whose key is not in "values".
/// At input, "values" must be sorted and unique, and all stats in "stats_in"
/// must have "key" defined.  At output, pointers to Clusterable* in stats_out
/// are not newly allocated-- they are the same as the ones in stats_in.
void FilterStatsByKey(const BuildTreeStatsType &stats_in,
                      EventKeyType key,
                      std::vector<EventValueType> &values,
                      bool include_if_present,  // true-> retain only if in "values",
                      // false-> retain only if not in "values".
                      BuildTreeStatsType *stats_out);


/// Sums stats, or returns NULL stats_in has no non-NULL stats.
/// Stats are newly allocated, owned by caller.
Clusterable *SumStats(const BuildTreeStatsType &stats_in);

/// Sums the normalizer [typically, data-count] over the stats.
BaseFloat SumNormalizer(const BuildTreeStatsType &stats_in);

/// Sums the objective function over the stats.
BaseFloat SumObjf(const BuildTreeStatsType &stats_in);


/// Sum a vector of stats.  Leaves NULL as pointer if no stats available.
/// The pointers in stats_out are owned by caller.  At output, there may be
/// NULLs in the vector stats_out.
void SumStatsVec(const std::vector<BuildTreeStatsType> &stats_in, std::vector<Clusterable*> *stats_out);

/// Cluster the stats given the event map return the total objf given those clusters.
BaseFloat ObjfGivenMap(const BuildTreeStatsType &stats_in, const EventMap &e);


/// FindAllKeys puts in *keys the (sorted, unique) list of all key identities in the stats.
/// If type == kAllKeysInsistIdentical, it will insist that this set of keys is the same for all the
///   stats (else exception is thrown).
/// if type == kAllKeysIntersection, it will return the smallest common set of keys present in
///   the set of stats
/// if type== kAllKeysUnion (currently probably not so useful since maps will return "undefined"
///   if key is not present), it will return the union of all the keys present in the stats.
void FindAllKeys(const BuildTreeStatsType &stats, AllKeysType keys_type,
                 std::vector<EventKeyType> *keys);


/// @}


/**
 \defgroup tree_group_intermediate Intermediate-level functions used in building the tree
    These functions are are used in top-level tree-building code (\ref tree_group_top); see
     \ref tree_internals for documentation.
 \ingroup tree_group
 @{
*/


/// Returns a tree with just one node.  Used @ start of tree-building process.
/// Not really used in current recipes.
inline EventMap *TrivialTree(int32 *num_leaves) {
  KALDI_ASSERT(*num_leaves == 0);  // in envisaged usage.
  return new ConstantEventMap( (*num_leaves)++ );
}

/// DoTableSplit does a complete split on this key (e.g. might correspond to central phone
/// (key = P-1), or HMM-state position (key == kPdfClass == -1).  Stats used to work out possible
/// values of the event. "num_leaves" is used to allocate new leaves.   All stats must have
/// this key defined, or this function will crash.
EventMap *DoTableSplit(const EventMap &orig, EventKeyType key,
                       const BuildTreeStatsType &stats, int32 *num_leaves);


/// DoTableSplitMultiple does a complete split on all the keys, in order from keys[0],
/// keys[1]
/// and so on.  The stats are used to work out possible values corresponding to the key.
/// "num_leaves" is used to allocate new leaves.   All stats must have
/// the keys defined, or this function will crash.
/// Returns a newly allocated event map.
EventMap *DoTableSplitMultiple(const EventMap &orig,
                               const std::vector<EventKeyType> &keys,
                               const BuildTreeStatsType &stats,
                               int32 *num_leaves);


/// "ClusterEventMapGetMapping" clusters the leaves of the EventMap, with "thresh" a delta-likelihood
/// threshold to control how many leaves we combine (might be the same as the delta-like
/// threshold used in splitting.
// The function returns the #leaves we combined.  The same leaf-ids of the leaves being clustered
// will be used for the clustered leaves (but other than that there is no special rule which
// leaf-ids should be used at output).
// It outputs the mapping for leaves, in "mapping", which may be empty at the start
// but may also contain mappings for other parts of the tree, which must contain
// disjoint leaves from this part.  This is so that Cluster can
// be called multiple times for sub-parts of the tree (with disjoint sets of leaves),
// e.g. if we want to avoid sharing across phones.  Afterwards you can use Copy function
// of EventMap to apply the mapping, i.e. call e_in.Copy(mapping) to get the new map.
// Note that the application of Cluster creates gaps in the leaves.  You should then
// call RenumberEventMap(e_in.Copy(mapping), num_leaves).
// *If you only want to cluster a subset of the leaves (e.g. just non-silence, or just
// a particular phone, do this by providing a set of "stats" that correspond to just
// this subset of leaves*.  Leaves with no stats will not be clustered.
// See build-tree.cc for an example of usage.
int ClusterEventMapGetMapping(const EventMap &e_in, const BuildTreeStatsType &stats,
                              BaseFloat thresh, std::vector<EventMap*> *mapping);

/// This is as ClusterEventMapGetMapping but a more convenient interface
/// that exposes less of the internals.  It uses a bottom-up clustering to
/// combine the leaves, until the log-likelihood decrease from combinging two
/// leaves exceeds the threshold.
EventMap *ClusterEventMap(const EventMap &e_in, const BuildTreeStatsType &stats,
                          BaseFloat thresh, int32 *num_removed);

/// This is as ClusterEventMap, but first splits the stats on the keys specified
/// in "keys" (e.g. typically keys = [ -1, P ]), and only clusters within the
/// classes defined by that splitting.
/// Note-- leaves will be non-consecutive at output, use RenumberEventMap.
EventMap *ClusterEventMapRestrictedByKeys(const EventMap &e_in,
                                          const BuildTreeStatsType &stats,
                                          BaseFloat thresh,
                                          const std::vector<EventKeyType> &keys,
                                          int32 *num_removed);


/// This version of ClusterEventMapRestricted restricts the clustering to only
/// allow things that "e_restrict" maps to the same value to be clustered
/// together.
EventMap *ClusterEventMapRestrictedByMap(const EventMap &e_in,
                                         const BuildTreeStatsType &stats,
                                         BaseFloat thresh,
                                         const EventMap &e_restrict,
                                         int32 *num_removed);


/// RenumberEventMap [intended to be used after calling ClusterEventMap] renumbers
/// an EventMap so its leaves are consecutive.
/// It puts the number of leaves in *num_leaves.  If later you need the mapping of
/// the leaves, modify the function and add a new argument.
EventMap *RenumberEventMap(const EventMap &e_in, int32 *num_leaves);

/// This function remaps the event-map leaves using this mapping,
/// indexed by the number at leaf.
EventMap *MapEventMapLeaves(const EventMap &e_in,
                            const std::vector<int32> &mapping);



/// ShareEventMapLeaves performs a quite specific function that allows us to
/// generate trees where, for a certain list of phones, and for all states in
/// the phone, all the pdf's are shared.
/// Each element of "values" contains a list of phones (may be just one phone),
/// all states of which we want shared together).  Typically at input, "key" will
/// equal P, the central-phone position, and "values" will contain just one
/// list containing the silence phone.
/// This function renumbers the event map leaves after doing the sharing, to
/// make the event-map leaves contiguous.
EventMap *ShareEventMapLeaves(const EventMap &e_in, EventKeyType key,
                              std::vector<std::vector<EventValueType> > &values,
                              int32 *num_leaves);



/// Does a decision-tree split at the leaves of an EventMap.
/// @param orig [in] The EventMap whose leaves we want to split. [may be either a trivial or a
///           non-trivial one].
/// @param stats [in] The statistics for splitting the tree; if you do not want a particular
///          subset of leaves to be split, make sure the stats corresponding to those leaves
///          are not present in "stats".
/// @param qcfg [in] Configuration class that contains initial questions (e.g. sets of phones)
///          for each key and says whether to refine these questions during tree building.
/// @param thresh [in] A log-likelihood threshold (e.g. 300) that can be used to
///           limit the number of leaves; you can use zero and set max_leaves instead.
/// @param max_leaves [in] Will stop leaves being split after they reach this number.
/// @param num_leaves [in,out] A pointer used to allocate leaves; always corresponds to the
///             current number of leaves (is incremented when this is increased).
/// @param objf_impr_out [out] If non-NULL, will be set to the objective improvement due to splitting
///           (not normalized by the number of frames).
/// @param smallest_split_change_out If non-NULL, will be set to the smallest objective-function
///         improvement that we got from splitting any leaf; useful to provide a threshold
///         for ClusterEventMap.
/// @return The EventMap after splitting is returned; pointer is owned by caller.
EventMap *SplitDecisionTree(const EventMap &orig,
                            const BuildTreeStatsType &stats,
                            Questions &qcfg,
                            BaseFloat thresh,
                            int32 max_leaves,  // max_leaves<=0 -> no maximum.
                            int32 *num_leaves,
                            BaseFloat *objf_impr_out,
                            BaseFloat *smallest_split_change_out);

/// CreateRandomQuestions will initialize a Questions randomly, in a reasonable
/// way [for testing purposes, or when hand-designed questions are not available].
/// e.g. num_quest = 5 might be a reasonable value if num_iters > 0, or num_quest = 20 otherwise.
void CreateRandomQuestions(const BuildTreeStatsType &stats, int32 num_quest, Questions *cfg_out);


/// FindBestSplitForKey is a function used in DoDecisionTreeSplit.
/// It finds the best split for this key, given these stats.
/// It will return 0 if the key was not always defined for the stats.
BaseFloat FindBestSplitForKey(const BuildTreeStatsType &stats,
                              const Questions &qcfg,
                              EventKeyType key,
                              std::vector<EventValueType> *yes_set);


/// GetStubMap is used in tree-building functions to get the initial
/// to-states map, before the decision-tree-building process.  It creates
/// a simple map that splits on groups of phones.  For the set of phones in
/// phone_sets[i] it creates either: if share_roots[i] == true, a single
/// leaf node, or if share_roots[i] == false, separate root nodes for
/// each HMM-position (it goes up to the highest position for any
/// phone in the set, although it will warn if you share roots between
/// phones with different numbers of states, which is a weird thing to
/// do but should still work.  If any phone is present
/// in "phone_sets" but "phone2num_pdf_classes" does not map it to a length,
/// it is an error.  Note that the behaviour of the resulting map is
/// undefined for phones not present in "phone_sets".
/// At entry, this function should be called with (*num_leaves == 0).
/// It will number the leaves starting from (*num_leaves).

EventMap *GetStubMap(int32 P,
                     const std::vector<std::vector<int32> > &phone_sets,
                     const std::vector<int32> &phone2num_pdf_classes,
                     const std::vector<bool> &share_roots,  // indexed by index into phone_sets.
                     int32 *num_leaves);
/// Note: GetStubMap with P = 0 can be used to get a standard monophone system.

/// @}


}// end namespace kaldi

#endif
