// tree/build-tree.h

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

#ifndef KALDI_TREE_BUILD_TREE_H_
#define KALDI_TREE_BUILD_TREE_H_

// The file build-tree.h contains outer-level routines used in tree-building
// and related tasks, that are directly called by the command-line tools.

#include "tree/build-tree-utils.h"
#include "tree/context-dep.h"
namespace kaldi {

/// \defgroup tree_group_top Top-level tree-building functions
/// See \ref tree_internals for context.
/// \ingroup tree_group
/// @{

// Note, in tree_group_top we also include AccumulateTreeStats, in
// ../hmm/tree-accu.h (it has some extra dependencies so we didn't
// want to include it here).

/**
 *  BuildTree is the normal way to build a set of decision trees.
 *  The sets "phone_sets" dictate how we set up the roots of the decision trees.
 *  each set of phones phone_sets[i] has shared decision-tree roots, and if
 *  the corresponding variable share_roots[i] is true, the root will be shared
 *  for the different HMM-positions in the phone.  All phones in "phone_sets"
 *  should be in the stats (use FixUnseenPhones to ensure this).
 *  if for any i, do_split[i] is false, we will not do any tree splitting for
 *  phones in that set.
 * @param qopts [in] Questions options class, contains questions for each key
 *                   (e.g. each phone position)
 * @param phone_sets [in] Each element of phone_sets is a set of phones whose
 *                 roots are shared together (prior to decision-tree splitting).
 * @param phone2num_pdf_classes [in] A map from phones to the number of
 *                 \ref pdf_class "pdf-classes"
 *                 in the phone (this info is derived from the HmmTopology object)
 * @param share_roots [in] A vector the same size as phone_sets; says for each
 *                phone set whether the root should be shared among all the
 *                pdf-classes or not.
 * @param do_split [in] A vector the same size as phone_sets; says for each
 *                phone set whether decision-tree splitting should be done
 *                 (generally true for non-silence phones).
 * @param stats [in] The statistics used in tree-building.
 * @param thresh [in] Threshold used in decision-tree splitting (e.g. 1000),
 *                   or you may use 0 in which case max_leaves becomes the
 *                    constraint.
 * @param max_leaves [in] Maximum number of leaves it will create; set this
 *                  to a large number if you want to just specify  "thresh".
 * @param cluster_thresh [in] Threshold for clustering leaves after decision-tree
 *                  splitting (only within each phone-set); leaves will be combined
 *                  if log-likelihood change is less than this.  A value about equal
 *                  to "thresh" is suitable
 *                  if thresh != 0; otherwise, zero will mean no clustering is done,
 *                  or a negative value (e.g. -1) sets it to the smallest likelihood
 *                  change seen during the splitting algorithm; this typically causes
 *                  about a 20% reduction in the number of leaves.
 
 * @param P [in] The central position of the phone context window, e.g. 1 for a
 *                triphone system.
 * @return  Returns a pointer to an EventMap object that is the tree.

*/

EventMap *BuildTree(Questions &qopts,
                    const std::vector<std::vector<int32> > &phone_sets,
                    const std::vector<int32> &phone2num_pdf_classes,
                    const std::vector<bool> &share_roots,
                    const std::vector<bool> &do_split,
                    const BuildTreeStatsType &stats,
                    BaseFloat thresh,
                    int32 max_leaves,
                    BaseFloat cluster_thresh,  // typically == thresh.  If negative, use smallest split.
                    int32 P);


/**
 *
 *  BuildTreeTwoLevel builds a two-level tree, useful for example in building tied mixture
 *  systems with multiple codebooks.  It first builds a small tree by splitting to
 *  "max_leaves_first".  It then splits at the leaves of "max_leaves_first" (think of this
 *  as creating multiple little trees at the leaves of the first tree), until the total
 *  number of leaves reaches "max_leaves_second".  It then outputs the second tree, along
 *  with a mapping from the leaf-ids of the second tree to the leaf-ids of the first tree.
 *  Note that the interface is similar to BuildTree, and in fact it calls BuildTree
 *  internally.
 *
 *  The sets "phone_sets" dictate how we set up the roots of the decision trees.
 *  each set of phones phone_sets[i] has shared decision-tree roots, and if
 *  the corresponding variable share_roots[i] is true, the root will be shared
 *  for the different HMM-positions in the phone.  All phones in "phone_sets"
 *  should be in the stats (use FixUnseenPhones to ensure this).
 *  if for any i, do_split[i] is false, we will not do any tree splitting for
 *  phones in that set.
 *
 * @param qopts [in] Questions options class, contains questions for each key
 *                   (e.g. each phone position)
 * @param phone_sets [in] Each element of phone_sets is a set of phones whose
 *                 roots are shared together (prior to decision-tree splitting).
 * @param phone2num_pdf_classes [in] A map from phones to the number of
 *                 \ref pdf_class "pdf-classes"
 *                 in the phone (this info is derived from the HmmTopology object)
 * @param share_roots [in] A vector the same size as phone_sets; says for each
 *                phone set whether the root should be shared among all the
 *                pdf-classes or not.
 * @param do_split [in] A vector the same size as phone_sets; says for each
 *                phone set whether decision-tree splitting should be done
 *                 (generally true for non-silence phones).
 * @param stats [in] The statistics used in tree-building.
 * @param max_leaves_first [in] Maximum number of leaves it will create in first
 *                  level of decision tree. 
 * @param max_leaves_second [in] Maximum number of leaves it will create in second
 *                  level of decision tree.  Must be > max_leaves_first.
 * @param cluster_leaves [in] Boolean value; if true, we post-cluster the leaves produced
 *                  in the second level of decision-tree split; if false, we don't.
 *                  The threshold for post-clustering is the log-like change of the last
 *                  decision-tree split; this typically causes about a 20% reduction in
 *                  the number of leaves.
 * @param P [in]   The central position of the phone context window, e.g. 1 for a
 *                 triphone system.
 * @param leaf_map [out]  Will be set to be a mapping from the leaves of the
 *                 "big" tree to the leaves of the "little" tree, which you can
 *                 view as cluster centers.
 * @return  Returns a pointer to an EventMap object that is the (big) tree.

*/

EventMap *BuildTreeTwoLevel(Questions &qopts,
                            const std::vector<std::vector<int32> > &phone_sets,
                            const std::vector<int32> &phone2num_pdf_classes,
                            const std::vector<bool> &share_roots,
                            const std::vector<bool> &do_split,
                            const BuildTreeStatsType &stats,
                            int32 max_leaves_first,
                            int32 max_leaves_second,
                            bool cluster_leaves,
                            int32 P,
                            std::vector<int32> *leaf_map);


/// GenRandStats generates random statistics of the form used by BuildTree.
/// It tries to do so in such a way that they mimic "real" stats.  The event keys
/// and their corresponding values are:
/// - key == -1 == kPdfClass -> pdf-class, generally corresponds to
///       zero-based position in HMM (0, 1, 2 .. hmm_lengths[phone]-1)
/// - key == 0 -> phone-id of left-most context phone.
/// - key == 1 -> phone-id of one-from-left-most context phone.
/// - key == P-1 -> phone-id of central phone.
/// - key == N-1 -> phone-id of right-most context phone.
/// GenRandStats is useful only for testing but it serves to document the format of
/// stats used by BuildTreeDefault.
/// if is_ctx_dep[phone] is set to false, GenRandStats will not define the keys for
/// other than the P-1'th phone.

/// @param dim [in] dimension of features.
/// @param num_stats [in] approximate number of separate phones-in-context wanted.
/// @param N [in] context-size (typically 3)
/// @param P [in] central-phone position in zero-based numbering (typically 1)
/// @param phone_ids [in] integer ids of phones
/// @param hmm_lengths [in] lengths of hmm for phone, indexed by phone.
/// @param is_ctx_dep [in] boolean array indexed by phone, saying whether each phone
///     is context dependent.
/// @param ensure_all_phones_covered [in] Boolean argument: if true, GenRandStats
///     ensures that every phone is seen at least once in the central position (P).
/// @param stats_out [out] The statistics that this routine outputs.

void GenRandStats(int32 dim, int32 num_stats, int32 N, int32 P,
                  const std::vector<int32> &phone_ids,
                  const std::vector<int32> &hmm_lengths,
                  const std::vector<bool> &is_ctx_dep,
                  bool ensure_all_phones_covered,
                  BuildTreeStatsType *stats_out);


/// included here because it's used in some tree-building
/// calling code.  Reads an OpenFst symbl table,
/// discards the symbols and outputs the integers
void ReadSymbolTableAsIntegers(std::string filename,
                               bool include_eps,
                               std::vector<int32> *syms);



/**
 *  Outputs sets of phones that are reasonable for questions
 *  to ask in the tree-building algorithm.  These are obtained by tree
 *  clustering of the phones; for each node in the tree, all the leaves
 *  accessible from that node form one of the sets of phones.
 *    @param stats [in] The statistics as used for normal tree-building.
 *    @param phone_sets_in [in] All the phones, pre-partitioned into sets.
 *       The output sets will be various unions of these sets.  These sets
 *       will normally correspond to "real phones", in cases where the phones
 *       have stress and position markings.
 *    @param all_pdf_classes_in [in] All the \ref pdf_class "pdf-classes"
 *      that we consider for clustering.  In the normal case this is the singleton
 *       set {1}, which means that we only consider the central hmm-position
 *       of the standard 3-state HMM, for clustering purposes.
 *    @param P [in] The central position in the phone context window; normally
 *       1 for triphone system.s
 *    @param questions_out [out] The questions (sets of phones) are output to here.
 **/
void AutomaticallyObtainQuestions(BuildTreeStatsType &stats,
                                  const std::vector<std::vector<int32> > &phone_sets_in,
                                  const std::vector<int32> &all_pdf_classes_in,
                                  int32 P,
                                  std::vector<std::vector<int32> > *questions_out);

/// This function clusters the phones (or some initially specified sets of phones)
/// into sets of phones, using a k-means algorithm.  Useful, for example, in building
/// simple models for purposes of adaptation.

void KMeansClusterPhones(BuildTreeStatsType &stats,
                         const std::vector<std::vector<int32> > &phone_sets_in,
                         const std::vector<int32> &all_pdf_classes_in,
                         int32 P,
                         int32 num_classes,
                         std::vector<std::vector<int32> > *sets_out);

/// Reads the roots file (throws on error).  Format is lines like:
///  "shared split 1 2 3 4",
///  "not-shared not-split 5",
/// and so on.  The numbers are indexes of phones.
void ReadRootsFile(std::istream &is,
                   std::vector<std::vector<int32> > *phone_sets,
                   std::vector<bool> *is_shared_root,
                   std::vector<bool> *is_split_root);


/// @}

}// end namespace kaldi

#endif
