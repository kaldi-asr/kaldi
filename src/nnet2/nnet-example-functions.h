// nnet2/nnet-example-functions.h

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET2_NNET_EXAMPLE_FUNCTIONS_H_
#define KALDI_NNET2_NNET_EXAMPLE_FUNCTIONS_H_

/** @file
    Note on how to parse this filename: it contains functions relatied to
    neural-net training examples, mostly discriminative neural-net training examples,
   i.e. type DiscriminativeNnetExample    
*/

#include "nnet2/nnet-nnet.h"
#include "util/table-types.h"
#include "lat/kaldi-lattice.h"
#include "nnet2/nnet-example.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet2 {

// Glossary: mmi = Maximum Mutual Information,
//          mpfe = Minimum Phone Frame Error
//          smbr = State-level Minimum Bayes Risk


// This file relates to the creation of examples for discriminative training
// (see struct DiscriminativeNnetExample, in ./nnet-example.h).


/** Config structure for SplitExample, for splitting discriminative
    training examples.
*/
struct SplitDiscriminativeExampleConfig {
  // This is the maximum length in frames that any example is allowed to have.
  // We will split training examples to ensure that they are no longer than
  // this.  Note: if you make this too short it may have bad effects because
  // the posteriors start to become inaccurate at the edges of the training
  // example (since they will be based on the acoustic model that was used to
  // generate the lattices, not the current one).
  int32 max_length;

  // criterion can be "smbr" or "mpfe" or "mmi".  This info is only needed to
  // determine which parts of the lattices will not contribute to training and
  // can be discarded (for mpe/smbr, any part where the den-lat has only one
  // path or all den-lat paths map to the same pdf can be discareded; for mmi,
  // any part where the den-lat's pdfs all have the same value as the num-lat
  // pdf for that frame, can be discarded.
  std::string criterion;

  bool collapse_transition_ids;

  bool determinize;

  bool minimize; // we'll push and minimize if this is true.
  
  bool test;

  bool drop_frames; // For MMI, true if we will eventually drop frames in which
                    // the numerator does not appear in the denominator lattice.
                    // (i.e. we won't backpropagate any derivatives on those
                    // frames).  We may still need to include those frames in
                    // the computation in order to get correct posteriors for
                    // other parts of the lattice.

  bool split; // if false, we won't split at all.

  bool excise; // if false, we will skip the "excise" step.
  
  SplitDiscriminativeExampleConfig():
      max_length(1024), criterion("smbr"), collapse_transition_ids(true),
      determinize(true), minimize(true), test(false), drop_frames(false),
      split(true), excise(true) { }

  void Register(OptionsItf *po) {

    po->Register("max-length", &max_length, "Maximum length allowed for any "
                "segment (i.e. max #frames for any example");
    //po->Register("target-length", &target_length, "Target length for a "
    // "segment");
    po->Register("criterion", &criterion, "Criterion, 'mmi'|'mpfe'|'smbr'. "
                 "Determines which frames may be dropped from lattices.");
    po->Register("collapse-transition-ids", &collapse_transition_ids,
                 "This option included for debugging purposes");
    po->Register("determinize", &determinize, "If true, we determinize "
                 "lattices (as Lattice) before splitting and possibly minimize");
    po->Register("minimize", &minimize, "If true, we push and "
                 "minimize lattices (as Lattice) before splitting");
    po->Register("test", &test, "If true, activate self-testing code.");
    // See "Sequence-discriminative training of deep neural networks", Vesely et al,
    // ICASSP 2013 for explanation of frame dropping.
    po->Register("drop-frames", &drop_frames, "For MMI, if true we drop frames "
                 "with no overlap of num and den frames");
    po->Register("split", &split, "Set to false to disable lattice-splitting.");
    po->Register("excise", &excise, "Set to false to disable excising un-needed "
                 "frames (option included for debug purposes)");
  }
};

/// This struct exists only for diagnostic purposes.  Note: the stats assume
/// that you call SplitDiscriminative and ExciseDiscriminativeExample in the
/// same program, and the info printed out will be wrong if this is not the
/// case... this isn't ideal but it was more convenient.
struct SplitExampleStats {
  int32 num_lattices;
  int32 longest_lattice;
  int32 num_segments;
  int32 num_kept_segments;
  int64 num_frames_orig;
  int64 num_frames_must_keep;
  int64 num_frames_kept_after_split;
  int32 longest_segment_after_split;
  int64 num_frames_kept_after_excise;
  int32 longest_segment_after_excise;
  
  SplitExampleStats() { memset(this, 0, sizeof(*this)); }
  void Print();
};

/** Converts lattice to discriminative training example.  returns true on
    success, false on failure such as mismatched input (will also warn in this
    case). */
bool LatticeToDiscriminativeExample(
    const std::vector<int32> &alignment,
    const Vector<BaseFloat> &spk_vec,
    const Matrix<BaseFloat> &feats,
    const CompactLattice &clat,
    BaseFloat weight,
    int32 left_context,
    int32 right_context,
    DiscriminativeNnetExample *eg);


/** Split a "discriminative example" into multiple pieces,
    splitting where the lattice has "pinch points".
 */
void SplitDiscriminativeExample(
    const SplitDiscriminativeExampleConfig &config,
    const TransitionModel &tmodel,
    const DiscriminativeNnetExample &eg,
    std::vector<DiscriminativeNnetExample> *egs_out,
    SplitExampleStats *stats_out);

/** Remove unnecessary frames from discriminative training
    example.  The output egs_out will be of size zero or one
    (usually one) after being called. */
void ExciseDiscriminativeExample(
    const SplitDiscriminativeExampleConfig &config,
    const TransitionModel &tmodel,
    const DiscriminativeNnetExample &eg,
    std::vector<DiscriminativeNnetExample> *egs_out,
    SplitExampleStats *stats_out);


/** Appends the given vector of examples (which must be non-empty) into 
    a single output example (called by CombineExamples, which might be
    a more convenient interface).

   When combining examples it directly appends the features, and then adds a
   "fake" segment to the lattice and alignment in between, padding with
   transition-ids that are all ones.  This is necessary in case the network
   needs acoustic context, and only because of a kind of limitation in the nnet
   training code that doesn't support varying 'chunk' sizes within a minibatch.

   Will fail if all the input examples don't have the same weight (this will
   normally be 1.0 anyway).

   Will crash if the spk_info variables are non-empty (don't call it in that
   case).
*/
void AppendDiscriminativeExamples(
    const std::vector<const DiscriminativeNnetExample*> &input,
    DiscriminativeNnetExample *output);

/**
   This function is used to combine multiple discriminative-training
   examples (each corresponding to a segment of a lattice), into one.
   
   It combines examples into groups such that each group will have a
   total length (number of rows of the feature matrix) less than or
   equal to max_length.  However, if individual examples are longer
   than max_length they will still be processed; they will be given
   their own group.
   
   See also the documentation for AppendDiscriminativeExamples() which
   gives more details on how we append the examples.

   Will fail if all the input examples don't have the same weight (this will
   normally be 1.0 anyway).

   If the spk_info variables are non-empty, it won't attempt to combine the
   examples, it will just copy them to the output.  If we later need to
   extend it to work with spk_info data (e.g. combining examples from the
   same speaker), we can do that.
*/
void CombineDiscriminativeExamples(
    int32 max_length,
    const std::vector<DiscriminativeNnetExample> &input,
    std::vector<DiscriminativeNnetExample> *output);
                     
/**
   This function solves the "packing problem" using the "first fit" algorithm.
   It groups together the indices 0 through sizes.size() - 1, such that the sum
   of cost within each group does not exceed max_lcost.  [However, if there
   are single examples that exceed max_cost, it puts them in their own bin].
   The algorithm is not particularly efficient-- it's more n^2 than n log(n)
   which it should be.  */
void SolvePackingProblem(BaseFloat max_cost,
                         const std::vector<BaseFloat> &costs,
                         std::vector<std::vector<size_t> > *groups);



/**
   Given a discriminative training example, this function works out posteriors
   at the pdf level (note: these are "discriminative-training posteriors" that
   may be positive or negative.  The denominator lattice "den_lat" in the
   example "eg" should already have had acoustic-rescoring done so that its
   acoustic probs are up to date, and any acoustic scaling should already have
   been applied.

   "criterion" may be "mmi" or "mpfe" or "smbr".  If criterion
   is "mmi", "drop_frames" means we don't include derivatives for frames
   where the numerator pdf is not in the denominator lattice.

   "silence_phones" is a list of silence phones (this is only relevant for mpfe
   or smbr, if we want to treat silence specially).
 */
void ExampleToPdfPost(
    const TransitionModel &tmodel,
    const std::vector<int32> &silence_phones,
    std::string criterion,
    bool drop_frames,
    const DiscriminativeNnetExample &eg,
    Posterior *post);

/**
   This function is used in code that tests the functionality that we provide
   here, about splitting and excising nnet examples.  It adds to a "hash
   function" that is a function of a set of examples; the hash function is of
   dimension (number of pdf-ids x features dimension).  The hash function
   consists of the (denominator - numerator) posteriors over pdf-ids, times the
   average over the context-window (left-context on the left, right-context on
   the right), of the features.  This is useful because the various
   manipulations we do are supposed to preserve this, and if there is a bug
   it will most likely cause the hash function to change.

   This function will resize the matrix if it is empty.

   Any acoustic scaling of the lattice should be done before you call this
   function.

   'criterion' should be 'mmi', 'mpfe', or 'smbr'.
   
   You should set drop_frames to true if you are doing MMI with drop-frames
   == true.  Then it will not compute the hash for frames where the numerator
   pdf-id is not in the denominator lattice.

   The function will also accumulate the total numerator and denominator weights
   used as num_weight and den_weight, for an additional diagnostic, and the total
   number of frames, as tot_t.
*/
void UpdateHash(
    const TransitionModel &tmodel,
    const DiscriminativeNnetExample &eg,
    std::string criterion,
    bool drop_frames,
    Matrix<double> *hash,
    double *num_weight,
    double *den_weight,
    double *tot_t);



} // namespace nnet2
} // namespace kaldi

#endif // KALDI_NNET2_NNET_EXAMPLE_FUNCTIONS_H_
