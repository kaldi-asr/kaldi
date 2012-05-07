// lat/lattice-functions.h

// Copyright 2009-2011   Saarland University
// Author: Arnab Ghoshal

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


#ifndef KALDI_LAT_LATTICE_FUNCTIONS_H_
#define KALDI_LAT_LATTICE_FUNCTIONS_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {

/// This function iterates over the states of a topologically sorted lattice
/// and counts the time instance corresponding to each state. The times are
/// returned in a vector of integers 'times' which is resized to have a size
/// equal to the number of states in the lattice. The function also returns
/// the maximum time in the lattice (this will equal the #frames in the file).
int32 LatticeStateTimes(const Lattice &lat, std::vector<int32> *times);

/// As LatticeStateTimes, but in the CompactLattice format.  Note: must
/// be topologically sorted.  Returns length of the utterance in frames, which
/// may not be the same as the maximum time in the lattice, due to frames
/// in the final-prob.
int32 CompactLatticeStateTimes(const CompactLattice &lat,
                               std::vector<int32> *times);

/// This function does the forward-backward over lattices and computes the
/// posterior probabilities of the arcs. It returns the total log-probability
/// of the lattice.
BaseFloat LatticeForwardBackward(const Lattice &lat, Posterior *arc_post);

/// Given a lattice, and a transition model to map pdf-ids to phones,
/// outputs for each frame the set of phones active on that frame.  If
/// sil_phones (which must be sorted and uniq) is nonempty, it excludes
/// phones in this list.
void LatticeActivePhones(const Lattice &lat, const TransitionModel &trans,
                         const std::vector<int32> &sil_phones,
                         std::vector<std::set<int32> > *active_phones);

/// Given a lattice, and a transition model to map pdf-ids to phones,
/// replace the output symbols (presumably words), with phones; we
/// use the TransitionModel to work out the phone sequence.  Note
/// that the phone labels are not exactly aligned with the phone
/// boundaries.  We put a phone label to coincide with any transition
/// to the final, nonemitting state of a phone (this state always exists,
/// we ensure this in HmmTopology::Check()).  This would be the last
/// transition-id in the phone if reordering is not done (but typically
/// we do reorder).
void ConvertLatticeToPhones(const TransitionModel &trans_model,
                            Lattice *lat);

/// Boosts LM probabilities by b * [#frame errors]; equivalently, adds
/// -b*[#frame errors] to the graph-component of the cost of each arc/path.
/// There is a frame error if a particular transition-id on a particular frame
/// corresponds to a phone not appearining in active_phones for that frame.
/// This is used in "margin-inspired" discriminative training, esp. Boosted MMI.
/// The TransitionModel is used to map transition-ids in the lattice
/// input-side to phones; the phones appearing in
/// "silence_phones" are treated specially in that we replace the frame error f
/// (either zero or 1) for a frame, with the minimum of f or max_silence_error.
/// For the normal recipe, max_silence_error would be zero.
/// Returns true on success, false if there was some kind of mismatch.
/// At input, silence_phones must be sorted and unique.
bool LatticeBoost(const TransitionModel &trans,
                  const std::vector<std::set<int32> > &active_phones,
                  const std::vector<int32> &silence_phones,
                  BaseFloat b,
                  BaseFloat max_silence_error,
                  Lattice *lat);

int32 LatticePhoneFrameAccuracy(const Lattice &hyp, const TransitionModel &trans,
                               const std::vector< std::map<int32, int32> > &ref,
                               std::vector< std::map<int32, char> > *arc_accs,
                               std::vector<int32> *state_times);

BaseFloat LatticeForwardBackwardMpe(const Lattice &lat,
                                    const TransitionModel &trans,
                                    const vector< std::map<int32, char> > &arc_accs,
                                    Posterior *arc_post);
}  // namespace kaldi

#endif  // KALDI_LAT_LATTICE_FUNCTIONS_H_
