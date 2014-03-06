// lat/kws-functions.h

// Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)

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


#ifndef KALDI_LAT_KWS_FUNCTIONS_H_
#define KALDI_LAT_KWS_FUNCTIONS_H_

#include "lat/kaldi-lattice.h"
#include "lat/kaldi-kws.h"

namespace kaldi {

// We store the time information of the arc into class "Interval". "Interval"
// has a public function "int32 Overlap(Interval interval)" which takes in
// another interval and returns the overlap of that interval and the current
// interval.
class Interval {
 public:
  Interval() {}
  Interval(int32 start, int32 end) : start_(start), end_(end) {}
  Interval(const Interval &interval) : start_(interval.Start()), end_(interval.End()) {}
  int32 Overlap(Interval interval) {
    return std::max<int32>(0, std::min(end_, interval.end_) -
                              std::max(start_, interval.start_));
  }
  int32 Start() const {return start_;}
  int32 End() const {return end_;}
  ~Interval() {}

 private:
  int32 start_;
  int32 end_;
};

// We define a function bool CompareInterval(const Interval &i1, const Interval
// &i2) to compare the Interval defined above. If interval i1 is in front of
// interval i2, then return true; otherwise return false.
bool CompareInterval(const Interval &i1, 
                     const Interval &i2); 

// This function clusters the arcs with same word id and overlapping time-spans.
// Examples of clusters:
// 0 1 a a (0.1s ~ 0.5s) and 2 3 a a (0.2s ~ 0.4s) are within the same cluster; 
// 0 1 a a (0.1s ~ 0.5s) and 5 6 b b (0.2s ~ 0.4s) are in different clusters; 
// 0 1 a a (0.1s ~ 0.5s) and 7 8 a a (0.9s ~ 1.4s) are also in different clusters.
// It puts disambiguating symbols in the olabels, leaving the words on the
// ilabels.
bool ClusterLattice(CompactLattice *clat, 
                    const vector<int32> &state_times);

// This function is something similar to LatticeForwardBackward(), but it is on
// the CompactLattice lattice format. Also we only need the alpha in the forward 
// path, not the posteriors.
bool ComputeCompactLatticeAlphas(const CompactLattice &lat,
                                 vector<double> *alpha);

// A sibling of the function CompactLatticeAlphas()... We compute the beta from
// the backward path here.
bool ComputeCompactLatticeBetas(const CompactLattice &lat,
                                vector<double> *beta);

// This function contains two steps: weight pushing and factor generation. The
// original ShortestDistance() is not very efficient, so we do the weight
// pushing and shortest path manually by computing the alphas and betas. The
// factor generation step expand the lattice to the LXTXT' semiring, with
// additional start state and end state (and corresponding arcs) added. 
bool CreateFactorTransducer(const CompactLattice &clat,
                            const vector<int32> &state_times, 
                            int32 utterance_id, 
                            KwsProductFst *factor_transducer);

// This function removes the arcs with long silence. By "long" we mean arcs with
// #frames exceeding the given max_silence_frames. We do this filtering because
// the gap between adjacent words in a keyword must be <= 0.5 second. 
// Note that we should not remove the arcs created in the factor generation
// step, so the "search area" is limited to the original arcs before factor
// generation. 
void RemoveLongSilences(int32 max_silence_frames, 
                        const vector<int32> &state_times, 
                        KwsProductFst *factor_transducer);

// Do the factor merging part: encode input and output, and apply weighted
// epsilon removal, determinization and minimization.  Modifies factor_transducer.
void DoFactorMerging(KwsProductFst *factor_transducer,
                     KwsLexicographicFst *index_transducer);

// Do the factor disambiguation step: remove the cluster id's for the non-final
// arcs and insert disambiguation symbols for the final arcs
void DoFactorDisambiguation(KwsLexicographicFst *index_transducer);

// Do the optimization: do encoded determinization, minimization
void OptimizeFactorTransducer(KwsLexicographicFst *index_transducer,
                              int32 max_states,
                              bool allow_partial);

// the following two functions will, if GetVerboseLevel() >= 2, check that the
// cost of the second-best path in the transducers is not negative, and print
// out some associated debugging info if GetVerboseLevel() >= 3.  The best path
// in the transducers will typically be for the empty word sequence, and it may
// have negative cost (i.e. probability more than one), but the second-best one
// should not have negative cost.  A warning will be printed if
// GetVerboseLevel() >= 2 and a substantially negative cost is found.
void MaybeDoSanityCheck(const KwsProductFst &factor_transducer);
void MaybeDoSanityCheck(const KwsLexicographicFst &index_transducer);


} // namespace kaldi


#endif  // KALDI_LAT_KWS_FUNCTIONS_H_
