// lat/sausages.h

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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


#ifndef KALDI_LAT_SAUSAGES_H_
#define KALDI_LAT_SAUSAGES_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {

/// The implementation of the Minimum Bayes Risk decoding method described in
///  "Minimum Bayes Risk decoding and system combination based on a recursion for
///  edit distance", Haihua Xu, Daniel Povey, Lidia Mangu and Jie Zhu, Computer
///  Speech and Language, 2011
/// This is a slightly more principled way to do Minimum Bayes Risk (MBR) decoding
/// than the standard "Confusion Network" method.  Note: MBR decoding aims to
/// minimize the expected word error rate, assuming the lattice encodes the
/// true uncertainty about what was spoken; standard Viterbi decoding gives the
/// most likely utterance, which corresponds to minimizing the expected sentence
/// error rate.
///
/// In addition to giving the MBR output, we also provide a way to get a
/// "Confusion Network" or informally "sausage"-like structure.  This is a
/// linear sequence of bins, and in each bin, there is a distribution over
/// words (or epsilon, meaning no word).  This is useful for estimating
/// confidence.  Note: due to the way these sausages are made, typically there
/// will be, between each bin representing a high-confidence word, a bin
/// in which epsilon (no word) is the most likely word.  Inside these bins
/// is where we put possible insertions. 


/// This class does the word-level Minimum Bayes Risk computation, and gives you
/// either the 1-best MBR output together with the expected Bayes Risk,
/// or a sausage-like structure.
class MinimumBayesRisk {
 public:
  /// Initialize with compact lattice-- any acoustic scaling etc., is assumed
  /// to have been done already.
  /// This does the whole computation.  You get the output with
  /// GetOneBest(), GetBayesRisk(), and GetSausageStats().
  MinimumBayesRisk(const CompactLattice &clat, bool do_mbr = true); // if do_mbr == false,
  // it will just use the MAP recognition output, but will get the MBR stats for things
  // like confidences.
  
  const std::vector<int32> &GetOneBest() const { // gets one-best (with no epsilons)
    return R_;
  }

  const std::vector<std::pair<BaseFloat, BaseFloat> > GetSausageTimes() const {
    return times_; // returns average (start,end) times for each bin (each entry
    // of GetSausageStats()).  Note: if you want the times for the one best,
    // you can work out the one best yourself from the sausage stats and get the times
    // at the same time.
  }

  const std::vector<std::pair<BaseFloat, BaseFloat> > &GetOneBestTimes() const {
    return one_best_times_; // returns average (start,end) times for each bin corresponding
    // to an entry in the one-best output.  This is just the appropriate
    // subsequence of the times in SausageTimes().
  }

  /// Outputs the confidences for the one-best transcript.
  const std::vector<BaseFloat> &GetOneBestConfidences() const {
    return one_best_confidences_;
  }

  /// Returns the expected WER over this sentence (assuming
  /// model correctness.
  BaseFloat GetBayesRisk() const { return L_; }
  
  const std::vector<std::vector<std::pair<int32, BaseFloat> > > &GetSausageStats() const {
    return gamma_;
  }  

 private:
  /// Minimum-Bayes-Risk Decode. Top-level algorithm.  Figure 6 of the paper.
  void MbrDecode(); 

  /// The basic edit-distance function l(a,b), as in the paper.
  inline double l(int32 a, int32 b) { return (a == b ? 0.0 : 1.0); }
  
  /// returns r_q, in one-based indexing, as in the paper.
  inline int32 r(int32 q) { return R_[q-1]; }
  
  
  /// Figure 4 of the paper; called from AccStats (Fig. 5)
  double EditDistance(int32 N, int32 Q,
                      Vector<double> &alpha,
                      Matrix<double> &alpha_dash,
                      Vector<double> &alpha_dash_arc);

  /// Figure 5 of the paper.  Outputs to gamma_ and L_.
  void AccStats(); 

  /// Removes epsilons (symbol 0) from a vector
  static void RemoveEps(std::vector<int32> *vec); 

  // Ensures that between each word in "vec" and at the beginning and end, is
  // epsilon (0).  (But if no words in vec, just one epsilon)
  static void NormalizeEps(std::vector<int32> *vec);   

  static inline BaseFloat delta() { return 1.0e-05; } // A constant
  // used in the algorithm.

  /// Function used to increment map.
  static inline void AddToMap(int32 i, double d, std::map<int32, double> *gamma) {
    if (d == 0) return;
    std::pair<const int32, double> pr(i, d);
    std::pair<std::map<int32, double>::iterator, bool> ret = gamma->insert(pr);
    if (!ret.second) // not inserted, so add to contents.
      ret.first->second += d;
  }
    
  struct Arc {
    int32 word;
    int32 start_node;
    int32 end_node;
    BaseFloat loglike;
  };

  /// Boolean configuration parameter: if true, we actually update the hypothesis
  /// to do MBR decoding (if false, our output is the MAP decoded output, but we
  /// output the stats too).
  bool do_mbr_;
  
  /// Arcs in the topologically sorted acceptor form of the word-level lattice,
  /// with one final-state.  Contains (word-symbol, log-likelihood on arc ==
  /// negated cost).  Indexed from zero.
  std::vector<Arc> arcs_;

  /// For each node in the lattice, a list of arcs entering that node. Indexed
  /// from 1 (first node == 1).
  std::vector<std::vector<int32> > pre_;

  std::vector<int32> state_times_; // time of each state in the word lattice,
  // indexed from 1 (same index as into pre_)
  
  std::vector<int32> R_; // current 1-best word sequence, normalized to have
  // epsilons between each word and at the beginning and end.  R in paper...
  // caution: indexed from zero, not from 1 as in paper.

  double L_; // current averaged edit-distance between lattice and R_.
  // \hat{L} in paper.
  
  std::vector<std::vector<std::pair<int32, BaseFloat> > > gamma_;
  // The stats we accumulate; these are pairs of (posterior, word-id), and note
  // that word-id may be epsilon.  Caution: indexed from zero, not from 1 as in
  // paper.  We sort in reverse order on the second member (posterior), so more
  // likely word is first.

  std::vector<std::pair<BaseFloat, BaseFloat> > times_;
  // The average start and end times for each confusion-network bin.  This
  // is like an average over words, of the tau_b and tau_e quantities in
  // Appendix C of the paper.  Indexed from zero, like gamma_ and R_.

  std::vector<std::pair<BaseFloat, BaseFloat> > one_best_times_;
  // one_best_times_ is a subsequence of times_, corresponding to
  // (start,end) times of words in the one best output.  Actually these
  // times are averages over the bin that each word came from.

  std::vector<BaseFloat> one_best_confidences_;
  // vector of confidences for the 1-best output (which could be
  // the MAP output if do_mbr_ == false, or the MBR output otherwise).
  // Indexed by the same index as one_best_times_.
  
  struct GammaCompare{
    // should be like operator <.  But we want reverse order
    // on the 2nd element (posterior), so it'll be like operator
    // > that looks first at the posterior.
    bool operator () (const std::pair<int32, BaseFloat> &a,
                      const std::pair<int32, BaseFloat> &b) const {
      if (a.second > b.second) return true;
      else if (a.second < b.second) return false;
      else return a.first > b.first;
    }
  };
};

}  // namespace kaldi

#endif  // KALDI_LAT_SAUSAGES_H_
