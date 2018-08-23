// decoder/lattice-faster-online-decoder.h

// Copyright 2009-2013  Microsoft Corporation;  Mirko Hannemann;
//           2013-2014  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen
//                2018  Zhehuai Chen

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

// see note at the top of lattice-faster-decoder.h, about how to maintain this
// file in sync with lattice-faster-decoder.h


#ifndef KALDI_DECODER_LATTICE_FASTER_ONLINE_DECODER_H_
#define KALDI_DECODER_LATTICE_FASTER_ONLINE_DECODER_H_

#include "util/stl-utils.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "fstext/fstext-lib.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"
// Use the same configuration class as LatticeFasterDecoder.
#include "decoder/lattice-faster-decoder.h"

namespace kaldi {



/** LatticeFasterOnlineDecoder is as LatticeFasterDecoder but also supports an
    efficient way to get the best path (see the function BestPathEnd()), which
    is useful in endpointing.
 */
class LatticeFasterOnlineDecoder {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  struct BestPathIterator {
    void *tok;
    int32 frame;
    // note, "frame" is the frame-index of the frame you'll get the
    // transition-id for next time, if you call TraceBackBestPath on this
    // iterator (assuming it's not an epsilon transition).  Note that this
    // is one less than you might reasonably expect, e.g. it's -1 for
    // the nonemitting transitions before the first frame.
    BestPathIterator(void *t, int32 f): tok(t), frame(f) { }
    bool Done() { return tok == NULL; }
  };

  // instantiate this class once for each thing you have to decode.
  LatticeFasterOnlineDecoder(const fst::Fst<fst::StdArc> &fst,
                             const LatticeFasterDecoderConfig &config);

  // This version of the initializer "takes ownership" of the fst,
  // and will delete it when this object is destroyed.
  LatticeFasterOnlineDecoder(const LatticeFasterDecoderConfig &config,
                             fst::Fst<fst::StdArc> *fst);


  void SetOptions(const LatticeFasterDecoderConfig &config) {
    config_ = config;
  }

  const LatticeFasterDecoderConfig &GetOptions() const {
    return config_;
  }

  ~LatticeFasterOnlineDecoder();

  /// Decodes until there are no more frames left in the "decodable" object..
  /// note, this may block waiting for input if the "decodable" object blocks.
  /// Returns true if any kind of traceback is available (not necessarily from a
  /// final state).
  bool Decode(DecodableInterface *decodable);


  /// says whether a final-state was active on the last frame.  If it was not, the
  /// lattice (or traceback) will end with states that are not final-states.
  bool ReachedFinal() const {
    return FinalRelativeCost() != std::numeric_limits<BaseFloat>::infinity();
  }

  /// Outputs an FST corresponding to the single best path through the lattice.
  /// This is quite efficient because it doesn't get the entire raw lattice and find
  /// the best path through it; insterad, it uses the BestPathEnd and BestPathIterator
  /// so it basically traces it back through the lattice.
  /// Returns true if result is nonempty (using the return status is deprecated,
  /// it will become void).  If "use_final_probs" is true AND we reached the
  /// final-state of the graph then it will include those as final-probs, else
  /// it will treat all final-probs as one.
  bool GetBestPath(Lattice *ofst,
                   bool use_final_probs = true) const;


  /// This function does a self-test of GetBestPath().  Returns true on
  /// success; returns false and prints a warning on failure.
  bool TestGetBestPath(bool use_final_probs = true) const;


  /// This function returns an iterator that can be used to trace back
  /// the best path.  If use_final_probs == true and at least one final state
  /// survived till the end, it will use the final-probs in working out the best
  /// final Token, and will output the final cost to *final_cost (if non-NULL),
  /// else it will use only the forward likelihood, and will put zero in
  /// *final_cost (if non-NULL).
  /// Requires that NumFramesDecoded() > 0.
  BestPathIterator BestPathEnd(bool use_final_probs,
                               BaseFloat *final_cost = NULL) const;


  /// This function can be used in conjunction with BestPathEnd() to trace back
  /// the best path one link at a time (e.g. this can be useful in endpoint
  /// detection).  By "link" we mean a link in the graph; not all links cross
  /// frame boundaries, but each time you see a nonzero ilabel you can interpret
  /// that as a frame.  The return value is the updated iterator.  It outputs
  /// the ilabel and olabel, and the (graph and acoustic) weight to the "arc" pointer,
  /// while leaving its "nextstate" variable unchanged.
  BestPathIterator TraceBackBestPath(
      BestPathIterator iter, LatticeArc *arc) const;

  /// Outputs an FST corresponding to the raw, state-level
  /// tracebacks.  Returns true if result is nonempty.
  /// If "use_final_probs" is true AND we reached the final-state
  /// of the graph then it will include those as final-probs, else
  /// it will treat all final-probs as one.
  /// The raw lattice will be topologically sorted.
  bool GetRawLattice(Lattice *ofst,
                     bool use_final_probs = true) const;

  /// Behaves the same like GetRawLattice but only processes tokens whose
  /// extra_cost is smaller than the best-cost plus the specified beam.
  /// It is only worthwhile to call this function if beam is less than
  /// the lattice_beam specified in the config; otherwise, it would
  /// return essentially the same thing as GetRawLattice, but more slowly.
  bool GetRawLatticePruned(Lattice *ofst,
                           bool use_final_probs,
                           BaseFloat beam) const;


  /// InitDecoding initializes the decoding, and should only be used if you
  /// intend to call AdvanceDecoding().  If you call Decode(), you don't need to
  /// call this.  You can also call InitDecoding if you have already decoded an
  /// utterance and want to start with a new utterance.
  void InitDecoding();

  /// This will decode until there are no more frames ready in the decodable
  /// object.  You can keep calling it each time more frames become available.
  /// If max_num_frames is specified, it specifies the maximum number of frames
  /// the function will decode before returning.
  void AdvanceDecoding(DecodableInterface *decodable,
                       int32 max_num_frames = -1);

  /// This function may be optionally called after AdvanceDecoding(), when you
  /// do not plan to decode any further.  It does an extra pruning step that
  /// will help to prune the lattices output by GetRawLattice more accurately,
  /// particularly toward the end of the utterance.  It does this by using the
  /// final-probs in pruning (if any final-state survived); it also does a final
  /// pruning step that visits all states (the pruning that is done during
  /// decoding may fail to prune states that are within kPruningScale = 0.1
  /// outside of the beam).  If you call this, you cannot call AdvanceDecoding
  /// again (it will fail), and you cannot call GetRawLattice() and related
  /// functions with use_final_probs = false.  Used to be called
  /// PruneActiveTokensFinal().
  void FinalizeDecoding();

  /// FinalRelativeCost() serves the same purpose as ReachedFinal(), but gives
  /// more information.  It returns the difference between the best (final-cost
  /// plus cost) of any token on the final frame, and the best cost of any token
  /// on the final frame.  If it is infinity it means no final-states were
  /// present on the final frame.  It will usually be nonnegative.  If it not
  /// too positive (e.g. < 5 is my first guess, but this is not tested) you can
  /// take it as a good indication that we reached the final-state with
  /// reasonable likelihood.
  BaseFloat FinalRelativeCost() const;

  // Returns the number of frames decoded so far.  The value returned changes
  // whenever we call ProcessEmitting().
  inline int32 NumFramesDecoded() const { return active_toks_.size() - 1; }

 private:
  // ForwardLinks are the links from a token to a token on the next frame.
  // or sometimes on the current frame (for input-epsilon links).
  struct Token;
  struct ForwardLink {
    Token *next_tok; // the next token [or NULL if represents final-state]
    Label ilabel; // ilabel on link.
    Label olabel; // olabel on link.
    BaseFloat graph_cost; // graph cost of traversing link (contains LM, etc.)
    BaseFloat acoustic_cost; // acoustic cost (pre-scaled) of traversing link
    ForwardLink *next; // next in singly-linked list of forward links from a
                       // token.
    inline ForwardLink(Token *next_tok, Label ilabel, Label olabel,
                       BaseFloat graph_cost, BaseFloat acoustic_cost,
                       ForwardLink *next):
        next_tok(next_tok), ilabel(ilabel), olabel(olabel),
        graph_cost(graph_cost), acoustic_cost(acoustic_cost),
        next(next) { }
  };

  // Token is what's resident in a particular state at a particular time.
  // In this decoder a Token actually contains *forward* links.
  // When first created, a Token just has the (total) cost.    We add forward
  // links from it when we process the next frame.
  struct Token {
    BaseFloat tot_cost; // would equal weight.Value()... cost up to this point.
    BaseFloat extra_cost; // >= 0.  After calling PruneForwardLinks, this equals
    // the minimum difference between the cost of the best path, and the cost of
    // this is on, and the cost of the absolute best path, under the assumption
    // that any of the currently active states at the decoding front may
    // eventually succeed (e.g. if you were to take the currently active states
    // one by one and compute this difference, and then take the minimum).

    ForwardLink *links; // Head of singly linked list of ForwardLinks

    Token *next; // Next in list of tokens for this frame.

    Token *backpointer; // best preceding Token (could be on this frame or a
                        // previous frame).  This is only required for an
                        // efficient GetBestPath function, it plays no part in
                        // the lattice generation (the "links" list is what
                        // stores the forward links, for that).

    inline Token(BaseFloat tot_cost, BaseFloat extra_cost, ForwardLink *links,
                 Token *next, Token *backpointer):
        tot_cost(tot_cost), extra_cost(extra_cost), links(links), next(next),
        backpointer(backpointer) { }
    inline void DeleteForwardLinks() {
      ForwardLink *l = links, *m;
      while (l != NULL) {
        m = l->next;
        delete l;
        l = m;
      }
      links = NULL;
    }
  };

  // head of per-frame list of Tokens (list is in topological order),
  // and something saying whether we ever pruned it using PruneForwardLinks.
  struct TokenList {
    Token *toks;
    bool must_prune_forward_links;
    bool must_prune_tokens;
    TokenList(): toks(NULL), must_prune_forward_links(true),
                 must_prune_tokens(true) { }
  };

  typedef HashList<StateId, Token*>::Elem Elem;

  void PossiblyResizeHash(size_t num_toks);

  // FindOrAddToken either locates a token in hash of toks_, or if necessary
  // inserts a new, empty token (i.e. with no forward links) for the current
  // frame.  [note: it's inserted if necessary into hash toks_ and also into the
  // singly linked list of tokens active on this frame (whose head is at
  // active_toks_[frame]).  The frame_plus_one argument is the acoustic frame
  // index plus one, which is used to index into the active_toks_ array.
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true if the
  // token was newly created or the cost changed.
  inline Token *FindOrAddToken(StateId state, int32 frame_plus_one,
                               BaseFloat tot_cost, Token *backpointer,
                               bool *changed);

  // prunes outgoing links for all tokens in active_toks_[frame]
  // it's called by PruneActiveTokens
  // all links, that have link_extra_cost > lattice_beam are pruned
  // delta is the amount by which the extra_costs must change
  // before we set *extra_costs_changed = true.
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned
  void PruneForwardLinks(int32 frame_plus_one, bool *extra_costs_changed,
                         bool *links_pruned,
                         BaseFloat delta);

  // This function computes the final-costs for tokens active on the final
  // frame.  It outputs to final-costs, if non-NULL, a map from the Token*
  // pointer to the final-prob of the corresponding state, for all Tokens
  // that correspond to states that have final-probs.  This map will be
  // empty if there were no final-probs.  It outputs to
  // final_relative_cost, if non-NULL, the difference between the best
  // forward-cost including the final-prob cost, and the best forward-cost
  // without including the final-prob cost (this will usually be positive), or
  // infinity if there were no final-probs.  [c.f. FinalRelativeCost(), which
  // outputs this quanitity].  It outputs to final_best_cost, if
  // non-NULL, the lowest for any token t active on the final frame, of
  // forward-cost[t] + final-cost[t], where final-cost[t] is the final-cost in
  // the graph of the state corresponding to token t, or the best of
  // forward-cost[t] if there were no final-probs active on the final frame.
  // You cannot call this after FinalizeDecoding() has been called; in that
  // case you should get the answer from class-member variables.
  void ComputeFinalCosts(unordered_map<Token*, BaseFloat> *final_costs,
                         BaseFloat *final_relative_cost,
                         BaseFloat *final_best_cost) const;

  // PruneForwardLinksFinal is a version of PruneForwardLinks that we call
  // on the final frame.  If there are final tokens active, it uses
  // the final-probs for pruning, otherwise it treats all tokens as final.
  void PruneForwardLinksFinal();

  // Prune away any tokens on this frame that have no forward links.
  // [we don't do this in PruneForwardLinks because it would give us
  // a problem with dangling pointers].
  // It's called by PruneActiveTokens if any forward links have been pruned
  void PruneTokensForFrame(int32 frame_plus_one);


  // Go backwards through still-alive tokens, pruning them if the
  // forward+backward cost is more than lat_beam away from the best path.  It's
  // possible to prove that this is "correct" in the sense that we won't lose
  // anything outside of lat_beam, regardless of what happens in the future.
  // delta controls when it considers a cost to have changed enough to continue
  // going backward and propagating the change.  larger delta -> will recurse
  // less far.
  void PruneActiveTokens(BaseFloat delta);

  /// Gets the weight cutoff.  Also counts the active tokens.
  BaseFloat GetCutoff(Elem *list_head, size_t *tok_count,
                      BaseFloat *adaptive_beam, Elem **best_elem);

  /// Processes emitting arcs for one frame.  Propagates from prev_toks_ to cur_toks_.
  /// Returns the cost cutoff for subsequent ProcessNonemitting() to use.
  /// Templated on FST type for speed; called via ProcessEmittingWrapper().
  template <typename FstType> BaseFloat ProcessEmitting(DecodableInterface *decodable);

  BaseFloat ProcessEmittingWrapper(DecodableInterface *decodable);

  /// Processes nonemitting (epsilon) arcs for one frame.  Called after
  /// ProcessEmitting() on each frame.  The cost cutoff is computed by the
  /// preceding ProcessEmitting().
  /// the templated design is similar to ProcessEmitting()
  template <typename FstType> void ProcessNonemitting(BaseFloat cost_cutoff);

  void ProcessNonemittingWrapper(BaseFloat cost_cutoff);

  // HashList defined in ../util/hash-list.h.  It actually allows us to maintain
  // more than one list (e.g. for current and previous frames), but only one of
  // them at a time can be indexed by StateId.  It is indexed by frame-index
  // plus one, where the frame-index is zero-based, as used in decodable object.
  // That is, the emitting probs of frame t are accounted for in tokens at
  // toks_[t+1].  The zeroth frame is for nonemitting transition at the start of
  // the graph.
  HashList<StateId, Token*> toks_;

  std::vector<TokenList> active_toks_; // Lists of tokens, indexed by
  // frame (members of TokenList are toks, must_prune_forward_links,
  // must_prune_tokens).
  std::vector<StateId> queue_;  // temp variable used in ProcessNonemitting,
  std::vector<BaseFloat> tmp_array_;  // used in GetCutoff.
  // make it class member to avoid internal new/delete.
  const fst::Fst<fst::StdArc> &fst_;
  bool delete_fst_;
  std::vector<BaseFloat> cost_offsets_; // This contains, for each
  // frame, an offset that was added to the acoustic log-likelihoods on that
  // frame in order to keep everything in a nice dynamic range i.e.  close to
  // zero, to reduce roundoff errors.
  LatticeFasterDecoderConfig config_;
  int32 num_toks_; // current total #toks allocated...
  bool warned_;

  /// decoding_finalized_ is true if someone called FinalizeDecoding().  [note,
  /// calling this is optional].  If true, it's forbidden to decode more.  Also,
  /// if this is set, then the output of ComputeFinalCosts() is in the next
  /// three variables.  The reason we need to do this is that after
  /// FinalizeDecoding() calls PruneTokensForFrame() for the final frame, some
  /// of the tokens on the last frame are freed, so we free the list from toks_
  /// to avoid having dangling pointers hanging around.
  bool decoding_finalized_;
  /// For the meaning of the next 3 variables, see the comment for
  /// decoding_finalized_ above., and ComputeFinalCosts().
  unordered_map<Token*, BaseFloat> final_costs_;
  BaseFloat final_relative_cost_;
  BaseFloat final_best_cost_;

  // There are various cleanup tasks... the the toks_ structure contains
  // singly linked lists of Token pointers, where Elem is the list type.
  // It also indexes them in a hash, indexed by state (this hash is only
  // maintained for the most recent frame).  toks_.Clear()
  // deletes them from the hash and returns the list of Elems.  The
  // function DeleteElems calls toks_.Delete(elem) for each elem in
  // the list, which returns ownership of the Elem to the toks_ structure
  // for reuse, but does not delete the Token pointer.  The Token pointers
  // are reference-counted and are ultimately deleted in PruneTokensForFrame,
  // but are also linked together on each frame by their own linked-list,
  // using the "next" pointer.  We delete them manually.
  void DeleteElems(Elem *list);

  // This function takes a singly linked list of tokens for a single frame, and
  // outputs a list of them in topological order (it will crash if no such order
  // can be found, which will typically be due to decoding graphs with epsilon
  // cycles, which are not allowed).  Note: the output list may contain NULLs,
  // which the caller should pass over; it just happens to be more efficient for
  // the algorithm to output a list that contains NULLs.
  static void TopSortTokens(Token *tok_list,
                            std::vector<Token*> *topsorted_list);

  void ClearActiveTokens();


  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeFasterOnlineDecoder);
};



} // end namespace kaldi.

#endif
