// decoder/lattice-incremental-decoder.h

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

#ifndef KALDI_DECODER_LATTICE_INCREMENTAL_DECODER_H_
#define KALDI_DECODER_LATTICE_INCREMENTAL_DECODER_H_

#include "util/stl-utils.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "fstext/fstext-lib.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"
#include "decoder/grammar-fst.h"
#include "lattice-faster-decoder.h"

namespace kaldi {

struct LatticeIncrementalDecoderConfig {
  BaseFloat beam;
  int32 max_active;
  int32 min_active;
  BaseFloat lattice_beam;
  int32 prune_interval;
  int32 determinize_delay;
  bool determinize_lattice; // not inspected by this class... used in
                            // command-line program.
  BaseFloat beam_delta;     // has nothing to do with beam_ratio
  BaseFloat hash_ratio;
  BaseFloat
      prune_scale; // Note: we don't make this configurable on the command line,
                   // it's not a very important parameter.  It affects the
                   // algorithm that prunes the tokens as we go.
  // Most of the options inside det_opts are not actually queried by the
  // LatticeIncrementalDecoder class itself, but by the code that calls it, for
  // example in the function DecodeUtteranceLatticeIncremental.
  int32 max_word_id; // for GetLattice
  fst::DeterminizeLatticePhonePrunedOptions det_opts;

  LatticeIncrementalDecoderConfig()
      : beam(16.0),
        max_active(std::numeric_limits<int32>::max()),
        min_active(200),
        lattice_beam(10.0),
        prune_interval(25),
        determinize_delay(25),
        determinize_lattice(true),
        beam_delta(0.5),
        hash_ratio(2.0),
        prune_scale(0.1),
        max_word_id(1e7) {}
  void Register(OptionsItf *opts) {
    det_opts.Register(opts);
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("max-active", &max_active,
                   "Decoder max active states.  Larger->slower; "
                   "more accurate");
    opts->Register("min-active", &min_active, "Decoder minimum #active states.");
    opts->Register("lattice-beam", &lattice_beam,
                   "Lattice generation beam.  Larger->slower, "
                   "and deeper lattices");
    opts->Register("prune-interval", &prune_interval,
                   "Interval (in frames) at "
                   "which to prune tokens");
    opts->Register("determinize-delay", &determinize_delay,
                   "delay (in frames) at "
                   "which to incrementally determinize lattices");
    opts->Register("determinize-lattice", &determinize_lattice,
                   "If true, "
                   "determinize the lattice (lattice-determinization, keeping only "
                   "best pdf-sequence for each word-sequence).");
    opts->Register("beam-delta", &beam_delta,
                   "Increment used in decoding-- this "
                   "parameter is obscure and relates to a speedup in the way the "
                   "max-active constraint is applied.  Larger is more accurate.");
    opts->Register("hash-ratio", &hash_ratio,
                   "Setting used in decoder to "
                   "control hash behavior");
  }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && max_active > 1 && lattice_beam > 0.0 &&
                 min_active <= max_active && prune_interval > 0 &&
                 beam_delta > 0.0 && hash_ratio >= 1.0 && prune_scale > 0.0 &&
                 prune_scale < 1.0);
  }
};

/** This is the "normal" lattice-generating decoder.
    See \ref lattices_generation \ref decoders_faster and \ref decoders_simple
     for more information.

   The decoder is templated on the FST type and the token type.  The token type
   will normally be StdToken, but also may be BackpointerToken which is to support
   quick lookup of the current best path (see lattice-faster-online-decoder.h)

   The FST you invoke this decoder with is expected to equal
   Fst::Fst<fst::StdArc>, a.k.a. StdFst, or GrammarFst.  If you invoke it with
   FST == StdFst and it notices that the actual FST type is
   fst::VectorFst<fst::StdArc> or fst::ConstFst<fst::StdArc>, the decoder object
   will internally cast itself to one that is templated on those more specific
   types; this is an optimization for speed.
 */
template <typename FST, typename Token = decoder::StdToken>
class LatticeIncrementalDecoderTpl {
 public:
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using ForwardLinkT = decoder::ForwardLink<Token>;

  // Instantiate this class once for each thing you have to decode.
  // This version of the constructor does not take ownership of
  // 'fst'.
  LatticeIncrementalDecoderTpl(const FST &fst, const TransitionModel &trans_model,
                               const LatticeIncrementalDecoderConfig &config);

  // This version of the constructor takes ownership of the fst, and will delete
  // it when this object is destroyed.
  LatticeIncrementalDecoderTpl(const LatticeIncrementalDecoderConfig &config,
                               FST *fst, const TransitionModel &trans_model);

  void SetOptions(const LatticeIncrementalDecoderConfig &config) {
    config_ = config;
  }

  const LatticeIncrementalDecoderConfig &GetOptions() const { return config_; }

  ~LatticeIncrementalDecoderTpl();

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
  /// Returns true if result is nonempty (using the return status is deprecated,
  /// it will become void).  If "use_final_probs" is true AND we reached the
  /// final-state of the graph then it will include those as final-probs, else
  /// it will treat all final-probs as one.  Note: this just calls GetRawLattice()
  /// and figures out the shortest path.
  bool GetBestPath(Lattice *ofst, bool use_final_probs = true) const;

  /// Outputs an FST corresponding to the raw, state-level
  /// tracebacks.  Returns true if result is nonempty.
  /// If "use_final_probs" is true AND we reached the final-state
  /// of the graph then it will include those as final-probs, else
  /// it will treat all final-probs as one.
  /// The raw lattice will be topologically sorted.
  ///
  /// See also GetRawLatticePruned in lattice-faster-online-decoder.h,
  /// which also supports a pruning beam, in case for some reason
  /// you want it pruned tighter than the regular lattice beam.
  /// We could put that here in future needed.
  bool GetRawLattice(Lattice *ofst, bool use_final_probs = true) const;
  bool GetCompactLattice(CompactLattice *ofst) const;

  /// InitDecoding initializes the decoding, and should only be used if you
  /// intend to call AdvanceDecoding().  If you call Decode(), you don't need to
  /// call this.  You can also call InitDecoding if you have already decoded an
  /// utterance and want to start with a new utterance.
  void InitDecoding();

  /// This will decode until there are no more frames ready in the decodable
  /// object.  You can keep calling it each time more frames become available.
  /// If max_num_frames is specified, it specifies the maximum number of frames
  /// the function will decode before returning.
  void AdvanceDecoding(DecodableInterface *decodable, int32 max_num_frames = -1);

  /// This function may be optionally called after AdvanceDecoding(), when you
  /// do not plan to decode any further.  It does an extra pruning step that
  /// will help to prune the lattices output by GetLattice and (particularly)
  /// GetRawLattice more accurately, particularly toward the end of the
  /// utterance.  It does this by using the final-probs in pruning (if any
  /// final-state survived); it also does a final pruning step that visits all
  /// states (the pruning that is done during decoding may fail to prune states
  /// that are within kPruningScale = 0.1 outside of the beam).  If you call
  /// this, you cannot call AdvanceDecoding again (it will fail), and you
  /// cannot call GetLattice() and related functions with use_final_probs =
  /// false.
  /// Used to be called PruneActiveTokensFinal().
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

 protected:
  // we make things protected instead of private, as code in
  // LatticeIncrementalOnlineDecoderTpl, which inherits from this, also uses the
  // internals.

  // Deletes the elements of the singly linked list tok->links.
  inline static void DeleteForwardLinks(Token *tok);

  // head of per-frame list of Tokens (list is in topological order),
  // and something saying whether we ever pruned it using PruneForwardLinks.
  struct TokenList {
    Token *toks;
    bool must_prune_forward_links;
    bool must_prune_tokens;
    TokenList()
        : toks(NULL), must_prune_forward_links(true), must_prune_tokens(true) {}
  };

  using Elem = typename HashList<StateId, Token *>::Elem;
  // Equivalent to:
  //  struct Elem {
  //    StateId key;
  //    Token *val;
  //    Elem *tail;
  //  };

  void PossiblyResizeHash(size_t num_toks);

  // FindOrAddToken either locates a token in hash of toks_, or if necessary
  // inserts a new, empty token (i.e. with no forward links) for the current
  // frame.  [note: it's inserted if necessary into hash toks_ and also into the
  // singly linked list of tokens active on this frame (whose head is at
  // active_toks_[frame]).  The frame_plus_one argument is the acoustic frame
  // index plus one, which is used to index into the active_toks_ array.
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true if the
  // token was newly created or the cost changed.
  // If Token == StdToken, the 'backpointer' argument has no purpose (and will
  // hopefully be optimized out).
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
                         bool *links_pruned, BaseFloat delta);

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
  void ComputeFinalCosts(unordered_map<Token *, BaseFloat> *final_costs,
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
  BaseFloat GetCutoff(Elem *list_head, size_t *tok_count, BaseFloat *adaptive_beam,
                      Elem **best_elem);

  /// Processes emitting arcs for one frame.  Propagates from prev_toks_ to
  /// cur_toks_.  Returns the cost cutoff for subsequent ProcessNonemitting() to
  /// use.
  BaseFloat ProcessEmitting(DecodableInterface *decodable);

  /// Processes nonemitting (epsilon) arcs for one frame.  Called after
  /// ProcessEmitting() on each frame.  The cost cutoff is computed by the
  /// preceding ProcessEmitting().
  void ProcessNonemitting(BaseFloat cost_cutoff);

  // HashList defined in ../util/hash-list.h.  It actually allows us to maintain
  // more than one list (e.g. for current and previous frames), but only one of
  // them at a time can be indexed by StateId.  It is indexed by frame-index
  // plus one, where the frame-index is zero-based, as used in decodable object.
  // That is, the emitting probs of frame t are accounted for in tokens at
  // toks_[t+1].  The zeroth frame is for nonemitting transition at the start of
  // the graph.
  HashList<StateId, Token *> toks_;

  std::vector<TokenList> active_toks_; // Lists of tokens, indexed by
  // frame (members of TokenList are toks, must_prune_forward_links,
  // must_prune_tokens).
  std::vector<StateId> queue_;       // temp variable used in ProcessNonemitting,
  std::vector<BaseFloat> tmp_array_; // used in GetCutoff.

  // fst_ is a pointer to the FST we are decoding from.
  const FST *fst_;
  // delete_fst_ is true if the pointer fst_ needs to be deleted when this
  // object is destroyed.
  bool delete_fst_;

  std::vector<BaseFloat> cost_offsets_; // This contains, for each
  // frame, an offset that was added to the acoustic log-likelihoods on that
  // frame in order to keep everything in a nice dynamic range i.e.  close to
  // zero, to reduce roundoff errors.
  LatticeIncrementalDecoderConfig config_;
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
  unordered_map<Token *, BaseFloat> final_costs_;
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
  static void TopSortTokens(Token *tok_list, std::vector<Token *> *topsorted_list);

  void ClearActiveTokens();

  /// The following part is specifically designed for incremental determinization
  ///
  /// The function obtains a CompactLattice for the part of this utterance that has
  /// been decoded so far.  If you call this multiple times (calling it on
  /// every frame would not make sense,
  /// but every, say, 10, to 40 frames might make sense) it will spread out
  /// the
  /// work of determinization over time,which might be useful for online
  /// applications.
  ///
  ///   @param [in]  use_final_probs  If true *and* at least one final-state in HCLG
  ///                         was active on the final frame, include final-probs from
  ///                         HCLG
  ///                         in the lattice.  Otherwise treat all final-costs of
  ///                         states active
  ///                         on the most recent frame as zero (i.e.  Weight::One()).
  ///   @param [in]  redeterminize    If true, re-determinize the CompactLattice
  ///                         after appending the most recently decoded chunk to it,
  ///                         to
  ///                         ensure that the output is fully deterministic.
  ///                         This does extra work, but not nearly as much as
  ///                         determinizing
  ///                          a RawLattice from scratch.
  ///  @param [out] lat   The CompactLattice representing what has been decoded
  ///                          so far.
  ///   @return reached_final    This function will returns true if a state that was
  ///   final in
  ///                         HCLG was active on the most recent frame, and false
  ///                         otherwise.
  ///                         CAUTION: this is not the same meaning as the return
  ///                         value of
  ///                         LatticeFasterDecoder::GetLattice().
  bool GetLattice(bool use_final_probs, bool redeterminize,
                  int32 last_frame_of_chunk, CompactLattice *olat);
  CompactLattice lat_;                            // the compact lattice we obtain
  int32 last_get_lattice_frame_;                  // the last time we call GetLattice
  unordered_map<Token *, int32> state_label_map_; // between Token and state_label
  int32 state_label_available_idx_;    // we allocate a unique id for each Token
  const TransitionModel &trans_model_; // keep it for determinization
  std::vector<std::pair<StateId, size_t>> final_arc_list_; // keep final_arc
  std::vector<std::pair<StateId, size_t>> final_arc_list_prev_;
  // We keep tot_cost or extra_cost for each state_label (Token) in final and
  // initial arcs. We need them before determinization
  // We cancel them after determinization
  // TODO use 2 vector to replace this map, since state_label is continuous
  // in each frame, and we need 2 frames of them
  unordered_map<int32, BaseFloat> state_label_initial_cost_;
  unordered_map<int32, BaseFloat> state_label_final_cost_;

  /// This function is modified from LatticeFasterDecoderTpl::GetRawLattice()
  /// and specific design for incremental GetLattice
  /// It does the same thing as GetRawLattice in lattice-faster-decoder.cc except:
  ///
  /// i) it creates a initial state, and connect
  /// all the tokens in the first frame of this chunk to the initial state
  /// by an arc with a per-token state-label as its olabel
  /// ii) it creates a final state, and connect
  /// all the tokens in the last frame of this chunk to the final state
  /// by an arc with a per-token state-label as its olabel
  /// the state-label for a token in both i) and ii) should be the same
  /// frame_begin and frame_end are the first and last frame of this chunk
  /// if create_initial_state == false, we will not create initial state and
  /// the corresponding state-label arcs. Similar for create_final_state
  /// In incremental GetLattice, we do not create the initial state in
  /// the first chunk, and we do not create the final state in the last chunk
  bool GetRawLattice(Lattice *ofst, bool use_final_probs, int32 frame_begin,
                     int32 frame_end, bool create_initial_state,
                     bool create_final_state);

  // Take care of the step 3 in GetLattice, which is to
  // appending the new chunk in clat to the old one in olat
  // If not_first_chunk == false, we do not need to append and just copy
  // clat into olat
  // Otherwise, we need to connect the last frame state of
  // last chunk to the first frame state of this chunk.
  // These begin and final states are corresponding to the same Token,
  // guaranteed by unique state labels.
  void AppendLatticeChunks(CompactLattice clat, bool not_first_chunk,
                           int32 last_frame_of_chunk, CompactLattice *olat);

  BaseFloat best_cost_in_chunk_;                // for sanity check
  unordered_set<int32> initial_state_in_chunk_; // for sanity check
  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeIncrementalDecoderTpl);
};

typedef LatticeIncrementalDecoderTpl<fst::StdFst, decoder::StdToken>
    LatticeIncrementalDecoder;

} // end namespace kaldi.

#endif
