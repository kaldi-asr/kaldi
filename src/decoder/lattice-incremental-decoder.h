// decoder/lattice-incremental-decoder.h

// Copyright      2019  Zhehuai Chen

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
  int32 determinize_chunk_size;
  int32 determinize_max_active;
  bool redeterminize;
  int32 redeterminize_max_frames;
  bool epsilon_removal;
  BaseFloat beam_delta; // has nothing to do with beam_ratio
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
        determinize_chunk_size(20),
        determinize_max_active(std::numeric_limits<int32>::max()),
        redeterminize(false),
        redeterminize_max_frames(std::numeric_limits<int32>::max()),
        epsilon_removal(false),
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
                   "delay (in frames, typically larger than --prune-interval) "
                   "at which to incrementally determinize lattices.");
    opts->Register("determinize-chunk-size", &determinize_chunk_size,
                   "the size (in frames) of chunk to do incrementally "
                   "determinization. If working with --determinize-max-active,"
                   "it will become a lower bound of the size of chunk.");
    opts->Register("determinize-max-active", &determinize_max_active,
                   "This option is to adaptively decide the size of the chunk "
                   "to be determinized. "
                   "If the number of active tokens(in a certain frame) is less "
                   "than this number (typically 50), we will start to "
                   "incrementally determinize lattices from the last frame we "
                   "determinized up to this frame. It can work with "
                   "--determinize-delay to further reduce the computation "
                   "introduced by incremental determinization. ");
    opts->Register("redeterminize", &redeterminize,
                   "whether to re-determinize the lattice after incremental "
                   "determinization.");
    opts->Register("redeterminize_max_frames", &redeterminize_max_frames,
                   "To impose a limit on how far back in time we will "
                   "redeterminize states.  This is mainly intended to avoid "
                   "pathological cases. You could set it infinite to get a fully "
                   "determinized lattice.");
    opts->Register("epsilon-removal", &epsilon_removal,
                   "whether to remove epsilon when appending two adjacent chunks.");
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
                 determinize_delay >= 0 && determinize_max_active >= 0 &&
                 determinize_chunk_size >= 0 &&
                 redeterminize_max_frames >= 0 && beam_delta > 0.0 &&
                 hash_ratio >= 1.0 && prune_scale > 0.0 && prune_scale < 1.0);
  }
};

template <typename FST>
class LatticeIncrementalDeterminizer;

/* This is an extention to the "normal" lattice-generating decoder.
   See \ref lattices_generation \ref decoders_faster and \ref decoders_simple
    for more information.

   The main difference is the incremental determinization which will be
   discussed in the function GetLattice().

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
  bool GetBestPath(Lattice *ofst, bool use_final_probs = true);

  // The following function is specifically designed for incremental
  // determinization. The function obtains a CompactLattice for
  // the part of this utterance up to the frame last_frame_of_chunk.
  // If you call this multiple times
  // (calling it on every frame would not make sense,
  // but every, say, 10 to 40 frames might make sense) it will spread out the
  // work of determinization over time, which might be useful for online
  // applications.
  //
  // The procedure of incremental determinization is as follow:
  // step 1: Get lattice chunk with initial and final states and arcs, called `raw
  // lattice`.
  // Here, we define a `final arc` as an arc to a final-state, and the source state
  // of it as a `pre-final state`
  // Similarly, we define a `initial arc` as an arc from a initial-state, and the
  // destination state of it as a `post-initial state`
  // The post-initial states are constructed corresponding to pre-final states
  // in the determinized and appended lattice before this chunk
  // The pre-final states are constructed correponding to tokens in the last frames
  // of this chunk.
  // Since the StateId can change during determinization, we need to give permanent
  // unique labels (as olabel) to these
  // raw-lattice states for latter appending.
  // We give each token an olabel id, called `token_label`, and each determinized and
  // appended state an olabel id, called `state_label`
  // step 2: Determinize the chunk of above raw lattice using determinization
  // algorithm the same as LatticeFasterDecoder. Benefit from above `state_label` and
  // `token_label` in initial and final arcs, each pre-final state in the last chunk
  // w.r.t the initial arc of this chunk can be treated uniquely and each token in
  // the last frame of this chunk can also be treated uniquely. We call the
  // determinized new
  // chunk `compact lattice (clat)`
  // step 3: Appending the new chunk `clat` to the determinized lattice
  // before this chunk. First, for each StateId in clat except its
  // initial state, allocate a new StateId in the appended
  // compact lattice. Copy the arcs except whose incoming state is initial
  // state. Secondly, for each initial arcs, change its source state to the state
  // corresponding to its `state_label`, which is a determinized and appended state
  // Finally, we make the previous final arcs point to a "dead state"
  // step 4 (optional): We re-determinize the appended lattice if needed.
  //
  // In our implementation, step 1 is done in GetIncrementalRawLattice(),
  // step 2-4 is taken care by the class
  // LatticeIncrementalDeterminizer
  //
  //   @param [in]  use_final_probs  If true *and* at least one final-state in HCLG
  //                         was active on the final frame, include final-probs from
  //                         HCLG
  //                         in the lattice.  Otherwise treat all final-costs of
  //                         states active
  //                         on the most recent frame as zero (i.e.  Weight::One()).
  //   @param [in]  redeterminize    If true, re-determinize the CompactLattice
  //                         after appending the most recently decoded chunk to it,
  //                         to
  //                         ensure that the output is fully deterministic.
  //                         This does extra work, but not nearly as much as
  //                         determinizing
  //                          a RawLattice from scratch.
  //   @param [in]  last_frame_of_chunk  Pass the last frame of this chunk to
  //                       the function. We make it not always equal to
  //                         NumFramesDecoded() to have a delay on the
  //                       deteriminization
  //   @param [out] olat   The CompactLattice representing what has been decoded
  //                          so far.
  //                       If lat == NULL, the CompactLattice won't be outputed.
  //   @return ret   This function will returns true if the chunk is processed
  //                       successfully
  bool GetLattice(bool use_final_probs, bool redeterminize,
                  int32 last_frame_of_chunk, CompactLattice *olat = NULL);
  /// Specifically design when decoding_finalized_==true
  bool GetLattice(CompactLattice *olat);

  /// This function is to keep forwards compatibility.
  /// It outputs an FST corresponding to the raw, state-level
  /// tracebacks.  Returns true if result is nonempty.
  /// If "use_final_probs" is true AND we reached the final-state
  /// of the graph then it will include those as final-probs, else
  /// it will treat all final-probs as one.
  /// Notably, the raw lattice from this incremental determinization decoder
  /// has already been partially determinized
  bool GetRawLattice(Lattice *ofst, bool use_final_probs = true);

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
  // we make things protected instead of private, as future code in
  // LatticeIncrementalOnlineDecoderTpl, which inherits from this, also will
  // use the internals.

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

  // The following part is specifically designed for incremental determinization
  // This function is modified from LatticeFasterDecoderTpl::GetRawLattice()
  // and specific design for step 1 of incremental determinization
  // introduced before above GetLattice()
  // It does the same thing as GetRawLattice in lattice-faster-decoder.cc except:
  //
  // i) it creates a initial state, and connect
  // each token in the first frame of this chunk to the initial state
  // by one or more arcs with a state_label correponding to the pre-final state w.r.t
  // this token(the pre-final state is appended in the last chunk) as its olabel
  // ii) it creates a final state, and connect
  // all the tokens in the last frame of this chunk to the final state
  // by an arc with a per-token token_label as its olabel
  // `frame_begin` and `frame_end` are the first and last frame of this chunk
  // if `create_initial_state` == false, we will not create initial state and
  // the corresponding initial arcs. Similar for `create_final_state`
  // In incremental GetLattice, we do not create the initial state in
  // the first chunk, and we do not create the final state in the last chunk
  bool GetIncrementalRawLattice(Lattice *ofst, bool use_final_probs,
                                int32 frame_begin, int32 frame_end,
                                bool create_initial_state, bool create_final_state);
  // Get the number of tokens in each frame
  // It is useful, e.g. in using config_.determinize_max_active
  int32 GetNumToksForFrame(int32 frame);

  // The incremental lattice determinizer to take care of step 2-4
  LatticeIncrementalDeterminizer<FST> determinizer_;
  int32 last_get_lattice_frame_; // the last time we call GetLattice
  // a map from Token to its token_label
  unordered_map<Token *, int32> token_label_map_;
  // we allocate a unique id for each Token
  int32 token_label_available_idx_;
  // We keep cost_offset for each token_label (Token) in final arcs. We need them to
  // guide determinization
  // We cancel them after determinization
  unordered_map<int32, BaseFloat> token_label_final_cost_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeIncrementalDecoderTpl);
};

typedef LatticeIncrementalDecoderTpl<fst::StdFst, decoder::StdToken>
    LatticeIncrementalDecoder;

// This class is designed for step 2-4 and part of step 1 of incremental
// determinization
// introduced before above GetLattice()
template <typename FST>
class LatticeIncrementalDeterminizer {
 public:
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  LatticeIncrementalDeterminizer(const LatticeIncrementalDecoderConfig &config,
                                 const TransitionModel &trans_model);
  // Reset the lattice determinization data for an utterance
  void Init();
  // Output the resultant determinized lattice in the form of CompactLattice
  const CompactLattice &GetDeterminizedLattice() const { return lat_; }

  // Part of step 1 of incremental determinization,
  // where the post-initial states are constructed corresponding to
  // redeterminized states (see the description in redeterminized_states_) in the
  // determinized and appended lattice before this chunk.
  // We give each determinized and appended state an olabel id, called `state_label`
  // We maintain a map (`token_label2last_state_map`) from token label (obtained from
  // final arcs) to the destination state of the last of the sequence of initial arcs
  // w.r.t the token label here
  // Notably, we have multiple states for one token label after determinization,
  // hence we use multiset here
  // We need `token_label_final_cost` to cancel out the cost offset used in guiding
  // DeterminizeLatticePhonePrunedWrapper
  void GetInitialRawLattice(
      Lattice *olat,
      unordered_multimap<int, LatticeArc::StateId> *token_label2last_state_map,
      const unordered_map<int32, BaseFloat> &token_label_final_cost);
  // This function consumes raw_fst generated by step 1 of incremental
  // determinization with specific initial and final arcs.
  // It does step 2-4 and outputs the resultant CompactLattice if
  // needed. Otherwise, it keeps the resultant lattice in lat_
  bool ProcessChunk(Lattice &raw_fst, int32 first_frame, int32 last_frame);

  // Step 3 of incremental determinization,
  // which is to append the new chunk in clat to the old one in lat_
  // If not_first_chunk == false, we do not need to append and just copy
  // clat into olat
  // Otherwise, we need to connect states of the last frame of
  // the last chunk to states of the first frame of this chunk.
  // These post-initial and pre-final states are corresponding to the same Token,
  // guaranteed by unique state labels.
  bool AppendLatticeChunks(CompactLattice clat, bool not_first_chunk);

  // Step 4 of incremental determinization,
  // which either re-determinize above lat_, or simply remove the dead
  // states of lat_
  bool Finalize(bool redeterminize);
  std::vector<BaseFloat> &GetForwardCosts() { return forward_costs_; }

 private:
  // This function either locates a redeterminized state w.r.t nextstate previously
  // added, or if necessary inserts a new one.
  // The new one is inserted in olat and kept by the map (redeterminized_states_)
  // which is from the state in the appended compact lattice to the state_copy in the
  // raw lattice. The function returns whether a new one is inserted
  // The StateId of the redeterminized state will be outputed by nextstate_copy
  bool AddRedeterminizedState(Lattice::StateId nextstate, Lattice *olat,
                              Lattice::StateId *nextstate_copy = NULL);
  // Sub function of GetInitialRawLattice(). Refer to description there
  void GetRawLatticeForRedeterminizedStates(
      StateId start_state, StateId state,
      const unordered_map<int32, BaseFloat> &token_label_final_cost,
      unordered_multimap<int, LatticeArc::StateId> *token_label2last_state_map,
      Lattice *olat);
  // This function is to preprocess the appended compact lattice before
  // generating raw lattices for the next chunk.
  // After identifying pre-final states, for any such state that is separated by
  // more than config_.redeterminize_max_frames from the end of the current
  // appended lattice, we create an extra state for it; we add an epsilon arc
  // from that pre-final state to the extra state; we copy any final arcs from
  // the pre-final state to its extra state and we remove those final arcs from
  // the original pre-final state.  Now this extra state is the pre-final state to
  // redeterminize and the original pre-final state does not need to redeterminize
  // The epsilon would be removed later on in AppendLatticeChunks, while
  // splicing the compact lattices together
  void GetRedeterminizedStates();

  const LatticeIncrementalDecoderConfig config_;
  const TransitionModel &trans_model_; // keep it for determinization

  // Record whether we have finished determinized the whole utterance
  // (including re-determinize)
  bool determinization_finalized_;
  // keep final_arc for appending later
  std::vector<std::pair<StateId, size_t>> final_arc_list_;
  std::vector<std::pair<StateId, size_t>> final_arc_list_prev_;
  // alpha of each state in lat_
  std::vector<BaseFloat> forward_costs_;
  // we allocate a unique id for each source-state of the last arc of a series of
  // initial arcs in GetInitialRawLattice
  int32 state_last_initial_offset_;
  // We define a state in the appended lattice as a 'redeterminized-state' (meaning:
  // one that will be redeterminized), if it is: a pre-final state, or there
  // exists an arc from a redeterminized state to this state. We keep reapplying
  // this rule until there are no more redeterminized states. The final state
  // is not included. These redeterminized states will be stored in this map
  // which is a map from the state in the appended compact lattice to the
  // state_copy in the newly-created raw lattice.
  unordered_map<StateId, StateId> redeterminized_states_;
  // It is a map used in GetRedeterminizedStates (see the description there)
  // A map from the original pre-final state to the pre-final states (i.e. the
  // original pre-final state or an extra state generated by
  // GetRedeterminizedStates) used for generating raw lattices of the next chunk.
  unordered_map<StateId, StateId> processed_prefinal_states_;

  // The compact lattice we obtain. It should be reseted before processing a
  // new utterance
  CompactLattice lat_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeIncrementalDeterminizer);
};

} // end namespace kaldi.

#endif
