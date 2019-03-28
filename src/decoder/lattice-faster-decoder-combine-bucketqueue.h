// decoder/lattice-faster-decoder-combine.h

// Copyright 2009-2013  Microsoft Corporation;  Mirko Hannemann;
//           2013-2019  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen
//                2018  Zhehuai Chen
//                2019  Hang Lyu

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

#ifndef KALDI_DECODER_LATTICE_FASTER_DECODER_COMBINE_H_
#define KALDI_DECODER_LATTICE_FASTER_DECODER_COMBINE_H_


#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "fstext/fstext-lib.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"
#include "decoder/grammar-fst.h"
#include "decoder/lattice-faster-decoder.h"

namespace kaldi {

struct LatticeFasterDecoderCombineConfig {
  BaseFloat beam;
  int32 max_active;
  int32 min_active;
  BaseFloat lattice_beam;
  int32 prune_interval;
  bool determinize_lattice; // not inspected by this class... used in
                            // command-line program.
  BaseFloat beam_delta; // has nothing to do with beam_ratio
  BaseFloat hash_ratio;
  BaseFloat cost_scale;
  BaseFloat prune_scale;   // Note: we don't make this configurable on the command line,
                           // it's not a very important parameter.  It affects the
                           // algorithm that prunes the tokens as we go.
  // Most of the options inside det_opts are not actually queried by the
  // LatticeFasterDecoder class itself, but by the code that calls it, for
  // example in the function DecodeUtteranceLatticeFaster.
  fst::DeterminizeLatticePhonePrunedOptions det_opts;

  LatticeFasterDecoderCombineConfig(): beam(16.0),
                                       max_active(std::numeric_limits<int32>::max()),
                                       min_active(200),
                                       lattice_beam(10.0),
                                       prune_interval(25),
                                       determinize_lattice(true),
                                       beam_delta(0.5),
                                       hash_ratio(2.0),
                                       cost_scale(1.0),
                                       prune_scale(0.1) { }
  void Register(OptionsItf *opts) {
    det_opts.Register(opts);
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("max-active", &max_active, "Decoder max active states.  Larger->slower; "
                   "more accurate");
    opts->Register("min-active", &min_active, "Decoder minimum #active states.");
    opts->Register("lattice-beam", &lattice_beam, "Lattice generation beam.  Larger->slower, "
                   "and deeper lattices");
    opts->Register("prune-interval", &prune_interval, "Interval (in frames) at "
                   "which to prune tokens");
    opts->Register("determinize-lattice", &determinize_lattice, "If true, "
                   "determinize the lattice (lattice-determinization, keeping only "
                   "best pdf-sequence for each word-sequence).");
    opts->Register("beam-delta", &beam_delta, "Increment used in decoding-- this "
                   "parameter is obscure and relates to a speedup in the way the "
                   "max-active constraint is applied.  Larger is more accurate.");
    opts->Register("hash-ratio", &hash_ratio, "Setting used in decoder to "
                   "control hash behavior");
    opts->Register("cost-scale", &cost_scale, "A scale that we multiply the "
                   "token costs by before intergerizing; a larger value means "
                   "more buckets and precise.");

  }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && max_active > 1 && lattice_beam > 0.0
                 && min_active <= max_active
                 && prune_interval > 0 && beam_delta > 0.0 && hash_ratio >= 1.0
                 && prune_scale > 0.0 && prune_scale < 1.0);
  }
};


namespace decodercombine {
// We will template the decoder on the token type as well as the FST type; this
// is a mechanism so that we can use the same underlying decoder code for
// versions of the decoder that support quickly getting the best path
// (LatticeFasterOnlineDecoder, see lattice-faster-online-decoder.h) and also
// those that do not (LatticeFasterDecoder).


// ForwardLinks are the links from a token to a token on the next frame.
// or sometimes on the current frame (for input-epsilon links).
template <typename Token>
struct ForwardLink {
  using Label = fst::StdArc::Label;

  Token *next_tok;  // the next token [or NULL if represents final-state]
  Label ilabel;  // ilabel on arc
  Label olabel;  // olabel on arc
  BaseFloat graph_cost;  // graph cost of traversing arc (contains LM, etc.)
  BaseFloat acoustic_cost;  // acoustic cost (pre-scaled) of traversing arc
  ForwardLink *next;  // next in singly-linked list of forward arcs (arcs
                      // in the state-level lattice) from a token.
  inline ForwardLink(Token *next_tok, Label ilabel, Label olabel,
                     BaseFloat graph_cost, BaseFloat acoustic_cost,
                     ForwardLink *next):
      next_tok(next_tok), ilabel(ilabel), olabel(olabel),
      graph_cost(graph_cost), acoustic_cost(acoustic_cost),
      next(next) { }
};

template <typename Fst>
struct StdToken {
  using ForwardLinkT = ForwardLink<StdToken>;
  using Token = StdToken;
  using StateId = typename Fst::Arc::StateId;

  // Standard token type for LatticeFasterDecoder.  Each active HCLG
  // (decoding-graph) state on each frame has one token.

  // tot_cost is the total (LM + acoustic) cost from the beginning of the
  // utterance up to this point.  (but see cost_offset_, which is subtracted
  // to keep it in a good numerical range).
  BaseFloat tot_cost;

  // exta_cost is >= 0.  After calling PruneForwardLinks, this equals
  // the minimum difference between the cost of the best path, and the cost of
  // this is on, and the cost of the absolute best path, under the assumption
  // that any of the currently active states at the decoding front may
  // eventually succeed (e.g. if you were to take the currently active states
  // one by one and compute this difference, and then take the minimum).
  BaseFloat extra_cost;

  // Record the state id of the token
  StateId state_id;

  // 'links' is the head of singly-linked list of ForwardLinks, which is what we
  // use for lattice generation.
  ForwardLinkT *links;

  //'next' is the next in the singly-linked list of tokens for this frame.
  Token *next;

  // identitfy the token is in current queue or not to prevent duplication in
  // function ProcessOneFrame().
  bool in_queue;


  // This function does nothing and should be optimized out; it's needed
  // so we can share the regular LatticeFasterDecoderTpl code and the code
  // for LatticeFasterOnlineDecoder that supports fast traceback.
  inline void SetBackpointer (Token *backpointer) { }

  // This constructor just ignores the 'backpointer' argument.  That argument is
  // needed so that we can use the same decoder code for LatticeFasterDecoderTpl
  // and LatticeFasterOnlineDecoderTpl (which needs backpointers to support a
  // fast way to obtain the best path).
  inline StdToken(BaseFloat tot_cost, BaseFloat extra_cost, StateId state_id,
                  ForwardLinkT *links, Token *next, Token *backpointer):
      tot_cost(tot_cost), extra_cost(extra_cost), state_id(state_id), 
      links(links), next(next), in_queue(false) { }
};

template <typename Fst>
struct BackpointerToken {
  using ForwardLinkT = ForwardLink<BackpointerToken>;
  using Token = BackpointerToken;
  using StateId = typename Fst::Arc::StateId;

  // BackpointerToken is like Token but also
  // Standard token type for LatticeFasterDecoder.  Each active HCLG
  // (decoding-graph) state on each frame has one token.

  // tot_cost is the total (LM + acoustic) cost from the beginning of the
  // utterance up to this point.  (but see cost_offset_, which is subtracted
  // to keep it in a good numerical range).
  BaseFloat tot_cost;

  // exta_cost is >= 0.  After calling PruneForwardLinks, this equals
  // the minimum difference between the cost of the best path, and the cost of
  // this is on, and the cost of the absolute best path, under the assumption
  // that any of the currently active states at the decoding front may
  // eventually succeed (e.g. if you were to take the currently active states
  // one by one and compute this difference, and then take the minimum).
  BaseFloat extra_cost;

  // Record the state id of the token
  StateId state_id;

  // 'links' is the head of singly-linked list of ForwardLinks, which is what we
  // use for lattice generation.
  ForwardLinkT *links;

  //'next' is the next in the singly-linked list of tokens for this frame.
  BackpointerToken *next;

  // Best preceding BackpointerToken (could be a on this frame, connected to
  // this via an epsilon transition, or on a previous frame).  This is only
  // required for an efficient GetBestPath function in
  // LatticeFasterOnlineDecoderTpl; it plays no part in the lattice generation
  // (the "links" list is what stores the forward links, for that).
  Token *backpointer;

  // identitfy the token is in current queue or not to prevent duplication in
  // function ProcessOneFrame().
  bool in_queue;

  inline void SetBackpointer (Token *backpointer) {
    this->backpointer = backpointer;
  }

  inline BackpointerToken(BaseFloat tot_cost, BaseFloat extra_cost,
                          StateId state_id, ForwardLinkT *links,
                          Token *next, Token *backpointer):
      tot_cost(tot_cost), extra_cost(extra_cost), state_id(state_id), 
      links(links), next(next), backpointer(backpointer),
      in_queue(false) { }
};

}  // namespace decoder


template<typename Token>
class BucketQueue {
 public:
  // Constructor. 'cost_scale' is a scale that we multiply the token costs by
  // before intergerizing; a larger value means more buckets.
  // 'bucket_offset_' is initialized to "15 * cost_scale_". It is an empirical
  // value in case we trigger the re-allocation in normal case, since we do in
  // fact normalize costs to be not far from zero on each frame. 
  BucketQueue(BaseFloat cost_scale = 1.0);

  // Adds Token to the queue; sets the field tok->in_queue to true (it is not
  // an error if it was already true).
  // If a Token was already in the queue but its cost improves, you should
  // just Push it again. It will be added to (possibly) a different bucket, but
  // the old entry will remain. We use "tok->in_queue" to decide
  // an entry is nonexistent or not. When pop a Token off, the field
  // 'tok->in_queue' is set to false. So the old entry in the queue will be
  // considered as nonexistent when we try to pop it.
  void Push(Token *tok);

  // Removes and returns the next Token 'tok' in the queue, or NULL if there
  // were no Tokens left. Sets tok->in_queue to false for the returned Token.
  Token* Pop();

  // Clears all the individual buckets. Sets 'first_nonempty_bucket_index_' to
  // the end of buckets_.
  void Clear();

 private:
  // Configuration value that is multiplied by tokens' costs before integerizing
  // them to determine the bucket index
  BaseFloat cost_scale_;

  // buckets_ is a list of Tokens 'tok' for each bucket.
  // If tok->in_queue is false, then the item is considered as not
  // existing (this is to avoid having to explicitly remove Tokens when their
  // costs change). The index into buckets_ is determined as follows:
  // bucket_index = std::floor(tok->cost * cost_scale_);
  // vec_index = bucket_index - bucket_storage_begin_;
  // then access buckets_[vec_index].
  std::vector<std::vector<Token*> > buckets_;

  // An offset that determines how we index into the buckets_ vector;
  // In the constructor this will be initialized to something like
  // "15 * cost_scale_" which will make it unlikely that we have to change this
  // value in future if we get a much better Token (this is expensive because it
  // involves reallocating 'buckets_').
  int32 bucket_offset_;

  // first_nonempty_bucket_index_ is an integer in the range [0,
  // buckets_.size() - 1] which is not larger than the index of the first
  // nonempty element of buckets_.
  int32 first_nonempty_bucket_index_;

  // Synchronizes with first_nonempty_bucket_index_.
  std::vector<Token*> *first_nonempty_bucket_;
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
template <typename FST, typename Token = decodercombine::StdToken<FST> >
class LatticeFasterDecoderCombineTpl {
 public:
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using ForwardLinkT = decodercombine::ForwardLink<Token>;

  using StateIdToTokenMap = typename std::unordered_map<StateId, Token*>;
  //using StateIdToTokenMap = typename std::unordered_map<StateId, Token*,
  //      std::hash<StateId>, std::equal_to<StateId>,
  //      fst::PoolAllocator<std::pair<const StateId, Token*> > >;
  using IterType = typename StateIdToTokenMap::const_iterator;

  using BucketQueue = typename kaldi::BucketQueue<Token>;

  // Instantiate this class once for each thing you have to decode.
  // This version of the constructor does not take ownership of
  // 'fst'.
  LatticeFasterDecoderCombineTpl(const FST &fst,
                                 const LatticeFasterDecoderCombineConfig &config);

  // This version of the constructor takes ownership of the fst, and will delete
  // it when this object is destroyed.
  LatticeFasterDecoderCombineTpl(const LatticeFasterDecoderCombineConfig &config,
                                 FST *fst);

  void SetOptions(const LatticeFasterDecoderCombineConfig &config) {
    config_ = config;
  }

  const LatticeFasterDecoderCombineConfig &GetOptions() const {
    return config_;
  }

  ~LatticeFasterDecoderCombineTpl();

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
  bool GetBestPath(Lattice *ofst,
                   bool use_final_probs = true);

  /// Outputs an FST corresponding to the raw, state-level
  /// tracebacks.  Returns true if result is nonempty.
  /// If "use_final_probs" is true AND we reached the final-state
  /// of the graph then it will include those as final-probs, else
  /// it will treat all final-probs as one.
  /// The raw lattice will be topologically sorted.
  /// The function can be called during decoding, it will process non-emitting
  /// arcs from "cur_toks_" map to get tokens from both non-emitting and 
  /// emitting arcs for getting raw lattice. Then recover it to ensure the
  /// consistency of ProcessForFrame().
  ///
  /// See also GetRawLatticePruned in lattice-faster-online-decoder.h,
  /// which also supports a pruning beam, in case for some reason
  /// you want it pruned tighter than the regular lattice beam.
  /// We could put that here in future needed.
  bool GetRawLattice(Lattice *ofst, bool use_final_probs = true);



  /// [Deprecated, users should now use GetRawLattice and determinize it
  /// themselves, e.g. using DeterminizeLatticePhonePrunedWrapper].
  /// Outputs an FST corresponding to the lattice-determinized
  /// lattice (one path per word sequence).   Returns true if result is nonempty.
  /// If "use_final_probs" is true AND we reached the final-state of the graph
  /// then it will include those as final-probs, else it will treat all
  /// final-probs as one.
  bool GetLattice(CompactLattice *ofst,
                  bool use_final_probs = true);

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
  // whenever we call ProcessForFrame().
  inline int32 NumFramesDecoded() const { return active_toks_.size() - 1; }

 protected:
  // we make things protected instead of private, as code in
  // LatticeFasterOnlineDecoderTpl, which inherits from this, also uses the
  // internals.

  // Deletes the elements of the singly linked list tok->links.
  inline static void DeleteForwardLinks(Token *tok);

  // head of per-frame list of Tokens (list is in topological order),
  // and something saying whether we ever pruned it using PruneForwardLinks.
  struct TokenList {
    Token *toks;
    bool must_prune_forward_links;
    bool must_prune_tokens;
    TokenList(): toks(NULL), must_prune_forward_links(true),
                 must_prune_tokens(true) { }
  };

  // FindOrAddToken either locates a token in hash map "token_map", or if necessary
  // inserts a new, empty token (i.e. with no forward links) for the current
  // frame.  [note: it's inserted if necessary into hash map and also into the
  // singly linked list of tokens active on this frame (whose head is at
  // active_toks_[frame]).  The frame_plus_one argument is the acoustic frame
  // index plus one, which is used to index into the active_toks_ array.
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true if the
  // token was newly created or the cost changed.
  // If Token == StdToken, the 'backpointer' argument has no purpose (and will
  // hopefully be optimized out).
  inline Token *FindOrAddToken(StateId state, int32 token_list_index,
                               BaseFloat tot_cost, Token *backpointer,
                               StateIdToTokenMap *token_map,
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

  /// Processes non-emitting (epsilon) arcs and emitting arcs for one frame
  /// together. It takes the emittion tokens in "prev_toks_" from last frame.
  /// Generates non-emitting tokens for previous frame and emitting tokens for
  /// next frame.
  /// Notice: The emitting tokens for the current frame means the token take
  /// acoustic scores of the current frame. (i.e. the destnations of emitting
  /// arcs.)
  void ProcessForFrame(DecodableInterface *decodable);

  /// Processes nonemitting (epsilon) arcs for one frame.
  /// This function is called from FinalizeDecoding(), and also from
  /// GetRawLattice() if GetRawLattice() is called before FinalizeDecoding() is
  /// called.
  void ProcessNonemitting();

  /// The "prev_toks_" and "cur_toks_" actually allow us to maintain current
  /// and next frames. They are indexed by StateId. It is indexed by frame-index
  /// plus one, where the frame-index is zero-based, as used in decodable object.
  /// That is, the emitting probs of frame t are accounted for in tokens at
  /// toks_[t+1].  The zeroth frame is for nonemitting transition at the start of
  /// the graph.
  StateIdToTokenMap prev_toks_;
  StateIdToTokenMap cur_toks_;

  /// Gets the weight cutoff.
  /// Notice: In traiditional version, the histogram prunning method is applied
  /// on a complete token list on one frame. But, in this version, it is used
  /// on a token list which only contains the emittion part. So the max_active
  /// and min_active values might be narrowed.
  std::vector<TokenList> active_toks_; // Lists of tokens, indexed by
  // frame (members of TokenList are toks, must_prune_forward_links,
  // must_prune_tokens).

  // fst_ is a pointer to the FST we are decoding from.
  const FST *fst_;
  // delete_fst_ is true if the pointer fst_ needs to be deleted when this
  // object is destroyed.
  bool delete_fst_;

  std::vector<BaseFloat> cost_offsets_; // This contains, for each
  // frame, an offset that was added to the acoustic log-likelihoods on that
  // frame in order to keep everything in a nice dynamic range i.e.  close to
  // zero, to reduce roundoff errors.
  // Notice: It will only be added to emitting arcs (i.e. cost_offsets_[t] is
  // added to arcs from "frame t" to "frame t+1").
  LatticeFasterDecoderCombineConfig config_;
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

  BaseFloat adaptive_beam_;  // will be set to beam_ when we start
  BucketQueue cur_queue_;  // temp variable used in 
                           // ProcessForFrame/ProcessNonemitting

  // This function takes a singly linked list of tokens for a single frame, and
  // outputs a list of them in topological order (it will crash if no such order
  // can be found, which will typically be due to decoding graphs with epsilon
  // cycles, which are not allowed).  Note: the output list may contain NULLs,
  // which the caller should pass over; it just happens to be more efficient for
  // the algorithm to output a list that contains NULLs.
  static void TopSortTokens(Token *tok_list,
                            std::vector<Token*> *topsorted_list);

  void ClearActiveTokens();

  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeFasterDecoderCombineTpl);
};

typedef LatticeFasterDecoderCombineTpl<fst::StdFst,
        decodercombine::StdToken<fst::StdFst> > LatticeFasterDecoderCombine;



} // end namespace kaldi.

#endif
