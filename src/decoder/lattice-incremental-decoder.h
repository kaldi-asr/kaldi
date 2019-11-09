// decoder/lattice-incremental-decoder.h

// Copyright      2019  Zhehuai Chen, Hainan Xu, Daniel Povey

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
/**
   The normal decoder, lattice-faster-decoder.h, sometimes has an issue when
   doing real-time applications with long utterances, that each time you get the
   lattice the lattice determinization can take a considerable amount of time;
   this introduces latency.  This version of the decoder spreads the work of
   lattice determinization out throughout the decoding process.

   NOTE:

   Please see https://www.danielpovey.com/files/ *TBD* .pdf for a technical
   explanation of what is going on here.

   GLOSSARY OF TERMS:
      chunk: We do the determinization on chunks of frames; these
          may coincide with the chunks on which the user calls
          AdvanceDecoding().  The basic idea is to extract chunks
          of the raw lattice and determinize them individually, but
          it gets much more complicated than that.  The chunks
          should normally be at least as long as a word (let's say,
          at least 20 frames), or the overhead of this algorithm
          might become excessive and affect RTF.

      raw lattice chunk: A chunk of raw (i.e. undeterminized) lattice
          that we will determinize.  In the paper this corresponds
          to the FST B that is described in Section 5.2.

      token_label, state_label:  In the paper these are both
          referred to as `state labels` (these are special, large integer
          id's that refer to states in the undeterminized lattice
          and in the the determinized lattice);
          but we use two separate terms here, for more clarity,
          when referring to the undeterminized vs. determinized lattice.

           token_label conceptually refers to states in the
           raw lattice, but we don't materialize the entire
           raw lattice as a physical FST and and these tokens
           are actually tokens (template type Token) held by
           the decoder

           state_label when used in this code refers specifically
           to labels that identify states in the determinized
           lattice (i.e. state indexes in lat_).

       redeterminized-non-splice-state, aka ns_redet:
         A redeterminized state which is not also a splice state;
         refer to the paper for explanation.  In the already-determinized
         part this means a redeterminized state which is not final.

       canonical appended lattice:  This is the appended compact lattice
         that we conceptually have (i.e. what we described in the paper).
         The difference from the "actual appended lattice" is that the
         actual appended lattice has all its final-arcs replaced with
         final-probs (we keep the real final-arcs "on the side" in a
         separate data structure).

       final-arc:  An arc in the canonical appended CompactLattice which
         goes to a final-state.  These arcs will have `state-labels` as
         their labels.

 */
struct LatticeIncrementalDecoderConfig {
  // All the configuration values until det_opts are the same as in
  // LatticeFasterDecoder.  For clarity we repeat them rather than inheriting.
  BaseFloat beam;
  int32 max_active;
  int32 min_active;
  BaseFloat lattice_beam;
  int32 prune_interval;
  BaseFloat beam_delta; // has nothing to do with beam_ratio
  BaseFloat hash_ratio;
  BaseFloat prune_scale; // Note: we don't make this configurable on the command line,
                         // it's not a very important parameter.  It affects the
                         // algorithm that prunes the tokens as we go.
  // Most of the options inside det_opts are not actually queried by the
  // LatticeIncrementalDecoder class itself, but by the code that calls it, for
  // example in the function DecodeUtteranceLatticeIncremental.
  fst::DeterminizeLatticePhonePrunedOptions det_opts;

  // The configuration values from this point on are specific to the
  // incremental determinization.
  // TODO: explain the following.
  int32 determinize_delay;
  int32 determinize_period;
  int32 determinize_max_active;
  int32 redeterminize_max_frames;
  bool final_prune_after_determinize;
  int32 max_word_id; // for GetLattice


  LatticeIncrementalDecoderConfig()
      : beam(16.0),
        max_active(std::numeric_limits<int32>::max()),
        min_active(200),
        lattice_beam(10.0),
        prune_interval(25),
        beam_delta(0.5),
        hash_ratio(2.0),
        prune_scale(0.1),
        determinize_delay(25),
        determinize_period(20),
        determinize_max_active(std::numeric_limits<int32>::max()),
        redeterminize_max_frames(std::numeric_limits<int32>::max()),
        final_prune_after_determinize(true),
        max_word_id(1e8) {}
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
    // TODO: check the following.
    opts->Register("determinize-delay", &determinize_delay,
                   "Delay (in frames) at which to incrementally determinize "
                   "lattices. A larger delay reduces the computational "
                   "overhead of incremental deteriminization while increasing"
                   "the length of the last chunk which may increase latency.");
    opts->Register("determinize-period", &determinize_period,
                   "The size (in frames) of chunk to do incrementally "
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
    opts->Register("redeterminize-max-frames", &redeterminize_max_frames,
                   "To impose a limit on how far back in time we will "
                   "redeterminize states.  This is mainly intended to avoid "
                   "pathological cases. Smaller value leads to less "
                   "deterministic but less likely to blow up the processing"
                   "time in bad cases. You could set it infinite to get a fully "
                   "determinized lattice.");
    opts->Register("final-prune-after-determinize", &final_prune_after_determinize,
                   "prune lattice after determinization ");
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
                 determinize_period >= 0 && redeterminize_max_frames >= 0 &&
                 beam_delta > 0.0 && hash_ratio >= 1.0 && prune_scale > 0.0 &&
                 prune_scale < 1.0);
  }
};



/**
   This class is used inside LatticeIncrementalDecoderTpl; it handles
   some of the details of incremental determinization.
   https://www.danielpovey.com/files/ *TBD*.pdf for the paper.

*/
class LatticeIncrementalDeterminizer2 {
 public:
  using Label = typename LatticeArc::Label;  /* Actualy the same labels appear
                                                in both lattice and compact
                                                lattice, so we don't use the
                                                specific type all the time but
                                                just say 'Label' */

  LatticeIncrementalDeterminizer2(const LatticeIncrementalDecoderConfig &config,
                                  const TransitionModel &trans_model);

  // Resets the lattice determinization data for new utterance
  void Init();

  // Returns the current determinized lattice.
  const CompactLattice &GetDeterminizedLattice() const { return clat_; }

  /**
     Starts the process of creating a raw lattice chunk.  (Search the glossary
     for "raw lattice chunk").  This just sets up the initial states and
     redeterminized-states in the chunk.  Relates to sec. 5.2 in the paper,
     specifically the initial-state i and the redeterminized-states.

     After calling this, the caller would add the remaining arcs and states
     to `olat` and then call AcceptChunk() with the result.

        @param [out] olat    The lattice to be (partially) created

        @param [out] token_label2state  This function outputs to here
                a map from `token-label` to the state we created for
                it in *olat.  See glossary for `token-label`.
                The keys actually correspond to the .nextstate fields
                in the arcs in final_arcs_; values are states in `olat`.
                See the last bullet point before Sec. 5.3 in the paper.
  */
  void InitializeRawLatticeChunk(
      Lattice *olat,
      unordered_map<Label, LatticeArc::StateId> *token_label2state);

  /**
     This function accepts the raw FST (state-level lattice) corresponding to a
     single chunk of the lattice, determinizes it and appends it to this->clat_.
     Unless this was the

     Note: final-probs in `raw_fst` are treated specially: they are used to
     guide the pruned determinization, but when you call GetLattice() it will be
     -- except for pruning effects-- as if all nonzero final-probs in `raw_fst`
     were: One() if final_costs == NULL; else the value present in `final_costs`.

       @param [in] raw_fst  (Consumed destructively).  The input
                  raw (state-level) lattice.  Would correspond to the
                  FST A in the paper if first_frame == 0, and B
                  otherwise.
       @param [in] final_costs  Final-costs that the user wants to
                  be included in clat_.  These replace the values present
                  in the Final() probs in raw_fst whenever there was
                  a nonzero final-prob in raw_fst.  (States in raw_fst
                  that had a final-prob will still be non-final).

     @return returns false if determinization finished earlier than the beam,
         true otherwise.
  */
  bool AcceptRawLatticeChunk(Lattice *raw_fst,
                             const std::unordered_map<LatticeArc::StateId, BaseFloat> *final_costs = NULL);


  const CompactLattice &GetLattice() { return clat_; }

 private:

  // kTokenLabelOffset is where we start allocating labels corresponding to Tokens
  // (these correspond with raw lattice states);
  // kStateLabelOffset is what we add to state-ids in clat_ to produce labels
  // to identify them in the raw lattice chunk
  enum  { kStateLabelOffset = (int)1e8,  kTokenLabelOffset = (int)2e8, kMaxTokenLabel = (int)3e8 };

  Label next_state_label_;

  // clat_ is the appended lattice (containing all chunks processed so
  // far), except its `final-arcs` (i.e. arcs which in the canonical
  // lattice would go to final-states) are not present (they are stored
  // separately in final_arcs_) and states which in the canonical lattice
  // should have final-arcs leaving them will instead have a final-prob.
  CompactLattice clat_;

  // The elements of this set are the redeterminized-states which are not final in
  // the canonical appended lattice.  This means the set of .first elements in
  // final_arcs, plus whatever states in clat_ are reachable from such states.
  // (The final redeterminized states/splice-states are never actually
  // materialized.)
  std::unordered_set<CompactLatticeArc::StateId> non_final_redet_states_;


  // final_arcs_ contains arcs which would appear in the canonical appended
  // lattice but for implementation reasons are not physically present in clat_.
  // These are arcs to final states in the canonical appended lattice.  The
  // .first elements are the source states in clat_ (these will all be elements
  // of non_final_redet_states_); the .nextstate elements of the arcs does not
  // contain a physical state, but contain state-labels allocated by
  // AllocateNewStateLabel().
  std::vector<CompactLatticeArc> final_arcs_;


  // final_weights_ contain the final-probs of states that are final in the
  // canonical compact lattice.  Physically it maps from the state-labels which
  // are allocated by AllocateNewStateLabel() and are stored in the .nextstate
  // in final_arcs_, to the weight that would be on that final-state in the
  // canonical compact lattice.
  std::unordered_map<Label, CompactLatticeWeight> final_weights_;

  // forward_costs_, indexed by the state-id in clat_, stores the alpha
  // (forward) costs, i.e. the minimum cost from the start state to each state
  // in clat_.  This is relevant for pruned determinization.  The BaseFloat can
  // be thought of as the sum of a Value1() + Value2() in a LatticeWeight.
  std::vector<BaseFloat> forward_costs_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeIncrementalDeterminizer2);
};


/** This is an extention to the "normal" lattice-generating decoder.
   See \ref lattices_generation \ref decoders_faster and \ref decoders_simple
    for more information.

   The main difference is the incremental determinization which will be
   discussed in the function GetLattice().  This means that the work of determinizatin
   isn't done all at once at the end of the file, but incrementally while decoding.
   See the comment at the top of this file for more explanation.

   The decoder is templated on the FST type and the token type.  The token type
   will normally be StdToken, but also may be BackpointerToken which is to support
   quick lookup of the current best path (see lattice-faster-online-decoder.h)

   The FST you invoke this decoder with is expected to be of type
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

  void SetOptions(const LatticeIncrementalDecoderConfig &config) { config_ = config; }

  const LatticeIncrementalDecoderConfig &GetOptions() const { return config_; }

  ~LatticeIncrementalDecoderTpl();

  /**
     CAUTION: this function is provided only for testing and instructional
     purposes.  In a scenario where you have the entire file and just want
     to decode it, there is no point using this decoder.

     An example of how to do decoding together with incremental
     determinization. It decodes until there are no more frames left in the
     "decodable" object.

     In this example, config_.determinize_delay, config_.determinize_period
     and config_.determinize_max_active are used to determine the time to
     call GetLattice().

     Users will probably want to use appropriate combinations of
     AdvanceDecoding() and GetLattice() to build their application; this just
     gives you some idea how.

     The function returns true if any kind of traceback is available (not
     necessarily from a final state).
  */
  bool Decode(DecodableInterface *decodable);

  /// says whether a final-state was active on the last frame.  If it was not,
  /// the lattice (or traceback) will end with states that are not final-states.
  bool ReachedFinal() const {
    return FinalRelativeCost() != std::numeric_limits<BaseFloat>::infinity();
  }

  /**
     Outputs an FST corresponding to the single best path through the lattice.
     If "use_final_probs" is true AND we reached the
     final-state of the graph then it will include those as final-probs, else
     it will treat all final-probs as one.

     Note: this gets the traceback from the compact lattice, which will not
     include the most recently decoded frames if determinize_delay > 0 and
     FinalizeDecoding() has not been called.  If you'll be wanting to call
     GetBestPath() a lot and need it to be up to date, you may prefer to
     use LatticeIncrementalOnlineDecoder.
  */
  void GetBestPath(Lattice *ofst, bool use_final_probs = true);

  /**
     This GetLattice() function is the main way you will interact with the
     incremental determinization that this class provides.  Note that the
     interface is slightly different from that of other decoders.  For example,
     if olat is NULL it will do the work of incremental determinization without
     actually giving you the lattice (which can save it some time).

     Note: calling it on every frame doesn't make sense as it would
     still have to do a fair amount of work; calling it every, say,
     10 to 40 frames would make sense though.

      @param [in] use_final_probs  True if you want the final-probs
                     of HCLG to be included in the output lattice.
                     (However, if no state was final on frame
                     `num_frames_to_include` they won't be included regardless
                     of use_final_probs; if this equals NumFramesDecoded() you
                     can test this with ReachedFinal()).  Caution:
                     it is an error to call this function with
                     the same num_frames_to_include and different values
                     of `use_final_probs`.  (This is not a fundamental
                     limitation but just the way we coded it.)

      @param [in] num_frames_to_include  The number of frames that you want
                     to be included in the lattice.  Must be >0 and
                     <= NumFramesDecoded().  If you are calling this just to
                     keep the incremental lattice determinization up to date and
                     don't really need the lattice now or don't need it to be up
                     to date, you will probably want to make
                     num_frames_to_include at least 5 or 10 frames less than
                     NumFramessDecoded(); search for determinize-delay in the
                     paper and for determinize_delay in the configuration class
                     and the code.  You may not call this with a
                     num_frames_to_include that is smaller than the largest
                     value previously provided.  Calling it with an
                     only-slightly-larger version than the last time (e.g. just
                     a few frames larger) is probably not a good use of
                     computational resources.

      @return clat   The CompactLattice representing what has been decoded
                     up until `num_frames_to_include` (e.g., LatticeStateTimes()
                     on this lattice would return `num_frames_to_include`).

  */
  const CompactLattice &GetLattice(bool use_final_probs,
                                   int32 num_frames_to_include);



  /**
     InitDecoding initializes the decoding, and should only be used if you
     intend to call AdvanceDecoding().  If you call Decode(), you don't need to
     call this.  You can also call InitDecoding if you have already decoded an
     utterance and want to start with a new utterance.
  */
  void InitDecoding();

  /**
     This will decode until there are no more frames ready in the decodable
     object.  You can keep calling it each time more frames become available
     (this is the normal pattern in a real-time/online decoding scenario).
     If max_num_frames is specified, it specifies the maximum number of frames
     the function will decode before returning.
  */
  void AdvanceDecoding(DecodableInterface *decodable, int32 max_num_frames = -1);


  /**
     This function may be optionally called after AdvanceDecoding(), when you
     do not plan to decode any further.  It does an extra pruning step that
     will help to prune the lattices output by GetLattice more accurately,
     particularly toward the end of the utterance.
     It does this by using the final-probs in pruning (if any
     final-state survived); it also does a final pruning step that visits all
     states (the pruning that is done during decoding may fail to prune states
     that are within kPruningScale = 0.1 outside of the beam).  If you call
     this, you cannot call AdvanceDecoding again (it will fail), and you
     cannot call GetLattice() and related functions with use_final_probs =
     false.
  */
  void FinalizeDecoding();

  /** FinalRelativeCost() serves the same purpose as ReachedFinal(), but gives
      more information.  It returns the difference between the best (final-cost
      plus cost) of any token on the final frame, and the best cost of any token
      on the final frame.  If it is infinity it means no final-states were
      present on the final frame.  It will usually be nonnegative.  If it not
      too positive (e.g. < 5 is my first guess, but this is not tested) you can
      take it as a good indication that we reached the final-state with
      reasonable likelihood. */
  BaseFloat FinalRelativeCost() const;

  /** Returns the number of frames decoded so far. */
  inline int32 NumFramesDecoded() const { return active_toks_.size() - 1; }

 protected:
  /* Some protected things are needed in LatticeIncrementalOnlineDecoderTpl. */

  /** NOTE: for parts the internal implementation that are shared with LatticeFasterDecoer,
      we have removed the comments.*/
  inline static void DeleteForwardLinks(Token *tok);
  struct TokenList {
    Token *toks;
    bool must_prune_forward_links;
    bool must_prune_tokens;
    TokenList()
        : toks(NULL), must_prune_forward_links(true), must_prune_tokens(true) {}
  };
  using Elem = typename HashList<StateId, Token *>::Elem;
  void PossiblyResizeHash(size_t num_toks);
  inline Token *FindOrAddToken(StateId state, int32 frame_plus_one,
                               BaseFloat tot_cost, Token *backpointer, bool *changed);
  void PruneForwardLinks(int32 frame_plus_one, bool *extra_costs_changed,
                         bool *links_pruned, BaseFloat delta);
  void ComputeFinalCosts(unordered_map<Token *, BaseFloat> *final_costs,
                         BaseFloat *final_relative_cost,
                         BaseFloat *final_best_cost) const;
  void PruneForwardLinksFinal();
  void PruneTokensForFrame(int32 frame_plus_one);
  void PruneActiveTokens(BaseFloat delta);
  BaseFloat GetCutoff(Elem *list_head, size_t *tok_count, BaseFloat *adaptive_beam,
                      Elem **best_elem);
  BaseFloat ProcessEmitting(DecodableInterface *decodable);
  void ProcessNonemitting(BaseFloat cost_cutoff);

  HashList<StateId, Token *> toks_;
  std::vector<TokenList> active_toks_;  // indexed by frame.
  std::vector<StateId> queue_;       // temp variable used in ProcessNonemitting,
  std::vector<BaseFloat> tmp_array_; // used in GetCutoff.
  const FST *fst_;
  bool delete_fst_;
  std::vector<BaseFloat> cost_offsets_;
  int32 num_toks_;
  bool warned_;
  bool decoding_finalized_;
  unordered_map<Token *, BaseFloat> final_costs_;
  BaseFloat final_relative_cost_;
  BaseFloat final_best_cost_;

  /*** Variables below this point relate to the incremental
       determinization. ***/
  LatticeIncrementalDecoderConfig config_;
  /** Much of the the incremental determinization algorithm is encapsulated in
      the determinize_ object.  */
  LatticeIncrementalDeterminizer2 determinizer_;

  /** num_frames_in_lattice_ is the highest `num_frames_to_include_` argument
      for any prior call to GetLattice(). */
  int32 num_frames_in_lattice_;
  // a map from Token to its token_label
  unordered_map<Token *, int32> token2label_map_;
  // we allocate a unique id for each Token
  int32 token_label_available_idx_;
  // We keep cost_offset for each token_label (Token) in final arcs. We need them to
  // guide determinization
  // We cancel them after determinization
  unordered_map<int32, BaseFloat> token_label2final_cost_;


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
  // Returns the number of active tokens on frame `frame`.
  int32 GetNumToksForFrame(int32 frame);

  // DeterminizeLattice() is just a wrapper for GetLattice() that uses the various
  // heuristics specified in the config class to decide when, and with what arguments,
  // to call GetLattice() in order to make sure that the incremental determinization
  // is kept up to date.  It is mainly of use for documentation (it is called inside
  // Decode() which is not recommended for users to call in most scenarios).
  // We may at some point decide to make this public.
  void DeterminizeLattice();

  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeIncrementalDecoderTpl);
};

typedef LatticeIncrementalDecoderTpl<fst::StdFst, decoder::StdToken>
    LatticeIncrementalDecoder;


} // end namespace kaldi.

#endif
