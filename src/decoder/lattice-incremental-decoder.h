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

       redeterminized-non-splice-state, aka redetnss:
         A redeterminized state which is not also a splice state;
         refer to the paper for explanation.

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

template <typename FST>
class LatticeIncrementalDeterminizer;

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

      @param [in] use_final_probs  If true *and* at least one final-state in HCLG
                     was active on the most recently decoded frame, include the
                     final-probs from the decoding FST (HCLG) in the lattice.
                     Otherwise treat all final-costs of states active on the
                     most recent frame as zero (i.e. use Weight::One()).  You
                     can tell whether a final-prob was active on the most
                     recent frame by calling ReachedFinal().
                     Setting use_final_probs will not affect the lattices
                     output by subsequent calls to this function.  (TODO:
                     verify this).

      @param [in] num_frames_to_include  The number of frames that you want
                     to be included in the lattice.  Must be >0 and
                     <= NumFramesDecoded().  If you are calling this
                     just to keep the incremental lattice determinization up to
                     date and don't really need the lattice (olat == NULL), you
                     will probably want to give it some delay (at least 5 or 10
                     frames); search for determinize-delay in the paper
                     and for determinize_delay in the configuration class and the
                     code.  You may not call this with a num_frames_to_include
                     that is smaller than the largest value previously
                     provided.

      @param [out] olat  The CompactLattice representing what has been decoded
                     up until `num_frames_to_include` (e.g., LatticeStateTimes()
                     on this lattice would return `num_frames_to_include`).
                     If NULL, the lattice won't be output, and this will save
                     the work of copying it, but the incremental determinization
                     will still be done.
  */
  void GetLattice(bool use_final_probs, int32 num_frames_to_include,
                  CompactLattice *olat = NULL);


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
  LatticeIncrementalDeterminizer<FST> determinizer_;
  /** last_get_lattice_frame_ is the highest `num_frames_to_include_` argument
      for any prior call to GetLattice(). */
  int32 last_get_lattice_frame_;
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

/**
   This class is used inside LatticeIncrementalDecoderTpl; it handles
   some of the details of incremental determinization.
   https://www.danielpovey.com/files/ *TBD*.pdf for the paper.

*/
template <typename FST>
class LatticeIncrementalDeterminizer {
 public:
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  LatticeIncrementalDeterminizer(const LatticeIncrementalDecoderConfig &config,
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
        @param [in] token_label2final_cost   For each token-label,
                contains a cost that we will need to negate and then
                introduce into newly created arcs in 'olat' that correspond
                to arcs with that token-label on in the previous
                determinized chunk.  (They won't actually have the token
                label as we remove them at this point).  This relates
                to an issue not discussed in the paper, which is
                that to get pruned determinization to work right we
                have to introduce special final-probs when determinizing
                the previous chunk (think of the previous chunk as
                the FST A in the paper).  This map allows us to cancel
                out those final-probs.
        @param [out] token_label2state  For each token-label (say, t)
                that appears in lat_ (i.e. the result of determinizing previous
                chunks), this identifies the state in `olat` that we allocate
                for that token-label.  This is a so-called `splice-state`; they
                will never have arcs leaving them within `lat_`.  When the
                calling code processes the arcs in the raw lattice, it will add
                arcs leaving these splice states.
                See the last bullet point before Sec. 5.3 in the paper.
  */
  void InitializeRawLatticeChunk(
      Lattice *olat,
      const unordered_map<int32, BaseFloat> &token_label2final_cost,
      unordered_map<int, LatticeArc::StateId> *token_label2state);

  /**
     This function accepts the raw FST (state-level lattice) corresponding
     to a single chunk of the lattice, determinizes it and appends it to
     this->clat_.

       @param [in] first_frame  The start frame-index, which equals the
                  total number of frames in all chunks previous to this one.
                  Only needed to ask "is this the first chunk", plus
                  debug info.
       @param [in] last_frame  The end frame-index, which equals the
                  total number of frames in all previous chunks plus
                  this one.  Only needed for debug.
       @param [in] raw_fst  (Consumed destructively).  The input
                  raw (state-level) lattice.  Would correspond to the
                  FST A in the paper if first_frame == 0, and B
                  otherwise.

     @return returns false if determinization finished earlier than the beam,
         true otherwise.
  */
  bool AcceptRawLatticeChunk(int32 first_frame, int32 last_frame, Lattice *raw_fst);


  /**
     Finalize incremental decoding by pruning the lattice (if
     config_.final_prune_after_determinize), otherwise just removing unreachable
     states.
  */
  void Finalize();


 private:

  /**
  // Step 3 of incremental determinization,
  // which is to append the new chunk in clat to the old one in lat_
  // If not_first_chunk == false, we do not need to append and just copy
  // clat into olat
  // Otherwise, we need to connect states of the last frame of
  // the last chunk to states of the first frame of this chunk.
  // These post-initial and pre-final states are corresponding to the same Token,
  // guaranteed by unique state labels.
     NOTE clat must be top sorted.
   */
  void AppendLatticeChunks(const CompactLattice &clat, bool first_chunk);


  /**
     In the paper, recall from Sec. 5.2 that some states in det(A) (specifically:
     redeterminized state) are also included in B.  In the paper we just assumed
     that the same state-ids were used, but in practice they are different numbers;
     in redeterminized_state_map_ we store the mapping from state-id in det(A)==clat_ to
     the state-id in B==raw_lat_chunk.  The map is re-initialized each time we
     process a new chunk.  This function maps from the the state-id in clat_
     to the state_id in `raw_lat_chunk`, adding to the map and creating a new state
     in `raw_lat_chunk` if it was not already present.

          @param [in] redet_state  State-id of a redeterminized-state in clat_
          @param [in] raw_lat_chunk  The raw lattice that we are creating;
                        this function may add a new state to it.
          @param [out] state_id   If non-NULL, the state-id in `raw_lat_chunk`
                       will be output to here
          @return    Returns true if a new state was created and added to the map
  */
  bool FindOrAddRedeterminizedState(
      CompactLattice::StateId redet_state,
      Lattice *raw_lat_chunk,
      Lattice::StateId *state_id = NULL);

  /**
     TODO
   */
  void ProcessRedeterminizedState(
      Lattice::StateId state,
      const unordered_map<int32, BaseFloat> &token_label2final_cost,
      unordered_map<int, LatticeArc::StateId> *token_label2state,
      Lattice *raw_lat_chunk);

  // This function is to preprocess the appended compact lattice before
  // generating raw lattices for the next chunk.
  // After identifying redeterminized states, for any such state that is separated by
  // more than config_.redeterminize_max_frames from the end of the current
  // appended lattice, we create an extra state for it; we add an epsilon arc
  // from that redeterminized state to the extra state; we copy any final arcs from
  // the pre-final state to its extra state and we remove those final arcs from
  // the original pre-final state.
  // We also copy arcs meet the following requirements: i) destination-state of the
  // arc is prefinal state. ii) destination-state of the arc is no further than than
  // redeterminize_max_frames from the most recent frame we are determinizing.
  // Now this extra state is the pre-final state to
  // redeterminize and the original pre-final state does not need to redeterminize
  // The epsilon would be removed later on in AppendLatticeChunks, while
  // splicing the compact lattices together
  void GetRedeterminizedStates();

  const LatticeIncrementalDecoderConfig config_;
  const TransitionModel &trans_model_; // keep it for determinization

  // Record whether we have finished determinized the whole utterance
  // (including re-determinize)
  bool determinization_finalized_;
  /**
  // A map from the prefinal state to its correponding first final arc (there could be
  // multiple final arcs). We keep final arc information for GetRedeterminizedStates()
  // later. It can also be used to identify whether a state is a prefinal state.
  */
  unordered_map<StateId, size_t> final_arc_list_;
  unordered_map<StateId, size_t> final_arc_list_prev_;
  // alpha of each state in lat_
  std::vector<BaseFloat> forward_costs_;
  // we allocate a unique id for each source-state of the last arc of a series of
  // initial arcs in InitializeRawLatticeChunk
  int32 state_last_initial_offset_;

  // We define a state in the appended lattice as a 'redeterminized-state' (meaning:
  // one that will be redeterminized), if it is: a pre-final state, or there
  // exists an arc from a redeterminized state to this state. We keep reapplying
  // this rule until there are no more redeterminized states. The final state
  // is not included. These redeterminized states will be stored in this map
  // which is a map from the state in the appended compact lattice to the
  // state_copy in the newly-created raw lattice.
  unordered_map<StateId, StateId> redeterminized_state_map_;

  /**
  // It is a map used in GetRedeterminizedStates (see the description there)
  // A map from the original pre-final state to the pre-final states (i.e. the
  // original pre-final state or an extra state generated by
  // GetRedeterminizedStates) used for generating raw lattices of the next chunk.
  */
  unordered_map<StateId, StateId> processed_prefinal_states_;

  // The compact lattice we obtain. It should be cleared before processing a new
  // utterance
  CompactLattice clat_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeIncrementalDeterminizer);
};

} // end namespace kaldi.

#endif
