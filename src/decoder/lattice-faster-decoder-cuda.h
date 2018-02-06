// decoder/lattice-faster-decoder-cuda.h

// Copyright      2018  Zhehuai Chen

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

#ifndef KALDI_DECODER_LATTICE_FASTER_DECODER_CUDA_H_
#define KALDI_DECODER_LATTICE_FASTER_DECODER_CUDA_H_


#include "util/stl-utils.h"
#include "util/hash-list.h"
#include "itf/decodable-itf.h"

#include "cuda-lattice-decoder.h"

namespace kaldi {

/** A bit more optimized version of the lattice decoder.
   See \ref lattices_generation \ref decoders_faster and \ref decoders_simple
    for more information.
 */
class LatticeFasterDecoderCuda {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  typedef CudaLatticeDecoder::Token cuToken;
  typedef CudaLatticeDecoder::TokenMergeVector cuTokenVector;
  typedef CudaLatticeDecoder::TokenState TokenState;
  typedef CudaLatticeDecoder::LatLink LatLink;

  // instantiate this class once for each thing you have to decode.
  LatticeFasterDecoderCuda(const CudaFst &fst,
                           const CudaLatticeDecoderConfig &config);
  // This version of the initializer "takes ownership" of the fst,
  // and will delete it when this object is destroyed.

  ~LatticeFasterDecoderCuda();

  void InitDecoding(); // CPU decoding init

  const CudaLatticeDecoderConfig& GetOptions() const {
    return config_;
  }
  const CudaLatticeDecoder &Decoder() const { return decoder_; }

  /// Decodes until there are no more frames left in the "decodable" object..
  /// note, this may block waiting for input if the "decodable" object blocks.
  /// Returns true if any kind of traceback is available (not necessarily from a
  /// final state).
  // the main procedure is done in GPU
  bool Decode(DecodableInterface *decodable);

  // the same to the version in lattice-faster-decoder.h
  /// says whether a final-state was active on the last frame.  If it was not, the
  /// lattice (or traceback) will end with states that are not final-states.
  bool ReachedFinal() const {
    return FinalRelativeCost() != std::numeric_limits<BaseFloat>::infinity();
  }

  // the same to the version in lattice-faster-decoder.h
  /// Outputs an FST corresponding to the single best path through the lattice.
  /// Returns true if result is nonempty (using the return status is deprecated,
  /// it will become void).  If "use_final_probs" is true AND we reached the
  /// final-state of the graph then it will include those as final-probs, else
  /// it will treat all final-probs as one.  Note: this just calls GetRawLattice()
  /// and figures out the shortest path.
  bool GetBestPath(Lattice *ofst,
                   bool use_final_probs = true) const;

  // the same to the version in lattice-faster-decoder.h
  /// Outputs an FST corresponding to the raw, state-level
  /// tracebacks.  Returns true if result is nonempty.
  /// If "use_final_probs" is true AND we reached the final-state
  /// of the graph then it will include those as final-probs, else
  /// it will treat all final-probs as one.
  /// The raw lattice will be topologically sorted.
  bool GetRawLattice(Lattice *ofst,
                     bool use_final_probs = true) const;

  // the same to the version in lattice-faster-decoder.h
  /// [Deprecated, users should now use GetRawLattice and determinize it
  /// themselves, e.g. using DeterminizeLatticePhonePrunedWrapper].
  /// Outputs an FST corresponding to the lattice-determinized
  /// lattice (one path per word sequence).   Returns true if result is nonempty.
  /// If "use_final_probs" is true AND we reached the final-state of the graph
  /// then it will include those as final-probs, else it will treat all
  /// final-probs as one.
  bool GetLattice(CompactLattice *ofst,
                  bool use_final_probs = true) const;

  // the same to the version in lattice-faster-decoder.h
  /// FinalRelativeCost() serves the same purpose as ReachedFinal(), but gives
  /// more information.  It returns the difference between the best (final-cost
  /// plus cost) of any token on the final frame, and the best cost of any token
  /// on the final frame.  If it is infinity it means no final-states were
  /// present on the final frame.  It will usually be nonnegative.  If it not
  /// too positive (e.g. < 5 is my first guess, but this is not tested) you can
  /// take it as a good indication that we reached the final-state with
  /// reasonable likelihood.
  BaseFloat FinalRelativeCost() const;

  // the same to the version in lattice-faster-decoder.h
  // Returns the number of frames decoded so far.  The value returned changes
  // whenever we call ProcessEmitting().
  inline int32 NumFramesDecoded() const { return active_toks_.size() - 1; }

 private:

  // Token and ForwardLink are the same to CPU decoder in lattice-faster-decoder.h

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
    BaseFloat extra_cost; // >= 0.  This is used in pruning a way tokens.
    // there is a comment in lattice-faster-decoder.cc explaining this;
    // search for "a note on the definition of extra_cost".

    ForwardLink *links; // Head of singly linked list of ForwardLinks

    Token *next; // Next in list of tokens for this frame.

    inline Token(BaseFloat tot_cost, BaseFloat extra_cost, ForwardLink *links,
                 Token *next):
      tot_cost(tot_cost), extra_cost(extra_cost), links(links), next(next) { }
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


  // a map from packed uint64 to the corresponding CPU Token address
  inline Token* ActiveToksMap(void*) const;

  // a map from (frame, idx) pair to the corresponding CPU Token address
  // in both CPU & GPU, We use frame index t and vector index i to trace a node
  inline Token* ActiveToksMap(int32 frame, int32 i) const;

  // create a CPU token based on a GPU token, link it into the linked list
  inline bool CreateAndLinkTok(BaseFloat cost, Token *&toks, Token* newtok,
                               bool last);

  // iteration on lattice arcs transfered from GPU, link the prev lattice node
  // and next lattice node recorded in the arc; after that, unlinked nodes
  // are implicitly pruned
  int32 AddLatticeArcs(int32 proc_frame);

  // final process lattice in CPU
  void FinalProcessLattice(cuTokenVector* last_toks, cuToken* toks_buf,
                           int* toks_sidx, LatLink* arcs_buf, int* arcs_size, 
                           int32 proc_frame);

  // the same to the version in lattice-faster-decoder.h
  // FinalizeDecoding() is a version of PruneActiveTokens that we call
  // (optionally) on the final frame.  Takes into account the final-prob of
  // tokens.  This function used to be called PruneActiveTokensFinal().
  void FinalizeDecoding();

  // the same to the version in lattice-faster-decoder.h
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

  // slightly different from the version in lattice-faster-decoder.h:
  // the iteration of last frame tokens is applied on GPU version TokenState
  // and it is transfered to CPU
  // details of the algorithm can be referred to lattice-faster-decoder.h:
  void ComputeFinalCosts(unordered_map<Token*, BaseFloat> *final_costs,
                         BaseFloat *final_relative_cost,
                         BaseFloat *final_best_cost) const;

  // the same to the version in lattice-faster-decoder.h
  // PruneForwardLinksFinal is a version of PruneForwardLinks that we call
  // on the final frame.  If there are final tokens active, it uses
  // the final-probs for pruning, otherwise it treats all tokens as final.
  void PruneForwardLinksFinal();

  // the same to the version in lattice-faster-decoder.h
  // Prune away any tokens on this frame that have no forward links.
  // [we don't do this in PruneForwardLinks because it would give us
  // a problem with dangling pointers].
  // It's called by PruneActiveTokens if any forward links have been pruned
  void PruneTokensForFrame(int32 frame_plus_one);

  // the same to the version in lattice-faster-decoder.h
  // Go backwards through still-alive tokens, pruning them if the
  // forward+backward cost is more than lat_beam away from the best path.  It's
  // possible to prove that this is "correct" in the sense that we won't lose
  // anything outside of lat_beam, regardless of what happens in the future.
  // delta controls when it considers a cost to have changed enough to continue
  // going backward and propagating the change.  larger delta -> will recurse
  // less far.
  void PruneActiveTokens(BaseFloat delta);

  // the same to the version in lattice-faster-decoder.h
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

  // the same to the version in lattice-faster-decoder.h
  // This function takes a singly linked list of tokens for a single frame, and
  // outputs a list of them in topological order (it will crash if no such order
  // can be found, which will typically be due to decoding graphs with epsilon
  // cycles, which are not allowed).  Note: the output list may contain NULLs,
  // which the caller should pass over; it just happens to be more efficient for
  // the algorithm to output a list that contains NULLs.
  static void TopSortTokens(Token *tok_list,
                            std::vector<Token*> *topsorted_list);

  // different from the version in lattice-faster-decoder.h:
  // as we pre-allocate all the token memory, we do not really clear them
  void ClearActiveTokens();

 private:
  const CudaFst& fst_;
  bool delete_fst_;

  CudaLatticeDecoder decoder_; // GPU decoder
  const CudaLatticeDecoderConfig &config_;
  int32 num_toks_; // current total number of toks allocated
  int32 num_frames_decoded_;

  // the TokenState of the last frame, used in ComputeFinalCosts()
  cuTokenVector* last_toks_;
  // used to index tokens by (frame, index), see ActiveToksMap() for details
  std::vector<Token*> active_toks_perframe_;
  std::vector<int> active_toks_size_perframe_; // size of toks in each frame
  // used to index arcs by (frame, index), see AddLatticeArcs() for details
  std::vector<ForwardLink*> active_arcs_perframe_;
  std::vector<int> active_arcs_size_perframe_; // size of arcs in each frame
  Token* toks_buf_; //as GPU is so fast, we need to pre-allocate toks
  int32 toks_buf_used_;

  // below definitions are the same to lattice-faster-decoder.h

  bool warned_;
  std::vector<TokenList> active_toks_; // Lists of tokens, indexed by
  // frame (members of TokenList are toks, must_prune_forward_links,
  // must_prune_tokens).
  std::vector<StateId> queue_;  // temp variable used in ProcessNonemitting,

  // make it class member to avoid internal new/delete.

  std::vector<BaseFloat> cost_offsets_; // This contains, for each
  // frame, an offset that was added to the acoustic log-likelihoods on that
  // frame in order to keep everything in a nice dynamic range i.e.  close to
  // zero, to reduce roundoff errors.

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

  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeFasterDecoderCuda);
};


} // end namespace kaldi.

#endif
