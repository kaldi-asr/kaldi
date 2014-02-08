// decoder/lattice-tracking-decoder.h

// Copyright 2012 BUT (Author: Mirko Hannemann) Johns Hopkins University
// (Author: Daniel Povey)
//           2014 Guoguo Chen

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

#ifndef KALDI_DECODER_LATTICE_TRACKING_DECODER_H_
#define KALDI_DECODER_LATTICE_TRACKING_DECODER_H_

#include "util/stl-utils.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "fstext/fstext-lib.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {

struct LatticeTrackingDecoderConfig {
  BaseFloat beam;
  int32 max_active;
  BaseFloat lattice_beam;
  int32 prune_interval;
  bool determinize_lattice; // not inspected by this class... used in
  // command-line program.
  BaseFloat beam_delta; // has nothing to do with beam_ratio
  BaseFloat hash_ratio;
  BaseFloat extra_beam; // added to beam of tracked tokens from first pass
  BaseFloat max_beam; // maximum beam (in case tracked tokens go too far from beam)
  fst::DeterminizeLatticePhonePrunedOptions det_opts;

  LatticeTrackingDecoderConfig(): beam(16.0),
                                max_active(std::numeric_limits<int32>::max()),
                                lattice_beam(10.0),
                                prune_interval(25),
                                determinize_lattice(true),
                                beam_delta(0.5),
                                hash_ratio(2.0),
                                extra_beam(4.0),
                                max_beam(40.0) { }
  void Register(OptionsItf *po) {
    det_opts.Register(po);
    po->Register("beam", &beam, "Decoding beam.");
    po->Register("max-active", &max_active, "Decoder max active states.");
    po->Register("lattice-beam", &lattice_beam, "Lattice generation beam");
    po->Register("prune-interval", &prune_interval, "Interval (in frames) at "
                 "which to prune tokens");
    po->Register("determinize-lattice", &determinize_lattice, "If true, "
                 "determinize the lattice (in a special sense, keeping only "
                 "best pdf-sequence for each word-sequence).");
    po->Register("beam-delta", &beam_delta, "Increment used in decoding");
    po->Register("hash-ratio", &hash_ratio, "Setting used in decoder to control"
                 " hash behavior");
    po->Register("extra-beam", &extra_beam, "Increment used in decoding (added "
                 "to worst tracked token from first pass)");
    po->Register("max-beam", &max_beam, "Maximum beam (in case tracked tokens "
                 "go too far from beam)");

  }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && max_active > 1 && lattice_beam > 0.0 
                 && prune_interval > 0 && beam_delta > 0.0 && hash_ratio >= 1.0
                 && extra_beam >= 0.0 && max_beam >= beam);
  }
};


/**
   This is the "tracking" version of the lattice-generating decoder.  It relates
   to a "forward-backward decoding" concept that Mirko Hannemann and Dan Povey
   are pursuing.  The idea is to use information from, say, a forward pass over
   the data to inform the search of a backward pass over the same data.
   Note: this code is modified from lattice-faster-decoder.h.

   The basic idea is we take a lattice, say from a backward-in-time decoding
   (for complementarity with the forward-in-time one), and for the forward-in-time
   HCLG graph, we work out which arcs in the graph each lattice arc corresponds to...
   this is done in a shortest-path sense, in that we work out for each path through
   the lattice the best path through HCLG that it corresponds to... then when
   decoding forward in time, we make sure on each time to, to keep each of the
   arcs in HCLG that the backwards lattice would require us to keep active..

   The lattice that we give to the decoder to help the search, has a format where
   the ilabels contain the state in HCLG (corresponding to the preceding-state of
   the arc in the lattice), and the olabels contain the arc index corresponding to
   that transition in the lattice, i.e. the offset into the list of arcs transitioning
   from that state in HCLG.   
 */
class LatticeTrackingDecoder {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  // instantiate this class once for each thing you have to decode.
  LatticeTrackingDecoder(const fst::Fst<fst::StdArc> &fst,
                         const LatticeTrackingDecoderConfig &config);

  void SetOptions(const LatticeTrackingDecoderConfig &config) {
    config_ = config;
  }

  LatticeTrackingDecoderConfig GetOptions() {
    return config_;
  }

  ~LatticeTrackingDecoder() { ClearActiveTokens(); }

  /// Returns true if any kind of traceback is available (not necessarily from
  /// a final state).
  /// The "arc_graph" argument is a specially processed lattice derived from
  /// a normal lattice (typically from decoding in an opposite direction in time),
  /// and the HCLG decoding graph we're decoding with in this decoder.
  /// Look at the comment for LatticeTrackingDecoder for more details on its
  /// format.
  bool Decode(DecodableInterface *decodable,
              const fst::StdVectorFst &arc_graph);
  
  /// says whether a final-state was active on the last frame.  If it was not, the
  /// lattice (or traceback) will end with states that are not final-states.
  bool ReachedFinal() const { return final_active_; }

  // Outputs an FST corresponding to the single best path
  // through the lattice.
  bool GetBestPath(fst::MutableFst<LatticeArc> *ofst) const;

  // Outputs an FST corresponding to the raw, state-level
  // tracebacks.
  bool GetRawLattice(fst::MutableFst<LatticeArc> *ofst) const;

  // This function is now deprecated, since now we do determinization from
  // outside the LatticeTrackingDecoder class.
  // Outputs an FST corresponding to the lattice-determinized
  // lattice (one path per word sequence).
  bool GetLattice(fst::MutableFst<CompactLatticeArc> *ofst) const;
  
 private:
  struct Token;
  // ForwardLinks are the links from a token to a token on the next frame.
  // or sometimes on the current frame (for input-epsilon links).
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
  // links to it when we process the next frame.
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
    
    StateId lat_state; // current state in graph arc lattice from first pass decoding
    // lat_state == fst::kNoStateId means that this token is not tracked
    
    inline Token(BaseFloat tot_cost, BaseFloat extra_cost, ForwardLink *links,
                 Token *next, StateId lat_state): tot_cost(tot_cost), extra_cost(extra_cost),
                 links(links), next(next), lat_state(lat_state) { }
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
  
  // head and tail of per-frame list of Tokens (list is in topological order),
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

  // FindOrAddToken either locates a token in hash of toks_,
  // or if necessary inserts a new, empty token (i.e. with no forward links)
  // for the current frame.  [note: it's inserted if necessary into hash toks_
  // and also into the singly linked list of tokens active on this frame
  // (whose head is at active_toks_[frame]).
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
  // if the token was newly created or the cost was changed,
  // or when the token inherits the status "tracked"
  // this will be needed when deciding whether to put it to the queue
  // lat_state is the next state in the arc graph lattice
  inline Token *FindOrAddToken(StateId state, StateId lat_state, int32 frame,
                               BaseFloat tot_cost, bool *changed);
  
  // prunes outgoing links for all tokens in active_toks_[frame]
  // it's called by PruneActiveTokens
  // all links, that have link_extra_cost > lattice_beam are pruned
  // delta is the amount by which the extra_costs must change
  // before we set *extra_costs_changed = true.
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned
  void PruneForwardLinks(int32 frame, bool *extra_costs_changed,
                         bool *links_pruned,
                         BaseFloat delta);


  // PruneForwardLinksFinal is a version of PruneForwardLinks that we call
  // on the final frame.  If there are final tokens active, it uses
  // the final-probs for pruning, otherwise it treats all tokens as final.
  void PruneForwardLinksFinal(int32 frame);
  
  // Prune away any tokens on this frame that have no forward links.
  // [we don't do this in PruneForwardLinks because it would give us
  // a problem with dangling pointers].
  // It's called by PruneActiveTokens if any forward links have been pruned
  void PruneTokensForFrame(int32 frame);
  
  // Go backwards through still-alive tokens, pruning them.  note: cur_frame is
  // where hash toks_ are (so we do not want to mess with it because these tokens
  // don't yet have forward pointers), but we do all previous frames, unless we
  // know that we can safely ignore them because the frame after them was unchanged.
  // delta controls when it considers a cost to have changed enough to continue
  // going backward and propagating the change.
  // for a larger delta, we will recurse less far back
  void PruneActiveTokens(int32 cur_frame, BaseFloat delta);

  /// Version of PruneActiveTokens that we call on the final frame.
  /// Takes into account the final-prob of tokens.
  void PruneActiveTokensFinal(int32 cur_frame);
  
  /// Gets the weight cutoff.  Also counts the active tokens.
  BaseFloat GetCutoff(Elem *list_head, size_t *tok_count,
                      BaseFloat *adaptive_beam, Elem **best_elem);

  /// Processes emitting arcs for one frame.  Propagates from prev_toks_ to cur_toks_.
  void ProcessEmitting(DecodableInterface *decodable, int32 frame);

  /// Processes nonemitting (epsilon) arcs for one frame.
  /// Ccalled after ProcessEmitting on each frame.
  /// TODO: could possibly add adaptive_beam back as an argument here (was
  /// returned from ProcessEmitting, in faster-decoder.h).
  void ProcessNonemitting(int32 frame);

  // HashList defined in ../util/hash-list.h.  It actually allows us to maintain
  // more than one list (e.g. for current and previous frames), but only one of
  // them at a time can be indexed by StateId.
  HashList<StateId, Token*> toks_;
  std::vector<TokenList> active_toks_; // Lists of tokens, indexed by
  // frame (members of TokenList are toks, must_prune_forward_links,
  // must_prune_tokens).
  std::vector<StateId> queue_;  // temp variable used in ProcessNonemitting,
  std::vector<BaseFloat> tmp_array_;  // used in GetCutoff.
  // make it class member to avoid internal new/delete.
  const fst::Fst<fst::StdArc> &fst_;
  std::vector<BaseFloat> cost_offsets_; // This contains, for each
  // frame, an offset that was added to the acoustic likelihoods on that
  // frame in order to keep everything in a nice dynamic range.
  LatticeTrackingDecoderConfig config_;
  int32 num_toks_; // current total #toks allocated...
  bool warned_;
  bool final_active_; // use this to say whether we found active final tokens
  // on the last frame.
  std::map<Token*, BaseFloat> final_costs_; // A cache of final-costs
  // of tokens on the last frame-- it's just convenient to store it this way.

  const fst::StdVectorFst *arc_graph_; // graph arc lattice from first pass

  // It might seem unclear why we call ClearToks(toks_.Clear()).
  // There are two separate cleanup tasks we need to do at when we start a new file.
  // one is to delete the Token objects in the list; the other is to delete
  // the Elem objects.  toks_.Clear() just clears them from the hash and gives ownership
  // to the caller, who then has to call toks_.Delete(e) for each one.  It was designed
  // this way for convenience in propagating tokens from one frame to the next.
  void ClearToks(Elem *list);
  
  void ClearActiveTokens();
  
};


// This function DecodeUtteranceLatticeFaster is used in several decoders, and
// we have moved it here.  Note: this is really "binary-level" code as it
// involves table readers and writers; we've just put it here as there is no
// other obvious place to put it.  If determinize == false, it writes to
// lattice_writer, else to compact_lattice_writer.  The writers for
// alignments and words will only be written to if they are open.
bool DecodeUtteranceLatticeTracking(
    LatticeTrackingDecoder &decoder, // not const but is really an input.
    DecodableInterface &decodable, // not const but is really an input.
    const TransitionModel &trans_model,
    const fst::StdVectorFst &arc_graph, // contains graph arcs from forward pass lattice
    const fst::SymbolTable *word_syms,
    std::string utt,
    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignments_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr);  // puts utterance's likelihood in like_ptr on success.


} // end namespace kaldi.

#endif
