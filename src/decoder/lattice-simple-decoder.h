// decoder/lattice-simple-decoder.h

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen

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

#ifndef KALDI_DECODER_LATTICE_SIMPLE_DECODER_H_
#define KALDI_DECODER_LATTICE_SIMPLE_DECODER_H_


#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "fstext/fstext-lib.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"

#include <algorithm>
#ifdef _MSC_VER
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
using std::tr1::unordered_map;

namespace kaldi {

struct LatticeSimpleDecoderConfig {
  BaseFloat beam;
  BaseFloat lattice_beam;
  int32 prune_interval;
  bool determinize_lattice; // not inspected by this class... used in
  // command-line program.
  bool prune_lattice;
  BaseFloat beam_ratio;
  fst::DeterminizeLatticePhonePrunedOptions det_opts;
  LatticeSimpleDecoderConfig(): beam(16.0),
                                lattice_beam(10.0),
                                prune_interval(25),
                                determinize_lattice(true),
                                beam_ratio(0.9) { }
  void Register(OptionsItf *po) {
    det_opts.Register(po);
    po->Register("beam", &beam, "Decoding beam.");
    po->Register("lattice-beam", &lattice_beam, "Lattice generation beam");
    po->Register("prune-interval", &prune_interval, "Interval (in frames) at "
                 "which to prune tokens");
    po->Register("determinize-lattice", &determinize_lattice, "If true, "
                 "determinize the lattice (in a special sense, keeping only "
                 "best pdf-sequence for each word-sequence).");
  }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && lattice_beam > 0.0 && prune_interval > 0);
  }
};


/** Simplest possible decoder, included largely for didactic purposes and as a
    means to debug more highly optimized decoders.  See \ref decoders_simple
    for more information.
 */
class LatticeSimpleDecoder {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  // instantiate this class onece for each thing you have to decode.
  LatticeSimpleDecoder(const fst::Fst<fst::StdArc> &fst,
                       const LatticeSimpleDecoderConfig &config):
      fst_(fst), config_(config), num_toks_(0) { config.Check(); }
  
  ~LatticeSimpleDecoder() { ClearActiveTokens(); }

  LatticeSimpleDecoderConfig GetOptions() {
    return config_;
  }

  // Returns true if any kind of traceback is available (not necessarily from
  // a final state).
  bool Decode(DecodableInterface *decodable);

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
    ForwardLink(Token *next_tok, Label ilabel, Label olabel,
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
    // the minimum difference between the cost of the best path this is on,
    // and the cost of the absolute best path, under the assumption
    // that any of the currently active states at the decoding front may
    // eventually succeed (e.g. if you were to take the currently active states
    // one by one and compute this difference, and then take the minimum).
    
    ForwardLink *links; // Head of singly linked list of ForwardLinks
    
    Token *next; // Next in list of tokens for this frame.
    
    Token(BaseFloat tot_cost, BaseFloat extra_cost, ForwardLink *links,
          Token *next): tot_cost(tot_cost), extra_cost(extra_cost), links(links),
                        next(next) { }
    Token() {}
    void DeleteForwardLinks() {
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
  

  // FindOrAddToken either locates a token in cur_toks_, or if necessary inserts a new,
  // empty token (i.e. with no forward links) for the current frame.  [note: it's
  // inserted if necessary into cur_toks_ and also into the singly linked list
  // of tokens active on this frame (whose head is at active_toks_[frame]).
  //
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
  // if the token was newly created or the cost changed.
  inline Token *FindOrAddToken(StateId state, int32 frame, BaseFloat tot_cost,
                               bool emitting, bool *changed);
  
  // delta is the amount by which the extra_costs must
  // change before it sets "extra_costs_changed" to true.  If delta is larger,
  // we'll tend to go back less far toward the beginning of the file.
  void PruneForwardLinks(int32 frame, bool *extra_costs_changed,
                         bool *links_pruned,
                         BaseFloat delta);

  // PruneForwardLinksFinal is a version of PruneForwardLinks that we call
  // on the final frame.  If there are final tokens active, it uses the final-probs
  // for pruning, otherwise it treats all tokens as final.
  void PruneForwardLinksFinal(int32 frame);
  
  // Prune away any tokens on this frame that have no forward links. [we don't do
  // this in PruneForwardLinks because it would give us a problem with dangling
  // pointers].
  void PruneTokensForFrame(int32 frame);
  
  // Go backwards through still-alive tokens, pruning them.  note: cur_frame is
  // where cur_toks_ are (so we do not want to mess with it because these tokens
  // don't yet have forward pointers), but we do all previous frames, unless we
  // know that we can safely ignore them becaus the frame after them was unchanged.
  // delta controls when it considers a cost to have changed enough to continue
  // going backward and propagating the change.  larger delta -> will recurse less
  // far.
  void PruneActiveTokens(int32 cur_frame, BaseFloat delta);

  // Version of PruneActiveTokens that we call on the final
  // frame.  Takes into account the final-prob of tokens.
  // returns true if there were final states active (else it treats
  // all states as final while doing the pruning, and returns false--
  // this can be useful if you want partial lattice output, although
  // it can be dangerous, depending what you want the lattices for).
  // final_active_ is set intenally (by PruneForwardLinksFinal),
  // and final_probs_ (a hash) is also set by PruneForwardLinksFinal.
  void PruneActiveTokensFinal(int32 cur_frame);

  void ProcessEmitting(DecodableInterface *decodable, int32 frame);

  void ProcessNonemitting(int32 frame);

  void ClearActiveTokens(); // a cleanup routine, at utt end/begin

  // PruneCurrentTokens deletes the tokens from the "toks" map, but not
  // from the active_toks_ list, which could cause dangling forward pointers
  // (will delete it during regular pruning operation).
  void PruneCurrentTokens(BaseFloat beam, unordered_map<StateId, Token*> *toks);
  
  unordered_map<StateId, Token*> cur_toks_;
  unordered_map<StateId, Token*> prev_toks_;
  std::vector<TokenList> active_toks_; // Lists of tokens, indexed by
  // frame (members of TokenList are toks, must_prune_forward_links,
  // must_prune_tokens).
  const fst::Fst<fst::StdArc> &fst_;
  LatticeSimpleDecoderConfig config_;
  int32 num_toks_; // current total #toks allocated...
  bool warned_;
  bool final_active_; // use this to say whether we found active final tokens
  // on the last frame.
  std::map<Token*, BaseFloat> final_costs_; // A cache of final-costs
  // of tokens on the last frame-- it's just convenient to storeit this way.
  
};

// This function DecodeUtteranceLatticeSimple is used in several decoders, and
// we have moved it here.  Note: this is really "binary-level" code as it
// involves table readers and writers; we've just put it here as there is no
// other obvious place to put it.  If determinize == false, it writes to
// lattice_writer, else to compact_lattice_writer.  The writers for
// alignments and words will only be written to if they are open.
bool DecodeUtteranceLatticeSimple(
    LatticeSimpleDecoder &decoder, // not const but is really an input.
    DecodableInterface &decodable, // not const but is really an input.
    const TransitionModel &trans_model,
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
