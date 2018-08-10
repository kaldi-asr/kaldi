// decoder/lattice-faster-decoder-cuda.cc

// Copyright      2018  Zhehuai Chen

// See ../../COPYING for clarification regarding multiple authors
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http:// www.apache.org/licenses/LICENSE-2.0
// 
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/timer.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

#include "cuda-decoder-utils.h"
#include "decoder/lattice-faster-decoder-cuda.h"

namespace kaldi {

#define GET_ITEM_FROM_BUF(ret,buf,offset,nsize,size) { \
    assert((nsize)+(offset)<size); \
    ret=(buf)+(offset); \
    offset+=(nsize); \
}

// instantiate this class once for each thing you have to decode.
LatticeFasterDecoderCuda::LatticeFasterDecoderCuda(const CudaFst &fst, 
    const TransitionModel &trans_model, const CudaLatticeDecoderConfig &config):
  config_(config), fst_(fst), delete_fst_(false), num_toks_(0),
  decoder_(fst, trans_model, config_) {
  toks_buf_ = (Token*)malloc(sizeof(Token) * config_.max_tokens);
  toks_buf_used_ = 0;
}

LatticeFasterDecoderCuda::~LatticeFasterDecoderCuda() {
  // DeleteElems(toks_.Clear());
  ClearActiveTokens();
  if (toks_buf_) free(toks_buf_);
  if (delete_fst_) delete &(fst_);
}

// CPU decoding init
void LatticeFasterDecoderCuda::InitDecoding() {
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  // only need to memset the the length we use in last decoding
  memset(toks_buf_, 0, toks_buf_used_ * sizeof(Token));

  // below is the same to lattice-faster-decoder.cc
  cost_offsets_.clear(); // used in ComputeFinalCosts()
  ClearActiveTokens();
  warned_ = false;
  decoding_finalized_ = false;
  final_costs_.clear();
}

// Returns true if any kind of traceback is available (not necessarily from
// a final state).  It should only very rarely return false; this indicates
// an unusual search error.
bool LatticeFasterDecoderCuda::Decode(MatrixChunker *decodable) {
  PUSH_RANGE("CudaLatticeDecoder::Decode::init_search", 0);
  InitDecoding(); // CPU init
  decoder_.InitDecoding(); // GPU init
  decoder_.Decode(decodable);
  POP_RANGE;

  PUSH_RANGE("CudaLatticeDecoder::Decode::final", 1);
  cuToken* toks_buf;
  int* toks_sidx;
  LatLink* arcs_buf;
  int* arcs_size;
  cuTokenVector* last_tokv;
  // GPU lattice processing and D2H data trasnfer
  int num_frames_decoded;
  decoder_.FinalProcessLattice(&toks_buf, &toks_sidx, &arcs_buf, 
                               &arcs_size, &last_tokv, &num_frames_decoded);
  // CPU lattice processing
  FinalProcessLattice(last_tokv, toks_buf, toks_sidx, arcs_buf, arcs_size,
                      num_frames_decoded);
  // final lattice arc pruning and state trims
  // it is the same to CPU decoder in lattice-faster-decoder.h
  FinalizeDecoding();   
  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  POP_RANGE;
  assert(NumFramesDecoded() == NumFramesDecoded());
  return !active_toks_.empty() && active_toks_.back().toks != NULL;
}

// a map from packed uint64 to the corresponding CPU Token address
inline LatticeFasterDecoderCuda::Token* LatticeFasterDecoderCuda::
                                        ActiveToksMap(void* p) const {
  int32 frame, id;
  // the mapping is used in GPU code, see cuda-decoder-utils.h for details
  DECODE_TOK_IDX_PAIR(frame, id, p); 
  return ActiveToksMap(frame, id);
}

// a map from (frame, idx) pair to the corresponding CPU Token address
// in both CPU & GPU, We use frame index t and vector index i to trace a node
inline LatticeFasterDecoderCuda::Token* LatticeFasterDecoderCuda::
                                    ActiveToksMap(int32 frame, int32 id) const {
  assert(frame < active_toks_perframe_.size()); // toks of frame 0 is in index 0
  if (id >= active_toks_size_perframe_[frame]) KALDI_ERR << id<<"\t"<<frame<<"\t"<<active_toks_size_perframe_[frame]<<"\t"<<active_toks_perframe_.size();
  assert(id < active_toks_size_perframe_[frame]); // index < the size of toks
  Token* tok = active_toks_perframe_[frame] + id;
  assert(tok);
  return tok;
}

// create a CPU token based on a GPU token, link it into the linked list
inline bool LatticeFasterDecoderCuda::CreateAndLinkTok(BaseFloat cost,
                                  Token *&toks, Token* newtok, bool last) {
  Token *new_tok = newtok;
  {
    if (!new_tok->links && !last) return false;
    new_tok->tot_cost = cost;
    new_tok->next = toks;
    new_tok->extra_cost = 0;
  }
  toks = new_tok; // add into active_toks_;
  return true;
}

// iteration on lattice arcs transfered from GPU, link the prev lattice node 
// and next lattice node recorded in the arc; after that, unlinked nodes
// are implicitly pruned
inline int32 LatticeFasterDecoderCuda::AddLatticeArcs(int32 proc_frame) {
  PUSH_RANGE("AddLatticeArcs", 1);

  int32 num_arcs = 0;
  num_arcs = active_arcs_size_perframe_[proc_frame];
  ForwardLink* newarcs = active_arcs_perframe_[proc_frame];  
  for ( int32 j = 0; j < num_arcs; j++) {
    ForwardLink* newarc = newarcs + j;
    {
      newarc->next_tok = ActiveToksMap(newarc->next_tok); // LatLink.p1
      Token* prev_tok = ActiveToksMap(newarc->next); // LatLink.p2
      newarc->next = prev_tok->links;
      prev_tok->links = newarc;
    }
  }

  POP_RANGE;
  return num_arcs;
}

// final process lattice in CPU
void LatticeFasterDecoderCuda::FinalProcessLattice(cuTokenVector* last_toks,
    cuToken* toks_buf, int* toks_sidx, LatLink* arcs_buf, int* arcs_size, 
    int32 proc_frame) {
  if (proc_frame < 0) return;
  PUSH_RANGE("FinalProcessLattice", 3);

  assert(proc_frame <= config_.max_len);
  active_toks_.resize(proc_frame + 1);
  assert(proc_frame < active_toks_.size());

  last_toks_ = last_toks;
  for (int32 i = 0; i <= NumFramesDecoded(); i++) {
    Token* newtoks;
    int32 cur_toks_size = toks_sidx[i + 1] - toks_sidx[i];
    // get token from pre-allocated buffer
    GET_ITEM_FROM_BUF(newtoks, toks_buf_, toks_buf_used_, cur_toks_size,
                 config_.max_tokens);
    active_toks_perframe_.push_back(newtoks);
    active_toks_size_perframe_.push_back(cur_toks_size);
    if (config_.verbose > 3 || (cur_toks_size == 0)) KALDI_LOG << i << " " << cur_toks_size;
  }
  assert(proc_frame == active_toks_perframe_.size() - 1); // frame 0 has idx 0
  
  // as arcs_buf is in reverse sequence, we firstly store the toks address
  // and then reverse it
  int32 acc = 0;
  std::vector<ForwardLink*> active_arcs_perframe_tmp;
  std::vector<int> active_arcs_size_perframe_tmp;
  for (int32 i = NumFramesDecoded(); i >= 0; i--) {
    int32 num_arcs = arcs_size[i];
    if (config_.verbose > 3 || (num_arcs == 0 && i != 0)) 
      KALDI_LOG << i << " " << num_arcs;
    active_arcs_perframe_tmp.push_back((ForwardLink*)arcs_buf + acc);
    acc += num_arcs;
    active_arcs_size_perframe_tmp.push_back(num_arcs);
  }
  // reverse the vector
  for (int32 i = NumFramesDecoded(); i >= 0; i--) {
    active_arcs_perframe_.push_back(active_arcs_perframe_tmp[i]);
    active_arcs_size_perframe_.push_back(active_arcs_size_perframe_tmp[i]);
  }
  assert(proc_frame == active_arcs_perframe_.size() - 1); // frame 0 has idx 0
  
  // iteration on arcs frame by frame to add lattice arcs to lattice nodes
  // arcs are pruned in GPU
  for (int32 j = 0; j <= NumFramesDecoded(); j++) AddLatticeArcs(j);

  // trim lattice nodes  
  for (int32 j = 0; j <= NumFramesDecoded(); j++) {
    int32 cur_toks_size = toks_sidx[j + 1] - toks_sidx[j];
    Token* newtoks = active_toks_perframe_[j];
    int32 survive = 0;
    for (int32 i = 0; i < cur_toks_size;
         i++) { // always add into active_toks_map_, the newer key will replace the older
      cuToken& cur_tok = toks_buf[toks_sidx[j] + i];
      survive += CreateAndLinkTok(cur_tok.cost_, active_toks_[j].toks, newtoks + i,
                                      j == NumFramesDecoded());
    }
    // debug
    if (survive)
      num_toks_ += survive;
    else
      KALDI_WARN << "no survive after GPU prune @ frame " << j;
    KALDI_VLOG(4) << j << " " << cur_toks_size << " " << survive << " " << num_toks_;
  }
  KALDI_VLOG(3) << "tok after GPU prune " << num_toks_;

  POP_RANGE;
}

// Outputs an FST corresponding to the single best path through the lattice.
bool LatticeFasterDecoderCuda::GetBestPath(Lattice *olat,
    bool use_final_probs) const {
  Lattice raw_lat;
  GetRawLattice(&raw_lat, use_final_probs);
  ShortestPath(raw_lat, olat);
  // decoder_.GetBestPath(best_path, true);
  return (olat->NumStates() != 0);
}

// Outputs an FST corresponding to the raw, state-level
// tracebacks. :
bool LatticeFasterDecoderCuda::GetRawLattice(Lattice *ofst,
    bool use_final_probs) const {
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;

  // Note: you can't use the old interface (Decode()) if you want to
  // get the lattice with use_final_probs = false.  You'd have to do
  // InitDecoding() and then AdvanceDecoding().
  if (decoding_finalized_ && !use_final_probs)
    KALDI_ERR << "You cannot call FinalizeDecoding() and then call "
              << "GetRawLattice() with use_final_probs == false";

  unordered_map<Token*, BaseFloat> final_costs_local;

  const unordered_map<Token*, BaseFloat> &final_costs =
    (decoding_finalized_ ? final_costs_ : final_costs_local);
  if (!decoding_finalized_ && use_final_probs)
    ComputeFinalCosts(&final_costs_local, NULL, NULL);

  ofst->DeleteStates();
  // num-frames plus one (since frames are one-based, and we have
  // an extra frame for the start-state).
  int32 num_frames = active_toks_.size() - 1;
  KALDI_ASSERT(num_frames > 0);
  const int32 bucket_count = num_toks_ / 2 + 3;
  unordered_map<Token*, StateId> tok_map(bucket_count);
  // First create all states.
  std::vector<Token*> token_list;
  for (int32 f = 0; f <= num_frames; f++) {
    if (active_toks_[f].toks == NULL) {
      KALDI_WARN << "GetRawLattice: no tokens active on frame " << f
                 << ": not producing lattice.\n";
      return false;
    }
    TopSortTokens(active_toks_[f].toks, &token_list);
    for (size_t i = 0; i < token_list.size(); i++)
      if (token_list[i] != NULL)
        tok_map[token_list[i]] = ofst->AddState();
  }
  // The next statement sets the start state of the output FST.  Because we
  // topologically sorted the tokens, state zero must be the start-state.
  ofst->SetStart(0);

  KALDI_VLOG(4) << "init:" << num_toks_ / 2 + 3 << " buckets:"
                << tok_map.bucket_count() << " load:" << tok_map.load_factor()
                << " max:" << tok_map.max_load_factor();
  // Now create all arcs.
  for (int32 f = 0; f <= num_frames; f++) {
    for (Token *tok = active_toks_[f].toks; tok != NULL; tok = tok->next) {
      StateId cur_state = tok_map[tok];
      for (ForwardLink *l = tok->links;
           l != NULL;
           l = l->next) {
        unordered_map<Token*, StateId>::const_iterator iter =
          tok_map.find(l->next_tok);
        StateId nextstate = iter->second;
        KALDI_ASSERT(iter != tok_map.end());
        BaseFloat cost_offset = 0.0;
        if (l->ilabel != 0) {  // emitting..
          // KALDI_ASSERT(f >= 0 && f < cost_offsets_.size());
          cost_offset = 0; // cost_offsets_[f];
        }
        Arc arc(l->ilabel, l->olabel,
                Weight(l->graph_cost, l->acoustic_cost - cost_offset),
                nextstate);
        ofst->AddArc(cur_state, arc);
      }
      if (f == num_frames) {
        if (use_final_probs && !final_costs.empty()) {
          unordered_map<Token*, BaseFloat>::const_iterator iter =
            final_costs.find(tok);
          if (iter != final_costs.end())
            ofst->SetFinal(cur_state, LatticeWeight(iter->second, 0));
        } else {
          ofst->SetFinal(cur_state, LatticeWeight::One());
        }
      }
    }
  }
  return (ofst->NumStates() > 0);
}

// FinalizeDecoding() is a version of PruneActiveTokens that we call
// (optionally) on the final frame.  Takes into account the final-prob of
// tokens.  This function used to be called PruneActiveTokensFinal().
void LatticeFasterDecoderCuda::FinalizeDecoding() {
  int32 final_frame_plus_one = NumFramesDecoded();
  int32 num_toks_begin = num_toks_;
  // PruneForwardLinksFinal() prunes final frame (with final-probs), and
  // sets decoding_finalized_.
  PruneForwardLinksFinal();
  for (int32 f = final_frame_plus_one - 1; f >= 0; f--) {
    bool b1, b2; // values not used.
    BaseFloat dontcare = 0.0; // delta of zero means we must always update
    PruneForwardLinks(f, &b1, &b2, dontcare);
    PruneTokensForFrame(f + 1);
  }
  PruneTokensForFrame(0);
  KALDI_VLOG(4) << "pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}


// prunes outgoing links for all tokens in active_toks_[frame]
// it's called by PruneActiveTokens
// all links, that have link_extra_cost > lattice_beam are pruned
void LatticeFasterDecoderCuda::PruneForwardLinks(
  int32 frame_plus_one, bool *extra_costs_changed,
  bool *links_pruned, BaseFloat delta) {
  Timer tm;
  // delta is the amount by which the extra_costs must change
  // If delta is larger,  we'll tend to go back less far
  // toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned

  *extra_costs_changed = false;
  *links_pruned = false;
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  if (active_toks_[frame_plus_one].toks ==
      NULL) {  // empty list; should not happen.
    if (!warned_) {
      KALDI_WARN << "No tokens alive [doing pruning].. warning first "
                 "time only for each utterance\n";
      warned_ = true;
    }
  }

  // We have to iterate until there is no more change, because the links
  // are not guaranteed to be in topological order.
  bool changed = true;  // difference new minus old extra cost >= delta ?
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame_plus_one].toks;
         tok != NULL; tok = tok->next) {
      ForwardLink *link, *prev_link = NULL;
      // will recompute tok_extra_cost for tok.
      BaseFloat tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      // tok_extra_cost is the best (min) of link_extra_cost of outgoing links
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost = next_tok->extra_cost +
                                    ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
                                     - next_tok->tot_cost);  // difference in brackets is >= 0
        // link_exta_cost is the difference in score between the best paths
        // through link source state and through link destination state
        KALDI_ASSERT(link_extra_cost == link_extra_cost);  // check for NaN
        if (link_extra_cost > config_.lattice_beam) {  // excise link
          ForwardLink *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          // delete link;
          link = next_link;  // advance link but leave prev_link the same.
          *links_pruned = true;
        } else {   // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) {  // this is just a precaution.
            if (config_.verbose > 1 && link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost) {
            tok_extra_cost = link_extra_cost;
          }
          prev_link = link;  // move to next link
          link = link->next;
        }
      }  // for all outgoing links
      if (fabs(tok_extra_cost - tok->extra_cost) > delta)
        changed = true;   // difference new minus old is bigger than delta
      tok->extra_cost = tok_extra_cost;
      // will be +infinity or <= lattice_beam_.
      // infinity indicates, that no forward link survived pruning
    }  // for all Token on active_toks_[frame]
    if (changed) *extra_costs_changed = true;

    // Note: it's theoretically possible that aggressive compiler
    // optimizations could cause an infinite loop here for small delta and
    // high-dynamic-range scores.
  } // while changed
}

// PruneForwardLinksFinal is a version of PruneForwardLinks that we call
// on the final frame.  If there are final tokens active, it uses
// the final-probs for pruning, otherwise it treats all tokens as final.
void LatticeFasterDecoderCuda::PruneForwardLinksFinal() {
  KALDI_ASSERT(!active_toks_.empty());
  int32 frame_plus_one = active_toks_.size() - 1;
  if (active_toks_[frame_plus_one].toks == NULL)  // empty list; should not happen.
    KALDI_WARN << "No tokens alive at end of file";

  typedef unordered_map<Token*, BaseFloat>::const_iterator IterType;
  ComputeFinalCosts(&final_costs_, &final_relative_cost_, &final_best_cost_);
  decoding_finalized_ = true;
  // We call DeleteElems() as a nicety, not because it's really necessary;
  // otherwise there would be a time, after calling PruneTokensForFrame() on the
  // final frame, when toks_.GetList() or toks_.Clear() would contain pointers
  // to nonexistent tokens.
  // DeleteElems(toks_.Clear());

  // Now go through tokens on this frame, pruning forward links...  may have to
  // iterate a few times until there is no more change, because the list is not
  // in topological order.  This is a modified version of the code in
  // PruneForwardLinks, but here we also take account of the final-probs.
  bool changed = true;
  BaseFloat delta = 1.0e-05;
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame_plus_one].toks;
         tok != NULL; tok = tok->next) {
      ForwardLink *link, *prev_link = NULL;
      // will recompute tok_extra_cost.  It has a term in it that corresponds
      // to the "final-prob", so instead of initializing tok_extra_cost to infinity
      // below we set it to the difference between the (score+final_prob) of this token,
      // and the best such (score+final_prob).
      BaseFloat final_cost;
      if (final_costs_.empty()) {
        final_cost = 0.0;
      } else {
        IterType iter = final_costs_.find(tok);
        if (iter != final_costs_.end())
          final_cost = iter->second;
        else
          final_cost = std::numeric_limits<BaseFloat>::infinity();
      }
      BaseFloat tok_extra_cost = tok->tot_cost + final_cost - final_best_cost_;
      // tok_extra_cost will be a "min" over either directly being final, or
      // being indirectly final through other links, and the loop below may
      // decrease its value:
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost = next_tok->extra_cost +
                                    ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
                                     - next_tok->tot_cost);
        if (link_extra_cost > config_.lattice_beam) {  // excise link
          ForwardLink *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          // delete link;
          link = next_link; // advance link but leave prev_link the same.
        } else { // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) { // this is just a precaution.
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost)
            tok_extra_cost = link_extra_cost;
          prev_link = link;
          link = link->next;
        }
      }
      // prune away tokens worse than lattice_beam above best path.  This step
      // was not necessary in the non-final case because then, this case
      // showed up as having no forward links.  Here, the tok_extra_cost has
      // an extra component relating to the final-prob.
      if (tok_extra_cost > config_.lattice_beam)
        tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      // to be pruned in PruneTokensForFrame

      if (!ApproxEqual(tok->extra_cost, tok_extra_cost, delta))
        changed = true;
      tok->extra_cost = tok_extra_cost; // will be +infinity or <= lattice_beam_.
    }
  } // while changed
}

BaseFloat LatticeFasterDecoderCuda::FinalRelativeCost() const {
  if (!decoding_finalized_) {
    BaseFloat relative_cost;
    ComputeFinalCosts(NULL, &relative_cost, NULL);
    return relative_cost;
  } else {
    // we're not allowed to call that function if FinalizeDecoding() has
    // been called; return a cached value.
    return final_relative_cost_;
  }
}


// Prune away any tokens on this frame that have no forward links.
// [we don't do this in PruneForwardLinks because it would give us
// a problem with dangling pointers].
// It's called by PruneActiveTokens if any forward links have been pruned
void LatticeFasterDecoderCuda::PruneTokensForFrame(int32 frame_plus_one) {
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  int32 num_toks_s = num_toks_;
  Token *&toks = active_toks_[frame_plus_one].toks;
  if (toks == NULL)
    KALDI_WARN << "frame: " << frame_plus_one << " No tokens alive [doing pruning]";
  Token *tok, *next_tok, *prev_tok = NULL;
  for (tok = toks; tok != NULL; tok = next_tok) {
    next_tok = tok->next;
    if (tok->extra_cost == std::numeric_limits<BaseFloat>::infinity()) {
      // token is unreachable from end of graph; (no forward links survived)
      // excise tok from list and delete tok.
      if (prev_tok != NULL) prev_tok->next = tok->next;
      else toks = tok->next;
      // delete tok;
      num_toks_--;
    } else {  // fetch next Token
      prev_tok = tok;
    }
  }
  KALDI_VLOG(5) << "PR: " << frame_plus_one << "," << num_toks_s - num_toks_;
}

// Go backwards through still-alive tokens, pruning them, starting not from
// the current frame (where we want to keep all tokens) but from the frame before
// that.  We go backwards through the frames and stop when we reach a point
// where the delta-costs are not changing (and the delta controls when we consider
// a cost to have "not changed").
void LatticeFasterDecoderCuda::PruneActiveTokens(BaseFloat delta) {
  PUSH_RANGE("PruneActiveTokens", 4);
  int32 cur_frame_plus_one = NumFramesDecoded(); // till last frame
  int32 num_toks_begin = num_toks_;
  // The index "f" below represents a "frame plus one", i.e. you'd have to subtract
  // one to get the corresponding index for the decodable object.
  for (int32 f = cur_frame_plus_one - 1; f >= 0; f--) {
    // Reason why we need to prune forward links in this situation:
    // (1) we have never pruned them (new TokenList)
    // (2) we have not yet pruned the forward links to the next f,
    // after any of those tokens have changed their extra_cost.
    if (active_toks_[f].must_prune_forward_links) {
      bool extra_costs_changed = false, links_pruned = false;
      PruneForwardLinks(f, &extra_costs_changed, &links_pruned, delta);
      if (extra_costs_changed && f > 0) // any token has changed extra_cost
        active_toks_[f - 1].must_prune_forward_links = true;
      if (links_pruned) // any link was pruned
        active_toks_[f].must_prune_tokens = true;
      active_toks_[f].must_prune_forward_links = false; // job done
    }
    if (f + 1 < cur_frame_plus_one &&    // except for last f (no forward links)
        active_toks_[f + 1].must_prune_tokens) {
      PruneTokensForFrame(f + 1);
      active_toks_[f + 1].must_prune_tokens = false;
    }
  }
  KALDI_VLOG(4) << "PruneActiveTokens: pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
  POP_RANGE;
}

void LatticeFasterDecoderCuda::ComputeFinalCosts(
  unordered_map<Token*, BaseFloat> *final_costs,
  BaseFloat *final_relative_cost,
  BaseFloat *final_best_cost) const {
  KALDI_ASSERT(!decoding_finalized_);
// *final_relative_cost=0;
// *final_best_cost=0;
// KALDI_WARN<<"unfinished here";
  if (final_costs != NULL)
    final_costs->clear();
  BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat best_cost = infinity, best_cost_with_final = infinity;

  cuTokenVector& cur_toks = *this->last_toks_;
  for (int32 i = 0; i < cur_toks.Size(); i++) {
    StateId state = cur_toks[i].state;
    Token* tok = ActiveToksMap(NumFramesDecoded(), i);

    BaseFloat final_cost = fst_.Final(state);
    BaseFloat cost = tok->tot_cost,
              cost_with_final = cost + final_cost;
    best_cost = std::min(cost, best_cost);
    best_cost_with_final = std::min(cost_with_final, best_cost_with_final);
    if (final_costs != NULL && final_cost != infinity)
      (*final_costs)[tok] = final_cost;
  }

  if (final_relative_cost != NULL) {
    if (best_cost == infinity && best_cost_with_final == infinity) {
      // Likely this will only happen if there are no tokens surviving.
      // This seems the least bad way to handle it.
      *final_relative_cost = infinity;
    } else {
      *final_relative_cost = best_cost_with_final - best_cost;
    }
  }
  if (final_best_cost != NULL) {
    if (best_cost_with_final != infinity) { // final-state exists.
      *final_best_cost = best_cost_with_final;
    } else { // no final-state exists.
      *final_best_cost = best_cost;
    }
  }
}



void LatticeFasterDecoderCuda::ClearActiveTokens() { // a cleanup routine, at utt end/begin
  // for (size_t i = 0; i < active_toks_.size(); i++) {
  // Delete all tokens alive on this frame, and any forward
  // links they may have.
  // for (Token *tok = active_toks_[i].toks; tok != NULL; ) {
  // tok->DeleteForwardLinks();
  // Token *next_tok = tok->next;
  // delete tok;
  // num_toks_--;
  // tok = next_tok;
  // }
  // }
  // for (auto i:active_toks_perframe_) free(i);
  // for (auto i:active_arcs_perframe_) free(i);
  active_toks_size_perframe_.clear();
  active_toks_size_perframe_.reserve(2000);
  active_arcs_size_perframe_.clear();
  active_arcs_size_perframe_.reserve(2000);
  active_toks_perframe_.clear();
  active_toks_perframe_.reserve(2000);
  active_arcs_perframe_.clear();
  active_arcs_perframe_.reserve(2000);
  num_toks_ = 0;
  active_toks_.clear();
  active_toks_.reserve(2000);
  toks_buf_used_ = 0;
  KALDI_ASSERT(num_toks_ == 0);
}

// static
void LatticeFasterDecoderCuda::TopSortTokens(Token *tok_list,
    std::vector<Token*> *topsorted_list) {
  unordered_map<Token*, int32> token2pos;
  typedef unordered_map<Token*, int32>::iterator IterType;
  int32 num_toks = 0;
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    num_toks++;
  int32 cur_pos = 0;
  // We assign the tokens numbers num_toks - 1, ... , 2, 1, 0.
  // This is likely to be in closer to topological order than
  // if we had given them ascending order, because of the way
  // new tokens are put at the front of the list.
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    token2pos[tok] = num_toks - ++cur_pos;

  unordered_set<Token*> reprocess;

  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter) {
    Token *tok = iter->first;
    int32 pos = iter->second;
    for (ForwardLink *link = tok->links; link != NULL; link = link->next) {
      if (link->ilabel == 0) {
        // We only need to consider epsilon links, since non-epsilon links
        // transition between frames and this function only needs to sort a list
        // of tokens from a single frame.
        IterType following_iter = token2pos.find(link->next_tok);
        if (following_iter != token2pos.end()) { // another token on this frame,
          // so must consider it.
          int32 next_pos = following_iter->second;
          if (next_pos < pos) { // reassign the position of the next Token.
            following_iter->second = cur_pos++;
            reprocess.insert(link->next_tok);
          }
        }
      }
    }
    // In case we had previously assigned this token to be reprocessed, we can
    // erase it from that set because it's "happy now" (we just processed it).
    reprocess.erase(tok);
  }

  size_t max_loop = 1000000, loop_count; // max_loop is to detect epsilon cycles.
  for (loop_count = 0;
       !reprocess.empty() && loop_count < max_loop; ++loop_count) {
    std::vector<Token*> reprocess_vec;
    for (unordered_set<Token*>::iterator iter = reprocess.begin();
         iter != reprocess.end(); ++iter)
      reprocess_vec.push_back(*iter);
    reprocess.clear();
    for (std::vector<Token*>::iterator iter = reprocess_vec.begin();
         iter != reprocess_vec.end(); ++iter) {
      Token *tok = *iter;
      int32 pos = token2pos[tok];
      // Repeat the processing we did above (for comments, see above).
      for (ForwardLink *link = tok->links; link != NULL; link = link->next) {
        if (link->ilabel == 0) {
          IterType following_iter = token2pos.find(link->next_tok);
          if (following_iter != token2pos.end()) {
            int32 next_pos = following_iter->second;
            if (next_pos < pos) {
              following_iter->second = cur_pos++;
              reprocess.insert(link->next_tok);
            }
          }
        }
      }
    }
  }
  KALDI_ASSERT(loop_count < max_loop && "Epsilon loops exist in your decoding "
               "graph (this is not allowed!)");

  topsorted_list->clear();
  topsorted_list->resize(cur_pos, NULL);  // create a list with NULLs in between.
  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter)
    (*topsorted_list)[iter->second] = iter->first;
}

} // end namespace kaldi.
