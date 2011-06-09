// decoder/kaldi-decoder-inl.h

// Copyright 2009-2011  Mirko Hannemann, Lukas Burget

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

#ifndef KALDI_DECODER_KALDI_DECODER_INL_H_
#define KALDI_DECODER_KALDI_DECODER_INL_H_

#include <cstdio>
#include <ctime>
#include <iomanip>

namespace kaldi {
  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  KaldiDecoder<Decodable, Fst>::KaldiDecoder(KaldiDecoderOptions opts) :
      options_(opts) {
    tokens_ = new TokenStore(&link_store_);
    tokens_next_ = new TokenStore(&link_store_);
    active_tokens_.Init(tokens_, tokens_next_);
    frame_index_ = 0;
    p_decodable_ = NULL;
    reconet_     = NULL;
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  KaldiDecoder<Decodable, Fst>::~KaldiDecoder() {
    delete tokens_;
    delete tokens_next_;
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  fst::VectorFst<typename fst::StdArc>* KaldiDecoder<Decodable, Fst>::Decode(
      const Fst &fst, Decodable *decodable) {
    // the decoding main routine

    std::clock_t start = std::clock();
    InitDecoding(fst, decodable); // insert first token to active tokens list
    frame_index_ = -1;

    do {  // go over all feature frames and decode
      frame_index_++;
      DEBUG_OUT1("==== FRAME " << frame_index_)
      if ((frame_index_%50) == 0) KALDI_LOG << "==== FRAME " << frame_index_;

      ProcessToDoList();
      // all active tokens from last frame are by now processed
      // tokens for next frame are on the priority queue
      ProcessTokens(false);  // processes priority queue and active tokens
      // all tokens within beam are forwarded to all arcs:
      //   a) non-emitting arcs go again onto the queue in topological order,
      //   b) emitting arcs go to active_tokens_(next)
      // adds for each word a link to the lattice link record

      EvaluateAndPrune();  // evaluates all tokens on active_tokens_(next)
      // adds GMM likelihood score (based on state) to each token
      // computes best likelihood and new beamwidth for pruning
      // new states are explored using depth first network visitor

      // pruning: removes all tokens outside the beam from active_tokens_
      //   a) main loop of EvaluateAndPrune
      //   b) main loop of ProcessTokens
      //   c) always when PassTokenThroughArc
      //  beam_threshold_: determined within EvaluateAndPrune
      //  options_.beamwidth: parameter, on which we compute beam_threshold_
      //  options_.max_active_tokens: maximum number of active states
    } while (!(p_decodable_->IsLastFrame(frame_index_)));

    DEBUG_OUT1("==== FINISH FRAME " << frame_index_)
    KALDI_LOG << "==== FINISH FRAME " << frame_index_;
    FinalizeDecoding(); // processes the active states and the priority queue
    // forwards to final state, backtracking, build output FST, memory clean-up
    double diff = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    KALDI_LOG << "TIME:" << std::setprecision(4) << diff<<std::setprecision(6);
    return output_arcs_;  // or maybe frame_index_ ? number of decoded frames?
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  void KaldiDecoder<Decodable, Fst>::InitDecoding(const Fst &fst,
                                                  Decodable *decodable) {
    reconet_ = &fst;  // recognition network FST
    assert(reconet_ != NULL);
    assert(reconet_->Start() != fst::kNoStateId);  // gets the initial state
    // kNoState means empty FST
    p_decodable_ = decodable;  // acoustic model with features
    assert(p_decodable_ != NULL);
    if ((Weight::Properties() & (fst::kPath | fst::kRightSemiring)) !=
        (fst::kPath | fst::kRightSemiring)) {
      KALDI_ERR << "Weight must have path property and be right distributive: "
                << Weight::Type();
    }
    // pruning
    assert(options_.max_active_tokens > 1);
    scores_.reserve(options_.max_active_tokens * 3);  // a heuristical size
    assert(options_.beamwidth >= 0.0);
    assert(options_.beamwidth2 >= 0.0);
    // scoring
    assert(options_.lm_scale > 0.0);
    // assert(options_.word_penalty <= 0.0);  // does it have to be >0 or <0?
    DEBUG_OUT2("BeamWidth:" << options_.beamwidth << " LmScale:"
              << options_.lm_scale << " WordPenalty: " << options_.word_penalty)

    // filling Queue will be done by VisitNode
    link_store_.Clear();
    active_tokens_.Clear();
    final_token_.weight = Weight::Zero();
    final_token_.arcs = NULL;
    frame_index_ = 0;
    beam_threshold_ = options_.beamwidth;

    KALDI_LOG << "START DECODING";
    // init decoding queue with source state and initial token
    StateId source = reconet_->Start();  // start state for search
    DEBUG_OUT2("Initial state: " << source)
    Token *token = tokens_->NewToken(source);
    token->weight = Weight::One();
    link_store_.LinkToken(token, 0, fst::kNoLabel);
    // create an initial backtrack link
    active_tokens_.ThisPush(token);
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  void KaldiDecoder<Decodable, Fst>::ProcessToDoList() {
    // process active states list, create lattice links
    // new states are explored using depth first network visitor
    // to find topological state order using depth first network visitor
    DEBUG_OUT2("ProcessToDolist")
    size_t n_tokens = active_tokens_.Size();
    for(size_t ntok = 0; ntok < n_tokens; ntok++) {
      Token *token = active_tokens_.Get(ntok); // internal iterator not valid!!
      DEBUG_CMD(assert(token != NULL))
      DEBUG_OUT2("get:" << token->state << ":" << token->unique)
      if (!is_less(beam_threshold_, token->weight)) VisitNode(token);
    }
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  void KaldiDecoder<Decodable, Fst>::VisitNode(Token *token) {
    // recursive depth first search visitor, builds state priority queue
    // checks, that recogn. network has all properties required by KaldiDecoder
    StateId state = token->state;
    DEBUG_OUT2("visit:" << state << ":" << token->ilabel)
    ArcIterator aiter(*reconet_, state);
    DEBUG_CMD(assert(!(aiter.Done() && reconet_->Final(state)==Weight::Zero())))
    // check for states without outgoing links

    do { // go through all outgoing arcs
      const MyArc &arc = aiter.Value();
      DEBUG_OUT2("state " << state << " follow link:" << arc.nextstate << " "
                << arc.ilabel << ":" << arc.olabel << "/" << arc.weight)

      Token *next_token = active_tokens_.HashCheck(arc.nextstate, arc.ilabel);
      // gets token and state color from hash
      // checks also that label for arc.nextstate is always the same
      // unexplored emitting links are pushed to front of queue
      if (next_token->ilabel < 0) { // unexplored non-emitting link
        // == 0 means explored (and queued)
        VisitNode(next_token); // go recursively down, follow links
        active_tokens_.PushQueue(next_token);
        active_tokens_.ThisPush(next_token); // only to delete tokens later
      }
      next_token->AddInputArc(token, arc, &link_store_);
      // save all incoming arcs into next state
      aiter.Next();
    } while(!aiter.Done()); // for arc iterator
    return;
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  void KaldiDecoder<Decodable, Fst>::VisitNode2(Token *token) {
    // recursive depth first search visitor, builds state priority queue
    // checks, that recogn. network has all properties required by KaldiDecoder
    StateId state = token->state;
    DEBUG_OUT2("visit:" << state << ":" << token->ilabel)
    ArcIterator aiter(*reconet_, state);
    DEBUG_CMD(assert(!(aiter.Done() && reconet_->Final(state)==Weight::Zero())))
    // check for states without outgoing links

    do { // go through all outgoing arcs
      const MyArc &arc = aiter.Value();
      if (arc.ilabel > 0) { aiter.Next(); continue; } // follow only non-emitting arcs
      DEBUG_OUT2("state " << state << " follow link:" << arc.nextstate << " "
                << arc.ilabel << ":" << arc.olabel << "/" << arc.weight)

      Token *next_token = active_tokens_.HashCheck(arc.nextstate, arc.ilabel);
      // gets token and state color from hash
      // checks also that label for arc.nextstate is always the same
      // unexplored emitting links are pushed to front of queue
      if (next_token->ilabel < 0) { // unexplored non-emitting link
        // == 0 means explored (and queued)
        VisitNode2(next_token); // go recursively down, follow links
        active_tokens_.PushQueue(next_token);
        active_tokens_.ThisPush(next_token); // only to delete tokens later
      }
      next_token->AddInputArc(token, arc, &link_store_);
      // save all incoming arcs into next state
      aiter.Next();
    } while(!aiter.Done()); // for arc iterator
    return;
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  void KaldiDecoder<Decodable, Fst>::ProcessTokens(bool final_frame) {
    // all active tokens from last frame are by now processed
    // now processes arcs for next frame from priority queue
    // every destination state is EITHER emitting XOR non-emitting
    // non-emitting arcs go to the queue in topological order in the same frame
    // emitting arcs go to active_tokens_(next)
    DEBUG_OUT2("ProcessTokens")
    while (!active_tokens_.QueueEmpty()) {
      Token *dest = active_tokens_.PopQueue();  // get next state from queue
      //if (!dest) continue;
      DEBUG_OUT2("pop destination: " << dest->state << "(" << dest->weight<<")")

      Weight before_w;
      Label best_label = fst::kNoLabel;
      Link *best_link = NULL;
      Link *link = dest->arcs;
      // viterbi operation for all incoming links
      while(link != NULL) {
        Token *source = link->source;
        // source cannot be on hash, hash is only for new frame
        DEBUG_OUT2("from:" << source->state << "(" << source->weight
	  << "):" << link->olabel << "/" << link->weight)

// Weight new_w=Times(source->weight,link->weight.Value()*options_.lm_scale);
// if (link->olabel > 0) {
//  new_w = Times(new_w, options_.word_penalty);  // add word penalty
//  DEBUG_OUT2("word penalty: " << options_.word_penalty) }

	Weight new_w = Times(source->weight, link->weight); // Times (log plus)
        DEBUG_OUT2(dest->state << " new: " << new_w)
	if (is_less(beam_threshold_, new_w)) {
	  DEBUG_OUT2("prune arc!")
	} else {
          before_w = dest->weight;
	  dest->weight = Plus(before_w, new_w); // always log-add (log-semiring)
          if (before_w != dest->weight) { // if the new path is better
            DEBUG_OUT2("update token")
            best_label = link->olabel; // remember best incoming arc (olabel)
            best_link = source->arcs;  // remember corresponding lattice link
          }
        }
        link = link->next;
      }
      if (best_link) {
        // process lattice links
        dest->arcs = best_link;
        best_link->refs++;       // remember new usage in ref counter
        DEBUG_OUT3("remember:"<< best_link->unique << " ("<<best_link->refs<<"x)")
        if (best_label > 0) { // creates Link for word arcs (olabel > 0)
          Link *ans = link_store_.SlowNewLink();
          //?? no time info at the moment !! ans->state = frame_index_;
          ans->source = NULL;
          ans->olabel = best_label;
          ans->weight = dest->weight;
          ans->next = best_link;  // lattice backward pointer
          ans->refs = 1;  // can it be zero if not yet assigned?
          DEBUG_CMD(ans->unique = lnumber++)
          dest->arcs = ans;  // store new link in token
        }
        if (final_frame) {
          ReachedFinalState(dest);
        } else {
          if (dest->ilabel > 0) active_tokens_.NextPush(dest);
          // all emitting targets for evaluation of acoustic models
        }
      } else {
        dest->arcs = NULL;
        if (dest->ilabel > 0) active_tokens_.NextDelete(dest);
      }    
      //token should only be really deleted, if all arcs from it have been processed
    }
    //active_tokens_.DefragmentThis();
    link_store_.FastDefragment(); // free all temporary links

    // clear old tokens from current frame
    do {
      Token *token = active_tokens_.PopNext();
      DEBUG_OUT2("kill token:" << token->state << "(" << token->weight<<")")
      active_tokens_.ThisDelete(token);
    } while (!active_tokens_.Empty());
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  inline bool KaldiDecoder<Decodable, Fst>::ReachedFinalState(Token *token) {
    // checks if a token is in a final state and computes the path weight

    Weight fw = reconet_->Final(token->state);  // get final weight of the state
    if (fw == Weight::Zero()) return false;  // not a final state?

    // yes: it is a final state: compute the new path score
    Weight w = Times(token->weight, fw); // Times (log-domain plus) final weight
    // Weight w = Times(token->weight, fw.Value() * options_.lm_scale);
    DEBUG_OUT1("final state reached: path weight:" << w)
    DEBUG_OUT3("final link:" << token->arcs->unique)

    // get best token in final state
    Weight before_w = final_token_.weight;
    final_token_.weight = Plus(final_token_.weight, w);  // update final score
    // Plus is either mininum or log-add (in case of log-semiring)
    if (final_token_.weight == before_w) return false; // lower score: quit

    // a better score was achieved: update the link of destination state
    final_token_.UpdateLink(token->arcs, &link_store_); // copy link
    // not yet save in links (LinkToken):
    // wait until all active/queued states have been seen
    return true;  // it's a final state and new token is better than old one
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  void KaldiDecoder<Decodable, Fst>::EvaluateAndPrune() {
    // evaluates and prunes all tokens on active_tokens_(next)
    // adds GMM likelihood score (based on state) to each token
    // computes best likelihood and new beamwidth for pruning

    // the priority queue is empty, so flip active tokens and take next frame
    DEBUG_OUT2("EvaluateAndPrune")
    active_tokens_.Swap();  // flip dual token lists
    if (active_tokens_.Empty()) KALDI_ERR << "no token survived!";
    // on the hash are all tokens that are on active tokens (this)

    bool limit_tokens = false;  // did we set the max_active_tokens?
    if (options_.max_active_tokens != std::numeric_limits<int32>::max()) {
      limit_tokens = true;
      scores_.clear();
      scores_.reserve(active_tokens_.Size());
    }
    // search for best token on active_tokens_ before evaluating acoustic model
    Weight best_active_score_ = Weight::Zero();
    active_tokens_.Start();
    do {
        Token *token = active_tokens_.GetNext();
        DEBUG_CMD(assert(token != NULL))
        DEBUG_OUT2("get:" << token->state << ":" << token->unique)
        active_tokens_.HashRemove(token); //some of them we still need as begin of queue for next frame

        // compute minimum score:
        best_active_score_ = Plus(best_active_score_, token->weight);
        // save in scores_ array to determine n-best tokens:
        if (limit_tokens) scores_.push_back(token->weight.Value());
    } while(!active_tokens_.Ended()); 
    DEBUG_CMD(assert(active_tokens_.NextEmpty()))
    DEBUG_CMD(active_tokens_.AssertHashEmpty())
    //!! ?? at the moment, we still have the non-emitting states on the hash
    active_tokens_.Defragment();  // memory defragmentation for next frame

    // compute new beam threshold for this frame
    beam_threshold_ = Times(best_active_score_, options_.beamwidth);
    if (limit_tokens &&
       (scores_.size() > static_cast<size_t>(options_.max_active_tokens))) {
      active_tokens_.NextReserve(floor(options_.max_active_tokens*1.001));
      // sort scores array (only considering the n-best, leaving rest unsorted)
      std::nth_element(scores_.begin(),
                       scores_.begin() + options_.max_active_tokens,
                       scores_.end());
      // read the n-th best score
      BaseFloat ans = *(scores_.begin() + options_.max_active_tokens - 1);

      // compute the tighter of the two beams
      if (is_less(ans, beam_threshold_)) beam_threshold_ = ans;
    } else {
      active_tokens_.NextReserve(scores_.size());
    }
    DEBUG_OUT1("FRAME:" << frame_index_ << " before pdf "
               << active_tokens_.Size() << " active tokens; best score:"
               << best_active_score_ << " pruning threshold:"
               << beam_threshold_ )
    active_tokens_.QueueReset();  // the emitting states remain still marked!
    DEBUG_CMD(assert(active_tokens_.QueueEmpty()))

    // evaluate acoustic models and again find best score on active_tokens_
    best_active_score_ = Weight::Zero();
    KALDI_ASSERT(NULL != p_decodable_);
    do {
      // go through all active tokens, write compact new list
      Token *token = active_tokens_.PopNext();

      if (!is_less(beam_threshold_, token->weight)) {
        DEBUG_OUT2("evaluate state " << token->state << " (" << token->weight
                   << ") : " << token->ilabel)
        DEBUG_CMD(assert(token->ilabel > 0))
        BaseFloat score 
          = -p_decodable_->LogLikelihood(frame_index_, token->ilabel);
        // add negative loglikelihood to previous token score
        token->weight = Times(token->weight, score);
        DEBUG_OUT2("new: " << token->weight)
        // compute minimum score
        best_active_score_ = Plus(best_active_score_, token->weight);
        //token->state is already marked on color_
        active_tokens_.NextPush(token);  // write new, compact token list
      } else {  // prune token
        DEBUG_OUT2("prune token:" << token->state << "(" << token->weight<<")" << ":" << token->unique)
        active_tokens_.ThisDelete(token);
      }
    } while (!active_tokens_.Empty());
    active_tokens_.SwapMembers();
    DEBUG_CMD(assert(active_tokens_.NextEmpty()))
    active_tokens_.NextReserve(
      std::min(scores_.size(), size_t(floor(options_.max_active_tokens*1.001))));

    // compute new beam after evaluation of acoustic models
    beam_threshold_ = Times(best_active_score_, options_.beamwidth2);
    DEBUG_OUT1("FRAME:" << frame_index_ 
               << " after pdf " << active_tokens_.Size()
               << ", " << active_tokens_.QueueSize()
               << " best active score:" << best_active_score_
               << " pruning threshold:" << beam_threshold_)
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  void KaldiDecoder<Decodable, Fst>::FinalizeDecoding() {
    // processes todo-list, processes priority queue and then
    // forwards tokens to final states, back-tracks the word links
    // builds the output FST and cleans-up the memory

    final_token_.arcs = NULL;
    final_token_.weight = Weight::Zero();
    // in case no final state is reached, take the best token in the last frame
    Token best_token;  // best active token in last frame
    best_token.arcs = NULL;
    best_token.weight = Weight::Zero();

    // Analog to ProcessToDoList
    // to find non-emitting state order using depth first network visitor
    DEBUG_OUT1("Final ProcessToDoList:")
    size_t n_tokens = active_tokens_.Size();
    for(size_t ntok = 0; ntok < n_tokens; ntok++) {
      Token *token = active_tokens_.Get(ntok); // internal iterator not valid!!
      DEBUG_CMD(assert(token != NULL))
      if (!is_less(beam_threshold_, token->weight)) {
        ReachedFinalState(token);  // stores as final_token_
        VisitNode2(token); // follows only non-emitting links

        // compute best active token in last frame
        Weight before_w = best_token.weight;
        best_token.weight = Plus(best_token.weight, token->weight);
        // remember new best score; always do the Plus
        if (best_token.weight != before_w) {  // new path from s better than old
          best_token.UpdateLink(token->arcs, &link_store_);
          // copy back-track pointer from source
        }
      }
    }

    ProcessTokens(true);

    DEBUG_OUT1("decoding finished!")
    DEBUG_OUT1("FRAME:" << frame_index_ << " " << active_tokens_.NextSize())
    assert(active_tokens_.NextEmpty());
    active_tokens_.AssertHashEmpty();
    active_tokens_.QueueReset();
    // active_tokens_.AssertQueueEmpty();

    // either take final_token or best_token if no final state was reached
    if (final_token_.weight == Weight::Zero()) {  // take only best_token
      assert(final_token_.arcs == NULL &&
             best_token.weight != Weight::Zero() &&
             best_token.arcs != NULL);
      final_token_.weight = best_token.weight;
      final_token_.arcs = best_token.arcs;
      KALDI_WARN << "Warning: no final state reached!";
    } else {  // final token exists
      link_store_.SlowDelete(best_token.arcs);  // delete other best path
    }

    // build output FST
    output_arcs_ = new fst::VectorFst<MyArc>;
    // output_arcs_->SetOutputSymbols(reconet_->OutputSymbols());
    // in case we'd have symbol tables

    // back-track word links in best path
    assert(final_token_.arcs != NULL);
    StateId wlstate = output_arcs_->AddState();
    output_arcs_->SetFinal(wlstate, final_token_.weight);
    DEBUG_OUT1("set final state of Links:" << wlstate << " total score:"
              << final_token_.weight)
    Link *wl = final_token_.arcs;
    while (wl != NULL && (wl->olabel >= 0)) {
      StateId new_wlstate = output_arcs_->AddState();
      // add corresponding arc
      BaseFloat arc_weight = (wl->next != NULL) ?
        wl->weight.Value() - wl->next->weight.Value() : wl->weight.Value();
      // difference between scores at word labels
      output_arcs_->AddArc(new_wlstate,
                           MyArc(0, wl->olabel, arc_weight, wlstate));
      //!!??               MyArc(wl->state, wl->olabel, arc_weight, wlstate));
      std::string word = "";
      // if (reconet_->OutputSymbols())
        // word = reconet_->OutputSymbols()->Find(wl->olabel);
      DEBUG_OUT1(new_wlstate << "->" << wlstate << " " << wl->source << ":"
               << wl->olabel << "( " << word << " )/"  << wl->weight)
      wlstate = new_wlstate;
      wl = wl->next;
    }
    output_arcs_->SetStart(wlstate);  // create an initial state
    link_store_.SlowDelete(final_token_.arcs);  // clean-up final path
    DEBUG_OUT1("FRAME:" << frame_index_ << " " << active_tokens_.NextSize())

    // memory clean-up
    link_store_.Clear();
    tokens_->Clear();
    tokens_next_->Clear();
    active_tokens_.Clear();
  }
};  // namespace kaldi

#endif
