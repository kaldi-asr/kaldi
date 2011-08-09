// decoder/kaldi-decoder-inl.h

// Copyright 2011  Mirko Hannemann;  Lukas Burget

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
    tokens_ = new TokenStore(&wl_store_);
    tokens_next_ = new TokenStore(&wl_store_);
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
    // initially, insert the first token to the active tokens list
    InitDecoding(fst, decodable);
    // go over all feature frames and decode
    frame_index_ = -1;

    do {
      frame_index_++;
      DEBUG_OUT1("==== FRAME " << frame_index_ << " =====")

      // all active tokens from last frame are by now processed
      // tokens for next frame are on active_tokens_(next) or on priority queue
      // now process the priority queue, until it's empty
      ProcessNonEmitting();
      // takes a token from the queue and forwards the token to all links
      // (operates on active_tokens_(next))
      // all non-emitting targets go again onto the queue,
      // all emitting targets go to active_tokens_(next)
      // (for words we add a token to the word link record)

      // the priority queue is empty, so flip active tokens and take next frame
      // evaluate all tokens on active_tokens_(this)
      EvaluateAndPrune();
      // computes GMM likelihood (based on state) and adds score to each token
      // also computes the best likelihood and the new beam width for pruning

      // processes active_tokens(this) and forward tokens to active_tokens(next)
      ProcessEmitting();
      // all tokens in active_tokens(this) within beam are forwarded to all arcs
      // new nodes are explored using depth first network visitor
      // puts tokens either to active_tokens_(next)
      // or to non-emitting queue in topological state order
    } while (!(p_decodable_->IsLastFrame(frame_index_)));

    DEBUG_OUT1("==== FINISH FRAME " << frame_index_ << " =====")
    // FinalizeDecoding also processes the active states and the priority queue
    FinalizeDecoding();
    // forwards to final state, backtracking, build output FST, memory clean-up
    double diff = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    KALDI_LOG << "TIME:" << std::setprecision(4)<<diff<<std::setprecision(6);
    return output_arcs_;  // or maybe frame_index_ ? number of decoded frames?
  }
  // pruning: remove all tokens outside the beam from active_tokens_(this)
  //  options_.beamwidth: parameter, on which we compute beam_threshold_
  //  beam_threshold_: determined within EvaluateAndPrune
  //  options_.max_active_tokens: maximum number of active states

  // pruning with beam_threshold_ and max_active_tokens_ is applied:
  // a) main loop before EvaluateAndPrune
  // b) main loop of ProcessNonEmitting, ProcessEmitting
  // c) always when PassTokenThroughArc
  // (in ProcessNonEmitting/ProcessEmitting and PassTokenThroughArc)


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
    wl_store_.Clear();
    active_tokens_.Clear();
    final_token_.weight = Weight::Zero();
    final_token_.previous = NULL;
    frame_index_ = 0;
    beam_threshold_ = options_.beamwidth;

    DEBUG_OUT("start decoding")
    // init decoding queues with source state and initial token
    StateId source = reconet_->Start();  // start state for search
    DEBUG_OUT2("Push initial state: " << source)
    active_tokens_.ResizeHash(source);
    Token *token = active_tokens_.HashLookup(source);
    token->weight = Weight::One();
    // token->olabel = fst::kNoLabel;
    wl_store_.LinkToken(token, 0);  // create an initial backtrack wordlink

    // explore non-emitting states
    if (VisitNode(source) >= kBlack) {
      active_tokens_.Enqueue(token->state);
      DEBUG_OUT2("to queue")
    } else {
      active_tokens_.NextPush(token);
      DEBUG_OUT2("to active tokens")
    }
    // allocate memory with highest index of all visited states
    active_tokens_.ResizeHash(active_tokens_.ColorSize() - 1);
    DEBUG_OUT1("FRAME:" << frame_index_ << " " << active_tokens_.NextSize()
               << ", " << active_tokens_.QueueSize())
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  int KaldiDecoder<Decodable, Fst>::VisitNode(StateId s) {
    // recursive depth first search visitor

    // visitor status for each state in Queue:
    // color kWhite:  not yet discovered              not in map or -3
    // color kGray:   discovered & unfinished         == -2
    // color kBrown:  finished visit, emitting        == -1
    // color kBlack:  finished visit, non-emitting    a topological number

    //int color = active_tokens_.GetKey(s);
    //if (color > kWhite) return color;  // state was already explored
    DEBUG_CMD(assert(active_tokens_.GetKey(s) == kWhite))
    DEBUG_OUT2("visit node:" << s)
        
    // an unexplored state: go through recursively through all arcs
    for (ArcIterator aiter(*reconet_, s); !aiter.Done(); aiter.Next()) {
      // idea: we could use an EpsilonArcFilter (or any) in the ArcIterator
      const MyArc &arc = aiter.Value();
      DEBUG_OUT2("node " << s << " follow link: " << arc.nextstate << " "
                << arc.ilabel << ":" << arc.olabel << "/" << arc.weight)

      if (arc.ilabel > 0) {  // an emitting link
        active_tokens_.Brown(s);  // mark as kBrown (emitting)
        // emitting states are only marked - they are not needed in the queue
        DEBUG_OUT2("touched:" << s << "; toporder: " << active_tokens_.GetKey(s))
        return kBrown;  // means: explored as emitting
        // if we assume, that all further links are emitting,
        // we don't need to go through all links
      } else {  // a non-emitting link
        if (active_tokens_.GetKey(arc.nextstate) == kWhite)
          VisitNode(arc.nextstate);
        // go recursively down, follow links
      }
    }
    // assign topological number (color kBlack)
    active_tokens_.Finish(s);  // put non-emitting states to the queue
    DEBUG_OUT2("finished:" << s << "; toporder: " << active_tokens_.GetKey(s))
    return kBlack;  // means: explored as non-emitting
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  int KaldiDecoder<Decodable, Fst>::VerifyNode(StateId s) {
    // recursive depth first search visitor
    // checks, that recogn. network has all properties required by KaldiDecoder

    int color = active_tokens_.GetKey(s);
    DEBUG_OUT2("visit node:" << s << ":" << color)
    if (color > kWhite) {
      // if there is a loop in the FST that was reached by a non-emitting link
      if (color == kGray) KALDI_ERR << "transducer contains epsilon loop!";
      // root reaches a state, that was already visited by other roots
      return color;
    }
    active_tokens_.Touch(s);  // insert color = kGray
    bool emitting_links = false, nonemitting_links = false;
    Label mylabel = 0;

    // an unexplored state: go through recursively through all arcs
    for (ArcIterator aiter(*reconet_, s); !aiter.Done(); aiter.Next()) {
      // idea: we could use an EpsilonArcFilter (or any) in the ArcIterator
      const MyArc &arc = aiter.Value();
      DEBUG_OUT2("node " << s << " follow link: " << arc.nextstate << " "
                << arc.ilabel << ":" << arc.olabel << "/" << arc.weight)

      if (arc.ilabel > 0) {  // emitting link
        if (emitting_links && (arc.ilabel != mylabel)) {
          KALDI_ERR << "we assume all emitting links have the same input label";
        }
        mylabel = arc.ilabel;
        emitting_links = true;
        // do not break, go through all links,
        // assuming that all are emitting, and check their labels

      } else {  // non-emitting link

        if (emitting_links)
          KALDI_ERR << "we assume non-emitting links come before emitting ones";
        nonemitting_links = true;
        VerifyNode(arc.nextstate);  // go recursively down, follow links
      }
    }

    bool is_final = (reconet_->Final(s) != Weight::Zero());
    if ((!emitting_links) && (!nonemitting_links) && (!is_final))
      KALDI_ERR << "dead node: no outgoing links";
    // assign color and topological number for explored states
    if (nonemitting_links || is_final) {
      active_tokens_.Finish(s);
      DEBUG_OUT2("finished:" << s << "; toporder: " << active_tokens_.GetKey(s))
      return kBlack;
    } else {
      active_tokens_.Brown(s);
      DEBUG_OUT2("finished:" << s)
      return kBrown;
    }
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  void KaldiDecoder<Decodable, Fst>::ProcessNonEmitting() {
    // processes the priority queue, until it's empty
    // takes a token from the queue and forwards the token to all links
    // all non-emitting targets go again onto the queue,
    // all emitting targets go to active_tokens_(next)
    // for words we add a token to the word link record

    // main-loop: until queue empty
    while (!active_tokens_.QueueEmpty()) {
      Token *token = active_tokens_.PopQueue();  // get next state from queue
      if (!token) continue;
      DEBUG_OUT2("pop node: " << token->state << " weight:" << token->weight)

      if (!is_less(beam_threshold_, token->weight)) {  // if token within beam
        wl_store_.LinkToken(token, frame_index_);
        // creates a word link for each word label

        // go through all arcs of the state s
        for (ArcIterator aiter(*reconet_, token->state);
             !aiter.Done(); aiter.Next()) {
          const MyArc &arc = aiter.Value();
          DEBUG_OUT2("link: " << arc.nextstate << " " << arc.ilabel << ":"
                    << arc.olabel << "/" << arc.weight)

          DEBUG_CMD(assert(arc.ilabel <= 0))
          // compute new score and if better than old, remember token:
          if (!PassTokenThroughArc(token, arc)) {
          // inside PassTokenThroughArc, we dispatch to queue or active tokens
            // if (active_tokens_.GetKey(arc.nextstate) == kBrown)
            //  active_tokens_.Untouch(arc.nextstate);  // pruned: reset color
          }
        } // for all links
      } // if within beamwidth
      active_tokens_.NextDelete(token);
    } // while not empty
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  inline bool KaldiDecoder<Decodable, Fst>::PassTokenThroughArc(
                                              Token *token, const MyArc &arc) {
    // computes new score, adds penalties
    // if the new score for arc.nextstate is better: update the word link
    // dispatches token if arc.nextstate should be further followed
    // puts tokens either to active tokens or to non-emitting queue
    // and finds topological state order using depth first network visitor

    // add arc weight and penalty
    // Weight new_w = Times(token->weight, arc.weight.Value()*options_.lm_scale);
    Weight new_w = Times(token->weight, arc.weight);
    // score of source token Times (log plus) arc weight
    // if (arc.olabel > 0) {
    //  new_w = Times(new_w, options_.word_penalty);  // add word penalty
    //  DEBUG_OUT2("word penalty: " << options_.word_penalty)
    //}
    DEBUG_OUT2(arc.nextstate << " new: " << new_w)

    // arc pruning
    if (is_less(beam_threshold_, new_w)) {  // not within beam
      DEBUG_OUT2("prune! (arc)")
      return false;  // don't follow the link
    }

    // get token from destination state (if necessary create new one)
    Token *next_tok = active_tokens_.HashLookup(arc.nextstate);
    // be careful, that this pointer stays valid
    Weight before_w = next_tok->weight;
    DEBUG_OUT2(" old: " << before_w)

    if (before_w == Weight::Zero()) {  // new token was created
      int color = active_tokens_.GetKey(arc.nextstate);

      if (color == kWhite) {  // unexplored state?
        // the assumption is, that in ProcessEmitting all states are kWhite
        color = VisitNode(arc.nextstate);
      }

      // dispatch new token to the right queue
      if (color < kBlack) {  // next state emitting
        active_tokens_.NextPush(next_tok);
        DEBUG_OUT2("state " << arc.nextstate << " to active states")
      } else {  // emitting state:
        // it is a non-emitting state:
        active_tokens_.Enqueue(arc.nextstate);
        DEBUG_OUT2("state " << arc.nextstate << " to queue")
      }
      next_tok->weight = new_w;  // assign weight to new token

    } else {  // token already exists

      next_tok->weight = Plus(before_w, new_w);  // update score for destination
      // Plus is either mininum or log-add
      // we should always do the Plus (log-add in case of log-semiring)
      if (next_tok->weight == before_w) return false;
      // don't follow the link if the new path has a lower score
    }

    // a better score was achieved: update the wordlink of destination state
    DEBUG_OUT2("update token")
    next_tok->olabel = arc.olabel;  // remember best incoming arc (so far label)
    next_tok->UpdateWordLink(token, &wl_store_);  // copy wordlink
    // not yet save in wordlinks (LinkToken):
    // wait until all active/queued states have been seen

    return true;  // yes, follow the link
    // we need to return that, so that we put the token to the right queue
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  inline bool KaldiDecoder<Decodable, Fst>::ReachedFinalState(Token *token) {
    // checks if a token is in a final state and computes the path weight
    // returns true, if it's a final state and new token is better than old one

    Weight fw = reconet_->Final(token->state);  // get final weight of the state
    if (fw == Weight::Zero()) return false;  // final state? (if finalweight !=0)

    // yes: it is a final state: compute the new path score
    // Weight w = Times(token->weight, fw.Value() * options_.lm_scale);
    Weight w = Times(token->weight, fw);
    // path score Times (plus in log-domain) final weight
    DEBUG_OUT2("final state reached: " << token->state << " path weight:" << w)
    DEBUG_OUT3("final WordLink:" << token->previous->unique)

    // get best token in final state
    Weight before_w = final_token_.weight;
    final_token_.weight = Plus(final_token_.weight, w);  // update final score
    // Plus is either mininum or log-add
    // we should always do the Plus (log-add in case of log-semiring)

    if (final_token_.weight != before_w) {
      // a better score was achieved: update the wordlink of destination state
      final_token_.UpdateWordLink(token, &wl_store_);
      // not yet save in wordlinks (LinkToken):
      // wait until all active/queued states have been seen
      return true;
    } else {
      return false;
    }
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  void KaldiDecoder<Decodable, Fst>::EvaluateAndPrune() {
    // computes GMM likelihood (based on state) and adds score to each token
    // also computes the best likelihood and the new beam width for pruning

    // the priority queue is empty, so flip active tokens and take next frame
    DEBUG_OUT2("EvaluateAndPrune")
    active_tokens_.Swap();  // flip dual token lists
    if (active_tokens_.Empty()) KALDI_ERR << "no token survived!";

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
        active_tokens_.HashRemove(token);
        // active_tokens_.Untouch(token->state);  // clean up state color

        // compute minimum score:
        best_active_score_ = Plus(best_active_score_, token->weight);
        // save in scores_ array to determine n-best tokens:
        if (limit_tokens) scores_.push_back(token->weight.Value());
    } while(!active_tokens_.Ended()); 
    DEBUG_CMD(active_tokens_.AssertNextEmpty())
    //active_tokens_.Defragment();  // memory defragmentation for next frame

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
    active_tokens_.QueueReset();  // the emitting states remain still kBrown!
    //DEBUG_CMD(active_tokens_.AssertQueueEmpty())

    // evaluate acoustic models and again find best score on active_tokens_
    best_active_score_ = Weight::Zero();
    KALDI_ASSERT(NULL != p_decodable_);
    do {
      // go through all active tokens, write compact new list
      Token *token = active_tokens_.PopNext();

      if (!is_less(beam_threshold_, token->weight)) {
        ArcIterator aiter(*reconet_, token->state);
        const MyArc &arc = aiter.Value();
        // get acoustic model index from first emitting arc -> arc.ilabel
        Label mLabel = arc.ilabel;
        DEBUG_OUT2("evaluate state " << token->state << " (" << token->weight
                   << ") : " << mLabel)
        DEBUG_CMD(assert(mLabel > 0))
        BaseFloat score = -p_decodable_->LogLikelihood(frame_index_, mLabel);
        // add negative loglikelihood to previous token score
        token->weight = Times( token->weight, score);
        DEBUG_OUT2("new: " << token->weight)
        // compute minimum score
        best_active_score_ = Plus(best_active_score_, token->weight);
        active_tokens_.NextPush(token);  // write new, compact token list
      } else {  // prune token
        DEBUG_OUT2("kill token:" << token->state << "(" << token->weight<<")")
        active_tokens_.ThisDelete(token);
        // active_tokens_.KillThis();  //remove from members_ and hash_ and Delete
      }
    } while (!active_tokens_.Empty());

    active_tokens_.SwapMembers();
    DEBUG_CMD(active_tokens_.AssertNextEmpty())
    active_tokens_.Defragment();  // memory defragmentation for next frame
    active_tokens_.NextReserve(
      std::min(scores_.size(), size_t(floor(options_.max_active_tokens*1.001))));
    // compute new beam after evaluation of acoustic models
    beam_threshold_ = Times(best_active_score_, options_.beamwidth2);
    DEBUG_OUT1("FRAME:" << frame_index_ << " after pdf "
               << active_tokens_.Size() << " best active score:"
               << best_active_score_ << " pruning threshold:"
               << beam_threshold_)
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  void KaldiDecoder<Decodable, Fst>::ProcessEmitting() {
    // processes active_tokens_(this) and forwards tokens to active_tokens(next)
    // all tokens in active_tokens_(this) within beam are forwarded to all arcs:
    // new nodes are explored using depth first network visitor
    // emitting states go to active_tokens_(next)
    // non-emitting states go to the queue in topological state order

    if (active_tokens_.Empty()) KALDI_ERR << "no token survived!";
    do {  // go through all active tokens
      Token *token = active_tokens_.PopNext();
      DEBUG_OUT2("process state " << token->state << " weight:" << token->weight)

      if (!is_less(beam_threshold_, token->weight)) {
        wl_store_.LinkToken(token, frame_index_);
        // create wordlink for each word label

        for (ArcIterator aiter(*reconet_, token->state);
             !aiter.Done(); aiter.Next()) {
          const MyArc &arc = aiter.Value();
          // look at first emitting arc -> arc.ilabel
          DEBUG_OUT2("link: " << arc.nextstate << " " << arc.ilabel << ":"
                    << arc.olabel << "/" << arc.weight)

          DEBUG_CMD(assert(arc.ilabel > 0))
          active_tokens_.ResizeHash(arc.nextstate);  // AllocateLists
          // memory allocation for data structures that index on states

          // compute new score and if better than old, remember token info:
          PassTokenThroughArc(token, arc);
          // emitting & non-emitting states are dispatched by ProcessToDoList
        } // for all links
      }
      active_tokens_.ThisDelete(token);
    } while (!active_tokens_.Empty());

    // allocate memory with highest index of all visited states
    active_tokens_.ResizeHash(active_tokens_.ColorSize() - 1);
    DEBUG_OUT1("FRAME:" << frame_index_ << " " << active_tokens_.NextSize()
               << ", " << active_tokens_.QueueSize())
  }

  //***************************************************************************
  //***************************************************************************
  template<class Decodable, class Fst>
  void KaldiDecoder<Decodable, Fst>::FinalizeDecoding() {
    // processes todo-list, processes priority queue and then
    // forwards tokens to final states, back-tracks the word links
    // builds the output FST and cleans-up the memory

    final_token_.previous = NULL;
    final_token_.weight = Weight::Zero();
    // in case no final state is reached, take the best token in the last frame
    Token best_token;  // best active token in last frame
    best_token.previous = NULL;
    best_token.weight = Weight::Zero();

    // this part is analog to ProcessEmitting,
    // but checks for final states and computes best active token
    // it processes the priority queue, until it's empty
    // takes a token from the queue and forwards the token to all links
    // all non-emitting targets go again onto the queue,
    // all emitting targets are no longer needed
    // for words we add a token to the word link record

    // main-loop: until queue empty
    while (!active_tokens_.QueueEmpty()) {
      Token *token = active_tokens_.PopQueue();  // get next state from queue
      if (!token) continue;
      DEBUG_OUT2("pop node: " << token->state << " weight:" << token->weight)

      // should we prune with beam_threshold_ here?
      //// if (!is_less(beam_threshold_, token->weight)) {  // if token within beam
      wl_store_.LinkToken(token, frame_index_);
      // creates a wordlink for each word label
      ReachedFinalState(token);  // store as final_token_ if final state and better

      // we go on, because even a final state can have outgoing links (non-emit)
      // go through all arcs of the state s
      for (ArcIterator aiter(*reconet_, token->state);
           !aiter.Done(); aiter.Next()) {
        const MyArc &arc = aiter.Value();
        DEBUG_OUT2("link: " << arc.nextstate << " " << arc.ilabel << ":"
                  << arc.olabel << "/" << arc.weight)

        // compute new score and if better than old, remember token:
        if (active_tokens_.GetKey(arc.nextstate) >= kBlack) {
          PassTokenThroughArc(token, arc);  // propagate non-emitting or final
        } else {  // delete emitting
          // active_tokens_.Untouch(arc.nextstate);
        }
      } // for all links
      ////} // if within beamwidth

      // compute best active token in last frame
      Weight before_w = best_token.weight;
      best_token.weight = Plus(best_token.weight, token->weight);
      // remember new best score; always do the Plus
      if (best_token.weight != before_w) {  // new path from s better than old one
        best_token.UpdateWordLink(token, &wl_store_);
        // copy back-track pointer from source
      }
      DEBUG_OUT2("clear state " << token->state << " weight:" << token->weight)
      active_tokens_.NextDelete(token);  // delete token
    } // while not empty
    // end ProcessEmitting
    active_tokens_.QueueReset();  // the emitting states are still kBrown

    // analog to ProcessEmitting, but only check emitting nodes for final states
    active_tokens_.Swap();
    do {
      Token *token = active_tokens_.PopNext();  // remove state
      active_tokens_.HashRemove(token);
      DEBUG_CMD(assert(active_tokens_.GetKey(token->state) < kBlack))
      DEBUG_OUT2("pop queue: " << token->state << " weight:" << token->weight)

      // compute the best token so far
      Weight before_w = best_token.weight;
      // compute new best score; always do Plus (log-add)
      best_token.weight = Plus(best_token.weight, token->weight);
      if (best_token.weight != before_w) {
        // if new path from s better than old one: copy word link
        best_token.UpdateWordLink(token, &wl_store_);
      }
      // evaluates final state weight and if better stores as final_token_
      ReachedFinalState(token);
      DEBUG_OUT2("clear active state "<<token->state<<" weight:"<<token->weight)
      // active_tokens_.Untouch(token->state);
      active_tokens_.ThisDelete(token);
    } while (!active_tokens_.Empty());

    DEBUG_OUT1("decoding finished!")
    DEBUG_OUT1("FRAME:" << frame_index_ << " " << active_tokens_.NextSize())
    active_tokens_.AssertNextEmpty();
    // active_tokens_.AssertQueueEmpty();

    // either take final_token or best_token if no final state was reached
    if (final_token_.weight == Weight::Zero()) {  // take only best_token
      assert(final_token_.previous == NULL &&
             best_token.weight != Weight::Zero() &&
             best_token.previous != NULL);
      final_token_.weight = best_token.weight;
      final_token_.previous = best_token.previous;
      KALDI_WARN << "Warning: no final state reached!";
    } else {  // final token exists
      wl_store_.Delete(best_token.previous);  // delete other best path
    }

    // build output FST
    output_arcs_ = new fst::VectorFst<MyArc>;
    // output_arcs_->SetOutputSymbols(reconet_->OutputSymbols());
    // in case we'd have symbol tables

    // back-track word links in best path
    assert(final_token_.previous != NULL);
    StateId wlstate = output_arcs_->AddState();
    output_arcs_->SetFinal(wlstate, final_token_.weight);
    DEBUG_OUT1("set final state of WordLinks:" << wlstate << " total score:"
              << final_token_.weight)
    WordLink *wl = final_token_.previous;
    while (wl != NULL && (wl->olabel >= 0)) {
      StateId new_wlstate = output_arcs_->AddState();
      // add corresponding arc
      BaseFloat arc_weight = (wl->previous != NULL) ?
        wl->weight.Value() - wl->previous->weight.Value() : wl->weight.Value();
      // difference between scores at word labels
      output_arcs_->AddArc(new_wlstate,
                           MyArc(wl->state, wl->olabel, arc_weight, wlstate));
      std::string word = "";
      // if (reconet_->OutputSymbols())
        // word = reconet_->OutputSymbols()->Find(wl->olabel);
      DEBUG_OUT1(new_wlstate << "->" << wlstate << " " << wl->state << ":"
               << wl->olabel << "( " << word << " )/"  << wl->weight)
      wlstate = new_wlstate;
      wl = wl->previous;
    }
    output_arcs_->SetStart(wlstate);  // create an initial state
    wl_store_.Delete(final_token_.previous);  // clean-up final path
    DEBUG_OUT1("FRAME:" << frame_index_ << " " << active_tokens_.NextSize())

    // memory clean-up
    wl_store_.Clear();
    tokens_->Clear();
    tokens_next_->Clear();
    active_tokens_.Clear();
  }
};  // namespace kaldi

#endif
