// decoder/deterministic-composer.h

// Copyright 2011 Gilles Boulianne

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

#ifndef KALDI_DECODER_DETERMINISTIC_COMPOSE_H_
#define KALDI_DECODER_DETERMINISTIC_COMPOSE_H_

#include "fst/fstlib.h"
#include "fstext/deterministic-fst.h"

namespace kaldi {

class DeterministicComposer {
 public:
  friend class ArcIterator;  // needs access to private state
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;  
  typedef std::pair<StateId,StateId> StatePair;
  
  DeterministicComposer(const fst::Fst<fst::StdArc> &fst,
                        const fst::DeterministicOnDemandFst<fst::StdArc> &gdiff):
        fst1_(fst), fst2_(gdiff) {
  }
  
  // interface for large lm decoding
  StateId Start() {
    // start state of composed fst1_ and fst2_
    StateId s1 = fst1_.Start();
    StateId s2 = fst2_.Start();
    if (s1 == fst::kNoStateId || s2 == fst::kNoStateId) return fst::kNoStateId;
    if (composedState_.size()==0) AddComposedState(s1,s2);	
    return 0;
  }
  
  Weight Final(StateId s) {
    StatePair sp = composedState_[s];
    Weight w = Times(fst1_.Final(sp.first),fst2_.Final(sp.second));
    return w;
  }
  
  class ArcIterator {
   public:
    ArcIterator(DeterministicComposer* dc, StateId s) : dc_(dc) {
      assert(dc_!=NULL);
      StatePair sp = dc_->composedState_[s];
      s1_ = sp.first;
      s2_ = sp.second;
      aiter_ = new fst::ArcIterator<fst::Fst<Arc> >(dc_->fst1_,s1_);
      done_ = computeArcValue();
    }
    
    ~ArcIterator() {if (aiter_) delete aiter_;}
    
    bool Done() {return done_;}
    
    Arc Value() {return value_;}
    
    void Next() {
      // position to next arc in fst1_
      aiter_->Next();
      done_ = computeArcValue();
    }
    
  private:
    DeterministicComposer* dc_;
    StateId s1_, s2_;
    Arc value_;
    bool done_;
    fst::ArcIterator<fst::Fst<Arc> >* aiter_;
    
    bool computeArcValue() {
      // advance through arcs of fst1_ until a composed arc is found
      // cerr << "computeArcValue currently at state ("<<s1_<<","<<s2_<<")"<<endl;
      while (!aiter_->Done()) {
        Arc arc2, arc1 = aiter_->Value();
        // cerr << "   going through arc1 ["<<arc1.ilabel<<","<<arc1.olabel<<"]"<<endl;
        if (arc1.olabel==0) {
          // output epsilon in fst1_ : don't move in fst2_
          value_ = Arc(arc1.ilabel, 0, arc1.weight, 
                       dc_->AddComposedState(arc1.nextstate,s2_));
          return false;  // got a match, so we're still not done
        }
        if (dc_->fst2_.GetArc(s2_, arc1.olabel, &arc2)) {
          value_ = Arc(arc1.ilabel, arc2.olabel, Times(arc1.weight,arc2.weight),
                       dc_->AddComposedState(arc1.nextstate, arc2.nextstate));
          return false;  // got a match, so we're still not done
        }
        aiter_->Next();
      }
      // went through all arcs, we're done
      return true;
    }    
  }; // end ArcIterator class    

private:

  // State pair management
  class StatePairEqual {
  public:
    bool operator()(const StatePair &x, const StatePair &y) const {
      return x.first == y.first && x.second == y.second;
    }
  };
  
  class StatePairKey{
  public:
    size_t operator()(const StatePair &x) const {
      return static_cast<size_t>(x.first*kPrime+x.second);
    }
  private:
    static const int kPrime = 7853;
  };

  // component FSTs
  const fst::Fst<fst::StdArc> &fst1_;
  const fst::DeterministicOnDemandFst<fst::StdArc> &fst2_;
  // resulting FST
  typedef unordered_map<StatePair, StateId, StatePairKey, StatePairEqual> StateMap;  // map to composed StateId
  StateMap state_map_;                        // map from state in fst1_ and fst2_ to composed state
  std::vector<StatePair> composedState_;      // indexed by composed StateId 
  
  // add composed state to internal data (create if new)
  StateId AddComposedState(StateId s1, StateId s2) {
    StatePair sp = make_pair(s1, s2);
    StateMap::iterator mit = state_map_.find(sp);
    StateId cs;
    if (mit == state_map_.end()) {
      // new, add it
      cs = composedState_.size();
      composedState_.push_back(sp);
      state_map_[sp] = cs;
      //cerr << "Adding composed state ("<<s1<<","<<s2<<") = "<<cs<<endl;
    } else {
      cs = (*mit).second;
    }
    return cs;
  }
  
}; // end DeterministicComposer class definition

  
} // end namespace kaldi.

#endif
