// fstext/determinize-star-inl.h

// Copyright 2009-2011  Microsoft Corporation;  Jan Silovsky

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

#ifndef KALDI_FSTEXT_DETERMINIZE_STAR_INL_H_
#define KALDI_FSTEXT_DETERMINIZE_STAR_INL_H_
// Do not include this file directly.  It is included by determinize-star.h

#include "base/kaldi-error.h"

#ifdef _MSC_VER
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
using std::tr1::unordered_map;
#include <vector>
#include <climits>

namespace fst {

// This class maps back and forth from/to integer id's to sequences of strings.
// used in determinization algorithm.

template<class Label, class StringId> class StringRepository {
  // Label and StringId are both integer types, possibly the same.
  // This is a utility that maps back and forth between a vector<Label> and StringId
  // representation of sequences of Labels.  It is to save memory, and to save compute.
  // We treat sequences of length zero and one separately, for efficiency.

 public:
  class VectorKey { // Hash function object.
   public:
    size_t operator()(const vector<Label> *vec) const {
      assert(vec != NULL);
      size_t hash = 0, factor = 1;
      for (typename vector<Label>::const_iterator it = vec->begin(); it != vec->end(); it++)
        hash += factor*(*it); factor*=103333;  // just an arbitrary prime number.
      return hash;
    }
  };
  class VectorEqual {  // Equality-operator function object.
   public:
    size_t operator()(const vector<Label> *vec1, const vector<Label> *vec2) const {
      return (*vec1 == *vec2);
    }
  };

  typedef unordered_map<const vector<Label>*, StringId, VectorKey, VectorEqual> MapType;

  StringId IdOfEmpty() { return no_symbol; }

  StringId IdOfLabel(Label l) {
    if (l>= 0 && l <= (Label) single_symbol_range) {
      return l + single_symbol_start;
    } else {
      // l is out of the allowed range so we have to treat it as a sequence of length one.  Should be v. rare.
      vector<Label> v; v.push_back(l);
      return IdOfSeqInternal(v);
    }
  }

  StringId IdOfSeq(const vector<Label> &v) {  // also works for sizes 0 and 1.
    size_t sz = v.size();
    if (sz == 0) return no_symbol;
    else if (v.size() == 1) return IdOfLabel(v[0]);
    else return IdOfSeqInternal(v);
  }

  inline bool IsEmptyString(StringId id) {
    return id == no_symbol;
  }
  void SeqOfId(StringId id, vector<Label> *v) {
    if (id == no_symbol) v->clear();
    else if (id>=single_symbol_start) {
      v->resize(1); (*v)[0] = id - single_symbol_start;
    } else {
      assert(id >= string_start && id < static_cast<StringId>(vec_.size()));
      *v = *(vec_[id]);
    }
  }
  StringId RemovePrefix(StringId id, size_t prefix_len) {
    if (prefix_len == 0) return id;
    else {
      vector<Label> v;
      SeqOfId(id, &v);
      size_t sz = v.size();
      assert(sz >= prefix_len);
      vector<Label> v_noprefix(sz - prefix_len);
      for (size_t i = 0;i < sz-prefix_len;i++) v_noprefix[i] = v[i+prefix_len];
      return IdOfSeq(v_noprefix);
    }
  }

  StringRepository() {
    // The following are really just constants but don't want to complicate compilation so make them
    // class variables.  Due to the brokenness of <limits>, they can't be accessed as constants.
    string_end = (numeric_limits<StringId>::max() / 2) - 1;  // all hash values must be <= this.
    no_symbol = (numeric_limits<StringId>::max() / 2);  // reserved for empty sequence.
    single_symbol_start =  (numeric_limits<StringId>::max() / 2) + 1;
    single_symbol_range =  numeric_limits<StringId>::max() - single_symbol_start;
  }
  void Destroy() {
    for (typename vector<vector<Label>* >::iterator iter = vec_.begin(); iter != vec_.end(); ++iter)
      delete *iter;
    vector<vector<Label>* > tmp_vec;
    tmp_vec.swap(vec_);    
    MapType tmp_map;
    tmp_map.swap(map_);
  }
  ~StringRepository() {
    Destroy();
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(StringRepository);

  StringId IdOfSeqInternal(const vector<Label> &v) {
    typename MapType::iterator iter = map_.find(&v);
    if (iter != map_.end()) {
      return iter->second;
    } else {  // must add it to map.
      StringId this_id = (StringId) vec_.size();
      vector<Label> *v_new = new vector<Label> (v);
      vec_.push_back(v_new);
      map_[v_new] = this_id;
      assert(this_id < string_end);  // or we used up the labels.
      return this_id;
    }
  }

  vector<vector<Label>* > vec_;
  MapType map_;

  static const StringId string_start = (StringId) 0;  // This must not change.  It's assumed.
  StringId string_end;  // = (numeric_limits<StringId>::max() / 2) - 1; // all hash values must be <= this.
  StringId no_symbol;  // = (numeric_limits<StringId>::max() / 2); // reserved for empty sequence.
  StringId single_symbol_start;  // =  (numeric_limits<StringId>::max() / 2) + 1;
  StringId single_symbol_range;  // =  numeric_limits<StringId>::max() - single_symbol_start;
};


template<class Arc> class DeterminizerStar {
 public:
  // Output to Gallic acceptor (so the strings go on weights, and there is a 1-1 correspondence
  // between our states and the states in ofst.  If destroy == true, release memory as we go
  // (but we cannot output again).
  void Output(MutableFst<GallicArc<Arc> >  *ofst, bool destroy = true) {
    assert(determinized_);
    if (destroy) determinized_ = false;
    typedef GallicWeight<Label, Weight> ThisGallicWeight;
    typedef typename Arc::StateId StateId;
    if (destroy)
      FreeMostMemory();
    StateId nStates = static_cast<StateId>(output_arcs_.size());
    ofst->DeleteStates();
    ofst->SetStart(kNoStateId);
    if (nStates == 0) {
      return;
    }
    for (StateId s = 0;s < nStates;s++) {
      OutputStateId news = ofst->AddState();
      assert(news == s);
    }
    ofst->SetStart(0);
    // now process transitions.
    for (StateId this_state = 0; this_state < nStates; this_state++) {
      vector<TempArc> &this_vec(output_arcs_[this_state]);
      typename vector<TempArc>::const_iterator iter = this_vec.begin(), end = this_vec.end();

      for (;iter != end; ++iter) {
        const TempArc &temp_arc(*iter);
        GallicArc<Arc> new_arc;
        vector<Label> seq;
        repository_.SeqOfId(temp_arc.ostring, &seq);
        StringWeight<Label, STRING_LEFT> string_weight;
        for (size_t i = 0;i < seq.size();i++) string_weight.PushBack(seq[i]);
        ThisGallicWeight gallic_weight(string_weight, temp_arc.weight);

        if (temp_arc.nextstate == kNoStateId) {  // is really final weight.
          ofst->SetFinal(this_state, gallic_weight);
        } else {  // is really an arc.
          new_arc.nextstate = temp_arc.nextstate;
          new_arc.ilabel = temp_arc.ilabel;
          new_arc.olabel = temp_arc.ilabel;  // acceptor.  input == output.
          new_arc.weight = gallic_weight;  // includes string and weight.
          ofst->AddArc(this_state, new_arc);
        }
      }
      // Free up memory.  Do this inside the loop as ofst is also allocating memory
      if (destroy) { vector<TempArc> temp; temp.swap(this_vec); }
    }
    if (destroy) { vector<vector<TempArc> > temp; temp.swap(output_arcs_); }
  }

  // Output to standard FST.  We will create extra states to handle sequences of symbols
  // on the output.  If destroy == true, release memory as we go
  // (but we cannot output again).

  void  Output(MutableFst<Arc> *ofst, bool destroy = true) {
    assert(determinized_);
    if (destroy) determinized_ = false;
    // Outputs to standard fst.
    OutputStateId nStates = static_cast<OutputStateId>(output_arcs_.size());
    if (destroy)
      FreeMostMemory();
    ofst->DeleteStates();
    if (nStates == 0) {
      ofst->SetStart(kNoStateId);
      return;
    }
    // Add basic states-- but will add extra ones to account for strings on output.
    for (OutputStateId s = 0;s < nStates;s++) {
      OutputStateId news = ofst->AddState();
      assert(news == s);
    }
    ofst->SetStart(0);
    for (OutputStateId this_state = 0; this_state < nStates; this_state++) {
      vector<TempArc> &this_vec(output_arcs_[this_state]);

      typename vector<TempArc>::const_iterator iter = this_vec.begin(), end = this_vec.end();
      for (;iter != end; ++iter) {
        const TempArc &temp_arc(*iter);
        vector<Label> seq;
        repository_.SeqOfId(temp_arc.ostring, &seq);

        if (temp_arc.nextstate == kNoStateId) {  // Really a final weight.
          // Make a sequence of states going to a final state, with the strings as labels.
          // Put the weight on the first arc.
          OutputStateId cur_state = this_state;
          for (size_t i = 0;i < seq.size();i++) {
            OutputStateId next_state = ofst->AddState();
            Arc arc;
            arc.nextstate = next_state;
            arc.weight = (i == 0 ? temp_arc.weight : Weight::One());
            arc.ilabel = 0;  // epsilon.
            arc.olabel = seq[i];
            ofst->AddArc(cur_state, arc);
            cur_state = next_state;
          }
          ofst->SetFinal(cur_state, (seq.size() == 0 ? temp_arc.weight : Weight::One()));
        } else {  // Really an arc.
          OutputStateId cur_state = this_state;
          // Have to be careful with this integer comparison (i+1 < seq.size()) because unsigned.
          // i < seq.size()-1 could fail for zero-length sequences.
          for (size_t i = 0; i+1 < seq.size();i++) {
            // for all but the last element of seq, create new state.
            OutputStateId next_state = ofst->AddState();
            Arc arc;
            arc.nextstate = next_state;
            arc.weight = (i == 0 ? temp_arc.weight : Weight::One());
            arc.ilabel = (i == 0 ? temp_arc.ilabel : 0);  // put ilabel on first element of seq.
            arc.olabel = seq[i];
            ofst->AddArc(cur_state, arc);
            cur_state = next_state;
          }
          // Add the final arc in the sequence.
          Arc arc;
          arc.nextstate = temp_arc.nextstate;
          arc.weight = (seq.size() <= 1 ? temp_arc.weight : Weight::One());
          arc.ilabel = (seq.size() <= 1 ? temp_arc.ilabel : 0);
          arc.olabel = (seq.size() > 0 ? seq.back() : 0);
          ofst->AddArc(cur_state, arc);
        }
      }
      // Free up memory.  Do this inside the loop as ofst is also allocating memory
      if (destroy) { vector<TempArc> temp; temp.swap(this_vec); }
    }
    if (destroy) {
      vector<vector<TempArc> > temp;
      temp.swap(output_arcs_);
      repository_.Destroy();
    }
  }


  // Initializer.  After initializing the object you will typically call one of
  // the Output functions.
  DeterminizerStar(const Fst<Arc> &ifst, float delta = kDelta,
                   int max_states = -1, bool allow_partial = false):
      ifst_(ifst.Copy()), delta_(delta), max_states_(max_states),
      determinized_(false), allow_partial_(allow_partial),
      is_partial_(false), equal_(delta),
      hash_(ifst.Properties(kExpanded, false) ? down_cast<const ExpandedFst<Arc>*, const Fst<Arc> >(&ifst)->NumStates()/2 + 3 : 20, hasher_, equal_) { }

  void Determinize(bool *debug_ptr) {
    assert(!determinized_);
    // This determinizes the input fst but leaves it in the "special format"
    // in "output_arcs_".
    InputStateId start_id = ifst_->Start();
    if (start_id == kNoStateId) { determinized_ = true; return; } // Nothing to do.
    else {  // Insert start state into hash and queue.
      Element elem;
      elem.state = start_id;
      elem.weight = Weight::One();
      elem.string = repository_.IdOfEmpty();  // Id of empty sequence.
      vector<Element> vec;
      vec.push_back(elem);
      OutputStateId cur_id = SubsetToStateId(vec);
      assert(cur_id == 0 && "Do not call Determinize twice.");
    }
    while (!Q_.empty()) {
      pair<vector<Element>*, OutputStateId> cur_pair = Q_.front();
      Q_.pop_front();
      ProcessSubset(cur_pair);
      if (debug_ptr && *debug_ptr) Debug();  // will exit.
      if (max_states_ > 0 && output_arcs_.size() > max_states_) {
        if (allow_partial_ == false) {
          std::cerr << "Determinization aborted since passed " << max_states_
                    << " states.\n";
          throw std::runtime_error("max-states reached in determinization");
        } else {
          KALDI_WARN << "Determinization terminated since passed " << max_states_
                     << " states, partial results will be generated.";
          is_partial_ = true;
          break;
        }
      }
    }
    determinized_ = true;
  }

  bool IsPartial() {
    return is_partial_;
  }
  
  // frees all except output_arcs_, which contains the important info
  // we need to output.
  void FreeMostMemory() {
    if (ifst_) {
      delete ifst_;
      ifst_ = NULL;
    }
    for (typename SubsetHash::iterator iter = hash_.begin();
        iter != hash_.end(); ++iter)
      delete iter->first;
    SubsetHash tmp;
    tmp.swap(hash_); 
  }
  
  ~DeterminizerStar() {
    FreeMostMemory();
  }
 private:
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId InputStateId;
  typedef typename Arc::StateId OutputStateId;  // same as above but distinguish states in output Fst.
  typedef typename Arc::Label StringId;  // Id type used in the StringRepository
  typedef StringRepository<Label, StringId> StringRepositoryType;


  // Element of a subset [of original states]

  struct Element {
    InputStateId state;
    StringId string;
    Weight weight;
    bool operator != (const Element &other) const  {
      return (state != other.state || string != other.string ||
              weight != other.weight);
    }
  };

  // Arcs in the format we temporarily create in this class (a representation, essentially of
  // a Gallic Fst).
  struct TempArc {
    Label ilabel;
    StringId ostring;  // Look it up in the StringRepository, it's a sequence of Labels.
    OutputStateId nextstate;  // or kNoState for final weights.
    Weight weight;
  };


  // Hashing function used in hash of subsets.
  // A subset is a pointer to vector<Element>.
  // The Elements are in sorted order on state id, and without repeated states.
  // Because the order of Elements is fixed, we can use a hashing function that is
  // order-dependent.  However the weights are not included in the hashing function--
  // we hash subsets that differ only in weight to the same key.  This is not optimal
  // in terms of the O(N) performance but typically if we have a lot of determinized
  // states that differ only in weight then the input probably was pathological in some way,
  // or even non-determinizable.
  //   We don't quantize the weights, in order to avoid inexactness in simple cases.
  // Instead we apply the delta when comparing subsets for equality, and allow a small
  // difference.

  class SubsetKey {
   public:
    size_t operator ()(const vector<Element> * subset) const {  // hashes only the state and string.
      size_t hash = 0, factor = 1;
      for (typename vector<Element>::const_iterator iter= subset->begin(); iter != subset->end(); ++iter) {
        hash *= factor;
        hash += iter->state + 103333*iter->string;
        factor *= 23531;  // these numbers are primes.
      }
      return hash;
    }
  };

  // This is the equality operator on subsets.  It checks for exact match on state-id
  // and string, and approximate match on weights.
  class SubsetEqual {
   public:
    bool operator ()(const vector<Element> * s1, const vector<Element> * s2) const {
      size_t sz = s1->size();
      assert(sz>=0);
      if (sz != s2->size()) return false;
      typename vector<Element>::const_iterator iter1 = s1->begin(),
          iter1_end = s1->end(), iter2=s2->begin();
      for (; iter1 < iter1_end; ++iter1, ++iter2) {
        if (iter1->state != iter2->state ||
           iter1->string != iter2->string ||
           ! ApproxEqual(iter1->weight, iter2->weight, delta_)) return false;
      }
      return true;
    }
    float delta_;
    SubsetEqual(float delta): delta_(delta) {}
    SubsetEqual(): delta_(kDelta) {}
  };

  // Operator that says whether two Elements have the same states.
  // Used only for debug.
  class SubsetEqualStates {
   public:
    bool operator ()(const vector<Element> * s1, const vector<Element> * s2) const {
      size_t sz = s1->size();
      assert(sz>=0);
      if (sz != s2->size()) return false;
      typename vector<Element>::const_iterator iter1 = s1->begin(),
          iter1_end = s1->end(), iter2=s2->begin();
      for (; iter1 < iter1_end; ++iter1, ++iter2) {
        if (iter1->state != iter2->state) return false;
      }
      return true;
    }
  };

  // Define the hash type we use to store subsets.
  typedef unordered_map<const vector<Element>*, OutputStateId, SubsetKey, SubsetEqual> SubsetHash;


  // This function computes epsilon closure of subset of states by following epsilon links.
  // Called by ProcessSubset.
  // Has no side effects except on the repository.

  void EpsilonClosure(const vector<Element> & input_subset,
                      vector<Element> *output_subset) {
    // input_subset must have only one example of each StateId.

    std::map<InputStateId, Element> cur_subset;
    typedef typename std::map<InputStateId, Element>::iterator MapIter;
    {
      MapIter iter = cur_subset.end();
      for (size_t i = 0;i < input_subset.size();i++) {
        std::pair<const InputStateId, Element> pr(input_subset[i].state, input_subset[i]);
        iter = cur_subset.insert(iter, pr);
        // By providing iterator where we inserted last one, we make insertion more efficient since
        // input subset was already in sorted order.
      }
    }
    // find whether input fst is known to be sorted in input label.
    bool sorted = ((ifst_->Properties(kILabelSorted, false) & kILabelSorted) != 0);
    
    vector<Element> queue(input_subset);  // queue of things to be processed.
    bool replaced_elems = false; // relates to an optimization, see below.
    int counter = 0; // relates to max-states option, used for test.
    while (queue.size() != 0) {
      Element elem = queue.back();
      queue.pop_back();
      // The next if-statement is a kind of optimization.  It's to prevent us
      // unnecessarily repeating the processing of a state.  "cur_subset" always
      // contains only one Element with a particular state.  The issue is that
      // whenever we modify the Element corresponding to that state in "cur_subset",
      // both the new (optimal) and old (less-optimal) Element will still be in
      // "queue".  The next if-statement stops us from wasting compute by
      // processing the old Element.
      if (replaced_elems && cur_subset[elem.state] != elem)
        continue;
      if (max_states_ > 0 && counter++ > max_states_) {
        std::cerr << "Determinization aborted since looped more than "
                  << max_states_ << " times during epsilon closure.\n";
        throw std::runtime_error("looped more than max-states times in determinization");
      }      
      for (ArcIterator<Fst<Arc> > aiter(*ifst_, elem.state); !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (sorted && arc.ilabel != 0) break;  // Break from the loop: due to sorting there will be no
        // more transitions with epsilons as input labels.
        if (arc.ilabel == 0) {  // Epsilon transition.
          Element next_elem;
          next_elem.state = arc.nextstate;
          next_elem.weight = Times(elem.weight, arc.weight);
          // now must append strings
          if (arc.olabel == 0)
            next_elem.string = elem.string;
          else {
            vector<Label> seq;
            repository_.SeqOfId(elem.string, &seq);
            if (arc.olabel != 0)
              seq.push_back(arc.olabel);
            next_elem.string = repository_.IdOfSeq(seq);
          }
          typename std::map<InputStateId, Element>::iterator
              iter = cur_subset.find(next_elem.state);
          if (iter == cur_subset.end()) {
            // was no such StateId: insert and add to queue.
            cur_subset[next_elem.state] = next_elem;
            queue.push_back(next_elem);
          } else {  // one is already there.  Add weights.
            if (iter->second.string != next_elem.string) {
              std::cerr << "DeterminizerStar: FST was not functional -> not determinizable\n";
              { // Print some debugging information.  Can be helpful to debug
                // the inputs when FSTs are mysteriously non-functional.
                vector<Label> tmp_seq;
                repository_.SeqOfId(iter->second.string, &tmp_seq);
                std::cerr << "First string: ";
                for (size_t i = 0; i < tmp_seq.size(); i++) std::cerr << tmp_seq[i] << " ";
                std::cerr << "\nSecond string: ";
                repository_.SeqOfId(next_elem.string, &tmp_seq);
                for (size_t i = 0; i < tmp_seq.size(); i++) std::cerr << tmp_seq[i] << " ";
                std::cerr << "\n";
              }
              throw std::runtime_error("Non-functional FST: cannot determinize.\n");
            }
            Weight weight = Plus(iter->second.weight, next_elem.weight);
            if (! ApproxEqual(weight, iter->second.weight, delta_)) {  // add extra part of weight to queue.
              queue.push_back(next_elem);
              replaced_elems = true;
            }
            iter->second.weight = weight; // Update weight in map.
          }
        }
      }
    }

    {  // copy cur_subset to output_subset.
      // sorted order is automatic.
      output_subset->clear();
      output_subset->reserve(cur_subset.size());
      MapIter iter = cur_subset.begin(), end = cur_subset.end();
      for (; iter != end; ++iter) output_subset->push_back(iter->second);
    }
  }


  // This function works out the final-weight of the determinized state.
  // called by ProcessSubset.
  // Has no side effects except on the variable repository_, and output_arcs_.

  void ProcessFinal(const vector<Element> &closed_subset, OutputStateId state) {
    // processes final-weights for this subset.
    bool is_final = false;
    StringId final_string = 0;  // = 0 to keep compiler happy.
    Weight final_weight;
    typename vector<Element>::const_iterator iter = closed_subset.begin(), end = closed_subset.end();
    for (; iter != end; ++iter) {
      const Element &elem = *iter;
      Weight this_final_weight = ifst_->Final(elem.state);
      if (this_final_weight != Weight::Zero()) {
        if (!is_final) {  // first final-weight
          final_string = elem.string;
          final_weight = Times(elem.weight, this_final_weight);
          is_final = true;
        } else {  // already have one.
          if (final_string != elem.string) {
            std::cerr << "DeterminizerStar: FST was not functional -> not determinizable\n";
            throw std::runtime_error("Non-functional FST: cannot determinize.\n");
          }            
          final_weight = Plus(final_weight, Times(elem.weight, this_final_weight));
        }
      }
    }
    if (is_final) {
      // store final weights in TempArc structure, just like a transition.
      TempArc temp_arc;
      temp_arc.ilabel = 0;
      temp_arc.nextstate = kNoStateId;  // special marker meaning "final weight".
      temp_arc.ostring = final_string;
      temp_arc.weight = final_weight;
      output_arcs_[state].push_back(temp_arc);
    }
  }

  // ProcessTransition is called from "ProcessTransitions".  Broken out for clarity.
  // Has side effects on output_arcs_, and (via SubsetToStateId) Q_ and hash_

  void ProcessTransition(OutputStateId state, Label ilabel, vector<Element> *subset) {
    // At input, "subset" may contain duplicates for a given dest state (but in sorted
    // order).  This function removes duplicates from "subset", normalizes it, and adds
    // a transition to the dest. state (possibly affecting Q_ and hash_, if state did not
    // exist).

    typedef typename vector<Element>::iterator IterType;
    {  // This block makes the subset have one unique Element per state, adding the weights.
      IterType cur_in = subset->begin(), cur_out = cur_in, end = subset->end();
      size_t num_out = 0;
      // Merge elements with same state-id
      while (cur_in != end) {  // while we have more elements to process.
        // At this point, cur_out points to location of next place we want to put an element,
        // cur_in points to location of next element we want to process.
        if (cur_in != cur_out) *cur_out = *cur_in;
        cur_in++;
        while (cur_in != end && cur_in->state == cur_out->state) {  // merge elements.
          if (cur_in->string != cur_out->string) {
            std::cerr << "DeterminizerStar: FST was not functional -> not determinizable\n";
            throw std::runtime_error("Non-functional FST: cannot determinize.\n");
          }            
          cur_out->weight = Plus(cur_out->weight, cur_in->weight);
          cur_in++;
        }
        cur_out++;
        num_out++;
      }
      subset->resize(num_out);
    }

    StringId common_str;
    Weight tot_weight;
    {  // This block computes common_str and tot_weight (essentially: the common divisor)
      // and removes them from the elements.
      vector<Label> seq;

      IterType begin = subset->begin(), iter, end = subset->end();
      {  // This block computes "seq", which is the common prefix, and "common_str",
        // which is the StringId version of "seq".
        vector<Label> tmp_seq;
        for (iter = begin; iter!= end; ++iter) {
          if (iter == begin) {
            repository_.SeqOfId(iter->string, &seq);
          } else {
            repository_.SeqOfId(iter->string, &tmp_seq);
            if (tmp_seq.size() < seq.size()) seq.resize(tmp_seq.size());  // size of shortest one.
            for (size_t i = 0;i < seq.size(); i++) // seq.size() is the shorter one at this point.
              if (tmp_seq[i] != seq[i]) seq.resize(i);
          }
          if (seq.size() == 0) break;  // will not get any prefix.
        }
        common_str = repository_.IdOfSeq(seq);
      }

      {  // This block computes "tot_weight".
        iter = begin;
        tot_weight = iter->weight;
        for (++iter; iter!= end; ++iter)
          tot_weight = Plus(tot_weight, iter->weight);
      }

      // Now divide out common stuff from elements.
      size_t prefix_len = seq.size();
      for (iter = begin; iter != end; ++iter) {
        iter->weight = Divide(iter->weight, tot_weight);
        iter->string = repository_.RemovePrefix(iter->string, prefix_len);
      }
    }

    // Now add an arc to the state that the subset represents.
    // We may create a new state id for this (in SubsetToStateId).
    TempArc temp_arc;
    temp_arc.ilabel = ilabel;
    temp_arc.nextstate = SubsetToStateId(*subset);  // may or may not really add the subset.
    temp_arc.ostring = common_str;
    temp_arc.weight = tot_weight;
    output_arcs_[state].push_back(temp_arc);  // record the arc.
  }


  // "less than" operator for pair<Label, Element>.   Used in ProcessTransitions.
  // Lexicographical order, with comparing the state only for "Element".

  class PairComparator {
   public:
    inline bool operator () (const pair<Label, Element> &p1, const pair<Label, Element> &p2) {
      if (p1.first < p2.first) return true;
      else if (p1.first > p2.first) return false;
      else {
        return p1.second.state < p2.second.state;
      }
    }
  };


  // ProcessTransitions handles transitions out of this subset of states.
  // Ignores epsilon transitions (epsilon closure already handled that).
  // Does not consider final states.  Breaks the transitions up by ilabel,
  // and creates a new transition in determinized FST, for each ilabel.
  // Does this by creating a big vector of pairs <Label, Element> and then sorting them
  // using a lexicographical ordering, and calling ProcessTransition for each range
  // with the same ilabel.
  // Side effects on repository, and (via ProcessTransition) on Q_, hash_,
  // and output_arcs_.

  void ProcessTransitions(const vector<Element> &closed_subset, OutputStateId state) {
    vector<pair<Label, Element> > all_elems;
    {  // Push back into "all_elems", elements corresponding to all non-epsilon-input transitions
      // out of all states in "closed_subset".
      typename vector<Element>::const_iterator iter = closed_subset.begin(), end = closed_subset.end();
      for (;iter != end; ++iter) {
        const Element &elem = *iter;
        for (ArcIterator<Fst<Arc> > aiter(*ifst_, elem.state); ! aiter.Done(); aiter.Next()) {
          const Arc &arc = aiter.Value();
          if (arc.ilabel != 0) {  // Non-epsilon transition -- ignore epsilons here.
            pair<Label, Element> this_pr;
            this_pr.first = arc.ilabel;
            Element &next_elem(this_pr.second);
            next_elem.state = arc.nextstate;
            next_elem.weight = Times(elem.weight, arc.weight);
            if (arc.olabel == 0) // output epsilon-- this is simple case so
                                 // handle separately for efficiency
              next_elem.string = elem.string;
            else {
              vector<Label> seq;
              repository_.SeqOfId(elem.string, &seq);
              seq.push_back(arc.olabel);
              next_elem.string = repository_.IdOfSeq(seq);
            }
            all_elems.push_back(this_pr);
          }
        }
      }
    }
    PairComparator pc;
    std::sort(all_elems.begin(), all_elems.end(), pc);
    // now sorted first on input label, then on state.
    typedef typename vector<pair<Label, Element> >::const_iterator PairIter;
    PairIter cur = all_elems.begin(), end = all_elems.end();
    vector<Element> this_subset;
    while (cur != end) {
      // Process ranges that share the same input symbol.
      Label ilabel = cur->first;
      this_subset.clear();
      while (cur != end && cur->first == ilabel) {
        this_subset.push_back(cur->second);
        cur++;
      }
      // We now have a subset for this ilabel.
      ProcessTransition(state, ilabel, &this_subset);
    }
  }

  // SubsetToStateId converts a subset (vector of Elements) to a StateId in the output
  // fst.  This is a hash lookup; if no such state exists, it adds a new state to the hash
  // and adds a new pair to the queue.
  // Side effects on hash_ and Q_, and on output_arcs_ [just affects the size].

  OutputStateId SubsetToStateId(const vector<Element> &subset) {  // may add the subset to the queue.
    typedef typename SubsetHash::iterator IterType;
    IterType iter = hash_.find(&subset);
    if (iter == hash_.end()) {  // was not there.
      vector<Element> *new_subset = new vector<Element>(subset);
      OutputStateId new_state_id = (OutputStateId) output_arcs_.size();
      hash_[new_subset] = new_state_id;
      output_arcs_.push_back(vector<TempArc>());
      if (allow_partial_ == false) {
        // If --allow-partial is not requested, we do the old way.
        Q_.push_front(pair<vector<Element>*, OutputStateId>(new_subset,  new_state_id));
      } else {
        // If --allow-partial is requested, we do breadth first search. This
        // ensures that when we return partial results, we return the states
        // that are reachable by the fewest steps from the start state.
        Q_.push_back(pair<vector<Element>*, OutputStateId>(new_subset,  new_state_id));
      }
      return new_state_id;
    } else {
      return iter->second;  // the OutputStateId.
    }
  }


  // ProcessSubset does the processing of a determinized state, i.e. it creates
  // transitions out of it and adds new determinized states to the queue if necessary.
  // The first stage is "EpsilonClosure" (follow epsilons to get a possibly larger set
  // of (states, weights)).  After that we ignore epsilons.  We process the final-weight
  // of the state, and then handle transitions out (this may add more determinized states
  // to the queue).

  void ProcessSubset(const pair<vector<Element>*, OutputStateId> & pair) {
    const vector<Element> *subset = pair.first;
    OutputStateId state = pair.second;

    vector<Element> closed_subset;  // subset after epsilon closure.
    EpsilonClosure(*subset, &closed_subset);

    // Now follow non-epsilon arcs [and also process final states]
    ProcessFinal(closed_subset, state);

    // Now handle transitions out of these states.
    ProcessTransitions(closed_subset, state);
  }

  void Debug() {  // this function called if you send a signal
    // SIGUSR1 to the process (and it's caught by the handler in
    // fstdeterminizestar).  It prints out some traceback
    // info and exits.

    std::cerr << "Debug function called (probably SIGUSR1 caught).\n";
    // free up memory from the hash as we need a little memory
    { SubsetHash hash_tmp; std::swap(hash_tmp, hash_); }

    if (output_arcs_.size() <= 2) {
      std::cerr << "Nothing to trace back";
      exit(1);
    }
    size_t max_state = output_arcs_.size() - 2;  // don't take the last
    // one as we might be halfway into constructing it.

    vector<OutputStateId> predecessor(max_state+1, kNoStateId);
    for (size_t i = 0; i < max_state; i++) {
      for (size_t j = 0; j < output_arcs_[i].size(); j++) {
        OutputStateId nextstate = output_arcs_[i][j].nextstate;
        // always find an earlier-numbered prececessor; this
        // is always possible because of the way the algorithm
        // works.
        if (nextstate <= max_state && nextstate > i)
          predecessor[nextstate] = i;
      }
    }
    vector<pair<Label, StringId> > traceback;
    // traceback is a pair of (ilabel, olabel-seq).
    OutputStateId cur_state = max_state;  // a recently constructed state.

    while (cur_state != 0 && cur_state != kNoStateId) {
      OutputStateId last_state = predecessor[cur_state];
      pair<Label, StringId> p;
      size_t i;
      for (i = 0; i < output_arcs_[last_state].size(); i++) {
        if (output_arcs_[last_state][i].nextstate == cur_state) {
          p.first = output_arcs_[last_state][i].ilabel;
          p.second = output_arcs_[last_state][i].ostring;
          traceback.push_back(p);
          break;
        }
      }
      assert(i != output_arcs_[last_state].size());  // or fell off loop.
      cur_state = last_state;
    }
    if (cur_state == kNoStateId)
      std::cerr << "Traceback did not reach start state (possibly debug-code error)";

    std::cerr << "Traceback below (or on standard error) in format ilabel (olabel olabel) ilabel (olabel) ...\n";
    for (ssize_t i = traceback.size() - 1; i >= 0; i--) {
      std::cerr << traceback[i].first << ' ' << "( ";
      vector<Label> seq;
      repository_.SeqOfId(traceback[i].second, &seq);
      for (size_t j = 0; j < seq.size(); j++)
        std::cerr << seq[j] << ' ';
      std::cerr << ") ";
    }
    std::cerr << '\n';
    exit(1);
  }


  DISALLOW_COPY_AND_ASSIGN(DeterminizerStar);
  deque<pair<vector<Element>*, OutputStateId> > Q_;  // queue of subsets to be processed.

  vector<vector<TempArc> > output_arcs_;  // essentially an FST in our format.

  const Fst<Arc> *ifst_;
  float delta_;
  int max_states_;
  bool determinized_; // used to check usage.
  bool allow_partial_;  // output paritial results or not
  bool is_partial_;     // if we get partial results or not
  SubsetKey hasher_;  // object that computes keys-- has no data members.
  SubsetEqual equal_;  // object that compares subsets-- only data member is delta_.
  SubsetHash hash_;  // hash from Subset to StateId in final Fst.

  StringRepository<Label, StringId> repository_;  // associate integer id's with sequences of labels.
};


template<class Arc>
bool DeterminizeStar(Fst<Arc> &ifst, MutableFst<Arc> *ofst,
                     float delta, bool *debug_ptr, int max_states,
                     bool allow_partial) {
  ofst->SetOutputSymbols(ifst.OutputSymbols());
  ofst->SetInputSymbols(ifst.InputSymbols());
  DeterminizerStar<Arc> det(ifst, delta, max_states, allow_partial);
  det.Determinize(debug_ptr);
  det.Output(ofst);
  return det.IsPartial();
}


template<class Arc>
bool DeterminizeStar(Fst<Arc> &ifst, MutableFst<GallicArc<Arc> > *ofst, float delta,
                     bool *debug_ptr, int max_states,
                     bool allow_partial) {
  ofst->SetOutputSymbols(ifst.InputSymbols());
  ofst->SetInputSymbols(ifst.InputSymbols());
  DeterminizerStar<Arc> det(ifst, delta, max_states, allow_partial);
  det.Determinize(debug_ptr);
  det.Output(ofst);
  return det.IsPartial();
}



}


#endif
