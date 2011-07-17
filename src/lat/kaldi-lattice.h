// transform/kaldi-lattice.h

// Copyright 2009-2011  Microsoft Corporation

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


#ifndef KALDI_LAT_KALDI_LATTICE_H_
#define KALDI_LAT_KALDI_LATTICE_H_

#include "fstext/fstext-lib.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"

namespace fst {

// Declare weight type for lattice... will import to namespace kaldi.
// has two members, a_ and b_, of type BaseFloat (normally equals float).
// It is basically the same as the tropical semiring on a_+b_, except it keeps
// track of a and b separately.
// More precisely, it is equivalent to the lexicographic semiring on
// (a_+b_), (a_-b_)

template<class FloatType>
class LatticeWeightTpl {
 public:
  typedef FloatType T; // normally float.
  typedef LatticeWeightTpl ReverseWeight;
  
  LatticeWeightTpl() { }

  LatticeWeightTpl(T a, T b): a_(a), b_(b) {}

  LatticeWeightTpl(const LatticeWeightTpl &other): a_(other.a_), b_(other.b_) { }

  LatticeWeightTpl &operator=(const LatticeWeightTpl &w) {
    a_ = w.a_;
    b_ = w.b_;
    return *this;
  }

  static const LatticeWeightTpl Zero() {
    return LatticeWeightTpl(FloatLimits<T>::kPosInfinity, FloatLimits<T>::kPosInfinity);
  }

  static const LatticeWeightTpl One() {
    return LatticeWeightTpl(0.0, 0.0);
  }
  
  static const string &Type() {
    static const string type = (sizeof(T) == 4 ? "lattice" : "lattice_dbl") ;
    return type;
  }

  bool Member() const {
    // a_ == a_ tests for NaN.
    // also test for no -inf, and either both or neither
    // must be +inf, and
    if(a_ != a_ || b_ != b_) return false; // NaN
    if(a_ == FloatLimits<T>::kNegInfinity  ||
       b_ == FloatLimits<T>::kNegInfinity) return false; // -infty not allowed
    if(a_ == FloatLimits<T>::kPosInfinity ||
       b_ == FloatLimits<T>::kPosInfinity) {
      if(a_ != FloatLimits<T>::kPosInfinity ||
         b_ != FloatLimits<T>::kPosInfinity) return false; // both must be +infty;
      // this is necessary so that the semiring has only one zero.
    }
    return true;
  }
  
  LatticeWeightTpl Quantize(float delta = kDelta) const {
    if(a_+b_ == FloatLimits<T>::kNegInfinity) {
      return LatticeWeightTpl(FloatLimits<T>::kNegInfinity,FloatLimits<T>::kNegInfinity);
    } else if(a_+b_ == FloatLimits<T>::kPosInfinity) {
      return LatticeWeightTpl(FloatLimits<T>::kPosInfinity,FloatLimits<T>::kPosInfinity);
    } else if(a_+b_ != a_+b_) { // NaN
      return LatticeWeightTpl(a_+b_, a_+b_);
    } else {
      return LatticeWeightTpl(floor(a_/delta + 0.5F)*delta, floor(b_/delta + 0.5F) * delta);
    }
  }

  // This is used in OpenFst for binary I/O.  This is OpenFst-style,
  // not Kaldi-style, I/O.
  istream &Read(istream &strm) {
    // Always read/write as float, even if T is double,
    // so we can use OpenFst-style read/write and still maintain
    // compatibility when compiling with different FloatTypes
    ReadType(strm, &a_);
    ReadType(strm, &b_);
    return strm;
  }


  // This is used in OpenFst for binary I/O.  This is OpenFst-style,
  // not Kaldi-style, I/O.
  ostream &Write(ostream &strm) const {
    WriteType(strm, a_);
    WriteType(strm, b_);
    return strm;
  }

  size_t Hash() const {
    size_t ans;
    union {
      T f;
      size_t s;
    } u;
    u.s = 0;
    u.f = a_;
    ans = u.s;
    u.f = b_;
    ans += u.s;
    return ans;
  }

  T a_;
  T b_;
};


template<class FloatType>
inline bool operator==(const LatticeWeightTpl<FloatType> &w1,
                       const LatticeWeightTpl<FloatType> &w2 ) {
  return (w1.a_ == w2.a_ && w1.b_ == w2.b_);
}


// We define a Compare function LatticeWeightTpl even though it's
// not required by the semiring standard-- it's just more efficient
// to do it this way rather than using the NaturalLess template.

/// Compare returns -1 if w1 < w2, +1 if w1 > w2, and 0 if w1 == w2.

template<class FloatType>
inline int Compare (const  LatticeWeightTpl<FloatType> &w1,
                    const LatticeWeightTpl<FloatType> &w2) {
  FloatType f1 = w1.a_ + w1.b_,
      f2 = w2.a_ + w2.b_;
  if(f1 < f2) { return 1; } // having smaller cost means you're larger
  // in the semiring [higher probability]
  else if(f1 > f2) { return -1; }
  // mathematically we should be comparing (w1.a_-w1.b_ < w2.a_-w2.b_)
  // in the next line, but add w1.a_+w1.b_ = w2.a_+w2.b_ to both sides and
  // divide by two, and we get the simpler equivalent form w1.a_ < w2.a_.
  else if(w1.a_ < w2.a_) { return 1; } 
  else { return 0; }
}


template<class FloatType>
inline LatticeWeightTpl<FloatType> Plus(const LatticeWeightTpl<FloatType> &w1,
                             const LatticeWeightTpl<FloatType> &w2) {
  return (Compare(w1,w2) >= 0 ? w1 : w2); //
}


template<class FloatType>
inline LatticeWeightTpl<FloatType> Times(const LatticeWeightTpl<FloatType> &w1,
                              const LatticeWeightTpl<FloatType> &w2) {
  return LatticeWeightTpl<FloatType>(w1.a_+w2.a_, w1.b_+w2.b_);  
}

// divide w1 by w2 (on left/right/any doesn't matter as
// commutative).
template<class FloatType>
inline LatticeWeightTpl<FloatType> Divide(const LatticeWeightTpl<FloatType> &w1,
                               const LatticeWeightTpl<FloatType> &w2,
                               DivideType typ = DIVIDE_ANY) {
  typedef FloatType T;
  T a = w1.a_ - w2.a_, b = w1.b_ - w2.b_;
  if(a!=a || b!=b || a == FloatLimits<T>::kNegInfinity
     || b == FloatLimits<T>::kNegInfinity) {
    std::cerr << "LatticeWeightTpl::Divide, NaN or invalid number produced. "
              << "[dividing by zero?]  Returning zero.";
    return LatticeWeightTpl<T>::Zero();
  }
  if(a == FloatLimits<T>::kPosInfinity ||
     b == FloatLimits<T>::kPosInfinity)
    return LatticeWeightTpl<T>::Zero(); // problems if only one is infinite.
  return LatticeWeightTpl<T>(a, b);
}


template<class FloatType>
inline bool ApproxEqual(const LatticeWeightTpl<FloatType> &w1,
                        const LatticeWeightTpl<FloatType> &w2,
                        float delta = kDelta) {
  return (fabs(w1.a_ - w2.a_) <= delta && fabs(w1.b_ - w2.b_) <= delta);
}

template <class FloatType>
inline ostream &operator <<(ostream &strm, LatticeWeightTpl<FloatType> &w1) {
  typedef FloatType T;
  if(w1.a_ == FloatLimits<T>::kPosInfinity)
    strm << "Infinity";
  else if (w1.a_ == FloatLimits<T>::kNegInfinity
           || w1.a_ != w1.a_)  // -infty not a valid weight so treat as NaN
    strm << "BadNumber";
  else
    strm << w1.a_;
  strm << ';'; // hard-code separator as ';'

  if(w1.b_ == FloatLimits<T>::kPosInfinity)
    strm << "Infinity";
  else if (w1.b_ == FloatLimits<T>::kNegInfinity
           || w1.b_ != w1.b_)  // -infty not a valid weight so treat as NaN
    strm << "BadNumber";
  else
    strm << w1.b_;
  return strm;
}

template <class FloatType>
inline istream &operator >>(istream &strm, LatticeWeightTpl<FloatType> &w1) {
  // TODO...
  return strm;
}



// CompactLattice will be an acceptor (accepting the words/output-symbols),
// with the weights and input-symbol-seqs on the arcs.
// There must be a total order on W.  We assume for the sake of efficiency
// that there is a function
// Compare(W w1, W w2) that returns -1 if w1 < w2, +1 if w1 > w2, and
// zero if w1 == w2, and Plus for type W returns (Compare(w1,w2) >= 0 ? w1 : w2).

template<class WeightType, class IntType>
class CompactLatticeWeightTpl {
 public:
  typedef WeightType W;
  W w_;
  vector<IntType> s_; 

  // Plus is like LexicographicWeight on the pair (w_, s_), but where we
  // use standard lexicographic order on s_ [this is not the same as
  // NaturalLess on the StringWeight equivalent, which does not define a
  // total order].
  // Times, Divide obvious... (support both left & right division..)
  // CommonDivisor would need to be coded separately.

  CompactLatticeWeightTpl() { }

  CompactLatticeWeightTpl(const WeightType &w, const std::vector<IntType> &s):
      w_(w), s_(s) { }

  CompactLatticeWeightTpl &operator=(const CompactLatticeWeightTpl<WeightType,IntType> &w) {
    w_ = w.w_;
    s_ = w.s_;
    return *this;
  }

  static const CompactLatticeWeightTpl<WeightType,IntType> Zero() {
    return CompactLatticeWeightTpl<WeightType,IntType>(
        WeightType::Zero(), std::vector<IntType>());
  }

  static const CompactLatticeWeightTpl<WeightType,IntType> One() {
    return CompactLatticeWeightTpl<WeightType,IntType>(
        WeightType::One(), std::vector<IntType>());
  }

  inline static string GetIntSizeString() {
    char buf[2];
    buf[0] = '0' + sizeof(IntType);
    buf[1] = '\0';
    return buf;
  }
  static const string &Type() {
    static const string type = "compact" + WeightType::Type()
        + GetIntSizeString();
    return type;
  }
  
  bool Member() const {
    // a semiring has only one zero, this is the important property
    // we're trying to maintain here.  So force s_ to be empty if
    // w_ == zero.
    if(!w_.Member()) return false;
    if(w_ == WeightType::Zero())
      return s_.empty();
    else
      return true;
  }

  CompactLatticeWeightTpl Quantize(float delta = kDelta) const {
    return CompactLatticeWeightTpl(w_.Quantize(delta), s_);
  }

  // This is used in OpenFst for binary I/O.  This is OpenFst-style,
  // not Kaldi-style, I/O.
  istream &Read(istream &strm) {
    w_.Read(strm);
    if(strm.fail()){ return strm; }
    IntType sz;
    ReadType(strm, &sz);
    if(strm.fail()){ return strm; }
    if(sz < 0) {
      std::cerr << "Negative string size!  Read failure.";
      strm.clear(std::ios::badbit);
      return strm;
    }
    s_.resize(sz);
    for(IntType i = 0; i < sz; i++) {
      ReadType(strm, &(s_[i]));
    }
    return strm;
  }

  // This is used in OpenFst for binary I/O.  This is OpenFst-style,
  // not Kaldi-style, I/O.
  ostream &Write(ostream &strm) const {
    w_.Write(strm);
    if(strm.fail()){ return strm; }
    IntType sz = static_cast<IntType>(s_.size());
    for(IntType i = 0; i < sz; i++)
      WriteType(strm, &(s_[i]));
  }        
};

template<class WeightType, class IntType>
inline bool operator==(const CompactLatticeWeightTpl<WeightType,IntType> &w1,
                       const CompactLatticeWeightTpl<WeightType,IntType> &w2 ) {
  return (w1.w_ == w2.w_ && w1.s_ == w2.s_);
}

// Compare is not part of the standard for weight types, but used internally for
// efficiency.  The comparison here first compares the weight; if this is the
// same, it compares the string.  The comparison on strings is: first compare
// the length, if this is the same, use lexicographical order.  We can't just
// use the lexicographical order because this would destroy the distributive
// property of multiplication over addition, taking into account that addition
// uses Compare.  The string element of "Compare" isn't super-important in
// practical terms; it's only needed to ensure that Plus always give consistent
// answers and is symmetric.  It's essentially for tie-breaking, but we need to
// make sure all the semiring axioms are satisfied otherwise OpenFst might
// break.

template<class WeightType, class IntType>
inline int Compare(const CompactLatticeWeightTpl<WeightType,IntType> &w1,
                   const CompactLatticeWeightTpl<WeightType,IntType> &w2) {
  int c1 = Compare(w1.w_, w2.w_);
  if(c1 != 0) return c1;
  int l1 = w1.v_.length(), l2 = w2.v_.length();
  if(l1 < l2) return -1;
  else if(l1 > l2) return 1;
  for(int i = 0; i < l1; i++) {
    if(w1.v_[i] < w2.v_[i]) return -1;
    else if(w1.v_[i] > w2.v_[i]) return 1;
  }
  return 0;
}


template<class WeightType, class IntType>
inline CompactLatticeWeightTpl<WeightType,IntType> Plus(
    const CompactLatticeWeightTpl<WeightType,IntType> &w1,
    const CompactLatticeWeightTpl<WeightType,IntType> &w2) {
  return (Compare(w1,w2) >= 0 ? w1 : w2); 
}

template<class WeightType, class IntType>
inline CompactLatticeWeightTpl<WeightType,IntType> Times(
    const CompactLatticeWeightTpl<WeightType,IntType> &w1,
    const CompactLatticeWeightTpl<WeightType,IntType> &w2) {
  typedef WeightType T;
  WeightType w = Times(w1.w_, w2.w_);
  if(w.IsZero()) {
    return CompactLatticeWeightTpl<WeightType,IntType>::Zero();
    // special case to ensure zero is unique
  } else {
    std::vector<IntType> v;
    v.resize(w1.v_.size() + w2.v_.size());
    typename std::vector<IntType>::iterator iter = v.begin();
    iter = std::copy(w1.v_.begin(), w1.v_.end(), iter); // returns end of first range.
    std::copy(w2.v_.begin(), w2.v_.end(), iter);
    return CompactLatticeWeightTpl<WeightType,IntType>(w, v);
  }
}

template<class WeightType, class IntType>
inline CompactLatticeWeightTpl<WeightType,IntType> Divide(const CompactLatticeWeightTpl<WeightType,IntType> &w1,
                                                          const CompactLatticeWeightTpl<WeightType,IntType> &w2,
                                                          DivideType div) {
  if(w1 == WeightType::Zero()) { return CompactLatticeWeightTpl<WeightType,IntType>::Zero(); }
  WeightType w = Divide(w1.w_, w2.w_);

  const std::vector<IntType> v1 = w1.v_, v2 = w2.v_;
  if(v2.size() > v1.size()) {
    std::cerr << "Error in Divide (CompactLatticeWeighTpl): cannot divide, length mismatch.\n";
    exit(1);
  }
  typename std::vector<IntType>::const_iterator v1b = v1.begin(),
      v1e = v1.end(), v2b = v2.begin(), v2e = v2.end();
  if(div == DIVIDE_LEFT) {
    if(!std::equal(v2b, v2e, v1b)) { // v2 must be identical to first part of v1.
      std::cerr << "Error in Divide (CompactLatticeWeighTpl): cannot divide, data mismatch.\n";
      exit(1);
    }
    return CompactLatticeWeightTpl<WeightType,IntType>(
        w, std::vector<IntType>(v1b+(v2e-v2b), v1e)); // return last part of v1.
  } else if(div == DIVIDE_RIGHT) {
    if(!std::equal(v2b, v2e, v1e-(v2e-v2b))) { // v2 must be identical to last part of v1.
      std::cerr << "Error in Divide (CompactLatticeWeighTpl): cannot divide, data mismatch.\n";
      exit(1);
    }
    return CompactLatticeWeightTpl<WeightType,IntType>(
        w, std::vector<IntType>(v1b, v1e-(v2e-v2b))); // return first part of v1.

  } else {
    std::cerr << "Cannot divide CompactLatticeWeightTpl with DIVIDE_ANY.\n";
    exit(1);
  }
  return CompactLatticeWeightTpl<WeightType,IntType>::Zero(); // keep compiler happy.
}

} // end namespace fst

namespace kaldi {
// will import some things above...

typedef fst::LatticeWeightTpl<BaseFloat> LatticeWeight;

// careful: kaldi::int32 is not always the same C type as fst::int32
typedef fst::CompactLatticeWeightTpl<LatticeWeight, int32> CompactLatticeWeight;




} // namespace kaldi

#endif  // KALDI_LAT_KALDI_LATTICE_H_
