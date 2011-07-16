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

class LatticeWeight {
 public:
  typedef typename kaldi::BaseFloat T; // normally float.
  typedef LatticeWeight ReverseWeight;
  
  LatticeWeight() { }

  LatticeWeight(T a, T b): a_(a), b_(b) {}

  LatticeWeight(const LatticeWeight &other): a_(other.a_), b_(other.b_) { }

  LatticeWeight &operator=(const LatticeWeight &w) {
    a_ = w.a_;
    b_ = w.b_;
    return *this;
  }

  static const LatticeWeight Zero() {
    return LatticeWeight(FloatLimits<T>::kPosInfinity, FloatLimits<T>::kPosInfinity);
  }

  static const LatticeWeight One() {
    return LatticeWeight(0.0, 0.0);
  }
  
  static const string &Type() {
    static const string type = "lattice";
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
  
  LatticeWeight Quantize(float delta = kDelta) const {
    if(a_+b_ == FloatLimits<T>::kNegInfinity) {
      return LatticeWeight(FloatLimits<T>::kNegInfinity,FloatLimits<T>::kNegInfinity);
    } else if(a_+b_ == FloatLimits<T>::kPosInfinity) {
      return LatticeWeight(FloatLimits<T>::kPosInfinity,FloatLimits<T>::kPosInfinity);
    } else if(a_+b_ != a_+b_) { // NaN
      return LatticeWeight(a_+b_, a_+b_);
    } else {
      return LatticeWeight(floor(a_/delta + 0.5F)*delta, floor(b_/delta + 0.5F) * delta);
    }
  }

  // This is used in OpenFst for binary I/O.  This is OpenFst-style,
  // not Kaldi-style, I/O.
  istream &Read(istream &strm) {
    // Always read/write as float, even if T is double,
    // so we can use OpenFst-style read/write and still maintain
    // compatibility when compiling with different BaseFloats.
    float a,b;
    ReadType(strm, &a);
    ReadType(strm, &b);
    a_ = static_cast<T>(a);
    b_ = static_cast<T>(b);
    WriteType(strm, a);
    return strm;
  }


  // This is used in OpenFst for binary I/O.  This is OpenFst-style,
  // not Kaldi-style, I/O.
  ostream &Write(ostream &strm) const {
    float a(a_), b(b_);
    WriteType(strm, a);
    WriteType(strm, b);
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

inline bool operator==(const LatticeWeight &w1,
                       const LatticeWeight &w2 ) {
  return (w1.a_ == w2.a_ && w1.b_ == w2.b_);
}


// We define a Compare function LatticeWeight even though it's
// not required by the semiring standard-- it's just more efficient
// to do it this way rather than using the NaturalLess template.

/// Compare returns -1 if w1 < w2, +1 if w1 > w2, and 0 if w1 == w2.
inline int Compare (const  LatticeWeight &w1,
                    const LatticeWeight &w2) {
  LatticeWeight::T f1 = w1.a_ + w1.b_,
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


inline LatticeWeight Plus(const LatticeWeight &w1,
                          const LatticeWeight &w2) {
  return (Compare(w1,w2) >= 0 ? w1 : w2); //
}

inline LatticeWeight Times(const LatticeWeight &w1,
                           const LatticeWeight &w2) {
  return LatticeWeight(w1.a_+w2.a_, w1.b_+w2.b_);  
}

// divide w1 by w2 (on left/right/any doesn't matter as
// commutative).
inline LatticeWeight Divide(const LatticeWeight &w1,
                            const LatticeWeight &w2,
                            DivideType typ = DIVIDE_ANY) {
  typedef LatticeWeight::T T;
  T a = w1.a - w2.a, b = w1.a - w2.a;
  if(a!=a || b!=b || a == FloatLimits<T>::kNegInfinity
     || b == FloatLimits<T>::kNegInfinity) {
    KALDI_WARN << "LatticeWeight::Divide, NaN or invalid number produced. "
               << "[dividing by zero?]  Returning zero.";
    return LatticeWeight::Zero();
  }
  if(a == FloatLimits<T>::kPosInfinity ||
     b == FloatLimits<T>::kPosInfinity)
    return LatticeWeight::Zero(); // problems if only one is infinite.
  return LatticeWeight(a, b);
}

inline bool ApproxEqual(const LatticeWeight &w1,
                        const LatticeWeight &w2,
                        float delta = kDelta) {
  return (fabs(w1.a_ - w2.a_) <= delta && fabs(w1.b_ - w2.b_) <= delta);
}

inline ostream &operator <<(ostream &strm, LatticeWeight &w1) {
  typedef LatticeWeight::T T;
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

inline istream &operator >>(istream &strm, LatticeWeight &w1) {
  // TODO...  
}



// CompactLattice will be an acceptor (accepting the words/output-symbols),
// with the weights and input-symbol-seqs on the arcs.

class CompactLatticeWeight {
 public:
  LatticeWeight w_;
  vector<int32> s_; // watch out: fst::int32 and kaldi::int32 are not always compatible.

  // Plus is like LexicographicWeight on the pair (w_, s_), but where we
  // use standard lexicographic order on s_ [this is not the same as
  // NaturalLess on the StringWeight equivalent, which does not define a
  // total order].
  // Times, Divide obvious... (support both left & right division..)
  // CommonDivisor would need to be coded separately.

  CompactLatticeWeight() { }

  CompactLatticeWeight(const LatticeWeight &w, const std::vector<int32> &s):
      w_(w), s_(s) { }

  CompactLatticeWeight(const LatticeWeight &w, const std::vector<int32> &s):
      w_(w), s_(s) { }

  // Note: LatticeWeight::T is kaldi::BaseFloat which is normally float.
  CompactLatticeWeight(LatticeWeight::T a, LatticeWeight::T b,
                       const std::vector<int32> &s):
      w_(a,b), s_(s) { }
  
  CompactLatticeWeight &operator=(const CompactLatticeWeight &w):
      w_(w.w_), s(w.s_) { }

  static const CompactLatticeWeight Zero() {
    return CompactLatticeWeight(LatticeWeight::Zero(), std::vector<int32>());
  }

  static const CompactLatticeWeight One() {
    return CompactLatticeWeight(LatticeWeight::One(), std::vector<int32>());
  }

  static const string &Type() {
    static const string type = "compactlattice";
    return type;
  }

  bool Member() const {
    // a semiring has only one zero, this is the important property
    // we're trying to maintain here.  So force s_ to be empty if
    // w_ == zero.
    if(!w_.Member()) return false;
    if(w_ == LatticeWeight::Zero())
      return s_.empty();
    else
      return true;
  }

  CompactLatticeWeight Quantize(float delta = kDelta) const {
    return CompactLatticeWeight(w_.Quantize(delta), s_);
  }

  // This is used in OpenFst for binary I/O.  This is OpenFst-style,
  // not Kaldi-style, I/O.
  istream &Read(istream &strm) {
    w_.Read(strm);
    if(strm.fail()){ return strm; }
    int32 sz;
    ReadType(strm, &sz);
    if(strm.fail()){ return strm; }
    KALDI_ASSERT(sz>=0);
    v_.resize(sz);
    for(int32 i = 0; i < sz; i++) {
      ReadType(strm, &(v_[i]));
    }
    return strm;
  }

  // This is used in OpenFst for binary I/O.  This is OpenFst-style,
  // not Kaldi-style, I/O.
  ostream &Write(ostream &strm) const {
    w_.Write(strm);
    if(strm.fail()){ return strm; }
    int32 sz = static_cast<int32>(v_.size());
    for(int32 i = 0; i < sz; i++)
      WriteType(strm, &(v_[i]));
  }        
};

inline bool operator==(const CompactLatticeWeight &w1,
                       const CompactLatticeWeight &w2 ) {
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

inline int Compare(const CompactLatticeWeight &w1,
                   const CompactLatticeWeight &w2) {
  int c1 = Compare(w1.w_, w2.w_);
  if(c1 != 0) return c1;
  int l1 = w1.v_.length(), l2 = w2.v_.length();
  if(l1 < l2) return -1;
  else if(l1 > l2) return 1;
  for(int i = 0; i < l1; i++) {
    if(l1[i] < l2[i]) return -1;
    else if(l1[i] > l2[i]) return 1;
  }
  return 0;
}


inline CompactLatticeWeight Plus(const CompactLatticeWeight &w1,
                                 const CompactLatticeWeight &w2) {
  return (Compare(w1,w2) >= 0 ? w1 : w2); 
}

inline CompactLatticeWeight Times(const CompactLatticeWeight &w1,
                                  const CompactLatticeWeight &w2) {
  typedef LatticeWeight::T T;
  if(w1.w_.a_ == FloatLimits<T>::kPosInfinity
     || w2.w_.a_ == FloatLimits<T>::kPosInfinity) // if either w1 or w2 are zero...
    return CompactLatticeWeight::Zero(); // special case to ensure zero is unique
  std::vector<int32> v;
  v.reserve(w1.v_.size() + w2.v_.size());
  std::vector<int32>::iterator iter = v.begin();
  iter = std::copy(w1.v_.begin(), w1.v_.end(), iter); // returns end of first range.
  std::copy(w2.v_.begin(), s2.v_.end(), iter);
  return CompactLatticeWeight(Times(w1.w_, w2.w_), v);
}

inline CompactLatticeWeight Divide(const CompactLatticeWeight &w1,
                                   const CompactLatticeWeight &w2,
                                   DivideType div) {
  
  
}



namespace kaldi {
// will import some things above...






} // namespace kaldi

#endif  // KALDI_LAT_KALDI_LATTICE_H_
