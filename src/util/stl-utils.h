// util/stl-utils.h

// Copyright 2009-2011  Microsoft Corporation  Arnab Ghoshal

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

#ifndef KALDI_UTIL_STL_UTILS_H_
#define KALDI_UTIL_STL_UTILS_H_

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "base/kaldi-common.h"

namespace kaldi {


/// Sorts and uniq's (removes duplicates) from a vector.
template<typename T>
inline void SortAndUniq(std::vector<T> *vec) {
  std::sort(vec->begin(), vec->end());
  Uniq(vec);
}


/// Returns true if the vector is sorted.
template<typename T>
inline bool IsSorted(const std::vector<T> &vec) {
  typename std::vector<T>::const_iterator iter = vec.begin(), end = vec.end();
  if (iter == end) return true;
  while (1) {
    typename std::vector<T>::const_iterator next_iter = iter;
    ++next_iter;
    if (next_iter == end) return true;  // end of loop and nothing out of order
    if (*next_iter < *iter) return false;
    iter = next_iter;
  }
}


/// Returns true if the vector is sorted and contains each element
/// only once.
template<typename T>
inline bool IsSortedAndUniq(const std::vector<T> &vec) {
  typename std::vector<T>::const_iterator iter = vec.begin(), end = vec.end();
  if (iter == end) return true;
  while (1) {
    typename std::vector<T>::const_iterator next_iter = iter;
    ++next_iter;
    if (next_iter == end) return true;  // end of loop and nothing out of order
    if (*next_iter <= *iter) return false;
    iter = next_iter;
  }
}


/// Removes duplicate elements from a sorted list.
template<typename T>
inline void Uniq(std::vector<T> *vec) {  // must be already sorted.
#ifdef KALDI_PARANOID
  assert(IsSorted(*vec));
#endif
  assert(vec);
  vec->erase(std::unique(vec->begin(), vec->end()), vec->end());
}

/// Copies the elements of a set to a vector.
template<class T>
void CopySetToVector(const std::set<T> &s, std::vector<T> *v) {
  // adds members of s to v, in sorted order from lowest to highest
  // (because the set was in sorted order).
  assert(v != NULL);
  v->resize(s.size());
  typename std::set<T>::const_iterator siter = s.begin(), send = s.end();
  typename std::vector<T>::iterator viter = v->begin();
  for (; siter != send; ++siter, ++viter) {
    *viter = *siter;
  }
}


/// Copies the (key, value) pairs in a map to a vector of pairs.
template<class A, class B>
void CopyMapToVector(const std::map<A, B> &m,
                     std::vector<std::pair<A, B> > *v) {
  assert(v != NULL);
  v->resize(m.size());
  typename std::map<A, B>::const_iterator miter = m.begin(), mend = m.end();
  typename std::vector<std::pair<A, B> >::iterator viter = v->begin();
  for (; miter != mend; ++miter, ++viter) {
    *viter = std::make_pair(miter->first, miter->second);
    // do it like this because of const casting.
  }
}

/// Copies the keys in a map to a vector.
template<class A, class B>
void CopyMapKeysToVector(const std::map<A, B> &m, std::vector<A> *v) {
  assert(v != NULL);
  v->resize(m.size());
  typename std::map<A, B>::const_iterator miter = m.begin(), mend = m.end();
  typename std::vector<A>::iterator viter = v->begin();
  for (; miter != mend; ++miter, ++viter) {
    *viter = miter->first;
  }
}

/// Copies the values in a map to a vector.
template<class A, class B>
void CopyMapValuesToVector(const std::map<A, B> &m, std::vector<B> *v) {
  assert(v != NULL);
  v->resize(m.size());
  typename std::map<A, B>::const_iterator miter = m.begin(), mend = m.end();
  typename std::vector<B>::iterator viter = v->begin();
  for (; miter != mend; ++miter, ++viter) {
    *viter = miter->second;
  }
}

/// Copies the keys in a map to a set.
template<class A, class B>
void CopyMapKeysToSet(const std::map<A, B> &m, std::set<A> *s) {
  assert(s != NULL);
  s->clear();
  typename std::map<A, B>::const_iterator miter = m.begin(), mend = m.end();
  for (; miter != mend; ++miter) {
    s->insert(s->end(), miter->first);
  }
}

/// Copies the values in a map to a set.
template<class A, class B>
void CopyMapValuesToSet(const std::map<A, B> &m, std::set<B> *s) {
  assert(s != NULL);
  s->clear();
  typename std::map<A, B>::const_iterator miter = m.begin(), mend = m.end();
  for (; miter != mend; ++miter)
    s->insert(s->end(), miter->second);
}


/// Copies the contents of a vector to a set.
template<class A>
void CopyVectorToSet(const std::vector<A> &v, std::set<A> *s) {
  assert(s != NULL);
  s->clear();
  typename std::vector<A>::const_iterator iter = v.begin(), end = v.end();
  for (; iter != end; ++iter)
    s->insert(s->end(), *iter);
  // s->end() is a hint in case v was sorted.  will work regardless.
}

/// Deletes any non-NULL pointers in the vector v, and sets
/// the corresponding entries of v to NULL
template<class A>
void DeletePointers(std::vector<A*> *v) {
  assert(v != NULL);
  typename std::vector<A*>::iterator iter = v->begin(), end = v->end();
  for (; iter != end; ++iter) {
    if (*iter != NULL) {
      delete *iter;
      *iter = NULL;  // set to NULL for extra safety.
    }
  }
}

/// Returns true if the vector of pointers contains NULL pointers.
template<class A>
bool ContainsNullPointers(const std::vector<A*> &v) {
  typename std::vector<A*>::const_iterator iter = v.begin(), end = v.end();
  for (; iter != end; ++iter)
    if (*iter == static_cast<A*> (NULL)) return true;
  return false;
}

/// Copies the contents a vector of one type to a vector
/// of another type.
template<typename A, typename B>
void CopyVectorToVector(const std::vector<A> &vec_in, std::vector<B> *vec_out) {
  assert(vec_out != NULL);
  vec_out->resize(vec_in.size());
  for (size_t i = 0; i < vec_in.size(); i++)
    (*vec_out)[i] = static_cast<B> (vec_in[i]);
}

/// A hashing function-object for vectors.
template<typename Int>
struct VectorHasher {  // hashing function for vector<Int>.
  size_t operator()(const std::vector<Int> &x) const {
    size_t ans = 0;
    typename std::vector<Int>::const_iterator iter = x.begin(), end = x.end();
    for (; iter != end; ++iter) {
      ans *= kPrime;
      ans += *iter;
    }
    return ans;
  }
  VectorHasher() {  // Check we're instantiated with an integer type.
    KALDI_ASSERT_IS_INTEGER_TYPE(Int);
  }
 private:
  static const int kPrime = 7853;
};

/// A hashing function object for strings.
struct StringHasher {  // hashing function for std::string
  size_t operator()(const std::string &str) const {
    size_t ans = 0;
    const char *c = str.c_str();
    for (; *c != '\0'; c++) {
      ans *= kPrime;
      ans += *c;
    }
    return ans;
  }
 private:
  static const int kPrime = 7853;
};

/// Reverses the contents of a vector.
template<typename T>
inline void ReverseVector(std::vector<T> *vec) {
  assert(vec != NULL);
  size_t sz = vec->size();
  for (size_t i = 0; i < sz/2; i++)
    std::swap( (*vec)[i], (*vec)[sz-1-i]);
}


}  // namespace kaldi

#endif  // KALDI_UTIL_STL_UTILS_H_

