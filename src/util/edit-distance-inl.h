// util/edit-distance-inl.h

// Copyright 2009-2011  Microsoft Corporation;  Haihua Xu;  Yanmin Qian

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_UTIL_EDIT_DISTANCE_INL_H_
#define KALDI_UTIL_EDIT_DISTANCE_INL_H_
#include <algorithm>
#include <utility>
#include <vector>
#include "util/stl-utils.h"

namespace kaldi {

template<class T>
int32 LevenshteinEditDistance(const std::vector<T> &a,
                              const std::vector<T> &b) {
  // Algorithm:
  //  write A and B for the sequences, with elements a_0 ..
  //  let |A| = M and |B| = N be the lengths, and have
  //  elements a_0 ... a_{M-1} and b_0 ... b_{N-1}.
  //  We are computing the recursion
  //     E(m, n) = min(  E(m-1, n-1) + (1-delta(a_{m-1}, b_{n-1})),
  //                    E(m-1, n),
  //                    E(m, n-1) ).
  //  where E(m, n) is defined for m = 0..M and n = 0..N and out-of-
  //  bounds quantities are considered to be infinity (i.e. the
  //  recursion does not visit them).

  // We do this computation using a vector e of size N+1.
  // The outer iterations range over m = 0..M.

  int M = a.size(), N = b.size();
  std::vector<int32> e(N+1);
  std::vector<int32> e_tmp(N+1);
  // initialize e.
  for (size_t i = 0; i < e.size(); i++)
    e[i] = i;
  for (int32 m = 1; m <= M; m++) {
    // computing E(m, .) from E(m-1, .)
    // handle special case n = 0:
    e_tmp[0] = e[0] + 1;

    for (int32 n = 1; n <= N; n++) {
      int32 term1 = e[n-1] + (a[m-1] == b[n-1] ? 0 : 1);
      int32 term2 = e[n] + 1;
      int32 term3 = e_tmp[n-1] + 1;
      e_tmp[n] = std::min(term1, std::min(term2, term3));
    }
    e = e_tmp;
  }
  return e.back();
}

//
struct Particle {
  int32 ins_num;
  int32 del_num;
  int32 sub_num;
  int32 total_cost;  // minimum total cost to the current alignment.
  int32 a_i;
  int32 b_i;
};
// Note that both hyp and ref should not contain noise word in
// the following implementation.
// The current algorithm is symmetric, so `hyp` and `ref` are named `a` and `b`

template<class T>
int32 LevenshteinEditDistance(const std::vector<T> &a,
                              const std::vector<T> &b,
                              int32 *ins, int32 *del, int32 *sub) {
  size_t A = a.size(), B = b.size();
  //There are A+B+1 diagonals, and at most that many Particles.
  //We denote uninitialized Particle as located in imaginary location (a_i,b_i)=(-1,-1).
  std::vector<Particle> diagonals(A+B+1),future_diagonals;
  for (int d = 0;d < A+B+1;++d) {
    Particle &p = diagonals[d];
    p.b_i = p.a_i = -1;
  }
  future_diagonals = diagonals;
  //We start with a single Particle in the upper left corner: (0,0).
  //Mapping between position (a_i,b_i) and diagonal index is index=a_i-b_i+B.
  Particle &start = diagonals[0-0+B];
  start.a_i = start.b_i = 0;
  start.total_cost = 0;
  start.del_num = 0;
  start.ins_num = 0;
  start.sub_num = 0;
  //We maintain the list of indexs of active particles, which never shrinks, and initially contains the single particle.
  std::vector<int> diagonals_indexes;
  diagonals_indexes.push_back(0-0+B);
  while (true) {
    //We progress all Particles along their diagonals as far as possible without incurring any cost.
    for (size_t d = diagonals_indexes.size();d--;) {
      Particle &p = diagonals[diagonals_indexes[d]];
      while (p.a_i < A && p.b_i < B && a[p.a_i] == b[p.b_i]) {
        p.a_i++;
        p.b_i++;
      }
      if (p.a_i == A && p.b_i == B) {
        *ins = p.ins_num;
        *del = p.del_num;
        *sub = p.sub_num;
        return p.total_cost;
      }
    }
    //None of particles can go any further without paying.
    //Moving one cell in any direction will cost 1 unit.
    //There are three interesting directions which we will consider:
    //Case 1) along the diagonal (a_i+1,b_i+1), which increases sub_num
    //Case 2) to the next diagonal (a_i+1,b_i), which increases del_num
    //Case 3) to the previous diagonal (a_i,b_i-1), which increases ins_num
    //Therefore, each old particle in diagonals[] can spawn at most three new particles in future_particles[].
    //We merge these propositions, and maintain only the best particle for each diagonal.
    //Case 1)
    for (size_t d = diagonals_indexes.size();d--;) {
      const int diagonal_index = diagonals_indexes[d];
      Particle &future = future_diagonals[diagonal_index];
      future = diagonals[diagonal_index];
      future.total_cost++;
      if (future.a_i < A && future.b_i < B) {
        future.a_i++;
        future.b_i++;
        future.sub_num++;
      }
    }
    for (size_t d = diagonals_indexes.size();d--;) {
      const int diagonal_index = diagonals_indexes[d];
      const Particle &p = diagonals[diagonal_index];
      //Case 2)
      if (p.a_i < A) {
        const int nbr_diagonal_index = diagonal_index+1;
        assert(nbr_diagonal_index < A+B+1);
        Particle &nbr = future_diagonals[nbr_diagonal_index];
        if (nbr.b_i == -1) {
          //uninitialized particle
          diagonals_indexes.push_back(nbr_diagonal_index);
        } else {
          //a particle at this diagonal is already found, it should be from the future generation
          assert(nbr.total_cost == p.total_cost+1);
        }
        if (nbr.b_i < p.b_i || (nbr.b_i == p.b_i && nbr.sub_num < p.sub_num) ) {
          nbr = p;
          nbr.a_i++;
          nbr.del_num++;
          nbr.total_cost++;
        }
      }
      //Case 3)
      if (p.b_i < B) {
        const int nbr_diagonal_index = diagonal_index-1;
        assert(0 <= nbr_diagonal_index);
        Particle &nbr = future_diagonals[nbr_diagonal_index];
        if (nbr.a_i == -1) {
          diagonals_indexes.push_back(nbr_diagonal_index);
        } else {
          assert(nbr.total_cost == p.total_cost+1);
        }
        if (nbr.a_i < p.a_i || (nbr.a_i == p.a_i && nbr.sub_num < p.sub_num) ) {
          nbr = p;
          nbr.b_i++;
          nbr.ins_num++;
          nbr.total_cost++;
        }
      }
    }
    swap(diagonals,future_diagonals);
  }
}



template<class T>
int32 LevenshteinAlignment(const std::vector<T> &a,
                           const std::vector<T> &b,
                           T eps_symbol,
                           std::vector<std::pair<T, T> > *output) {
  // Check inputs:
  {
    KALDI_ASSERT(output != NULL);
    for (size_t i = 0; i < a.size(); i++) KALDI_ASSERT(a[i] != eps_symbol);
    for (size_t i = 0; i < b.size(); i++) KALDI_ASSERT(b[i] != eps_symbol);
  }
  output->clear();
  // This is very memory-inefficiently implemented using a vector of vectors.
  size_t M = a.size(), N = b.size();
  size_t m, n;
  std::vector<std::vector<int32> > e(M+1);
  for (m = 0; m <=M; m++) e[m].resize(N+1);
  for (n = 0; n <= N; n++)
    e[0][n]  = n;
  for (m = 1; m <= M; m++) {
    e[m][0] = e[m-1][0] + 1;
    for (n = 1; n <= N; n++) {
      int32 sub_or_ok = e[m-1][n-1] + (a[m-1] == b[n-1] ? 0 : 1);
      int32 del = e[m-1][n] + 1;  // assumes a == ref, b == hyp.
      int32 ins = e[m][n-1] + 1;
      e[m][n] = std::min(sub_or_ok, std::min(del, ins));
    }
  }
  // get time-reversed output first: trace back.
  m = M;
  n = N;
  while (m != 0 || n != 0) {
    size_t last_m, last_n;
    if (m == 0) {
      last_m = m;
      last_n = n-1;
    } else if (n == 0) {
      last_m = m-1;
      last_n = n;
    } else {
      int32 sub_or_ok = e[m-1][n-1] + (a[m-1] == b[n-1] ? 0 : 1);
      int32 del = e[m-1][n] + 1;  // assumes a == ref, b == hyp.
      int32 ins = e[m][n-1] + 1;
      // choose sub_or_ok if all else equal.
      if (sub_or_ok <= std::min(del, ins)) {
        last_m = m-1;
        last_n = n-1;
      } else {
        if (del <= ins) {  // choose del over ins if equal.
          last_m = m-1;
          last_n = n;
        } else {
          last_m = m;
          last_n = n-1;
        }
      }
    }
    T a_sym, b_sym;
    a_sym = (last_m == m ? eps_symbol : a[last_m]);
    b_sym = (last_n == n ? eps_symbol : b[last_n]);
    output->push_back(std::make_pair(a_sym, b_sym));
    m = last_m;
    n = last_n;
  }
  ReverseVector(output);
  return e[M][N];
}


}  // end namespace kaldi

#endif  // KALDI_UTIL_EDIT_DISTANCE_INL_H_
