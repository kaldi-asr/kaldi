// nnet/nnet-randomizer-test.cc

// Copyright 2013  Brno University of Technology (author: Karel Vesely)

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

#include "nnet/nnet-randomizer.h"

#include <numeric>
#include <vector>
#include <algorithm>

using namespace kaldi;
using namespace kaldi::nnet1;

//////////////////////////////////////////////////

template<class Real> 
static void InitRand(VectorBase<Real> *v) {
  for (MatrixIndexT i = 0;i < v->Dim();i++)
	(*v)(i) = RandGauss();
}

template<class Real> 
static void InitRand(MatrixBase<Real> *M) {
  do {
    for (MatrixIndexT i = 0;i < M->NumRows();i++)
      for (MatrixIndexT j = 0;j < M->NumCols();j++)
        (*M)(i, j) = RandGauss();
  } while (M->NumRows() != 0 && M->Cond() > 100);
}


template<class Real> 
static void AssertEqual(const VectorBase<Real> &A, const VectorBase<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i=0; i < A.Dim(); i++)
    KALDI_ASSERT(std::abs(A(i)-B(i)) < tol);
}


template<class RandomAccessIterator> 
static void AssertEqual(RandomAccessIterator begin1, RandomAccessIterator end1,
                        RandomAccessIterator begin2, RandomAccessIterator end2) {
  KALDI_ASSERT((end1 - begin1) == (end2 - begin2));
  KALDI_ASSERT(end1 > begin1);
  for ( ; begin1 < end1; ++begin1,++begin2) {
    KALDI_ASSERT(*begin1 == *begin2);
  }
}


//////////////////////////////////////////////////

void UnitTestRandomizerMask() {
  NnetDataRandomizerOptions c;
  RandomizerMask r;
  r.Init(c);
  const std::vector<int32>& m = r.Generate(5);
  KALDI_ASSERT(m.size() == 5);
  int32 sum_of_elems = std::accumulate(m.begin(),m.end(),0);
  KALDI_ASSERT(sum_of_elems == 4 + 3 + 2 + 1 + 0);
}

void UnitTestMatrixRandomizer() {
  Matrix<BaseFloat> m(1111,10);
  InitRand(&m);
  CuMatrix<BaseFloat> m2(m);
  // config
  NnetDataRandomizerOptions c;
  c.randomizer_size = 1000;
  c.minibatch_size = 100;
  // randomizer
  MatrixRandomizer r;
  r.Init(c);
  r.AddData(m2);
  KALDI_ASSERT(r.IsFull());
  // create vector with consecutive indices
  std::vector<int32> mask(1111);
  for(int32 i=0; i<1111; i++) { mask[i]=i; }
  r.Randomize(mask); // no shuffling
  // make sure we get same data we put to randomizer
  int32 i=0;
  for( ; !r.Done(); r.Next(), i++) {
    KALDI_LOG << i;
    const CuMatrixBase<BaseFloat> &m3 = r.Value();
    Matrix<BaseFloat> m4(m3.NumRows(),m3.NumCols()); m3.CopyToMat(&m4);
    AssertEqual(m4,m.RowRange(i*c.minibatch_size,c.minibatch_size));
  }
  KALDI_ASSERT(i == 11); // 11 minibatches

  KALDI_LOG << "Filling for 2nd time";
  // try to fill buffer one more time, and empty it
  KALDI_ASSERT(!r.IsFull());
  r.AddData(m2);
  KALDI_ASSERT(r.IsFull());
  KALDI_ASSERT(r.NumFrames() == 11 + 1111);
  { // check last 11 rows were copied to the front in the buffer
    const CuMatrixBase<BaseFloat> &m3 = r.Value();
    Matrix<BaseFloat> m4(m3.NumRows(),m3.NumCols()); m3.CopyToMat(&m4);
    AssertEqual(m4.RowRange(0,11),m.RowRange(1100,11));
  }
  KALDI_ASSERT(!r.Done());
  for( ; !r.Done(); r.Next(), i++) {
    KALDI_LOG << i;
    const CuMatrixBase<BaseFloat> &m3 = r.Value();
  }
  KALDI_ASSERT(i == 22); // 22 minibatches
}

void UnitTestVectorRandomizer() {
  Vector<BaseFloat> v(1111);
  InitRand(&v);
  // config
  NnetDataRandomizerOptions c;
  c.randomizer_size = 1000;
  c.minibatch_size = 100;
  // randomizer
  VectorRandomizer r;
  r.Init(c);
  r.AddData(v);
  KALDI_ASSERT(r.IsFull());
  // create vector with consecutive indices
  std::vector<int32> mask(1111);
  for(int32 i=0; i<1111; i++) { mask[i]=i; }
  r.Randomize(mask); // no shuffling
  // make sure we get same data we put to randomizer
  int32 i=0;
  for( ; !r.Done(); r.Next(), i++) {
    KALDI_LOG << i;
    const VectorBase<BaseFloat> &v2 = r.Value();
    AssertEqual(v2, v.Range(i*c.minibatch_size,c.minibatch_size));
  }
  KALDI_ASSERT(i == 11); // 11 minibatches

  KALDI_LOG << "Filling for 2nd time";
  // try to fill buffer one more time, and empty it
  KALDI_ASSERT(!r.IsFull());
  r.AddData(v);
  KALDI_ASSERT(r.IsFull());
  KALDI_ASSERT(r.NumFrames() == 11 + 1111);
  { // check last 11 rows were copied to the front in the buffer
    const VectorBase<BaseFloat> &v2 = r.Value();
    AssertEqual(v2.Range(0,11),v.Range(1100,11));
  }
  KALDI_ASSERT(!r.Done());
  for( ; !r.Done(); r.Next(), i++) {
    KALDI_LOG << i;
    const VectorBase<BaseFloat> &v2 = r.Value();
  }
  KALDI_ASSERT(i == 22); // 22 minibatches
}

void UnitTestStdVectorRandomizer() {
  //prepare vector with some data
  std::vector<int32> v(1111);
  for (int32 i=0; i<v.size(); i++) {
    v.at(i) = i;
  }
  std::random_shuffle(v.begin(),v.end());

  // config
  NnetDataRandomizerOptions c;
  c.randomizer_size = 1000;
  c.minibatch_size = 100;
  // randomizer
  Int32VectorRandomizer r;
  r.Init(c);
  r.AddData(v);
  KALDI_ASSERT(r.IsFull());
  // create vector with consecutive indices
  std::vector<int32> mask(1111);
  for(int32 i=0; i<1111; i++) { mask[i]=i; }
  r.Randomize(mask); // no shuffling
  // make sure we get same data we put to randomizer
  int32 i=0;
  for( ; !r.Done(); r.Next(), i++) {
    KALDI_LOG << i;
    std::vector<int32> v2 = r.Value();
    AssertEqual(v2.begin(), v2.end(), v.begin()+(i*c.minibatch_size), v.begin()+((i+1)*c.minibatch_size));
  }
  KALDI_ASSERT(i == 11); // 11 minibatches

  KALDI_LOG << "Filling for 2nd time";
  // try to fill buffer one more time, and empty it
  KALDI_ASSERT(!r.IsFull());
  r.AddData(v);
  KALDI_ASSERT(r.IsFull());
  KALDI_ASSERT(r.NumFrames() == 11 + 1111);
  { // check last 11 rows were copied to the front in the buffer
    std::vector<int32> v2 = r.Value();
    AssertEqual(v2.begin(), v2.begin()+11, v.begin()+1100, v.begin()+1100+11);
  }
  KALDI_ASSERT(!r.Done());
  for( ; !r.Done(); r.Next(), i++) {
    KALDI_LOG << i;
    std::vector<int32> v2 = r.Value();
  }
  KALDI_ASSERT(i == 22); // 22 minibatches
}


int main() {
  UnitTestRandomizerMask();
  UnitTestMatrixRandomizer();
  UnitTestVectorRandomizer();
  UnitTestStdVectorRandomizer();
  
  std::cout << "Tests succeeded.\n";
}

