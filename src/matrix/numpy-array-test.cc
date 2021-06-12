// matrix/numpy-array-test.cc

// Copyright 2020   Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)

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

#include "matrix/numpy-array.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>


using namespace kaldi;

namespace {

template <typename Real>
void test(const char* filename, int dim) {
  NumpyArray<Real> arr;
  std::ifstream in(filename);
  arr.Read(in, true);

  KALDI_ASSERT(arr.NumElements() == 8);
  if (dim == 1) {
    KALDI_ASSERT(arr.Shape().size() == 1);
  } else {
    KALDI_ASSERT(arr.Shape().size() == 2);
    KALDI_ASSERT(arr.Shape()[0] == 2);
    KALDI_ASSERT(arr.Shape()[1] == 4);
  }

  int i = 0;
  for (auto d : arr) {
    KALDI_ASSERT(i == d);
    i++;
  }
}

void test_numpy() {
  test<float>("test_data/float_vector.npy", 1);
  test<float>("test_data/float_matrix.npy", 2);

  test<double>("test_data/double_vector.npy", 1);
  test<double>("test_data/double_matrix.npy", 2);

  test<float>("test_data/float_vector_big_endian.npy", 1);
  test<float>("test_data/float_matrix_big_endian.npy", 2);

  test<double>("test_data/double_vector_big_endian.npy", 1);
  test<double>("test_data/double_matrix_big_endian.npy", 2);

  NumpyArray<float> a;
  std::ifstream in("test_data/float_matrix.npy");
  a.Read(in, true);

  std::ofstream os("numpy-array.tmp", std::ios::binary);
  a.Write(os, true);
  os.close();

  // Make sure to use external double quotes for cmd.exe on Windows:
  // `python -c "print('Hi')"` good, `python -c 'print("Hi")'` bad.
  int rc = std::system("python3 -c \"import numpy\"");
  if (rc != 0) {
    rc = 0;
    KALDI_LOG << "python3 or numpy unavailable, array file load test SKIPPED";
  } else {
    rc = std::system("python3 -c \"import numpy; "
                     "numpy.load('numpy-array.tmp')\"");
    KALDI_LOG << "python3 numpy array file load test "
              << (rc == 0 ? "PASSED" : "FAILED");
  }
  std::remove("numpy-array.tmp");
  KALDI_ASSERT(rc == 0);
}

template <typename Real>
void test_vector() {
  Vector<Real> v(2);
  v(0) = 10;
  v(1) = 20;

  NumpyArray<Real> arr(v);

  KALDI_ASSERT(arr.NumElements() == v.Dim());
  int i = 0;
  for (auto d : arr) {
    KALDI_ASSERT(d == v(i));
    i += 1;
  }

  SubVector<Real> sub = arr;
  KALDI_ASSERT(sub.Dim() == v.Dim());
  sub(0) = 1;
  sub(1) = 2;

  KALDI_ASSERT(arr[0] == 1);
  KALDI_ASSERT(arr[1] == 2);
}

template <typename Real>
void test_matrix() {
  Matrix<Real> m(1, 2);
  m(0, 0) = 10;
  m(0, 1) = 20;

  NumpyArray<Real> arr(m);

  KALDI_ASSERT(arr.NumElements() == m.NumRows() * m.NumCols());
  KALDI_ASSERT(arr.Shape().size() == 2);
  KALDI_ASSERT(arr.Shape()[0] == 1);
  KALDI_ASSERT(arr.Shape()[1] == 2);

  KALDI_ASSERT(arr[0] == 10);
  KALDI_ASSERT(arr[1] == 20);

  SubMatrix<Real> sub = arr;
  KALDI_ASSERT(sub.NumRows() == m.NumRows());
  KALDI_ASSERT(sub.NumCols() == m.NumCols());

  sub(0, 0) = 1;
  sub(0, 1) = 2;

  KALDI_ASSERT(arr[0] == 1);
  KALDI_ASSERT(arr[1] == 2);
}

template <typename Real>
void test_read_write() {
  Vector<Real> v(2);
  v(0) = 10;
  v(1) = 20;

  NumpyArray<Real> arr(v);

  std::stringstream os;
  arr.Write(os, true);

  NumpyArray<Real> a;
  a.Read(os, true);

  KALDI_ASSERT(a.NumElements() == 2);
  KALDI_ASSERT(a.Shape().size() == 1);
  KALDI_ASSERT(a.Shape()[0] == 2);
  KALDI_ASSERT(a[0] == 10);
  KALDI_ASSERT(a[1] == 20);
}

}  // namespace

int main() {
  test_numpy();

  test_vector<float>();
  test_vector<double>();

  test_matrix<float>();
  test_matrix<double>();

  test_read_write<float>();
  test_read_write<double>();

  return 0;
}
