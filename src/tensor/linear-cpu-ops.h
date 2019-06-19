// tensor/linear-cpu-ops.h

// Copyright      2019  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_TENSOR_LINEAR_SPECIAL_OPS_H_
#define KALDI_TENSOR_LINEAR_SPECIAL_OPS_H_ 1

#include "tensor/tensor.h"
#include "tensor/linear-special-ops.h"
#include "matrix/kaldi-blas.h"

// This Ops are more specialized forms of the Ops declared in linear-ops.h;
// these correspond to more specific combinations of Tensor shapes.  These Ops
// are only intended to be created from inside other more generic Ops.
namespace kaldi {
namespace tensor {

/**
   Does a += b for a and b both scalar, on CPU.
 */
template <class T>
class ScalarPlusEqScalarCpuOp<T>: public Op {

  ScalarPlusEqScalarOp(const Tensor &a, const Tensor &b): a_(a), b_(b) { }

  Op *Copy() {
    return new ScalarPlusEqScalar<T>(a_, b_);
  }

  void Do() {
    DebugNormalOp(a, kReadWrite, b_, kRead);
    *a_.GetData<T>() += *b_.GetData<T>();
  }

  Tensor a_;
  Tensor b_;
};


/**
   Does a += b for a and b both possibly-strided vectors (Stvector), on CPU.

   They must be normalized form, i.e. all axes trivial except raxis 0,
   and they must have the same dimension.

   This generic form of the template works for integer types (and would work,
   if used, for float and double).  We will separately instantiate this
   template for float and double, to use BLAS calls
*/
template <class T>
class StvectorPlusEqStvectorCpuOp<T>: public Op {

  StvectorPlusEqStvectorOp(const Tensor &a, const Tensor &b): a_(a), b_(b) { }

  int32 Properties() { return kConcreteOp; }

  Op *Copy() {
    return new StvectorPlusEqStvectorCpuOp<T>(a_, b_);
  }

  void Do() {
    DebugNormalOp(a, kReadWrite, b_, kRead);
    const Pattern &a_pattern = a_.Pattern(),
        &b_pattern = b_.Pattern();
    int32 dim = a_pattern.dims[0],
        a_stride = a_pattern.strides[0],
        b_stride = b_pattern.strides[0];

    bool uninitialized;
    T *a_data = a_.GetData<T>(&uninitialized),
        *b_data = a_.GetData<T>();
    if (uninitialized) {
      // This branch is an optimization to avoid writing, and reading, zeros
      // to/from memory.
      DebugNormalOp(a, kWrite, b_, kRead);
      // In future could look into unrolling this loop if it becomes a bottleneck.
      for (int32 i = 0; i < dim; i++)
        a_data[i * a_stride] = b_data[i * b_stride];
    } else {
      DebugNormalOp(a, kReadWrite, b_, kRead);
      // In future could look into unrolling this loop if it becomes a bottleneck.
      for (int32 i = 0; i < dim; i++)
        a_data[i * a_stride] += b_data[i * b_stride];
    }
  }
  Tensor a_;
  Tensor b_;
};


// override for float that uses BLAS
template <>
class StvectorPlusEqStvectorCpuOp<float>: public Op {
  SvectorPlusEqSvectorOp(const Tensor &a, const Tensor &b): a_(a), b_(b) { }
  int32 Properties() { return kConcreteOp; }
  Op *Copy() {
    return new SvectorPlusEqSvectorCpuOp<float>(a_, b_);
  }
  void Do() {
    const Pattern &a_pattern = a_.Pattern(),
        &b_pattern = b_.Pattern();
    bool uninitialized;
    float *a_data = a_.GetData<float>(&uninitialized),
        *b_data = a_.GetData<float>();
    if (uninitialized) {
      // This branch is an optimization to avoid writing, and reading, zeros
      // to/from memory.
      DebugNormalOp(a, kWrite, b_, kRead);
      cblas_scopy(a_pattern.dims[0], 1.0,
                  b_.GetData<float>(), b_pattern.strides[0],
                  a_.GetData<float>(), a_pattern.strides[0]);
    } else {
      DebugNormalOp(a, kReadWrite, b_, kRead);
      cblas_saxpy(a_pattern.dims[0], 1.0,
                  b_.GetData<float>(), b_pattern.strides[0],
                  a_.GetData<float>(), a_pattern.strides[0]);
    }
  }
  Tensor a_;
  Tensor b_;
};

// override for double that uses BLAS
template <>
class StvectorPlusEqStvectorCpuOp<double>: public Op {
  SvectorPlusEqSvectorOp(const Tensor &a, const Tensor &b): a_(a), b_(b) { }
  int32 Properties() { return kConcreteOp; }
  Op *Copy() {
    return new SvectorPlusEqSvectorCpuOp<double>(a_, b_);
  }
  void Do() {
    const Pattern &a_pattern = a_.Pattern(),
        &b_pattern = b_.Pattern();
    bool uninitialized;
    double *a_data = a_.GetData<double>(&uninitialized),
        *b_data = a_.GetData<double>();
    if (uninitialized) {
      // This branch is an optimization to avoid writing, and reading, zeros
      // to/from memory.
      DebugNormalOp(a, kWrite, b_, kRead);
      cblas_dcopy(a_pattern.dims[0], 1.0,
                  b_.GetData<double>(), b_pattern.strides[0],
                  a_.GetData<double>(), a_pattern.strides[0]);
    } else {
      DebugNormalOp(a, kReadWrite, b_, kRead);
      cblas_daxpy(a_pattern.dims[0], 1.0,
                  b_.GetData<double>(), b_pattern.strides[0],
                  a_.GetData<double>(), a_pattern.strides[0]);
    }
  }
  Tensor a_;
  Tensor b_;
};


/**
   Does a += b for a scalar and b a vector or strided vector, on CPU.
   (i.e. a += sum(b)).

   They must be normalized form, i.e. all axes trivial except raxis 0
   of b, and b must not have negative stride.  (This is to allow
   the BLAS template overrides).

   This generic form of the template works for integer types (and would work,
   if used, for float and double).  We will separately instantiate this
   template for float and double, to use BLAS calls.
*/
template <class T>
class ScalarPlusEqStvectorCpuOp<T>: public Op {

  StvectorPlusEqStvectorOp(const Tensor &a, const Tensor &b): a_(a), b_(b) { }

  int32 Properties() { return kConcreteOp; }

  Op *Copy() { return new ScalarPlusEqStvectorCpuOp<T>(a_, b_); }

  void Do() {
    DebugNormalOp(a, kReadWrite, b_, kRead);
    const Pattern &a_pattern = a_.Pattern(),
        &b_pattern = b_.Pattern();
    int32 dim = b_pattern.dims[0],
        b_stride = b_pattern.strides[0];
    T *a_data = a_.GetData<T>(),
        *b_data = a_.GetData<T>();
    T sum(0);
    // In future could look into unrolling this loop if it becomes a bottleneck.
    for (int32 i = 0; i < dim; i++)
      sum += b_data[i * b_stride];
    *a_data += sum;
  }
  Tensor a_;
  Tensor b_;
};



// Override for T = float.
template <>
class ScalarPlusEqStvectorCpuOp<float>: public Op {
  StvectorPlusEqStvectorOp(const Tensor &a, const Tensor &b): a_(a), b_(b) { }

  int32 Properties() { return kConcreteOp; }

  Op *Copy() { return new ScalarPlusEqStvectorCpuOp<float>(a_, b_); }

  void Do() {
    DebugNormalOp(a, kReadWrite, b_, kRead);
    const Pattern &a_pattern = a_.Pattern(),
        &b_pattern = b_.Pattern();
    int32 dim = b_pattern.dims[0],
        b_stride = b_pattern.strides[0];
    float *a_data = a_.GetData<T>(),
        *b_data = a_.GetData<T>();
    *a_data += cblas_sasum(dim, b_data, b_stride);
  }
  Tensor a_;
  Tensor b_;
};

// Override for T = double
template <>
class ScalarPlusEqStvectorCpuOp<double>: public Op {
  ScalarPlusEqStvectorOp(const Tensor &a, const Tensor &b): a_(a), b_(b) { }

  int32 Properties() { return kConcreteOp; }

  Op *Copy() { return new ScalarPlusEqStvectorCpuOp<double>(a_, b_); }

  void Do() {
    DebugNormalOp(a, kReadWrite, b_, kRead);
    const Pattern &a_pattern = a_.Pattern(),
        &b_pattern = b_.Pattern();
    int32 dim = b_pattern.dims[0],
        b_stride = b_pattern.strides[0];
    double *a_data = a_.GetData<T>(),
        *b_data = a_.GetData<T>();
    *a_data += cblas_dasum(dim, b_data, b_stride);
  }
  Tensor a_;
  Tensor b_;
};

/**
   Operation doing a += b with a a vector and b a scalar.  (I.e. add
   a constant elementwise to a vector).

   May not be used if a and b overlap.
*/
template <class T>
class StvectorPlusEqScalarCpuOp<T>: public Op {
  StvectorPlusEqScalarOp(const Tensor &a, const Tensor &b): a_(a), b_(b) { }

  int32 Properties() { return kConcreteOp; }

  Op *Copy() { return new StvectorPlusEqScalarCpuOp<T>(a_, b_); }

  void Do() {
    const Pattern &a_pattern = a_.Pattern(),
        &b_pattern = b_.Pattern();
    int32 dim = a_pattern.dims[0],
        a_stride = a_pattern.strides[0];
    bool uninitialized;
    T *a_data = a_.GetData<T>(&uninitialized),
        *b_data = a_.GetData<T>();

    if (uninitialized) {
      DebugNormalOp(a, kWrite, b_, kRead);
      T b = *b_data;
#pragma unroll (4)
      for (int32 i = 0; i < dim; i++)
        a_data[i * a_stride] = b;
    } else {
      DebugNormalOp(a, kReadWrite, b_, kRead);
      T b = *b_data;
#pragma unroll (4)
      for (int32 i = 0; i < dim; i++)
        a_data[i * a_stride] += b;
    }
  }
  Tensor a_;
  Tensor b_;
};


/**
   Operation doing a += b with a a vector or strided vector (implicitly,
   interpreted as a row vector) and b a matrix, so it sums up the rows of the
   matrix.

   May not be used if a and b overlap.
*/
template <class T>
class StvectorPlusEqMatrixCpuOp<T>: public Op {
  StvectorPlusEqMatrixCpuOp(const Tensor &a, const Tensor &b): a_(a), b_(b) { }

  int32 Properties() { return kConcreteOp; }

  Op *Copy() { return new StvectorPlusEqMatrixCpuOp<T>(a_, b_); }

  void Do() {
    const Pattern &a_pattern = a_.Pattern(),
        &b_pattern = b_.Pattern();
    int32 a_dim = a_pattern.dims[0],
        b_num_cols = b_pattern.dims[0],
        b_num_rows = b_pattern.dims[1],
        a_stride = a_pattern.strides[0],
        b_stride =  b_pattern.strides[1];
    KALDI_PARANOID_ASSERT(b_pattern.strides[0] == 1 &&
                          a_dim == b_num_cols);

    bool uninitialized;
    T *a_data = a_.GetData<T>(&uninitialized),
        *b_data = a_.GetData<T>();

    if (uninitialized) {
      DebugNormalOp(a, kWrite, b_, kRead);
      T b = *b_data;
#pragma unroll (4)
      for (int32 i = 0; i < dim; i++)
        a_data[i * a_stride] = b;
    } else {
      DebugNormalOp(a, kReadWrite, b_, kRead);
      T b = *b_data;
#pragma unroll (4)
      for (int32 i = 0; i < dim; i++)
        a_data[i * a_stride] += b;
    }
  }
  Tensor a_;
  Tensor b_;
};




}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_LINEAR_OPS_H_
