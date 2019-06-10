// tensor/linear-ops.cc

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

#include "tensor/linear-ops.h"

namespace kaldi {
namespace tensor {

void PlusEqOp::Expand(std::vector<std::unique_ptr<Op> > *ops) const {

  if (ReferenceMode() && a_.DeviceType() == kCpuDevice) {
    // In reference mode, always use the reference implementation.
    Op *ans;
    SET_TO_TEMPLATED_CPU_OP_ALL(ans, a_.Dtype(), a_, b_);
    return ans;
  }

  // The generic implementation requires us to first normalize the patterns.
  Pattern a_pattern = a_.Impl().pattern,
      b_pattern = b_.Impl().pattern;
  NormalizePatterns({a_pattern, b_pattern});

  KALDI_ASSERT(Compatible(a_, b_));  // dtype and device, check they match.

  Tensor a(a_), b(b_);

  if (a_pattern != a_.Impl().pattern)
    a = WithPattern(a, a_pattern);
  if (b_pattern != b_.Impl().pattern)
    b = WithPattern(b, b_pattern);

  /*
    The case-statement values in the switch statement below may be interpreted
    in groups of 3 hex characters, are 0xAAABBB, pertaining to Tensors a and b
    respectively.  See GetPatternCode() in pattern-utils.h for documentation on
    the meanings of the values and our notation with X,x,1.

  */
  int64 combined_code = CombineCodes(a_pattern.GetCode(),
                                     b_pattern.GetCode());

  Op *new_op;

  /*
    The case-statement values in the switch statement below may be interpreted
    in groups of 3 hex characters, are 0xAAABBB, pertaining to Tensors a and b.
    See ComputePatternCode() in pattern-utils.h for documentation on the meanings of
    the values and our notation with X,x,1.
       Quick legend:
             X means dim >1, stride = 1
             x means dim >1, stride != 1
             1 means dim == 1, stride = 0.
                 (Note: the numbers in case-statements below exclude negative
                 strides because bit 11 of the 12-bit chunks would be set if
                 there were a negative stride).
   */

  // We are doing a += b.
  switch(combined_code) {
    // A scalar += scalar,
    case 0x000000:   // () +=  ()
      SET_TO_TEMPLATED_OP_REAL(new_op, a.Dtype(), a.DeviceType(), ScalarPlusEqScalarOp, a, b);
      break;
    // We may split apart some of the following cases in future.
    // They all represent, vector += vector.
    case 0x101101:  //  (X) += (X)
    case 0x001001:  //  (x) += (x)
    case 0x101001:  //  (X) += (x)
    case 0x001101:  //  (X) += (x)
      SET_TO_TEMPLATED_OP_REAL(new_op, a.Dtype(), a.DeviceType(), StvectorPlusEqStvectorOp, a, b);
      break;
    // Scalar += (sum of) vector or strided vector
    case 0x000101:  //  () += (X)
    case 0x000001:  //  () += (X)
      SET_TO_TEMPLATED_OP_REAL(new_op, a.Dtype(), a.DeviceType(), ScalarPlusEqStvectorOp, a, b);
      break;
    // vector or strided vector += scalar.
    // We could later split apart the strided and non-strided cases.
    case 0x101000:  //  (x) += ()
    case 0x001000:  //  (X) += ()
      SET_TO_TEMPLATED_OP_REAL(new_op, a.Dtype(), a.DeviceType(), StvectorPlusEqScalarOp, a, b);
      break;
    // scalar += matrix
    case 0x000103: { // () += (xX)
      int32 num_rows = b.Pattern().dims[1];
      // Create a temporary- a column vector, which is what we call
      // a vector whose nontrivial axis is raxis 1 instead of raxis 0.
      Tensor temp({num_rows, 1}, {a.Dtype(), a.Device()});
      Op *temp_op;
      // Below we do temp += b.  We could use PlusEqOp for this and also for the
      // following reduction, but doing it this way avoids an unnecessary layer
      // of expansion.
      SET_TO_TEMPLATED_OP_REAL(temp_op, a.Dtype(), a.DeviceType(),
                               ColVectorEqMatrixOp, temp, b);
      ops->push_back(temp_op);
      // Normalize the temporary vector so its nontrivial axis is raxis 0, by
      // removing the current raxis 0 and having current raxis 1 shift down.
      Tensor temp_normalized = Squeeze(temp, 0);
      SET_TO_TEMPLATED_OP_REAL(new_op, a.Dtype(), a.DeviceType(),
                               ScalarPlusEqStvectorOp, a, temp_normalized);
    }


    default:
      // Later we can add a more generic implementation that handles arbitrary
      // patterns.
      KALDI_ERR << "Unhandled code: " << std::hex << combined_code;
  }
  ops->push_back(new_op);
}



inline static void AddProductScalar3(
    float alpha, float beta,
    const TensorImpl &a, const TensorImpl &b, const TensorImpl *c) {
  switch (a.device.device_type) {
    case kCpuDevice:
      AddProductScalar3Cpu(alpha, beta, a, b, c);
      return;
#ifdef HAVE_CUDA
    case kCudaDevice:
      AddProductScalar3Gpu(alpha, beta, a, b, c);
      return;
#endif
    default:
      KALDI_ERR << "Unsupported device type " << a.ToString();
  }
}


void AddProduct(float alpha, float beta,
                const TensorImpl &a, const TensorImpl &b, const TensorImpl *c){

  if (a.pattern.code < b.pattern.code) {
    // Ensure, via a recursion, that a.pattern.code >= b.pattern.code.
    // This avoids us having to test for the swapped versions of the patterns.
    AddProduct(alpha, beta, b, a, c);
    return;
  }

  CheckDeviceAndDtype(a, b, *c);


  int64 combined_code = CombineCodes(a.pattern.code, b.pattern.code,
                                     c->pattern.code);

  /*
    The case-statement values in the switch statement below may be
    interpreted in groups of 3 hex characters, are 0xAAABBBCCC,
    pertaining to Tensors a, b and c respectively.  See
    GetPatternCode() in pattern-utils.h for documentation on
    the meanings of the values and our notation with X,x,1.
   */
  switch(combined_code) {
    case 0x000000000:
      // () * () -> ()
      // scalar * scalar -> scalar
      AddProductScalar3(a, b, c);
      return;
    case 0x101000101:
      //  (X) * ()-> (X)
      // vector * scalar -> vector
      AddProductVecScalarVec(a, b, c);
      return;
    case 0x101101101:
      // (X) * (X) -> (X)
      // vector .* vector -> vector
      AddProductVec3(a, b, c);
      return;
    case 0x103101202:
      // (x,X) * (X)  -> (X,1)
      // vector * matrix -> vector.unsqueeze(-1)
      AddProductMatVecVec(a, b, c);
      return;
    case 0x203101202:
      // (X,x) * (X) -> (X,1)
      // transposed-matrix * vector -> vector.unsqueeze(-1)
      AddProductTmatVecVec(a, b, c);
      return;
    case 0x202101103:
      // (X,1) * (X) -> (x,X)
      // vector * vector -> matrix (outer product)
      AddProductVec2Mat(a, b, c);
      return;


    default:
      break;

  }

  // If we reached this point, it means we could
  // not handle this request with any of the basic operations above.
  // Something is a little differ


  SubTensor a_temp(a), b_temp(b), c_temp(*c);

  PadAxes(&(a.pattern), &(b.pattern), &(c.pattern));

  CompressPatterns({&a_temp, &b_temp, &c_temp});
}



}  // namespace kaldi
}  // namespace tensor
