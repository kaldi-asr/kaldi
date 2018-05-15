// cudamatrix/cu-math.cc

// Copyright 2009-2012  Karel Vesely
//                      Johns Hopkins University (author: Daniel Povey)

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

#include "base/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"

namespace kaldi {

namespace cu {

/*
 * templated functions wrapping the ANSI-C CUDA kernel functions
 */


template<typename Real>
void RegularizeL1(CuMatrixBase<Real> *weight, CuMatrixBase<Real> *grad, Real l1, Real lr) {
  KALDI_ASSERT(SameDim(*weight, *grad));
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(weight->NumCols(), CU2DBLOCK), n_blocks(weight->NumRows(), CU2DBLOCK));

    cuda_regularize_l1(dimGrid, dimBlock, weight->Data(), grad->Data(), l1, lr,
                       weight->Dim(), grad->Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    MatrixBase<Real> &weight2 = weight->Mat();
    MatrixBase<Real> &grad2 = grad->Mat();
    for(MatrixIndexT r=0; r<weight2.NumRows(); r++) {
      for(MatrixIndexT c=0; c<weight2.NumCols(); c++) {

        if(weight2(r,c)==0.0) continue; // skip L1 if zero weightght!

        Real l1_signed = l1;
        if (weight2(r, c) < 0.0)
          l1_signed = -l1;

        Real before = weight2(r, c);
        Real after = weight2(r, c) - lr*grad2(r, c) - l1_signed;
        if ((after > 0.0) ^ (before > 0.0)) {
          weight2(r, c) = 0.0;
          grad2(r, c) = 0.0;
        } else {
          weight2(r, c) -= l1_signed;
        }
      }
    }
  }
}


template<typename Real>
void Randomize(const CuMatrixBase<Real> &src,
               const CuArray<int32> &copy_from_idx,
               CuMatrixBase<Real> *tgt) {

  KALDI_ASSERT(src.NumCols() == tgt->NumCols());
  KALDI_ASSERT(src.NumRows() == tgt->NumRows());
  KALDI_ASSERT(copy_from_idx.Dim() <= tgt->NumRows());

  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    /*
    Note: default 16x16 block-size limits the --cachesize to matrix size 16*65535 x 16*65535
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(tgt->NumCols(), CU2DBLOCK), n_blocks(copy_from_idx.Dim(), CU2DBLOCK));
    */

    /*
     * Let's use blocksize 4 x 128 (512 threads/block)
     * and extend the randomizable matrices to: col 4*65535, row 128*65535
     * (ie. max-cols:262140 (dim), max-rows:8388480 (datapoints))
     */
    dim3 dimBlock(4, 128);
    dim3 dimGrid(n_blocks(tgt->NumCols(), 4), n_blocks(copy_from_idx.Dim(), 128));
    /*
     */

    MatrixDim dimsrc = src.Dim(); dimsrc.rows=copy_from_idx.Dim();
    MatrixDim dimtgt = tgt->Dim(); dimtgt.rows=copy_from_idx.Dim();

    cuda_randomize(dimGrid, dimBlock, tgt->Data(), src.Data(),
                   copy_from_idx.Data(), dimtgt, dimsrc);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    // randomize in CPU
    const MatrixBase<Real> &srcmat = src.Mat();
    const int32 *copy_from_idxvec = copy_from_idx.Data();
    MatrixBase<Real> &tgtmat = tgt->Mat();
    for(int32 i=0; i<copy_from_idx.Dim(); i++) {
      tgtmat.Row(i).CopyFromVec(srcmat.Row(copy_from_idxvec[i]));
    }
  }
}



template<typename Real>
void Splice(const CuMatrixBase<Real> &src, const CuArray<int32> &frame_offsets,
            CuMatrixBase<Real> *tgt) {

  KALDI_ASSERT(src.NumCols()*frame_offsets.Dim() == tgt->NumCols());
  KALDI_ASSERT(src.NumRows() == tgt->NumRows());

  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(tgt->NumCols(), CU2DBLOCK), n_blocks(tgt->NumRows(), CU2DBLOCK));

    cuda_splice(dimGrid, dimBlock, tgt->Data(), src.Data(),
                frame_offsets.Data(), tgt->Dim(), src.Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    // expand in CPU
    const MatrixBase<Real> &srcmat = src.Mat();
    const int32 *frame_offsetvec = frame_offsets.Data();
    int32 dim = frame_offsets.Dim();
    MatrixBase<Real> &tgtmat = tgt->Mat();
    //
    for(int32 r=0; r < tgtmat.NumRows(); r++) {
      for(int32 off=0; off < dim; off++) {
        int32 r_off = r + frame_offsetvec[off];
        if(r_off < 0) r_off = 0;
        if(r_off >= srcmat.NumRows()) r_off = srcmat.NumRows()-1;
        memcpy(tgtmat.RowData(r)+off*srcmat.NumCols(),srcmat.RowData(r_off),sizeof(Real)*srcmat.NumCols());
      }
    }
  }
}



template<typename Real>
void Copy(const CuMatrixBase<Real> &src, const CuArray<int32> &copy_from_indices,
          CuMatrixBase<Real> *tgt) {

  KALDI_ASSERT(copy_from_indices.Dim() == tgt->NumCols());
  KALDI_ASSERT(src.NumRows() == tgt->NumRows());

  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(tgt->NumCols(), CU2DBLOCK), n_blocks(tgt->NumRows(), CU2DBLOCK));

    cuda_copy(dimGrid, dimBlock, tgt->Data(), src.Data(),
              copy_from_indices.Data(), tgt->Dim(), src.Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    // expand in CPU
    const MatrixBase<Real> &srcmat = src.Mat();
    const int32 *copy_from_indicesvec = copy_from_indices.Data();
    int32 dim = copy_from_indices.Dim();
    MatrixBase<Real> &tgtmat = tgt->Mat();
    //
    for(int32 r = 0; r < tgtmat.NumRows(); r++) {
      for(int32 c = 0; c < dim; c++) {
        tgtmat(r,c) = srcmat(r,copy_from_indicesvec[c]);
      }
    }
  }
}

template <typename Real>
void EnsureNonzero(const CuMatrixBase<Real> &src,
                   Real epsilon,
                   CuMatrixBase<Real> *dest) {
  KALDI_ASSERT(SameDim(*dest, src) && epsilon > 0.0);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(src.NumRows(), src.NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_ensure_nonzero(dimGrid, dimBlock, src.Data(), src.Dim(),
                        epsilon, dest->Stride(), dest->Data());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    int32 num_rows = src.NumRows(), num_cols = src.NumCols();
    for (int32 r = 0; r < num_rows; r++) {
      const Real *src_data = src.RowData(r);
      Real *dest_data = dest->RowData(r);
      for (int32 c = 0; c < num_cols; c++) {
        Real x = src_data[c], y;
        if (x <= -epsilon || x >= epsilon) y = x;
        else if (x >= 0.0) y = epsilon;
        else y = -epsilon;
        dest_data[c] = y;
      }
    }
  }
}


// instantiate the templates.
template
void RegularizeL1(CuMatrixBase<float> *weight, CuMatrixBase<float> *grad, float l1, float lr);
template
void RegularizeL1(CuMatrixBase<double> *weight, CuMatrixBase<double> *grad, double l1, double lr);

template
void Splice(const CuMatrixBase<float> &src, const CuArray<int32> &frame_offsets,
            CuMatrixBase<float> *tgt);
template
void Splice(const CuMatrixBase<double> &src, const CuArray<int32> &frame_offsets,
            CuMatrixBase<double> *tgt);
template
void Copy(const CuMatrixBase<float> &src, const CuArray<int32> &copy_from_indices,
          CuMatrixBase<float> *tgt);
template
void Copy(const CuMatrixBase<double> &src, const CuArray<int32> &copy_from_indices,
          CuMatrixBase<double> *tgt);

template
void Randomize(const CuMatrixBase<float> &src,
               const CuArray<int32> &copy_from_idx,
               CuMatrixBase<float> *tgt);
template
void Randomize(const CuMatrixBase<double> &src,
               const CuArray<int32> &copy_from_idx,
               CuMatrixBase<double> *tgt);

// The output y_i = scale * x_i,
// and we want to RMS value of the y_i to equal target_rms,
// so y^t y = D * target_rms^2 (if y is one row of the input).
// we need to have scale = 1.0 / sqrt(x^t x / (D * target_rms^2)).
// there is also flooring involved, to avoid division-by-zero
// problems.  It's important for the backprop, that the floor's
// square root is exactly representable as float.
// If add_log_stddev_ is true, log(max(epsi, sqrt(x^t x / D)))
// is an extra dimension of the output.
template<typename Real>
void NormalizePerRow(const CuMatrixBase<Real>& in, const Real target_rms,
                     const bool add_log_stddev, CuMatrixBase<Real>* out) {
  const Real kSquaredNormFloor = 1.3552527156068805425e-20; // 2^-66
  if (add_log_stddev) {
    KALDI_ASSERT(in.NumRows() == out->NumRows());
    KALDI_ASSERT(in.NumCols() + 1 == out->NumCols());
  } else {
    KALDI_ASSERT(SameDim(in, *out));
  }

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    size_t dimBlock = CU1DBLOCK;
    size_t dimGrid = out->NumRows();
    cuda_normalize_per_row(dimGrid, dimBlock, out->Data(), out->Stride(),
                           in.Data(), in.Dim(), target_rms, add_log_stddev);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    CuSubMatrix<Real> out_no_log(*out, 0, out->NumRows(), 0, in.NumCols());
    if (in.Data() != out_no_log.Data())
      out_no_log.CopyFromMat(in);
    CuVector<Real> in_norm(in.NumRows());
    Real d_scaled = in.NumCols() * target_rms * target_rms;
    in_norm.AddDiagMat2(1.0 / d_scaled, in, kNoTrans, 0.0);
    in_norm.ApplyFloor(kSquaredNormFloor);
    in_norm.ApplyPow(-0.5);
    out_no_log.MulRowsVec(in_norm);
    if (add_log_stddev) {
      in_norm.ApplyLog();
      in_norm.Scale(-1.0);
      in_norm.Add(log(target_rms));
      out->CopyColFromVec(in_norm, in.NumCols());
    }
  }
}

template
void NormalizePerRow(const CuMatrixBase<float>& in, const float target_rms,
                     const bool add_log_stddev, CuMatrixBase<float>* out);
template
void NormalizePerRow(const CuMatrixBase<double>& in, const double target_rms,
                     const bool add_log_stddev, CuMatrixBase<double>* out);


// A note on the derivative of NormalizeComponent...
// let both row_in and row_out be vectors of dimension D.
// Let p = row_in^T row_in / (D * target_rms^2), and let
// f = 1.0 / sqrt(max(kSquaredNormFloor, p)), and we compute row_out as:
// row_out = f row_in.
// Suppose we have a quantity deriv_out which is the derivative
// of the objective function w.r.t. row_out.  We want to compute
// deriv_in which is the derivative of the objective function w.r.t.
// row_in.  Let the objective function be F.  One term is obvious: we have
// deriv_in = f deriv_out + ....
// next we have to take into account the derivative that gets back-propagated
// through f.  Obviously, dF/df = deriv_out^T row_in.
// And df/dp = (p <= kSquaredNormFloor ? 0.0 : -0.5 p^{-1.5}) = (f == 1.0 / sqrt(kSquaredNormFloor) ? 0.0 : -0.5 f^3),
// and dp/d(row_in) = 2/(D * target_rms^2) row_in. [it's vector_valued].
// So this term in dF/d(row_in) equals:
// dF/df df/dp dp/d(row_in)   =    2/(D * target_rms^2) (f == 1.0 / sqrt(kSquaredNormFloor)  ? 0.0 : -0.5 f^3) (deriv_out^T row_in) row_in
// So
// deriv_in = f deriv_out + (f == 1.0 ? 0.0 : -f^3  / (D * target_rms^2) ) (deriv_out^T row_in) row_in
//  if add_log_stddev_ true, the deriv_in has another term as
// dF/dx_i = dF/df . df/dx_i => df/dx_i = x_i/(x^T x)
template<typename Real>
void DiffNormalizePerRow(const CuMatrixBase<Real> &in_value,
                         const CuMatrixBase<Real> &out_deriv,
                         const Real target_rms, const bool add_log_stddev,
                         CuMatrixBase<Real>* in_deriv) {
  const Real kSquaredNormFloor = 1.3552527156068805425e-20; // 2^-66
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    size_t dimBlock = CU1DBLOCK;
    size_t dimGrid = in_deriv->NumRows();
    cuda_diff_normalize_per_row(dimGrid, dimBlock, in_deriv->Data(),
                                in_deriv->Stride(), in_value.Data(),
                                in_value.Dim(), out_deriv.Data(),
                                out_deriv.Stride(), target_rms, add_log_stddev);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    const CuSubMatrix<Real> out_deriv_no_log(out_deriv, 0, out_deriv.NumRows(),
                                             0, in_value.NumCols());
    CuVector<Real> dot_products(out_deriv.NumRows());
    dot_products.AddDiagMatMat(1.0, out_deriv_no_log, kNoTrans, in_value,
                               kTrans, 0.0);
    CuVector<Real> in_norm(in_value.NumRows());
    Real d_scaled = (in_value.NumCols() * target_rms * target_rms);
    in_norm.AddDiagMat2(1.0, in_value, kNoTrans, 0.0);

    if (add_log_stddev) {
      CuVector<Real> log_stddev_deriv(in_norm), // log_stddev deriv as dF/dy .* (x^T x)^-1
      out_deriv_for_stddev(out_deriv.NumRows(), kUndefined);
      // f = log(sqrt(max(epsi, x^T x / D)))
      // df/dx = epsi^2 * D < x^T x ? (1/(x^T x)) * x  : 0.
      // we don't compute this exactly below for the case when x^2 x is very
      // small, but we do make sure that the deriv isn't infinity when the input
      // is zero.
      log_stddev_deriv.ApplyFloor(in_value.NumCols() * kSquaredNormFloor);
      log_stddev_deriv.ApplyPow(-1.0);
      out_deriv_for_stddev.CopyColFromMat(out_deriv, (out_deriv.NumCols() - 1));
      log_stddev_deriv.MulElements(out_deriv_for_stddev);
      if (in_deriv)
        in_deriv->AddDiagVecMat(1.0, log_stddev_deriv, in_value, kNoTrans, 1.0);
    }
    in_norm.Scale(1.0 / d_scaled);
    in_norm.ApplyFloor(kSquaredNormFloor);
    in_norm.ApplyPow(-0.5);
    if (in_deriv) {
      if (in_deriv->Data() != out_deriv_no_log.Data())
        in_deriv->AddDiagVecMat(1.0, in_norm, out_deriv_no_log, kNoTrans, 1.0);
      else
        in_deriv->MulRowsVec(in_norm);
      in_norm.ReplaceValue(1.0 / sqrt(kSquaredNormFloor), 0.0);
      in_norm.ApplyPow(3.0);
      dot_products.MulElements(in_norm);

      in_deriv->AddDiagVecMat(-1.0 / d_scaled, dot_products, in_value, kNoTrans,
                              1.0);
    }
  }
}

template
void DiffNormalizePerRow(const CuMatrixBase<float> &in_value,
                         const CuMatrixBase<float> &out_deriv,
                         const float target_rms, const bool add_log_stddev,
                         CuMatrixBase<float>* in_deriv);
template
void DiffNormalizePerRow(const CuMatrixBase<double> &in_value,
                         const CuMatrixBase<double> &out_deriv,
                         const double target_rms, const bool add_log_stddev,
                         CuMatrixBase<double>* in_deriv);


// not calling this Sigmoid to reduce the chance of future collisions.
template<typename Real>
static inline Real ScalarSigmoid(Real a) {
  if (a > Real(0)) {
    return Real(1) / (Real(1) + Exp(-a));
  } else {
    Real x = Exp(a);
    return x / (x + Real(1));
  }
}

template<typename Real>
static inline Real ScalarTanh(Real a) {
  if (a > Real(0)) {
    Real inv_expa = Exp(-a);
    return -Real(1) + Real(2) / (Real(1) + inv_expa * inv_expa);
  } else {
    Real expa = Exp(a);
    return Real(1) - Real(2) / (Real(1) + expa * expa);
  }
}

template<typename Real>
void CpuComputeLstmNonlinearity(const MatrixBase<Real> &input_mat,
                                const MatrixBase<Real> &params_mat,
                                MatrixBase<Real> *output) {
  int32 num_rows = input_mat.NumRows(),
      input_cols = input_mat.NumCols(),
        cell_dim = input_cols / 5;
  KALDI_ASSERT(input_cols == (cell_dim * 5) || input_cols == (cell_dim * 5) + 3);
  KALDI_ASSERT(output->NumRows() == num_rows);
  KALDI_ASSERT(params_mat.NumRows() == 3);
  KALDI_ASSERT(params_mat.NumCols() == cell_dim);
  KALDI_ASSERT(output->NumCols() == 2 * cell_dim);

  MatrixBase<Real> &output_mat = *output;
  const Real *params_data = params_mat.Data();
  int32 params_stride = params_mat.Stride();
  for (int32 r = 0; r < num_rows; r++) {
    const Real *input_row = input_mat.RowData(r);
    // i_scale and f_scale relate to dropout, they will normally be 1.0.
    Real i_scale = (input_cols == cell_dim*5 ? 1.0:input_row[cell_dim*5]),
         f_scale = (input_cols == cell_dim*5 ? 1.0:input_row[cell_dim*5 + 1]),
         o_scale = (input_cols == cell_dim*5 ? 1.0:input_row[cell_dim*5 + 2]);

    Real *output_row = output_mat.RowData(r);
    for (int32 c = 0; c < cell_dim; c++) {
      Real i_part = input_row[c];
      Real f_part = input_row[c + cell_dim];
      Real c_part = input_row[c + 2 * cell_dim];
      Real o_part = input_row[c + 3 * cell_dim];
      Real c_prev = input_row[c + 4 * cell_dim];
      Real w_ic = params_data[c];
      Real w_fc = params_data[c + params_stride];
      Real w_oc = params_data[c + params_stride * 2];
      Real i_t = ScalarSigmoid(i_part + w_ic * c_prev);
      Real f_t = ScalarSigmoid(f_part + w_fc * c_prev);
      Real c_t = f_t * f_scale * c_prev + i_t * i_scale * ScalarTanh(c_part);
      Real o_t = ScalarSigmoid(o_part + w_oc * c_t);
      Real m_t = o_t * o_scale * ScalarTanh(c_t);
      output_row[c] = c_t;
      output_row[c + cell_dim] = m_t;
    }
  }
}

template<typename Real>
void ComputeLstmNonlinearity(const CuMatrixBase<Real> &input,
                             const CuMatrixBase<Real> &params,
                             CuMatrixBase<Real> *output) {
  int32 num_rows = input.NumRows(),
      input_cols = input.NumCols(),
        cell_dim = input_cols / 5;
  KALDI_ASSERT(input_cols == (cell_dim * 5) || input_cols == (cell_dim * 5) + 3);
  KALDI_ASSERT(output->NumRows() == num_rows);
  KALDI_ASSERT(params.NumRows() == 3);
  KALDI_ASSERT(params.NumCols() == cell_dim);
  KALDI_ASSERT(output->NumCols() == 2 * cell_dim);

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    int have_dropout_mask = (input_cols == (cell_dim * 5) + 3);

    // Each thread block is working on 1 row of the data.
    // It's best that cell dim is a multiple fo CU1DBLOCK
    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(num_rows);

    cuda_lstm_nonlinearity(dimGrid, dimBlock, input.Data(), input.Stride(),
                           params.Data(), params.Stride(), output->Stride(),
                           cell_dim, have_dropout_mask, num_rows, output->Data());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    CpuComputeLstmNonlinearity(input.Mat(), params.Mat(), &output->Mat());
  }
}

template
void CpuComputeLstmNonlinearity(const MatrixBase<float> &input_mat,
                                const MatrixBase<float> &params_mat,
                                MatrixBase<float> *output);
template
void CpuComputeLstmNonlinearity(const MatrixBase<double> &input_mat,
                                const MatrixBase<double> &params_mat,
                                MatrixBase<double> *output);
template
void ComputeLstmNonlinearity(const CuMatrixBase<float> &input,
                             const CuMatrixBase<float> &params,
                             CuMatrixBase<float> *output);
template
void ComputeLstmNonlinearity(const CuMatrixBase<double> &input,
                             const CuMatrixBase<double> &params,
                             CuMatrixBase<double> *output);

template<typename Real>
void CpuBackpropLstmNonlinearity(const MatrixBase<Real> &input,
                                 const MatrixBase<Real> &params,
                                 const MatrixBase<Real> &output_deriv,
                                 const MatrixBase<double> &deriv_sum_in,
                                 const VectorBase<Real> &self_repair_config,
                                 double count_in,
                                 MatrixBase<Real> *input_deriv,
                                 MatrixBase<Real> *params_deriv,
                                 MatrixBase<double> *value_sum_out,
                                 MatrixBase<double> *deriv_sum_out,
                                 MatrixBase<Real> *self_repair_sum_out) {
  int32 num_rows = input.NumRows(),
      input_cols = input
                   .NumCols(),
        cell_dim = input.NumCols() / 5;
  // Check dimensions.
  KALDI_ASSERT(input_cols == (cell_dim * 5) || input_cols == (cell_dim * 5) + 3);
  KALDI_ASSERT(params.NumRows() == 3);
  KALDI_ASSERT(params.NumCols() == cell_dim);
  KALDI_ASSERT(output_deriv.NumRows() == num_rows);
  KALDI_ASSERT(output_deriv.NumCols() == 2 * cell_dim);
  KALDI_ASSERT(deriv_sum_in.NumRows() == 5);
  KALDI_ASSERT(deriv_sum_in.NumCols() == cell_dim);
  KALDI_ASSERT(self_repair_config.Dim() == 10);
  if (input_deriv != NULL) {
    KALDI_ASSERT(SameDim(input, *input_deriv));
  }
  if (params_deriv == NULL) {
    KALDI_ASSERT(value_sum_out == NULL);
    KALDI_ASSERT(deriv_sum_out == NULL);
    KALDI_ASSERT(self_repair_sum_out == NULL);
  } else {
    KALDI_ASSERT(value_sum_out != NULL);
    KALDI_ASSERT(deriv_sum_out != NULL);
    KALDI_ASSERT(self_repair_sum_out != NULL);
    KALDI_ASSERT(SameDim(params, *params_deriv));
    KALDI_ASSERT(value_sum_out->NumRows() == 5);
    KALDI_ASSERT(value_sum_out->NumCols() == cell_dim);
    KALDI_ASSERT(SameDim(*value_sum_out, *deriv_sum_out));
    KALDI_ASSERT(self_repair_sum_out->NumRows() == 5);
    KALDI_ASSERT(self_repair_sum_out->NumCols() == cell_dim);
  }

  const MatrixBase<Real> &input_mat = input;
  const MatrixBase<Real> &params_mat = params;
  const MatrixBase<Real> &output_deriv_mat = output_deriv;
  const MatrixBase<double> &deriv_sum_in_mat = deriv_sum_in;
  const VectorBase<Real> &sr_config = self_repair_config;
  MatrixBase<Real> *input_deriv_mat = (
      input_deriv == NULL ? NULL : input_deriv);
  MatrixBase<Real> *params_deriv_mat = NULL;
  MatrixBase<Real> *self_repair_sum_out_mat = NULL;
  MatrixBase<double> *value_sum_out_mat = NULL;
  MatrixBase<double> *deriv_sum_out_mat = NULL;
  if (params_deriv != NULL) {
    params_deriv_mat = params_deriv;
    value_sum_out_mat = value_sum_out;
    deriv_sum_out_mat = deriv_sum_out;
    self_repair_sum_out_mat = self_repair_sum_out;
  }


  // We add 1.0 (i.e. a small value) to the count to avoid division by zero.
  Real count = 1.0 + count_in;
  for (int32 c = 0; c < cell_dim; c++) {
    // parameters
    Real w_ic = params_mat(0, c);
    Real w_fc = params_mat(1, c);
    Real w_oc = params_mat(2, c);
    // derivative sums w.r.t. parameters.
    Real w_ic_deriv_sum = 0.0;
    Real w_fc_deriv_sum = 0.0;
    Real w_oc_deriv_sum = 0.0;

    // average derivatives, for self-repair.
    // The 5 nonlinearities that are subject to self-repair are written as:
    //  Sigmoid(i_t_input), Sigmoid(f_t_input),
    //  Tanh(c_part), Sigmoid(o_t_input),  Tanh(c_t)
    Real i_t_self_repair = (
        deriv_sum_in_mat(0, c) / count < sr_config(0) ? sr_config(5) : 0.0);
    Real f_t_self_repair = (
        deriv_sum_in_mat(1, c) / count < sr_config(1) ? sr_config(6) : 0.0);
    Real c_part_self_repair = (
        deriv_sum_in_mat(2, c) / count < sr_config(2) ? sr_config(7) : 0.0);
    Real o_t_self_repair = (
        deriv_sum_in_mat(3, c) / count < sr_config(3) ? sr_config(8) : 0.0);
    Real c_t_self_repair = (
        deriv_sum_in_mat(4, c) / count < sr_config(4) ? sr_config(9) : 0.0);
    // Note on how we add self-repair for sigmoids/tanh's.  If self-repair
    // is activated for this unit, then...
    // For sigmoids we'd add -self_repair_scale * (2 * sigmoid(x) - 1.0)
    // ... to the input-deriv;
    // For tanh's we'd add -self_repair_scale * tanh(x)
    // If self-repair is not activated, the 'self_repair' scales are set to zero.

    // The following variables are for the accumulation of stats on the
    // sigmoid and tanh units.
    Real i_t_value_sum = 0.0, i_t_deriv_sum = 0.0;
    Real f_t_value_sum = 0.0, f_t_deriv_sum = 0.0;
    Real c_part_value_sum = 0.0, c_part_deriv_sum = 0.0;
    Real o_t_value_sum = 0.0, o_t_deriv_sum = 0.0;
    Real c_t_value_sum = 0.0, c_t_deriv_sum = 0.0;


    for (int32 r = 0; r < num_rows; r++) {
      Real i_part = input_mat(r, c),
          f_part = input_mat(r, c + cell_dim),
          c_part = input_mat(r, c + 2 * cell_dim),
          o_part = input_mat(r, c + 3 * cell_dim),
          c_prev = input_mat(r, c + 4 * cell_dim);

      Real i_scale = (input_cols == cell_dim * 5 ? 1.0 :
                      input_mat(r, cell_dim * 5)),
           f_scale = (input_cols == cell_dim * 5 ? 1.0 :
                      input_mat(r, cell_dim * 5 + 1)),
           o_scale = (input_cols == cell_dim * 5 ? 1.0 :
                      input_mat(r, cell_dim * 5 + 2));

      // For greater clarity, we give some of the quantities in the
      // forward equations their own names.
      Real i_t_input = i_part + w_ic * c_prev,
          i_t = ScalarSigmoid(i_t_input),
          f_t_input = f_part + w_fc * c_prev,
          f_t = ScalarSigmoid(f_t_input),
          tanh_c_part = ScalarTanh(c_part),
          c_t = f_t * f_scale * c_prev + i_t * i_scale * tanh_c_part,
          o_t_input = o_part + w_oc * c_t,
          o_t = ScalarSigmoid(o_t_input),
          tanh_c_t = ScalarTanh(c_t);
      // we'd also compute, in the forward pass,
      //   m_t = o_t * tanh_c_t;
      // but this variable is not needed.

      // Accumulate nonlinearity value and derivative stats.
      // Note:
      //    tanh'(x)  = sech^2(x) = -(tanh(x)+1) (tanh(x)-1) = 1 - tanh^2(x)
      //  sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)).
      i_t_value_sum += i_t;
      i_t_deriv_sum += i_t * (1.0F - i_t);
      f_t_value_sum += f_t;
      f_t_deriv_sum += f_t * (1.0F - f_t);
      c_part_value_sum += tanh_c_part;
      c_part_deriv_sum += 1.0F - tanh_c_part * tanh_c_part;
      o_t_value_sum += o_t;
      o_t_deriv_sum += o_t * (1.0F - o_t);
      c_t_value_sum += tanh_c_t;
      c_t_deriv_sum += 1.0F - tanh_c_t * tanh_c_t;


      // the derivative of the objective function w.r.t. a particular quantity
      // will be written by prepending "d" to the name.
      // We compute these derivatives in the reverse of the order in which
      // we computed the original quantities.
      // dc_t_out is the part of the derivative w.r.t. c_t that
      // comes directly from the output of this function.
      Real dc_t_out = output_deriv_mat(r, c);
      Real dm_t = output_deriv_mat(r, c + cell_dim);
      Real dtanh_c_t = o_t * o_scale * dm_t;
      Real do_t = o_scale * tanh_c_t * dm_t;
      Real do_t_input = (o_t * (1.0F - o_t) * do_t
          - (2.0F * o_t - 1.0F) * o_t_self_repair);
      Real dc_t = ((1.0F - tanh_c_t * tanh_c_t) * dtanh_c_t + dc_t_out
          + do_t_input * w_oc) - tanh_c_t * c_t_self_repair;
      Real dtanh_c_part = i_t * i_scale * dc_t;
      Real df_t = dc_t * f_scale * c_prev;
      Real df_t_input = ((df_t * f_t * (1.0F - f_t)
                          - (2.0F * f_t - 1.0F) * f_t_self_repair));
      Real di_t = dc_t * i_scale * tanh_c_part;
      Real di_t_input = ((di_t * i_t * (1.0F - i_t)
                          - (2.0F * i_t - 1.0F) * i_t_self_repair));

      w_ic_deriv_sum += c_prev * di_t_input;
      w_fc_deriv_sum += c_prev * df_t_input;
      w_oc_deriv_sum += c_t * do_t_input;

      Real dc_prev = w_ic * di_t_input + w_fc * df_t_input + f_t * f_scale * dc_t;
      Real do_part = do_t_input;
      Real dc_part = ((1.0F - tanh_c_part * tanh_c_part) * dtanh_c_part
          - tanh_c_part * c_part_self_repair);
      Real df_part = df_t_input;
      Real di_part = di_t_input;

      if (input_deriv_mat != NULL) {
        (*input_deriv_mat)(r, c) = di_part;
        (*input_deriv_mat)(r, c + cell_dim) = df_part;
        (*input_deriv_mat)(r, c + 2 * cell_dim) = dc_part;
        (*input_deriv_mat)(r, c + 3 * cell_dim) = do_part;
        (*input_deriv_mat)(r, c + 4 * cell_dim) = dc_prev;
      }
    }

    if (params_deriv != NULL) {
      // note: for optimizing things you can assume that params_deriv and
      // input_deriv_mat are non-NULL (i.e. all the output matrices are
      // non-NULL).  The situations when some of the output matrices are NULL
      // does not happen often (mainly only in testing code).

      (*params_deriv_mat)(0, c) = w_ic_deriv_sum;
      (*params_deriv_mat)(1, c) = w_fc_deriv_sum;
      (*params_deriv_mat)(2, c) = w_oc_deriv_sum;

      (*value_sum_out_mat)(0, c) += i_t_value_sum;
      (*value_sum_out_mat)(1, c) += f_t_value_sum;
      (*value_sum_out_mat)(2, c) += c_part_value_sum;
      (*value_sum_out_mat)(3, c) += o_t_value_sum;
      (*value_sum_out_mat)(4, c) += c_t_value_sum;

      // need to update self_repair_sum_out before deriv_sum_out, because
      // deriv_sum_out and deriv_sum_in might point to the same memory.
      for (int32 i = 0; i < 5; i++)
        (*self_repair_sum_out_mat)(i, c) =
            (deriv_sum_in_mat(i, c) / count < sr_config(i) ? num_rows : 0);

      (*deriv_sum_out_mat)(0, c) += i_t_deriv_sum;
      (*deriv_sum_out_mat)(1, c) += f_t_deriv_sum;
      (*deriv_sum_out_mat)(2, c) += c_part_deriv_sum;
      (*deriv_sum_out_mat)(3, c) += o_t_deriv_sum;
      (*deriv_sum_out_mat)(4, c) += c_t_deriv_sum;
    }
  }
}



template<typename Real>
void BackpropLstmNonlinearity(const CuMatrixBase<Real> &input,
                              const CuMatrixBase<Real> &params,
                              const CuMatrixBase<Real> &output_deriv,
                              const CuMatrixBase<double> &deriv_sum_in,
                              const CuVectorBase<Real> &self_repair_config,
                              double count_in,
                              CuMatrixBase<Real> *input_deriv,
                              CuMatrixBase<Real> *params_deriv,
                              CuMatrixBase<double> *value_sum_out,
                              CuMatrixBase<double> *deriv_sum_out,
                              CuMatrixBase<Real> *self_repair_sum_out) {
  int32 num_rows = input.NumRows(),
        cell_dim = input.NumCols() / 5,
      input_cols = input.NumCols();
  // Check dimensions.
  KALDI_ASSERT(input_cols == (cell_dim * 5) || input_cols == (cell_dim*5) + 3);
  KALDI_ASSERT(params.NumRows() == 3);
  KALDI_ASSERT(params.NumCols() == cell_dim);
  KALDI_ASSERT(output_deriv.NumRows() == num_rows);
  KALDI_ASSERT(output_deriv.NumCols() == 2 * cell_dim);
  KALDI_ASSERT(deriv_sum_in.NumRows() == 5);
  KALDI_ASSERT(deriv_sum_in.NumCols() == cell_dim);
  KALDI_ASSERT(self_repair_config.Dim() == 10);
  if (input_deriv != NULL) {
    KALDI_ASSERT(SameDim(input, *input_deriv));
  }
  if (params_deriv == NULL) {
    KALDI_ASSERT(value_sum_out == NULL);
    KALDI_ASSERT(deriv_sum_out == NULL);
    KALDI_ASSERT(self_repair_sum_out == NULL);
  } else {
    KALDI_ASSERT(value_sum_out != NULL);
    KALDI_ASSERT(deriv_sum_out != NULL);
    KALDI_ASSERT(self_repair_sum_out != NULL);
    KALDI_ASSERT(SameDim(params, *params_deriv));
    KALDI_ASSERT(value_sum_out->NumRows() == 5);
    KALDI_ASSERT(value_sum_out->NumCols() == cell_dim);
    KALDI_ASSERT(SameDim(*value_sum_out, *deriv_sum_out));
    KALDI_ASSERT(self_repair_sum_out->NumRows() == 5);
    KALDI_ASSERT(self_repair_sum_out->NumCols() == cell_dim);
  }


#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    // Each thread block is working on 1 row of the data.
    // It's best that cell dim is a multiple fo CU1DBLOCK

    int have_dropout_mask = (input_cols == (cell_dim * 5) + 3);

    // Use 2D block (8x32 threads) as we need to compute column sum.
    // Use 1D grid to cover the data matrix width `cell_dim`.
    const int kWarpSize = 32;
    dim3 dimBlock(kWarpSize, CU1DBLOCK / kWarpSize);
//    dim3 dimGrid(n_blocks(cell_dim, dimBlock.x),
//                 n_blocks(num_rows, dimBlock.y));
//    if (dimGrid.x * dimGrid.y > 1024) {
//      dimGrid.y = std::max(1024 / dimGrid.x, 1);
//    }
    dim3 dimGrid(n_blocks(cell_dim, dimBlock.x));
    if (input_deriv == NULL) {
      if (params_deriv == NULL) {
        cuda_diff_lstm_nonlinearity(dimGrid, dimBlock, cell_dim,
                                    have_dropout_mask, num_rows,
                                    input.Data(), input.Stride(), params.Data(),
                                    params.Stride(), output_deriv.Data(),
                                    output_deriv.Stride(), deriv_sum_in.Data(),
                                    deriv_sum_in.Stride(),
                                    self_repair_config.Data(), count_in + 1,
                                    NULL,
                                    0,
                                    NULL,
                                    0,
                                    NULL,
                                    0,
                                    NULL,
                                    0,
                                    NULL,
                                    0);

      } else {
        cuda_diff_lstm_nonlinearity(dimGrid, dimBlock, cell_dim,
                                    have_dropout_mask, num_rows,
                                    input.Data(), input.Stride(), params.Data(),
                                    params.Stride(), output_deriv.Data(),
                                    output_deriv.Stride(), deriv_sum_in.Data(),
                                    deriv_sum_in.Stride(),
                                    self_repair_config.Data(), count_in + 1,
                                    NULL,
                                    0, params_deriv->Data(),
                                    params_deriv->Stride(),
                                    value_sum_out->Data(),
                                    value_sum_out->Stride(),
                                    deriv_sum_out->Data(),
                                    deriv_sum_out->Stride(),
                                    self_repair_sum_out->Data(),
                                    self_repair_sum_out->Stride());
      }
    } else {
      if (params_deriv == NULL) {
        cuda_diff_lstm_nonlinearity(dimGrid, dimBlock, cell_dim,
                                    have_dropout_mask, num_rows,
                                    input.Data(), input.Stride(), params.Data(),
                                    params.Stride(), output_deriv.Data(),
                                    output_deriv.Stride(), deriv_sum_in.Data(),
                                    deriv_sum_in.Stride(),
                                    self_repair_config.Data(), count_in + 1,
                                    input_deriv->Data(), input_deriv->Stride(),
                                    NULL,
                                    0, NULL, 0, NULL, 0, NULL, 0);
      } else {
        cuda_diff_lstm_nonlinearity(dimGrid, dimBlock, cell_dim,
                                    have_dropout_mask, num_rows,
                                    input.Data(), input.Stride(), params.Data(),
                                    params.Stride(), output_deriv.Data(),
                                    output_deriv.Stride(), deriv_sum_in.Data(),
                                    deriv_sum_in.Stride(),
                                    self_repair_config.Data(), count_in + 1,
                                    input_deriv->Data(), input_deriv->Stride(),
                                    params_deriv->Data(),
                                    params_deriv->Stride(),
                                    value_sum_out->Data(),
                                    value_sum_out->Stride(),
                                    deriv_sum_out->Data(),
                                    deriv_sum_out->Stride(),
                                    self_repair_sum_out->Data(),
                                    self_repair_sum_out->Stride());
      }
    }

    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    CpuBackpropLstmNonlinearity(input.Mat(), params.Mat(), output_deriv.Mat(),
                                deriv_sum_in.Mat(), self_repair_config.Vec(),
                                count_in, &(input_deriv->Mat()),
                                &(params_deriv->Mat()), &(value_sum_out->Mat()),
                                &(deriv_sum_out->Mat()),
                                &(self_repair_sum_out->Mat()));
  }
}

template <typename Real>
void EnsureNonzero(const CuVectorBase<Real> &src,
                   Real epsilon,
                   CuVectorBase<Real> *dest) {
  KALDI_ASSERT(src.Dim() == dest->Dim());
  int32 dim = src.Dim();
  // fake it with a 1-row matrix.
  CuSubMatrix<Real> src_mat(src.Data(), 1,  dim, dim),
      dest_mat(dest->Data(), 1, dim, dim);
  EnsureNonzero(src_mat, epsilon, &dest_mat);
}

// Instantiate the templates we defined above.

template
void EnsureNonzero(const CuMatrixBase<float> &src,
                   float epsilon,
                   CuMatrixBase<float> *dest);
template
void EnsureNonzero(const CuMatrixBase<double> &src,
                   double epsilon,
                   CuMatrixBase<double> *dest);

template
void EnsureNonzero(const CuVectorBase<float> &src,
                   float epsilon,
                   CuVectorBase<float> *dest);
template
void EnsureNonzero(const CuVectorBase<double> &src,
                   double epsilon,
                   CuVectorBase<double> *dest);

template
void CpuBackpropLstmNonlinearity(const MatrixBase<float> &input,
                                 const MatrixBase<float> &params,
                                 const MatrixBase<float> &output_deriv,
                                 const MatrixBase<double> &deriv_sum_in,
                                 const VectorBase<float> &self_repair_config,
                                 double count_in,
                                 MatrixBase<float> *input_deriv,
                                 MatrixBase<float> *params_deriv,
                                 MatrixBase<double> *value_sum_out,
                                 MatrixBase<double> *deriv_sum_out,
                                 MatrixBase<float> *self_repair_sum_out);
template
void CpuBackpropLstmNonlinearity(const MatrixBase<double> &input,
                                 const MatrixBase<double> &params,
                                 const MatrixBase<double> &output_deriv,
                                 const MatrixBase<double> &deriv_sum_in,
                                 const VectorBase<double> &self_repair_config,
                                 double count_in,
                                 MatrixBase<double> *input_deriv,
                                 MatrixBase<double> *params_deriv,
                                 MatrixBase<double> *value_sum_out,
                                 MatrixBase<double> *deriv_sum_out,
                                 MatrixBase<double> *self_repair_sum_out);
template
void BackpropLstmNonlinearity(const CuMatrixBase<float> &input,
                              const CuMatrixBase<float> &params,
                              const CuMatrixBase<float> &output_deriv,
                              const CuMatrixBase<double> &deriv_sum_in,
                              const CuVectorBase<float> &self_repair_config,
                              double count_in,
                              CuMatrixBase<float> *input_deriv,
                              CuMatrixBase<float> *params_deriv,
                              CuMatrixBase<double> *value_sum_out,
                              CuMatrixBase<double> *deriv_sum_out,
                              CuMatrixBase<float> *self_repair_sum_out);
template
void BackpropLstmNonlinearity(const CuMatrixBase<double> &input,
                              const CuMatrixBase<double> &params,
                              const CuMatrixBase<double> &output_deriv,
                              const CuMatrixBase<double> &deriv_sum_in,
                              const CuVectorBase<double> &self_repair_config,
                              double count_in,
                              CuMatrixBase<double> *input_deriv,
                              CuMatrixBase<double> *params_deriv,
                              CuMatrixBase<double> *value_sum_out,
                              CuMatrixBase<double> *deriv_sum_out,
                              CuMatrixBase<double> *self_repair_sum_out);



} //namespace cu

} //namespace kaldi
