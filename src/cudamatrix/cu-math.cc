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
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(weight->NumCols(), CU2DBLOCK), n_blocks(weight->NumRows(), CU2DBLOCK));

    cuda_regularize_l1(dimGrid, dimBlock, weight->Data(), grad->Data(), l1, lr,
                       weight->Dim(), grad->Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
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
    Timer tim;

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

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
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
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(tgt->NumCols(), CU2DBLOCK), n_blocks(tgt->NumRows(), CU2DBLOCK));

    cuda_splice(dimGrid, dimBlock, tgt->Data(), src.Data(),
                frame_offsets.Data(), tgt->Dim(), src.Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
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
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(tgt->NumCols(), CU2DBLOCK), n_blocks(tgt->NumRows(), CU2DBLOCK));

    cuda_copy(dimGrid, dimBlock, tgt->Data(), src.Data(),
              copy_from_indices.Data(), tgt->Dim(), src.Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
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


// not calling this Sigmoid to reduce the chance of future collisions.
static inline BaseFloat ScalarSigmoid(BaseFloat a) {
  if (a > 0.0) {
    return 1.0 / (1.0 + Exp(-a));
  } else {
    BaseFloat x = Exp(a);
    return x / (x + 1.0);
  }
}

static inline BaseFloat ScalarTanh(BaseFloat a) {
  if (a > 0.0) {
    BaseFloat inv_expa = Exp(-a);
    return -1.0 + 2.0 / (1.0 + inv_expa * inv_expa);
  } else {
    BaseFloat expa = Exp(a);
    return 1.0 - 2.0 / (1.0 + expa * expa);
  }
}


void ComputeLstmNonlinearity(const CuMatrixBase<BaseFloat> &input,
                             const CuMatrixBase<BaseFloat> &params,
                             CuMatrixBase<BaseFloat> *output) {
  int32 num_rows = input.NumRows(),
      cell_dim = input.NumCols() / 5;
  KALDI_ASSERT(output->NumRows() == num_rows &&
               input.NumCols() % 5 == 0 &&
               params.NumRows() == 3 && params.NumCols() == cell_dim &&
               output->NumCols() == 2 * cell_dim);

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ERR << "CUDA version not implemented";
  } else
#endif
  {
    const MatrixBase<BaseFloat> &input_mat = input.Mat(),
        &params_mat = params.Mat();
    MatrixBase<BaseFloat> &output_mat = output->Mat();
    const BaseFloat *params_data = params_mat.Data();
    int32 params_stride = params_mat.Stride();
    for (int32 r = 0; r < num_rows; r++) {
      const BaseFloat *input_row = input_mat.RowData(r);
      BaseFloat *output_row = output_mat.RowData(r);
      for (int32 c = 0; c < cell_dim; c++) {
        BaseFloat i_part = input_row[c], f_part = input_row[c + cell_dim],
            c_part = input_row[c + 2 * cell_dim],
            o_part = input_row[c + 3 * cell_dim],
            c_prev = input_row[c + 4 * cell_dim],
            w_ic = params_data[c], w_fc = params_data[c + params_stride],
            w_oc = params_data[c + params_stride * 2];
        BaseFloat i_t = ScalarSigmoid(i_part + w_ic * c_prev),
            f_t = ScalarSigmoid(f_part + w_fc * c_prev),
            c_t = f_t * c_prev + i_t * ScalarTanh(c_part),
            o_t = ScalarSigmoid(o_part + w_oc * c_t),
            m_t = o_t * ScalarTanh(c_t);
        output_row[c] = c_t;
        output_row[c + cell_dim] = m_t;
      }
    }
  }
}


void BackpropLstmNonlinearity(const CuMatrixBase<BaseFloat> &input,
                              const CuMatrixBase<BaseFloat> &params,
                              const CuMatrixBase<BaseFloat> &output_deriv,
                              const CuMatrixBase<double> &deriv_sum_in,
                              const CuVectorBase<BaseFloat> &self_repair_config,
                              BaseFloat count_in,
                              CuMatrixBase<BaseFloat> *input_deriv,
                              CuMatrixBase<BaseFloat> *params_deriv,
                              CuMatrixBase<double> *value_sum_out,
                              CuMatrixBase<double> *deriv_sum_out,
                              CuMatrixBase<BaseFloat> *self_repair_sum_out) {
  int32 num_rows = input.NumRows(),
      cell_dim = input.NumCols() / 5;
  // Check dimensions.
  KALDI_ASSERT(output_deriv.NumRows() == num_rows &&
               input.NumCols() % 5 == 0 &&
               params.NumRows() == 3 && params.NumCols() == cell_dim &&
               output_deriv.NumCols() == 2 * cell_dim &&
               deriv_sum_in.NumRows() == 5 && deriv_sum_in.NumCols() == cell_dim
               && self_repair_config.Dim() == 10 && count_in >= 0);
  if (input_deriv != NULL) {
    KALDI_ASSERT(SameDim(input, *input_deriv));
  }
  if (params_deriv == NULL) {
    KALDI_ASSERT(value_sum_out == NULL && deriv_sum_out == NULL &&
                 self_repair_sum_out == NULL);
  } else {
    KALDI_ASSERT(value_sum_out != NULL && deriv_sum_out != NULL &&
                 self_repair_sum_out != NULL && SameDim(params, *params_deriv) &&
                 value_sum_out->NumRows() == 5 &&
                 value_sum_out->NumCols() == cell_dim &&
                 SameDim(*value_sum_out, *deriv_sum_out) &&
                 self_repair_sum_out->NumRows() == 5 &&
                 self_repair_sum_out->NumCols() == cell_dim);
  }


#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ERR << "CUDA version not implemented";
  // notes for Shiyin:
    //  You could do an 'easy' initial version where we have have one thread per dimension,
    //  and you can try optimizing this later on.
    //  Since the cell-dim is usually quite large, like 1024, this is fairly reasonable.
    // But up to you.
  } else
#endif
  {
    const MatrixBase<BaseFloat> &input_mat = input.Mat(),
        &params_mat = params.Mat(),
        &output_deriv_mat = output_deriv.Mat();
    const MatrixBase<double> &deriv_sum_in_mat = deriv_sum_in.Mat();
    const VectorBase<BaseFloat> &sr_config =
        self_repair_config.Vec();
    MatrixBase<BaseFloat> *input_deriv_mat =
        (input_deriv == NULL ? NULL : &(input_deriv->Mat()));
    MatrixBase<BaseFloat> *params_deriv_mat = NULL,
        *self_repair_sum_out_mat = NULL;
    MatrixBase<double> *value_sum_out_mat = NULL, *deriv_sum_out_mat = NULL;
    if (params_deriv != NULL) {
      params_deriv_mat = &(params_deriv->Mat());
      value_sum_out_mat = &(value_sum_out->Mat());
      deriv_sum_out_mat = &(deriv_sum_out->Mat());
      self_repair_sum_out_mat = &(self_repair_sum_out->Mat());
    }


    // We add 1.0 (i.e. a small value) to the count to avoid division by zero.
    BaseFloat count = 1.0 + count_in;
    for (int32 c = 0; c < cell_dim; c++) {
      // parameters
      BaseFloat w_ic = params_mat(0, c),
          w_fc = params_mat(1, c),
          w_oc = params_mat(2, c);
      // derivative sums w.r.t. parameters.
      BaseFloat w_ic_deriv_sum = 0.0,
          w_fc_deriv_sum = 0.0,
          w_oc_deriv_sum = 0.0;

      // average derivatives, for self-repair.
      // The 5 nonlinearities that are subject to self-repair are written as:
      //  Sigmoid(i_t_input), Sigmoid(f_t_input),
      //  Tanh(c_part), Sigmoid(o_t_input),  Tanh(c_t)
      BaseFloat i_t_self_repair = (deriv_sum_in(0, c) / count < sr_config(0) ?
                                   sr_config(5) : 0.0),
          f_t_self_repair = (deriv_sum_in(1, c) / count < sr_config(1) ?
                             sr_config(6) : 0.0),
          c_part_self_repair = (deriv_sum_in(2, c) / count < sr_config(2) ?
                                sr_config(7) : 0.0),
          o_t_self_repair = (deriv_sum_in(3, c) / count < sr_config(3) ?
                             sr_config(8) : 0.0),
          c_t_self_repair = (deriv_sum_in(4, c) / count < sr_config(4) ?
                             sr_config(9) : 0.0);
      // Note on how we add self-repair for sigmoids/tanh's.  If self-repair
      // is activated for this unit, then...
      // For sigmoids we'd add -self_repair_scale * (2 * sigmoid(x) - 1.0)
      // ... to the input-deriv;
      // For tanh's we'd add -self_repair_scale * tanh(x)
      // If self-repair is not activated, the 'self_repair' scales are set to zero.

      // The following variables are for the accumulation of stats on the
      // sigmoid and tanh units.
      BaseFloat i_t_value_sum = 0.0, i_t_deriv_sum = 0.0,
          f_t_value_sum = 0.0, f_t_deriv_sum = 0.0,
          c_part_value_sum = 0.0, c_part_deriv_sum = 0.0,
          o_t_value_sum = 0.0, o_t_deriv_sum = 0.0,
          c_t_value_sum = 0.0, c_t_deriv_sum = 0.0;


      for (int32 r = 0; r < num_rows; r++) {
        BaseFloat i_part = input_mat(r, c),
            f_part = input_mat(r, c + cell_dim),
            c_part = input_mat(r, c + 2 * cell_dim),
            o_part = input_mat(r, c + 3 * cell_dim),
            c_prev = input_mat(r, c + 4 * cell_dim);
        // For greater clarity, we give some of the quantities in the
        // forward equations their own names.
        BaseFloat i_t_input = i_part + w_ic * c_prev,
            i_t = ScalarSigmoid(i_t_input),
            f_t_input = f_part + w_fc * c_prev,
            f_t = ScalarSigmoid(f_t_input),
            tanh_c_part = ScalarTanh(c_part),
            c_t = f_t * c_prev + i_t * tanh_c_part,
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
        BaseFloat dc_t_out = output_deriv_mat(r, c),
            dm_t = output_deriv_mat(r, c + cell_dim),
            dtanh_c_t = o_t * dm_t,
            do_t = tanh_c_t * dm_t,
            do_t_input = (o_t * (1.0F - o_t) * do_t -
                          (2.0F * o_t - 1.0F) * o_t_self_repair),
            dc_t = ((1.0F - tanh_c_t * tanh_c_t) * dtanh_c_t +
                    dc_t_out + do_t_input * w_oc) - tanh_c_t * c_t_self_repair,
            dtanh_c_part = i_t * dc_t,
            df_t = dc_t * c_prev,
            df_t_input = (df_t * f_t * (1.0F - f_t) -
                          (2.0F * f_t - 1.0F) * f_t_self_repair),
            di_t = dc_t * tanh_c_part,
            di_t_input = (di_t * i_t * (1.0F - i_t) -
                          (2.0F * i_t - 1.0F) * i_t_self_repair);

        w_ic_deriv_sum += c_prev * di_t_input;
        w_fc_deriv_sum += c_prev * df_t_input;
        w_oc_deriv_sum += c_t * do_t_input;

        BaseFloat dc_prev = w_ic * di_t_input + w_fc * df_t_input + f_t * dc_t,
            do_part = do_t_input,
            dc_part = ((1.0F - tanh_c_part * tanh_c_part) * dtanh_c_part -
                       tanh_c_part * c_part_self_repair),
            df_part = df_t_input,
            di_part = di_t_input;

        if (input_deriv_mat != NULL) {
          (*input_deriv_mat)(r, c) += di_part;
          (*input_deriv_mat)(r, c + cell_dim) += df_part;
          (*input_deriv_mat)(r, c + 2 * cell_dim) += dc_part;
          (*input_deriv_mat)(r, c + 3 * cell_dim) += do_part;
          (*input_deriv_mat)(r, c + 4 * cell_dim) += dc_prev;
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
          (*self_repair_sum_out_mat)(i, c) +=
              (deriv_sum_in(i, c) / count < sr_config(i) ? num_rows : 0);

        (*deriv_sum_out_mat)(0, c) += i_t_deriv_sum;
        (*deriv_sum_out_mat)(1, c) += f_t_deriv_sum;
        (*deriv_sum_out_mat)(2, c) += c_part_deriv_sum;
        (*deriv_sum_out_mat)(3, c) += o_t_deriv_sum;
        (*deriv_sum_out_mat)(4, c) += c_t_deriv_sum;
      }
    }
  }
}




} //namespace cu

} //namespace kaldi
