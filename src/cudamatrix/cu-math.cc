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

template<typename Real>
void CpuAddSpatialRegularizationDeriv(const MatrixBase<Real>& out_value,
                                      Real scale, MatrixBase<Real>* deriv,
                                      Real* regularization_sumsq) {
  const int kSectWidth = 16;
  KALDI_ASSERT(SameDim(*deriv, out_value));

  const int rows = out_value.NumRows();
  const int cols = out_value.NumCols();
  const int ext_cols = ((cols + (kSectWidth - 1)) / kSectWidth) * kSectWidth;
  Matrix<Real> val_ext(rows, ext_cols);
  Matrix<Real> conv1(rows, ext_cols);

  val_ext.ColRange(0, cols).CopyFromMat(out_value);
  const int image_rows = ext_cols / kSectWidth;
  for (int32 i = 0; i < rows; i++) {
    Real* conv1_i = conv1.Data() + i * conv1.Stride();
    Real* val_ext_i = val_ext.Data() + i * val_ext.Stride();
    for (int32 j = 0; j < ext_cols; j++) {
      int image_x = j % kSectWidth;
      int image_x_left = (image_x + (kSectWidth - 1)) % kSectWidth;
      int image_x_right = (image_x + 1) % kSectWidth;
      int image_y = j / kSectWidth;
      int image_y_up = (image_y + (image_rows - 1)) % image_rows;
      int image_y_down = (image_y + 1) % image_rows;
      conv1_i[j] = val_ext_i[j]
          - Real(0.125)
              * (val_ext_i[image_y_up * kSectWidth + image_x_left]
                  + val_ext_i[image_y_up * kSectWidth + image_x]
                  + val_ext_i[image_y_up * kSectWidth + image_x_right]
                  + val_ext_i[image_y * kSectWidth + image_x_left]
                  + val_ext_i[image_y * kSectWidth + image_x_right]
                  + val_ext_i[image_y_down * kSectWidth + image_x_left]
                  + val_ext_i[image_y_down * kSectWidth + image_x]
                  + val_ext_i[image_y_down * kSectWidth + image_x_right]);
    }
  }
  if (regularization_sumsq) {
    *regularization_sumsq = TraceMatMat(conv1.ColRange(0, cols),
                                        conv1.ColRange(0, cols), kTrans);
  }
  for (int32 i = 0; i < rows; i++) {
    Real* conv1_i = conv1.Data() + i * conv1.Stride();
    Real* out_deriv_i = deriv->Data() + i * deriv->Stride();
    for (int32 j = 0; j < cols; j++) {
      int image_x = j % kSectWidth;
      int image_x_left = (image_x + (kSectWidth - 1)) % kSectWidth;
      int image_x_right = (image_x + 1) % kSectWidth;
      int image_y = j / kSectWidth;
      int image_y_up = (image_y + (image_rows - 1)) % image_rows;
      int image_y_down = (image_y + 1) % image_rows;
      out_deriv_i[j] -= scale
          * (conv1_i[j]
              - Real(0.125)
                  * (conv1_i[image_y_up * kSectWidth + image_x_left]
                      + conv1_i[image_y_up * kSectWidth + image_x]
                      + conv1_i[image_y_up * kSectWidth + image_x_right]
                      + conv1_i[image_y * kSectWidth + image_x_left]
                      + conv1_i[image_y * kSectWidth + image_x_right]
                      + conv1_i[image_y_down * kSectWidth + image_x_left]
                      + conv1_i[image_y_down * kSectWidth + image_x]
                      + conv1_i[image_y_down * kSectWidth + image_x_right]));
    }
  }
}

template<typename Real>
void AddSpatialRegularizationDeriv(const CuMatrixBase<Real>& out_value,
                                   Real scale, CuMatrixBase<Real>* deriv,
                                   Real* regularization_sumsq) {
  const unsigned int kBlockSize = 256;
  const unsigned int kSectWidth = 16;
  const int kActiveSize = kBlockSize - 2 * kSectWidth;
  KALDI_ASSERT(SameDim(*deriv, out_value));
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    // The reshaped image need to have at least 2 rows.
    // This is an implementation limit of the CUDA kernel function.
    KALDI_ASSERT(deriv->NumCols() > kSectWidth);
    Timer tim;
    dim3 dimBlock(kSectWidth, kBlockSize / kSectWidth);
    dim3 dimGrid(n_blocks(deriv->NumCols(), kActiveSize), deriv->NumRows());
    if (regularization_sumsq) {
      CuVector<Real> sumsq(dimGrid.x * dimGrid.y);
      cuda_add_spatial_regularization_deriv(dimGrid, dimBlock, out_value.Data(),
                                            out_value.Dim(), deriv->Data(),
                                            deriv->Stride(), scale,
                                            sumsq.Data());
      *regularization_sumsq = sumsq.Sum();
    } else {
      cuda_add_spatial_regularization_deriv(dimGrid, dimBlock, out_value.Data(),
                                            out_value.Dim(), deriv->Data(),
                                            deriv->Stride(), scale, NULL);
    }
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    CpuAddSpatialRegularizationDeriv(out_value.Mat(), scale, &(deriv->Mat()),
                                     regularization_sumsq);
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

template
void CpuAddSpatialRegularizationDeriv(const MatrixBase<float>& out_value,
                                      float scale, MatrixBase<float>* deriv,
                                      float* regularization_sumsq);
template
void CpuAddSpatialRegularizationDeriv(const MatrixBase<double>& out_value,
                                      double scale, MatrixBase<double>* deriv,
                                      double* regularization_sumsq);

template
void AddSpatialRegularizationDeriv(const CuMatrixBase<float>& out_value,
                                   float scale, CuMatrixBase<float>* deriv,
                                   float* regularization_sumsq);
template
void AddSpatialRegularizationDeriv(const CuMatrixBase<double>& out_value,
                                   double scale, CuMatrixBase<double>* deriv,
                                   double* regularization_sumsq);



} //namespace cu

} //namespace kaldi

