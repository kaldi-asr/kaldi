// cudamatrix/cu-block-matrix.h

// Copyright 2013      Johns Hopkins University (author: Daniel Povey)

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



#ifndef KALDI_CUDAMATRIX_CU_BLOCK_MATRIX_H_
#define KALDI_CUDAMATRIX_CU_BLOCK_MATRIX_H_

#include <sstream>

#include <vector>
#include "cudamatrix/cu-common.h"


namespace kaldi {


/**
   The class CuBlockMatrix holds a vector of objects of type CuMatrix,
   say, M_1, M_2, .. M_N
   and it represents the matrix diag(M_1, M_2, ... M_N).  Note:
   the individual matrices do not have to be square.  The reason the
   class is needed is mostly so that we can efficiently multiply by this
   block-diagonal structure in a parallel way.

   If we have a GPU available, CuBlockMatrix will store a copy of the
   individual CuMatrix quantities M_1 .. M_N on the GPU, but their
   'primary' home remains on the CPU.. what we mean by this is that
   while the data remains on the GPU, the "primary" version of the
   Matrix object that holds the pointers will remain on the CPU.
   We just copy it over to the GPU whenever it is changed.
 */

template<typename Real>
class CuBlockMatrix {
 public:
  friend class CuMatrixBase<Real>;
  
  CuBlockMatrix();

  CuBlockMatrix(const std::vector<CuMatrix<Real> > &data);

  ~CuBlockMatrix() { Destroy(); }
  
  /// Copy constructor
  CuBlockMatrix(const CuBlockMatrix &other); 

  /// Assignment operator
  CuBlockMatrix &operator= (const CuBlockMatrix &other); 

  void Write(std::ostream &os, bool binary) const;
  
  void Read(std::istream &is, bool binary);

  MatrixIndexT NumRows() const { return num_rows_; }

  MatrixIndexT NumCols() const { return data_.num_cols_; }

  MatrixIndexT NumBlocks() const { return block_data_.size(); }
  
  // Returns max num-columns of any block
  MatrixIndexT MaxBlockCols() const ;

  // Returns max num-rows of any block
  MatrixIndexT MaxBlockRows() const;
    
  const CuSubMatrix<Real> Block(MatrixIndexT b) const;

  CuSubMatrix<Real> Block(MatrixIndexT b); // return CuMatrixBase to disallow resizes.


  /// Does *this = alpha A B + beta * *this, discarding elements of the product outside
  /// the block structure of the *this matrix.  The transA and transB parameters
  /// can be used to substitute A^T for A and B^T for B, respectively.
  void AddMatMat(BaseFloat alpha,
                 const CuMatrix<Real> &A, MatrixTransposeType transA,
                 const CuMatrix<Real> &B, MatrixTransposeType transB,
                 BaseFloat beta);


  /// Copies elements within the block structure from matrix M, discarding others.
  /// Note: this has not been implemented in a very efficient way, it's used only
  /// for testing.
  void CopyFromMat(const CuMatrix<Real> &M);

  /// Normalizes the columns of *this so that each one sums to one.
  /// On error (e.g. inf's), will set the column to a constant value that
  /// sums to one.
  void NormalizeColumns();

  void Swap(CuBlockMatrix *other);
  
 protected:
  CuMatrix<Real> data_; // This is a single matrix into which
  // we pack all the blocks (possibly with spaces left over)

  struct BlockMatrixData{
    MatrixIndexT num_rows;
    MatrixIndexT num_cols;
    MatrixIndexT row_offset;
    MatrixIndexT col_offset;
  };
  

#if HAVE_CUDA == 1
  const CuBlockMatrixData* CuData() const { return cu_data_; }
#endif
 private:
  
  /// If using GPU and cu_data_ != NULL, free cu_data_ and set it to NULL
  void FreeCudaData();
  /// If using GPU, allocate and set cu_data_ on the GPU to reflect "data_".
  void SetCudaData();


  /// Frees and deinitializes everything.
  void Destroy();

  std::vector<BlockMatrixData> block_data_;
  
  MatrixIndexT num_rows_; // sum of num_rows of elements of block_data_.
#if HAVE_CUDA == 1
  CuBlockMatrixData *cu_data_; // We store the pointers and some additional info
                               // on the GPU card in a form more suited to
                               // use by CUDA kernels.
#endif
}; // class CuBlockMatrix

template<typename Real>
std::ostream &operator << (std::ostream &out, const CuBlockMatrix<Real> &mat);


} // namespace Kaldi
#endif
