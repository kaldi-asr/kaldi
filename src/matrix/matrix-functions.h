// matrix/matrix-functions.h

// Copyright 2009-2011  Microsoft Corporation;  Go Vivace Inc.;  Jan Silovsky;
//                      Yanmin Qian;   1991 Henrique (Rico) Malvar (*)
//
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
//
// (*) incorporates, with permission, FFT code from his book
// "Signal Processing with Lapped Transforms", Artech, 1992.



#ifndef KALDI_MATRIX_MATRIX_FUNCTIONS_H_
#define KALDI_MATRIX_MATRIX_FUNCTIONS_H_

#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {

/// @addtogroup matrix_funcs_misc
/// @{

/** The function ComplexFft does an Fft on the vector argument v.
   v is a vector of even dimension, interpreted for both input
   and output as a vector of complex numbers i.e.
   \f[ v = ( re_0, im_0, re_1, im_1, ... )    \f]

   If "forward == true" this routine does the Discrete Fourier Transform
   (DFT), i.e.:
   \f[   vout[m] \leftarrow \sum_{n = 0}^{N-1} vin[i] exp( -2pi m n / N )  \f]

   If "backward" it does the Inverse Discrete Fourier Transform (IDFT)
   *WITHOUT THE FACTOR 1/N*,
   i.e.:
   \f[   vout[m] <-- \sum_{n = 0}^{N-1} vin[i] exp(  2pi m n / N )   \f]
   [note the sign difference on the 2 pi for the backward one.]

   Note that this is the definition of the FT given in most texts, but
   it differs from the Numerical Recipes version in which the forward
   and backward algorithms are flipped.

   Note that you would have to multiply by 1/N after the IDFT to get
   back to where you started from.  We don't do this because
   in some contexts, the transform is made symmetric by multiplying
   by sqrt(N) in both passes.   The user can do this by themselves.

   See also SplitRadixComplexFft, declared in srfft.h, which is more efficient
   but only works if the length of the input is a power of 2.
 */
template<typename Real> void ComplexFft (VectorBase<Real> *v, bool forward, Vector<Real> *tmp_work = NULL);

/// ComplexFt is the same as ComplexFft but it implements the Fourier
/// transform in an inefficient way.  It is mainly included for testing purposes.
/// See comment for ComplexFft to describe the input and outputs and what it does.
template<typename Real> void ComplexFt (const VectorBase<Real> &in,
                                     VectorBase<Real> *out, bool forward);

/// RealFft is a fourier transform of real inputs.  Internally it uses
/// ComplexFft.  The input dimension N must be even.  If forward == true,
/// it transforms from a sequence of N real points to its complex fourier
/// transform; otherwise it goes in the reverse direction.  If you call it
/// in the forward and then reverse direction and multiply by 1.0/N, you
/// will get back the original data.
/// The interpretation of the complex-FFT data is as follows: the array
/// is a sequence of complex numbers C_n of length N/2 with (real, im) format,
/// i.e. [real0, real_{N/2}, real1, im1, real2, im2, real3, im3, ...].
/// See also SplitRadixRealFft, declared in srfft.h, which is more efficient
/// but only works if the length of the input is a power of 2.

template<typename Real> void RealFft (VectorBase<Real> *v, bool forward);


/// RealFt has the same input and output format as RealFft above, but it is
/// an inefficient implementation included for testing purposes.
template<typename Real> void RealFftInefficient (VectorBase<Real> *v, bool forward);

/// ComputeDctMatrix computes a matrix corresponding to the DCT, such that
/// M * v equals the DCT of vector v.  M must be square at input.
/// This is the type = III DCT with normalization, corresponding to the
/// following equations, where x is the signal and X is the DCT:
/// X_0 = 1/sqrt(2*N) \sum_{n = 0}^{N-1} x_n
/// X_k = 1/sqrt(N) \sum_{n = 0}^{N-1} x_n cos( \pi/N (n + 1/2) k )
/// This matrix's transpose is its own inverse, so transposing this
/// matrix will give the inverse DCT.
/// Caution: the type III DCT is generally known as the "inverse DCT" (with the
/// type II being the actual DCT), so this function is somewhatd mis-named.  It
/// was probably done this way for HTK compatibility.  We don't change it
/// because it was this way from the start and changing it would affect the
/// feature generation.

template<typename Real> void ComputeDctMatrix(Matrix<Real> *M);


/// ComplexMul implements, inline, the complex multiplication b *= a.
template<typename Real> inline void ComplexMul(const Real &a_re, const Real &a_im,
                                            Real *b_re, Real *b_im);

/// ComplexMul implements, inline, the complex operation c += (a * b).
template<typename Real> inline void ComplexAddProduct(const Real &a_re, const Real &a_im,
                                                   const Real &b_re, const Real &b_im,
                                                   Real *c_re, Real *c_im);


/// ComplexImExp implements a <-- exp(i x), inline.
template<typename Real> inline void ComplexImExp(Real x, Real *a_re, Real *a_im);



/**
    ComputePCA does a PCA computation, using either outer products
    or inner products, whichever is more efficient.  Let D be
    the dimension of the data points, N be the number of data
    points, and G be the PCA dimension we want to retain.  We assume
    G <= N and G <= D.

    @param X [in]  An N x D matrix.  Each row of X is a point x_i.
    @param U [out] A G x D matrix.  Each row of U is a basis element u_i.
    @param A [out] An N x D matrix, or NULL.  Each row of A is a set of coefficients
         in the basis for a point x_i, so A(i, g) is the coefficient of u_i
         in x_i.
    @param print_eigs [in] If true, prints out diagnostic information about the
         eigenvalues.
    @param exact [in] If true, does the exact computation; if false, does
         a much faster (but almost exact) computation based on the Lanczos
         method.
*/

template<typename Real>
void ComputePca(const MatrixBase<Real> &X,
                MatrixBase<Real> *U,
                MatrixBase<Real> *A,
                bool print_eigs = false,
                bool exact = true);



// This function does: *plus += max(0, a b^T),
// *minus += max(0, -(a b^T)).
template<typename Real>
void AddOuterProductPlusMinus(Real alpha,
                              const VectorBase<Real> &a,
                              const VectorBase<Real> &b,
                              MatrixBase<Real> *plus,
                              MatrixBase<Real> *minus);

template<typename Real1, typename Real2>
inline void AssertSameDim(const MatrixBase<Real1> &mat1, const MatrixBase<Real2> &mat2) {
  KALDI_ASSERT(mat1.NumRows() == mat2.NumRows()
               && mat1.NumCols() == mat2.NumCols());
}


/*
   This class allows you to compute the class of function described in
    http://www.danielpovey.com/files/2018_svd_derivative.pdf
   and to backprop through that computation.
   Short summary: it allows you to apply some kind of scalar function
   to the singular values of a matrix, reconstruct it, and then backprop
   through that operation.

   This class is quite general-purpose in the sense that you can
   provide any scalar function; but in order to avoid things like
   passing function-pointers around, we had give it a rather clunky
   interface.  The way you are supposed to use it is as follows
   (to give an example):

      Matrix<BaseFloat> A(...);  // set it somehow.
      SvdRescaler rescaler(A);
      const VectorBase<BaseFloat> &lambda_in = A.InputSingularValues();
      VectorBase<BaseFloat> &lambda_out = *(A.OutputSingularValues());
      VectorBase<BaseFloat> &lambda_out_deriv = *(A.OutputSingularValueDerivs());
      for (int32 i = 0; i < lambda_in.size(); i++) {
        // compute the scalar function and its derivative for the singular
        // values.
        lambda_out(i) = some_func(lambda_in(i));
        lambda_out_deriv(i) = some_func_deriv(lambda_in(i));
      }
      Matrix<BaseFloat> B(A.NumRows(), A.NumCols(), kUndefined);
      rescaler.GetOutput(&B);
      // Do something with B.
      Matrix<BaseFloat> B_deriv(...);  // Get the derivative w.r.t. B
                                       // somehow.
      Matrix<BaseFloat> A_deriv(A.NumRows(), A.NumCols());  // Get the derivative w.r.t. A.


 */
class SvdRescaler {
 public:
  /*
    Constructor.
    'A' is the input matrix.  See class-level documentation above for
     more information.

    If 'symmetric' is set to true, then the user is asserting that A is
    symmetric, and that that symmetric structure needs to be preserved in the
    output.  In this case, we use code for the symmetric eigenvalue problem to
    do the decomposition instead of the SVD.  I.e. decompose A = P diag(s) P^T
    instead of A = U diag(s) V^T, using SpMatrix::Eig().  You can view this as a
    special case of SVD.
  */
  SvdRescaler(const MatrixBase<BaseFloat> &A, bool symmetric);

  // Constructor that takes no args.  In this case you are supposed to
  // call Init()
  SvdRescaler();

  // An alternative to the constructor that takes args.  Should only be called
  // directly after initializing the object with no args.  Warning: this object
  // keeps a reference to this matrix, so don't modify it during the lifetime
  // of this object.
  // This program assumes the input matrix (num_rows >= num_cols).
  void Init(const MatrixBase<BaseFloat> *A, bool symmetric);

  // Get the singular values of A, which will have been computed in the
  // constructor.  The reason why this is not const is that there may be
  // situations where you discover that the input matrix has some very small
  // singular values, and you want to (say) floor them somehow and reconstruct,
  // and have the derivatives be valid assuming you had given that 'repaired'
  // matrix A as input.  Modifying the elements of this vector gives you
  // a way to do that, although currently this class doesn't provide a way
  // for you to access that 'fixed-up' A directly.
  // We hope you know what you are doing if you modify these singular values.
  VectorBase<BaseFloat> &InputSingularValues();

  // Returns a pointer to a place that you can write the
  // modified singular values f(lambda).
  VectorBase<BaseFloat> *OutputSingularValues();

  // Returns a pointer to a place that you can write the
  // values of f'(lambda) (the function-derivative of f).
  VectorBase<BaseFloat> *OutputSingularValueDerivs();

  // Outputs F(A) to 'output', which must have the correct size.
  // It's OK if 'output' contains NaNs on entry.
  // Before calling this, you must have set the values in
  // 'OutputSingularValues()'.
  void GetOutput(MatrixBase<BaseFloat> *output);

  // Computes the derivative of some function g w.r.t. the input A,
  // given that dg/d(output) is provided in 'output_deriv'.
  // This derivative is *added* to 'input_deriv', so you need
  // to zero 'input_deriv' or otherwise set it, beforehand.
  // It is acceptable to call ComputeInputDeriv (with possibly different
  // values of 'output_deriv' and 'input_deriv' as many times as you want,
  // on the same object.
  void ComputeInputDeriv(const MatrixBase<BaseFloat> &output_deriv,
                         MatrixBase<BaseFloat> *input_deriv);

 protected:
    Matrix<BaseFloat> input_matrix_A_;
    bool symmetric_;
    Matrix<BaseFloat> U_, Vt_;
    Vector<BaseFloat> lambda_in_;
    Vector<BaseFloat> *lambda_out_, *lambda_out_deriv_;
};

/// @} end of "addtogroup matrix_funcs_misc"

} // end namespace kaldi

#include "matrix/matrix-functions-inl.h"


#endif
