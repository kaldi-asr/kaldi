#if HAVE_CUDA == 1
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include "util/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cublas-wrappers.h"

namespace kaldi {

template<typename Real>
void CuSpMatrix<Real>::CopyFromMat(const CuMatrixBase<Real> &M,
                                   SpCopyType copy_type) {
  KALDI_ASSERT(this->num_rows_ == M.NumRows() &&
               this->num_rows_ == M.NumCols());
  if (this->num_rows_ == 0)
    return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    MatrixIndexT D = this->NumRows();
    if (D == 0)
      return;
    switch (copy_type) {
      case kTakeMeanAndCheck:
        KALDI_ERR << "kTakeMeanAndCheck not supported!";
      // The grid/block dimensions have been very roughly tuned for the
      // individual cases.
      case kTakeMean:
        {
          dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
          dim3 dimGrid(n_blocks(D, CU2DBLOCK), n_blocks(D, CU2DBLOCK));
          cuda_take_mean(dimGrid, dimBlock, M.Data(), this->data_, M.Dim());
          CU_SAFE_CALL(cudaGetLastError());
        }
        break;
      case kTakeLower:
        {
          dim3 dimBlock(1, CU1DBLOCK);
          dim3 dimGrid(D, n_blocks(D, CU1DBLOCK));
          cuda_take_lower(dimGrid, dimBlock, M.Data(), this->data_, M.Dim());
          CU_SAFE_CALL(cudaGetLastError());
        }
        break;
      case kTakeUpper:
        {
          dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
          dim3 dimGrid(n_blocks(D, CU2DBLOCK), n_blocks(D, CU2DBLOCK));
          cuda_take_upper(dimGrid, dimBlock, M.Data(), this->data_, M.Dim());
          CU_SAFE_CALL(cudaGetLastError());
        }
        break;
      default:
        KALDI_ASSERT("Invalid argument to CuSpMatrix::CopyFromMat");
    }
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::CopyFromMat(from CuMatrixBase)", tim.Elapsed());
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), copy_type);
  }
}

template<typename Real>
void CuSpMatrix<Real>::Invert() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuMatrix<Real> mat(this->num_rows_, this->num_rows_);
    mat.CopyFromSp(*this);
    mat.SymInvertPosDef();
    this->CopyFromMat(mat);
  } else
#endif
  { // Use inversion of CPU-based SpMatrix.
    Mat().Invert();
  }
}



template<typename Real>
void CuSpMatrix<Real>::InvertPosDefApprox(BaseFloat max_error) {
  if (this->num_rows_ == 0) return;
  MatrixIndexT dim = this->num_rows_;
  CuMatrix<Real> temp(dim * 5, dim);
  CuSubMatrix<Real> A(temp, 0, dim, 0, dim),
      AA(temp, dim, dim, 0, dim),
      AAA(temp, 2 * dim, dim, 0, dim),
      AAAA(temp, 3 * dim, dim, 0, dim);
  Real prescale = dim / this->Trace();
  this->Scale(prescale); // We'll compute the inverse of the prescaled A, and then
                         // put that factor back later.  This is useful since we
                         // deal with high powers of A that could get large or small.
  A.CopyFromSp(*this);
  // use *this as a temporary SpMatrix; we've stored its contents in "A".
  this->AddMat2(1.0, A, kNoTrans, 0.0);
  AA.CopyFromSp(*this);
  { // now create AAA and AAAA using a single multiplication.
    CuSubMatrix<Real> A_and_AA(temp, 0, dim * 2, 0, dim),
        AAA_and_AAAA(temp, dim * 2, dim * 2, 0, dim);
    // Note: below, the transpose-ness of AA is arbitrary since it's symmetric;
    // I guess that transposed may be faster.
    AAA_and_AAAA.AddMatMat(1.0, A_and_AA, kNoTrans, AA, kTrans, 0.0);
  }

  // Note: below, trace_A equals dim because of the prescaling, we
  // ensured that.
  Vector<double> trace(8); // trace(i) is trace(A^(i+1))
  trace(0) = dim;
  {  
    CuVector<Real> trace_vec(dim * 5);
    CuSubVector<Real> trace_lower4(trace_vec, 0, dim * 4),
        trace_lower3(trace_vec, 0, dim * 3),
        trace1(trace_vec, 0, dim), trace2(trace_vec, dim, dim),
        trace3(trace_vec, dim * 2, dim), trace4(trace_vec, dim * 3, dim),
        ones(trace_vec, dim * 4, dim);
    trace_lower4.AddDiagMat2(1.0, temp.Range(0, dim * 4, 0, dim),
                             kNoTrans, 0.0);
    ones.Set(1.0);
    // TODO: can make these vecvec's faster as fake matrix multiplies.
    trace(1) = VecVec(trace1, ones);
    trace(3) = VecVec(trace2, ones);
    trace(5) = VecVec(trace3, ones);
    trace(7) = VecVec(trace4, ones);
    // Now we want to get odd-numbered trace quantities, so multiply the
    // rows of A through AAA with the rows of AA through AAA.
    CuSubMatrix<Real> lower_three(temp, 0, dim * 3, 0, dim),
        upper_three(temp, dim, dim * 3, 0, dim);
    trace_lower3.AddDiagMatMat(1.0, lower_three, kNoTrans, upper_three, kTrans, 0.0);
    trace(2) = VecVec(trace1, ones);
    trace(4) = VecVec(trace2, ones);
    trace(6) = VecVec(trace3, ones);
  }
  { // Check the trace values.
    CuMatrix<Real> power(A);
    for (int32 i = 0; i < 8; i++) {
      double this_trace = power.Trace();
      AssertEqual(this_trace, trace(i));
      CuMatrix<Real> temp_power(power);
      power.AddMatMat(1.0, temp_power, kNoTrans, A, kNoTrans, 0.0);
    }
  }
  
  // We'll use a and B to get the coefficients.  These operations are in very
  // tiny dimensions -> faster and more convenient to use CPU.
  SubVector<double> a(trace, 0, 4);
  SpMatrix<double> B(4);
  for (int32 r = 0; r < 4; r++)
    for (int32 c = 0; c <= r; c++)
      B(r, c) = trace(r + c + 1);

  TpMatrix<double> C(4);
  C.Cholesky(B);
  C.Invert();
  SpMatrix<double> Binv(4);
  Binv.AddTp2(1.0, C, kTrans, 0.0);
  Vector<double> v(4);
  v.AddSpVec(1.0, Binv, a, 0.0);
  
  Real av = VecVec(a, v), vBv = VecSpVec(v, B, v),
      error = (vBv + dim) - 2.0 * av;
  

  KALDI_ASSERT(error >= 0.0); // note: error is a squared Frobenius
                                      // norm.

  KALDI_VLOG(5) << "a is " << a << ", B is " << B;
  KALDI_VLOG(5) << "Dim is " << dim << ", error norm is " << sqrt(error);
    
  if (error <= max_error) {
    // It's sufficient to return with the approximation up to A^3.
    A.Scale(v(1));
    A.AddToDiag(v(0));
    A.AddMat(v(2), AA);
    A.AddMat(v(3), AAA);
    this->CopyFromMat(A, kTakeLower);
    this->Scale(prescale);
    return;
  } else {
    // Let X be the approximate inverse of A: X = v(0) I + v(1) A + v(2) A^2 + v(3) A^3.
    // Let AX be A times X: AX = v(0) A + v(1) A^2 + v(2) A^3 + v(3) A^4.
    // We can construct both X and AX out of quantities we've already computed.

    CuSubMatrix<Real> X(temp, dim * 4, dim, 0, dim),
        AX(temp, dim * 3, dim, 0, dim);

    AX.Scale(v(3));  // AX re-uses memory of AAAA: scale that.
    AX.AddMat(v(2), AAA);
    AX.AddMat(v(1), AA);
    AX.AddMat(v(0), A);

    X.AddMat(v(3), AAA); // X was zero before; space never used.
    X.AddMat(v(2), AA);
    X.AddMat(v(1), A);
    X.AddToDiag(v(0));

    int32 num_iters = 10;
    for (int32 i = 0; i < num_iters; i++) {
      CuSubMatrix<Real> AX_and_X(temp, dim * 3, dim * 2, 0, dim),
          AAXX_and_AXX(temp, dim, dim * 2, 0, dim); // Note: in our variable-naming
      // conventions we put the A's first; since all quantities commute it doesn't
      // matter which order we put them in.  Note: the transpose of AX below is
      // arbitrary (it's symmetric); I think it might be more efficient.`
      AAXX_and_AXX.AddMatMat(1.0, AX_and_X, kNoTrans, AX, kTrans, 0.0);

      // The iteration now is X' <--- X (2I - AX).  This is the iteration of
      // Schulz/Hoteling/whatever.  To get the objf (and for the next iteration)
      // we also want AX'.  Use X' <-- 2X - AXX, and AX' <-- 2AX - AAXX.
      // They go in the same place as before.  For now on, forget about the dash
      // on the X, we'll just call it X.
      AX_and_X.Scale(2.0);
      AX_and_X.AddMat(-1.0, AAXX_and_AXX);

      // The squared error is  ||AX - I||^2 = tr((AX - I)(AX - I)) = tr(AX^T AX) + dim - 2 tr(AX)
      Real a = TraceMatMat(AX, AX, kTrans), b = AX.Trace();
      error = a + dim - 2 * b;
      
      KALDI_VLOG(5) << "Better-inverse error is "
                    <<  sqrt(error);
      if (error <= max_error) {
        this->CopyFromMat(X, kTakeLower);
        this->Scale(prescale);
        return;
      }
    }
    KALDI_ASSERT("Error: max iters reached."); // TODO
  }
}


template<typename Real>
void CuSpMatrix<Real>::AddVec2(const Real alpha, const CuVectorBase<Real> &v) {
  KALDI_ASSERT(v.Dim() == this->NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    size_t nr = this->num_rows_;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(nr, CU2DBLOCK), n_blocks(nr, CU2DBLOCK));

    cublas_spr('U', this->num_rows_, alpha, v.Data(),
               1, this->Data());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::AddVec2", tim.Elapsed());
  } else
#endif
  {
    Mat().AddVec2(alpha, v.Vec());
  }
}

template<typename Real>
void CuSpMatrix<Real>::AddMat2(const Real alpha, const CuMatrixBase<Real> &M,
                               MatrixTransposeType transM, const Real beta) {
  KALDI_ASSERT((transM == kNoTrans && this->NumRows() == M.NumRows())
               || (transM == kTrans && this->NumRows() == M.NumCols()));

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    MatrixIndexT this_dim = this->NumRows(),
        m_other_dim = (transM == kNoTrans ? M.NumCols() : M.NumRows());

    if (this_dim == 0) return;
    if (alpha == 0.0) {
      if (beta != 1.0) this->Scale(beta);
      return;
    }

    char trans = (transM == kTrans ? 'N' : 'T');

    CuMatrix<Real> tmp_mat(*this);
    cublas_syrk('U', trans, this_dim, m_other_dim, alpha, M.Data(),
                M.Stride(), beta, tmp_mat.Data(), tmp_mat.Stride());
    this->CopyFromMat(tmp_mat, kTakeLower);
    
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::AddMat2", tim.Elapsed());
  } else
#endif
  {
    Mat().AddMat2(alpha, M.Mat(), transM, beta);
  }
}

/**
 * C++ templatd wrapper of ANSI-C CUBLAS function GEMM (matrix multiply)
 */

template<typename Real, typename OtherReal>
Real TraceSpSp(const CuSpMatrix<Real> &A, const CuSpMatrix<OtherReal> &B) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    MatrixIndexT nr = A.NumRows(), size = nr * (nr+1) / 2;
    CuVector<Real> Adiag(nr, kUndefined);
    CuVector<OtherReal> Bdiag(nr, kUndefined);
    Adiag.CopyDiagFromPacked(A);
    Bdiag.CopyDiagFromPacked(B);
    CuSubVector<Real> Aall(A.Data(), size);
    CuSubVector<OtherReal> Ball(B.Data(), size);
    // Below, we subtrace VecVec(Adiag, Bdiag) to remove double-counting
    // on the diagonal.
    return 2.0 * VecVec(Aall, Ball) - VecVec(Adiag, Bdiag);
  } else
#endif
  {
    return TraceSpSp(A.Mat(), B.Mat());
  }
}
template
float TraceSpSp(const CuSpMatrix<float> &A, const CuSpMatrix<float> &B);
template
float TraceSpSp(const CuSpMatrix<float> &A, const CuSpMatrix<double> &B);
template
double TraceSpSp(const CuSpMatrix<double> &A, const CuSpMatrix<float> &B);
template
double TraceSpSp(const CuSpMatrix<double> &A, const CuSpMatrix<double> &B);


template<typename Real>
bool CuSpMatrix<Real>::ApproxEqual(const CuSpMatrix<Real> &B, Real tol) const {
  KALDI_ASSERT(this->NumRows() == B.NumRows());
  CuSpMatrix<Real> diff(*this);
  diff.AddSp(-1.0, B);
  Real a = this->FrobeniusNorm(), b = B.FrobeniusNorm(),
      d = diff.FrobeniusNorm();
  return (d <= tol * std::max(a, b));
}

template<typename Real>
bool CuSpMatrix<Real>::IsUnit(Real tol) const {
  // want to return:
  //FrobeniusNorm(*this - I) <= tol * NumRows(), i.e.:
  //sqrt (trace((*this - I)(*this-I)) <= tol * NumRows()
  //    trace((*this - I)(*this - I)) <= tol * NumRows()
  // trace(*this * *this) + trace(I) - 2 * trace(*this) <= tol * NumRows()
  // trace(*this * *this) + dim - 2*this.Trace() <= tol * NumRows()

  // Note: we could do this more efficiently still, by slightly changing the
  // definition of IsUnit and getting rid of the extra stuff inside TraceSpSp
  // that corrects for the diagonal being counted twice.
  
  return (TraceSpSp(*this, *this) + this->NumRows() - 2.0 * this->Trace() <=
          tol * this->NumRows());
}


template class CuSpMatrix<float>;
template class CuSpMatrix<double>;



} // namespace
