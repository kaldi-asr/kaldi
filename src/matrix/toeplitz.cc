// matrix/toeplitz.cc

// Copyright 2015  Hakan Erdogan

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "matrix/toeplitz.h"

namespace kaldi {

// Solve the nxn Toeplitz system Rx=y
// R is Toeplitz, does not have to be symmetric
// r is of size n and contains first row of R from 0 to n-1
// c is of size n and contains first column of R from 0 to n-1, r[0]==c[0] (duplicated)
// r and c can be the same (symmetric-Toeplitz case)
// y is of size n from 0 to n-1
// x is the solution from 0 to n-1

template<typename Real>
void toeplitz_solve(const Vector<Real> &rvec, const Vector<Real> &cvec, const Vector<Real> &yvec, Vector<Real> *xvec, Real tol_factor)
{
    int32 j,k,m,mp1,mp2,m2,nm1;
    Real pp,pt1,pt2,qq,qt1,qt2,sd,sgd,sgn,shn,sxn;
    const Real *r,*c,*y;
    Real *x;

    int32 n=rvec.Dim();
    r=rvec.Data();
    c=cvec.Data();
    y=yvec.Data();
    x=xvec->Data();
    Real tolerance=r[0]/tol_factor; // tol_factor larger means lower tolerance
    nm1=n-1;
    KALDI_ASSERT(r[0] == c[0]);
    if (r[0] == 0.0) KALDI_ERR << "toeplitz_solve: singular input matrix, detection 1";
    x[0]=y[0]/r[0];
    if (nm1 == 0) return;
    Real g[nm1];
    Real h[nm1];

    g[0]=r[1]/r[0];
    h[0]=c[1]/r[0];
    for (m=0; m<n; m++) {
        mp1=m+1;
        mp2=mp1+1;
        sxn = -y[mp1];
        sd = -r[0];
        for (j=0; j<mp1; j++) {
            sxn += c[mp1-j]*x[j];
            sd += c[mp1-j]*g[m-j];
        }
        if (std::abs(sd) <= tolerance) KALDI_ERR << "toeplitz_solve: singular input matrix, detection 2";
        x[mp1]=sxn/sd; // init x[1] through x[n-1]
        for (j=0; j<mp1; j++)
            x[j] -= x[mp1]*g[m-j]; // update x[j] for j less than or equal to m
        if (mp1 == nm1) return; // returns from function here
        sgn = -r[mp2];
        shn = -c[mp2];
        sgd = -r[0];
        for (j=0; j<mp1; j++) {
            sgn += r[mp1-j]*g[j];
            shn += c[mp1-j]*h[j];
            sgd += r[mp1-j]*h[m-j];
        }
        if (std::abs(sgd) <= tolerance) KALDI_ERR << "toeplitz_solve: singular input matrix, detection 3";
        g[mp1]=sgn/sgd;
        h[mp1]=shn/sd;
        k=m;
        m2=(mp2) >> 1;
        pp=g[mp1];
        qq=h[mp1];
        for (j=0; j<m2; j++) {
            pt1=g[j];
            pt2=g[k];
            qt1=h[j];
            qt2=h[k];
            g[j]=pt1-pp*qt2;
            g[k]=pt2-pp*qt1;
            h[j]=qt1-qq*pt2;
            h[k--]=qt2-qq*pt1;
        }
    }
    KALDI_ERR << "toeplitz_solve: out of the return loop, you should not be here!";
}

template<typename Real>
void make_toeplitz_matrix(const Vector<Real> &r, Matrix<Real> *rmat)
{
  int32 n=r.Dim();
  for (int32 i=0; i<n; i++)
    for (int32 j=0; j<n; j++)
      (*rmat)(i,j)= r(std::abs(i-j));
}

template<typename Real>
void make_nonsym_toeplitz_matrix(const Vector<Real> &r, const Vector<Real> &c,  Matrix<Real> *rmat)
{
  int32 n=r.Dim();
  KALDI_ASSERT(n==c.Dim());
  KALDI_ASSERT(r(0)==c(0));
  for (int32 i=0; i<n; i++)
    for (int32 j=i; j<n; j++)
      (*rmat)(i,j)= r(j-i);
  for (int32 i=0; i<n; i++)
    for (int32 j=0; j<i; j++)
      (*rmat)(i,j)= c(i-j);
}

template void toeplitz_solve(const Vector<float> &rvec, const Vector<float> &cvec, const Vector<float> &yvec, Vector<float> *xvec, float tol_factor);
template void toeplitz_solve(const Vector<double> &rvec, const Vector<double> &cvec, const Vector<double> &yvec, Vector<double> *xvec, double tol_factor);
template void make_toeplitz_matrix(const Vector<float> &r, Matrix<float> *rmat);
template void make_toeplitz_matrix(const Vector<double> &r, Matrix<double> *rmat);
template void make_nonsym_toeplitz_matrix(const Vector<float> &r, const Vector<float> &c, Matrix<float> *rmat);
template void make_nonsym_toeplitz_matrix(const Vector<double> &r, const Vector<double> &c, Matrix<double> *rmat);

}  // namespace kaldi
