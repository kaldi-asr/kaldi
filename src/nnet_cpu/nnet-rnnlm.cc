// nnet/nnet-rnnlm.cc

// Copyright 2011  Karel Vesely

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


#include "nnet_cpu/nnet-rnnlm.h"


///////////////////// TODO HACK, REWRITE AS INLINE!!!! ///////////////
///// fast exp() implementation
static union{
  double d;
  struct{
    int j, i;
  } n;
} d2i;
#define EXP_A (1048576/M_LN2)
#define EXP_C 60801
#define FAST_EXP(y) (d2i.n.i = EXP_A*(y)+(1072693248-EXP_C), d2i.d)
///////////////////// TODO HACK, REWRITE AS INLINE!!!! ///////////////




namespace kaldi {

template<typename Real>
static void MyApplySoftMax(VectorBase<Real>& v) {
  Real max = v.Max();
  v.Add(-max);
  v.ApplyExp();
  Real sum = v.Sum();
  v.Scale(1.0/sum);
}

template<typename Real>
static void MyApplySoftMax2(VectorBase<Real>& v) {

  Real* p=v.Data();
  Real max = -1e20;
  for(int32 i=0; i<v.Dim(); i++) {
    Real v = *p++;
    if (v > max) max = v;
  }

  p = v.Data();
  Real sum = 0.0;
  for(int32 i=0; i<v.Dim(); i++) {
    // Real v = exp(*p-max);
    Real v = FAST_EXP(*p-max);
    *p = v; sum += v;
    p++;
  }
  v.Scale(1.0/sum);

}


void Rnnlm::Propagate(const std::vector<int32>& in, Matrix<BaseFloat>* out) {
  // resize hidden buffer
  h2_.Resize(in.size(), V1_.NumCols(), kSetZero);

  // forward pass over recurrent layer
  for(int32 r=0; r<h2_.NumRows(); r++) {
    h2_.Row(r).CopyFromVec(b1_);
    if(in[r] > 0) { // don't add V1 row for OOV words
      h2_.Row(r).AddVec(1.0, V1_.Row(in[r]-1));
    }
    if (r==0 && preserve_) {
      h2_.Row(r).AddMatVec(1.0, U1_, kTrans, h2_last_, 1.0);
    }
    if (r>0) {
      h2_.Row(r).AddMatVec(1.0, U1_, kTrans, h2_.Row(r-1), 1.0);
    }
    for(int32 c=0; c<h2_.NumCols(); c++) {
      // h2_(r,c) = 1.0/(1.0+exp(-h2_(r,c)));
      h2_(r, c) = 1.0/(1.0+FAST_EXP(-h2_(r, c)));
    }
  }

  // TODO factorization

  // forward pass second layer
  out->Resize(in.size(), W2_.NumCols(), kSetZero);
  for(int32 r=0; r<out->NumRows(); r++) {
    out->Row(r).CopyFromVec(b2_);
  }
  out->AddMatMat(1.0, h2_, kNoTrans, W2_, kNoTrans, 1.0);
  for(int32 r=0; r<out->NumRows(); r++) {
    // out->Row(r).ApplySoftMax();
    SubVector<BaseFloat> v(out->Row(r));
    MyApplySoftMax2(v);
  }
  
  // copy the input sequence
  in_seq_ = in;
  // copy the last state of hidden layer
  h2_last_.CopyFromVec(h2_.Row(h2_.NumRows()-1));
}


void Rnnlm::Backpropagate(const MatrixBase<BaseFloat>& in_err) {

  // prepare error buffer
  e2_.Resize(in_err.NumRows(), h2_.NumCols(), kSetZero);

  // LAYER2
  // backpropagate error
  e2_.AddMatMat(1.0, in_err, kNoTrans, W2_, kTrans, 0.0);
  for(int32 r=0; r<e2_.NumRows(); r++) {
    for(int32 c=0; c<e2_.NumCols(); c++) {
      e2_(r, c) *= h2_(r, c)*(1.0-h2_(r, c));
    }
  }
  // update layer2
  W2_.AddMatMat(-learn_rate_, h2_, kTrans, in_err, kNoTrans, 1.0);
  b2_corr_.SetZero();
  b2_corr_.AddRowSumMat(in_err);
  b2_.AddVec(-learn_rate_, b2_corr_);

  // LAYER1
  V1_corr_.SetZero();
  U1_corr_.SetZero();
  b1_corr_.SetZero();
  // accumulate gradient for layer1
  b1_corr_.AddRowSumMat(e2_);
  for(int32 r=0; r<e2_.NumRows();r++) {
    V1_corr_.Row(in_seq_[r]-1).AddVec(1.0, e2_.Row(r));
  }
  // shift the hidden layer forward in time!!!
  // (ie. compensate recurrence delay of 1 timestep)
  e2_.RemoveRow(0);
  // h2_.RemoveRow(h2_.NumRows()-1);
  {
    SubMatrix<BaseFloat> h2win(h2_, 0, h2_.NumRows()-1, 0, h2_.NumCols());
    U1_corr_.AddMatMat(1.0, h2win, kTrans, e2_, kNoTrans, 0.0);
  }


  ///////////////////////////////////////////
  // accumulate BPTT graident
  e2_bptt_[0].Resize(e2_.NumRows(), e2_.NumCols());
  e2_bptt_[0].CopyFromMat(e2_);
  for(int32 step=1;step<=bptt_;step++) {
    // time shift
    e2_bptt_[(step+1)%2].RemoveRow(0);
    // h2_.RemoveRow(h2_.NumRows()-1);
    SubMatrix<BaseFloat> h2win(h2_, 0, h2_.NumRows()-1-step, 0, h2_.NumCols());

    // BPTT time step
    e2_bptt_[step%2].Resize(e2_bptt_[(step+1)%2].NumRows(), U1_.NumRows());
    e2_bptt_[step%2].AddMatMat(1.0, e2_bptt_[(step+1)%2], kNoTrans, U1_, kTrans, 0.0);
    // apply y*(1-y) -sigmoid derivative
    Matrix<BaseFloat>& E = e2_bptt_[step%2];
    KALDI_ASSERT(E.NumRows() == h2win.NumRows());
    KALDI_ASSERT(E.NumCols() == h2win.NumCols());
    for(int32 r=0; r<E.NumRows(); r++) {
      for(int32 c=0; c<E.NumCols(); c++) {
        E(r, c) *= h2win(r, c)*(1.0-h2win(r, c));
      }
    }

    // accumulate graidient
    b1_corr_.AddRowSumMat(E);
    for(int32 r=0; r<E.NumRows();r++) {
      // :TODO: IS IT CORRECT?
      // V1_corr_.Row(in_seq_[r+step]).AddVec(1.0,E.Row(r));
      V1_corr_.Row(in_seq_[r]-1).AddVec(1.0, E.Row(r));
    }
    U1_corr_.AddMatMat(1.0, h2win, kTrans, E, kNoTrans, 0.0);
  }
  ///////////////////////////////////////////

  // update layer1
  V1_.AddMat(-learn_rate_, V1_corr_);
  U1_.AddMat(-learn_rate_, U1_corr_);
  b1_.AddVec(-learn_rate_, b1_corr_);

}

BaseFloat Rnnlm::Score(const std::vector<int32>& seq, const VectorBase<BaseFloat>* hid, int32 prev_wrd) {
  
  std::vector<int32>in;
  in.push_back(prev_wrd);
  in.insert(in.end(), seq.begin(), seq.end());

  if (NULL != hid) {
    HidVector(*hid);
  }

  Matrix<BaseFloat> out;
  Propagate(in, &out);

  BaseFloat llk = 0.0;
  for(size_t i=0; i<seq.size(); i++) {
    llk += log(out(i, seq[i]-1));
  }

  return llk;
}

} // namespace
