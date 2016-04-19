// nnet/nnet-lhuc.h

#ifndef KALDI_NNET_NNET_LHUC_H_
#define KALDI_NNET_NNET_LHUC_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-matrix.h"

namespace kaldi {
namespace nnet1 {
class LhucComp : public UpdatableComponent {
 public:
  LhucComp(int32 dim_in, float lhuc_const, float learn_rate_coef)
     : UpdatableComponent(dim_in, dim_in),
       lhuc_const_(lhuc_const),
       lhuc_vec_(dim_in), lhuc_scale_vec_(dim_in),
       lhuc_vec_grad_(dim_in),
       learn_rate_coef_(learn_rate_coef) { }
  ~LhucComp() { }
  Component* Copy() const { return new LhucComp(*this); }
  ComponentType GetType() const { return kRescale; }
  
  void InitData(std::istream &is) { }
  void ReadData(std::istream &is, bool binary) { }
  void WriteData(std::ostream &os, bool binary) const { } 
  int32 NumParams() const { return lhuc_vec_.Dim(); }
  void GetParams(Vector<BaseFloat>* lhuc_vec_copy) const {
    lhuc_vec_copy->Resize(InputDim());
    lhuc_vec_.CopyToVec(lhuc_vec_copy);
  }
  void SetScaleVec(const Vector<BaseFloat> *lhuc_vec) {
    if(lhuc_vec != NULL) {
      KALDI_ASSERT(lhuc_vec->Dim() == lhuc_vec_.Dim());
      lhuc_vec_.CopyFromVec(*lhuc_vec);
    }
    Vector<BaseFloat> cpu_lhuc_vec(lhuc_vec_.Dim());
    lhuc_vec_.CopyToVec(&cpu_lhuc_vec);
    Vector<BaseFloat> cpu_scale_vec(lhuc_vec_.Dim());
    cpu_scale_vec.Sigmoid(cpu_lhuc_vec);
    cpu_scale_vec.Scale(lhuc_const_);
    lhuc_scale_vec_.CopyFromVec(cpu_scale_vec);
  }

  std::string Info() const {
    return std::string("\n lhuc_vec") + MomentStatistics(lhuc_vec_);
  }
  std::string InfoGradient() const {
    return std::string("\n lhuc_vec_grad") + MomentStatistics(lhuc_vec_grad_) +
            ", lr-coef " + ToString(learn_rate_coef_);
  }
  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    out->CopyFromMat(in);
    out->MulColsVec(lhuc_scale_vec_); 
  }
  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    in_diff->CopyFromMat(out_diff);
    in_diff->MulColsVec(lhuc_scale_vec_);
  }
  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {

    CuVector<BaseFloat> grad_vec(lhuc_scale_vec_);
    grad_vec.Scale(1.0/lhuc_const_); 
    CuVector<BaseFloat> sigmoid_vec(grad_vec);
    grad_vec.Scale(-1.0);
    grad_vec.Add(1.0);
    grad_vec.MulElements(sigmoid_vec);
    grad_vec.Scale(lhuc_const_);

    
    CuMatrix<BaseFloat> scaled_input(input);
    scaled_input.MulColsVec(grad_vec);

    CuMatrix<BaseFloat> gradient_aux(diff);
    gradient_aux.MulElements(scaled_input);
    lhuc_vec_grad_.AddRowSumMat(1.0, gradient_aux, 0.0);
    const BaseFloat lr = opts_.learn_rate;
    lhuc_vec_.AddVec(-lr*learn_rate_coef_, lhuc_vec_grad_);
    SetScaleVec(NULL);
  }
  void GetLhucVec(VectorBase<BaseFloat> *vec) {
    lhuc_vec_.CopyToVec(vec);
  }
 private:
  float lhuc_const_;
  CuVector<BaseFloat> lhuc_vec_;
  CuVector<BaseFloat> lhuc_scale_vec_; 
  CuVector<BaseFloat> lhuc_vec_grad_;
  BaseFloat learn_rate_coef_;    
};

} // namespace nnet1
} // namespace kaldi
#endif
