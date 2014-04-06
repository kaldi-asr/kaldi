// nnet/nnet-rbm.h

// Copyright 2012-2013  Brno University of Technology (Author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_RBM_H_
#define KALDI_NNET_NNET_RBM_H_


#include "nnet/nnet-component.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-various.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

class RbmBase : public Component {
 public:
  typedef enum {
    Bernoulli,
    Gaussian
  } RbmNodeType;
 
  RbmBase(int32 dim_in, int32 dim_out) 
   : Component(dim_in, dim_out)
  { }
  
  /*Is included in Component:: itf
  virtual void Propagate(
    const CuMatrix<BaseFloat> &vis_probs, 
    CuMatrix<BaseFloat> *hid_probs
  ) = 0;
  */

  virtual void Reconstruct(
    const CuMatrix<BaseFloat> &hid_state, 
    CuMatrix<BaseFloat> *vis_probs
  ) = 0;
  virtual void RbmUpdate(
    const CuMatrix<BaseFloat> &pos_vis, 
    const CuMatrix<BaseFloat> &pos_hid, 
    const CuMatrix<BaseFloat> &neg_vis, 
    const CuMatrix<BaseFloat> &neg_hid
  ) = 0;

  virtual RbmNodeType VisType() const = 0;
  virtual RbmNodeType HidType() const = 0;

  virtual void WriteAsNnet(std::ostream& os, bool binary) const = 0;

  /// Set training hyper-parameters to the network and its UpdatableComponent(s)
  void SetRbmTrainOptions(const RbmTrainOptions& opts) {
    rbm_opts_ = opts;
  }
  /// Get training hyper-parameters from the network
  const RbmTrainOptions& GetRbmTrainOptions() const {
    return rbm_opts_;
  }
  
 protected:
  RbmTrainOptions rbm_opts_;

 //// Make these methods inaccessible for descendants.
 //
 private:
  // For RBMs we use Reconstruct(.)
  void Backpropagate(const CuMatrix<BaseFloat> &in, const CuMatrix<BaseFloat> &out,
                     const CuMatrix<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) { }
  void BackpropagateFnc(const CuMatrix<BaseFloat> &in, const CuMatrix<BaseFloat> &out,
                        const CuMatrix<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) { }
 //
 ////

};



class Rbm : public RbmBase {
 public:
  Rbm(int32 dim_in, int32 dim_out) 
   : RbmBase(dim_in, dim_out)
  { } 
  ~Rbm()
  { }  
  
  Component* Copy() const { return new Rbm(*this); }
  ComponentType GetType() const { return kRbm; }

  void InitData(std::istream &is) {
    // define options
    std::string vis_type;
    std::string hid_type;
    float vis_bias_mean = 0.0, vis_bias_range = 0.0, 
          hid_bias_mean = 0.0, hid_bias_range = 0.0, 
          param_stddev = 0.1;
    std::string vis_bias_cmvn_file; // initialize biases to logit(p_active)
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<VisibleType>") ReadToken(is, false, &vis_type);
      else if (token == "<HiddenType>") ReadToken(is, false, &hid_type);
      else if (token == "<VisibleBiasMean>") ReadBasicType(is, false, &vis_bias_mean);
      else if (token == "<VisibleBiasRange>") ReadBasicType(is, false, &vis_bias_range);
      else if (token == "<HiddenBiasMean>") ReadBasicType(is, false, &hid_bias_mean);
      else if (token == "<HiddenBiasRange>") ReadBasicType(is, false, &hid_bias_range);
      else if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<VisibleBiasCmvnFilename>") ReadToken(is, false, &vis_bias_cmvn_file);
      else KALDI_ERR << "Unknown token " << token << " Typo in config?";
      is >> std::ws; // eat-up whitespace
    }

    //
    // initialize
    //
    if (vis_type == "bern" || vis_type == "Bernoulli") vis_type_ = RbmBase::Bernoulli;
    else if (vis_type == "gauss" || vis_type == "Gaussian") vis_type_ = RbmBase::Gaussian;
    else KALDI_ERR << "Wrong <VisibleType>" << vis_type;
    //
    if (hid_type == "bern" || hid_type == "Bernoulli") hid_type_ = RbmBase::Bernoulli;
    else if (hid_type == "gauss" || hid_type == "Gaussian") hid_type_ = RbmBase::Gaussian;
    else KALDI_ERR << "Wrong <HiddenType>" << hid_type;
    // visible-hidden connections
    Matrix<BaseFloat> mat(output_dim_, input_dim_);
    for (int32 r=0; r<output_dim_; r++) {
      for (int32 c=0; c<input_dim_; c++) {
        mat(r,c) = param_stddev * RandGauss(); // 0-mean Gauss with given std_dev
      }
    }
    vis_hid_ = mat;
    // hidden-bias
    Vector<BaseFloat> vec(output_dim_);
    for (int32 i=0; i<output_dim_; i++) {
      // +/- 1/2*bias_range from bias_mean:
      vec(i) = hid_bias_mean + (RandUniform() - 0.5) * hid_bias_range; 
    }
    hid_bias_ = vec;
    // visible-bias
    if (vis_bias_cmvn_file == "") {
      Vector<BaseFloat> vec2(input_dim_);
      for (int32 i=0; i<input_dim_; i++) {
        // +/- 1/2*bias_range from bias_mean:
        vec2(i) = vis_bias_mean + (RandUniform() - 0.5) * vis_bias_range; 
      }
      vis_bias_ = vec2;
    } else {
      KALDI_LOG << "Initializing from <VisibleBiasCmvnFilename> " << vis_bias_cmvn_file;
      Nnet cmvn;
      cmvn.Read(vis_bias_cmvn_file);
      // getting probablity that neuron fires:
      Vector<BaseFloat> p(dynamic_cast<AddShift&>(cmvn.GetComponent(0)).GetShiftVec());
      p.Scale(-1.0);
      // compute logit:
      Vector<BaseFloat> logit_p(p.Dim());
      for(int32 d = 0; d < p.Dim(); d++) {
        if(p(d) < 0.0001) p(d) = 0.0001;
        if(p(d) > 0.9999) p(d) = 0.9999;
        logit_p(d) = log(p(d)) - log(1.0 - p(d));
      }
      vis_bias_ = logit_p;
      KALDI_ASSERT(vis_bias_.Dim() == InputDim());
    }
    //
  }


  void ReadData(std::istream &is, bool binary) {
    std::string vis_node_type, hid_node_type;
    ReadToken(is, binary, &vis_node_type);
    ReadToken(is, binary, &hid_node_type);
    
    if(vis_node_type == "bern") {
      vis_type_ = RbmBase::Bernoulli;
    } else if(vis_node_type == "gauss") {
      vis_type_ = RbmBase::Gaussian;
    }
    if(hid_node_type == "bern") {
      hid_type_ = RbmBase::Bernoulli;
    } else if(hid_node_type == "gauss") {
      hid_type_ = RbmBase::Gaussian;
    }

    vis_hid_.Read(is, binary);
    vis_bias_.Read(is, binary);
    hid_bias_.Read(is, binary);

    KALDI_ASSERT(vis_hid_.NumRows() == output_dim_);
    KALDI_ASSERT(vis_hid_.NumCols() == input_dim_);
    KALDI_ASSERT(vis_bias_.Dim() == input_dim_);
    KALDI_ASSERT(hid_bias_.Dim() == output_dim_);
  }
  
  void WriteData(std::ostream &os, bool binary) const {
    switch (vis_type_) {
      case Bernoulli : WriteToken(os,binary,"bern"); break;
      case Gaussian  : WriteToken(os,binary,"gauss"); break;
      default : KALDI_ERR << "Unknown type " << vis_type_;
    }
    switch (hid_type_) {
      case Bernoulli : WriteToken(os,binary,"bern"); break;
      case Gaussian  : WriteToken(os,binary,"gauss"); break;
      default : KALDI_ERR << "Unknown type " << hid_type_;
    }
    vis_hid_.Write(os, binary);
    vis_bias_.Write(os, binary);
    hid_bias_.Write(os, binary);
  }


  // Component API
  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // precopy bias
    out->AddVecToRows(1.0, hid_bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, vis_hid_, kTrans, 1.0);
    // optionally apply sigmoid
    if (hid_type_ == RbmBase::Bernoulli) {
      out->Sigmoid(*out);
    }
  }

  // RBM training API
  void Reconstruct(const CuMatrix<BaseFloat> &hid_state, CuMatrix<BaseFloat> *vis_probs) {
    // check the dim
    if (output_dim_ != hid_state.NumCols()) {
      KALDI_ERR << "Nonmatching dims, component:" << output_dim_ << " data:" << hid_state.NumCols();
    }
    // optionally allocate buffer
    if (input_dim_ != vis_probs->NumCols() || hid_state.NumRows() != vis_probs->NumRows()) {
      vis_probs->Resize(hid_state.NumRows(), input_dim_);
    }

    // precopy bias
    vis_probs->AddVecToRows(1.0, vis_bias_, 0.0);
    // multiply by weights
    vis_probs->AddMatMat(1.0, hid_state, kNoTrans, vis_hid_, kNoTrans, 1.0);
    // optionally apply sigmoid
    if (vis_type_ == RbmBase::Bernoulli) {
      vis_probs->Sigmoid(*vis_probs);
    }
  }
  
  void RbmUpdate(const CuMatrix<BaseFloat> &pos_vis, const CuMatrix<BaseFloat> &pos_hid, const CuMatrix<BaseFloat> &neg_vis, const CuMatrix<BaseFloat> &neg_hid) {

    KALDI_ASSERT(pos_vis.NumRows() == pos_hid.NumRows() &&
           pos_vis.NumRows() == neg_vis.NumRows() &&
           pos_vis.NumRows() == neg_hid.NumRows() &&
           pos_vis.NumCols() == neg_vis.NumCols() &&
           pos_hid.NumCols() == neg_hid.NumCols() &&
           pos_vis.NumCols() == input_dim_ &&
           pos_hid.NumCols() == output_dim_);

    //lazy initialization of buffers
    if ( vis_hid_corr_.NumRows() != vis_hid_.NumRows() ||
         vis_hid_corr_.NumCols() != vis_hid_.NumCols() ||
         vis_bias_corr_.Dim()    != vis_bias_.Dim()    ||
         hid_bias_corr_.Dim()    != hid_bias_.Dim()     ){
      vis_hid_corr_.Resize(vis_hid_.NumRows(),vis_hid_.NumCols(),kSetZero);
      //vis_bias_corr_.Resize(vis_bias_.Dim(),kSetZero);
      //hid_bias_corr_.Resize(hid_bias_.Dim(),kSetZero);
      vis_bias_corr_.Resize(vis_bias_.Dim());
      hid_bias_corr_.Resize(hid_bias_.Dim());
    }

    //
    // ANTI-WEIGHT-EXPLOSION PROTECTION
    // in the following section we detect that the weights in Gaussian-Bernoulli RBM
    // are about to explode. The weight explosion is caused by large variance of the
    // reconstructed data, which causes increase of weight variance towards the explosion.
    //
    // To avoid explosion, the variance of the visible-data and reconstructed-data
    // should be about the same. The model is particularly sensitive at the very
    // beginning of the CD-1 training.
    //
    // We compute variance of a)input mini-batch b)reconstruction. 
    // When the ratio b)/a) is larger than 2, we:
    // 1. scale down the weights and biases by b)/a) (for next mini-batch b)/a) gets 1.0)
    // 2. shrink learning rate by 0.9x
    // 3. reset the momentum buffer  
    //
    // Wa also display a warning. Note that in later stage 
    // the training returns back to higher learning rate.
    // 
    if (vis_type_ == RbmBase::Gaussian) {
      //get the standard deviations of pos_vis and neg_vis data

      //pos_vis
      CuMatrix<BaseFloat> pos_vis_pow2(pos_vis);
      pos_vis_pow2.MulElements(pos_vis);
      CuVector<BaseFloat> pos_vis_second(pos_vis.NumCols());
      pos_vis_second.AddRowSumMat(1.0,pos_vis_pow2,0.0);
      CuVector<BaseFloat> pos_vis_mean(pos_vis.NumCols());
      pos_vis_mean.AddRowSumMat(1.0/pos_vis.NumRows(),pos_vis,0.0);

      Vector<BaseFloat> pos_vis_second_h(pos_vis_second.Dim());
      pos_vis_second.CopyToVec(&pos_vis_second_h);
      Vector<BaseFloat> pos_vis_mean_h(pos_vis_mean.Dim());
      pos_vis_mean.CopyToVec(&pos_vis_mean_h);
      
      Vector<BaseFloat> pos_vis_stddev(pos_vis_mean_h);
      pos_vis_stddev.MulElements(pos_vis_mean_h);
      pos_vis_stddev.Scale(-1.0);
      pos_vis_stddev.AddVec(1.0/pos_vis.NumRows(),pos_vis_second_h);
      /* set negative values to zero before the square root */
      for (int32 i=0; i<pos_vis_stddev.Dim(); i++) {
        if(pos_vis_stddev(i) < 0.0) { 
          KALDI_WARN << "Forcing the variance to be non-negative! (set to zero)" 
                     << pos_vis_stddev(i);
          pos_vis_stddev(i) = 0.0;
        }
      }
      pos_vis_stddev.ApplyPow(0.5);

      //neg_vis
      CuMatrix<BaseFloat> neg_vis_pow2(neg_vis);
      neg_vis_pow2.MulElements(neg_vis);
      CuVector<BaseFloat> neg_vis_second(neg_vis.NumCols());
      neg_vis_second.AddRowSumMat(1.0,neg_vis_pow2,0.0);
      CuVector<BaseFloat> neg_vis_mean(neg_vis.NumCols());
      neg_vis_mean.AddRowSumMat(1.0/neg_vis.NumRows(),neg_vis,0.0);

      Vector<BaseFloat> neg_vis_second_h(neg_vis_second.Dim());
      neg_vis_second.CopyToVec(&neg_vis_second_h);
      Vector<BaseFloat> neg_vis_mean_h(neg_vis_mean.Dim());
      neg_vis_mean.CopyToVec(&neg_vis_mean_h);
      
      Vector<BaseFloat> neg_vis_stddev(neg_vis_mean_h);
      neg_vis_stddev.MulElements(neg_vis_mean_h);
      neg_vis_stddev.Scale(-1.0);
      neg_vis_stddev.AddVec(1.0/neg_vis.NumRows(),neg_vis_second_h);
      /* set negative values to zero before the square root */
      for (int32 i=0; i<neg_vis_stddev.Dim(); i++) {
        if(neg_vis_stddev(i) < 0.0) { 
          KALDI_WARN << "Forcing the variance to be non-negative! (set to zero)" 
                     << neg_vis_stddev(i);
          neg_vis_stddev(i) = 0.0;
        }
      }
      neg_vis_stddev.ApplyPow(0.5);

      //monitor the standard deviation discrepancy between pos_vis and neg_vis
      if (pos_vis_stddev.Sum() * 2 < neg_vis_stddev.Sum()) {
        //1) scale-down the weights and biases
        BaseFloat scale = pos_vis_stddev.Sum() / neg_vis_stddev.Sum();
        vis_hid_.Scale(scale);
        vis_bias_.Scale(scale);
        hid_bias_.Scale(scale);
        //2) reduce the learning rate           
        rbm_opts_.learn_rate *= 0.9;
        //3) reset the momentum buffers
        vis_hid_corr_.SetZero();
        vis_bias_corr_.SetZero();
        hid_bias_corr_.SetZero();

        KALDI_WARN << "Discrepancy between pos_hid and neg_hid variances, "
                   << "danger of weight explosion. a) Reducing weights with scale " << scale
                   << " b) Lowering learning rate to " << rbm_opts_.learn_rate
                   << " [pos_vis_stddev(~1.0):" << pos_vis_stddev.Sum()/pos_vis.NumCols()
                   << ",neg_vis_stddev:" << neg_vis_stddev.Sum()/neg_vis.NumCols() << "]";
        return; /* i.e. don't update weights with current stats */
      }
    }
    //
    // End of Gaussian-Bernoulli weight-explosion check


    //  We use these training hyper-parameters
    //
    const BaseFloat lr = rbm_opts_.learn_rate;
    const BaseFloat mmt = rbm_opts_.momentum;
    const BaseFloat l2 = rbm_opts_.l2_penalty;
    
    //  UPDATE vishid matrix
    //  
    //  vishidinc = momentum*vishidinc + ...
    //              epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    //
    //  vishidinc[t] = -(epsilonw/numcases)*negprods + momentum*vishidinc[t-1]
    //                 +(epsilonw/numcases)*posprods
    //                 -(epsilonw*weightcost)*vishid[t-1]
    //
    BaseFloat N = static_cast<BaseFloat>(pos_vis.NumRows());
    vis_hid_corr_.AddMatMat(-lr/N, neg_hid, kTrans, neg_vis, kNoTrans, mmt);
    vis_hid_corr_.AddMatMat(+lr/N, pos_hid, kTrans, pos_vis, kNoTrans, 1.0);
    vis_hid_corr_.AddMat(-lr*l2, vis_hid_);
    vis_hid_.AddMat(1.0, vis_hid_corr_);

    //  UPDATE visbias vector
    //
    //  visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    //
    vis_bias_corr_.AddRowSumMat(-lr/N, neg_vis, mmt);
    vis_bias_corr_.AddRowSumMat(+lr/N, pos_vis, 1.0);
    vis_bias_.AddVec(1.0, vis_bias_corr_, 1.0);
    
    //  UPDATE hidbias vector
    //
    // hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
    //
    hid_bias_corr_.AddRowSumMat(-lr/N, neg_hid, mmt);
    hid_bias_corr_.AddRowSumMat(+lr/N, pos_hid, 1.0);
    hid_bias_.AddVec(1.0, hid_bias_corr_, 1.0);
  }



  RbmNodeType VisType() const { 
    return vis_type_; 
  }

  RbmNodeType HidType() const { 
    return hid_type_; 
  }

  void WriteAsNnet(std::ostream& os, bool binary) const {
    //header
    WriteToken(os,binary,Component::TypeToMarker(Component::kAffineTransform));
    WriteBasicType(os,binary,OutputDim());
    WriteBasicType(os,binary,InputDim());
    if(!binary) os << "\n";
    //data
    vis_hid_.Write(os,binary);
    hid_bias_.Write(os,binary);
    //optionally sigmoid activation
    if(HidType() == Bernoulli) {
      WriteToken(os,binary,Component::TypeToMarker(Component::kSigmoid));
      WriteBasicType(os,binary,OutputDim());
      WriteBasicType(os,binary,OutputDim());
    }
    if(!binary) os << "\n";
  }

protected:
  CuMatrix<BaseFloat> vis_hid_;        ///< Matrix with neuron weights
  CuVector<BaseFloat> vis_bias_;       ///< Vector with biases
  CuVector<BaseFloat> hid_bias_;       ///< Vector with biases

  CuMatrix<BaseFloat> vis_hid_corr_;   ///< Matrix for linearity updates
  CuVector<BaseFloat> vis_bias_corr_;  ///< Vector for bias updates
  CuVector<BaseFloat> hid_bias_corr_;  ///< Vector for bias updates

  RbmNodeType vis_type_;
  RbmNodeType hid_type_;

};



} // namespace nnet1
} // namespace kaldi

#endif
