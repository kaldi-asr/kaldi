// nnet/nnet-rbm.h

// Copyright 2012  Karel Vesely

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


#ifndef KALDI_NNET_RBM_H
#define KALDI_NNET_RBM_H


#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {

class RbmBase : public UpdatableComponent {
 public:
  typedef enum {
    BERNOULLI,
    GAUSSIAN
  } RbmNodeType;
 
  RbmBase(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet) 
   : UpdatableComponent(dim_in, dim_out, nnet)
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
};



class Rbm : public RbmBase {
 public:
  Rbm(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet) 
   : RbmBase(dim_in, dim_out, nnet)
  { } 
  ~Rbm()
  { }  
  
  ComponentType GetType() const {
    return kRbm;
  }

  void ReadData(std::istream &is, bool binary) {
    std::string vis_node_type, hid_node_type;
    ReadToken(is, binary, &vis_node_type);
    ReadToken(is, binary, &hid_node_type);
    
    if(vis_node_type == "bern") {
      vis_type_ = RbmBase::BERNOULLI;
    } else if(vis_node_type == "gauss") {
      vis_type_ = RbmBase::GAUSSIAN;
    }
    if(hid_node_type == "bern") {
      hid_type_ = RbmBase::BERNOULLI;
    } else if(hid_node_type == "gauss") {
      hid_type_ = RbmBase::GAUSSIAN;
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
      case BERNOULLI : WriteToken(os,binary,"bern"); break;
      case GAUSSIAN  : WriteToken(os,binary,"gauss"); break;
      default : KALDI_ERR << "Unknown type " << vis_type_;
    }
    switch (hid_type_) {
      case BERNOULLI : WriteToken(os,binary,"bern"); break;
      case GAUSSIAN  : WriteToken(os,binary,"gauss"); break;
      default : KALDI_ERR << "Unknown type " << hid_type_;
    }
    vis_hid_.Write(os, binary);
    vis_bias_.Write(os, binary);
    hid_bias_.Write(os, binary);
  }


  // UpdatableComponent API
  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // precopy bias
    out->AddScaledRow(1.0, hid_bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, vis_hid_, kTrans, 1.0);
    // optionally apply sigmoid
    if (hid_type_ == RbmBase::BERNOULLI) {
      cu::Sigmoid(*out, out);
    }
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    KALDI_ERR << "Cannot backpropagate through RBM!"
              << "Better convert it to <BiasedLinearity>";
  }
  virtual void Update(const CuMatrix<BaseFloat> &input,
                      const CuMatrix<BaseFloat> &err) {
    KALDI_ERR << "Cannot update RBM by backprop!"
              << "Better convert it to <BiasedLinearity>";
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
    vis_probs->AddScaledRow(1.0, vis_bias_, 0.0);
    // multiply by weights
    vis_probs->AddMatMat(1.0, hid_state, kNoTrans, vis_hid_, kNoTrans, 1.0);
    // optionally apply sigmoid
    if (vis_type_ == RbmBase::BERNOULLI) {
      cu::Sigmoid(*vis_probs, vis_probs);
    }
  }
  
  void RbmUpdate(const CuMatrix<BaseFloat> &pos_vis, const CuMatrix<BaseFloat> &pos_hid, const CuMatrix<BaseFloat> &neg_vis, const CuMatrix<BaseFloat> &neg_hid) {

    assert(pos_vis.NumRows() == pos_hid.NumRows() &&
           pos_vis.NumRows() == neg_vis.NumRows() &&
           pos_vis.NumRows() == neg_hid.NumRows() &&
           pos_vis.NumCols() == neg_vis.NumCols() &&
           pos_hid.NumCols() == neg_hid.NumCols() &&
           pos_vis.NumCols() == input_dim_ &&
           pos_hid.NumCols() == output_dim_);

    //lazy initialization of buffers (possibly reduces to no-op)
    vis_hid_corr_.Resize(vis_hid_.NumRows(),vis_hid_.NumCols());
    vis_bias_corr_.Resize(vis_bias_.Dim());
    hid_bias_corr_.Resize(hid_bias_.Dim());
    

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

    // vis_hid_corr_.Gemm('T','N',-learn_rate_/N,neg_vis,neg_hid,momentum_);
    vis_hid_corr_.AddMatMat(-learn_rate_/N, neg_vis, kTrans, neg_hid, kNoTrans, momentum_);
    // vis_hid_corr_.Gemm('T','N',+learn_rate_/N,pos_vis,pos_hid,1.0);
    vis_hid_corr_.AddMatMat(+learn_rate_/N, pos_vis, kTrans, pos_hid, kNoTrans, 1.0);
    vis_hid_corr_.AddMat(-learn_rate_*l2_penalty_, vis_hid_, 1.0);
    vis_hid_.AddMat(1.0, vis_hid_corr_, 1.0);

    //  UPDATE visbias vector
    //
    //  visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    //
    vis_bias_corr_.AddColSum(-learn_rate_/N, neg_vis, momentum_);
    vis_bias_corr_.AddColSum(+learn_rate_/N, pos_vis, 1.0);
    vis_bias_.AddVec(1.0, vis_bias_corr_, 1.0);
    
    //  UPDATE hidbias vector
    //
    // hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
    //
    hid_bias_corr_.AddColSum(-learn_rate_/N, neg_hid, momentum_);
    hid_bias_corr_.AddColSum(+learn_rate_/N, pos_hid, 1.0);
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
    WriteToken(os,binary,Component::TypeToMarker(Component::kBiasedLinearity));
    WriteBasicType(os,binary,OutputDim());
    WriteBasicType(os,binary,InputDim());
    if(!binary) os << "\n";
    //data
    vis_hid_.Write(os,binary);
    hid_bias_.Write(os,binary);
    //optionally sigmoid activation
    if(HidType() == BERNOULLI) {
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

  // CuMatrix<BaseFloat> backprop_err_buf_;

  RbmNodeType vis_type_;
  RbmNodeType hid_type_;

};



} // namespace

#endif
