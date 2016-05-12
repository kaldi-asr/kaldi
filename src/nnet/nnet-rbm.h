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

#include <string>

#include "nnet/nnet-component.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-utils.h"
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

  RbmBase(int32 dim_in, int32 dim_out):
    Component(dim_in, dim_out)
  { }

  // Inherited from Component::
  // void Propagate(...)
  // virtual void PropagateFnc(...) = 0

  virtual void Reconstruct(
    const CuMatrixBase<BaseFloat> &hid_state,
    CuMatrix<BaseFloat> *vis_probs
  ) = 0;
  virtual void RbmUpdate(
    const CuMatrixBase<BaseFloat> &pos_vis,
    const CuMatrixBase<BaseFloat> &pos_hid,
    const CuMatrixBase<BaseFloat> &neg_vis,
    const CuMatrixBase<BaseFloat> &neg_hid
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

 private:
  //// Make inherited methods inaccessible,
  //   as for RBMs we use Reconstruct(.)
  void Backpropagate(const CuMatrixBase<BaseFloat> &in,
                     const CuMatrixBase<BaseFloat> &out,
                     const CuMatrixBase<BaseFloat> &out_diff,
                     CuMatrix<BaseFloat> *in_diff)
  { }
  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff)
  { }
  ////
};



class Rbm : public RbmBase {
 public:
  Rbm(int32 dim_in, int32 dim_out):
    RbmBase(dim_in, dim_out)
  { }

  ~Rbm()
  { }

  Component* Copy() const {
    return new Rbm(*this);
  }

  ComponentType GetType() const {
    return kRbm;
  }

  void InitData(std::istream &is) {
    // define options,
    std::string vis_type;
    std::string hid_type;
    float vis_bias_mean = 0.0, vis_bias_range = 0.0,
          hid_bias_mean = 0.0, hid_bias_range = 0.0,
          param_stddev = 0.1;
    std::string vis_bias_cmvn_file;  // initialize biases to logit(p_active)
    // parse config,
    std::string token;
    while (is >> std::ws, !is.eof()) {
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
    }

    // Translate the 'node' types,
    if (vis_type == "bern" || vis_type == "Bernoulli") vis_type_ = RbmBase::Bernoulli;
    else if (vis_type == "gauss" || vis_type == "Gaussian") vis_type_ = RbmBase::Gaussian;
    else KALDI_ERR << "Wrong <VisibleType>" << vis_type;
    //
    if (hid_type == "bern" || hid_type == "Bernoulli") hid_type_ = RbmBase::Bernoulli;
    else if (hid_type == "gauss" || hid_type == "Gaussian") hid_type_ = RbmBase::Gaussian;
    else KALDI_ERR << "Wrong <HiddenType>" << hid_type;

    //
    // Initialize trainable parameters,
    //
    // visible-hidden connections,
    vis_hid_.Resize(OutputDim(), InputDim());
    RandGauss(0.0, param_stddev, &vis_hid_);
    // hidden-bias,
    hid_bias_.Resize(OutputDim());
    RandUniform(hid_bias_mean, hid_bias_range, &hid_bias_);
    // visible-bias,
    if (vis_bias_cmvn_file == "") {
      vis_bias_.Resize(InputDim());
      RandUniform(vis_bias_mean, vis_bias_range, &vis_bias_);
    } else {
      KALDI_LOG << "Initializing from <VisibleBiasCmvnFilename> "
                << vis_bias_cmvn_file;
      // Reading Nnet with 'global-cmvn' components,
      Nnet cmvn;
      cmvn.Read(vis_bias_cmvn_file);
      KALDI_ASSERT(InputDim() == cmvn.InputDim());
      // The parameters from <AddShift> correspond to 'negative' mean values,
      Vector<BaseFloat> p(cmvn.InputDim());
      dynamic_cast<AddShift&>(cmvn.GetComponent(0)).GetParams(&p);
      p.Scale(-1.0);  // 'un-do' negation of mean values,
      p.ApplyFloor(0.0001);
      p.ApplyCeiling(0.9999);
      // Getting the logit,
      Vector<BaseFloat> logit_p(p.Dim());
      for (int32 d = 0; d < p.Dim(); d++) {
        logit_p(d) = Log(p(d)) - Log(1.0 - p(d));
      }
      vis_bias_ = logit_p;
      KALDI_ASSERT(vis_bias_.Dim() == InputDim());
    }
  }


  void ReadData(std::istream &is, bool binary) {
    std::string vis_node_type, hid_node_type;
    ReadToken(is, binary, &vis_node_type);
    ReadToken(is, binary, &hid_node_type);

    if (vis_node_type == "bern") {
      vis_type_ = RbmBase::Bernoulli;
    } else if (vis_node_type == "gauss") {
      vis_type_ = RbmBase::Gaussian;
    }
    if (hid_node_type == "bern") {
      hid_type_ = RbmBase::Bernoulli;
    } else if (hid_node_type == "gauss") {
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
      case Bernoulli : WriteToken(os,binary, "bern"); break;
      case Gaussian  : WriteToken(os,binary, "gauss"); break;
      default : KALDI_ERR << "Unknown type " << vis_type_;
    }
    switch (hid_type_) {
      case Bernoulli : WriteToken(os,binary, "bern"); break;
      case Gaussian  : WriteToken(os,binary, "gauss"); break;
      default : KALDI_ERR << "Unknown type " << hid_type_;
    }
    vis_hid_.Write(os, binary);
    vis_bias_.Write(os, binary);
    hid_bias_.Write(os, binary);
  }


  // Component API
  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // pre-fill with bias
    out->AddVecToRows(1.0, hid_bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, vis_hid_, kTrans, 1.0);
    // optionally apply sigmoid
    if (hid_type_ == RbmBase::Bernoulli) {
      out->Sigmoid(*out);
    }
  }

  // RBM training API
  void Reconstruct(const CuMatrixBase<BaseFloat> &hid_state,
                   CuMatrix<BaseFloat> *vis_probs) {
    // check the dim
    if (output_dim_ != hid_state.NumCols()) {
      KALDI_ERR << "Nonmatching dims, component:" << output_dim_
                << " data:" << hid_state.NumCols();
    }
    // optionally allocate buffer
    if (input_dim_ != vis_probs->NumCols() ||
        hid_state.NumRows() != vis_probs->NumRows()) {
      vis_probs->Resize(hid_state.NumRows(), input_dim_);
    }

    // pre-fill with bias
    vis_probs->AddVecToRows(1.0, vis_bias_, 0.0);
    // multiply by weights
    vis_probs->AddMatMat(1.0, hid_state, kNoTrans, vis_hid_, kNoTrans, 1.0);
    // optionally apply sigmoid
    if (vis_type_ == RbmBase::Bernoulli) {
      vis_probs->Sigmoid(*vis_probs);
    }
  }

  void RbmUpdate(const CuMatrixBase<BaseFloat> &pos_vis,
                 const CuMatrixBase<BaseFloat> &pos_hid,
                 const CuMatrixBase<BaseFloat> &neg_vis,
                 const CuMatrixBase<BaseFloat> &neg_hid) {
    // dims
    KALDI_ASSERT(pos_vis.NumRows() == pos_hid.NumRows() &&
           pos_vis.NumRows() == neg_vis.NumRows() &&
           pos_vis.NumRows() == neg_hid.NumRows() &&
           pos_vis.NumCols() == neg_vis.NumCols() &&
           pos_hid.NumCols() == neg_hid.NumCols() &&
           pos_vis.NumCols() == input_dim_ &&
           pos_hid.NumCols() == output_dim_);

    // lazy initialization of buffers
    if ( vis_hid_corr_.NumRows() != vis_hid_.NumRows() ||
         vis_hid_corr_.NumCols() != vis_hid_.NumCols() ||
         vis_bias_corr_.Dim()    != vis_bias_.Dim()    ||
         hid_bias_corr_.Dim()    != hid_bias_.Dim()     ) {
      vis_hid_corr_.Resize(vis_hid_.NumRows(), vis_hid_.NumCols(), kSetZero);
      vis_bias_corr_.Resize(vis_bias_.Dim(), kSetZero);
      hid_bias_corr_.Resize(hid_bias_.Dim(), kSetZero);
    }

    // ANTI-WEIGHT-EXPLOSION PROTECTION (Gaussian-Bernoulli RBM)
    //
    // in the following section we detect that the weights in
    // Gaussian-Bernoulli RBM are almost exploding. The weight
    // explosion is caused by large variance of the reconstructed data,
    // which causes a feed-back loop that keeps increasing the weights.
    //
    // To avoid explosion, the standard-deviation of the visible-data
    // and reconstructed-data should be about the same.
    // The model is particularly sensitive at the very
    // beginning of the CD-1 training.
    //
    // We compute the standard deviations on
    // * 'A' : input mini-batch
    // * 'B' : reconstruction.
    // When 'B > 2*A', we stabilize the training in this way:
    // 1. we scale down the weights and biases by 'A/B',
    // 2. we shrink learning rate by 0.9x,
    // 3. we reset the momentum buffer,
    //
    // A warning message is put to the log. In later stage
    // the learning-rate returns back to its original value.
    //
    // To avoid the issue, we make sure that the weight-matrix
    // is sensibly initialized.
    //
    if (vis_type_ == RbmBase::Gaussian) {
      // check the data have no nan/inf:
      CheckNanInf(pos_vis, "pos_vis");
      CheckNanInf(pos_hid, "pos_hid");
      CheckNanInf(neg_vis, "neg_vis");
      CheckNanInf(neg_hid, "pos_hid");

      // get standard deviations of pos_vis and neg_vis:
      BaseFloat pos_vis_std = ComputeStdDev(pos_vis);
      BaseFloat neg_vis_std = ComputeStdDev(neg_vis);

      // monitor the standard deviation mismatch : data vs. reconstruction
      if (pos_vis_std * 2 < neg_vis_std) {
        // 1) scale-down the weights and biases
        BaseFloat scale = pos_vis_std / neg_vis_std;
        vis_hid_.Scale(scale);
        vis_bias_.Scale(scale);
        hid_bias_.Scale(scale);
        // 2) reduce the learning rate
        rbm_opts_.learn_rate *= 0.9;
        // 3) reset the momentum buffers
        vis_hid_corr_.SetZero();
        vis_bias_corr_.SetZero();
        hid_bias_corr_.SetZero();

        KALDI_WARN << "Mismatch between pos_vis and neg_vis variances, "
                   << "danger of weight explosion."
                   << " a) Reducing weights with scale " << scale
                   << " b) Lowering learning rate to " << rbm_opts_.learn_rate
                   << " [pos_vis_std:" << pos_vis_std
                   << ",neg_vis_std:" << neg_vis_std << "]";
        return; /* i.e. don't update now, the update would be too BIG */
      }
    }
    //
    // End of weight-explosion check


    //  We use these training hyper-parameters
    //
    const BaseFloat lr = rbm_opts_.learn_rate;
    const BaseFloat mmt = rbm_opts_.momentum;
    const BaseFloat l2 = rbm_opts_.l2_penalty;

    //  UPDATE vishid matrix
    //
    //  vishidinc = momentum*vishidinc + ...
    //              epsilonw*( (posprods-negprods)/numcases - weightcost*vishid)
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
    //  visbiasinc = momentum*visbiasinc +
    //               (epsilonvb/numcases)*(posvisact-negvisact);
    //
    vis_bias_corr_.AddRowSumMat(-lr/N, neg_vis, mmt);
    vis_bias_corr_.AddRowSumMat(+lr/N, pos_vis, 1.0);
    vis_bias_.AddVec(1.0, vis_bias_corr_, 1.0);

    //  UPDATE hidbias vector
    //
    // hidbiasinc = momentum*hidbiasinc +
    //              (epsilonhb/numcases)*(poshidact-neghidact);
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
    // header,
    WriteToken(os, binary, Component::TypeToMarker(Component::kAffineTransform));
    WriteBasicType(os, binary, OutputDim());
    WriteBasicType(os, binary, InputDim());
    if (!binary) os << "\n";
    // data,
    vis_hid_.Write(os, binary);
    hid_bias_.Write(os, binary);
    // sigmoid activation,
    if (HidType() == Bernoulli) {
      WriteToken(os, binary, Component::TypeToMarker(Component::kSigmoid));
      WriteBasicType(os, binary, OutputDim());
      WriteBasicType(os, binary, OutputDim());
    }
    if (!binary) os << "\n";
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



}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_RBM_H_
