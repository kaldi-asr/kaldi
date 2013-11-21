// nnet2/nnet-compute.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/nnet-compute.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet2 {

/*
  This class does the forward and possibly backward computation for (typically)
  a whole utterance of contiguous features.  You'll instantiate one of
  these classes each time you want to do this computation.
*/
class NnetComputer {
 public:
  /* Initializer.  If pad == true, pad input with nnet.LeftContext() frames on
     the left and nnet.RightContext() frames on the right (duplicate the first
     and last frames.) */
  NnetComputer(const Nnet &nnet,
               const CuMatrixBase<BaseFloat> &input_feats,
               const CuVectorBase<BaseFloat> &spk_info,
               bool pad, 
               Nnet *nnet_to_update = NULL);
  
  /// The forward-through-the-layers part of the computation.
  void Propagate();
  
  void Backprop(CuMatrix<BaseFloat> *tmp_deriv);
                
  
  /// Computes objf derivative at last layer, and returns objective
  /// function summed over labels and multiplied by utterance_weight.
  /// [Note: utterance_weight will normally be 1.0].
  BaseFloat ComputeLastLayerDeriv(const Posterior &pdf_post,
                                  CuMatrix<BaseFloat> *deriv) const;
  
  CuMatrixBase<BaseFloat> &GetOutput() { return forward_data_.back(); }
  
 private:  
  const Nnet &nnet_;
  CuVector<BaseFloat> spk_info_;
  std::vector<CuMatrix<BaseFloat> > forward_data_;
  Nnet *nnet_to_update_; // May be NULL, if just want objective function
  // but no gradient info or SGD.
};

NnetComputer::NnetComputer(const Nnet &nnet,
                           const CuMatrixBase<BaseFloat> &input_feats,
                           const CuVectorBase<BaseFloat> &spk_info,
                           bool pad,
                           Nnet *nnet_to_update):
    nnet_(nnet), spk_info_(spk_info), nnet_to_update_(nnet_to_update) {
  int32 feature_dim = input_feats.NumCols(),
            spk_dim = spk_info.Dim(),
            tot_dim = feature_dim + spk_dim;  
  KALDI_ASSERT(tot_dim == nnet.InputDim());

  forward_data_.resize(nnet.NumComponents() + 1);

  int32 left_context = (pad ? nnet_.LeftContext() : 0),
       right_context = (pad ? nnet_.RightContext() : 0);

  int32 num_rows = left_context + input_feats.NumRows() + right_context;
  CuMatrix<BaseFloat> &input(forward_data_[0]);
  input.Resize(num_rows, tot_dim);
  input.Range(left_context, input_feats.NumRows(),
              0, feature_dim).CopyFromMat(input_feats);
  for (int32 i = 0; i < left_context; i++)
    input.Row(i).Range(0, feature_dim).CopyFromVec(input_feats.Row(0));
  int32 last_row = input_feats.NumRows() - 1;
  for (int32 i = 0; i < right_context; i++)
    input.Row(num_rows - i - 1).
        Range(0, feature_dim).CopyFromVec(input_feats.Row(last_row));
  if (spk_dim != 0)
    input.Range(0, input.NumRows(),
                feature_dim, spk_dim).CopyRowsFromVec(spk_info);
}


/// This is the forward part of the computation.
void NnetComputer::Propagate() {
  for (int32 c = 0; c < nnet_.NumComponents(); c++) {
    const Component &component = nnet_.GetComponent(c);
    CuMatrix<BaseFloat> &input = forward_data_[c],
                     &output = forward_data_[c+1];
        
    component.Propagate(input, 1, &output);
    const Component *prev_component = (c == 0 ? NULL : &(nnet_.GetComponent(c-1)));
    bool will_do_backprop = (nnet_to_update_ != NULL),
         keep_last_output = will_do_backprop &&
                             ((c>0 && prev_component->BackpropNeedsOutput()) ||
                              component.BackpropNeedsInput());
    if (!keep_last_output)
      forward_data_[c].Resize(0, 0); // We won't need this data; save memory.
  }
}

BaseFloat NnetComputer::ComputeLastLayerDeriv(const Posterior &pdf_post,
                                              CuMatrix<BaseFloat> *deriv) const {
  // TODO: convert this to proper CUDA code, c.f. ComputeObjfAndDeriv
  // in nnet-update.cc (I'm not sure, though, that this code is ever reached.)
  int32 num_components = nnet_.NumComponents();
  double tot_objf = 0.0, tot_weight = 0.0;
  const CuMatrix<BaseFloat> &last_layer_output = forward_data_[num_components];
  int32 num_frames = last_layer_output.NumRows(),
          num_pdfs = last_layer_output.NumCols();
  KALDI_ASSERT(pdf_post.size() == static_cast<size_t>(num_frames));
  deriv->Resize(num_frames, num_pdfs); // will zero it.
  for (int32 i = 0; i < deriv->NumRows(); i++) {
    for (size_t j = 0; j < pdf_post[i].size(); j++) {
      int32 label = pdf_post[i][j].first;
      BaseFloat weight = pdf_post[i][j].second;
      KALDI_ASSERT(label >= 0 && label < num_pdfs);
      BaseFloat this_prob = last_layer_output(i, label);
      KALDI_ASSERT(this_prob > 0.99e-20); // We floored to 1.0e-20 in SoftmaxLayer.
      tot_objf += weight * log(this_prob);
      tot_weight += weight;
      (*deriv)(i, label) += weight / this_prob; // could be "=", assuming the
      // labels are all distinct.
    }
  }
  KALDI_VLOG(4) << "Objective function is " << (tot_objf/tot_weight) <<
      " per frame over " << tot_weight << " samples.";
  return tot_objf;  
}


void NnetComputer::Backprop(CuMatrix<BaseFloat> *tmp_deriv) {
  KALDI_ASSERT(nnet_to_update_ != NULL); // Or why do backprop?
  // If later this reasoning changes, we can change this
  // statement and add logic to make component_to_update, below,
  // NULL if necessary.
  int32 num_chunks = 1;
  
  for (int32 c = nnet_.NumComponents() - 1; c >= 0; c--) {
    const Component &component = nnet_.GetComponent(c);
    Component *component_to_update = &(nnet_to_update_->GetComponent(c));
    const CuMatrix<BaseFloat>  &input = forward_data_[c],
                            &output = forward_data_[c+1],
                      &output_deriv = *tmp_deriv;
    CuMatrix<BaseFloat> input_deriv;
    component.Backprop(input, output, output_deriv, num_chunks,
                       component_to_update, &input_deriv);
    *tmp_deriv = input_deriv;
  }
}

void NnetComputation(const Nnet &nnet,
                     const CuMatrixBase<BaseFloat> &input,  // features
                     const CuVectorBase<BaseFloat> &spk_info,
                     bool pad_input,
                     CuMatrixBase<BaseFloat> *output) {
  NnetComputer nnet_computer(nnet, input, spk_info, pad_input, NULL);
  nnet_computer.Propagate();
  output->CopyFromMat(nnet_computer.GetOutput());
}

BaseFloat NnetGradientComputation(const Nnet &nnet,
                                  const CuMatrixBase<BaseFloat> &input,
                                  const CuVectorBase<BaseFloat> &spk_info,
                                  bool pad_input,
                                  const Posterior &pdf_post,
                                  Nnet *nnet_to_update) {
  NnetComputer nnet_computer(nnet, input, spk_info, pad_input, nnet_to_update);
  nnet_computer.Propagate();
  CuMatrix<BaseFloat> deriv;
  BaseFloat ans;
  ans = nnet_computer.ComputeLastLayerDeriv(pdf_post, &deriv);  
  nnet_computer.Backprop(&deriv);
  return ans;
}


} // namespace nnet2
} // namespace kaldi
