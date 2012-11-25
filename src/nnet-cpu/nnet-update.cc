// nnet/nnet-update.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet-cpu/nnet-update.h"

namespace kaldi {


// This class NnetUpdater contains functions for updating the neural net or
// computing its gradient, given a set of NnetTrainingExamples.  Its
// functionality is exported by DoBackprop(), and by ComputeNnetObjf(), so we
// define it in the .cc file.
class NnetUpdater {
 public:
  // Note: in the case of training with SGD, "nnet" and "nnet_to_update" will
  // be identical.  They'll be different if we're accumulating the gradient
  // for a held-out set and don't want to update the model.  Note: nnet_to_update
  // may be NULL if you don't want do do backprop.
  NnetUpdater(const Nnet &nnet,
              Nnet *nnet_to_update);
  
  BaseFloat ComputeForMinibatch(const std::vector<NnetTrainingExample> &data);
  // returns average objective function over this minibatch.
  
 private:

  /// takes the input and formats as a single matrix, in forward_data_[0].
  void FormatInput(const std::vector<NnetTrainingExample> &data);
  
  // Possibly splices input together from forward_data_[component].
  //   MatrixBase<BaseFloat> &GetSplicedInput(int32 component, Matrix<BaseFloat> *temp_matrix);


  void Propagate();

  /// Computes objective function and derivative at output layer.
  BaseFloat ComputeObjfAndDeriv(const std::vector<NnetTrainingExample> &data,
                                Matrix<BaseFloat> *deriv) const;
  
  /// Returns objf summed (and weighted) over samples.
  /// Note: "deriv" will contain, at input, the derivative w.r.t. the
  /// output layer but will be used as a temporary variable by
  /// this function.
  void Backprop(const std::vector<NnetTrainingExample> &data,
                Matrix<BaseFloat> *deriv);

  const Nnet &nnet_;
  Nnet *nnet_to_update_;
  int32 num_chunks_; // same as the minibatch size.
  
  std::vector<Matrix<BaseFloat> > forward_data_; // The forward data
  // for the outputs of each of the components.

  // These weights are one per parameter; they equal to the "weight"
  // member variables in the NnetTrainingExample structures.  These
  // will typically be about one on average.
  Vector<BaseFloat> chunk_weights_;
};

NnetUpdater::NnetUpdater(const Nnet &nnet,
                         Nnet *nnet_to_update):
    nnet_(nnet), nnet_to_update_(nnet_to_update) {
}
 

BaseFloat NnetUpdater::ComputeForMinibatch(
    const std::vector<NnetTrainingExample> &data) {
  FormatInput(data);
  Propagate();
  Matrix<BaseFloat> tmp_deriv;
  BaseFloat ans = ComputeObjfAndDeriv(data, &tmp_deriv);
  if (nnet_to_update_ != NULL)
    Backprop(data, &tmp_deriv); // this is summed (after weighting), not
                                // averaged.
  return ans;
}

void NnetUpdater::Propagate() {
  int32 num_components = nnet_.NumComponents();
  for (int32 c = 0; c < num_components; c++) {
    const Component &component = nnet_.GetComponent(c);
    const Matrix<BaseFloat> &input = forward_data_[c];
    Matrix<BaseFloat> &output = forward_data_[c+1];
    // Note: the Propagate function will automatically resize the
    // output.
    component.Propagate(input, 1, &output);
    // If we won't need the output of the previous layer for
    // backprop, delete it to save memory.
    bool need_last_output =
        (c>0 && nnet_.GetComponent(c-1).BackpropNeedsOutput()) ||
        component.BackpropNeedsInput();
    if (!need_last_output)
      forward_data_[c].Resize(0, 0); // We won't need this data.
  }
}

BaseFloat NnetUpdater::ComputeObjfAndDeriv(
    const std::vector<NnetTrainingExample> &data,
    Matrix<BaseFloat> *deriv) const {
  const BaseFloat floor = 1.0e-20; // Avoids division by zero.
  double tot_objf = 0.0, tot_weight = 0.0;
  int32 num_components = nnet_.NumComponents();  
  deriv->Resize(num_chunks_, nnet_.OutputDim()); // sets to zero.
  const Matrix<BaseFloat> &output(forward_data_[num_components]);
  KALDI_ASSERT(SameDim(output, *deriv));
  for (int32 m = 0; m < num_chunks_; m++) {
    int32 label = data[m].label;
    BaseFloat weight = data[m].weight;
    KALDI_ASSERT(label >= 0 && label < nnet_.OutputDim());
    BaseFloat this_prob = output(m, label);
    if (this_prob < floor) {
      KALDI_WARN << "Probability is " << this_prob << ", flooring to "
                 << floor;
      this_prob = floor;
    }
    tot_objf += weight * log(this_prob);
    tot_weight += weight;
    (*deriv)(m, label) = weight / this_prob;
    
  }
  KALDI_VLOG(4) << "Objective function is " << (tot_objf/tot_weight) << " over "
                << tot_weight << " samples (weighted).";
  return tot_objf;
}


void NnetUpdater::Backprop(const std::vector<NnetTrainingExample> &data,
                           Matrix<BaseFloat> *deriv) {
  Vector<BaseFloat> sample_weights(data.size());
  for (size_t i = 0; i < data.size(); i++)
    sample_weights(i) = data[i].weight;
  // We assume ComputeObjfAndDeriv has already been called.
  for (int32 c = nnet_.NumComponents() - 1; c >= 0; c--) {
    const Component &component = nnet_.GetComponent(c);
    Component *component_to_update = (nnet_to_update_ == NULL ? NULL :
                                      &(nnet_to_update_->GetComponent(c)));
    Matrix<BaseFloat> &input = forward_data_[c],
                     &output = forward_data_[c+1];
    Matrix<BaseFloat> input_deriv(input.NumRows(), input.NumCols());
    const Matrix<BaseFloat> &output_deriv(*deriv);
 
    component.Backprop(input, output, output_deriv, sample_weights,
                       component_to_update, &input_deriv);
    *deriv = input_deriv;
  }
}


void NnetUpdater::FormatInput(const std::vector<NnetTrainingExample> &data) {
  KALDI_ASSERT(data.size() > 0);
  int32 num_splice = nnet_.LeftContext() + 1 + nnet_.RightContext();
  KALDI_ASSERT(data[0].input_frames.NumRows() == num_splice);

  int32 feat_dim = data[0].input_frames.NumCols(),
         spk_dim = data[0].spk_info.Dim(),
         tot_dim = feat_dim + spk_dim; // we append these at the neural net
                                       // input... note, spk_dim might be 0.
  KALDI_ASSERT(tot_dim == nnet_.InputDim());
  num_chunks_ = data.size();

  forward_data_.resize(nnet_.NumComponents() + 1);
  forward_data_[0].Resize(num_splice * num_chunks_,
                          tot_dim);
  for (int32 chunk = 0; chunk < num_chunks_; chunk++) {
    SubMatrix<BaseFloat> dest(forward_data_[0],
                              chunk * num_splice, num_splice,
                              0, feat_dim);
    const Matrix<BaseFloat> &src(data[chunk].input_frames);
    dest.CopyFromMat(src);
    if (spk_dim != 0) {
      SubMatrix<BaseFloat> spk_dest(forward_data_[0],
                                    chunk * num_splice, num_splice,
                                    feat_dim, spk_dim);
      spk_dest.CopyRowsFromVec(data[chunk].spk_info);
    }
  }
}

BaseFloat TotalNnetTrainingWeight(const std::vector<NnetTrainingExample> &egs) {
  double ans = 0.0;
  for (size_t i = 0; i < egs.size(); i++)
    ans += egs[i].weight;
  return ans;
}

BaseFloat DoBackprop(const Nnet &nnet,
                     const std::vector<NnetTrainingExample> &examples,
                     Nnet *nnet_to_update) {
  KALDI_ASSERT(nnet_to_update != NULL && "Call ComputeNnetObjf() instead.");
  NnetUpdater updater(nnet, nnet_to_update);
  return updater.ComputeForMinibatch(examples);  
}

BaseFloat ComputeNnetObjf(const Nnet &nnet,
                          const std::vector<NnetTrainingExample> &examples) {
  NnetUpdater updater(nnet, NULL);
  return updater.ComputeForMinibatch(examples);
}


BaseFloat ComputeNnetGradient(
    const Nnet &nnet,
    const std::vector<NnetTrainingExample> &validation_set,
    int32 batch_size,
    Nnet *gradient) {
  bool treat_as_gradient = true;
  gradient->SetZero(treat_as_gradient);
  std::vector<NnetTrainingExample> batch;
  batch.reserve(batch_size);
  BaseFloat tot_objf = 0.0;
  for (int32 start_pos = 0;
       start_pos < static_cast<int32>(validation_set.size());
       start_pos += batch_size) {
    batch.clear();
    for (int32 i = start_pos;
         i < std::min(start_pos + batch_size,
                      static_cast<int32>(validation_set.size()));
         i++) {
      batch.push_back(validation_set[i]);
    }
    tot_objf += DoBackprop(nnet,
                           batch,
                           gradient);
  }
  return tot_objf / TotalNnetTrainingWeight(validation_set);
}


void NnetTrainingExample::Write(std::ostream &os, bool binary) const {
  // Note: weight, label, input_frames and spk_info are members.  This is a
  // struct.
  WriteToken(os, binary, "<NnetTrainingExample>");
  WriteToken(os, binary, "<Weight>");
  WriteBasicType(os, binary, weight);
  WriteToken(os, binary, "<Label>");
  WriteBasicType(os, binary, label);
  WriteToken(os, binary, "<InputFrames>");
  input_frames.Write(os, binary);
  WriteToken(os, binary, "<SpkInfo>");
  spk_info.Write(os, binary);
  WriteToken(os, binary, "</NnetTrainingExample>");
}
void NnetTrainingExample::Read(std::istream &is, bool binary) {
  // Note: weight, label, input_frames and spk_info are members.  This is a
  // struct.
  ExpectToken(is, binary, "<NnetTrainingExample>");
  ExpectToken(is, binary, "<Weight>");
  ReadBasicType(is, binary, &weight);
  ExpectToken(is, binary, "<Label>");
  ReadBasicType(is, binary, &label);
  ExpectToken(is, binary, "<InputFrames>");
  input_frames.Read(is, binary);
  ExpectToken(is, binary, "<SpkInfo>");
  spk_info.Read(is, binary);
  ExpectToken(is, binary, "</NnetTrainingExample>");
}


  
  
} // namespace
