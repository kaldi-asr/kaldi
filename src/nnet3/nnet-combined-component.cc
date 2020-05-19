// nnet3/nnet-combined-component.cc

// Copyright 2015-2018  Johns Hopkins University (author: Daniel Povey)
//                2015  Daniel Galvez
//                2018  Hang Lyu

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

#include <iterator>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include "nnet3/nnet-combined-component.h"
#include "nnet3/nnet-parse.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet3 {

// Constructors for the convolution component
ConvolutionComponent::ConvolutionComponent():
    UpdatableComponent(),
    input_x_dim_(0), input_y_dim_(0), input_z_dim_(0),
    filt_x_dim_(0), filt_y_dim_(0),
    filt_x_step_(0), filt_y_step_(0),
    input_vectorization_(kZyx) { }

ConvolutionComponent::ConvolutionComponent(
    const ConvolutionComponent &component):
    UpdatableComponent(component),
    input_x_dim_(component.input_x_dim_),
    input_y_dim_(component.input_y_dim_),
    input_z_dim_(component.input_z_dim_),
    filt_x_dim_(component.filt_x_dim_),
    filt_y_dim_(component.filt_y_dim_),
    filt_x_step_(component.filt_x_step_),
    filt_y_step_(component.filt_y_step_),
    input_vectorization_(component.input_vectorization_),
    filter_params_(component.filter_params_),
    bias_params_(component.bias_params_) { }

ConvolutionComponent::ConvolutionComponent(
    const CuMatrixBase<BaseFloat> &filter_params,
    const CuVectorBase<BaseFloat> &bias_params,
    int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
    int32 filt_x_dim, int32 filt_y_dim,
    int32 filt_x_step, int32 filt_y_step,
    TensorVectorizationType input_vectorization,
    BaseFloat learning_rate):
    input_x_dim_(input_x_dim),
    input_y_dim_(input_y_dim),
    input_z_dim_(input_z_dim),
    filt_x_dim_(filt_x_dim),
    filt_y_dim_(filt_y_dim),
    filt_x_step_(filt_x_step),
    filt_y_step_(filt_y_step),
    input_vectorization_(input_vectorization),
    filter_params_(filter_params),
    bias_params_(bias_params){
  KALDI_ASSERT(filter_params.NumRows() == bias_params.Dim() &&
               bias_params.Dim() != 0);
  KALDI_ASSERT(filter_params.NumCols() == filt_x_dim * filt_y_dim * input_z_dim);
  SetUnderlyingLearningRate(learning_rate);
  is_gradient_ = false;
}

// aquire input dim
int32 ConvolutionComponent::InputDim() const {
  return input_x_dim_ * input_y_dim_ * input_z_dim_;
}

// aquire output dim
int32 ConvolutionComponent::OutputDim() const {
  int32 num_x_steps = (1 + (input_x_dim_ - filt_x_dim_) / filt_x_step_);
  int32 num_y_steps = (1 + (input_y_dim_ - filt_y_dim_) / filt_y_step_);
  int32 num_filters = filter_params_.NumRows();
  return num_x_steps * num_y_steps * num_filters;
}

// initialize the component using hyperparameters
void ConvolutionComponent::Init(
    int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
    int32 filt_x_dim, int32 filt_y_dim,
    int32 filt_x_step, int32 filt_y_step, int32 num_filters,
    TensorVectorizationType input_vectorization,
    BaseFloat param_stddev, BaseFloat bias_stddev) {
  input_x_dim_ = input_x_dim;
  input_y_dim_ = input_y_dim;
  input_z_dim_ = input_z_dim;
  filt_x_dim_ = filt_x_dim;
  filt_y_dim_ = filt_y_dim;
  filt_x_step_ = filt_x_step;
  filt_y_step_ = filt_y_step;
  input_vectorization_ = input_vectorization;
  KALDI_ASSERT((input_x_dim_ - filt_x_dim_) % filt_x_step_ == 0);
  KALDI_ASSERT((input_y_dim_ - filt_y_dim_) % filt_y_step_ == 0);
  int32 filter_dim = filt_x_dim_ * filt_y_dim_ * input_z_dim_;
  filter_params_.Resize(num_filters, filter_dim);
  bias_params_.Resize(num_filters);
  KALDI_ASSERT(param_stddev >= 0.0 && bias_stddev >= 0.0);
  filter_params_.SetRandn();
  filter_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
}

// initialize the component using predefined matrix file
void ConvolutionComponent::Init(
    int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
    int32 filt_x_dim, int32 filt_y_dim,
    int32 filt_x_step, int32 filt_y_step,
    TensorVectorizationType input_vectorization,
    std::string matrix_filename) {
  input_x_dim_ = input_x_dim;
  input_y_dim_ = input_y_dim;
  input_z_dim_ = input_z_dim;
  filt_x_dim_ = filt_x_dim;
  filt_y_dim_ = filt_y_dim;
  filt_x_step_ = filt_x_step;
  filt_y_step_ = filt_y_step;
  input_vectorization_ = input_vectorization;
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat);
  int32 filter_dim = (filt_x_dim_ * filt_y_dim_ * input_z_dim_);
  int32 num_filters = mat.NumRows();
  KALDI_ASSERT(mat.NumCols() == (filter_dim + 1));
  filter_params_.Resize(num_filters, filter_dim);
  bias_params_.Resize(num_filters);
  filter_params_.CopyFromMat(mat.Range(0, num_filters, 0, filter_dim));
  bias_params_.CopyColFromMat(mat, filter_dim);
}

// display information about component
std::string ConvolutionComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", input-x-dim=" << input_x_dim_
         << ", input-y-dim=" << input_y_dim_
         << ", input-z-dim=" << input_z_dim_
         << ", filt-x-dim=" << filt_x_dim_
         << ", filt-y-dim=" << filt_y_dim_
         << ", filt-x-step=" << filt_x_step_
         << ", filt-y-step=" << filt_y_step_
         << ", input-vectorization=" << input_vectorization_
         << ", num-filters=" << filter_params_.NumRows();
  PrintParameterStats(stream, "filter-params", filter_params_);
  PrintParameterStats(stream, "bias-params", bias_params_, true);
  return stream.str();
}

// initialize the component using configuration file
void ConvolutionComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  int32 input_x_dim = -1, input_y_dim = -1, input_z_dim = -1,
        filt_x_dim = -1, filt_y_dim = -1,
        filt_x_step = -1, filt_y_step = -1,
        num_filters = -1;
  std::string input_vectorization_order = "zyx";
  InitLearningRatesFromConfig(cfl);
  ok = ok && cfl->GetValue("input-x-dim", &input_x_dim);
  ok = ok && cfl->GetValue("input-y-dim", &input_y_dim);
  ok = ok && cfl->GetValue("input-z-dim", &input_z_dim);
  ok = ok && cfl->GetValue("filt-x-dim", &filt_x_dim);
  ok = ok && cfl->GetValue("filt-y-dim", &filt_y_dim);
  ok = ok && cfl->GetValue("filt-x-step", &filt_x_step);
  ok = ok && cfl->GetValue("filt-y-step", &filt_y_step);

  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
  // optional argument
  TensorVectorizationType input_vectorization;
  cfl->GetValue("input-vectorization-order", &input_vectorization_order);
  if (input_vectorization_order.compare("zyx") == 0) {
    input_vectorization = kZyx;
  } else if (input_vectorization_order.compare("yzx") == 0) {
    input_vectorization = kYzx;
  } else {
    KALDI_ERR << "Unknown or unsupported input vectorization order "
              << input_vectorization_order
              << " accepted candidates are 'yzx' and 'zyx'";
  }

  if (cfl->GetValue("matrix", &matrix_filename)) {
    // initialize from prefined parameter matrix
    Init(input_x_dim, input_y_dim, input_z_dim,
         filt_x_dim, filt_y_dim,
         filt_x_step, filt_y_step,
         input_vectorization,
         matrix_filename);
  } else {
    ok = ok && cfl->GetValue("num-filters", &num_filters);
    if (!ok)
      KALDI_ERR << "Bad initializer " << cfl->WholeLine();
    // initialize from configuration
    int32 filter_input_dim = filt_x_dim * filt_y_dim * input_z_dim;
    BaseFloat param_stddev = 1.0 / std::sqrt(filter_input_dim), bias_stddev = 1.0;
    cfl->GetValue("param-stddev", &param_stddev);
    cfl->GetValue("bias-stddev", &bias_stddev);
    Init(input_x_dim, input_y_dim, input_z_dim,
         filt_x_dim, filt_y_dim, filt_x_step, filt_y_step, num_filters,
         input_vectorization, param_stddev, bias_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

// Inline methods to convert from tensor index i.e., (x,y,z) index
// to index in yzx or zyx vectorized tensors
inline int32 YzxVectorIndex(int32 x, int32 y, int32 z,
                            int32 input_x_dim,
                            int32 input_y_dim,
                            int32 input_z_dim) {
  KALDI_PARANOID_ASSERT(x < input_x_dim && y < input_y_dim && z < input_z_dim);
  return (input_y_dim * input_z_dim) * x + (input_y_dim) * z + y;
}

inline int32 ZyxVectorIndex(int32 x, int32 y, int32 z,
                            int32 input_x_dim,
                            int32 input_y_dim,
                            int32 input_z_dim) {
  KALDI_PARANOID_ASSERT(x < input_x_dim && y < input_y_dim && z < input_z_dim);
  return (input_y_dim * input_z_dim) * x + (input_z_dim) * y + z;
}

// Method to convert from a matrix representing a minibatch of vectorized
// 3D tensors to patches for convolution, each patch corresponds to
// one dot product in the convolution
void ConvolutionComponent::InputToInputPatches(
    const CuMatrixBase<BaseFloat>& in,
    CuMatrix<BaseFloat> *patches) const{
  int32 num_x_steps = (1 + (input_x_dim_ - filt_x_dim_) / filt_x_step_);
  int32 num_y_steps = (1 + (input_y_dim_ - filt_y_dim_) / filt_y_step_);
  const int32 filt_x_step = filt_x_step_,
              filt_y_step = filt_y_step_,
              filt_x_dim = filt_x_dim_,
              filt_y_dim = filt_y_dim_,
              input_x_dim = input_x_dim_,
              input_y_dim = input_y_dim_,
              input_z_dim = input_z_dim_,
              filter_dim = filter_params_.NumCols();

  std::vector<int32> column_map(patches->NumCols());
  int32 column_map_size = column_map.size();
  for (int32 x_step = 0; x_step < num_x_steps; x_step++) {
    for (int32 y_step = 0; y_step < num_y_steps; y_step++)  {
      int32 patch_number = x_step * num_y_steps + y_step;
      int32 patch_start_index = patch_number * filter_dim;
      for (int32 x = 0, index = patch_start_index; x < filt_x_dim; x++)  {
        for (int32 y = 0; y < filt_y_dim; y++)  {
          for (int32 z = 0; z < input_z_dim; z++, index++)  {
            KALDI_ASSERT(index < column_map_size);
            if (input_vectorization_ == kZyx)  {
              column_map[index] = ZyxVectorIndex(x_step * filt_x_step + x,
                                                 y_step * filt_y_step + y, z,
                                                 input_x_dim, input_y_dim,
                                                 input_z_dim);
            } else if (input_vectorization_ == kYzx)  {
              column_map[index] = YzxVectorIndex(x_step * filt_x_step + x,
                                                  y_step * filt_y_step + y, z,
                                                  input_x_dim, input_y_dim,
                                                  input_z_dim);
            }
          }
        }
      }
    }
  }
  CuArray<int32> cu_cols(column_map);
  patches->CopyCols(in, cu_cols);
}


// propagation function
// see function declaration in nnet-simple-component.h for details
void* ConvolutionComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                         const CuMatrixBase<BaseFloat> &in,
                                         CuMatrixBase<BaseFloat> *out) const {
  const int32 num_x_steps = (1 + (input_x_dim_ - filt_x_dim_) / filt_x_step_),
              num_y_steps = (1 + (input_y_dim_ - filt_y_dim_) / filt_y_step_),
              num_filters = filter_params_.NumRows(),
              num_frames = in.NumRows(),
              filter_dim = filter_params_.NumCols();
  KALDI_ASSERT((*out).NumRows() == num_frames &&
               (*out).NumCols() == (num_filters * num_x_steps * num_y_steps));

  CuMatrix<BaseFloat> patches(num_frames,
                              num_x_steps * num_y_steps * filter_dim,
                              kUndefined);
  InputToInputPatches(in, &patches);
  CuSubMatrix<BaseFloat>* filter_params_elem = new CuSubMatrix<BaseFloat>(
      filter_params_, 0, filter_params_.NumRows(), 0, filter_params_.NumCols());
  std::vector<CuSubMatrix<BaseFloat>* > tgt_batch, patch_batch,
      filter_params_batch;

  for (int32 x_step = 0; x_step < num_x_steps; x_step++)  {
    for (int32 y_step = 0; y_step < num_y_steps; y_step++)  {
      int32 patch_number = x_step * num_y_steps + y_step;
      tgt_batch.push_back(new CuSubMatrix<BaseFloat>(
              out->ColRange(patch_number * num_filters, num_filters)));
      patch_batch.push_back(new CuSubMatrix<BaseFloat>(
              patches.ColRange(patch_number * filter_dim, filter_dim)));
      filter_params_batch.push_back(filter_params_elem);
      tgt_batch[patch_number]->AddVecToRows(1.0, bias_params_, 1.0); // add bias
    }
  }
  // apply all filters
  AddMatMatBatched<BaseFloat>(1.0, tgt_batch, patch_batch,
                              kNoTrans, filter_params_batch,
                              kTrans, 1.0);
  // release memory
  delete filter_params_elem;
  for (int32 p = 0; p < tgt_batch.size(); p++) {
    delete tgt_batch[p];
    delete patch_batch[p];
  }
  return NULL;
}

// scale the parameters
void ConvolutionComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    filter_params_.SetZero();
    bias_params_.SetZero();
  } else {
    filter_params_.Scale(scale);
    bias_params_.Scale(scale);
  }
}

// add another convolution component
void ConvolutionComponent::Add(BaseFloat alpha, const Component &other_in) {
  const ConvolutionComponent *other =
      dynamic_cast<const ConvolutionComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  filter_params_.AddMat(alpha, other->filter_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

/*
 This function transforms a vector of lists into a list of vectors,
 padded with -1.
 @param[in] The input vector of lists. Let in.size() be D, and let
            the longest list length (i.e. the max of in[i].size()) be L.
 @param[out] The output list of vectors. The length of the list will
            be L, each vector-dimension will be D (i.e. out[i].size() == D),
            and if in[i] == j, then for some k we will have that
            out[k][j] = i. The output vectors are padded with -1
            where necessary if not all the input lists have the same side.
*/
void RearrangeIndexes(const std::vector<std::vector<int32> > &in,
                                                std::vector<std::vector<int32> > *out) {
  int32 D = in.size();
  int32 L = 0;
  for (int32 i = 0; i < D; i++)
    if (in[i].size() > L)
      L = in[i].size();
  out->resize(L);
  for (int32 i = 0; i < L; i++)
    (*out)[i].resize(D, -1);
  for (int32 i = 0; i < D; i++) {
    for (int32 j = 0; j < in[i].size(); j++) {
      (*out)[j][i] = in[i][j];
    }
  }
}

// Method to compute the input derivative matrix from the input derivatives
// for patches, where each patch corresponds to one dot product
// in the convolution
void ConvolutionComponent::InderivPatchesToInderiv(
    const CuMatrix<BaseFloat>& in_deriv_patches,
    CuMatrixBase<BaseFloat> *in_deriv) const {

  const int32 num_x_steps = (1 + (input_x_dim_ - filt_x_dim_) / filt_x_step_),
              num_y_steps = (1 + (input_y_dim_ - filt_y_dim_) / filt_y_step_),
              filt_x_step = filt_x_step_,
              filt_y_step = filt_y_step_,
              filt_x_dim = filt_x_dim_,
              filt_y_dim = filt_y_dim_,
              input_x_dim = input_x_dim_,
              input_y_dim = input_y_dim_,
              input_z_dim = input_z_dim_,
              filter_dim = filter_params_.NumCols();

  // Compute the reverse column_map from the matrix with input
  // derivative patches to input derivative matrix
  std::vector<std::vector<int32> > reverse_column_map(in_deriv->NumCols());
  int32 rev_col_map_size = reverse_column_map.size();
  for (int32 x_step = 0; x_step < num_x_steps; x_step++) {
    for (int32 y_step = 0; y_step < num_y_steps; y_step++)  {
      int32 patch_number = x_step * num_y_steps + y_step;
      int32 patch_start_index = patch_number * filter_dim;
      for (int32 x = 0, index = patch_start_index; x < filt_x_dim; x++)  {
        for (int32 y = 0; y < filt_y_dim; y++)  {
          for (int32 z = 0; z < input_z_dim; z++, index++)  {
            int32 vector_index;
            if (input_vectorization_ == kZyx)  {
              vector_index = ZyxVectorIndex(x_step * filt_x_step + x,
                                            y_step * filt_y_step + y, z,
                                            input_x_dim, input_y_dim,
                                            input_z_dim);
            } else {
              KALDI_ASSERT(input_vectorization_ == kYzx);
              vector_index = YzxVectorIndex(x_step * filt_x_step + x,
                                            y_step * filt_y_step + y, z,
                                            input_x_dim, input_y_dim,
                                            input_z_dim);
            }
            KALDI_ASSERT(vector_index < rev_col_map_size);
            reverse_column_map[vector_index].push_back(index);
          }
        }
      }
    }
  }
  std::vector<std::vector<int32> > rearranged_column_map;
  RearrangeIndexes(reverse_column_map, &rearranged_column_map);
  for (int32 p = 0; p < rearranged_column_map.size(); p++) {
    CuArray<int32> cu_cols(rearranged_column_map[p]);
    in_deriv->AddCols(in_deriv_patches, cu_cols);
  }
}

// back propagation function
// see function declaration in nnet-simple-component.h for details
void ConvolutionComponent::Backprop(const std::string &debug_info,
                                    const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &in_value,
                                    const CuMatrixBase<BaseFloat> &, // out_value,
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    void *memo,
                                    Component *to_update_in,
                                    CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("ConvolutionComponent::Backprop");
  ConvolutionComponent *to_update =
      dynamic_cast<ConvolutionComponent*>(to_update_in);
  const int32 num_x_steps = (1 + (input_x_dim_ - filt_x_dim_) / filt_x_step_),
              num_y_steps = (1 + (input_y_dim_ - filt_y_dim_) / filt_y_step_),
              num_filters = filter_params_.NumRows(),
              num_frames = out_deriv.NumRows(),
              filter_dim = filter_params_.NumCols();

  KALDI_ASSERT(out_deriv.NumRows() == num_frames &&
               out_deriv.NumCols() ==
               (num_filters * num_x_steps * num_y_steps));

  // Compute inderiv patches
  CuMatrix<BaseFloat> in_deriv_patches(num_frames,
                                       num_x_steps * num_y_steps * filter_dim,
                                       kSetZero);

  std::vector<CuSubMatrix<BaseFloat>* > patch_deriv_batch, out_deriv_batch,
      filter_params_batch;
  CuSubMatrix<BaseFloat>* filter_params_elem = new CuSubMatrix<BaseFloat>(
      filter_params_, 0, filter_params_.NumRows(), 0, filter_params_.NumCols());

  for (int32 x_step = 0; x_step < num_x_steps; x_step++)  {
    for (int32 y_step = 0; y_step < num_y_steps; y_step++)  {
      int32 patch_number = x_step * num_y_steps + y_step;

      patch_deriv_batch.push_back(new CuSubMatrix<BaseFloat>(
              in_deriv_patches.ColRange(
              patch_number * filter_dim, filter_dim)));
      out_deriv_batch.push_back(new CuSubMatrix<BaseFloat>(out_deriv.ColRange(
              patch_number * num_filters, num_filters)));
      filter_params_batch.push_back(filter_params_elem);
    }
  }
  AddMatMatBatched<BaseFloat>(1.0, patch_deriv_batch,
                              out_deriv_batch, kNoTrans,
                              filter_params_batch, kNoTrans, 0.0);

  if (in_deriv) {
    // combine the derivatives from the individual input deriv patches
    // to compute input deriv matrix
    InderivPatchesToInderiv(in_deriv_patches, in_deriv);
  }

  if (to_update != NULL)  {
    to_update->Update(debug_info, in_value, out_deriv, out_deriv_batch);
  }

  // release memory
  delete filter_params_elem;
  for (int32 p = 0; p < patch_deriv_batch.size(); p++) {
    delete patch_deriv_batch[p];
    delete out_deriv_batch[p];
  }
}


// update parameters
// see function declaration in nnet-simple-component.h for details
void ConvolutionComponent::Update(const std::string &debug_info,
                                  const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  const std::vector<CuSubMatrix<BaseFloat> *>& out_deriv_batch) {
  // useful dims
  const int32 num_x_steps = (1 + (input_x_dim_ - filt_x_dim_) / filt_x_step_),
              num_y_steps = (1 + (input_y_dim_ - filt_y_dim_) / filt_y_step_),
              num_filters = filter_params_.NumRows(),
              num_frames = out_deriv.NumRows(),
              filter_dim = filter_params_.NumCols();
  KALDI_ASSERT(out_deriv.NumRows() == num_frames &&
               out_deriv.NumCols() ==
               (num_filters * num_x_steps * num_y_steps));


  CuMatrix<BaseFloat> filters_grad;
  CuVector<BaseFloat> bias_grad;

  CuMatrix<BaseFloat> input_patches(num_frames,
                                    filter_dim * num_x_steps * num_y_steps,
                                    kUndefined);
  InputToInputPatches(in_value, &input_patches);

  filters_grad.Resize(num_filters, filter_dim, kSetZero); // reset
  bias_grad.Resize(num_filters, kSetZero); // reset

  // create a single large matrix holding the smaller matrices
  // from the vector container filters_grad_batch along the rows
  CuMatrix<BaseFloat> filters_grad_blocks_batch(
      num_x_steps * num_y_steps * filters_grad.NumRows(),
      filters_grad.NumCols());

  std::vector<CuSubMatrix<BaseFloat>* > filters_grad_batch, input_patch_batch;

  for (int32 x_step = 0; x_step < num_x_steps; x_step++)  {
    for (int32 y_step = 0; y_step < num_y_steps; y_step++)  {
      int32 patch_number = x_step * num_y_steps + y_step;
      filters_grad_batch.push_back(new CuSubMatrix<BaseFloat>(
          filters_grad_blocks_batch.RowRange(
              patch_number * filters_grad.NumRows(), filters_grad.NumRows())));

      input_patch_batch.push_back(new CuSubMatrix<BaseFloat>(
              input_patches.ColRange(patch_number * filter_dim, filter_dim)));
    }
  }

  AddMatMatBatched<BaseFloat>(1.0, filters_grad_batch, out_deriv_batch, kTrans,
                              input_patch_batch, kNoTrans, 1.0);

  // add the row blocks together to filters_grad
  filters_grad.AddMatBlocks(1.0, filters_grad_blocks_batch);

  // create a matrix holding the col blocks sum of out_deriv
  CuMatrix<BaseFloat> out_deriv_col_blocks_sum(out_deriv.NumRows(),
                                               num_filters);

  // add the col blocks together to out_deriv_col_blocks_sum
  out_deriv_col_blocks_sum.AddMatBlocks(1.0, out_deriv);

  bias_grad.AddRowSumMat(1.0, out_deriv_col_blocks_sum, 1.0);

  // release memory
  for (int32 p = 0; p < input_patch_batch.size(); p++) {
    delete filters_grad_batch[p];
    delete input_patch_batch[p];
  }

  //
  // update
  //
  filter_params_.AddMat(learning_rate_, filters_grad);
  bias_params_.AddVec(learning_rate_, bias_grad);
}

void ConvolutionComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // Read opening tag and learning rate.
  ExpectToken(is, binary, "<InputXDim>");
  ReadBasicType(is, binary, &input_x_dim_);
  ExpectToken(is, binary, "<InputYDim>");
  ReadBasicType(is, binary, &input_y_dim_);
  ExpectToken(is, binary, "<InputZDim>");
  ReadBasicType(is, binary, &input_z_dim_);
  ExpectToken(is, binary, "<FiltXDim>");
  ReadBasicType(is, binary, &filt_x_dim_);
  ExpectToken(is, binary, "<FiltYDim>");
  ReadBasicType(is, binary, &filt_y_dim_);
  ExpectToken(is, binary, "<FiltXStep>");
  ReadBasicType(is, binary, &filt_x_step_);
  ExpectToken(is, binary, "<FiltYStep>");
  ReadBasicType(is, binary, &filt_y_step_);
  ExpectToken(is, binary, "<InputVectorization>");
  int32 input_vectorization;
  ReadBasicType(is, binary, &input_vectorization);
  input_vectorization_ = static_cast<TensorVectorizationType>(input_vectorization);
  ExpectToken(is, binary, "<FilterParams>");
  filter_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  std::string tok;
  ReadToken(is, binary, &tok);
  if (tok == "<IsGradient>") {
    ReadBasicType(is, binary, &is_gradient_);
    ExpectToken(is, binary, "</ConvolutionComponent>");
  } else {
    is_gradient_ = false;
    KALDI_ASSERT(tok == "</ConvolutionComponent>");
  }
}

void ConvolutionComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // write opening tag and learning rate.
  WriteToken(os, binary, "<InputXDim>");
  WriteBasicType(os, binary, input_x_dim_);
  WriteToken(os, binary, "<InputYDim>");
  WriteBasicType(os, binary, input_y_dim_);
  WriteToken(os, binary, "<InputZDim>");
  WriteBasicType(os, binary, input_z_dim_);
  WriteToken(os, binary, "<FiltXDim>");
  WriteBasicType(os, binary, filt_x_dim_);
  WriteToken(os, binary, "<FiltYDim>");
  WriteBasicType(os, binary, filt_y_dim_);
  WriteToken(os, binary, "<FiltXStep>");
  WriteBasicType(os, binary, filt_x_step_);
  WriteToken(os, binary, "<FiltYStep>");
  WriteBasicType(os, binary, filt_y_step_);
  WriteToken(os, binary, "<InputVectorization>");
  WriteBasicType(os, binary, static_cast<int32>(input_vectorization_));
  WriteToken(os, binary, "<FilterParams>");
  filter_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</ConvolutionComponent>");
}

BaseFloat ConvolutionComponent::DotProduct(const UpdatableComponent &other_in) const {
  const ConvolutionComponent *other =
      dynamic_cast<const ConvolutionComponent*>(&other_in);
  return TraceMatMat(filter_params_, other->filter_params_, kTrans)
         + VecVec(bias_params_, other->bias_params_);
}

Component* ConvolutionComponent::Copy() const {
  ConvolutionComponent *ans = new ConvolutionComponent(*this);
  return ans;
}

void ConvolutionComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_filter_params(filter_params_);
  temp_filter_params.SetRandn();
  filter_params_.AddMat(stddev, temp_filter_params);

  CuVector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

void ConvolutionComponent::SetParams(const VectorBase<BaseFloat> &bias,
                                     const MatrixBase<BaseFloat> &filter) {
  bias_params_ = bias;
  filter_params_ = filter;
  KALDI_ASSERT(bias_params_.Dim() == filter_params_.NumRows());
}

int32 ConvolutionComponent::NumParameters() const {
  return (filter_params_.NumCols() + 1) * filter_params_.NumRows();
}

void ConvolutionComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  int32 num_filter_params = filter_params_.NumCols() * filter_params_.NumRows();
  params->Range(0, num_filter_params).CopyRowsFromMat(filter_params_);
  params->Range(num_filter_params, bias_params_.Dim()).CopyFromVec(bias_params_);
}
void ConvolutionComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  int32 num_filter_params = filter_params_.NumCols() * filter_params_.NumRows();
  filter_params_.CopyRowsFromVec(params.Range(0, num_filter_params));
  bias_params_.CopyFromVec(params.Range(num_filter_params, bias_params_.Dim()));
}

// aquire input dim
int32 MaxpoolingComponent::InputDim() const {
  return input_x_dim_ * input_y_dim_ * input_z_dim_;
}

MaxpoolingComponent::MaxpoolingComponent(
    const MaxpoolingComponent &component):
    input_x_dim_(component.input_x_dim_),
    input_y_dim_(component.input_y_dim_),
    input_z_dim_(component.input_z_dim_),
    pool_x_size_(component.pool_x_size_),
    pool_y_size_(component.pool_y_size_),
    pool_z_size_(component.pool_z_size_),
    pool_x_step_(component.pool_x_step_),
    pool_y_step_(component.pool_y_step_),
    pool_z_step_(component.pool_z_step_) { }

// aquire output dim
int32 MaxpoolingComponent::OutputDim() const {
  int32 num_pools_x = 1 + (input_x_dim_ - pool_x_size_) / pool_x_step_;
  int32 num_pools_y = 1 + (input_y_dim_ - pool_y_size_) / pool_y_step_;
  int32 num_pools_z = 1 + (input_z_dim_ - pool_z_size_) / pool_z_step_;
  return num_pools_x * num_pools_y * num_pools_z;
}

// check the component parameters
void MaxpoolingComponent::Check() const {
  // sanity check of the max pooling parameters
  KALDI_ASSERT(input_x_dim_ > 0);
  KALDI_ASSERT(input_y_dim_ > 0);
  KALDI_ASSERT(input_z_dim_ > 0);
  KALDI_ASSERT(pool_x_size_ > 0);
  KALDI_ASSERT(pool_y_size_ > 0);
  KALDI_ASSERT(pool_z_size_ > 0);
  KALDI_ASSERT(pool_x_step_ > 0);
  KALDI_ASSERT(pool_y_step_ > 0);
  KALDI_ASSERT(pool_z_step_ > 0);
  KALDI_ASSERT(input_x_dim_ >= pool_x_size_);
  KALDI_ASSERT(input_y_dim_ >= pool_y_size_);
  KALDI_ASSERT(input_z_dim_ >= pool_z_size_);
  KALDI_ASSERT(pool_x_size_ >= pool_x_step_);
  KALDI_ASSERT(pool_y_size_ >= pool_y_step_);
  KALDI_ASSERT(pool_z_size_ >= pool_z_step_);
  KALDI_ASSERT((input_x_dim_ - pool_x_size_) % pool_x_step_  == 0);
  KALDI_ASSERT((input_y_dim_ - pool_y_size_) % pool_y_step_  == 0);
  KALDI_ASSERT((input_z_dim_ - pool_z_size_) % pool_z_step_  == 0);
}

// initialize the component using configuration file
void MaxpoolingComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;

  ok = ok && cfl->GetValue("input-x-dim", &input_x_dim_);
  ok = ok && cfl->GetValue("input-y-dim", &input_y_dim_);
  ok = ok && cfl->GetValue("input-z-dim", &input_z_dim_);
  ok = ok && cfl->GetValue("pool-x-size", &pool_x_size_);
  ok = ok && cfl->GetValue("pool-y-size", &pool_y_size_);
  ok = ok && cfl->GetValue("pool-z-size", &pool_z_size_);
  ok = ok && cfl->GetValue("pool-x-step", &pool_x_step_);
  ok = ok && cfl->GetValue("pool-y-step", &pool_y_step_);
  ok = ok && cfl->GetValue("pool-z-step", &pool_z_step_);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();

  Check();
}

// Method to convert from a matrix representing a minibatch of vectorized
// 3D tensors to patches for 3d max pooling, each patch corresponds to
// the nodes having the same local coordinatenodes from each pool
void MaxpoolingComponent::InputToInputPatches(
    const CuMatrixBase<BaseFloat>& in,
    CuMatrix<BaseFloat> *patches) const{
  int32 num_pools_x = 1 + (input_x_dim_ - pool_x_size_) / pool_x_step_;
  int32 num_pools_y = 1 + (input_y_dim_ - pool_y_size_) / pool_y_step_;
  int32 num_pools_z = 1 + (input_z_dim_ - pool_z_size_) / pool_z_step_;

  std::vector<int32> column_map(patches->NumCols());
  int32 column_map_size = column_map.size();
  for (int32 x = 0, index =0; x < pool_x_size_; x++) {
    for (int32 y = 0; y < pool_y_size_; y++) {
      for (int32 z = 0; z < pool_z_size_; z++) {
        // given the local node coordinate, group them from each pool
        // to form a patch
        for (int32 x_pool = 0; x_pool < num_pools_x; x_pool++) {
          for (int32 y_pool = 0; y_pool < num_pools_y; y_pool++) {
            for (int32 z_pool = 0; z_pool < num_pools_z; z_pool++, index++) {
              KALDI_ASSERT(index < column_map_size);
              column_map[index] = (x_pool * pool_x_step_ + x) * input_y_dim_ * input_z_dim_ +
                                  (y_pool * pool_y_step_ + y) * input_z_dim_ +
                                  (z_pool * pool_z_step_ + z);

            }
          }
        }
      }
    }
  }
  CuArray<int32> cu_cols(column_map);
  patches->CopyCols(in, cu_cols);
}

/*
  This is the 3d max pooling propagate function.
  It is assumed that each row of the input matrix
  is a vectorized 3D-tensor of type zxy.
  Similar to the propagate function of ConvolutionComponent,
  the input matrix is first arranged into patches so that
  pools (with / without overlapping) could be
  processed in a parallelizable manner.
  The output matrix is also a vectorized 3D-tensor of type zxy.
*/

void* MaxpoolingComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &in,
                                    CuMatrixBase<BaseFloat> *out) const {
  int32 num_frames = in.NumRows();
  int32 num_pools = OutputDim();
  int32 pool_size = pool_x_size_ * pool_y_size_ * pool_z_size_;
  CuMatrix<BaseFloat> patches(num_frames, num_pools * pool_size, kUndefined);
  InputToInputPatches(in, &patches);

  out->Set(-1e20); // reset a large negative value
  for (int32 q = 0; q < pool_size; q++)
    out->Max(patches.ColRange(q * num_pools, num_pools));
  return NULL;
}

// Method to compute the input derivative matrix from the input derivatives
// for patches, where each patch corresponds to
// the nodes having the same local coordinatenodes from each pool
void MaxpoolingComponent::InderivPatchesToInderiv(
    const CuMatrix<BaseFloat>& in_deriv_patches,
    CuMatrixBase<BaseFloat> *in_deriv) const {

  int32 num_pools_x = 1 + (input_x_dim_ - pool_x_size_) / pool_x_step_;
  int32 num_pools_y = 1 + (input_y_dim_ - pool_y_size_) / pool_y_step_;
  int32 num_pools_z = 1 + (input_z_dim_ - pool_z_size_) / pool_z_step_;

  std::vector<std::vector<int32> > reverse_column_map(in_deriv->NumCols());
  int32 rev_col_map_size = reverse_column_map.size();
  for (int32 x = 0, index = 0; x < pool_x_size_; x++) {
    for (int32 y = 0; y < pool_y_size_; y++) {
      for (int32 z = 0; z < pool_z_size_; z++) {

        for (int32 x_pool = 0; x_pool < num_pools_x; x_pool++) {
          for (int32 y_pool = 0; y_pool < num_pools_y; y_pool++) {
            for (int32 z_pool = 0; z_pool < num_pools_z; z_pool++, index++) {
              int32 vector_index = (x_pool * pool_x_step_ + x) * input_y_dim_ * input_z_dim_ +
                                  (y_pool * pool_y_step_ + y) * input_z_dim_ +
                                  (z_pool * pool_z_step_ + z);

              KALDI_ASSERT(vector_index < rev_col_map_size);
              reverse_column_map[vector_index].push_back(index);
            }
          }
        }
      }
    }
  }
  std::vector<std::vector<int32> > rearranged_column_map;
  RearrangeIndexes(reverse_column_map, &rearranged_column_map);
  for (int32 p = 0; p < rearranged_column_map.size(); p++) {
    CuArray<int32> cu_cols(rearranged_column_map[p]);
    in_deriv->AddCols(in_deriv_patches, cu_cols);
  }
}

/*
  3d max pooling backpropagate function
  This function backpropagate the error from
  out_deriv to in_deriv.
  In order to select the node in each pool to
  backpropagate the error, it has to compare
  the output pool value stored in the out_value
  matrix with each of its input pool member node
  stroed in the in_value matrix.
*/
void MaxpoolingComponent::Backprop(const std::string &debug_info,
                                   const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in_value,
                                   const CuMatrixBase<BaseFloat> &out_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv,
                                   void *memo,
                                   Component *, // to_update,
                                   CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("MaxpoolingComponent::Backprop");
  if (!in_deriv)
    return;

  int32 num_frames = in_value.NumRows();
  int32 num_pools = OutputDim();
  int32 pool_size = pool_x_size_ * pool_y_size_ * pool_z_size_;
  CuMatrix<BaseFloat> patches(num_frames, num_pools * pool_size, kUndefined);
  InputToInputPatches(in_value, &patches);

  for (int32 q = 0; q < pool_size; q++) {
    // zero-out mask
    CuMatrix<BaseFloat> mask;
    out_value.EqualElementMask(patches.ColRange(q * num_pools, num_pools), &mask);
    mask.MulElements(out_deriv);
    patches.ColRange(q * num_pools, num_pools).CopyFromMat(mask);
  }

  // combine the derivatives from the individual input deriv patches
  // to compute input deriv matrix
  InderivPatchesToInderiv(patches, in_deriv);
}

void MaxpoolingComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<MaxpoolingComponent>", "<InputXDim>");
  ReadBasicType(is, binary, &input_x_dim_);
  ExpectToken(is, binary, "<InputYDim>");
  ReadBasicType(is, binary, &input_y_dim_);
  ExpectToken(is, binary, "<InputZDim>");
  ReadBasicType(is, binary, &input_z_dim_);
  ExpectToken(is, binary, "<PoolXSize>");
  ReadBasicType(is, binary, &pool_x_size_);
  ExpectToken(is, binary, "<PoolYSize>");
  ReadBasicType(is, binary, &pool_y_size_);
  ExpectToken(is, binary, "<PoolZSize>");
  ReadBasicType(is, binary, &pool_z_size_);
  ExpectToken(is, binary, "<PoolXStep>");
  ReadBasicType(is, binary, &pool_x_step_);
  ExpectToken(is, binary, "<PoolYStep>");
  ReadBasicType(is, binary, &pool_y_step_);
  ExpectToken(is, binary, "<PoolZStep>");
  ReadBasicType(is, binary, &pool_z_step_);
  ExpectToken(is, binary, "</MaxpoolingComponent>");
  Check();
}

void MaxpoolingComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<MaxpoolingComponent>");
  WriteToken(os, binary, "<InputXDim>");
  WriteBasicType(os, binary, input_x_dim_);
  WriteToken(os, binary, "<InputYDim>");
  WriteBasicType(os, binary, input_y_dim_);
  WriteToken(os, binary, "<InputZDim>");
  WriteBasicType(os, binary, input_z_dim_);
  WriteToken(os, binary, "<PoolXSize>");
  WriteBasicType(os, binary, pool_x_size_);
  WriteToken(os, binary, "<PoolYSize>");
  WriteBasicType(os, binary, pool_y_size_);
  WriteToken(os, binary, "<PoolZSize>");
  WriteBasicType(os, binary, pool_z_size_);
  WriteToken(os, binary, "<PoolXStep>");
  WriteBasicType(os, binary, pool_x_step_);
  WriteToken(os, binary, "<PoolYStep>");
  WriteBasicType(os, binary, pool_y_step_);
  WriteToken(os, binary, "<PoolZStep>");
  WriteBasicType(os, binary, pool_z_step_);
  WriteToken(os, binary, "</MaxpoolingComponent>");
}

// display information about component
std::string MaxpoolingComponent::Info() const {
  std::ostringstream stream;
  stream << Type()
         << ", input-x-dim=" << input_x_dim_
         << ", input-y-dim=" << input_y_dim_
         << ", input-z-dim=" << input_z_dim_
         << ", pool-x-size=" << pool_x_size_
         << ", pool-y-size=" << pool_y_size_
         << ", pool-z-size=" << pool_z_size_
         << ", pool-x-step=" << pool_x_step_
         << ", pool-y-step=" << pool_y_step_
         << ", pool-z-step=" << pool_z_step_;
  return stream.str();
}


int32 LstmNonlinearityComponent::InputDim() const {
  int32 cell_dim = value_sum_.NumCols();
  return cell_dim * 5 + (use_dropout_ ? 3 : 0);
}

int32 LstmNonlinearityComponent::OutputDim() const {
  int32 cell_dim = value_sum_.NumCols();
  return cell_dim * 2;
}


void LstmNonlinearityComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // Read opening tag and learning rate.
  ExpectToken(is, binary, "<Params>");
  params_.Read(is, binary);
  ExpectToken(is, binary, "<ValueAvg>");
  value_sum_.Read(is, binary);
  ExpectToken(is, binary, "<DerivAvg>");
  deriv_sum_.Read(is, binary);
  ExpectToken(is, binary, "<SelfRepairConfig>");
  self_repair_config_.Read(is, binary);
  ExpectToken(is, binary, "<SelfRepairProb>");
  self_repair_total_.Read(is, binary);

  std::string tok;
  ReadToken(is, binary, &tok);
  if (tok == "<UseDropout>") {
    ReadBasicType(is, binary, &use_dropout_);
    ReadToken(is, binary, &tok);
  } else {
    use_dropout_ = false;
  }
  KALDI_ASSERT(tok == "<Count>");
  ReadBasicType(is, binary, &count_);

  // For the on-disk format, we normalze value_sum_, deriv_sum_ and
  // self_repair_total_ by dividing by the count, but in memory they are scaled
  // by the count.  [for self_repair_total_, the scaling factor is count_ *
  // cell_dim].
  value_sum_.Scale(count_);
  deriv_sum_.Scale(count_);
  int32 cell_dim = params_.NumCols();
  self_repair_total_.Scale(count_ * cell_dim);

  InitNaturalGradient();

  ExpectToken(is, binary, "</LstmNonlinearityComponent>");

}

void LstmNonlinearityComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Read opening tag and learning rate.

  WriteToken(os, binary, "<Params>");
  params_.Write(os, binary);
  WriteToken(os, binary, "<ValueAvg>");
  {
    Matrix<BaseFloat> value_avg(value_sum_);
    if (count_ != 0.0)
      value_avg.Scale(1.0 / count_);
    value_avg.Write(os, binary);
  }
  WriteToken(os, binary, "<DerivAvg>");
  {
    Matrix<BaseFloat> deriv_avg(deriv_sum_);
    if (count_ != 0.0)
      deriv_avg.Scale(1.0 / count_);
    deriv_avg.Write(os, binary);
  }
  WriteToken(os, binary, "<SelfRepairConfig>");
  self_repair_config_.Write(os, binary);
  WriteToken(os, binary, "<SelfRepairProb>");
  {
    int32 cell_dim = params_.NumCols();
    Vector<BaseFloat> self_repair_prob(self_repair_total_);
    if (count_ != 0.0)
      self_repair_prob.Scale(1.0 / (count_ * cell_dim));
    self_repair_prob.Write(os, binary);
  }
  if (use_dropout_) {
    // only write this if true; we have back-compat code in reading anyway.
    // this makes the models without dropout easier to read with older code.
    WriteToken(os, binary, "<UseDropout>");
    WriteBasicType(os, binary, use_dropout_);
  }
  WriteToken(os, binary, "<Count>");
  WriteBasicType(os, binary, count_);
  WriteToken(os, binary, "</LstmNonlinearityComponent>");
}



std::string LstmNonlinearityComponent::Info() const {
  std::ostringstream stream;
  int32 cell_dim = params_.NumCols();
  stream << UpdatableComponent::Info() << ", cell-dim=" << cell_dim
         << ", use-dropout=" << (use_dropout_ ? "true" : "false");
  PrintParameterStats(stream, "w_ic", params_.Row(0));
  PrintParameterStats(stream, "w_fc", params_.Row(1));
  PrintParameterStats(stream, "w_oc", params_.Row(2));

  // Note: some of the following code mirrors the code in
  // UpdatableComponent::Info(), in nnet-component-itf.cc.
  if (count_ > 0) {
    stream << ", count=" << std::setprecision(3) << count_
           << std::setprecision(6);
  }
  static const char *nonlin_names[] = { "i_t_sigmoid", "f_t_sigmoid", "c_t_tanh",
                                        "o_t_sigmoid", "m_t_tanh" };
  for (int32 i = 0; i < 5; i++) {
    stream << ", " << nonlin_names[i] << "={";
    stream << " self-repair-lower-threshold=" << self_repair_config_(i)
           << ", self-repair-scale=" << self_repair_config_(i + 5);

    if (count_ != 0) {
      BaseFloat self_repaired_proportion =
          self_repair_total_(i) / (count_ * cell_dim);
      stream << ", self-repaired-proportion=" << self_repaired_proportion;
      Vector<double> value_sum(value_sum_.Row(i)),
          deriv_sum(deriv_sum_.Row(i));
      Vector<BaseFloat> value_avg(value_sum), deriv_avg(deriv_sum);
      value_avg.Scale(1.0 / count_);
      deriv_avg.Scale(1.0 / count_);
      stream << ", value-avg=" << SummarizeVector(value_avg)
             << ", deriv-avg=" << SummarizeVector(deriv_avg);
    }
    stream << " }";
  }
  return stream.str();
}


Component* LstmNonlinearityComponent::Copy() const {
  return new LstmNonlinearityComponent(*this);
}

void LstmNonlinearityComponent::ZeroStats() {
  value_sum_.SetZero();
  deriv_sum_.SetZero();
  self_repair_total_.SetZero();
  count_ = 0.0;
}

void LstmNonlinearityComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    params_.SetZero();
    value_sum_.SetZero();
    deriv_sum_.SetZero();
    self_repair_total_.SetZero();
    count_ = 0.0;
  } else {
    params_.Scale(scale);
    value_sum_.Scale(scale);
    deriv_sum_.Scale(scale);
    self_repair_total_.Scale(scale);
    count_ *= scale;
  }
}

void LstmNonlinearityComponent::Add(BaseFloat alpha,
                                    const Component &other_in) {
  const LstmNonlinearityComponent *other =
      dynamic_cast<const LstmNonlinearityComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  params_.AddMat(alpha, other->params_);
  value_sum_.AddMat(alpha, other->value_sum_);
  deriv_sum_.AddMat(alpha, other->deriv_sum_);
  self_repair_total_.AddVec(alpha, other->self_repair_total_);
  count_ += alpha * other->count_;
}

void LstmNonlinearityComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_params(params_.NumRows(), params_.NumCols());
  temp_params.SetRandn();
  params_.AddMat(stddev, temp_params);
}

BaseFloat LstmNonlinearityComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const LstmNonlinearityComponent *other =
      dynamic_cast<const LstmNonlinearityComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  return TraceMatMat(params_, other->params_, kTrans);
}

int32 LstmNonlinearityComponent::NumParameters() const {
  return params_.NumRows() * params_.NumCols();
}

void LstmNonlinearityComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == NumParameters());
  params->CopyRowsFromMat(params_);
}


void LstmNonlinearityComponent::UnVectorize(
    const VectorBase<BaseFloat> &params)  {
  KALDI_ASSERT(params.Dim() == NumParameters());
  params_.CopyRowsFromVec(params);
}


void* LstmNonlinearityComponent::Propagate(
    const ComponentPrecomputedIndexes *, // indexes
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  cu::ComputeLstmNonlinearity(in, params_, out);
  return NULL;
}


void LstmNonlinearityComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &, // out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("LstmNonlinearityComponent::Backprop");

  if (to_update_in == NULL) {
    cu::BackpropLstmNonlinearity(in_value, params_, out_deriv,
                                 deriv_sum_, self_repair_config_,
                                 count_, in_deriv,
                                 (CuMatrixBase<BaseFloat>*) NULL,
                                 (CuMatrixBase<double>*) NULL,
                                 (CuMatrixBase<double>*) NULL,
                                 (CuMatrixBase<BaseFloat>*) NULL);
  } else {
    LstmNonlinearityComponent *to_update =
        dynamic_cast<LstmNonlinearityComponent*>(to_update_in);
    KALDI_ASSERT(to_update != NULL);

    int32 cell_dim = params_.NumCols();
    CuMatrix<BaseFloat> params_deriv(3, cell_dim, kUndefined);
    CuMatrix<BaseFloat> self_repair_total(5, cell_dim, kUndefined);

    cu::BackpropLstmNonlinearity(in_value, params_, out_deriv,
                                 deriv_sum_, self_repair_config_,
                                 count_, in_deriv, &params_deriv,
                                 &(to_update->value_sum_),
                                 &(to_update->deriv_sum_),
                                 &self_repair_total);

    CuVector<BaseFloat> self_repair_total_sum(5);
    self_repair_total_sum.AddColSumMat(1.0, self_repair_total, 0.0);
    to_update->self_repair_total_.AddVec(1.0, self_repair_total_sum);
    to_update->count_ += static_cast<double>(in_value.NumRows());

    BaseFloat scale = 1.0;
    if (!to_update->is_gradient_) {
      to_update->preconditioner_.PreconditionDirections(
          &params_deriv, &scale);
    }
    to_update->params_.AddMat(to_update->learning_rate_ * scale,
                              params_deriv);
  }
}

LstmNonlinearityComponent::LstmNonlinearityComponent(
    const LstmNonlinearityComponent &other):
    UpdatableComponent(other),
    params_(other.params_),
    use_dropout_(other.use_dropout_),
    value_sum_(other.value_sum_),
    deriv_sum_(other.deriv_sum_),
    self_repair_config_(other.self_repair_config_),
    self_repair_total_(other.self_repair_total_),
    count_(other.count_),
    preconditioner_(other.preconditioner_) { }

void LstmNonlinearityComponent::Init(
    int32 cell_dim, bool use_dropout,
    BaseFloat param_stddev,
    BaseFloat tanh_self_repair_threshold,
    BaseFloat sigmoid_self_repair_threshold,
    BaseFloat self_repair_scale) {
  KALDI_ASSERT(cell_dim > 0 && param_stddev >= 0.0 &&
               tanh_self_repair_threshold >= 0.0 &&
               tanh_self_repair_threshold <= 1.0 &&
               sigmoid_self_repair_threshold >= 0.0 &&
               sigmoid_self_repair_threshold <= 0.25 &&
               self_repair_scale >= 0.0 && self_repair_scale <= 0.1);
  use_dropout_ = use_dropout;
  params_.Resize(3, cell_dim);
  params_.SetRandn();
  params_.Scale(param_stddev);
  value_sum_.Resize(5, cell_dim);
  deriv_sum_.Resize(5, cell_dim);
  self_repair_config_.Resize(10);
  self_repair_config_.Range(0, 5).Set(sigmoid_self_repair_threshold);
  self_repair_config_(2) = tanh_self_repair_threshold;
  self_repair_config_(4) = tanh_self_repair_threshold;
  self_repair_config_.Range(5, 5).Set(self_repair_scale);
  self_repair_total_.Resize(5);
  count_ = 0.0;
  InitNaturalGradient();

}

void LstmNonlinearityComponent::InitNaturalGradient() {
  // As regards the configuration for the natural-gradient preconditioner, we
  // don't make it configurable from the command line-- it's unlikely that any
  // differences from changing this would be substantial enough to effectively
  // tune the configuration.  Because the preconditioning code doesn't 'see' the
  // derivatives from individual frames, but only averages over the minibatch,
  // there is a fairly small amount of data available to estimate the Fisher
  // information matrix, so we set the rank, update period and
  // num-samples-history to smaller values than normal.
  preconditioner_.SetRank(20);
  preconditioner_.SetUpdatePeriod(2);
  preconditioner_.SetNumSamplesHistory(1000.0);
}

/// virtual
void LstmNonlinearityComponent::FreezeNaturalGradient(bool freeze) {
  preconditioner_.Freeze(freeze);
}

void LstmNonlinearityComponent::InitFromConfig(ConfigLine *cfl) {
  InitLearningRatesFromConfig(cfl);
  bool ok = true;
  bool use_dropout = false;
  int32 cell_dim;
  // these self-repair thresholds are the normal defaults for tanh and sigmoid
  // respectively.  If, later on, we decide that we want to support different
  // self-repair config values for the individual sigmoid and tanh
  // nonlinearities, we can modify this code then.
  BaseFloat tanh_self_repair_threshold = 0.2,
      sigmoid_self_repair_threshold = 0.05,
      self_repair_scale = 1.0e-05;
  // param_stddev is the stddev of the parameters.  it may be better to
  // use a smaller value but this was the default in the python scripts
  // for a while.
  BaseFloat param_stddev = 1.0;
  ok = ok && cfl->GetValue("cell-dim", &cell_dim);
  cfl->GetValue("param-stddev", &param_stddev);
  cfl->GetValue("tanh-self-repair-threshold",
                &tanh_self_repair_threshold);
  cfl->GetValue("sigmoid-self-repair-threshold",
                &sigmoid_self_repair_threshold);
  cfl->GetValue("self-repair-scale", &self_repair_scale);
  cfl->GetValue("use-dropout", &use_dropout);

  // We may later on want to make it possible to initialize the different
  // parameters w_ic, w_fc and w_oc with different biases.  We'll implement
  // that when and if it's needed.

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (ok) {
    Init(cell_dim, use_dropout, param_stddev, tanh_self_repair_threshold,
         sigmoid_self_repair_threshold, self_repair_scale);
  } else {
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  }
}

void LstmNonlinearityComponent::ConsolidateMemory() {
  OnlineNaturalGradient preconditioner_temp(preconditioner_);
  preconditioner_.Swap(&preconditioner_);
}


int32 GruNonlinearityComponent::InputDim() const {
  if (recurrent_dim_ == cell_dim_) {
    // non-projected GRU.
    return 4 * cell_dim_;
  } else {
    return 3 * cell_dim_ + 2 * recurrent_dim_;
  }
}

int32 GruNonlinearityComponent::OutputDim() const {
  return 2 * cell_dim_;
}


std::string GruNonlinearityComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", cell-dim=" << cell_dim_
         << ", recurrent-dim=" << recurrent_dim_;
  PrintParameterStats(stream, "w_h", w_h_);
  stream << ", self-repair-threshold=" << self_repair_threshold_
         << ", self-repair-scale=" << self_repair_scale_;
  if (count_ > 0) {  // c.f. NonlinearComponent::Info().
    stream << ", count=" << std::setprecision(3) << count_
           << std::setprecision(6);
    stream << ", self-repaired-proportion="
           << (self_repair_total_ / (count_ * cell_dim_));
    Vector<double> value_avg_dbl(value_sum_);
    Vector<BaseFloat> value_avg(value_avg_dbl);
    value_avg.Scale(1.0 / count_);
    stream << ", value-avg=" << SummarizeVector(value_avg);
    Vector<double> deriv_avg_dbl(deriv_sum_);
    Vector<BaseFloat> deriv_avg(deriv_avg_dbl);
    deriv_avg.Scale(1.0 / count_);
    stream << ", deriv-avg=" << SummarizeVector(deriv_avg);
  }
  // natural-gradient parameters.
  stream << ", alpha=" << preconditioner_in_.GetAlpha()
         << ", rank-in=" << preconditioner_in_.GetRank()
         << ", rank-out=" << preconditioner_out_.GetRank()
         << ", update-period="
         << preconditioner_in_.GetUpdatePeriod();
  return stream.str();
}

void GruNonlinearityComponent::InitFromConfig(ConfigLine *cfl) {
  cell_dim_ = -1;
  recurrent_dim_ = -1;
  self_repair_threshold_ = 0.2;
  self_repair_scale_ = 1.0e-05;

  InitLearningRatesFromConfig(cfl);
  if (!cfl->GetValue("cell-dim", &cell_dim_) || cell_dim_ <= 0)
    KALDI_ERR << "cell-dim > 0 is required for GruNonlinearityComponent.";

  BaseFloat param_stddev = 1.0 / std::sqrt(cell_dim_),
      alpha = 4.0;
  int32 rank_in = 20, rank_out = 80,
      update_period = 4;

  cfl->GetValue("recurrent-dim", &recurrent_dim_);
  cfl->GetValue("self-repair-threshold", &self_repair_threshold_);
  cfl->GetValue("self-repair-scale", &self_repair_scale_);
  cfl->GetValue("param-stddev", &param_stddev);
  cfl->GetValue("alpha", &alpha);
  cfl->GetValue("rank-in", &rank_in);
  cfl->GetValue("rank-out", &rank_out);
  cfl->GetValue("update-period", &update_period);

  if (recurrent_dim_ < 0)
    recurrent_dim_ = cell_dim_;
  if (recurrent_dim_ == 0 || recurrent_dim_ > cell_dim_)
    KALDI_ERR << "Invalid values for cell-dim and recurrent-dim";

  w_h_.Resize(cell_dim_, recurrent_dim_);
  w_h_.SetRandn();
  w_h_.Scale(param_stddev);

  preconditioner_in_.SetAlpha(alpha);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_in_.SetUpdatePeriod(update_period);
  preconditioner_out_.SetAlpha(alpha);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_out_.SetUpdatePeriod(update_period);

  count_ = 0.0;
  self_repair_total_ = 0.0;
  value_sum_.Resize(cell_dim_);
  deriv_sum_.Resize(cell_dim_);

  Check();
}

void* GruNonlinearityComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumRows() == out->NumRows() &&
               in.NumCols() == InputDim() &&
               out->NumCols() == OutputDim());
  // If recurrent_dim_ != cell_dim_, this is projected GRU and we
  // are computing:
  //  (z_t, r_t, hpart_t, c_{t-1}, s_{t-1}) -> (h_t, c_t).
  // Otherwise (no projection), it's
  //  (z_t, r_t, hpart_t, y_{t-1},) -> (h_t, y_t).
  // but to understand this code, it's better to rename y to c:
  //  (z_t, r_t, hpart_t, c_{t-1}) -> (h_t, c_t).
  int32 num_rows = in.NumRows(),
      c = cell_dim_,
      r =  recurrent_dim_;
  CuSubMatrix<BaseFloat> z_t(in, 0, num_rows, 0, c),
      r_t(in, 0, num_rows, c, r),
      hpart_t(in, 0, num_rows, c + r, c),
      c_t1(in, 0, num_rows, c + r + c, c);
  // note: the variable named 'c_t1' actually represents
  // y_{t-1} for non-projected GRUs.

  // By setting s_t1 to the last recurrent_dim_ rows of 'in', we get something
  // that represents s_{t-1} for recurrent setups and y_{t-1} (which we're
  // renaming to c_{t-1}) for non-projected GRUs.  The key thing is that
  // in the non-projected case, the variables c_t1 and s_t1 point to the
  // same memory.
  CuSubMatrix<BaseFloat> s_t1(in, 0, num_rows, in.NumCols() - r, r);

  // note: for non-projected GRUs, c_t below is actually y_t.
  CuSubMatrix<BaseFloat> h_t(*out, 0, num_rows, 0, c),
      c_t(*out, 0, num_rows, c, c);

  // sdotr is the only temporary storage we need in the forward pass.
  CuMatrix<BaseFloat> sdotr(num_rows, r);
  sdotr.AddMatMatElements(1.0, r_t, s_t1, 0.0);
  // now sdotr = r_t \dot s_{t-1}.
  h_t.CopyFromMat(hpart_t);
  // now h_t = hpart_t (note: hpart_t actually means U^h x_t).
  h_t.AddMatMat(1.0, sdotr, kNoTrans, w_h_, kTrans, 1.0);
  // now h_t = hpart_t + W^h (s_{t-1} \dot r_t).
  h_t.Tanh(h_t);
  // now, h_t = tanh(hpart_t + W^h (s_{t-1} \dot r_t)).

  c_t.CopyFromMat(h_t);
  // now c_t = h_t
  c_t.AddMatMatElements(-1.0, z_t, h_t, 1.0);
  // now c_t = (1 - z_t) \dot h_t.
  c_t.AddMatMatElements(1.0, z_t, c_t1, 1.0);
  // now c_t = (1 - z_t) \dot h_t  +  z_t \dot c_{t-1}.
  return NULL;
}

void GruNonlinearityComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *, // indexes
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("GruNonlinearityComponent::Backprop");
  KALDI_ASSERT(SameDim(out_value, out_deriv) &&
               in_value.NumRows() == out_value.NumRows() &&
               in_value.NumCols() == InputDim() &&
               out_value.NumCols() == OutputDim() &&
               (in_deriv == NULL || SameDim(in_value, *in_deriv)) &&
               memo == NULL);
  GruNonlinearityComponent *to_update =
      dynamic_cast<GruNonlinearityComponent*>(to_update_in);
  KALDI_ASSERT(in_deriv != NULL || to_update != NULL);
  int32 num_rows = in_value.NumRows(),
      c = cell_dim_,
      r = recurrent_dim_;

  // To understand what's going on here, compare this code with the
  // corresponding 'forward' code in Propagate().


  CuSubMatrix<BaseFloat> z_t(in_value, 0, num_rows, 0, c),
      r_t(in_value, 0, num_rows, c, r),
      hpart_t(in_value, 0, num_rows, c + r, c),
      c_t1(in_value, 0, num_rows, c + r + c, c),
      s_t1(in_value, 0, num_rows, in_value.NumCols() - r, r);


  // The purpose of this 'in_deriv_ptr' is so that we can create submatrices
  // like z_t_deriv without the code crashing.  If in_deriv is NULL these point
  // to 'in_value', and we'll be careful never to actually write to these
  // sub-matrices, which aside from being conceptually wrong would violate the
  // const semantics of this function.
  const CuMatrixBase<BaseFloat> *in_deriv_ptr =
      (in_deriv == NULL ? &in_value : in_deriv);
  CuSubMatrix<BaseFloat> z_t_deriv(*in_deriv_ptr, 0, num_rows, 0, c),
      r_t_deriv(*in_deriv_ptr, 0, num_rows, c, r),
      hpart_t_deriv(*in_deriv_ptr, 0, num_rows, c + r, c),
      c_t1_deriv(*in_deriv_ptr, 0, num_rows, c + r + c, c),
      s_t1_deriv(*in_deriv_ptr, 0, num_rows, in_value.NumCols() - r, r);

  // Note: the output h_t is never actually used in the GRU computation (we only
  // output it because we want the value to be cached to save computation in the
  // backprop), so we expect that the 'h_t_deriv', if we extracted it in the
  // obvious way, would be all zeros.
  // We create a different, local h_t_deriv
  // variable that backpropagates the derivative from c_t_deriv.
  CuSubMatrix<BaseFloat> h_t(out_value, 0, num_rows, 0, c),
      c_t(out_value, 0, num_rows, c, c),
      c_t_deriv(out_deriv, 0, num_rows, c, c);
  CuMatrix<BaseFloat> h_t_deriv(num_rows, c, kUndefined);

  {  // we initialize h_t_deriv with the derivative from 'out_deriv'.
    // In real life in a GRU, this would always be zero; but in testing
    // code it may be nonzero and we include this term so that
    // the tests don't fail.  Note: if you were to remove these
    // lines, you'd have to change 'h_t_deriv.AddMat(1.0, c_t_deriv);' below
    // to a CopyFromMat() call.
    CuSubMatrix<BaseFloat> h_t_deriv_in(out_deriv, 0, num_rows, 0, c);
    h_t_deriv.CopyFromMat(h_t_deriv_in);
  }


  // sdotr is the same variable as used in the forward pass, it will contain
  // r_t \dot s_{t-1}.
  CuMatrix<BaseFloat> sdotr(num_rows, r);
  sdotr.AddMatMatElements(1.0, r_t, s_t1, 0.0);


  { // This block does the
    // backprop corresponding to the
    // forward-pass expression: c_t = (1 - z_t) \dot h_t + z_t \dot c_{t-1}.

    // First do: h_t_deriv = c_t_deriv \dot (1 - z_t).
    h_t_deriv.AddMat(1.0, c_t_deriv);
    h_t_deriv.AddMatMatElements(-1.0, c_t_deriv, z_t, 1.0);

    if (in_deriv) {
      // these should be self-explanatory if you study
      // the expression "c_t = (1 - z_t) \dot h_t + z_t \dot c_{t-1}".
      z_t_deriv.AddMatMatElements(-1.0, c_t_deriv, h_t, 1.0);
      z_t_deriv.AddMatMatElements(1.0, c_t_deriv, c_t1, 1.0);
      c_t1_deriv.AddMatMatElements(1.0, c_t_deriv, z_t, 1.0);
    }
  }

  h_t_deriv.DiffTanh(h_t, h_t_deriv);
  if (to_update)
    to_update->TanhStatsAndSelfRepair(h_t, &h_t_deriv);


  if (to_update)
    to_update->UpdateParameters(sdotr, h_t_deriv);

  // At this point, 'h_t_deriv' contains the derivative w.r.t.
  // the argument of the tanh function, i.e. w.r.t. the expression:
  //    hpart_t + W^h (s_{t-1} \dot r_t).
  // The next block propagates this to the derivatives for
  // hpart_t, s_{t-1} and r_t.
  if (in_deriv) {
    hpart_t_deriv.AddMat(1.0, h_t_deriv);

    // We re-use the memory that we used for s_{t-1} \dot r_t,
    // for its derivative.
    CuMatrix<BaseFloat> &sdotr_deriv(sdotr);
    sdotr_deriv.AddMatMat(1.0, h_t_deriv, kNoTrans, w_h_, kNoTrans, 0.0);

    // we add to all the input-derivatives instead of setting them,
    // because we chose to export the flag kBackpropAdds.
    r_t_deriv.AddMatMatElements(1.0, sdotr_deriv, s_t1, 1.0);
    s_t1_deriv.AddMatMatElements(1.0, sdotr_deriv, r_t, 1.0);
  }
}


void GruNonlinearityComponent::TanhStatsAndSelfRepair(
    const CuMatrixBase<BaseFloat> &h_t,
    CuMatrixBase<BaseFloat> *h_t_deriv) {
  KALDI_ASSERT(SameDim(h_t, *h_t_deriv));

  // we use this probability (hardcoded for now) to limit the stats accumulation
  // and self-repair code to running on about half of the minibatches.
  BaseFloat repair_and_stats_probability = 0.5;
  if (RandUniform() > repair_and_stats_probability)
    return;

  // OK, accumulate stats.
  // For the next few lines, compare with TanhComponent::StoreStats(), which is where
  // we got this code.
  // tanh_deriv is the function derivative of the tanh function,
  // tanh'(x) = tanh(x) * (1.0 - tanh(x)).  h_t corresponds to tanh(x).
  CuMatrix<BaseFloat> tanh_deriv(h_t);
  tanh_deriv.ApplyPow(2.0);
  tanh_deriv.Scale(-1.0);
  tanh_deriv.Add(1.0);

  count_ += h_t.NumRows();
  CuVector<BaseFloat> temp(cell_dim_);
  temp.AddRowSumMat(1.0, h_t, 0.0);
  value_sum_.AddVec(1.0, temp);
  temp.AddRowSumMat(1.0, tanh_deriv, 0.0);
  deriv_sum_.AddVec(1.0, temp);

  if (count_ <= 0.0) {
    // this would be rather pathological if it happened.
    return;
  }

  // The rest of this function contains code modified from
  // TanhComponent::RepairGradients().

  // thresholds_vec is actually a 1-row matrix.  (the ApplyHeaviside
  // function isn't defined for vectors).
  CuMatrix<BaseFloat> thresholds(1, cell_dim_);
  CuSubVector<BaseFloat> thresholds_vec(thresholds, 0);
  thresholds_vec.AddVec(-1.0, deriv_sum_);
  thresholds_vec.Add(self_repair_threshold_ * count_);
  thresholds.ApplyHeaviside();
  self_repair_total_ += thresholds_vec.Sum();

  // there is a comment explaining what we are doing with
  // 'thresholds_vec', at this point in TanhComponent::RepairGradients().
  // We won't repeat it here.

  h_t_deriv->AddMatDiagVec(-self_repair_scale_ / repair_and_stats_probability,
                           h_t, kNoTrans, thresholds_vec);
}

void GruNonlinearityComponent::UpdateParameters(
    const CuMatrixBase<BaseFloat> &sdotr,
    const CuMatrixBase<BaseFloat> &h_t_deriv) {
  if (is_gradient_) {
    // 'simple' update, no natural gradient.  Compare
    // with AffineComponent::UpdateSimple().
    w_h_.AddMatMat(learning_rate_, h_t_deriv, kTrans,
                   sdotr, kNoTrans, 1.0);
  } else {
    // the natural-gradient update.
    CuMatrix<BaseFloat> in_value_temp(sdotr),
        out_deriv_temp(h_t_deriv);

    // These "scale" values get will get multiplied into the learning rate.
    BaseFloat in_scale, out_scale;

    preconditioner_in_.PreconditionDirections(&in_value_temp, &in_scale);
    preconditioner_out_.PreconditionDirections(&out_deriv_temp, &out_scale);

    BaseFloat local_lrate = learning_rate_ * in_scale * out_scale;
    w_h_.AddMatMat(local_lrate, out_deriv_temp, kTrans,
                   in_value_temp, kNoTrans, 1.0);
  }
}



void GruNonlinearityComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);
  ExpectToken(is, binary, "<CellDim>");
  ReadBasicType(is, binary, &cell_dim_);
  ExpectToken(is, binary, "<RecurrentDim>");
  ReadBasicType(is, binary, &recurrent_dim_);
  ExpectToken(is, binary, "<w_h>");
  w_h_.Read(is, binary);
  ExpectToken(is, binary, "<ValueAvg>");
  value_sum_.Read(is, binary);
  ExpectToken(is, binary, "<DerivAvg>");
  deriv_sum_.Read(is, binary);
  ExpectToken(is, binary, "<SelfRepairTotal>");
  ReadBasicType(is, binary, &self_repair_total_);
  ExpectToken(is, binary, "<Count>");
  ReadBasicType(is, binary, &count_);
  value_sum_.Scale(count_);  // we read in the averages, not the sums.
  deriv_sum_.Scale(count_);
  ExpectToken(is, binary, "<SelfRepairThreshold>");
  ReadBasicType(is, binary, &self_repair_threshold_);
  ExpectToken(is, binary, "<SelfRepairScale>");
  ReadBasicType(is, binary, &self_repair_scale_);
  BaseFloat alpha;
  int32 rank_in, rank_out, update_period;
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha);
  ExpectToken(is, binary, "<RankInOut>");
  ReadBasicType(is, binary, &rank_in);
  ReadBasicType(is, binary, &rank_out);
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetAlpha(alpha);
  preconditioner_out_.SetAlpha(alpha);
  preconditioner_in_.SetUpdatePeriod(update_period);
  preconditioner_out_.SetUpdatePeriod(update_period);
  ExpectToken(is, binary, "</GruNonlinearityComponent>");
}

void GruNonlinearityComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);
  WriteToken(os, binary, "<CellDim>");
  WriteBasicType(os, binary, cell_dim_);
  WriteToken(os, binary, "<RecurrentDim>");
  WriteBasicType(os, binary, recurrent_dim_);
  WriteToken(os, binary, "<w_h>");
  w_h_.Write(os, binary);
  {
    // Write the value and derivative stats in a count-normalized way, for
    // greater readability in text form.
    WriteToken(os, binary, "<ValueAvg>");
    Vector<BaseFloat> temp(value_sum_);
    if (count_ != 0.0) temp.Scale(1.0 / count_);
    temp.Write(os, binary);
    WriteToken(os, binary, "<DerivAvg>");
    temp.CopyFromVec(deriv_sum_);
    if (count_ != 0.0) temp.Scale(1.0 / count_);
    temp.Write(os, binary);
  }
  WriteToken(os, binary, "<SelfRepairTotal>");
  WriteBasicType(os, binary, self_repair_total_);
  WriteToken(os, binary, "<Count>");
  WriteBasicType(os, binary, count_);
  WriteToken(os, binary, "<SelfRepairThreshold>");
  WriteBasicType(os, binary, self_repair_threshold_);
  WriteToken(os, binary, "<SelfRepairScale>");
  WriteBasicType(os, binary, self_repair_scale_);

  BaseFloat alpha = preconditioner_in_.GetAlpha();
  int32 rank_in = preconditioner_in_.GetRank(),
      rank_out = preconditioner_out_.GetRank(),
      update_period = preconditioner_in_.GetUpdatePeriod();
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha);
  WriteToken(os, binary, "<RankInOut>");
  WriteBasicType(os, binary, rank_in);
  WriteBasicType(os, binary, rank_out);
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, update_period);
  WriteToken(os, binary, "</GruNonlinearityComponent>");
}

void GruNonlinearityComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    w_h_.SetZero();
    value_sum_.SetZero();
    deriv_sum_.SetZero();
    self_repair_total_ = 0.0;
    count_ = 0.0;
  } else {
    w_h_.Scale(scale);
    value_sum_.Scale(scale);
    deriv_sum_.Scale(scale);
    self_repair_total_ *= scale;
    count_ *= scale;
  }
}

void GruNonlinearityComponent::Add(BaseFloat alpha,
                                   const Component &other_in) {
  const GruNonlinearityComponent *other =
      dynamic_cast<const GruNonlinearityComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  w_h_.AddMat(alpha, other->w_h_);
  value_sum_.AddVec(alpha, other->value_sum_);
  deriv_sum_.AddVec(alpha, other->deriv_sum_);
  self_repair_total_ += alpha * other->self_repair_total_;
  count_ += alpha * other->count_;
}

void GruNonlinearityComponent::ZeroStats() {
  value_sum_.SetZero();
  deriv_sum_.SetZero();
  self_repair_total_ = 0.0;
  count_ = 0.0;
}

void GruNonlinearityComponent::Check() const {
  KALDI_ASSERT(cell_dim_ > 0 && recurrent_dim_ > 0 &&
               recurrent_dim_ <= cell_dim_ &&
               self_repair_threshold_ >= 0.0 &&
               self_repair_scale_ >= 0.0 );
  KALDI_ASSERT(w_h_.NumRows() == cell_dim_ &&
               w_h_.NumCols() == recurrent_dim_);
  KALDI_ASSERT(value_sum_.Dim() == cell_dim_ &&
               deriv_sum_.Dim() == cell_dim_);
}

void GruNonlinearityComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_params(w_h_.NumRows(), w_h_.NumCols());
  temp_params.SetRandn();
  w_h_.AddMat(stddev, temp_params);
}

BaseFloat GruNonlinearityComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const GruNonlinearityComponent *other =
      dynamic_cast<const GruNonlinearityComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  return TraceMatMat(w_h_, other->w_h_, kTrans);
}

int32 GruNonlinearityComponent::NumParameters() const {
  return w_h_.NumRows() * w_h_.NumCols();
}

void GruNonlinearityComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == NumParameters());
  params->CopyRowsFromMat(w_h_);
}


void GruNonlinearityComponent::UnVectorize(
    const VectorBase<BaseFloat> &params)  {
  KALDI_ASSERT(params.Dim() == NumParameters());
  w_h_.CopyRowsFromVec(params);
}

void GruNonlinearityComponent::FreezeNaturalGradient(bool freeze) {
  preconditioner_in_.Freeze(freeze);
  preconditioner_out_.Freeze(freeze);
}

GruNonlinearityComponent::GruNonlinearityComponent(
    const GruNonlinearityComponent &other):
    UpdatableComponent(other),
    cell_dim_(other.cell_dim_),
    recurrent_dim_(other.recurrent_dim_),
    w_h_(other.w_h_),
    value_sum_(other.value_sum_),
    deriv_sum_(other.deriv_sum_),
    self_repair_total_(other.self_repair_total_),
    count_(other.count_),
    self_repair_threshold_(other.self_repair_threshold_),
    self_repair_scale_(other.self_repair_scale_),
    preconditioner_in_(other.preconditioner_in_),
    preconditioner_out_(other.preconditioner_out_) {
  Check();
}


int32 OutputGruNonlinearityComponent::InputDim() const {
  return 3 * cell_dim_;
}

int32 OutputGruNonlinearityComponent::OutputDim() const {
  return 2 * cell_dim_;
}


std::string OutputGruNonlinearityComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", cell-dim=" << cell_dim_;
  PrintParameterStats(stream, "w_h", w_h_);
  stream << ", self-repair-threshold=" << self_repair_threshold_
         << ", self-repair-scale=" << self_repair_scale_;
  if (count_ > 0) {  // c.f. NonlinearComponent::Info().
    stream << ", count=" << std::setprecision(3) << count_
           << std::setprecision(6);
    stream << ", self-repaired-proportion="
           << (self_repair_total_ / (count_ * cell_dim_));
    Vector<double> value_avg_dbl(value_sum_);
    Vector<BaseFloat> value_avg(value_avg_dbl);
    value_avg.Scale(1.0 / count_);
    stream << ", value-avg=" << SummarizeVector(value_avg);
    Vector<double> deriv_avg_dbl(deriv_sum_);
    Vector<BaseFloat> deriv_avg(deriv_avg_dbl);
    deriv_avg.Scale(1.0 / count_);
    stream << ", deriv-avg=" << SummarizeVector(deriv_avg);
  }
  // natural-gradient parameters.
  stream << ", alpha=" << preconditioner_.GetAlpha()
         << ", rank=" << preconditioner_.GetRank()
         << ", update-period="
         << preconditioner_.GetUpdatePeriod();
  return stream.str();
}

void OutputGruNonlinearityComponent::InitFromConfig(ConfigLine *cfl) {
  cell_dim_ = -1;
  self_repair_threshold_ = 0.2;
  self_repair_scale_ = 1.0e-05;

  InitLearningRatesFromConfig(cfl);
  if (!cfl->GetValue("cell-dim", &cell_dim_) || cell_dim_ <= 0)
    KALDI_ERR << "cell-dim > 0 is required for GruNonlinearityComponent.";

  BaseFloat param_mean = 0.0, param_stddev = 1.0, 
      alpha = 4.0;
  int32 rank=8,
      update_period = 10;

  cfl->GetValue("self-repair-threshold", &self_repair_threshold_);
  cfl->GetValue("self-repair-scale", &self_repair_scale_);
  cfl->GetValue("param-mean", &param_mean);
  cfl->GetValue("param-stddev", &param_stddev);
  cfl->GetValue("alpha", &alpha);
  cfl->GetValue("rank", &rank);
  cfl->GetValue("update-period", &update_period);


  w_h_.Resize(cell_dim_);
  w_h_.SetRandn();
  w_h_.Scale(param_stddev);
  w_h_.Add(param_mean);

  preconditioner_.SetAlpha(alpha);
  preconditioner_.SetRank(rank);
  preconditioner_.SetUpdatePeriod(update_period);

  count_ = 0.0;
  self_repair_total_ = 0.0;
  value_sum_.Resize(cell_dim_);
  deriv_sum_.Resize(cell_dim_);

  Check();
}

void* OutputGruNonlinearityComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumRows() == out->NumRows() &&
               in.NumCols() == InputDim() &&
               out->NumCols() == OutputDim());
  // This component implements the function
  // (z_t, hpart_t, c_{t-1}) -> (h_t, c_t)
  // of dimensions
  // (cell_dim, cell_dim, cell_dim) -> (cell_dim, cell_dim),
  // where:
  // h_t = \tanh( hpart_t + W^h \dot c_{t-1} )
  // c_t = (1 - z_t) \dot h_t + z_t \dot c_{t-1}.
  int32 num_rows = in.NumRows(),
      c = cell_dim_;
  CuSubMatrix<BaseFloat> z_t(in, 0, num_rows, 0, c),
      hpart_t(in, 0, num_rows, c, c),
      c_t1(in, 0, num_rows, c + c, c);

  CuSubMatrix<BaseFloat> h_t(*out, 0, num_rows, 0, c),
      c_t(*out, 0, num_rows, c, c);

  h_t.CopyFromMat(c_t1);
  // now h_t = c_{t-1}
  h_t.MulColsVec(w_h_);
  // now h_t = W^h \dot c_{t-1}
  h_t.AddMat(1.0, hpart_t, kNoTrans);
  // now h_t = hpart_t + W^h \dot c_{t-1}.(note: hpart_t actually means U^h x_t).
  h_t.Tanh(h_t);
  // now, h_t = tanh(hpart_t + W^h \dot c_{t-1}).

  c_t.CopyFromMat(h_t);
  // now c_t = h_t
  c_t.AddMatMatElements(-1.0, z_t, h_t, 1.0);
  // now c_t = (1 - z_t) \dot h_t.
  c_t.AddMatMatElements(1.0, z_t, c_t1, 1.0);
  // now c_t = (1 - z_t) \dot h_t  +  z_t \dot c_{t-1}.
  return NULL;
}

void OutputGruNonlinearityComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *, // indexes
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("OutputGruNonlinearityComponent::Backprop");
  KALDI_ASSERT(SameDim(out_value, out_deriv) &&
               in_value.NumRows() == out_value.NumRows() &&
               in_value.NumCols() == InputDim() &&
               out_value.NumCols() == OutputDim() &&
               (in_deriv == NULL || SameDim(in_value, *in_deriv)) &&
               memo == NULL);
  OutputGruNonlinearityComponent *to_update =
      dynamic_cast<OutputGruNonlinearityComponent*>(to_update_in);
  KALDI_ASSERT(in_deriv != NULL || to_update != NULL);
  int32 num_rows = in_value.NumRows(),
      c = cell_dim_;

  // To understand what's going on here, compare this code with the
  // corresponding 'forward' code in Propagate().


  CuSubMatrix<BaseFloat> z_t(in_value, 0, num_rows, 0, c),
      hpart_t(in_value, 0, num_rows, c, c),
      c_t1(in_value, 0, num_rows, c + c, c);

  // The purpose of this 'in_deriv_ptr' is so that we can create submatrices
  // like z_t_deriv without the code crashing.  If in_deriv is NULL these point
  // to 'in_value', and we'll be careful never to actually write to these
  // sub-matrices, which aside from being conceptually wrong would violate the
  // const semantics of this function.
  const CuMatrixBase<BaseFloat> *in_deriv_ptr =
      (in_deriv == NULL ? &in_value : in_deriv);
  CuSubMatrix<BaseFloat> z_t_deriv(*in_deriv_ptr, 0, num_rows, 0, c),
      hpart_t_deriv(*in_deriv_ptr, 0, num_rows, c, c),
      c_t1_deriv(*in_deriv_ptr, 0, num_rows, c + c, c);

  // Note: the output h_t is never actually used in the GRU computation (we only
  // output it because we want the value to be cached to save computation in the
  // backprop), so we expect that the 'h_t_deriv', if we extracted it in the
  // obvious way, would be all zeros.
  // We create a different, local h_t_deriv
  // variable that backpropagates the derivative from c_t_deriv.
  CuSubMatrix<BaseFloat> h_t(out_value, 0, num_rows, 0, c),
      c_t(out_value, 0, num_rows, c, c),
      c_t_deriv(out_deriv, 0, num_rows, c, c);
  CuMatrix<BaseFloat> h_t_deriv(num_rows, c, kUndefined);

  {  // we initialize h_t_deriv with the derivative from 'out_deriv'.
    // In real life in a GRU, this would always be zero; but in testing
    // code it may be nonzero and we include this term so that
    // the tests don't fail.  Note: if you were to remove these
    // lines, you'd have to change 'h_t_deriv.AddMat(1.0, c_t_deriv);' below
    // to a CopyFromMat() call.
    CuSubMatrix<BaseFloat> h_t_deriv_in(out_deriv, 0, num_rows, 0, c);
    h_t_deriv.CopyFromMat(h_t_deriv_in);
  }


  { // This block does the
    // backprop corresponding to the
    // forward-pass expression: c_t = (1 - z_t) \dot h_t + z_t \dot c_{t-1}.

    // First do: h_t_deriv = c_t_deriv \dot (1 - z_t).
    h_t_deriv.AddMat(1.0, c_t_deriv);
    h_t_deriv.AddMatMatElements(-1.0, c_t_deriv, z_t, 1.0);

    if (in_deriv) {
      // these should be self-explanatory if you study
      // the expression "c_t = (1 - z_t) \dot h_t + z_t \dot c_{t-1}".
      z_t_deriv.AddMatMatElements(-1.0, c_t_deriv, h_t, 1.0);
      z_t_deriv.AddMatMatElements(1.0, c_t_deriv, c_t1, 1.0);
      c_t1_deriv.AddMatMatElements(1.0, c_t_deriv, z_t, 1.0);
    }
  }

  h_t_deriv.DiffTanh(h_t, h_t_deriv);
  if (to_update)
    to_update->TanhStatsAndSelfRepair(h_t, &h_t_deriv);
  
  if (to_update)
    to_update->UpdateParameters(c_t1, h_t_deriv);
  // At this point, 'h_t_deriv' contains the derivative w.r.t.
  // the argument of the tanh function, i.e. w.r.t. the expression:
  //    hpart_t + W^h \dot c_{t-1}.
  // The next block propagates this to the derivative for h_part_t and c_t1
  // The derivative of z_t has already been finished.
  if (in_deriv) {
    hpart_t_deriv.AddMat(1.0, h_t_deriv);

    // Currently, c_t1_deriv contains the derivative from
    // c_t = (1 - z_t) \dot h_t + z_t \dot c_{t-1}
    // Now compute the h_t = \tanh(hpart_t + W^h \dot c_{t-1}) part
    h_t_deriv.MulColsVec(w_h_);
    // Combine the two parts
    c_t1_deriv.AddMat(1.0, h_t_deriv);
  }
}


void OutputGruNonlinearityComponent::TanhStatsAndSelfRepair(
    const CuMatrixBase<BaseFloat> &h_t,
    CuMatrixBase<BaseFloat> *h_t_deriv) {
  KALDI_ASSERT(SameDim(h_t, *h_t_deriv));

  // we use this probability (hardcoded for now) to limit the stats accumulation
  // and self-repair code to running on about half of the minibatches.
  BaseFloat repair_and_stats_probability = 0.5;
  if (RandUniform() > repair_and_stats_probability)
    return;

  // OK, accumulate stats.
  // For the next few lines, compare with TanhComponent::StoreStats(), which is where
  // we got this code.
  // tanh_deriv is the function derivative of the tanh function,
  // tanh'(x) = tanh(x) * (1.0 - tanh(x)).  h_t corresponds to tanh(x).
  CuMatrix<BaseFloat> tanh_deriv(h_t);
  tanh_deriv.ApplyPow(2.0);
  tanh_deriv.Scale(-1.0);
  tanh_deriv.Add(1.0);

  count_ += h_t.NumRows();
  CuVector<BaseFloat> temp(cell_dim_);
  temp.AddRowSumMat(1.0, h_t, 0.0);
  value_sum_.AddVec(1.0, temp);
  temp.AddRowSumMat(1.0, tanh_deriv, 0.0);
  deriv_sum_.AddVec(1.0, temp);

  if (count_ <= 0.0) {
    // this would be rather pathological if it happened.
    return;
  }

  // The rest of this function contains code modified from
  // TanhComponent::RepairGradients().

  // thresholds_vec is actually a 1-row matrix.  (the ApplyHeaviside
  // function isn't defined for vectors).
  CuMatrix<BaseFloat> thresholds(1, cell_dim_);
  CuSubVector<BaseFloat> thresholds_vec(thresholds, 0);
  thresholds_vec.AddVec(-1.0, deriv_sum_);
  thresholds_vec.Add(self_repair_threshold_ * count_);
  thresholds.ApplyHeaviside();
  self_repair_total_ += thresholds_vec.Sum();

  // there is a comment explaining what we are doing with
  // 'thresholds_vec', at this point in TanhComponent::RepairGradients().
  // We won't repeat it here.

  h_t_deriv->AddMatDiagVec(-self_repair_scale_ / repair_and_stats_probability,
                           h_t, kNoTrans, thresholds_vec);
}

void OutputGruNonlinearityComponent::UpdateParameters(
    const CuMatrixBase<BaseFloat> &c_t1_value,
    const CuMatrixBase<BaseFloat> &h_t_deriv) {
  if (is_gradient_) {
    // 'simple' update, no natural gradient.  Compare
    // with PerElementScaleComponent::UpdateSimple().
    w_h_.AddDiagMatMat(learning_rate_, h_t_deriv, kTrans,
                       c_t1_value, kNoTrans, 1.0);
  } else {
    // the natural-gradient update.
    CuMatrix<BaseFloat> derivs_per_frame(c_t1_value);
    derivs_per_frame.MulElements(h_t_deriv);

    // This "scale" value gets will get multiplied into the learning rate.
    BaseFloat scale;

    preconditioner_.PreconditionDirections(&derivs_per_frame, &scale);

    CuVector<BaseFloat> delta_w_h(w_h_.Dim());
    delta_w_h.AddRowSumMat(scale * learning_rate_, derivs_per_frame);
    w_h_.AddVec(1.0, delta_w_h);
  }
}



void OutputGruNonlinearityComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);
  ExpectToken(is, binary, "<CellDim>");
  ReadBasicType(is, binary, &cell_dim_);
  ExpectToken(is, binary, "<w_h>");
  w_h_.Read(is, binary);
  ExpectToken(is, binary, "<ValueAvg>");
  value_sum_.Read(is, binary);
  ExpectToken(is, binary, "<DerivAvg>");
  deriv_sum_.Read(is, binary);
  ExpectToken(is, binary, "<SelfRepairTotal>");
  ReadBasicType(is, binary, &self_repair_total_);
  ExpectToken(is, binary, "<Count>");
  ReadBasicType(is, binary, &count_);
  value_sum_.Scale(count_);  // we read in the averages, not the sums.
  deriv_sum_.Scale(count_);
  ExpectToken(is, binary, "<SelfRepairThreshold>");
  ReadBasicType(is, binary, &self_repair_threshold_);
  ExpectToken(is, binary, "<SelfRepairScale>");
  ReadBasicType(is, binary, &self_repair_scale_);
  BaseFloat alpha;
  int32 rank, update_period;
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha);
  ExpectToken(is, binary, "<Rank>");
  ReadBasicType(is, binary, &rank);
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period);
  preconditioner_.SetRank(rank);
  preconditioner_.SetAlpha(alpha);
  preconditioner_.SetUpdatePeriod(update_period);
  ExpectToken(is, binary, "</OutputGruNonlinearityComponent>");
}

void OutputGruNonlinearityComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);
  WriteToken(os, binary, "<CellDim>");
  WriteBasicType(os, binary, cell_dim_);
  WriteToken(os, binary, "<w_h>");
  w_h_.Write(os, binary);
  {
    // Write the value and derivative stats in a count-normalized way, for
    // greater readability in text form.
    WriteToken(os, binary, "<ValueAvg>");
    Vector<BaseFloat> temp(value_sum_);
    if (count_ != 0.0) temp.Scale(1.0 / count_);
    temp.Write(os, binary);
    WriteToken(os, binary, "<DerivAvg>");
    temp.CopyFromVec(deriv_sum_);
    if (count_ != 0.0) temp.Scale(1.0 / count_);
    temp.Write(os, binary);
  }
  WriteToken(os, binary, "<SelfRepairTotal>");
  WriteBasicType(os, binary, self_repair_total_);
  WriteToken(os, binary, "<Count>");
  WriteBasicType(os, binary, count_);
  WriteToken(os, binary, "<SelfRepairThreshold>");
  WriteBasicType(os, binary, self_repair_threshold_);
  WriteToken(os, binary, "<SelfRepairScale>");
  WriteBasicType(os, binary, self_repair_scale_);

  BaseFloat alpha = preconditioner_.GetAlpha();
  int32 rank = preconditioner_.GetRank(),
      update_period = preconditioner_.GetUpdatePeriod();
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha);
  WriteToken(os, binary, "<Rank>");
  WriteBasicType(os, binary, rank);
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, update_period);
  WriteToken(os, binary, "</OutputGruNonlinearityComponent>");
}

void OutputGruNonlinearityComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    w_h_.SetZero();
    value_sum_.SetZero();
    deriv_sum_.SetZero();
    self_repair_total_ = 0.0;
    count_ = 0.0;
  } else {
    w_h_.Scale(scale);
    value_sum_.Scale(scale);
    deriv_sum_.Scale(scale);
    self_repair_total_ *= scale;
    count_ *= scale;
  }
}

void OutputGruNonlinearityComponent::Add(BaseFloat alpha,
                                   const Component &other_in) {
  const OutputGruNonlinearityComponent *other =
      dynamic_cast<const OutputGruNonlinearityComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  w_h_.AddVec(alpha, other->w_h_);
  value_sum_.AddVec(alpha, other->value_sum_);
  deriv_sum_.AddVec(alpha, other->deriv_sum_);
  self_repair_total_ += alpha * other->self_repair_total_;
  count_ += alpha * other->count_;
}

void OutputGruNonlinearityComponent::ZeroStats() {
  value_sum_.SetZero();
  deriv_sum_.SetZero();
  self_repair_total_ = 0.0;
  count_ = 0.0;
}

void OutputGruNonlinearityComponent::Check() const {
  KALDI_ASSERT(cell_dim_ > 0 &&
               self_repair_threshold_ >= 0.0 &&
               self_repair_scale_ >= 0.0 );
  KALDI_ASSERT(w_h_.Dim() == cell_dim_);
  KALDI_ASSERT(value_sum_.Dim() == cell_dim_ &&
               deriv_sum_.Dim() == cell_dim_);
}

void OutputGruNonlinearityComponent::PerturbParams(BaseFloat stddev) {
  CuVector<BaseFloat> temp_params(w_h_.Dim());
  temp_params.SetRandn();
  w_h_.AddVec(stddev, temp_params);
}

BaseFloat OutputGruNonlinearityComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const OutputGruNonlinearityComponent *other =
      dynamic_cast<const OutputGruNonlinearityComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  return VecVec(w_h_, other->w_h_);
}

int32 OutputGruNonlinearityComponent::NumParameters() const {
  return w_h_.Dim();
}

void OutputGruNonlinearityComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == NumParameters());
  params->CopyFromVec(w_h_);
}


void OutputGruNonlinearityComponent::UnVectorize(
    const VectorBase<BaseFloat> &params)  {
  KALDI_ASSERT(params.Dim() == NumParameters());
  w_h_.CopyFromVec(params);
}

void OutputGruNonlinearityComponent::FreezeNaturalGradient(bool freeze) {
  preconditioner_.Freeze(freeze);
}

OutputGruNonlinearityComponent::OutputGruNonlinearityComponent(
    const OutputGruNonlinearityComponent &other):
    UpdatableComponent(other),
    cell_dim_(other.cell_dim_),
    w_h_(other.w_h_),
    value_sum_(other.value_sum_),
    deriv_sum_(other.deriv_sum_),
    self_repair_total_(other.self_repair_total_),
    count_(other.count_),
    self_repair_threshold_(other.self_repair_threshold_),
    self_repair_scale_(other.self_repair_scale_),
    preconditioner_(other.preconditioner_) {
  Check();
}

} // namespace nnet3
} // namespace kaldi
