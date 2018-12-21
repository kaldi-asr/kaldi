// nnet3/nnet-chaina-training.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)
//                2016    Xiaohui Zhang

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

#include "nnet3/nnet-utils.h"
#include "nnet3a/nnet-chaina-training.h"
#include "nnet3a/nnet-chaina-utils.h"

namespace kaldi {
namespace nnet3 {

NnetChainaTopTrainer::NnetChainaTopTrainer(
    const std::string &lang_name,
    const NnetChainaTrainingOptions &config,
    const fst::StdVectorFst &den_fst,
    const differentiable_transform::DifferentiableTransform &transform,
    Nnet *nnet):
    lang_name_(lang_name),
    opts_(config),
    den_graph_(den_fst, nnet->OutputDim("output")),
    transform_(transform),
    compiler_(*nnet, opts_.nnet_config.optimize_config,
              opts_.nnet_config.compiler_config),
    nnet_(nnet),
    delta_nnet_(nnet->Copy()),
    num_minibatches_processed_(0),
    max_change_stats_si_(*nnet),
    max_change_stats_(*nnet) {

  if (opts_.nnet_config.zero_component_stats)
    ZeroComponentStats(nnet);

  ScaleNnet(0.0, delta_nnet_);
  if (opts_.nnet_config.read_cache != "") {
    // It would be complicated to implement, as there are various top nnets
    // and they would all try to read and write the same cache files.
    // To implement this, the best way would be to
    KALDI_WARN << "The read-cache options are not currently supported.";
  }
  KALDI_ASSERT(opts_.nnet_config.momentum >= 0.0 &&
               opts_.nnet_config.max_param_change >= 0.0);
}


/**
   TODO: include this somewhere.
   if (num_minibatches_processed_ == 0) {
    ConsolidateMemory(nnet_);
    ConsolidateMemory(delta_nnet_);
  }
*/


std::shared_ptr<const NnetComputation> NnetChainaTopTrainer::GetComputation(
    const ComputationStructure &s) {
  {
    auto iter = computation_map_.find(s);
    if (iter != computation_map_.end())
      return iter->second;
  }
  int32 num_sequences = s.num_sequences,
      frames_per_sequence_in = s.frames_per_sequence_in,
      frames_per_sequence_out = s.frames_per_sequence_out,
      first_input_t = s.first_input_t,
      first_output_t = 0,
      top_subsampling_factor = s.top_subsampling_factor;

  ComputationRequest request;
  request.need_model_derivative = opts_.train_top_nnet;

  request.store_component_stats = true;
  request.inputs.resize(1);
  request.inputs[0].name = "input";
  request.inputs[0].indexes.resize(frames_per_sequence_in * num_sequences);
  request.inputs[0].has_deriv = s.need_input_deriv;
  // The inputs are in the order: all frames of sequence 0; then all frames of
  // sequence 1; and so on.  This is done
  auto iter = request.inputs[0].indexes.begin();
  for (int32 n = 0; n < num_sequences; n++) {
    for (int32 t = first_input_t;
         t < first_input_t + frames_per_sequence_in; ++t,++iter) {
      iter->n = n;
      iter->t = t;
    }
  }
  // ... but the outputs are in the order: the first frame of all sequences;
  // the second frame of all sequences; and so on.
  request.outputs.resize(2);
  request.outputs[0].name = (s.adapted ? "output" : "output-si");
  request.outputs[0].has_deriv = true;
  request.outputs[0].indexes.resize(frames_per_sequence_out * num_sequences);
  int32 t_stride_out = top_subsampling_factor;
  iter = request.outputs[0].indexes.begin();
  for (int32 t = first_output_t;
       t < first_output_t + frames_per_sequence_out * t_stride_out;
       t += t_stride_out) {
    for (int32 n = 0; n < num_sequences; ++n,++iter) {
      iter->n = n;
      iter->t = t;
    }
  }
  request.outputs[1].has_deriv = true;
  request.outputs[1].name = (s.adapted ? "output-xent" : "output-xent-si");
  request.outputs[1].indexes = request.outputs[0].indexes;
  std::shared_ptr<const NnetComputation> computation = compiler_.Compile(
      request);
  computation_map_[s] = computation;
  return computation;
}

bool NnetChainaTopTrainer::TrainUnadapted(
    const CuMatrixBase<BaseFloat> &input,
    const NnetComputation &computation,
    const chain::Supervision &supervision,
    const CuVectorBase<BaseFloat> &deriv_weights,
    Posterior *posterior,
    CuMatrixBase<BaseFloat> *input_deriv) {

  const NnetTrainerOptions &nnet_config = opts_.nnet_config;

  // note: because we give the 1st arg (nnet_) as a pointer to the
  // constructor of 'computer', it will use that copy of the nnet to
  // store stats.
  NnetComputer computer(nnet_config.compute_config, computation,
                        nnet_, delta_nnet_);

  // Freeze the natural gradient.  We dont want to update the NG scatter
  // matrices on this data because we'll next be running the same nnet on the
  // speaker-adapted version of the same data, and it would violate the
  // independence assumptions needed for NG to work if we updated them.
  FreezeNaturalGradient(true, delta_nnet_);

  // give the inputs to the computer object.
  CuMatrix<BaseFloat> input_copy(input);
  computer.AcceptInput("input", &input_copy);
  computer.Run();

  const CuMatrixBase<BaseFloat>
      &output = computer.GetOutput("output-si"),
      &output_xent = computer.GetOutput("output-si-xent");
  CuMatrix<BaseFloat> output_deriv(output.NumRows(),
                                   output.NumCols(),
                                   kUndefined),
      output_xent_deriv;

  // Note: we don't normally use the l2 term any more, parameter-level
  // regularization seems to work better.
  BaseFloat tot_objf, tot_l2_term, tot_weight;

  ComputeChainObjfAndDeriv(opts_.chain_config, den_graph_,
                           supervision, output,
                           &tot_objf, &tot_l2_term, &tot_weight,
                           &output_deriv, &output_xent_deriv,
                           posterior);

  {
    // this block computes and keeps track of the cross-entropy objective.
    // at this point, xent_deriv is posteriors derived from the numerator
    // computation.  note, xent_objf has a factor of '.supervision.weight'
    BaseFloat xent_objf = TraceMatMat(output_xent, output_xent_deriv, kTrans);
    output_si_xent_objf_.UpdateStats(lang_name_ + ":output-si-xent",
                                  opts_.nnet_config.print_interval,
                                  num_minibatches_processed_,
                                  tot_weight, xent_objf);
  }

  if (opts_.apply_deriv_weights && deriv_weights.Dim() != 0) {
    output_deriv.MulRowsVec(deriv_weights);
    output_xent_deriv.MulRowsVec(deriv_weights);
  }

  if (opts_.unadapted_deriv_scale != 1.0)
    output_deriv.Scale(opts_.unadapted_deriv_scale);

  computer.AcceptInput("output-si", &output_deriv);

  output_xent_deriv.Scale(opts_.chain_config.xent_regularize *
                          opts_.unadapted_deriv_scale);
  computer.AcceptInput("output-si-xent", &output_xent_deriv);

  output_si_objf_.UpdateStats(lang_name_ + ":output-si",
                              opts_.nnet_config.print_interval,
                              num_minibatches_processed_,
                              tot_weight, tot_objf, tot_l2_term);

  // Do the backprop.  We know we're either updating the nnet or need the
  // input derivatives (else, what point is there in training), so there
  // must be a backprop pass.
  computer.Run();

  if (input_deriv != NULL) {
    input_deriv->AddMat(opts_.unadapted_backprop_scale,
                        computer.GetOutput("input"));
  }

  // Updates the parameters of nnet.  Since the derivatives will all be scaled
  // with "unadapted_deriv_scale" it makes sense to apply that same factor to
  // the max-change, to keep the max-change in proportion with how much we
  // expect the net to change (so smaller max-change values don't lead to more
  // emphasize on the unadapted model's derivatives)
  bool success = UpdateNnetWithMaxChange(
      *delta_nnet_,
      nnet_config.max_param_change,
      opts_.unadapted_deriv_scale,
      1.0 - nnet_config.momentum,  // normally momentum is 0.0.
      nnet_, &max_change_stats_si_);

  // Un-freeze the natural gradient.
  FreezeNaturalGradient(false, delta_nnet_);

  if (!success)
    ScaleNnet(nnet_config.momentum, delta_nnet_);
  else
    ScaleNnet(0.0, delta_nnet_);
  return success;
}

bool NnetChainaTopTrainer::TrainAdapted(
    const CuMatrixBase<BaseFloat> &input,
    const NnetComputation &computation,
    const chain::Supervision &supervision,
    const CuVectorBase<BaseFloat> &deriv_weights,
    CuMatrixBase<BaseFloat> *input_deriv) {

  const NnetTrainerOptions &nnet_config = opts_.nnet_config;

  // note: because we give the 1st arg (nnet_) as a pointer to the
  // constructor of 'computer', it will use that copy of the nnet to
  // store stats.
  NnetComputer computer(nnet_config.compute_config, computation,
                        nnet_, delta_nnet_);

  // give the inputs to the computer object.
  CuMatrix<BaseFloat> input_copy(input);
  computer.AcceptInput("input", &input_copy);
  computer.Run();

  const CuMatrixBase<BaseFloat>
      &output = computer.GetOutput("output"),
      &output_xent = computer.GetOutput("output-xent");
  CuMatrix<BaseFloat> output_deriv(output.NumRows(),
                                   output.NumCols(),
                                   kUndefined),
      output_xent_deriv;

  // Note: we don't normally use the l2 term any more, parameter-level
  // regularization seems to work better.
  BaseFloat tot_objf, tot_l2_term, tot_weight;

  ComputeChainObjfAndDeriv(opts_.chain_config, den_graph_,
                           supervision, output,
                           &tot_objf, &tot_l2_term, &tot_weight,
                           &output_deriv, &output_xent_deriv);

  {
    // this block computes and keeps track of the cross-entropy objective.
    // at this point, xent_deriv is posteriors derived from the numerator
    // computation.  note, xent_objf has a factor of '.supervision.weight'
    BaseFloat xent_objf = TraceMatMat(output_xent, output_xent_deriv, kTrans);
    output_xent_objf_.UpdateStats(lang_name_ + ":output-xent",
                                  opts_.nnet_config.print_interval,
                                  num_minibatches_processed_,
                                  tot_weight, xent_objf);
  }

  if (opts_.apply_deriv_weights && deriv_weights.Dim() != 0) {
    output_deriv.MulRowsVec(deriv_weights);
    output_xent_deriv.MulRowsVec(deriv_weights);
  }

  computer.AcceptInput("output", &output_deriv);
  output_xent_deriv.Scale(opts_.chain_config.xent_regularize);
  computer.AcceptInput("output-xent", &output_xent_deriv);

  output_objf_.UpdateStats(lang_name_ + ":output",
                           opts_.nnet_config.print_interval,
                           num_minibatches_processed_,
                           tot_weight, tot_objf, tot_l2_term);

  if (input_deriv == NULL && !opts_.train_top_nnet) {
    // We're neither training the top model nor need the input derivatives.
    // E.g., we might be just getting stats for batch normalization after
    // training the model.
    return true;
  }

  // Do the backprop.  We know we're either updating the nnet or need the
  // input derivatives (else, what point is there in training), so there
  // must be a backprop pass.
  computer.Run();

  if (input_deriv != NULL) {
    input_deriv->AddMat(1.0, computer.GetOutput("input"));
  }

  // Update the parameters of nnet.
  bool success = UpdateNnetWithMaxChange(
      *delta_nnet_,
      nnet_config.max_param_change,
      1.0,
      1.0 - nnet_config.momentum,  // normally momentum is 0.0.
      nnet_, &max_change_stats_);

  // Scale down the batchnorm stats (keeps them fresh... this affects what
  // happens when, later on, we use the model with batchnorm test-mode set).
  ScaleBatchnormStats(nnet_config.batchnorm_stats_scale, nnet_);

  // The following will only do something if we have a LinearComponent
  // or AffineComponent with orthonormal-constraint set to a nonzero value.
  ConstrainOrthonormal(nnet_);

  if (!success)
    ScaleNnet(nnet_config.momentum, delta_nnet_);
  else
    ScaleNnet(0.0, delta_nnet_);
  return success;
}


bool NnetChainaTopTrainer::Train(const CuMatrixBase<BaseFloat> &input,
                                 int32 num_sequences,
                                 int32 num_spk,
                                 int32 first_input_t,
                                 int32 top_subsampling_factor,
                                 const VectorBase<BaseFloat> &deriv_weights_in,
                                 const chain::Supervision &supervision,
                                 CuMatrixBase<BaseFloat> *input_deriv) {
  KALDI_ASSERT(input.NumRows() != 0 && input.NumRows() % num_sequences != 0);
  int32 frames_per_sequence_in = input.NumRows() / num_sequences,
      frames_per_sequence_out = supervision.frames_per_sequence;

  bool adapted = false;
  ComputationStructure structure(
      adapted, (input_deriv != NULL),
      num_sequences, frames_per_sequence_in, frames_per_sequence_out,
      first_input_t, top_subsampling_factor);

  Posterior post;

  CuVector<BaseFloat> deriv_weights(deriv_weights_in);

  std::shared_ptr<const NnetComputation> computation_unadapted =
      GetComputation(structure);
  if (!TrainUnadapted(input, *computation_unadapted, supervision,
                      deriv_weights, &post, input_deriv)) {
    num_minibatches_processed_++;
    if (input_deriv)
      input_deriv->SetZero();
    return false;
  }


  Posterior post_padded(input.NumRows());
  ConvertPosterior(post, num_sequences, first_input_t,
                   top_subsampling_factor, &post_padded);

  structure.adapted = true;
  std::shared_ptr<const NnetComputation> computation_adapted =
      GetComputation(structure);

  CuMatrix<BaseFloat> adapted_input(input.NumRows(), input.NumCols(),
                                    kUndefined),
      adapted_input_deriv(input.NumRows(), input.NumCols());

  using namespace differentiable_transform;
  MinibatchInfoItf *minibatch_info = transform_.TrainingForward(
      input, num_sequences, num_spk, post_padded, &adapted_input);

  if (!TrainAdapted(adapted_input, *computation_adapted, supervision,
                    deriv_weights, &adapted_input_deriv)) {
    num_minibatches_processed_++;
    if (input_deriv)
      input_deriv->SetZero();
    return false;
  }

  if (input_deriv == NULL) {
    delete minibatch_info;
  } else {
    transform_.TrainingBackward(input, adapted_input_deriv,
                                num_sequences, num_spk, post_padded,
                                minibatch_info, input_deriv);
  }
  num_minibatches_processed_++;
  return true;
}


bool NnetChainaTopTrainer::PrintTotalStats() const {
  bool ans = false;
  if (output_si_objf_.PrintTotalStats(lang_name_ + ":output-si"))
    ans = true;
  if (output_objf_.PrintTotalStats(lang_name_ + ":output"))
    ans = true;
  if (output_si_xent_objf_.PrintTotalStats(lang_name_ + ":output-si-xent"))
    ans = true;
  if (output_xent_objf_.PrintTotalStats(lang_name_ + ":output-xent"))
    ans = true;
  KALDI_LOG << "Speaker-independent max-change stats for language "
            << lang_name_ << ":";
  max_change_stats_si_.Print(*nnet_);
  KALDI_LOG << "Speaker-dependent max-change stats for language "
            << lang_name_ << ":";
  max_change_stats_.Print(*nnet_);
  return ans;
}


NnetComputer* NnetChainaBottomTrainer::Forward(
    int32 num_sequences,
    int32 first_input_t,
    int32 first_output_t,
    int32 frames_per_sequence_out,
    CuMatrix<BaseFloat> *input,
    CuMatrix<BaseFloat> *output) {
  KALDI_ASSERT(input->NumRows() != 0 && input->NumRows() % num_sequences == 0);
  int32 frames_per_sequence_in = input->NumRows() / num_sequences;
  ComputationStructure s(opts_.train_bottom_nnet,
                         num_sequences,
                         frames_per_sequence_in,
                         frames_per_sequence_out,
                         first_input_t, first_output_t);
  // Note: this will be cached in the unordered_map owned by this class, so we
  // don't have to worry about it being deleted before we're done with the
  // NnetComputer object.
  std::shared_ptr<const NnetComputation> computation = GetComputation(s);

  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  NnetComputer *computer = new NnetComputer(nnet_config.compute_config,
                                            *computation, nnet_, delta_nnet_);
  computer->AcceptInput("input", input);
  computer->Run();
  computer->GetOutputDestructive("output", output);
  return computer;
}


void NnetChainaBottomTrainer::Backward(NnetComputer *computer,
                                       CuMatrix<BaseFloat> *output_deriv) {
  computer->AcceptInput("output", output_deriv);
  computer->Run();

  const NnetTrainerOptions &nnet_config = opts_.nnet_config;

  // we may later provide a way to set a different max-change for the bottom
  // nnet than on the top nnet.
  bool success = UpdateNnetWithMaxChange(
      *delta_nnet_,
      nnet_config.max_param_change,
      1.0,
      1.0 - nnet_config.momentum,  // normally momentum is 0.0.
      nnet_,
      &max_change_stats_);

  // Scale down the batchnorm stats (keeps them fresh... this affects what
  // happens when, later on, we use the model with batchnorm test-mode set).
  ScaleBatchnormStats(nnet_config.batchnorm_stats_scale, nnet_);

  // The following will only do something if we have a LinearComponent
  // or AffineComponent with orthonormal-constraint set to a nonzero value.
  ConstrainOrthonormal(nnet_);

  if (!success)
    ScaleNnet(nnet_config.momentum, delta_nnet_);
  else
    ScaleNnet(0.0, delta_nnet_);

  num_minibatches_processed_++;
}


NnetChainaBottomTrainer::NnetChainaBottomTrainer(
    const NnetChainaTrainingOptions &opts,
    Nnet *nnet):
    opts_(opts),
    nnet_(nnet),
    delta_nnet_(nnet->Copy()),
    compiler_(*nnet, opts_.nnet_config.optimize_config,
              opts_.nnet_config.compiler_config),
    max_change_stats_(*nnet) {
  if (opts_.nnet_config.zero_component_stats)
    ZeroComponentStats(nnet);
  ScaleNnet(0.0, delta_nnet_);
  if (opts_.nnet_config.read_cache != "") {
    // It would be complicated to implement, as there are various top nnets
    // and they would all try to read and write the same cache files.
    // To implement this, the best way would be to
    KALDI_WARN << "The read-cache options are not currently supported.";
  }
  KALDI_ASSERT(opts_.nnet_config.momentum >= 0.0 &&
               opts_.nnet_config.max_param_change >= 0.0 &&
               opts_.bottom_subsampling_factor >= 1);
}

std::shared_ptr<const NnetComputation> NnetChainaBottomTrainer::GetComputation(
    const ComputationStructure &s) {
  {
    auto iter = computation_map_.find(s);
    if (iter != computation_map_.end())
      return iter->second;
  }
  int32 num_sequences = s.num_sequences,
      frames_per_sequence_in = s.frames_per_sequence_in,
      frames_per_sequence_out = s.frames_per_sequence_out,
      first_input_t = s.first_input_t,
      first_output_t = s.first_output_t;

  ComputationRequest request;
  request.need_model_derivative = opts_.train_bottom_nnet;
  request.store_component_stats = true;
  request.inputs.resize(1);
  request.inputs[0].name = "input";
  request.inputs[0].indexes.resize(frames_per_sequence_in * num_sequences);
  // The inputs are in the order: all frames of sequence 0; then all frames of
  // sequence 1; and so on.  This is how the example-merging code does it, since
  // it's more convenient when dealing with compressed matrices and the like.
  auto iter = request.inputs[0].indexes.begin();
  for (int32 n = 0; n < num_sequences; n++) {
    for (int32 t = first_input_t;
         t < first_input_t + frames_per_sequence_in; ++t,++iter) {
      iter->n = n;
      iter->t = t;
    }
  }
  // ... but the outputs are in the order: the first frame of all sequences;
  // the second frame of all sequences; and so on.
  request.outputs.resize(1);
  request.outputs[0].name = "output";
  request.outputs[1].has_deriv = opts_.train_bottom_nnet;
  request.outputs[0].indexes.resize(frames_per_sequence_out * num_sequences);
  int32 t_stride_out = opts_.bottom_subsampling_factor;
  iter = request.outputs[0].indexes.begin();
  for (int32 t = first_output_t;
       t < first_output_t  +  frames_per_sequence_out * t_stride_out;
       t += t_stride_out) {
    for (int32 n = 0; n < num_sequences; ++n,++iter) {
      iter->n = n;
      iter->t = t;
    }
  }
  std::shared_ptr<const NnetComputation> computation = compiler_.Compile(
      request);
  computation_map_[s] = computation;
  return computation;
}


bool NnetChainaTrainer::PrintTotalStats() const {
  bottom_trainer_.PrintTotalStats();
  bool ans = false;
  for (auto iter = top_trainers_.begin(); iter != top_trainers_.end();
       ++iter)
    if (iter->second->PrintTotalStats())
      ans = true;
  return ans;
}

NnetChainaTrainer::NnetChainaTrainer(
    const NnetChainaTrainingOptions &config,
    NnetChainaModels *models):
    opts_(config),
    models_(models),
    bottom_trainer_(opts_, models->GetBottomNnet()) {
}


void NnetChainaTrainer::Train(const NnetChainExample &eg) {
  // TODO.  work out structure, etc.
}

NnetChainaTrainer::~NnetChainaTrainer() {
  for (auto iter = top_trainers_.begin(); iter != top_trainers_.end();
       ++iter)
    delete iter->second;
}


} // namespace nnet3
} // namespace kaldi
