// nnet3/nnet-chaina-training.cc

// Copyright      2018    Johns Hopkins University (author: Daniel Povey)

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

NnetChainaModels::NnetChainaModels(
    bool zero_component_stats,
    bool bottom_model_test_mode,
    bool top_model_test_mode,
    const std::string &model_dir,
    const std::string &den_fst_dir,
    const std::string &transform_dir):
    zero_component_stats_(zero_component_stats),
    bottom_model_test_mode_(bottom_model_test_mode),
    top_model_test_mode_(top_model_test_mode),
    model_dir_(model_dir),
    den_fst_dir_(den_fst_dir),
    transform_dir_(transform_dir) {
  std::string bottom_nnet_name; // model_dir/bottom.raw
  GetPathname(model_dir, "bottom", "raw", &bottom_nnet_name);
  ReadKaldiObject(bottom_nnet_name, &bottom_nnet_);
  if (zero_component_stats_ && !bottom_model_test_mode_)
    ZeroComponentStats(&bottom_nnet_);
  ComputeSimpleNnetContext(bottom_nnet_,
                           &bottom_nnet_left_context_,
                           &bottom_nnet_right_context_);
  if (bottom_model_test_mode_) {
    SetBatchnormTestMode(true, &bottom_nnet_);
    SetDropoutTestMode(true, &bottom_nnet_);
    // The following is for efficiency in evaluating the bottom nnet,
    // it may combine certain component types.
    CollapseModel(CollapseModelConfig(), &bottom_nnet_);
  }
}

void NnetChainaModels::GetPathname(const std::string &dir,
                                   const std::string &name,
                                   const std::string &suffix,
                                   std::string *pathname) {
  std::ostringstream str;
  str << dir << '/' << name << '.' << suffix;
  *pathname = str.str();
}

void NnetChainaModels::GetPathname(const std::string &dir,
                                   const std::string &name,
                                   int32 job_id,
                                   const std::string &suffix,
                                   std::string *pathname) {
  std::ostringstream str;
  str << dir << '/' << name << '.' << job_id << '.' << suffix;
  *pathname = str.str();
}

NnetChainaModels::LanguageInfo *NnetChainaModels::GetInfoForLang(
    const std::string &lang) {
  auto iter = lang_info_.find(lang);
  if (iter != lang_info_.end()) {
    return iter->second;
  } else {
    LanguageInfo *info = new LanguageInfo();

    std::string model_filename, den_fst_filename, transform_filename;
    GetPathname(model_dir_, lang, "mdl", &model_filename);
    GetPathname(den_fst_dir_, lang, "fst", &den_fst_filename);
    GetPathname(transform_dir_, lang, "ada", &transform_filename);

    {
      bool binary;
      Input ki(model_filename, &binary);
      info->trans_model.Read(ki.Stream(), binary);
      info->am_nnet.Read(ki.Stream(), binary);
      if (zero_component_stats_ && !top_model_test_mode_) {
        ZeroComponentStats(&(info->am_nnet.GetNnet()));
      }
      if (top_model_test_mode_) {
        Nnet &nnet = info->am_nnet.GetNnet();
        SetBatchnormTestMode(true, &nnet);
        SetDropoutTestMode(true, &nnet);
        // The following is for efficiency in evaluating the top nnet,
        // it may combine certain component types.
        CollapseModel(CollapseModelConfig(), &bottom_nnet_);
      }
    }
    ReadFstKaldi(den_fst_filename, &(info->den_fst));
    ReadKaldiObject(transform_filename, &(info->transform));
    lang_info_[lang] = info;
    return info;
  }
}

Nnet* NnetChainaModels::GetBottomNnet() {
  return &bottom_nnet_;
}


AmNnetSimple* NnetChainaModels::GetNnetForLang(
    const std::string &language_name) {
  LanguageInfo *info = GetInfoForLang(language_name);
  return &(info->am_nnet);
}

TransitionModel* NnetChainaModels::GetTransitionModelForLang(
    const std::string &language_name) {
  LanguageInfo *info = GetInfoForLang(language_name);
  return &(info->trans_model);
}

fst::StdVectorFst* NnetChainaModels::GetDenFstForLang(
       const std::string &language_name) {
  LanguageInfo *info = GetInfoForLang(language_name);
  return &(info->den_fst);
}

Nnet* NnetChainaModels::GetRawNnetForLang(
       const std::string &language_name) {
  LanguageInfo *info = GetInfoForLang(language_name);
  return &(info->am_nnet.GetNnet());
}

differentiable_transform::DifferentiableTransformMapped*
NnetChainaModels::GetTransformForLang(
    const std::string &language_name) {
  LanguageInfo *info = GetInfoForLang(language_name);
  return &(info->transform);
}



void NnetChainaModels::WriteRawModels(const std::string &model_out_dir,
                                      bool binary,
                                      int32 job_id) {
  if (!bottom_model_test_mode_) {
    std::string bottom_model_name;
    GetPathname(model_out_dir, "bottom", job_id, "raw", &bottom_model_name);
    WriteKaldiObject(bottom_nnet_, bottom_model_name, binary);
  }
  std::ostringstream lang_names_ss;
  for (auto iter = lang_info_.begin(); iter != lang_info_.end(); ++iter) {
    const std::string &lang_name = iter->first;
    lang_names_ss << lang_name << " ";
    LanguageInfo *info = iter->second;
    {
      // we write it as a 'raw' model without the TransitionModel or
      // the AmNnetSimple wrapper, since we can reconstruct those parts
      // from the previous iter's model.
      std::string top_model_name;
      GetPathname(model_out_dir, lang_name, job_id, "raw", &top_model_name);
      WriteKaldiObject(info->am_nnet.GetNnet(), top_model_name, binary);
    }
  }
  KALDI_LOG << "Wrote " << (bottom_model_test_mode_ ? "" : " bottom nnet and ")
            << "nnets for languages " << lang_names_ss.str() << "to "
            << model_out_dir;
}


NnetChainaModels::~NnetChainaModels() {
  for (auto iter = lang_info_.begin(); iter != lang_info_.end(); ++iter)
    delete iter->second;
}

NnetChainaTopTrainer::NnetChainaTopTrainer(
    const std::string &lang_name,
    const NnetChainaTrainingOptions &config,
    const fst::StdVectorFst &den_fst,
    const differentiable_transform::DifferentiableTransformMapped &transform,
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
    KALDI_WARN << "The read-cache options are not currently supported here.";
  }
  KALDI_ASSERT(opts_.nnet_config.momentum >= 0.0);
}


NnetChainaTopTrainer::ComputationStructure::ComputationStructure(
    bool adapted,
    bool train_model,
    bool need_input_deriv,
    int32 num_sequences,
    int32 frames_per_sequence_in,
    int32 frames_per_sequence_out,
    int32 first_input_t,
    int32 top_subsampling_factor):
    adapted(adapted), train_model(train_model),
    need_input_deriv(need_input_deriv), num_sequences(num_sequences),
    frames_per_sequence_in(frames_per_sequence_in),
    frames_per_sequence_out(frames_per_sequence_out),
    first_input_t(first_input_t),
    top_subsampling_factor(top_subsampling_factor) { }


void NnetChainaTopTrainer::ConsolidateMemory() {
  ::kaldi::nnet3::ConsolidateMemory(nnet_);
  ::kaldi::nnet3::ConsolidateMemory(delta_nnet_);
}


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

  if (nnet_->InputDim("input") < 0 ||
      nnet_->OutputDim("output") < 0 ||
      nnet_->OutputDim("output-si") < 0 ||
      nnet_->OutputDim("output-xent") < 0 ||
      nnet_->OutputDim("output-si-xent") < 0) {
    KALDI_ERR << "Top neural net for chaina training must have an input called "
        "'input' and outputs called 'output', 'output-xent', 'output-si', and "
        "'output-si-xent'.";
  }

  ComputationRequest request;
  request.need_model_derivative = s.train_model;
  request.store_component_stats = !opts_.top_model_test_mode;
  request.inputs.resize(1);
  request.inputs[0].name = "input";
  request.inputs[0].indexes.resize(frames_per_sequence_in * num_sequences);
  request.inputs[0].has_deriv = s.need_input_deriv;
  // The inputs are in the order: the first frame of all sequences; the second
  // frame of all sequences; and so on.
  auto iter = request.inputs[0].indexes.begin();
  for (int32 t = first_input_t;
       t < first_input_t + frames_per_sequence_in; ++t) {
    for (int32 n = 0; n < num_sequences; ++n,++iter) {
      iter->n = n;
      iter->t = t;
      // the x values will already be 0, thanks to the default constructor of
      // Index().
    }
  }
  // The outputs are also in the order: the first frame of all sequences;
  // the second frame of all sequences; and so on.
  request.outputs.resize(2);
  request.outputs[0].name = (s.adapted ? "output" : "output-si");
  request.outputs[0].has_deriv = !opts_.top_model_test_mode;
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
  request.outputs[1].has_deriv = !opts_.top_model_test_mode;
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
    BaseFloat model_training_scale,
    const CuVectorBase<BaseFloat> &deriv_weights,
    Posterior *posterior,
    CuMatrix<BaseFloat> *input_deriv) {

  const NnetTrainerOptions &nnet_config = opts_.nnet_config;

  // note: because we give the 1st arg (nnet_) as a pointer to the
  // constructor of 'computer', it will use that copy of the nnet to
  // store stats.
  NnetComputer computer(nnet_config.compute_config, computation,
                        nnet_, delta_nnet_);

  // Give the inputs to the computer object.
  CuMatrix<BaseFloat> input_copy(input);
  computer.AcceptInput("input", &input_copy);
  // Do the forward propagation.
  computer.Run();

  const CuMatrixBase<BaseFloat>
      &output = computer.GetOutput("output-si"),
      &output_xent = computer.GetOutput("output-si-xent");
  // It's not optimal that we compute these derivatives even when we're not
  // training, but the 'compute-prob' phase doesn't dominate.
  CuMatrix<BaseFloat> output_deriv(output.NumRows(),
                                   output.NumCols(),
                                   kUndefined),
      output_xent_deriv;

  // Note: we normally turn the chain l2 regularization (which is l2 on the
  // output of the nnet) off now, since parameter-level l2 regularization seems
  // to work better.  So expect 'tot_l2_term' to be zero.
  BaseFloat tot_objf, tot_l2_term, tot_weight;

  ComputeChainObjfAndDeriv(opts_.chain_config, den_graph_,
                           supervision, output,
                           &tot_objf, &tot_l2_term, &tot_weight,
                           &output_deriv, &output_xent_deriv,
                           posterior);

  if (!(tot_objf - tot_objf == 0.0)) {
    // A NaN or inf was encountered in the objective computation.
    // The input_deriv won't be used, so no need to set it.
    // Un-freeze the natural gradient and return.
    return false;
  }

  {
    // this block computes and keeps track of the cross-entropy objective.
    // at this point, xent_deriv is posteriors derived from the numerator
    // computation.  note, xent_objf has a factor of '.supervision.weight',
    // which is also included in 'tot_weight'.
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

  output_si_objf_.UpdateStats(lang_name_ + ":output-si",
                              opts_.nnet_config.print_interval,
                              num_minibatches_processed_,
                              tot_weight, tot_objf, tot_l2_term);

  if (input_deriv == NULL && model_training_scale == 0.0)
    return true;

  // Freeze the natural gradient.  We don't want to update the NG scatter
  // matrices on this data because we'll next be running the same nnet on the
  // speaker-adapted version of the same data, and it would violate the
  // independence assumptions needed for NG to work if we updated them.
  if (model_training_scale != 0.0)
    FreezeNaturalGradient(true, delta_nnet_);

  computer.AcceptInput("output-si", &output_deriv);

  output_xent_deriv.Scale(opts_.chain_config.xent_regularize);
  computer.AcceptInput("output-si-xent", &output_xent_deriv);

  // Do the backprop.
  computer.Run();

  if (input_deriv != NULL)
    computer.GetOutputDestructive("input", input_deriv);

  static bool warned_momentum = false;
  if (model_training_scale != 1.0 &&
      nnet_config.momentum != 0.0 && !warned_momentum) {
    KALDI_WARN << "Momentum does not interact correctly with top_weight or "
        "bottom_weight values.  Will not warn again.";
    warned_momentum = true;
  }

  if (model_training_scale != 0.0) {
    // If we're actually training the top model...

    // Update the parameters of nnet.
    // Note: normally momentum is 0.0.
    bool success = UpdateNnetWithMaxChange(
        *delta_nnet_,
        nnet_config.max_param_change,
        1.0,
        model_training_scale * (1.0 - nnet_config.momentum),
        nnet_, &max_change_stats_si_);

    // Un-freeze the natural gradient.
    FreezeNaturalGradient(false, delta_nnet_);

    if (success)
      ScaleNnet(nnet_config.momentum, delta_nnet_);
    else
      ScaleNnet(0.0, delta_nnet_);
    return success;
  } else {
    return true;
  }
}

bool NnetChainaTopTrainer::TrainAdapted(
    const NnetComputation &computation,
    const chain::Supervision &supervision,
    BaseFloat model_training_scale,
    const CuVectorBase<BaseFloat> &deriv_weights,
    CuMatrix<BaseFloat> *input,
    CuMatrix<BaseFloat> *input_deriv) {

  const NnetTrainerOptions &nnet_config = opts_.nnet_config;

  // note: because we give the 1st arg (nnet_) as a pointer to the
  // constructor of 'computer', it will use that copy of the nnet to
  // store stats.
  NnetComputer computer(nnet_config.compute_config, computation,
                        nnet_, delta_nnet_);

  // give the input to the computer object.
  computer.AcceptInput("input", input);
  // Do the forward computation
  computer.Run();

  const CuMatrixBase<BaseFloat>
      &output = computer.GetOutput("output"),
      &output_xent = computer.GetOutput("output-xent");
  CuMatrix<BaseFloat> output_deriv(output.NumRows(),
                                   output.NumCols(),
                                   kUndefined),
      output_xent_deriv;

  // Note: we don't normally use the l2 term any more; parameter-level
  // regularization seems to work better than regularization of the
  // nnet output.
  BaseFloat tot_objf, tot_l2_term, tot_weight;

  ComputeChainObjfAndDeriv(opts_.chain_config, den_graph_,
                           supervision, output,
                           &tot_objf, &tot_l2_term, &tot_weight,
                           &output_deriv, &output_xent_deriv);

  if (!(tot_objf - tot_objf == 0.0)) {
    // A NaN or inf was encountered in the objective computation.  the input_deriv
    // won't be used by the calling code, so no need to set it.
    return false;
  }

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
  output_objf_.UpdateStats(lang_name_ + ":output",
                           opts_.nnet_config.print_interval,
                           num_minibatches_processed_,
                           tot_weight, tot_objf, tot_l2_term);

  if (input_deriv == NULL && model_training_scale == 0.0)
    return true;

  if (opts_.apply_deriv_weights && deriv_weights.Dim() != 0) {
    output_deriv.MulRowsVec(deriv_weights);
    output_xent_deriv.MulRowsVec(deriv_weights);
  }

  computer.AcceptInput("output", &output_deriv);
  output_xent_deriv.Scale(opts_.chain_config.xent_regularize);
  computer.AcceptInput("output-xent", &output_xent_deriv);

  // Do the backprop.
  computer.Run();

  if (input_deriv != NULL)
    computer.GetOutputDestructive("input", input_deriv);

  if (model_training_scale != 0.0) {
    // If we're actually training the top model...

    // Update the parameters of nnet.
    // Note: normally, momentum is 0.0.
    bool success = UpdateNnetWithMaxChange(
        *delta_nnet_,
        nnet_config.max_param_change,
        1.0,
        model_training_scale * (1.0 - nnet_config.momentum),
        nnet_, &max_change_stats_);

    // Scale down the batchnorm stats (keeps them fresh... this affects what
    // happens when, later on, we use the model with batchnorm test-mode set).
    ScaleBatchnormStats(nnet_config.batchnorm_stats_scale, nnet_);

    // The following will only do something if we have a LinearComponent
    // or AffineComponent with orthonormal-constraint set to a nonzero value.
    ConstrainOrthonormal(nnet_);

    if (success)
      ScaleNnet(nnet_config.momentum, delta_nnet_);
    else
      ScaleNnet(0.0, delta_nnet_);
    return success;
  } else {
    return true;
  }
}


bool NnetChainaTopTrainer::Train(const CuMatrixBase<BaseFloat> &input,
                                 int32 num_sequences,
                                 int32 num_spk,
                                 int32 first_input_t,
                                 int32 top_subsampling_factor,
                                 const VectorBase<BaseFloat> &deriv_weights_in,
                                 const chain::Supervision &supervision,
                                 BaseFloat model_training_scale,
                                 CuMatrix<BaseFloat> *input_deriv) {
  KALDI_ASSERT(input.NumRows() != 0 && input.NumRows() % num_sequences != 0);
  int32 frames_per_sequence_in = input.NumRows() / num_sequences,
      frames_per_sequence_out = supervision.frames_per_sequence;

  bool adapted = false;
  ComputationStructure structure(
      adapted, (model_training_scale != 0.0), (input_deriv != NULL),
      num_sequences, frames_per_sequence_in, frames_per_sequence_out,
      first_input_t, top_subsampling_factor);

  // Will be the numerator posterior from the unadapted pass, which will be
  // padded with l/r context and used to estimate the adapted features.
  Posterior post;

  CuVector<BaseFloat> deriv_weights;
  if (opts_.apply_deriv_weights)
    deriv_weights = deriv_weights_in;

  std::shared_ptr<const NnetComputation> computation_unadapted =
      GetComputation(structure);
  bool success = TrainUnadapted(
      input, *computation_unadapted, supervision,
      model_training_scale * opts_.unadapted_top_weight,
      deriv_weights, &post, input_deriv);

  if (!success) {
    num_minibatches_processed_++;
    return false;
  }

  if (input_deriv) {
    // Apply the scale from --unadapted-bottom-weight.  We'll supply the other
    // factor that comes from from the language-specific bottom_weight ("bw")
    // ito UpdateNnetWithMaxChange() later on when we train the bottom nnet.
    input_deriv->Scale(opts_.unadapted_bottom_weight);
  }

  Posterior post_padded(input.NumRows());
  ConvertPosterior(post, num_sequences, first_input_t,
                   top_subsampling_factor,
                   transform_.pdf_map,
                   transform_.transform->NumClasses(),
                   &post_padded);

  structure.adapted = true;
  std::shared_ptr<const NnetComputation> computation_adapted =
      GetComputation(structure);

  CuMatrix<BaseFloat> adapted_input(input.NumRows(), input.NumCols(),
                                    kUndefined),
      adapted_input_deriv;

  using namespace differentiable_transform;
  MinibatchInfoItf *minibatch_info = transform_.transform->TrainingForward(
      input, num_sequences, num_spk, post_padded, &adapted_input);

  success = TrainAdapted(
      *computation_adapted, supervision,
      model_training_scale, deriv_weights,
      &adapted_input, &adapted_input_deriv);

  num_minibatches_processed_++;
  if (!success)
    return false;

  if (input_deriv == NULL)
    delete minibatch_info;
  else
    transform_.transform->TrainingBackward(input, adapted_input_deriv,
                                           num_sequences, num_spk, post_padded,
                                           minibatch_info, input_deriv);
  return true;
}


/**
   This helper function for ConvertPosterior() converts from pdf-ids to
   cluster-ids using the map provided in pdf_map, if it is nonempty.
   If pdf_map is empty, it just copies the pairs over unchanged.
 */
static inline void ConvertPosteriorElement(
    const std::vector<int32> &pdf_map,
    int32 num_classes,
    const std::vector<std::pair<int32, BaseFloat> > &post_elem_in,
    std::vector<std::pair<int32, BaseFloat> > *post_elem_out) {
  if (pdf_map.empty()) {
    *post_elem_out = post_elem_in;
    if (!post_elem_in.empty()) {
      // We just check the first int32-- this is a spot-check that the
      // pdf-ids are in the correct range.
      KALDI_ASSERT(post_elem_in[0].first < num_classes);
    }
  } else {
    int32 num_classes_in = pdf_map.size();
    size_t num_pairs = post_elem_in.size();
    post_elem_out->resize(num_pairs);
    for (size_t i =0; i < num_pairs; i++) {
      int32 pdf_id = post_elem_in[i].first;
      BaseFloat weight = post_elem_in[i].second;
      KALDI_ASSERT(pdf_id < num_classes_in);
      int32 cluster_id = pdf_map[pdf_id];
      KALDI_ASSERT(cluster_id < num_classes);
      (*post_elem_out)[i].first = cluster_id;
      (*post_elem_out)[i].second = weight;
    }
  }
}

void NnetChainaTopTrainer::ConvertPosterior(
    const Posterior &post_at_output,
    int32 num_sequences,
    int32 first_input_t,
    int32 top_subsampling_factor,
    const std::vector<int32> &pdf_map,
    int32 num_classes,
    Posterior *post_at_input) {
  int32 output_post_size = post_at_output.size(),
      input_post_size = post_at_input->size(),
      s = top_subsampling_factor;
  KALDI_ASSERT(input_post_size % num_sequences == 0 &&
               output_post_size % num_sequences == 0 &&
               input_post_size >= output_post_size * top_subsampling_factor &&
               top_subsampling_factor > 0);
  int32 num_frames_out = output_post_size / num_sequences,
      num_frames_in = input_post_size / num_sequences,
      last_input_t = first_input_t + (num_frames_in - 1),
      first_output_t = 0,
      last_output_t = first_output_t + s * (num_frames_out - 1);

  int32 half_s = s / 2;  // note: this will round down, which is intended.

  for (int32 t_in = first_input_t; t_in <= last_input_t; t_in++) {
    // find the corresponding output frame by rounding t to the closest
    // t that's a multiple of top_subsampling_factor (rounding down in
    // case of ties).  We do this by adding half_s and rounding down.
    int32 t_out = s * DivideRoundingDown(t_in + half_s, s);
    if (t_out >= first_output_t && t_out <= last_output_t) {
      for (int32 n = 0; n < num_sequences; n++) {
        int32 input_index = num_sequences * (t_in - first_input_t) + n,
            output_index = num_sequences * ((t_out - first_output_t) / s) + n;
        ConvertPosteriorElement(pdf_map, num_classes,
                                post_at_output[output_index],
                                &((*post_at_input)[input_index]));
      }
    }
    // else just leave the input posterior for this frame empty.  This will
    // happen for most of the frames that were added for left and right context.
  }
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


NnetChainaTopTrainer::~NnetChainaTopTrainer() {
  delete delta_nnet_;
}

void NnetChainaBottomTrainer::ConsolidateMemory() {
  ::kaldi::nnet3::ConsolidateMemory(nnet_);
  ::kaldi::nnet3::ConsolidateMemory(delta_nnet_);
}

NnetComputer* NnetChainaBottomTrainer::Forward(
    int32 num_sequences,
    int32 first_input_t,
    int32 first_output_t,
    int32 frames_per_sequence_out,
    bool train_model,
    CuMatrix<BaseFloat> *input,
    CuMatrix<BaseFloat> *output) {
  KALDI_ASSERT(input->NumRows() != 0 && input->NumRows() % num_sequences == 0);
  int32 frames_per_sequence_in = input->NumRows() / num_sequences;
  ComputationStructure s(train_model,
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
  if (!train_model) {
    delete computer;
    return NULL;
  } else {
    return computer;
  }
}


void NnetChainaBottomTrainer::Backward(BaseFloat model_training_scale,
                                       NnetComputer *computer,
                                       CuMatrix<BaseFloat> *output_deriv) {
  // if model_training_scale was 0.0, this function should not have been called.
  KALDI_ASSERT(model_training_scale > 0.0);
  computer->AcceptInput("output", output_deriv);
  computer->Run();

  delete computer;

  const NnetTrainerOptions &nnet_config = opts_.nnet_config;

  // we may later provide a way to set a different max-change for the bottom
  // nnet than on the top nnet.
  // Note: normally, momentum is 0.0.
  bool success = UpdateNnetWithMaxChange(
      *delta_nnet_,
      nnet_config.max_param_change,
      1.0,
      model_training_scale * (1.0 - nnet_config.momentum),
      nnet_,
      &max_change_stats_);

  // Scale down the batchnorm stats (keeps them fresh... this affects what
  // happens when, later on, we use the model with batchnorm test-mode set).
  ScaleBatchnormStats(nnet_config.batchnorm_stats_scale, nnet_);

  // The following will only do something if we have a LinearComponent
  // or AffineComponent with orthonormal-constraint set to a nonzero value.
  ConstrainOrthonormal(nnet_);

  if (success)
    ScaleNnet(nnet_config.momentum, delta_nnet_);
  else
    ScaleNnet(0.0, delta_nnet_);

  static bool warned_momentum = false;
  if (model_training_scale != 1.0 && nnet_config.momentum != 0.0 &&
      !warned_momentum) {
    KALDI_WARN << "Momentum does not interact correctly with top_weight or "
        "bottom_weight values.  Will not warn again.";
    warned_momentum = true;
  }
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
  { // Check in the cache, in case we already handled this computation.
    auto iter = computation_map_.find(s);
    if (iter != computation_map_.end())
      return iter->second;
  }

  if (opts_.bottom_model_test_mode) {
    KALDI_ASSERT(!s.train_model);
  }

  int32 num_sequences = s.num_sequences,
      frames_per_sequence_in = s.frames_per_sequence_in,
      frames_per_sequence_out = s.frames_per_sequence_out,
      first_input_t = s.first_input_t,
      first_output_t = s.first_output_t;

  if (nnet_->InputDim("input") < 0 ||
      nnet_->OutputDim("output") < 0) {
    KALDI_ERR << "Bottom neural net for chaina training must have an input "
        "called 'input' and an output called 'output'.";
  }

  ComputationRequest request;
  request.need_model_derivative = s.train_model;
  // If the user supplied the option --train-bottom-model false, then we
  // are using test-mode for the batch-norm on the bottom model, and we
  // don't want to overwrite the batch-norm stats.
  request.store_component_stats = !opts_.bottom_model_test_mode;
  request.inputs.resize(1);
  request.inputs[0].name = "input";
  request.inputs[0].indexes.resize(frames_per_sequence_in * num_sequences);
  // The inputs are in the order: all frames of sequence 0; then all frames of
  // sequence 1; and so on.  This is how the example-merging code does it, since
  // it's more convenient when dealing with compressed matrices.
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
  request.outputs[0].has_deriv = s.train_model;
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

void NnetChainaBottomTrainer::PrintTotalStats() const {
  KALDI_LOG << "Max-change stats for bottom nnet:";
  max_change_stats_.Print(*nnet_);
}
NnetChainaBottomTrainer::~NnetChainaBottomTrainer() {
  delete delta_nnet_;
}


void NnetChainaTrainer::GetContextInfo(
    const std::string &lang,
    int32 *bottom_left_context,
    int32 *bottom_right_context,
    int32 *top_left_context,
    int32 *top_right_context) {

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
  ComputeSimpleNnetContext(*models->GetBottomNnet(),
                           &bottom_left_context_,
                           &bottom_right_context_);
}


NnetChainaTopTrainer* NnetChainaTrainer::GetTopTrainerForLang(
    const std::string &lang) {
  auto iter = top_trainers_.find(lang);
  if (iter != top_trainers_.end())
    return iter->second;
  NnetChainaTopTrainer *ans =
      new NnetChainaTopTrainer(
          lang, opts_,
          *(models_->GetDenFstForLang(lang)),
          *(models_->GetTransformForLang(lang)),
          models_->GetRawNnetForLang(lang));
  top_trainers_[lang] = ans;
  return ans;
}

// 'key' might be something like "afsdadsfds12345?lang=english&tw=1.0&bw=0.5"
// expressing how much we want this eg to be used to train the top, and bottom,
// models respectively.
void NnetChainaTrainer::Train(const std::string &key,
                              const NnetChainExample &eg) {
  size_t num_top_trainers = top_trainers_.size();
  std::string lang_name = "default";
  // 'top_weight' is a weight on the derivatives and max-change
  // when training the top model, 'bottom_weight' is the same
  // for the bottom model.
  BaseFloat top_weight = 1.0,
      bottom_weight = 1.0;
  ParseFromQueryString(key, "lang", &lang_name);
  ParseFromQueryString(key, "tw", &top_weight);
  ParseFromQueryString(key, "bw", &bottom_weight);
  KALDI_ASSERT(top_weight >= 0.0 && bottom_weight >= 0.0);

  if (opts_.bottom_model_test_mode)
    bottom_weight = 0.0;
  if (opts_.top_model_test_mode)
    top_weight = 0.0;

  int32 num_sequences, chunks_per_spk, first_input_t,
      num_input_frames, num_output_frames,
      frame_subsampling_factor,
      eg_left_context, eg_right_context;
  FindChainaExampleStructure(eg, &num_sequences, &chunks_per_spk,
                             &first_input_t,
                             &num_input_frames, &num_output_frames,
                             &frame_subsampling_factor,
                             &eg_left_context, &eg_right_context);
  KALDI_ASSERT(chunks_per_spk % num_sequences == 0);
  int32 num_spk = num_sequences / chunks_per_spk;

  AmNnetSimple *top_am_nnet = models_->GetNnetForLang(lang_name);
  int32 top_left_context = top_am_nnet->LeftContext(),
      top_right_context = top_am_nnet->RightContext();

  int32 first_embedding_t,
      num_embedding_frames;
  ComputeEmbeddingTimes(first_input_t, num_input_frames, num_output_frames,
                        frame_subsampling_factor,
                        opts_.bottom_subsampling_factor,
                        bottom_left_context_, bottom_right_context_,
                        top_left_context, top_right_context,
                        opts_.keep_embedding_context,
                        &first_embedding_t, &num_embedding_frames);

  const GeneralMatrix &eg_input = eg.inputs[0].features;
  CuMatrix<BaseFloat> cu_input(eg_input.NumRows(), eg_input.NumCols(),
                               kUndefined),
      cu_embedding;
  eg_input.CopyToMat(&cu_input);
  bool train_bottom_nnet = bottom_weight != 1.0;
  KALDI_ASSERT(cu_input.NumRows() == num_input_frames * num_sequences);

  NnetComputer *computer = bottom_trainer_.Forward(
      num_sequences, first_input_t,
      first_embedding_t, num_embedding_frames,
      train_bottom_nnet,
      &cu_input, &cu_embedding);

  int32 b = opts_.bottom_subsampling_factor,
      first_embedding_t_subsampled = first_embedding_t / b,
      top_subsampling_factor = frame_subsampling_factor / b;

  NnetChainaTopTrainer *top_trainer = GetTopTrainerForLang(lang_name);

  CuMatrix<BaseFloat> cu_embedding_deriv;
  if (train_bottom_nnet)
    cu_embedding_deriv.Resize(cu_embedding.NumRows(), cu_embedding.NumCols());


  bool success = top_trainer->Train(cu_embedding, num_sequences,
                                    num_spk,
                                    first_embedding_t_subsampled,
                                    top_subsampling_factor,
                                    eg.outputs[0].deriv_weights,
                                    eg.outputs[0].supervision,
                                    top_weight,
                                    (train_bottom_nnet ?
                                     &cu_embedding_deriv : NULL));

  if (success && train_bottom_nnet) {
    bottom_trainer_.Backward(bottom_weight, computer,
                             &cu_embedding_deriv);
  } else {
    delete computer;  // if it's NULL, this will do nothing.
  }

  if (top_trainers_.size() != num_top_trainers) {
    // Move any permanently held bits of GPU memory to low addresses, to reduce
    // fragmentation.
    bottom_trainer_.ConsolidateMemory();
    top_trainer->ConsolidateMemory();
  }

}


NnetChainaTrainer::~NnetChainaTrainer() {
  for (auto iter = top_trainers_.begin(); iter != top_trainers_.end();
       ++iter)
    delete iter->second;
}



} // namespace nnet3
} // namespace kaldi
