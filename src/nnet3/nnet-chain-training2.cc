// nnet3/nnet-chain-training.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)
//                2016    Xiaohui Zhang
//                2019    Idiap Research Institute (author: Srikanth Madikeri)

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

#include "nnet3/nnet-chain-training2.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetChainTrainer2::NnetChainTrainer2(const NnetChainTraining2Options &opts,
                                   const NnetChainModel2 &model, 
                                   Nnet *nnet):
    opts_(opts),
    model_(model),
    nnet_(nnet),
    compiler_(*nnet, opts_.nnet_config.optimize_config,
              opts_.nnet_config.compiler_config),
    num_minibatches_processed_(0),
    max_change_stats_(*nnet),
    srand_seed_(RandInt(0, 100000)) {

  if (opts.nnet_config.zero_component_stats)
    ZeroComponentStats(nnet);
  KALDI_ASSERT(opts.nnet_config.momentum >= 0.0 &&
               opts.nnet_config.max_param_change >= 0.0 &&
               opts.nnet_config.backstitch_training_interval > 0);
  delta_nnet_ = nnet_->Copy();
  ScaleNnet(0.0, delta_nnet_);

  if (opts.nnet_config.read_cache != "") {
    bool binary;
    try {
      Input ki(opts.nnet_config.read_cache, &binary);
      compiler_.ReadCache(ki.Stream(), binary);
      KALDI_LOG << "Read computation cache from " << opts.nnet_config.read_cache;
    } catch (...) {
      KALDI_WARN << "Could not open cached computation. "
                    "Probably this is the first training iteration.";
    }
  }
}


void NnetChainTrainer2::Train(const std::string &key, NnetChainExample &chain_eg) {
  bool need_model_derivative = true;
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  bool use_xent_regularization = (opts_.chain_config.xent_regularize != 0.0);
  ComputationRequest request;
  std::string lang_name = "default";
  ParseFromQueryString(key, "lang", &lang_name);
  for (size_t i = 0; i < chain_eg.outputs.size(); i++) {
    // there will normally be exactly one output , named "output"
      if(chain_eg.outputs[i].name.compare("output")==0)
          chain_eg.outputs[i].name = "output-" + lang_name;
  }
  GetChainComputationRequest(*nnet_, chain_eg, need_model_derivative,
                             nnet_config.store_component_stats,
                             use_xent_regularization, need_model_derivative,
                             &request);
  std::shared_ptr<const NnetComputation> computation = compiler_.Compile(request);

  if (nnet_config.backstitch_training_scale > 0.0 && num_minibatches_processed_
      % nnet_config.backstitch_training_interval ==
      srand_seed_ % nnet_config.backstitch_training_interval) {
    // backstitch training is incompatible with momentum > 0
    KALDI_ASSERT(nnet_config.momentum == 0.0);
    FreezeNaturalGradient(true, delta_nnet_);
    bool is_backstitch_step1 = true;
    srand(srand_seed_ + num_minibatches_processed_);
    ResetGenerators(nnet_);
    TrainInternalBackstitch(key, chain_eg, *computation, is_backstitch_step1);
    FreezeNaturalGradient(false, delta_nnet_); // un-freeze natural gradient
    is_backstitch_step1 = false;
    srand(srand_seed_ + num_minibatches_processed_);
    ResetGenerators(nnet_);
    TrainInternalBackstitch(key, chain_eg, *computation, is_backstitch_step1);
  } else { // conventional training
    TrainInternal(key, chain_eg, *computation, lang_name);
  }
  if (num_minibatches_processed_ == 0) {
    ConsolidateMemory(nnet_);
    ConsolidateMemory(delta_nnet_);
  }
  num_minibatches_processed_++;
}

void NnetChainTrainer2::TrainInternal(const std::string &key,
                                     const NnetChainExample &eg,
                                     const NnetComputation &computation,
                                     const std::string &lang_name) {
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  // note: because we give the 1st arg (nnet_) as a pointer to the
  // constructor of 'computer', it will use that copy of the nnet to
  // store stats.
  NnetComputer computer(nnet_config.compute_config, computation,
                        nnet_, delta_nnet_);

  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, eg.inputs);
  computer.Run();

  this->ProcessOutputs(false, lang_name, eg, &computer);
  computer.Run();

  // If relevant, add in the part of the gradient that comes from
  // parameter-level L2 regularization.
  ApplyL2Regularization(*nnet_,
                        GetNumNvalues(eg.inputs, false) *
                        nnet_config.l2_regularize_factor,
                        delta_nnet_);

  // Updates the parameters of nnet
  bool success = UpdateNnetWithMaxChange(
      *delta_nnet_,
      nnet_config.max_param_change,
      1.0, 1.0 - nnet_config.momentum, nnet_,
      &max_change_stats_);

  // Scale down the batchnorm stats (keeps them fresh... this affects what
  // happens when we use the model with batchnorm test-mode set).
  ScaleBatchnormStats(nnet_config.batchnorm_stats_scale, nnet_);

  // The following will only do something if we have a LinearComponent
  // or AffineComponent with orthonormal-constraint set to a nonzero value.
  ConstrainOrthonormal(nnet_);

  // Scale delta_nnet
  if (success)
    ScaleNnet(nnet_config.momentum, delta_nnet_);
  else
    ScaleNnet(0.0, delta_nnet_);
}

void NnetChainTrainer2::TrainInternalBackstitch(const std::string key, const NnetChainExample &eg,
                                               const NnetComputation &computation,
                                               bool is_backstitch_step1) {
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  // note: because we give the 1st arg (nnet_) as a pointer to the
  // constructor of 'computer', it will use that copy of the nnet to
  // store stats.
  NnetComputer computer(nnet_config.compute_config, computation,
                        nnet_, delta_nnet_);
  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, eg.inputs);
  computer.Run();

  bool is_backstitch_step2 = !is_backstitch_step1;
  this->ProcessOutputs(is_backstitch_step2, key, eg, &computer);
  computer.Run();

  BaseFloat max_change_scale, scale_adding;
  if (is_backstitch_step1) {
    // max-change is scaled by backstitch_training_scale;
    // delta_nnet is scaled by -backstitch_training_scale when added to nnet;
    max_change_scale = nnet_config.backstitch_training_scale;
    scale_adding = -nnet_config.backstitch_training_scale;
  } else {
    // max-change is scaled by 1 + backstitch_training_scale;
    // delta_nnet is scaled by 1 + backstitch_training_scale when added to nnet;
    max_change_scale = 1.0 + nnet_config.backstitch_training_scale;
    scale_adding = 1.0 + nnet_config.backstitch_training_scale;
    // If relevant, add in the part of the gradient that comes from L2
    // regularization.  It may not be optimally inefficient to do it on both
    // passes of the backstitch, like we do here, but it probably minimizes
    // any harmful interactions with the max-change.
    ApplyL2Regularization(*nnet_,
        1.0 / scale_adding * GetNumNvalues(eg.inputs, false) *
        nnet_config.l2_regularize_factor, delta_nnet_);
  }

  // Updates the parameters of nnet
  UpdateNnetWithMaxChange(
      *delta_nnet_, nnet_config.max_param_change,
      max_change_scale, scale_adding, nnet_,
      &max_change_stats_);

  if (is_backstitch_step1) {
    // The following will only do something if we have a LinearComponent or
    // AffineComponent with orthonormal-constraint set to a nonzero value. We
    // choose to do this only on the 1st backstitch step, for efficiency.
    ConstrainOrthonormal(nnet_);
  }

  if (!is_backstitch_step1) {
    // Scale down the batchnorm stats (keeps them fresh... this affects what
    // happens when we use the model with batchnorm test-mode set).  Do this
    // after backstitch step 2 so that the stats are scaled down before we start
    // the next minibatch.
    ScaleBatchnormStats(nnet_config.batchnorm_stats_scale, nnet_);
  }

  ScaleNnet(0.0, delta_nnet_);
}

void NnetChainTrainer2::ProcessOutputs(bool is_backstitch_step2,
                                      const std::string &lang_name,
                                      const NnetChainExample &eg,
                                      NnetComputer *computer) {
  // normally the eg will have just one output named 'output', but
  // we don't assume this.
  // In backstitch training, the output-name with the "_backstitch" suffix is
  // the one computed after the first, backward step of backstitch.
  const std::string suffix = (is_backstitch_step2 ? "_backstitch" : "");
  std::vector<NnetChainSupervision>::const_iterator iter = eg.outputs.begin(),
      end = eg.outputs.end();
  for (; iter != end; ++iter) {
    const NnetChainSupervision &sup = *iter;
    std::string node_name = "output-" + lang_name;
    /* sup.name = node_name; */
    int32 node_index = nnet_->GetNodeIndex(node_name);
    if (node_index < 0 ||
        !nnet_->IsOutputNode(node_index))
      KALDI_ERR << "Network has no output named " << node_name;

    const CuMatrixBase<BaseFloat> &nnet_output = computer->GetOutput(node_name);
    CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
                                          nnet_output.NumCols(),
                                          kUndefined);

    bool use_xent = (opts_.chain_config.xent_regularize != 0.0);
    std::string xent_name = node_name + "-xent";  // "output-${lang_name}-xent".
    CuMatrix<BaseFloat> xent_deriv;

    BaseFloat tot_objf, tot_l2_term, tot_weight;

    ComputeChainObjfAndDeriv(opts_.chain_config, *(model_.GetDenGraphForLang(lang_name)),
                             sup.supervision, nnet_output,
                             &tot_objf, &tot_l2_term, &tot_weight,
                             &nnet_output_deriv,
                             (use_xent ? &xent_deriv : NULL));

    if (use_xent) {
      // this block computes the cross-entropy objective.
      const CuMatrixBase<BaseFloat> &xent_output = computer->GetOutput(
          xent_name);
      // at this point, xent_deriv is posteriors derived from the numerator
      // computation.  note, xent_objf has a factor of '.supervision.weight'
      BaseFloat xent_objf = TraceMatMat(xent_output, xent_deriv, kTrans);
      objf_info_[xent_name + suffix].UpdateStats(xent_name + suffix,
                                        opts_.nnet_config.print_interval,
                                        num_minibatches_processed_,
                                        tot_weight, xent_objf);
    }

    if (opts_.apply_deriv_weights && sup.deriv_weights.Dim() != 0) {
      CuVector<BaseFloat> cu_deriv_weights(sup.deriv_weights);
      nnet_output_deriv.MulRowsVec(cu_deriv_weights);
      if (use_xent)
        xent_deriv.MulRowsVec(cu_deriv_weights);
    }

    /* computer->AcceptInput(sup.name, &nnet_output_deriv); */
    computer->AcceptInput(node_name, &nnet_output_deriv);

    /* objf_info_[sup.name + suffix].UpdateStats(sup.name + suffix, */
    objf_info_[node_name + suffix].UpdateStats(sup.name + suffix,
                                     opts_.nnet_config.print_interval,
                                     num_minibatches_processed_,
                                     tot_weight, tot_objf, tot_l2_term);

    if (use_xent) {
      xent_deriv.Scale(opts_.chain_config.xent_regularize);
      computer->AcceptInput(xent_name, &xent_deriv);
    }
  }
}

bool NnetChainTrainer2::PrintTotalStats() const {
  unordered_map<std::string, ObjectiveFunctionInfo, StringHasher>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  bool ans = false;
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    const ObjectiveFunctionInfo &info = iter->second;
    ans = info.PrintTotalStats(name) || ans;
  }
  max_change_stats_.Print(*nnet_);
  return ans;
}

NnetChainTrainer2::~NnetChainTrainer2() {
  if (opts_.nnet_config.write_cache != "") {
    Output ko(opts_.nnet_config.write_cache, opts_.nnet_config.binary_write_cache);
    compiler_.WriteCache(ko.Stream(), opts_.nnet_config.binary_write_cache);
    KALDI_LOG << "Wrote computation cache to " << opts_.nnet_config.write_cache;
  }
  delete delta_nnet_;
}

NnetChainModel2::NnetChainModel2(
    const NnetChainTraining2Options &opts,
    Nnet *nnet,
    const std::string &den_fst_dir
    ):
    opts_(opts),
    nnet(nnet),
    den_fst_dir_(den_fst_dir) {
}

NnetChainModel2::~NnetChainModel2() {
}

NnetChainModel2::LanguageInfo::LanguageInfo(
    const NnetChainModel2::LanguageInfo &other):
    name(other.name),
    den_graph(other.den_graph)
     { }


NnetChainModel2::LanguageInfo::LanguageInfo(
    const std::string &name,
    const fst::StdVectorFst &den_fst, 
    int32 num_pdfs):
    name(name),
    den_graph(den_fst, num_pdfs){
}

void NnetChainModel2::GetPathname(const std::string &dir,
                                   const std::string &name,
                                   const std::string &suffix,
                                   std::string *pathname) {
  std::ostringstream str;
  str << dir << '/' << name << '.' << suffix;
  *pathname = str.str();
}

void NnetChainModel2::GetPathname(const std::string &dir,
                                   const std::string &name,
                                   int32 job_id,
                                   const std::string &suffix,
                                   std::string *pathname) {
  std::ostringstream str;
  str << dir << '/' << name << '.' << job_id << '.' << suffix;
  *pathname = str.str();
}

NnetChainModel2::LanguageInfo *NnetChainModel2::GetInfoForLang(
    const std::string &lang) {
  auto iter = lang_info_.find(lang);
  if (iter != lang_info_.end()) {
    return iter->second;
  } else {
    std::string den_fst_filename;
    GetPathname(den_fst_dir_, lang, "den.fst", &den_fst_filename);
    fst::StdVectorFst den_fst;
    ReadFstKaldi(den_fst_filename, &den_fst);
    std::string outputname = "output-" + lang;

    LanguageInfo *info = new LanguageInfo(lang, den_fst, nnet->OutputDim(outputname));
    lang_info_[lang] = info;
    return info;
  }
}

/* fst::StdVectorFst* NnetChainModel2::GetDenFstForLang( */
/*        const std::string &language_name) { */
/*   LanguageInfo *info = GetInfoForLang(language_name); */
/*   return &(info->den_fst); */
/* } */

chain::DenominatorGraph *NnetChainModel2::GetDenGraphForLang(const std::string &language_name){
  LanguageInfo *info = GetInfoForLang(language_name);
  return &(info->den_graph);
}
} // namespace nnet3
} // namespace kaldi

