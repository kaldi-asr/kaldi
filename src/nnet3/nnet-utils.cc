// nnet3/nnet-utils.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2016  Daniel Galvez
//
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

#include <iomanip>
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-graph.h"
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-normalize-component.h"
#include "nnet3/nnet-general-component.h"
#include "nnet3/nnet-convolutional-component.h"
#include "nnet3/nnet-parse.h"
#include "nnet3/nnet-computation-graph.h"
#include "nnet3/nnet-diagnostics.h"

namespace kaldi {
namespace nnet3 {

int32 NumOutputNodes(const Nnet &nnet) {
  int32 ans = 0;
  for (int32 n = 0; n < nnet.NumNodes(); n++)
    if (nnet.IsOutputNode(n))
      ans++;
  return ans;
}

int32 NumInputNodes(const Nnet &nnet) {
  int32 ans = 0;
  for (int32 n = 0; n < nnet.NumNodes(); n++)
    if (nnet.IsInputNode(n))
      ans++;
  return ans;
}


bool IsSimpleNnet(const Nnet &nnet) {
  // check that we have an output node and called "output".
  if (nnet.GetNodeIndex("output") == -1 ||
      !nnet.IsOutputNode(nnet.GetNodeIndex("output")))
    return false;
  // check that there is an input node named "input".
  if (nnet.GetNodeIndex("input") == -1 ||
      !nnet.IsInputNode(nnet.GetNodeIndex("input")))
    return false;
  // if there was just one input, then it was named
  // "input" and everything checks out.
  if (NumInputNodes(nnet) == 1)
    return true;
  // Otherwise, there should be input node with name "input" and one
  // should be called "ivector".
  return nnet.GetNodeIndex("ivector") != -1 &&
      nnet.IsInputNode(nnet.GetNodeIndex("ivector"));
}

void EvaluateComputationRequest(
    const Nnet &nnet,
    const ComputationRequest &request,
    std::vector<std::vector<bool> > *is_computable) {
  ComputationGraph graph;
  ComputationGraphBuilder builder(nnet, &graph);
  builder.Compute(request);
  builder.GetComputableInfo(is_computable);
  if (GetVerboseLevel() >= 4) {
    std::ostringstream graph_pretty;
    graph.Print(graph_pretty, nnet.GetNodeNames());
    KALDI_VLOG(4) << "Graph is " << graph_pretty.str();
  }
}

// this non-exported function is used in ComputeSimpleNnetContext
// to compute the left and right context of the nnet for a particular
// window size and shift-length.
static void ComputeSimpleNnetContextForShift(
    const Nnet &nnet,
    int32 input_start,
    int32 window_size,
    int32 *left_context,
    int32 *right_context) {

  int32 input_end = input_start + window_size;
  IoSpecification input;
  input.name = "input";
  IoSpecification output;
  output.name = "output";
  IoSpecification ivector;  // we might or might not use this.
  ivector.name = "ivector";

  int32 n = rand() % 10;
  // in the IoSpecification for now we we will request all the same indexes at
  // output that we requested at input.
  for (int32 t = input_start; t < input_end; t++) {
    input.indexes.push_back(Index(n, t));
    output.indexes.push_back(Index(n, t));
  }

  // most networks will just require the ivector at time t = 0,
  // but this might not always be the case, and some might use rounding
  // descriptors with the iVector which might require it at an earlier
  // frame than the regular input, so we provide the iVector in as wide a range
  // as it might possibly be needed.
  for (int32 t = input_start - nnet.Modulus(); t < input_end; t++) {
    ivector.indexes.push_back(Index(n, t));
  }


  ComputationRequest request;
  request.inputs.push_back(input);
  request.outputs.push_back(output);
  if (nnet.GetNodeIndex("ivector") != -1)
    request.inputs.push_back(ivector);
  std::vector<std::vector<bool> > computable;
  EvaluateComputationRequest(nnet, request, &computable);

  KALDI_ASSERT(computable.size() == 1);
  std::vector<bool> &output_ok = computable[0];
  std::vector<bool>::iterator iter =
      std::find(output_ok.begin(), output_ok.end(), true);
  int32 first_ok = iter - output_ok.begin();
  int32 first_not_ok = std::find(iter, output_ok.end(), false) -
      output_ok.begin();
  if (first_ok == window_size || first_not_ok <= first_ok)
    KALDI_ERR << "No outputs were computable (perhaps not a simple nnet?)";
  *left_context = first_ok;
  *right_context = window_size - first_not_ok;
}

void ComputeSimpleNnetContext(const Nnet &nnet,
                              int32 *left_context,
                              int32 *right_context) {
  KALDI_ASSERT(IsSimpleNnet(nnet));
  int32 modulus = nnet.Modulus();
  // modulus >= 1 is a number such that the network ought to be
  // invariant to time shifts (of both the input and output) that
  // are a multiple of this number.  We need to test all shifts modulo
  // this number in case the left and right context vary at all within
  // this range.

  std::vector<int32> left_contexts(modulus + 1);
  std::vector<int32> right_contexts(modulus + 1);

  // This will crash if the total context (left + right) is greater
  // than window_size.
  int32 window_size = 200;

  // by going "<= modulus" instead of "< modulus" we do one more computation
  // than we really need; it becomes a sanity check.
  for (int32 input_start = 0; input_start <= modulus; input_start++)
    ComputeSimpleNnetContextForShift(nnet, input_start, window_size,
                                     &(left_contexts[input_start]),
                                     &(right_contexts[input_start]));
  KALDI_ASSERT(left_contexts[0] == left_contexts[modulus] &&
               "nnet does not have the properties we expect.");
  KALDI_ASSERT(right_contexts[0] == right_contexts[modulus] &&
               "nnet does not have the properties we expect.");
  *left_context =
      *std::max_element(left_contexts.begin(), left_contexts.end());
  *right_context =
      *std::max_element(right_contexts.begin(), right_contexts.end());
}

void PerturbParams(BaseFloat stddev,
                   Nnet *nnet) {
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *comp = nnet->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      UpdatableComponent *u_comp = dynamic_cast<UpdatableComponent*>(comp);
      KALDI_ASSERT(u_comp != NULL);
      u_comp->PerturbParams(stddev);
    }
  }
}

void ComponentDotProducts(const Nnet &nnet1,
                          const Nnet &nnet2,
                          VectorBase<BaseFloat> *dot_prod) {
  KALDI_ASSERT(nnet1.NumComponents() == nnet2.NumComponents());
  int32 updatable_c = 0;
  for (int32 c = 0; c < nnet1.NumComponents(); c++) {
    const Component *comp1 = nnet1.GetComponent(c),
                    *comp2 = nnet2.GetComponent(c);
    if (comp1->Properties() & kUpdatableComponent) {
      const UpdatableComponent
          *u_comp1 = dynamic_cast<const UpdatableComponent*>(comp1),
          *u_comp2 = dynamic_cast<const UpdatableComponent*>(comp2);
      KALDI_ASSERT(u_comp1 != NULL && u_comp2 != NULL);
      dot_prod->Data()[updatable_c] = u_comp1->DotProduct(*u_comp2);
      updatable_c++;
    }
  }
  KALDI_ASSERT(updatable_c == dot_prod->Dim());
}

std::string PrintVectorPerUpdatableComponent(const Nnet &nnet,
                                             const VectorBase<BaseFloat> &vec) {
  std::ostringstream os;
  os << "[ ";
  KALDI_ASSERT(NumUpdatableComponents(nnet) == vec.Dim());
  int32 updatable_c = 0;
  for (int32 c = 0; c < nnet.NumComponents(); c++) {
    const Component *comp = nnet.GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      const std::string &component_name = nnet.GetComponentName(c);
      os << component_name << ':' << vec(updatable_c) << ' ';
      updatable_c++;
    }
  }
  KALDI_ASSERT(updatable_c == vec.Dim());
  os << ']';
  return os.str();
}

BaseFloat DotProduct(const Nnet &nnet1,
                     const Nnet &nnet2) {
  KALDI_ASSERT(nnet1.NumComponents() == nnet2.NumComponents());
  BaseFloat ans = 0.0;
  for (int32 c = 0; c < nnet1.NumComponents(); c++) {
    const Component *comp1 = nnet1.GetComponent(c),
                    *comp2 = nnet2.GetComponent(c);
    if (comp1->Properties() & kUpdatableComponent) {
      const UpdatableComponent
          *u_comp1 = dynamic_cast<const UpdatableComponent*>(comp1),
          *u_comp2 = dynamic_cast<const UpdatableComponent*>(comp2);
      KALDI_ASSERT(u_comp1 != NULL && u_comp2 != NULL);
      ans += u_comp1->DotProduct(*u_comp2);
    }
  }
  return ans;
}


void ZeroComponentStats(Nnet *nnet) {
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *comp = nnet->GetComponent(c);
    comp->ZeroStats();  // for some components, this won't do anything.
  }
}

void SetLearningRate(BaseFloat learning_rate,
                     Nnet *nnet) {
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *comp = nnet->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      // For now all updatable components inherit from class UpdatableComponent.
      // If that changes in future, we will change this code.
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
            "UpdatableComponent; change this code.";
      uc->SetUnderlyingLearningRate(learning_rate);
    }
  }
}

void SetNnetAsGradient(Nnet *nnet) {
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *comp = nnet->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      UpdatableComponent *u_comp = dynamic_cast<UpdatableComponent*>(comp);
      KALDI_ASSERT(u_comp != NULL);
      u_comp->SetAsGradient();
    }
  }
}

void ScaleNnet(BaseFloat scale, Nnet *nnet) {
  if (scale == 1.0) return;
  else {
    for (int32 c = 0; c < nnet->NumComponents(); c++) {
      Component *comp = nnet->GetComponent(c);
      comp->Scale(scale);
    }
  }
}

void AddNnetComponents(const Nnet &src, const Vector<BaseFloat> &alphas,
                       BaseFloat scale, Nnet *dest) {
  if (src.NumComponents() != dest->NumComponents())
    KALDI_ERR << "Trying to add incompatible nnets.";
  int32 i = 0;
  for (int32 c = 0; c < src.NumComponents(); c++) {
    const Component *src_comp = src.GetComponent(c);
    Component *dest_comp = dest->GetComponent(c);
    if (src_comp->Properties() & kUpdatableComponent) {
      // For now all updatable components inherit from class UpdatableComponent.
      // If that changes in future, we will change this code.
      const UpdatableComponent *src_uc =
          dynamic_cast<const UpdatableComponent*>(src_comp);
      UpdatableComponent *dest_uc =
          dynamic_cast<UpdatableComponent*>(dest_comp);
      if (src_uc == NULL || dest_uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
            "UpdatableComponent; change this code.";
      KALDI_ASSERT(i < alphas.Dim());
      dest_uc->Add(alphas(i++), *src_uc);
    } else { // add stored stats
      dest_comp->Add(scale, *src_comp);
    }
  }
  KALDI_ASSERT(i == alphas.Dim());
}

void AddNnet(const Nnet &src, BaseFloat alpha, Nnet *dest) {
  if (src.NumComponents() != dest->NumComponents())
    KALDI_ERR << "Trying to add incompatible nnets.";
  for (int32 c = 0; c < src.NumComponents(); c++) {
    const Component *src_comp = src.GetComponent(c);
    Component *dest_comp = dest->GetComponent(c);
    dest_comp->Add(alpha, *src_comp);
  }
}

int32 NumParameters(const Nnet &src) {
  int32 ans = 0;
  for (int32 c = 0; c < src.NumComponents(); c++) {
    const Component *comp = src.GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      // For now all updatable components inherit from class UpdatableComponent.
      // If that changes in future, we will change this code.
      const UpdatableComponent *uc =
          dynamic_cast<const UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
            "UpdatableComponent; change this code.";
      ans += uc->NumParameters();
    }
  }
  return ans;
}


void VectorizeNnet(const Nnet &src,
                   VectorBase<BaseFloat> *parameters) {
  KALDI_ASSERT(parameters->Dim() == NumParameters(src));
  int32 dim_offset = 0;
  for (int32 c = 0; c < src.NumComponents(); c++) {
    const Component *comp = src.GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      // For now all updatable components inherit from class UpdatableComponent.
      // If that changes in future, we will change this code.
      const UpdatableComponent *uc =
          dynamic_cast<const UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
            "UpdatableComponent; change this code.";
      int32 this_dim = uc->NumParameters();
      SubVector<BaseFloat> this_part(*parameters, dim_offset, this_dim);
      uc->Vectorize(&this_part);
      dim_offset += this_dim;
    }
  }
}


void UnVectorizeNnet(const VectorBase<BaseFloat> &parameters,
                     Nnet *dest) {
  KALDI_ASSERT(parameters.Dim() == NumParameters(*dest));
  int32 dim_offset = 0;
  for (int32 c = 0; c < dest->NumComponents(); c++) {
    Component *comp = dest->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      // For now all updatable components inherit from class UpdatableComponent.
      // If that changes in future, we will change this code.
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
            "UpdatableComponent; change this code.";
      int32 this_dim = uc->NumParameters();
      const SubVector<BaseFloat> this_part(parameters, dim_offset, this_dim);
      uc->UnVectorize(this_part);
      dim_offset += this_dim;
    }
  }
}

int32 NumUpdatableComponents(const Nnet &dest) {
  int32 ans = 0;
  for (int32 c = 0; c < dest.NumComponents(); c++) {
      const Component *comp = dest.GetComponent(c);
    if (comp->Properties() & kUpdatableComponent)
      ans++;
  }
  return ans;
}

void FreezeNaturalGradient(bool freeze, Nnet *nnet) {
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *comp = nnet->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      // For now all updatable components inherit from class UpdatableComponent.
      // If that changes in future, we will change this code.
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
            "UpdatableComponent; change this code.";
      uc->FreezeNaturalGradient(freeze);
    }
  }
}

void ConvertRepeatedToBlockAffine(CompositeComponent *c_component) {
  for(int32 i = 0; i < c_component->NumComponents(); i++) {
    const Component *c = c_component->GetComponent(i);
    KALDI_ASSERT(c->Type() != "CompositeComponent" &&
                 "Nesting CompositeComponent within CompositeComponent is not allowed.\n"
                 "(We may change this as more complicated components are introduced.)");

    if(c->Type() == "RepeatedAffineComponent" ||
       c->Type() == "NaturalGradientRepeatedAffineComponent") {
      // N.B.: NaturalGradientRepeatedAffineComponent is a subclass of
      // RepeatedAffineComponent.
      const RepeatedAffineComponent *rac =
        dynamic_cast<const RepeatedAffineComponent*>(c);
      KALDI_ASSERT(rac != NULL);
      BlockAffineComponent *bac = new BlockAffineComponent(*rac);
      // following call deletes rac
      c_component->SetComponent(i, bac);
    }
  }
}

void ConvertRepeatedToBlockAffine(Nnet *nnet) {
  for(int32 i = 0; i < nnet->NumComponents(); i++) {
    const Component *const_c = nnet->GetComponent(i);
    if(const_c->Type() == "RepeatedAffineComponent" ||
       const_c->Type() == "NaturalGradientRepeatedAffineComponent") {
      // N.B.: NaturalGradientRepeatedAffineComponent is a subclass of
      // RepeatedAffineComponent.
      const RepeatedAffineComponent *rac =
        dynamic_cast<const RepeatedAffineComponent*>(const_c);
      KALDI_ASSERT(rac != NULL);
      BlockAffineComponent *bac = new BlockAffineComponent(*rac);
      // following call deletes rac
      nnet->SetComponent(i, bac);
    } else if (const_c->Type() == "CompositeComponent") {
      // We must modify the composite component, so we use the
      // non-const GetComponent() call here.
      Component *c = nnet->GetComponent(i);
      CompositeComponent *cc = dynamic_cast<CompositeComponent*>(c);
      KALDI_ASSERT(cc != NULL);
      ConvertRepeatedToBlockAffine(cc);
    }
  }
}

std::string NnetInfo(const Nnet &nnet) {
  std::ostringstream ostr;
  if (IsSimpleNnet(nnet)) {
    int32 left_context, right_context;
    // this call will crash if the nnet is not 'simple'.
    ComputeSimpleNnetContext(nnet, &left_context, &right_context);
    ostr << "left-context: " << left_context << "\n";
    ostr << "right-context: " << right_context << "\n";
  }
  ostr << "input-dim: " << nnet.InputDim("input") << "\n";
  ostr << "ivector-dim: " << nnet.InputDim("ivector") << "\n";
  ostr << "output-dim: " << nnet.OutputDim("output") << "\n";
  ostr << "# Nnet info follows.\n";
  ostr << nnet.Info();
  return ostr.str();
}

void SetDropoutProportion(BaseFloat dropout_proportion,
                          Nnet *nnet) {
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *comp = nnet->GetComponent(c);
    DropoutComponent *dc = dynamic_cast<DropoutComponent*>(comp);
    if (dc != NULL)
      dc->SetDropoutProportion(dropout_proportion);
    DropoutMaskComponent *mc =
        dynamic_cast<DropoutMaskComponent*>(nnet->GetComponent(c));
    if (mc != NULL)
      mc->SetDropoutProportion(dropout_proportion);
    GeneralDropoutComponent *gdc =
        dynamic_cast<GeneralDropoutComponent*>(nnet->GetComponent(c));
    if (gdc != NULL)
      gdc->SetDropoutProportion(dropout_proportion);
  }
}

bool HasBatchnorm(const Nnet &nnet) {
  for (int32 c = 0; c < nnet.NumComponents(); c++) {
    const Component *comp = nnet.GetComponent(c);
    if (dynamic_cast<const BatchNormComponent*>(comp) != NULL)
      return true;
  }
  return false;
}

void ScaleBatchnormStats(BaseFloat batchnorm_stats_scale,
                         Nnet *nnet) {
  KALDI_ASSERT(batchnorm_stats_scale >= 0.0 && batchnorm_stats_scale <= 1.0);
  if (batchnorm_stats_scale == 1.0)
    return;
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *comp = nnet->GetComponent(c);
    BatchNormComponent *bc = dynamic_cast<BatchNormComponent*>(comp);
    if (bc != NULL)
      bc->Scale(batchnorm_stats_scale);
  }
}


void RecomputeStats(const std::vector<NnetExample> &egs, Nnet *nnet) {
  KALDI_LOG << "Recomputing stats on nnet (affects batch-norm)";
  ZeroComponentStats(nnet);
  NnetComputeProbOptions opts;
  opts.store_component_stats = true;
  NnetComputeProb prob_computer(opts, nnet);
  for (size_t i = 0; i < egs.size(); i++)
    prob_computer.Compute(egs[i]);
  prob_computer.PrintTotalStats();
  KALDI_LOG << "Done recomputing stats.";
}



void SetBatchnormTestMode(bool test_mode,  Nnet *nnet) {
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *comp = nnet->GetComponent(c);
    BatchNormComponent *bc = dynamic_cast<BatchNormComponent*>(comp);
    if (bc != NULL)
      bc->SetTestMode(test_mode);
  }
}

void SetDropoutTestMode(bool test_mode,  Nnet *nnet) {
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *comp = nnet->GetComponent(c);
    RandomComponent *rc = dynamic_cast<RandomComponent*>(comp);
    if (rc != NULL)
      rc->SetTestMode(test_mode);
  }
}

void ResetGenerators(Nnet *nnet){
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *comp = nnet->GetComponent(c);
    RandomComponent *rc = dynamic_cast<RandomComponent*>(comp);
    if (rc != NULL)
      rc->ResetGenerator();
  }
}

void FindOrphanComponents(const Nnet &nnet, std::vector<int32> *components) {
  int32 num_components = nnet.NumComponents(), num_nodes = nnet.NumNodes();
  std::vector<bool> is_used(num_components, false);
  for (int32 i = 0; i < num_nodes; i++) {
    if (nnet.IsComponentNode(i)) {
      int32 c = nnet.GetNode(i).u.component_index;
      KALDI_ASSERT(c >= 0 && c < num_components);
      is_used[c] = true;
    }
  }
  components->clear();
  for (int32 i = 0; i < num_components; i++)
    if (!is_used[i])
      components->push_back(i);
}

void FindOrphanNodes(const Nnet &nnet, std::vector<int32> *nodes) {

  std::vector<std::vector<int32> > depend_on_graph, dependency_graph;
  NnetToDirectedGraph(nnet, &depend_on_graph);
  // depend_on_graph[i] is a list of all the nodes that depend on i.
  ComputeGraphTranspose(depend_on_graph, &dependency_graph);
  // dependency_graph[i] is a list of all the nodes that i depends on,
  // to be computed.

  // Find all nodes required to produce the outputs.
  int32 num_nodes = nnet.NumNodes();
  assert(num_nodes == static_cast<int32>(dependency_graph.size()));
  std::vector<bool> node_is_required(num_nodes, false);
  std::vector<int32> queue;
  for (int32 i = 0; i < num_nodes; i++) {
    if (nnet.IsOutputNode(i))
      queue.push_back(i);
  }
  while (!queue.empty()) {
    int32 i = queue.back();
    queue.pop_back();
    if (!node_is_required[i]) {
      node_is_required[i] = true;
      for (size_t j = 0; j < dependency_graph[i].size(); j++)
        queue.push_back(dependency_graph[i][j]);
    }
  }
  nodes->clear();
  for (int32 i = 0; i < num_nodes; i++) {
    if (!node_is_required[i])
      nodes->push_back(i);
  }
}


// this class implements the internals of the edit directive 'apply-svd'.
class SvdApplier {
 public:
  SvdApplier(const std::string component_name_pattern,
             int32 bottleneck_dim,
             Nnet *nnet): nnet_(nnet),
                          bottleneck_dim_(bottleneck_dim),
                          component_name_pattern_(component_name_pattern) { }
  void ApplySvd() {
    DecomposeComponents();
    if (!modified_component_info_.empty())
      ModifyTopology();
    KALDI_LOG << "Decomposed " << modified_component_info_.size()
              << " components with SVD dimension " << bottleneck_dim_;
  }

 private:
  // This function finds components to decompose and decomposes them,  adding _a and
  // _b versions of those components to the nnet while not removing the original
  // ones.  Does not affect the graph topology.
  void DecomposeComponents() {
    int32 num_components = nnet_->NumComponents();
    modification_index_.resize(num_components, -1);
    for (int32 c = 0; c < num_components; c++) {
      Component *component = nnet_->GetComponent(c);
      std::string component_name = nnet_->GetComponentName(c);
      if (NameMatchesPattern(component_name.c_str(),
                             component_name_pattern_.c_str())) {
        AffineComponent *affine =  dynamic_cast<AffineComponent*>(component);
        if (affine == NULL) {
          KALDI_WARN << "Not decomposing component " << component_name
                     << " as it is not an AffineComponent.";
          continue;
        }
        int32 input_dim = affine->InputDim(),
            output_dim = affine->OutputDim();
        if (input_dim <= bottleneck_dim_ || output_dim <= bottleneck_dim_) {
          KALDI_WARN << "Not decomposing component " << component_name
                     << " with SVD to rank " << bottleneck_dim_
                     << " because its dimension is " << input_dim
                     << " -> " << output_dim;
          continue;
        }
        size_t n = modified_component_info_.size();
        modification_index_[c] = n;
        modified_component_info_.resize(n + 1);
        ModifiedComponentInfo &info = modified_component_info_[n];
        info.component_index = c;
        info.component_name = component_name;
        Component *component_a = NULL, *component_b = NULL;
        info.component_name_a = component_name + "_a";
        info.component_name_b = component_name + "_b";
        if (nnet_->GetComponentIndex(info.component_name_a) >= 0)
          KALDI_ERR << "Neural network already has a component named "
                    << info.component_name_a;
        if (nnet_->GetComponentIndex(info.component_name_b) >= 0)
          KALDI_ERR << "Neural network already has a component named "
                    << info.component_name_b;
        DecomposeComponent(component_name, *affine, &component_a, &component_b);
        info.component_a_index = nnet_->AddComponent(info.component_name_a,
                                                     component_a);
        info.component_b_index = nnet_->AddComponent(info.component_name_b,
                                                     component_b);
      }
    }
    KALDI_LOG << "Converted " << modified_component_info_.size()
              << " components to FixedAffineComponent.";
  }

  void DecomposeComponent(const std::string &component_name,
                          const AffineComponent &affine,
                          Component **component_a_out,
                          Component **component_b_out) {
    int32 input_dim = affine.InputDim(), output_dim = affine.OutputDim();
    Matrix<BaseFloat> linear_params(affine.LinearParams());
    Vector<BaseFloat> bias_params(affine.BiasParams());

    int32 bottleneck_dim = bottleneck_dim_,
        middle_dim = std::min<int32>(input_dim, output_dim);
    KALDI_ASSERT(bottleneck_dim < middle_dim);

    // note: 'linear_params' is of dimension output_dim by input_dim.
    Vector<BaseFloat> s(middle_dim);
    Matrix<BaseFloat> A(middle_dim, input_dim),
        B(output_dim, middle_dim);
    linear_params.Svd(&s, &B, &A);
    // make sure the singular values are sorted from greatest to least value.
    SortSvd(&s, &B, &A);
    BaseFloat s_sum_orig = s.Sum();
    s.Resize(bottleneck_dim, kCopyData);
    A.Resize(bottleneck_dim, input_dim, kCopyData);
    B.Resize(output_dim, bottleneck_dim, kCopyData);
    BaseFloat s_sum_reduced = s.Sum();
    KALDI_LOG << "For component " << component_name
              << " singular value sum changed by "
              << (s_sum_orig - s_sum_reduced)
              << " (from " << s_sum_orig << " to " << s_sum_reduced << ")";

    // we'll divide the singular values equally between the two
    // parameter matrices.
    s.ApplyPow(0.5);
    A.MulRowsVec(s);
    B.MulColsVec(s);

    CuMatrix<BaseFloat> A_cuda(A), B_cuda(B);
    CuVector<BaseFloat> bias_params_cuda(bias_params);

    LinearComponent *component_a = new LinearComponent(A_cuda);
    NaturalGradientAffineComponent *component_b =
        new NaturalGradientAffineComponent(B_cuda, bias_params_cuda);
    // set the learning rates, max-change, and so on.
    component_a->SetUpdatableConfigs(affine);
    component_b->SetUpdatableConfigs(affine);
    *component_a_out = component_a;
    *component_b_out = component_b;
  }

  // This function modifies the topology of the neural network, splitting
  // up the components we're modifying into two parts.
  // Suppose we have something like:
  //  component-node name=some_node component=some_component input=
  void ModifyTopology() {
    // nodes_to_split will be a list of component-node indexes that we
    // need to split into two.  These will be nodes like
    // component-node name=component_node_name component=component_name input=xxx
    // where 'component_name' is one of the components that we're splitting.
    std::set<int32> nodes_to_modify;


    // node_names_modified is nnet_->node_names_ except with, for the nodes that
    // we are splitting in two, "some_node_name" replaced with
    // "some_node_name_b" (the second of the two split nodes).
    std::vector<std::string> node_names_orig = nnet_->GetNodeNames(),
        node_names_modified = node_names_orig;

    // The following loop sets up 'nodes_to_modify' and 'node_names_modified'.
    for (int32 n = 0; n < nnet_->NumNodes(); n++) {
      if (nnet_->IsComponentNode(n)) {
        NetworkNode &node = nnet_->GetNode(n);
        int32 component_index = node.u.component_index,
            modification_index = modification_index_[component_index];
        if (modification_index >= 0) {
          // This is a component-node for one of the components that we're
          // splitting in two.
          nodes_to_modify.insert(n);
          std::string node_name = node_names_orig[n],
              node_name_b = node_name + "_b";
          node_names_modified[n] = node_name_b;
        }
      }
    }


    // config_os is a stream to which we are printing lines that we'll later
    // read using nnet_->ReadConfig().
    std::ostringstream config_os;
    // The following loop writes to 'config_os'. The the code is modified from
    // the private function Nnet::GetAsConfigLine(), and from
    // Nnet::GetConfigLines().
    for (int32 n = 0; n < nnet_->NumNodes(); n++) {
      if (nnet_->IsComponentInputNode(n) || nnet_->IsInputNode(n)) {
        // component-input descriptor nodes aren't handled separately from their
        // associated components (we deal with them along with their
        // component-node); and input-nodes won't be affected so we don't have
        // to print anything.
        continue;
      }
      const NetworkNode &node = nnet_->GetNode(n);
      int32 c = node.u.component_index;  // 'c' will only be meaningful if the
                                         // node is a component-node.
      std::string node_name = node_names_orig[n];
      if (node.node_type == kComponent &&  modification_index_[c] >= 0) {
        ModifiedComponentInfo &info = modified_component_info_[
            modification_index_[c]];
        std::string node_name_a = node_name + "_a",
            node_name_b = node_name + "_b";
        // we print two component-nodes, the "a" an "b".  The original
        // one will later be removed when we call RemoveOrphanNodes().
        config_os << "component-node name=" << node_name_a << " component="
                  << info.component_name_a << " input=";
        nnet_->GetNode(n-1).descriptor.WriteConfig(config_os, node_names_modified);
        config_os << "\n";
        config_os << "component-node name=" << node_name_b << " component="
                  << info.component_name_b << " input=" << node_name_a << "\n";
      } else {
        // This code is modified from Nnet::GetAsConfigLine().  The key difference
        // is that we're using node_names_modified, which will replace all the
        // nodes we're splitting with their "b" versions.
        switch (node.node_type) {
          case kDescriptor:
            // assert that it's an output-descriptor, not one describing the input to
            // a component-node.
            KALDI_ASSERT(nnet_->IsOutputNode(n));
            config_os << "output-node name=" << node_name << " input=";
            node.descriptor.WriteConfig(config_os, node_names_modified);
            config_os << " objective=" << (node.u.objective_type == kLinear ?
                                           "linear" : "quadratic");
            break;
          case kComponent:
            config_os << "component-node name=" << node_name << " component="
                      << nnet_->GetComponentName(node.u.component_index)
                      << " input=";
            nnet_->GetNode(n-1).descriptor.WriteConfig(config_os,
                                                       node_names_modified);
            break;
          case kDimRange:
            config_os << "dim-range-node name=" << node_name << " input-node="
                      << node_names_modified[node.u.node_index]
                      << " dim-offset=" << node.dim_offset
                      << " dim=" << node.dim;
            break;
          default:
            KALDI_ERR << "Unexpected node type.";
        }
        config_os << "\n";
      }
    }
    std::istringstream config_is(config_os.str());
    nnet_->ReadConfig(config_is);
    nnet_->RemoveOrphanNodes();
    nnet_->RemoveOrphanComponents();
  }

  // modification_index_ is a vector with dimension equal to the number of
  // components nnet_ had at entry.  For each component that we are decomposing,
  // it contains an index >= 0 into the 'component_info_' vector; for each
  // component that we are not decomposing, it contains -1.
  // with SVD.
  std::vector<int32> modification_index_;

  struct ModifiedComponentInfo {
    int32 component_index;  // Index of the component we are modifying.
    std::string component_name;  // The original name of the component,
                                 // e.g. "some_component".
    std::string component_name_a;  // The original name of the component, plus "_a"
                                   // e.g. "some_component_a".
    std::string component_name_b;  // The original name of the component, plus "_b"
                                   // e.g. "some_component_b".
    int32 component_a_index;  // component-index of the left part of the
                              // decomposed component, which will have a name
                              // like "some_component_a".
    int32 component_b_index;  // component-index of the right part of the
                              // decomposed component, which will have a name
                              // like "some_component_b".

  };
  std::vector<ModifiedComponentInfo> modified_component_info_;


  Nnet *nnet_;
  int32 bottleneck_dim_;
  std::string component_name_pattern_;
};

/*
  Does an update that moves M closer to being a (matrix with orthonormal rows)
  times 'scale'.  Note: this will diverge if we start off with singular values
  too far from 'scale'.

  This function requires 'scale' to be nonzero.  If 'scale' is negative, then it
  will be set internally to the value that ensures the change in M is orthogonal to
  M (viewed as a vector).
*/
void ConstrainOrthonormalInternal(BaseFloat scale, CuMatrixBase<BaseFloat> *M) {
  KALDI_ASSERT(scale != 0.0);

  // We'd like to enforce the rows of M to be orthonormal.
  // define P = M M^T.  If P is unit then M has orthonormal rows.
  // We actually want P to equal scale^2 * I, so that M's rows are
  // orthogonal with 2-norms equal to 'scale'.
  // We (notionally) add to the objective function, the value
  // -alpha times the sum of squared elements of Q = (P - scale^2 * I).
  int32 rows = M->NumRows(), cols = M->NumCols();
  CuMatrix<BaseFloat> M_update(rows, cols);
  CuMatrix<BaseFloat> P(rows, rows);
  P.SymAddMat2(1.0, *M, kNoTrans, 0.0);
  P.CopyLowerToUpper();

  // The 'update_speed' is a constant that determines how fast we approach a
  // matrix with the desired properties (larger -> faster).  Larger values will
  // update faster but will be more prone to instability.  0.125 (1/8) is the
  // value that gives us the fastest possible convergence when we are already
  // close to be a semi-orthogonal matrix (in fact, it will lead to quadratic
  // convergence).
  // See  http://www.danielpovey.com/files/2018_interspeech_tdnnf.pdf
  // for more details.
  BaseFloat update_speed = 0.125;

  if (scale < 0.0) {
    // If scale < 0.0 then it's like letting the scale "float",
    // as in Sec. 2.3 of
    // http://www.danielpovey.com/files/2018_interspeech_tdnnf.pdf,
    // where 'scale' here is written 'alpha' in the paper.
    //
    // We pick the scale that will give us an update to M that is
    // orthogonal to M (viewed as a vector): i.e., if we're doing
    // an update M := M + X, then we want to have tr(M X^T) == 0.
    // The following formula is what gives us that.
    // With P = M M^T, our update formula is doing to be:
    //  M := M + (-4 * alpha * (P - scale^2 I) * M).
    // (The math below explains this update formula; for now, it's
    // best to view it as an established fact).
    // So X (the change in M) is -4 * alpha * (P - scale^2 I) * M,
    // where alpha == update_speed / scale^2.
    // We want tr(M X^T) == 0.  First, forget the -4*alpha, because
    // we don't care about constant factors.  So we want:
    //  tr(M * M^T * (P - scale^2 I)) == 0.
    // Since M M^T == P, that means:
    //  tr(P^2 - scale^2 P) == 0,
    // or scale^2 = tr(P^2) / tr(P).
    // Note: P is symmetric so it doesn't matter whether we use tr(P P) or
    // tr(P^T P); we use tr(P^T P) because I believe it's faster to compute.

    BaseFloat trace_P = P.Trace(), trace_P_P = TraceMatMat(P, P, kTrans);

    scale = std::sqrt(trace_P_P / trace_P);

    // The following is a tweak to avoid divergence when the eigenvalues aren't
    // close to being the same.  trace_P is the sum of eigenvalues of P, and
    // trace_P_P is the sum-square of eigenvalues of P.  Treat trace_P as a sum
    // of positive values, and trace_P_P as their sumsq.  Then mean = trace_P /
    // dim, and trace_P_P cannot be less than dim * (trace_P / dim)^2,
    // i.e. trace_P_P >= trace_P^2 / dim.  If ratio = trace_P_P * dim /
    // trace_P^2, then ratio >= 1.0, and the excess above 1.0 is a measure of
    // how far we are from convergence.  If we're far from convergence, we make
    // the learning rate slower to reduce the risk of divergence, since the
    // update may not be stable for starting points far from equilibrium.
    BaseFloat ratio = (trace_P_P * P.NumRows() / (trace_P * trace_P));
    KALDI_ASSERT(ratio > 0.999);
    if (ratio > 1.02) {
      update_speed *= 0.5;  // Slow down the update speed to reduce the risk of divergence.
    }
  }

  // see Sec. 2.2 of http://www.danielpovey.com/files/2018_interspeech_tdnnf.pdf
  // for explanation of the 1/(scale*scale) factor, but there is a difference in
  // notation; 'scale' here corresponds to 'alpha' in the paper, and
  // 'update_speed' corresponds to 'nu' in the paper.
  BaseFloat alpha = update_speed / (scale * scale);

  P.AddToDiag(-1.0 * scale * scale);

  if (GetVerboseLevel() >= 1) {
    BaseFloat error = P.FrobeniusNorm();
    KALDI_VLOG(2) << "Error in orthogonality is " << error;
  }

  // At this point, the matrix P contains what, in the math, would be Q =
  // P-scale^2*I.  The derivative of the objective function w.r.t. an element q(i,j)
  // of Q is now equal to -2*alpha*q(i,j), i.e. we could write q_deriv(i,j)
  // = -2*alpha*q(i,j) This is also the derivative of the objective function
  // w.r.t. p(i,j): i.e. p_deriv(i,j) = -2*alpha*q(i,j).
  // Suppose we have define this matrix as 'P_deriv'.
  // The derivative of the objective w.r.t M equals
  // 2 * P_deriv * M, which equals -4*alpha*(P-scale^2*I)*M.
  // (Currently the matrix P contains what, in the math, is P-scale^2*I).
  M_update.AddMatMat(-4.0 * alpha, P, kNoTrans, *M, kNoTrans, 0.0);
  M->AddMat(1.0, M_update);
}

/**
   This function, to be called after processing every minibatch, is responsible
   for enforcing the orthogonality constraint for any components of type
   LinearComponent or inheriting from AffineComponent that have the
   "orthonormal_constraint" value set.
 */
void ConstrainOrthonormal(Nnet *nnet) {

  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *component = nnet->GetComponent(c);
    LinearComponent *lc = dynamic_cast<LinearComponent*>(component);
    if (lc != NULL && lc->OrthonormalConstraint() != 0.0) {
      if (RandInt(0, 3) != 0)
        continue;  // For efficiency, only do this every 4 minibatches-- it won't
                   // stray far.
      BaseFloat scale = lc->OrthonormalConstraint();

      CuMatrixBase<BaseFloat> &params = lc->Params();
      int32 rows = params.NumRows(), cols = params.NumCols();
      if (rows <= cols) {
        ConstrainOrthonormalInternal(scale, &params);
      } else {
        CuMatrix<BaseFloat> params_trans(params, kTrans);
        ConstrainOrthonormalInternal(scale, &params_trans);
        params.CopyFromMat(params_trans, kTrans);
      }
    }

    AffineComponent *ac = dynamic_cast<AffineComponent*>(component);
    if (ac != NULL && ac->OrthonormalConstraint() != 0.0) {
      if (RandInt(0, 3) != 0)
        continue;  // For efficiency, only do this every 4 minibatches-- it won't
                   // stray far.
      BaseFloat scale = ac->OrthonormalConstraint();
      CuMatrixBase<BaseFloat> &params = ac->LinearParams();
      int32 rows = params.NumRows(), cols = params.NumCols();
      if (rows <= cols) {
        ConstrainOrthonormalInternal(scale, &params);
      } else {
        CuMatrix<BaseFloat> params_trans(params, kTrans);
        ConstrainOrthonormalInternal(scale, &params_trans);
        params.CopyFromMat(params_trans, kTrans);
      }
    }
  }
}


// This code has been broken out of ReadEditConfig as it's quite long.
// It implements the internals of the edit directive 'reduce-rank'.
// See also the related direcive 'apply-svd'.
void ReduceRankOfComponents(const std::string component_name_pattern,
                            int32 rank,
                            Nnet *nnet) {
  int32 num_components_changed = 0;
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *component = nnet->GetComponent(c);
    std::string component_name = nnet->GetComponentName(c);
    if (NameMatchesPattern(component_name.c_str(),
                           component_name_pattern.c_str())) {
      AffineComponent *affine =  dynamic_cast<AffineComponent*>(component);
      if (affine == NULL) {
        KALDI_WARN << "Not reducing rank of component " << component_name
                   << " as it is not an AffineComponent.";
        continue;
      }
      int32 input_dim = affine->InputDim(),
          output_dim = affine->OutputDim();
      if (input_dim <= rank || output_dim <= rank) {
        KALDI_WARN << "Not reducing rank of component " << component_name
                   << " with SVD to rank " << rank
                   << " because its dimension is " << input_dim
                   << " -> " << output_dim;
        continue;
      }
      Matrix<BaseFloat> linear_params(affine->LinearParams());
      Vector<BaseFloat> bias_params(affine->BiasParams());

      // note: 'linear_params' is of dimension output_dim by input_dim.
      int32 middle_dim = std::min<int32>(input_dim, output_dim);
      Vector<BaseFloat> s(middle_dim);
      Matrix<BaseFloat> U(output_dim, middle_dim),
          Vt(middle_dim, input_dim);
      linear_params.Svd(&s, &U, &Vt);
      // make sure the singular values are sorted from greatest to least value.
      SortSvd(&s, &U, &Vt);
      BaseFloat s_sum_orig = s.Sum();
      s.Resize(rank, kCopyData);
      U.Resize(output_dim, rank, kCopyData);
      Vt.Resize(rank, input_dim, kCopyData);
      BaseFloat s_sum_reduced = s.Sum();
      KALDI_LOG << "For component " << component_name
                << " singular value sum changed by reduce-rank command "
                << (s_sum_orig - s_sum_reduced)
                << " (from " << s_sum_orig << " to " << s_sum_reduced << ")";
      U.MulColsVec(s);
      Matrix<BaseFloat> linear_params_reduced_rank(output_dim, input_dim);
      linear_params_reduced_rank.AddMatMat(1.0, U, kNoTrans, Vt, kNoTrans, 0.0);
      CuMatrix<BaseFloat> linear_params_reduced_rank_cuda;
      linear_params_reduced_rank_cuda.Swap(&linear_params_reduced_rank);
      CuVector<BaseFloat> bias_params_cuda;
      bias_params_cuda.Swap(&bias_params);
      affine->SetParams(bias_params_cuda, linear_params_reduced_rank_cuda);
      num_components_changed++;
    }
  }
  KALDI_LOG << "Reduced rank of parameters of " << num_components_changed
            << " components.";
}




void ReadEditConfig(std::istream &edit_config_is, Nnet *nnet) {
  std::vector<std::string> lines;
  ReadConfigLines(edit_config_is, &lines);
  // we process this as a sequence of lines.
  std::vector<ConfigLine> config_lines;
  ParseConfigLines(lines, &config_lines);
  for (size_t i = 0; i < config_lines.size(); i++) {
    ConfigLine &config_line = config_lines[i];
    const std::string &directive = config_lines[i].FirstToken();
    if (directive == "convert-to-fixed-affine") {
      std::string name_pattern = "*";
      // name_pattern defaults to '*' if none is given.  Note: this pattern
      // matches names of components, not nodes.
      config_line.GetValue("name", &name_pattern);
      int32 num_components_changed = 0;
      for (int32 c = 0; c < nnet->NumComponents(); c++) {
        Component *component = nnet->GetComponent(c);
        AffineComponent *affine = NULL;
        if (NameMatchesPattern(nnet->GetComponentName(c).c_str(),
                               name_pattern.c_str()) &&
            (affine = dynamic_cast<AffineComponent*>(component))) {
          nnet->SetComponent(c, new FixedAffineComponent(*affine));
          num_components_changed++;
        }
      }
      KALDI_LOG << "Converted " << num_components_changed
                << " components to FixedAffineComponent.";
    } else if (directive == "remove-orphan-nodes") {
      bool remove_orphan_inputs = false;
      config_line.GetValue("remove-orphan-inputs", &remove_orphan_inputs);
      nnet->RemoveOrphanNodes(remove_orphan_inputs);
    } else if (directive == "remove-orphan-components") {
      nnet->RemoveOrphanComponents();
    } else if (directive == "remove-orphans") {
      bool remove_orphan_inputs = false;
      config_line.GetValue("remove-orphan-inputs", &remove_orphan_inputs);
      nnet->RemoveOrphanNodes(remove_orphan_inputs);
      nnet->RemoveOrphanComponents();
    } else if (directive == "set-learning-rate") {
      std::string name_pattern = "*";
      // name_pattern defaults to '*' if none is given.  This pattern
      // matches names of components, not nodes.
      config_line.GetValue("name", &name_pattern);
      BaseFloat learning_rate = -1;
      if (!config_line.GetValue("learning-rate", &learning_rate)) {
        KALDI_ERR << "In edits-config, expected learning-rate to be set in line: "
                  << config_line.WholeLine();
      }
      // Note: the learning rate you provide will be multiplied by any
      // 'learning-rate-factor' that is defined in the component,
      // so if you call SetUnderlyingLearningRate(), the actual learning
      // rate (learning_rate_) is set to the value you provide times
      // learning_rate_factor_.
      UpdatableComponent *component = NULL;
      int32 num_learning_rates_set = 0;
      for (int32 c = 0; c < nnet->NumComponents(); c++) {
        if (NameMatchesPattern(nnet->GetComponentName(c).c_str(),
                               name_pattern.c_str()) &&
            (component =
             dynamic_cast<UpdatableComponent*>(nnet->GetComponent(c)))) {
          component->SetUnderlyingLearningRate(learning_rate);
          num_learning_rates_set++;
        }
      }
      KALDI_LOG << "Set learning rates for " << num_learning_rates_set << " components.";
    } else if (directive == "set-learning-rate-factor") {
      std::string name_pattern = "*";
      // name_pattern defaults to '*' if none is given.
      config_line.GetValue("name", &name_pattern);
      BaseFloat learning_rate_factor = -1;
      if (!config_line.GetValue("learning-rate-factor", &learning_rate_factor)) {
        KALDI_ERR << "In edits-config, expected learning-rate-factor to be set in line: "
                  << config_line.WholeLine();
      }
      // Note: the learning_rate_factor_  defined in the component
      // sets to the value you provided, so if you call SetUnderlyingLearningRate(),
      // the actual learning rate (learning_rate_) is set to the value you provided
      // times learning_rate.
      UpdatableComponent *component = NULL;
      int32 num_learning_rate_factors_set = 0;
      for (int32 c = 0; c < nnet->NumComponents(); c++) {
        if (NameMatchesPattern(nnet->GetComponentName(c).c_str(),
            name_pattern.c_str()) &&
            (component =
            dynamic_cast<UpdatableComponent*>(nnet->GetComponent(c)))) {
          component->SetLearningRateFactor(learning_rate_factor);
          num_learning_rate_factors_set++;
        }
      }
      KALDI_LOG << "Set learning rate factors for " << num_learning_rate_factors_set
                << " components.";
    } else if (directive == "rename-node") {
      // this is a shallow renaming of a node, and it requires that the name used is
      // not the name of another node.
      std::string old_name, new_name;
      if (!config_line.GetValue("old-name", &old_name) ||
          !config_line.GetValue("new-name", &new_name) ||
          config_line.HasUnusedValues()) {
        KALDI_ERR << "In edits-config, could not make sense of this rename-node "
                  << "directive (expect old-name=xxx new-name=xxx) "
                  << config_line.WholeLine();
      }
      if (nnet->GetNodeIndex(old_name) < 0)
        KALDI_ERR << "Could not rename node from " << old_name << " to "
                  << new_name << " because there is no node called "
                  << old_name;
      // further checks will happen inside SetNodeName().
      nnet->SetNodeName(nnet->GetNodeIndex(old_name), new_name);
    } else if (directive == "remove-output-nodes") {
      // note: after remove-output-nodes you probably want to do 'remove-orphans'.
      std::string name_pattern;
      if (!config_line.GetValue("name", &name_pattern) ||
          config_line.HasUnusedValues())
        KALDI_ERR << "In edits-config, could not make sense of "
                  << "remove-output-nodes directive: "
                  << config_line.WholeLine();
      std::vector<int32> nodes_to_remove;
      int32 outputs_remaining = 0;
      for (int32 n = 0; n < nnet->NumNodes(); n++) {
        if (nnet->IsOutputNode(n)) {
          if (NameMatchesPattern(nnet->GetNodeName(n).c_str(),
                                 name_pattern.c_str()))
            nodes_to_remove.push_back(n);
          else
            outputs_remaining++;
        }
      }
      KALDI_LOG << "Removing " << nodes_to_remove.size() << " output nodes.";
      if (outputs_remaining == 0)
        KALDI_ERR << "All outputs were removed.";
      nnet->RemoveSomeNodes(nodes_to_remove);
    } else if (directive == "set-dropout-proportion") {
      std::string name_pattern = "*";
      // name_pattern defaults to '*' if none is given.  This pattern
      // matches names of components, not nodes.
      config_line.GetValue("name", &name_pattern);
      BaseFloat proportion = -1;
      if (!config_line.GetValue("proportion", &proportion)) {
        KALDI_ERR << "In edits-config, expected proportion to be set in line: "
                  << config_line.WholeLine();
      }
      int32 num_dropout_proportions_set = 0;
      for (int32 c = 0; c < nnet->NumComponents(); c++) {
        if (NameMatchesPattern(nnet->GetComponentName(c).c_str(),
                               name_pattern.c_str())) {
          DropoutComponent *dropout_component =
             dynamic_cast<DropoutComponent*>(nnet->GetComponent(c));
          DropoutMaskComponent *mask_component =
             dynamic_cast<DropoutMaskComponent*>(nnet->GetComponent(c));
          GeneralDropoutComponent *general_dropout_component =
             dynamic_cast<GeneralDropoutComponent*>(nnet->GetComponent(c));
          if (dropout_component != NULL) {
            dropout_component->SetDropoutProportion(proportion);
            num_dropout_proportions_set++;
          } else if (mask_component != NULL){
            mask_component->SetDropoutProportion(proportion);
            num_dropout_proportions_set++;
          } else if (general_dropout_component != NULL){
            general_dropout_component->SetDropoutProportion(proportion);
            num_dropout_proportions_set++;
          }
        }
      }
      KALDI_LOG << "Set dropout proportions for "
                << num_dropout_proportions_set << " components.";
    } else if (directive == "apply-svd") {
      std::string name_pattern;
      int32 bottleneck_dim = -1;
      if (!config_line.GetValue("name", &name_pattern) ||
          !config_line.GetValue("bottleneck-dim", &bottleneck_dim))
        KALDI_ERR << "Edit directive apply-svd requires 'name' and "
            "'bottleneck-dim' to be specified.";
      if (bottleneck_dim <= 0)
        KALDI_ERR << "Bottleneck-dim must be positive in apply-svd command.";
      SvdApplier applier(name_pattern, bottleneck_dim, nnet);
      applier.ApplySvd();
    } else if (directive == "reduce-rank") {
      std::string name_pattern;
      int32 rank = -1;
      if (!config_line.GetValue("name", &name_pattern) ||
          !config_line.GetValue("rank", &rank))
        KALDI_ERR << "Edit directive reduce-rank requires 'name' and "
            "'rank' to be specified.";
      if (rank <= 0)
        KALDI_ERR << "Rank must be positive in reduce-rank command.";
      ReduceRankOfComponents(name_pattern, rank, nnet);
    } else {
      KALDI_ERR << "Directive '" << directive << "' is not currently "
          "supported (reading edit-config).";
    }
    if (config_line.HasUnusedValues()) {
      KALDI_ERR << "Could not interpret '" << config_line.UnusedValues()
                << "' in edit config line " << config_line.WholeLine();
    }
  }
}


/// Returns true if 'nnet' has some kind of recurrency.
bool NnetIsRecurrent(const Nnet &nnet) {
  std::vector<std::vector<int32> > graph;
  NnetToDirectedGraph(nnet, &graph);
  return GraphHasCycles(graph);
}

class ModelCollapser {
 public:
  ModelCollapser(const CollapseModelConfig &config,
                 Nnet *nnet):
      config_(config), nnet_(nnet) { }
  void Collapse() {
    bool changed = true;
    int32 num_nodes = nnet_->NumNodes(),
        num_iters = 0;
    int32 num_components1 = nnet_->NumComponents();
    for (; changed; num_iters++) {
      changed = false;
      for (int32 n = 0; n < num_nodes; n++)
        if (OptimizeNode(n))
          changed = true;
      // we shouldn't iterate more than a couple of times.
      if (num_iters >= 10)
        KALDI_ERR << "Something went wrong collapsing model.";
    }
    int32 num_components2 = nnet_->NumComponents();
    nnet_->RemoveOrphanNodes();
    nnet_->RemoveOrphanComponents();
    int32 num_components3 = nnet_->NumComponents();
    if (num_components2 != num_components1 ||
        num_components3 != num_components2)
      KALDI_LOG << "Added " << (num_components2 - num_components1)
                << " components, removed "
                << (num_components2 - num_components3);
  }
 private:
  /**
     This function tries to collapse two successive components, where
     the component 'component_index1' appears as the input of 'component_index2'.
     If the two components can be collapsed in that way, it returns the index
     of a combined component.

     Note: in addition to the two components simply being chained together, this
     function supports the case where different time-offsets of the first
     component are appendend together as the input of the second component.
     So the input-dim of the second component may be a multiple of
     the output-dim of the first component.

     The function returns the component-index of a (newly created or existing)
     component that combines both of these components, if it's possible to
     combine them; or it returns -1 if it's not possible.
   */
  int32 CollapseComponents(int32 component_index1,
                           int32 component_index2) {
    int32 ans;
    if (config_.collapse_dropout &&
        (ans = CollapseComponentsDropout(component_index1,
                                         component_index2)) != -1)
      return ans;
    if (config_.collapse_batchnorm &&
        (ans = CollapseComponentsBatchnorm(component_index1,
                                           component_index2)) != -1)
      return ans;
    if (config_.collapse_affine &&
        (ans = CollapseComponentsAffine(component_index1,
                                        component_index2)) != -1)
      return ans;
    if (config_.collapse_scale &&
        (ans = CollapseComponentsScale(component_index1,
                                       component_index2)) != -1)
      return ans;
    return -1;
  }


  // If the SumDescriptor has exactly one part that is either a
  // SimpleForwardingDescriptor or an OffsetForwardingDescriptor containing a
  // SimpleForwardingDescriptor, returns the node-index that the
  // SimpleForwardingDescriptor contains.  Otherwise returns -1.
  //
  // E.g. of the SumDescriptor represents something like "foo" it returns
  // the index for "foo"; if it represents "Offset(foo, -2)" it returns
  // the index for "foo"; if it represents something else like
  // "Sum(foo, bar)" or "IfDefined(foo)", then it returns -1.
  int32 SumDescriptorIsCollapsible(const SumDescriptor &sum_desc) {
    // I don't much like having to use dynamic_cast here.
    const SimpleSumDescriptor *ss = dynamic_cast<const SimpleSumDescriptor*>(
        &sum_desc);
    if (ss == NULL) return -1;
    const ForwardingDescriptor *fd = &(ss->Src());
    const OffsetForwardingDescriptor *od =
        dynamic_cast<const OffsetForwardingDescriptor*>(fd);
    if (od != NULL)
      fd = &(od->Src());
    const SimpleForwardingDescriptor *sd =
        dynamic_cast<const SimpleForwardingDescriptor*>(fd);
    if (sd == NULL) return -1;
    else {
      // the following is a rather roundabout way to get the node-index from a
      // SimpleForwardingDescriptor, but it works (it avoids adding other stuff
      // to the interface).
      std::vector<int32> v;
      sd->GetNodeDependencies(&v);
      int32 node_index = v[0];
      return node_index;
    }
  }

  // If the Descriptor is a sum over different offsets of a particular node,
  // e.g. something of the form "Sum(Offset(foo, -2), Offset(foo, 2))" or in the
  // most degenerate case just "foo", then this function returns the index for
  // foo; otherwise it returns -1.
  int32 DescriptorIsCollapsible(const Descriptor &desc) {
    int32 ans = SumDescriptorIsCollapsible(desc.Part(0));
    for (int32 i = 1; i < desc.NumParts(); i++) {
      if (ans != -1) {
        int32 node_index = SumDescriptorIsCollapsible(desc.Part(i));
        if (node_index != ans)
          ans = -1;
      }
    }
    // note: ans is only >= 0 if the answers from all parts of
    // the SumDescriptors were >=0 and identical to each other.
    // Otherwise it will be -1.
    return ans;
  }

  // Replaces all the nodes with index 'node_to_replace' in 'src' with the
  // descriptor 'expr', and returns the appropriately modified Descriptor.  For
  // example, if 'src' is 'Append(Offset(foo, -1), Offset(foo, 1))' and 'expr'
  // is 'Offset(bar, -1)', this should give you: 'Append(Offset(bar, -2), bar)'.
  Descriptor ReplaceNodeInDescriptor(const Descriptor &src,
                                     int32 node_to_replace,
                                     const Descriptor &expr) {
    // The way we replace it is at the textual level: we create a "fake" vector
    // of node-names where the printed form of 'expr' appears as the
    // node name in node_names[node_to_replace]; we print the descriptor
    // in 'src' using that faked node-names vector; and we parse it again
    // using the real node-names vector.
    std::vector<std::string> node_names = nnet_->GetNodeNames();
    std::ostringstream expr_os;
    expr.WriteConfig(expr_os, node_names);
    node_names[node_to_replace] = expr_os.str();
    std::ostringstream src_replaced_os;
    src.WriteConfig(src_replaced_os, node_names);
    std::vector<std::string> tokens;
    // now, in the example, src_replaced_os.str() would equal
    //  Append(Offset(Offset(bar, -1), -1), Offset(Offset(bar, -1), 1)).
    bool b = DescriptorTokenize(src_replaced_os.str(),
                                  &tokens);
    KALDI_ASSERT(b);
    // 'tokens' might now contain something like [ "Append", "(", "Offset", ..., ")" ].
    tokens.push_back("end of input");
    const std::string *next_token = &(tokens[0]);
    Descriptor ans;
    // parse using the un-modified node names.
    ans.Parse(nnet_->GetNodeNames(), &next_token);
    KALDI_ASSERT(*next_token == "end of input");
    // Note: normalization of expressions in Descriptors, such as conversion of
    // Offset(Offset(bar, -1), -1) to Offset(bar, -2), takes place inside the
    // Descriptor parsing code.
    return ans;
  }



  /**
     This function modifies the neural network in the case where 'node_index' is a
     component-input node whose component (in the node at 'node_index + 1),
     if a bunch of other conditions also apply.

     First, he descriptor in the node at 'node_index' has to have
     a certain limited structure, e.g.:
        - the input-descriptor is a component-node name like 'foo' or:
        - the input-descriptor is a combination of Append and/or and Offset
          expressions, like:
           'Append(Offset(foo, -3), foo, Offset(foo, 3))',
          referring to only a single node 'foo'.

     ALSO the components need to be collapsible by the function
     CollapseComponents(), which will only be possible for certain pairs of
     component types (like, say, a dropout node preceding an affine or
     convolutional node); see that function for details.

     This function will (if it does anything), modify the node to replace the
     component at 'node_index + 1' with a newly created component that combines
     the two components involved.
     It will also modify the node at 'node_index' by
     replacing its Descriptor with a modified input descriptor, so that if the
     input-descriptor of node 'foo' was 'bar', the descriptor for our node would
     now look like:
        'Append(Offset(bar, -3), bar, Offset(bar, 3))'...
     and note that 'bar' itself doesn't have to be just a node-name, it can
     be a more general expression.
     This function returns true if it changed something in the neural net, and false
     otherwise.
   */
  bool OptimizeNode(int32 node_index) {
    NetworkNode &descriptor_node = nnet_->GetNode(node_index);
    if (descriptor_node.node_type != kDescriptor ||
        node_index + 1 >= nnet_->NumNodes())
      return false;
    NetworkNode &component_node = nnet_->GetNode(node_index + 1);
    if (component_node.node_type != kComponent)
      return false;
    Descriptor &descriptor = descriptor_node.descriptor;
    int32 component_index = component_node.u.component_index;

    int32 input_node_index = DescriptorIsCollapsible(descriptor);
    if (input_node_index == -1)
      return false;  // do nothing, the expression in the Descriptor is too
                     // general for this code to handle.
    const NetworkNode &input_node = nnet_->GetNode(input_node_index);
    if (input_node.node_type != kComponent)
      return false;
    int32 input_component_index = input_node.u.component_index;
    int32 combined_component_index = CollapseComponents(input_component_index,
                                                        component_index);
    if (combined_component_index == -1)
      return false;  // these components were not of types that can be
                     // collapsed.
    component_node.u.component_index = combined_component_index;

    // 'input_descriptor_node' is the input descriptor of the component
    // that's the input to the node in "node_index".  (e.g. the component for
    // the node "foo" in our example above).
    const NetworkNode &input_descriptor_node = nnet_->GetNode(input_node_index - 1);
    const Descriptor &input_descriptor = input_descriptor_node.descriptor;

    // The next statement replaces the descriptor in the network node with one
    // in which the component 'input_component_index' has been replaced with its
    // input, thus bypassing the component in 'input_component_index'.
    // We'll later remove that component and its node from the network, if
    // needed by RemoveOrphanNodes() and RemoveOrphanComponents().
    descriptor = ReplaceNodeInDescriptor(descriptor,
                                         input_node_index,
                                         input_descriptor);
    return true;
  }


  /**
     Tries to produce a component that's equivalent to running the component
     'component_index2' with input given by 'component_index1'.  This handles
     the case where 'component_index1' is of type DropoutComponent or
     GeneralDropoutComponent, and where 'component_index2' is of type
     AffineComponent, NaturalGradientAffineComponent or
     TimeHeightConvolutionComponent.

     Returns -1 if this code can't produce a combined component (normally
     because the components have the wrong types).
   */
  int32 CollapseComponentsDropout(int32 component_index1,
                                  int32 component_index2) {
    const DropoutComponent *dropout_component =
        dynamic_cast<const DropoutComponent*>(
            nnet_->GetComponent(component_index1));
    const GeneralDropoutComponent *general_dropout_component =
        dynamic_cast<const GeneralDropoutComponent*>(
            nnet_->GetComponent(component_index1));

    if (dropout_component == NULL && general_dropout_component == NULL)
      return -1;
    BaseFloat scale;  // the scale we have to apply to correct for removing
                      // this dropout comonent.
    if (dropout_component != NULL) {
      BaseFloat dropout_proportion = dropout_component->DropoutProportion();
      scale = 1.0 / (1.0 - dropout_proportion);
    } else {
      // for GeneralDropoutComponent, it's done in such a way that the expectation
      // is always 1.  (When it's nonzero, we give it a value 1/(1-dropout_proportion).
      // So no scaling is needed.
      scale = 1.0;
    }
    // note: if the 2nd component is not of a type that we can scale, the
    // following function call will return -1, which is OK.
    return GetScaledComponentIndex(component_index2,
                                   scale);
  }



  /**
     Tries to produce a component that's equivalent to running the component
     'component_index2' with input given by 'component_index1'.  This handles
     the case where 'component_index1' is of type BatchnormComponent, and where
     'component_index2' is of type AffineComponent or
     NaturalGradientAffineComponent.

     Returns -1 if this code can't produce a combined component (normally
     because the components have the wrong types).
   */
  int32 CollapseComponentsBatchnorm(int32 component_index1,
                                    int32 component_index2) {
    const BatchNormComponent *batchnorm_component =
        dynamic_cast<const BatchNormComponent*>(
            nnet_->GetComponent(component_index1));
    if (batchnorm_component == NULL)
      return -1;

    if (batchnorm_component->Offset().Dim() == 0) {
      KALDI_ERR << "Expected batch-norm components to have test-mode set.";
    }
    std::string batchnorm_component_name = nnet_->GetComponentName(
        component_index1);
    return GetDiagonallyPreModifiedComponentIndex(batchnorm_component->Offset(),
                                                  batchnorm_component->Scale(),
                                                  batchnorm_component_name,
                                                  component_index2);
  }


  /**
     Tries to produce a component that's equivalent to running the component
     'component_index2' with input given by 'component_index1'.  This handles
     the case where 'component_index1' is of type FixedAffineComponent,
     AffineComponent or NaturalGradientAffineComponent, and 'component_index2'
     is of type AffineComponent or NaturalGradientAffineComponent.

     Returns -1 if this code can't produce a combined component.
   */
  int32 CollapseComponentsAffine(int32 component_index1,
                                 int32 component_index2) {

    const FixedAffineComponent *fixed_affine_component1 =
        dynamic_cast<const FixedAffineComponent*>(
            nnet_->GetComponent(component_index1));
    const AffineComponent *affine_component1 =
        dynamic_cast<const AffineComponent*>(
            nnet_->GetComponent(component_index1)),
        *affine_component2 =
        dynamic_cast<const AffineComponent*>(
            nnet_->GetComponent(component_index2));
    if (affine_component2 == NULL ||
        (fixed_affine_component1 == NULL && affine_component1 == NULL))
      return -1;

    std::ostringstream new_component_name_os;
    new_component_name_os << nnet_->GetComponentName(component_index1)
                          << "." << nnet_->GetComponentName(component_index2);
    std::string new_component_name = new_component_name_os.str();
    int32 new_component_index = nnet_->GetComponentIndex(new_component_name);
    if (new_component_index >= 0)
      return new_component_index;  // we previously created this.

    const CuMatrix<BaseFloat> *linear_params1;
    const CuVector<BaseFloat> *bias_params1;
    if (fixed_affine_component1 != NULL) {
      if (fixed_affine_component1->InputDim() >
          fixed_affine_component1->OutputDim()) {
        // first affine component is dimension-reducing, so combining the two
        // might be inefficient.
        return -1;
      }
      linear_params1 = &(fixed_affine_component1->LinearParams());
      bias_params1 = &(fixed_affine_component1->BiasParams());
    } else {
      if (affine_component1->InputDim() >
          affine_component1->OutputDim()) {
        // first affine component is dimension-reducing, so combining the two
        // might be inefficient.
        return -1;
      }
      linear_params1 = &(affine_component1->LinearParams());
      bias_params1 = &(affine_component1->BiasParams());
    }

    int32 input_dim1 = linear_params1->NumCols(),
        output_dim1 = linear_params1->NumRows(),
        input_dim2 = affine_component2->InputDim(),
        output_dim2 = affine_component2->OutputDim();
    KALDI_ASSERT(input_dim2 % output_dim1 == 0);
    // with typical configurations for TDNNs, like Append(-3, 0, 3) [in xconfigs], a.k.a.
    // Append(Offset(foo, -3), foo, Offset(foo, 3)), the first component's output may
    // be smaller than the second component's input.  We construct a single
    // transform with a block-diagonal structure in this case.
    int32 multiple = input_dim2 / output_dim1;
    CuVector<BaseFloat> bias_params1_full(input_dim2);
    CuMatrix<BaseFloat> linear_params1_full(input_dim2,
                                            multiple * input_dim1);
    for (int32 i = 0; i < multiple; i++) {
      bias_params1_full.Range(i * output_dim1,
                              output_dim1).CopyFromVec(*bias_params1);
      linear_params1_full.Range(i * output_dim1, output_dim1,
                                i * input_dim1, input_dim1).CopyFromMat(
                                    *linear_params1);
    }
    const CuVector<BaseFloat> &bias_params2 = affine_component2->BiasParams();
    const CuMatrix<BaseFloat> &linear_params2 = affine_component2->LinearParams();

    int32 new_input_dim = multiple * input_dim1,
        new_output_dim = output_dim2;
    CuMatrix<BaseFloat> new_linear_params(new_output_dim,
                                          new_input_dim);
    CuVector<BaseFloat> new_bias_params(bias_params2);
    new_bias_params.AddMatVec(1.0, linear_params2, kNoTrans,
                              bias_params1_full, 1.0);
    new_linear_params.AddMatMat(1.0, linear_params2, kNoTrans,
                                linear_params1_full, kNoTrans, 0.0);

    AffineComponent *new_component = new AffineComponent();
    new_component->Init(new_input_dim, new_output_dim, 0.0, 0.0);
    new_component->SetParams(new_bias_params, new_linear_params);
    return nnet_->AddComponent(new_component_name, new_component);
  }



  /**
     Tries to produce a component that's equivalent to running the component
     'component_index2' with input given by 'component_index1'.  This handles
     the case where 'component_index1' is of type AffineComponent or
     NaturalGradientAffineComponent, and 'component_index2' is of type
     FixedScaleComponent, and the output dim of the first is the same as the
     input dim of the second.  This situation is common in output layers.  Later
     if it's needed, we could easily enable the code to support
     PerElementScaleComponent.

     Returns -1 if this code can't produce a combined component.
   */
  int32 CollapseComponentsScale(int32 component_index1,
                                int32 component_index2) {

    const AffineComponent *affine_component1 =
        dynamic_cast<const AffineComponent*>(
            nnet_->GetComponent(component_index1));
    const FixedScaleComponent *fixed_scale_component2 =
        dynamic_cast<const FixedScaleComponent*>(
                    nnet_->GetComponent(component_index2));
    if (affine_component1 == NULL ||
        fixed_scale_component2 == NULL ||
        affine_component1->OutputDim() !=
        fixed_scale_component2->InputDim())
      return -1;

    std::ostringstream new_component_name_os;
    new_component_name_os << nnet_->GetComponentName(component_index1)
                          << "." << nnet_->GetComponentName(component_index2);
    std::string new_component_name = new_component_name_os.str();
    int32 new_component_index = nnet_->GetComponentIndex(new_component_name);
    if (new_component_index >= 0)
      return new_component_index;  // we previously created this.

    CuMatrix<BaseFloat> linear_params(affine_component1->LinearParams());
    CuVector<BaseFloat> bias_params(affine_component1->BiasParams());
    const CuVector<BaseFloat> &scales = fixed_scale_component2->Scales();

    bias_params.MulElements(scales);
    linear_params.MulRowsVec(scales);

    AffineComponent *new_affine_component =
        dynamic_cast<AffineComponent*>(affine_component1->Copy());
    new_affine_component->SetParams(bias_params, linear_params);
    return nnet_->AddComponent(new_component_name,
                               new_affine_component);
  }


  /**
     This function finds, or creates, a component which is like
     'component_index' but is combined with a diagonal offset-and-scale
     transform *before* the component.  (We may later create a function called
     GetDiagonallyPostModifiedComponentIndex if we need to apply the
     transform *after* the component.

     This function doesn't work for convolutional components, because
     due to zero-padding, it's not possible to represent an offset/scale
     on the input filters via changes in the convolutional parameters.
     [the scale, yes; but we don't bother doing that.]

     This may require modifying its linear and
     bias parameters.

     @param [in] offset   The offset term 'b' in the diagnonal transform
                          y = a x + b.
     @param [in] scale    The scale term 'a' in the diagnonal transform
                          y = a x + b.  Must have the same dimension as
                          'offset'.
     @param [in] src_identifier   A string that uniquely identifies 'offset'
                          and 'scale'.  In practice it will be the component-index
                          from where 'offset' and 'scale' were taken.

     @param [in] component_index  The component to be modified (not in-place, but
                 as a copy).  The component described in 'component_index' must
                 be AffineComponent or NaturalGradientAffineComponent, and
                 case the dimension of 'offset'/'scale' should divide the
                 component input dimension, otherwise it's an error.
                      of 'offset' and 'scale' should equal 'scale_input'
                      (else it's an error).
     @return  Returns the component-index of a suitably modified component.
              If one like this already exists, the existing one will be returned.
              If the component in 'component_index' was not of a type that can
              be modified in this way, returns -1.

   */
  int32 GetDiagonallyPreModifiedComponentIndex(
      const CuVectorBase<BaseFloat> &offset,
      const CuVectorBase<BaseFloat> &scale,
      const std::string &src_identifier,
      int32 component_index) {
    int32 transform_dim = offset.Dim();
    KALDI_ASSERT(offset.Dim() > 0 && offset.Dim() == scale.Dim());
    if (offset.Max() == 0.0 && offset.Min() == 0.0 &&
        scale.Max() == 1.0 && scale.Min() == 1.0)
      return component_index;  // identity transform.
    std::ostringstream new_component_name_os;
    new_component_name_os << src_identifier
                          << "."
                          << nnet_->GetComponentName(component_index);
    std::string new_component_name = new_component_name_os.str();
    int32 new_component_index = nnet_->GetComponentIndex(new_component_name);
    if (new_component_index >= 0)
      return new_component_index;  // we previously created this.
    const Component *component = nnet_->GetComponent(component_index);
    const AffineComponent *affine_component =
        dynamic_cast<const AffineComponent*>(component);
    if (affine_component == NULL)
      return -1;  // we can't do this.


    int32 input_dim = affine_component->InputDim();
    if (input_dim % transform_dim != 0) {
      KALDI_ERR << "Dimension mismatch when modifying affine component.";
    }
    // 'full_offset' and 'full_scale' may be repeated versions of
    // 'offset' and 'scale' in case input_dim > transform_dim.
    CuVector<BaseFloat> full_offset(input_dim),
        full_scale(input_dim);
    for (int32 d = 0; d < input_dim; d += transform_dim) {
      full_offset.Range(d, transform_dim).CopyFromVec(offset);
      full_scale.Range(d, transform_dim).CopyFromVec(scale);
    }
    CuVector<BaseFloat> bias_params(affine_component->BiasParams());
    CuMatrix<BaseFloat> linear_params(affine_component->LinearParams());
    // Image the affine component does y = a x + b, and by applying
    // the pre-transform we are replacing x with s x + o
    // s for scale and o for offset), so we have:
    //  y = a s x + (b + a o).
    // do: b += a o.
    bias_params.AddMatVec(1.0, linear_params, kNoTrans, full_offset, 1.0);
    // do: a = a * s.
    linear_params.MulColsVec(full_scale);
    AffineComponent *new_affine_component =
        dynamic_cast<AffineComponent*>(affine_component->Copy());
    new_affine_component->SetParams(bias_params, linear_params);
    return nnet_->AddComponent(new_component_name, new_affine_component);
  }


  /**
      Given a component 'component_index', returns a component which
      will give the same output as the current component gives when its input
      is scaled by 'scale'.   This will generally mean applying
      the scale to the linear parameters in the component, if it is
      an affine or convolutional component.

      If the component referred to in 'component_index' is not an
      affine or convolutional component, and therefore cannot
      be scaled (by this code), then this function returns -1.
  */
  int32 GetScaledComponentIndex(int32 component_index,
                                BaseFloat scale) {
    if (scale == 1.0)
      return component_index;
    std::ostringstream os;
    os << nnet_->GetComponentName(component_index)
       << ".scale" << std::setprecision(3) << scale;
    std::string new_component_name = os.str();  // e.g. foo.s2.0
    int32 ans = nnet_->GetComponentIndex(new_component_name);
    if (ans >= 0)
      return ans;  // one already exists, no need to create it.
    const Component *current_component = nnet_->GetComponent(component_index);
    const AffineComponent *affine_component =
        dynamic_cast<const AffineComponent*>(current_component);
    const TimeHeightConvolutionComponent *conv_component =
        dynamic_cast<const TimeHeightConvolutionComponent*>(current_component);
    if (affine_component != NULL) {
      // AffineComponent or NaturalGradientAffineComponent.
      CuVector<BaseFloat> bias_params(affine_component->BiasParams());
      CuMatrix<BaseFloat> linear_params(affine_component->LinearParams());
      linear_params.Scale(scale);
      AffineComponent *new_affine_component =
          dynamic_cast<AffineComponent*>(current_component->Copy());
      new_affine_component->SetParams(bias_params, linear_params);
      return nnet_->AddComponent(new_component_name, new_affine_component);
    } else if (conv_component != NULL) {
      TimeHeightConvolutionComponent *new_conv_component =
          dynamic_cast<TimeHeightConvolutionComponent*>(
              current_component->Copy());
      // scale the linear but not the bias parameters.
      new_conv_component->ScaleLinearParams(scale);
      return nnet_->AddComponent(new_component_name, new_conv_component);
    } else {
      // We can't scale this component (at least, not using this code).
      return -1;
    }
  }

  const CollapseModelConfig &config_;
  Nnet *nnet_;
};


void CollapseModel(const CollapseModelConfig &config,
                   Nnet *nnet) {
  ModelCollapser c(config, nnet);
  c.Collapse();
}

bool UpdateNnetWithMaxChange(const Nnet &delta_nnet,
                             BaseFloat max_param_change,
                             BaseFloat max_change_scale,
                             BaseFloat scale, Nnet *nnet,
                             std::vector<int32> *
                             num_max_change_per_component_applied,
                             int32 *num_max_change_global_applied) {
  KALDI_ASSERT(nnet != NULL);
  // computes scaling factors for per-component max-change
  const int32 num_updatable = NumUpdatableComponents(delta_nnet);
  Vector<BaseFloat> scale_factors = Vector<BaseFloat>(num_updatable);
  BaseFloat param_delta_squared = 0.0;
  int32 num_max_change_per_component_applied_per_minibatch = 0;
  BaseFloat min_scale = 1.0;
  std::string component_name_with_min_scale;
  BaseFloat max_change_with_min_scale;
  int32 i = 0;
  for (int32 c = 0; c < delta_nnet.NumComponents(); c++) {
    const Component *comp = delta_nnet.GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      const UpdatableComponent *uc =
          dynamic_cast<const UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
                  << "UpdatableComponent; change this code.";
      BaseFloat max_param_change_per_comp = uc->MaxChange();
      KALDI_ASSERT(max_param_change_per_comp >= 0.0);
      BaseFloat dot_prod = uc->DotProduct(*uc);
      if (max_param_change_per_comp != 0.0 &&
          std::sqrt(dot_prod) * std::abs(scale) >
          max_param_change_per_comp * max_change_scale) {
        scale_factors(i) = max_param_change_per_comp * max_change_scale /
            (std::sqrt(dot_prod) * std::abs(scale));
        (*num_max_change_per_component_applied)[i]++;
        num_max_change_per_component_applied_per_minibatch++;
        KALDI_VLOG(2) << "Parameters in " << delta_nnet.GetComponentName(c)
                      << " change too big: " << std::sqrt(dot_prod) << " * "
                      << scale << " > " << "max-change * max-change-scale="
                      << max_param_change_per_comp << " * " << max_change_scale
                      << ", scaling by " << scale_factors(i);
      } else {
        scale_factors(i) = 1.0;
      }
      if (i == 0 || scale_factors(i) < min_scale) {
        min_scale =  scale_factors(i);
        component_name_with_min_scale = delta_nnet.GetComponentName(c);
        max_change_with_min_scale = max_param_change_per_comp;
      }
      param_delta_squared += std::pow(scale_factors(i),
                                      static_cast<BaseFloat>(2.0)) * dot_prod;
      i++;
    }
  }
  KALDI_ASSERT(i == scale_factors.Dim());
  BaseFloat param_delta = std::sqrt(param_delta_squared);
  // computes the scale for global max-change
  param_delta *= std::abs(scale);
  if (max_param_change != 0.0) {
    if (param_delta > max_param_change * max_change_scale) {
      if (param_delta - param_delta != 0.0) {
        KALDI_WARN << "Infinite parameter change, will not apply.";
        return false;
      } else {
        scale *= max_param_change * max_change_scale / param_delta;
        (*num_max_change_global_applied)++;
      }
    }
  }
  if ((max_param_change != 0.0 &&
      param_delta > max_param_change * max_change_scale &&
      param_delta - param_delta == 0.0) || min_scale < 1.0) {
    std::ostringstream ostr;
    if (min_scale < 1.0)
      ostr << "Per-component max-change active on "
           << num_max_change_per_component_applied_per_minibatch
           << " / " << num_updatable << " Updatable Components."
           << "(smallest factor=" << min_scale << " on "
           << component_name_with_min_scale
           << " with max-change=" << max_change_with_min_scale <<"). ";
    if (param_delta > max_param_change * max_change_scale)
      ostr << "Global max-change factor was "
           << max_param_change * max_change_scale / param_delta
           << " with max-change=" << max_param_change << ".";
    KALDI_LOG << ostr.str();
  }
  // applies both of the max-change scalings all at once, component by component
  // and updates parameters
  scale_factors.Scale(scale);
  AddNnetComponents(delta_nnet, scale_factors, scale, nnet);
  return true;
}

int32 GetNumNvalues(const std::vector<NnetIo> &io_vec,
                   bool exhaustive) {
  int32 num_n_values = -1;
  for (size_t i = 0; i < io_vec.size(); i++) {
    const NnetIo &io = io_vec[i];
    int32 this_num_n_values;
    const std::vector<Index> &index_vec = io.indexes;
    KALDI_ASSERT(!index_vec.empty() &&
                 "Empty input or output in ComputationRequest?");
    if (exhaustive) {
      int32 lowest_n_value = std::numeric_limits<int32>::max(),
          highest_n_value = std::numeric_limits<int32>::min();
      std::vector<Index>::const_iterator
          iter = index_vec.begin(), end = index_vec.end();
      for (; iter != end; ++iter) {
        int32 n = iter->n;
        if (n < lowest_n_value) { lowest_n_value = n; }
        if (n > highest_n_value) { highest_n_value = n; }
      }
      this_num_n_values = highest_n_value + 1 - lowest_n_value;
    } else {
      // we assume that the 'n' values range from zero to N-1,
      // where N is the number of distinct 'n' values.
      this_num_n_values = index_vec.back().n + 1;
    }
    if (num_n_values == -1) {
      num_n_values = this_num_n_values;
    } else {
      if (num_n_values != this_num_n_values) {
        KALDI_ERR << "Different inputs/outputs of ComputationRequest have "
            "different numbers of n values: " << num_n_values
                  << " vs. " << this_num_n_values;
      }
    }
  }
  if (!exhaustive && RandInt(0, 100) == 0) {
    int32 num_n_values_check = GetNumNvalues(io_vec, true);
    if (num_n_values != num_n_values_check) {
      KALDI_ERR << "Exhaustive and quick checks returned different "
          "answers: " << num_n_values << " vs. "
                << num_n_values_check;
    }
  }
  return num_n_values;
}

void ApplyL2Regularization(const Nnet &nnet,
                           BaseFloat l2_regularize_scale,
                           Nnet *delta_nnet) {
  if (l2_regularize_scale == 0.0)
    return;
  for (int32 c = 0; c < nnet.NumComponents(); c++) {
    const Component *src_component_in = nnet.GetComponent(c);
    if (src_component_in->Properties() & kUpdatableComponent) {
      const UpdatableComponent *src_component =
          dynamic_cast<const UpdatableComponent*>(src_component_in);
      UpdatableComponent *dest_component =
          dynamic_cast<UpdatableComponent*>(delta_nnet->GetComponent(c));
      // The following code will segfault if they aren't both updatable, which
      // would be a bug in the calling code.
      BaseFloat lrate = dest_component->LearningRate(),
          l2_regularize = dest_component->L2Regularization();
      KALDI_ASSERT(lrate >= 0 && l2_regularize >= 0);
      BaseFloat scale = -2.0 * l2_regularize_scale * lrate * l2_regularize;
      if (scale != 0.0)
        dest_component->Add(scale, *src_component);
    }
  }
}


} // namespace nnet3
} // namespace kaldi
