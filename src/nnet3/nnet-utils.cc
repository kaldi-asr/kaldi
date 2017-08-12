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
  int32 window_size = 150;
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
  }
}

bool HasBatchnorm(const Nnet &nnet) {
  for (int32 c = 0; c < nnet.NumComponents(); c++) {
    const Component *comp = nnet.GetComponent(c);
    const BatchNormComponent *bc =
        dynamic_cast<const BatchNormComponent*>(comp);
    if (bc != NULL)
      return true;
  }
  return false;
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
      KALDI_LOG << "Set learning rates for " << num_learning_rates_set << " nodes.";
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
          if (dropout_component != NULL) {
            dropout_component->SetDropoutProportion(proportion);
            num_dropout_proportions_set++;
          } else if (mask_component != NULL){
            mask_component->SetDropoutProportion(proportion);
            num_dropout_proportions_set++;
          }
        }
      }
      KALDI_LOG << "Set dropout proportions for "
                << num_dropout_proportions_set << " components.";
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
     the case where 'component_index1' is of type DropoutComponent, and where
     'component_index2' is of type AffineComponent,
     NaturalGradientAffineComponent or TimeHeightConvolutionComponent.

     Returns -1 if this code can't produce a combined component (normally
     because the components have the wrong types).
   */
  int32 CollapseComponentsDropout(int32 component_index1,
                                  int32 component_index2) {
    const DropoutComponent *dropout_component =
        dynamic_cast<const DropoutComponent*>(
            nnet_->GetComponent(component_index1));
    if (dropout_component == NULL)
      return -1;
    BaseFloat dropout_proportion = dropout_component->DropoutProportion();
    BaseFloat scale = 1.0 / (1.0 - dropout_proportion);
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
  std::string info_before_collapse;
  if (GetVerboseLevel() >= 4)
    info_before_collapse = nnet->Info();
  c.Collapse();
  if (GetVerboseLevel() >= 4) {
    std::string info_after_collapse = nnet->Info();
    if (info_after_collapse != info_before_collapse) {
      KALDI_VLOG(4) << "Collapsing model: info before collapse was: "
                    << info_before_collapse
                    << ", info after collapse was:"
                    << info_after_collapse;
    }
  }
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

} // namespace nnet3
} // namespace kaldi
