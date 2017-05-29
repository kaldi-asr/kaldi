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

#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-graph.h"
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-general-component.h"
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
  int32 window_size = 100;
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



} // namespace nnet3
} // namespace kaldi
