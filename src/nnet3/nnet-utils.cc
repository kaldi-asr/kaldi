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
#include "nnet3/nnet-simple-component.h"

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

void CheckStateIoForStatePreserving(const Nnet &nnet) {
  const std::string suffix = "_STATE_PREVIOUS_MINIBATCH";
  const std::vector<std::string> &node_names = nnet.GetNodeNames();
  for (int32 n = 0; n < node_names.size(); n++) {
    const std::string node_name = node_names[n];
    if (node_name.size() >= suffix.size() &&
        node_name.substr(node_name.size() - suffix.size()) == suffix) {
      const std::string node_name_no_suffix = 
          node_name.substr(0, node_name.size() - suffix.size());
      int32 node_index = nnet.GetNodeIndex(node_name_no_suffix);
      if (node_index == -1 || !nnet.IsInputNode(n) ||
          !nnet.IsOutputNode(node_index))
        KALDI_ERR << "The name \"" << node_name_no_suffix << "\" with suffix \""
                  << "_STATE_PREVIOUS_MINIBATCH\" is reserved only for input "
                  << "node. Further such an input node should be accompanied "
                  << "by an output node of the name \"" << node_name_no_suffix
                  << "\". These nodes are used to implement state preserving "
                  << "RNN training.";
    }
  }
}

void GetRecurrentOutputNodeNames(const Nnet &nnet,
                                 std::vector<std::string>
                                 *recurrent_output_names,
                                 std::vector<std::string>
                                 *recurrent_node_names) {
  recurrent_output_names->clear();
  recurrent_node_names->clear();
  const std::string suffix = "_STATE_PREVIOUS_MINIBATCH";
  for (int32 i = 0; i < nnet.NumNodes(); i++) {
    const std::string node_name = nnet.GetNodeName(i);
    int32 suffixed_node_index = nnet.GetNodeIndex(node_name + suffix);
    if (nnet.IsOutputNode(i) && node_name != "output" &&
        suffixed_node_index > -1 && nnet.IsInputNode(suffixed_node_index)) {
      recurrent_output_names->push_back(node_name);
      std::vector<int32> node_indexes;
      nnet.GetNode(i).descriptor.GetNodeDependencies(&node_indexes);
      KALDI_ASSERT(node_indexes.size() == 1);
      recurrent_node_names->push_back(nnet.GetNodeName(node_indexes[0]));
    }
  }
}

void GetRecurrentNodeOffsets(const Nnet &nnet,
                             const std::vector<std::string>
                             &recurrent_node_names,
                             std::vector<int32> *recurrent_offsets) {
  const std::vector<std::string> &node_names = nnet.GetNodeNames();
  recurrent_offsets->clear();
  recurrent_offsets->resize(recurrent_node_names.size(), 0);
  for (size_t i = 0; i < recurrent_node_names.size(); i++) {
    int32 node_index = nnet.GetNodeIndex(recurrent_node_names[i]);
    if (node_index == -1)
      continue;
    for (size_t n = 0; n < nnet.NumNodes(); n++) {
      const NetworkNode &node = nnet.GetNode(n);
      // skip output node to circumvent the problem when a recurrent output 
      // is "label delayed" and thus also contain keyword "Offset", which is not
      // what we are trying to look at for our "offset"  
      if (node.node_type == kDescriptor && !nnet.IsOutputNode(n)) {
        std::ostringstream ostr;
        node.descriptor.WriteConfig(ostr, node_names);
        std::vector<std::string> tokens;
        DescriptorTokenize(ostr.str(), &tokens);
        tokens.push_back("end of input");
        const std::string *next_token = &(tokens[0]);
        GeneralDescriptor *gen_desc = GeneralDescriptor::Parse(node_names,
                                                               &next_token);
        if (*next_token != "end of input")
          KALDI_ERR << "Parsing Descriptor, expected end of input but got "
                    << "'" <<  *next_token << "'";
        int32 offset = 0;
        GetDescriptorRecurrentNodeOffset(*gen_desc, node_index, &offset);
        delete gen_desc;
        if (std::abs(offset) > std::abs((*recurrent_offsets)[i]))
          (*recurrent_offsets)[i] = offset;
      }
    }
  }
}

bool GetDescriptorRecurrentNodeOffset(const GeneralDescriptor &gen_desc,
                                      int32 recurrent_node_index, int32 *offset) {
  switch (gen_desc.descriptor_type_) {
    case GeneralDescriptor::kAppend: case GeneralDescriptor::kSum:
    case GeneralDescriptor::kFailover: case GeneralDescriptor::kIfDefined:
    case GeneralDescriptor::kSwitch: case GeneralDescriptor::kRound:
    case GeneralDescriptor::kReplaceIndex: {
      bool node_found = false;
      for (size_t i = 0; i < gen_desc.descriptors_.size(); i++) {
        int32 inner_offset = 0;
        bool inner_node_found =
            GetDescriptorRecurrentNodeOffset(*(gen_desc.descriptors_[i]),
                recurrent_node_index, &inner_offset);
        node_found = node_found || inner_node_found;
        if (inner_node_found && std::abs(inner_offset) > std::abs(*offset))
          *offset = inner_offset;
      }
      return node_found;
    }
    case GeneralDescriptor::kOffset: {
      KALDI_ASSERT(gen_desc.descriptors_.size() == 1);
      int32 inner_offset = 0;
      bool node_found =
          GetDescriptorRecurrentNodeOffset(*(gen_desc.descriptors_[0]),
              recurrent_node_index, &inner_offset);
      if (node_found)
        *offset = inner_offset + gen_desc.value1_;
      return node_found;
    }
    case GeneralDescriptor::kNodeName: {
      return gen_desc.value1_ == recurrent_node_index;
    }
    default:
      KALDI_ERR << "Invalid descriptor type.";
  }
  return false;
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
  std::vector<std::string> recurrent_output_names;
  std::vector<std::string> recurrent_node_names;
  GetRecurrentOutputNodeNames(nnet, &recurrent_output_names,
                              &recurrent_node_names);
  // if there was just one input (excluding recurrent inputs), then it was named
  // "input" and everything checks out.
  if (NumInputNodes(nnet) == 1 + recurrent_output_names.size())
    return true;
  // Otherwise, there should be 2 inputs and one
  // should be called "ivector".
  return NumInputNodes(nnet) == 2 + recurrent_output_names.size() &&
      nnet.GetNodeIndex("ivector") != -1 &&
      nnet.IsInputNode(nnet.GetNodeIndex("ivector"));
}

void EvaluateComputationRequest(
    const Nnet &nnet,
    const ComputationRequest &request,
    std::vector<std::vector<bool> > *is_computable) {
  ComputationGraph graph;
  ComputationGraphBuilder builder(nnet, request, &graph);
  builder.Compute();
  builder.GetComputableInfo(is_computable);
  if (GetVerboseLevel() >= 2) {
    std::ostringstream graph_pretty;
    graph.Print(graph_pretty, nnet.GetNodeNames());
    KALDI_VLOG(2) << "Graph is " << graph_pretty.str();
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

  // extract recurrent output names from nnet 
  std::vector<std::string> recurrent_output_names;
  std::vector<std::string> recurrent_node_names; 
  GetRecurrentOutputNodeNames(nnet, &recurrent_output_names,
                              &recurrent_node_names);

  // create IoSpecification for all recurrent inputs
  std::vector<IoSpecification> r_inputs;
  for (int32 i = 0; i < nnet.NumNodes(); i++) {
    std::string node_name = nnet.GetNodeName(i);
    if (nnet.IsInputNode(i) && node_name != "input" && node_name != "ivector")
      for (int32 j = 0; j < recurrent_output_names.size(); j++) {
        if (recurrent_output_names[j] + "_STATE_PREVIOUS_MINIBATCH" 
            == node_name) {
          // note that here IoSpecification::indexes is empty
          r_inputs.push_back(IoSpecification(node_name, 0, 0));
          break;
        }
      }
  }
  // create IoSpecification for all recurrent outputs
  std::vector<IoSpecification> r_outputs;
  for (int32 j = 0; j < recurrent_output_names.size(); j++)
    // note that here IoSpecification::indexes is empty
    r_outputs.push_back(IoSpecification(recurrent_output_names[j], 0, 0));

  int32 n = rand() % 10;
  // in the IoSpecification for now we we will request all the same indexes at
  // output that we requested at input.
  for (int32 t = input_start; t < input_end; t++) {
    input.indexes.push_back(Index(n, t));
    output.indexes.push_back(Index(n, t));
    //for (int32 i = 0; i < r_inputs.size(); i++)
      //r_inputs[i].indexes.push_back(Index(n, t));
    for (int32 i = 0; i < r_outputs.size(); i++)
      r_outputs[i].indexes.push_back(Index(n, t));
  }
  for (int32 t = input_start - 100; t < input_end + 100; t++) {// an ugly hack
    for (int32 i = 0; i < r_inputs.size(); i++)
      r_inputs[i].indexes.push_back(Index(n, t));
  }
  // the assumption here is that the network just requires the ivector at time
  // t=0.
  ivector.indexes.push_back(Index(n, 0));

  ComputationRequest request;
  request.inputs.push_back(input);
  request.outputs.push_back(output);
  if (nnet.GetNodeIndex("ivector") != -1)
    request.inputs.push_back(ivector);
  // add recurrent inputs to request
  for (int32 i = 0; i < r_inputs.size(); i++)
    request.inputs.push_back(r_inputs[i]);
  // add recurrent outputs to request
  for (int32 i = 0; i < r_outputs.size(); i++)
    request.outputs.push_back(r_outputs[i]);

  std::vector<std::vector<bool> > computable;
  EvaluateComputationRequest(nnet, request, &computable);

  *left_context = 0;
  *right_context = 0;
  for (int32 i = 0; i < computable.size(); i++) {
    std::vector<bool> &output_ok = computable[i];
    std::vector<bool>::iterator iter =
        std::find(output_ok.begin(), output_ok.end(), true);
    int32 first_ok = iter - output_ok.begin();
    int32 first_not_ok = std::find(iter, output_ok.end(), false) -
        output_ok.begin();
    if (first_ok == window_size || first_not_ok <= first_ok)
      KALDI_ERR << "No outputs were computable (perhaps not a simple nnet?)";
    *left_context = std::max(first_ok, *left_context);
    *right_context = std::max(window_size - first_not_ok, *right_context);
  }
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

void SetZero(bool is_gradient,
             Nnet *nnet) {
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *comp = nnet->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      UpdatableComponent *u_comp = dynamic_cast<UpdatableComponent*>(comp);
      KALDI_ASSERT(u_comp != NULL);
      u_comp->SetZero(is_gradient);
    }
  }
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

void ScaleLearningRate(BaseFloat learning_rate_scale,
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
      uc->SetActualLearningRate(uc->LearningRate() * learning_rate_scale);
    }
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

void SetLearningRates(const Vector<BaseFloat> &learning_rates,
                     Nnet *nnet) {
  int32 i = 0;
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *comp = nnet->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      // For now all updatable components inherit from class UpdatableComponent.
      // If that changes in future, we will change this code.
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
            "UpdatableComponent; change this code.";
      KALDI_ASSERT(i < learning_rates.Dim());
      uc->SetActualLearningRate(learning_rates(i++));
    }
  }
  KALDI_ASSERT(i == learning_rates.Dim());
}

void GetLearningRates(const Nnet &nnet, 
                      Vector<BaseFloat> *learning_rates) {
  learning_rates->Resize(NumUpdatableComponents(nnet));
  int32 i = 0;
  for (int32 c = 0; c < nnet.NumComponents(); c++) {
    const Component *comp = nnet.GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      // For now all updatable components inherit from class UpdatableComponent.
      // If that changes in future, we will change this code.
      const UpdatableComponent *uc = dynamic_cast<const UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
            "UpdatableComponent; change this code.";
      (*learning_rates)(i++) = uc->LearningRate();
    }
  }
  KALDI_ASSERT(i == learning_rates->Dim());
}

void ScaleNnetComponents(const Vector<BaseFloat> &scale_factors,
                         Nnet *nnet) {
  int32 i = 0;
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    Component *comp = nnet->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      // For now all updatable components inherit from class UpdatableComponent.
      // If that changes in future, we will change this code.
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
            "UpdatableComponent; change this code.";
      KALDI_ASSERT(i < scale_factors.Dim());
      uc->Scale(scale_factors(i++));
    }
  }
  KALDI_ASSERT(i == scale_factors.Dim());
}

void ScaleNnet(BaseFloat scale, Nnet *nnet) {
  if (scale == 1.0) return;
  else if (scale == 0.0) {
    SetZero(false, nnet);
  } else {
    for (int32 c = 0; c < nnet->NumComponents(); c++) {
      Component *comp = nnet->GetComponent(c);
      comp->Scale(scale);
    }
  }
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


} // namespace nnet3
} // namespace kaldi
