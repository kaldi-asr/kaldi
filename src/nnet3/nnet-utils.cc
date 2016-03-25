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

int32 GetInputInterval(const Nnet &nnet, std::string input_name) {
  const std::vector<std::string> &node_names = nnet.GetNodeNames();
  int32 interval = -1;
  for (int32 i = 0; i < nnet.NumNodes(); i++) {
    const NetworkNode &node = nnet.GetNode(i);
    if (node.node_type == kDescriptor) {
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
      int32 new_interval =
          GetInputIntervalInternal(*gen_desc, node_names, input_name);
      if (new_interval != -1) {
        if (new_interval == 0) {
          if (interval > 0) {
            // if input is used only at t=0 from the arg we are looking at, and
            // is used from previous args at some interval > 0, set interval to
            // 1 as a "hard" case
            interval = 1;
          } else {
            // if input is used only at t=0 from the arg we are looking at, and
            // input has not been used from the previous args) or was used only
            // at t=0, set interval to 0
            interval = 0;
          }
        } else if (interval == 0) {
          // if input is used at some interval > 0 from the arg we are looking
          // at, and is used only at t=0 from previous args, set interval to 1
          // as a "hard" case
          interval = 1;
        } else if (interval == -1) {
          // if input is used at some interval > 0 from the arg we are looking
          // at, and is not used from previous args, set interval to that
          // interval
          interval = new_interval;
        } else {
          // if input is used at some interval > 0 from the arg we are looking
          // at and is also used from prvious args at another interval > 0, set
          // interval to GCD of the two
          interval = Gcd(interval, new_interval);
        }
      } 
      delete gen_desc;
    }
  }
  return interval;
}

int32 GetInputIntervalInternal(const GeneralDescriptor &gen_desc,
                               const std::vector<std::string> &node_names,
                               const std::string &input_name) {
  switch (gen_desc.descriptor_type_) {
    case GeneralDescriptor::kAppend: case GeneralDescriptor::kSum:
    case GeneralDescriptor::kFailover: case GeneralDescriptor::kIfDefined:
    case GeneralDescriptor::kSwitch: {
      int32 interval = -1;
      for (size_t i = 0; i < gen_desc.descriptors_.size(); i++) {
        int32 new_interval =
            GetInputIntervalInternal(*(gen_desc.descriptors_[i]), node_names,
                                     input_name);
        if (new_interval != -1) {
          if (new_interval == 0) {
            if (interval > 0) {
              // if input is used only at t=0 from the arg we are looking at,
              // and is used from previous args at some interval > 0, set
              // interval to 1 as a "hard" case
              interval = 1;
            } else {
              // if input is used only at t=0 from the arg we are looking at,
              // and input has not been used from the previous args) or was used
              // only at t=0, set interval to 0
              interval = 0;
            }
          } else if (interval == 0) {
            // if input is used at some interval > 0 from the arg we are looking
            // at, and is used only at t=0 from previous args, set interval to 1
            // as a "hard" case
            interval = 1;
          } else if (interval == -1) {
            // if input is used at some interval > 0 from the arg we are looking
            // at, and is not used from previous args, set interval to that
            // interval
            interval = new_interval;
          } else {
            // if input is used at some interval > 0 from the arg we are looking
            // at and is also used from prvious args at another interval > 0,
            // set interval to GCD of the two
            interval = Gcd(interval, new_interval);
          }
        }
      }
      return interval;
    }
    case GeneralDescriptor::kOffset: {
      KALDI_ASSERT(gen_desc.descriptors_.size() == 1);
      int32 interval =
          GetInputIntervalInternal(*(gen_desc.descriptors_[0]), node_names,
                                   input_name);
      if (interval == 0) {
        // in the case where input is used only at t=0 from the arg of "Offset",
        // if offset by 0, return 0; otherwise return 1 as a "hard" case
        return gen_desc.value1_ == 0 ? 0 : 1;
      } else if (interval > 0 && gen_desc.value1_ % interval != 0) {
        // if input is used from the arg at some interval > 0 and offset by
        // other than multiple of interval, return 1 as a "hard" case
        return 1;
      } else {
        // if input is used from the arg at some interval > 0 and offset by a
        // multiple of interval, or input is not used from the arg, just keep
        // interval 
        return interval;
      }
    }
    case GeneralDescriptor::kRound: {
      KALDI_ASSERT(gen_desc.descriptors_.size() == 1);
      int32 interval =
          GetInputIntervalInternal(*(gen_desc.descriptors_[0]), node_names,
                                   input_name);
      if (interval > 0)
        if (gen_desc.value1_ >= interval) {
          // if t is rounded with a larger modulus than interval from the arg,
          // we will instead have t at all multiples of that larger number
          return gen_desc.value1_;
        } else {
          if (interval % gen_desc.value1_ == 0) {
            // if the modulus is smaller than interval from the arg,
            // and interval is a multiple of the modulus, we will still have
            // interval
            return interval;
          } else {
            // if the modulus is smaller than interval from the arg,
            // and interval is not a multiple of the modulus,  we will no
            // longer have t at all multiples of either number of the two, then
            // return 1 as a "hard" case
            return 1;
          }
        } else {
        // if interval <= 0, just keep interval as "Round" has no effect on it
        return interval;
      }
    }
    case GeneralDescriptor::kReplaceIndex: {
      KALDI_ASSERT(gen_desc.descriptors_.size() == 1);
      int32 interval =
          GetInputIntervalInternal(*(gen_desc.descriptors_[0]), node_names,
                                   input_name);
      if (gen_desc.value1_ == int32(ReplaceIndexForwardingDescriptor::kT) &&
          interval >= 0) {
        // in the case where the input is used at least once from the arg of
        // "ReplaceIndex" and the replacement takes place on t, if replace t
        // with 0, then return 0; otherwise, i.e. if replace t with other
        // values, return 1 as a "hard" case 
        return gen_desc.value2_ == 0 ? 0 : 1;
      } else {
        // if replacement is on x, or input is not used from the arg,
        // just keep interval
        return interval;
      }
    }
    case GeneralDescriptor::kNodeName: {
      // if the input is used as NodeName, return 1, otherwise -1 
      if (node_names[gen_desc.value1_] == input_name)
        return 1;
      else
        return -1;
    }
    default:
      KALDI_ERR << "Invalid descriptor type.";
      return -1;
  }
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
  // Otherwise, there should be 2 inputs and one
  // should be called "ivector".
  return NumInputNodes(nnet) == 2 &&
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

  int32 n = rand() % 10;
  // in the IoSpecification for now we we will request all the same indexes at
  // output that we requested at input.
  for (int32 t = input_start; t < input_end; t++) {
    input.indexes.push_back(Index(n, t));
    output.indexes.push_back(Index(n, t));
  }
  // push the indexes for ivector(s)
  int32 ivector_period = GetInputInterval(nnet,"ivector");
  if (ivector_period == 0)
    // single ivector for the entire chunk
    // the assumption here is that the network just requires the ivector at time
    // t=0
    ivector.indexes.push_back(Index(n, 0));
  else {
    // multiple ivectors, so push multiple indexes
    int32 t_begin = std::floor(1.0 * input_start / ivector_period);
    int32 t_end = std::floor((input_end - 1.0) / ivector_period);
    for (int32 t = t_begin; t <= t_end; t++)
      ivector.indexes.push_back(Index(n, t * ivector_period));
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
  ostr << "ivector-period: " << GetInputInterval(nnet, "ivector") << "\n";
  ostr << "output-dim: " << nnet.OutputDim("output") << "\n";
  ostr << "# Nnet info follows.\n";
  ostr << nnet.Info();
  return ostr.str();
}


} // namespace nnet3
} // namespace kaldi
