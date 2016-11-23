// nnet3/nnet-nnet.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2016  Daniel Galvez
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
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-parse.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-simple-component.h"

namespace kaldi {
namespace nnet3 {

// returns dimension that this node outputs.
int32 NetworkNode::Dim(const Nnet &nnet) const {
  int32 ans;
  switch (node_type) {
    case kInput: case kDimRange:
      ans = dim;
      break;
    case kDescriptor:
      ans = descriptor.Dim(nnet);
      break;
    case kComponent:
      ans = nnet.GetComponent(u.component_index)->OutputDim();
      break;
    default:
      ans = 0;  // suppress compiler warning
      KALDI_ERR << "Invalid node type.";
  }
  KALDI_ASSERT(ans > 0);
  return ans;
}

void Nnet::SetNodeName(int32 node_index, const std::string &new_name) {
  if (!(static_cast<size_t>(node_index) < nodes_.size()))
    KALDI_ERR << "Invalid node index";
  if (GetNodeIndex(new_name) != -1)
    KALDI_ERR << "You cannot rename a node to create a duplicate node name";
  if (!IsValidName(new_name))
    KALDI_ERR << "Node name " << new_name << " is not allowed.";
  node_names_[node_index] = new_name;
}

const std::vector<std::string> &Nnet::GetNodeNames() const {
  return node_names_;
}

const std::vector<std::string> &Nnet::GetComponentNames() const {
  return component_names_;
}

std::string Nnet::GetAsConfigLine(int32 node_index, bool include_dim) const {
  std::ostringstream ans;
  KALDI_ASSERT(node_index < nodes_.size() &&
               nodes_.size() == node_names_.size());
  const NetworkNode &node = nodes_[node_index];
  const std::string &name = node_names_[node_index];
  switch (node.node_type) {
    case kInput:
      ans << "input-node name=" << name << " dim=" << node.dim;
      break;
    case kDescriptor:
      // assert that it's an output-descriptor, not one describing the input to
      // a component-node.
      KALDI_ASSERT(IsOutputNode(node_index));
      ans << "output-node name=" << name << " input=";
      node.descriptor.WriteConfig(ans, node_names_);
      if (include_dim)
        ans << " dim=" << node.Dim(*this);
      ans << " objective=" << (node.u.objective_type == kLinear ? "linear" :
                               "quadratic");
      break;
    case kComponent:
      ans << "component-node name=" << name << " component="
          << component_names_[node.u.component_index] << " input=";
      KALDI_ASSERT(nodes_[node_index-1].node_type == kDescriptor);
      nodes_[node_index-1].descriptor.WriteConfig(ans, node_names_);
      if (include_dim)
        ans << " input-dim=" << nodes_[node_index-1].Dim(*this)
            << " output-dim=" << node.Dim(*this);
      break;
    case kDimRange:
      ans << "dim-range-node name=" << name << " input-node="
          << node_names_[node.u.node_index] << " dim-offset="
          << node.dim_offset << " dim=" << node.dim;
      break;
    default:
      KALDI_ERR << "Unknown node type.";
  }
  return ans.str();
}

bool Nnet::IsOutputNode(int32 node) const {
  int32 size = nodes_.size();
  KALDI_ASSERT(node >= 0 && node < size);
  return (nodes_[node].node_type == kDescriptor &&
          (node + 1 == size ||
           nodes_[node + 1].node_type != kComponent));
}

bool Nnet::IsInputNode(int32 node) const {
  int32 size = nodes_.size();
  KALDI_ASSERT(node >= 0 && node < size);
  return (nodes_[node].node_type == kInput);
}

bool Nnet::IsDescriptorNode(int32 node) const {
  int32 size = nodes_.size();
  KALDI_ASSERT(node >= 0 && node < size);
  return (nodes_[node].node_type == kDescriptor);
}

bool Nnet::IsComponentNode(int32 node) const {
  int32 size = nodes_.size();
  KALDI_ASSERT(node >= 0 && node < size);
  return (nodes_[node].node_type == kComponent);
}

bool Nnet::IsDimRangeNode(int32 node) const {
  int32 size = nodes_.size();
  KALDI_ASSERT(node >= 0 && node < size);
  return (nodes_[node].node_type == kDimRange);
}


const Component *Nnet::GetComponent(int32 c) const {
  KALDI_ASSERT(static_cast<size_t>(c) < components_.size());
  return components_[c];
}

Component *Nnet::GetComponent(int32 c) {
  KALDI_ASSERT(static_cast<size_t>(c) < components_.size());
  return components_[c];
}

void Nnet::SetComponent(int32 c, Component *component) {
  KALDI_ASSERT(static_cast<size_t>(c) < components_.size());
  delete components_[c];
  components_[c] = component;
}

/// Returns true if this is component-input node, i.e. a node of type kDescriptor
/// that immediately precedes a node of type kComponent.
bool Nnet::IsComponentInputNode(int32 node) const {
  int32 size = nodes_.size();
  KALDI_ASSERT(node >= 0 && node < size);
  return (node + 1 < size &&
          nodes_[node].node_type == kDescriptor &&
          nodes_[node+1].node_type == kComponent);
}

void Nnet::GetConfigLines(bool include_dim,
                          std::vector<std::string> *config_lines) const {
  config_lines->clear();
  for (int32 n = 0; n < NumNodes(); n++)
    if (!IsComponentInputNode(n))
      config_lines->push_back(GetAsConfigLine(n, include_dim));

}

void Nnet::ReadConfig(std::istream &config_is) {

  std::vector<std::string> lines;
  // Write into "lines" a config file corresponding to whatever
  // nodes we currently have.  Because the numbering of nodes may
  // change, it's most convenient to convert to the text representation
  // and combine the existing and new config lines in that representation.
  const bool include_dim = false;
  GetConfigLines(include_dim, &lines);

  // we'll later regenerate what we need from nodes_ and node_name_ from the
  // string representation.
  nodes_.clear();
  node_names_.clear();

  int32 num_lines_initial = lines.size();

  ReadConfigLines(config_is, &lines);
  // now "lines" will have comments removed and empty lines stripped out

  std::vector<ConfigLine> config_lines(lines.size());

  ParseConfigLines(lines, &config_lines);

  // the next line will possibly remove some elements from "config_lines" so no
  // node or component is doubly defined, always keeping the second repeat.
  // Things being doubly defined can happen when a previously existing node or
  // component is redefined in a new config file.
  RemoveRedundantConfigLines(num_lines_initial, &config_lines);

  int32 initial_num_components = components_.size();
  for (int32 pass = 0; pass <= 1; pass++) {
    for (size_t i = 0; i < config_lines.size(); i++) {
      const std::string &first_token = config_lines[i].FirstToken();
      if (first_token == "component") {
        if (pass == 0)
          ProcessComponentConfigLine(initial_num_components,
                                     &(config_lines[i]));
      } else if (first_token == "component-node") {
        ProcessComponentNodeConfigLine(pass,  &(config_lines[i]));
      } else if (first_token == "input-node") {
        if (pass == 0)
          ProcessInputNodeConfigLine(&(config_lines[i]));
      } else if (first_token == "output-node") {
        ProcessOutputNodeConfigLine(pass, &(config_lines[i]));
      } else if (first_token == "dim-range-node") {
        ProcessDimRangeNodeConfigLine(pass, &(config_lines[i]));
      } else {
        KALDI_ERR << "Invalid config-file line ('" << first_token
                  << "' not expected): " << config_lines[i].WholeLine();
      }
    }
  }
  Check();
}


// called only on pass 0 of ReadConfig.
void Nnet::ProcessComponentConfigLine(
    int32 initial_num_components,
    ConfigLine *config) {
  std::string name, type;
  if (!config->GetValue("name", &name))
    KALDI_ERR << "Expected field name=<component-name> in config line: "
              << config->WholeLine();
  if (!IsToken(name)) // e.g. contains a space.
    KALDI_ERR << "Component name '" << name << "' is not allowed, in line: "
              << config->WholeLine();
  if (!config->GetValue("type", &type))
    KALDI_ERR << "Expected field type=<component-type> in config line: "
              << config->WholeLine();
  Component *new_component = Component::NewComponentOfType(type);
  if (new_component == NULL)
    KALDI_ERR << "Unknown component-type '" << type
              << "' in config file.  Check your code version and config.";
  // the next call will call KALDI_ERR or KALDI_ASSERT and die if something
  // went wrong.
  new_component->InitFromConfig(config);
  int32 index = GetComponentIndex(name);
  if (index != -1) {  // Replacing existing component.
    if (index >= initial_num_components) {
      // that index was something we added from this config.
      KALDI_ERR << "You are adding two components with the same name: '"
                << name << "'";
    }
    delete components_[index];
    components_[index] = new_component;
  } else {
    components_.push_back(new_component);
    component_names_.push_back(name);
  }
  if (config->HasUnusedValues())
    KALDI_ERR << "Unused values '" << config->UnusedValues()
              << " in config line: " << config->WholeLine();
}


void Nnet::ProcessComponentNodeConfigLine(
    int32 pass,
    ConfigLine *config) {

  std::string name;
  if (!config->GetValue("name", &name))
    KALDI_ERR << "Expected field name=<component-name> in config line: "
              << config->WholeLine();

  std::string input_name = name + std::string("_input");
  int32 input_node_index = GetNodeIndex(input_name),
      node_index = GetNodeIndex(name);

  if (pass == 0) {
    KALDI_ASSERT(input_node_index == -1 && node_index == -1);
    // just set up the node types and names for now, we'll properly set them up
    // on pass 1.
    nodes_.push_back(NetworkNode(kDescriptor));
    nodes_.push_back(NetworkNode(kComponent));
    node_names_.push_back(input_name);
    node_names_.push_back(name);
    return;
  } else {
    KALDI_ASSERT(input_node_index != -1 && node_index == input_node_index + 1);
    std::string component_name, input_descriptor;
    if (!config->GetValue("component", &component_name))
      KALDI_ERR << "Expected component=<component-name>, in config line: "
                << config->WholeLine();
    int32 component_index = GetComponentIndex(component_name);
    if (component_index == -1)
      KALDI_ERR << "No component named '" << component_name
                << "', in config line: " << config->WholeLine();
    nodes_[node_index].u.component_index = component_index;

    if (!config->GetValue("input", &input_descriptor))
      KALDI_ERR << "Expected input=<input-descriptor>, in config line: "
                << config->WholeLine();
    std::vector<std::string> tokens;
    if (!DescriptorTokenize(input_descriptor, &tokens))
      KALDI_ERR << "Error tokenizing descriptor in config line "
                << config->WholeLine();
    std::vector<std::string> node_names_temp;
    GetSomeNodeNames(&node_names_temp);
    tokens.push_back("end of input");
    const std::string *next_token = &(tokens[0]);
    if (!nodes_[input_node_index].descriptor.Parse(node_names_temp,
                                                   &next_token))
      KALDI_ERR << "Error parsing Descriptor in config line: "
                << config->WholeLine();
    if (config->HasUnusedValues())
      KALDI_ERR << "Unused values '" << config->UnusedValues()
                << " in config line: " << config->WholeLine();
  }
}

// called only on pass 0 of ReadConfig.
void Nnet::ProcessInputNodeConfigLine(
    ConfigLine *config) {
  std::string name;
  if (!config->GetValue("name", &name))
    KALDI_ERR << "Expected field name=<input-name> in config line: "
              << config->WholeLine();
  int32 dim;
  if (!config->GetValue("dim", &dim))
    KALDI_ERR << "Expected field dim=<input-dim> in config line: "
              << config->WholeLine();

  if (config->HasUnusedValues())
    KALDI_ERR << "Unused values '" << config->UnusedValues()
              << " in config line: " << config->WholeLine();

  KALDI_ASSERT(GetNodeIndex(name) == -1);
  if (dim <= 0)
    KALDI_ERR << "Invalid dimension in config line: " << config->WholeLine();

  int32 node_index = nodes_.size();
  nodes_.push_back(NetworkNode(kInput));
  nodes_[node_index].dim = dim;
  node_names_.push_back(name);
}


void Nnet::ProcessOutputNodeConfigLine(
    int32 pass,
    ConfigLine *config) {
  std::string name;
  if (!config->GetValue("name", &name))
    KALDI_ERR << "Expected field name=<input-name> in config line: "
              << config->WholeLine();
  int32 node_index = GetNodeIndex(name);
  if (pass == 0) {
    KALDI_ASSERT(node_index == -1);
    nodes_.push_back(NetworkNode(kDescriptor));
    node_names_.push_back(name);
  } else {
    KALDI_ASSERT(node_index != -1);
    std::string input_descriptor;
    if (!config->GetValue("input", &input_descriptor))
      KALDI_ERR << "Expected input=<input-descriptor>, in config line: "
                << config->WholeLine();
    std::vector<std::string> tokens;
    if (!DescriptorTokenize(input_descriptor, &tokens))
      KALDI_ERR << "Error tokenizing descriptor in config line "
                << config->WholeLine();
    tokens.push_back("end of input");
    // if the following fails it will die.
    std::vector<std::string> node_names_temp;
    GetSomeNodeNames(&node_names_temp);
    const std::string *next_token = &(tokens[0]);
    if (!nodes_[node_index].descriptor.Parse(node_names_temp, &next_token))
      KALDI_ERR << "Error parsing descriptor (input=...) in config line "
                << config->WholeLine();
    std::string objective_type;
    if (config->GetValue("objective", &objective_type)) {
      if (objective_type == "linear") {
        nodes_[node_index].u.objective_type = kLinear;
      } else if (objective_type == "quadratic") {
        nodes_[node_index].u.objective_type = kQuadratic;
      } else {
        KALDI_ERR << "Invalid objective type: " << objective_type;
      }
    } else {
      // the default objective type is linear.  This is what we use
      // for softmax objectives; the LogSoftmaxLayer is included as the
      // last layer, in this case.
      nodes_[node_index].u.objective_type = kLinear;
    }
    if (config->HasUnusedValues())
      KALDI_ERR << "Unused values '" << config->UnusedValues()
                << " in config line: " << config->WholeLine();
  }
}


void Nnet::ProcessDimRangeNodeConfigLine(
    int32 pass,
    ConfigLine *config) {
  std::string name;
  if (!config->GetValue("name", &name))
    KALDI_ERR << "Expected field name=<input-name> in config line: "
              << config->WholeLine();
  int32 node_index = GetNodeIndex(name);
  if (pass == 0) {
    KALDI_ASSERT(node_index == -1);
    nodes_.push_back(NetworkNode(kDimRange));
    node_names_.push_back(name);
  } else {
    KALDI_ASSERT(node_index != -1);
    std::string input_node_name;
    if (!config->GetValue("input-node", &input_node_name))
      KALDI_ERR << "Expected input-node=<input-node-name>, in config line: "
                << config->WholeLine();
    int32 dim, dim_offset;
    if (!config->GetValue("dim", &dim))
      KALDI_ERR << "Expected dim=<feature-dim>, in config line: "
                << config->WholeLine();
    if (!config->GetValue("dim-offset", &dim_offset))
      KALDI_ERR << "Expected dim-offset=<dimension-offset>, in config line: "
                << config->WholeLine();

    int32 input_node_index = GetNodeIndex(input_node_name);
    if (input_node_index == -1 ||
        !(nodes_[input_node_index].node_type == kComponent ||
          nodes_[input_node_index].node_type == kInput))
      KALDI_ERR << "invalid input-node " << input_node_name
                << ": " << config->WholeLine();

    if (config->HasUnusedValues())
      KALDI_ERR << "Unused values '" << config->UnusedValues()
                << " in config line: " << config->WholeLine();

    NetworkNode &node = nodes_[node_index];
    KALDI_ASSERT(node.node_type == kDimRange);
    node.u.node_index = input_node_index;
    node.dim = dim;
    node.dim_offset = dim_offset;
  }
}


int32 Nnet::GetNodeIndex(const std::string &node_name) const {
  size_t size = node_names_.size();
  for (size_t i = 0; i < size; i++)
    if (node_names_[i] == node_name)
      return static_cast<int32>(i);
  return -1;
}

int32 Nnet::GetComponentIndex(const std::string &component_name) const {
  size_t size = component_names_.size();
  for (size_t i = 0; i < size; i++)
    if (component_names_[i] == component_name)
      return static_cast<int32>(i);
  return -1;
}


// note: the input to this function is a config generated from the nnet,
// containing the node info, concatenated with a config provided by the user.
//static
void Nnet::RemoveRedundantConfigLines(int32 num_lines_initial,
                                      std::vector<ConfigLine> *config_lines) {
  int32 num_lines = config_lines->size();
  KALDI_ASSERT(num_lines_initial <= num_lines);
  // node names and component names live in different namespaces.
  unordered_map<std::string, int32, StringHasher> node_name_to_most_recent_line;
  unordered_set<std::string, StringHasher> component_names;
  typedef unordered_map<std::string, int32, StringHasher>::iterator IterType;

  std::vector<bool> to_remove(num_lines, false);
  for (int32 line = 0; line < num_lines; line++) {
    ConfigLine &config_line = (*config_lines)[line];
    std::string name;
    if (!config_line.GetValue("name", &name))
      KALDI_ERR << "Config line has no field 'name=xxx': "
                << config_line.WholeLine();
    if (!IsValidName(name))
      KALDI_ERR << "Name '" << name << "' is not allowable, in line: "
                << config_line.WholeLine();
    if (config_line.FirstToken() == "component") {
      // a line starting with "component"... components live in their own
      // namespace.  No repeats are allowed because we never wrote them
      // to the config generated from the nnet.
      if (!component_names.insert(name).second) {
        // we could not insert it because it was already there.
        KALDI_ERR << "Component name " << name
                  << " appears twice in the same config file.";
      }
    } else {
      // the line defines some sort of network node, e.g. component-node.
      IterType iter = node_name_to_most_recent_line.find(name);
      if (iter != node_name_to_most_recent_line.end()) {
        // name is repeated.
        int32 prev_line = iter->second;
        if (prev_line >= num_lines_initial) {
          // user-provided config contained repeat of node with this name.
          KALDI_ERR << "Node name " << name
                    << " appears twice in the same config file.";
        }
        // following assert checks that the config-file generated
        // from an actual nnet does not contain repeats.. that
        // would be a bug so check it with assert.
        KALDI_ASSERT(line >= num_lines_initial);
        to_remove[prev_line] = true;
      }
      node_name_to_most_recent_line[name] = line;
    }
  }
  // Now remove any lines with to_remove[i] = true.
  std::vector<ConfigLine> config_lines_out;
  config_lines_out.reserve(num_lines);
  for (int32 i = 0; i < num_lines; i++) {
    if (!to_remove[i])
      config_lines_out.push_back((*config_lines)[i]);
  }
  config_lines->swap(config_lines_out);
}

// copy constructor.
NetworkNode::NetworkNode(const NetworkNode &other):
    node_type(other.node_type),
    descriptor(other.descriptor),
    dim(other.dim),
    dim_offset(other.dim_offset) {
  u.component_index = other.u.component_index;
}


void Nnet::Destroy() {
  for (size_t i = 0; i < components_.size(); i++)
    delete components_[i];
  component_names_.clear();
  components_.clear();
  node_names_.clear();
  nodes_.clear();
}

void Nnet::GetSomeNodeNames(
    std::vector<std::string> *modified_node_names) const {
  modified_node_names->resize(node_names_.size());
  const std::string invalid_name = "**";
  size_t size = node_names_.size();
  for (size_t i = 0; i < size; i++) {
    if (nodes_[i].node_type == kComponent ||
        nodes_[i].node_type == kInput ||
        nodes_[i].node_type == kDimRange) {
      (*modified_node_names)[i] = node_names_[i];
    } else {
      (*modified_node_names)[i] = invalid_name;
    }
  }
}

void Nnet::Read(std::istream &is, bool binary) {
  Destroy();
  ExpectToken(is, binary, "<Nnet3>");
  std::ostringstream config_file_out;
  std::string cur_line;
  getline(is, cur_line);  // Eat up a single newline.
  if (!(cur_line == "" || cur_line == "\r"))
    KALDI_ERR << "Expected newline in config file, got " << cur_line;
  while (getline(is, cur_line)) {
    // config-file part of file is terminated by an empty line.
    if (cur_line == "" || cur_line == "\r")
      break;
    config_file_out << cur_line << std::endl;
  }
  // Now we read the Components; later we try to parse the config_lines.
  ExpectToken(is, binary, "<NumComponents>");
  int32 num_components;
  ReadBasicType(is, binary, &num_components);
  KALDI_ASSERT(num_components >= 0 && num_components < 100000);
  components_.resize(num_components, NULL);
  component_names_.resize(num_components);
  for (int32 c = 0; c < num_components; c++) {
    ExpectToken(is, binary, "<ComponentName>");
    ReadToken(is, binary, &(component_names_[c]));
    components_[c] = Component::ReadNew(is, binary);
  }
  ExpectToken(is, binary, "</Nnet3>");
  std::istringstream config_file_in(config_file_out.str());
  this->ReadConfig(config_file_in);
}

void Nnet::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Nnet3>");
  os << std::endl;
  std::vector<std::string> config_lines;
  const bool include_dim = false;
  GetConfigLines(include_dim, &config_lines);
  for (size_t i = 0; i < config_lines.size(); i++) {
    KALDI_ASSERT(!config_lines[i].empty());
    os << config_lines[i] << std::endl;
  }
  // A blank line terminates the config-like section of the file.
  os << std::endl;
  // Now write the Components
  int32 num_components = components_.size();
  WriteToken(os, binary, "<NumComponents>");
  WriteBasicType(os, binary, num_components);
  if (!binary)
    os << std::endl;
  for (int32 c = 0; c < num_components; c++) {
    WriteToken(os, binary, "<ComponentName>");
    WriteToken(os, binary, component_names_[c]);
    components_[c]->Write(os, binary);
    if (!binary)
      os << std::endl;
  }
  WriteToken(os, binary, "</Nnet3>");
}

int32 Nnet::Modulus() const {
  int32 ans = 1;
  for (int32 n = 0; n < NumNodes(); n++) {
    const NetworkNode &node = nodes_[n];
    if (node.node_type == kDescriptor)
      ans = Lcm(ans, node.descriptor.Modulus());
  }
  return ans;
}


int32 Nnet::InputDim(const std::string &input_name) const {
  int32 n = GetNodeIndex(input_name);
  if (n == -1) return -1;
  const NetworkNode &node = nodes_[n];
  if (node.node_type != kInput) return -1;
  return node.dim;
}

int32 Nnet::OutputDim(const std::string &input_name) const {
  int32 n = GetNodeIndex(input_name);
  if (n == -1 || !IsOutputNode(n)) return -1;
  const NetworkNode &node = nodes_[n];
  return node.Dim(*this);
}

const std::string& Nnet::GetNodeName(int32 node_index) const {
  KALDI_ASSERT(static_cast<size_t>(node_index) < node_names_.size());
  return node_names_[node_index];
}

const std::string& Nnet::GetComponentName(int32 component_index) const {
  KALDI_ASSERT(static_cast<size_t>(component_index) < component_names_.size());
  return component_names_[component_index];
}

void Nnet::Check(bool warn_for_orphans) const {
  int32 num_nodes = nodes_.size(),
    num_input_nodes = 0,
    num_output_nodes = 0;
  KALDI_ASSERT(num_nodes != 0);
  for (int32 n = 0; n < num_nodes; n++) {
    const NetworkNode &node = nodes_[n];
    std::string node_name = node_names_[n];
    KALDI_ASSERT(GetNodeIndex(node_name) == n);
    switch (node.node_type) {
      case kInput:
        KALDI_ASSERT(node.dim > 0);
        num_input_nodes++;
        break;
      case kDescriptor: {
        if (IsOutputNode(n))
          num_output_nodes++;
        std::vector<int32> node_deps;
        node.descriptor.GetNodeDependencies(&node_deps);
        SortAndUniq(&node_deps);
        for (size_t i = 0; i < node_deps.size(); i++) {
          int32 src_node = node_deps[i];
          KALDI_ASSERT(src_node >= 0 && src_node < num_nodes);
          NodeType src_type = nodes_[src_node].node_type;
          if (src_type != kInput && src_type != kDimRange &&
              src_type != kComponent)
            KALDI_ERR << "Invalid source node type in Descriptor: source node "
                      << node_names_[src_node];
        }
        break;
      }
      case kComponent: {
        KALDI_ASSERT(n > 0 && nodes_[n-1].node_type == kDescriptor);
        const NetworkNode &src_node = nodes_[n-1];
        const Component *c = GetComponent(node.u.component_index);
        int32 src_dim = src_node.Dim(*this), input_dim = c->InputDim();
        if (src_dim != input_dim) {
          KALDI_ERR << "Dimension mismatch for network-node "
                    << node_name << ": input-dim "
                    << src_dim << " versus component-input-dim "
                    << input_dim;
        }
        break;
      }
      case kDimRange: {
        int32 input_node = node.u.node_index;
        KALDI_ASSERT(input_node >= 0 && input_node < num_nodes);
        NodeType input_type = nodes_[input_node].node_type;
        if (input_type != kInput && input_type != kComponent)
          KALDI_ERR << "Invalid source node type in DimRange node: source node "
                    << node_names_[input_node];
        int32 input_dim = nodes_[input_node].Dim(*this);
        if (!(node.dim > 0 && node.dim_offset >= 0 &&
              node.dim + node.dim_offset <= input_dim)) {
          KALDI_ERR << "Invalid node dimensions for DimRange node: " << node_name
                    << ": input-dim=" << input_dim << ", dim=" << node.dim
                    << ", dim-offset=" << node.dim_offset;
        }
        break;
      }
      default:
        KALDI_ERR << "Invalid node type for node " << node_name;
    }
  }

  int32 num_components = components_.size();
  for (int32 c = 0; c < num_components; c++) {
    const std::string &component_name = component_names_[c];
    KALDI_ASSERT(GetComponentIndex(component_name) == c &&
                 "Duplicate component names?");
  }
  KALDI_ASSERT(num_input_nodes > 0);
  KALDI_ASSERT(num_output_nodes > 0);


  if (warn_for_orphans) {
    std::vector<int32> orphans;
    FindOrphanComponents(*this, &orphans);
    for (size_t i = 0; i < orphans.size(); i++) {
      KALDI_WARN << "Component " << GetComponentName(orphans[i])
                 << " is never used by any node.";
    }
    FindOrphanNodes(*this, &orphans);
    for (size_t i = 0; i < orphans.size(); i++) {
      if (!IsComponentInputNode(orphans[i])) {
        // There is no point warning about component-input nodes, since the
        // warning will be printed for the corresponding component nodes..  a
        // duplicate warning might be confusing to the user, as the
        // component-input nodes are implicit and usually hidden from users.
        KALDI_WARN << "Node " << GetNodeName(orphans[i])
                   << " is never used to compute any output.";
      }
    }
  }
}

// copy constructor
Nnet::Nnet(const Nnet &nnet):
    component_names_(nnet.component_names_),
    components_(nnet.components_.size()),
    node_names_(nnet.node_names_),
    nodes_(nnet.nodes_) {
  for (size_t i = 0; i < components_.size(); i++)
    components_[i] = nnet.components_[i]->Copy();
  Check();
}

Nnet& Nnet::operator =(const Nnet &nnet) {
  if (this == &nnet)
    return *this;
  Destroy();
  component_names_ = nnet.component_names_;
  components_.resize(nnet.components_.size());
  node_names_ = nnet.node_names_;
  nodes_ = nnet.nodes_;
  for (size_t i = 0; i < components_.size(); i++)
    components_[i] = nnet.components_[i]->Copy();
  Check();
  return *this;
}

std::string Nnet::Info() const {
  std::ostringstream os;

  if(IsSimpleNnet(*this))  {
    int32 left_context, right_context;
    ComputeSimpleNnetContext(*this, &left_context, &right_context);
    os << "left-context: " << left_context << "\n";
    os << "right-context: " << right_context << "\n";
  }
  os << "num-parameters: " << NumParameters(*this) << "\n";
  os << "modulus: " << this->Modulus() << "\n";
  std::vector<std::string> config_lines;
  bool include_dim = true;
  GetConfigLines(include_dim, &config_lines);
  for (size_t i = 0; i < config_lines.size(); i++)
    os << config_lines[i] << "\n";
  // Get component info.
  for (size_t i = 0; i < components_.size(); i++)
    os << "component name=" << component_names_[i]
       << " type=" << components_[i]->Info() << "\n";
  return os.str();
}

void Nnet::RemoveOrphanComponents() {
  std::vector<int32> orphan_components;
  FindOrphanComponents(*this, &orphan_components);
  KALDI_LOG << "Removing " << orphan_components.size()
            << " orphan components.";
  if (orphan_components.empty())
    return;
  int32 old_num_components = components_.size(),
      new_num_components = 0;
  std::vector<int32> old2new_map(old_num_components, 0);
  for (size_t i = 0; i < orphan_components.size(); i++)
    old2new_map[orphan_components[i]] = -1;
  std::vector<Component*> new_components;
  std::vector<std::string> new_component_names;
  for (int32 c = 0; c < old_num_components; c++) {
    if (old2new_map[c] != -1) {
      old2new_map[c] = new_num_components++;
      new_components.push_back(components_[c]);
      new_component_names.push_back(component_names_[c]);
    } else {
      delete components_[c];
      components_[c] = NULL;
    }
  }
  for (int32 n = 0; n < NumNodes(); n++) {
    if (IsComponentNode(n)) {
      int32 old_c = nodes_[n].u.component_index,
          new_c = old2new_map[old_c];
      KALDI_ASSERT(new_c >= 0);
      nodes_[n].u.component_index = new_c;
    }
  }
  components_ = new_components;
  component_names_ = new_component_names;
  Check();
}

void Nnet::RemoveSomeNodes(const std::vector<int32> &nodes_to_remove) {
  if (nodes_to_remove.empty())
    return;
  int32 old_num_nodes = nodes_.size(),
      new_num_nodes = 0;
  std::vector<int32> old2new_map(old_num_nodes, 0);
  for (size_t i = 0; i < nodes_to_remove.size(); i++)
    old2new_map[nodes_to_remove[i]] = -1;
  std::vector<NetworkNode> new_nodes;
  std::vector<std::string> new_node_names;
  for (int32 n = 0; n < old_num_nodes; n++) {
    if (old2new_map[n] != -1) {
      old2new_map[n] = new_num_nodes++;
      new_nodes.push_back(nodes_[n]);
      new_node_names.push_back(node_names_[n]);
    }
  }
  for (int32 n = 0; n < new_num_nodes; n++) {
    if (new_nodes[n].node_type == kDescriptor) {
      // we need to renumber the node indexes inside the descriptor.  It's
      // easiest to do this by converting back and forth to text format.  This
      // is inefficient, of course, but these graphs are typically quite small.
      std::ostringstream os;
      new_nodes[n].descriptor.WriteConfig(os, node_names_);
      std::vector<std::string> tokens;
      DescriptorTokenize(os.str(), &tokens);
      KALDI_ASSERT(!tokens.empty());
      tokens.push_back("end of input");
      const std::string *token = &(tokens[0]);
      Descriptor new_descriptor;
      // this should work; if it doesn't, there was a programming error.
      if (!new_nodes[n].descriptor.Parse(new_node_names, &token)) {
        KALDI_ERR << "Code error removing orphan nodes.";
      }
    } else if (new_nodes[n].node_type == kDimRange) {
      int32 old_node_index = new_nodes[n].u.node_index,
          new_node_index = old2new_map[old_node_index];
      KALDI_ASSERT(new_node_index >= 0 && new_node_index <= new_num_nodes);
      new_nodes[n].u.node_index = new_node_index;
    }
  }
  KALDI_LOG << "Removed " << (old_num_nodes - new_num_nodes)
            << " orphan nodes.";
  nodes_ = new_nodes;
  node_names_ = new_node_names;
  bool warn_for_orphans = false;
  // don't warn about orphans, because at this stage we may have
  // orphan components that will later be removed by calling
  // RemoveOrphanComponents().
  Check(warn_for_orphans);
}


void Nnet::RemoveOrphanNodes(bool remove_orphan_inputs) {
  std::vector<int32> orphan_nodes;
  FindOrphanNodes(*this, &orphan_nodes);
  if (!remove_orphan_inputs)
    for (int32 i = 0; i < orphan_nodes.size(); i++)
      if (IsInputNode(orphan_nodes[i]))
        orphan_nodes.erase(orphan_nodes.begin() + i);
  RemoveSomeNodes(orphan_nodes);
}

void Nnet::ResetGenerators() {
  // resets random-number generators for all random
  // components.
  for (int32 c = 0; c < NumComponents(); c++) {
    RandomComponent *rc = dynamic_cast<RandomComponent*>(GetComponent(c));
    if (rc != NULL)
      rc->ResetGenerator();
  }
}

} // namespace nnet3
} // namespace kaldi
