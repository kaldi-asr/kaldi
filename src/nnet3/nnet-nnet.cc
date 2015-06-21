// nnet3/nnet-nnet.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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
      KALDI_ERR << "Invalid node type.";
  }
  KALDI_ASSERT(ans > 0);
  return ans;
}

const std::vector<std::string> &Nnet::GetNodeNames() const {
  return node_names_;
}

const std::vector<std::string> &Nnet::GetComponentNames() const {
  return component_names_;
}

std::string Nnet::GetAsConfigLine(int32 node_index) const {
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
      KALDI_ASSERT(IsOutput(node_index));
      ans << "output-node name=" << name << " input=";
      node.descriptor.WriteConfig(ans, node_names_);
      break;
    case kComponent:
      ans << "component-node name=" << name << " component="
          << component_names_[node.u.component_index] << " input=";
      KALDI_ASSERT(nodes_[node_index-1].node_type == kDescriptor);
      nodes_[node_index-1].descriptor.WriteConfig(ans, node_names_);
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

bool Nnet::IsOutput(int32 node) const {
  int32 size = nodes_.size();  
  KALDI_ASSERT(node >= 0 && node < size);  
  return (nodes_[node].node_type == kDescriptor &&
          (node + 1 == size ||
           nodes_[node + 1].node_type != kComponent));
}

const Component *Nnet::GetComponent(int32 c) const {
  KALDI_ASSERT(static_cast<size_t>(c) < components_.size());
  return components_[c];
}

Component *Nnet::GetComponent(int32 c) {
  KALDI_ASSERT(static_cast<size_t>(c) < components_.size());
  return components_[c];
}

/// Returns true if this is component-input node, i.e. a node of type kDescriptor
/// that immediately precedes a node of type kComponent.
bool Nnet::IsComponentInput(int32 node) const {
  int32 size = nodes_.size();
  KALDI_ASSERT(node >= 0 && node < size);
  return (node + 1 < size &&
          nodes_[node].node_type == kDescriptor &&
          nodes_[node+1].node_type == kComponent);
}

void Nnet::GetConfigLines(std::vector<std::string> *config_lines) const {
  config_lines->clear();
  for (int32 n = 0; n < NumNodes(); n++)
    if (!IsComponentInput(n))
      config_lines->push_back(GetAsConfigLine(n));
  
}

void Nnet::ReadConfig(std::istream &config_is) {
  
  std::vector<std::string> lines;
  GetConfigLines(&lines);
  
  // we'll later regenerate what we need from nodes_ and node_name_ from the
  // string representation.
  nodes_.clear();
  node_names_.clear();

  int32 num_lines_initial = lines.size();
    
  // add new lines corresponding to what is in the proi
  ReadConfigFile(config_is, &lines);
  // now "lines" will have comments removed and empty lines stripped out

  std::vector<std::string> first_tokens(lines.size());
  std::vector<ConfigLine> config_lines(lines.size());
  for (size_t i = 0; i < lines.size(); i++) {
    std::istringstream is(lines[i]);
    std::string first_token;
    is >> first_token;
    first_tokens[i] = first_token;
    std::string rest_of_line;
    getline(is, rest_of_line);
    if (!config_lines[i].ParseLine(rest_of_line))
      KALDI_ERR << "Could not parse config-file line '" << lines[i] << "'";

  }

  // the next line will possibly remove some elements from "first_tokens" and
  // "config_lines" so nothing is doubly defined.
  RemoveRedundantConfigLines(num_lines_initial, &first_tokens, &config_lines);

  
  int32 initial_num_components = components_.size();
  
  for (int32 pass = 0; pass <= 1; pass++) {
    for (size_t i = 0; i < lines.size(); i++) {
      if (first_tokens[i] == "component") {
        if (pass == 0)
          ProcessComponentConfigLine(initial_num_components,
                                     lines[i], &(config_lines[i]));
      } else if (first_tokens[i] == "component-node") {
        ProcessComponentNodeConfigLine(pass, lines[i], &(config_lines[i]));
      } else if (first_tokens[i] == "input-node") {
        if (pass == 0)
          ProcessInputNodeConfigLine(lines[i], &(config_lines[i]));
      } else if (first_tokens[i] == "output-node") {
        ProcessOutputNodeConfigLine(pass, lines[i], &(config_lines[i]));
      } else if (first_tokens[i] == "dim-range-node") {
        ProcessDimRangeNodeConfigLine(pass, lines[i], &(config_lines[i]));
      } else {
        KALDI_ERR << "Invalid config-file line: " << lines[i];
      }
    }
  }  
}

// called only on pass 0.
void Nnet::ProcessComponentConfigLine(
    int32 initial_num_components, const std::string &whole_line,
    ConfigLine *config) {
  std::string name, type;
  if (!config->GetValue("name", &name))
    KALDI_ERR << "Expected field name=<component-name> in config line: "
              << whole_line;
  if (!IsToken(name)) // e.g. contains a space.
    KALDI_ERR << "Component name '" << name << "' is not allowed, in line: "
              << whole_line;
  if (!config->GetValue("type", &type))
    KALDI_ERR << "Expected field type=<component-type> in config line: "
              << whole_line;
  Component *new_component = Component::NewComponentOfType(type);
  if (new_component == NULL)
    KALDI_ERR << "Unknown component-type '" << type
              << "' in config file.  Check your code version and config.";
  // the next call will call KALDI_ERR or KALDI_ASSERT and die if something
  // went wrong.
  new_component->InitFromConfig(config);
  int32 index = GetNodeIndex(name);
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
              << " in config line: " << whole_line;
}


void Nnet::ProcessComponentNodeConfigLine(
    int32 pass, const std::string &whole_line,
    ConfigLine *config) {
  
  std::string name;
  if (!config->GetValue("name", &name))
    KALDI_ERR << "Expected field name=<component-name> in config line: "
              << whole_line;
  
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
                << whole_line;
    int32 component_index = GetComponentIndex(component_name);
    if (component_index == -1)
      KALDI_ERR << "No component named '" << component_name
                << "', in config line: " << whole_line;
    nodes_[node_index].u.component_index = component_index;
    
    if (!config->GetValue("input", &input_descriptor))
      KALDI_ERR << "Expected input=<input-descriptor>, in config line: "
                << whole_line;
    std::vector<std::string> tokens;
    if (!DescriptorTokenize(input_descriptor, &tokens))
      KALDI_ERR << "Error tokenizing descriptor in config line "
                << whole_line;
    std::vector<std::string> node_names_temp;
    GetSomeNodeNames(&node_names_temp);
    tokens.push_back("end of input");
    const std::string *next_token = &(tokens[0]);
    if (!nodes_[input_node_index].descriptor.Parse(node_names_temp, &next_token))
      KALDI_ERR << "Error parsing Descriptor in config line: "
                << whole_line;
    if (config->HasUnusedValues())
      KALDI_ERR << "Unused values '" << config->UnusedValues()
                << " in config line: " << whole_line;
  }
}

// called only on pass 0.
void Nnet::ProcessInputNodeConfigLine(
    const std::string &whole_line,
    ConfigLine *config) {
  std::string name;
  if (!config->GetValue("name", &name))
    KALDI_ERR << "Expected field name=<input-name> in config line: "
              << whole_line;
  int32 dim;
  if (!config->GetValue("dim", &dim))
    KALDI_ERR << "Expected field dim=<input-dim> in config line: "
              << whole_line;

  if (config->HasUnusedValues())
    KALDI_ERR << "Unused values '" << config->UnusedValues()
              << " in config line: " << whole_line;
  
  KALDI_ASSERT(GetNodeIndex(name) == -1);
  int32 node_index = nodes_.size();    
  nodes_.push_back(NetworkNode(kInput));
  if (dim <= 0)
    KALDI_ERR << "Invalid dimension in config line: " << whole_line;
  nodes_[node_index].dim = dim;
}


void Nnet::ProcessOutputNodeConfigLine(
    int32 pass,
    const std::string &whole_line,
    ConfigLine *config) {
  std::string name;
  if (!config->GetValue("name", &name))
    KALDI_ERR << "Expected field name=<input-name> in config line: "
              << whole_line;
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
                << whole_line;
    std::vector<std::string> tokens;
    if (!DescriptorTokenize(input_descriptor, &tokens))
      KALDI_ERR << "Error tokenizing descriptor in config line "
                << whole_line;
    tokens.push_back("end of input");
    // if the following fails it will die.
    std::vector<std::string> node_names_temp;
    GetSomeNodeNames(&node_names_temp);
    const std::string *next_token = &(tokens[0]);
    if (!nodes_[node_index].descriptor.Parse(node_names_temp, &next_token))
      KALDI_ERR << "Error parsing descriptor (input=...) in config line "
                << whole_line;
    if (config->HasUnusedValues())
      KALDI_ERR << "Unused values '" << config->UnusedValues()
                << " in config line: " << whole_line;
  }
}


void Nnet::ProcessDimRangeNodeConfigLine(
    int32 pass,
    const std::string &whole_line,
    ConfigLine *config) {
  std::string name;
  if (!config->GetValue("name", &name))
    KALDI_ERR << "Expected field name=<input-name> in config line: "
              << whole_line;
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
                << whole_line;
    int32 dim, dim_offset;
    if (!config->GetValue("dim", &dim))
      KALDI_ERR << "Expected dim=<feature-dim>, in config line: "
                << whole_line;
    if (!config->GetValue("dim-offset", &dim_offset))
      KALDI_ERR << "Expected dim-offset=<dimension-offset>, in config line: "
                << whole_line;

    int32 input_node_index = GetNodeIndex(input_node_name);
    if (input_node_index == -1 ||
        !(nodes_[input_node_index].node_type == kComponent ||
          nodes_[input_node_index].node_type == kInput))
      KALDI_ERR << "invalid input-node " << input_node_name
                << ": " << whole_line;

    if (config->HasUnusedValues())
      KALDI_ERR << "Unused values '" << config->UnusedValues()
                << " in config line: " << whole_line;

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
                                      std::vector<std::string> *first_tokens,
                                      std::vector<ConfigLine> *configs) {
  int32 num_lines = first_tokens->size();
  KALDI_ASSERT(configs->size() == num_lines &&
               num_lines_initial <= num_lines);
  // node names and component names live in different namespaces.
  unordered_map<std::string, int32, StringHasher> node_name_to_most_recent_line;
  unordered_set<std::string, StringHasher> component_names;
  typedef unordered_map<std::string, int32, StringHasher>::iterator IterType;

  std::vector<bool> to_remove(num_lines, false);
  for (int32 line = 0; line < num_lines; line++) {
    std::string first_token = (*first_tokens)[line];
    ConfigLine &config_line = (*configs)[line];
    std::string name;
    if (!config_line.GetValue("name", &name))
      KALDI_ERR << "Config line has no field 'name=xxx': "
                << config_line.WholeLine();
    if (!IsValidName(name))
      KALDI_ERR << "Name '" << name << "' is not allowable, in line: "
                << config_line.WholeLine();
    if ((*first_tokens)[line] == "component") {
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
  std::vector<std::string> first_tokens_out;
  std::vector<ConfigLine> configs_out;
  first_tokens_out.reserve(num_lines);
  configs_out.reserve(num_lines);
  for (int32 i = 0; i < num_lines; i++) {
    if (!to_remove[i]) {
      first_tokens_out.push_back((*first_tokens)[i]);
      configs_out.push_back((*configs)[i]);
    }
  }
  first_tokens->swap(first_tokens_out);
  configs->swap(configs_out);
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
  std::vector<std::string> config_lines;
  std::string cur_line;
  getline(is, cur_line);  // Eat up a single newline.
  if (!(cur_line == "" || cur_line == "\r"))
    KALDI_ERR << "Expected newline in config file, got " << cur_line;
  while (getline(is, cur_line)) {
    // config-file part of file is terminated by an empty line.
    if (cur_line == "" || cur_line == "\r")
      break;
    config_lines.push_back(cur_line);
  }
  // Now we read the Components; later we try to parse the config_lines.
  ExpectToken(is, binary, "<NumComponents>");
  int32 num_components;
  ReadBasicType(is, binary, &num_components);
  KALDI_ASSERT(num_components >= 0 && num_components < 100000);
  components_.resize(num_components, NULL);
  component_names_.resize(num_components);
  for (int32 c = 0; c < num_components; c++) {
    ExpectToken(is, binary, "<ComponentName> ");
    ReadToken(is, binary, &(component_names_[c]));
    components_[c] = Component::ReadNew(is, binary);
  }
  ExpectToken(is, binary, "</Nnet3>");  
}

void Nnet::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Nnet3>");
  os << std::endl;
  std::vector<std::string> config_lines;
  GetConfigLines(&config_lines);
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
    WriteToken(os, binary, "<ComponentName> ");
    WriteToken(os, binary, component_names_[c]);
    components_[c]->Write(os, binary);
    if (!binary)
      os << std::endl;
  }
  WriteToken(os, binary, "</Nnet3>");
}


} // namespace nnet3
} // namespace kaldi
