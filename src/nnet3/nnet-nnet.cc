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
    case kComponentOutput:
      ans = nnet.GetComponent(u.component_index)->OutputDim();
      break;
    default:
      KALDI_ERR << "Invalid node type.";
  }
  KALDI_ASSERT(ans > 0);
  return ans;
}

void Nnet::ReadConfig(std::istream &config_is) {
  std::vector<std::string> lines;
  ReadConfigFile(config_is, &lines);
  // now "lines" will have comments removed and empty lines stripped out
  
  std::vector<std::string> first_token(lines.size());
  std::vector<ConfigLine> config_lines(lines.size());
  for (size_t i = 0; i < lines.size(); i++) {
    std::istringstream is(lines[i]);
    std::string first_token;
    is >> first_token;
    std::string rest_of_line;
    getline(is, rest_of_line);
    if (!config_lines[i].ParseLine(rest_of_line))
      KALDI_ERR << "Could not parse config-file line " << lines[i];
  }

  int32 initial_num_nodes = nodes_.size(),
      initial_num_components = components_.size();
  
  for (int32 pass = 0; pass <= 1; pass++) {
    for (size_t i = 0; i < lines.size(); i++) {
      if (first_token == "component") {
        if (pass == 0)
          ProcessComponentConfigLine(initial_num_components,
                                     lines[i], &(config_lines[i]));
      } else if (first_token == "component-node") {
        ProcessComponentNodeConfigLine(pass, initial_num_nodes,
                                       lines[i], &(config_lines[i]));
      } else if (first_token == "input-node") {
        if (pass == 0)
          ProcessInputNodeConfigLine(initial_num_nodes,
                                     lines[i], &(config_lines[i]));
      } else if (first_token == "output-node") {
        ProcessOutputNodeConfigLine(pass, initial_num_nodes,
                                    lines[i], &(config_lines[i]));
      } else if (first_token == "dim-range-node") {
        ProcessDimRangeNodeConfigLine(pass, initial_num_nodes,
                                      lines[i], &(config_lines[i]));
      } else {
        KALDI_ERR << "Invalid config-file line: " << lines[i];
      }
    }
  }

  
}


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
  if (component == NULL)
    KALDI_ERR << "Unknown component-type '" << type
              << "' in config file.  Check your code version and config.";
  // the next call will call KALDI_ERR or KALDI_ASSERT and die if something
  // went wrong.
  component->InitFromConfig(config);
  int32 index = IndexOfNode(name);
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
}


void Nnet::ProcessComponentNodeConfigLine(
    int32 pass, int32 initial_num_nodes, const std::string &whole_line,
    ConfigLine *config) {
  std::string name, type;
  if (!config->GetValue("name", &name))
    KALDI_ERR << "Expected field name=<component-name> in config line: "
              << whole_line;
  if (!IsToken(name)) // e.g. contains a space.
    KALDI_ERR << "Node name '" << name << "' is not allowed, in line: "
              << whole_line;

  std::string input_name = name + std::string("-input");
  int32 input_node_index = IndexOfNode(input_name),
      node_index = IndexOfNode(name);
  if (node_index != -1 && input_node_index == -1) {
    // this wouldn't be hard to handle but I don't see it being needed
    // for the time being.  It would involve renumbering nodes.
    KALDI_ERR << "Component-node " << name << " already exists but not "
              << "its input " << input_name << ".  Currently we don't "
              << "support changing non-component node to component node. ";
  }
  KALDI_ASSERT((input_node_index == -1) == (node_index == -1));
  // HERE-- todo.
}



} // namespace nnet3
} // namespace kaldi
