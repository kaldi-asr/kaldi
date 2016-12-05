#include <iterator>
#include <sstream>
#include "rnnlm/rnnlm-nnet.h"
#include "nnet3/nnet-parse.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {

//using nnet3::AffineComponent;
//using nnet3::NonlinearComponent;
//using nnet3::LogSoftmaxComponent;

namespace rnnlm {

std::string LmNnet::Info() const {
  std::ostringstream os;
  int num_params_this = 0;
  LmInputComponent *p;
  if ((p = dynamic_cast<LmInputComponent*>(input_projection_)) != NULL) {
    num_params_this += p->NumParameters();
  }
  LmOutputComponent *p2;
  if ((p2 = dynamic_cast<LmOutputComponent*>(output_projection_)) != NULL) {
    num_params_this += p2->NumParameters();
  }
  os << "num-parameters: " << NumParameters(*this->nnet_) + num_params_this << "\n";
  os << "internal nnet info: \n" 
     << nnet_->Info();
//  os << "modulus: " << this->Modulus() << "\n";
//  std::vector<std::string> config_lines;
//  bool include_dim = true;
//  nnet_->GetConfigLines(include_dim, &config_lines);
//  for (size_t i = 0; i < config_lines.size(); i++)
//    os << config_lines[i] << "\n";
//  // Get component info.
//  for (size_t i = 0; i < nnet_->components_.size(); i++)
//    os << "component name=" << nnet_->component_names_[i]
//       << " type=" << nnet_->components_[i]->Info() << "\n";
  return os.str();
}

void LmNnet::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<LmNnet>");
  input_projection_ = dynamic_cast<LmInputComponent*>(LmComponent::ReadNew(is, binary));
  output_projection_ = dynamic_cast<LmOutputComponent*>(LmComponent::ReadNew(is, binary));

  nnet_->Read(is, binary);

  ExpectToken(is, binary, "</LmNnet>");

  KALDI_LOG << "Successfully Read LmNnet";

}

void LmNnet::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<LmNnet>");
  input_projection_->Write(os, binary);
  output_projection_->Write(os, binary);

  nnet_->Write(os, binary);
  WriteToken(os, binary, "</LmNnet>");

  KALDI_LOG << "Successfully Write LmNnet";

  //
}

void LmNnet::ReadConfig(std::istream &config_is) {

  // TODO(hxu) will allow for more flexible types
//  input_projection_ = new LmLinearComponent();
//  output_projection_ = new AffineComponent();
//  output_layer_ =  new LogSoftmaxComponent();

  std::vector<string> type(3);
  std::vector<string> lines(3);
  std::vector<ConfigLine> config_lines(3);
  
  for (int i = 0; i < 3; i++) {
    config_is >> type[i];
    getline(config_is, lines[i]);
    config_lines[i].ParseLine(lines[i]);
  }

  int i = 0; 
  input_projection_ = dynamic_cast<LmInputComponent*>(LmComponent::NewComponentOfType(type[i]));
  input_projection_->InitFromConfig(&config_lines[i++]);

  output_projection_ = dynamic_cast<LmOutputComponent*>(LmComponent::NewComponentOfType(type[i]));
  output_projection_->InitFromConfig(&config_lines[i++]);

  nnet_->ReadConfig(config_is);

}

}
}
