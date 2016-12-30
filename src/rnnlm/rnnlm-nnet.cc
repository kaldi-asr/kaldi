#include <iterator>
#include <sstream>
#include "rnnlm/rnnlm-nnet.h"
#include "nnet3/nnet-parse.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
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
}

void LmNnet::ReadConfig(std::istream &config_is) {
  std::vector<string> type(2);
  std::vector<string> lines(2);
  std::vector<ConfigLine> config_lines(2);
  
  for (int i = 0; i < 2; i++) {
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
