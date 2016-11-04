#include <iterator>
#include <sstream>
#include "rnnlm/rnnlm-nnet.h"
#include "nnet3/nnet-parse.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

void LmNnet::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<LmNnet>");
  input_projection_ = AffineComponent::ReadNew(is, binary);
  nnet_->Read(is, binary);
  output_projection_ = AffineComponent::ReadNew(is, binary);
  output_layer_ = NonlinearComponent::ReadNew(is, binary);
  ExpectToken(is, binary, "</LmNnet>");

}

void LmNnet::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<LmNnet>");
  input_projection_->Write(os, binary);
  nnet_->Write(os, binary);
  output_projection_->Write(os, binary);
  output_layer_->Write(os, binary);
  WriteToken(os, binary, "</LmNnet>");

}

void LmNnet::ReadConfig(std::istream &config_is) {

  // TODO(hxu) will allow for more flexible types
  input_projection_ = new NaturalGradientAffineComponent();
  output_projection_ = new NaturalGradientAffineComponent();
  output_layer_ =  new LogSoftmaxComponent();

  std::vector<string> lines(3);
  std::vector<ConfigLine> config_lines(3);
  
  for (int i = 0; i < 3; i++) {
    getline(config_is, lines[i]);
    config_lines[i].ParseLine(lines[i]);
  }

  int i = 0; 
  input_projection_->InitFromConfig(&config_lines[i++]);
  output_projection_->InitFromConfig(&config_lines[i++]);
  output_layer_->InitFromConfig(&config_lines[i++]);

  nnet_->ReadConfig(config_is);

}

}
}
