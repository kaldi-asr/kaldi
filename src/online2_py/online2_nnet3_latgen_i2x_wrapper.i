%module online2_nnet3_latgen_i2x_wrapper
%begin %{
#define SWIG_PYTHON_STRICT_BYTE_CHAR
%}

%{
#include "../online2/online2-nnet3-latgen-i2x-wrapper.h"
%}

%include stdint.i
%include std_string.i
using std::string;
%newobject DecoderFactory::StartDecodingSession();

%feature("autodoc", "3");
%include "../online2/online2-nnet3-latgen-i2x-wrapper.h"

