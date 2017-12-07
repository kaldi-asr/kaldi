%module online2_nnet3_latgen_i2x_wrapper
%{
#include "online2-nnet3-latgen-i2x-wrapper.h"

int16_t* Cast(intptr_t ptr) {
  return (int16_t*)(ptr);
}
std::string* CreateString() {
  return new std::string;
}
void FreeString(std::string* str) {
  delete str;
}
std::string DereferenceStringPtr(std::string* str) {
  return *str;
}
%}

%include stdint.i
%include cpointer.i
%include std_string.i
using std::string;
%include "online2-nnet3-latgen-i2x-wrapper.h"
extern int16_t* Cast(int ptr);
