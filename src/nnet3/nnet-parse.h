// nnet3/nnet-parse.h

// Copyright 2015    Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_PARSE_H_
#define KALDI_NNET3_NNET_PARSE_H_

#include "util/text-utils.h"
#include "matrix/kaldi-vector.h"

namespace kaldi {
namespace nnet3 {


/**
   This function tokenizes input when parsing Descriptor configuration
   values.  A token in this context is not the same as a generic Kaldi token,
   e.g. as defined in IsToken() in util/text_utils.h, which just means a non-empty
   whitespace-free string.  Here a token is more like a programming-language token,
   and currently the following are allowed as tokens:
    "("
    ")"
    ","
   - A nonempty string beginning with A-Za-z_, and containing only -_A-Za-z0-9.
   - An integer, optionally beginning with - or + and then a nonempty sequence of 0-9.

   This function should return false and print an informative error with local
   context if it can't tokenize the input.
 */
bool DescriptorTokenize(const std::string &input,
                        std::vector<std::string> *tokens);


/*
  Returns true if name 'name' matches pattern 'pattern'.  The pattern
  format is: everything is literal, except '*' matches zero or more
  characters.  (Like filename globbing in UNIX).
 */
bool NameMatchesPattern(const char *name,
                        const char *pattern);


/**
  Return a string used in error messages.  Here, "is" will be from an
  istringstream derived from a single line or part of a line.
  If "is" is at EOF or in error state, this should just say "end of line",
  else if the contents of "is" before EOF is <20 characters it should return
  it all, else it should return the first 20 characters followed by "...".
*/
std::string ErrorContext(std::istream &is);

std::string ErrorContext(const std::string &str);

/** Returns a string that summarizes a vector fairly succintly, for
    printing stats in info lines.  For example:
   "[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.001,0.003,0.003,0.004 \
      0.005,0.01,0.07,0.11,0.14 0.18,0.24,0.29,0.39), mean=0.0745, stddev=0.0611]"
*/
std::string SummarizeVector(const VectorBase<float> &vec);

std::string SummarizeVector(const VectorBase<double> &vec);

std::string SummarizeVector(const CuVectorBase<BaseFloat> &vec);

/** Print to 'os' some information about the mean and standard deviation of
    some parameters, used in Info() functions in nnet-simple-component.cc.
    For example:
     PrintParameterStats(os, "bias", bias_params_, true);
    would print to 'os' something like the string
     ", bias-{mean,stddev}=-0.013,0.196".  If 'include_mean = false',
    it will print something like
     ", bias-rms=0.2416", and this represents and uncentered standard deviation.
 */
void PrintParameterStats(std::ostringstream &os,
                         const std::string &name,
                         const CuVectorBase<BaseFloat> &params,
                         bool include_mean = false);

/** Print to 'os' some information about the mean and standard deviation of
    some parameters, used in Info() functions in nnet-simple-component.cc.
    For example:
     PrintParameterStats(os, "linear-params", linear_params_;
    would print to 'os' something like the string
     ", linear-params-rms=0.239".
    If you set 'include_mean' to true, it will print something like
    ", linear-params-{mean-stddev}=0.103,0.183".
    If you set 'include_row_norms' to true, it will print something
    like
    ", linear-params-row-norms=[percentiles(0,1........, stddev=0.0508]"
    If you set 'include_column_norms' to true, it will print something
    like
    ", linear-params-col-norms=[percentiles(0,1........, stddev=0.0508]"
    If you set 'include_singular_values' to true, it will print something
    like
    ", linear-params-singular-values=[percentiles(0,1........, stddev=0.0508]"
 */
void PrintParameterStats(std::ostringstream &os,
                         const std::string &name,
                         const CuMatrix<BaseFloat> &params,
                         bool include_mean = false,
                         bool include_row_norms = false,
                         bool include_column_norms = false,
                         bool include_singular_values = false);


} // namespace nnet3
} // namespace kaldi


#endif
