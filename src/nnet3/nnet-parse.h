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
   This class is responsible for parsing input like
    hi-there xx=yyy a=b c empty= f-oo=Append(bar, sss) ba_z=123 bing='a b c' baz="a b c d='a b' e"
   and giving you access to the fields, in this case

   FirstToken() == "hi-there", and key->value pairs:

   xx->yyy, a->"b c", empty->"", f-oo->"Append(bar, sss)", ba_z->"123",
   bing->"a b c", baz->"a b c d='a b' e"

   The first token is optional, if the line started with a key-value pair then
   FirstValue() will be empty.

   Note: it can parse value fields with space inside them only if they are free of the '='
   character.  If values are going to contain the '=' character, you need to quote them
   with either single or double quotes.

   Key values may contain -_a-zA-Z0-9, but must begin with a-zA-Z_.
 */
class ConfigLine {
 public:
  // Tries to parse the line as a config-file line.  Returns false
  // if it could not for some reason, e.g. parsing failure.  In most cases
  // prints no warnings; the user should do this.  Does not expect comments.
  bool ParseLine(const std::string &line);

  // the GetValue functions are overloaded for various types.  They return true
  // if the key exists with value that can be converted to that type, and false
  // otherwise.  They also mark the key-value pair as having been read.  It is
  // not an error to read values twice.
  bool GetValue(const std::string &key, std::string *value);
  bool GetValue(const std::string &key, BaseFloat *value);
  bool GetValue(const std::string &key, int32 *value);
  // Values may be separated by ":" or by ",".
  bool GetValue(const std::string &key, std::vector<int32> *value);
  bool GetValue(const std::string &key, bool *value);

  bool HasUnusedValues() const;
  /// returns e.g. foo=bar xxx=yyy if foo and xxx were not consumed by one
  /// of the GetValue() functions.
  std::string UnusedValues() const;

  const std::string &FirstToken() const { return first_token_; }

  const std::string WholeLine() { return whole_line_; }
  // use default assignment operator and copy constructor.
 private:
  std::string whole_line_;
  // the first token of the line, e.g. if line is
  // foo-bar baz=bing
  // then first_token_ would be "foo-bar".
  std::string first_token_;

  // data_ maps from key to (value, is-this-value-consumed?).
  std::map<std::string, std::pair<std::string, bool> > data_;

};

// Note: the ParseFromString functions are to be removed after we switch over to
// using the ConfigLine mechanism.


/// \file nnet-parse.h
///   This header contains a few parsing-related functions that are used
///    while reading parsing neural network files and config files.

/// Function used in Init routines.  Suppose name=="foo", if "string" has a
/// field like foo=12, this function will set "param" to 12 and remove that
/// element from "string".  It returns true if the parameter was read.
bool ParseFromString(const std::string &name, std::string *string,
                     int32 *param);

/// This version of ParseFromString is for parameters of type BaseFloat.
bool ParseFromString(const std::string &name, std::string *string,
                     BaseFloat *param);

/// This version of ParseFromString is for parameters of type bool, which can
/// appear as any string beginning with f, F, t or T.
bool ParseFromString(const std::string &name, std::string *string,
                     bool *param);

/// This version of ParseFromString is for parsing strings.  (these
/// should not contain space).
bool ParseFromString(const std::string &name, std::string *string,
                     std::string *param);

/// This version of ParseFromString handles colon-separated or comma-separated
/// lists of integers.
bool ParseFromString(const std::string &name, std::string *string,
                     std::vector<int32> *param);

/// This function is like ExpectToken but for two tokens, and it will either
/// accept token1 and then token2, or just token2.  This is useful in Read
/// functions where the first token may already have been consumed.
void ExpectOneOrTwoTokens(std::istream &is, bool binary,
                          const std::string &token1,
                          const std::string &token2);

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

/// Returns true if 'name' would be a valid name for a component or node in a
/// Nnet.  This is a nonempty string beginning with A-Za-z_, and containing only
/// '-', '_', '.', A-Z, a-z, or 0-9.
bool IsValidName(const std::string &name);


/**
   This function reads in a config file and *appends* its contents to a vector of
   lines; it is responsible for removing comments (anything after '#') and
   stripping out any lines that contain only whitespace after comment removal.
 */
void ReadConfigLines(std::istream &is,
                     std::vector<std::string> *lines);


/**
   This function converts config-lines from a simple sequence of strings
   as output by ReadConfigLines(), into a sequence of first-tokens and
   name-value pairs.  The general format is:
      "command-type bar=baz xx=yyy"
   etc., although there are subtleties as to what exactly is allowed, see
   documentation for class ConfigLine for details.
   This function will die if there was a parsing failure.
 */
void ParseConfigLines(const std::vector<std::string> &lines,
                      std::vector<ConfigLine> *config_lines);

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
std::string SummarizeVector(const VectorBase<BaseFloat> &vec);

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
