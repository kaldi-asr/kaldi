// util/parse-options.h

// Copyright 2009-2011  Karel Vesely;  Microsoft Corporation;
//                      Saarland University

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

#ifndef KALDI_UTIL_PARSE_OPTIONS_H_
#define KALDI_UTIL_PARSE_OPTIONS_H_

#include "base/kaldi-common.h"
#include <map>


namespace kaldi {

/// The class ParseOptions is for parsing command-line options; see
/// \ref parse_options for more documentation.
class ParseOptions {
 public:
  explicit ParseOptions(const char *usage)
      : print_args_(true), help_(false), usage_(usage), argc_(0), argv_(NULL) {
#ifndef _MSC_VER // This is just a convenient place to set the stderr to line 
    setlinebuf(stderr); // buffering mode, since it's called at program start.
#endif // This helps ensure different programs' output is not mixed up.
    Register("config", &config_, "Configuration file with options");
    Register("print-args", &print_args_, "Print the command line arguments (to stderr)");
    Register("help", &help_, "Print out usage message");
    Register("verbose", &g_kaldi_verbose_level, "Verbose level (higher->more warnings)");
  }

  ~ParseOptions() {}

 public:
  /// Register boolean variable
  void Register(const std::string &name, bool *b, const std::string &doc);
  /// Register int32 variable
  void Register(const std::string &name, int32 *i, const std::string &doc);
  /// Register unsinged  int32 variable
  void Register(const std::string &name, uint32 *u,
                const std::string &doc);
  /// Register float variable
  void Register(const std::string &name, float *f, const std::string &doc);
  /// Register double variable [useful as we change BaseFloat type].
  void Register(const std::string &name, double *f, const std::string &doc);
  /// Register string variable
  void Register(const std::string &name, std::string *s,
                const std::string &doc);

  /**
   * Parses the command line options and fills the ParseOptions-registered
   * variables. This must be called after all the variables were registered!!!
   *
   * Initially the variables have implicit values,
   * then the config file values are set-up,
   * finally the command line vaues given.
   * Returns the first position in argv that was not used.
   * [typically not useful: use NumParams() and GetParam(). ]
   */
  int Read(int argc, const char*const *argv);

  /// Prints the usage documentation [provided in the constructor].
  void PrintUsage(bool print_command_line = false);
  /// Prints the actual configuration of all the registered variables
  void PrintConfig(std::ostream &os);

  /// Number of positional parameters (c.f. argc-1).
  int NumArgs();

  /// Returns one of the positional parameters; 1-based indexing for argc/argv
  /// compatibility. Will crash if param is not >=1 and <=NumArgs().
  std::string GetArg(int param);

  std::string GetOptArg(int param) {
    return (param <= NumArgs() ? GetArg(param) : "");
  }

  /// The following function will return a possibly quoted and escaped
  /// version of "str", according to the current shell.  Currently
  /// this is just hardwired to bash.  It's useful for debug output.
  static std::string Escape(const std::string &str);

 private:
  /// Reads the options values from a config file
  void ReadConfigFile(const std::string &filename);

  void SplitLongArg(std::string in, std::string *key, std::string *value);
  void NormalizeArgName(std::string *str);

  bool SetOption(const std::string &key, const std::string &value);

  bool ToBool(std::string str);
  int32 ToInt(std::string str);
  uint32 ToUInt(std::string str);
  float ToFloat(std::string str);
  double ToDouble(std::string str);

  // maps for option variables
  std::map<std::string, bool*> bool_map_;
  std::map<std::string, int32*> int_map_;
  std::map<std::string, uint32*> uint_map_;
  std::map<std::string, float*> float_map_;
  std::map<std::string, double*> double_map_;
  std::map<std::string, std::string*> string_map_;

  /**
   * Structure for options' documentation
   */
  struct DocInfo {
    DocInfo() {}
    DocInfo(const std::string &name, const std::string &usemsg)
      : name_(name), use_msg_(usemsg) {}

    std::string name_;
    std::string use_msg_;
  };
  typedef std::map<std::string, DocInfo> DocMapType;
  DocMapType doc_map_;  ///< map for the documentation

  bool print_args_;     ///< variable for the implicit --print-args parameter
  bool help_;           ///< variable for the implicit --help parameter
  std::string config_;  ///< variable for the implicit --config parameter
  std::vector<std::string> positional_args_;
  const char *usage_;
  int argc_;
  const char*const *argv_;
};

}  // namespace kaldi

#endif  // KALDI_UTIL_PARSE_OPTIONS_H_
