// pybind/util/parse_options_pybind.cc

// Copyright 2020   Mobvoi AI Lab, Beijing, China
//                  (author: Fangjun Kuang, Yaguang Hu, Jian Wang)

// See ../../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "util/parse_options_pybind.h"

#include "util/parse-options.h"

using namespace kaldi;

void pybind_parse_options(py::module& m) {
  using PyClass = ParseOptions;
  py::class_<PyClass, OptionsItf>(m, "ParseOptions")
      .def(py::init<const char*>(), py::arg("usage"))
      .def("Read",
           [](PyClass* opts, const std::vector<std::string>& args) {
             int argc = static_cast<int>(args.size());
             const char** argv = new const char*[argc + 1];
             for (int i = 0; i < argc; ++i) {
               argv[i] = args[i].c_str();
             }
             argv[argc] = nullptr;
             opts->Read(argc, argv);
             delete[] argv;
           },
           "Parses the command line options and fills the "
           "ParseOptions-registered variables. This must be called after all "
           "the variables were registered!!!"
           "\n"
           "Initially the variables have implicit values, then the config file "
           "values are set-up, finally the command line values given. Returns "
           "the first position in argv that was not used. [typically not "
           "useful: use NumParams() and GetParam(). ]",
           py::arg("args"))
      .def("PrintUsage", &PyClass::PrintUsage,
           "Prints the usage documentation [provided in the constructor].",
           py::arg("print_command_line") = false)
      .def("ReadConfigFile", &PyClass::ReadConfigFile,
           "Reads the options values from a config file.  Must be called after "
           "registering all options.  This is usually used internally after "
           "the standard --config option is used, but it may also be called "
           "from a program.",
           py::arg("filename"))
      .def("NumArgs", &PyClass::NumArgs,
           "Number of positional parameters (c.f. argc-1).")
      .def("GetArg", &PyClass::GetArg,
           "Returns one of the positional parameters; 1-based indexing for "
           "argc/argv compatibility. Will crash if param is not >=1 and "
           "<=NumArgs().",
           py::arg("param"))
      .def("GetOptArg", &PyClass::GetOptArg, py::arg("param"))
      .def("__str__", [](PyClass* opts) {
        std::ostringstream os;
        opts->PrintConfig(os);
        return os.str();
      });
}
