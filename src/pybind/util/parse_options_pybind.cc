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

namespace {

template <typename Type>
struct ArgName;

#define DEFINE_ARG_NAME(type, name)             \
  template <>                                   \
  struct ArgName<type> {                        \
    static constexpr const char* value = #name; \
  }

DEFINE_ARG_NAME(bool, BoolArg);
DEFINE_ARG_NAME(int32, Int32Arg);
DEFINE_ARG_NAME(uint32, UInt32Arg);
DEFINE_ARG_NAME(float, FloatArg);
DEFINE_ARG_NAME(double, DoubleArg);
DEFINE_ARG_NAME(std::string, StringArg);

#undef DEFINE_ARG_NAME

template <typename Type>
struct Arg {
  Type value{};
  Arg() = default;
  Arg(const Type& v) : value(v) {}
};

template <typename Type, typename Opt>
void pybind_arg(py::module& m, Opt& opt) {
  using PyClass = Arg<Type>;
  py::class_<PyClass>(m, ArgName<Type>::value)
      .def(py::init<>())
      .def(py::init<const Type&>(), py::arg("v"))
      .def_readwrite("value", &PyClass::value)
      .def("__str__", [](const PyClass& arg) {
        std::ostringstream os;
        os << arg.value;
        return os.str();
      });

  opt.def("Register",
          [](typename Opt::type* o, const std::string& name, PyClass* arg,
             const std::string& doc) { o->Register(name, &arg->value, doc); },
          py::arg("name"), py::arg("arg"), py::arg("doc"));
}

}  // namespace

void pybind_parse_options(py::module& m) {
  using PyClass = ParseOptions;

  auto opt =
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
               "ParseOptions-registered variables. This must be called after "
               "all "
               "the variables were registered!!!"
               "\n"
               "Initially the variables have implicit values, then the config "
               "file "
               "values are set-up, finally the command line values given. "
               "Returns "
               "the first position in argv that was not used. [typically not "
               "useful: use NumParams() and GetParam(). ]",
               py::arg("args"))
          .def("PrintUsage", &PyClass::PrintUsage,
               "Prints the usage documentation [provided in the constructor].",
               py::arg("print_command_line") = false)
          .def(
              "ReadConfigFile", &PyClass::ReadConfigFile,
              "Reads the options values from a config file.  Must be called "
              "after "
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

  pybind_arg<bool>(m, opt);
  pybind_arg<int32>(m, opt);
  pybind_arg<uint32>(m, opt);
  pybind_arg<float>(m, opt);
  pybind_arg<double>(m, opt);
  pybind_arg<std::string>(m, opt);
}
