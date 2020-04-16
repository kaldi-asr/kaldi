import os
import sys
import re
import fnmatch
import argparse

# earily parse, will refernece args globally
parser = argparse.ArgumentParser()
parser.add_argument("working_dir")
parser.add_argument("--quiet", default=False, action="store_true")
args = parser.parse_args()

def print_wrapper(*args_, **kwargs):
    if not args.quiet:
        print(*args_, **kwargs)

def get_subdirectories(d):
    return [name for name in os.listdir(d) if os.path.isdir(os.path.join(d, name))]

def is_bin_dir(d):
    return d.endswith("bin")

def get_files(d):
    return [name for name in os.listdir(d) if os.path.isfile(os.path.join(d, name))]

def is_header(f):
    return f.endswith(".h")

def is_cu_source(f):
    return f.endswith(".cu")

def is_test_source(f):
    return f.endswith("-test.cc")

def is_source(f):
    return f.endswith(".cc") and not is_test_source(f)

def lib_dir_name_to_lib_target(dir_name):
    return "kaldi-" + dir_name

def bin_dir_name_to_lib_target(dir_name):
    """return the primary lib target for all executable targets in this bin dir"""
    assert is_bin_dir(dir_name)
    if dir_name == "bin":
        # NOTE: "kaldi-util" might be a more strict primary lib target...
        return "kaldi-hmm"
    elif dir_name == "fstbin":
        return "kaldi-fstext"
    else:
        return "kaldi-" + dir_name[:-3]

def wrap_notwin32_condition(should_wrap, lines):
    if isinstance(lines, str):
        lines = [lines]
    if should_wrap:
        return ["if(NOT WIN32)"] + list(map(lambda l: "    " + l, lines)) + ["endif()"]
    else:
        return lines


def get_exe_additional_depends(t):
    additional = {
        # solve bin
        "align-*": ["decoder"],
        "compile-*graph*": ["decoder"],
        "decode-faster": ["decoder"],
        "latgen-faster-mapped": ["decoder"],
        "latgen-faster-mapped-parallel": ["decoder"],
        "latgen-incremental-mapped": ["decoder"],
        "decode-faster-mapped": ["decoder"],
        "sum-lda-accs": ["transform"],
        "sum-mllt-accs": ["transform"],
        "est-mllt": ["transform"],
        "est-lda": ["transform"],
        "acc-lda": ["transform"],
        "build-pfile-from-ali": ["gmm"],
        "make-*-transducer": ["fstext"],
        "phones-to-prons": ["fstext"],

        # solve gmmbin
        "post-to-feats" : ["hmm"],
        "append-post-to-feats" : ["hmm"],
        "gmm-*": ["hmm", "transform"],
        "gmm-latgen-*": ["decoder"],
        "gmm-decode-*": ["decoder"],
        "gmm-align": ["decoder"],
        "gmm-align-compiled": ["decoder"],
        "gmm-est-fmllr-gpost": ["sgmm2", "hmm"],
        "gmm-rescore-lattice": ["hmm", "lat"],

        # solve fstbin
        "make-grammar-fst": ["decoder"],

        # solve sgmm2bin
        "sgmm2-*": ["hmm"],
        "sgmm2-latgen-faster*": ["decoder"],
        "sgmm2-align-compiled": ["decoder"],
        "sgmm2-rescore-lattice": ["lat"],
        "init-ubm": ["hmm"],

        # solve nnetbin
        "nnet-train-mmi-sequential": ["lat"],
        "nnet-train-mpe-sequential": ["lat"],

        # solve nnet2bin
        "nnet-latgen-faster*": ["fstext", "decoder"],
        "nnet-align-compiled": ["decoder"],
        "nnet1-to-raw-nnet": ["nnet"],

        # solve chainbin
        "nnet3-chain-*": ["nnet3"],

        # solve latbin
        "lattice-compose": ["fstext"],
        "lattice-lmrescore": ["fstext"],
        "lattice-lmrescore-*": ["fstext", "rnnlm"],

        # solve ivectorbin
        "ivector-extract*": ["hmm"],

        # solve kwsbin
        "generate-proxy-keywords": ["fstext"],
        "transcripts-to-fsts": ["fstext"],
    }
    l = []
    for pattern in additional.keys():
        if fnmatch.fnmatch(t, pattern):
            l.extend(list(map(lambda name: lib_dir_name_to_lib_target(name), additional[pattern])))
    return sorted(list(set(l)))

def disable_for_win32(t):
    disabled = [
        "online-audio-client",
        "online-net-client",
        "online2-tcp-nnet3-decode-faster",
        "online-server-gmm-decode-faster",
        "online-audio-server-decode-faster"
    ]
    return t in disabled

class CMakeListsHeaderLibrary(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.target_name = lib_dir_name_to_lib_target(self.dir_name)
        self.header_list = []

    def add_header(self, filename):
        self.header_list.append(filename)

    def add_source(self, filename):
        pass

    def add_cuda_source(self, filename):
        pass

    def add_test_source(self, filename):
        pass

    def gen_code(self):
        ret = []
        if len(self.header_list) > 0:
            ret.append("set(PUBLIC_HEADERS")
            for f in self.header_list:
                ret.append("    " + f)
            ret.append(")\n")

        ret.append("add_library(" + self.target_name + " INTERFACE)")
        ret.append("target_include_directories(" + self.target_name + " INTERFACE ")
        ret.append("    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
        ret.append("    $<INSTALL_INTERFACE:include/kaldi>")
        ret.append(")\n")

        ret.append("""
install(TARGETS {tgt} EXPORT kaldi-targets)

install(FILES ${{PUBLIC_HEADERS}} DESTINATION include/kaldi/{dir})
""".format(tgt=self.target_name, dir=self.dir_name))

        return "\n".join(ret)

class CMakeListsLibrary(object):

    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.target_name = lib_dir_name_to_lib_target(self.dir_name)
        self.header_list = []
        self.source_list = []
        self.cuda_source_list = []
        self.test_source_list = []
        self.depends = []

    def add_header(self, filename):
        self.header_list.append(filename)

    def add_source(self, filename):
        self.source_list.append(filename)

    def add_cuda_source(self, filename):
        self.cuda_source_list.append(filename)

    def add_test_source(self, filename):
        self.test_source_list.append(filename)

    def load_dependency_from_makefile(self, filename):
        with open(filename) as f:
            makefile = f.read()
            if "ADDLIBS" not in makefile:
                print_wrapper("WARNING: non-standard", filename)
                return
            libs = makefile.split("ADDLIBS")[-1].split("\n\n")[0]
            libs = re.findall("[^\s\\\\=]+", libs)
            for l in libs:
                self.depends.append(os.path.splitext(os.path.basename(l))[0])

    def gen_code(self):
        ret = []

        if len(self.header_list) > 0:
            ret.append("set(PUBLIC_HEADERS")
            for f in self.header_list:
                ret.append("    " + f)
            ret.append(")\n")

        if len(self.cuda_source_list) > 0:
            self.source_list.append("${CUDA_OBJS}")
            ret.append("if(CUDA_FOUND)")
            ret.append("    cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)")
            ret.append("    cuda_compile(CUDA_OBJS")
            for f in self.cuda_source_list:
                ret.append("        " + f)
            ret.append("    )")
            ret.append("endif()\n")

        ret.append("add_library(" + self.target_name)
        for f in self.source_list:
            ret.append("    " + f)
        ret.append(")\n")
        ret.append("target_include_directories(" + self.target_name + " PUBLIC ")
        ret.append("     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
        ret.append("     $<INSTALL_INTERFACE:include/kaldi>")
        ret.append(")\n")

        if len(self.depends) > 0:
            ret.append("target_link_libraries(" + self.target_name + " PUBLIC")
            for d in self.depends:
                ret.append("    " + d)
            ret.append(")\n")

        def get_test_exe_name(filename):
            exe_name = os.path.splitext(f)[0]
            if self.dir_name.startswith("nnet") and exe_name.startswith("nnet"):
                return self.dir_name + "-" + exe_name.split("-", 1)[1]
            else:
                return exe_name

        if len(self.test_source_list) > 0:
            ret.append("if(KALDI_BUILD_TEST)")
            for f in self.test_source_list:
                exe_target = get_test_exe_name(f)
                depends = (self.target_name + " " + " ".join(get_exe_additional_depends(exe_target))).strip()
                ret.extend(wrap_notwin32_condition(disable_for_win32(self.target_name),
                    "    add_kaldi_test_executable(NAME " + exe_target + " SOURCES " + f + " DEPENDS " + depends + ")"))
            ret.append("endif()")

        ret.append("""
install(TARGETS {tgt}
    EXPORT kaldi-targets
    ARCHIVE DESTINATION ${{CMAKE_INSTALL_LIBDIR}}
    LIBRARY DESTINATION ${{CMAKE_INSTALL_LIBDIR}}
    RUNTIME DESTINATION ${{CMAKE_INSTALL_BINDIR}}
)

install(FILES ${{PUBLIC_HEADERS}} DESTINATION include/kaldi/{dir})
""".format(tgt=self.target_name, dir=self.dir_name))

        return "\n".join(ret)



class CMakeListsExecutable(object):

    def __init__(self, dir_name, filename):
        assert(dir_name.endswith("bin"))
        self.list = []
        exe_name = os.path.splitext(os.path.basename(filename))[0]
        file_name = filename
        depend = bin_dir_name_to_lib_target(dir_name)
        self.list.append((exe_name, file_name, depend))

    def gen_code(self):
        ret = []
        for exe_name, file_name, depend in self.list:
            depends = (depend + " " + " ".join(get_exe_additional_depends(exe_name))).strip()
            ret.extend(wrap_notwin32_condition(disable_for_win32(exe_name),
                       "add_kaldi_executable(NAME " + exe_name + " SOURCES " + file_name + " DEPENDS " + depends + ")"))

        return "\n".join(ret)

class CMakeListsFile(object):

    GEN_CMAKE_HEADER = "# generated with cmake/gen_cmake_skeleton.py, DO NOT MODIFY.\n"

    def __init__(self, directory):
        self.path = os.path.realpath(os.path.join(directory, "CMakeLists.txt"))
        self.sections = []

    def add_section(self, section):
        self.sections.append(section)

    def write_file(self):
        with open(self.path, "w", newline='\n') as f: # good luck for python2
            f.write(CMakeListsFile.GEN_CMAKE_HEADER)
            for s in self.sections:
                code = s.gen_code()
                f.write(code)
                f.write("\n")
        print_wrapper("  Writed", self.path)


if __name__ == "__main__":
    os.chdir(args.working_dir)
    print_wrapper("Working in ", args.working_dir)

    subdirs = get_subdirectories(".")
    for d in subdirs:
        if d.startswith('tfrnnlm'):
            continue
        cmakelists = CMakeListsFile(d)
        if is_bin_dir(d):
            for f in get_files(d):
                if is_source(f):
                    dir_name = os.path.basename(d)
                    filename = os.path.basename(f)
                    exe = CMakeListsExecutable(dir_name, filename)
                    cmakelists.add_section(exe)
        else:
            dir_name = os.path.basename(d)
            lib = None
            makefile = os.path.join(d, "Makefile")
            if not os.path.exists(makefile):
                lib = CMakeListsHeaderLibrary(dir_name)
            else:
                lib = CMakeListsLibrary(dir_name)
                lib.load_dependency_from_makefile(makefile)
            cmakelists.add_section(lib)
            for f in sorted(get_files(d)):
                filename = os.path.basename(f)
                if is_source(filename):
                    lib.add_source(filename)
                elif is_cu_source(filename):
                    lib.add_cuda_source(filename)
                elif is_test_source(filename):
                    lib.add_test_source(filename)
                elif is_header(filename):
                    lib.add_header(filename)

        cmakelists.write_file()
