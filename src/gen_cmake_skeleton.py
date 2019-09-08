import os
import sys
import re

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

# class CMakeListsHeaders(object)

class CMakeListsLibrary(object):

    def __init__(self, library_name):
        self.library_name = "kaldi-" + library_name
        self.file_list = []
        self.cuda_file_list = []
        self.test_file_list = []
        self.depends = []

    def add_test_source(self, filename):
        self.test_file_list.append(filename)

    def add_source(self, filename):
        self.file_list.append(filename)

    def add_cuda_source(self, filename):
        self.cuda_file_list.append(filename)

    def load_dependency_from_makefile(self, filename):
        with open(filename) as f:
            makefile = f.read()
            if "ADDLIBS" not in makefile:
                print("WARNING: non-standard", filename)
                return
            libs = makefile.split("ADDLIBS")[-1].split("\n\n")[0]
            libs = re.findall("[^\s\\\\=]+", libs)
            for l in libs:
                self.depends.append(os.path.splitext(os.path.basename(l))[0])



    def gen_code(self):
        ret = []
        if len(self.cuda_file_list) > 0:
            self.file_list.append("${CUDA_OBJS}")
            ret.append("cuda_compile(${CUDA_OBJS}")
            for f in self.cuda_file_list:
                ret.append("    " + f)
            ret.append(")\n")

        ret.append("add_library(" + self.library_name)
        for f in self.file_list:
            ret.append("    " + f)
        ret.append(")\n")
        ret.append("target_include_directories(" + self.library_name + " PUBLIC ")
        ret.append("     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
        ret.append("     $<INSTALL_INTERFACE:include/kaldi>")
        ret.append(")\n")

        if len(self.depends) > 0:
            ret.append("target_link_libraries(" + self.library_name + " PUBLIC")
            for d in self.depends:
                ret.append("    " + d)
            ret.append(")\n")

        if len(self.test_file_list) > 0:
            ret.append("if(KALDI_BUILD_TEST)")
            for f in self.test_file_list:
                ret.append("    add_kaldi_test_executable(NAME " + os.path.splitext(f)[0] + " SOURCES " + f + " DEPENDS " + self.library_name + ")")
            ret.append("endif()")

        return "\n".join(ret)



class CMakeListsExecutable(object):

    def __init__(self, dir_name, filename):
        assert(dir_name.endswith("bin"))
        self.list = []
        exe_name = os.path.splitext(os.path.basename(filename))[0]
        file_name = filename
        depend = "kaldi-" + dir_name[:-3]
        self.list.append((exe_name, file_name, depend))

    def gen_code(self):
        ret = []
        for exe_name, file_name, depend in self.list:
            ret.append("add_kaldi_executable(NAME " + exe_name + " SOURCES " + file_name + " DEPENDS " + depend + ")")
        return "\n".join(ret)

class CMakeListsFile(object):

    def __init__(self, directory):
        self.path = os.path.realpath(os.path.join(directory, "CMakeLists.txt"))
        self.sections = []

    def add_section(self, section):
        self.sections.append(section)

    def write_file(self):
        with open(self.path, "w") as f:
            for s in self.sections:
                code = s.gen_code()
                f.write(code)
                f.write("\n")
        print("  Writed", self.path)


if __name__ == "__main__":

    subdirs = get_subdirectories(".")
    for d in subdirs:
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
            lib = CMakeListsLibrary(dir_name)
            makefile = os.path.join(d, "Makefile")
            if not os.path.exists(makefile):
                continue
            lib.load_dependency_from_makefile(makefile)
            cmakelists.add_section(lib)
            for f in get_files(d):
                filename = os.path.basename(f)
                if is_source(filename):
                    lib.add_source(filename)
                elif is_cu_source(filename):
                    lib.add_cuda_source(filename)
                elif is_test_source(filename):
                    lib.add_test_source(filename)

        cmakelists.write_file()
