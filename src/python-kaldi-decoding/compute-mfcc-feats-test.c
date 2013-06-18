#include "test_cffi_python_dyn.h"

int main(int argc, char **argv) {
    return testSharedLib("libkaldi-cffi.so", "compute_mfcc_feats_like_main", argc, argv);
}
