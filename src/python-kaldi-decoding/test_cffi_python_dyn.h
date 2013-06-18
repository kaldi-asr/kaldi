#ifndef TEST_CFFI_PYTHON_H
#define TEST_CFFI_PYTHON_H
#include <dlfcn.h>
#include <stdio.h>


/** Links ******
 - http://www.isotton.com/devel/docs/C++-dlopen-mini-HOWTO/C++-dlopen-mini-HOWTO.html
 - http://stackoverflow.com/questions/12762910/c-undefined-symbols-when-loading-shared-library-with-dlopen
 - http://kaldi.sourceforge.net/matrixwrap.html  # see Missing the ATLAS implementation of (parts of) CLAPACK
 - you have to choose lapack_atlas / lapack /clapack.. check symbols

**********************/

typedef int (*f_t)(int c, char **ar);

int testSharedLib(char *nameLib, char *nameFce, int argc, char ** argv) {
    void *lib = dlopen(nameLib, RTLD_NOW);
    if (!lib) {
        printf("Cannot open library: %s\n", dlerror());
        return 1;
    }   

    dlerror();  // reset errors
    f_t f = (f_t)dlsym(lib, nameFce); 
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
        printf("Cannot load symbol '%s', %s\n", nameFce, dlsym_error );
        dlclose(lib);
        return 1;
    }

    // using the function
    int retval = f(argc, argv);
    
    dlclose(lib);
    return retval;
}
#endif // #ifndef TEST_CFFI_PYTHON_H
