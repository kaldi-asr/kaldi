find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenFst REQUIRED fst)

set(OPENFST_LIBRARIES ${OpenFst_LIBRARIES})
set(OPENFST_INCLUDE_DIR ${OpenFst_INCLUDE_DIRS})
set(OPENFST_VERSION ${OpenFst_VERSION})

