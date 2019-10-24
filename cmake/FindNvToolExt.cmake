# The following variables are optionally searched for defaults
#  NvToolExt_ROOT_DIR:
#
# The following are set after configuration is done:
#  NvToolExt_FOUND
#  NvToolExt_INCLUDE_DIR
#  NvToolExt_LIBRARIES
#  NvToolExt_LIBRARY_DIR
#  NvToolExt:                   a target

include(FindPackageHandleStandardArgs)

set(NvToolExt_SEARCH_DIRS ${CUDA_TOOLKIT_ROOT_DIR})
if(WIN32)
    list(APPEND NvToolExt_SEARCH_DIRS "C:/Program Files/NVIDIA Corporation/NvToolsExt")
endif()
set(NvToolExt_SEARCH_DIRS ${NvToolExt_ROOT_DIR} ${NvToolExt_SEARCH_DIRS})


find_path(NvToolExt_INCLUDE_DIR nvToolsExt.h HINTS ${NvToolExt_SEARCH_DIRS} PATH_SUFFIXES include)

# 32bit not considered
set(NvToolExt_LIBNAME nvToolsExt libnvToolsExt.so libnvToolsExt.a libnvToolsExt.so nvToolsExt64_1.lib)
find_library(NvToolExt_LIBRARIES NAMES ${NvToolExt_LIBNAME} HINTS ${NvToolExt_SEARCH_DIRS}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

find_package_handle_standard_args(NvToolExt REQUIRED_VARS NvToolExt_INCLUDE_DIR NvToolExt_LIBRARIES)

add_library(NvToolExt INTERFACE)
target_include_directories(NvToolExt INTERFACE ${NvToolExt_INCLUDE_DIR})
# target_link_directories(NvToolExt INTERFACE ${NvToolExt_INCLUDE_DIR})
target_link_libraries(NvToolExt INTERFACE ${NvToolExt_LIBRARIES})

unset(NvToolExt_SEARCH_DIRS)
unset(NvToolExt_LIBNAME)
