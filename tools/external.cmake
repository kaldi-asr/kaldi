include(ExternalProject)

set(CUB_VERSION 1.8.0)

ExternalProject_Add(cub
  URL https://github.com/NVlabs/cub/archive/${CUB_VERSION}.zip
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/cub
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND :
  BUILD_COMMAND :
  INSTALL_COMMAND :
  )

set(OPENFST_VERSION 1.6.7)

ExternalProject_Add(openfst
  URL http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.7.tar.gz
  # PREFIX ${CMAKE_CURRENT_BINARY_DIR}/openfst
  BUILD_IN_SOURCE 1
  #SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/openfst
  INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/openfst-install
  CONFIGURE_COMMAND ./configure --prefix=${CMAKE_CURRENT_BINARY_DIR}/openfst-install --enable-static --enable-shared --enable-far --enable-ngram-fsts
  BUILD_COMMAND make -j 8
  INSTALL_COMMAND make install
  )
set(OPENFST_BIN_DIRS ${CMAKE_CURRENT_BINARY_DIR}/openfst-install/bin)
set(OPENFST_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/openfst-install/include)
set(OPENFST_LIBRARY_DIRS ${CMAKE_CURRENT_BINARY_DIR}/openfst-install/lib)
