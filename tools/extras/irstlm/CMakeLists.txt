#CMake 2.6+ is recommended to an improved Boost module
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.0 FATAL_ERROR)

PROJECT (irstlm)

SET(IRSTLM_VERSION_MAJOR "5")
SET(IRSTLM_VERSION_MINOR "80")
SET(IRSTLM_VERSION_PATCH "08")

SET(CMAKE_C_COMPILER "gcc" )
SET(CMAKE_CXX_COMPILER "g++" )

OPTION(CXX0 "Enable/Disable std=c++0" ON)
OPTION(ASSERT "Enable/Disable MY_ASSERT_FLAG" ON)

MATH(EXPR IRSTLM_INT_VERSION "(${IRSTLM_VERSION_MAJOR} * 10000) + (${IRSTLM_VERSION_MINOR} * 100) + (${IRSTLM_VERSION_PATCH} * 1)" )

SET(IRSTLM_VERSION "${IRSTLM_VERSION_MAJOR}.${IRSTLM_VERSION_MINOR}.${IRSTLM_VERSION_PATCH}")

#ADD_DEFINITIONS(-DPS_CACHE_ENABLE)

IF(COMMAND cmake_policy)
  cmake_policy(SET CMP0017 NEW)
ENDIF(COMMAND cmake_policy)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX
        "${CMAKE_SOURCE_DIR}/inst" CACHE PATH "IRSTLM install prefix" FORCE
        )
    MESSAGE(STATUS "You have not set the install dir, default to ${CMAKE_INSTALL_PREFIX}, if
    you want to set it, use cmake -DCMAKE_INSTALL_PREFIX to do so")
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

#set various platform specific global options
IF(WIN32)
 SET(CMAKE_DEBUG_POSTFIX "d")
 OPTION( USE_64_BIT "Set to on if you want to compile Win64" OFF )
ENDIF(WIN32)


# include specific modules
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

ADD_SUBDIRECTORY (src)
ADD_SUBDIRECTORY (scripts)
ADD_SUBDIRECTORY (doc)

IF (WIN32)
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
MESSAGE( STATUS "USE_64_BIT = ${USE_64_BIT}" )
MESSAGE( STATUS "Change a value with: cmake -D<Variable>=<Value>" )
MESSAGE( STATUS "-------------------------------------------------------------------------------" )


SET( USE_64_BIT "${USE_64_BIT}" CACHE BOOL
    "Set to ON to build Win64" FORCE )
ENDIF()

