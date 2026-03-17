cmake_minimum_required(VERSION 3.16)
project(ucminf VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# ucminf_core — the pure C++ optimization library
# ---------------------------------------------------------------------------
add_library(ucminf_core STATIC
    src/ucminf_core.cpp
)
target_include_directories(ucminf_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/ucminf>
)
target_compile_features(ucminf_core PUBLIC cxx_std_17)

# Aggressive optimisation flags for numerical workloads.
# Applied only for GCC-compatible compilers (GCC, Clang, AppleClang).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(ucminf_core PRIVATE
        -O3
        -funroll-loops
    )
endif()

# ---------------------------------------------------------------------------
# Python bindings (optional, requires pybind11)
# ---------------------------------------------------------------------------
option(BUILD_PYTHON "Build Python pybind11 bindings" OFF)
if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

# ---------------------------------------------------------------------------
# Julia bindings (optional, requires CxxWrap / libcxxwrap-julia)
# ---------------------------------------------------------------------------
option(BUILD_JULIA "Build Julia CxxWrap bindings" OFF)
if(BUILD_JULIA)
    add_subdirectory(julia)
endif()

# ---------------------------------------------------------------------------
# C++ unit tests (optional, uses CMake FetchContent to pull Catch2)
# ---------------------------------------------------------------------------
option(BUILD_TESTS "Build C++ unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS ucminf_core
    EXPORT ucminfTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES src/ucminf_core.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ucminf
)

install(EXPORT ucminfTargets
    FILE ucminfTargets.cmake
    NAMESPACE ucminf::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ucminf
)

# ---------------------------------------------------------------------------
# Package configuration for find_package(ucminf)
# ---------------------------------------------------------------------------
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ucminfConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ucminfConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ucminfConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ucminf
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ucminfConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ucminfConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ucminf
)
