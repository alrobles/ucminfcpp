cmake_minimum_required(VERSION 3.16)

# ---------------------------------------------------------------------------
# Python extension module via pybind11
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(ucminf ucminf_pybind.cpp ../src/ucminf_core.cpp)

target_include_directories(ucminf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(ucminf PRIVATE cxx_std_17)
