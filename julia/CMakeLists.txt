cmake_minimum_required(VERSION 3.16)
project(ucminf_julia LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Locate Julia and JlCxx
# ---------------------------------------------------------------------------
# JlCxx is provided by the CxxWrap.jl package.  Run:
#   julia -e 'using CxxWrap; println(CxxWrap.prefix_path())'
# and pass the result as -DJlCxx_DIR=<prefix>/lib/cmake/JlCxx
# ---------------------------------------------------------------------------
find_package(JlCxx REQUIRED)

# ---------------------------------------------------------------------------
# Shared library exposed to Julia
# ---------------------------------------------------------------------------
add_library(ucminf_julia SHARED
    ucminf_julia.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/ucminf_core.cpp
)

target_include_directories(ucminf_julia PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(ucminf_julia PUBLIC JlCxx::cxxwrap_julia)

# On macOS link as a bundle so Julia can dlopen() it
if(APPLE)
    set_target_properties(ucminf_julia PROPERTIES
        BUNDLE FALSE
        SUFFIX ".dylib"
    )
endif()
